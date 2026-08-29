// overlap_bench.cu — how much AllReduce can hide behind compute?
//
// Phase 7B. The goal is NOT to make NCCL faster. It is to measure how much
// communication disappears behind useful compute and how much stays exposed on
// the critical path, and to reproduce the DDP bucket scheduling principle.
//
// WHAT PHASES 6 AND 7A ESTABLISHED, AND THIS BUILDS ON
//   * the host-clock-plus-barrier timing scheme is sound (7A calibration:
//     every harness component is single-digit microseconds);
//   * device-wide barriers are expensive -- 91-93% of the naive ring's runtime;
//   * asynchronous submission does NOT imply overlap, so concurrency is
//     verified on a timeline rather than assumed;
//   * a P2P capability bit is not a functional test.
//
// STRUCTURE
//   compute      cuBLAS SGEMM repeated on a dedicated stream -- representative
//                of training compute, deterministic, duration tunable
//   T_comm       NCCL AllReduce alone
//   T_compute    GEMM alone
//   T_seq        compute, barrier, AllReduce  (fully serialised)
//   T_overlap    both issued concurrently on separate streams
//   T_ideal      max(T_compute, T_comm)
//
//   overlap_efficiency     = (T_seq - T_overlap) / (T_seq - T_ideal)
//   effective_exposed_comm = max(0, T_overlap - T_compute)
//
// Both are END-TO-END metrics. Neither proves what the communication kernel
// itself did; the timeline is what settles that.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nccl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#if defined(USE_NVTX)
#include <nvtx3/nvToolsExt.h>
#define NVTX_PUSH(n) nvtxRangePushA(n)
#define NVTX_POP()   nvtxRangePop()
#else
#define NVTX_PUSH(n) ((void)0)
#define NVTX_POP()   ((void)0)
#endif

#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); std::exit(1);} }while(0)
#define NK(x) do{ ncclResult_t r_=(x); if(r_!=ncclSuccess){ \
  std::fprintf(stderr,"NCCL %s @%d: %s\n",#x,__LINE__,ncclGetErrorString(r_)); std::exit(1);} }while(0)
#define BK(x) do{ cublasStatus_t s_=(x); if(s_!=CUBLAS_STATUS_SUCCESS){ \
  std::fprintf(stderr,"cuBLAS %s @%d: %d\n",#x,__LINE__,(int)s_); std::exit(1);} }while(0)

static int NG = 4;
static std::vector<int> DEV;

// ---------------------------------------------------------------------------
// Phase 8: three workload classes with deliberately different resource
// fingerprints. All three are calibrated to the SAME standalone duration for a
// given target, so "which one degrades more under overlap" is a fair question —
// otherwise a longer workload would appear to suffer simply by overlapping more
// of the collective.
//
//   COMPUTE  cuBLAS SGEMM            high arithmetic intensity, SM/FLOP bound
//   MEMORY   streaming triad         ~0.17 flop/byte, DRAM-bandwidth bound
//   MIXED    triad + inner FMA loop  ~8 flop/byte, between the two
//
// The classification is by DESIGN (arithmetic intensity by construction). It is
// labelled counter-verified only if Nsight Compute is actually available.
// ---------------------------------------------------------------------------
enum Workload { WL_COMPUTE = 0, WL_MEMORY = 1, WL_MIXED = 2 };
static const char* WL_NAME[] = {"compute-gemm", "memory-triad", "mixed"};

// d[i] = a[i] + s*b[i] : 12 bytes moved, 2 flops -> ~0.17 flop/byte
__global__ void streamTriad(float* d, const float* a, const float* b, float s, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t st = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += st) d[i] = a[i] + s * b[i];
}

// 8 bytes moved, 2*inner flops -> ~inner/4 flop/byte
__global__ void mixedKernel(float* d, const float* a, size_t n, int inner) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t st = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += st) {
    float v = a[i];
    #pragma unroll 8
    for (int k = 0; k < inner; ++k) v = fmaf(v, 1.0000001f, 0.0000001f);
    d[i] = v;
  }
}

struct Ctx {
  std::vector<cudaStream_t> sCompute, sComm;
  std::vector<cublasHandle_t> blas;
  std::vector<float*> A, B, C;      // GEMM operands
  std::vector<float*> grad;         // communication buffer
  std::vector<cudaEvent_t> evCompS, evCompE, evCommS, evCommE;
  std::vector<ncclComm_t> comm;
  int gemmN = 512;
  size_t gradElems = 0;
  // streaming buffers for the memory-bound and mixed classes
  std::vector<float*> mA, mB, mD;
  size_t memElems = 0;
  int mixedInner = 32;
};

// ---------------------------------------------------------------------------
// timing: the Phase 7A-validated scheme -- host clock bracketed by a full
// device barrier, warmup outside, setup outside.
// ---------------------------------------------------------------------------
static void barrier() {
  for (int d : DEV) { CK(cudaSetDevice(d)); CK(cudaDeviceSynchronize()); }
}
template <typename F>
static double timeUs(F&& body, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) body();
  barrier();
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) body();
  barrier();
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}
static double medianOf(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : v[v.size()/2];
}

// ---------------------------------------------------------------------------
// workloads
// ---------------------------------------------------------------------------
static void launchCompute(Ctx& c, int reps, Workload wl = WL_COMPUTE) {
  const float alpha = 1.0f, beta = 0.0f;
  const int n = c.gemmN;
  for (int i = 0; i < NG; ++i) {
    CK(cudaSetDevice(DEV[i]));
    for (int r = 0; r < reps; ++r) {
      switch (wl) {
        case WL_COMPUTE:
          BK(cublasSgemm(c.blas[i], CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                         &alpha, c.A[i], n, c.B[i], n, &beta, c.C[i], n));
          break;
        case WL_MEMORY:
          streamTriad<<<512, 256, 0, c.sCompute[i]>>>(c.mD[i], c.mA[i], c.mB[i],
                                                      2.0f, c.memElems);
          break;
        case WL_MIXED:
          mixedKernel<<<512, 256, 0, c.sCompute[i]>>>(c.mD[i], c.mA[i],
                                                      c.memElems, c.mixedInner);
          break;
      }
    }
  }
}
static void launchComm(Ctx& c, size_t elems) {
  NK(ncclGroupStart());
  for (int i = 0; i < NG; ++i) {
    CK(cudaSetDevice(DEV[i]));
    NK(ncclAllReduce(c.grad[i], c.grad[i], elems, ncclFloat, ncclSum,
                     c.comm[i], c.sComm[i]));
  }
  NK(ncclGroupEnd());
}
static void syncStreams(Ctx& c, bool compute, bool comm) {
  for (int i = 0; i < NG; ++i) {
    CK(cudaSetDevice(DEV[i]));
    if (compute) CK(cudaStreamSynchronize(c.sCompute[i]));
    if (comm)    CK(cudaStreamSynchronize(c.sComm[i]));
  }
}

// ---------------------------------------------------------------------------
// per-stream duration during a run, so interference can be measured rather
// than inferred: events bracket each stream's own work.
// ---------------------------------------------------------------------------
static void recordStart(Ctx& c) {
  for (int i = 0; i < NG; ++i) {
    CK(cudaSetDevice(DEV[i]));
    CK(cudaEventRecord(c.evCompS[i], c.sCompute[i]));
    CK(cudaEventRecord(c.evCommS[i], c.sComm[i]));
  }
}
static void recordEnd(Ctx& c) {
  for (int i = 0; i < NG; ++i) {
    CK(cudaSetDevice(DEV[i]));
    CK(cudaEventRecord(c.evCompE[i], c.sCompute[i]));
    CK(cudaEventRecord(c.evCommE[i], c.sComm[i]));
  }
}
static void elapsed(Ctx& c, double& compUs, double& commUs) {
  float best_c = 0, best_m = 0, t = 0;
  for (int i = 0; i < NG; ++i) {
    CK(cudaSetDevice(DEV[i]));
    CK(cudaEventElapsedTime(&t, c.evCompS[i], c.evCompE[i])); best_c = std::max(best_c, t);
    CK(cudaEventElapsedTime(&t, c.evCommS[i], c.evCommE[i])); best_m = std::max(best_m, t);
  }
  compUs = best_c * 1000.0; commUs = best_m * 1000.0;   // ms -> us, slowest rank
}

int main(int argc, char** argv) {
  int warmup = 3, iters = 10, repeats = 3, gemmN = 512;
  bool doBuckets = true, doMicro = true;
  size_t memElems = 8u<<20;            // 32 MiB per array, ~96 MiB traffic per pass
  int mixedInner = 32;
  std::vector<int> workloads = {WL_COMPUTE};   // Phase 7B default
  std::vector<size_t> commSizes = {1u<<20, 16u<<20, 128u<<20};
  std::vector<double> ratios = {0.25, 0.5, 1.0, 2.0};
  std::vector<size_t> bucketSizes = {4u<<20, 8u<<20, 16u<<20, 32u<<20, 64u<<20};
  size_t totalGrad = 128u<<20;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto nx = [&]() -> std::string {
      if (i+1 >= argc) { std::fprintf(stderr,"missing value after %s\n", argv[i]); std::exit(2);} 
      return std::string(argv[++i]); };
    if (a=="-g") NG = std::stoi(nx());
    else if (a=="-w") warmup = std::stoi(nx());
    else if (a=="-n") iters = std::stoi(nx());
    else if (a=="--repeats") repeats = std::stoi(nx());
    else if (a=="--gemm") gemmN = std::stoi(nx());
    else if (a=="--mem-elems") memElems = (size_t)std::stoull(nx());
    else if (a=="--mixed-inner") mixedInner = std::stoi(nx());
    else if (a=="--workloads") {
      std::string s = nx(), cur; workloads.clear();
      for (size_t k=0;k<=s.size();++k) {
        if (k==s.size() || s[k]==',') {
          if (cur=="compute"||cur=="gemm") workloads.push_back(WL_COMPUTE);
          else if (cur=="memory"||cur=="mem") workloads.push_back(WL_MEMORY);
          else if (cur=="mixed") workloads.push_back(WL_MIXED);
          else if (!cur.empty()) { std::fprintf(stderr,"unknown workload %s\n",cur.c_str()); return 2; }
          cur.clear();
        } else cur += s[k];
      }
    }
    else if (a=="--only-micro") doBuckets = false;
    else if (a=="--only-buckets") doMicro = false;
    // Observability filters: profiling the whole matrix makes an unreadable
    // trace. These pick one case so its timeline can be inspected.
    else if (a=="--sizes") {
      std::string s = nx(), cur; commSizes.clear();
      for (size_t k=0;k<=s.size();++k) {
        if (k==s.size() || s[k]==',') {
          if (!cur.empty()) { size_t m=1; char c2=cur.back();
            if (c2=='K'||c2=='k') m=1u<<10; else if (c2=='M'||c2=='m') m=1u<<20;
            if (m>1) cur.pop_back();
            commSizes.push_back((size_t)std::stoull(cur)*m); }
          cur.clear();
        } else cur += s[k];
      }
    }
    else if (a=="--ratios") {
      std::string s = nx(), cur; ratios.clear();
      for (size_t k=0;k<=s.size();++k) {
        if (k==s.size() || s[k]==',') { if(!cur.empty()) ratios.push_back(std::stod(cur)); cur.clear(); }
        else cur += s[k];
      }
    }
    else if (a=="--buckets") {
      std::string s = nx(), cur; bucketSizes.clear();
      for (size_t k=0;k<=s.size();++k) {
        if (k==s.size() || s[k]==',') {
          if (!cur.empty()) { size_t m=1; char c2=cur.back();
            if (c2=='K'||c2=='k') m=1u<<10; else if (c2=='M'||c2=='m') m=1u<<20;
            if (m>1) cur.pop_back();
            bucketSizes.push_back((size_t)std::stoull(cur)*m); }
          cur.clear();
        } else cur += s[k];
      }
    }
  }

  int avail=0; CK(cudaGetDeviceCount(&avail));
  if (avail < NG) { std::fprintf(stderr,"need %d GPUs, have %d\n",NG,avail); return 1; }
  DEV.resize(NG); std::iota(DEV.begin(), DEV.end(), 0);

  // --- environment / transport evidence (never assume a capability bit) -----
  std::printf("# overlap benchmark: %d GPUs\n", NG);
  for (int d : DEV) { cudaDeviceProp p{}; CK(cudaGetDeviceProperties(&p,d));
    std::printf("# device %d: %s sm_%d%d\n", d, p.name, p.major, p.minor); }
  for (int i=0;i<NG;++i) for (int j=0;j<NG;++j) if (i!=j) {
    int can=0; CK(cudaDeviceCanAccessPeer(&can, DEV[i], DEV[j]));
    if (i==0) std::printf("# canAccessPeer %d->%d: %s\n", DEV[i], DEV[j], can?"yes":"no");
  }
  std::printf("# NCCL's own transport choice is the authority here; run with\n"
              "# NCCL_DEBUG=INFO to capture it alongside these results.\n");

  Ctx c; c.gemmN = gemmN;
  c.sCompute.resize(NG); c.sComm.resize(NG); c.blas.resize(NG);
  c.A.resize(NG); c.B.resize(NG); c.C.resize(NG); c.grad.resize(NG);
  c.evCompS.resize(NG); c.evCompE.resize(NG); c.evCommS.resize(NG); c.evCommE.resize(NG);
  c.comm.resize(NG);
  c.mA.resize(NG); c.mB.resize(NG); c.mD.resize(NG);
  c.memElems = memElems; c.mixedInner = mixedInner;
  const size_t maxElems = totalGrad / sizeof(float);
  const size_t gn = (size_t)gemmN * gemmN;

  for (int i=0;i<NG;++i) {
    CK(cudaSetDevice(DEV[i]));
    CK(cudaStreamCreate(&c.sCompute[i]));
    CK(cudaStreamCreate(&c.sComm[i]));
    BK(cublasCreate(&c.blas[i]));
    BK(cublasSetStream(c.blas[i], c.sCompute[i]));
    CK(cudaMalloc(&c.A[i], gn*sizeof(float)));
    CK(cudaMalloc(&c.B[i], gn*sizeof(float)));
    CK(cudaMalloc(&c.C[i], gn*sizeof(float)));
    CK(cudaMemset(c.A[i], 1, gn*sizeof(float)));
    CK(cudaMemset(c.B[i], 1, gn*sizeof(float)));
    CK(cudaMalloc(&c.grad[i], maxElems*sizeof(float)));
    CK(cudaMemset(c.grad[i], 0, maxElems*sizeof(float)));
    CK(cudaMalloc(&c.mA[i], c.memElems*sizeof(float)));
    CK(cudaMalloc(&c.mB[i], c.memElems*sizeof(float)));
    CK(cudaMalloc(&c.mD[i], c.memElems*sizeof(float)));
    CK(cudaMemset(c.mA[i], 1, c.memElems*sizeof(float)));
    CK(cudaMemset(c.mB[i], 1, c.memElems*sizeof(float)));
    CK(cudaEventCreate(&c.evCompS[i])); CK(cudaEventCreate(&c.evCompE[i]));
    CK(cudaEventCreate(&c.evCommS[i])); CK(cudaEventCreate(&c.evCommE[i]));
  }
  NK(ncclCommInitAll(c.comm.data(), NG, DEV.data()));
  barrier();

  // --- calibrate the compute workload --------------------------------------
  NVTX_PUSH("calibrate");
  double usPerRep[3];
  for (int w = 0; w < 3; ++w)
    usPerRep[w] = timeUs([&](){ launchCompute(c,1,(Workload)w); syncStreams(c,true,false); }, 5, 20);
  NVTX_POP();
  // Standalone characterisation, printed as evidence rather than assumed.
  const double memBytesPerPass = 12.0 * (double)c.memElems;                    // triad: read a,b write d
  const double mixBytesPerPass = 8.0  * (double)c.memElems;                    // read a, write d
  const double gemmFlops = 2.0 * (double)gemmN * gemmN * gemmN;
  std::printf("# workload standalone characterisation (per repetition, per GPU):\n");
  std::printf("#   %-14s %8.2f us  gemm %dx%d  %.2f GFLOP  -> %.1f GFLOP/s  (AI ~ %.0f flop/byte, design)\n",
              WL_NAME[WL_COMPUTE], usPerRep[WL_COMPUTE], gemmN, gemmN, gemmFlops/1e9,
              gemmFlops/(usPerRep[WL_COMPUTE]*1e-6)/1e9, gemmFlops/(3.0*gemmN*gemmN*4.0));
  std::printf("#   %-14s %8.2f us  %.0f MiB moved -> %.1f GB/s          (AI ~ 0.17 flop/byte, design)\n",
              WL_NAME[WL_MEMORY], usPerRep[WL_MEMORY], memBytesPerPass/(1<<20),
              memBytesPerPass/(usPerRep[WL_MEMORY]*1e-6)/1e9);
  std::printf("#   %-14s %8.2f us  %.0f MiB moved -> %.1f GB/s          (AI ~ %.0f flop/byte, design)\n",
              WL_NAME[WL_MIXED], usPerRep[WL_MIXED], mixBytesPerPass/(1<<20),
              mixBytesPerPass/(usPerRep[WL_MIXED]*1e-6)/1e9, mixedInner/4.0);
  std::printf("# arithmetic intensity is BY CONSTRUCTION; it is counter-verified only if\n"
              "# Nsight Compute was available on this host (recorded separately).\n");
  auto repsFor = [&](Workload w, double targetUs){
    int r=(int)std::lround(targetUs/std::max(usPerRep[w],1e-6)); return std::max(r,1); };

  std::printf("experiment,workload,comm_bytes,ratio_target,gemm_reps,bucket_bytes,n_buckets,"
              "t_compute_us,t_comm_us,t_seq_us,t_overlap_us,t_ideal_us,"
              "overlap_efficiency,exposed_comm_us,comp_during_overlap_us,comm_during_overlap_us,repeat\n");

  // =========================================================================
  // controlled microbenchmark
  // =========================================================================
  if (doMicro) for (size_t bytes : commSizes) {
    const size_t elems = bytes/sizeof(float);
    double tComm = timeUs([&](){ launchComm(c,elems); syncStreams(c,false,true); }, warmup, iters);

    for (int wi : workloads) {
      const Workload wl = (Workload)wi;
      for (double ratio : ratios) {
        // Every workload class is calibrated to the SAME target duration, so a
        // slowdown comparison between classes is not confounded by one of them
        // simply running longer.
        const int reps = repsFor(wl, tComm*ratio);
        for (int rep = 0; rep < repeats; ++rep) {
          double tCompute = timeUs([&](){ launchCompute(c,reps,wl); syncStreams(c,true,false); }, warmup, iters);
          double tCommHere = timeUs([&](){ launchComm(c,elems); syncStreams(c,false,true); }, warmup, iters);

          NVTX_PUSH("sequential");
          double tSeq = timeUs([&](){
            launchCompute(c,reps,wl); syncStreams(c,true,false);
            launchComm(c,elems);      syncStreams(c,false,true);
          }, warmup, iters);
          NVTX_POP();

          NVTX_PUSH("overlapped");
          double tOv = timeUs([&](){
            recordStart(c);
            launchComm(c,elems);
            launchCompute(c,reps,wl);
            recordEnd(c);
            syncStreams(c,true,true);
          }, warmup, iters);
          NVTX_POP();
          double cDur=0, mDur=0; elapsed(c, cDur, mDur);

          double tIdeal = std::max(tCompute, tCommHere);
          double denom  = tSeq - tIdeal;
          double eff    = (denom > 1.0) ? (tSeq - tOv)/denom : NAN;
          double exposed= std::max(0.0, tOv - tCompute);

          std::printf("micro,%s,%zu,%.2f,%d,0,0,%.3f,%.3f,%.3f,%.3f,%.3f,%.4f,%.3f,%.3f,%.3f,%d\n",
                      WL_NAME[wl], bytes, ratio, reps, tCompute, tCommHere, tSeq, tOv, tIdeal,
                      eff, exposed, cDur, mDur, rep);
          std::fflush(stdout);
        }
      }
    }
  }

  // =========================================================================
  // DDP-like bucket pipeline
  //
  // Simulated backward: B stages, each producing one gradient bucket. As soon
  // as a bucket is ready its AllReduce is launched asynchronously and the next
  // stage's compute continues. One final synchronisation at the end -- exactly
  // the scheduling principle DDP uses, without any of PyTorch's machinery.
  // =========================================================================
  if (doBuckets) for (size_t bucket : bucketSizes) {
    if (bucket > totalGrad) continue;
    const int nb = (int)(totalGrad / bucket);
    const size_t be = bucket/sizeof(float);

    // total compute is held at roughly one standalone AllReduce of the whole
    // gradient, so the pipeline has a realistic amount of work to hide behind
    double tCommAll = timeUs([&](){ launchComm(c, totalGrad/sizeof(float)); syncStreams(c,false,true); },
                             warmup, std::max(iters/2,3));
    const int repsPerStage = repsFor(WL_COMPUTE, tCommAll / nb);

    for (int rep = 0; rep < repeats; ++rep) {
      // compute-only reference for the whole simulated backward pass
      double tCompute = timeUs([&](){
        for (int b=0;b<nb;++b) launchCompute(c, repsPerStage);
        syncStreams(c,true,false);
      }, warmup, std::max(iters/2,3));

      // standalone communication of the same total volume, in nb collectives
      double tComm = timeUs([&](){
        for (int b=0;b<nb;++b) launchComm(c, be);
        syncStreams(c,false,true);
      }, warmup, std::max(iters/2,3));

      double tSeq = timeUs([&](){
        for (int b=0;b<nb;++b) launchCompute(c, repsPerStage);
        syncStreams(c,true,false);
        for (int b=0;b<nb;++b) launchComm(c, be);
        syncStreams(c,false,true);
      }, warmup, std::max(iters/2,3));

      NVTX_PUSH("ddp-pipeline");
      double tOv = timeUs([&](){
        recordStart(c);
        for (int b=0;b<nb;++b) {
          launchCompute(c, repsPerStage);   // backward stage b
          launchComm(c, be);                // bucket b ready -> async AllReduce
        }
        recordEnd(c);
        syncStreams(c,true,true);           // final synchronisation
      }, warmup, std::max(iters/2,3));
      NVTX_POP();
      double cDur=0, mDur=0; elapsed(c, cDur, mDur);

      double tIdeal = std::max(tCompute, tComm);
      double denom  = tSeq - tIdeal;
      double eff    = (denom > 1.0) ? (tSeq - tOv)/denom : NAN;
      double exposed= std::max(0.0, tOv - tCompute);

      std::printf("bucket,%s,%zu,1.00,%d,%zu,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.4f,%.3f,%.3f,%.3f,%d\n",
                  WL_NAME[WL_COMPUTE], totalGrad, repsPerStage, bucket, nb, tCompute, tComm, tSeq, tOv,
                  tIdeal, eff, exposed, cDur, mDur, rep);
      std::fflush(stdout);
    }
  }

  for (int i=0;i<NG;++i) ncclCommDestroy(c.comm[i]);
  for (int i=0;i<NG;++i) { CK(cudaSetDevice(DEV[i]));
    cublasDestroy(c.blas[i]);
    cudaFree(c.A[i]); cudaFree(c.B[i]); cudaFree(c.C[i]); cudaFree(c.grad[i]);
    cudaFree(c.mA[i]); cudaFree(c.mB[i]); cudaFree(c.mD[i]);
    cudaStreamDestroy(c.sCompute[i]); cudaStreamDestroy(c.sComm[i]);
    cudaEventDestroy(c.evCompS[i]); cudaEventDestroy(c.evCompE[i]);
    cudaEventDestroy(c.evCommS[i]); cudaEventDestroy(c.evCommE[i]); }
  return 0;
}
