// harness_calibration.cu — what does the Phase 6 benchmark harness actually cost?
//
// Phase 6 reported a ~4.4 ms floor for every implementation including NCCL,
// while Phase 5 measured the same NCCL collective at 35 us with nccl-tests.
// This program measures the incremental cost of each harness component in
// isolation so the floor can be attributed by evidence rather than by guess.
//
// Every case is timed with the SAME host-clock-plus-barrier scheme the Phase 6
// harness uses, so the numbers compose.
//
// Build:  make -f Makefile calibration
// Run:    ./build/harness_calibration -g 4 -n 200

#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA %s at %d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); std::exit(1);} } while(0)
#define NK(x) do { ncclResult_t r_=(x); if(r_!=ncclSuccess){ \
  std::fprintf(stderr,"NCCL %s at %d: %s\n",#x,__LINE__,ncclGetErrorString(r_)); std::exit(1);} } while(0)

__global__ void emptyKernel() {}
__global__ void addInto(float* d, const float* s, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t st = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += st) d[i] += s[i];
}

static int NG = 4, ITERS = 200;
static std::vector<int> DEV;

// The Phase 6 timing scheme, reproduced exactly.
template <typename F>
static double timeUs(F&& body, int iters) {
  for (int i = 0; i < 5; ++i) body();
  for (int d : DEV) { CK(cudaSetDevice(d)); CK(cudaDeviceSynchronize()); }
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) body();
  for (int d : DEV) { CK(cudaSetDevice(d)); CK(cudaDeviceSynchronize()); }
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

static void report(const char* label, double us, const char* note = "") {
  std::printf("%-52s %12.3f us   %s\n", label, us, note);
}

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-g" && i + 1 < argc) NG = std::atoi(argv[++i]);
    else if (a == "-n" && i + 1 < argc) ITERS = std::atoi(argv[++i]);
  }
  int avail = 0; CK(cudaGetDeviceCount(&avail));
  if (avail < NG) { std::fprintf(stderr, "need %d GPUs, have %d\n", NG, avail); return 1; }
  DEV.resize(NG); std::iota(DEV.begin(), DEV.end(), 0);

  // Warm every context first: the first touch of a device creates its context,
  // which is expensive and would otherwise be charged to whichever case ran
  // first rather than to setup.
  for (int d : DEV) { CK(cudaSetDevice(d)); CK(cudaFree(nullptr)); CK(cudaDeviceSynchronize()); }

  std::printf("# harness calibration: %d GPUs, %d iterations per case\n", NG, ITERS);
  std::printf("# every case uses the Phase 6 timing scheme (host clock + full device barrier)\n");
  std::printf("%-52s %15s   %s\n", "case", "per iteration", "note");
  std::printf("%s\n", std::string(96, '-').c_str());

  // --- A: empty host region --------------------------------------------------
  report("A. empty host body (chrono + loop overhead)",
         timeUs([](){ }, ITERS * 50), "lower bound of the method");

  // --- C: cudaSetDevice ------------------------------------------------------
  report("C1. cudaSetDevice x1 (same device, no switch)",
         timeUs([&](){ CK(cudaSetDevice(DEV[0])); }, ITERS * 20), "");
  report("C2. cudaSetDevice cycling over all GPUs",
         timeUs([&](){ for (int d : DEV) CK(cudaSetDevice(d)); }, ITERS * 20),
         "the pattern syncAll() uses");

  // --- D: synchronization ----------------------------------------------------
  CK(cudaSetDevice(DEV[0]));
  report("D1. cudaDeviceSynchronize x1 (device idle)",
         timeUs([&](){ CK(cudaSetDevice(DEV[0])); CK(cudaDeviceSynchronize()); }, ITERS * 5), "");
  report("D2. syncAll() — setDevice+deviceSync over all GPUs",
         timeUs([&](){ for (int d : DEV) { CK(cudaSetDevice(d)); CK(cudaDeviceSynchronize()); } },
                ITERS * 5), "Phase 6 calls this per ring step in V1");

  std::vector<cudaStream_t> st(NG);
  for (int i = 0; i < NG; ++i) { CK(cudaSetDevice(DEV[i])); CK(cudaStreamCreate(&st[i])); }
  report("D3. cudaStreamSynchronize over all GPUs (idle)",
         timeUs([&](){ for (int i = 0; i < NG; ++i) { CK(cudaSetDevice(DEV[i]));
                        CK(cudaStreamSynchronize(st[i])); } }, ITERS * 5),
         "the pattern the NCCL body uses");

  // --- E: kernel launch ------------------------------------------------------
  CK(cudaSetDevice(DEV[0]));
  report("E1. empty kernel launch x1 (async, no sync)",
         timeUs([&](){ emptyKernel<<<1,32>>>(); }, ITERS * 20), "");
  report("E2. empty kernel launch on all GPUs, then syncAll",
         timeUs([&](){ for (int i=0;i<NG;++i){ CK(cudaSetDevice(DEV[i])); emptyKernel<<<1,32,0,st[i]>>>(); }
                       for (int i=0;i<NG;++i){ CK(cudaSetDevice(DEV[i])); CK(cudaStreamSynchronize(st[i])); } },
                ITERS * 2), "");

  // --- F: events -------------------------------------------------------------
  std::vector<cudaEvent_t> ev(NG);
  for (int i = 0; i < NG; ++i) { CK(cudaSetDevice(DEV[i])); CK(cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming)); }
  report("F1. event record + cross-device stream wait, all GPUs",
         timeUs([&](){ for (int i=0;i<NG;++i){ CK(cudaSetDevice(DEV[i])); CK(cudaEventRecord(ev[i], st[i])); }
                       for (int i=0;i<NG;++i){ CK(cudaSetDevice(DEV[i]));
                         CK(cudaStreamWaitEvent(st[i], ev[(i+1)%NG], 0)); } }, ITERS * 5), "");

  // --- G: one representative peer copy ---------------------------------------
  const size_t CH = (16u << 20) / sizeof(float) / NG;   // a 16 MiB message's chunk
  std::vector<float*> buf(NG), stage(NG);
  for (int i = 0; i < NG; ++i) {
    CK(cudaSetDevice(DEV[i]));
    CK(cudaMalloc(&buf[i], CH * sizeof(float)));
    CK(cudaMalloc(&stage[i], CH * sizeof(float)));
    CK(cudaMemset(buf[i], 0, CH * sizeof(float)));
  }
  report("G1. one cudaMemcpyPeer, 16MiB/N chunk, blocking",
         timeUs([&](){ CK(cudaMemcpyPeer(stage[1], DEV[1], buf[0], DEV[0], CH*sizeof(float))); }, 50), "");
  report("G2. one cudaMemcpyPeerAsync + streamSync, same size",
         timeUs([&](){ CK(cudaSetDevice(DEV[0]));
                       CK(cudaMemcpyPeerAsync(stage[1], DEV[1], buf[0], DEV[0], CH*sizeof(float), st[0]));
                       CK(cudaStreamSynchronize(st[0])); }, 50), "");
  report("G3. reduce kernel over the same chunk + sync",
         timeUs([&](){ CK(cudaSetDevice(DEV[1]));
                       addInto<<<256,256,0,st[1]>>>(buf[1], stage[1], CH);
                       CK(cudaStreamSynchronize(st[1])); }, 50), "");

  // --- H/I: NCCL through the Phase 6 body pattern -----------------------------
  std::vector<ncclComm_t> comm(NG);
  NK(ncclCommInitAll(comm.data(), NG, DEV.data()));

  for (size_t bytes : {(size_t)1024, (size_t)(16u<<20)}) {
    size_t elems = bytes / sizeof(float);
    std::vector<float*> nb(NG);
    for (int i = 0; i < NG; ++i) { CK(cudaSetDevice(DEV[i]));
      CK(cudaMalloc(&nb[i], elems*sizeof(float))); CK(cudaMemset(nb[i], 0, elems*sizeof(float))); }

    char lbl[128];
    // exactly the Phase 6 body
    std::snprintf(lbl, sizeof lbl, "H1. NCCL AllReduce %zuB — Phase 6 body verbatim", bytes);
    report(lbl, timeUs([&](){
      NK(ncclGroupStart());
      for (int i=0;i<NG;++i){ CK(cudaSetDevice(DEV[i]));
        NK(ncclAllReduce(nb[i], nb[i], elems, ncclFloat, ncclSum, comm[i], st[i])); }
      NK(ncclGroupEnd());
      for (int i=0;i<NG;++i){ CK(cudaSetDevice(DEV[i])); CK(cudaStreamSynchronize(st[i])); }
    }, 50), "8x setDevice + 4x streamSync per iter");

    // same collective, minimal host work: no per-rank setDevice, one sync pass
    std::snprintf(lbl, sizeof lbl, "H2. NCCL AllReduce %zuB — no per-rank setDevice", bytes);
    report(lbl, timeUs([&](){
      NK(ncclGroupStart());
      for (int i=0;i<NG;++i)
        NK(ncclAllReduce(nb[i], nb[i], elems, ncclFloat, ncclSum, comm[i], st[i]));
      NK(ncclGroupEnd());
      for (int i=0;i<NG;++i){ CK(cudaSetDevice(DEV[i])); CK(cudaStreamSynchronize(st[i])); }
    }, 50), "isolates the setDevice cost");

    // amortised: enqueue K collectives, then synchronise once (what nccl-tests does)
    const int K = 20;
    std::snprintf(lbl, sizeof lbl, "H3. NCCL AllReduce %zuB — %d enqueued, 1 sync", bytes, K);
    report(lbl, timeUs([&](){
      for (int k=0;k<K;++k){
        NK(ncclGroupStart());
        for (int i=0;i<NG;++i)
          NK(ncclAllReduce(nb[i], nb[i], elems, ncclFloat, ncclSum, comm[i], st[i]));
        NK(ncclGroupEnd());
      }
      for (int i=0;i<NG;++i){ CK(cudaSetDevice(DEV[i])); CK(cudaStreamSynchronize(st[i])); }
    }, 20) / K, "per collective; the nccl-tests pattern");

    for (int i = 0; i < NG; ++i) { CK(cudaSetDevice(DEV[i])); cudaFree(nb[i]); }
  }

  for (int i = 0; i < NG; ++i) ncclCommDestroy(comm[i]);
  for (int i = 0; i < NG; ++i) { CK(cudaSetDevice(DEV[i])); cudaFree(buf[i]); cudaFree(stage[i]);
                                 cudaStreamDestroy(st[i]); cudaEventDestroy(ev[i]); }
  std::printf("\n# H1 vs H2 isolates cudaSetDevice; H1 vs H3 isolates per-iteration synchronisation.\n");
  return 0;
}
