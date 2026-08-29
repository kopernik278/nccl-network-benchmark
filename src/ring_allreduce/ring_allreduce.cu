// ring_allreduce.cu — a simplified Ring AllReduce written from scratch.
//
// PURPOSE
//   Teach the algorithm and expose its costs, not to replace NCCL. NCCL is
//   linked here only as a correctness oracle and a performance reference; it is
//   never used to implement the custom ring.
//
// THE ALGORITHM
//   AllReduce(SUM) over N ranks is built from two ring phases over a tensor
//   split into N equal chunks:
//
//     1. ReduceScatter — N-1 steps. Afterwards rank r owns ONE chunk that holds
//        the full sum across all ranks.
//     2. AllGather     — N-1 steps. Afterwards every rank owns every chunk.
//
//   Ring order is explicit: ring[i] sends to ring[(i+1) % N] and receives from
//   ring[(i-1+N) % N]. Nothing about the schedule is hidden in a library.
//
//   Chunk schedule, for rank index r (position in the ring) and step s:
//
//     ReduceScatter  send = (r - s     + N) % N     recv = (r - s - 1 + N) % N
//     AllGather      send = (r - s + 1 + N) % N     recv = (r - s     + N) % N
//
//   Worked example, N = 4, rank 0:
//     RS  s=0: send c0, recv c3 (reduce into c3)
//         s=1: send c3, recv c2
//         s=2: send c2, recv c1     -> rank 0 now owns the final c1 = (r+1)%N
//     AG  s=0: send c1, recv c0
//         s=1: send c0, recv c3
//         s=2: send c3, recv c2     -> rank 0 holds c0..c3, all final
//
//   Invariant after ReduceScatter: rank r owns the finished chunk (r + 1) % N.
//   Every step moves exactly M/N bytes per rank, so per rank each phase moves
//   (N-1)/N * M and the whole AllReduce moves 2(N-1)/N * M — the same factor
//   used to convert algorithmic to bus bandwidth in earlier phases.
//
// VERSIONS
//   v1  naive      — blocking copies, a device sync at every boundary
//   v2  async      — per-device streams + events, double-buffered staging
//   v3  pipelined  — v2 plus subchunking so copy and reduce overlap
//
// Build: see Makefile.

#include <cuda_runtime.h>
#include <nccl.h>

// NVTX ranges for Nsight Systems. Compiled in only with -DUSE_NVTX=1 so the
// timed binary and the profiled binary differ by exactly one flag, and the
// numbers in the results file are never taken from an instrumented run.
#if defined(USE_NVTX)
#include <nvtx3/nvToolsExt.h>
#define NVTX_PUSH(name) nvtxRangePushA(name)
#define NVTX_POP() nvtxRangePop()
struct NvtxScope {
  explicit NvtxScope(const char* n) { nvtxRangePushA(n); }
  ~NvtxScope() { nvtxRangePop(); }
};
#define NVTX_SCOPE(name) NvtxScope nvtx_scope_##__LINE__(name)
#else
#define NVTX_PUSH(name) ((void)0)
#define NVTX_POP() ((void)0)
#define NVTX_SCOPE(name) ((void)0)
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// error handling
// ---------------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t err_ = (expr);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",                   \
                   cudaGetErrorString(err_), __FILE__, __LINE__, #expr);       \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define NCCL_CHECK(expr)                                                       \
  do {                                                                         \
    ncclResult_t res_ = (expr);                                                \
    if (res_ != ncclSuccess) {                                                 \
      std::fprintf(stderr, "NCCL error %s at %s:%d -> %s\n",                   \
                   ncclGetErrorString(res_), __FILE__, __LINE__, #expr);       \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

// dst[i] += src[i]. Grid-stride so one launch covers any length.
__global__ void addInto(float* dst, const float* src, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) dst[i] += src[i];
}

// Deterministic input so the exact AllReduce result is independently known.
//   value(rank, j) = (rank + 1) * ((j % 8) + 1)
// Summing over ranks 0..N-1 gives ((j % 8) + 1) * N(N+1)/2 — a small integer,
// exactly representable in fp32, so the oracle has no rounding slack.
__global__ void fillDeterministic(float* buf, size_t n, int rank) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) buf[i] = (float)((rank + 1) * ((int)(i % 8) + 1));
}

static inline void launchAdd(float* dst, const float* src, size_t n,
                             cudaStream_t s) {
  int threads = 256;
  int blocks = (int)std::min<size_t>((n + threads - 1) / threads, 1024);
  if (blocks < 1) blocks = 1;
  addInto<<<blocks, threads, 0, s>>>(dst, src, n);
}

// ---------------------------------------------------------------------------
// ring context
// ---------------------------------------------------------------------------
struct Ring {
  int n = 0;                       // ranks
  std::vector<int> dev;            // dev[i] is the CUDA device at ring position i
  size_t elems = 0;                // elements in the full tensor
  size_t chunk = 0;                // elements per chunk = elems / n
  std::vector<float*> buf;         // full tensor, one per rank
  // One staging buffer PER REDUCESCATTER STEP, not a parity pair.
  //
  // Parity double-buffering was the first design and it is WRONG: it separates
  // adjacent steps, but steps s-1 and s+1 share a parity and therefore share a
  // buffer. Nothing in the event graph orders rank r's copy at step s+1 against
  // rank next(r)'s reduce at step s-1 — the dependency chain runs backwards
  // around the ring and never reaches forward — so the copy can overwrite
  // staging that is still being read. It showed up exactly as a race should:
  // v1 correct, v2/v3 wrong, and the mismatch count varying run to run for the
  // same input. Giving every step its own buffer removes the hazard
  // structurally instead of paying for another synchronisation edge.
  std::vector<std::vector<float*>> stage;  // [step][rank], one chunk each
  std::vector<cudaStream_t> comm;  // copy stream per rank
  std::vector<cudaStream_t> calc;  // compute stream per rank
  // One event per (rank, step). Reusing a single event per rank would alias
  // across steps: rank 0 recording its step-s copy before rank 1 has waited on
  // the step-(s-1) copy makes rank 1 wait on the wrong (later) event. That is
  // still correct but silently over-synchronises, which would show up as a
  // performance result rather than a bug.
  std::vector<std::vector<cudaEvent_t>> copyDone;  // [rank][step], 2N slots
  std::vector<std::vector<cudaEvent_t>> calcDone;

  int next(int i) const { return (i + 1) % n; }
  int prev(int i) const { return (i - 1 + n) % n; }
  float* chunkPtr(int rank, int idx) const { return buf[rank] + (size_t)idx * chunk; }
};

// Bytes actually moved by the last run, accumulated per rank. Compared against
// the 2(N-1)/N * M prediction rather than assumed to match it.
static std::vector<size_t> g_bytesMoved;
static void resetByteCounter(int n) { g_bytesMoved.assign(n, 0); }

// ---------------------------------------------------------------------------
// setup
// ---------------------------------------------------------------------------

// Peer-access matrix. can[i][j] means device i can read device j's memory
// directly. A ring only needs its own edges, not the full matrix.
static std::vector<std::vector<int>> peerMatrix(const std::vector<int>& devs) {
  size_t n = devs.size();
  std::vector<std::vector<int>> can(n, std::vector<int>(n, 0));
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j)
      if (i != j) CUDA_CHECK(cudaDeviceCanAccessPeer(&can[i][j], devs[i], devs[j]));
  return can;
}

// cudaDeviceCanAccessPeer answers "is peer access permitted", NOT "does a peer
// copy actually deliver the bytes". On at least one RunPod L4 host the query
// returns 1 for every pair, yet an enabled peer copy silently produces NaN --
// and enabling peer access breaks a plain cudaMemcpy D2D on the same pair too.
// NCCL survives this because it validates P2P itself and falls back; a naive
// implementation that trusts the capability bit produces wrong answers with no
// error returned anywhere.
//
// So: functionally test the edge with a known pattern before believing it.
static bool p2pFunctional(int src, int dst) {
  const int n = 256;
  const size_t bytes = n * sizeof(float);
  float *s = nullptr, *d = nullptr;
  std::vector<float> host(n, -1.0f), back(n, -1.0f);
  for (int i = 0; i < n; ++i) host[i] = 1.0f + (float)i;

  if (cudaSetDevice(src) != cudaSuccess) return false;
  if (cudaMalloc(&s, bytes) != cudaSuccess) return false;
  if (cudaMemcpy(s, host.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(s); return false;
  }
  if (cudaSetDevice(dst) != cudaSuccess) { cudaFree(s); return false; }
  if (cudaMalloc(&d, bytes) != cudaSuccess) { cudaFree(s); return false; }
  cudaMemset(d, 0, bytes);

  cudaError_t pe = cudaDeviceEnablePeerAccess(src, 0);
  bool enabled = (pe == cudaSuccess || pe == cudaErrorPeerAccessAlreadyEnabled);
  cudaGetLastError();

  bool ok = false;
  if (enabled && cudaMemcpyPeer(d, dst, s, src, bytes) == cudaSuccess &&
      cudaDeviceSynchronize() == cudaSuccess &&
      cudaMemcpy(back.data(), d, bytes, cudaMemcpyDeviceToHost) == cudaSuccess) {
    ok = true;
    for (int i = 0; i < n; ++i)
      if (back[i] != host[i]) { ok = false; break; }
  }

  // Leave the machine as we found it: an enabled-but-broken peer mapping also
  // corrupts ordinary device-to-device copies on the same pair.
  if (enabled) { cudaSetDevice(dst); cudaDeviceDisablePeerAccess(src); }
  cudaGetLastError();
  cudaSetDevice(src); cudaFree(s);
  cudaSetDevice(dst); cudaFree(d);
  cudaGetLastError();
  return ok;
}

// Find a ring permutation whose every edge supports direct peer access. The
// identity order is tried first; otherwise permutations of the tail (rank 0 is
// fixed, since a ring is rotation-invariant). Returns empty if none exists —
// the caller must then say so rather than fall back silently.
static std::vector<int> findP2PRing(const std::vector<int>& devs,
                                    const std::vector<std::vector<int>>& can) {
  size_t n = devs.size();
  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  auto ringOk = [&](const std::vector<int>& order) {
    for (size_t i = 0; i < n; ++i) {
      int a = order[i], b = order[(i + 1) % n];
      if (!can[a][b]) return false;                       // capability
      if (!p2pFunctional(devs[a], devs[b])) return false; // and it actually works
    }
    return true;
  };
  if (ringOk(idx)) {
    std::vector<int> r(n);
    for (size_t i = 0; i < n; ++i) r[i] = devs[idx[i]];
    return r;
  }
  std::vector<int> tail(idx.begin() + 1, idx.end());
  std::sort(tail.begin(), tail.end());
  do {
    std::vector<int> order{idx[0]};
    order.insert(order.end(), tail.begin(), tail.end());
    if (ringOk(order)) {
      std::vector<int> r(n);
      for (size_t i = 0; i < n; ++i) r[i] = devs[order[i]];
      return r;
    }
  } while (std::next_permutation(tail.begin(), tail.end()));
  return {};
}

static void ringInit(Ring& R, const std::vector<int>& devOrder, size_t elems) {
  R.n = (int)devOrder.size();
  R.dev = devOrder;
  R.elems = elems;
  R.chunk = elems / R.n;
  R.buf.resize(R.n);
  const int nsteps = std::max(1, R.n - 1);
  R.stage.assign(nsteps, std::vector<float*>(R.n, nullptr));
  R.comm.resize(R.n);
  R.calc.resize(R.n);
  const int slots = 2 * R.n + 2;
  R.copyDone.assign(R.n, std::vector<cudaEvent_t>(slots));
  R.calcDone.assign(R.n, std::vector<cudaEvent_t>(slots));
  for (int i = 0; i < R.n; ++i) {
    CUDA_CHECK(cudaSetDevice(R.dev[i]));
    CUDA_CHECK(cudaMalloc(&R.buf[i], elems * sizeof(float)));
    for (int st = 0; st < nsteps; ++st)
      CUDA_CHECK(cudaMalloc(&R.stage[st][i], R.chunk * sizeof(float)));
    CUDA_CHECK(cudaStreamCreate(&R.comm[i]));
    CUDA_CHECK(cudaStreamCreate(&R.calc[i]));
    for (int k = 0; k < slots; ++k) {
      CUDA_CHECK(cudaEventCreateWithFlags(&R.copyDone[i][k], cudaEventDisableTiming));
      CUDA_CHECK(cudaEventCreateWithFlags(&R.calcDone[i][k], cudaEventDisableTiming));
    }
  }
}

static void ringFree(Ring& R) {
  for (int i = 0; i < R.n; ++i) {
    CUDA_CHECK(cudaSetDevice(R.dev[i]));
    cudaFree(R.buf[i]);
    for (size_t st = 0; st < R.stage.size(); ++st) cudaFree(R.stage[st][i]);
    cudaStreamDestroy(R.comm[i]);
    cudaStreamDestroy(R.calc[i]);
    for (size_t k = 0; k < R.copyDone[i].size(); ++k) {
      cudaEventDestroy(R.copyDone[i][k]);
      cudaEventDestroy(R.calcDone[i][k]);
    }
  }
}

static void ringReset(Ring& R) {
  for (int i = 0; i < R.n; ++i) {
    CUDA_CHECK(cudaSetDevice(R.dev[i]));
    fillDeterministic<<<256, 256>>>(R.buf[i], R.elems, i);
  }
  for (int i = 0; i < R.n; ++i) {
    CUDA_CHECK(cudaSetDevice(R.dev[i]));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

static void syncAll(const Ring& R) {
  for (int i = 0; i < R.n; ++i) {
    CUDA_CHECK(cudaSetDevice(R.dev[i]));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

// ---------------------------------------------------------------------------
// V1 — naive reference
//
// Deliberately simple: blocking peer copies and a full device barrier at every
// phase boundary. Every step is: all ranks copy, everyone waits, all ranks
// reduce, everyone waits. The barriers are what make it easy to reason about,
// and they are exactly what v2 removes.
// ---------------------------------------------------------------------------
static void ringAllReduceV1(Ring& R) {
  const size_t bytes = R.chunk * sizeof(float);
  NVTX_SCOPE("v1-allreduce");

  // --- ReduceScatter: N-1 steps -------------------------------------------
  NVTX_PUSH("v1-reducescatter");
  for (int s = 0; s < R.n - 1; ++s) {
    for (int r = 0; r < R.n; ++r) {
      int sendIdx = (r - s + R.n) % R.n;
      int nx = R.next(r);
      CUDA_CHECK(cudaMemcpyPeer(R.stage[0][nx], R.dev[nx],
                                R.chunkPtr(r, sendIdx), R.dev[r], bytes));
      g_bytesMoved[r] += bytes;
    }
    NVTX_PUSH("v1-sync"); syncAll(R); NVTX_POP();
    NVTX_PUSH("v1-reduce");
    for (int r = 0; r < R.n; ++r) {
      // The chunk this rank just received is the one its predecessor sent.
      int recvIdx = (r - s - 1 + R.n) % R.n;
      CUDA_CHECK(cudaSetDevice(R.dev[r]));
      launchAdd(R.chunkPtr(r, recvIdx), R.stage[0][r], R.chunk, 0);
    }
    NVTX_POP();
    NVTX_PUSH("v1-sync"); syncAll(R); NVTX_POP();
  }
  NVTX_POP();

  // --- AllGather: N-1 steps ------------------------------------------------
  // No reduction and no staging: the finished chunk is written straight into
  // the destination's slot, because sender and receiver agree on the index.
  NVTX_PUSH("v1-allgather");
  for (int s = 0; s < R.n - 1; ++s) {
    for (int r = 0; r < R.n; ++r) {
      int sendIdx = (r - s + 1 + R.n) % R.n;
      int nx = R.next(r);
      CUDA_CHECK(cudaMemcpyPeer(R.chunkPtr(nx, sendIdx), R.dev[nx],
                                R.chunkPtr(r, sendIdx), R.dev[r], bytes));
      g_bytesMoved[r] += bytes;
    }
    NVTX_PUSH("v1-sync"); syncAll(R); NVTX_POP();
  }
  NVTX_POP();
}

// ---------------------------------------------------------------------------
// V2 — asynchronous
//
// Dependencies that actually exist, and how each is expressed:
//
//   (a) rank r's copy at step s must follow rank r's reduce at step s-1,
//       because it sends the chunk that reduce just finished.
//       -> same rank, ordered by making the copy stream wait on calcDone[r].
//
//   (b) rank r+1's reduce at step s must follow rank r's copy at step s,
//       because it consumes what that copy wrote.
//       -> cross-device RAW dependency, expressed with copyDone[r].
//
//   (c) rank r's copy must not overwrite staging that rank next(r)'s reduce is
//       still reading (a WAR hazard).
//       -> removed structurally by giving every ReduceScatter step its own
//          staging buffer. Parity double-buffering is NOT enough: steps s-1 and
//          s+1 share a parity, and no event orders them, so that version
//          produced nondeterministic corruption while v1 stayed correct.
//
// What is NOT assumed: that using async APIs produces overlap on its own. The
// only overlap this version can express is between different ranks' copies and
// reduces; within a rank, step s+1's copy still waits on step s's reduce.
// ---------------------------------------------------------------------------
static void ringAllReduceV2(Ring& R) {
  const size_t bytes = R.chunk * sizeof(float);
  NVTX_SCOPE("v2-allreduce");

  NVTX_PUSH("v2-reducescatter");
  for (int s = 0; s < R.n - 1; ++s) {
    const int cur = s;                       // one staging buffer per step
    for (int r = 0; r < R.n; ++r) {
      int sendIdx = (r - s + R.n) % R.n;
      int nx = R.next(r);
      CUDA_CHECK(cudaSetDevice(R.dev[r]));
      if (s > 0)
        CUDA_CHECK(cudaStreamWaitEvent(R.comm[r], R.calcDone[r][s - 1], 0));    // (a)
      CUDA_CHECK(cudaMemcpyPeerAsync(R.stage[cur][nx], R.dev[nx],
                                     R.chunkPtr(r, sendIdx), R.dev[r], bytes,
                                     R.comm[r]));
      CUDA_CHECK(cudaEventRecord(R.copyDone[r][s], R.comm[r]));
      g_bytesMoved[r] += bytes;
    }
    for (int r = 0; r < R.n; ++r) {
      int recvIdx = (r - s - 1 + R.n) % R.n;
      int pv = R.prev(r);
      CUDA_CHECK(cudaSetDevice(R.dev[r]));
      CUDA_CHECK(cudaStreamWaitEvent(R.calc[r], R.copyDone[pv][s], 0));         // (b)
      launchAdd(R.chunkPtr(r, recvIdx), R.stage[cur][r], R.chunk, R.calc[r]);
      CUDA_CHECK(cudaEventRecord(R.calcDone[r][s], R.calc[r]));
    }
  }

  // AllGather writes straight into the peer's slot; the only ordering needed is
  // that a rank forwards a chunk after it has actually received it. Event slots
  // continue past the ReduceScatter steps so no index is reused.
  NVTX_POP();
  NVTX_PUSH("v2-allgather");
  const int agBase = R.n;
  for (int s = 0; s < R.n - 1; ++s) {
    for (int r = 0; r < R.n; ++r) {
      int sendIdx = (r - s + 1 + R.n) % R.n;
      int nx = R.next(r);
      CUDA_CHECK(cudaSetDevice(R.dev[r]));
      if (s == 0)
        CUDA_CHECK(cudaStreamWaitEvent(R.comm[r], R.calcDone[r][R.n - 2], 0));
      else
        CUDA_CHECK(cudaStreamWaitEvent(R.comm[r], R.copyDone[R.prev(r)][agBase + s - 1], 0));
      CUDA_CHECK(cudaMemcpyPeerAsync(R.chunkPtr(nx, sendIdx), R.dev[nx],
                                     R.chunkPtr(r, sendIdx), R.dev[r], bytes,
                                     R.comm[r]));
      CUDA_CHECK(cudaEventRecord(R.copyDone[r][agBase + s], R.comm[r]));
      g_bytesMoved[r] += bytes;
    }
  }
  NVTX_POP();
  NVTX_PUSH("v2-final-sync"); syncAll(R); NVTX_POP();
}

// ---------------------------------------------------------------------------
// V3 — pipelined
//
// v2 still moves a whole chunk before any of it is reduced. v3 splits each
// chunk into `sub` subchunks so the receiver can start reducing subchunk k
// while subchunk k+1 is still in flight.
//
// The tradeoff this exposes:
//   sub too small -> one copy and one kernel launch per subchunk, so launch
//                    and event overhead dominates
//   sub too large -> little overlap; degenerates towards v2
// ---------------------------------------------------------------------------
static void ringAllReduceV3(Ring& R, int sub) {
  if (sub < 1) sub = 1;
  const size_t base = R.chunk / sub;
  if (base == 0) { ringAllReduceV2(R); return; }

  std::vector<cudaEvent_t> subCopy(R.n);
  for (int r = 0; r < R.n; ++r) {
    CUDA_CHECK(cudaSetDevice(R.dev[r]));
    CUDA_CHECK(cudaEventCreateWithFlags(&subCopy[r], cudaEventDisableTiming));
  }

  NVTX_SCOPE("v3-allreduce");
  NVTX_PUSH("v3-reducescatter");
  for (int s = 0; s < R.n - 1; ++s) {
    const int cur = s;                       // one staging buffer per step
    for (int k = 0; k < sub; ++k) {
      size_t off = (size_t)k * base;
      size_t len = (k == sub - 1) ? (R.chunk - off) : base;
      for (int r = 0; r < R.n; ++r) {
        int sendIdx = (r - s + R.n) % R.n;
        int nx = R.next(r);
        CUDA_CHECK(cudaSetDevice(R.dev[r]));
        if (s > 0 && k == 0)
          CUDA_CHECK(cudaStreamWaitEvent(R.comm[r], R.calcDone[r][s - 1], 0));
        CUDA_CHECK(cudaMemcpyPeerAsync(R.stage[cur][nx] + off, R.dev[nx],
                                       R.chunkPtr(r, sendIdx) + off, R.dev[r],
                                       len * sizeof(float), R.comm[r]));
        CUDA_CHECK(cudaEventRecord(subCopy[r], R.comm[r]));
        g_bytesMoved[r] += len * sizeof(float);
      }
      for (int r = 0; r < R.n; ++r) {
        int recvIdx = (r - s - 1 + R.n) % R.n;
        int pv = R.prev(r);
        CUDA_CHECK(cudaSetDevice(R.dev[r]));
        CUDA_CHECK(cudaStreamWaitEvent(R.calc[r], subCopy[pv], 0));
        launchAdd(R.chunkPtr(r, recvIdx) + off, R.stage[cur][r] + off, len,
                  R.calc[r]);
        if (k == sub - 1) CUDA_CHECK(cudaEventRecord(R.calcDone[r][s], R.calc[r]));
      }
    }
  }

  NVTX_POP();
  NVTX_PUSH("v3-allgather");
  for (int s = 0; s < R.n - 1; ++s) {
    for (int r = 0; r < R.n; ++r) {
      int sendIdx = (r - s + 1 + R.n) % R.n;
      int nx = R.next(r);
      CUDA_CHECK(cudaSetDevice(R.dev[r]));
      if (s == 0) CUDA_CHECK(cudaStreamWaitEvent(R.comm[r], R.calcDone[r][R.n - 2], 0));
      else        CUDA_CHECK(cudaStreamWaitEvent(R.comm[r], R.copyDone[R.prev(r)][R.n + s - 1], 0));
      CUDA_CHECK(cudaMemcpyPeerAsync(R.chunkPtr(nx, sendIdx), R.dev[nx],
                                     R.chunkPtr(r, sendIdx), R.dev[r],
                                     R.chunk * sizeof(float), R.comm[r]));
      CUDA_CHECK(cudaEventRecord(R.copyDone[r][R.n + s], R.comm[r]));
      g_bytesMoved[r] += R.chunk * sizeof(float);
    }
  }
  NVTX_POP();
  NVTX_PUSH("v3-final-sync"); syncAll(R); NVTX_POP();
  for (int r = 0; r < R.n; ++r) {
    CUDA_CHECK(cudaSetDevice(R.dev[r]));
    cudaEventDestroy(subCopy[r]);
  }
}

// ---------------------------------------------------------------------------
// correctness oracle
// ---------------------------------------------------------------------------
struct Verdict {
  double maxAbsErr = 0.0;
  long long mismatches = 0;
  bool ok() const { return mismatches == 0; }
};

static Verdict verify(const Ring& R) {
  Verdict v;
  const int n = R.n;
  // expected[j] = ((j % 8) + 1) * sum_{r=1..n} r
  const double rankSum = (double)n * (n + 1) / 2.0;
  std::vector<float> host(R.elems);
  for (int r = 0; r < n; ++r) {
    CUDA_CHECK(cudaSetDevice(R.dev[r]));
    CUDA_CHECK(cudaMemcpy(host.data(), R.buf[r], R.elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (size_t j = 0; j < R.elems; ++j) {
      double want = ((double)(j % 8) + 1.0) * rankSum;
      double err = std::abs((double)host[j] - want);
      if (err > v.maxAbsErr) v.maxAbsErr = err;
      if (err > 1e-3) ++v.mismatches;
    }
  }
  return v;
}

// ---------------------------------------------------------------------------
// timing
//
// Host wall clock bracketed by a full device barrier on both sides, identical
// for every implementation including NCCL. Device-side event timing would be
// per-device and would not capture the collective's true end-to-end cost.
// ---------------------------------------------------------------------------
template <typename F>
static double timeUs(Ring& R, F&& body, int warmup, int iters) {
  NVTX_PUSH("warmup");
  for (int i = 0; i < warmup; ++i) body();
  syncAll(R);
  NVTX_POP();
  NVTX_PUSH("timed-region");
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) body();
  syncAll(R);
  NVTX_POP();
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  int ngpu = 4, warmup = 5, iters = 20;
  std::vector<size_t> sizesBytes = {1 << 10, 1 << 15, 1 << 20, 1 << 24, 1 << 27};
  std::vector<int> subchunks = {2, 4, 8, 16};
  bool allowHostStaged = false;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value after %s\n", argv[i]); std::exit(2); }
      return std::string(argv[++i]);
    };
    if (a == "-g") ngpu = std::stoi(next());
    else if (a == "-w") warmup = std::stoi(next());
    else if (a == "-n") iters = std::stoi(next());
    else if (a == "--allow-host-staged") allowHostStaged = true;
    else if (a == "-h") {
      std::printf("usage: %s [-g ngpus] [-w warmup] [-n iters] [--allow-host-staged]\n", argv[0]);
      return 0;
    }
  }

  int avail = 0;
  CUDA_CHECK(cudaGetDeviceCount(&avail));
  if (avail < ngpu) {
    std::fprintf(stderr, "requested %d GPUs but only %d visible\n", ngpu, avail);
    return 1;
  }
  std::vector<int> devs(ngpu);
  std::iota(devs.begin(), devs.end(), 0);

  // --- topology report (printed before anything is measured) ---------------
  std::printf("# ring-allreduce custom implementation\n");
  for (int d : devs) {
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, d));
    std::printf("# device %d: %s  sm_%d%d  pci=%04x:%02x:%02x\n", d, p.name,
                p.major, p.minor, p.pciDomainID, p.pciBusID, p.pciDeviceID);
  }
  auto can = peerMatrix(devs);
  std::printf("# peer-access matrix (row can access column):\n");
  for (int i = 0; i < ngpu; ++i) {
    std::printf("#   dev%d:", devs[i]);
    for (int j = 0; j < ngpu; ++j)
      std::printf(" %s", i == j ? "-" : (can[i][j] ? "yes" : "NO "));
    std::printf("\n");
  }

  std::printf("# functional peer test on the identity ring (capability bit is not enough):\n");
  for (int i = 0; i < ngpu; ++i) {
    int a = i, b = (i + 1) % ngpu;
    bool cap = can[a][b] != 0;
    bool works = cap && p2pFunctional(devs[a], devs[b]);
    std::printf("#   dev%d -> dev%d : canAccessPeer=%s  transfers_correct=%s\n",
                devs[a], devs[b], cap ? "yes" : "no", works ? "yes" : "NO");
  }

  std::vector<int> ring = findP2PRing(devs, can);
  const char* transport = "p2p-direct";
  if (ring.empty()) {
    // Never pretend a host-staged path is P2P. Either shrink the ring or say so.
    std::printf("# NO ring permutation of %d GPUs has WORKING direct peer access\n", ngpu);
    if (!allowHostStaged) {
      std::printf("# refusing to run: pass --allow-host-staged to measure the\n"
                  "# host-staged transport explicitly, or use fewer GPUs.\n");
      return 2;
    }
    ring = devs;
    transport = "host-staged";
    std::printf("# running in EXPLICIT host-staged mode (NOT direct P2P)\n");
  }
  std::printf("# ring order:");
  for (int d : ring) std::printf(" %d", d);
  std::printf("  transport=%s\n", transport);

  if (std::string(transport) == "p2p-direct") {
    for (int i = 0; i < ngpu; ++i) {
      CUDA_CHECK(cudaSetDevice(ring[i]));
      cudaError_t e = cudaDeviceEnablePeerAccess(ring[(i + 1) % ngpu], 0);
      if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(e);
      cudaGetLastError();
    }
  }

  if (ngpu > 16) {
    std::fprintf(stderr, "this harness supports at most 16 ranks\n");
    return 1;
  }
  ncclComm_t comms[16];
  NCCL_CHECK(ncclCommInitAll(comms, ngpu, ring.data()));

  std::printf("impl,subchunks,transport,ranks,size_bytes,elements,warmup,iters,"
              "latency_us,algbw_gbps,busbw_gbps,bytes_moved_per_rank,"
              "bytes_expected_per_rank,max_abs_err,mismatches\n");

  for (size_t bytes : sizesBytes) {
    size_t elems = bytes / sizeof(float);
    if (elems % (size_t)ngpu != 0) elems -= elems % ngpu;   // chunk divisibility
    if (elems == 0) continue;
    size_t realBytes = elems * sizeof(float);

    Ring R;
    NVTX_PUSH("alloc"); ringInit(R, ring, elems); NVTX_POP();

    // theoretical per-rank movement for the whole AllReduce: 2(N-1)/N * M
    const double expectBytes = 2.0 * (ngpu - 1) / ngpu * (double)realBytes;

    struct Case { std::string name; int sub; };
    std::vector<Case> cases = {{"v1-naive", 0}, {"v2-async", 0}};
    for (int s : subchunks) cases.push_back({"v3-pipelined", s});
    cases.push_back({"nccl-reference", 0});

    for (const Case& c : cases) {
      auto run = [&]() {
        if (c.name == "v1-naive") ringAllReduceV1(R);
        else if (c.name == "v2-async") ringAllReduceV2(R);
        else if (c.name == "v3-pipelined") ringAllReduceV3(R, c.sub);
        else {
          NVTX_SCOPE("nccl-allreduce");
          NCCL_CHECK(ncclGroupStart());
          for (int r = 0; r < ngpu; ++r) {
            CUDA_CHECK(cudaSetDevice(R.dev[r]));
            NCCL_CHECK(ncclAllReduce(R.buf[r], R.buf[r], elems, ncclFloat,
                                     ncclSum, comms[r], R.calc[r]));
          }
          NCCL_CHECK(ncclGroupEnd());
          for (int r = 0; r < ngpu; ++r) {
            CUDA_CHECK(cudaSetDevice(R.dev[r]));
            CUDA_CHECK(cudaStreamSynchronize(R.calc[r]));
          }
        }
      };

      // correctness first, on a freshly initialised buffer, always
      NVTX_PUSH("reset"); ringReset(R); NVTX_POP();
      resetByteCounter(ngpu);
      NVTX_PUSH("correctness-run"); run(); NVTX_POP();
      syncAll(R);
      NVTX_PUSH("validate");
      Verdict v = verify(R);
      NVTX_POP();
      size_t moved = g_bytesMoved[0];

      double us = -1.0, algbw = 0.0, busbw = 0.0;
      if (v.ok()) {
        // Timed iterations deliberately do NOT re-initialise the buffer.
        // Repeating AllReduce in place makes the values grow, but the work is
        // identical every iteration — same transfer sizes, same kernel shape,
        // and float addition is constant time regardless of magnitude. This is
        // what nccl-tests does, and it keeps the measurement free of the
        // re-initialisation cost that a per-iteration reset would fold in.
        us = timeUs(R, run, warmup, iters);
        {
          algbw = (double)realBytes / (us * 1e-6) / 1e9;
          busbw = algbw * 2.0 * (ngpu - 1) / ngpu;
        }
      }

      std::printf("%s,%d,%s,%d,%zu,%zu,%d,%d,%.3f,%.4f,%.4f,%zu,%.0f,%.6f,%lld\n",
                  c.name.c_str(), c.sub, transport, ngpu, realBytes, elems,
                  warmup, iters, us, algbw, busbw,
                  (c.name == "nccl-reference" ? (size_t)0 : moved), expectBytes,
                  v.maxAbsErr, v.mismatches);
      std::fflush(stdout);
    }
    ringFree(R);
  }

  for (int r = 0; r < ngpu; ++r) ncclCommDestroy(comms[r]);
  return 0;
}
