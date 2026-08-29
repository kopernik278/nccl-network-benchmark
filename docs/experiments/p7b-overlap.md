# Phase 7B — Communication / Compute Overlap

Status: **completed**
Experiment ID: `p7b-overlap-20260829T040506Z-ec065cb`
Date (UTC): 2026-08-29 · Repo commit: `ec065cb`
Builds on: [Phase 7A](p7a-harness-validation.md) · [Phase 6](p6-ring-allreduce.md)

> The goal is **not** to make NCCL faster. It is to measure how much
> communication hides behind compute and how much stays exposed. All numbers are
> measured; profiler runs are used for mechanism only, never as timing samples.

---

## 1. Setup and transport

| Field | Value |
|-------|-------|
| Pod | `tzoq8bjacpooab`, RunPod SECURE, EUR-IS-1 |
| GPU | NVIDIA L4 × 4, sm_89, driver 550.127.05, CUDA 12.8 |
| Topology | every GPU pair `SYS` |
| **Transport** | **`SHM/direct`** — NCCL reports `intraNodeP2pSupport 0` |
| NCCL | 2.25.1+cuda12.8 |
| **Price / cost** | **$1.96/hour** × ≈ 9 min ≈ **$0.30** |

Following Phases 6 and 7A, the capability bit was not trusted: NCCL's own
`Check P2P Type intraNodeP2pSupport 0` and `via SHM/direct/direct` are the
evidence. **Phase 7B therefore measures overlap on the SHM transport.** Nothing
here should be generalised to NVLink or a working direct-P2P system — on those
the communication side would be far faster and every ratio would shift.

---

## 2. Design

**Compute** is cuBLAS SGEMM (512×512) repeated on a dedicated stream —
representative of training compute, deterministic, and tunable. The benchmark
calibrates it at runtime (measured **20.9 µs per repetition**) and then solves
for the repetition count that hits each target duration. Allocation, handle
creation, communicator creation and calibration are all outside every timed
region, per the Phase 7A audit.

**Measured independently before any overlap run:**
`T_compute`, `T_comm`, and `T_seq` (compute → barrier → AllReduce).

**Overlapped** issues the AllReduce on a communication stream and the GEMMs on a
compute stream, then waits on both. Phase 7A established that *asynchronous
submission does not imply overlap*, so CUDA events bracket each stream's own
work to measure what each actually took, and Nsight settles whether the kernels
really ran concurrently.

```
overlap_efficiency     = (T_seq − T_overlap) / (T_seq − T_ideal),  T_ideal = max(T_compute, T_comm)
effective_exposed_comm = max(0, T_overlap − T_compute)
```

Both are **end-to-end** metrics. Neither proves what the collective's own kernel
did.

---

## 3. Controlled microbenchmark

Median of 3 repeats; run-to-run spread 0.0–3.0%.

| comm | ratio | T_compute | T_comm | T_seq | T_overlap | T_ideal | efficiency | exposed |
|-----:|------:|----------:|-------:|------:|----------:|--------:|-----------:|--------:|
| 1 MiB | 0.25 | 64.9 | 285.5 | 352.9 | 309.5 | 285.5 | 0.623 | 245.5 |
| 1 MiB | 0.5 | 119.3 | 281.9 | 406.2 | 307.4 | 281.9 | 0.814 | 187.8 |
| 1 MiB | 1.0 | 278.4 | 282.9 | 566.2 | 355.9 | 282.9 | 0.748 | 77.5 |
| 1 MiB | 2.0 | 488.5 | 284.9 | 776.2 | 536.4 | 488.5 | 0.836 | 47.2 |
| 16 MiB | 0.25 | 694.2 | 3540.8 | 4171.0 | 3529.2 | 3540.8 | 0.944 | 2825.1 |
| 16 MiB | 0.5 | 1383.2 | 3462.2 | 4823.8 | 3568.7 | 3462.2 | 0.928 | 2192.2 |
| 16 MiB | 1.0 | 2793.6 | 3498.7 | 6254.3 | 3700.1 | 3498.7 | 0.927 | 912.9 |
| 16 MiB | 2.0 | 5607.8 | 3521.5 | 8951.1 | 5804.8 | 5607.8 | 0.942 | 196.9 |
| 128 MiB | 0.25 | 5527.7 | 27653.1 | 33053.2 | 28078.2 | 27653.1 | 0.939 | 22550.5 |
| 128 MiB | 0.5 | 11069.8 | 27744.1 | 38787.7 | 28259.9 | 27744.1 | 0.953 | 17201.6 |
| 128 MiB | 1.0 | 22135.6 | 27775.7 | 49404.3 | 30440.6 | 27775.7 | 0.877 | 8298.0 |
| **128 MiB** | **2.0** | 47117.8 | 27895.0 | 75376.3 | **64469.8** | 47117.8 | **0.386** | 17332.0 |

**OBSERVATION.** Overlap is real and substantial. In 11 of 12 cases 62–95% of
the theoretical opportunity is realised. At 16 MiB / ratio 2.0 the exposed
communication falls from 3.5 ms standalone to **197 µs** — 94% of the collective
disappears behind compute.

**OBSERVATION.** One case collapses: **128 MiB at ratio 2.0, efficiency 0.386**,
and exposed communication *rises* to 17.3 ms from 8.3 ms at ratio 1.0 — adding
compute made the exposure worse, not better.

---

## 4. Interference — compute pays, communication mostly does not

CUDA events on each stream give each side's own duration during the overlapped
run, so contention is measured rather than inferred.

| comm | ratio | compute alone | compute under overlap | slowdown | comm alone | comm under overlap | slowdown |
|-----:|------:|--------------:|----------------------:|---------:|-----------:|-------------------:|---------:|
| 1 MiB | 0.25 | 64.9 | 126.5 | **1.95×** | 285.5 | 290.7 | 1.02× |
| 1 MiB | 1.0 | 278.4 | 331.2 | 1.19× | 282.9 | 291.4 | 1.03× |
| 16 MiB | 0.25 | 694.2 | 1450.0 | **2.09×** | 3540.8 | 3552.1 | 1.00× |
| 16 MiB | 1.0 | 2793.6 | 3732.2 | 1.34× | 3498.7 | 3714.8 | 1.06× |
| 128 MiB | 0.5 | 11069.8 | 16354.2 | 1.48× | 27744.1 | 28105.1 | 1.01× |
| **128 MiB** | **2.0** | 47117.8 | 64710.1 | **1.37×** | 27895.0 | 51007.2 | **1.83×** |

**OBSERVATION.** Compute is slowed in *every* case, by 1.03×–2.09×.
Communication is essentially unaffected (1.00×–1.06×) — **except** at
128 MiB / ratio 2.0, where it is slowed **1.83×**.

**INTERPRETATION.** Overlap is not free: the NCCL kernel occupies SMs and
competes with the GEMMs, and on this SHM transport it also drives host-memory
traffic that the GEMMs need. In most regimes communication wins the contention
outright and compute absorbs the cost — which is the right trade, since the
point is to hide communication. The 128 MiB / ratio 2.0 case is different:
there is enough of *both* that neither yields, both slow down, and the
end-to-end result is worse than either alone would predict.

**CONCLUSION.** "Overlap efficiency" and "free" are not the same thing. A step
can show 0.94 efficiency while its compute is running 1.5× slower than it would
alone.

---

## 5. Timeline evidence

Nsight Systems + NVTX, three cases. **Not used as timing samples.**

| case | NVTX `sequential` | NVTX `overlapped` |
|---|---:|---:|
| good overlap (16 MiB, ratio 1.0) | 26.76 ms | **15.61 ms** |
| poor overlap (128 MiB, ratio 2.0) | 292.53 ms | 176.58 ms |

**OBSERVATION — the kernels really do run concurrently.** In the good case the
NCCL kernel (`ncclDevKernel_AllReduce_Sum_f32_RING_LL`, 64 instances, avg
3.92 ms) and the cuBLAS GEMM (`cutlass_80_simt_sgemm_128x64_8x5_nn_align1`,
2020 instances, avg 26.5 µs) both occupy the window that the `overlapped` range
covers, and that range is 1.71× shorter than `sequential`. Concurrency is
observed, not assumed.

**OBSERVATION — contention is visible in the kernel distributions.** The GEMM's
median stays ~21.6 µs in all three traces, but its **maximum** is 397 µs (good),
**564 µs (poor)** and 418 µs (DDP). The NCCL kernel at 128 MiB ranges from
27.5 ms (min) to **44.7 ms (max)** — a 1.63× spread within one run.

**INTERPRETATION.** Individual kernels are being stretched when they collide,
which is the mechanism behind the aggregate slowdowns in §4. The timeline shows
*that* they interfere; it does not identify which resource (SM occupancy,
host-memory bandwidth, copy engines) is the binding constraint, and this report
does not claim one.

**DDP pipeline trace.** 400 NCCL AllReduce calls, kernel median 3.59 ms but
**maximum 53.8 ms** — a 15× tail. Buckets queue behind one another on the single
communication stream; the last bucket's collective is what the final
synchronisation waits for.

---

## 6. DDP-like bucket pipeline

Simulated backward: *N* stages, each producing one gradient bucket whose
AllReduce is launched asynchronously while the next stage's compute continues,
with one final synchronisation. Total gradient volume held at **128 MiB**.

| bucket | collectives | T_compute | T_comm | T_seq | **overlapped step** | efficiency | exposed | spread |
|-------:|------------:|----------:|-------:|------:|--------------------:|-----------:|--------:|-------:|
| 4 MiB | 32 | 15461 | 29690 | 44360 | **30510** | 0.944 | 15035 | 0.6% |
| 8 MiB | 16 | 15003 | 28814 | 43068 | **30114** | 0.909 | 15104 | 0.3% |
| 16 MiB | 8 | 15999 | 27719 | 42733 | **30203** | 0.835 | 14204 | 1.0% |
| 32 MiB | 4 | 16886 | 27705 | 43897 | 31776 | 0.749 | 14891 | 0.6% |
| 64 MiB | 2 | 18261 | 28008 | 45608 | **35099** | 0.596 | 16840 | 0.5% |

**OBSERVATION — both sides of the predicted tradeoff appear.**
Many small buckets cost more to communicate: 32 collectives take **29 690 µs**
against 27 705 µs for 4 collectives of the same total volume — **+7% from
per-collective overhead alone**. Few large buckets hide worse: efficiency falls
monotonically from 0.944 to 0.596.

**OBSERVATION — step time is NOT monotonic, and it disagrees with efficiency.**
4, 8 and 16 MiB land within **1.3%** of each other (30 510 / 30 114 / 30 203 µs)
while run-to-run spread is 0.3–1.0%. 32 MiB is +5% and 64 MiB is **+16%**.

**CONCLUSION.** There *is* an intermediate region that performs best, and it is
a **plateau, not a point**: on this system anywhere in **4–16 MiB** is
equivalent within noise, and the data does not support naming 8 MiB the optimum
even though it is nominally lowest. Above 16 MiB step time degrades clearly and
monotonically.

**INTERPRETATION.** Efficiency keeps improving as buckets shrink because the
denominator `T_seq − T_ideal` grows — smaller buckets make the *sequential*
baseline worse faster than they make the overlapped step better. **Step time is
the metric that matters; efficiency alone would have picked 4 MiB and been
wrong.**

---

## 7. Answers to the research questions

1. **How much AllReduce hides?** Up to **94%** (16 MiB, ratio 2.0: 3.5 ms
   standalone → 197 µs exposed). Typically 62–95% of the theoretical
   opportunity.
2. **How does overlap depend on the ratio?** Efficiency is high and fairly flat
   from 0.25 to 1.0. Exposed communication falls steadily as compute grows —
   that is the useful trend, not efficiency.
3. **When is communication fully hidden?** Never quite, here. The closest is
   16 MiB / ratio 2.0 at 197 µs exposed out of 3 522 µs.
4. **When does it stay exposed?** Whenever `T_comm > T_compute` — exposure is
   roughly `T_comm − T_compute` — and, more sharply, when both are large enough
   to contend (128 MiB / ratio 2.0).
5. **Does concurrency slow either side?** **Yes, measured.** Compute 1.03×–2.09×
   slower in every case; communication 1.00×–1.06× except 1.83× in the
   contended case.
6. **How does bucket size affect the pipeline?** Step time is flat over
   4–16 MiB and rises 16% by 64 MiB.
7. **Why can many small buckets be worse?** They are not worse here — but they
   cost **+7% total communication time** from per-collective overhead, which is
   what would make them worse once that overhead outgrew the earlier start.
8. **What does this imply for step time?** §8.

---

## 8. Implications for distributed training

Framed as interpretation of microbenchmarks, not measurements of a real trainer.

- **Overlap is worth a lot and is not free.** Hiding communication cost the
  compute stream 1.3×–2.1× in wall time here. A step-time model that assumes
  overlap is free will be optimistic.
- **Bucket size has a plateau, and tuning inside it is wasted effort.** The
  practical guidance from this data is "avoid buckets that are too large",
  not "find the exact optimum" — 4–16 MiB were indistinguishable.
- **Optimise the metric you care about.** Overlap efficiency improved
  monotonically as buckets shrank while step time did not. Efficiency is a
  diagnostic; step time is the objective.
- **The exposed tail is what remains.** In the DDP pipeline, 14–17 ms of the
  ~30 ms step is communication that no amount of scheduling hid, because the
  last bucket cannot start before the last gradient exists. Reducing that
  requires less data or a faster fabric, not better overlap.

---

## 9. Limitations

- **SHM transport only.** Direct P2P was functionally unavailable, as in
  Phases 6 and 7A. On NVLink the communication side would be far faster, every
  ratio would move, and the contention picture could differ qualitatively.
  **No result here generalises to a healthy direct-P2P system.**
- **The custom Phase 6 ring was not re-studied.** It was optional; it is
  host-staged, the task directs that conclusions not rest on it, and Phases 6
  and 7A already characterised its stream/event dependency behaviour. Spending
  GPU time to repeat that was judged poor value.
- **Compute is SGEMM, not a real backward pass** — no activation memory
  traffic, no optimiser, no framework overhead.
- **`overlap_efficiency` is end-to-end.** It does not prove what the collective
  itself did; §4 and §5 are what support the mechanism claims.
- Two efficiency values exceeded 1.0 in the raw data (1.19–1.23 at ratio 0.25),
  meaning the overlapped run beat `max(T_compute, T_comm)`. These are within
  measurement noise of a degenerate denominator and are reported as-is rather
  than trimmed; medians over 3 repeats bring them back below 1.
- **Which resource is contended is not identified** (§5). Nsight Compute was
  not used.
- Single node, 4 ranks, fp32.

---

## 10. Cleanup

Completeness was confirmed before deletion: the full 51-row matrix with 3
repeats, three NVTX traces with extracted statistics, and the transport probe.
`delete-pod` → 204; `list-pods` returned **0**. `.nsys-rep` traces were left on
the pod; only extracted CSV summaries were retained.

---

## 11. Next

The next phase has **not** been started.
