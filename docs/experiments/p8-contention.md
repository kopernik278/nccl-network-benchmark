# Phase 8 — Communication / Compute Resource Contention

Status: **completed**
Experiment IDs:
`p8-interference-matrix-20260829T0522Z-22a53e1` ·
`p8-intensity-sweep-20260829T0522Z-22a53e1` ·
`p8-collapse-recreate-20260829T0530Z-22a53e1`
Date (UTC): 2026-08-29 · Repo commit: `22a53e1`
Builds on: [Phase 7B](p7b-overlap.md) · [Phase 7A](p7a-harness-validation.md)

> Phase 7B measured *that* concurrent communication and computation interfere.
> Phase 8 asks *which resource* they are fighting over. The method is to vary
> the compute workload's arithmetic intensity while holding message size,
> transport, rank count and target duration fixed, and to see which workload
> class the interference tracks.
>
> Sections are labelled **OBSERVATION** (measured), **INTERPRETATION**
> (reasoning from the measurement), **CONCLUSION** and **LIMITATION**. The
> distinction is deliberate: the headline result of this phase is a *negative*
> one, and negative results are easy to overstate.

---

## 1. Setup and transport

| Field | Value |
|-------|-------|
| Pods | `8k78uif3jx90kc` (matrix + sweep), `1urcvdqtp3p4yw` (collapse) — RunPod SECURE, EU-SE-1 |
| GPU | NVIDIA A40 × 4, sm_86, driver 580.159.03, CUDA 12.8 |
| CPU | Intel Xeon Gold 6342 @ 2.80 GHz, 96 threads, 2 NUMA nodes |
| NCCL | 2.25.1+cuda12.8 (`/usr/include/nccl.h`) |
| Compiler | gcc 13.3.0, nvcc 12.8, built `SM=86` |
| **Transport** | **`SHM/direct` on every ring link** — proven from `NCCL_DEBUG=INFO` |
| Network | none — single node, 4 GPUs, one process |
| **Price / cost** | **$1.76/hour** × (≈ 38 min + ≈ 18 min) ≈ **$1.65** — see §11 |

Topology is *not* uniform on this host, unlike the all-`SYS` L4 host of Phase 7B:

```
        GPU0  GPU1  GPU2  GPU3   CPU Affinity     NUMA
GPU0     X    SYS   SYS   SYS    0-23,48-71       0
GPU1    SYS    X    PIX   PXB    24-47,72-95      1
GPU2    SYS   PIX    X    PXB    24-47,72-95      1
GPU3    SYS   PXB   PXB    X     24-47,72-95      1
```

### 1.1 The P2P transport hangs on this host

**OBSERVATION.** With NCCL left to choose transports, `cudaDeviceCanAccessPeer`
returns *yes* for every pair and NCCL builds a mixed ring:

```
Channel 00 : 0[0] -> 1[1] via SHM/direct/direct
Channel 00/0 : 1[1] -> 2[2] via P2P/direct pointer
Channel 00/0 : 2[2] -> 3[3] via P2P/direct pointer
Channel 00 : 3[3] -> 0[0] via SHM/direct/direct
Connected all rings, use ring PXN 0 GDR 1
```

Initialisation completes, and then the first `ncclAllReduce` never returns. Three
runs (80 s, 90 s and 180 s of wall time, warmup 0 and warmup 5, 1 MiB and 16 MiB)
produced **zero** data rows with all four GPUs pinned at 100 % utilisation. The
identical command with `NCCL_P2P_DISABLE=1` completed in under a minute.

**INTERPRETATION.** This is the Phase 6 lesson again in a new form: the P2P
*capability bit* is not evidence that the P2P *path works*. In Phase 6 the
capability bit was set and peer copies silently returned NaN; here the
capability bit is set and the collective deadlocks. A `PIX`/`PXB` link across a
PCIe switch under a container tenant is exactly where ACS / IOMMU configuration
decides whether peer traffic actually flows, and the tenant does not control it.

**Consequence for this phase.** Every Phase 8 number was measured with
`NCCL_P2P_DISABLE=1`, so all eight ring links are `SHM/direct`
(`nccl_debug_transport_128m.txt`). That is the same transport class Phase 7B
measured, which makes the two phases comparable — but it also means nothing here
describes NVLink or a working direct-P2P system.

---

## 2. Design — three workload classes, one calibrated duration

The Phase 7B benchmark (`src/overlap/overlap_bench.cu`) was extended rather than
rewritten: the compute side became a switch over three classes, and the
calibration loop now measures each class separately so that all three can be
driven to the *same* target duration.

| Class | Kernel | Design arithmetic intensity | Measured standalone (per rep, per GPU) |
|-------|--------|-----------------------------|-----------------------------------------|
| `compute-gemm` | cuBLAS SGEMM 512×512 | ~85 flop/byte | **51.4 µs** → 5.22 TFLOP/s |
| `memory-triad` | `d = a + s·b` over 8 M floats | ~0.17 flop/byte | **218.9 µs** → 96 MiB at **460 GB/s** |
| `mixed` | 32 FMAs per element over 8 M floats | ~8 flop/byte | **165.4 µs** → 64 MiB at **406 GB/s** |

Repetition counts are chosen per class so that every cell targets the same
compute duration; the `reps` column below shows the result. Interference is
measured with CUDA events *on each stream*, not inferred from wall clock:

```
compute_slowdown = compute_during_overlap_us / t_compute_us
comm_slowdown    = comm_during_overlap_us    / t_comm_us
```

**LIMITATION (design).** The classes differ in two ways at once: arithmetic
intensity *and* kernel granularity. A 51 µs GEMM must be launched ~4× as often
as a 219 µs triad to fill the same window. Section 6 shows this confound is
load-bearing, and Phase 8 cannot separate the two variables.

**LIMITATION (workload realism).** `compute-gemm` reaches 5.22 TFLOP/s, roughly
14 % of the A40's ~37 TFLOP/s FP32 peak. A 512³ SGEMM is small; it is
launch- and occupancy-limited, not ALU-saturated. It stands for *"a stream of
many short compute kernels"*, which is a realistic training-step shape, but it
is not a proxy for a large well-tuned GEMM.

Arithmetic intensity is **by construction**, not counter-verified — see §7.

---

## 3. Interference matrix — 3 classes × 3 message sizes

Compute calibrated to the collective's own duration (ratio 1.0), 3 repeats,
5 warmup + 3 timed iterations, medians shown.

| message | workload | reps | t_comp (µs) | t_comm (µs) | **compute ×** | **comm ×** | ovl. eff | exposed (µs) |
|---------|----------|-----:|------------:|------------:|--------------:|-----------:|---------:|-------------:|
| 1 MiB | compute-gemm | 6 | 246.2 | 341.1 | **1.40** | 1.03 | 0.894 | 122.1 |
| 1 MiB | memory-triad | 2 | 429.6 | 338.8 | 1.17 | **1.68** | 0.533 | 161.5 |
| 1 MiB | mixed | 2 | 323.9 | 342.2 | 1.19 | **1.39** | 0.530 | 168.7 |
| 16 MiB | compute-gemm | 93 | 3440.8 | 4742.8 | **1.52** | 1.01 | 0.861 | 1762.0 |
| 16 MiB | memory-triad | 22 | 4628.2 | 4760.2 | 1.17 | 1.09 | 0.841 | 873.1 |
| 16 MiB | mixed | 29 | 4548.4 | 4784.7 | 1.19 | 0.98 | 0.867 | 843.1 |
| 128 MiB | compute-gemm | 724 | 26610.0 | 37403.3 | **1.61** | 1.01 | 0.801 | 16081.6 |
| 128 MiB | memory-triad | 171 | 35885.1 | 37301.1 | 1.17 | 1.08 | 0.874 | 5983.1 |
| 128 MiB | mixed | 227 | 35520.4 | 37298.4 | 1.20 | 1.00 | 0.855 | 6962.3 |

**OBSERVATION 1.** The compute stream is slowed in every one of the nine cells.
The **compute-heavy** class is slowed the most (1.40×–1.61×); the
**memory-heavy** class is slowed the least (1.17× everywhere).

**OBSERVATION 2.** The collective is almost never slowed at ratio 1.0
(1.00×–1.09×), with one exception: at **1 MiB**, the memory-heavy class slows it
**1.68×** and the mixed class **1.39×**, while the compute-heavy class leaves it
at 1.03×. The 1 MiB row is the only place in the matrix where the memory-bound
workload is the aggressor.

**INTERPRETATION.** If a single shared resource explained all of this, one
workload class would dominate both columns. It does not: the class that hurts
*compute* the most is the one that hurts *communication* the least, and the
ranking flips between 1 MiB and 128 MiB. At least two distinct mechanisms are
present, and which one dominates depends on the message-size regime.

---

## 4. Compute-intensity sweep at 128 MiB

Compute duration set to 25 / 50 / 75 / 100 % of the collective's duration, then
extended to 150 % and 200 % to reach the Phase 7B collapse point (§5).

| ratio | class | reps | t_comp (µs) | **compute ×** | **comm ×** | ovl. eff |
|------:|-------|-----:|------------:|--------------:|-----------:|---------:|
| 0.25 | compute-gemm | 181 | 6650.1 | **2.00** | 1.01 | 0.952 |
| 0.50 | compute-gemm | 362 | 13311.2 | **2.00** | 1.02 | 0.955 |
| 0.75 | compute-gemm | 543 | 19911.5 | **1.96** | 1.03 | 0.939 |
| 1.00 | compute-gemm | 724 | 26610.0 | 1.61 | 1.01 | 0.801 |
| 1.50 | compute-gemm | 1054 | 38546.0 | 1.31 | 1.00 | 0.668 |
| 2.00 | compute-gemm | 1406 | 60390.7 | 1.26 | **1.42** | **0.573** |
| 0.25 | memory-triad | 43 | 9039.7 | 1.26 | 1.03 | 0.903 |
| 0.50 | memory-triad | 86 | 18045.6 | 1.21 | 1.05 | 0.887 |
| 0.75 | memory-triad | 129 | 27071.6 | 1.22 | 1.09 | 0.883 |
| 1.00 | memory-triad | 171 | 35885.1 | 1.17 | 1.08 | 0.874 |
| 1.50 | memory-triad | 248 | 52039.2 | 1.10 | 1.08 | 0.854 |
| 2.00 | memory-triad | 331 | 70099.6 | 1.07 | 1.07 | 0.855 |
| 0.25 | mixed | 57 | 8927.0 | 1.29 | 1.00 | 0.995 |
| 0.50 | mixed | 114 | 17834.6 | 1.30 | 1.01 | 0.970 |
| 0.75 | mixed | 170 | 26598.5 | 1.28 | 1.01 | 0.977 |
| 1.00 | mixed | 227 | 35520.4 | 1.20 | 1.00 | 0.855 |
| 1.50 | mixed | 327 | 51759.4 | 1.11 | 0.97 | 0.841 |
| 2.00 | mixed | 436 | 69094.1 | 1.08 | 0.97 | 0.850 |

![Phase 8 contention](../../results/plots/p8-contention.png)

**OBSERVATION 3.** For the compute-heavy class the slowdown is **2.00×** while
the compute fits entirely inside the collective's window (ratios ≤ 0.75), and
then *falls* as the ratio grows. The memory-heavy and mixed classes never
exceed 1.30×.

**INTERPRETATION.** The falling slowdown at high ratios is an averaging effect,
not relief: once compute outlasts the collective, its tail runs undisturbed and
dilutes the ratio. Taking the ratio-1.0 compute-gemm cell — 26.6 ms of work
finishing at 43.8 ms while the collective occupies ~37.4 ms — the contended
portion implies a slowdown of ≈ 1.8×, consistent with the 2.00× measured where
the whole kernel stream is inside the window. The interference factor is
roughly *constant while the collective is resident*, which is what a fixed
fractional loss of execution resources looks like, and not what a
bandwidth-saturation effect (which would worsen as the compute stream's demand
grows) looks like.

---

## 5. Recreating the Phase 7B collapse

Phase 7B's one bad case was **128 MiB at ratio 2.0**: efficiency 0.386, and the
collective itself slowed **1.83×**. Ratio 2.0 lies outside the 25–100 % sweep,
so it needed its own run on a second pod of the same GPU model and topology,
with the repo pinned to the same commit.

| class | ratio | t_comp (µs) | t_comm (µs) | compute × | **comm ×** | **ovl. eff** |
|-------|------:|------------:|------------:|----------:|-----------:|-------------:|
| compute-gemm | 1.50 | 38546.0 | 36411.7 | 1.31 | 1.00 | 0.668 |
| **compute-gemm** | **2.00** | 60390.7 | 36572.8 | 1.26 | **1.42** | **0.573** |
| memory-triad | 2.00 | 70099.6 | 36498.1 | 1.07 | 1.07 | 0.855 |
| mixed | 2.00 | 69094.1 | 36562.8 | 1.08 | 0.97 | 0.850 |

**OBSERVATION 4.** The collapse **reproduces**, on different silicon (A40 vs
L4) and a different pod, and it is **specific to the compute-heavy class**.
At the same message size, ratio, transport and rank count, the memory-heavy and
mixed classes sit at efficiency 0.85 with the collective essentially undisturbed.
Only `compute-gemm` degrades — efficiency 0.952 → 0.573 across the sweep — and
only there does the collective slow down (1.42×).

**INTERPRETATION.** Phase 7B could not tell whether the collapse was a property
of the message size, the ratio, or the workload. It is the workload: the same
128 MiB collective at the same ratio is fine when the concurrent work is a
smaller number of longer, memory-bound kernels.

**Reconciliation with Phase 7B.** Phase 7B's finding — "at 128 MiB / ratio 2.0
overlap collapses and communication is slowed" — is **STILL VALID** and now
**generalised**: it is a compute-workload-class effect, not a message-size
effect. The magnitudes differ (eff 0.386 vs 0.573, comm 1.83× vs 1.42×), which
is expected across L4 and A40.

---

## 6. Which resource? What the evidence rules out

### 6.1 DRAM bandwidth — ruled out for the compute-heavy case

**OBSERVATION 5 (arithmetic).** A ring AllReduce moves `2(N-1)/N` of the buffer
in and out of each rank. Counting a device-memory read *and* write for every such
byte — a deliberately generous upper bound — gives:

| message | t_comm | ≤ device-memory traffic | share of the 460 GB/s measured achievable |
|---------|-------:|------------------------:|------------------------------------------:|
| 1 MiB | 341 µs | ≤ 9.22 GB/s | **2.00 %** |
| 16 MiB | 4.74 ms | ≤ 10.61 GB/s | **2.31 %** |
| 128 MiB | 37.7 ms | ≤ 10.68 GB/s | **2.32 %** |

The SHM transport stages through host memory over PCIe, so the collective's
device-memory demand is bounded by its PCIe rate — about 2 % of what the A40's
GDDR6 can deliver.

**OBSERVATION 6 (the decisive contrast).** At 128 MiB / ratio 2.0, the workload
that consumes ~460 GB/s of DRAM slows the collective **1.07×**, while the
workload that consumes very little DRAM slows it **1.42×**.

**CONCLUSION.** **DRAM bandwidth does not explain the interference measured
here, and specifically does not explain the collapse.** The resource ranking is
the opposite of the DRAM-contention prediction, and the collective's own
device-memory footprint is ~2 % of capacity. This phase therefore does *not*
claim "DRAM is the bottleneck" — the evidence points the other way.

### 6.2 Kernel-level distributions

Nsight Systems, NVTX-instrumented binary, `NCCL_P2P_DISABLE=1`. In each trace
the two workload classes that are not being exercised appear only in the
calibration phase, which gives an uncontended baseline *inside the same trace*.

| trace | kernel | instances | med (µs) | avg (µs) | max (µs) | sd (µs) |
|-------|--------|----------:|---------:|---------:|---------:|--------:|
| *(uncontended baseline)* | `ampere_sgemm_64x32_sliced1x4_nn` | 100 | 23.46 | 23.51 | 24.67 | **0.23** |
| 128 MiB r1.0 compute | `ampere_sgemm_64x32_sliced1x4_nn` | 37 000 | **23.17** | 29.17 | **319.94** | **15.00** |
| 128 MiB r2.0 compute | `ampere_sgemm_64x32_sliced1x4_nn` | 71 620 | **23.14** | 25.83 | **358.95** | 10.75 |
| *(uncontended baseline)* | `streamTriad` | 100 | 201.76 | 201.83 | 205.67 | **0.98** |
| 16 MiB r1.0 memory | `streamTriad` | 1 360 | **202.58** | 211.46 | 382.88 | **22.50** |
| 128 MiB r2.0 memory | `streamTriad` | 19 840 | **202.24** | 207.01 | 495.43 | 17.79 |

**OBSERVATION 7.** Under contention the **median** kernel duration is
unchanged — 23.17 µs vs a 23.46 µs baseline for SGEMM, 202.6 µs vs 201.8 µs for
triad — while the standard deviation grows by 45–65× and the maximum by 13×
(SGEMM) and 2.4× (triad).

**INTERPRETATION.** A shared-bandwidth mechanism would slow *every* kernel:
the median would move. It does not. Instead a minority of launches is delayed
enormously. The aggregate stream inflation measured in §3–§4 is therefore built
out of *episodic stalls*, not a uniform reduction in execution rate.

**OBSERVATION 8.** The collective is one persistent kernel,
`ncclDevKernel_AllReduce_Sum_f32_RING_LL`, resident for the whole collective —
median 36.7 ms at 128 MiB. The `RING_LL` name also confirms NCCL chose the LL
protocol here, consistent with Phase 5.

### 6.3 What the evidence supports

**CONCLUSION.** The measurements are consistent with contention for **GPU
execution resources — SM residency and the work scheduler** — between a
long-lived NCCL kernel and a stream of concurrent compute kernels:

1. the collective's memory demand is ~2 % of DRAM capacity, so it cannot be
   starving anything of bandwidth (§6.1);
2. the workload with the *lowest* memory demand is the one that disturbs the
   collective the most (§5);
3. per-kernel medians are unaffected while tails explode — the signature of
   delayed scheduling, not of slower execution (§6.2);
4. the compute-side slowdown is ~constant while the collective is resident and
   dilutes only when compute outlasts it (§4).

**LIMITATION (the confound, stated plainly).** Kernel granularity and
arithmetic intensity are not independent in this design. At 128 MiB / ratio 2.0
the compute-heavy case issues **1 406** kernels per step against **331** for the
memory-heavy case. "Many short kernels contend for scheduling slots with a
persistent collective kernel" and "low-arithmetic-intensity work is immune"
predict the same data here. Phase 8 shows the effect is *not* bandwidth; it
cannot rank granularity against intensity. Separating them needs one workload
class run at several kernel granularities with intensity held fixed — the
natural next experiment.

---

## 7. Nsight Compute — attempted once, not available

**OBSERVATION.** One `ncu` invocation was attempted, requesting
`dram__bytes.sum`, `l1tex__t_bytes.sum` and
`sm__throughput.avg.pct_of_peak_sustained_elapsed`:

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 3.
```

Hardware performance counters require `CAP_SYS_ADMIN` or the host-level
`NVreg_RestrictProfilingToAdminUsers=0` module option. A container tenant
controls neither. **The attempt was made once and not retried**, per the
failure policy. Raw output: `profile-ncu_attempt.txt`.

**Consequence.** No measured DRAM byte counts, no measured achieved occupancy,
no counter-verified arithmetic intensity. §6 is built on counter-free evidence:
an arithmetic upper bound on the collective's memory traffic, a workload-class
contrast, and kernel-duration *distributions* from Nsight Systems (which needs
only tracing, not counters). Every intensity figure in this report is labelled
**by construction**.

---

## 8. Answers to the phase questions

1. **Which resource explains Phase 7B's interference?** Not DRAM bandwidth —
   ruled out arithmetically (~2 % of capacity) and empirically (the
   highest-bandwidth workload is the *least* disruptive). The evidence supports
   contention for GPU execution resources between the persistent NCCL kernel and
   concurrent compute kernels.
2. **Does the workload class matter?** Decisively. Compute slowdown 1.40×–2.00×
   for `compute-gemm` against 1.07×–1.30× for `memory-triad`.
3. **Is the collective ever the victim?** Yes, in two places: small messages
   against a memory-bound workload (1 MiB, **1.68×**), and the collapse case
   (128 MiB / ratio 2.0 against many short GEMMs, **1.42×**).
4. **Does the Phase 7B collapse reproduce?** Yes, on different silicon, and it is
   a property of the compute workload class, not of the message size.
5. **Was SHM transport revalidated?** Yes — all eight ring links `SHM/direct`,
   captured in every run's `NCCL_DEBUG=INFO` output.
6. **Were hardware counters used?** No — unavailable, attempted once, recorded.

---

## 9. Implications for distributed training

- **Overlapping communication with many small kernels is the bad case.** The
  same collective that hides perfectly behind a few long kernels degrades
  sharply behind a long queue of short ones. Kernel fusion and larger fused
  GEMMs help the *communication* schedule, not only the compute schedule.
- **A gradient AllReduce is not free even when it looks hidden.** At ratio ≤ 0.75
  the end-to-end efficiency is 0.94–0.99 while the compute stream is running at
  half speed. Wall-clock overlap metrics alone hide this; per-stream event
  measurement exposes it.
- **Do not tune for DRAM here.** On a host where the collective travels over SHM
  or any PCIe-class path, its bandwidth footprint is negligible. Effort belongs
  in kernel granularity and launch scheduling.
- Everything above is measured on the **SHM** transport. On NVLink the
  collective is far faster and far more SM-hungry per unit time; the balance
  could shift, and this phase does not measure that.

---

## 10. Limitations

1. **Transport.** `SHM/direct` only. The P2P path hangs on this host and was
   disabled; NVLink was never present. No NVLink, RoCE or InfiniBand claim.
2. **Counters.** No Nsight Compute. Arithmetic intensity is by construction;
   DRAM traffic for the collective is an upper bound, not a measurement.
3. **Confound.** Kernel granularity and arithmetic intensity covary (§6.3).
4. **Workload realism.** The SGEMM runs at ~14 % of FP32 peak (§2).
5. **Scale.** 4 GPUs, 1 node, 1 process, `float`, AllReduce/sum only.
6. **Calibration variance.** The triad calibration measured 460 GB/s in most
   runs but 311 GB/s once (`smoke_gate.txt`), i.e. calibration is sensitive to
   what ran immediately before. Reported cells all come from runs whose
   calibration landed at ~460 GB/s.
7. **Repeat count.** 3 repeats × 3 timed iterations per cell. Spread within a
   cell is well under 1 %, but this is not a statistically deep sample.
8. **Two pods.** The collapse run is a second pod (§5). Same GPU model, same
   topology, same commit, same transport — but not the same machine instance.

---

## 11. Cleanup

Both pods were terminated after all GPU-dependent work was complete:
`8k78uif3jx90kc` and `1urcvdqtp3p4yw`. `list-pods` returns **0** items. Large
`.nsys-rep` and `.sqlite` traces were deleted on the pods and never fetched;
only their derived CSV summaries are in the repository, per the repo policy on
large profiling artifacts.

Cost: pod `8k78uif3jx90kc` ran from 04:48:40Z until just before 05:26:49Z (≈ 38 min)
and pod `1urcvdqtp3p4yw` from 05:26:49Z for ≈ 18 min, at $1.76/hour — **≈ $1.65**.
One 4-GPU pod ran at a time, so the configuration stayed inside the $3.00/hour
autonomous threshold throughout.

**Updated 2026-08-29 (during Phase 9):** the figure above was computed from pod
lifetime × the posted rate, because the billing API had not yet posted these
hours when this report was written. Billing has since posted **$1.23** —
`8k78uif3jx90kc` $1.03 and `1urcvdqtp3p4yw` $0.20 — so the lifetime-based
computation was **high by about 26 %**. The billed number is the correct one.

**Process note.** The collapse point was missed on the first pod because the
sweep stopped at ratio 1.0 while Phase 7B's collapse sat at ratio 2.0, so a
second pod was needed — the same "terminated before all planned GPU work was
finished" failure recorded in Phase 5. The check that would have caught it is
to re-read the *prior* phase's headline case before writing the run matrix, not
after.

---

## 12. Next

The experiment this phase now clearly demands: hold arithmetic intensity fixed
and vary **kernel granularity** alone — the same total GEMM work as 1 fused
kernel, 10 kernels, 100 kernels, 1000 kernels, overlapped with the same 128 MiB
collective. That separates the one confound §6.3 leaves standing, and it is a
single-node 4-GPU experiment on the cheapest suitable hardware.
