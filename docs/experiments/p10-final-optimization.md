# Phase 10 — Final Optimization and Synthesis

Status: **completed**
Experiment ID: `p10-final-ddp-20260830T0741Z-502811c`
Date (UTC): 2026-08-30 · Repo commit: `502811c`
Builds on: [Phase 9](p9-ddp-training.md) · [Phase 8](p8-contention.md) · [Phase 7B](p7b-overlap.md)

> This phase opens no new research direction. It takes the configuration Phase 9
> identified from measurement, runs it against its baseline and against a
> deliberately bad negative control on one host in one session, and reports what
> the change is actually worth.
>
> The headline is smaller than Phase 9 predicted, and that is the finding.

---

## 1. Setup, transport and the preflight that chose it

| Field | Value |
|-------|-------|
| Pod | `p3wii8aj59d11e`, RunPod SECURE, EU-SE-1 |
| GPU | NVIDIA A40 × 4, 46 068 MiB each, driver 550.127.05 |
| PyTorch / CUDA / NCCL | 2.8.0+cu128 / 12.8 / **2.27.3** · BF16 supported |
| **Transport** | **`P2P/CUMEM` + `SHM/direct` — NCCL's own default, functionally validated** |
| Network | none — single node, 4 GPUs, 4 processes (`torchrun`) |
| **Price / cost** | **$1.76/hour** × ≈ 32 min ≈ **$0.94** — see §12 |

```
        GPU0  GPU1  GPU2  GPU3   CPU Affinity     NUMA
GPU0     X    NODE  NODE  NODE   24-47,72-95      1
GPU1    NODE   X    PXB   PXB    24-47,72-95      1
GPU2    NODE  PXB    X    PXB    24-47,72-95      1
GPU3    NODE  PXB   PXB    X     24-47,72-95      1
```

### 1.1 The preflight earned its keep — in the opposite direction

`scripts/preflight_ddp.sh` runs the reliability chain this project learned the
hard way: **topology → capability → functional collective → correctness gate →
transport decision.** The gate is functional, not declarative: a real DDP step
under a timeout, accepted only if `param_sync_max_abs_diff == 0`.

On this host it **accepted NCCL's defaults**:

```
--- candidate: <NCCL default, P2P enabled> ---
  rc=0  transport: via P2P/CUMEM via SHM/direct/direct
  # CORRECTNESS loss_finite=True grads_finite=True param_sync_max_abs_diff=0.000e+00
  VERDICT: accepted (param_sync_max_abs_diff=0.000e+00)
chosen: NCCL defaults (P2P functional on this host)
```

**OBSERVATION.** On the three hosts of Phases 8 and 9 the identical gate caught a
deadlock and forced `NCCL_P2P_DISABLE=1`. Here the P2P path works.

**INTERPRETATION.** The right lesson from Phases 6/8/9 was never "P2P is broken
on A40" — it was "**do not trust the capability bit; test the path**". A
preflight that hard-coded the Phase 9 workaround would have silently thrown away
a working, faster transport on this machine. The check must be dynamic, and this
run is the case where the dynamic answer differed.

**LIMITATION.** Four hosts is not a survey. Nothing here says how often P2P is
healthy on this provider; it says the capability bit predicted neither outcome.

### 1.2 Comparability with Phase 9

Because the selected transport differs from Phase 9's, the same three capacities
were **also** run with `NCCL_P2P_DISABLE=1` on this host, as a like-for-like
anchor (§5). Every other variable — model, parameter count, sequence length,
per-GPU batch, BF16 autocast, FP32 master weights and gradients, AdamW, resident
synthetic batches, 4-GPU DDP, timing semantics — is unchanged from Phase 9.

---

## 2. Workload (unchanged from Phase 9)

| Field | Value |
|-------|-------|
| Model | compact GPT, 8 layers / 12 heads / d_model 768 / vocab 16 384 / seq 1024 |
| **Parameters** | **82 601 472** |
| **Gradient volume** | **330 405 888 B = 315.09 MiB** per step per rank (float32) |
| Precision | BF16 autocast, FP32 master weights and gradients |
| Batch per GPU / tokens per step | 16 / 65 536 (4 GPU) |
| Optimizer | AdamW (fused), lr 3e-4, β (0.9, 0.95), wd 0.1 |
| Peak memory (rank 0) | 7.53 GiB single-GPU, 7.84–8.14 GiB DDP, 8.45 GiB serialised — of 45 GiB |

Measurement: 12 warmup steps (DDP finishes rebuilding its buckets), then 30
measured steps, **3 independent `torchrun` launches** per configuration. CUDA
event pairs are buffered and read after the window closes, so no synchronisation
is injected into the region being measured. Profiled runs are never timing
samples.

---

## 3. Correctness gate

Every configuration passed before any timing was accepted:

| configuration | loss finite | grads finite | `param_sync_max_abs_diff` | optimizer steps |
|---|---|---|---|---|
| 4 / 25 / 64 MiB, both transports | True | True | **0.000e+00** | 45 per launch |

A non-zero `param_sync_max_abs_diff` — the invariant a corrupted transport
breaks — would have blocked the run. No NCCL hang occurred on the selected
transport; the one hang this project can still reproduce is in the preflight
logs of Phases 8 and 9, preserved there.

---

## 4. Final results

### 4.1 On the transport the preflight selected (P2P + SHM)

| configuration | collectives/step | **step (ms)** | spread | fwd | bwd | bwd (no-sync) | **sync cost** | opt | **tokens/s** | scaling eff. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single GPU (no comm) | — | 141.20 | 0.38 | 46.64 | 87.36 | — | — | 7.19 | 116 030 | — |
| **B. optimized, 4 MiB** | 26 | **151.62** | 0.15 | 46.58 | 97.82 | 87.98 | **9.81** | 7.23 | **432 126** | **93.1 %** |
| A. reference, 25 MiB | 10 | 152.35 | 0.81 | 46.51 | 98.56 | 87.84 | 10.72 | 7.19 | 430 181 | 92.7 % |
| C. negative control, 64 MiB | 4 | 157.12 | 0.19 | 46.48 | 103.46 | 88.15 | 15.38 | 7.13 | 417 116 | 89.9 % |
| non-overlapped control | 1 | 189.82 | 0.34 | 46.28 | 136.71 | — | — | 6.75 | 345 679 | 74.5 % |

### 4.2 Anchor — the same three capacities forced onto SHM

| configuration | **step (ms)** | spread | bwd | bwd (no-sync) | **sync cost** | tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| 4 MiB | **162.73** | 0.10 | 108.89 | 88.33 | **20.53** | 402 456 |
| 25 MiB | 164.25 | 0.14 | 110.78 | 88.16 | 22.57 | 398 899 |
| 64 MiB | 172.58 | 0.19 | 119.13 | 88.32 | 30.83 | 379 767 |

![Phase 10 final results](../../results/plots/p10-final.png)

### 4.3 Baseline → optimized

| metric | 25 MiB → 4 MiB (selected transport) | 25 MiB → 4 MiB (SHM anchor) |
|---|---|---|
| step time | 152.35 → **151.62 ms** (**−0.48 %**) | 164.25 → **162.73 ms** (**−0.93 %**) |
| throughput | 430 181 → **432 126 tok/s** (**+0.45 %**) | 398 899 → **402 456 tok/s** (+0.89 %) |
| **synchronisation penalty** | 10.72 → **9.81 ms** (**−8.5 %**) | 22.57 → **20.53 ms** (−9.0 %) |
| scaling efficiency | 92.7 % → **93.1 %** (+0.42 pp) | 85.9 % → **86.7 %** (+0.77 pp) |
| collectives per step | 10 → 26 (**2.6× more calls** — the cost side) | 10 → 26 |

### 4.4 Is the optimization resolvable?

Noise floor = the largest spread across independent launches of any capacity.

| | selected transport (floor 0.81 ms) | SHM anchor (floor 0.19 ms) |
|---|---|---|
| 4 MiB | reference | reference |
| 25 MiB | +0.73 ms (+0.48 %) — **within noise** | +1.52 ms (+0.93 %) — resolvable |
| 64 MiB | +5.50 ms (+3.62 %) — **resolvable** | +9.85 ms (+6.05 %) — **resolvable** |

**CONCLUSION.** On the fast transport the 4 MiB optimization is **not
statistically distinguishable** from the 25 MiB reference: the gap (0.73 ms) is
smaller than the run-to-run spread (0.81 ms, driven by the 25 MiB launches
themselves). On SHM it is resolvable but still under 1 %.

**The robust result of this phase is the negative control, not the
optimization.** Avoiding 64 MiB is worth 3.6 % on the fast transport and 6.1 %
on the slow one, and is resolvable by a wide margin on both. Moving from 25 MiB
to 4 MiB is worth at most ~1 % and may be worth nothing measurable.

---

## 5. Timeline evidence for the mechanism

Nsight Systems, NVTX, one steady-state step on rank 0, on the selected
transport. The window is bounded by the *next* backward on the same rank, so the
tail cannot absorb the following step's collectives.

| bucket | collectives | first AllReduce start | NCCL resident during backward | **exposed tail** | total NCCL kernel time / step |
|---|---:|---:|---:|---:|---:|
| **4 MiB** | 26 | +38.20 ms (34.8 %) | 54.56 ms (**49.8 %**) | **6.51 ms** | **61.1 ms** |
| 25 MiB | 10 | +37.81 ms (34.8 %) | 48.81 ms (44.9 %) | 9.42 ms | 58.2 ms |
| 64 MiB | 4 | +41.94 ms (38.8 %) | 34.21 ms (**31.6 %**) | **13.20 ms** | **47.4 ms** |

NCCL kernel instances across each trace: 1572 / 612 / 252 — exactly 26 / 10 / 4
per step per rank, matching DDP's own `rebuilt_bucket_sizes`. Collective count
is measured on both sides, not assumed. Median NCCL kernel duration rises
1.90 → 5.32 → 11.32 ms as buckets grow.

---

## 6. The optimization mechanism, tested

The causal chain Phase 9 proposed, checked link by link against §5:

| link | prediction | measured on this host | verdict |
|---|---|---|---|
| 1 | smaller buckets → gradients communicable earlier | first AllReduce at 34.8 % of backward for both 4 and 25 MiB; 38.8 % for 64 MiB | **partly** — see below |
| 2 | → more NCCL work executes during backward | residency 49.8 / 44.9 / 31.6 % for 4 / 25 / 64 MiB | **yes** |
| 3 | → exposed tail decreases | 6.51 / 9.42 / 13.20 ms | **yes** |
| 4 | → step time decreases | 151.62 / 152.35 / 157.12 ms | **yes, monotone** |

**OBSERVATION.** Link 1 behaves differently here than on SHM. In Phase 9's SHM
run the first AllReduce at 4 MiB began *before* backward (−5.1 ms) because the
previous step's collectives were still draining; at 25 MiB it began at 24 %. On
this faster transport the queue has drained by the time backward starts, so
**4 and 25 MiB begin communicating at the same point (34.8 %)** — the moment the
first gradients are ready, not a queueing artefact.

**INTERPRETATION.** With a fast collective, "start earlier" stops being the
active mechanism between 4 and 25 MiB; what still differs is *residency and
tail*. That is exactly why the 4 MiB win shrinks from 0.93 % (SHM) to 0.48 %
(P2P) and falls inside the noise. Only 64 MiB still delays the first collective,
and it is the only configuration that clearly loses.

**And the cost side, measured:** 4 MiB issues **2.6× more collectives** and
spends **13.7 ms more total NCCL kernel time per step** than 64 MiB (61.1 vs
47.4 ms). The optimization succeeds only because reclaiming 6.7 ms of tail
(13.20 → 6.51 ms) outweighs that extra aggregate collective time — and against
25 MiB the two nearly cancel.

---

## 7. The negative control is the important result

64 MiB produces **fewer, larger, individually more efficient collectives**:

- 4 collectives per step instead of 26;
- median NCCL kernel 11.32 ms instead of 1.90 ms;
- **the least total NCCL kernel time of any configuration** — 47.4 ms/step
  against 61.1 ms for 4 MiB, a 22 % reduction in aggregate collective work.

And it is **the slowest configuration**: 157.12 ms against 151.62 ms.

**CONCLUSION.** Isolated communication efficiency is not the optimization
objective. A configuration can win every collective-level metric and lose the
training step, because what matters is *when* the bytes move relative to
backward, not how efficiently they move. This is the clearest single
demonstration in the project of why Phases 1–5's collective benchmarks, on their
own, cannot tell you how to configure training.

---

## 8. Overlap benefit and scaling

| | step | tokens/s | scaling efficiency |
|---|---:|---:|---:|
| single GPU | 141.20 ms | 116 030 | — |
| 4-GPU DDP, 25 MiB reference | 152.35 ms | 430 181 | 92.7 % |
| **4-GPU DDP, 4 MiB optimized** | **151.62 ms** | **432 126** | **93.1 %** |
| 4-GPU, non-overlapped reduction | 189.82 ms | 345 679 | 74.5 % |

Per-GPU batch is held constant, so the 4-GPU global batch is **4× the
single-GPU batch** (65 536 vs 16 384 tokens per step); the scaling efficiency
above is `4-GPU tok/s ÷ (4 × single-GPU tok/s)`.

**Overlap is worth 38.20 ms/step (20.1 %)** against the non-overlapped control
on this transport — down from 30.4 % on Phase 9's SHM host, for the same reason
the bucket effect shrank: there is less communication to hide.

---

## 9. Final performance claim

> On 4 × NVIDIA A40 with a functionally validated `P2P/CUMEM` + `SHM/direct`
> transport, training a 82.6 M-parameter GPT under PyTorch DDP, reducing
> `bucket_cap_mb` from 25 to 4 **reduced the exposed synchronisation penalty by
> 8.5 % (10.72 → 9.81 ms)** and step time by **0.48 % (152.35 → 151.62 ms)**,
> improving throughput by **0.45 % (430 181 → 432 126 tokens/s)**. The step-time
> difference is **within run-to-run noise on this transport** and should not be
> presented as a reliable speedup. On the same host with P2P disabled the
> equivalent change is **−0.93 % step time**, resolvable but still under 1 %.
>
> The reliable result is the negative control: **`bucket_cap_mb = 64` costs
> 3.6 % of step time on this transport and 6.1 % with P2P disabled**, despite
> producing 22 % less total NCCL kernel work per step.

This is not a claim about NCCL bandwidth; no NCCL bandwidth was optimized. It
applies to **this model, this gradient volume, this topology, this transport and
this hardware**, and 4 MiB is not asserted to be universally optimal.

---

## 10. Cross-phase synthesis

| Phase | Observation | Lesson | Influence on the next phase |
|---|---|---|---|
| **1–2** | 2→4 GPU: small-message latency ×1.9 (a tree, not a ring); large messages keep only ~28 % of bus bandwidth across a host-bridge hop | Topology, not rank count, governs collective scaling — and NCCL's algorithm choice is visible in timing data | Motivated forcing algorithm and protocol explicitly |
| **5** | Ring/Tree × Simple/LL/LL128 characterised; the automatic choice tracks the measured crossovers | NCCL's defaults are good, but the crossover is measurable rather than mysterious | Gave a reference for reading `RING_LL` out of later traces |
| **6** | A hand-written ring: the peer-capability bit was set, yet peer copies returned NaN; V2/V3 pipelining beat V1 on paper and not in measurement | Capability bits are not functional validation; a faster design must be measured | Introduced the functional P2P test; left a "4.4 ms harness floor" claim that needed checking |
| **7A** | That floor was a property of the host (~27 µs actual); 91–93 % of the naive ring's time was synchronisation | Validate the measuring apparatus before trusting what it measures | **Invalidated two Phase 6 conclusions**; established the NVTX + Nsight workflow |
| **7B** | 62–95 % of the overlap opportunity realised; compute slowed 1.03–2.09×; one 128 MiB case collapsed | Overlap is real but not free | Posed the "which resource?" question |
| **8** | Compute-heavy work slowed 2.0×, memory-heavy only 1.2×; the collective's DRAM demand is ~2 % of capacity; medians flat, tails inflate | **DRAM bandwidth ruled out** — contention is for execution resources | Gave Phase 9 a signature to look for in real kernels |
| **9** | Real DDP: 50–83 % NCCL residency during backward; step time monotone in bucket capacity; **requested capacity ≠ collective size** | Microbenchmarks predicted the real collective within 7.8 %; 4 MiB best, 64 MiB 6.1 % worse | Selected 4 MiB as the Phase 10 candidate |
| **10** | 4 MiB wins by 0.48 % — **inside the noise** on a healthy P2P host; 64 MiB still loses by 3.6 %; the preflight **accepted** P2P here | The tail-reclaiming optimization shrinks as the interconnect improves; the cliff does not. Test the transport, never assume it | — |

---

## 11. Reliability synthesis — the preflight, as executed

```
  hardware / topology discovery      nvidia-smi -L, topo -m
            ↓
  software version capture           torch, CUDA, NCCL, BF16 support
            ↓
  capability discovery               cudaDeviceCanAccessPeer  ← recorded, NOT believed
            ↓
  functional collective validation   a real DDP step under a hard timeout
            ↓
  correctness gate                   loss finite ∧ grads finite ∧
                                     param_sync_max_abs_diff == 0
            ↓
  transport decision                 first candidate that PASSES; otherwise
                                     exit non-zero and benchmark nothing
            ↓
  benchmark  →  profiling  →  result validation (schema + correctness flags)
```

Across four hosts, the capability bit said *yes* every time. The functional gate
disagreed with it on three of them (a deadlock in Phases 8 and 9) and agreed on
the fourth (this one). **The capability query predicted neither outcome.** This
is an observation about the four cloud hosts tested here and is not generalised
to A40, to this provider, or to PCIe P2P at large.

---

## 12. Cost and cleanup

Two pods were created. The first, `n6lt2dke8xvvhf`, never received a public port
mapping and was terminated within about a minute without running anything; it is
recorded here rather than omitted. All work ran on `p3wii8aj59d11e` in one
session: preflight, single-GPU reference, three DDP capacities × 3 launches on
the selected transport, the same three on the SHM anchor, the non-overlapped
control, transport evidence for both, three Nsight traces, and the per-step
timeline extraction — before the pod was released.

`list-pods` returns **0** items. Large `.nsys-rep` and `.sqlite` files were
deleted on the pod and never fetched; only derived CSV summaries and timeline
JSON are in the repository.

Cost: ≈ 32 min at $1.76/hour ≈ **$0.94**, computed from pod lifetime × the
posted rate — the billing API had not posted this hour when the report was
written. One 4-GPU pod at a time, inside the $3.00/hour autonomous threshold
throughout.

---

## 13. Limitations and generalization boundary

1. **Transport.** Results are for a mixed `P2P/CUMEM` + `SHM/direct` ring, with
   an SHM-only anchor. **No NVLink, RoCE or InfiniBand result is claimed** and
   none was measured anywhere in this project.
2. **The optimization is within noise on the fast transport** (§4.4). The
   negative control is the robust finding.
3. **Model shape drives the bucket structure.** A single 50 MiB output
   projection is one indivisible bucket, so no capacity below ~48 MiB can touch
   about a third of this model's gradient traffic (Phase 9 §4.1). A model with a
   smaller vocabulary or tied embeddings would give the knob more room.
4. **Scale.** 4 GPUs, 1 node, 82.6 M parameters. Nothing here speaks to
   multi-node reduction or to large models.
5. **The `sync_cost` metric bundles the exposed tail with interference**, by
   construction (Phase 9 §2.1). The trace tail and `sync_cost` differ by 3–4 ms
   in the same direction, which is that interference term.
6. **Four hosts.** The P2P reliability observation is an observation about four
   machines, not a statistic.
7. **Individual bucket-ready points are not plotted** — only the first start
   offset was exported. The figure shows measured spans and draws no inferred
   markers.
8. **Kernel-granularity isolation remains deferred** (Phase 8 §6.3): granularity
   and arithmetic intensity are still confounded in the synthetic experiments.

---

## 14. Preserved negative results

None of these were removed or softened as the project progressed:

- **Phase 6** — the peer-capability bit was set while peer copies returned NaN;
  the V2/V3 pipelined ring showed no benefit over the naive V1.
- **Phase 7A** — the Phase 6 "~4.4 ms harness floor" was **invalidated**; two
  Phase 6 performance conclusions fell with it, and the Phase 6 report is
  preserved unedited with a pointer.
- **Phase 8** — Nsight Compute was unavailable (`ERR_NVGPUCTRPERM`), attempted
  once and not retried; DRAM bandwidth was **ruled out** rather than confirmed.
- **Phases 8–9** — NCCL's P2P transport deadlocked on three separate hosts.
- **Phase 9 / 10** — the large-bucket cliff, and Phase 10's own finding that the
  proposed optimization is within noise on a healthy interconnect.
- **Process** — Phases 5 and 8 released a pod before all GPU-dependent work was
  finished, and both reports say so.

---

## 15. Files

- `scripts/preflight_ddp.sh` — the reliability chain of §11, executable
- `scripts/run_final_ddp.sh` — the final benchmark driver
- `scripts/analyze_ddp.py`, `scripts/plot_final_ddp.py` — analysis and figure
- `src/ddp/{model.py,train_ddp.py}` — unchanged from Phase 9
- `results/raw/p10-final-ddp-20260830T0741Z-502811c/` — raw stdout for all 24
  launches, preflight log, env, topology, NCCL transport evidence for both
  transports, Nsight summaries, per-step timeline JSON, driver scripts
- `results/summary/p10-final-ddp-20260830T0741Z-502811c/` — 42 rows
- `results/plots/p10-final.png`

---

## 16. Next

Nothing in Phase 10 requires a follow-up on this hardware. The two open threads,
both recorded rather than pursued, are: repeating the bucket study on a genuine
NVLink system, where §6 predicts the optimization shrinks further while the
64 MiB cliff persists; and the deferred kernel-granularity isolation from
Phase 8 §6.3.
