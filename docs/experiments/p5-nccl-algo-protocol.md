# Phase 5 — NCCL Algorithm and Protocol Characterization

Status: **completed**
Experiment IDs:
`p5-algo-proto-20260828T153618Z-21cf57f` (coarse matrix + spot checks) ·
`p5-dense-20260828T154940Z-21cf57f` (dense crossover sweep)
Date (UTC): 2026-08-28 · Repo commit: `21cf57f`
Previous: [Phase 2](p2-multigpu-scaling.md) · [Phase 3 design](phase3_multinode_tcp_baseline.md)

> All numbers are **measured** and traceable to raw nccl-tests output under
> `results/raw/`. **No optimization was applied** — this phase characterizes,
> it does not tune. No RoCE, InfiniBand, RDMA, or NVLink behaviour is measured
> or claimed; none of that hardware was present.

> **Scope note added in Phase 11 (2026-08-31).** The statement in §3 that "Tree
> is defined for AllReduce/Broadcast/Reduce only" was established from a single
> runtime rejection on **NCCL 2.25.1 with this build of nccl-tests**, and holds
> for that configuration. It is a record of what this runtime rejected, not a
> claim about NCCL in general or about other versions. §12's AUTO conclusions
> are already labelled as *inferred from timings, not read from a log* (§9), and
> that labelling stands.


---

## 1. Purpose

Understand experimentally how NCCL's **algorithm** (Ring, Tree) and **protocol**
(Simple, LL, LL128) selection interacts with message size, and how close NCCL's
automatic choice is to the best forced configuration.

Deliberately excluded: NVLS, CollNet, PAT. They are outside the research
question and carry hardware-specific requirements. NCCL reported
`NVLS multicast support is not available` on all four devices anyway.

---

## 2. Hardware and Environment

| Field | Value |
|-------|-------|
| Provider | RunPod SECURE, US-MO-2 |
| Pods | `9siimeiuttzwee` (coarse), `5cf4pg4vldy305` (dense) |
| GPU | NVIDIA L4 × **4**, single node |
| Compute capability | 8.9 (Ada) |
| PCIe | Gen4 ×16 (max) on all four |
| **Price** | **$0.49/GPU/hr → $1.96/hour total** (under the $3/hr threshold → autonomous) |
| Driver | 570.195.03 |
| CUDA | 12.8 |
| **NCCL** | **2.25.1+cuda12.8** (from the runtime `NCCL_DEBUG=VERSION` banner, not the header) |
| nccl-tests | commit `b4d5beebca8a76cf01335f724d154b9b9d394d96` |

### Topology — asymmetric, and it matters

```text
        GPU0    GPU1    GPU2    GPU3    CPU Affinity      NUMA
GPU0     X      SYS     SYS     SYS     0-31,64-95        0
GPU1    SYS      X      NODE    NODE    32-63,96-127      1
GPU2    SYS     NODE     X      NODE    32-63,96-127      1
GPU3    SYS     NODE    NODE     X      32-63,96-127      1
```

**GPU0 sits alone on NUMA node 0**; GPU1–3 share NUMA node 1. Every path from
GPU0 to another GPU is `SYS` — across the CPU's SMP interconnect.

The consequence is visible in the runtime log: NCCL used
**`via SHM/direct/direct` on every channel** — shared-memory host staging, *not*
peer-to-peer. Phase 1B's 2-GPU L4 node had `isAllDirectP2p 1` and reached
11 GB/s bus bandwidth; this 4-GPU node peaks at 2.4–3.4 GB/s. Same GPU model,
different data path.

**The dense sweep ran on a second pod with an identical topology matrix** (same
host IP, same NUMA split), which is what makes the two experiments comparable at
all — see §7.

---

## 3. Configurations Tested

| Algorithm | Protocol | Label | Accepted? |
|-----------|----------|-------|-----------|
| *(automatic)* | *(automatic)* | `auto` | — reference |
| Ring | Simple | `ring-simple` | yes |
| Ring | LL | `ring-ll` | yes |
| Ring | LL128 | `ring-ll128` | yes |
| Tree | Simple | `tree-simple` | yes |
| Tree | LL | `tree-ll` | yes |
| Tree | LL128 | `tree-ll128` | yes |

**LL128 was not forced blindly.** It was included only after runtime evidence:
NCCL accepted every LL128 combination, each passed the correctness gate, and the
three protocols produced **three clearly distinct timings** under Tree at 1 MiB
(Simple 1804 µs, LL 981 µs, LL128 529 µs). Identical timings would have
suggested a silent fallback; distinct ones show the setting took effect.

### Rejected combination — recorded, not retried

| Collective | Algorithm | Result |
|------------|-----------|--------|
| AllGather | Tree | **REJECTED** — `Test NCCL failure common.cu:645` |
| ReduceScatter | Tree | **REJECTED** — `Test NCCL failure common.cu:645` |

Tree is defined for AllReduce/Broadcast/Reduce only. This was **established from
runtime evidence rather than assumed**, tested once, and recorded as unsupported
rather than retried.

---

## 4. Method

Two-stage, as designed:

1. **Coarse logarithmic scan** — `-b 8 -e 128M -f 8` → 9 points (8 B, 64 B,
   512 B, 4 KiB, 32 KiB, 256 KiB, 2 MiB, 16 MiB, 128 MiB), 7 configurations,
   3 repeats, AllReduce.
2. **Dense sweep** over the crossover region the coarse scan identified —
   `-b 16K -e 512K -f 2` → 6 points, same 7 configurations, 3 repeats.
3. **Spot checks** on AllGather and ReduceScatter at representative sizes,
   Ring protocols only (Tree being unsupported).

Every configuration passed the smoke correctness gate before its measured runs.
`NCCL_DEBUG=INFO` diagnostics were captured **per configuration** and kept out
of the timed set; measured runs used `NCCL_DEBUG=VERSION`.

fp32, out-of-place and in-place both recorded, validation on (`-c 1`).

---

## 5. Correctness

| | Coarse | Dense |
|---|---|---|
| Rows parsed | **468** | **252** |
| `value_kind` | all `measured` | all `measured` |
| `#wrong` maximum | **0** | **0** |
| `correctness_ok = false` | **0** | **0** |
| Bus/algorithmic bandwidth self-check | pass | pass |
| Schema validation | pass | pass |

720 measured rows, zero validation errors, across every algorithm/protocol
combination.

---

## 6. Results

![NCCL algorithm and protocol characterization](../../results/plots/p5-algo-protocol.png)

### 6.1 AllReduce latency (µs), median of 3 — coarse scan

| size | auto | ring-simple | ring-ll | ring-ll128 | tree-simple | tree-ll | **tree-ll128** |
|-----:|-----:|------------:|--------:|-----------:|------------:|--------:|---------------:|
| 8 B | 42.68 | 38.45 | **29.75** | 33.44 | 44.91 | 42.50 | 42.90 |
| 64 B | 29.48 | 44.09 | 30.22 | 32.00 | 45.16 | 33.12 | 30.85 |
| 512 B | 30.54 | 44.07 | 30.32 | 32.31 | 44.98 | **30.26** | 30.83 |
| 4 KiB | 29.81 | 44.10 | **23.85** | 32.44 | 46.80 | 32.45 | 29.09 |
| 32 KiB | 36.65 | 64.48 | 39.02 | 42.05 | 70.44 | 48.49 | 44.79 |
| 256 KiB | 197.93 | 189.24 | 196.01 | 220.62 | 395.30 | 248.84 | **144.81** |
| 2 MiB | 1415.29 | 1464.12 | 1350.15 | 1360.61 | 1617.19 | 1763.37 | **925.14** |
| 16 MiB | 11291.0 | 11387.3 | 10749.2 | 10484.5 | 13360.2 | 13904.7 | **9251.5** |
| 128 MiB | 99398.7 | 100870.0 | 86309.6 | 84214.3 | 118336.0 | 110299.0 | **74432.6** |

### 6.2 Bus bandwidth (GB/s), median of 3

| size | auto | ring-simple | ring-ll | ring-ll128 | tree-simple | tree-ll | **tree-ll128** |
|-----:|-----:|------------:|--------:|-----------:|------------:|--------:|---------------:|
| 32 KiB | 1.34 | 0.76 | 1.26 | 1.17 | 0.70 | 1.01 | 1.10 |
| 256 KiB | 1.99 | 2.08 | 2.01 | 1.78 | 0.99 | 1.58 | **2.72** |
| 2 MiB | 2.22 | 2.15 | 2.33 | 2.31 | 1.95 | 1.78 | **3.40** |
| 16 MiB | 2.23 | 2.21 | 2.34 | 2.40 | 1.88 | 1.81 | **2.72** |
| 128 MiB | 2.03 | 2.00 | 2.33 | 2.39 | 1.70 | 1.83 | **2.70** |

### 6.3 Run-to-run spread (latency, 3 repeats)

| config | median | max |
|--------|-------:|----:|
| auto | 4.47% | 23.85% |
| ring-simple | 3.82% | 17.84% |
| ring-ll | 4.22% | 14.02% |
| ring-ll128 | 2.36% | 42.55% |
| tree-simple | 1.53% | 6.08% |
| tree-ll | 3.75% | 41.03% |
| tree-ll128 | 4.73% | 31.47% |

Variance here is **far higher than Phase 2's sub-1%**. That is a property of this
host and this data path (SHM staging, cross-NUMA, 4 ranks), and it governs how
strongly any of the following can be stated.

---

## 7. The dense sweep did not localize the crossover — and that is the result

**OBSERVATION.** The dense 16 KiB–512 KiB sweep was run to pin the crossover the
coarse scan had bracketed. It did not succeed. Of six size points, **five show no
statistically meaningful winner**:

| size | winner | runner-up | lead | run-to-run noise | significant? |
|-----:|--------|-----------|-----:|-----------------:|--------------|
| 16 KiB | tree-ll | ring-ll128 | 8.6% | 40.8% | **no — tie** |
| 32 KiB | ring-ll | ring-ll128 | 0.7% | 6.5% | **no — tie** |
| 64 KiB | ring-ll128 | ring-ll | 11.2% | 9.4% | marginal yes |
| 128 KiB | ring-ll128 | tree-ll128 | 11.3% | 17.4% | **no — tie** |
| 256 KiB | ring-simple | tree-ll128 | 3.8% | 28.4% | **no — tie** |
| 512 KiB | ring-simple | tree-ll128 | 1.6% | 20.1% | **no — tie** |

**OBSERVATION.** An anchor check at the two sizes present in both sweeps shows
coarse-vs-dense deviations up to **35.8%** (tree-simple at 256 KiB) and **21.4%**
(ring-simple at 256 KiB), despite an identical topology matrix on the second pod.

**INTERPRETATION.** In the 16 KiB–512 KiB window, absolute times are 30–300 µs —
short enough that host jitter on a shared-memory, cross-NUMA data path dominates
the differences between configurations. This is a property of the measurement
regime, not of a defect in the sweep.

**CONCLUSION.** On this host the crossover **cannot be localized more precisely
than "somewhere between 32 KiB and 256 KiB"**. Reporting a sharper number would
be reading signal out of noise. The dense sweep earned its cost by establishing
that limit rather than by narrowing the bracket.

---

## 8. The three regimes

### A. Latency-dominated regime — up to ≈ 4 KiB

**OBSERVATION.** Latency is flat at ≈ 24–45 µs and independent of size. Protocol
choice, not algorithm, separates the configurations:

- **LL**: 23.9–30.3 µs
- **LL128**: 29.1–33.4 µs
- **Simple**: 38.5–46.8 µs

`ring-simple` and `tree-simple` sit at ~44 µs across the whole flat region while
`ring-ll` reaches 23.85 µs at 4 KiB — Simple is **~45–85% slower**, far outside
the 2–4% spread at these sizes.

**INTERPRETATION.** LL ("low latency") uses inline flag-based synchronization
that avoids the separate handshake Simple performs per step. When the payload is
too small to matter, that fixed overhead is the entire cost.

**CONCLUSION.** Best algorithm: either (they tie). Best protocol: **LL**.
AUTO tracks LL closely here.

### B. Transition regime — ≈ 32 KiB to ≈ 512 KiB

**OBSERVATION.** Latency begins to scale with size; no configuration wins
significantly (§7).

**CONCLUSION.** No ranking is defensible on this host. **LIMITATION**, not a
finding about NCCL.

### C. Bandwidth-dominated regime — ≈ 2 MiB and above

**OBSERVATION.** `tree-ll128` wins decisively and consistently:

| size | tree-ll128 | auto | auto ÷ best | auto spread |
|-----:|-----------:|-----:|------------:|------------:|
| 256 KiB | 144.81 µs | 197.93 µs | **1.37×** | 4.5% |
| 2 MiB | 925.14 µs | 1415.29 µs | **1.53×** | 4.7% |
| 16 MiB | 9251.5 µs | 11291.0 µs | **1.22×** | 1.9% |
| 128 MiB | 74432.6 µs | 99398.7 µs | **1.34×** | 2.6% |

Peak bus bandwidth: **tree-ll128 3.40 GB/s** vs auto 2.22 GB/s — **+53%**.

Gaps of 22–53% against spreads of 2–5% are far outside noise.

**INTERPRETATION.** Two mechanisms plausibly compound, and this experiment
separates neither: Tree moves less total data than Ring for AllReduce, and LL128
carries a larger useful payload per 128-byte line than LL while keeping inline
synchronization. On a host where every hop is host-memory staging, reducing
bytes moved matters more than it would over a fast direct link.

**CONCLUSION.** Best algorithm: **Tree**. Best protocol: **LL128**. AUTO is
**materially suboptimal** throughout.

---

## 9. AUTO versus best forced

**OBSERVATION.** By regime:

| regime | AUTO verdict | evidence |
|--------|--------------|----------|
| ≤ 4 KiB | **effectively optimal** at most sizes; one materially suboptimal point at 8 B (1.44×), but with 23.9% spread | tracks `ring-ll` |
| 32 KiB – 512 KiB | **indeterminate** — differences inside noise | §7 |
| ≥ 256 KiB | **materially suboptimal**, 1.22×–1.53× | §8C |

**OBSERVATION.** AUTO's timings track the **Ring** family at large sizes
(2 MiB: auto 1415 µs, ring-simple 1464, ring-ll 1350 — while tree-ll128 is 925).

**LIMITATION.** NCCL's `NCCL_DEBUG=INFO` output prints the *header* of its tuning
table — every algorithm × protocol combination — but **not which cell it selected
for a given size**. AUTO's choice is therefore *inferred from its timings*, not
read from a log statement. The correlation between AUTO and the Ring family is
consistent across sizes, but it is an inference.

**INTERPRETATION.** NCCL's internal cost model does not appear to favour
Tree+LL128 on this SHM/cross-NUMA topology, yet empirically that combination is
22–53% faster at ≥ 256 KiB. This is the kind of case where NCCL's tuning
heuristics — calibrated across a wide range of hardware — do not match an unusual
topology.

**CONCLUSION.** On this host AUTO leaves a real, reproducible 22–53% on the table
for large AllReduce. **This is a characterization, not a recommendation**;
turning it into a tuning change belongs to Phase 8.

---

## 10. AllGather and ReduceScatter spot checks

Ring only — Tree rejected by NCCL (§3). Latency (µs), median:

| collective | size | auto | ring-simple | ring-ll | ring-ll128 |
|------------|-----:|-----:|------------:|--------:|-----------:|
| AllGather | 32 KiB | 43.25 | 43.27 | **30.27** | 32.66 |
| AllGather | 128 MiB | 44219 | 51081 | 48868 | **43659** |
| ReduceScatter | 32 KiB | 40.03 | 40.57 | **30.64** | 44.42 |
| ReduceScatter | 128 MiB | 53934 | 56719 | 56376 | **49515** |

**OBSERVATION.** The same protocol pattern holds: LL fastest at 32 KiB (~30 µs
vs ~40–43 µs), LL128 fastest at 128 MiB. AUTO is close to best on AllGather at
128 MiB (1.3%) but 8.9% behind on ReduceScatter.

**CONCLUSION.** The protocol story generalizes beyond AllReduce; the algorithm
story cannot, because Tree does not apply. The full matrix was **not** repeated
for these collectives — the AllReduce results gave no reason to.

---

## 11. Relationship to Phase 2

Phase 2 interpreted its 2→4 GPU results as small-message latency following
**tree-style `log₂(n)` scaling** and large-message behaviour being governed by
**bandwidth and topology**.

**Phase 5 strengthens the second half and complicates the first.**

**Strengthened.** Large-message behaviour is indeed topology-governed. Phase 5
shows the same GPU model (L4) reaching 11 GB/s bus bandwidth with 2 ranks and
direct P2P (Phase 1B) but only 2.4–3.4 GB/s with 4 ranks over SHM staging. The
data path, not the rank count, sets the ceiling — exactly Phase 2's reading.

**Complicated.** Phase 2 inferred tree selection at small messages from a
measured 1.89× latency growth matching `log₂(n)` better than the ring prediction
of 3×. Phase 5 shows that at small sizes on this hardware the **algorithm barely
matters** — `ring-ll` (29.75 µs) and `tree-ll` (42.50 µs) differ, but `ring-ll`
vs `ring-simple` (38.45 µs) differ by a comparable amount from **protocol alone**.
Phase 2's `log₂(n)` observation remains consistent with tree selection, but
Phase 5 shows protocol is a confounding factor Phase 2 did not control.

**LIMITATION — explicitly.** Phase 5 does **not** resolve Phase 2's
rank-count/topology confound. That requires the deferred Phase 4 topology-isolation
experiment. Phase 5 was run on a *different* topology (4 GPUs, cross-NUMA, SHM)
than Phase 2 (4 GPUs, two PHB pairs, P2P), so it cannot even be differenced
against it.

---

## 12. What this teaches about NCCL selection

1. **Protocol dominates the small-message regime; algorithm dominates the large.**
   Below ~4 KiB, LL vs Simple is worth ~45–85%; algorithm is nearly irrelevant.
   Above ~2 MiB, Tree vs Ring is worth up to 53%.
2. **NCCL's automatic choice is good but not optimal, and where it misses is
   systematic** — here, consistently at large sizes, consistently by choosing a
   Ring variant over Tree+LL128.
3. **LL128 is not exotic on this hardware.** It was available and correct on
   PCIe L4s with no NVLink, contrary to a common assumption, and was the best
   protocol at large sizes.
4. **A transition region can be too noisy to characterize.** Measuring one is
   still worthwhile: it bounds what any tuning claim in that range can assert.
5. **Runtime evidence beats assumption, repeatedly** — Tree's rejection for
   AllGather/ReduceScatter, LL128's availability, and the SHM transport were all
   established by observation, and two of the three contradicted a plausible
   prior.

---

## 13. Limitations

- **One host topology.** 4× L4, cross-NUMA, SHM staging. The bandwidth-regime
  conclusion may not transfer to a P2P or NVLink node; on a fast direct link the
  balance between "bytes moved" and "protocol overhead" changes.
- **High variance.** Median run-to-run spread 1.5–4.7%, maxima to 42%. The
  transition regime is unresolvable here.
- **AUTO's choice is inferred**, not read from a log (§9).
- **Coarse and dense sweeps ran on different pod instances.** Topology matrices
  were identical, but anchor deviations reached 35.8%; the two are not pooled.
- **fp32 only, AllReduce-centric**, 4 ranks, single node.
- **NVLS/CollNet/PAT untested** by design.
- **No optimization applied.** Phase 5 characterizes only.
- Phase 2's confound remains open (§11).

---

## 14. Cost and cleanup

| Item | Value |
|------|-------|
| Coarse pod `9siimeiuttzwee` | ≈ 17 min @ $1.96/hr ≈ **$0.56** |
| Dense pod `5cf4pg4vldy305` | ≈ 8 min @ $1.96/hr ≈ **$0.26** |
| **Total (derived)** | **≈ $0.82** |

A process note: the dense sweep required a **second pod** because the first was
terminated before it ran — the coarse-then-dense plan called for both on one
instance. The re-provision cost ~$0.26 and forced an anchor check that would
otherwise have been unnecessary. The correct order is to complete every
GPU-dependent stage before releasing the node.

**Cleanup verified.** Both pods terminated (`delete-pod` → 204); `list-pods`
returned **0**. No network volumes or other persistent paid resources. All
parsing, plotting, analysis and documentation were done locally **after** GPU
compute was released.

---

## 15. Next

Phase 6 (simplified Ring AllReduce) has **not** been started. The Phase 5 result
that most deserves follow-up is §9: AUTO's systematic 22–53% shortfall at large
AllReduce on this topology is a concrete, measured optimization target — but
acting on it belongs to Phase 8, not here.
