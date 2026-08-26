# Phase 1B — First Measured 2-GPU NCCL Baseline

Status: **completed**
Experiment ID: `p1-nccl-baseline-20260826T181721Z-16cea6a`
Date (UTC): 2026-08-26
Design: [`phase1_nccl_baseline.md`](phase1_nccl_baseline.md)
Schema: [`../design/RFC-001-result-schema.md`](../design/RFC-001-result-schema.md)

> All numbers in this report are **measured** and traceable to raw nccl-tests
> output preserved under `results/raw/p1-nccl-baseline-20260826T181721Z-16cea6a/`.
> The one vendor specification quoted is explicitly labelled as such.
> **No RoCE, InfiniBand, RDMA, GPUDirect RDMA, or NVLink behaviour is claimed
> or measured here** — none of that hardware was present.

---

## 1. Purpose

First real GPU experiment of the project. Validate the complete Phase 1
pipeline end to end — infrastructure → environment capture → nccl-tests build
→ correctness → benchmark → raw preservation → parsing → schema validation →
summary → cleanup — and establish the baseline later phases compare against.

Not an optimization experiment. No tuning was applied.

---

## 2. Infrastructure

| Field | Value |
|-------|-------|
| Provider | RunPod |
| Pod ID | `xhbh0qcn3gsqvp` |
| Cloud | SECURE |
| Data centre | EU-RO-1 |
| GPU | NVIDIA L4 × **2** (same node) |
| **Quoted price** | **$0.49 / GPU / hour → $0.98 / hour total** |
| Approval | Under the $3.00/hour threshold → autonomous, no approval required |
| Container disk | 20 GB (no persistent/network volume) |
| Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| Pod lifetime | ≈ 11 min (created 18:11:14 UTC, terminated ≈ 18:22 UTC) |
| Benchmark wall time | ≈ 3 min (runner 18:17:21 → 18:20) |

### Provisioning history (both attempts recorded honestly)

A first pod was created on the cheapest available configuration —
**2 × RTX 3070, community cloud, $0.13/GPU/hr = $0.26/hr** (pod `924mr5hxcjdx2n`).
That host had **no reachable SSH path**: it exposed no public IP, and the
`ssh.runpod.io` proxy timed out forwarding to the pod's internal address
(`100.65.24.78:2002`). Rather than debug networking on paid hardware, the pod
was terminated after ≈ 8 minutes and the experiment moved to secure cloud,
which provides a public IP and direct SSH.

A second choice, 2 × RTX 2000 Ada secure ($0.48/hr), returned
`no longer any instances available`. L4 was the next step up the fallback
chain. **The Phase 1 design's primary pick, RTX A5000, was out of stock at
execution time** — the design's own note that availability must be re-checked
at provisioning time proved correct.

---

## 3. Environment (captured, not assumed)

| Field | Value |
|-------|-------|
| Hostname | `5c44f3e9a47e` |
| OS / kernel | Ubuntu 24.04.3 LTS / 6.8.0-106-generic (x86_64) |
| CPU | AMD EPYC 9254 24-Core (48 logical cores) |
| GPU memory | 23034 MiB per GPU |
| GPU PCI IDs | `0000:03:00.0`, `0000:c3:00.0` |
| NVIDIA driver | 580.126.20 |
| CUDA (nvcc) | 12.8 (V12.8.93) |
| CUDA (driver ceiling) | 13.0 |
| **NCCL** | **2.25.1+cuda12.8** (from the `NCCL_DEBUG=VERSION` banner) |
| **nccl-tests** | version 2.19.7, commit `717b68318278e93f371d8ffb46b076069d7c7851` |
| Compiler | gcc 13.3.0 |
| MPI | **not installed** — intentional; Phase 1 uses single-process multi-GPU (`-g 2`) |
| RDMA devices | **none detected** |
| Repo commit | `16cea6aed21d649453b65981c72d4b8ea7e32605` (clean tree) |

### Topology — determined from evidence, not inferred

```text
        GPU0    GPU1    NIC0    CPU Affinity    NUMA Affinity
GPU0     X      NODE    NODE    0-47            0
GPU1    NODE     X      NODE    0-47            0
```

- `NODE` = traffic traverses PCIe **and** the interconnect between PCIe host
  bridges within a single NUMA node.
- **No NVLink.** The matrix shows no `NV#` entry, and `nvlink_present` was
  recorded as `false`. Both GPUs sit on one NUMA node (affinity 0-47).

The `NCCL_DEBUG=INFO` probe shows what NCCL actually did with that topology:

```text
Check P2P Type intraNodeP2pSupport 1 directMode 1
Channel 00/0 : 0[0] -> 1[1] via P2P/direct pointer
Channel 01/0 : 0[0] -> 1[1] via P2P/direct pointer
```

So **peer-to-peer DMA is enabled and used** over the PCIe path, with **2 ring
channels**. This matters: it means the measured bandwidth reflects GPU-to-GPU
P2P across PCIe host bridges, not a host-memory staging fallback.

---

## 4. Commands executed

```bash
# build
scripts/setup_nccl_tests.sh -d /root/nccl-tests      # CUDA_HOME=/usr/local/cuda, MPI disabled

# benchmark (runner drives all three collectives, both tiers)
scripts/run_nccl_baseline.sh -g 2 -t both

# per-run command shape, e.g. AllReduce full tier:
/root/nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 2 -w 20 -n 50 -c 1 -d float

# parse (locally, after the pod was terminated)
python3 scripts/parse_nccl_output.py --raw-dir results/raw/p1-nccl-baseline-20260826T181721Z-16cea6a --strict
```

Build succeeded on the first attempt; **no script defect was found during
execution**. The smoke tier (`-b 8 -e 128M -f 4096`, `-w 5 -n 20`) ran first
and gated the full tier, as designed.

---

## 5. Correctness

| Check | Result |
|-------|--------|
| Runs executed | 12 (3 smoke + 9 full) |
| Failed runs | **0** |
| `#wrong` across every row | **0** (max observed 0) |
| `Out of bounds values : 0 OK` trailer | present on every run |
| Process exit codes | all 0 |
| CUDA / NCCL errors | none (all stderr files empty) |
| Rows parsed | **468**, all `value_kind = measured` |
| Schema conformance | **468/468 valid** against `schemas/nccl_result.schema.json` |
| Rows with `correctness_ok = false` | **0** |

The correctness gate passed, so the full baseline was permitted to proceed.

---

## 6. n = 2 sanity checks (H5)

Expected `busbw / algbw` = `2(n−1)/n` = 1.0 for AllReduce, `(n−1)/n` = 0.5 for
AllGather and ReduceScatter. Restricted to points with algbw ≥ 1.0 GB/s, where
nccl-tests' two-decimal output precision is not the limiting factor:

| Collective | Expected | Mean | Min | Max | Points | Parser verdict |
|------------|----------|------|-----|-----|--------|----------------|
| AllReduce | 1.00 | **1.0000** | 1.0000 | 1.0000 | 94 | 136/136 pass |
| AllGather | 0.50 | **0.5000** | 0.4957 | 0.5043 | 94 | 136/136 pass |
| ReduceScatter | 0.50 | **0.5000** | 0.4976 | 0.5045 | 93 | 136/136 pass |

**All three pass.** Parsing, rank count, and units are confirmed correct. The
small residual spread on AllGather/ReduceScatter is rounding at two decimals,
not a real deviation.

---

## 7. Results (out-of-place, median of 3 repeats)

Bandwidth is **GB/s = 10⁹ bytes/s**, not gigabits.

### AllReduce

| size | latency (µs) | algbw (GB/s) | busbw (GB/s) |
|-----:|-------------:|-------------:|-------------:|
| 8 B | 7.07 | 0.00 | 0.00 |
| 8 KiB | 7.67 | 1.07 | 1.07 |
| 64 KiB | 17.73 | 3.70 | 3.70 |
| 512 KiB | 65.93 | 7.95 | 7.95 |
| 4 MiB | 390.18 | 10.75 | 10.75 |
| 32 MiB | 3059.61 | 10.97 | 10.97 |
| **128 MiB** | **12182.60** | **11.02** | **11.02** |

### AllGather

| size | latency (µs) | algbw (GB/s) | busbw (GB/s) |
|-----:|-------------:|-------------:|-------------:|
| 64 B | 7.09 | 0.01 | 0.00 |
| 8 KiB | 6.98 | 1.17 | 0.59 |
| 64 KiB | 12.11 | 5.41 | 2.71 |
| 512 KiB | 44.00 | 11.92 | 5.96 |
| 4 MiB | 211.40 | 19.84 | 9.92 |
| 32 MiB | 1624.55 | 20.65 | 10.33 |
| **128 MiB** | **6458.89** | **20.78** | **10.39** |

### ReduceScatter

| size | latency (µs) | algbw (GB/s) | busbw (GB/s) |
|-----:|-------------:|-------------:|-------------:|
| 64 B | 7.30 | 0.01 | 0.00 |
| 8 KiB | 7.08 | 1.16 | 0.58 |
| 64 KiB | 12.22 | 5.36 | 2.68 |
| 512 KiB | 44.94 | 11.67 | 5.83 |
| 4 MiB | 225.67 | 18.59 | 9.29 |
| 32 MiB | 1740.03 | 19.28 | 9.64 |
| **128 MiB** | **6838.88** | **19.63** | **9.81** |

### Peak bus bandwidth

| Collective | Peak busbw (GB/s) |
|------------|------------------:|
| AllReduce | 11.02 |
| AllGather | 10.39 |
| ReduceScatter | 9.81 |

---

## 8. Analysis

### 8.1 Latency: flat floor, then linear growth (H2, H3 supported)

All three collectives show a **fixed latency floor of ≈ 7.1 µs**:

| Collective | Floor | Flat (≤ +15%) up to | First point > +50% |
|------------|------:|--------------------:|-------------------:|
| AllReduce | 7.07 µs @ 8 B | 8 KiB | 32 KiB |
| AllGather | 7.24 µs @ 32 B | 16 KiB | 64 KiB |
| ReduceScatter | 7.14 µs @ 32 B | 16 KiB | 64 KiB |

From 8 B to 1 KiB, AllReduce latency changes by a factor of **1.00** despite a
128× increase in payload. Below ~8–16 KiB the cost is entirely fixed overhead
— kernel launch, synchronization, protocol handshake — and the message size is
irrelevant. The knee sits at 32–64 KiB; beyond it latency grows essentially
linearly with size, which is the bandwidth-bound regime.

H2 and H3 are both supported by the data.

### 8.2 Bandwidth plateaus cleanly (H3 supported)

Across the largest four message sizes, algorithmic bandwidth is flat:

| Collective | algbw over last 4 sizes | Variation |
|------------|------------------------|----------:|
| AllReduce | 10.93 → 11.02 GB/s | 0.09 GB/s |
| AllGather | 20.50 → 20.78 GB/s | 0.28 GB/s |
| ReduceScatter | 19.18 → 19.63 GB/s | 0.45 GB/s |

Saturation is reached by roughly 4 MiB and holds to 128 MiB. The link, not the
algorithm, is the limit in this regime.

### 8.3 Bus bandwidth converges across collectives

This is the point of bus bandwidth. The three collectives have very different
algorithmic bandwidths at 128 MiB (11.02 / 20.78 / 19.63 GB/s), but their
**bus** bandwidths cluster tightly at **9.8–11.0 GB/s**. Once the per-collective
correction factor is applied, all three are pushing the same physical link at
about the same rate — exactly what one expects when the interconnect is the
bottleneck and the collectives are otherwise healthy.

### 8.4 Does the topology explain the numbers?

The topology is consistent with the measurements, with one caveat.

Both GPUs are on one NUMA node, connected `NODE` (PCIe + inter-host-bridge),
with **P2P/direct enabled** and 2 ring channels. A sustained ≈ 10–11 GB/s of
bus traffic is a plausible figure for GPU-to-GPU P2P crossing PCIe host
bridges.

**Caveat — this run did not capture the PCIe link state.** NVIDIA specifies the
L4 as PCIe Gen4 ×16 (**vendor specification, not measured here**), which would
be ≈ 31.5 GB/s theoretical per direction; the measured ≈ 11 GB/s is well below
that. Attributing the gap requires knowing the negotiated link generation and
width, which `collect_env.sh` did not record. That gap is now fixed (§10), but
for *this* experiment the honest statement is: **the measured bandwidth is
consistent with a PCIe-mediated P2P path, and the specific limiting factor was
not established.**

No bottleneck claim is made. This is a baseline.

### 8.5 Measurement quality

Run-to-run spread across the 3 independent repeats, for messages ≥ 1 MiB:

| Collective | Median spread | Max spread |
|------------|--------------:|-----------:|
| AllReduce | 0.48% | 0.99% |
| AllGather | 0.58% | 0.75% |
| ReduceScatter | 0.31% | 0.58% |

Under 1% throughout. The dedicated secure-cloud host produced far more stable
measurements than the community-cloud noise the design anticipated.

In-place and out-of-place agree closely at 128 MiB (AllReduce 12182.70 vs
12182.60 µs; AllGather 6435.26 vs 6458.89 µs; ReduceScatter 6892.96 vs
6838.88 µs) — differences well inside run-to-run spread.

### 8.6 Data-quality note: zero-byte rows

28 parsed rows have `message_size_bytes = 0`, all from AllGather and
ReduceScatter. With `-b 8` and 2 ranks the per-rank element count rounds down
to zero, and nccl-tests emits a degenerate 0-byte row (≈ 0.15 µs). These are
genuine tool output, are preserved, and pass their correctness check, but they
are **not meaningful latency measurements** and were excluded from the analysis
above. Future sweeps should start at a size that divides evenly by the rank
count.

---

## 9. Limitations

- **2 ranks, 1 node.** Nothing here generalizes to multi-node behaviour.
- **No NVLink, no NVSwitch.** Absolute bandwidth is not representative of
  NVLink-class systems. Interconnect comparison is Phase 2.
- **No RoCE, InfiniBand, RDMA, or GPUDirect RDMA.** No such hardware was
  present (`rdma_devices: none detected`); no claim about them is made.
- **PCIe link generation and width were not captured**, so the ≈ 11 GB/s
  plateau cannot be attributed to a specific link limit (§8.4).
- **fp32 only**; reduced-precision datatypes were not studied.
- **Single-process multi-GPU** (`-g 2`); no MPI, no multi-process contention.
- Only 3 of the 5 project collectives were run; Broadcast and P2P are pending.
- Cost figures are derived from pod lifetime × quoted rate; the RunPod billing
  API had not yet posted records for this window.

---

## 10. Framework change made as a result of this run

`scripts/collect_env.sh` now records **PCIe link generation and width**
(current and max, per GPU) via
`nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current,...`,
plus a `Gen<N> x<W>` summary line in `env.txt`.

Rationale: on a PCIe-connected system, a measured GB/s figure cannot be
compared against the link it actually crossed without this. Its absence is
precisely what blocked a firm conclusion in §8.4. No other framework change
was needed — the harness ran end to end on the first attempt.

---

## 11. Cost and cleanup

| Item | Value |
|------|-------|
| Pod 1 (2× RTX 3070, community) | ≈ 8 min @ $0.26/hr ≈ **$0.03** |
| Pod 2 (2× L4, secure) | ≈ 11 min @ $0.98/hr ≈ **$0.19** |
| **Total (derived)** | **≈ $0.22** |
| Billing API | no records posted for this window yet |

**Cleanup status: complete and verified.** Both pods were terminated
(`delete-pod` → 204). A follow-up `list-pods` returned **0 pods**. No network
volumes or other persistent paid resources were created. All parsing, analysis,
and documentation were performed locally **after** GPU compute was released.

---

## 12. Next recommended experiment

**Phase 2 — single-node multi-GPU topology and collective benchmarks.** The
open question this baseline raises is what limits the ≈ 11 GB/s plateau. The
natural next step is a topology comparison: repeat this identical sweep on a
node whose GPUs are connected by NVLink (for example an SXM configuration) and
on a node with a different PCIe arrangement, with PCIe link state now captured,
and compare bus bandwidth across interconnects.

Phase 2 has **not** been started.
