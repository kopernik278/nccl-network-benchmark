# Phase 1 — NCCL Baseline Experiment Design

Status: designed, not yet executed
Author: Claude Code (infrastructure/execution agent)
Design date: 2026-08-26
Related: `docs/design/RFC-001-result-schema.md`

> This document contains **no measured results**. It defines what will be
> measured, how, and what would count as success. Every number appearing in
> this file is either a configuration parameter, a vendor/catalog figure, or
> an explicitly labelled expectation — never a measurement.

---

## 1. Purpose

Establish the first reproducible NCCL communication baseline for this project,
so that every later phase (NVLink topology, multi-node TCP, RoCE, InfiniBand,
custom Ring AllReduce, optimization) has a trustworthy point of comparison.

Phase 1 is a **methodology** milestone, not a performance milestone. The
deliverable is a harness that produces correct, fully-annotated, machine-
readable measurements — not a high bandwidth number.

Concretely, Phase 1 must answer:

1. Does NCCL initialize and run correctly on the target environment?
2. Do AllReduce, AllGather, and ReduceScatter produce numerically correct
   results across ranks?
3. What are latency, algorithmic bandwidth, and bus bandwidth as a function
   of message size on the cheapest viable 2-GPU node?
4. Can the entire run be reproduced from recorded metadata alone?

Explicit non-goals for Phase 1: peak bandwidth, NVLink/NVSwitch analysis,
multi-node behaviour, RDMA of any kind, algorithm/protocol tuning.

---

## 2. Hypotheses

Stated so they can be falsified by the data.

| ID | Hypothesis | How it is tested |
|----|------------|------------------|
| H1 | NCCL initializes across 2 ranks on one node, and all three collectives complete with zero validation errors. | `#wrong == 0` on every row; exit code 0. |
| H2 | For small messages (≲ 1 KiB), latency is approximately flat — dominated by fixed per-collective overhead (launch, sync, protocol), not by size. | Latency vs. size curve is flat in the small-size region. |
| H3 | Above a transition size, latency grows approximately linearly with message size and algorithmic bandwidth saturates. | Bandwidth vs. size curve rises then plateaus. |
| H4 | On a PCIe-attached 2-GPU node with no NVLink, saturated bus bandwidth is bounded by the PCIe / host-staging path and will be far below NVLink-class figures. | Compare plateau busbw against the link the topology matrix reports. |
| H5 | At n = 2 ranks, `busbw == algbw` for AllReduce, and `busbw == 0.5 × algbw` for AllGather and ReduceScatter. | Arithmetic self-check on parsed rows (see §9). |

H5 is a **parser/units correctness check**, not a claim about hardware: it
follows from the bus-bandwidth definitions and must hold exactly. If it fails,
the parser or the rank count is wrong, and no performance conclusion may be
drawn from that run.

H2/H3 describe the classic latency-bound → bandwidth-bound transition. If the
measured curve disagrees, that is a finding to investigate and report — not a
result to discard.

---

## 3. Hardware Requirements

Minimum viable configuration for Phase 1:

- 1 node
- 2 NVIDIA GPUs visible to one process
- Any CUDA-capable architecture; no specific compute capability required
- **NVLink is NOT required.** NVSwitch is NOT required.
- No RDMA NIC, no InfiniBand, no RoCE.

Rationale: every Phase 1 question (init, correctness, harness, metadata,
size-sweep shape) is answerable with two of the cheapest available GPUs on a
single host. Interconnect quality changes the *values* but not the *validity*
of the methodology. Interconnect comparison is deliberately deferred to
Phase 2.

Whatever interconnect the node actually has will be recorded from
`nvidia-smi topo -m` and reported. Phase 1 makes no claim about a link type it
did not run on.

---

## 4. Software Requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| NVIDIA driver | any version supporting the GPU | recorded |
| CUDA toolkit | ≥ 11.x, `nvcc` needed to build nccl-tests | recorded |
| NCCL | shipped with the base image | version recorded via `NCCL_DEBUG=VERSION` |
| nccl-tests | built from source on the node | commit SHA recorded |
| MPI | **not required in Phase 1** | see below |
| Python 3 | for local parsing only | not needed on the GPU node |

**MPI is intentionally avoided in Phase 1.** `nccl-tests` supports
single-process / multi-GPU mode via `-g <ngpus>`, where one process drives
both GPUs. This removes MPI as a variable and as an install step, which is
both cheaper and easier to make correct. Multi-process and multi-node runs
(which do need MPI or an equivalent launcher) start in Phase 3.

The `mpi_version` metadata field is therefore expected to be `null` for
Phase 1 runs. That is recorded honestly rather than left blank.

---

## 5. Collectives

| Collective | nccl-tests binary | Datatype | Reduction |
|------------|-------------------|----------|-----------|
| AllReduce | `all_reduce_perf` | `float` (fp32) | `sum` |
| AllGather | `all_gather_perf` | `float` (fp32) | n/a |
| ReduceScatter | `reduce_scatter_perf` | `float` (fp32) | `sum` |

Broadcast and Point-to-Point are in the project scope but deferred: AllReduce,
AllGather, and ReduceScatter are the three that matter most for data-parallel
and sharded (ZeRO/FSDP-style) training, and they exercise the full
reduce + gather machinery. Adding more collectives to the same harness later
is a config change, not a code change.

Note on size semantics: nccl-tests reports the *buffer* size, whose meaning
differs per collective (for AllGather/ReduceScatter the reported size relates
to the total gathered/scattered buffer, not the per-rank contribution). The
harness records the reported size verbatim and does not renormalize across
collectives, to avoid introducing a silent unit error. Cross-collective
comparison must use bus bandwidth, which is what bus bandwidth exists for.

---

## 6. Message-Size Strategy

Two tiers, per the project's "small representative matrix first" rule.

### Tier A — smoke sweep (always run first)

3 points, one invocation per collective:

```
-b 8 -e 128M -f 4096      →   8 B, 32 KiB, 128 MiB
```

Chosen to hit one clearly latency-bound point, one transition-region point,
and one clearly bandwidth-bound point. Runtime is seconds. Its job is to prove
the whole pipeline works — binaries, GPUs, correctness check, output capture,
parser — before any larger sweep is worth paying for.

### Tier B — full sweep (only after Tier A is clean)

25 points, powers of two:

```
-b 8 -e 128M -f 2         →   8 B … 128 MiB
```

128 MiB upper bound: large enough to saturate any Phase 1-class link, small
enough to fit comfortably in the smallest candidate GPU's memory alongside
in-place and out-of-place buffers.

Tier B is only launched if Tier A passes every correctness criterion in §10.

---

## 7. Warmup Strategy

| Tier | Warmup iterations (`-w`) |
|------|--------------------------|
| A | 5 |
| B | 20 |

Warmup exists to exclude one-time costs from the measurement: NCCL channel
and ring setup, buffer registration, CUDA context and module load, memory
pool growth, and GPU clock/power ramp. The first iterations of a cold
collective are systematically slower and are not representative of
steady-state training traffic.

Tier B uses more warmup because its small-message points are the most
sensitive to residual cold-start effects.

---

## 8. Iteration Strategy

| Tier | Measured iterations (`-n`) | Whole-sweep repeats |
|------|----------------------------|---------------------|
| A | 20 | 1 |
| B | 50 | 3 |

nccl-tests reports the mean over `-n` iterations. That captures within-run
noise but not run-to-run variance (process placement, clock state, other
tenants on a shared community-cloud host).

Tier B therefore repeats the entire sweep 3 times as **separate process
launches**, recorded with a `repeat_index`. Reported figures will state
median across repeats and the observed spread. A single run is not treated as
a measurement of the machine.

---

## 9. Metrics

All three come directly from nccl-tests and are recorded verbatim; the parser
recomputes them only as a consistency check.

| Metric | Field | Definition |
|--------|-------|------------|
| Latency | `latency_us` | Mean wall time of one collective, microseconds. |
| Algorithmic bandwidth | `algorithmic_bandwidth_gbps` | `message_size_bytes / time` — throughput as the *application* sees it. |
| Bus bandwidth | `bus_bandwidth_gbps` | `algbw × correction_factor` — approximates bytes actually crossing the interconnect, making different collectives and rank counts comparable to hardware peak. |

Bus-bandwidth correction factors for `n` ranks:

| Collective | Factor | At n = 2 |
|------------|--------|----------|
| AllReduce | `2(n−1)/n` | 1.0 |
| AllGather | `(n−1)/n` | 0.5 |
| ReduceScatter | `(n−1)/n` | 0.5 |

**Units:** GB/s means 10⁹ bytes/s (decimal), matching nccl-tests. It is *not*
gigabits per second. The field suffix `_gbps` is retained for schema
compatibility but the schema documents the unit explicitly to prevent an
8× misreading.

Both out-of-place and in-place variants are recorded as separate rows
(`placement` field). They are not averaged together.

Derived/secondary (computed locally, no extra GPU time): scaling behaviour vs.
message size, latency-vs-bandwidth transition point, and repeat-to-repeat
spread.

GPU utilization, CPU utilization, and GPU idle time are project-level metrics
but require profiling; they are deferred to Phase 7.

---

## 10. Correctness Criteria

nccl-tests validates results against a CPU-computed reference on every
iteration and reports a `#wrong` count per row. Phase 1 treats correctness as
a gate, not a footnote.

A run is **correct** only if all of the following hold:

1. Process exit code is 0.
2. `#wrong == 0` on **every** data row, for **both** in-place and out-of-place.
3. The trailer line `# Out of bounds values : 0 OK` is present.
4. No CUDA error and no NCCL error (`WARN`/`ERROR`) appears in stderr.
5. The number of ranks NCCL reports matches the number requested.
6. H5 holds: the busbw/algbw ratio matches the expected factor for the
   collective and rank count, within floating-point rounding of the reported
   precision.

If any check fails, the affected rows are marked `correctness_ok = false` and
are **excluded from all performance analysis**. Per project rules, a benchmark
result whose correctness was not established is not a result.

Failures are preserved (raw output kept, reason recorded) rather than deleted,
so the failure itself remains reproducible and diagnosable.

---

## 11. Experiment Metadata

Captured by `scripts/collect_env.sh` on the node into `env.json` + `env.txt`,
and merged into every result row by the parser.

Required: experiment_id, timestamp (UTC, ISO 8601), git_commit, provider,
gpu_model, gpu_count, node_count, topology, network, cuda_version,
driver_version, nccl_version, mpi_version, collective, datatype,
message_size_bytes, warmup_iterations, measured_iterations, latency_us,
algorithmic_bandwidth_gbps, bus_bandwidth_gbps, command, environment variables.

Also captured: hostname, OS/kernel, CPU model and core count, per-GPU PCI bus
IDs and memory, full `nvidia-smi topo -m` matrix, NVLink status where present,
nccl-tests commit SHA, and how the NCCL version was detected.

Unavailable values are recorded as `null` with the reason, never guessed and
never silently omitted.

**Environment variables are captured through an allowlist**
(`NCCL_*`, `CUDA_*`, `NVIDIA_*`, `UCX_*`, `OMPI_*`, plus non-secret RunPod
identifiers) with a hard redaction filter on anything whose name contains
KEY / TOKEN / SECRET / PASSWORD / CREDENTIAL / AUTH / PRIVATE. A blanket
environment dump is never written, so credentials cannot leak into results or
into git.

Schema details: `docs/design/RFC-001-result-schema.md`.

---

## 12. Output Structure

```text
results/raw/<experiment-id>/          # verbatim, never edited
    env.json                          # machine-readable environment
    env.txt                           # human-readable environment
    run_manifest.json                 # every command, exit code, timing
    <collective>.<tier>.r<N>.stdout.txt
    <collective>.<tier>.r<N>.stderr.txt
    nccl_debug_info.txt               # one NCCL_DEBUG=INFO run

results/summary/<experiment-id>/      # derived, regenerable
    results.jsonl                     # one row per measurement point
    results.csv                       # same data, flat
    summary.md                        # written after analysis
```

Raw and parsed data are strictly separated: `results/raw/` is append-only
evidence; `results/summary/` is regenerable from it by re-running the parser.
Results are never overwritten — a new run gets a new experiment ID.

**Experiment ID format:**

```
p1-<slug>-<UTC timestamp>-<git short SHA>
e.g. p1-nccl-baseline-20260826T141530Z-b1f095c
```

---

## 13. Cost Strategy

Governing rule: cheapest resource that can correctly answer the question.

Phase 1 needs 2 GPUs on 1 node and nothing else, so the selection criterion is
simply the lowest hourly price for a GPU type with ≥ 2 units available on a
single host.

RunPod catalog snapshot taken 2026-08-26 (community cloud, per-GPU/hour, as
reported by the catalog API — these are **vendor prices, not measurements**,
and must be re-checked at provisioning time):

| Candidate | $/GPU/hr | 2× GPU $/hr | VRAM | Notes |
|-----------|----------|-------------|------|-------|
| **RTX A5000 (primary)** | 0.16 | **0.32** | 24 GB | Professional Ampere; PCIe P2P generally available |
| RTX 4000 Ada | 0.20 | 0.40 | 20 GB | fallback |
| RTX 3090 | 0.22 | 0.44 | 24 GB | GeForce — P2P typically disabled |
| V100 SXM2 | 0.23 | 0.46 | 16 GB | SXM2 ⇒ NVLink; more interesting for Phase 2 |
| RTX 4090 | 0.34 | 0.68 | 24 GB | fallback |

Selection: **2× RTX A5000, community cloud**, with RTX 4000 Ada / RTX 3090 as
fallbacks if stock is unavailable. Availability was reported LOW at snapshot
time for several of these, so the runner must re-query and fall back rather
than wait on one type.

Premium GPUs (A100, H100, H200, B200) are explicitly **excluded** from
Phase 1: none of the four Phase 1 questions requires them.

Budget and controls:

- Expected wall clock: ≤ 45 min (pod start, nccl-tests build ~2–3 min, Tier A,
  Tier B ×3, collection).
- Expected spend: **≈ $0.25 at $0.32/hr**; hard ceiling **$1.00**. Investigate
  and stop if exceeded.
- Everything that can be built locally is built locally *before* provisioning:
  design, scripts, parser, schema, directory layout, and the exact command
  lines. The pod does compilation and measurement only.
- No interactive debugging on the pod. Prepare → provision → execute →
  collect → **terminate** → analyze locally.
- The pod is terminated immediately after results are pulled back, *before*
  parsing, plotting, or writing the summary.
- Cleanup is verified with a pod listing, and the result of that check is
  reported.

---

## 14. Success Criteria

Phase 1 is complete when all of the following are true:

1. `env.json` is captured with every required metadata field populated or
   explicitly `null` with a reason.
2. AllReduce, AllGather, and ReduceScatter all run at Tier A and Tier B with
   **zero** validation errors (§10).
3. `results.jsonl` validates against `schemas/nccl_result.schema.json`, and
   `results.csv` carries the same rows.
4. H5 (bus/algorithmic bandwidth ratio) holds exactly — proving units and rank
   count are handled correctly.
5. Latency and bandwidth are plotted against message size for all three
   collectives, with the latency-bound → bandwidth-bound transition either
   visible (H2/H3 supported) or explicitly discussed if absent.
6. Raw output is preserved unmodified and the run is reproducible from the
   recorded command + metadata alone.
7. Total spend is within budget, the pod is terminated, and termination is
   verified.
8. Limitations are documented explicitly (§15).

Phase 1 is **not** complete merely because nccl-tests printed numbers.

---

## 15. Known Limitations (to be restated with the results)

- 2 ranks, 1 node. Nothing here generalizes to multi-node behaviour.
- The likely-PCIe interconnect means absolute bandwidth is not representative
  of NVLink or NVSwitch systems.
- Community-cloud hosts are shared; run-to-run variance may reflect
  neighbouring tenants. This is why Tier B repeats 3×.
- fp32 only. Reduction datatype effects (bf16/fp16) are not studied here.
- Single-process multi-GPU mode; no MPI, no multi-process contention.
- **No RoCE, InfiniBand, RDMA, or GPUDirect RDMA result may be claimed from
  this phase.** Phase 1 runs on none of that hardware.
