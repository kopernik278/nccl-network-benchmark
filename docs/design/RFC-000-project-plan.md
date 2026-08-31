# RFC-000: NCCL Communication Benchmark Project Plan

## Motivation

Distributed LLM training depends heavily on GPU communication.

As training scales across multiple GPUs and multiple nodes,
collective communication can become a major performance bottleneck.

This project studies GPU communication from several levels:

- collective communication algorithms;
- NCCL runtime behavior;
- GPU topology;
- intra-node interconnects;
- inter-node networking;
- RDMA;
- RoCE;
- InfiniBand;
- communication/computation overlap.

The goal is not only to run existing benchmarks.

The project should develop the ability to understand, measure,
profile, diagnose, implement, and optimize GPU communication systems
used by distributed AI training.

---

## Goals

- Benchmark important NCCL collectives.
- Understand latency and bandwidth behavior across message sizes.
- Analyze GPU topology effects.
- Compare intra-node and inter-node communication.
- Study TCP, RoCE, and InfiniBand communication.
- Investigate NCCL algorithms, protocols, and communication paths.
- Implement a simplified Ring AllReduce.
- Compare the simplified implementation with NCCL.
- Profile communication behavior.
- Identify real communication bottlenecks.
- Investigate communication/computation overlap.
- Perform measurable communication optimization.

---

## Non-Goals

Initially this project will not:

- build a complete replacement for NCCL;
- implement a production-quality RDMA stack;
- implement a production InfiniBand or RoCE driver;
- reproduce hyperscale datacenter benchmarks;
- claim RoCE performance without real RoCE infrastructure;
- claim InfiniBand performance without real InfiniBand infrastructure;
- present theoretical or vendor bandwidth as measured benchmark data.

---

## Architecture

```text
                    Benchmark Platform
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ↓                ↓                ↓
      NCCL Tests       Custom Code      Profiling
          │                │                │
          │          ┌─────┴──────┐         │
          │          ↓            ↓         │
          │         P2P      Ring AllReduce │
          │                               │
          └───────────────┬───────────────┘
                          ↓
                    Result Collection
                          ↓
                    Result Parser
                          ↓
                     Analysis
                          ↓
                       Plots
```

Raw tool output enters at Result Collection and is never edited. Everything
downstream of Result Parser is derived and regenerable.

---

## Experimental Layers

The project escalates through progressively more capable — and more
expensive — infrastructure. A layer is used only when the question genuinely
cannot be answered at a lower one.

```text
L0  Local development        no GPU; design, parsers, plots, scripts, tests
        ↓
L1  Single GPU               CUDA build, single-device correctness
        ↓
L2  Single-node multi-GPU    NCCL collectives, PCIe / NVLink / NVSwitch
        ↓
L3  Multi-node TCP           inter-node communication without RDMA
        ↓
L4  RoCE                     RDMA over Converged Ethernet
        ↓
L5  InfiniBand               IB transport, GPUDirect RDMA
```

Mapping the layer to the question is a hard rule: a parser bug does not
require a GPU, an NCCL initialization test does not require a cluster, and an
intra-node AllReduce benchmark does not require RoCE.

L4 and L5 require infrastructure that genuinely exposes those fabrics. Whether
such infrastructure is reachable on the chosen provider at acceptable cost is
an open question to be verified before those phases begin, not an assumption
this plan rests on.

---

## Collectives

Primary operations under study:

| Collective | Why it matters |
|------------|----------------|
| AllReduce | Gradient synchronization in data-parallel training |
| AllGather | Parameter/activation gathering in sharded (ZeRO/FSDP-style) training |
| ReduceScatter | The reduce half of sharded optimizers; pairs with AllGather |
| Broadcast | Parameter and state distribution |
| Point-to-Point | Pipeline-parallel stage transfer; building block for everything else |

Phase 1 begins with AllReduce, AllGather, and ReduceScatter — the three that
dominate data-parallel and sharded training and that exercise the full
reduce-plus-gather machinery. Broadcast and P2P follow once the harness is
established; adding them is a configuration change, not a code change.

Datatypes start at fp32 for correctness clarity. Reduced-precision datatypes
(fp16, bf16) are a later variable, changed one at a time.

---

## Correctness

Correctness is a gate, not a footnote. Performance work does not begin until
it passes.

Requirements before any performance number is reported:

- program correctness verified;
- collective communication results validated against a reference;
- tensor/data consistency verified across all ranks;
- return codes checked;
- CUDA errors checked;
- NCCL errors checked;
- deterministic tests preserved where feasible.

Operationally, for nccl-tests-based runs this means a zero validation-error
count on every measured row, a clean out-of-bounds trailer, a zero exit code,
and no CUDA or NCCL error output. Rows failing any check are marked incorrect
and excluded from performance analysis.

Failed runs are preserved with their logs rather than deleted, so the failure
itself stays diagnosable and reproducible.

**A benchmark result whose correctness has not been established is not a
result.**

---

## Performance Measurements

Primary metrics:

| Metric | Meaning |
|--------|---------|
| Latency | Wall time of one collective operation |
| Algorithmic bandwidth | Message size / time — throughput as the application sees it |
| Bus bandwidth | Algorithmic bandwidth × a collective-specific factor, approximating bytes actually crossing the interconnect |
| Scaling efficiency | How performance holds up as ranks and nodes increase |
| Communication time | Time attributable to communication within a workload |
| GPU / CPU utilization | Resource occupancy during communication |
| GPU idle time | Time GPUs spend waiting |
| Communication/computation overlap | Degree to which the two proceed concurrently |

Bus bandwidth exists so that different collectives and rank counts can be
compared against hardware peak on equal terms. Its correction factor is
collective-specific — `2(n−1)/n` for AllReduce, `(n−1)/n` for AllGather and
ReduceScatter — and the relationship serves as a units self-check on the
measurement pipeline.

Bandwidth is recorded in GB/s meaning 10⁹ **bytes** per second, not gigabits.

Utilization, idle time, and overlap require profiling and are introduced in
Phase 7 rather than estimated earlier.

---

## Benchmark Methodology

Every measurement follows the same discipline:

1. **Warmup before measurement.** Early iterations include channel setup,
   buffer registration, context and module load, memory pool growth, and clock
   ramp. They are excluded.
2. **Repeated iterations.** Reported figures are means over a defined
   iteration count, not single samples.
3. **Repeated runs.** Whole sweeps are repeated as separate process launches
   to expose run-to-run variance. A single run is not treated as a measurement
   of the machine; spread is reported alongside central tendency.
4. **Small matrix first.** A minimal representative sweep — small, medium, and
   large message sizes — validates the pipeline before any larger sweep is
   paid for. The small sweep gates the large one.
5. **One variable at a time.** Message size, collective, rank count, node
   count, transport, algorithm, and protocol are changed independently where
   practical, so an observed difference has one candidate cause.
6. **Full metadata capture.** Every run records the environment needed to
   reproduce it (see Reproducibility).
7. **Analysis off the clock.** Parsing, plotting, and interpretation happen
   locally after compute is released.

---

## Ring AllReduce

A simplified Ring AllReduce is implemented from scratch and compared against
NCCL.

The purpose is understanding, not replacement: implementing the algorithm
forces engagement with chunking, the reduce-scatter plus all-gather structure,
ring ordering, synchronization, and buffer management — the mechanics that
NCCL otherwise hides.

Scope:

- implement the algorithm;
- verify numerical correctness against a reference;
- verify consistency across all ranks;
- measure latency and bandwidth on the same harness as NCCL;
- compare bandwidth utilization against NCCL on identical hardware;
- analyze the gap: synchronization cost, chunk sizing, protocol overhead,
  lack of topology awareness, absence of multi-channel parallelism.

The expected outcome is that NCCL wins, and that the project can **explain
precisely why**. A quantified, well-understood gap is the deliverable; beating
NCCL is not a goal.

---

## Profiling

Profiling answers specific performance questions. Traces are not collected
speculatively — they are expensive to gather and to store.

**Nsight Systems** — system-level timeline:

- CPU/GPU timeline correlation
- NCCL communication behavior
- GPU idle time
- synchronization points
- communication/computation overlap
- kernel launch gaps

**Nsight Compute** — individual kernel analysis:

- occupancy
- memory throughput
- register pressure
- instruction behavior
- Tensor Core utilization where relevant

**NCCL logs** (`NCCL_DEBUG=INFO` and subsystem filters) reveal ring and tree
construction, channel counts, and the algorithm and protocol NCCL selected —
often the fastest route to understanding an unexpected result.

**Topology tools** (`nvidia-smi topo -m`, `nvidia-smi nvlink`, `lspci`) record
the physical paths that explain the numbers.

Large raw traces stay out of version control.

---

## Cost Strategy

Cloud GPU resources are real money. The governing principle:

> **Use the cheapest resource that can correctly answer the current question.**

Escalation hierarchy:

```text
Local
  → cheapest suitable single GPU
    → smallest suitable multi-GPU node
      → smallest suitable multi-node configuration
        → premium hardware only when technically necessary
```

### Cost Approval Threshold

Claude Code operates the project's cloud infrastructure autonomously and may
create, purchase, start, stop, configure, and terminate resources.

- **Total planned compute cost ≤ USD 3.00 per hour** — no user approval
  required; Claude proceeds autonomously.
- **Total planned compute cost > USD 3.00 per hour** — Claude must obtain user
  approval **before** provisioning or starting the resource.

The threshold applies to the **total** configuration cost, not the per-GPU
price:

| Configuration | Total | Decision |
|---------------|-------|----------|
| 2 GPUs × $0.50/GPU/hr | $1.00/hr | autonomous |
| 8 GPUs × $0.50/GPU/hr | $4.00/hr | approval required |

If pricing cannot be determined reliably before provisioning and the
configuration could plausibly exceed $3.00/hour, Claude asks first.

### The threshold does not relax cost efficiency

Being under $3.00/hour is permission, not justification. Independently of the
threshold, Claude must still:

- choose the cheapest hardware capable of answering the engineering question;
- prefer local work over paid GPU work;
- prefer a single GPU over multi-GPU when sufficient;
- prefer a single node over multi-node when sufficient;
- avoid premium H100 / H200 / B200-class hardware unless technically
  justified;
- terminate unused paid resources promptly;
- avoid offline analysis, plotting, or documentation while GPUs are running.

Premium hardware is justified only when required by topology, interconnect,
GPU count, RDMA / RoCE / InfiniBand capability, GPUDirect RDMA, memory
capacity, architecture-specific investigation, or a final representative
benchmark.

### Before provisioning

Determine: what question the experiment answers; minimum GPU count; minimum
node count; required network/interconnect; cheapest suitable hardware;
benchmark scope; expected runtime; total hourly cost; and the termination
condition.

### Execution pattern

```text
prepare → provision → execute → collect → terminate → analyze locally
```

rather than provisioning first and developing on paid hardware. Design,
documentation, parsers, plotting, sweep generation, scripts, and static
analysis are all built at L0 before any GPU is started.

---

## Failure Policy

Cloud failures waste money quickly. On unexpected failure of an expensive
experiment:

1. capture the failure;
2. preserve logs;
3. stop repeated execution;
4. diagnose the root cause;
5. fix locally or on cheaper hardware where possible;
6. rerun the expensive experiment only after a plausible fix exists.

Repeatedly retrying the same failing multi-node job is prohibited. Interactive
debugging on expensive multi-node hardware is avoided; failures are reproduced
at the lowest layer that still exhibits them.

---

## Experimental Data

```text
results/raw/<experiment-id>/       verbatim evidence, append-only
results/summary/<experiment-id>/   parsed, derived, regenerable
```

Rules:

- benchmark results are never overwritten;
- every experiment gets a unique experiment ID;
- raw measurements and processed summaries stay distinguishable;
- summaries carry enough metadata to reproduce the run;
- failed runs are retained, not discarded;
- large binary artifacts (profiling traces, archives) stay out of git, while
  the small text artifacts that constitute the evidence for a result are
  committed with it.

Result rows are machine-readable and self-contained, carrying their full
environment so that rows from different phases and different hardware remain
comparable and unambiguously distinguishable. Each row records a `value_kind`
so that measured, estimated, theoretical, and synthetic numbers can never be
silently confused.

---

## Optimization Strategy

Optimization follows measurement and bottleneck analysis. Nothing is tuned
before a profile identifies why it is slow.

Techniques to investigate:

- communication/computation overlap;
- message and bucket size tuning;
- collective selection and restructuring;
- NCCL environment configuration (channels, buffer sizes, thresholds);
- topology-aware placement and configuration;
- algorithm and protocol selection where supported (ring vs. tree; simple,
  low-latency, and LL128 protocols).

Each optimization is evaluated the same way: baseline, change one variable,
re-benchmark under identical conditions, report before and after with the
correctness gate still passing. Optimizations that do not help are reported as
such — a negative result honestly measured is a finding.

---

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| RoCE / InfiniBand infrastructure unavailable or unaffordable | Phases 4–5 blocked | Verify catalog availability cheaply before committing; if genuinely unavailable, document as an explicit limitation rather than substituting TCP |
| Shared/community cloud hosts produce noisy measurements | Misleading conclusions | Repeat whole sweeps as separate launches; report spread, not just means |
| Cloud spend escalates | Project cost | $3.00/hour approval threshold; cheapest-sufficient hardware; terminate immediately after collection; verify cleanup |
| Multi-node setup complexity burns paid time | Wasted spend | Prepare everything at L0; no interactive debugging on expensive hardware; smoke test gates full sweeps |
| Consumer GPUs disable peer-to-peer access | Unrepresentative intra-node results | Record topology and P2P status with every run; interpret results against the interconnect actually present |
| Benchmark tool version differences change output format | Silent parsing errors | Parse tool output by header rather than fixed offsets; pin and record tool commit |
| Confusing theoretical and measured figures | Integrity failure | Explicit `value_kind` on every result row; vendor numbers always labelled |
| Scope creep across ten phases | Nothing finished well | Phases gate each other; correctness gates performance; small matrices gate large ones |

---

## Reproducibility

Every important experiment records:

- experiment ID
- timestamp (UTC)
- git commit
- cloud provider
- GPU model, GPU count, node count
- CPU where relevant
- GPU topology
- NIC / network type
- CUDA version
- NVIDIA driver version
- NCCL version
- compiler version
- MPI version where applicable
- message size
- collective
- datatype
- iteration count
- warmup iterations
- environment variables
- exact benchmark command

Environment variables are captured through an allowlist with credential
redaction, so secrets never reach results, logs, documentation, or version
control.

Unavailable values are recorded as null with a reason — never guessed, never
silently omitted.

---

## Project Phases / Roadmap

Status reflects the repository as of this revision. Later phases have **not**
been executed.

### Phase 0 — Development platform and infrastructure automation

Repository structure, version control, development environment, cloud provider
integration, and the autonomous infrastructure workflow.

*Status: complete.*

### Phase 1 — NCCL baseline

Establish the first reproducible NCCL baseline: environment validation, NCCL
initialization, collective correctness, benchmark harness, result capture, and
reproducible metadata. Collectives: AllReduce, AllGather, ReduceScatter.

*Status: complete. First baseline measured 2026-08-26 on 2 × NVIDIA L4
(single node, PCIe, no NVLink) — see
`docs/experiments/p1b-first-2gpu-nccl-baseline.md`.*

### Phase 2 — Single-node multi-GPU topology and collective benchmarks

PCIe, NVLink, and NVSwitch where available; GPU topology analysis; collective
scaling with rank count; message-size behavior across the latency-bound to
bandwidth-bound transition.

*Status: complete for the 2 vs 4 GPU PCIe case, measured 2026-08-26 — see
`docs/experiments/p2-multigpu-scaling.md`. NVLink and NVSwitch remain
unmeasured; no such hardware has been used.*

### Phase 3 — Multi-node TCP baseline

Inter-node communication baseline that assumes no RDMA. Establishes the
reference against which RDMA benefit is later measured, and introduces
multi-process launching.

*Status: Phase 3A (preparation) complete 2026-08-28 — design, MPI launcher,
network discovery, transport verification and schema extension built and
tested locally at zero cost; see
`docs/experiments/phase3_multinode_tcp_baseline.md`. Phase 3B (measurement)
**deferred on cost**: RunPod Instant Clusters require 2 nodes x 8 GPUs on
B200/H200/H100/A100 only, making the cheapest viable cluster $25.44/hour —
8.5x the project's $3.00/hour threshold. Verified against the live API, not
assumed; no cluster was provisioned and nothing was billed.*

### Phase 4 — Single-node topology isolation

Repeat the Phase 2 rank-count sweep on a 4-GPU node whose GPUs sit in a single
P2P domain (or an NVLink domain), to separate the rank-count effect from the
topology effect that Phase 2 confounded. Phase 2's ~75% bandwidth loss moved
rank count and topology together; only this experiment can attribute it.

*Status: **deferred by the user**. The design is preserved here as a future
extension — it remains the single most valuable follow-up to Phase 2, and the
Phase 2 confound stands unresolved until it runs.*

### Phase 5 — NCCL algorithm and protocol characterization

How NCCL's algorithm (Ring, Tree) and protocol (Simple, LL, LL128) selection
interacts with message size, and how close NCCL's automatic choice is to the
best forced configuration. Single-node, no new fabric required — which is why
it can proceed while the multi-node phases are blocked.

*Status: complete, measured 2026-08-28 on 4 x NVIDIA L4 — see
`docs/experiments/p5-nccl-algo-protocol.md`. Protocol dominates below ~4 KiB
(LL vs Simple worth 45-85%); Tree+LL128 dominates above ~2 MiB and NCCL's
automatic selection is 1.22x-1.53x slower there; the transition region proved
too noisy on this host to rank.*

### Phase 6 — Simplified Ring AllReduce

Implement, verify, and benchmark a simplified Ring AllReduce; compare against
NCCL; analyze algorithm, correctness, bandwidth utilization, synchronization,
and implementation overhead.

*Status: complete, measured 2026-08-28 on 4 x NVIDIA L4 — see
`docs/experiments/p6-ring-allreduce.md`. Three versions (naive, async,
pipelined), all correct against an exact oracle; measured per-rank movement
matches 2(N-1)/N*M exactly. Removing device barriers gave 1.48x-3.04x;
subchunking gave 1.22x at 16 MiB and was 2x slower at 1 KiB. Direct P2P was
functionally broken on the host, so everything ran in explicitly named
host-staged mode, and a ~4.4 ms harness floor makes sizes below a few MiB
unusable for ranking.*

### Phase 7 — Communication profiling

NCCL logs, Nsight Systems, Nsight Compute where appropriate, and topology
tools, used to identify real bottlenecks rather than suspected ones.

*Status: complete. Phase 7A (measurement validation, 2026-08-29) found that
Phase 6's "~4.4 ms harness floor" was a property of that host, not of the
harness (measured at ~27 us), and confirmed from the timeline that 91-93% of
the naive ring's runtime is synchronization barriers; two Phase 6 performance
conclusions are invalidated and three revised there. Phase 7B (overlap,
2026-08-29) measured that 62-95% of the theoretical overlap opportunity is
realised, that overlap costs the compute stream 1.03x-2.09x, and that DDP-like
bucket size has a plateau over 4-16 MiB rather than a sharp optimum. See
`docs/experiments/p7a-harness-validation.md` and `p7b-overlap.md`.*

### Phase 8 — Communication/compute resource contention

Which resource concurrent communication and computation actually fight over.
Three compute workload classes at fixed arithmetic intensity by construction
(compute-heavy SGEMM, memory-heavy triad, mixed), driven to a common duration
and run against the same collective, so the interference can be attributed to
a workload property rather than to the message size.

*Status: complete (2026-08-29). Measured on 4 x NVIDIA A40, `SHM/direct`
transport. The compute stream is slowed in every cell (1.07x-2.00x) and the
slowdown tracks the workload class, not the message size: the compute-heavy
class loses the most and the memory-heavy class the least. **DRAM bandwidth is
ruled out** as the mechanism — the collective's own device-memory traffic is
bounded at ~2% of achievable bandwidth, and the workload consuming ~460 GB/s
disturbs the collective less (1.07x) than the workload consuming almost none
(1.42x). Phase 7B's 128 MiB overlap collapse reproduced on different silicon
and proved to be specific to the compute-heavy class. Nsight Compute was
unavailable (`ERR_NVGPUCTRPERM`), so no counter-verified intensity exists;
kernel granularity and arithmetic intensity remain confounded. NCCL's P2P
transport deadlocks on this host despite `cudaDeviceCanAccessPeer` returning
true — the Phase 6 lesson in a new form. See
`docs/experiments/p8-contention.md`.*

### Phase 9 — Real DDP training workload validation

The microbenchmark conclusions of Phases 7B and 8 re-tested inside a real
PyTorch DistributedDataParallel training step: a compact GPT trained with a real
autograd backward and DDP's own reducer, swept over `bucket_cap_mb`.

*Status: complete (2026-08-29). Measured on 4 x NVIDIA A40, `SHM/direct`,
82.6M parameters, 315 MiB of gradients per step. DDP does overlap: a collective
is resident for 50-83% of backward, and overlap is worth **30.4%** of step time
against an otherwise identical serialised reduction. Step time rises
monotonically with bucket capacity (157.3 -> 166.8 ms for 4 -> 64 MiB); every
gap is resolvable above a 0.39 ms across-launch noise floor, so there is no flat
plateau, but 4/16/25 MiB span only 1.5% while 64 MiB is 6.1% worse. **The
requested bucket capacity is not the collective size**: a single 50 MiB output
projection sets a floor no cap below 48 MiB can cross, which blunts the lever on
large-vocabulary models. Five of six microbenchmark predictions are supported
outright and Phase 8's collective timing predicted the real AllReduce within
7.8%. NCCL's P2P transport deadlocked on a **third** distinct host. See
`docs/experiments/p9-ddp-training.md`.*

### Phase 10 — Final optimization and synthesis

The validated findings turned into a final DDP configuration, measured against
its baseline and against a deliberately bad negative control on one host.

*Status: complete (2026-08-30). Measured on 4 x NVIDIA A40. The functional
preflight **accepted NCCL's default P2P/CUMEM + SHM ring on this host**, unlike
the three hosts of Phases 8 and 9 where the same gate caught a deadlock — the
lesson was always "test the path", never "P2P is broken". Reducing
`bucket_cap_mb` from 25 to 4 cut the synchronisation penalty by 8.5%
(10.72 -> 9.81 ms) but moved step time only 0.48% (152.35 -> 151.62 ms), which
is **inside the run-to-run noise** on this transport; with P2P disabled the same
change is a resolvable but sub-1% 0.93%. The robust result is the negative
control: `bucket_cap_mb = 64` costs 3.6% (6.1% on SHM) despite producing 22%
less total NCCL kernel work per step — isolated collective efficiency is not the
optimization objective. Overlap is worth 20.1% of step time against a
non-overlapped reduction; scaling efficiency 93.1%. See
`docs/experiments/p10-final-optimization.md`.*

### Phase 11 — Portfolio and reproducibility finalization

Reproducible experiments, final benchmark tables, plots, architecture
documentation, bottleneck analysis, before/after optimization comparison,
explicit limitations, and portfolio-quality documentation.

*Status: complete (2026-08-31). No paid GPU resources were provisioned. The
project was renamed **NCCL Communication Performance Lab** and the README
rewritten as a portfolio landing page with a causal 1-10 phase narrative, ten
headline results, six figures, a Mermaid flow diagram, an experiment index, the
reliability preflight, a cost summary distinguishing billed from
lifecycle-estimated, and a "what did not work" section. Added
`docs/REPRODUCING.md` (environment assumptions, dependencies, a six-step minimal
path, and a GPU-free verification procedure) and `docs/PORTFOLIO.md` (summary,
resume bullets, three-level interview guide). A claim audit added dated
correction notes to Phases 5, 7B, 8 and 9 rather than rewriting them.
**All 13 experiment summaries were confirmed to regenerate byte-for-byte from
committed raw data (2 469 rows).** Two `FAILED-RUN-NOTE.md` files that the
`.gitignore` allowlist had silently excluded were found and tracked.*

---

## Roadmap revision (2026-08-28)

The roadmap was revised after Phase 3. The original plan placed **RoCE at
Phase 4 and InfiniBand at Phase 5**; both have been moved to Future Work
below, and those slots now hold experiments that are actually reachable.

The reason is measured, not assumed. Phase 3B established that the cheapest
schedulable RunPod Instant Cluster is 2 nodes x 8 GPUs on A100/H100/H200/B200
only, at **$25.44/hour — 8.5x the project's $3.00/hour threshold** (see
`docs/experiments/phase3_multinode_tcp_baseline.md` section 9.2b). RoCE and
InfiniBand require that same multi-node infrastructure or better, so they are
further out of reach than the TCP baseline that already exceeded budget.

Phase numbering elsewhere in this document and in the experiment reports
follows the revised list. No completed phase changed number.

---

## Future Work

**RoCE experiments** — RDMA over Converged Ethernet: RDMA behaviour,
bandwidth, latency, NCCL behaviour, scaling, topology, and GPUDirect RDMA.
Requires multi-node infrastructure that genuinely exposes RoCE. *Blocked on
cost, per the revision note above. No RoCE measurement exists or is claimed.*

**InfiniBand experiments** — IB transport, GPUDirect RDMA, collective
performance, scaling, topology, on matching real infrastructure. *Blocked on
the same cost constraint. No InfiniBand measurement exists or is claimed.*

Beyond those, natural extensions include:

- reduced-precision collectives (fp16, bf16) and their reduction accuracy
  implications;
- larger scale-out behavior and scaling-efficiency limits across many nodes;
- collective behavior under realistic training workloads rather than
  microbenchmarks;
- interaction between communication and framework-level features such as
  gradient bucketing and sharded optimizer states;
- NCCL algorithm and protocol selection modelling across message-size regimes;
- alternative communication libraries for comparison;
- network congestion and multi-tenant interference effects.

These are explicitly out of scope for the current plan and are recorded so
that scope boundaries stay deliberate.

---

## Success Criteria

The project is not complete merely because NCCL executes. Completion requires:

- correct implementation;
- correctness tests;
- a working benchmark harness;
- actual measured benchmark results;
- GPU topology analysis;
- profiling evidence;
- bottleneck analysis grounded in that evidence;
- at least one meaningful, measured optimization;
- a before/after comparison under identical conditions;
- a simplified Ring AllReduce implementation compared against NCCL;
- reproducible experiments with complete metadata;
- clear documentation;
- explicit, honest limitations.

The strongest final evidence is a single demonstrable claim:

> "I identified a real GPU communication bottleneck, measured it, understood
> its cause, changed the system or configuration, and demonstrated the
> resulting performance difference using reproducible experiments."

Any result that cannot be reproduced from recorded metadata, or whose
correctness was not established, does not count toward these criteria.
