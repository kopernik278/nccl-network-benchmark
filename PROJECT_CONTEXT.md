# Project Context

## Project Name

NCCL / RDMA Communication Benchmark

## Career Target

This project is designed primarily as portfolio evidence for:

- AI Training Infrastructure Engineer
- Distributed Training Engineer
- HPC Engineer
- GPU Performance Engineer

## Project Goal

Build a systems-oriented GPU communication benchmark, implementation,
profiling, and optimization platform for studying communication used in
distributed LLM training.

The project should demonstrate the ability to:

- understand GPU communication;
- implement communication components;
- benchmark real hardware;
- analyze topology;
- profile communication behavior;
- identify bottlenecks;
- optimize communication;
- explain engineering trade-offs.

---

## Main Research Questions

1. How do NCCL collectives behave across different message sizes?
2. How does GPU topology affect collective performance?
3. How do intra-node and inter-node communication differ?
4. Where do communication latency and bandwidth bottlenecks originate?
5. How do TCP, RoCE, and InfiniBand differ in distributed GPU training?
6. How does NCCL choose algorithms, protocols, and communication paths?
7. How does Ring AllReduce work internally?
8. How close can a simplified implementation get to NCCL?
9. When can communication overlap with computation?
10. What factors limit distributed training scaling efficiency?

---

## Primary Collectives

Study and benchmark:

- Point-to-Point
- AllReduce
- AllGather
- ReduceScatter
- Broadcast

---

## Primary Metrics

Measure where applicable:

- latency
- algorithmic bandwidth
- bus bandwidth
- scaling efficiency
- communication time
- GPU utilization
- CPU utilization
- synchronization
- GPU idle time
- communication/computation overlap

---

## Project Phases

### Phase 0 — Development Platform

Build:

- VS Code
- Claude Code
- GitHub
- RunPod integration
- repository structure
- autonomous infrastructure workflow

### Phase 1 — NCCL Baseline

Establish basic NCCL functionality and benchmark methodology.

Validate:

- environment
- NCCL initialization
- correctness
- benchmark harness
- result capture

### Phase 2 — Single-Node Multi-GPU Communication

Study:

- PCIe
- NVLink
- NVSwitch where available
- GPU topology
- collective scaling
- message-size behavior

### Phase 3 — Multi-Node TCP Baseline

Establish inter-node communication baseline without assuming RDMA.

### Phase 4 — RoCE

Run real RoCE experiments only on infrastructure that actually exposes
RoCE networking.

Study:

- RDMA behavior
- bandwidth
- latency
- NCCL behavior
- scaling
- topology
- GPUDirect RDMA where available

### Phase 5 — InfiniBand

Run real InfiniBand experiments on matching infrastructure.

Study:

- IB transport
- GPUDirect RDMA
- collective performance
- scaling behavior
- topology

### Phase 6 — Simplified Ring AllReduce

Implement and verify a simplified Ring AllReduce.

Compare with NCCL.

Analyze:

- algorithm
- correctness
- bandwidth utilization
- synchronization
- implementation overhead

### Phase 7 — Communication Profiling

Use:

- NCCL logs
- Nsight Systems
- Nsight Compute where appropriate
- GPU/system topology tools

Identify real bottlenecks.

### Phase 8 — Communication Optimization

Investigate techniques such as:

- communication/computation overlap
- message/bucket tuning
- collective selection
- NCCL configuration
- topology-aware configuration
- algorithm/protocol tuning where supported

### Phase 9 — Final Reproducibility

Produce:

- reproducible experiments
- final benchmark tables
- plots
- architecture documentation
- bottleneck analysis
- before/after optimization comparison
- limitations
- portfolio-quality README

---

## Engineering Workflow

Every important component should follow:

Theory
→ Architecture
→ Design
→ Implementation
→ Correctness
→ Baseline
→ Profiling
→ Bottleneck
→ Optimization
→ Benchmark Again
→ Documentation

Correctness comes before optimization.

Measurements come before performance claims.

---

## Infrastructure Architecture

Development:

Mac
+ VS Code
+ Claude Code

Source control:

GitHub

Cloud execution:

RunPod

Claude Code acts as the infrastructure execution agent.

Claude may autonomously:

- choose suitable RunPod hardware;
- create resources;
- configure resources;
- start resources;
- stop resources;
- terminate resources;
- deploy code;
- execute experiments;
- collect results.

Individual infrastructure operations do not require user approval when
the total planned compute cost is at or below USD 3.00 per hour.

Above that threshold, Claude must obtain user approval before provisioning
or starting the resource. See "Cost Philosophy" below.

---

## Cost Philosophy

Cloud GPU resources represent real monetary cost.

Infrastructure must therefore be used economically.

The governing principle is:

USE THE CHEAPEST RESOURCE THAT CAN CORRECTLY ANSWER THE CURRENT QUESTION.

Escalation hierarchy:

Local
→ cheapest suitable single GPU
→ smallest suitable multi-GPU node
→ smallest suitable multi-node system
→ premium cluster only when required

### Cost Approval Threshold

- Total planned compute cost <= USD 3.00 per hour:
  no user approval required; Claude proceeds autonomously.

- Total planned compute cost > USD 3.00 per hour:
  Claude must ask the user for approval BEFORE provisioning or starting
  the resource.

The threshold applies to the TOTAL resource configuration, not the per-GPU
price.

2 GPUs x $0.50/GPU/hour = $1.00/hour -> autonomous execution allowed.
8 GPUs x $0.50/GPU/hour = $4.00/hour -> user approval required.

If pricing cannot be determined reliably before provisioning and the
configuration could plausibly exceed $3.00/hour, Claude must ask first.

The threshold is permission, not justification. It does not relax the
cheapest-sufficient-resource principle above: Claude must still prefer local
work over paid GPU work, single GPU over multi-GPU when sufficient,
single-node over multi-node when sufficient, avoid premium
H100/H200/B200-class hardware unless technically justified, terminate unused
paid resources promptly, and avoid offline analysis or documentation while
GPUs remain running.

Development and debugging should be moved away from expensive hardware
whenever possible.

Multi-node RDMA hardware should primarily be used for experiments that
cannot be reproduced on cheaper infrastructure.

Expensive machines should primarily perform measurements rather than
general development.

---

## Infrastructure Cleanup

After a GPU-dependent experiment:

results
→ logs
→ metadata
→ repository
→ terminate compute
→ verify cleanup

Do not leave GPU resources running during offline analysis,
documentation, or plotting.

---

## Benchmark Integrity

Never fabricate performance results.

Measured numbers must come from actual experiments.

Vendor specifications must be labelled as specifications.

Estimated results must be labelled as estimates.

RoCE measurements require real RoCE infrastructure.

InfiniBand measurements require real InfiniBand infrastructure.

Do not silently substitute TCP results for RDMA measurements.

---

## Reproducibility

Every important experiment should record:

- experiment ID
- Git commit
- hardware
- GPU count
- node count
- topology
- network
- CUDA
- driver
- NCCL
- MPI where relevant
- compiler
- message sizes
- datatype
- iteration count
- environment variables
- command
- timestamp

---

## AI Collaboration

### User

Role:

Engineer / learner / project owner

Responsibilities:

- understand system design;
- understand important code;
- understand experiments;
- make high-level project decisions;
- develop independent engineering ability.

The user does not need to manually operate RunPod during normal project
execution.

### ChatGPT

Role:

AI Infra tutor and systems/performance reasoning partner

Responsibilities:

- explain theory;
- explain implementation;
- design architecture;
- design experiments;
- review technical decisions;
- interpret benchmark results;
- analyze profiling;
- identify bottlenecks;
- propose optimizations;
- connect project work to interview expectations.

### Claude Code

Role:

Coding and infrastructure execution agent

Responsibilities:

- repository navigation;
- implementation;
- tests;
- debugging;
- scripts;
- builds;
- Git workflow;
- RunPod infrastructure;
- environment setup;
- benchmark execution;
- profiling execution;
- result collection;
- resource cleanup.

Claude must remain cost-aware while operating infrastructure
autonomously.

---

## Final Project Standard

The project is not complete merely because NCCL executes.

Completion requires:

- correct implementation;
- correctness tests;
- benchmark harness;
- actual benchmark results;
- topology analysis;
- profiling evidence;
- bottleneck analysis;
- at least one meaningful optimization;
- before/after comparison;
- simplified Ring AllReduce implementation;
- reproducibility;
- clear documentation;
- explicit limitations.

The strongest final evidence should be:

"I identified a real GPU communication bottleneck, measured it,
understood its cause, changed the system or configuration, and
demonstrated the resulting performance difference using reproducible
experiments."
