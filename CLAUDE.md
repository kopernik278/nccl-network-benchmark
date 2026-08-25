# Claude Code Project Instructions

## Project

NCCL / RDMA Communication Benchmark

Target roles:
- AI Training Infrastructure Engineer
- Distributed Training Engineer
- HPC / GPU Performance Engineer

## Core Engineering Workflow

For every major feature or experiment follow:

Theory
→ Architecture
→ Design
→ Implementation
→ Correctness
→ Baseline
→ Profiling
→ Bottleneck Analysis
→ Optimization
→ Benchmark Again
→ Documentation

Do not optimize without measurements.

Do not claim performance results that were not actually measured.

## Correctness First

Before performance optimization:

- verify program correctness;
- verify collective communication correctness;
- verify tensor/data consistency across ranks;
- check return codes;
- check NCCL/CUDA errors;
- keep deterministic tests where feasible.

## Benchmark Rules

Every real benchmark must record:

- date/time
- git commit
- GPU model
- GPU count
- node count
- topology
- NIC/network
- CUDA version
- NCCL version
- driver version
- message size
- collective
- number of iterations
- warmup iterations
- environment variables
- command used

Never fabricate benchmark numbers.

Theoretical or expected values must be explicitly labelled as theoretical/estimated.

## Cost-Aware Infrastructure Policy

RunPod and other GPU cloud resources cost real money.

Default rule:

LOCAL FIRST
→ CHEAPEST GPU VALIDATION
→ SINGLE NODE
→ MULTI GPU
→ MULTI NODE RDMA

Use the smallest resource capable of validating the current hypothesis.

Claude must NOT:

- create RunPod resources autonomously;
- resize or scale a cluster autonomously;
- start expensive multi-node experiments without explicit user approval;
- leave GPU instances running unnecessarily;
- launch long benchmark sweeps without first estimating experiment scope;
- repeat failed expensive experiments without diagnosing the failure first.

Before proposing a paid GPU experiment, provide:

1. Purpose
2. Minimum hardware required
3. GPU count
4. Node count
5. Expected commands
6. Number of benchmark runs
7. Expected output
8. Stop condition

Prefer development, compilation, unit tests, parsing, plotting,
documentation, and static analysis on local hardware whenever possible.

## Experimental Discipline

Change one important variable at a time.

Keep raw benchmark output separate from processed summaries.

Do not overwrite benchmark results.

Each experiment should have an identifiable experiment ID.

## Code Changes

Prefer small incremental changes.

Do not rewrite large parts of the repository unnecessarily.

Preserve working baselines.

Performance-sensitive code should be isolated and measurable.

Explain non-obvious CUDA, NCCL, MPI, RDMA, RoCE, and InfiniBand behavior.

## Git

Use small logical commits.

Do not commit:

- large Nsight traces;
- build artifacts;
- secrets;
- SSH keys;
- API tokens.

## Communication With User

After important changes report:

1. What changed
2. Why
3. How to test
4. Expected result
5. Whether GPU resources are required
6. Estimated experiment scope
7. Next step
