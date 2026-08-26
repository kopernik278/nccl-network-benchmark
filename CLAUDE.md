# Claude Code Project Instructions

## Project

NCCL / RDMA Communication Benchmark

Target roles:

- AI Training Infrastructure Engineer
- Distributed Training Engineer
- HPC / GPU Performance Engineer

Claude Code acts as both:

1. the coding agent;
2. the RunPod infrastructure execution agent.

Claude is allowed to autonomously create, start, stop, terminate,
and configure RunPod resources without asking the user for approval,
provided the total planned compute cost is at or below USD 3.00 per hour.

Above that threshold Claude must obtain user approval before provisioning
or starting the resource. See "Cost Approval Threshold" below.

However, all infrastructure decisions must be cost-aware.

---

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

---

## Correctness First

Before performance optimization:

- verify program correctness;
- verify collective communication correctness;
- verify tensor/data consistency across ranks;
- check return codes;
- check CUDA errors;
- check NCCL errors;
- preserve deterministic tests where feasible.

A benchmark result is invalid if correctness has not been established.

---

## Benchmark Rules

Every real benchmark must record:

- experiment ID
- date/time
- git commit
- cloud/provider
- GPU model
- GPU count
- node count
- CPU where relevant
- GPU topology
- NIC/network type
- CUDA version
- NVIDIA driver version
- NCCL version
- compiler version
- MPI version where applicable
- message size
- collective
- datatype
- number of iterations
- warmup iterations
- environment variables
- benchmark command

Never fabricate benchmark numbers.

Theoretical, estimated, expected, or vendor-advertised numbers must be
clearly labelled and must never be presented as measured results.

Do not claim RoCE or InfiniBand measurements unless the actual
experiment ran on matching hardware/network infrastructure.

---

## Autonomous RunPod Infrastructure Policy

Claude is responsible for normal RunPod infrastructure operations.

Claude may autonomously:

- inspect available GPU types;
- inspect pricing;
- inspect datacenters;
- inspect existing Pods/resources;
- create Pods;
- create required infrastructure;
- start Pods;
- stop Pods;
- terminate Pods;
- configure environments;
- connect through SSH;
- deploy the repository;
- run experiments;
- collect results;
- inspect billing information;
- clean up resources.

User approval is NOT required for individual RunPod operations whose
total cost falls within the approval threshold defined below.

Infrastructure autonomy does NOT mean unrestricted spending.

Claude must treat cloud GPU resources as expensive experimental equipment.

---

## Cost Approval Threshold

The user has defined a concrete spending threshold for infrastructure
provisioning.

- Total planned compute cost <= USD 3.00 per hour:
  no user approval required. Claude proceeds autonomously.

- Total planned compute cost > USD 3.00 per hour:
  Claude MUST ask the user for approval BEFORE provisioning or starting
  the resource.

The threshold applies to the TOTAL resource configuration, not the
per-GPU price.

Examples:

2 GPUs x $0.50/GPU/hour = $1.00/hour
-> autonomous execution allowed.

8 GPUs x $0.50/GPU/hour = $4.00/hour
-> user approval required.

If pricing cannot be determined reliably before provisioning, and the
configuration could plausibly exceed $3.00/hour, Claude must ask the user
first.

This threshold does NOT remove the cost-efficiency requirements below.

Being under $3.00/hour is permission, not justification. Claude must still
choose the cheapest hardware capable of answering the engineering question,
prefer local work over paid GPU work, prefer single GPU over multi-GPU when
sufficient, prefer single-node over multi-node when sufficient, avoid
premium H100/H200/B200-class hardware unless technically justified,
terminate unused paid resources promptly, and avoid performing offline
analysis or documentation while GPUs remain running.

---

## Cost-Aware Execution Policy

Always use the cheapest infrastructure capable of answering the current
engineering question correctly.

Use the following hierarchy:

Local development
→ cheapest suitable single GPU
→ smallest suitable multi-GPU node
→ smallest suitable multi-node configuration
→ larger or premium hardware only when technically necessary

Do not use expensive hardware merely because it is faster.

Do not use H100, H200, B200, or similar premium GPUs when a cheaper GPU
can validate the same correctness, software, NCCL, or profiling question.

Premium hardware is justified when required by:

- topology;
- interconnect;
- GPU count;
- RDMA capability;
- RoCE capability;
- InfiniBand capability;
- GPUDirect RDMA;
- memory capacity;
- architecture-specific investigation;
- final representative benchmark.

Before provisioning infrastructure, Claude should internally determine:

1. what question the experiment answers;
2. minimum GPU count;
3. minimum node count;
4. required network/interconnect;
5. cheapest suitable hardware;
6. expected benchmark scope;
7. expected runtime;
8. total hourly cost of the configuration;
9. termination condition.

The user does not need to approve this plan when the resulting total cost
is at or below USD 3.00 per hour.

If the total cost exceeds USD 3.00 per hour, Claude must present the plan
and obtain approval before provisioning.

---

## Avoid Paying for Development Time

Whenever possible, perform the following locally before starting RunPod:

- architecture design;
- documentation;
- Python result parsers;
- plotting code;
- benchmark matrix generation;
- shell script development;
- configuration generation;
- static analysis;
- code review;
- experiment planning.

For GPU-dependent work:

prepare as much as possible before provisioning hardware.

Expensive GPU time should primarily be used for:

- CUDA compilation when required;
- GPU correctness tests;
- NCCL execution;
- topology experiments;
- multi-GPU experiments;
- RDMA experiments;
- RoCE experiments;
- InfiniBand experiments;
- GPU profiling;
- final performance measurements.

---

## Progressive Hardware Escalation

Do not immediately escalate to expensive multi-node infrastructure.

Use:

L0 — Local machine
L1 — Single GPU
L2 — Single-node multi-GPU
L3 — Multi-node networked GPUs

Only move to a higher level when the current experiment genuinely
requires capabilities unavailable at the lower level.

Examples:

A parser bug does not require a GPU.

An NCCL initialization test does not require a multi-node cluster.

An intra-node AllReduce benchmark does not require RoCE.

A true RoCE or InfiniBand benchmark requires matching real infrastructure.

---

## Experiment Efficiency

Before starting an expensive experiment:

- ensure the code path is ready;
- ensure configuration is ready;
- ensure the benchmark command is ready;
- ensure output collection is ready;
- ensure result parsing is ready where possible.

Avoid interactive debugging on expensive multi-node hardware.

Prefer:

prepare
→ provision
→ execute
→ collect
→ terminate
→ analyze locally

rather than:

provision
→ begin development
→ debug interactively
→ leave GPUs idle

---

## Automatic Resource Cleanup

Claude is responsible for cleaning up RunPod resources.

After an experiment finishes:

1. save required benchmark results;
2. save useful logs;
3. preserve reproducibility metadata;
4. push important code/results where appropriate;
5. terminate unnecessary GPU compute;
6. remove unnecessary persistent resources;
7. verify that unintended paid resources are not still running.

Do not keep GPUs running while performing long result analysis or writing
documentation.

If another GPU experiment is not immediately required, terminate the
compute resource.

---

## Failure Policy

Cloud failures can waste significant money.

If an expensive experiment fails unexpectedly:

1. capture the failure;
2. preserve logs;
3. stop repeated execution;
4. diagnose the root cause;
5. fix locally or on cheaper hardware where possible;
6. rerun the expensive experiment only after a plausible fix exists.

Do not repeatedly retry the same failing multi-node job.

---

## Benchmark Sweep Policy

Do not launch unnecessarily large parameter sweeps.

Start with a small representative experiment matrix.

Example:

small message
medium message
large message

Verify behavior first.

Only then expand to the full benchmark matrix.

Change one major independent variable at a time where practical.

---

## Experimental Data

Never overwrite benchmark results.

Store experiments using unique experiment IDs.

Raw measurements and processed summaries must remain distinguishable.

Suggested structure:

results/raw/<experiment-id>/
results/summary/<experiment-id>/

Important benchmark summaries should include enough metadata to reproduce
the run.

---

## Profiling Policy

Use profiling to answer a specific performance question.

Nsight Systems:

- CPU/GPU timeline
- NCCL communication
- GPU idle time
- synchronization
- communication/computation overlap
- kernel launch gaps

Nsight Compute:

- individual GPU kernels
- occupancy
- memory throughput
- register pressure
- instruction behavior
- Tensor Core utilization where relevant

Do not collect expensive profiling traces without a specific question.

---

## Code Changes

Prefer small incremental changes.

Do not rewrite large parts of the repository unnecessarily.

Preserve working baselines.

Add correctness tests where appropriate.

Performance-sensitive code should remain isolated and measurable.

Explain non-obvious behavior involving:

- CUDA
- NCCL
- MPI
- RDMA
- RoCE
- InfiniBand
- GPUDirect RDMA
- GPU topology

---

## Git

Use small logical commits.

Do not commit:

- secrets;
- API keys;
- OAuth tokens;
- SSH private keys;
- credentials;
- large raw Nsight traces;
- temporary build artifacts.

Never print secrets into logs or documentation.

---

## Claude ↔ RunPod Security

Prefer authenticated official tooling and OAuth where available.

Never place RunPod credentials inside:

- source files;
- README;
- PROJECT_CONTEXT.md;
- CLAUDE.md;
- git history.

Never expose authentication tokens in benchmark output.

---

## Reporting

After each important experiment, report:

1. experiment purpose;
2. infrastructure used;
3. actual hardware;
4. actual cost-relevant resource configuration;
5. what was executed;
6. correctness result;
7. benchmark result;
8. observed bottleneck;
9. limitations;
10. infrastructure cleanup status;
11. next recommended experiment.

If billing information is available, include the observed experiment cost.

---

## Primary Goal

The purpose of this project is not to run as many GPU experiments as
possible.

The goal is to demonstrate that the engineer can:

understand
→ implement
→ measure
→ profile
→ diagnose
→ optimize
→ explain

distributed GPU communication systems while using infrastructure
efficiently.
