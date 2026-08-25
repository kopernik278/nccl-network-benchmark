# Project Context

## Project Name

NCCL / RDMA Communication Benchmark

## Goal

Build a systems-oriented GPU communication benchmark and analysis platform
for studying communication used by distributed LLM training.

The project is intended as portfolio evidence for:

- AI Training Infrastructure
- Distributed Training
- HPC
- GPU Performance Engineering

## Main Questions

1. How do NCCL collective operations behave across message sizes?
2. How does GPU topology affect communication performance?
3. What is the difference between intra-node and inter-node communication?
4. Where do latency and bandwidth bottlenecks come from?
5. How do TCP, RoCE and InfiniBand differ?
6. How does NCCL choose communication algorithms and paths?
7. How does Ring AllReduce work internally?
8. When can communication overlap with computation?

## Collectives

Primary:

- Point-to-Point
- AllReduce
- AllGather
- ReduceScatter
- Broadcast

## Metrics

Measure where applicable:

- latency
- algorithmic bandwidth
- bus bandwidth
- scaling efficiency
- GPU utilization
- CPU utilization
- communication time
- synchronization / idle time

## Project Phases

Phase 0
Development environment and repository setup

Phase 1
Single-node NCCL baseline

Phase 2
Topology-aware multi-GPU experiments

Phase 3
Multi-node TCP baseline

Phase 4
RoCE experiments

Phase 5
InfiniBand experiments

Phase 6
Simplified Ring AllReduce implementation

Phase 7
Communication profiling

Phase 8
Communication optimization

Phase 9
Final reproducibility and portfolio documentation

## Cost Policy

Cloud GPU hardware must be used economically.

Development hierarchy:

Local Mac
→ cheapest suitable single GPU
→ single-node multi-GPU
→ two-node cluster
→ larger cluster only when justified

Multi-node RDMA resources are reserved primarily for experiments that
cannot be reproduced on cheaper hardware.

GPU instances should be stopped immediately after required measurements
and artifacts have been saved.

## Project Standard

A phase is not considered complete merely because code executes.

Important phases should produce:

- implementation
- correctness evidence
- benchmark configuration
- real measurements
- analysis
- profiling evidence when applicable
- limitations
- reproducibility instructions
