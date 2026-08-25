# RFC-000: NCCL Communication Benchmark Project Plan

## Motivation

Distributed LLM training depends heavily on GPU communication.

This project studies GPU collective communication from the application,
runtime, topology, and network levels.

## Goals

- Benchmark important NCCL collectives.
- Understand latency/bandwidth behavior.
- Analyze topology effects.
- Study intra-node and inter-node communication.
- Experiment with RoCE and InfiniBand where real hardware is available.
- Implement a simplified Ring AllReduce.
- Profile communication behavior.
- Investigate communication/computation overlap.

## Non-Goals

Initially this project will not:

- build a complete replacement for NCCL;
- implement a production RDMA stack;
- reproduce data-center-scale measurements;
- claim RoCE/InfiniBand performance without real matching hardware.

## Architecture

Benchmark Runner
      |
      +-- NCCL tests
      |
      +-- Custom P2P
      |
      +-- Custom Collectives
      |
      +-- Ring AllReduce
      |
      +-- Profiling
      |
      +-- Result Parser
      |
      +-- Analysis / Plotting

## Experimental Layers

Layer 1:
single GPU / development validation

Layer 2:
single-node multi-GPU

Layer 3:
multi-node TCP

Layer 4:
multi-node RoCE

Layer 5:
multi-node InfiniBand

## Correctness

Before benchmark data is accepted:

- processes/ranks must initialize correctly;
- collective outputs must be verified;
- CUDA/NCCL errors must be checked;
- benchmark configuration must be recorded.

## Performance Measurements

Primary metrics:

- latency
- algorithmic bandwidth
- bus bandwidth
- scaling efficiency

Supporting metrics:

- GPU utilization
- CPU utilization
- synchronization
- communication timeline

## Cost Strategy

Paid GPU resources are only used when required by the experiment.

Before multi-node experiments:

1. implementation must already work;
2. command-line interface must already be validated;
3. output parser must already work;
4. experiment matrix must be defined;
5. expected run count must be known.

Expensive cluster time should primarily be measurement time,
not development/debugging time.

## Risks

- cloud providers may expose different network fabrics;
- GPU/NIC topology may vary;
- RDMA privileges may be restricted;
- Nsight hardware counters may be unavailable;
- multi-node debugging may consume excessive cloud time.

## Future Work

- NCCL algorithm comparison
- NCCL protocol comparison
- topology-aware analysis
- communication/computation overlap
- collective tuning
