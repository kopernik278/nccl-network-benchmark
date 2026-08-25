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
