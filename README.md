# NCCL / RDMA Communication Benchmark

A systems-oriented GPU communication benchmarking and optimization project
for distributed AI training.

The project investigates the communication stack behind multi-GPU and
multi-node LLM training, including NCCL collectives, GPU topology,
RDMA networking, RoCE, InfiniBand, communication profiling, and
collective algorithm implementation.

## Motivation

Distributed LLM training performance is determined not only by GPU compute
but also by communication.

At scale, operations such as:

- AllReduce
- AllGather
- ReduceScatter
- Point-to-Point communication

can become major training bottlenecks.

This project studies these mechanisms experimentally rather than treating
NCCL as a black box.

## Goals

The project aims to:

- benchmark NCCL collective communication;
- measure latency and bandwidth across message sizes;
- analyze intra-node GPU topology;
- study inter-node communication;
- investigate RoCE and InfiniBand;
- analyze NCCL algorithms and communication paths;
- implement a simplified Ring AllReduce;
- profile GPU communication behavior;
- investigate communication/computation overlap;
- identify and optimize real communication bottlenecks.

## Communication Operations

Primary collectives:

- Point-to-Point
- AllReduce
- AllGather
- ReduceScatter
- Broadcast

## Metrics

Important metrics include:

- latency
- algorithmic bandwidth
- bus bandwidth
- scaling efficiency
- communication time
- GPU utilization
- CPU utilization
- GPU idle time
- communication/computation overlap

## Experimental Layers

The project progresses through increasingly complex infrastructure:

```text
Local development
        ↓
Single GPU
        ↓
Single-node multi-GPU
        ↓
Multi-node TCP
        ↓
RoCE
        ↓
InfiniBand
