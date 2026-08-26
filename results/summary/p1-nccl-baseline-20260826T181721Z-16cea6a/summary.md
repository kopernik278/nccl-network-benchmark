# Summary — p1-nccl-baseline-20260826T181721Z-16cea6a

First measured NCCL baseline of the project. Full report:
[`docs/experiments/p1b-first-2gpu-nccl-baseline.md`](../../../docs/experiments/p1b-first-2gpu-nccl-baseline.md)

| | |
|---|---|
| Date (UTC) | 2026-08-26 |
| Repo commit | `16cea6aed21d649453b65981c72d4b8ea7e32605` |
| Hardware | 2 × NVIDIA L4, single node (RunPod SECURE, EU-RO-1) |
| Price | $0.98/hour total |
| Topology | `NODE` (PCIe via host bridges, one NUMA node) — **no NVLink** |
| Transport | P2P/direct pointer, 2 ring channels |
| CUDA / driver | 12.8 (nvcc) / 580.126.20 |
| NCCL | 2.25.1+cuda12.8 |
| nccl-tests | 2.19.7, commit `717b683` |
| Network | none-single-node (no RDMA devices detected) |

## Correctness

468 rows, all `value_kind = measured`, **0** correctness failures,
**0** validation errors (`#wrong` max = 0), 468/468 schema-valid.

## n = 2 sanity checks (busbw / algbw)

| Collective | Expected | Observed mean | Verdict |
|---|---|---|---|
| AllReduce | 1.00 | 1.0000 | pass |
| AllGather | 0.50 | 0.5000 | pass |
| ReduceScatter | 0.50 | 0.5000 | pass |

## Headline results (out-of-place, median of 3 repeats)

| Collective | Latency floor | Peak algbw | Peak busbw |
|---|---:|---:|---:|
| AllReduce | 7.07 µs | 11.02 GB/s | 11.02 GB/s |
| AllGather | 7.24 µs | 20.78 GB/s | 10.39 GB/s |
| ReduceScatter | 7.14 µs | 19.63 GB/s | 9.81 GB/s |

GB/s = 10⁹ bytes/s.

## Pattern

Flat ≈ 7.1 µs latency floor up to 8–16 KiB (fixed overhead dominates), knee at
32–64 KiB, then linear growth. Bandwidth saturates by ~4 MiB and holds to
128 MiB. Bus bandwidth converges to 9.8–11.0 GB/s across all three
collectives — the interconnect is the limit in that regime. Run-to-run spread
below 1%.

No bottleneck attribution is claimed: PCIe link generation and width were not
captured on this run.

## Files

- `results.jsonl` — 468 rows, canonical
- `results.csv` — same rows, flat
- `../../raw/p1-nccl-baseline-20260826T181721Z-16cea6a/` — verbatim evidence
