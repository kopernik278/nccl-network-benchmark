# Summary — p6-ring-allreduce-20260828T164556Z-911ccf0

Phase 6 simplified Ring AllReduce.
Full report: [`docs/experiments/p6-ring-allreduce.md`](../../../docs/experiments/p6-ring-allreduce.md)

| | |
|---|---|
| Date (UTC) | 2026-08-28 |
| Repo commit | `911ccf0` |
| Hardware | 4 × NVIDIA L4, RunPod SECURE US-MO-2, $1.96/hour |
| **Transport** | **`host-staged`** — direct P2P is functionally broken on this host |
| CUDA / driver | 12.8 / 570.195.03 |
| NCCL (reference only) | 2.25.1+cuda12.8 |

## Correctness
35 rows, all `value_kind = measured`, **0 mismatches**, max absolute error 0.0.
Every configuration was verified against an independently computed oracle
before being timed.

## Communication volume
Bytes actually copied per rank match the `2(N-1)/N · M` prediction **exactly**
at all five sizes.

## Headline
- **V1 → V2** (removing device barriers): **1.48×–3.04×**
- **V2 → V3** (subchunk pipelining): **1.22×** at 16 MiB; ≈1.0× at 128 MiB;
  **2.01× slower** at 1 KiB with 8 subchunks — both ends of the pipeline tradeoff
- Custom ring is faster than NCCL at ≥16 MiB **in this harness on this
  degraded-transport host** — see report §9 for why that is not a general claim

## Caveats that bound every number
- Host-staged transport only; no direct-P2P result exists here
- A **~4.4 ms fixed harness cost** makes all sizes ≤ 1 MiB unusable for ranking
- Single measurement per configuration; no run-to-run variance

## Files
- `results.jsonl` / `results.csv`
- `../../raw/p6-ring-allreduce-20260828T164556Z-911ccf0/` — verbatim evidence
