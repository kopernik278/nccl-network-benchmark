# Summary — p5-algo-proto-20260828T153618Z-21cf57f

Phase 5 NCCL algorithm/protocol characterization.
Full report: [`docs/experiments/p5-nccl-algo-protocol.md`](../../../docs/experiments/p5-nccl-algo-protocol.md)

| | |
|---|---|
| Date (UTC) | 2026-08-28 |
| Repo commit | `21cf57f` |
| Hardware | 4 × NVIDIA L4, single node (RunPod SECURE, US-MO-2), $1.96/hour |
| Topology | GPU0 on NUMA 0 (`SYS` to all others); GPU1–3 on NUMA 1 — **no NVLink**, **no P2P** |
| Transport | `SHM/direct` (shared-memory host staging) |
| CUDA / driver | 12.8 / 570.195.03 |
| NCCL | 2.25.1+cuda12.8 (runtime banner) |
| nccl-tests | `b4d5beeb` |

## Correctness
All rows `value_kind = measured`, **0** correctness failures, `#wrong` max 0,
bus/algorithmic bandwidth self-check passes, schema valid.

## Headline
- **≤ 4 KiB**: protocol dominates — LL ~24–30 µs vs Simple ~44 µs.
- **32 KiB – 512 KiB**: differences are inside run-to-run noise; no ranking defensible.
- **≥ 2 MiB**: `tree-ll128` best — 3.40 GB/s peak bus bandwidth vs AUTO's 2.22.
- **AUTO is 1.22×–1.53× slower than `tree-ll128`** at ≥ 256 KiB, far outside 2–5% spread.

Tree is **rejected by NCCL** for AllGather and ReduceScatter (runtime evidence).

## Files
- `results.jsonl` / `results.csv`
- `../../raw/p5-algo-proto-20260828T153618Z-21cf57f/` — verbatim evidence
