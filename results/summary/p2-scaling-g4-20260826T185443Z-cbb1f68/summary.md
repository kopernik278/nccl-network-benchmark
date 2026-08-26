# Summary — p2-scaling-g4-20260826T185443Z-cbb1f68

Phase 2 single-node scaling, **4-GPU** configuration. Full report and the
2-vs-4 comparison:
[`docs/experiments/p2-multigpu-scaling.md`](../../../docs/experiments/p2-multigpu-scaling.md)

| | |
|---|---|
| Date (UTC) | 2026-08-26 |
| Repo commit | `cbb1f68` |
| Hardware | 4 × NVIDIA RTX PRO 4500 Blackwell, single node (RunPod SECURE, EU-RO-1) |
| Ranks used | **4** |
| Price | $2.88/hour (4-GPU pod; both configs ran on this one pod) |
| Topology | GPU0–GPU1 `PHB`, GPU2–GPU3 `PHB`, cross-pair `NODE` — **no NVLink** |
| PCIe | Gen4 ×16 (max) on all four GPUs |
| CUDA / driver | 12.8 / 580.126.09 |
| NCCL | 2.31.2+cuda12.9 (runtime banner) |
| nccl-tests | 2.19.7, commit `717b683` |

## Correctness

468 rows, all `value_kind = measured`, **0** correctness failures, `#wrong` max 0,
468/468 schema-valid. n=4 bus/algorithmic bandwidth ratios match the expected
factors.

## Files

- `results.jsonl` — 468 rows, canonical
- `results.csv` — same rows, flat
- `../../raw/p2-scaling-g4-20260826T185443Z-cbb1f68/` — verbatim evidence
