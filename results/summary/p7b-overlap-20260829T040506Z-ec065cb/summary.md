# Summary — p7b-overlap-20260829T040506Z-ec065cb

Phase 7B communication / compute overlap.
Full report: [`docs/experiments/p7b-overlap.md`](../../../docs/experiments/p7b-overlap.md)

| | |
|---|---|
| Date (UTC) | 2026-08-29 |
| Repo commit | `ec065cb` |
| Hardware | 4 × NVIDIA L4, RunPod SECURE EUR-IS-1, $1.96/hour |
| **Transport** | **`SHM/direct`** — NCCL reports `intraNodeP2pSupport 0` |
| Compute | cuBLAS SGEMM 512×512, 20.9 µs per repetition (calibrated at runtime) |

## Headline
- **Overlap is real**: 62–95% of the theoretical opportunity realised in 11 of
  12 microbenchmark cases. Best case hides **94%** of a 3.5 ms AllReduce.
- **Overlap is not free**: compute runs **1.03×–2.09× slower** under
  concurrency in every case; communication is mostly unaffected (1.00×–1.06×)
  **except** at 128 MiB / ratio 2.0 where it slows **1.83×** and efficiency
  collapses to **0.386**.
- **Bucket size has a plateau, not a point**: 4, 8 and 16 MiB buckets are within
  **1.3%** of each other on step time (spread 0.3–1.0%); 64 MiB is **+16%**.
- **Efficiency and step time disagree**: efficiency rises monotonically as
  buckets shrink (0.596 → 0.944) while step time does not. Step time is the
  objective; efficiency alone would have chosen wrong.

## Timeline evidence
Kernels genuinely run concurrently (NVTX `overlapped` is 1.71× shorter than
`sequential`). Contention is visible in the distributions: GEMM median stays
~21.6 µs but its maximum reaches 564 µs, and the 128 MiB NCCL kernel spans
27.5–44.7 ms within one run.

## Files
- `results.jsonl` / `results.csv` — 51 rows, all `value_kind = measured`
- `nvtx_*.csv`, `kernels_*.csv`, `cudaapi_*.csv` — extracted trace summaries
- `../../raw/p7b-overlap-20260829T040506Z-ec065cb/` — verbatim evidence
