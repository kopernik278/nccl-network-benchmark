# Summary — p7a-harness-validation-20260829T032840Z-64c7410

Phase 7A profiling and harness validation.
Full report: [`docs/experiments/p7a-harness-validation.md`](../../../docs/experiments/p7a-harness-validation.md)

| | |
|---|---|
| Date (UTC) | 2026-08-29 |
| Repo commit | `64c7410` |
| Hardware | 4 × NVIDIA L4, RunPod SECURE **EUR-IS-1**, $1.96/hour |
| Transport | `host-staged` (direct P2P functionally broken here too) |
| Binary | identical to Phase 6; only the host differs |

## Headline
**The Phase 6 "~4.4 ms harness floor" was the host, not the harness.**
The same binary reports NCCL 1 KiB AllReduce at **42 µs** here versus
**4 443 µs** in Phase 6. Calibration puts every harness component at
single-digit microseconds and the whole Phase 6 NCCL body at 64.9 µs.

## Timeline (Nsight Systems + NVTX)
- **91–93% of V1's runtime is `syncAll` barriers** — the V1→V2 speedup is
  confirmed to be removed synchronization.
- NCCL kernel name is direct evidence of its choice:
  `ncclDevKernel_AllReduce_Sum_f32_RING_LL` at both 1 KiB and 128 MiB.
- `ncclCommInitAll` (548 ms) is far outside the timed region (0.74 ms) —
  NCCL setup was never in Phase 6 timing.

## Corrected results (median of 3 repeats)
- V1→V2: **3.28×–3.35×** at ≥1 MiB (Phase 6 said 1.48×–3.04×)
- V2→V3: **0.64×–1.01×** — the Phase 6 claim of 1.22× at 16 MiB **does not reproduce**
- NCCL is **6–8× faster** at 1–32 KiB (Phase 6 called it a tie)
- Custom is 0.67×–0.78× of NCCL at ≥16 MiB (Phase 6 said 0.27×–0.45×)

## Correctness
105 rows, all `value_kind = measured`, **0 mismatches**.

## Files
- `results.jsonl` / `results.csv`, `harness_calibration.txt`
- `nvtx_*.csv`, `cudaapi_*.csv`, `kernels_*.csv` — extracted trace summaries
- `../../raw/p7a-harness-validation-20260829T032840Z-64c7410/` — verbatim evidence
