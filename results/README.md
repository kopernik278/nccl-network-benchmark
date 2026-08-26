# Results Layout

Contains the Phase 1 baseline measured on 2026-08-26
(`p1-nccl-baseline-20260826T181721Z-16cea6a`): 2 × NVIDIA L4, single node,
AllReduce / AllGather / ReduceScatter, 468 rows, zero validation errors.

## Structure

```text
results/raw/<experiment-id>/       verbatim evidence — append-only, never edited
    env.json                       machine-readable environment metadata
    env.txt                        human-readable environment metadata
    run_manifest.json              every command, exit code, and timing
    <collective>.<tier>.r<N>.stdout.txt
    <collective>.<tier>.r<N>.stderr.txt
    nccl_debug_info.txt            NCCL_DEBUG=INFO diagnostic run

results/summary/<experiment-id>/   derived — regenerable from raw
    results.jsonl                  canonical, one row per measurement point
    results.csv                    same rows, flat
    summary.md                     written during analysis
```

## Rules

- **Raw is never modified.** It is the evidence a result rests on. Everything
  in `summary/` can be rebuilt by re-running the parser against `raw/`.
- **Results are never overwritten.** Each run gets a new experiment ID; the
  runner refuses to write into an existing directory.
- **Failed runs are kept.** A run that fails its correctness gate stays on
  disk with its logs, so the failure remains diagnosable.

## Experiment ID

```
p1-<slug>-<UTC timestamp>-<git short SHA>
e.g. p1-nccl-baseline-20260826T141530Z-b1f095c
```

## Regenerating a summary

```bash
python3 scripts/parse_nccl_output.py --raw-dir results/raw/<experiment-id>
```

## What is committed

Small text artifacts under `raw/` (`*.txt`, `*.json`) are committed, because
reproducibility requires the raw evidence to travel with the repository.
Large binaries — Nsight traces, archives, dumps — are excluded by
`.gitignore`; profiling traces belong in `profiles/raw/`, which is fully
ignored.

Schema: `docs/design/RFC-001-result-schema.md` and
`schemas/nccl_result.schema.json`.
