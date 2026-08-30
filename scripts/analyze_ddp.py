#!/usr/bin/env python3
"""Phase 9: summarise the DDP bucket-size study.

The primary target is end-to-end training step time. Overlap quality is a
diagnostic, so it is reported next to the step time rather than instead of it.

Two families of repeat are kept apart on purpose. The three repeats inside one
process share a warm NCCL communicator, a warm allocator and one memory layout,
so their spread UNDERSTATES run-to-run noise. The `relaunch-*` rows are
independent torchrun launches, and those are what the noise floor is taken from.

Plateau rule (stated, not assumed): the noise floor is the largest spread
observed across independent relaunches of any bucket capacity. A gap smaller
than that floor is not resolvable; a gap larger than it is real, which is not
the same as being large enough to care about, so both the absolute gap and the
percentage are printed and the verdict is left to the reader.

    python3 scripts/analyze_ddp.py results/summary/p9-*/results.jsonl

Standard library only.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def group(rows: list[dict], relaunch: bool = False) -> dict[str, dict]:
    """One entry per configuration, medians over repeats plus the spread.

    DDP rows are keyed by bucket capacity rather than by label so that the four
    bucket sizes line up; `relaunch` selects the independent-launch family
    instead of the in-process one.
    """
    by: dict[str, list[dict]] = {}
    for r in rows:
        label = str(r["config_label"])
        is_relaunch = label.startswith("relaunch-")
        if r["workload_kind"] == "ddp-training":
            if is_relaunch != relaunch:
                continue
            key = f"{r['bucket_cap_mb']:g} MiB"
        elif relaunch:
            continue
        else:
            key = label
        by.setdefault(key, []).append(r)

    out: dict[str, dict] = {}
    for label, g in by.items():
        def col(f):
            return [x[f] for x in g if x.get(f) is not None]

        step = col("step_time_ms")
        if not step:
            continue
        wall = col("step_wall_ms") or step
        out[label] = {
            "kind": g[0]["workload_kind"],
            "bucket_mb": g[0].get("bucket_cap_mb"),
            "n": len(g),
            "step_ms": statistics.median(step),
            "step_spread_ms": max(step) - min(step),
            "wall_ms": statistics.median(wall),
            "fwd_ms": statistics.median(col("forward_ms")),
            "bwd_ms": statistics.median(col("backward_ms")),
            "opt_ms": statistics.median(col("optimizer_ms")),
            "tokens_per_s": statistics.median(col("tokens_per_s")),
            "bwd_nosync_ms": (statistics.median(col("backward_nosync_ms"))
                              if col("backward_nosync_ms") else None),
            "sync_cost_ms": (statistics.median(col("sync_cost_ms"))
                             if col("sync_cost_ms") else None),
            "buckets": g[0].get("ddp_bucket_sizes"),
            "bucket_count": g[0].get("ddp_bucket_count"),
            "grad_bytes": g[0].get("gradient_bytes"),
            "tokens_per_step": g[0].get("tokens_per_step"),
            "gpus": g[0].get("gpu_count"),
        }
    return out


def bucket_table(cells: dict[str, dict]) -> str:
    ddp = {k: v for k, v in cells.items() if v["kind"] == "ddp-training"}
    head = (f"{'bucket':>8} {'buckets':>8} {'step ms':>9} {'±spread':>8} {'fwd':>7} "
            f"{'bwd':>8} {'bwd(no-sync)':>13} {'sync cost':>10} {'opt':>7} {'tok/s':>10}")
    lines = [head, "-" * len(head)]
    for label, v in sorted(ddp.items(), key=lambda kv: kv[1]["bucket_mb"] or 0):
        lines.append(
            f"{(v['bucket_mb'] or 0):>7.0f}M {(v['bucket_count'] or 0):>8} "
            f"{v['step_ms']:>9.2f} {v['step_spread_ms']:>8.2f} {v['fwd_ms']:>7.2f} "
            f"{v['bwd_ms']:>8.2f} "
            f"{(v['bwd_nosync_ms'] if v['bwd_nosync_ms'] is not None else float('nan')):>13.2f} "
            f"{(v['sync_cost_ms'] if v['sync_cost_ms'] is not None else float('nan')):>10.2f} "
            f"{v['opt_ms']:>7.2f} {v['tokens_per_s']:>10,.0f}")
    return "\n".join(lines)


def plateau(cells: dict[str, dict]) -> tuple[float, list[str], str]:
    ddp = {k: v for k, v in cells.items() if v["kind"] == "ddp-training"}
    if not ddp:
        return 0.0, [], "no DDP rows"
    noise = max((v["step_spread_ms"] for v in ddp.values()), default=0.0)
    best_label = min(ddp, key=lambda k: ddp[k]["step_ms"])
    best = ddp[best_label]["step_ms"]
    inside = sorted((k for k, v in ddp.items() if v["step_ms"] - best <= noise),
                    key=lambda k: ddp[k]["bucket_mb"] or 0)
    worst = max(ddp.values(), key=lambda v: v["step_ms"])
    verdict = (f"noise floor (largest repeat spread) = {noise:.2f} ms; "
               f"best = {best_label} at {best:.2f} ms; "
               f"worst = {worst['step_ms']:.2f} ms "
               f"({(worst['step_ms'] / best - 1) * 100:.1f} % slower)")
    return noise, inside, verdict


def bucket_distribution(cells: dict[str, dict]) -> str:
    lines = [f"{'bucket cap':>10} {'count':>6} {'min MiB':>9} {'median MiB':>11} "
             f"{'max MiB':>9} {'total MiB':>10}"]
    lines.append("-" * len(lines[0]))
    for label, v in sorted(cells.items(), key=lambda kv: kv[1]["bucket_mb"] or 0):
        if v["kind"] != "ddp-training" or not v["buckets"]:
            continue
        b = [x / (1 << 20) for x in v["buckets"]]
        lines.append(f"{(v['bucket_mb'] or 0):>9.0f}M {len(b):>6} {min(b):>9.2f} "
                     f"{statistics.median(b):>11.2f} {max(b):>9.2f} {sum(b):>10.1f}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="+", type=Path)
    ap.add_argument("--baseline", type=float, default=None,
                    help="bucket_cap_mb of the reference configuration")
    ap.add_argument("--optimized", type=float, default=None,
                    help="bucket_cap_mb of the configuration being proposed")
    a = ap.parse_args()

    rows = load(a.jsonl)
    cells = group(rows)
    relaunches = group(rows, relaunch=True)
    print(f"# rows={len(rows)} configurations={len(cells)} "
          f"relaunch groups={len(relaunches)}\n")

    single = next((v for v in cells.values() if v["kind"] == "single-gpu-training"), None)
    if single:
        print("## single-GPU reference (compute-only)\n")
        print(f"  step {single['step_ms']:.2f} ms  fwd {single['fwd_ms']:.2f}  "
              f"bwd {single['bwd_ms']:.2f}  opt {single['opt_ms']:.2f}  "
              f"{single['tokens_per_s']:,.0f} tok/s "
              f"({single['tokens_per_step']:,} tokens/step)\n")

    print("## DDP bucket matrix\n")
    print(bucket_table(cells))

    print("\n## actual bucket sizes DDP used (not the requested capacity)\n")
    print(bucket_distribution(cells))

    if relaunches:
        print("\n## across-launch variance (independent torchrun launches)\n")
        w = f"{'bucket':>8} {'launches':>9} {'median ms':>10} {'spread ms':>10}"
        print(w); print("-" * len(w))
        for k, v in sorted(relaunches.items(), key=lambda kv: kv[1]["bucket_mb"] or 0):
            print(f"{k:>8} {v['n']:>9} {v['step_ms']:>10.2f} {v['step_spread_ms']:>10.2f}")

    noise, _inside, _v = plateau(relaunches or cells)
    ddp = {k: v for k, v in cells.items() if v["kind"] == "ddp-training"}
    print("\n## plateau analysis\n")
    print(f"  noise floor (largest spread across independent relaunches) = {noise:.2f} ms")
    if ddp:
        best_k = min(ddp, key=lambda k: ddp[k]["step_ms"])
        best = ddp[best_k]["step_ms"]
        print(f"  reference = {best_k} at {best:.2f} ms")
        for k, v in sorted(ddp.items(), key=lambda kv: kv[1]["bucket_mb"] or 0):
            d = v["step_ms"] - best
            verdict = "within noise" if d <= noise else "resolvable"
            print(f"    {k:>8}: {d:+7.2f} ms ({d / best * 100:+5.2f} %)  {verdict}")

    ser = next((v for v in cells.values() if v["kind"] == "serial-reduction"), None)
    if ser:
        ddp = {k: v for k, v in cells.items() if v["kind"] == "ddp-training"}
        best = min(ddp.values(), key=lambda v: v["step_ms"]) if ddp else None
        print("\n## overlap benefit — DDP vs a deliberately serialised reduction\n")
        print(f"  serialised : step {ser['step_ms']:.2f} ms  {ser['tokens_per_s']:,.0f} tok/s")
        if best:
            print(f"  best DDP   : step {best['step_ms']:.2f} ms  {best['tokens_per_s']:,.0f} tok/s")
            print(f"  overlap saves {ser['step_ms'] - best['step_ms']:.2f} ms/step "
                  f"({(1 - best['step_ms'] / ser['step_ms']) * 100:.1f} %)")

    if a.baseline is not None and a.optimized is not None:
        ddp = {v["bucket_mb"]: v for v in cells.values() if v["kind"] == "ddp-training"}
        b, o = ddp.get(a.baseline), ddp.get(a.optimized)
        if b and o:
            print(f"\n## baseline ({a.baseline:g} MiB) -> optimized ({a.optimized:g} MiB)\n")

            def delta(name, bv, ov, unit="", higher_is_better=False):
                if bv is None or ov is None:
                    print(f"  {name:<28} n/a")
                    return
                pct = (ov - bv) / bv * 100.0
                arrow = "better" if ((pct > 0) == higher_is_better) else "worse"
                if abs(pct) < 1e-9:
                    arrow = "unchanged"
                print(f"  {name:<28} {bv:>10.2f}{unit} -> {ov:>10.2f}{unit}  "
                      f"{pct:+6.2f} %  ({arrow})")

            delta("step time", b["step_ms"], o["step_ms"], " ms")
            delta("throughput", b["tokens_per_s"], o["tokens_per_s"], " tok/s",
                  higher_is_better=True)
            delta("sync cost (comm penalty)", b["sync_cost_ms"], o["sync_cost_ms"], " ms")
            delta("backward", b["bwd_ms"], o["bwd_ms"], " ms")
            print(f"  {'collectives per step':<28} {b['bucket_count']:>10} -> "
                  f"{o['bucket_count']:>10}         "
                  f"({o['bucket_count'] / b['bucket_count']:.1f}x more calls — the cost side)")
            if single:
                eb = b["tokens_per_s"] / (b["gpus"] * single["tokens_per_s"])
                eo = o["tokens_per_s"] / (o["gpus"] * single["tokens_per_s"])
                print(f"  {'scaling efficiency':<28} {eb * 100:>10.2f} % -> "
                      f"{eo * 100:>10.2f} %  {(eo - eb) * 100:+6.2f} pp")

    if single:
        ddp = {k: v for k, v in cells.items() if v["kind"] == "ddp-training"}
        if ddp:
            best = max(ddp.values(), key=lambda v: v["tokens_per_s"])
            n = best["gpus"]
            eff = best["tokens_per_s"] / (n * single["tokens_per_s"])
            print(f"\n## scaling\n")
            print(f"  per-GPU batch held constant, so the {n}-GPU global batch is "
                  f"{n}x the single-GPU batch.")
            print(f"  {n}-GPU {best['tokens_per_s']:,.0f} tok/s vs "
                  f"{n} x {single['tokens_per_s']:,.0f} = "
                  f"{n * single['tokens_per_s']:,.0f} tok/s ideal")
            print(f"  scaling efficiency = {eff * 100:.1f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
