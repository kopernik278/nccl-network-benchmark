#!/usr/bin/env python3
"""Phase 8: turn overlap rows into a communication/compute contention view.

The overlap benchmark records, per case, both the standalone duration of each
stream's work and the duration that same work took while the other stream ran
(measured with CUDA events on the stream itself, not inferred from the wall
clock). The ratio of those two numbers is the only interference number this
project is willing to report:

    compute_slowdown = compute_during_overlap_us / t_compute_us
    comm_slowdown    = comm_during_overlap_us    / t_comm_us

A slowdown of 1.00 means the stream was not measurably disturbed. Nothing here
attributes a slowdown to a particular hardware resource — that argument needs
evidence this script does not have, and belongs in the experiment report.

    python3 scripts/analyze_contention.py results/summary/p8-*/results.jsonl

Standard library only.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

SIZE_NAMES = {1 << 20: "1 MiB", 16 << 20: "16 MiB", 128 << 20: "128 MiB"}


def size_name(b: int) -> str:
    return SIZE_NAMES.get(b, f"{b / (1 << 20):g} MiB")


def load(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def ratio_of(row: dict) -> float | None:
    """The ratio target lives in `notes` as written by the parser."""
    for tok in (row.get("notes") or "").split():
        if tok.startswith("ratio_target="):
            try:
                return float(tok.split("=", 1)[1])
            except ValueError:
                return None
    return None


def med(xs: list[float]) -> float:
    return statistics.median(xs)


def aggregate(rows: list[dict]) -> dict[tuple, dict]:
    """Group repeats of one (size, workload, ratio) cell and take medians."""
    cells: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["message_size_bytes"], r.get("workload_class") or "?", ratio_of(r))
        cells.setdefault(key, []).append(r)

    out: dict[tuple, dict] = {}
    for key, group in sorted(cells.items()):
        def g(field: str) -> list[float]:
            return [x[field] for x in group if x.get(field) is not None]

        t_comp, t_comm = g("t_compute_us"), g("t_comm_us")
        c_ov, m_ov = g("compute_during_overlap_us"), g("comm_during_overlap_us")
        if not (t_comp and t_comm and c_ov and m_ov):
            continue
        out[key] = {
            "n": len(group),
            "t_compute_us": med(t_comp),
            "t_comm_us": med(t_comm),
            "t_seq_us": med(g("t_seq_us")),
            "t_overlap_us": med(g("t_overlap_us")),
            "compute_slowdown": med(c_ov) / med(t_comp),
            "comm_slowdown": med(m_ov) / med(t_comm),
            "overlap_efficiency": med(g("overlap_efficiency")) if g("overlap_efficiency") else None,
            "exposed_comm_us": med(g("exposed_comm_us")),
            "compute_reps": group[0].get("compute_reps"),
        }
    return out


def fmt_table(cells: dict[tuple, dict], with_ratio: bool) -> str:
    head = (f"{'message':>9}  {'workload':<13} " + (f"{'ratio':>6} " if with_ratio else "")
            + f"{'reps':>5} {'t_comp':>9} {'t_comm':>9} {'comp x':>7} {'comm x':>7} "
              f"{'ovl_eff':>8} {'exposed':>9}")
    lines = [head, "-" * len(head)]
    for (size, wl, ratio), v in cells.items():
        lines.append(
            f"{size_name(size):>9}  {wl:<13} "
            + (f"{ratio if ratio is not None else float('nan'):>6.2f} " if with_ratio else "")
            + f"{v['compute_reps'] or 0:>5} {v['t_compute_us']:>9.1f} {v['t_comm_us']:>9.1f} "
              f"{v['compute_slowdown']:>7.2f} {v['comm_slowdown']:>7.2f} "
              f"{(v['overlap_efficiency'] if v['overlap_efficiency'] is not None else float('nan')):>8.3f} "
              f"{v['exposed_comm_us']:>9.1f}")
    return "\n".join(lines)


def collective_dram_share(size_bytes: int, ranks: int, t_comm_us: float,
                          achievable_gbps: float) -> tuple[float, float]:
    """Traffic a ring AllReduce forces through each rank's own memory.

    Ring AllReduce = reduce-scatter + all-gather: each rank sends and receives
    2(N-1)/N of the buffer. Counting one device-memory read plus one write per
    byte that crosses the rank boundary is the conservative-high estimate, so
    the resulting share is an upper bound, not a measurement.
    """
    moved = 2.0 * (ranks - 1) / ranks * size_bytes
    dram_bytes = 2.0 * moved
    gbps = dram_bytes / (t_comm_us * 1e-6) / 1e9
    return gbps, gbps / achievable_gbps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="+", type=Path)
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--achievable-dram-gbps", type=float, default=460.0,
                    help="measured standalone triad bandwidth, GB/s (10^9 B/s)")
    a = ap.parse_args()

    rows = load(a.jsonl)
    cells = aggregate(rows)
    single = {k: v for k, v in cells.items() if k[2] in (None, 1.0)}
    swept = {k: v for k, v in cells.items() if k[2] not in (None, 1.0)}

    print(f"# rows={len(rows)} cells={len(cells)}\n")
    print("## interference at matched duration (ratio 1.0)\n")
    print(fmt_table(single, with_ratio=False))
    if swept:
        print("\n## compute-intensity sweep\n")
        print(fmt_table(cells, with_ratio=True))

    print("\n## upper bound on the collective's own device-memory traffic")
    print(f"# ranks={a.ranks}, achievable DRAM bandwidth taken as {a.achievable_dram_gbps:g} GB/s")
    seen = set()
    for (size, _wl, ratio), v in cells.items():
        if size in seen:
            continue
        seen.add(size)
        gbps, share = collective_dram_share(size, a.ranks, v["t_comm_us"],
                                            a.achievable_dram_gbps)
        print(f"{size_name(size):>9}: t_comm {v['t_comm_us']:>9.1f} us -> "
              f"<= {gbps:6.2f} GB/s ({share * 100:5.2f} % of achievable DRAM bandwidth)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
