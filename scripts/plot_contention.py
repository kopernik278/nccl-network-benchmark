#!/usr/bin/env python3
"""Phase 8 figure: how overlap quality and interference vary with the workload.

Reads the same summary JSONL as analyze_contention.py and draws the 128 MiB
intensity series for each workload class. Requires matplotlib; every number
plotted is measured, so the script refuses to invent points for a missing cell.

    python3 scripts/plot_contention.py results/summary/p8-*/results.jsonl \
        -o results/plots/p8-contention.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

from analyze_contention import aggregate, load

STYLE = {
    "compute-gemm": ("#c1440e", "o", "compute-gemm (small SGEMM, AI ~85)"),
    "memory-triad": ("#1f6fb4", "s", "memory-triad (AI ~0.17)"),
    "mixed":        ("#2e7d32", "^", "mixed (AI ~8)"),
}
SIZE = 128 << 20


def series(cells: dict, workload: str, field: str) -> tuple[list[float], list[float]]:
    pts = sorted((k[2], v[field]) for k, v in cells.items()
                 if k[0] == SIZE and k[1] == workload and k[2] is not None)
    return [p[0] for p in pts], [p[1] for p in pts]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = aggregate(load(a.jsonl))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for wl, (color, marker, label) in STYLE.items():
        x, y = series(cells, wl, "overlap_efficiency")
        if x:
            ax1.plot(x, y, marker=marker, color=color, label=label)
        x, y = series(cells, wl, "compute_slowdown")
        if x:
            ax2.plot(x, y, marker=marker, color=color, label=f"{wl} compute")
        x, y = series(cells, wl, "comm_slowdown")
        if x:
            ax2.plot(x, y, marker=marker, color=color, linestyle="--", alpha=0.7,
                     label=f"{wl} comm")

    ax1.set_xlabel("compute : communication duration ratio")
    ax1.set_ylabel("overlap efficiency")
    ax1.set_title("Overlap quality — 128 MiB AllReduce, SHM, 4×A40")
    ax1.set_ylim(0.4, 1.05)
    ax1.axhline(1.0, color="grey", lw=0.6, ls=":")
    ax1.legend(fontsize=7.5, loc="lower left")
    ax1.grid(alpha=0.25)

    ax2.set_xlabel("compute : communication duration ratio")
    ax2.set_ylabel("slowdown under concurrency  (×)")
    ax2.set_title("Interference — solid: compute, dashed: comm")
    ax2.axhline(1.0, color="grey", lw=0.6, ls=":")
    ax2.legend(fontsize=7, ncol=2, loc="upper right")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
