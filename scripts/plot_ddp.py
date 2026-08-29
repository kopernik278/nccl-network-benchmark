#!/usr/bin/env python3
"""Phase 9 figures: bucket-size effect on step time, and a measured step timeline.

Panel A plots end-to-end step time against DDP bucket capacity, with the
single-GPU compute floor and the serialised-reduction ceiling drawn in, so the
bucket effect is read against the range it actually lives in.

Panel B is the step timeline. Every span drawn is measured — backward duration
and the first AllReduce start come from the Nsight trace, the tail from the
same trace. Individual bucket-ready points are NOT drawn: their start offsets
were not exported before the pod was released, and inventing evenly spaced
markers would put an inference into a figure of measurements.

    python3 scripts/plot_ddp.py results/summary/p9-*/results.jsonl \
        --timelines results/raw/p9-*/profile-ddp-*.timeline.json \
        -o results/plots/p9-ddp.png
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from analyze_ddp import group, load

COLORS = {"bwd": "#2e7d32", "nccl": "#c1440e", "tail": "#8b2500", "idle": "#d8d8d8"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="+", type=Path)
    ap.add_argument("--timelines", nargs="*", type=Path, default=[])
    ap.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    cells = group(load(a.jsonl))
    ddp = {k: v for k, v in cells.items() if v["kind"] == "ddp-training"}
    single = next((v for v in cells.values() if v["kind"] == "single-gpu-training"), None)
    ser = next((v for v in cells.values() if v["kind"] == "serial-reduction"), None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # ---- Panel A: step time vs bucket capacity ----------------------------
    pts = sorted(((v["bucket_mb"], v["step_ms"], v["bucket_count"]) for v in ddp.values()),
                 key=lambda t: t[0])
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    ax1.plot(x, y, marker="o", color=COLORS["nccl"], zorder=3, label="DDP step time")
    for bx, by, bc in pts:
        ax1.annotate(f"{bc} buckets", (bx, by), textcoords="offset points",
                     xytext=(6, 6), fontsize=7.5)
    if ser:
        ax1.axhline(ser["step_ms"], color="#666", ls="--", lw=1,
                    label=f"serialised reduction ({ser['step_ms']:.0f} ms)")
    if single:
        ax1.axhline(single["step_ms"], color="#1f6fb4", ls=":", lw=1.2,
                    label=f"single-GPU compute floor ({single['step_ms']:.0f} ms)")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(v)}" for v in x])
    ax1.set_xlabel("DDP bucket_cap_mb (MiB, requested)")
    ax1.set_ylabel("training step time (ms)")
    ax1.set_title("Step time vs bucket capacity — 4×A40, SHM")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=7.5, loc="center right")

    # ---- Panel B: measured step timeline ----------------------------------
    tl = []
    for p in a.timelines:
        d = json.loads(p.read_text())
        m = re.search(r"ddp-(\d+)mb", p.name)
        if m and "backward_span_ms" in d:
            tl.append((int(m.group(1)), d))
    tl.sort(key=lambda t: t[0])

    for i, (mb, d) in enumerate(tl):
        bwd = d["backward_span_ms"]
        first = d["first_allreduce_start_ms_after_backward_start"]
        tail = d["nccl_after_backward_ms"]
        resident = d["nccl_resident_during_backward_ms"]
        y0 = i
        ax2.barh(y0 + 0.18, bwd, height=0.3, left=0, color=COLORS["bwd"],
                 label="backward compute" if i == 0 else None)
        # NCCL band: from the first collective to the end of the tail. The
        # collectives inside it are not contiguous; the hatched portion is the
        # measured resident total, drawn from the first start.
        ax2.barh(y0 - 0.18, (bwd - first) + tail, height=0.3, left=first,
                 color=COLORS["idle"],
                 label="AllReduce window (first start -> last end)" if i == 0 else None)
        ax2.barh(y0 - 0.18, resident, height=0.3, left=first, color=COLORS["nccl"],
                 label="AllReduce resident (measured total)" if i == 0 else None)
        ax2.barh(y0 - 0.18, tail, height=0.3, left=bwd, color=COLORS["tail"],
                 label="exposed tail after backward" if i == 0 else None)
        ax2.plot([first], [y0 - 0.18], marker="v", color="black", markersize=6, zorder=5)
        ax2.annotate(f"first bucket ready {first:+.1f} ms", (first, y0 + 0.45),
                     fontsize=7, ha="left")
        ax2.annotate(f"tail {tail:.1f} ms", (bwd + tail + 2.0, y0 - 0.23), fontsize=7,
                     va="center")

    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_ylim(-0.75, len(tl) - 0.15)
    ax2.set_xlim(-12, max(d["backward_span_ms"] + d["nccl_after_backward_ms"]
                          for _m, d in tl) * 1.22)
    ax2.set_yticks(range(len(tl)))
    ax2.set_yticklabels([f"{mb} MiB\n({d['nccl_kernels_in_window']} collectives)"
                         for mb, d in tl], fontsize=8)
    ax2.set_xlabel("ms, relative to the start of backward")
    ax2.set_title("One measured steady-state step (rank 0, Nsight trace)")
    ax2.grid(axis="x", alpha=0.25)
    ax2.legend(handles=[Patch(color=COLORS["bwd"], label="backward compute"),
                        Patch(color=COLORS["nccl"], label="AllReduce resident (total)"),
                        Patch(color=COLORS["tail"], label="exposed tail"),
                        Patch(color=COLORS["idle"], label="AllReduce window")],
               fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13),
               ncol=4, frameon=False)

    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
