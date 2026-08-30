#!/usr/bin/env python3
"""Phase 10 figure: the final configuration, and why it wins.

Panel A puts the bucket sweep on both transports the preflight examined, between
the single-GPU compute floor and the non-overlapped ceiling. Plotting both is
the point: the bucket effect is real on each, but its size depends on how slow
the collective is, and that is the generalisation boundary of the result.

Panel B is the measured step timeline on the transport the preflight selected.
Only measured spans are drawn — backward, the first AllReduce start, aggregate
AllReduce residency, and the tail. Individual bucket-ready points are not drawn.

    python3 scripts/plot_final_ddp.py results/summary/p10-*/results.jsonl \
        --timelines results/raw/p10-*/profile-final-*.timeline.json \
        -o results/plots/p10-final.png
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from analyze_ddp import group, load

C = {"p2p": "#c1440e", "shm": "#1f6fb4", "bwd": "#2e7d32",
     "nccl": "#c1440e", "tail": "#8b2500", "idle": "#d8d8d8"}


def series(cells: dict) -> tuple[list[float], list[float], list[int]]:
    pts = sorted((v["bucket_mb"], v["step_ms"], v["bucket_count"])
                 for v in cells.values() if v["kind"] == "ddp-training")
    return [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]


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

    rows = load(a.jsonl)
    p2p = group(rows, prefix="final-")
    shm = group(rows, prefix="shm-")
    single = next((v for v in p2p.values() if v["kind"] == "single-gpu-training"), None)
    ser = next((v for v in p2p.values() if v["kind"] == "serial-reduction"), None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    for cells, key, label in ((p2p, "p2p", "P2P + SHM (preflight-selected)"),
                              (shm, "shm", "SHM only (P2P disabled)")):
        x, y, n = series(cells)
        if not x:
            continue
        ax1.plot(x, y, marker="o", color=C[key], zorder=3, label=label)
        for bx, by, bn in zip(x, y, n):
            ax1.annotate(f"{bn}", (bx, by), textcoords="offset points",
                         xytext=(5, -11), fontsize=7, color=C[key])
    if ser:
        ax1.axhline(ser["step_ms"], color="#666", ls="--", lw=1,
                    label=f"non-overlapped control ({ser['step_ms']:.0f} ms)")
    if single:
        ax1.axhline(single["step_ms"], color="#333", ls=":", lw=1.2,
                    label=f"single-GPU compute floor ({single['step_ms']:.0f} ms)")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks([4, 25, 64])
    ax1.set_xticklabels(["4", "25", "64"])
    ax1.set_xlabel("DDP bucket_cap_mb (MiB, requested) — labels are collectives/step")
    ax1.set_ylabel("training step time (ms)")
    ax1.set_title("Final configurations, both transports — 4×A40")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=7.5, loc="lower right")

    tl = []
    for p in a.timelines:
        d = json.loads(p.read_text())
        m = re.search(r"final-(\d+)mb", p.name)
        if m and "backward_span_ms" in d:
            tl.append((int(m.group(1)), d))
    tl.sort(key=lambda t: t[0])

    for i, (mb, d) in enumerate(tl):
        bwd = d["backward_span_ms"]
        first = d["first_allreduce_start_ms_after_backward_start"]
        tail = d["nccl_after_backward_ms"]
        resident = d["nccl_resident_during_backward_ms"]
        ax2.barh(i + 0.18, bwd, height=0.3, left=0, color=C["bwd"])
        ax2.barh(i - 0.18, (bwd - first) + tail, height=0.3, left=first, color=C["idle"])
        ax2.barh(i - 0.18, resident, height=0.3, left=first, color=C["nccl"])
        ax2.barh(i - 0.18, tail, height=0.3, left=bwd, color=C["tail"])
        ax2.plot([first], [i - 0.18], marker="v", color="black", markersize=6, zorder=5)
        ax2.annotate(f"first bucket ready {first:+.1f} ms", (first, i + 0.44), fontsize=7)
        ax2.annotate(f"tail {tail:.1f} ms", (bwd + tail + 2.0, i - 0.23),
                     fontsize=7, va="center")

    ax2.axvline(0, color="black", lw=0.8)
    if tl:
        ax2.set_ylim(-0.75, len(tl) - 0.15)
        ax2.set_xlim(-6, max(d["backward_span_ms"] + d["nccl_after_backward_ms"]
                             for _m, d in tl) * 1.26)
        ax2.set_yticks(range(len(tl)))
        ax2.set_yticklabels([f"{mb} MiB\n({d['nccl_kernels_in_window']} collectives)"
                             for mb, d in tl], fontsize=8)
    ax2.set_xlabel("ms, relative to the start of backward")
    ax2.set_title("Measured step timeline on the selected transport")
    ax2.grid(axis="x", alpha=0.25)
    ax2.legend(handles=[Patch(color=C["bwd"], label="backward compute"),
                        Patch(color=C["nccl"], label="AllReduce resident (total)"),
                        Patch(color=C["tail"], label="exposed tail"),
                        Patch(color=C["idle"], label="AllReduce window")],
               fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13),
               ncol=4, frameon=False)

    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
