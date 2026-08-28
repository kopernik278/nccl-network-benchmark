#!/usr/bin/env python3
"""Convert custom Ring AllReduce output into schema-conformant JSONL and CSV.

The Phase 6 benchmark emits its own CSV (it is not nccl-tests), so this is a
sibling of parse_nccl_output.py rather than an extension of it. It reuses the
same experiment layout, the same env.json, the same schema, and the same
integrity fields, so Phase 6 rows sit alongside Phases 1-5 in one dataset.

    python3 scripts/parse_ring_output.py --raw-dir results/raw/<experiment-id>

Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

# Bus bandwidth correction for AllReduce over a ring of n ranks. Derived in the
# Phase 6 report: ReduceScatter moves (n-1)/n * M per rank and AllGather moves
# the same, so the collective moves 2(n-1)/n * M.
def busbw_factor(n: int) -> float:
    return 2.0 * (n - 1) / n


CSV_COLUMNS = [
    "schema_version", "experiment_id", "phase", "timestamp", "value_kind",
    "git_commit", "provider", "hostname", "os", "cpu_model",
    "node_count", "gpu_model", "gpu_count", "topology_summary", "network",
    "nvlink_present", "p2p_enabled", "transport",
    "cuda_version", "driver_version", "nccl_version", "compiler_version",
    "benchmark_tool", "config_label", "collective", "datatype",
    "message_size_bytes", "count_elements", "placement",
    "warmup_iterations", "measured_iterations", "repeat_index",
    "latency_us", "algorithmic_bandwidth_gbps", "bus_bandwidth_gbps",
    "wrong_count", "correctness_ok", "command", "raw_output_path", "notes",
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        print(f"warning: could not read {path}: {exc}", file=sys.stderr)
        return {}


CSV_HEADER_PREFIX = "impl,"


def parse_rows(text: str) -> tuple[list[dict[str, str]], list[str]]:
    """Split the benchmark output into its CSV table and its '#' preamble.

    The table is located by its header line rather than by "first non-comment
    line": NCCL_DEBUG=VERSION prints a banner to stdout that starts with
    neither '#' nor 'impl,', and feeding that to DictReader silently makes the
    banner the header and turns every row into garbage.
    """
    lines = text.splitlines()
    preamble = [l for l in lines if l.startswith("#")]
    start = next((i for i, l in enumerate(lines)
                  if l.startswith(CSV_HEADER_PREFIX)), None)
    if start is None:
        return [], preamble
    body = [l for l in lines[start:] if l and not l.startswith("#")]
    return list(csv.DictReader(body)), preamble


def build_rows(raw_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    env = load_json(raw_dir / "env.json")
    manifest = load_json(raw_dir / "run_manifest.json")
    problems: list[str] = []

    out_path = raw_dir / "ring_allreduce.stdout.txt"
    if not out_path.is_file():
        return [], [f"missing {out_path.name}"]
    table, preamble = parse_rows(out_path.read_text(encoding="utf-8", errors="replace"))
    if not table:
        return [], ["no CSV rows in benchmark output"]

    synthetic = "SYNTHETIC" in out_path.read_text(encoding="utf-8", errors="replace")
    value_kind = "synthetic" if synthetic else manifest.get("value_kind", "measured")
    if synthetic:
        problems.append("SYNTHETIC marker present -> value_kind=synthetic")

    experiment_id = manifest.get("experiment_id") or raw_dir.name
    repo_root = Path(__file__).resolve().parent.parent
    try:
        rel_raw = str(out_path.resolve().relative_to(repo_root))
    except ValueError:
        rel_raw = str(out_path)

    rows: list[dict[str, Any]] = []
    for rec in table:
        def num(key, cast=float, default=None):
            v = rec.get(key)
            if v is None or v == "":
                return default
            try:
                return cast(v)
            except ValueError:
                return default

        ranks = num("ranks", int) or manifest.get("gpu_count") or 1
        mismatches = num("mismatches", int, 0)
        latency = num("latency_us", float)
        # The benchmark writes -1 when it refused to time an incorrect run.
        if latency is not None and latency < 0:
            latency = None
        correct = (mismatches == 0) and latency is not None
        if mismatches:
            problems.append(f"{rec.get('impl')}@{rec.get('size_bytes')}B: "
                            f"{mismatches} mismatches")

        # Independently recompute bandwidth rather than trusting the C++ print.
        size = num("size_bytes", int, 0) or 0
        algbw = busbw = None
        if latency and latency > 0 and size:
            algbw = size / (latency * 1e-6) / 1e9
            busbw = algbw * busbw_factor(int(ranks))

        label = rec.get("impl", "")
        sub = num("subchunks", int, 0) or 0
        if label == "v3-pipelined" and sub:
            label = f"{label}-sub{sub}"

        rows.append({
            "schema_version": SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "phase": "phase6",
            "timestamp": manifest.get("created_at_utc") or env.get("captured_at_utc"),
            "value_kind": value_kind,
            "git_commit": manifest.get("git_commit") or env.get("git_commit"),
            "git_dirty": env.get("git_dirty"),

            "provider": env.get("provider"),
            "provider_instance_id": env.get("provider_instance_id"),
            "hostname": env.get("hostname"),
            "os": env.get("os"),
            "kernel": env.get("kernel"),
            "cpu_model": env.get("cpu_model"),
            "cpu_cores": env.get("cpu_cores"),
            "node_count": manifest.get("node_count", 1),
            "gpu_model": env.get("gpu_model"),
            "gpu_count": int(ranks),
            "gpus": env.get("gpus"),
            "topology": env.get("topology"),
            "topology_summary": env.get("topology_summary"),
            "network": manifest.get("network", "none-single-node"),
            "nvlink_present": env.get("nvlink_present"),
            "p2p_enabled": rec.get("transport") == "p2p-direct",
            "transport": rec.get("transport"),
            "transport_verified": True,   # printed by the binary from cudaDeviceCanAccessPeer
            "hosts": None, "rank_to_host": None, "ranks_per_node": None,
            "net_interface": None, "mpi_implementation": None, "launcher": None,

            "cuda_version": env.get("cuda_version"),
            "driver_version": env.get("driver_version"),
            "nccl_version": env.get("nccl_version"),
            "nccl_version_source": env.get("nccl_version_source"),
            "mpi_version": env.get("mpi_version"),
            "compiler_version": env.get("compiler_version"),
            "nccl_tests_commit": None,

            "benchmark_tool": ("nccl" if label == "nccl-reference"
                               else "custom-ring-allreduce"),
            "config_label": label,
            "nccl_algo": "Ring" if label != "nccl-reference" else None,
            "nccl_proto": None,
            "nccl_extra_env": None,
            "collective": "all_reduce",
            "datatype": "float",
            "redop": "sum",
            "root": None,
            "message_size_bytes": size,
            "count_elements": num("elements", int),
            "placement": "in_place",
            "warmup_iterations": num("warmup", int, 0) or 0,
            "measured_iterations": num("iters", int, 1) or 1,
            "repeat_index": 0,
            "tier": "full",

            "latency_us": latency,
            "algorithmic_bandwidth_gbps": algbw,
            "bus_bandwidth_gbps": busbw,

            "wrong_count": mismatches,
            "correctness_ok": bool(correct),
            "out_of_bounds_ok": mismatches == 0,
            "bandwidth_ratio_ok": None,
            "exit_code": manifest.get("exit_code"),

            "command": manifest.get("command") or "<unrecorded>",
            "env": env.get("env"),
            "raw_output_path": rel_raw,
            "notes": (f"max_abs_err={rec.get('max_abs_err')} "
                      f"bytes_moved_per_rank={rec.get('bytes_moved_per_rank')} "
                      f"bytes_expected_per_rank={rec.get('bytes_expected_per_rank')}"),
        })
    return rows, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    if not args.raw_dir.is_dir():
        print(f"error: no such directory: {args.raw_dir}", file=sys.stderr)
        return 2

    rows, problems = build_rows(args.raw_dir)
    if not rows:
        print("error: no rows parsed", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    out_dir = args.out_dir or (args.raw_dir.parent.parent / "summary" /
                               rows[0]["experiment_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    with (out_dir / "results.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    bad = sum(1 for r in rows if not r["correctness_ok"])
    impls = sorted({r["config_label"] for r in rows})
    transports = sorted({r["transport"] for r in rows})
    print(f"parsed {len(rows)} row(s) from {args.raw_dir}")
    print(f"  implementations  : {', '.join(impls)}")
    print(f"  transport        : {', '.join(t for t in transports if t)}")
    print(f"  value_kind       : {sorted({r['value_kind'] for r in rows})}")
    print(f"  correctness fail : {bad}")
    print(f"  -> {out_dir / 'results.jsonl'}")
    print(f"  -> {out_dir / 'results.csv'}")
    if problems:
        print(f"\n{len(problems)} problem(s):", file=sys.stderr)
        for p in problems[:20]:
            print(f"  - {p}", file=sys.stderr)
    return 1 if (args.strict and bad) else 0


if __name__ == "__main__":
    sys.exit(main())
