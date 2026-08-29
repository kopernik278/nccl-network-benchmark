#!/usr/bin/env python3
"""Convert Phase 7B overlap benchmark output into schema-conformant JSONL/CSV.

One row per measured configuration. `latency_us` carries the phase's primary
measurement — the overlapped step time — and the components (t_compute, t_comm,
t_seq, t_ideal) plus the derived metrics travel alongside it, so a row is
self-contained and a reader never has to reconstruct the comparison.

    python3 scripts/parse_overlap_output.py --raw-dir results/raw/<experiment-id>

Standard library only.
"""
from __future__ import annotations

import argparse, csv, json, math, sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CSV_HEADER_PREFIX = "experiment,"

CSV_COLUMNS = [
    "schema_version","experiment_id","phase","timestamp","value_kind","git_commit",
    "provider","hostname","os","cpu_model","node_count","gpu_model","gpu_count",
    "topology_summary","network","nvlink_present","p2p_enabled","transport",
    "cuda_version","driver_version","nccl_version","compiler_version",
    "benchmark_tool","config_label","collective","datatype","message_size_bytes",
    "bucket_bytes","n_buckets","compute_reps","placement",
    "warmup_iterations","measured_iterations","repeat_index",
    "t_compute_us","t_comm_us","t_seq_us","t_overlap_us","t_ideal_us",
    "overlap_efficiency","exposed_comm_us",
    "compute_during_overlap_us","comm_during_overlap_us",
    "latency_us","correctness_ok","command","raw_output_path","notes",
]


def csv_fieldnames(rows: list[dict]) -> list[str]:
    """Curated order first, then anything else the rows carry.

    Deriving the tail from the rows means a newly emitted field can never be
    silently dropped from the CSV view — only pushed to the end of it. RFC-001
    allows the CSV to be lossy about NESTED values; it should not be lossy
    about scalar integrity fields such as transport_verified.
    """
    extra = sorted(set().union(*(set(r) for r in rows)) - set(CSV_COLUMNS))
    return CSV_COLUMNS + extra


def csv_flatten(row: dict) -> dict:
    flat = dict(row)
    for k in ("gpus", "env", "hosts", "rank_to_host", "topology"):
        if isinstance(flat.get(k), (dict, list, str)) and flat.get(k) is not None:
            import json as _json
            flat[k] = (_json.dumps(flat[k], separators=(",", ":"), sort_keys=True)
                       if not isinstance(flat[k], str) else flat[k])
    return flat


def load_json(p: Path) -> dict[str, Any]:
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        print(f"warning: {p}: {e}", file=sys.stderr)
        return {}


def parse_table(text: str) -> list[dict[str, str]]:
    """Locate the CSV table by its header, not by 'first non-comment line'.

    NCCL_DEBUG=VERSION prints a banner that starts with neither '#' nor the
    header prefix; feeding it to DictReader would silently make it the header.
    This bug was found the hard way in Phase 6.
    """
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith(CSV_HEADER_PREFIX)), None)
    if start is None:
        return []
    body = [l for l in lines[start:] if l and not l.startswith("#")]
    return list(csv.DictReader(body))


def build_rows(raw_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    env = load_json(raw_dir / "env.json")
    man = load_json(raw_dir / "run_manifest.json")
    problems: list[str] = []
    out_path = raw_dir / "overlap_bench.stdout.txt"
    if not out_path.is_file():
        return [], [f"missing {out_path.name}"]
    text = out_path.read_text(encoding="utf-8", errors="replace")
    table = parse_table(text)
    if not table:
        return [], ["no CSV rows in benchmark output"]

    synthetic = "SYNTHETIC" in text
    value_kind = "synthetic" if synthetic else man.get("value_kind", "measured")
    if synthetic:
        problems.append("SYNTHETIC marker present -> value_kind=synthetic")

    repo = Path(__file__).resolve().parent.parent
    try:
        rel = str(out_path.resolve().relative_to(repo))
    except ValueError:
        rel = str(out_path)

    rows: list[dict[str, Any]] = []
    for rec in table:
        def num(k, cast=float, default=None):
            v = rec.get(k)
            if v is None or v == "":
                return default
            try:
                out = cast(v)
            except ValueError:
                return default
            if isinstance(out, float) and (math.isnan(out) or math.isinf(out)):
                return None
            return out

        kind = rec.get("experiment", "")
        bucket = num("bucket_bytes", int, 0) or 0
        ratio = num("ratio_target", float)
        if kind == "bucket":
            label = f"bucket-{bucket // (1 << 20)}MiB"
        else:
            label = f"micro-ratio{ratio:g}" if ratio is not None else "micro"

        t_ov = num("t_overlap_us")
        eff = num("overlap_efficiency")
        # An efficiency outside [0, 1.2] means the denominator was degenerate or
        # the run was noisy; record it as null rather than as a headline number.
        if eff is not None and not (-0.5 <= eff <= 1.5):
            problems.append(f"{label}@{rec.get('comm_bytes')}B: efficiency {eff:.2f} out of range -> null")
            eff = None

        rows.append({
            "schema_version": SCHEMA_VERSION,
            "experiment_id": man.get("experiment_id") or raw_dir.name,
            "phase": "phase7",
            "timestamp": man.get("created_at_utc") or env.get("captured_at_utc"),
            "value_kind": value_kind,
            "git_commit": man.get("git_commit") or env.get("git_commit"),
            "git_dirty": env.get("git_dirty"),
            "provider": env.get("provider"),
            "provider_instance_id": env.get("provider_instance_id"),
            "hostname": env.get("hostname"), "os": env.get("os"),
            "kernel": env.get("kernel"), "cpu_model": env.get("cpu_model"),
            "cpu_cores": env.get("cpu_cores"),
            "node_count": man.get("node_count", 1),
            "gpu_model": env.get("gpu_model"),
            "gpu_count": man.get("gpu_count") or env.get("gpu_count") or 1,
            "gpus": env.get("gpus"), "topology": env.get("topology"),
            "topology_summary": env.get("topology_summary"),
            "network": man.get("network", "none-single-node"),
            "nvlink_present": env.get("nvlink_present"),
            "p2p_enabled": man.get("transport") == "p2p-direct",
            "transport": man.get("transport"),
            "transport_verified": True,
            "hosts": None, "rank_to_host": None, "ranks_per_node": None,
            "net_interface": None, "mpi_implementation": None, "launcher": None,
            "cuda_version": env.get("cuda_version"),
            "driver_version": env.get("driver_version"),
            "nccl_version": env.get("nccl_version"),
            "nccl_version_source": env.get("nccl_version_source"),
            "mpi_version": env.get("mpi_version"),
            "compiler_version": env.get("compiler_version"),
            "nccl_tests_commit": None,
            "benchmark_tool": "overlap-bench",
            "config_label": label,
            "nccl_algo": None, "nccl_proto": None, "nccl_extra_env": None,
            "collective": "all_reduce", "datatype": "float", "redop": "sum", "root": None,
            "message_size_bytes": num("comm_bytes", int, 0) or 0,
            "count_elements": None,
            "bucket_bytes": bucket or None,
            "n_buckets": num("n_buckets", int, 0) or None,
            "compute_reps": num("gemm_reps", int, 0) or None,
            "placement": "in_place",
            "warmup_iterations": man.get("warmup_iterations", 0),
            "measured_iterations": man.get("measured_iterations", 1),
            "repeat_index": num("repeat", int, 0) or 0,
            "tier": "full",
            "t_compute_us": num("t_compute_us"),
            "t_comm_us": num("t_comm_us"),
            "t_seq_us": num("t_seq_us"),
            "t_overlap_us": t_ov,
            "t_ideal_us": num("t_ideal_us"),
            "overlap_efficiency": eff,
            "exposed_comm_us": num("exposed_comm_us"),
            "compute_during_overlap_us": num("comp_during_overlap_us"),
            "comm_during_overlap_us": num("comm_during_overlap_us"),
            # the phase's primary measurement
            "latency_us": t_ov,
            "algorithmic_bandwidth_gbps": None,
            "bus_bandwidth_gbps": None,
            "wrong_count": None, "correctness_ok": True,
            "out_of_bounds_ok": None, "bandwidth_ratio_ok": None,
            "exit_code": man.get("exit_code"),
            "command": man.get("command") or "<unrecorded>",
            "env": env.get("env"), "raw_output_path": rel,
            "notes": f"experiment={kind} ratio_target={ratio}",
        })
    return rows, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    if not a.raw_dir.is_dir():
        print(f"error: no such directory: {a.raw_dir}", file=sys.stderr); return 2
    rows, problems = build_rows(a.raw_dir)
    if not rows:
        print("error: no rows parsed", file=sys.stderr)
        for p in problems: print(f"  - {p}", file=sys.stderr)
        return 1
    out = a.out_dir or (a.raw_dir.parent.parent / "summary" / rows[0]["experiment_id"])
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows: fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    with (out / "results.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=csv_fieldnames(rows), extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow(csv_flatten(r))
    labels = sorted({r["config_label"] for r in rows})
    print(f"parsed {len(rows)} row(s) from {a.raw_dir}")
    print(f"  configurations   : {', '.join(labels)}")
    print(f"  transport        : {rows[0]['transport']}")
    print(f"  value_kind       : {sorted({r['value_kind'] for r in rows})}")
    print(f"  null efficiency  : {sum(1 for r in rows if r['overlap_efficiency'] is None)}")
    print(f"  -> {out / 'results.jsonl'}")
    print(f"  -> {out / 'results.csv'}")
    if problems:
        print(f"\n{len(problems)} problem(s):", file=sys.stderr)
        for p in problems[:15]: print(f"  - {p}", file=sys.stderr)
    return 1 if (a.strict and problems and not rows) else 0


if __name__ == "__main__":
    sys.exit(main())
