#!/usr/bin/env python3
"""Parse raw nccl-tests output into schema-conformant JSONL and CSV.

Runs locally, after the GPU node has been terminated. Reads a raw experiment
directory produced by ``scripts/run_nccl_baseline.sh`` and emits one row per
measurement point.

    python3 scripts/parse_nccl_output.py --raw-dir results/raw/<experiment-id>

Design notes
------------
* Column positions are derived from the nccl-tests header line rather than
  hardcoded. Different nccl-tests versions include or omit the ``redop`` and
  ``root`` columns, and a fixed-offset parser silently mis-assigns values when
  that happens.
* Raw input is never modified. Output goes to ``results/summary/<id>/``.
* Integrity interlock: text containing a ``SYNTHETIC`` marker is tagged
  ``value_kind="synthetic"`` so fabricated numbers cannot enter a results file
  as measurements.

Standard library only; see docs/design/RFC-001-result-schema.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

BINARY_TO_COLLECTIVE = {
    "all_reduce_perf": "all_reduce",
    "all_gather_perf": "all_gather",
    "reduce_scatter_perf": "reduce_scatter",
    "broadcast_perf": "broadcast",
    "sendrecv_perf": "p2p",
}

# Bus bandwidth correction factor as a function of rank count.
# See docs/experiments/phase1_nccl_baseline.md section 9.
BUSBW_FACTOR = {
    "all_reduce": lambda n: 2.0 * (n - 1) / n,
    "all_gather": lambda n: (n - 1) / n,
    "reduce_scatter": lambda n: (n - 1) / n,
    "broadcast": lambda n: 1.0,
    "p2p": lambda n: 1.0,
}

CSV_COLUMNS = [
    "schema_version", "experiment_id", "phase", "timestamp", "value_kind",
    "git_commit", "git_dirty",
    "provider", "provider_instance_id", "hostname", "os", "kernel",
    "cpu_model", "cpu_cores",
    "node_count", "gpu_model", "gpu_count", "topology_summary", "network",
    "nvlink_present", "p2p_enabled",
    "cuda_version", "driver_version", "nccl_version", "nccl_version_source",
    "mpi_version", "compiler_version", "nccl_tests_commit",
    "hosts", "rank_to_host", "ranks_per_node", "net_interface",
    "transport", "transport_verified", "mpi_implementation", "launcher",
    "benchmark_tool", "collective", "datatype", "redop", "root",
    "message_size_bytes", "count_elements", "placement",
    "warmup_iterations", "measured_iterations", "repeat_index", "tier",
    "latency_us", "algorithmic_bandwidth_gbps", "bus_bandwidth_gbps",
    "wrong_count", "correctness_ok", "out_of_bounds_ok",
    "bandwidth_ratio_ok", "exit_code",
    "command", "raw_output_path", "notes",
    # nested, serialized as compact JSON in the CSV view
    "gpus", "topology", "env",
]

_NUMERIC_RE = re.compile(r"^-?\d+$")


# --------------------------------------------------------------------------
# nccl-tests output parsing
# --------------------------------------------------------------------------

def _is_header_line(tokens: list[str]) -> bool:
    return "size" in tokens and "count" in tokens and "busbw" in tokens


def _to_float(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _to_int(text: str) -> int | None:
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def parse_nccl_tests_output(text: str) -> dict[str, Any]:
    """Extract run metadata and per-size measurements from nccl-tests stdout."""
    meta: dict[str, Any] = {
        "n_gpus": None,
        "n_ranks": None,
        "warmup_iterations": None,
        "measured_iterations": None,
        "validation_enabled": None,
        "gpu_model": None,
        "hostname": None,
        "nccl_version": None,
        "out_of_bounds_ok": None,
        "avg_bus_bandwidth_gbps": None,
        "synthetic": "SYNTHETIC" in text,
    }
    points: list[dict[str, Any]] = []
    columns: list[str] | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue

        if line.lstrip().startswith("#"):
            body = line.lstrip()[1:].strip()
            tokens = body.split()

            if _is_header_line(tokens):
                columns = tokens
                continue

            m = re.search(r"\bnGpus\s+(\d+)", body)
            if m:
                meta["n_gpus"] = int(m.group(1))
            m = re.search(r"warmup iters:\s*(\d+)\s+iters:\s*(\d+)", body)
            if m:
                meta["warmup_iterations"] = int(m.group(1))
                meta["measured_iterations"] = int(m.group(2))
            m = re.search(r"validation:\s*(\d+)", body)
            if m:
                meta["validation_enabled"] = bool(int(m.group(1)))

            # "Rank  0 Group  0 Pid 1234 on host device  0 [0x01] NVIDIA RTX A5000"
            m = re.match(r"Rank\s+\d+\s+.*?\bon\s+(\S+)\s+device\s+\d+\s+\[[^\]]*\]\s+(.+)$", body)
            if m:
                meta["n_ranks"] = (meta["n_ranks"] or 0) + 1
                meta["hostname"] = meta["hostname"] or m.group(1)
                meta["gpu_model"] = meta["gpu_model"] or m.group(2).strip()

            m = re.search(r"Out of bounds values\s*:\s*(\d+)\s*(\w+)", body)
            if m:
                meta["out_of_bounds_ok"] = (m.group(1) == "0" and m.group(2).upper() == "OK")
            m = re.search(r"Avg bus bandwidth\s*:\s*([\d.]+)", body)
            if m:
                meta["avg_bus_bandwidth_gbps"] = float(m.group(1))
            continue

        # NCCL_DEBUG=VERSION banner can land on stdout in some builds.
        m = re.search(r"NCCL version\s+(\S+)", line)
        if m:
            meta["nccl_version"] = m.group(1).rstrip("+")
            continue

        # Data row.
        if columns is None:
            continue
        fields = line.split()
        if len(fields) != len(columns) or not _NUMERIC_RE.match(fields[0]):
            continue

        first_time = columns.index("time")
        prefix = dict(zip(columns[:first_time], fields[:first_time]))
        tail_cols = columns[first_time:]
        tail_vals = fields[first_time:]
        half = len(tail_cols) // 2
        groups = [
            dict(zip(tail_cols[:half], tail_vals[:half])),
            dict(zip(tail_cols[half:], tail_vals[half:])),
        ]

        for placement, group in zip(("out_of_place", "in_place"), groups):
            if not group:
                continue
            points.append({
                "placement": placement,
                "message_size_bytes": _to_int(prefix.get("size", "")),
                "count_elements": _to_int(prefix.get("count", "")),
                "datatype": prefix.get("type"),
                "redop": prefix.get("redop"),
                "root": _to_int(prefix.get("root", "")),
                "latency_us": _to_float(group.get("time", "")),
                "algorithmic_bandwidth_gbps": _to_float(group.get("algbw", "")),
                "bus_bandwidth_gbps": _to_float(group.get("busbw", "")),
                "wrong_count": _to_int(group.get("#wrong", "")),
            })

    if meta["n_ranks"] is None:
        meta["n_ranks"] = meta["n_gpus"]
    return {"meta": meta, "points": points}


def check_bandwidth_ratio(collective: str, ranks: int | None,
                          algbw: float | None, busbw: float | None) -> bool | None:
    """Hypothesis H5: busbw must equal algbw * factor(collective, ranks).

    Returns None when the printed precision makes the check meaningless
    (nccl-tests prints two decimals, so tiny messages report 0.00 GB/s).
    """
    factor_fn = BUSBW_FACTOR.get(collective)
    if factor_fn is None or not ranks or ranks < 2:
        return None
    if algbw is None or busbw is None or algbw < 0.01:
        return None
    expected = algbw * factor_fn(ranks)
    return abs(busbw - expected) <= max(0.02, 0.02 * expected)


# --------------------------------------------------------------------------
# Experiment assembly
# --------------------------------------------------------------------------

def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"warning: could not read {path}: {exc}", file=sys.stderr)
        return {}


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def discover_runs(raw_dir: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Prefer the manifest; fall back to filenames if it is missing."""
    runs = manifest.get("runs") or []
    if runs:
        return runs

    recovered = []
    for path in sorted(raw_dir.glob("*.stdout.txt")):
        stem = path.name[: -len(".stdout.txt")]
        parts = stem.split(".")
        collective = parts[0] if parts else stem
        tier = parts[1] if len(parts) > 1 else None
        repeat = 0
        if len(parts) > 2 and parts[2].startswith("r"):
            repeat = _to_int(parts[2][1:]) or 0
        recovered.append({
            "collective": collective,
            "tier": tier,
            "repeat_index": repeat,
            "stdout_file": path.name,
            "stderr_file": f"{stem}.stderr.txt",
            "command": None,
            "exit_code": None,
        })
    if recovered:
        print(f"note: no run_manifest.json; recovered {len(recovered)} run(s) from filenames",
              file=sys.stderr)
    return recovered


def build_rows(raw_dir: Path, phase: str) -> tuple[list[dict[str, Any]], list[str]]:
    env = load_json(raw_dir / "env.json")
    manifest = load_json(raw_dir / "run_manifest.json")
    problems: list[str] = []

    experiment_id = manifest.get("experiment_id") or raw_dir.name
    repo_root = Path(__file__).resolve().parent.parent

    rows: list[dict[str, Any]] = []
    for run in discover_runs(raw_dir, manifest):
        stdout_path = raw_dir / (run.get("stdout_file") or "")
        stdout_text = read_text(stdout_path)
        if not stdout_text:
            problems.append(f"{stdout_path.name}: empty or missing")
            continue
        stderr_text = read_text(raw_dir / (run.get("stderr_file") or ""))

        parsed = parse_nccl_tests_output(stdout_text)
        meta, points = parsed["meta"], parsed["points"]
        if not points:
            problems.append(f"{stdout_path.name}: no data rows parsed")

        collective = run.get("collective") or BINARY_TO_COLLECTIVE.get(
            run.get("binary", ""), "unknown")

        # The banner from NCCL_DEBUG=VERSION normally goes to stderr; it is the
        # definitive version for this measurement, so it wins over env probing.
        nccl_version = meta["nccl_version"]
        if not nccl_version:
            m = re.search(r"NCCL version\s+(\S+)", stderr_text)
            if m:
                nccl_version = m.group(1).rstrip("+")
        nccl_version_source = "NCCL_DEBUG=VERSION banner" if nccl_version else None
        if not nccl_version:
            nccl_version = env.get("nccl_version")
            nccl_version_source = env.get("nccl_version_source")

        exit_code = run.get("exit_code")
        ranks = meta["n_ranks"] or run.get("gpu_count") or env.get("gpu_count")
        synthetic = meta["synthetic"] or "SYNTHETIC" in stderr_text
        value_kind = "synthetic" if synthetic else manifest.get("value_kind", "measured")
        if synthetic:
            problems.append(f"{stdout_path.name}: SYNTHETIC marker -> value_kind=synthetic")

        try:
            rel_raw = str(stdout_path.resolve().relative_to(repo_root))
        except ValueError:
            rel_raw = str(stdout_path)

        for point in points:
            correctness_ok = (
                (exit_code in (0, None))
                and meta["out_of_bounds_ok"] is not False
                and (point["wrong_count"] in (0, None))
            )
            if not correctness_ok:
                problems.append(
                    f"{stdout_path.name}: correctness failure at "
                    f"{point['message_size_bytes']} B ({point['placement']})")

            ratio_ok = check_bandwidth_ratio(
                collective, ranks,
                point["algorithmic_bandwidth_gbps"], point["bus_bandwidth_gbps"])
            if ratio_ok is False:
                problems.append(
                    f"{stdout_path.name}: H5 bus/alg bandwidth ratio mismatch at "
                    f"{point['message_size_bytes']} B ({point['placement']})")

            rows.append({
                "schema_version": SCHEMA_VERSION,
                "experiment_id": experiment_id,
                "phase": phase,
                "timestamp": run.get("started_at_utc") or env.get("captured_at_utc")
                             or manifest.get("created_at_utc"),
                "value_kind": value_kind,
                "git_commit": manifest.get("git_commit") or env.get("git_commit"),
                "git_dirty": env.get("git_dirty"),

                "provider": env.get("provider"),
                "provider_instance_id": env.get("provider_instance_id"),
                "hostname": env.get("hostname") or meta["hostname"],
                "os": env.get("os"),
                "kernel": env.get("kernel"),
                "cpu_model": env.get("cpu_model"),
                "cpu_cores": env.get("cpu_cores"),
                "node_count": manifest.get("node_count") or env.get("node_count") or 1,
                "gpu_model": meta["gpu_model"] or env.get("gpu_model"),
                "gpu_count": ranks or 1,
                "gpus": env.get("gpus"),
                "topology": env.get("topology"),
                "topology_summary": env.get("topology_summary"),
                "network": manifest.get("network", "none-single-node"),
                "nvlink_present": env.get("nvlink_present"),
                "p2p_enabled": None,

                # Multi-node (phase 3+). Single-node manifests omit these, so
                # they stay null rather than being invented.
                "hosts": manifest.get("hosts"),
                "rank_to_host": manifest.get("rank_to_host"),
                "ranks_per_node": manifest.get("ranks_per_node"),
                "net_interface": manifest.get("net_interface"),
                "transport": manifest.get("transport"),
                "transport_verified": manifest.get("transport_verified"),
                "mpi_implementation": manifest.get("mpi_implementation"),
                "launcher": manifest.get("launcher"),

                "cuda_version": env.get("cuda_version"),
                "driver_version": env.get("driver_version"),
                "nccl_version": nccl_version,
                "nccl_version_source": nccl_version_source,
                "mpi_version": env.get("mpi_version"),
                "compiler_version": env.get("compiler_version"),
                "nccl_tests_commit": manifest.get("nccl_tests_commit"),

                "benchmark_tool": manifest.get("benchmark_tool", "nccl-tests"),
                "collective": collective,
                "datatype": point["datatype"] or run.get("datatype") or "float",
                "redop": point["redop"],
                "root": point["root"],
                "message_size_bytes": point["message_size_bytes"],
                "count_elements": point["count_elements"],
                "placement": point["placement"],
                "warmup_iterations": meta["warmup_iterations"]
                                     if meta["warmup_iterations"] is not None
                                     else run.get("warmup_iterations", 0),
                "measured_iterations": meta["measured_iterations"]
                                       if meta["measured_iterations"] is not None
                                       else run.get("measured_iterations", 1),
                "repeat_index": run.get("repeat_index", 0),
                "tier": run.get("tier"),

                "latency_us": point["latency_us"],
                "algorithmic_bandwidth_gbps": point["algorithmic_bandwidth_gbps"],
                "bus_bandwidth_gbps": point["bus_bandwidth_gbps"],

                "wrong_count": point["wrong_count"],
                "correctness_ok": bool(correctness_ok),
                "out_of_bounds_ok": meta["out_of_bounds_ok"],
                "bandwidth_ratio_ok": ratio_ok,
                "exit_code": exit_code,

                "command": run.get("command") or "<unrecorded>",
                "env": env.get("env"),
                "raw_output_path": rel_raw,
                "notes": None,
            })

    return rows, problems


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            for key in ("gpus", "env", "hosts", "rank_to_host"):
                if isinstance(flat.get(key), (dict, list)):
                    flat[key] = json.dumps(flat[key], separators=(",", ":"), sort_keys=True)
            if isinstance(flat.get("topology"), str):
                flat["topology"] = json.dumps(flat["topology"])
            writer.writerow(flat)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True, type=Path,
                    help="results/raw/<experiment-id>")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: results/summary/<experiment-id>")
    ap.add_argument("--phase", default="phase1")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any correctness or H5 problem was found")
    args = ap.parse_args()

    raw_dir: Path = args.raw_dir
    if not raw_dir.is_dir():
        print(f"error: no such directory: {raw_dir}", file=sys.stderr)
        return 2

    rows, problems = build_rows(raw_dir, args.phase)
    if not rows:
        print("error: no measurement rows parsed", file=sys.stderr)
        return 1

    out_dir: Path = args.out_dir or (
        raw_dir.parent.parent / "summary" / (rows[0]["experiment_id"] or raw_dir.name))
    out_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(rows, out_dir / "results.jsonl")
    write_csv(rows, out_dir / "results.csv")

    total = len(rows)
    bad = sum(1 for r in rows if not r["correctness_ok"])
    ratio_bad = sum(1 for r in rows if r["bandwidth_ratio_ok"] is False)
    multinode = sum(1 for r in rows if (r.get("node_count") or 1) > 1)
    unverified = sum(1 for r in rows
                     if (r.get("node_count") or 1) > 1 and not r.get("transport_verified"))
    kinds = sorted({r["value_kind"] for r in rows})
    collectives = sorted({r["collective"] for r in rows})

    print(f"parsed {total} row(s) from {raw_dir}")
    print(f"  collectives      : {', '.join(collectives)}")
    print(f"  value_kind       : {', '.join(kinds)}")
    print(f"  correctness fail : {bad}")
    print(f"  H5 ratio fail    : {ratio_bad}")
    if multinode:
        transports = sorted({r.get("transport") for r in rows if (r.get("node_count") or 1) > 1})
        ifaces = sorted({r.get("net_interface") for r in rows if (r.get("node_count") or 1) > 1})
        print(f"  multi-node rows  : {multinode}  transport={transports} iface={ifaces}")
        if unverified:
            print(f"  WARNING          : {unverified} multi-node row(s) have no verified "
                  "transport; they must not be used for a transport comparison",
                  file=sys.stderr)
    print(f"  -> {out_dir / 'results.jsonl'}")
    print(f"  -> {out_dir / 'results.csv'}")

    if problems:
        print(f"\n{len(problems)} problem(s):", file=sys.stderr)
        for p in problems[:20]:
            print(f"  - {p}", file=sys.stderr)
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more", file=sys.stderr)

    if args.strict and (bad or ratio_bad or unverified):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
