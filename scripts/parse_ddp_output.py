#!/usr/bin/env python3
"""Convert Phase 9 DDP training output into schema-conformant JSONL/CSV.

One row per measured repeat. `latency_us` carries the phase's primary
measurement — the end-to-end training step time — because that, not overlap
percentage, is what Phase 9 optimises.

The `nosync` rows are not benchmark results in their own right: they are the
compute reference used to derive `sync_cost_ms` for the DDP rows that share
their label. They are still emitted, marked `workload_kind=nosync-probe`, so
the derivation is auditable rather than hidden inside this script.

    python3 scripts/parse_ddp_output.py --raw-dir results/raw/<experiment-id>

Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CSV_HEADER_PREFIX = "experiment,"

KIND = {
    "ddp": "ddp-training",
    "single": "single-gpu-training",
    "serial": "serial-reduction",
    "nosync": "nosync-probe",
}

CSV_COLUMNS = [
    "schema_version", "experiment_id", "phase", "timestamp", "value_kind", "git_commit",
    "provider", "hostname", "os", "cpu_model", "node_count", "gpu_model", "gpu_count",
    "topology_summary", "network", "nvlink_present", "p2p_enabled", "transport",
    "transport_verified", "cuda_version", "driver_version", "nccl_version",
    "benchmark_tool", "workload_kind", "config_label",
    "model_name", "model_params", "precision", "batch_per_gpu", "sequence_length",
    "tokens_per_step", "bucket_cap_mb", "ddp_bucket_count", "gradient_bytes",
    "collective", "datatype", "message_size_bytes", "placement",
    "warmup_iterations", "measured_iterations", "repeat_index",
    "step_time_ms", "forward_ms", "backward_ms", "optimizer_ms", "step_wall_ms",
    "tokens_per_s", "backward_nosync_ms", "sync_cost_ms",
    "loss_finite", "grads_finite", "param_sync_max_abs_diff", "peak_memory_gib",
    "latency_us", "correctness_ok", "command", "raw_output_path", "notes",
]


def csv_fieldnames(rows: list[dict]) -> list[str]:
    """Curated order first, then anything else the rows carry.

    Same rule as the other parsers: a newly emitted scalar can be pushed to the
    end of the CSV view but never silently dropped from it.
    """
    extra = sorted(set().union(*(set(r) for r in rows)) - set(CSV_COLUMNS))
    return CSV_COLUMNS + extra


def csv_flatten(row: dict) -> dict:
    flat = dict(row)
    for k in ("gpus", "env", "hosts", "rank_to_host", "topology",
              "model_config", "ddp_bucket_sizes"):
        v = flat.get(k)
        if v is not None and not isinstance(v, str):
            flat[k] = json.dumps(v, separators=(",", ":"), sort_keys=True)
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
    """Find the CSV table by its header prefix, never by 'first non-# line'.

    torchrun and NCCL both print banners that start with neither '#' nor the
    header; Phase 6 lost a run to exactly that mistake.
    """
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith(CSV_HEADER_PREFIX)), None)
    if start is None:
        return []
    body = [l for l in lines[start:] if l and not l.startswith("#")]
    return list(csv.DictReader(body))


def parse_header(text: str) -> dict[str, Any]:
    """Pull the run's self-description out of the '#' comment block."""
    out: dict[str, Any] = {}
    for line in text.splitlines():
        if not line.startswith("#"):
            continue
        if m := re.search(r"params=(\d+) grad_bytes=(\d+)", line):
            out["model_params"] = int(m.group(1))
            out["gradient_bytes"] = int(m.group(2))
        if m := re.search(r"layers=(\d+) heads=(\d+) embd=(\d+) vocab=(\d+) seq=(\d+)", line):
            out["model_config"] = {
                "layers": int(m.group(1)), "heads": int(m.group(2)),
                "embd": int(m.group(3)), "vocab": int(m.group(4)),
                "seq": int(m.group(5)),
            }
            out["sequence_length"] = int(m.group(5))
        if m := re.search(r"batch_per_gpu=(\d+) tokens_per_step=(\d+) precision=(\S+)", line):
            out["batch_per_gpu"] = int(m.group(1))
            out["tokens_per_step"] = int(m.group(2))
            out["precision"] = m.group(3)
        if m := re.search(r"warmup=(\d+) steps=(\d+) repeats=(\d+)", line):
            out["warmup_iterations"] = int(m.group(1))
            out["measured_iterations"] = int(m.group(2))
        if m := re.search(r"loss_finite=(\w+) grads_finite=(\w+) "
                          r"param_sync_max_abs_diff=(\S+)", line):
            out["loss_finite"] = m.group(1) == "True"
            out["grads_finite"] = m.group(2) == "True"
            out["param_sync_max_abs_diff"] = float(m.group(3))
        if m := re.search(r"PEAK_MEM_GIB rank0 (\S+)", line):
            out["peak_memory_gib"] = float(m.group(1))
        if line.startswith("# DDP_LOGGING "):
            try:
                d = json.loads(line[len("# DDP_LOGGING "):])
            except json.JSONDecodeError:
                continue
            # DDP reports bucket sizes as a comma-separated string, and the
            # rebuilt list is the one that actually ran: DDP reorders buckets
            # after the first iteration to match gradient-ready order.
            raw = d.get("rebuilt_bucket_sizes") or d.get("bucket_sizes") or ""
            sizes = [int(x) for x in str(raw).split(",") if x.strip().isdigit()]
            if sizes:
                out["ddp_bucket_sizes"] = sizes
                out["ddp_bucket_count"] = len(sizes)
            out["ddp_logging_raw"] = d
    return out


def build_rows(raw_dir: Path, phase: str = "phase9") -> tuple[list[dict[str, Any]], list[str]]:
    env = load_json(raw_dir / "env.json")
    man = load_json(raw_dir / "run_manifest.json")
    problems: list[str] = []

    outs = sorted(raw_dir.glob("*.stdout.txt"))
    if not outs:
        return [], ["no *.stdout.txt in raw dir"]

    repo = Path(__file__).resolve().parent.parent
    rows: list[dict[str, Any]] = []

    for out_path in outs:
        text = out_path.read_text(encoding="utf-8", errors="replace")
        table = parse_table(text)
        if not table:
            problems.append(f"{out_path.name}: no CSV rows")
            continue
        head = parse_header(text)
        if "SYNTHETIC" in text:
            problems.append(f"{out_path.name}: SYNTHETIC marker -> value_kind=synthetic")
        value_kind = "synthetic" if "SYNTHETIC" in text else man.get("value_kind", "measured")
        if head.get("loss_finite") is False or head.get("grads_finite") is False:
            problems.append(f"{out_path.name}: correctness check failed")
        try:
            rel = str(out_path.resolve().relative_to(repo))
        except ValueError:
            rel = str(out_path)

        for rec in table:
            def num(k, cast=float, default=None):
                v = rec.get(k)
                if v is None or v == "":
                    return default
                try:
                    o = cast(v)
                except ValueError:
                    return default
                if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
                    return None
                return o

            mode = rec.get("mode", "")
            kind = KIND.get(mode, mode or None)
            world = num("world_size", int, 1) or 1
            grad_bytes = num("grad_bytes", int, 0) or head.get("gradient_bytes") or 0

            rows.append({
                "schema_version": SCHEMA_VERSION,
                "experiment_id": man.get("experiment_id") or raw_dir.name,
                "phase": phase,
                "timestamp": man.get("created_at_utc") or env.get("captured_at_utc"),
                "value_kind": value_kind,
                "git_commit": man.get("git_commit") or env.get("git_commit"),
                "git_dirty": env.get("git_dirty"),
                "provider": man.get("provider") or env.get("provider"),
                "provider_instance_id": man.get("provider_instance_id"),
                "hostname": env.get("hostname"), "os": env.get("os"),
                "kernel": env.get("kernel"), "cpu_model": env.get("cpu_model"),
                "cpu_cores": env.get("cpu_cores"),
                "node_count": man.get("node_count", 1),
                "gpu_model": env.get("gpu_model"),
                "gpu_count": world,
                "gpus": env.get("gpus"), "topology": env.get("topology"),
                "topology_summary": env.get("topology_summary"),
                "network": man.get("network", "none-single-node"),
                "nvlink_present": env.get("nvlink_present"),
                "p2p_enabled": man.get("transport") == "P2P/direct",
                "transport": man.get("transport") if world > 1 else None,
                "transport_verified": bool(man.get("transport_verified")) if world > 1 else None,
                "hosts": None, "rank_to_host": None, "ranks_per_node": None,
                "net_interface": None, "mpi_implementation": None,
                "launcher": man.get("launcher"),
                "cuda_version": env.get("cuda_version"),
                "driver_version": env.get("driver_version"),
                "nccl_version": env.get("nccl_version"),
                "nccl_version_source": env.get("nccl_version_source"),
                "mpi_version": None,
                "compiler_version": env.get("compiler_version"),
                "nccl_tests_commit": None,
                "benchmark_tool": "ddp-train-bench",
                "workload_kind": kind,
                "config_label": rec.get("label") or mode,
                "nccl_algo": None, "nccl_proto": None, "nccl_extra_env": man.get("nccl_extra_env"),
                "model_name": man.get("model_name", "gpt-compact"),
                "model_params": num("param_count", int) or head.get("model_params"),
                "model_config": head.get("model_config"),
                "precision": head.get("precision"),
                "batch_per_gpu": head.get("batch_per_gpu"),
                "sequence_length": head.get("sequence_length"),
                "tokens_per_step": head.get("tokens_per_step"),
                "bucket_cap_mb": num("bucket_mb", float),
                "ddp_bucket_sizes": head.get("ddp_bucket_sizes"),
                "ddp_bucket_count": head.get("ddp_bucket_count"),
                "gradient_bytes": grad_bytes,
                # A DDP step reduces the whole gradient set; the collective is
                # AllReduce/sum over float32 regardless of autocast dtype.
                "collective": "all_reduce", "datatype": "float", "redop": "sum", "root": None,
                "message_size_bytes": grad_bytes,
                "count_elements": None,
                "placement": "in_place",
                "warmup_iterations": head.get("warmup_iterations", 0),
                "measured_iterations": head.get("measured_iterations", 1),
                "repeat_index": num("repeat", int, 0) or 0,
                "tier": "full",
                "step_time_ms": num("step_ms"),
                "forward_ms": num("fwd_ms"),
                "backward_ms": num("bwd_ms"),
                "optimizer_ms": num("opt_ms"),
                "step_wall_ms": num("step_wall_ms"),
                "tokens_per_s": num("tokens_per_s"),
                "backward_nosync_ms": None,   # filled in below
                "sync_cost_ms": None,         # filled in below
                "loss_finite": head.get("loss_finite"),
                "grads_finite": head.get("grads_finite"),
                "param_sync_max_abs_diff": head.get("param_sync_max_abs_diff"),
                "peak_memory_gib": head.get("peak_memory_gib"),
                # the phase's primary measurement
                "latency_us": (num("step_ms") * 1e3) if num("step_ms") is not None else None,
                "algorithmic_bandwidth_gbps": None, "bus_bandwidth_gbps": None,
                "wrong_count": None,
                "correctness_ok": bool(head.get("loss_finite") and head.get("grads_finite")),
                "out_of_bounds_ok": None, "bandwidth_ratio_ok": None,
                "exit_code": man.get("exit_code"),
                "command": man.get("command") or "<unrecorded>",
                "env": env.get("env"), "raw_output_path": rel,
                "notes": f"mode={mode} label={rec.get('label')}",
            })

    attach_sync_cost(rows, problems)
    return rows, problems


def attach_sync_cost(rows: list[dict], problems: list[str]) -> None:
    """Pair each DDP row with the nosync probe that shares its label.

    The probe label is '<label>-nosync' by construction in train_ddp.py. A DDP
    row with no probe keeps null sync_cost_ms rather than borrowing another
    configuration's number.
    """
    probe = {r["config_label"][:-len("-nosync")]: r["backward_ms"]
             for r in rows
             if r["workload_kind"] == "nosync-probe"
             and str(r["config_label"]).endswith("-nosync")
             and r["backward_ms"] is not None}
    for r in rows:
        if r["workload_kind"] != "ddp-training":
            continue
        base = probe.get(r["config_label"])
        if base is None:
            problems.append(f"{r['config_label']}: no nosync probe -> sync_cost_ms null")
            continue
        r["backward_nosync_ms"] = base
        r["sync_cost_ms"] = round(r["backward_ms"] - base, 4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--phase", default="phase9")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    if not a.raw_dir.is_dir():
        print(f"error: {a.raw_dir} is not a directory", file=sys.stderr)
        return 2
    rows, problems = build_rows(a.raw_dir, a.phase)
    if not rows:
        print(f"error: no rows parsed from {a.raw_dir}", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    out_dir = a.out_dir or Path("results/summary") / a.raw_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    jl, cv = out_dir / "results.jsonl", out_dir / "results.csv"
    with jl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    fields = csv_fieldnames(rows)
    with cv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(csv_flatten(r))

    kinds = sorted({r["workload_kind"] for r in rows})
    print(f"parsed {len(rows)} row(s) from {a.raw_dir}")
    print(f"  workload kinds  : {', '.join(str(k) for k in kinds)}")
    print(f"  configurations  : {', '.join(sorted({str(r['config_label']) for r in rows}))}")
    print(f"  value_kind      : {sorted({r['value_kind'] for r in rows})}")
    print(f"  correctness_ok  : {all(r['correctness_ok'] for r in rows)}")
    for p in problems:
        print(f"  ! {p}")
    print(f"  -> {jl}")
    print(f"  -> {cv}")
    return 1 if (problems and a.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
