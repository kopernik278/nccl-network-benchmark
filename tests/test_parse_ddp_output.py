#!/usr/bin/env python3
"""Tests for the Phase 9 DDP training parser."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import parse_ddp_output as P  # noqa: E402

SCHEMA = Path(__file__).resolve().parent.parent / "schemas" / "nccl_result.schema.json"

# A torchrun banner precedes the table on purpose: the parser must locate the
# header by prefix, the bug Phase 6 was bitten by.
STDOUT = """\
W0829 NOTE: Redirects are currently not supported in MacOs.
NCCL version 2.25.1+cuda12.8
# phase9 ddp training benchmark
# mode=ddp label=ddp-25mb world_size=4 bucket_mb=25.0
# model: layers=8 heads=12 embd=768 vocab=16384 seq=1024 tie=False
# params=82000000 grad_bytes=328000000 (312.8 MiB, dtype=float32)
# batch_per_gpu=16 tokens_per_step=65536 precision=bf16 optimizer=AdamW(fused)
# warmup=10 steps=30 repeats=3 pool=4 seed=1234 nvtx=False
# CORRECTNESS loss_finite=True grads_finite=True param_sync_max_abs_diff=0.000e+00 steps_completed=13
# DDP_LOGGING {"bucket_cap_bytes": 26214400, "bucket_sizes": "1000,2000", \
"rebuilt_bucket_sizes": "26000000,26000000,20000000"}
experiment,mode,label,bucket_mb,world_size,repeat,step_ms,fwd_ms,bwd_ms,opt_ms,step_wall_ms,\
tokens_per_s,grad_bytes,param_count,loss
train,ddp,ddp-25mb,25,4,0,200.0000,40.0000,140.0000,20.0000,201.0000,326000.00,328000000,82000000,7.1
train,ddp,ddp-25mb,25,4,1,202.0000,40.0000,142.0000,20.0000,203.0000,323000.00,328000000,82000000,7.0
train,nosync,ddp-25mb-nosync,25,4,0,150.0000,40.0000,90.0000,20.0000,151.0000,434000.00,328000000,82000000,
# NOSYNC_PROBE done; params re-broadcast from rank 0
# SLOWEST_RANK_STEP_MS 202.5
# PEAK_MEM_GIB rank0 21.500
"""

MANIFEST = {
    "experiment_id": "p9-test",
    "created_at_utc": "2026-08-29T12:00:00Z",
    "git_commit": "deadbeef",
    "provider": "runpod",
    "provider_instance_id": "podX",
    "transport": "SHM/direct",
    "transport_verified": True,
    "value_kind": "measured",
    "launcher": "torchrun",
    "command": "torchrun --nproc_per_node=4 src/ddp/train_ddp.py",
    "exit_code": 0,
}

ENV = {
    "gpu_model": "NVIDIA A40", "hostname": "h", "os": "Ubuntu 24.04",
    "cuda_version": "12.8", "driver_version": "580", "nccl_version": "2.25.1",
    "topology_summary": "SYS", "nvlink_present": False, "cpu_model": "Xeon",
    "captured_at_utc": "2026-08-29T12:00:00Z",
}


def make_raw(td: Path, stdout: str = STDOUT) -> Path:
    (td / "ddp-25mb.stdout.txt").write_text(stdout, encoding="utf-8")
    (td / "run_manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    (td / "env.json").write_text(json.dumps(ENV), encoding="utf-8")
    return td


class TestDDPParsing(unittest.TestCase):
    def test_rows_parsed_past_the_torchrun_banner(self):
        with tempfile.TemporaryDirectory() as td:
            rows, _ = P.build_rows(make_raw(Path(td)))
            self.assertEqual(len(rows), 3)
            self.assertEqual({r["experiment_id"] for r in rows}, {"p9-test"})

    def test_workload_kinds_are_distinguished(self):
        with tempfile.TemporaryDirectory() as td:
            rows, _ = P.build_rows(make_raw(Path(td)))
            kinds = [r["workload_kind"] for r in rows]
            self.assertEqual(kinds.count("ddp-training"), 2)
            self.assertEqual(kinds.count("nosync-probe"), 1)

    def test_model_and_training_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            rows, _ = P.build_rows(make_raw(Path(td)))
            r = rows[0]
            self.assertEqual(r["model_config"],
                             {"layers": 8, "heads": 12, "embd": 768,
                              "vocab": 16384, "seq": 1024})
            self.assertEqual(r["precision"], "bf16")
            self.assertEqual(r["batch_per_gpu"], 16)
            self.assertEqual(r["tokens_per_step"], 65536)
            self.assertEqual(r["sequence_length"], 1024)
            self.assertEqual(r["warmup_iterations"], 10)
            self.assertEqual(r["measured_iterations"], 30)
            self.assertAlmostEqual(r["peak_memory_gib"], 21.5)

    def test_rebuilt_bucket_sizes_win_over_initial(self):
        # DDP reorders buckets after iteration 1; the rebuilt list is the one
        # that actually ran, so reporting the initial list would be wrong.
        with tempfile.TemporaryDirectory() as td:
            rows, _ = P.build_rows(make_raw(Path(td)))
            self.assertEqual(rows[0]["ddp_bucket_sizes"],
                             [26000000, 26000000, 20000000])
            self.assertEqual(rows[0]["ddp_bucket_count"], 3)

    def test_requested_bucket_cap_is_not_confused_with_actual_sizes(self):
        with tempfile.TemporaryDirectory() as td:
            rows, _ = P.build_rows(make_raw(Path(td)))
            r = rows[0]
            self.assertEqual(r["bucket_cap_mb"], 25.0)
            self.assertNotEqual(r["bucket_cap_mb"] * (1 << 20), r["ddp_bucket_sizes"][0])

    def test_sync_cost_is_derived_from_the_matching_probe(self):
        with tempfile.TemporaryDirectory() as td:
            rows, _ = P.build_rows(make_raw(Path(td)))
            ddp = [r for r in rows if r["workload_kind"] == "ddp-training"]
            self.assertAlmostEqual(ddp[0]["backward_nosync_ms"], 90.0)
            self.assertAlmostEqual(ddp[0]["sync_cost_ms"], 50.0)
            self.assertAlmostEqual(ddp[1]["sync_cost_ms"], 52.0)

    def test_missing_probe_leaves_sync_cost_null_and_is_reported(self):
        stdout = "\n".join(l for l in STDOUT.splitlines()
                           if not l.startswith("train,nosync"))
        with tempfile.TemporaryDirectory() as td:
            rows, problems = P.build_rows(make_raw(Path(td), stdout))
            ddp = [r for r in rows if r["workload_kind"] == "ddp-training"]
            self.assertTrue(all(r["sync_cost_ms"] is None for r in ddp))
            self.assertTrue(any("no nosync probe" in p for p in problems))

    def test_latency_us_carries_the_step_time(self):
        with tempfile.TemporaryDirectory() as td:
            rows, _ = P.build_rows(make_raw(Path(td)))
            self.assertAlmostEqual(rows[0]["latency_us"], 200000.0)

    def test_correctness_flags_travel_with_every_row(self):
        with tempfile.TemporaryDirectory() as td:
            rows, _ = P.build_rows(make_raw(Path(td)))
            self.assertTrue(all(r["correctness_ok"] for r in rows))
            self.assertTrue(all(r["loss_finite"] for r in rows))
            self.assertEqual(rows[0]["param_sync_max_abs_diff"], 0.0)

    def test_failed_correctness_is_flagged_not_swallowed(self):
        stdout = STDOUT.replace("grads_finite=True", "grads_finite=False")
        with tempfile.TemporaryDirectory() as td:
            rows, problems = P.build_rows(make_raw(Path(td), stdout))
            self.assertFalse(any(r["correctness_ok"] for r in rows))
            self.assertTrue(any("correctness check failed" in p for p in problems))

    def test_synthetic_marker_downgrades_value_kind(self):
        with tempfile.TemporaryDirectory() as td:
            rows, problems = P.build_rows(make_raw(Path(td), STDOUT + "\n# SYNTHETIC\n"))
            self.assertTrue(all(r["value_kind"] == "synthetic" for r in rows))
            self.assertTrue(any("SYNTHETIC" in p for p in problems))

    def test_single_gpu_rows_carry_no_transport_claim(self):
        stdout = STDOUT.replace("train,ddp,ddp-25mb,25,4,0", "train,single,single,,1,0")
        with tempfile.TemporaryDirectory() as td:
            rows, _ = P.build_rows(make_raw(Path(td), stdout))
            r = next(x for x in rows if x["workload_kind"] == "single-gpu-training")
            self.assertIsNone(r["transport"])
            self.assertIsNone(r["transport_verified"])
            self.assertEqual(r["gpu_count"], 1)


class TestDDPSchema(unittest.TestCase):
    def test_rows_conform(self):
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        props, req = set(schema["properties"]), set(schema["required"])
        with tempfile.TemporaryDirectory() as td:
            rows, _ = P.build_rows(make_raw(Path(td)))
        for r in rows:
            self.assertFalse(set(r) - props, f"unknown fields: {set(r) - props}")
            self.assertFalse(req - set(r), f"missing required: {req - set(r)}")
            self.assertEqual(r["phase"], "phase9")

    def test_csv_columns_cover_the_fields_this_parser_emits(self):
        with tempfile.TemporaryDirectory() as td:
            rows, _ = P.build_rows(make_raw(Path(td)))
        cols = set(P.csv_fieldnames(rows))
        for r in rows:
            self.assertFalse(set(r) - cols, f"dropped from CSV view: {set(r) - cols}")

    def test_nested_values_are_json_encoded_for_csv(self):
        with tempfile.TemporaryDirectory() as td:
            rows, _ = P.build_rows(make_raw(Path(td)))
        flat = P.csv_flatten(rows[0])
        self.assertIsInstance(flat["ddp_bucket_sizes"], str)
        self.assertIsInstance(flat["model_config"], str)


if __name__ == "__main__":
    unittest.main()
