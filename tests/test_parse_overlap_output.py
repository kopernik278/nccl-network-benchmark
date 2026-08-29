#!/usr/bin/env python3
"""Unit tests for scripts/parse_overlap_output.py. Zero-cost, no GPU."""
import json, sys, tempfile, unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import parse_overlap_output as P  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures"
SCHEMA = REPO / "schemas" / "nccl_result.schema.json"


def make_raw(tmp: Path) -> Path:
    raw = tmp / "results" / "raw" / "p7b-test-20260829T000000Z-deadbee"
    raw.mkdir(parents=True)
    (raw / "overlap_bench.stdout.txt").write_text(
        (FIX / "synthetic_overlap_output.txt").read_text(encoding="utf-8"), encoding="utf-8")
    (raw / "env.json").write_text(json.dumps({
        "schema_version": 1, "captured_at_utc": "2026-08-29T00:00:00Z",
        "provider": "runpod", "hostname": "synthetic", "os": "Ubuntu 24.04",
        "cpu_model": "Synthetic", "cpu_cores": 8, "gpu_count": 4,
        "gpu_model": "NVIDIA L4", "driver_version": "550.0", "cuda_version": "12.8",
        "nccl_version": "2.25.1", "compiler_version": "gcc 13",
        "topology_summary": "SYS", "nvlink_present": False,
        "git_commit": "deadbeef", "git_dirty": False, "env": {},
    }), encoding="utf-8")
    (raw / "run_manifest.json").write_text(json.dumps({
        "schema_version": 1, "experiment_id": "p7b-test-20260829T000000Z-deadbee",
        "phase": "phase7", "created_at_utc": "2026-08-29T00:00:00Z",
        "value_kind": "measured", "node_count": 1, "gpu_count": 4,
        "network": "none-single-node", "transport": "shm-nccl",
        "warmup_iterations": 3, "measured_iterations": 10,
        "command": "overlap_bench -g 4", "exit_code": 0,
    }), encoding="utf-8")
    return raw


class TestOverlapParsing(unittest.TestCase):
    def test_all_rows_parsed_past_the_nccl_banner(self):
        """The NCCL_DEBUG banner precedes the header; it must not become one."""
        with tempfile.TemporaryDirectory() as td:
            rows, _ = P.build_rows(make_raw(Path(td)))
            self.assertEqual(len(rows), 7)
            self.assertTrue(all(r["message_size_bytes"] > 0 for r in rows))

    def test_workload_class_separates_rows_at_the_same_ratio(self):
        """Two classes at ratio 1.0 must not collapse into one config_label."""
        with tempfile.TemporaryDirectory() as td:
            rows, _ = P.build_rows(make_raw(Path(td)))
            at16 = [r for r in rows if r["message_size_bytes"] == 16777216]
            self.assertEqual(len(at16), 2)
            self.assertEqual(sorted(r["config_label"] for r in at16),
                             ["micro-ratio1", "micro-ratio1-memory-triad"])
            self.assertEqual(sorted(r["workload_class"] for r in at16),
                             ["compute-gemm", "memory-triad"])

    def test_micro_and_bucket_labels(self):
        with tempfile.TemporaryDirectory() as td:
            rows, _ = P.build_rows(make_raw(Path(td)))
            labels = sorted({r["config_label"] for r in rows})
            self.assertIn("micro-ratio0.25", labels)
            self.assertIn("micro-ratio2", labels)
            self.assertIn("bucket-16MiB", labels)
            self.assertIn("bucket-4MiB", labels)

    def test_components_travel_with_the_row(self):
        with tempfile.TemporaryDirectory() as td:
            rows, _ = P.build_rows(make_raw(Path(td)))
            r = next(x for x in rows if x["config_label"] == "micro-ratio0.25")
            self.assertAlmostEqual(r["t_compute_us"], 100.0)
            self.assertAlmostEqual(r["t_comm_us"], 400.0)
            self.assertAlmostEqual(r["t_seq_us"], 500.0)
            self.assertAlmostEqual(r["t_overlap_us"], 420.0)
            self.assertAlmostEqual(r["t_ideal_us"], 400.0)
            self.assertAlmostEqual(r["overlap_efficiency"], 0.8)
            self.assertAlmostEqual(r["exposed_comm_us"], 320.0)
            # latency_us carries the phase's primary measurement
            self.assertAlmostEqual(r["latency_us"], r["t_overlap_us"])

    def test_zero_overlap_case_is_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            rows, _ = P.build_rows(make_raw(Path(td)))
            r = next(x for x in rows if x["message_size_bytes"] == 16777216)
            self.assertAlmostEqual(r["overlap_efficiency"], 0.0)
            self.assertAlmostEqual(r["exposed_comm_us"], 3000.0)

    def test_out_of_range_efficiency_is_nulled_not_reported(self):
        """A degenerate denominator must not become a headline number."""
        with tempfile.TemporaryDirectory() as td:
            rows, problems = P.build_rows(make_raw(Path(td)))
            # config_label alone does not identify a row — message size is
            # part of the key, and ratio 1.0 appears at two comm sizes here.
            bad = [r for r in rows if r["config_label"] == "micro-ratio1"
                   and r["message_size_bytes"] == 1048576]
            self.assertEqual(len(bad), 1)
            self.assertIsNone(bad[0]["overlap_efficiency"])
            self.assertTrue(any("out of range" in p for p in problems))

    def test_interference_fields_present(self):
        with tempfile.TemporaryDirectory() as td:
            rows, _ = P.build_rows(make_raw(Path(td)))
            r = next(x for x in rows if x["config_label"] == "micro-ratio0.25")
            self.assertAlmostEqual(r["compute_during_overlap_us"], 110.0)
            self.assertAlmostEqual(r["comm_during_overlap_us"], 410.0)

    def test_bucket_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            rows, _ = P.build_rows(make_raw(Path(td)))
            r = next(x for x in rows if x["config_label"] == "bucket-4MiB")
            self.assertEqual(r["bucket_bytes"], 4194304)
            self.assertEqual(r["n_buckets"], 32)
            self.assertEqual(r["compute_reps"], 240)

    def test_synthetic_marker_downgrades_value_kind(self):
        with tempfile.TemporaryDirectory() as td:
            rows, problems = P.build_rows(make_raw(Path(td)))
            self.assertTrue(all(r["value_kind"] == "synthetic" for r in rows))
            self.assertTrue(any("SYNTHETIC" in p for p in problems))


class TestOverlapSchema(unittest.TestCase):
    def test_rows_conform(self):
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        props, req = set(schema["properties"]), set(schema["required"])
        with tempfile.TemporaryDirectory() as td:
            rows, _ = P.build_rows(make_raw(Path(td)))
            for r in rows:
                self.assertFalse(req - set(r), f"missing: {req - set(r)}")
                self.assertFalse(set(r) - props, f"not in schema: {set(r) - props}")

    def test_csv_columns_cover_the_overlap_fields(self):
        for f in ("t_compute_us","t_comm_us","t_seq_us","t_overlap_us","t_ideal_us",
                  "overlap_efficiency","exposed_comm_us","bucket_bytes","n_buckets"):
            self.assertIn(f, P.CSV_COLUMNS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
