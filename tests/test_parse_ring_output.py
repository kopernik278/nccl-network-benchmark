#!/usr/bin/env python3
"""Unit tests for scripts/parse_ring_output.py.

Zero-cost, no GPU. Input is hand-written SYNTHETIC benchmark output, which the
parser tags value_kind="synthetic" so it can never be mistaken for a
measurement.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import parse_ring_output as P  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures"
SCHEMA_PATH = REPO_ROOT / "schemas" / "nccl_result.schema.json"


def make_raw(tmp: Path, fixture: str) -> Path:
    raw = tmp / "results" / "raw" / "p6-test-20260829T000000Z-deadbee"
    raw.mkdir(parents=True)
    (raw / "ring_allreduce.stdout.txt").write_text(
        (FIX / fixture).read_text(encoding="utf-8"), encoding="utf-8")
    (raw / "env.json").write_text(json.dumps({
        "schema_version": 1, "captured_at_utc": "2026-08-29T00:00:00Z",
        "provider": "runpod", "hostname": "synthetic-host",
        "os": "Ubuntu 24.04", "cpu_model": "Synthetic CPU", "cpu_cores": 8,
        "gpu_count": 2, "gpu_model": "NVIDIA SYNTHETIC",
        "driver_version": "570.0", "cuda_version": "12.8",
        "nccl_version": "2.25.1", "compiler_version": "gcc 13",
        "topology_summary": "PIX", "nvlink_present": False,
        "git_commit": "deadbeef", "git_dirty": False, "env": {},
    }), encoding="utf-8")
    (raw / "run_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "experiment_id": "p6-test-20260829T000000Z-deadbee",
        "phase": "phase6", "created_at_utc": "2026-08-29T00:00:00Z",
        "value_kind": "measured", "benchmark_tool": "custom-ring-allreduce",
        "git_commit": "deadbeef", "node_count": 1, "gpu_count": 2,
        "network": "none-single-node", "transport": "p2p-direct",
        "command": "./build/ring_allreduce -g 2 -w 5 -n 20", "exit_code": 0,
    }), encoding="utf-8")
    return raw


class TestRingParsing(unittest.TestCase):
    def test_all_implementations_parsed(self):
        with tempfile.TemporaryDirectory() as td:
            rows, _ = P.build_rows(make_raw(Path(td), "synthetic_ring_output.txt"))
            self.assertEqual(len(rows), 4)
            self.assertEqual(sorted(r["config_label"] for r in rows),
                             ["nccl-reference", "v1-naive", "v2-async", "v3-pipelined-sub4"])

    def test_subchunk_count_is_in_the_label(self):
        """v3 rows must be distinguishable by subchunk count, not collapsed."""
        with tempfile.TemporaryDirectory() as td:
            rows, _ = P.build_rows(make_raw(Path(td), "synthetic_ring_output.txt"))
            v3 = [r for r in rows if r["config_label"].startswith("v3")]
            self.assertEqual(v3[0]["config_label"], "v3-pipelined-sub4")

    def test_bandwidth_is_recomputed_not_trusted(self):
        with tempfile.TemporaryDirectory() as td:
            rows, _ = P.build_rows(make_raw(Path(td), "synthetic_ring_output.txt"))
            r = next(x for x in rows if x["config_label"] == "v1-naive")
            # 1048576 B / 500 us = 2.097 GB/s; busbw factor at n=2 is 2*(1)/2 = 1.0
            self.assertAlmostEqual(r["algorithmic_bandwidth_gbps"], 2.097152, places=4)
            self.assertAlmostEqual(r["bus_bandwidth_gbps"], 2.097152, places=4)

    def test_busbw_factor_matches_earlier_phases(self):
        self.assertAlmostEqual(P.busbw_factor(2), 1.0)
        self.assertAlmostEqual(P.busbw_factor(4), 1.5)
        self.assertAlmostEqual(P.busbw_factor(8), 1.75)

    def test_nccl_reference_is_tagged_as_a_different_tool(self):
        """The reference must not be mislabelled as our implementation."""
        with tempfile.TemporaryDirectory() as td:
            rows, _ = P.build_rows(make_raw(Path(td), "synthetic_ring_output.txt"))
            by = {r["config_label"]: r for r in rows}
            self.assertEqual(by["nccl-reference"]["benchmark_tool"], "nccl")
            self.assertEqual(by["v1-naive"]["benchmark_tool"], "custom-ring-allreduce")

    def test_synthetic_marker_downgrades_value_kind(self):
        with tempfile.TemporaryDirectory() as td:
            rows, problems = P.build_rows(make_raw(Path(td), "synthetic_ring_output.txt"))
            self.assertTrue(all(r["value_kind"] == "synthetic" for r in rows))
            self.assertTrue(any("SYNTHETIC" in p for p in problems))

    def test_failed_correctness_drops_the_timing(self):
        with tempfile.TemporaryDirectory() as td:
            rows, problems = P.build_rows(make_raw(Path(td), "synthetic_ring_output_bad.txt"))
            self.assertEqual(len(rows), 1)
            self.assertFalse(rows[0]["correctness_ok"])
            self.assertIsNone(rows[0]["latency_us"])
            self.assertIsNone(rows[0]["algorithmic_bandwidth_gbps"])
            self.assertEqual(rows[0]["wrong_count"], 4096)
            self.assertTrue(any("mismatches" in p for p in problems))

    def test_transport_is_carried_through(self):
        with tempfile.TemporaryDirectory() as td:
            rows, _ = P.build_rows(make_raw(Path(td), "synthetic_ring_output.txt"))
            self.assertTrue(all(r["transport"] == "p2p-direct" for r in rows))
            self.assertTrue(all(r["p2p_enabled"] for r in rows))


class TestRingSchemaConformance(unittest.TestCase):
    def test_rows_conform_to_the_shared_schema(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        props, required = set(schema["properties"]), set(schema["required"])
        with tempfile.TemporaryDirectory() as td:
            rows, _ = P.build_rows(make_raw(Path(td), "synthetic_ring_output.txt"))
            for r in rows:
                self.assertFalse(required - set(r), f"missing: {required - set(r)}")
                self.assertFalse(set(r) - props, f"not in schema: {set(r) - props}")
                self.assertIn(r["collective"], schema["properties"]["collective"]["enum"])
                self.assertIn(r["value_kind"], schema["properties"]["value_kind"]["enum"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
