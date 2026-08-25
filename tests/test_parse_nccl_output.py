#!/usr/bin/env python3
"""Unit tests for scripts/parse_nccl_output.py.

Runs locally with no GPU and no third-party dependencies:

    python3 tests/test_parse_nccl_output.py

All input comes from tests/fixtures/, which contains hand-written SYNTHETIC
nccl-tests output. Those numbers are not measurements and the parser tags them
value_kind="synthetic" so they can never be mistaken for one.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import parse_nccl_output as P  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SCHEMA_PATH = REPO_ROOT / "schemas" / "nccl_result.schema.json"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class TestOutputParsing(unittest.TestCase):
    def test_modern_header_metadata(self):
        parsed = P.parse_nccl_tests_output(fixture("synthetic_all_reduce.txt"))
        meta = parsed["meta"]
        self.assertEqual(meta["n_gpus"], 2)
        self.assertEqual(meta["n_ranks"], 2)
        self.assertEqual(meta["warmup_iterations"], 5)
        self.assertEqual(meta["measured_iterations"], 20)
        self.assertTrue(meta["validation_enabled"])
        self.assertEqual(meta["hostname"], "synthetic-host")
        self.assertEqual(meta["gpu_model"], "NVIDIA RTX A5000")
        self.assertTrue(meta["out_of_bounds_ok"])
        self.assertTrue(meta["synthetic"])

    def test_modern_header_points(self):
        parsed = P.parse_nccl_tests_output(fixture("synthetic_all_reduce.txt"))
        points = parsed["points"]
        # 3 sizes x {out_of_place, in_place}
        self.assertEqual(len(points), 6)

        oop = [p for p in points if p["placement"] == "out_of_place"]
        self.assertEqual([p["message_size_bytes"] for p in oop], [8, 32768, 134217728])
        self.assertEqual([p["count_elements"] for p in oop], [2, 8192, 33554432])

        largest = oop[-1]
        self.assertEqual(largest["datatype"], "float")
        self.assertEqual(largest["redop"], "sum")
        self.assertEqual(largest["root"], -1)
        self.assertAlmostEqual(largest["latency_us"], 11000.00)
        self.assertAlmostEqual(largest["algorithmic_bandwidth_gbps"], 12.20)
        self.assertAlmostEqual(largest["bus_bandwidth_gbps"], 12.20)
        self.assertEqual(largest["wrong_count"], 0)

        # In-place values must not be confused with out-of-place ones.
        ip = [p for p in points if p["placement"] == "in_place"]
        self.assertAlmostEqual(ip[-1]["latency_us"], 10900.00)
        self.assertAlmostEqual(ip[-1]["algorithmic_bandwidth_gbps"], 12.31)

    def test_legacy_header_without_redop_and_root(self):
        """Column positions must come from the header, not fixed offsets."""
        parsed = P.parse_nccl_tests_output(fixture("synthetic_all_gather_legacy.txt"))
        points = parsed["points"]
        self.assertEqual(len(points), 4)

        oop = [p for p in points if p["placement"] == "out_of_place"]
        self.assertEqual([p["message_size_bytes"] for p in oop], [32768, 33554432])
        first = oop[0]
        self.assertIsNone(first["redop"])
        self.assertIsNone(first["root"])
        self.assertEqual(first["datatype"], "float")
        # A fixed-offset parser would read 0.82 (algbw) as the latency here.
        self.assertAlmostEqual(first["latency_us"], 40.00)
        self.assertAlmostEqual(first["algorithmic_bandwidth_gbps"], 0.82)
        self.assertAlmostEqual(first["bus_bandwidth_gbps"], 0.41)

    def test_failed_validation_is_visible(self):
        parsed = P.parse_nccl_tests_output(fixture("synthetic_all_reduce_failed.txt"))
        self.assertFalse(parsed["meta"]["out_of_bounds_ok"])
        wrongs = [p["wrong_count"] for p in parsed["points"]]
        self.assertIn(17, wrongs)

    def test_empty_input_is_not_a_crash(self):
        parsed = P.parse_nccl_tests_output("")
        self.assertEqual(parsed["points"], [])
        self.assertIsNone(parsed["meta"]["n_gpus"])


class TestBandwidthRatio(unittest.TestCase):
    """Hypothesis H5: busbw == algbw * factor(collective, ranks)."""

    def test_all_reduce_factor_is_one_at_two_ranks(self):
        self.assertTrue(P.check_bandwidth_ratio("all_reduce", 2, 12.20, 12.20))
        self.assertFalse(P.check_bandwidth_ratio("all_reduce", 2, 12.20, 6.10))

    def test_all_gather_factor_is_half_at_two_ranks(self):
        self.assertTrue(P.check_bandwidth_ratio("all_gather", 2, 0.82, 0.41))
        self.assertFalse(P.check_bandwidth_ratio("all_gather", 2, 0.82, 0.82))

    def test_all_reduce_at_four_ranks(self):
        # factor = 2*(4-1)/4 = 1.5
        self.assertTrue(P.check_bandwidth_ratio("all_reduce", 4, 10.0, 15.0))
        self.assertFalse(P.check_bandwidth_ratio("all_reduce", 4, 10.0, 10.0))

    def test_undecidable_at_printed_precision(self):
        # nccl-tests prints 2 decimals; 0.00 GB/s carries no ratio information.
        self.assertIsNone(P.check_bandwidth_ratio("all_reduce", 2, 0.0, 0.0))
        self.assertIsNone(P.check_bandwidth_ratio("all_reduce", None, 1.0, 1.0))


class TestEndToEnd(unittest.TestCase):
    def _make_raw_dir(self, tmp: Path, fixture_name: str, exit_code: int = 0) -> Path:
        raw = tmp / "results" / "raw" / "p1-test-20260826T000000Z-deadbee"
        raw.mkdir(parents=True)
        (raw / "all_reduce.smoke.r0.stdout.txt").write_text(
            fixture(fixture_name), encoding="utf-8")
        (raw / "all_reduce.smoke.r0.stderr.txt").write_text(
            "NCCL version 2.21.5+cuda12.4\n", encoding="utf-8")
        (raw / "env.json").write_text(json.dumps({
            "schema_version": 1,
            "captured_at_utc": "2026-08-26T00:00:00Z",
            "provider": "runpod",
            "provider_instance_id": "synthetic-pod",
            "hostname": "synthetic-host",
            "os": "Ubuntu 22.04.4 LTS",
            "kernel": "5.15.0",
            "cpu_model": "Synthetic CPU",
            "cpu_cores": 16,
            "gpu_count": 2,
            "gpu_model": "NVIDIA RTX A5000",
            "gpus": [{"index": 0, "name": "NVIDIA RTX A5000",
                      "pci_bus_id": "00000000:01:00.0", "memory_total_mib": 24564}],
            "driver_version": "550.54.15",
            "cuda_version": "12.4",
            "nccl_version": None,
            "mpi_version": None,
            "compiler_version": "gcc (Ubuntu) 11.4.0",
            "topology": "GPU0 GPU1\nGPU0  X   PIX\nGPU1 PIX   X",
            "topology_summary": "PIX",
            "nvlink_present": False,
            "git_commit": "deadbeef",
            "git_dirty": False,
            "env": {"NCCL_DEBUG": "VERSION"},
        }), encoding="utf-8")
        (raw / "run_manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "experiment_id": "p1-test-20260826T000000Z-deadbee",
            "phase": "phase1",
            "created_at_utc": "2026-08-26T00:00:00Z",
            "value_kind": "measured",
            "benchmark_tool": "nccl-tests",
            "nccl_tests_commit": "cafebabe",
            "git_commit": "deadbeef",
            "node_count": 1,
            "gpu_count": 2,
            "network": "none-single-node",
            "runs": [{
                "collective": "all_reduce",
                "binary": "all_reduce_perf",
                "tier": "smoke",
                "repeat_index": 0,
                "warmup_iterations": 5,
                "measured_iterations": 20,
                "datatype": "float",
                "gpu_count": 2,
                "command": "./build/all_reduce_perf -b 8 -e 128M -f 4096 -g 2 -w 5 -n 20 -c 1 -d float",
                "exit_code": exit_code,
                "started_at_utc": "2026-08-26T00:00:01Z",
                "stdout_file": "all_reduce.smoke.r0.stdout.txt",
                "stderr_file": "all_reduce.smoke.r0.stderr.txt",
            }],
        }), encoding="utf-8")
        return raw

    def test_rows_are_built_and_tagged_synthetic(self):
        with tempfile.TemporaryDirectory() as td:
            raw = self._make_raw_dir(Path(td), "synthetic_all_reduce.txt")
            rows, problems = P.build_rows(raw, "phase1")

            self.assertEqual(len(rows), 6)
            row = rows[0]
            self.assertEqual(row["experiment_id"], "p1-test-20260826T000000Z-deadbee")
            self.assertEqual(row["collective"], "all_reduce")
            self.assertEqual(row["gpu_count"], 2)
            self.assertEqual(row["network"], "none-single-node")
            self.assertEqual(row["nccl_tests_commit"], "cafebabe")
            self.assertEqual(row["provider"], "runpod")
            self.assertEqual(row["tier"], "smoke")
            self.assertTrue(row["correctness_ok"])

            # The stderr banner is the definitive NCCL version for the run.
            self.assertEqual(row["nccl_version"], "2.21.5+cuda12.4".rstrip("+"))
            self.assertEqual(row["nccl_version_source"], "NCCL_DEBUG=VERSION banner")

            # Integrity interlock: fabricated input can never claim to be measured.
            self.assertTrue(all(r["value_kind"] == "synthetic" for r in rows))
            self.assertTrue(any("SYNTHETIC" in p for p in problems))

    def test_failed_run_marks_rows_incorrect(self):
        with tempfile.TemporaryDirectory() as td:
            raw = self._make_raw_dir(Path(td), "synthetic_all_reduce_failed.txt", exit_code=1)
            rows, problems = P.build_rows(raw, "phase1")
            self.assertTrue(rows)
            self.assertTrue(all(not r["correctness_ok"] for r in rows))
            self.assertTrue(any("correctness failure" in p for p in problems))

    def test_manifest_absent_falls_back_to_filenames(self):
        with tempfile.TemporaryDirectory() as td:
            raw = self._make_raw_dir(Path(td), "synthetic_all_reduce.txt")
            (raw / "run_manifest.json").unlink()
            rows, _ = P.build_rows(raw, "phase1")
            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[0]["collective"], "all_reduce")
            self.assertEqual(rows[0]["tier"], "smoke")
            self.assertEqual(rows[0]["repeat_index"], 0)
            self.assertEqual(rows[0]["command"], "<unrecorded>")

    def test_csv_and_jsonl_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            raw = self._make_raw_dir(Path(td), "synthetic_all_reduce.txt")
            rows, _ = P.build_rows(raw, "phase1")
            out = Path(td) / "summary"
            out.mkdir()
            P.write_jsonl(rows, out / "results.jsonl")
            P.write_csv(rows, out / "results.csv")

            lines = (out / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), len(rows))
            self.assertEqual(json.loads(lines[0])["collective"], "all_reduce")

            csv_text = (out / "results.csv").read_text(encoding="utf-8")
            self.assertEqual(len(csv_text.strip().splitlines()), len(rows) + 1)
            self.assertIn("experiment_id", csv_text.splitlines()[0])


class TestSchemaConformance(unittest.TestCase):
    """Validate emitted rows against schemas/nccl_result.schema.json.

    Uses jsonschema when available; otherwise falls back to checking required
    fields, unknown fields, and enum membership by hand so the test still has
    teeth in a bare Python environment.
    """

    def setUp(self):
        self.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    def _rows(self, td: str):
        raw = TestEndToEnd()._make_raw_dir(Path(td), "synthetic_all_reduce.txt")
        rows, _ = P.build_rows(raw, "phase1")
        return rows

    def test_rows_conform(self):
        with tempfile.TemporaryDirectory() as td:
            rows = self._rows(td)
            try:
                import jsonschema  # type: ignore
            except ImportError:
                jsonschema = None

            if jsonschema is not None:
                for row in rows:
                    jsonschema.validate(row, self.schema)
                return

            props = self.schema["properties"]
            required = set(self.schema["required"])
            for row in rows:
                missing = required - set(row)
                self.assertFalse(missing, f"missing required fields: {missing}")
                unknown = set(row) - set(props)
                self.assertFalse(unknown, f"fields absent from schema: {unknown}")
                for field in ("placement", "collective", "value_kind"):
                    allowed = props[field].get("enum")
                    if allowed:
                        self.assertIn(row[field], allowed)

    def test_csv_columns_cover_schema_fields(self):
        """A field in the schema but missing from CSV_COLUMNS is silently dropped."""
        schema_fields = set(self.schema["properties"])
        missing = schema_fields - set(P.CSV_COLUMNS)
        self.assertFalse(missing, f"CSV view would drop: {sorted(missing)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
