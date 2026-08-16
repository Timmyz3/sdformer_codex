#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("build_local5_compact_telemetry_preledger_v1.py")
SPEC = importlib.util.spec_from_file_location("compact_telemetry_v1", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def valid_line(heads: int = 3) -> str:
    partial = heads * heads * 450 * 32
    final = heads * 450 * 32
    values = {
        "memo": 0, "inplace": 0, "acc_backend": 0, "tx_service": 1,
        "seed": 17, "stage": 0, "block": 0, "window": 4,
        "cycles": 99, "token": heads * heads * 450,
        "token_delay_sum": MODULE.transaction_delay_sum(17, 0, heads * heads * 450),
        "weight_delay_sum": MODULE.transaction_delay_sum(17, 1, heads * heads * 32 * 32),
        "result_service": final, "hits": 0, "fallback": 0,
        "replay_records": 0, "partial": partial, "final": final,
        "child_results": partial,
        "weight_cycles": 2 * heads * heads * 32 * 32
            + MODULE.transaction_delay_sum(17, 1, heads * heads * 32 * 32),
        "frontend_cycles": 1, "readout_cycles": 3 * partial,
        "release_cycles": 2 * heads * heads,
        "rmw_cycles": heads * (heads - 1) * 450 * 32,
        "drain_cycles": 3 * final
            + MODULE.transaction_delay_sum(17, 2, final),
        "scheduler_cycles": 2 * heads * heads,
        "vector": 0, "token_service_hash": "1" * 16,
        "weight_service_hash": "2" * 16, "result_service_hash": "3" * 16,
    }
    return "PASS Local5 multi-tile " + " ".join(
        f"{key}={value}" for key, value in values.items()
    ) + "\n- tb.sv:1: Verilog $finish\n"


class CompactTelemetryV1Test(unittest.TestCase):
    def sealed_v2_fixture(self, directory: Path) -> tuple[dict, Path]:
        runtime_binary = directory / "python"
        numpy_file = directory / "numpy.py"
        runtime_binary.write_bytes(b"python-fixture")
        numpy_file.write_bytes(b"numpy-fixture")
        runtime = {
            "schema": "local5_numeric_batch_runtime_environment_v1",
            "status": "FROZEN_EXACT_RUNTIME",
            "resolved_executable": str(runtime_binary),
            "executable_sha256": hashlib.sha256(runtime_binary.read_bytes()).hexdigest(),
            "numpy_file": str(numpy_file),
            "numpy_file_sha256": hashlib.sha256(numpy_file.read_bytes()).hexdigest(),
        }
        runtime_path = directory / "runtime_environment.json"
        runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
        snapshot_dir = directory / "source"
        snapshot_dir.mkdir()
        snapshots = []
        for index in range(3):
            path = snapshot_dir / f"source{index}.py"
            path.write_text(f"source-{index}\n", encoding="utf-8")
            snapshots.append({
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        source = {"path": str(snapshots[0]["path"]), "sha256": snapshots[0]["sha256"]}
        batch = {
            "schema": "local5_numeric_sample_batch_complete_v2",
            "status": "PASS_NUMERIC_SAMPLE_BATCH_NOT_G0",
            "formal_g0": "DENY",
            "mismatch": 0,
            "rows": [],
            "source": source,
            "runtime_environment": runtime,
            "runtime_environment_sha256": hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
            "source_snapshots": snapshots,
            "origin_policy": "SELF_FIRST_EXECUTION",
        }
        batch_path = directory / "complete.json"
        batch_path.write_text(json.dumps(batch), encoding="utf-8")
        return batch, batch_path

    def test_accepts_env_sealed_v2_batch_header(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            batch, batch_path = self.sealed_v2_fixture(Path(name))
            self.assertIs(
                MODULE.validate_parent_batch_header(batch, batch_path),
                batch["source"],
            )

    def test_rejects_runtime_sidecar_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            batch, batch_path = self.sealed_v2_fixture(Path(name))
            (Path(name) / "runtime_environment.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "runtime sidecar"):
                MODULE.validate_parent_batch_header(batch, batch_path)

    def test_rejects_source_snapshot_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            batch, batch_path = self.sealed_v2_fixture(Path(name))
            Path(batch["source_snapshots"][1]["path"]).write_text("tampered")
            with self.assertRaisesRegex(ValueError, "source snapshot"):
                MODULE.validate_parent_batch_header(batch, batch_path)

    def test_rejects_unsealed_v2_batch_header(self) -> None:
        batch = {
            "schema": "local5_numeric_sample_batch_complete_v2",
            "status": "PASS_NUMERIC_SAMPLE_BATCH_NOT_G0",
            "formal_g0": "DENY",
            "mismatch": 0,
            "rows": [],
            "source": {},
        }
        with self.assertRaisesRegex(ValueError, "runtime/source seal"):
            MODULE.validate_parent_batch_header(batch, Path("/tmp/complete.json"))

    def test_parse_and_formula_accept(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sim.log"
            path.write_text(valid_line(), encoding="utf-8")
            row = MODULE.parse_telemetry_log(path)
            MODULE.validate_telemetry_formula(row, 3)

    def test_formula_rejects_partial_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sim.log"
            path.write_text(valid_line().replace("partial=129600", "partial=129599"))
            row = MODULE.parse_telemetry_log(path)
            with self.assertRaises(ValueError):
                MODULE.validate_telemetry_formula(row, 3)

    def test_parser_rejects_duplicate_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sim.log"
            path.write_text(valid_line() + valid_line())
            with self.assertRaises(ValueError):
                MODULE.parse_telemetry_log(path)

    def test_transaction_delay_matches_real_h3_anchor(self) -> None:
        self.assertEqual(MODULE.transaction_delay_sum(17828, 0, 4050), 10108)
        self.assertEqual(MODULE.transaction_delay_sum(17828, 1, 9216), 23156)
        self.assertEqual(MODULE.transaction_delay_sum(17828, 2, 43200), 108297)


if __name__ == "__main__":
    unittest.main()
