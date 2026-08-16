from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import local5_erep_g2_contract_v1 as runtime


def write(path: Path, value: str) -> str:
    path.write_text(value, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root: Path) -> tuple[Path, Path]:
    contract = json.loads(runtime.CONTRACT.read_text(encoding="utf-8"))
    required = contract["g2b_target_asic_receipt"]["required_fields"]
    shared = {}
    artifacts = {}
    for name in required:
        if name.endswith("_sha256"):
            artifact = root / f"{name}.txt"
            shared[name] = write(artifact, name)
            artifacts[name] = artifact.name
        else:
            shared[name] = f"value-{name}"
    shared.update(
        {
            "supply_voltage_v": 0.8,
            "junction_temperature_c": 25.0,
            "pvt_corner": "ss_0p8v_125c",
            "sdc_path": artifacts["sdc_sha256"],
        }
    )
    common = {
        "boundary_id": runtime.BOUNDARY,
        "clock_period_ns": 5.0,
        "pvt_corner": shared["pvt_corner"],
        "sdc_sha256": shared["sdc_sha256"],
        "memory_macro_policy_sha256": shared["sram_macro_port_latency_contract_sha256"],
        "common_activity_stimulus_sha256": shared["common_activity_stimulus_sha256"],
        "idle_clock_gating_policy_sha256": shared["idle_clock_gating_policy_sha256"],
    }
    candidates = []
    for candidate in runtime.CANDIDATES:
        candidates.append(
            {
                "id": candidate,
                **common,
                "rtl_sha256": "1" * 64,
                "filelist_sha256": "2" * 64,
                "parameter_sha256": "3" * 64,
            }
        )
    receipt = root / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "local5_erep_g2b_run_receipt_v1",
                "status": "READY",
                "shared": shared,
                "artifacts": artifacts,
                "candidates": candidates,
            }
        ),
        encoding="utf-8",
    )
    results = root / "results.json"
    rows = []
    for candidate in runtime.CANDIDATES:
        rows.append(
            {
                "id": candidate,
                "timing_pass": True,
                "weighted_energy_joule": "1.0",
                "weighted_latency_second": "1.0",
                "activity_annotation_coverage_percent": "99.0",
                "unknown_toggle_count": 0,
            }
        )
    rows[3]["weighted_energy_joule"] = "0.75"
    results.write_text(
        json.dumps(
            {
                "schema": "local5_erep_g2_result_bundle_v1",
                "g2b_receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                "candidates": rows,
            }
        ),
        encoding="utf-8",
    )
    return receipt, results


class Local5ErepG2RuntimeV1Test(unittest.TestCase):
    def test_complete_common_receipt_and_exact_gates_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt, results = fixture(Path(directory))
            preflight = runtime.validate_g2b_preflight(receipt)
            report = runtime.evaluate_g2_results(results, receipt)
            self.assertEqual(preflight["status"], "PASS")
            self.assertTrue(report["g2_passed"])
            self.assertEqual([gate["ratio_threshold"] for gate in report["gates"]], ["5/4", "20/19"])

    def test_empty_fake_sha_and_missing_artifact_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt, _ = fixture(root)
            value = json.loads(receipt.read_text())
            for replacement in (None, "", "not-a-sha"):
                changed = json.loads(receipt.read_text())
                changed["shared"]["dc_command_sha256"] = replacement
                receipt.write_text(json.dumps(changed))
                with self.subTest(replacement=replacement):
                    with self.assertRaises(ValueError):
                        runtime.validate_g2b_preflight(receipt)
            receipt, _ = fixture(root)
            value = json.loads(receipt.read_text())
            (root / value["artifacts"]["dc_command_sha256"]).unlink()
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                runtime.validate_g2b_preflight(receipt)

    def test_candidate_private_clock_pvt_sdc_macro_or_activity_fails(self) -> None:
        fields = (
            ("clock_period_ns", 6.0),
            ("pvt_corner", "ff"),
            ("sdc_sha256", "f" * 64),
            ("memory_macro_policy_sha256", "e" * 64),
            ("common_activity_stimulus_sha256", "d" * 64),
            ("idle_clock_gating_policy_sha256", "c" * 64),
        )
        for field, replacement in fields:
            with tempfile.TemporaryDirectory() as directory:
                receipt, _ = fixture(Path(directory))
                value = json.loads(receipt.read_text())
                value["candidates"][3][field] = replacement
                receipt.write_text(json.dumps(value))
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ValueError, "violates common"):
                        runtime.validate_g2b_preflight(receipt)

    def test_timing_unknown_and_low_annotation_fail_before_edp(self) -> None:
        changes = (
            ("timing_pass", False, "timing"),
            ("unknown_toggle_count", 1, "unknown"),
            ("activity_annotation_coverage_percent", "94.9", "coverage"),
        )
        for field, replacement, message in changes:
            with tempfile.TemporaryDirectory() as directory:
                receipt, results = fixture(Path(directory))
                value = json.loads(results.read_text())
                value["candidates"][3][field] = replacement
                results.write_text(json.dumps(value))
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ValueError, message):
                        runtime.evaluate_g2_results(results, receipt)

    def test_thresholds_are_exact_and_cannot_be_reinterpreted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt, results = fixture(Path(directory))
            value = json.loads(results.read_text())
            value["candidates"][3]["weighted_energy_joule"] = "0.8000000000000000000000000001"
            results.write_text(json.dumps(value))
            report = runtime.evaluate_g2_results(results, receipt)
            self.assertFalse(report["gates"][0]["passed"])
            self.assertFalse(report["g2_passed"])


if __name__ == "__main__":
    unittest.main()
