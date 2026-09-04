import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_h67_baseline_registry.py"
SPEC = importlib.util.spec_from_file_location("h67_baseline_registry", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BaselineRegistryTest(unittest.TestCase):
    def setUp(self):
        self.summary = {
            "status": MODULE.LEDGER_STATUS,
            "cycles_per_frame_model": {"fixed_total": 100, "rqtb_total": 80},
        }

    def test_native_and_blocked_external_rows(self):
        config = {
            "common_contract": {"frequency_mhz": 333},
            "baselines": [
                {"name": "Fixed2S", "kind": "native", "cycle_key": "fixed_total", "status": MODULE.NATIVE_STATUS},
                {"name": "External", "kind": "external_architecture", "status": "BLOCKED_TEST", "missing": ["adapter"]},
            ],
        }
        result = MODULE.build_registry(self.summary, config)
        self.assertEqual(result["baselines"][0]["cycles_per_frame_envelope"], 100)
        self.assertIsNone(result["baselines"][1]["cycles_per_frame_envelope"])
        self.assertFalse(result["paper_comparison_ready"])

    def test_external_row_cannot_silently_claim_admission(self):
        config = {
            "common_contract": {},
            "baselines": [
                {"name": "External", "kind": "external_architecture", "status": "ADMITTED", "missing": []}
            ],
        }
        with self.assertRaises(RuntimeError):
            MODULE.build_registry(self.summary, config)


if __name__ == "__main__":
    unittest.main()
