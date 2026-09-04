from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    "hw_autoresearch_nts07/system_handoff/scripts/"
    "run_m1177_motion_ep29_e1e8_closure_source.py"
)


def load_source():
    spec = importlib.util.spec_from_file_location("m1177_under_test", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class M1177SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_source()

    def test_source_contract_cannot_authorize_production(self):
        contract = self.m.strict_json(self.m.SOURCE_CONTRACT)
        self.assertEqual(contract["source"]["sha256"], self.m.sha256(SOURCE))
        with self.assertRaisesRegex(self.m.ClosureError, "source-only contract"):
            self.m.validate_launch(contract, self.m.SOURCE_CONTRACT)

    def test_fixed_deploy_modes_are_exact_and_do_not_mutate_source(self):
        source = {
            "experiment": "fixture",
            "bsa_attention": {"enabled": True, "hardware_quant_enabled": False},
            "runtime": {"unchanged": 7},
        }
        before = json.dumps(source, sort_keys=True)
        modes = self.m.fixed_deploy_configs(source)
        self.assertEqual(json.dumps(source, sort_keys=True), before)
        self.assertEqual(set(modes), {"dyadic", "hardware_order"})
        self.assertFalse(modes["dyadic"]["bsa_attention"]["hardware_rtl_shiftmax_enabled"])
        self.assertTrue(modes["hardware_order"]["bsa_attention"]["hardware_rtl_shiftmax_enabled"])
        for mode in modes.values():
            attention = mode["bsa_attention"]
            self.assertEqual(attention["hardware_score_step"], 1.0 / 128.0)
            self.assertEqual(attention["hardware_gate_step"], 1.0 / 128.0)
            self.assertFalse(mode["runtime"]["deployment_contract"][
                "candidate_selection_or_parameter_search"])

    def test_deploy_config_rejects_missing_attention(self):
        with self.assertRaisesRegex(self.m.ClosureError, "bsa_attention"):
            self.m.fixed_deploy_configs({"bsa_attention": {"enabled": False}})

    def test_dyadic_quantization_conv_and_transpose_axes(self):
        weight = np.asarray([
            [[1.0, -0.5], [0.0, 0.25]],
            [[2.0, -1.0], [0.5, 0.0]],
        ], dtype=np.float32)
        normal = self.m.quantize_dyadic_per_output(weight, 0)
        transposed = self.m.quantize_dyadic_per_output(weight, 1)
        self.assertEqual(normal["code"].shape, (2, 2, 2))
        self.assertEqual(transposed["code"].shape[0], 2)
        self.assertEqual(normal["preclip_violations"], 0)
        self.assertEqual(transposed["preclip_violations"], 0)
        self.assertTrue(np.all(normal["code"] >= -127))
        self.assertTrue(np.all(normal["code"] <= 127))
        self.assertNotIn(-128, normal["code"])
        self.assertLessEqual(normal["compression"]["selected_bytes"],
                             normal["compression"]["dense_int8_bytes"])

    def test_quantization_rejects_nonfinite_and_bad_axis(self):
        with self.assertRaisesRegex(self.m.ClosureError, "NaN/Infinity"):
            self.m.quantize_dyadic_per_output(
                np.asarray([[1.0, np.nan]], dtype=np.float32), 0)
        with self.assertRaisesRegex(self.m.ClosureError, "geometry"):
            self.m.quantize_dyadic_per_output(np.ones((2, 2), dtype=np.float32), 2)

    def test_tensor_summary_and_signed_width_boundaries(self):
        summary = self.m.tensor_summary(np.asarray([-3.0, 0.0, 2.0], dtype=np.float32))
        self.assertEqual(summary["zero"], 1)
        self.assertEqual(summary["maximum_absolute"], 3.0)
        self.assertEqual(self.m.signed_bits_for_bounds(-262144, 262143), 19)
        self.assertEqual(self.m.signed_bits_for_bounds(-262145, 262144), 20)
        with self.assertRaisesRegex(self.m.ClosureError, "NaN/Infinity"):
            self.m.tensor_summary(np.asarray([np.inf], dtype=np.float32))

    def test_bool_is_not_an_exact_integer(self):
        with self.assertRaisesRegex(self.m.ClosureError, "exact integer"):
            self.m.exact_int(True, "fixture")

    def test_strict_json_rejects_duplicate_and_nan(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            path.write_text('{"a":1,"a":2}', encoding="utf-8")
            with self.assertRaisesRegex(self.m.ClosureError, "duplicate"):
                self.m.strict_json(path)
            path.write_text('{"a":NaN}', encoding="utf-8")
            with self.assertRaisesRegex(self.m.ClosureError, "non-standard"):
                self.m.strict_json(path)

    def test_profile_identity_and_metric_attacks_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"checkpoint-fixture")
            config = root / "config.yml"
            config.write_text("fixture: true\n", encoding="utf-8")
            identity = {
                "config_path": str(config.resolve()),
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_size": checkpoint.stat().st_size,
                "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
            profile = {
                "samples": 825,
                "artifact_identity": identity,
                "checkpoint_load_audit": {"missing_count": 0, "unexpected_count": 0},
                "module_counts": {"ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12},
                "metrics": {"AEE": 1.2, "AAE": 5.4, "AAE_Benchmark": 5.1},
            }
            metrics = self.m.validate_profile(profile, expected_config=config,
                                              checkpoint=checkpoint)
            self.assertEqual(metrics["AEE"], 1.2)
            attacked = json.loads(json.dumps(profile))
            attacked["samples"] = True
            with self.assertRaisesRegex(self.m.ClosureError, "exact integer"):
                self.m.validate_profile(attacked, expected_config=config,
                                        checkpoint=checkpoint)
            attacked = json.loads(json.dumps(profile))
            attacked["metrics"]["AEE"] = float("nan")
            with self.assertRaisesRegex(self.m.ClosureError, "finite"):
                self.m.validate_profile(attacked, expected_config=config,
                                        checkpoint=checkpoint)

    def test_static_source_has_no_parameter_sweep_or_automatic_retry(self):
        text = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("for threshold in", text)
        self.assertNotIn("for epsilon in", text)
        self.assertNotIn("while retry", text)
        self.assertEqual(text.count("profile.build_model("), 1)
        self.assertIn("candidate_selection_or_parameter_search", text)

    def test_docs359_identity_unchanged(self):
        self.assertEqual(self.m.sha256(self.m.DOCS359), self.m.DOCS359_SHA256)


if __name__ == "__main__":
    unittest.main()
