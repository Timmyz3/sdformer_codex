from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / (
    "system_handoff/scripts/run_m1177r2_motion_ep29_e1e8_closure_source.py"
)
CONTRACT = HW / "contracts/m1177r2_motion_ep29_e1e8_source_contract_r1_20260830.json"


def load_source():
    spec = importlib.util.spec_from_file_location("m1177r2_under_test", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ATLIFTernaryPSN(torch.nn.Module):
    def forward(self, value):
        return value


class FixtureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 3, 3, bias=False)
        self.bn = torch.nn.BatchNorm2d(3)
        self.atlif = ATLIFTernaryPSN()
        self.deconv = torch.nn.ConvTranspose2d(3, 2, 3)
        self.fc = torch.nn.Linear(2, 2)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal_directory(directory: Path, members: dict[str, dict]) -> dict[str, str]:
    for name, payload in members.items():
        (directory / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                                      encoding="utf-8")
    manifest = directory / "SHA256SUMS"
    manifest.write_text("".join("{}  {}\n".format(sha(directory / name), name)
                                for name in sorted(members)), encoding="utf-8")
    outer = directory / "SHA256SUMS.seal.sha256"
    outer.write_text("{}  SHA256SUMS\n".format(sha(manifest)), encoding="utf-8")
    return {"manifest_sha256": sha(manifest), "outer_sha256": sha(outer)}


class FixtureAuthority:
    def __init__(self, case, mode="e1"):
        self.case = case
        self.m = case.m
        self.temporary = tempfile.TemporaryDirectory(prefix=".m1177r2_test.",
                                                     dir=HW / "results")
        self.root = Path(self.temporary.name)
        self.original = {name: getattr(self.m, name) for name in (
            "DOCS359", "DOCS359_SHA256", "M1175_REVIEW", "M1175_REVIEW_SHA256",
            "PROFILE", "PROFILE_SHA256", "EVALUATOR", "EVALUATOR_SHA256",
            "EXPECTED_SOURCE_HAMMER_REVIEW", "EXPECTED_CHECKPOINT",
            "EXPECTED_CONFIG_SHA256")}
        self.checkpoint = self.root / "checkpoint.pth"
        self.checkpoint.write_bytes(b"m1177r2-checkpoint-fixture")
        self.config = self.root / "config.yml"
        self.config.write_text("bsa_attention:\n  enabled: true\n", encoding="utf-8")
        self.docs = self.root / "docs359"
        self.docs.write_text("protected fixture\n", encoding="utf-8")
        self.profile = self.root / "profile.py"
        self.profile.write_text("# pinned fixture profiler\n", encoding="utf-8")
        self.evaluator = self.root / "evaluator.py"
        self.evaluator.write_text("# pinned fixture evaluator\n", encoding="utf-8")
        self.m.DOCS359 = self.docs
        self.m.DOCS359_SHA256 = sha(self.docs)
        self.m.PROFILE = self.profile
        self.m.PROFILE_SHA256 = sha(self.profile)
        self.m.EVALUATOR = self.evaluator
        self.m.EVALUATOR_SHA256 = sha(self.evaluator)
        self.m.EXPECTED_CHECKPOINT = {
            "epoch": 29, "sha256": sha(self.checkpoint),
            "size_bytes": self.checkpoint.stat().st_size,
            "mtime_ns": self.checkpoint.stat().st_mtime_ns,
        }
        self.m.EXPECTED_CONFIG_SHA256 = sha(self.config)
        self.m1175 = self.root / "m1175.json"
        self.write_m1175("PASS")
        self.m.M1175_REVIEW = self.m1175
        self.m.M1175_REVIEW_SHA256 = sha(self.m1175)
        self.hammer_dir = self.root / "hammer"
        self.hammer_dir.mkdir()
        self.hammer_review = self.hammer_dir / "review.json"
        self.m.EXPECTED_SOURCE_HAMMER_REVIEW = self.hammer_review
        self.hammer_declared = self.write_hammer(
            "PASS_SOURCE_HAMMER__RELEASE_AUTHORING_ALLOWED")
        self.launch_path = self.root / "launch.json"
        self.contract = self.make_contract(mode)

    def write_m1175(self, status):
        payload = {
            "schema": "m1175_m1171_motion_final_checkpoint_binder_result_hammer_r1_v1",
            "status": status,
            "verified": {"samples825_and_zero_load_audits": True,
                         "module_counts_atlif105_attention12": True,
                         "semantic_mutations_fail_closed": True},
            "selection": {"epoch": 29,
                          "checkpoint_sha256": self.m.EXPECTED_CHECKPOINT["sha256"],
                          "checkpoint_size_bytes": self.m.EXPECTED_CHECKPOINT["size_bytes"],
                          "checkpoint_mtime_ns": self.m.EXPECTED_CHECKPOINT["mtime_ns"],
                          "configuration_sha256": self.m.EXPECTED_CONFIG_SHA256,
                          "samples": 825, "AEE": "1.209876834190253",
                          "AAE": "5.406798340046045",
                          "AAE_Benchmark": "5.148612399245754"},
            "authorization_after_hammer": {
                "E0_final_checkpoint_and_deployment_identity": "ADMITTED"},
        }
        self.m1175.write_text(json.dumps(payload), encoding="utf-8")

    def write_hammer(self, status):
        for item in self.hammer_dir.iterdir():
            if item.is_file():
                item.unlink()
        review = {
            "schema": "m1181_m1177r2_motion_ep29_e1e8_source_hammer_review_r1_v1",
            "status": status, "production_authorized": False,
            "artifacts": {
                "source": {"path": str(SOURCE.relative_to(ROOT)), "sha256": sha(SOURCE)},
                "contract": {"path": str(CONTRACT.relative_to(ROOT)),
                             "sha256": sha(CONTRACT)},
                "tests": {"path": str(Path(__file__).resolve().relative_to(ROOT)),
                          "sha256": sha(Path(__file__).resolve())},
            },
            "verified": {"B{}".format(index): True for index in range(1, 9)},
        }
        seals = seal_directory(self.hammer_dir, {"review.json": review})
        return {"path": str(self.hammer_review.relative_to(ROOT)),
                "review_sha256": sha(self.hammer_review), **seals}

    def make_contract(self, mode):
        common = {
            "source": {"path": str(SOURCE.relative_to(ROOT)), "sha256": sha(SOURCE)},
            "selection": {"epoch": 29,
                          "checkpoint_sha256": self.m.EXPECTED_CHECKPOINT["sha256"],
                          "checkpoint_size_bytes": self.m.EXPECTED_CHECKPOINT["size_bytes"],
                          "checkpoint_mtime_ns": self.m.EXPECTED_CHECKPOINT["mtime_ns"],
                          "config_sha256": self.m.EXPECTED_CONFIG_SHA256,
                          "standard_valid825": self.m.EXPECTED_STANDARD},
            "checkpoint_path": str(self.checkpoint), "config_path": str(self.config),
            "m1175_result_hammer": {"path": str(self.m1175.relative_to(ROOT)),
                                    "sha256": sha(self.m1175)},
            "m1177r2_source_hammer": self.hammer_declared,
        }
        contract = {
            "schema": "m1177r2_motion_ep29_e1e8_launch_v2",
            "status": "HAMMERED_R2_SOURCE__M1175_BOUND__EXACTLY_ONE_MODE_AUTHORIZED",
            "mode": mode, "contract_path": str(self.launch_path.relative_to(ROOT)),
            "common": common, "output": {"path": "hw_autoresearch_nts07/results/fixture"},
            "one_shot": {"attempt_marker": "hw_autoresearch_nts07/results/.fixture"},
            "gpu_ownership": {"lease_path": str(self.m.LEASE.relative_to(ROOT))},
        }
        if mode == "e1":
            contract["e1"] = {"fixed_modes": ["dyadic", "hardware_order"],
                              "standard_valid825": self.m.EXPECTED_STANDARD,
                              "evaluator": {"path": str(self.evaluator.relative_to(ROOT)),
                                            "sha256": sha(self.evaluator)}}
        else:
            contract["e8"] = {"canonical_cohort_manifest": {
                "path": str(self.m.EXPECTED_COHORT.relative_to(ROOT)),
                "size_bytes": self.m.EXPECTED_COHORT_SIZE,
                "sha256": self.m.EXPECTED_COHORT_SHA256,
                "inner_sha256": self.m.EXPECTED_COHORT_INNER_SHA256,
                "outer_sha256": self.m.EXPECTED_COHORT_OUTER_SHA256},
                "profile": {"path": str(self.profile.relative_to(ROOT)),
                            "sha256": sha(self.profile)},
                "expected_dynamic_samples": 40}
        self.launch_path.write_text(json.dumps(contract), encoding="utf-8")
        return contract

    def close(self):
        for name, value in self.original.items():
            setattr(self.m, name, value)
        self.temporary.cleanup()


class M1177R2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_source()

    def authority(self, mode="e1"):
        authority = FixtureAuthority(self, mode)
        self.addCleanup(authority.close)
        return authority

    def test_source_contract_is_not_launch_authority(self):
        contract = self.m.strict_json(CONTRACT)
        self.assertEqual(contract["source"]["sha256"], sha(SOURCE))
        with self.assertRaisesRegex(self.m.ClosureError, "source-only"):
            self.m.validate_launch(contract, CONTRACT)

    def test_valid_e1_fixture_passes_exact_validation(self):
        fixture = self.authority("e1")
        result = self.m.validate_launch(fixture.contract, fixture.launch_path)
        self.assertEqual(result["mode"], "e1")

    def test_valid_e8_fixture_passes_exact_validation(self):
        fixture = self.authority("e8")
        result = self.m.validate_launch(fixture.contract, fixture.launch_path)
        self.assertEqual(result["mode"], "e8")

    def test_B1_arbitrary_or_semantically_bad_m1175_rejected(self):
        fixture = self.authority("e1")
        fixture.write_m1175("NOT_M1175_ADMISSION")
        fixture.m.M1175_REVIEW_SHA256 = sha(fixture.m1175)
        fixture.contract["common"]["m1175_result_hammer"]["sha256"] = sha(fixture.m1175)
        with self.assertRaisesRegex(self.m.ClosureError, "schema/status"):
            self.m.validate_launch(fixture.contract, fixture.launch_path)

    def test_B2_profile_and_evaluator_are_code_pinned(self):
        fixture = self.authority("e1")
        fixture.profile.write_text("# mutation\n", encoding="utf-8")
        with self.assertRaisesRegex(self.m.ClosureError, "profiler/evaluator"):
            self.m.validate_launch(fixture.contract, fixture.launch_path)

    def test_B3_noncanonical_lease_rejected(self):
        fixture = self.authority("e1")
        fixture.contract["gpu_ownership"]["lease_path"] = (
            "hw_autoresearch_nts07/results/private_bypass.lock")
        with self.assertRaisesRegex(self.m.ClosureError, "canonical shared GPU lease"):
            self.m.validate_launch(fixture.contract, fixture.launch_path)

    def test_B4_mode_mix_and_all_extra_keys_rejected(self):
        fixture = self.authority("e1")
        fixture.contract["e8"] = {}
        with self.assertRaisesRegex(self.m.ClosureError, "exact-key"):
            self.m.validate_launch(fixture.contract, fixture.launch_path)
        fixture = self.authority("e1")
        fixture.contract["common"]["cohort"] = []
        with self.assertRaisesRegex(self.m.ClosureError, "exact-key"):
            self.m.validate_launch(fixture.contract, fixture.launch_path)
        fixture = self.authority("e8")
        fixture.contract["e8"]["unknown"] = True
        with self.assertRaisesRegex(self.m.ClosureError, "exact-key"):
            self.m.validate_launch(fixture.contract, fixture.launch_path)

    def test_B5_canonical_cohort_is_exact_sealed_unique_and_ordered(self):
        rows = self.m.load_canonical_cohort(verify_source_bytes=False)
        self.assertEqual(len(rows), 40)
        self.assertEqual([row["global_sample_id"] for row in rows], list(range(40)))
        self.assertEqual(len({row["sha256"] for row in rows}), 40)

    def test_B5_duplicate_cohort_mutation_rejected_even_if_resealed(self):
        fixture = self.authority("e1")
        original = {name: getattr(self.m, name) for name in (
            "EXPECTED_COHORT", "EXPECTED_COHORT_SHA256", "EXPECTED_COHORT_SIZE",
            "EXPECTED_COHORT_INNER_SHA256", "EXPECTED_COHORT_OUTER_SHA256")}
        self.addCleanup(lambda: [setattr(self.m, key, value)
                                 for key, value in original.items()])
        payload = self.m.strict_json(self.m.EXPECTED_COHORT)
        payload["rows"][1] = deepcopy(payload["rows"][0])
        path = fixture.root / "bad_cohort.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        inner = path.with_name(path.name + ".sha256")
        inner.write_text("{}  {}\n".format(sha(path), path.name), encoding="utf-8")
        outer = path.with_name(path.name + ".sha256.seal.sha256")
        outer.write_text("{}  {}\n".format(sha(inner), inner.name), encoding="utf-8")
        self.m.EXPECTED_COHORT = path
        self.m.EXPECTED_COHORT_SHA256 = sha(path)
        self.m.EXPECTED_COHORT_SIZE = path.stat().st_size
        self.m.EXPECTED_COHORT_INNER_SHA256 = sha(inner)
        self.m.EXPECTED_COHORT_OUTER_SHA256 = sha(outer)
        with self.assertRaisesRegex(self.m.ClosureError, "order/id|duplicate"):
            self.m.load_canonical_cohort(verify_source_bytes=False)

    def test_B6_exact_model_census_and_static_export(self):
        model = FixtureModel()
        census = self.m.build_model_census(model)
        names = [row["name"] for row in census["dynamic"]]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual({row["kind"] for row in census["weights"]},
                         {"Conv2d", "ConvTranspose2d", "Linear"})
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary)
            rows = self.m.export_static(torch, model, staging, census["weights"])
            self.assertEqual(len(rows), census["counts"]["weights"])

    def test_B6_dynamic_requires_every_layer_exactly_once_per_sample(self):
        model = FixtureModel()
        census = self.m.build_model_census(model)
        capture = self.m.RangeCapture(torch)
        capture.attach(model, census["dynamic"])
        capture.begin({"global_sample_id": 0})
        with self.assertRaisesRegex(self.m.ClosureError, "every-layer-once"):
            capture.end({row["name"] for row in census["dynamic"]})
        capture.close()

    def test_B7_bn_exact_four_finite_channel_and_epsilon(self):
        model = FixtureModel()
        census = self.m.build_model_census(model)
        with tempfile.TemporaryDirectory() as temporary:
            rows = self.m.export_bn(model, Path(temporary), census["batch_norm"])
            self.assertEqual(len(rows), 1)
        broken = FixtureModel()
        broken.bn.running_var[0] = float("nan")
        census = self.m.build_model_census(broken)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(self.m.ClosureError, "channel/finite"):
                self.m.export_bn(broken, Path(temporary), census["batch_norm"])
        missing = torch.nn.Sequential(torch.nn.BatchNorm2d(3, track_running_stats=False))
        with self.assertRaisesRegex(self.m.ClosureError, "BN census exact"):
            self.m.build_model_census(missing)

    def test_B8_bad_hammer_status_and_artifact_binding_rejected(self):
        fixture = self.authority("e1")
        fixture.hammer_declared = fixture.write_hammer("NOT_A_PASS")
        fixture.contract["common"]["m1177r2_source_hammer"] = fixture.hammer_declared
        with self.assertRaisesRegex(self.m.ClosureError, "semantic status"):
            self.m.validate_launch(fixture.contract, fixture.launch_path)
        fixture = self.authority("e1")
        review = json.loads(fixture.hammer_review.read_text())
        review["artifacts"]["source"]["sha256"] = "0" * 64
        seals = seal_directory(fixture.hammer_dir, {"review.json": review})
        declared = {"path": str(fixture.hammer_review.relative_to(ROOT)),
                    "review_sha256": sha(fixture.hammer_review), **seals}
        fixture.contract["common"]["m1177r2_source_hammer"] = declared
        with self.assertRaisesRegex(self.m.ClosureError, "artifact binding"):
            self.m.validate_launch(fixture.contract, fixture.launch_path)

    def test_exact_key_helper_rejects_non_dict_and_extra(self):
        with self.assertRaisesRegex(self.m.ClosureError, "exact-key"):
            self.m.exact_keys({"a": 1, "b": 2}, {"a"}, "fixture")
        with self.assertRaisesRegex(self.m.ClosureError, "exact-key"):
            self.m.exact_keys([], set(), "fixture")

    def test_dyadic_quantization_and_width_helpers_retained(self):
        value = np.asarray([[1.0, -0.5], [0.0, 2.0]], dtype=np.float32)
        result = self.m.quantize_dyadic_per_output(value, 0)
        self.assertEqual(result["preclip_violations"], 0)
        self.assertNotIn(-128, result["code"])
        self.assertEqual(self.m.signed_bits_for_bounds(-262144, 262143), 19)

    def test_strict_json_bool_and_nonfinite_attacks(self):
        with self.assertRaisesRegex(self.m.ClosureError, "exact integer"):
            self.m.exact_int(True, "fixture")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            path.write_text('{"x":1,"x":2}', encoding="utf-8")
            with self.assertRaisesRegex(self.m.ClosureError, "duplicate"):
                self.m.strict_json(path)
            path.write_text('{"x":NaN}', encoding="utf-8")
            with self.assertRaisesRegex(self.m.ClosureError, "non-standard"):
                self.m.strict_json(path)

    def test_docs359_profile_evaluator_and_cohort_constants(self):
        self.assertEqual(sha(self.m.DOCS359), self.m.DOCS359_SHA256)
        self.assertEqual(sha(self.m.PROFILE), self.m.PROFILE_SHA256)
        self.assertEqual(sha(self.m.EVALUATOR), self.m.EVALUATOR_SHA256)
        self.assertEqual(sha(self.m.EXPECTED_COHORT), self.m.EXPECTED_COHORT_SHA256)


if __name__ == "__main__":
    unittest.main()
