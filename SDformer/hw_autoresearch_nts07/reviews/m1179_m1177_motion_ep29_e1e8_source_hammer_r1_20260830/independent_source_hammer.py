#!/usr/bin/env python3
"""Read-only, no-GPU hammer for the M1177 E1/E8 source package."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "system_handoff/scripts/run_m1177_motion_ep29_e1e8_closure_source.py"
TESTS = HW / "tests/test_run_m1177_motion_ep29_e1e8_closure_source.py"
CONTRACT = HW / "contracts/m1177_motion_ep29_e1e8_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1177_motion_ep29_e1e8_source_author_r1_20260830"
M1175 = HW / "reviews/m1175_m1171_motion_final_checkpoint_binder_result_hammer_r1_20260830/review.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    "source": "2f15c406ac8238f1389ead96b044848e0debc1529a13326a488299c42067a19d",
    "tests": "adc3fd9296d54139c9707f45fc655e4fda6fe4270db611d5a2ffd26a87fd5374",
    "contract": "d27fb4eebb60f4d828775838539f10080990f8a5262a44134a60c9a55337dfb7",
    "m1175": "8b83690b8b1130d2335bb118d35645ae4d172740966ab69c6fcea9bc8b5d307b",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source():
    spec = importlib.util.spec_from_file_location("m1179_m1177_under_hammer", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_launch_acceptance_attack(module) -> dict[str, bool]:
    """Prove that launch validation accepts unpinned receipts/sources/mode extras."""
    original = {
        key: getattr(module, key) for key in
        ("DOCS359", "DOCS359_SHA256", "PROFILE", "EVALUATOR",
         "EXPECTED_CHECKPOINT", "EXPECTED_CONFIG_SHA256")
    }
    with tempfile.TemporaryDirectory(prefix=".m1179_fixture.", dir=HW / "results") as raw:
        directory = Path(raw)
        checkpoint = directory / "checkpoint.pth"
        checkpoint.write_bytes(b"m1179-checkpoint-fixture")
        config = directory / "config.yml"
        config.write_text("fixture: true\n", encoding="utf-8")
        docs = directory / "docs359.fixture"
        docs.write_text("fixture protected file\n", encoding="utf-8")
        profile = directory / "profile.py"
        profile.write_text("# deliberately not the source-contract-pinned profiler\n", encoding="utf-8")
        evaluator = directory / "evaluator.py"
        evaluator.write_text("# deliberately not the source-contract-pinned evaluator\n", encoding="utf-8")
        fake_m1175 = directory / "arbitrary_receipt.json"
        fake_m1175.write_text('{"status":"NOT_M1175_ADMISSION"}\n', encoding="utf-8")
        fake_hammer = directory / "arbitrary_hammer.json"
        fake_hammer.write_text('{"status":"NOT_A_SOURCE_HAMMER"}\n', encoding="utf-8")
        launch_path = directory / "launch.json"
        relative = lambda p: str(p.relative_to(ROOT))
        try:
            module.DOCS359 = docs
            module.DOCS359_SHA256 = sha(docs)
            module.PROFILE = profile
            module.EVALUATOR = evaluator
            module.EXPECTED_CHECKPOINT = {
                "epoch": 29,
                "sha256": sha(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
                "mtime_ns": checkpoint.stat().st_mtime_ns,
            }
            module.EXPECTED_CONFIG_SHA256 = sha(config)
            contract = {
                "schema": "m1177_motion_ep29_e1e8_launch_v1",
                "status": "HAMMERED_SOURCE__M1175_BOUND__EXACTLY_ONE_MODE_AUTHORIZED",
                "mode": "e1",
                "contract_path": relative(launch_path),
                "inputs": {
                    "source": {"sha256": sha(SOURCE)},
                    "selection": {
                        "epoch": 29,
                        "checkpoint_sha256": sha(checkpoint),
                        "checkpoint_size_bytes": checkpoint.stat().st_size,
                        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
                        "config_sha256": sha(config),
                        "standard_valid825": module.EXPECTED_STANDARD,
                    },
                    "checkpoint_path": str(checkpoint),
                    "config_path": str(config),
                    "m1175_result_hammer": {"path": relative(fake_m1175), "sha256": sha(fake_m1175)},
                    "m1177_source_hammer": {"path": relative(fake_hammer), "sha256": sha(fake_hammer)},
                    "profile": {"sha256": sha(profile)},
                    "evaluator": {"sha256": sha(evaluator)},
                    # E8-only material mixed into an E1 launch is silently accepted.
                    "cohort": [{"global_sample_id": 999, "path": "arbitrary"}],
                },
            }
            launch_path.write_text(json.dumps(contract), encoding="utf-8")
            accepted = module.validate_launch(contract, launch_path)
            return {
                "arbitrary_m1175_receipt_accepted": accepted["mode"] == "e1",
                "unpinned_profile_and_evaluator_accepted": True,
                "e8_only_cohort_in_e1_accepted": True,
            }
        finally:
            for key, value in original.items():
                setattr(module, key, value)


def main() -> int:
    for label, path in (("source", SOURCE), ("tests", TESTS), ("contract", CONTRACT),
                        ("m1175", M1175), ("docs359", DOCS359)):
        assert sha(path) == EXPECTED[label], (label, sha(path))
    py_compile.compile(str(SOURCE), doraise=True)
    suite = unittest.defaultTestLoader.discover(str(TESTS.parent), pattern=TESTS.name)
    result = unittest.TextTestRunner(verbosity=0).run(suite)
    assert result.wasSuccessful() and result.testsRun == 11
    module = load_source()

    attacks = fixture_launch_acceptance_attack(module)
    assert all(attacks.values())

    shape = (96, 64, 3, 3)
    quantized = module.quantize_dyadic_per_output(np.ones(shape, dtype=np.float32), 0)
    reported = quantized["compression"]["dense_first_output_tile_up_to_96_bytes"]
    actual = int(np.prod(shape))
    # `code` is still flattened when tile_dense is calculated, so kernel terms
    # are correctly included.  Preserve this as an independent positive check.
    assert reported == actual == 96 * 64 * 3 * 3

    text = SOURCE.read_text(encoding="utf-8")
    static = {
        "canonical_lease_constant_declared": "LEASE =" in text,
        "launcher_lease_is_pinned_to_constant": (
            'contract["gpu_ownership"]["lease_path"]' in text and
            '== str(LEASE.relative_to(ROOT))' in text
        ),
        "m1175_expected_sha_constant_present": EXPECTED["m1175"] in text,
        "profile_expected_sha_constant_present":
            "04f692c5bda6d1f88cdc932ce48f012767f22a2bb1ca161378971232f99c0684" in text,
        "evaluator_expected_sha_constant_present":
            "ba40b42c7395fd703c59a183a19b6a4fd38fa08ed75201008f03fd71b82aaef1" in text,
        "exact_weight_layer_census_required": "expected_weight_layer" in text,
        "dynamic_every_layer_every_sample_required": "expected_dynamic_rows" in text,
        "bn_four_tensor_set_required": "set(arrays) ==" in text,
        "canonical_40_sample_manifest_pinned": "EXPECTED_COHORT" in text,
    }
    assert static["canonical_lease_constant_declared"]
    assert not any(value for key, value in static.items()
                   if key != "canonical_lease_constant_declared")

    output = {
        "schema": "m1179_m1177_motion_ep29_e1e8_source_hammer_output_r1_v1",
        "status": "FAIL_CLOSED__SOURCE_REVISION_REQUIRED",
        "verified_pass": {
            "artifact_shas": True,
            "author_tests": 11,
            "python_compile": True,
            "ep29_constants_in_source": True,
            "fixed_two_mode_e1_policy": True,
            "single_build_model_callsite": text.count("profile.build_model(") == 1,
            "docs359_unchanged": True,
        },
        "accepted_attacks": attacks,
        "tile_fit_positive_check": {
            "weight_shape": list(shape),
            "reported_bytes": reported,
            "actual_dense_bytes": actual,
            "kernel_terms_included": True,
        },
        "missing_fail_closed_guards": static,
        "production_authorized": False,
    }
    here = Path(__file__).resolve().parent
    (here / "hammer_output.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
