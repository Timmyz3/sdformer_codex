#!/usr/bin/env python3
"""Fresh different-author static hammer for M1174.

This program imports the source-only entry point but never calls main(),
run_capture(), CUDA, a remote host, an EDA tool, or a production namespace.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1174_motion_checkpoint_parametric_unified_hardware.py"
)
CONTRACT = HW / (
    "contracts/m1174_motion_checkpoint_parametric_unified_capture_source_contract_r1_20260830.json"
)
TEST = HW / "tests/test_m1174_motion_checkpoint_parametric_unified_capture_source.py"
AUTHOR = HW / "reviews/m1174_motion_checkpoint_parametric_unified_capture_author_r1_20260830"
M1175 = HW / "reviews/m1175_m1171_motion_final_checkpoint_binder_result_hammer_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_flat_seal(path: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    manifest = path / "SHA256SUMS"
    outer = path / "SHA256SUMS.seal.sha256"
    if not manifest.is_file() or not outer.is_file():
        return False, ["missing manifest or outer seal"]
    if outer.read_text(encoding="utf-8").split() != [sha256(manifest), "SHA256SUMS"]:
        errors.append("outer seal mismatch")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        if len(fields) != 2:
            errors.append("malformed manifest row")
            continue
        member = path / fields[1].lstrip("*")
        if not member.is_file() or sha256(member) != fields[0]:
            errors.append("payload mismatch: " + fields[1])
    return not errors, errors


def run_author_tests() -> dict[str, object]:
    module = load_module("m1174_author_tests_for_hammer", TEST)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "successful": result.wasSuccessful(),
        "log": stream.getvalue(),
    }


class Handle:
    def remove(self) -> None:
        pass


def hookable(class_name: str):
    def register_forward_hook(self, hook):
        self.hook = hook
        return Handle()
    return type(class_name, (), {"register_forward_hook": register_forward_hook})()


def attack_duplicate_arbitrary_cohort(module) -> bool:
    """True means the invalid forty-row cohort was accepted (a vulnerability)."""
    with tempfile.TemporaryDirectory(prefix="m1176_cohort_") as name:
        payload = Path(name) / "one.npy"
        payload.write_bytes(b"same source used forty times")
        digest = sha256(payload)
        rows = []
        sequences = ["not_the_frozen_c1_sequence"] * 10
        for seq in module.SEQUENCES:
            sequences.extend([seq] * 10)
        for index in range(40):
            rows.append({
                "global_sample_id": index,
                "cohort": "c1" if index < 10 else "arbitrary_decoder_label",
                "sequence": sequences[index],
                "sample_key": "arbitrary_{:02d}.npy".format(index),
                "path": "ignored/duplicate.npy",
                "bytes": payload.stat().st_size,
                "sha256": digest,
            })
        old_repo_path, old_regular, old_sha256 = module.repo_path, module.regular, module.sha256
        try:
            module.repo_path = lambda *_args, **_kwargs: payload
            module.regular = lambda *_args, **_kwargs: None
            module.sha256 = lambda path: sha256(Path(path))
            selected = module.selected_samples({"cohort": {"samples": rows}})
            return len(selected) == 40
        finally:
            module.repo_path, module.regular, module.sha256 = old_repo_path, old_regular, old_sha256


def attack_missing_layers(module) -> bool:
    """True means one C1 and one decoder layer satisfy whole-category coverage."""
    named = [
        (module.C1_TARGETS[0], hookable("Conv2d")),
        ("sttmultires_unet.decoders.0.0", hookable("ConvTranspose2d")),
        ("x.atlif", hookable("ATLIFTernaryPSN")),
        ("x.fc1", hookable("Linear")),
        ("x.fc2", hookable("Linear")),
        ("x.patch_embed.proj", hookable("Conv3d")),
        ("x.bn1", hookable("BatchNorm2d")),
        ("x.attn.q", hookable("Linear")),
        ("x.attn", hookable("ShiftmaxAttention")),
    ]
    model = type("Model", (), {"named_modules": lambda self: iter(named)})()
    writer = module.UnifiedHookWriter(object(), Path("/nonproduction"), {})
    try:
        writer.attach(model)
    except module.CaptureError:
        return False
    finally:
        writer.close()
    return (len(writer.module_inventory["c1_conv3x3"]) == 1 and
            len(writer.module_inventory["decoder_convtranspose"]) == 1)


def attack_nested_seal_self_incompatibility(module) -> bool:
    """True means M1174's writer emits a seal its verifier rejects."""
    with tempfile.TemporaryDirectory(prefix="m1176_nested_seal_") as name:
        root = Path(name)
        (root / "payloads").mkdir()
        (root / "payloads/item.bin").write_bytes(b"payload")
        (root / "manifest.json").write_text("{}\n", encoding="utf-8")
        module.write_double_seal(root)
        try:
            module.verify_double_seal(root)
        except module.CaptureError:
            return True
        return False


def main() -> int:
    module = load_module("m1174_source_for_independent_hammer", SOURCE)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    author_receipt = json.loads((AUTHOR / "author_receipt.json").read_text(encoding="utf-8"))
    author_seal_ok, author_seal_errors = verify_flat_seal(AUTHOR)
    m1175_seal_ok, m1175_seal_errors = verify_flat_seal(M1175)
    m1175_review = json.loads((M1175 / "review.json").read_text(encoding="utf-8"))
    validate_text = ast.get_source_segment(
        SOURCE.read_text(encoding="utf-8"),
        next(node for node in ast.parse(SOURCE.read_text(encoding="utf-8")).body
             if isinstance(node, ast.FunctionDef) and node.name == "validate_launch_contract"),
    ) or ""
    run_text = ast.get_source_segment(
        SOURCE.read_text(encoding="utf-8"),
        next(node for node in ast.parse(SOURCE.read_text(encoding="utf-8")).body
             if isinstance(node, ast.FunctionDef) and node.name == "run_capture"),
    ) or ""
    main_text = ast.get_source_segment(
        SOURCE.read_text(encoding="utf-8"),
        next(node for node in ast.parse(SOURCE.read_text(encoding="utf-8")).body
             if isinstance(node, ast.FunctionDef) and node.name == "main"),
    ) or ""

    author_tests = run_author_tests()
    findings = {
        "F0_author_controlled_tests_pass": bool(author_tests["successful"]),
        "F1_author_double_seal_valid": author_seal_ok,
        "F2_author_receipt_binds_current_contract": (
            author_receipt["authored_files"]["contract"]["sha256"] == sha256(CONTRACT)
        ),
        "F3_current_contract_binds_current_source": contract["source"]["sha256"] == sha256(SOURCE),
        "F4_m1175_independent_result_hammer_is_valid_and_passed": (
            m1175_seal_ok and
            m1175_review.get("schema") == "m1175_m1171_motion_final_checkpoint_binder_result_hammer_r1_v1" and
            m1175_review.get("status") == "PASS" and
            sha256(M1175 / "SHA256SUMS") ==
            "2a4481491d3d12bcba17263260a87e6511e523b4b410e18f3c7fecada07ab247"
        ),
        "F5_validator_requires_exact_m1175_schema_status_and_outer": (
            "m1175_m1171_motion_final_checkpoint_binder_result_hammer_r1_v1" in validate_text and
            "2a4481491d3d12bcba17263260a87e6511e523b4b410e18f3c7fecada07ab247" in validate_text
        ),
        "F6_validator_requires_fresh_m1174_source_hammer": "m1174_source_hammer" in validate_text,
        "F7_common_lease_path_is_not_contract_redirectable": (
            "exclusive_gpu_lease(LEASE)" in main_text and
            'contract["gpu_ownership"]["lease_path"]' not in main_text
        ),
        "F8_duplicate_or_arbitrary_fixed_cohort_rejected": not attack_duplicate_arbitrary_cohort(module),
        "F9_missing_c1_or_decoder_layers_rejected": not attack_missing_layers(module),
        "F10_attention_record_population_checked_before_publish": (
            "bit_writer.records" in run_text and "require" in run_text[run_text.find("bit_writer.records") - 100:
                                                               run_text.find("bit_writer.records") + 200]
        ),
        "F11_nested_payload_seal_self_verifies": not attack_nested_seal_self_incompatibility(module),
        "F12_docs359_unchanged": sha256(DOCS359) == DOCS359_SHA,
        "F13_one_build_model_callsite": SOURCE.read_text(encoding="utf-8").count("profile.build_model(") == 1,
        "F14_source_contract_cannot_launch": contract.get("production_authorization", {}).get("authorized_by_this_contract") is False,
    }
    attacks = {
        "arbitrary_duplicate_forty_source_cohort_accepted": not findings["F8_duplicate_or_arbitrary_fixed_cohort_rejected"],
        "one_of_four_c1_and_one_of_four_decoder_layers_accepted": not findings["F9_missing_c1_or_decoder_layers_rejected"],
        "nested_payload_double_seal_rejected_by_own_verifier": not findings["F11_nested_payload_seal_self_verifies"],
        "launch_contract_can_redirect_gpu_lock_path": not findings["F7_common_lease_path_is_not_contract_redirectable"],
        "m1175_admission_not_semantically_pinned": not findings["F5_validator_requires_exact_m1175_schema_status_and_outer"],
        "m1174_source_hammer_not_consumed": not findings["F6_validator_requires_fresh_m1174_source_hammer"],
        "zero_or_partial_attention_population_not_rejected": not findings["F10_attention_record_population_checked_before_publish"],
    }
    mandatory = [
        "F1_author_double_seal_valid", "F2_author_receipt_binds_current_contract",
        "F4_m1175_independent_result_hammer_is_valid_and_passed",
        "F5_validator_requires_exact_m1175_schema_status_and_outer",
        "F6_validator_requires_fresh_m1174_source_hammer",
        "F7_common_lease_path_is_not_contract_redirectable",
        "F8_duplicate_or_arbitrary_fixed_cohort_rejected",
        "F9_missing_c1_or_decoder_layers_rejected",
        "F10_attention_record_population_checked_before_publish",
        "F11_nested_payload_seal_self_verifies", "F12_docs359_unchanged",
    ]
    passed = all(findings[key] for key in mandatory)
    output = {
        "schema": "m1176_m1174_motion_unified_capture_source_hammer_output_r1_v1",
        "status": "PASS" if passed else "FAIL_CLOSED__M1174_R2_REQUIRED__NO_RELEASE_NO_GPU",
        "source_sha256": sha256(SOURCE),
        "contract_sha256": sha256(CONTRACT),
        "test_sha256": sha256(TEST),
        "author_receipt_sha256": sha256(AUTHOR / "author_receipt.json"),
        "author_manifest_sha256": sha256(AUTHOR / "SHA256SUMS"),
        "author_outer_file_sha256": sha256(AUTHOR / "SHA256SUMS.seal.sha256"),
        "m1175_manifest_sha256": sha256(M1175 / "SHA256SUMS"),
        "docs359_sha256": sha256(DOCS359),
        "author_seal_errors": author_seal_errors,
        "m1175_seal_errors": m1175_seal_errors,
        "author_tests": {
            key: value for key, value in author_tests.items() if key != "log"
        },
        "author_test_log_sha256": hashlib.sha256(
            str(author_tests["log"]).encode("utf-8")
        ).hexdigest(),
        "findings": findings,
        "mutation_attacks": attacks,
        "mandatory_checks": mandatory,
        "production_actions": {
            "remote": False, "gpu": False, "eda": False,
            "capture": False, "production_namespace": False,
        },
        "authorization": {
            "m1174_r1_release": False,
            "successor_launch_contract": False,
            "gpu_run": False,
            "required_next": "author M1174 r2, reseal, then fresh different-author hammer",
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
