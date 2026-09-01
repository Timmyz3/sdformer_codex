#!/usr/bin/env python3
"""Independent local-only rehammer of the M1564 permit-gate successor.

No checkpoint, GPU, SSH, capture, release, RTL, or EDA path is executed.
Compatible with CPython 3.6.
"""
from __future__ import print_function

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HW = HERE.parents[1]
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1558_motion_ep34_s2_tsbg_reduced_binary_source_r1.py")
TEST = HW / "tests/test_m1558_motion_ep34_s2_tsbg_reduced_binary_source.py"
CONTRACT = HW / (
    "contracts/m1564_m1558_reduced_binary_permit_gate_successor_source_"
    "contract_r1_20260901.json")
AUTHOR_REVIEW = HW / (
    "reviews/m1564_m1558_reduced_binary_permit_gate_successor_"
    "author_receipt_r1_20260901/review.json")
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

PINNED = {
    "source": "be827db3fb40650297d0c22e704b6b51e320808ad0b56df2ce6fe0255977fed6",
    "test": "b68becf1483fdc9cfb85ebc35f0deb4fa250b65e3f408314163d281747964283",
    "contract": "5788bded9893e6cbc751a77ec3b2c51ed53e21158a92c4b264f4dae16c8c23e1",
    "author_review": "e1b2e0816356ac6b04d685f0fbdc6dd8c6fe1231b8813d4953aa0345de624790",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expect_m1558_error(module, function, label):
    try:
        function()
    except module.M1558Error:
        return
    raise RuntimeError(label + " did not fail closed")


def main():
    inputs = {"source": sha256(SOURCE), "test": sha256(TEST),
              "contract": sha256(CONTRACT),
              "author_review": sha256(AUTHOR_REVIEW),
              "docs359": sha256(DOC359)}
    require(inputs == PINNED, "pinned M1564 input drift")
    module = load(SOURCE, "m1565_bound_m1564")
    test_module = load(TEST, "m1565_bound_m1564_test")

    require(not hasattr(module, "_mint_permit") and
            hasattr(module, "_checked_issue_permit") and
            hasattr(module, "_PreloadPermit"),
            "raw-mint removal identity drift")
    self_check = module.source_self_check()
    specs = module.frozen_layer_specs()
    estimate = module.estimate_from_specs(specs)
    require(len(specs) == 32 and
            sum(row["target"] == "FC1" for row in specs) == 12 and
            sum(row["target"] == "FC2" for row in specs) == 12 and
            sum(row["target"] == "PATCH" for row in specs) == 8 and
            estimate["fc_tokens"] == 44640000 and
            estimate["patch_tokens_histogram_only"] == 430080000 and
            estimate["raw_fc_payload_upper_bytes"] == 7528535874 and
            estimate["result_upper_bytes"] == 7598737368 and
            self_check["hardware_quantization_authority"] is False,
            "successor population/estimate drift")

    with tempfile.TemporaryDirectory(prefix="m1565_m1564_permit.") as directory:
        base = Path(directory)
        fake_specs = test_module.fake_specs()
        fake_samples = test_module.sample_order()
        fake_estimate = module.estimate_from_specs(fake_specs, 3)
        equal_free = (int(fake_estimate["result_upper_bytes"]) +
                      module.MIN_FREE_AFTER_BYTES)

        expect_m1558_error(module, lambda: module._checked_issue_permit(
            base / "zero", fake_specs, 3, 0), "zero-free checked issuer")
        expect_m1558_error(module, lambda: module._checked_issue_permit(
            base / "equal", fake_specs, 3, equal_free),
            "equal-free checked issuer")

        # A normal permit remains one-shot and path/inventory bound.
        path_bound = base / "path_bound"
        permit = module.issue_synthetic_permit(
            path_bound, fake_specs, 3, equal_free + 1)
        require(type(permit) is module._PreloadPermit,
                "checked issuer did not return exact permit type")
        inventory = module.canonical_sha(fake_specs)
        expect_m1558_error(module, lambda: permit.consume(
            base / "path_drift", inventory), "permit path drift")
        expect_m1558_error(module, lambda: permit.consume(
            path_bound, inventory + "x"), "permit inventory drift")
        consumed = permit.consume(path_bound, inventory)
        require(consumed["consumed"] is True and
                consumed["free_bytes_after_upper"] ==
                module.MIN_FREE_AFTER_BYTES + 1,
                "valid permit receipt drift")
        expect_m1558_error(module, lambda: permit.consume(
            path_bound, inventory), "permit reuse")

        # The class cannot be constructed with an arbitrary guessed token.
        signature = inspect.signature(module._PreloadPermit)
        require(list(signature.parameters) ==
                ["output", "inventory", "estimate", "free_bytes", "token"],
                "exact permit constructor signature drift")
        expect_m1558_error(module, lambda: module._PreloadPermit(
            base / "direct", inventory, fake_estimate, equal_free + 1,
            object()), "direct exact-permit construction")

        # P0: the production issuer still accepts a caller-controlled free
        # value. Prove the real disk query is not consulted by replacing it
        # with a sentinel that would fail if called.
        production_required = (int(estimate["result_upper_bytes"]) +
                               module.MIN_FREE_AFTER_BYTES + 1)
        original_disk_usage = module.shutil.disk_usage
        def disk_query_forbidden(_path):
            raise RuntimeError("real disk query unexpectedly consulted")
        module.shutil.disk_usage = disk_query_forbidden
        try:
            public_output = base / "public_free_override"
            public_permit = module.issue_preload_permit(
                public_output, free_bytes=production_required)
            checked_output = base / "checked_free_override"
            checked_permit = module._checked_issue_permit(
                checked_output, specs, 40, production_required)
            synthetic_output = base / "synthetic_production_identity"
            synthetic_permit = module.issue_synthetic_permit(
                synthetic_output, specs, 40, production_required)
        finally:
            module.shutil.disk_usage = original_disk_usage

        production_inventory = module.canonical_sha(specs)
        public_receipt = public_permit.consume(
            public_output, production_inventory)
        checked_receipt = checked_permit.consume(
            checked_output, production_inventory)
        synthetic_receipt = synthetic_permit.consume(
            synthetic_output, production_inventory)
        require(all(type(value) is module._PreloadPermit for value in
                    (public_permit, checked_permit, synthetic_permit)) and
                all(value["free_bytes_before"] == production_required for value in
                    (public_receipt, checked_receipt, synthetic_receipt)) and
                all(value["free_bytes_after_upper"] ==
                    module.MIN_FREE_AFTER_BYTES + 1 for value in
                    (public_receipt, checked_receipt, synthetic_receipt)),
                "caller-controlled free-space bypass proof drift")

    author_output = subprocess.check_output(
        [sys.executable, str(TEST)], stderr=subprocess.STDOUT).decode("utf-8")
    expected_author = (
        "PASS M1558 reduced-binary successor attacks=22 frames=6 "
        "fc_tokens=18 patch_rows=3 no_gpu=1 no_capture=1")
    require(expected_author in author_output, "author successor test did not pass")
    source_cli = subprocess.check_output(
        [sys.executable, str(SOURCE), "--source-self-check"],
        stderr=subprocess.STDOUT).decode("utf-8")
    require(json.loads(source_cli) == self_check,
            "source self-check CLI/module drift")

    result = {
        "schema": "m1565_m1564_reduced_binary_permit_gate_independent_rehammer_r1_v1",
        "status": "NO_GO_M1565_REMOTE_WRAPPER_AUTHORING__CALLER_CONTROLLED_FREE_SPACE_AND_SYNTHETIC_PROVENANCE_BYPASS",
        "runtime": {"executable": sys.executable,
                    "version": sys.version.split()[0]},
        "pinned_inputs": inputs,
        "passed": {
            "raw_mint_global_absent": True,
            "direct_constructor_with_arbitrary_token_rejected": True,
            "zero_free_rejected": True,
            "equal_free_rejected": True,
            "permit_path_drift_rejected": True,
            "permit_inventory_drift_rejected": True,
            "permit_reuse_rejected": True,
            "author_synthetic_regression": True,
            "source_self_check": True,
            "population_and_estimate_unchanged": True},
        "p0_finding": {
            "public_production_issuer_accepts_free_override": True,
            "checked_issuer_global_accepts_free_override": True,
            "synthetic_issuer_accepts_exact_production_inventory": True,
            "synthetic_and_production_permit_type_indistinguishable": True,
            "real_disk_query_bypassed_in_all_three_paths": True,
            "permit_gate_truly_enforced": False,
            "required_fix": (
                "production issuer must have no caller-controlled free-space "
                "parameter and must query disk itself; production permits must "
                "carry provenance distinct from synthetic permits, and "
                "production_inventory=True must reject synthetic provenance")},
        "bypass_receipt": {
            "caller_supplied_free_bytes": production_required,
            "result_upper_bytes": estimate["result_upper_bytes"],
            "reported_free_after_upper": module.MIN_FREE_AFTER_BYTES + 1,
            "real_disk_usage_consulted": False,
            "exact_permit_type_returned": True},
        "authorization": {
            "successor_permit_provenance_fix_authoring": True,
            "remote_integration_wrapper_authoring": False,
            "checkpoint_load": False, "gpu": False, "ssh": False,
            "capture": False, "release": False, "automatic_retry": False,
            "rtl": False, "eda": False},
        "release_ladder": {
            "independent_rehammer_after_fix_required": True,
            "actual_capture_requires_separate_one_shot_release": True,
            "production_result_hammer_required": True},
        "claim_boundary": {
            "local_source_and_synthetic_only": True,
            "checkpoint_loaded": False, "gpu": False, "ssh": False,
            "capture_executed": False, "release_executed": False,
            "aee": False, "cycles": False, "traffic": False,
            "energy": False, "speedup": False, "rtl": False,
            "eda": False, "paper_headline": False}}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
