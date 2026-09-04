#!/usr/bin/env python3
"""Receipt-blind synthetic/static hammer for M1284.

No author receipt, live prefix, canonical result, real preflight/production,
EDA, GPU, or remote resource is opened.  All filesystem mutations are confined
to temporary synthetic fixtures.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "system_simulator/scripts/build_m1284_decoder_completion_gate_diagnostic_annex_successor.py"
TEST = ROOT / "tests/test_m1284_decoder_completion_gate_diagnostic_annex_successor.py"
CONTRACT = ROOT / "contracts/m1284_decoder_completion_gate_diagnostic_annex_successor_source_contract_r1_20260830.json"
PREDECESSOR = ROOT / "system_simulator/scripts/build_m1278_decoder_completion_gate_and_diagnostic_annex.py"
PREDECESSOR_CONTRACT = ROOT / "contracts/m1278_decoder_completion_gate_diagnostic_annex_source_contract_r1_20260830.json"

EXPECTED_SHA = {
    SOURCE: "a0b5747b63f857cda594765fb7ed1d4837295327af477f73ef27f5a36635eb02",
    TEST: "3a42db04b51907a8df7c433c76ac0b5980efb8737866331d6f2bda7d16aafd6f",
    CONTRACT: "db774e9851343b1f79e9272199e933fe07a0fb837a6ccc1629e7e32add074008",
    PREDECESSOR: "52c0829927fb32211df86e0781049f202b2ed63297b3743f121267a6bfa5471d",
    PREDECESSOR_CONTRACT: "6987400c9adc638905675f1b1c3794095ec0ed2d63b887efadefa35ab105edfb",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def expect_error(action, fragment=None):
    try:
        action()
    except BaseException as exc:
        if fragment is not None:
            require(fragment in str(exc), "wrong rejection: " + str(exc))
        return type(exc).__name__ + ": " + str(exc)
    raise AssertionError("attack unexpectedly accepted")


def load_target():
    spec = importlib.util.spec_from_file_location("m1289_blind_m1284", SOURCE)
    require(spec is not None and spec.loader is not None, "cannot load target")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def layout_for(module, parent: Path, annex_name="annex"):
    predecessor = module.P
    return predecessor.Layout(parent, parent / predecessor.RESULT_NAME,
        parent / predecessor.ATTEMPT_NAME, parent / predecessor.LOCK_NAME,
        parent / predecessor.WORK_NAME, parent / annex_name)


def build_attempt(module, layout, runner, maximum=1, seal=True):
    layout.attempt.mkdir()
    payload = {
        "schema": "m1111dr2_decoder_production_attempt_v2",
        "status": "CONSUMED_BEFORE_CANONICAL_PAYLOAD_ACCESS",
        "maximum_attempts": maximum,
        "automatic_retry": False,
        "canonical_payload_opened_before_attempt": False,
        "runner_sha256": module.P.RUNNER_SHA256,
        "contract_sha256": runner.CONTRACT_ID[0],
    }
    (layout.attempt / "attempt.json").write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    if seal:
        runner.atomic_seal(layout.attempt)


def build_result(module, layout, runner):
    runner.build_publish_self_test_candidate(layout.result)
    runner.atomic_seal(layout.result)


def run():
    for path, expected in EXPECTED_SHA.items():
        require(sha256(path) == expected, "source identity drift: " + str(path))

    module = load_target()
    predecessor = module.P
    runner = predecessor.load_runner()
    findings = {}

    # Exact scalar types: attempt bool, row bool and result-claim integer zero.
    with tempfile.TemporaryDirectory(prefix="m1289.bool.") as name:
        layout = layout_for(module, Path(name))
        build_attempt(module, layout, runner, maximum=True); build_result(module, layout, runner)
        findings["bool_attempt"] = "REJECTED__PASS: " + expect_error(
            lambda: module.completion_capability(layout, runner, alive=lambda _: False),
            "exact integer")
    with tempfile.TemporaryDirectory(prefix="m1289.rowbool.") as name:
        layout = layout_for(module, Path(name))
        build_attempt(module, layout, runner); build_result(module, layout, runner)
        cap = module.completion_capability(layout, runner, alive=lambda _: False)
        cap.gate["rows"][0]["transaction_count"] = True
        findings["bool_row_counter"] = "REJECTED__PASS: " + expect_error(
            lambda: module.publish_with_capability(layout, runner, cap), "exact integer")
    with tempfile.TemporaryDirectory(prefix="m1289.claimbool.") as name:
        layout = layout_for(module, Path(name))
        build_attempt(module, layout, runner); build_result(module, layout, runner)
        gate = predecessor.completion_gate(layout, runner, alive=lambda _: False)
        gate["checked"]["payload"]["claim_boundary"]["speedup_admitted"] = 0
        findings["result_claim_integer_zero"] = "REJECTED__PASS: " + expect_error(
            lambda: module.validate_complete_gate(layout, runner, gate), "exact boolean")

    # Fake plain object is rejected, but a holder can mint a registered object
    # because Capability type and its private key are exposed on the instance.
    with tempfile.TemporaryDirectory(prefix="m1289.cap.") as name:
        layout = layout_for(module, Path(name))
        build_attempt(module, layout, runner); build_result(module, layout, runner)
        cap = module.completion_capability(layout, runner, alive=lambda _: False)
        findings["plain_fake_capability"] = "REJECTED__PASS: " + expect_error(
            lambda: module.publish_with_capability(layout, runner, object()), "capability")
        forged = type(cap)(cap.gate, cap.key)
        forged_gate = module._consume_capability(forged)
        require(forged_gate["state"] == "COMPLETE", "forged capability not accepted")
        findings["capability_mint_from_exposed_type_and_key"] = "ACCEPTED__CRITICAL"

    # One capability is single-consumption, but the issuer can be called twice
    # for one result.  The two capabilities can publish identical evidence into
    # two different annex paths because capability is not bound to annex/layout.
    with tempfile.TemporaryDirectory(prefix="m1289.repeat.") as name:
        parent = Path(name)
        first = layout_for(module, parent, "annex_a")
        second = layout_for(module, parent, "annex_b")
        build_attempt(module, first, runner); build_result(module, first, runner)
        cap1 = module.completion_capability(first, runner, alive=lambda _: False)
        cap2 = module.completion_capability(first, runner, alive=lambda _: False)
        out1 = module.publish_with_capability(first, runner, cap1)
        out2 = module.publish_with_capability(second, runner, cap2)
        require(first.annex.is_dir() and second.annex.is_dir() and
                out1["status"] == out2["status"], "duplicate publication failed")
        findings["repeat_publish_two_capabilities_two_paths"] = "ACCEPTED__HIGH"
        findings["same_capability_second_use"] = "REJECTED__PASS: " + expect_error(
            lambda: module.publish_with_capability(first, runner, cap1), "unused")

    # Mutation after capability issue but before publish is caught by the second
    # canonical completion validation.
    with tempfile.TemporaryDirectory(prefix="m1289.prepub.") as name:
        layout = layout_for(module, Path(name))
        build_attempt(module, layout, runner); build_result(module, layout, runner)
        cap = module.completion_capability(layout, runner, alive=lambda _: False)
        with (layout.result / predecessor.PAYLOAD).open("a", encoding="utf-8") as stream:
            stream.write("\n")
        findings["replacement_before_publish_entry"] = "REJECTED__PASS: " + expect_error(
            lambda: module.publish_with_capability(layout, runner, cap))

    # There remains a last-mile TOCTOU: mutate the result after validate_annex
    # returns but inside the delegated publisher.  The predecessor publisher
    # never reopens/revalidates canonical completion, so it publishes stale
    # source identity.
    with tempfile.TemporaryDirectory(prefix="m1289.toctou.") as name:
        layout = layout_for(module, Path(name))
        build_attempt(module, layout, runner); build_result(module, layout, runner)
        cap = module.completion_capability(layout, runner, alive=lambda _: False)
        original_publish = predecessor.publish_annex
        def late_replace(target_layout, payload):
            with (target_layout.result / predecessor.PAYLOAD).open(
                    "a", encoding="utf-8") as stream:
                stream.write("\n")
            return original_publish(target_layout, payload)
        predecessor.publish_annex = late_replace
        try:
            module.publish_with_capability(layout, runner, cap)
        finally:
            predecessor.publish_annex = original_publish
        annex_payload = json.loads((layout.annex / "annex.json").read_text(encoding="utf-8"))
        require(annex_payload["source_result"]["payload_sha256"] !=
                module.sha256(layout.result / predecessor.PAYLOAD),
                "late replacement unexpectedly bound")
        findings["replacement_after_final_validation_before_atomic_publish"] = (
            "ACCEPTED__CRITICAL")

    # Final payload schema and claim promotions are fail-closed.
    with tempfile.TemporaryDirectory(prefix="m1289.final.") as name:
        layout = layout_for(module, Path(name))
        build_attempt(module, layout, runner); build_result(module, layout, runner)
        cap = module.completion_capability(layout, runner, alive=lambda _: False)
        gate = copy.deepcopy(cap.gate)
        payload = module.build_annex(layout, runner, gate)
        attacks = {}
        value = copy.deepcopy(payload); value["schema"] = "table_a"
        attacks["schema"] = expect_error(
            lambda: module.validate_annex(layout, runner, gate["checked"], value),
            "schema/status")
        value = copy.deepcopy(payload); value["identity"]["checkpoint"] = "final"
        attacks["final_checkpoint"] = expect_error(
            lambda: module.validate_annex(layout, runner, gate["checked"], value),
            "ep35 identity")
        for key in ("table_a", "full_network", "system_speedup", "paper_headline"):
            value = copy.deepcopy(payload); value["claim_boundary"][key] = True
            attacks[key] = expect_error(
                lambda value=value: module.validate_annex(
                    layout, runner, gate["checked"], value), "annex claim")
        value = copy.deepcopy(payload); value["claim_boundary"]["table_a"] = 0
        attacks["claim_false_as_integer_zero"] = expect_error(
            lambda: module.validate_annex(layout, runner, gate["checked"], value),
            "exact boolean")
        value = copy.deepcopy(payload); value["extra"] = "PASS_TABLE_A"
        attacks["extra_top_level_key"] = expect_error(
            lambda: module.validate_annex(layout, runner, gate["checked"], value),
            "top-level schema")
        findings["final_schema_identity_and_claim_promotions"] = {
            "status": "ALL_REJECTED__PASS", "attacks": sorted(attacks)}

    # Static-authority check validates contract schema/source binding but not the
    # exact closed claim boundary.  A promoted temporary contract is accepted.
    with tempfile.TemporaryDirectory(prefix="m1289.contract.") as name:
        promoted_path = Path(name) / "promoted_contract.json"
        promoted = json.loads(CONTRACT.read_text(encoding="utf-8"))
        promoted["claim_boundary"]["table_a"] = True
        promoted["claim_boundary"]["system_speedup"] = True
        promoted_path.write_text(json.dumps(promoted, sort_keys=True) + "\n",
                                 encoding="utf-8")
        old_contract = module.CONTRACT
        module.CONTRACT = promoted_path
        try:
            module.verify_static_authorities()
        finally:
            module.CONTRACT = old_contract
        findings["promoted_source_contract_claim_boundary"] = "ACCEPTED__HIGH"

    # Missing authority/state components all fail closed.  These are synthetic
    # directories only; no canonical PID/result is inspected.
    missing = {}
    with tempfile.TemporaryDirectory(prefix="m1289.missing_attempt.") as name:
        layout = layout_for(module, Path(name))
        missing["attempt"] = expect_error(lambda: module.completion_capability(
            layout, runner, alive=lambda _: False))
    with tempfile.TemporaryDirectory(prefix="m1289.missing_seal.") as name:
        layout = layout_for(module, Path(name)); build_attempt(
            module, layout, runner, seal=False); build_result(module, layout, runner)
        missing["attempt_seal"] = expect_error(lambda: module.completion_capability(
            layout, runner, alive=lambda _: False))
    with tempfile.TemporaryDirectory(prefix="m1289.missing_result.") as name:
        layout = layout_for(module, Path(name)); build_attempt(module, layout, runner)
        missing["result"] = expect_error(lambda: module.completion_capability(
            layout, runner, alive=lambda _: False))
    with tempfile.TemporaryDirectory(prefix="m1289.left_lock.") as name:
        layout = layout_for(module, Path(name)); build_attempt(module, layout, runner)
        build_result(module, layout, runner); layout.lock.mkdir()
        missing["producer_absent_lock_present"] = expect_error(
            lambda: module.completion_capability(layout, runner, alive=lambda _: False))
    with tempfile.TemporaryDirectory(prefix="m1289.live_missing_lock.") as name:
        layout = layout_for(module, Path(name)); build_attempt(module, layout, runner)
        missing["pid_alive_missing_lock_work"] = expect_error(
            lambda: module.completion_capability(layout, runner, alive=lambda _: True,
                cmdline=lambda _: predecessor.EXPECTED_CMDLINE))
    with tempfile.TemporaryDirectory(prefix="m1289.pid.") as name:
        layout = layout_for(module, Path(name)); build_attempt(module, layout, runner)
        layout.work.mkdir(); layout.lock.mkdir()
        (layout.lock / "owner.json").write_text(json.dumps({
            "pid": predecessor.PRODUCER_PID, "maximum_attempts": 1,
            "automatic_retry": False}, sort_keys=True) + "\n", encoding="utf-8")
        missing["pid_cmdline_identity"] = expect_error(
            lambda: module.completion_capability(layout, runner, alive=lambda _: True,
                cmdline=lambda _: b"not-the-producer\0"))
    findings["missing_seal_pid_lock_result"] = {
        "status": "ALL_REJECTED__PASS", "cases": sorted(missing)}

    # Output writer is inherited from M1278 and therefore writes the M1278
    # completion token/seal namespace into an M1284 payload directory.
    with tempfile.TemporaryDirectory(prefix="m1289.token.") as name:
        layout = layout_for(module, Path(name))
        build_attempt(module, layout, runner); build_result(module, layout, runner)
        cap = module.completion_capability(layout, runner, alive=lambda _: False)
        module.publish_with_capability(layout, runner, cap)
        token = (layout.annex / "RUN_COMPLETE.txt").read_text(encoding="utf-8")
        require(token == predecessor.ANNEX_COMPLETE and "M1278" in token,
                "inherited completion token drift")
        findings["m1284_payload_with_inherited_m1278_completion_token"] = (
            "ACCEPTED__MEDIUM")

    return {
        "schema": "m1289_m1284_decoder_completion_receipt_blind_hammer_v1",
        "status": "CONDITIONAL_PASS_M1284_COMMON_PATH__STOP_PUBLICATION_UNTIL_CAPABILITY_TOCTOU_REPAIR",
        "score": 73,
        "receipt_blind": True,
        "live_prefix_opened": False,
        "real_preflight_or_production": False,
        "eda_gpu_remote": False,
        "source_identities": {str(path.relative_to(ROOT)): digest
                              for path, digest in EXPECTED_SHA.items()},
        "findings": findings,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))
