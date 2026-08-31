#!/usr/bin/env python3
"""Static author checks for the M682-r5 pre-attempt identity recheck."""

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = (ROOT / "system_handoff/scripts/"
          "run_m682r5_h67_layer_static_decoder_payload_one_shot.sh")
CONTRACT = (ROOT / "contracts/"
            "m682r5_h67_ep35_layer_static_decoder_payload_contract_r1_20260828.json")
PREFLIGHT = (ROOT / "results/"
             "m682r5_h67_ep35_cpu_exact_load_preflight_r1_20260828")
OUTPUT = (ROOT / "system_handoff/outgoing/"
          "m682r5_h67_ep35_layer_static_decoder_payload_s10_r1_20260828")
ATTEMPT = (ROOT / "results/"
           ".m682r5_h67_ep35_layer_static_decoder_payload_r1_attempt_consumed")

RUNNER_SHA = "047540d002f1812ed20097a03705d67f9260d10244d37401ed9a11c7643f631b"
CONTRACT_SHA = "099f27d16892f633ff5c0847c1e5958d9ba805668942c8d4e76f6d30692606aa"
PREFLIGHT_RECEIPT_SHA = "89381b8a8ecf8b9b3b8194fd5b77815b79cd1642ac2be2fd08412fa7ca54c78d"
PREFLIGHT_OUTER_SHA = "8b1c4c817a94a3c1fe438d8bdc5c8513a7852e2dd90b12f16638e1c13cf83966"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_frozen_identities_and_execution_mode():
    assert digest(RUNNER) == RUNNER_SHA
    assert digest(CONTRACT) == CONTRACT_SHA
    assert digest(PREFLIGHT / "preflight.json") == PREFLIGHT_RECEIPT_SHA
    assert digest(PREFLIGHT / "SHA256SUMS.seal.sha256") == PREFLIGHT_OUTER_SHA
    assert digest(ROOT / "docs/359_DATE终局冻结_20260813.md") == DOCS359_SHA
    assert os.access(RUNNER, os.X_OK)


def test_external_roots_are_checked_twice_and_second_is_adjacent_to_attempt():
    source = RUNNER.read_text(encoding="utf-8")
    required = source.index(
        "M660R3_EXPECTED_PREFLIGHT_RECEIPT_SHA256:-")
    receipt_compare = source.index(
        '"${M660R3_EXPECTED_PREFLIGHT_RECEIPT_SHA256}" &&', required)
    outer_compare = source.index(
        '"${M660R3_EXPECTED_PREFLIGHT_OUTER_SEAL_SHA256}" ]]',
        receipt_compare)
    first_exit = source.index("exit 41", outer_compare)
    semantic = source.index('"${m660r2_python}" - "${m660r2_preflight}/preflight.json"',
                            first_exit)
    second_receipt = source.index(
        '"${M660R3_EXPECTED_PREFLIGHT_RECEIPT_SHA256}" &&', semantic)
    second_outer = source.index(
        '"${M660R3_EXPECTED_PREFLIGHT_OUTER_SEAL_SHA256}" ]]',
        second_receipt)
    second_exit = source.index("exit 42", second_outer)
    attempt = source.index('mkdir "${m660r2_attempt}"', second_exit)
    capture = source.index("m660r2_capture_started=1", attempt)
    between = source[second_exit:attempt]
    assert required < receipt_compare < outer_compare < first_exit
    assert first_exit < semantic < second_receipt < second_outer < second_exit < attempt < capture
    assert "sha256sum" not in between and "python" not in between


def test_contract_and_runner_bind_each_other_and_r3_paths():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    runner = contract["inputs"]["runner"]
    assert runner == {
        "path": "hw_autoresearch_nts07/system_handoff/scripts/run_m682r5_h67_layer_static_decoder_payload_one_shot.sh",
        "sha256": RUNNER_SHA,
    }
    assert contract["cpu_exact_load_preflight"]["canonical_directory"] == (
        "hw_autoresearch_nts07/" + str(PREFLIGHT.relative_to(ROOT)))
    assert contract["one_shot"]["attempt_directory"] == (
        "hw_autoresearch_nts07/" + str(ATTEMPT.relative_to(ROOT)))
    assert contract["output"]["canonical_directory"] == (
        "hw_autoresearch_nts07/" + str(OUTPUT.relative_to(ROOT)))
    command = contract["only_candidate_command_after_fresh_hammer_explicit_go"]
    assert "M660R3_EXPECTED_PREFLIGHT_RECEIPT_SHA256" in command
    assert "M660R3_EXPECTED_PREFLIGHT_OUTER_SEAL_SHA256" in command
    assert "run_m682r5_h67_layer_static_decoder_payload_one_shot.sh" in command


def test_preflight_is_current_contract_and_canonical_is_unconsumed():
    receipt = json.loads((PREFLIGHT / "preflight.json").read_text(
        encoding="utf-8"))
    assert receipt["contract"]["sha256"] == CONTRACT_SHA
    assert receipt["checkpoint_load_audit"]["missing_count"] == 0
    assert receipt["checkpoint_load_audit"]["unexpected_count"] == 0
    assert receipt["forward_executed"] is False
    assert receipt["claim_boundary"]["gpu"] is False
    assert not OUTPUT.exists()
    assert not ATTEMPT.exists()


def test_r2_author_artifacts_remain_unchanged():
    assert digest(ROOT / "system_handoff/scripts/"
                  "run_m660r2_h67_layer_static_decoder_payload_one_shot.sh") == (
        "c8549148eed848fc0b8c6e58a5003f4b2c99f5822dce1ea89c5b31368ca78bb9")
    assert digest(ROOT / "contracts/"
                  "m660r2_h67_ep35_layer_static_decoder_payload_contract_r1_20260828.json") == (
        "0c6c22532ffa1a1cb70fd5a55cf94a75a594a20244ed878e6dc85f5ff47452fd")
