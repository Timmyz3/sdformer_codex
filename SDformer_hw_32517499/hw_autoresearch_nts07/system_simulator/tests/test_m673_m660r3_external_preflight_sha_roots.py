#!/usr/bin/env python3
"""Static author checks for the M660-r3 runner identity-root repair."""

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = (ROOT / "system_handoff/scripts/"
          "run_m660r3_h67_layer_static_decoder_payload_one_shot.sh")
CONTRACT = (ROOT / "contracts/"
            "m660r3_h67_ep35_layer_static_decoder_payload_contract_r1_20260828.json")
PREFLIGHT = (ROOT / "results/"
             "m660r3_h67_ep35_cpu_exact_load_preflight_r1_20260828")
OUTPUT = (ROOT / "system_handoff/outgoing/"
          "m660r3_h67_ep35_layer_static_decoder_payload_s10_r1_20260828")
ATTEMPT = (ROOT / "results/"
           ".m660r3_h67_ep35_layer_static_decoder_payload_r1_attempt_consumed")

RUNNER_SHA = "8fc347dc3ba8f8dba601a34938e1f5788c0c3c2153c3da9e0dbb09b7ecffdf55"
CONTRACT_SHA = "4acdfef539cdb03c26a3eeb9944842f94601e316676745bc36a1836f77705195"
PREFLIGHT_RECEIPT_SHA = "e773b5538ea39586b99a56c80f221df4f0e6e689fefc5648ecb6f413eb05f11b"
PREFLIGHT_OUTER_SHA = "97c565a9a458c7d8b793f0dbe9afb52a7a78566edc614706da2d935ec0bf5880"
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


def test_external_roots_precede_attempt_consumption():
    source = RUNNER.read_text(encoding="utf-8")
    required = source.index(
        "M660R3_EXPECTED_PREFLIGHT_RECEIPT_SHA256:-")
    receipt_compare = source.index(
        '"${M660R3_EXPECTED_PREFLIGHT_RECEIPT_SHA256}" &&', required)
    outer_compare = source.index(
        '"${M660R3_EXPECTED_PREFLIGHT_OUTER_SEAL_SHA256}" ]]',
        receipt_compare)
    mismatch_exit = source.index("exit 41", outer_compare)
    attempt = source.index('mkdir "${m660r2_attempt}"', mismatch_exit)
    capture = source.index("m660r2_capture_started=1", attempt)
    assert required < receipt_compare < outer_compare < mismatch_exit < attempt < capture


def test_contract_and_runner_bind_each_other_and_r3_paths():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    runner = contract["inputs"]["runner"]
    assert runner == {
        "path": "hw_autoresearch_nts07/system_handoff/scripts/run_m660r3_h67_layer_static_decoder_payload_one_shot.sh",
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
