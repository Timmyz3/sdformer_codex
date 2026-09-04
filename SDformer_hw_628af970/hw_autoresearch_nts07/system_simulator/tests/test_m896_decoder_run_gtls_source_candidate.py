#!/usr/bin/env python3
"""Bounded exactness, resident-state and fail-closed tests for M896."""

import importlib.util
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (ROOT / "system_simulator/scripts/"
          "analyze_m896_decoder_run_gtls_source_candidate.py")
SPEC = importlib.util.spec_from_file_location("m896_run_gtls", SCRIPT)
M896 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M896)


@pytest.mark.parametrize("count", [1000, 10000])
def test_synthetic_old_m861_m890_run_gtls_every_endpoint(count):
    result = M896.exact_miter(M896.M890.synthetic_transactions(count),
                              include_old=True)
    assert result["status"] == \
        "PASS_EXACT_M768_M861_M890_RUN_GTLS_MITER"
    assert result["expanded_requests"] == count


@pytest.mark.parametrize("count", [1000, 10000])
def test_real_old_m861_m890_run_gtls_every_endpoint(count):
    result = M896.exact_miter(M896.M890.real_prefix_transactions(count),
                              include_old=True)
    assert result["status"] == \
        "PASS_EXACT_M768_M861_M890_RUN_GTLS_MITER"
    assert result["expanded_requests"] == count


def test_real_100k_m861_m890_run_gtls_every_endpoint_calendar_and_class():
    result = M896.exact_miter(M896.M890.real_prefix_transactions(100000),
                              include_old=False)
    assert result["status"] == "PASS_EXACT_M861_M890_RUN_GTLS_MITER"
    assert result["expanded_requests"] == 100000
    assert result["terminal_readiness_sha256"] == \
        "a55d8cfa67f47863bc561323d01c674f1dd8d35555f3a972ab78d72bf44891ee"


def test_100k_measured_combined_state_projects_below_512mib():
    result = M896.measure_real_100k_state()
    assert result["status"] == \
        "PASS_RUN_GTLS_100K_COMBINED_STATE_PROJECTION_GATE"
    assert result["combined_live_event_state_bytes"] > 0
    assert result["conservative_projection_bytes"] <= 512 * 1024 * 1024
    assert result["serialized_or_compressed_file_size_used"] is False
    assert result["full_row_authorized"] is False


def test_priority_run_counted_ap_and_liveness_attacks():
    assert M896.priority_run_self_test()["status"] == \
        "PASS_PRIORITY_RUN_AND_COUNTED_AP_SELF_TEST"
    attack = M896.liveness_attack_self_test()
    assert attack["premature_retirement_rejected"] is True
    assert attack["post_retirement_rejected"] is True


def test_nonterminal_dependency_is_rejected():
    producer = M896.M890.synthetic_transactions(64)[0]
    consumer = M896.CompressedTransaction(
        transaction_id="bad_consumer", population_id="M890_SYNTHETIC",
        config="TYPED_SIGNED_K8", kind="compute", base_address=1 << 60,
        address_stride_bytes=0, count=1, bank_pattern=(0,),
        width_bytes=288,
        dependency_tokens=(M896.M890.token_for(producer, 0),),
        produces_token_prefix="bad_consumer:done")
    with pytest.raises(M896.Failure):
        M896.RunGroupIR(
            [producer, consumer],
            ("M890_SYNTHETIC", "TYPED_SIGNED_K8", 0, 0, 0))


@pytest.mark.parametrize("flag", ["--run-full-first-row", "--run-production"])
def test_unbounded_modes_are_refused(flag):
    completed = subprocess.run(
        ["/opt/anaconda3/envs/pytorch310/bin/python3.10", str(SCRIPT), flag],
        text=True, capture_output=True, check=False)
    assert completed.returncode != 0
    assert "refuses full-row/production replay" in completed.stderr


def test_docs359_m890_and_m893_authorities_are_unchanged():
    assert M896.sha256(ROOT / "docs/359_DATE终局冻结_20260813.md") == \
        M896.DOCS359_SHA256
    assert M896.sha256(M896.M890_PATH) == M896.M890_SHA256
    identity = M896.M785.verify_sealed_directory(M896.M893_DIR)
    assert M896.sha256(M896.M893_DIR / "review.json") == \
        M896.M893_IDENTITY[0]
    assert identity["manifest_sha256"] == M896.M893_IDENTITY[1]
    assert identity["outer_seal_file_sha256"] == M896.M893_IDENTITY[2]
