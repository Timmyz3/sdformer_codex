#!/usr/bin/env python3
"""Bounded exactness and fail-closed tests for M890 GTLS source."""

import importlib.util
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (ROOT / "system_simulator/scripts/"
          "analyze_m890_decoder_gtls_source_candidate.py")
SPEC = importlib.util.spec_from_file_location("m890_gtls", SCRIPT)
M890 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M890)


def test_synthetic_1k_old_m861_new_all_endpoints_and_classes():
    rows = M890.synthetic_transactions(1000)
    result = M890.exact_miter(rows, include_old=True)
    assert result["status"] == "PASS_EXACT_M768_M861_GTLS_MITER"
    assert result["expanded_requests"] == 1000
    assert result["live_token_peak"] > 0


def test_closed_form_and_fallback_are_exact_at_q_boundaries():
    assert M890.closed_form_boundary_self_test()["status"] == \
        "PASS_CLOSED_FORM_Q_BOUNDARIES"


def test_terminal_last_use_and_malicious_reuse_fail_closed():
    result = M890.liveness_attack_self_test()
    assert result["premature_retirement_rejected"] is True
    assert result["post_retirement_reuse_rejected"] is True


def test_dependency_on_nonterminal_token_is_rejected():
    producer = M890.synthetic_transactions(64)[0]
    consumer = M890.CompressedTransaction(
        transaction_id="bad_consumer", population_id="M890_SYNTHETIC",
        config="TYPED_SIGNED_K8", kind="compute", base_address=1 << 60,
        address_stride_bytes=0, count=1, bank_pattern=(0,),
        width_bytes=288, dependency_tokens=(M890.token_for(producer, 0),),
        produces_token_prefix="bad_consumer:done")
    with pytest.raises(M890.Failure):
        M890.PackedGroupIR([producer, consumer],
                           ("M890_SYNTHETIC", "TYPED_SIGNED_K8", 0, 0, 0))


def test_deterministic_row_shard_and_distinct_digest_schema():
    rows = M890.synthetic_transactions(128)
    ir = M890.PackedGroupIR(
        rows, ("M890_SYNTHETIC", "TYPED_SIGNED_K8", 17, 0, 9))
    assert ir.deterministic_shard(13) == ir.deterministic_shard(13)
    result = M890.GTLSScheduler(M890.M861._synthetic_resource()).schedule(
        ir, retain_details=False, retain_expanded_address_sha=False)
    assert result["compressed_group_ir_sha256"] == \
        ir.compressed_group_ir_sha256
    assert result["expanded_address_sha256"] is None
    assert result["transaction_address_sha256"] is None


def test_small_real_prefix_matches_m768_m861_new():
    rows = M890.real_prefix_transactions(1000)
    result = M890.exact_miter(rows, include_old=True)
    assert result["expanded_requests"] == 1000
    assert result["status"] == "PASS_EXACT_M768_M861_GTLS_MITER"


@pytest.mark.parametrize("flag", ["--run-full-first-row", "--run-production"])
def test_unbounded_modes_are_refused(flag):
    completed = subprocess.run(
        ["/opt/anaconda3/envs/pytorch310/bin/python3.10", str(SCRIPT), flag],
        text=True, capture_output=True, check=False)
    assert completed.returncode != 0
    assert "refuses full-row/production replay" in completed.stderr


def test_docs359_and_authorities_are_unchanged():
    assert M890.sha256(ROOT / "docs/359_DATE终局冻结_20260813.md") == \
        M890.DOCS359_SHA256
    for directory, filename, expected in (
            (M890.M883_DIR, "review.json", M890.M883_IDENTITY),
            (M890.M886_DIR, "review.json", M890.M886_IDENTITY),
            (M890.M887_DIR, "handoff.json", M890.M887_IDENTITY),
            (M890.M888_DIR, "request.json", M890.M888_IDENTITY)):
        identity = M890.M785.verify_sealed_directory(directory)
        assert M890.sha256(directory / filename) == expected[0]
        assert identity["manifest_sha256"] == expected[1]
        assert identity["outer_seal_file_sha256"] == expected[2]
