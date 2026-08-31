#!/usr/bin/env python3
"""Independent fail-closed hammer for the M1014 source-only package.

This hammer opens no real decoder payload and executes only the frozen
M890 synthetic transaction population.  Its mutation attacks are in-memory;
the frozen M785/M890/M896/M946 sources are never edited.
"""

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
PY_SOURCE = HW / "system_simulator/scripts/analyze_m1014_decoder_stratified_block_reset_windows_source.py"
CHECKER = HW / "system_simulator/scripts/check_m1014_decoder_stratified_block_reset_windows_source.py"
TEST = HW / "system_simulator/tests/test_m1014_decoder_stratified_block_reset_windows_source.py"
CONTRACT = HW / "contracts/m1014_decoder_stratified_block_reset_windows_source_contract_r1_20260829.json"
PLAN = HW / "contracts/m1009_decoder_stratified_window_source_plan_contract_r1_20260829.json"
RECEIPT = HW / "reviews/m1014_decoder_stratified_block_reset_windows_source_receipt_r1_20260829"
PLAN_REVIEW = HW / "reviews/m1009_decoder_stratified_window_source_plan_first_principles_r1_20260829"

EXPECTED = {
    PY_SOURCE: "c1fb987bd6d9921286fd9c53f3c9374d9c4779d9b3617946ab9b3d7ab11e2c64",
    CHECKER: "67f9ada909ee8c0fe72122d2b2b35722f88e3ced1bc5ccfaf03caaf8f02da01b",
    TEST: "9d85543a1442d28de07e8c0696798e17a6d70966a4f317f0040c8786202cfe95",
    CONTRACT: "c5a8f3ac9c9c919eaa44e6cf735ae84aba143957d46228d33264210eeb7769ce",
    PLAN: "d107c7e7ef1a8a0971a1bd4882e0ff7f46140787c7ad7afa2842366f6e5b6999",
    HW / "system_simulator/scripts/analyze_m768_h67_decoder_a1_k8_address_timed_cycles.py": "926069762c6274bae3aa7b88352e29fff8219cbbceba2f2be0ec46ee304a3f37",
    HW / "system_simulator/scripts/analyze_m861_decoder_streaming_event_sweep.py": "f72ed3b820051d624699152b784c05fa674106556ab73f452a2cf96a9f72d7a4",
    HW / "system_simulator/scripts/analyze_m785_h67_decoder_physical_residency_repair.py": "7fbd72d27e4733179d1d3037080c69ebc9e6ceb0aa5716cc497d3dfee81070f1",
    HW / "system_simulator/scripts/analyze_m890_decoder_gtls_source_candidate.py": "cacc118ea33616ae4284403ad69656bbeacaa7bc83d227c0d9b5a86c2ead459e",
    HW / "system_simulator/scripts/analyze_m896_decoder_run_gtls_source_candidate.py": "c877f70849eb254bd5b227c79e8120773a9c48aa7405a2e6564b7eb4647aae39",
    HW / "system_simulator/scripts/analyze_m946_decoder_multilayer_bounded_prefix_source_candidate.py": "0ffd1ee810f24d1a95b0df33ffe8eae43240920e12a2fccb86c947d2be51b6ac",
    HW / "docs/359_DATE终局冻结_20260813.md": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_flat_seal(directory):
    directory = Path(directory)
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    assert outer.read_text(encoding="utf-8") == sha256(manifest) + "  SHA256SUMS\n"
    listed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        assert name not in listed
        item = directory / name
        assert item.is_file() and not item.is_symlink() and sha256(item) == digest
        listed[name] = digest
    actual = {item.name for item in directory.iterdir()
              if item.is_file() and item.name not in
              ("SHA256SUMS", "SHA256SUMS.seal.sha256")}
    assert set(listed) == actual
    return {
        "manifest_sha256": sha256(manifest),
        "outer_seal_file_sha256": sha256(outer),
        "member_count": len(listed),
    }


def load_source():
    spec = importlib.util.spec_from_file_location("m1017_target", PY_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expect_runtime(callable_, contains):
    try:
        callable_()
    except RuntimeError as error:
        return contains in str(error)
    return False


def main():
    identities = {str(path.relative_to(HW)): sha256(path)
                  for path in EXPECTED}
    identity_pass = all(identities[str(path.relative_to(HW))] == expected
                        for path, expected in EXPECTED.items())
    receipt_seal = verify_flat_seal(RECEIPT)
    plan_seal = verify_flat_seal(PLAN_REVIEW)
    m = load_source()

    attacks = {}
    attacks["d1_rejected_before_scheduler"] = expect_runtime(
        lambda: m.frozen_route("D1"), "STRICT_COMMON_CHARGE")

    body_no_commit = m.M890.synthetic_transactions(64)
    no_commit_spec = m.WindowSpec("no-commit", "D0", "COMMIT_TAIL", 1)
    wrapped, _ = m.block_reset_transactions(body_no_commit, no_commit_spec,
                                             "candidate")
    attacks["commit_tail_zero_commit_rejected"] = expect_runtime(
        lambda: m.exact_replay(wrapped, no_commit_spec), "zero commit")

    too_large = m.M890.synthetic_transactions(9998)
    large_spec = m.WindowSpec("too-large", "D0", "COMPUTE_REGULAR", 1)
    attacks["window_above_10k_rejected"] = expect_runtime(
        lambda: m.block_reset_transactions(too_large, large_spec, "candidate"),
        "exceeds 10K")

    rows = [{"block_id": "b{:02d}".format(i), "compute_count": 1}
            for i in range(40)]
    attacks["selection_above_32_rejected"] = expect_runtime(
        lambda: m.deterministic_select(rows, "COMPUTE_REGULAR", 33),
        "frozen bound")

    mutated_contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    mutated_contract["sampling"]["pilot_per_noncensus_stratum"] = 7
    with tempfile.TemporaryDirectory(prefix="m1017_pilot_attack_") as temporary:
        path = Path(temporary) / "contract.json"
        path.write_text(json.dumps(mutated_contract), encoding="utf-8")
        attacks["pilot_not_8_rejected"] = expect_runtime(
            lambda: m.validate_source(path), "sampling drift")

    direct_cycle_rows = [
        {"block_id": "cycle-a", "compute_count": 1, "cycles": 1},
    ]
    attacks["literal_cycles_selector_rejected"] = expect_runtime(
        lambda: m.deterministic_select(direct_cycle_rows,
                                       "COMPUTE_REGULAR", 1),
        "cycle-derived")

    # Adversarial alias of an observed cycle field.  A truly cycle-blind
    # selector must allowlist metadata, not merely blacklist four spellings.
    aliased_cycle_rows = [
        {"block_id": "late", "compute_count": 1, "total_cycles": 999},
        {"block_id": "early", "compute_count": 1, "total_cycles": 1},
    ]
    attacks["aliased_total_cycles_selector_rejected"] = expect_runtime(
        lambda: m.deterministic_select(aliased_cycle_rows,
                                       "COMPUTE_REGULAR", 1),
        "cycle-derived")

    body = m.M890.synthetic_transactions(448)
    asym_spec = m.WindowSpec("asymmetric-reset", "D0", "COMMIT_TAIL", 1)
    original_wrapper = m.block_reset_transactions

    def asymmetric_wrapper(body_arg, spec_arg, side_arg):
        output, metadata = original_wrapper(body_arg, spec_arg, side_arg)
        if side_arg == "baseline":
            # Preserve count=3 so a count-only symmetry check cannot detect
            # that boundary moved from compute to external-read service.
            output[0] = replace(output[0], kind="external_read",
                                base_address=1 << 60, width_bytes=1)
        return output, metadata

    m.block_reset_transactions = asymmetric_wrapper
    asym_rejected = False
    asym_observation = {}
    try:
        result = m.paired_replay(body, body, asym_spec)
        asym_observation = {
            "accepted": True,
            "candidate_cycles": result["candidate_cycles"],
            "baseline_cycles": result["baseline_cycles"],
            "candidate_reset_count": result["candidate_reset"]["reset_expanded_request_count"],
            "baseline_reset_count": result["baseline_reset"]["reset_expanded_request_count"],
        }
    except RuntimeError as error:
        asym_rejected = True
        asym_observation = {"accepted": False, "error": str(error)}
    finally:
        m.block_reset_transactions = original_wrapper
    attacks["candidate_baseline_reset_semantics_asymmetry_rejected"] = asym_rejected

    high_ci = m.estimate_paired_totals([
        {"stratum": "COMPUTE_REGULAR", "population_blocks": 1000,
         "candidate_cycles": [1, 100, 1, 100, 1, 100, 1, 100],
         "baseline_cycles": [100, 1, 100, 1, 100, 1, 100, 1]},
    ])
    point = float(high_ci["paired_speedup_estimate"])
    low, high = (float(value) for value in high_ci["paired_speedup_ci95"])
    relative_halfwidth = max(point - low, high - point) / point
    # The current API always returns the point, so this attack is expected to
    # fail until an explicit hard-stop admission layer is added.
    attacks["ci_above_10pct_suppresses_point_estimate"] = (
        relative_halfwidth > 0.10 and
        high_ci.get("paired_speedup_estimate") is None)

    source_text = PY_SOURCE.read_text(encoding="utf-8")
    attacks["transaction_ratio_guard_present"] = (
        '"transaction_ratio_is_speedup": False' in source_text and
        "expanded_request_count/compressed_transaction_count" not in
        json.dumps(high_ci, sort_keys=True))

    required_fail_closed = [
        "d1_rejected_before_scheduler",
        "commit_tail_zero_commit_rejected",
        "window_above_10k_rejected",
        "selection_above_32_rejected",
        "pilot_not_8_rejected",
        "literal_cycles_selector_rejected",
        "aliased_total_cycles_selector_rejected",
        "candidate_baseline_reset_semantics_asymmetry_rejected",
        "ci_above_10pct_suppresses_point_estimate",
        "transaction_ratio_guard_present",
    ]
    failed = [name for name in required_fail_closed if not attacks[name]]
    output = {
        "schema": "m1017_m1014_decoder_stratified_block_reset_windows_independent_hammer_v1",
        "status": ("PASS_M1017_M1014_SOURCE_HAMMER" if not failed else
                   "FAIL_M1017_M1014_SOURCE_HAMMER__BLOCK_EXECUTION_RELEASE"),
        "identity_pass": identity_pass,
        "identities": identities,
        "receipt_seal": receipt_seal,
        "m1009_plan_seal": plan_seal,
        "attacks": attacks,
        "failed_attacks": failed,
        "asymmetric_reset_observation": asym_observation,
        "high_ci_observation": {
            "paired_speedup_estimate_returned": point,
            "paired_speedup_ci95": [low, high],
            "relative_halfwidth": relative_halfwidth,
            "hard_stop_threshold": 0.10,
        },
        "real_payload_opened": False,
        "real_window_execution": False,
        "eda_gpu_remote_used": False,
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
