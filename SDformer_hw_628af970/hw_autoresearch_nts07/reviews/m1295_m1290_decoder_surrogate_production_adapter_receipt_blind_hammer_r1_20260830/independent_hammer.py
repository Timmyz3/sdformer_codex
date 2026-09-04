#!/usr/bin/env python3
"""Receipt-blind, synthetic-only hammer for the M1290 repair.

This script intentionally does not open the M1290 author receipt, the live
M1111DR2 production result, or the canonical M1291 hammer.  It imports only the
frozen M1290 source under review and constructs sealed data in a temporary
directory.
"""
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HW / "system_simulator/scripts/build_m1290_decoder_surrogate_production_adapter_r1.py"
TEST = HW / "system_simulator/tests/test_m1290_decoder_surrogate_production_adapter_r1.py"
CONTRACT = HW / "contracts/m1290_m1281_decoder_surrogate_production_adapter_repair_source_contract_r1_20260830.json"
EXPECTED_SOURCE_SHA = "bf5b19e8740cb94a724133fec0f7c02af422aebfa5a66558876f4c7186493936"
EXPECTED_TEST_SHA = "c77a84bacf72d9efc3fc2c07fdfb657a26208ebdec1a9042a25a2dee121f82d8"
EXPECTED_CONTRACT_SHA = "7d8dd56ba78c292c051ef080823c76aa5fc378bd1d6340e223883226c88906c1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def load_module():
    spec = importlib.util.spec_from_file_location("m1290_blind_target", str(SOURCE))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


M = load_module()


def summary(count: int, traffic: int, base: int) -> Dict[str, Any]:
    return {
        "count": count,
        "traffic_bytes": traffic,
        "address_first": base,
        "address_last": base + max(count - 1, 0),
        "issue_first": base,
        "issue_last": base + max(count - 1, 0),
        "return_first": base + 1,
        "return_last": base + count,
        "commit_first": base + 1,
        "commit_last": base + count,
        "stall_events": {"none": count},
    }


def make_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    transaction = 0
    cycle = 0
    constants = (17, 23, 31, 41)
    for ordinal in range(120):
        sample = ordinal // 4
        module = ordinal % 4
        groups = 1100 + ordinal
        terms = groups * (2 + ordinal % 6)
        commit = M.EXPECTED_COMMIT_BYTES[module]
        counts = {
            "input_descriptor_read": groups,
            "weight_read": groups,
            "psum_read": groups,
            "compute": groups,
            "psum_write": groups,
            "output_commit": commit // 288,
        }
        traffic = {
            "input_descriptor_read": groups * 16,
            "weight_read": terms * 16,
            "psum_read": groups * 288,
            "compute": groups * 288,
            "psum_write": groups * 288,
            "output_commit": commit,
        }
        traffic.update({
            "total": sum(traffic.values()),
            "external": traffic["input_descriptor_read"] + commit,
            "onchip": traffic["weight_read"] + traffic["psum_read"] + traffic["psum_write"],
        })
        tx_count = sum(counts.values())
        cycles = 4 * groups + constants[module]
        row = {
            "schema": "m1111dr2_decoder_address_timed_call_schedule_v2",
            "global_call_ordinal": ordinal,
            "sequence_ordinal": sample // 10,
            "sequence": M.SEQUENCES[sample // 10],
            "sequence_sample_id": sample % 10,
            "module_ordinal": module,
            "module": M.MODULES[module],
            "configuration": M.CONFIGURATION,
            "d1_exact_theta": module == 1,
            "d1_theta_word_uint32": 1065353139 if module == 1 else None,
            "d1_weight_folding": False,
            "transaction_ordinal_first": transaction,
            "transaction_ordinal_last": transaction + tx_count - 1,
            "transaction_count": tx_count,
            "address_digest_sha256": digest(f"independent-address-{ordinal}"),
            "dependency_digest_sha256": digest(f"independent-dependency-{ordinal}"),
            "schedule_digest_sha256": digest(f"independent-schedule-{ordinal}"),
            "cycle_start": cycle,
            "cycle_end": cycle + cycles,
            "diagnostic_cycles": cycles,
            "diagnostic_traffic_bytes": traffic,
            "kind_summaries": {
                kind: summary(counts[kind], traffic[kind], ordinal * 1000000 + index * 1000)
                for index, kind in enumerate(M.KINDS)
            },
            "claim_boundary": {
                "diagnostic_only": True,
                "speedup_admitted": False,
                "system_speedup_admitted": False,
                "paper_ppa_ready": False,
                "final_checkpoint_rebind_required": True,
            },
        }
        rows.append(row)
        transaction += tx_count
        cycle += cycles
    return rows


def seal_result(directory: Path) -> None:
    bundle = directory / M.SEAL_DIR
    bundle.mkdir(exist_ok=True)
    for child in tuple(bundle.iterdir()):
        child.unlink()
    manifest = bundle / M.MANIFEST
    manifest.write_text("\n".join(sha(directory / name) + "  " + name for name in M.RESULT_FILES) + "\n")
    (bundle / M.OUTER).write_text(sha(manifest) + "  " + M.MANIFEST + "\n")


def write_result(directory: Path, rows: List[Dict[str, Any]]) -> None:
    directory.mkdir()
    calls = directory / M.CALLS
    calls.write_text("".join(canonical(row) + "\n" for row in rows))
    contract = json.loads((HW / "contracts/m1111dr2_m1105dr2_decoder_only_production_runner_source_contract_r2_20260830.json").read_text())
    aggregate = {key: sum(row["diagnostic_traffic_bytes"][key] for row in rows)
                 for key in (*M.KINDS, "total", "external", "onchip")}
    payload = {
        "schema": "m1111dr2_m1105dr2_decoder_only_address_timed_result_v2",
        "status": "PASS_M1111DR2_DECODER_ONLY_DIAGNOSTIC_RESULT__FINAL_RESULT_HAMMER_REQUIRED",
        "identity": {
            "checkpoint": "H67_ep35",
            "checkpoint_sha256": "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158",
            "source_sha256": "b2d8ef4139283de06b7e332429bdf752ad16122ffbeda0ff7d75bce6d816a5c4",
            "contract_sha256": "821819b00503b91a8fb8dfca8fe000208e10746e751a3815131dc8ff1cbed515",
            "m1110d_outer_seal_file_sha256": "9caf64e422b4cb696a600b69415bd8265dc4694066fae7ec67a5f34019f39e23",
            "final_checkpoint_rebind_required": True,
        },
        "population": {
            "calls": 120,
            "timesteps_per_call": 10,
            "transaction_count": sum(row["transaction_count"] for row in rows),
            "call_schedule_sha256": sha(calls),
            "call_row_stream_digest_sha256": sha(calls),
        },
        "common_resource": contract["common_resource"],
        "diagnostic": {"cycles": rows[-1]["cycle_end"], "traffic_bytes": aggregate,
                       "ratios_or_speedups": None},
        "claim_boundary": {
            "decoder_only": True,
            "address_timed_transactions_complete": True,
            "same_resource_schedule_complete": True,
            "diagnostic_cycles_only": True,
            "diagnostic_traffic_only": True,
            "speedup_admitted": False,
            "system_speedup_admitted": False,
            "paper_ppa_ready": False,
            "paper_citable_performance": False,
            "final_checkpoint_rebind_required": True,
            "independent_result_hammer_required": True,
        },
    }
    (directory / M.PAYLOAD).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (directory / M.COMPLETE).write_bytes(
        b"M1111DR2_DECODER_DIAGNOSTIC_COMPLETE__RESULT_HAMMER_REQUIRED\n")
    seal_result(directory)


def refresh_result(directory: Path, rows: List[Dict[str, Any]]) -> None:
    calls = directory / M.CALLS
    calls.write_text("".join(canonical(row) + "\n" for row in rows))
    payload = json.loads((directory / M.PAYLOAD).read_text())
    aggregate = {key: sum(row["diagnostic_traffic_bytes"][key] for row in rows)
                 for key in (*M.KINDS, "total", "external", "onchip")}
    payload["population"].update({
        "transaction_count": sum(row["transaction_count"] for row in rows),
        "call_schedule_sha256": sha(calls),
        "call_row_stream_digest_sha256": sha(calls),
    })
    payload["diagnostic"]["cycles"] = rows[-1]["cycle_end"]
    payload["diagnostic"]["traffic_bytes"] = aggregate
    (directory / M.PAYLOAD).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    seal_result(directory)


def write_hammer(directory: Path, result: Path) -> None:
    directory.mkdir(exist_ok=True)
    seal = M.verify_result_seal(result)
    review = {
        "schema": M.HAMMER_SCHEMA,
        "status": M.HAMMER_STATUS,
        "identity": {
            "result_manifest_sha256": seal["manifest_sha256"],
            "result_outer_seal_file_sha256": seal["outer_seal_file_sha256"],
            "result_payload_sha256": seal["members"][M.PAYLOAD],
            "result_calls_sha256": seal["members"][M.CALLS],
            "result_completion_sha256": seal["members"][M.COMPLETE],
        },
        "verification": {
            "exact_three_payload_files": True,
            "result_manifest_and_outer_seal": True,
            "strict_120_call_rows": True,
            "kind_summaries_and_digests": True,
            "diagnostic_claim_boundary": True,
        },
        "claim_boundary": {"diagnostic_only": True, "analytical_annex": False,
                           "speedup": False, "system_speedup": False,
                           "paper_ppa_ready": False},
    }
    (directory / "review.json").write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    (directory / "RUN_COMPLETE.txt").write_text(M.HAMMER_STATUS + "\n")
    manifest = directory / M.MANIFEST
    manifest.write_text("\n".join(sha(directory / name) + "  " + name
                                  for name in ("RUN_COMPLETE.txt", "review.json")) + "\n")
    (directory / M.OUTER).write_text(sha(manifest) + "  " + M.MANIFEST + "\n")


def rejected(function, contains: Optional[str] = None) -> str:
    try:
        function()
    except (M.CalibrationError, TypeError) as error:
        if contains is not None and contains not in str(error):
            raise AssertionError(f"wrong rejection: wanted {contains!r}, got {error!r}")
        return type(error).__name__ + ": " + str(error)
    raise AssertionError("attack unexpectedly accepted")


def shift_transaction_tail(rows: List[Dict[str, Any]], start: int, delta: int) -> None:
    rows[start]["transaction_ordinal_last"] += delta
    rows[start]["transaction_count"] += delta
    for index in range(start + 1, len(rows)):
        rows[index]["transaction_ordinal_first"] += delta
        rows[index]["transaction_ordinal_last"] += delta


def shift_cycle_tail(rows: List[Dict[str, Any]], start: int, delta: int) -> None:
    rows[start]["cycle_end"] += delta
    rows[start]["diagnostic_cycles"] += delta
    for index in range(start + 1, len(rows)):
        rows[index]["cycle_start"] += delta
        rows[index]["cycle_end"] += delta


def run() -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    def record(name: str, detail: str) -> None:
        checks.append({"name": name, "status": "PASS", "detail": detail})

    assert sha(SOURCE) == EXPECTED_SOURCE_SHA
    assert sha(TEST) == EXPECTED_TEST_SHA
    assert sha(CONTRACT) == EXPECTED_CONTRACT_SHA
    record("frozen_input_sha", "source/test/contract match closed M1290 identities")

    with tempfile.TemporaryDirectory(prefix="m1295_synthetic_only_") as raw:
        root = Path(raw)
        result, hammer = root / "result", root / "hammer"
        baseline = make_rows()
        write_result(result, baseline)
        write_hammer(hammer, result)

        projected, authority = M.verify_production_authorities(result, hammer)
        assert len(projected) == 120
        assert len({(row["sequence"], row["sequence_sample_id"], row["module"])
                    for row in projected}) == 120
        assert all(row["group_count"] <= row["active_source_terms"] <= 8 * row["group_count"]
                   for row in projected)
        assert {row["traffic"]["commit_bytes"] for row in projected} == set(M.EXPECTED_COMMIT_BYTES)
        for module in range(4):
            selected = [row for row in projected if row["module_ordinal"] == module]
            assert len(selected) == 30
            assert len({(row["address_digest_sha256"], row["dependency_digest_sha256"],
                         row["schedule_digest_sha256"], row["kind_summary_digest_sha256"],
                         row["group_count"], row["active_source_terms"], row["measured_cycles"],
                         row["traffic"]["commit_bytes"]) for row in selected}) == 30
        assert authority["hammer"]["status"] == M.HAMMER_STATUS
        record("baseline_semantic_projection", "120 calls; 3 sequences; 30 samples; 4 modules; 30 distinct observations/layer")
        record("result_internal_seal", "exact three result files plus nested manifest and outer seal accepted")
        record("future_hammer_linkage", "separately sealed hammer linked all three member digests and both result seals")
        record("real_field_projection", "kind summaries, three row digests, exact per-module commit bytes and term bounds projected")

        fixture_payload = {"schema": "m1290_projected_fixture_v1", "calls": projected,
                           "claim_boundary": {"synthetic_fixture": True,
                                              "analytical_cycle_annex": False}}
        fixture = M.calibrate_fixture(fixture_payload, True)
        assert fixture["cycle_surrogate"]["analytical_cycle_annex_allowed"] is False
        assert fixture["claim_boundary"]["analytical_cycle_annex"] is False
        record("fixture_annex_forced_false", "valid synthetic fixture cannot admit analytical annex")
        for bad in (0, 1, None, False, "true"):
            rejected(lambda bad=bad: M.calibrate_fixture(fixture_payload, bad))
        record("fixture_exact_bool", "0/1/None/False/string type-confusion attacks rejected")
        assert len(inspect.signature(M.calibrate_production).parameters) == 0
        rejected(lambda: M.calibrate_production({"pass": True}))
        record("zero_argument_production_api", "caller path/boolean injection raises TypeError")

        fake = root / "forged_authority"
        fake.mkdir()
        (fake / "authority.json").write_text(json.dumps({
            "result_outer_seal_pass": True, "result_hammer_pass": True,
            "result_sha256": "0" * 64}) + "\n")
        rejected(lambda: M.verify_production_authorities(fake, hammer))
        record("forged_sha_pass", "naked SHA/PASS JSON cannot substitute for exact sealed result")

        # A fully resealed result remains rejected by the pre-existing independent
        # hammer when all group/term/traffic/cycle fields are changed together.
        rows = copy.deepcopy(baseline)
        row = rows[0]
        old_tx, old_cycles = row["transaction_count"], row["diagnostic_cycles"]
        groups = row["kind_summaries"]["input_descriptor_read"]["count"] + 29
        terms = groups * 3
        for kind in ("input_descriptor_read", "weight_read", "psum_read", "compute", "psum_write"):
            row["kind_summaries"][kind]["count"] = groups
            row["kind_summaries"][kind]["stall_events"] = {"none": groups}
        new_traffic = {"input_descriptor_read": groups * 16,
                       "weight_read": terms * 16,
                       "psum_read": groups * 288,
                       "compute": groups * 288,
                       "psum_write": groups * 288,
                       "output_commit": row["diagnostic_traffic_bytes"]["output_commit"]}
        for kind in M.KINDS:
            row["diagnostic_traffic_bytes"][kind] = new_traffic[kind]
            row["kind_summaries"][kind]["traffic_bytes"] = new_traffic[kind]
        row["diagnostic_traffic_bytes"]["total"] = sum(new_traffic.values())
        row["diagnostic_traffic_bytes"]["external"] = new_traffic["input_descriptor_read"] + new_traffic["output_commit"]
        row["diagnostic_traffic_bytes"]["onchip"] = new_traffic["weight_read"] + new_traffic["psum_read"] + new_traffic["psum_write"]
        new_tx = sum(row["kind_summaries"][kind]["count"] for kind in M.KINDS)
        shift_transaction_tail(rows, 0, new_tx - old_tx)
        new_cycles = 4 * groups + 17
        shift_cycle_tail(rows, 0, new_cycles - old_cycles)
        refresh_result(result, rows)
        rejected(lambda: M.verify_production_authorities(result, hammer), "linkage")
        record("coordinated_group_term_traffic_cycle", "fully coordinated resealed mutation rejected by future-hammer cryptographic linkage")

        # Recreate baseline, then use a fresh hammer: the semantic commit constant
        # must still reject a huge internally coordinated commit mutation.
        result2, hammer2 = root / "result_commit", root / "hammer_commit"
        rows = copy.deepcopy(baseline)
        row = rows[3]
        old_tx = row["transaction_count"]
        huge = M.EXPECTED_COMMIT_BYTES[3] + 288 * 100000
        old_count = row["kind_summaries"]["output_commit"]["count"]
        new_count = huge // 288
        row["kind_summaries"]["output_commit"].update({
            "count": new_count, "traffic_bytes": huge, "stall_events": {"none": new_count}})
        row["diagnostic_traffic_bytes"]["output_commit"] = huge
        row["diagnostic_traffic_bytes"]["total"] += huge - M.EXPECTED_COMMIT_BYTES[3]
        row["diagnostic_traffic_bytes"]["external"] += huge - M.EXPECTED_COMMIT_BYTES[3]
        shift_transaction_tail(rows, 3, new_count - old_count)
        assert rows[3]["transaction_count"] == old_tx + new_count - old_count
        write_result(result2, rows)
        write_hammer(hammer2, result2)
        rejected(lambda: M.verify_production_authorities(result2, hammer2), "output_commit")
        record("huge_commit_fresh_hammer", "freshly sealed huge per-call commit rejected against all four exact module constants")

        # Fresh hammer cannot rescue an illegal term/group relationship.
        result3, hammer3 = root / "result_bounds", root / "hammer_bounds"
        rows = copy.deepcopy(baseline)
        row = rows[0]
        row["kind_summaries"]["weight_read"]["traffic_bytes"] = 0
        row["diagnostic_traffic_bytes"]["weight_read"] = 0
        row["diagnostic_traffic_bytes"]["total"] = sum(row["diagnostic_traffic_bytes"][kind] for kind in M.KINDS)
        row["diagnostic_traffic_bytes"]["onchip"] = row["diagnostic_traffic_bytes"]["psum_read"] + row["diagnostic_traffic_bytes"]["psum_write"]
        write_result(result3, rows)
        write_hammer(hammer3, result3)
        rejected(lambda: M.verify_production_authorities(result3, hammer3), "active_source_terms")
        record("group_term_bound_fresh_hammer", "fresh hammer cannot bypass group<=terms<=8*group")

        # Collapse two observations within D0 and repair the global cycle chain;
        # all seals are fresh, so rejection must be the 30-distinct semantic gate.
        result4, hammer4 = root / "result_collapse", root / "hammer_collapse"
        rows = copy.deepcopy(baseline)
        source, target = rows[0], rows[4]
        for key in ("address_digest_sha256", "dependency_digest_sha256",
                    "schedule_digest_sha256", "kind_summaries",
                    "diagnostic_traffic_bytes", "diagnostic_cycles"):
            target[key] = copy.deepcopy(source[key])
        new_tx = sum(target["kind_summaries"][kind]["count"] for kind in M.KINDS)
        tx_delta = new_tx - target["transaction_count"]
        shift_transaction_tail(rows, 4, tx_delta)
        old_cycles = baseline[4]["diagnostic_cycles"]
        target["cycle_end"] = target["cycle_start"] + target["diagnostic_cycles"]
        delta = old_cycles - target["diagnostic_cycles"]
        for index in range(5, 120):
            rows[index]["cycle_start"] -= delta
            rows[index]["cycle_end"] -= delta
        write_result(result4, rows)
        write_hammer(hammer4, result4)
        rejected(lambda: M.verify_production_authorities(result4, hammer4), "30 distinct")
        record("distinct_observation_collapse", "freshly sealed per-layer observation collapse rejected")

        # Claim promotion is rejected in the row even under a fresh seal/hammer.
        result5, hammer5 = root / "result_claim", root / "hammer_claim"
        rows = copy.deepcopy(baseline)
        rows[0]["claim_boundary"]["speedup_admitted"] = True
        write_result(result5, rows)
        write_hammer(hammer5, result5)
        rejected(lambda: M.verify_production_authorities(result5, hammer5), "claim")
        record("claim_promotion", "freshly sealed speedup promotion rejected")

    return {
        "schema": "m1295_m1290_decoder_surrogate_receipt_blind_hammer_mechanical_v1",
        "status": "PASS_M1295_M1290_DECODER_SURROGATE_RECEIPT_BLIND_HAMMER",
        "scope": {"source_static_synthetic_only": True, "author_receipt_read": False,
                  "live_prefix_read": False, "canonical_result_read": False,
                  "real_calibration": False, "eda": False, "gpu": False, "remote": False},
        "identity": {"source_sha256": sha(SOURCE), "test_sha256": sha(TEST),
                     "contract_sha256": sha(CONTRACT)},
        "checks": checks,
        "counts": {"checks": len(checks), "pass": len(checks), "fail": 0},
        "findings": {"P0": 0, "P1": 0, "P2": 0},
        "score": 100,
        "claim_boundary": {"production_calibration_run": False,
                           "analytical_annex_admitted": False,
                           "speedup_admitted": False,
                           "system_speedup_admitted": False,
                           "paper_ppa_ready": False},
    }


if __name__ == "__main__":
    output = run()
    (HERE / "mechanical_checks.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(output["status"], output["counts"], output["findings"], output["score"])
