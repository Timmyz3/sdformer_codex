#!/usr/bin/env python3
"""Receipt-blind M1286 source hammer for the M1281 calibration framework.

This script reads only the frozen M1281 source/contract and the M1111DR2
source/contract.  It never opens an M1281 receipt, a result directory, a live
work prefix, or a canonical calibration payload.  All dynamic checks use
synthetic in-memory fixtures.
"""

import ast
import copy
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
M1281_SOURCE = ROOT / "system_simulator/scripts/build_m1281_decoder_cycle_traffic_surrogate_calibration_source.py"
M1281_TEST = ROOT / "system_simulator/tests/test_m1281_decoder_cycle_traffic_surrogate_calibration_source.py"
M1281_CONTRACT = ROOT / "contracts/m1281_decoder_cycle_traffic_surrogate_calibration_source_contract_r1_20260830.json"
M1111_SOURCE = ROOT / "system_simulator/scripts/run_m1111dr2_m1105dr2_decoder_only_production_zero_arg.py"
M1111_CONTRACT = ROOT / "contracts/m1111dr2_m1105dr2_decoder_only_production_runner_source_contract_r2_20260830.json"

EXPECTED_SHA = {
    M1281_SOURCE: "098d7c0e96df18ed9eda2f43e26230b86ba5afbef3975c46d695ec8953e7a4ce",
    M1281_TEST: "c812b11c05d4fc00b30b4d029686e0d245aaefafb27ca1135c11fca78c14f170",
    M1281_CONTRACT: "829a0766f1d79a8acfdade0fd42853f445699e533b9ab918c745e8bc460501f9",
    M1111_SOURCE: "1167258c228631b73ca1784ae57db19e8f0fbe709efa34f369585c508bc9d746",
    M1111_CONTRACT: "821819b00503b91a8fb8dfca8fe000208e10746e751a3815131dc8ff1cbed515",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def expect_reject(module, payload, synthetic_fixture):
    try:
        module.calibrate_payload(payload, synthetic_fixture=synthetic_fixture)
    except module.CalibrationError:
        return True
    return False


def rewrite_row(module, row, group_count, active_terms, constant):
    row["group_count"] = group_count
    row["active_source_terms"] = active_terms
    row["measured_cycles"] = 4 * group_count + constant
    traffic = module.expected_traffic(group_count, active_terms)
    for key in ("descriptor_bytes", "weight_bytes", "psum_read_bytes",
                "compute_count", "psum_write_bytes", "commit_bytes"):
        row[key] = traffic[key]


def assignment_literal(source_text, name):
    tree = ast.parse(source_text)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name
                   for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError("missing literal assignment " + name)


def run():
    for path, expected in EXPECTED_SHA.items():
        require(sha256(path) == expected, "source identity drift: " + str(path))

    spec = importlib.util.spec_from_file_location("m1281_blind_target", M1281_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    findings = {}

    baseline = module.calibrate_payload(module.synthetic_payload(), synthetic_fixture=True)
    require(baseline["cycle_surrogate"]["error_gate_pass"] is True and
            baseline["cycle_surrogate"]["analytical_cycle_annex_allowed"] is False,
            "baseline synthetic boundary drift")
    findings["baseline_synthetic"] = "PASS_FIXTURE_ONLY_ANNEX_FALSE"

    # Pseudo-authority attack: syntactically valid SHA strings and pass booleans
    # are accepted without opening/verifying any sealed artifact.
    forged_authority = module.synthetic_payload()
    forged_authority["authority"].update({
        "result_sha256": "a" * 64,
        "result_outer_seal_file_sha256": "b" * 64,
        "result_hammer_review_sha256": "c" * 64,
        "result_outer_seal_pass": True,
        "result_hammer_pass": True,
        "synthetic_fixture": False,
    })
    forged_result = module.calibrate_payload(forged_authority, synthetic_fixture=False)
    require(forged_result["cycle_surrogate"]["analytical_cycle_annex_allowed"] is True,
            "forged authority unexpectedly rejected")
    findings["forged_seal_and_pass_strings"] = "ACCEPTED__CRITICAL"

    # `is synthetic_fixture` does not prove Boolean type.  The integer zero is
    # accepted on both sides and takes the non-fixture/annex branch.
    bool_confusion = module.synthetic_payload()
    bool_confusion["authority"]["synthetic_fixture"] = 0
    confused_result = module.calibrate_payload(bool_confusion, synthetic_fixture=0)
    require(confused_result["cycle_surrogate"]["analytical_cycle_annex_allowed"] is True and
            type(confused_result["synthetic_fixture"]) is int,
            "synthetic fixture type confusion unexpectedly rejected")
    findings["synthetic_fixture_bool_type_confusion"] = "ACCEPTED__HIGH"

    # Order checks are effective.
    ordinal_attack = module.synthetic_payload()
    ordinal_attack["calls"][1]["global_call_ordinal"] = 0
    require(expect_reject(module, ordinal_attack, True), "ordinal attack accepted")
    layer_attack = module.synthetic_payload()
    layer_attack["calls"][0]["layer"] = "D1"
    require(expect_reject(module, layer_attack, True), "layer order attack accepted")
    findings["ordinal_and_layer_order"] = "REJECTED__PASS"

    # Uncoordinated traffic mutation is rejected, but coordinated self-reported
    # group/term/traffic/cycle fabrication passes.  Positive groups with zero
    # source terms are also not a legal M1111 bank-unique group population.
    raw_traffic_attack = module.synthetic_payload()
    raw_traffic_attack["calls"][0]["psum_read_bytes"] += 1
    require(expect_reject(module, raw_traffic_attack, True),
            "uncoordinated traffic attack accepted")
    coordinated = module.synthetic_payload()
    rewrite_row(module, coordinated["calls"][0], 777, 0, 17)
    coordinated_result = module.calibrate_payload(coordinated, synthetic_fixture=True)
    require(coordinated_result["cycle_surrogate"]["error_gate_pass"] is True,
            "coordinated self-report attack unexpectedly rejected")
    findings["uncoordinated_traffic_forgery"] = "REJECTED__PASS"
    findings["coordinated_group_term_traffic_cycle_forgery"] = "ACCEPTED__HIGH"

    # Nominal 30/layer population can have only one unique observation per
    # layer.  There is no sequence/sample/module identity or effective-sample
    # diversity gate.
    collapsed = module.synthetic_payload()
    constants = {"D0": 17, "D1": 23, "D2": 31, "D3": 41}
    for row in collapsed["calls"]:
        layer_index = module.LAYERS.index(row["layer"])
        rewrite_row(module, row, 900 + layer_index, 7 + layer_index,
                    constants[row["layer"]])
    collapsed_result = module.calibrate_payload(collapsed, synthetic_fixture=True)
    distinct = {layer: len({(row["group_count"], row["active_source_terms"],
                            row["measured_cycles"])
                           for row in collapsed["calls"] if row["layer"] == layer})
                for layer in module.LAYERS}
    require(collapsed_result["cycle_surrogate"]["error_gate_pass"] is True and
            distinct == {layer: 1 for layer in module.LAYERS},
            "collapsed effective sample unexpectedly rejected")
    findings["nominal_120_but_one_unique_sample_per_layer"] = "ACCEPTED__HIGH"

    # Boundary is correctly <= 0.1%.  Build 29 residual-17 observations and
    # one residual-20 observation for D0: fitted residual is 17.1, so the last
    # error is 2.9/2900 = exactly 0.001.
    boundary = module.synthetic_payload()
    for row in boundary["calls"]:
        layer_index = module.LAYERS.index(row["layer"])
        if row["layer"] == "D0":
            rewrite_row(module, row, 720, 8, 17)
        else:
            rewrite_row(module, row, 800 + layer_index, 8,
                        constants[row["layer"]])
    boundary["calls"][116]["measured_cycles"] = 2900
    at_gate = module.calibrate_payload(boundary, synthetic_fixture=True)
    require(at_gate["cycle_surrogate"]["global_max_relative_error"] ==
            "0.001000000000000000" and
            at_gate["cycle_surrogate"]["error_gate_pass"] is True,
            "inclusive 0.1% boundary drift")
    above = copy.deepcopy(boundary)
    above["calls"][116]["measured_cycles"] = 2901
    above_gate = module.calibrate_payload(above, synthetic_fixture=True)
    require(Decimal(above_gate["cycle_surrogate"]["global_max_relative_error"]) >
            Decimal("0.001") and
            above_gate["cycle_surrogate"]["error_gate_pass"] is False,
            "above-gate error accepted")
    findings["error_gate_boundary"] = "EXACT_LTE_ACCEPTED_AND_ABOVE_REJECTED__PASS"

    promoted = module.synthetic_payload()
    promoted["claim_boundary"]["system_speedup_admitted"] = True
    require(expect_reject(module, promoted, True), "input claim promotion accepted")
    findings["input_claim_promotion"] = "REJECTED__PASS"

    # M1111DR2 transaction semantics.  The first five surrogate components are
    # compatible when projected from exact kind summaries; commit is not.
    m1111_text = M1111_SOURCE.read_text()
    geometry = assignment_literal(m1111_text, "MODULE_GEOMETRY")
    require("(desc_address,), (0,), 16" in m1111_text and
            "weight_addresses, weight_banks, 16" in m1111_text and
            "tuple(psum_base + bank * 48 for bank in range(6))" in m1111_text and
            'scheduler.issue(audit, "output_commit", (address,), (0,), 288' in m1111_text,
            "M1111 transaction source pattern drift")
    expected_commit = {}
    for module_index, (_, cout, _, _, hout, wout) in geometry.items():
        blocks = (cout + 95) // 96
        expected_commit["D" + str(module_index)] = 10 * hout * wout * blocks * 288
    require(expected_commit == {
        "D0": 13824000,
        "D1": 27648000,
        "D2": 55296000,
        "D3": 221184000,
    }, "M1111 commit derivation drift")
    require(all(value != module.COMMIT_BYTES_PER_CALL for value in expected_commit.values()),
            "M1281 fixed commit unexpectedly compatible")
    findings["m1111_formula_compatibility"] = {
        "compatible": ["descriptor_bytes", "weight_bytes", "psum_read_bytes",
                       "compute_count", "psum_write_bytes"],
        "incompatible": "commit_bytes",
        "m1281_commit_bytes_per_call": module.COMMIT_BYTES_PER_CALL,
        "m1111_commit_bytes_per_call_by_layer": expected_commit,
    }

    return {
        "schema": "m1286_m1281_receipt_blind_source_hammer_v1",
        "status": "STOP_M1281_REAL_ADAPTER_AND_ANALYTICAL_ANNEX__SOURCE_REPAIR_REQUIRED",
        "receipt_blind": True,
        "live_prefix_opened": False,
        "real_calibration_run": False,
        "eda_gpu_remote": False,
        "source_identities": {str(path.relative_to(ROOT)): digest
                              for path, digest in EXPECTED_SHA.items()},
        "findings": findings,
        "score": 61,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))
