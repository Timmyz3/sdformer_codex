#!/usr/bin/env python3
"""Receipt-blind M1332 source hammer.  This script never launches EDA."""
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
CHECKER = HW / "system_simulator/scripts/check_m1332_c2_headline_mapped_production_activity_source.py"
CONTRACT = HW / "contracts/m1332_c2_headline_mapped_production_activity_source_contract_r1_20260831.json"

spec = importlib.util.spec_from_file_location("m1332_blind_target", CHECKER)
M = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = M
spec.loader.exec_module(M)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def saif(duration, endpoint_tc=4, activity_inside_dut=True):
    payload = f'''(PORT
      (clk_core (T0 1) (T1 1) (TX 0) (TC 20))
      (rst_core (T0 1) (T1 0) (TX 0) (TC 0))
      (raw_valid (T0 1) (T1 1) (TX 0) (TC 2))
      (raw_accept (T0 1) (T1 1) (TX 0) (TC 2))
      (mem_req_accept[0] (T0 1) (T1 1) (TX 0) (TC {endpoint_tc}))
      (mem_rsp_accept[0] (T0 1) (T1 1) (TX 0) (TC {endpoint_tc}))
      (result_accumulator[0] (T0 1) (T1 1) (TX 0) (TC 6))
      (result_accept (T0 1) (T1 1) (TX 0) (TC 4))
      (token_done_accept (T0 1) (T1 1) (TX 0) (TC 2)))'''
    if activity_inside_dut:
        body = f"(INSTANCE core (INSTANCE dut {payload}))"
    else:
        body = f"(INSTANCE core (INSTANCE dut)) (INSTANCE assertions {payload})"
    return f"(SAIFILE (DURATION {duration}) (INSTANCE tb_m1332_c2_headline_mapped_production_activity {body}))"


def write_tmp(directory, name, text):
    path = Path(directory) / name
    path.write_text(text)
    return path


def mutant_contract(directory, omitted_suffix):
    data = json.loads(CONTRACT.read_text())
    data["source_files"] = [
        item for item in data["source_files"]
        if not item["path"].endswith(omitted_suffix)
    ]
    path = Path(directory) / ("contract_without_" + omitted_suffix.replace("/", "_") + ".json")
    path.write_text(json.dumps(data, sort_keys=True))
    return path


def static_mutation_accepted(directory, global_name, original_path,
                             omitted_suffix, mutated_text):
    old = getattr(M, global_name)
    try:
        mutated = write_tmp(directory, Path(original_path).name, mutated_text)
        setattr(M, global_name, mutated)
        contract = mutant_contract(directory, omitted_suffix)
        return M.validate_static(contract)["status"] == "PASS_M1332_SOURCE_ONLY__NO_EDA"
    finally:
        setattr(M, global_name, old)


def main():
    checks = []
    false_negatives = []

    baseline = M.validate_static()
    require(baseline["status"] == "PASS_M1332_SOURCE_ONLY__NO_EDA", "baseline static failed")
    checks.append("baseline_static_pass")

    expected_identity = {
        "k8_net": "6b745030df6c041a0501d041ee277459c726c52263b4eec6ab5712f14d156de5",
        "k8_sdc": "70a0d0e7700188f5a80f31b06c2f3d401f56c7d1e2a29428e3837064a722a96c",
        "k1x8_net": "65f89c13d0b181fd26708b385fc831bb4493328e24a15bbb07c2dc40f27677dc",
        "k1x8_sdc": "24806d5c2d5c0afae2c01d518927e3ca96ec977d000287b0a6bc62fc42a7e317",
    }
    actual_identity = {
        "k8_net": digest(M.BASE / "k8" / M.NET),
        "k8_sdc": digest(M.BASE / "k8" / M.SDC),
        "k1x8_net": digest(M.BASE / "k1x8" / M.NET),
        "k1x8_sdc": digest(M.BASE / "k1x8" / M.SDC),
    }
    require(actual_identity == expected_identity, "mapped M872/M903 identity drift")
    checks.append("mapped_k8_k1x8_netlist_sdc_sha_exact")

    # Positive rejection controls.
    k8_text = M.FILELISTS["k8"].read_text()
    for label, mutation in (
        ("diagnostic_k1", k8_text + "+define+M979_AXIS_K1\n"),
        ("opposite_axis", k8_text + "+define+M979_AXIS_K1X8\n"),
        ("named_old_memory", k8_text + "tb_m349/m349_fc2_scalar_bank_memory_model.sv\n"),
    ):
        try:
            M.validate_filelist(mutation, "k8")
        except RuntimeError:
            checks.append(label + "_rejected")
        else:
            raise AssertionError(label + " mutation unexpectedly accepted")

    # Five exact workload/cycle anchors on both axes, plus wrong-cycle rejection.
    with tempfile.TemporaryDirectory(prefix="m1333_saif_") as td:
        for axis in ("k8", "k1x8"):
            for case_id, cycles in enumerate(M.AXES[axis]["cycles"]):
                endpoint = 0 if case_id == 4 else 4
                path = write_tmp(td, f"{axis}_{case_id}.saif", saif(cycles * 3, endpoint))
                M.validate_saif(path, axis, case_id, cycles)
                try:
                    M.validate_saif(path, axis, case_id, cycles + 1)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("wrong cycle accepted")
        checks.append("ten_axis_case_cycle_anchors_and_wrong_cycle_rejection")

        # FN1: case 4 is contractually zero-endpoint, but nonzero traffic passes.
        bad_zero = write_tmp(td, "zero_with_endpoint.saif", saif(42, endpoint_tc=9))
        try:
            M.validate_saif(bad_zero, "k1x8", 4, 14)
        except RuntimeError:
            pass
        else:
            false_negatives.append("FN1_zero_event_case4_nonzero_endpoint_saif_accepted")

        # FN2: activity may live outside core.dut; the checker only searches names globally.
        bad_scope = write_tmp(td, "activity_outside_dut.saif", saif(153, 5, False))
        try:
            M.validate_saif(bad_scope, "k8", 0, 51)
        except RuntimeError:
            pass
        else:
            false_negatives.append("FN2_non_dut_activity_satisfies_dut_only_saif_cones")

    leaf = "m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.v"
    official = next(line for line in k8_text.splitlines() if line.endswith(leaf))
    forged = k8_text.replace(official, "/tmp/forged/k8/netlist/" + leaf)
    try:
        M.validate_filelist(forged, "k8")
    except RuntimeError:
        pass
    else:
        false_negatives.append("FN3_filelist_netlist_path_not_bound_to_sha_checked_object")

    alternate_old = k8_text + "/tmp/legacy/m349_fc2_scalar_bank_memory_model.sv\n"
    try:
        M.validate_filelist(alternate_old, "k8")
    except RuntimeError:
        pass
    else:
        false_negatives.append("FN4_alternate_old_memory_fallback_path_accepted")

    with tempfile.TemporaryDirectory(prefix="m1333_static_") as td:
        mem_text = M.MEM.read_text()
        mem_mut = mem_text.replace("epoch_q[slot] <= '0;", "// epoch_q[slot] <= '0;")
        require(mem_mut != mem_text, "memory reset mutation did not apply")
        if static_mutation_accepted(td, "MEM", M.MEM,
                "dc_handoff/tb/m1332_c2_production_activity_reset_safe_memory_model.sv",
                mem_mut):
            false_negatives.append("FN5_commented_out_memory_payload_reset_accepted")

        sva_text = M.SVA.read_text()
        sva_mut = sva_text.replace("cp_source: cover property (raw_accept);",
                                   "// cp_source: cover property (raw_accept);")
        require(sva_mut != sva_text, "cover mutation did not apply")
        if static_mutation_accepted(td, "SVA", M.SVA,
                "dc_handoff/tb/m1332_c2_production_activity_assertions.sv",
                sva_mut):
            false_negatives.append("FN6_commented_out_runtime_cover_accepted")

        ucli_text = M.UCLI.read_text()
        scope = "tb_m1332_c2_headline_mapped_production_activity.core.dut"
        ucli_mut = ucli_text.replace(
            "power " + scope,
            "# power " + scope + "\npower tb_m1332_c2_headline_mapped_production_activity.core")
        ucli_mut = ucli_mut.replace(
            "power -report $::env(M1332_SAIF_FILE) 1e-9 " + scope,
            "# power -report $::env(M1332_SAIF_FILE) 1e-9 " + scope
            + "\npower -report $::env(M1332_SAIF_FILE) 1e-9 tb_m1332_c2_headline_mapped_production_activity.core")
        require(ucli_mut != ucli_text, "scope mutation did not apply")
        if static_mutation_accepted(td, "UCLI", M.UCLI,
                "dc_handoff/scripts/m1332_c2_headline_mapped_production_activity.ucli.tcl",
                ucli_mut):
            false_negatives.append("FN7_commented_dut_scope_plus_live_core_scope_accepted")

    # Source-level protocol attack: a forced accept with valid 0/X and known slot
    # reaches the state-indexing branch.  Sticky faulting does not satisfy the
    # stronger contract that invalid/unknown requests never index state.
    mem_text = M.MEM.read_text()
    request_branch = mem_text[mem_text.index("if (mem_req_accept === 1'b1) begin"):]
    branch_head = request_branch[:request_branch.index("end\n", request_branch.index("if (!request_payload_known)")) + 4]
    if "mem_req_valid === 1'b1" not in branch_head and "pending_q[mem_req_slot]" in request_branch[:1800]:
        false_negatives.append("FN8_accept_with_invalid_or_unknown_valid_can_index_request_state")

    # Source contract says ten files, but there is no batch inventory/uniqueness gate.
    checker_text = CHECKER.read_text()
    if "headline_file_count" not in checker_text and "glob(" not in checker_text:
        false_negatives.append("FN9_no_executable_ten_file_inventory_or_axis_case_uniqueness_gate")

    # Payload/stability assertions have no fail-closed action and payload X is not
    # folded into the sticky unknown counter.  A future runner could only close
    # this by separately parsing assertion diagnostics, which M1332 does not author.
    sva_text = M.SVA.read_text()
    if "ap_result_payload_known:" in sva_text and "else $fatal" not in sva_text:
        false_negatives.append("FN10_runtime_payload_and_stability_sva_have_no_explicit_fatal_action")

    require(false_negatives, "hammer expected at least one false negative")
    out = {
        "schema": "m1333_m1332_c2_headline_mapped_production_activity_source_blind_hammer_r1",
        "status": "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED",
        "score": 62,
        "eda_executed": False,
        "author_baseline_passed": True,
        "positive_checks": checks,
        "false_negative_count": len(false_negatives),
        "false_negatives": false_negatives,
        "mapped_identity": actual_identity,
        "docs359_sha256": digest(HW / "docs/359_DATE终局冻结_20260813.md"),
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
