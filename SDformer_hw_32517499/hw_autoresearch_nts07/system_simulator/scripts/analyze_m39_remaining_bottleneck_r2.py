#!/usr/bin/env python3
"""Build M39-r2 fail-closed conditional DSE with recursive Synopsys evidence."""

import argparse
import csv
import hashlib
import importlib.util
import json
import re
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
DEFAULT_CONTRACT = (
    HW_ROOT / "contracts/m39_remaining_bottleneck_input_contract_r2_20260822.json"
)

CONTRACT_TOP_KEYS = {
    "schema", "identity", "supersedes", "claim_boundary", "reanchor_notice",
    "inputs", "frozen_dse_rules", "resource_and_admission_gates",
    "external_comparison_boundary",
}
INPUT_KEYS = {
    "m22_summary", "m25_cycle_ledger", "m26_factor_lower_bound",
    "m30_system_dse", "m32_threshold_carry", "m33_receipt", "m35_receipt",
    "m38_contract", "m38_result", "h67_dual_line_contract",
    "h67_operator_transactions",
}
M38_RESULT_TOP_KEYS = {
    "abstract_integrated_cycle_audit", "admission",
    "canonical_crc_and_fragment_protocol_audit", "claim_boundary",
    "conditional_theory", "configuration_bit_ledger", "identity",
    "rank3_q24_threshold_audit", "recursive_anchor_audit",
    "scalar_ternary_audit", "schema", "status", "unmeasured_nonzero_costs",
}
EXPECTED_CLAIM_BOUNDARY = (
    "BLOCKED_BY_STALE_M38_R2; EXPLORATORY_ONLY; CONDITIONAL_DSE_EVIDENCE_ONLY: "
    "frozen H67 profile100 compute-ledger decomposition and conditional same-resource "
    "arithmetic DSE only. The hashed M38-r2 artifact is auditable but its current "
    "fail-closed build is not reproducible after M31/M37 live-source drift, so it is "
    "not a current recursive admission. Local5 ep44 attention is missing and nonzero, "
    "so Local5 full-system cycles and speedup are unknown. M38 integrated RTL/VCS/DC/PPA, "
    "M33/M35 Local/Motion integration, address-timed memory, trained accuracy, power, "
    "energy, and headline performance remain unadmitted. This r2 is expected to be "
    "mechanically reanchored after M38-r3; it is not a final freeze or headline artifact."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(raw):
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def exact_keys(payload, expected, label):
    require(isinstance(payload, dict), "{} must be an object".format(label))
    require(set(payload) == set(expected), "{} population drift".format(label))


def fraction_json(value):
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def read_manifest(path):
    rows = []
    seen = set()
    for line_number, line in enumerate(
            Path(path).read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None,
                "malformed manifest line {} in {}".format(line_number, path))
        digest, raw = match.groups()
        require(raw not in seen, "duplicate manifest path in {}".format(path))
        seen.add(raw)
        rows.append((digest, raw))
    require(rows, "empty manifest {}".format(path))
    return rows


def verify_manifest(path, expected_sha, expected_count, base,
                    self_contained=False):
    path = Path(path)
    require(path.is_file(), "missing manifest {}".format(path))
    require(sha256(path) == expected_sha, "manifest hash drift: {}".format(path))
    rows = read_manifest(path)
    require(len(rows) == expected_count,
            "manifest population drift: {}".format(path))
    base = Path(base).resolve()
    verified = {}
    for digest, raw in rows:
        candidate = Path(raw)
        if self_contained:
            require(not candidate.is_absolute(),
                    "self-contained manifest contains absolute path")
            target = (base / candidate).resolve()
            try:
                target.relative_to(base)
            except ValueError:
                raise ValueError("self-contained manifest escapes snapshot")
        else:
            target = candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
        require(target.is_file(), "manifest member missing: {}".format(raw))
        require(sha256(target) == digest, "manifest member hash drift: {}".format(raw))
        verified[raw] = digest
    return verified


def count_properties(path):
    text = Path(path).read_text(encoding="utf-8")
    return (len(re.findall(r"\bassert\s+property\b", text)),
            len(re.findall(r"\bcover\s+property\b", text)))


def cover_matches(log_text):
    return [int(value) for value in re.findall(r"\d+ attempts, (\d+) match", log_text)]


def require_vcs_log(log_text, label, pass_fragments, expected_covers):
    require("SIMULATOR=Synopsys VCS" in log_text, "{} simulator marker missing".format(label))
    require("ASSERTIONS=enabled" in log_text, "{} assertion marker missing".format(label))
    for fragment in pass_fragments:
        require(fragment in log_text, "{} pass marker drift: {}".format(label, fragment))
    matches = cover_matches(log_text)
    require(matches == expected_covers, "{} cover population/matches drift".format(label))
    require(all(value > 0 for value in matches), "{} zero-match cover".format(label))
    require(not re.search(r"(?im)^.*(?:assertion failed|fatal|error:).*$", log_text),
            "{} failure signature present".format(label))


def parse_admission(text):
    rows = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            rows[key.strip()] = value.strip()
    return rows


def verify_receipt_pair(path_hash_pair, label):
    require(isinstance(path_hash_pair, list) and len(path_hash_pair) == 2,
            "{} identity must be [path, sha256]".format(label))
    path = resolve(path_hash_pair[0])
    require(path.is_file(), "{} identity file missing".format(label))
    require(sha256(path) == path_hash_pair[1], "{} identity hash drift".format(label))
    return path


def verify_m33_receipt(receipt):
    exact_keys(receipt, {
        "schema", "status", "date", "math_identity", "source_identity", "vcs_r2",
        "dc_sta_flat_r2", "formality_flat_r2", "claim_boundary",
        "paper_ppa_ready", "headline_admitted",
    }, "M33 receipt")
    require(receipt["schema"] == "m33_output_receipt_v1", "M33 receipt schema drift")
    require(receipt["status"] ==
            "PASS_FLAT_R2_UQ0P24_VCS_DC_STA_FORMALITY_NO_SYSTEM_OR_PAPER_PPA_CLAIM",
            "M33 receipt status drift")
    require(receipt["paper_ppa_ready"] is False and receipt["headline_admitted"] is False,
            "M33 receipt claim opened")
    exact_keys(receipt["math_identity"], {"contract", "result"}, "M33 math identity")
    math_contract_path = verify_receipt_pair(receipt["math_identity"]["contract"], "M33 math contract")
    math_result_path = verify_receipt_pair(receipt["math_identity"]["result"], "M33 math result")
    math_contract = load_json(math_contract_path)
    math_result = load_json(math_result_path)
    require(math_contract["schema"] == "m33_checkpoint_uq0p24_input_contract_v2",
            "M33 math contract schema drift")
    require(math_result["schema"] == "m33_checkpoint_uq0p24_cross_product_audit_v2"
            and math_result["status"] ==
            "PASS_EXACT_UQ0P24_AND_SIGNED_DIGIT_CROSS_PRODUCT_IDENTITY",
            "M33 math result drift")
    require(math_result["identity"]["input_contract_sha256"] ==
            receipt["math_identity"]["contract"][1], "M33 recursive math anchor drift")

    source_keys = {
        "multiplier_pool", "rtl", "assertions", "testbench", "vcs_filelist", "vcs_runner"
    }
    exact_keys(receipt["source_identity"], source_keys, "M33 source identity")
    source_paths = {}
    for name in sorted(source_keys):
        source_paths[name] = verify_receipt_pair(receipt["source_identity"][name],
                                                 "M33 {}".format(name))
    require(count_properties(source_paths["assertions"]) == (8, 4),
            "M33 SVA source population drift")

    vcs = receipt["vcs_r2"]
    vcs_dir = Path(vcs["directory"])
    input_rows = verify_manifest(vcs_dir / "input_sha256.txt",
                                 vcs["input_ledger_sha256"], 6, HW_ROOT)
    output_rows = verify_manifest(vcs_dir / "output_sha256.txt",
                                  vcs["output_ledger_sha256"], 3, vcs_dir)
    expected_input_hashes = {pair[1] for pair in receipt["source_identity"].values()}
    require(set(input_rows.values()) == expected_input_hashes,
            "M33 VCS input population does not equal source receipt")
    compile_path = vcs_dir / "compile.log"
    sim_path = vcs_dir / "sim.log"
    vector_path = vcs_dir / "vectors.txt"
    require(output_rows.get(str(compile_path)) == vcs["compile_log_sha256"],
            "M33 compile output ledger drift")
    require(output_rows.get(str(sim_path)) == vcs["sim_log_sha256"],
            "M33 sim output ledger drift")
    require(output_rows.get(str(vector_path)) == vcs["vector_sha256"],
            "M33 vector output ledger drift")
    compile_text = compile_path.read_text(encoding="utf-8", errors="replace")
    require("Chronologic VCS" in compile_text and "Error-" not in compile_text,
            "M33 compile log semantic drift")
    sim_text = sim_path.read_text(encoding="utf-8", errors="replace")
    require_vcs_log(sim_text, "M33", [
        "M33_UQ_SVA_BOUND=1", "M33_UQ_PASS packets=2048",
        "valid_scalar_products=4608", "digit_reconstruction_checks=8192",
        "negative_uq_digits=3029", "consecutive_full_rate=255",
    ], [112, 2, 193, 364])

    dc = receipt["dc_sta_flat_r2"]
    dc_dir = Path(dc["directory"])
    verify_manifest(dc_dir / "evidence.sha256", dc["evidence_ledger_sha256"], 31, dc_dir)
    verify_manifest(dc_dir / "sealed_dc_evidence.sha256",
                    dc["sealed_dc_evidence_sha256"], 33, dc_dir)
    admission_path = dc_dir / "admission.txt"
    require(sha256(admission_path) == dc["admission_sha256"], "M33 DC admission hash drift")
    admission = parse_admission(admission_path.read_text(encoding="utf-8"))
    require(admission.get("status") == "PASS_EXPLORATORY_FLAT_FAIR_AREA_DC"
            and admission.get("timing_status") == "MET"
            and admission.get("paper_ppa_ready") == "false",
            "M33 DC admission semantic drift")
    area_text = (dc_dir / "reports/area.rpt").read_text(encoding="utf-8")
    require("Total cell area:                 12997.403898" in area_text,
            "M33 DC area semantic drift")
    require(dc["setup_wns_ns"] >= 0 and dc["hold_wns_ns"] >= 0
            and dc["macro_count"] == 0 and dc["wire_area"] == "UNDEFINED",
            "M33 DC receipt qualification drift")

    fm = receipt["formality_flat_r2"]
    snapshot = fm["self_contained_snapshot"]
    require(snapshot["authority"] == "SELF_CONTAINED_SNAPSHOT_ONLY",
            "M33 Formality authority drift")
    outer = dc_dir / snapshot["evidence_ledger"]
    members = verify_manifest(outer, snapshot["evidence_ledger_sha256"], 21, dc_dir,
                              self_contained=True)
    prefix = snapshot["directory"] + "/outputs/"
    admission_name = prefix + "formality_admission_m33_flat_r2_fm1_20260822.txt"
    log_name = prefix + "formality_m33_flat_r2_fm1_20260822.log"
    manifest_name = prefix + "formality_run_manifest.json"
    require(members.get(log_name) == fm["log_sha256"], "M33 Formality log anchor drift")
    require(members.get(manifest_name) == fm["manifest_sha256"],
            "M33 Formality manifest anchor drift")
    fm_admission = parse_admission((dc_dir / admission_name).read_text(encoding="utf-8"))
    require(fm_admission == {
        "status": "PASS_RTL_TO_MAPPED_NETLIST_FORMALITY",
        "attempt_tag": "m33_flat_r2_fm1_20260822",
        "passing_compare_points": "655", "failing_compare_points": "0",
        "unmatched_compare_points": "0",
    }, "M33 Formality compare-point drift")
    require(fm["passing_compare_points"] == 655 and fm["failing_compare_points"] == 0
            and fm["unmatched_compare_points"] == 0 and fm["attempt_exit_status"] == 0,
            "M33 Formality receipt drift")
    return {"receipt_schema": receipt["schema"], "vcs_input_files": 6,
            "vcs_output_files": 3, "dc_live_files": 31, "dc_sealed_files": 33,
            "formality_snapshot_files": 21, "formality_compare_points": [655, 0, 0],
            "formality_authority": "SELF_CONTAINED_SNAPSHOT_ONLY",
            "standalone_area_um2": dc["cell_area_um2"]}


def verify_m35_receipt(receipt):
    exact_keys(receipt, {
        "schema", "status", "date", "supersedes", "math_identity", "source_identity",
        "vcs_r6", "dc_sta_r7", "formality_r7",
        "standalone_fair_flat_m33_comparison_only", "claim_boundary",
        "paper_ppa_ready", "independent_r7_review_required", "headline_admitted",
    }, "M35 receipt")
    require(receipt["schema"] == "m35_output_receipt_v2", "M35 receipt schema drift")
    require(receipt["status"] ==
            "PASS_STANDALONE_COMPLEMENT_CSD8_VCS_DC_R7_STA_FORMALITY_R7_NO_SYSTEM_OR_PAPER_PPA_CLAIM",
            "M35 receipt status drift")
    require(receipt["paper_ppa_ready"] is False and receipt["headline_admitted"] is False,
            "M35 receipt claim opened")
    math_contract_path = verify_receipt_pair(receipt["math_identity"]["contract"],
                                             "M35 math contract")
    math_result_path = verify_receipt_pair(receipt["math_identity"]["result"], "M35 math result")
    math_result = load_json(math_result_path)
    require(load_json(math_contract_path)["schema"] == "m35_complement_csd_input_contract_v3",
            "M35 math contract schema drift")
    require(math_result["status"] ==
            "PASS_TEN_CHECKPOINT_THRESHOLDS_EXACT_UP_TO_FOUR_TERM_COMPLEMENT_CSD_SIGNED42",
            "M35 math result status drift")
    require(len(math_result["thresholds"]) == 10
            and min(row["delta"] for row in math_result["thresholds"]) == 1
            and max(row["delta"] for row in math_result["thresholds"]) == 588
            and max(row["csd_nonzero_terms"] for row in math_result["thresholds"]) == 4,
            "M35 threshold population drift")
    source_paths = {}
    exact_keys(receipt["source_identity"], {"rtl", "assertions", "testbench"},
               "M35 source identity")
    for name, pair in receipt["source_identity"].items():
        source_paths[name] = verify_receipt_pair(pair, "M35 {}".format(name))
    require(count_properties(source_paths["assertions"]) == (9, 4),
            "M35 SVA source population drift")

    vcs = receipt["vcs_r6"]
    vcs_dir = Path(vcs["directory"])
    verify_manifest(vcs_dir / "input_sha256.txt", vcs["input_ledger_sha256"], 7, HW_ROOT)
    outputs = verify_manifest(vcs_dir / "output_sha256.txt", vcs["output_ledger_sha256"],
                              3, vcs_dir)
    compile_path = vcs_dir / "compile.log"
    sim_path = vcs_dir / "sim.log"
    vector_path = vcs_dir / "vectors.txt"
    require(outputs.get(str(compile_path)) == vcs["compile_log_sha256"]
            and outputs.get(str(sim_path)) == vcs["sim_log_sha256"]
            and outputs.get(str(vector_path)) == vcs["vector_sha256"],
            "M35 VCS output ledger drift")
    require("Chronologic VCS" in compile_path.read_text(encoding="utf-8", errors="replace"),
            "M35 compile log semantic drift")
    sim_text = sim_path.read_text(encoding="utf-8", errors="replace")
    require_vcs_log(sim_text, "M35", [
        "M35_SVA_BOUND=1", "M35_PASS packets=5120", "valid_products=23680",
        "config_loads=10", "config_releases=10", "consecutive_full_rate=630",
        "illegal_accepts=2", "illegal_rejections=2",
    ], [10, 859, 10, 2])

    dc = receipt["dc_sta_r7"]
    dc_dir = Path(dc["directory"])
    verify_manifest(dc_dir / "evidence.sha256", dc["evidence_ledger_sha256"], 30, dc_dir)
    verify_manifest(dc_dir / "sealed_dc_evidence.sha256",
                    dc["sealed_dc_evidence_sha256"], 32, dc_dir)
    admission_path = dc_dir / "admission.txt"
    require(sha256(admission_path) == dc["admission_sha256"], "M35 DC admission hash drift")
    admission = parse_admission(admission_path.read_text(encoding="utf-8"))
    require(admission.get("status") == "PASS_STANDALONE_COMPLEMENT_CSD8_DC"
            and admission.get("timing_status") == "MET"
            and admission.get("integer_multiplier_count") == "0"
            and admission.get("paper_ppa_ready") == "false",
            "M35 DC admission semantic drift")
    require(dc["setup_wns_ns"] >= 0 and dc["hold_wns_ns"] >= 0
            and dc["integer_multiplier_operators"] == 0
            and dc["macro_count"] == 0 and dc["wire_area"] == "UNDEFINED",
            "M35 DC receipt qualification drift")
    require("Total cell area:                 19633.571938" in
            (dc_dir / "reports/area.rpt").read_text(encoding="utf-8"),
            "M35 DC area semantic drift")

    fm = receipt["formality_r7"]
    snapshot = fm["self_contained_snapshot"]
    directory = snapshot["directory"]
    outer_name = directory.replace("sealed_formality_", "sealed_formality_evidence_", 1) + ".sha256"
    members = verify_manifest(dc_dir / outer_name, snapshot["evidence_ledger_sha256"],
                              20, dc_dir, self_contained=True)
    prefix = directory + "/outputs/"
    admission_name = prefix + "formality_admission_m35_r7_fm1_20260822.txt"
    log_name = prefix + "formality_m35_r7_fm1_20260822.log"
    manifest_name = prefix + "formality_run_manifest.json"
    require(members.get(log_name) == fm["log_sha256"], "M35 Formality log anchor drift")
    require(members.get(manifest_name) == fm["manifest_sha256"],
            "M35 Formality manifest anchor drift")
    fm_admission = parse_admission((dc_dir / admission_name).read_text(encoding="utf-8"))
    require(fm_admission == {
        "status": "PASS_RTL_TO_MAPPED_NETLIST_FORMALITY",
        "attempt_tag": "m35_r7_fm1_20260822",
        "passing_compare_points": "2333", "failing_compare_points": "0",
        "unmatched_compare_points": "0",
    }, "M35 Formality compare-point drift")
    require(fm["passing_compare_points"] == 2333 and fm["failing_compare_points"] == 0
            and fm["unmatched_compare_points"] == 0 and fm["attempt_exit_status"] == 0,
            "M35 Formality receipt drift")
    # Deliberately do not verify live Formality paths: the sealed snapshot is the authority.
    return {"receipt_schema": receipt["schema"], "vcs_input_files": 7,
            "vcs_output_files": 3, "dc_live_files": 30, "dc_sealed_files": 32,
            "formality_snapshot_files": 20, "formality_compare_points": [2333, 0, 0],
            "formality_authority":
            "SELF_CONTAINED_SNAPSHOT_ONLY_LIVE_WRAPPER_DRIFT_IGNORED",
            "standalone_area_um2": dc["cell_area_um2"]}


def verify_m38(contract_payload, result):
    exact_keys(contract_payload, {
        "schema", "identity", "claim_boundary", "inputs", "frozen_architecture",
        "canonical_configuration_frame", "abstract_cycle_protocol", "theory_rules",
    }, "M38-r2 contract")
    require(contract_payload["schema"] == "m38_rst_math_input_contract_v2"
            and contract_payload["identity"] ==
            "M31_r3_M37_r7_rank3_M38_RST_milestone2_fail_closed_math_crc_and_cycle_model_only",
            "M38-r2 contract identity drift")
    exact_keys(result, M38_RESULT_TOP_KEYS, "M38-r2 result")
    require(result["schema"] == "m38_rst_math_crc_and_abstract_cycle_audit_v2"
            and result["status"] ==
            "PASS_M38_RST_RECURSIVE_ANCHOR_MATH_CRC_AND_ABSTRACT_CYCLE_ONLY",
            "M38-r2 result identity drift")
    require(result["identity"]["contract_sha256"] == sha256(DEFAULT_CONTRACT.parent /
            "m38_rst_math_input_contract_r2_20260822.json"),
            "M38-r2 result-to-contract anchor drift")
    require(result["conditional_theory"] == {
        "finite_n_ratio": "10*N/(5+5*N)", "parallel_steady_ii": 5,
        "serialized_steady_ii": 10, "steady_t10_kernel_throughput_limit": 2.0,
        "system_speedup_admitted": False,
    }, "M38-r2 conditional theory drift")
    anchors = result["recursive_anchor_audit"]
    exact_keys(anchors, {"m31_r3", "m37_r7"}, "M38-r2 recursive anchors")
    require(anchors["m31_r3"]["receipt_sha256"] ==
            "3785a36272845bb5ea240d9aa7eed5bdc934b6cf453ebf2a90f5a16131109577"
            and anchors["m31_r3"]["receipt_schema"] == "m31_output_receipt_v3"
            and anchors["m31_r3"]["receipt_status"] ==
            "PASS_UNIFIED_T10_T2_EXACT_FIXED_POINT_SINGLE_SOURCE_MUL96_HIERARCHY_VCS_NO_PPA_OR_SYSTEM_CLAIM",
            "M38-r2 M31 recursive anchor drift")
    require(anchors["m37_r7"]["receipt_sha256"] ==
            "441531803e3f193bd1f348bacf16291bfab18db4903320549dd6f67d17b43344"
            and anchors["m37_r7"]["receipt_schema"] == "m37_output_receipt_v2"
            and anchors["m37_r7"]["receipt_status"] ==
            "PASS_STANDALONE_T10_CANONICAL_NAF_CSD_RECONSTRUCTION_EXACT_FIXED_POINT_VCS_NO_PPA_OR_SYSTEM_CLAIM",
            "M38-r2 M37 recursive anchor drift")
    admission = result["admission"]
    require(admission["recursive_anchor_identity_admitted"] is True
            and admission["abstract_integrated_cycle_safety_and_liveness_admitted"] is True
            and admission["integrated_rtl_admitted"] is False
            and admission["integrated_rtl_vcs_admitted"] is False
            and admission["dc_sta_formality_admitted"] is False
            and admission["system_speedup_admitted"] is False
            and admission["headline_admitted"] is False,
            "M38-r2 admission drift")
    analyzer_path = HW_ROOT / "system_simulator/scripts/analyze_m38_rst_math_crc_and_cycle_r2.py"
    require(sha256(analyzer_path) == result["identity"]["analyzer_sha256"],
            "M38-r2 analyzer identity drift")
    spec = importlib.util.spec_from_file_location("m38_r2_current_rebuild_probe", str(analyzer_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rebuild_error = None
    try:
        module.build(DEFAULT_CONTRACT.parent / "m38_rst_math_input_contract_r2_20260822.json")
    except ValueError as error:
        rebuild_error = str(error)
    require(rebuild_error == "receipt live source drift for unified_core_rtl",
            "M38-r2 current rebuild state changed; mechanically reanchor M39")
    return {"schema": result["schema"], "status": result["status"],
            "m31_receipt_sha256": anchors["m31_r3"]["receipt_sha256"],
            "m37_receipt_sha256": anchors["m37_r7"]["receipt_sha256"],
            "conditional_t10_ii": 5, "archival_artifact_hash_verified": True,
            "current_fail_closed_rebuild_admitted": False,
            "current_rebuild_error": rebuild_error,
            "recursive_current_dependency_admitted": False,
            "system_speedup_admitted": False}


def load_contract(path):
    path = Path(path)
    contract = load_json(path)
    exact_keys(contract, CONTRACT_TOP_KEYS, "M39-r2 contract")
    require(contract["schema"] == "m39_remaining_bottleneck_input_contract_v2",
            "M39-r2 schema drift")
    require(contract["identity"] ==
            "M39_r2_fail_closed_remaining_system_bottleneck_and_threshold_carry_sidecar_DSE",
            "M39-r2 identity drift")
    require(contract["claim_boundary"] == EXPECTED_CLAIM_BOUNDARY,
            "M39-r2 claim boundary drift")
    require(contract["supersedes"] == {
        "path": "hw_autoresearch_nts07/contracts/m39_remaining_bottleneck_input_contract_r1_20260822.json",
        "sha256": "e185ba742e0ae520ad1177168d8807dfee9671532b0b044a9462207d8cdaf9a8",
        "state": "NO_GO_DRAFT_SUPERSEDED_DO_NOT_CITE",
        "reason": "r1 did not recursively authenticate M33/M35 Synopsys evidence and anchored superseded M38/M35 receipts",
    }, "M39-r2 supersedes drift")
    require(contract["reanchor_notice"] == {
        "state": "EXPECTED_MECHANICAL_REANCHOR_AFTER_M38_R3",
        "reason": "M38-r2 currently closes M31-r3/M37-r7; later M31-r4/M37-r8 require an M38-r3 recursive anchor",
        "blocks_current_evidence_gate_repair": False, "final_freeze": False,
    }, "M39-r2 reanchor notice drift")
    exact_keys(contract["inputs"], INPUT_KEYS, "M39-r2 inputs")
    payloads, hashes, paths = {}, {}, {}
    for name, spec in sorted(contract["inputs"].items()):
        exact_keys(spec, {"path", "sha256"}, "M39-r2 input {}".format(name))
        source = resolve(spec["path"])
        require(source.is_file(), "M39-r2 input missing: {}".format(name))
        actual = sha256(source)
        require(actual == spec["sha256"], "M39-r2 input hash drift: {}".format(name))
        text = source.read_text(encoding="utf-8")
        payloads[name] = json.loads(text) if source.suffix == ".json" else text
        hashes[name], paths[name] = actual, str(source)
    return contract, payloads, hashes, paths


def validate_system_inputs(contract, payloads, hashes):
    rules = contract["frozen_dse_rules"]
    require(rules == {
        "fixed_compute_cycles": 620868243,
        "selected_m30_candidate": "dual256b_independent_output_packed24",
        "selected_m30_local_cycles": 305047198,
        "selected_m30_motion_cycles": 303376924,
        "m30_t10_cycles": 73183500, "m38_conditional_t10_cycles": 36591750,
        "m38_conditional_t10_ii": 5, "m33_outputs_per_cycle": 4,
        "m33_products_per_output": 20, "m33_multiplier_lanes_used": 80,
        "m35_outputs_per_cycle": 8, "m35_integer_multipliers": 0,
        "consumer_population_cycles": 105888197,
        "consumer_outputs_per_sample": 30456000,
        "bottleneck_population_cycles": 79630957,
        "bottleneck_outputs_per_sample": 9216000,
        "m4_profiled_population_cycles": 327131854,
        "minimum_saved_cycles_for_candidate": 50000000,
    }, "M39-r2 frozen DSE rule drift")
    require(contract["resource_and_admission_gates"] == {
        "clock_period_ns": 3.0, "signed_int8_multiplier_lanes": 96,
        "sram_preferred_kib": 240, "sram_hard_cap_kib": 408,
        "sram_row_bytes": 96, "sram_banks": 24,
        "sram_read_ports_per_bank": 1, "sram_write_ports_per_bank": 1,
        "minimum_system_speedup": {"numerator": 27, "denominator": 10},
        "stretch_system_speedup": {"numerator": 3, "denominator": 1},
        "maximum_integrated_area_delta_fraction": {"numerator": 15, "denominator": 100},
        "accuracy_primary_gate": "BIT_EXACT_TO_FROZEN_INTEGER_REFERENCE",
        "accuracy_fallback_delta_aee_max": {"numerator": 2, "denominator": 100},
        "energy_gate": "FUTURE_INTEGRATED_SAME_TRACE_PTPX_PLUS_MACRO_ENERGY_NOT_WORSE_THAN_FUTURE_M38_BASELINE",
        "m33_flat_r2_formality": "PASS_655_0_0_SELF_CONTAINED_SNAPSHOT",
        "m35_r7_formality": "PASS_2333_0_0_SELF_CONTAINED_SNAPSHOT",
    }, "M39-r2 resource/admission gate drift")
    require(contract["external_comparison_boundary"] == {
        "prosperity_real_domain_source": "M32_ONLY_EXACT_REAL_DOMAIN_NUMBERS",
        "prosperity_fixed_point_and_accuracy": "UNADMITTED",
        "prosperity_official_repository_commit": "6ee1c6f1cb419fcf942f2eda63db84ca28248f4b",
        "prosperity_repository_retrieved_date": "2026-08-22",
        "prosperity_repository_retrieval_method":
        "git ls-remote https://github.com/dubcyfor3/Prosperity HEAD",
        "prosperity_repository_file_sha256": "NOT_MEASURED_DO_NOT_INFER",
        "phi_like_adapter": "UNIMPLEMENTED_UNADMITTED",
    }, "M39-r2 external comparison boundary drift")

    m22 = payloads["m22_summary"]
    require(m22["status"] ==
            "PASS_FROZEN_INPUT_PARTIAL_TRANSACTION_LEDGER_NOT_DRAMSIM_OR_SPEEDUP",
            "M39-r2 M22 status drift")
    local_identity = m22["identities"]["local_ep44"]
    require(local_identity["attention_execution_records"] == 0
            and local_identity["attention_coverage_status"] ==
            "MISSING_FROM_EXECUTION_TRACE_NOT_ZERO_COST",
            "M39-r2 Local5 attention fail-close drift")
    require(m22["identities"]["h67_ep35"]["attention_execution_records"] == 120,
            "M39-r2 H67 attention population drift")

    m25 = payloads["m25_cycle_ledger"]
    require(m25["status"] == "PASS_FROZEN_C4_TILING_AND_CYCLE_ENVELOPE_HEADLINE_NO_GO",
            "M39-r2 M25 status drift")
    require(m25["attention_completeness"]["Local5"]["speedup"] == "UNKNOWN"
            and m25["attention_completeness"]["Local5"]["minimum_missing_module_calls"] == 120,
            "M39-r2 M25 Local5 attention drift")
    local = m25["compute_envelopes"]["local"]["10"]
    motion = m25["compute_envelopes"]["hybrid"]["10"]
    require(local["effective_m4_speed"] == 5.995180731292359
            and motion["effective_m4_speed"] == 6.203518497363532,
            "M39-r2 M25 effective speed drift")

    m26 = payloads["m26_factor_lower_bound"]
    m30 = payloads["m30_system_dse"]
    require(m26["schema"] == "m26_atlif_factor_arithmetic_lower_bound_v2"
            and m30["identity"]["m26_sha256"] == hashes["m26_factor_lower_bound"],
            "M39-r2 M26/M30 recursive identity drift")
    selected = {row["name"]: row for row in m30["port_candidates"]}[
        rules["selected_m30_candidate"]]
    require(selected["local_cycles"] == 305047198 and selected["motion_cycles"] == 303376924
            and selected["t10_cycles"] == 73183500,
            "M39-r2 M30 selected candidate drift")
    candidates = {row["name"]: row for row in m30["port_candidates"]}
    require(candidates["384b_independent_output_packed24"]["local_cycles"] == 305047222
            and candidates["384b_independent_output_packed24"]["motion_cycles"] == 303376948,
            "M39-r2 M30 384b comparison drift")

    m32 = payloads["m32_threshold_carry"]
    require(m32["status"] ==
            "PASS_H67_EP35_S10_EXACT_RUNTIME_DATAFLOW_REAL_DOMAIN_SEMANTIC_ADMISSION_ONLY",
            "M39-r2 M32 status drift")
    require(m32["candidate_census"]["semantically_admitted_operators"] == 10
            and m32["candidate_census"]["semantically_admitted_cycles_candidate_population"]
            == 105888197
            and m32["candidate_census"]["semantically_admitted_outputs_per_sample"]
            == 30456000,
            "M39-r2 M32 census drift")
    require(m32["admission"]["fixed_point_admitted"] is False
            and m32["admission"]["system_cycle_admitted"] is False,
            "M39-r2 M32 claim opened")
    balanced = {row["line"]: row
                for row in m32["control_charged_cycle_sensitivity"]["rows"]
                if row["variant"] == "balanced_radix20_exact_product"}
    expected = {
        "local": (17662220, 7614000, 1974013),
        "motion": (17069055, 7614000, 2026532),
    }
    for line, values in expected.items():
        row = balanced[line]
        require((row["event_accumulation_cycles_borrowed"],
                 row["late_scale_cycles_arithmetic"],
                 row["proportional_frontend_control_cycles"]) == values,
                "M39-r2 M32 {} row drift".format(line))

    dual = payloads["h67_dual_line_contract"]
    categories = dual["coverage"]["categories"]
    category_expected = {
        "bottleneck": (79630957, 0), "patch_embed": (199420620, 172321077),
        "ffn_expand": (118370114, 100895624),
        "downsample": (21012750, 12321697), "prediction": (271156, 179459),
        "attention_q_projection": (14536040, 14536040),
        "attention_k_projection": (14536040, 14536040),
    }
    for name, values in category_expected.items():
        require((categories[name]["cycles"], categories[name]["eligible_cycles"]) == values,
                "M39-r2 category drift: {}".format(name))
    operator_rows = list(csv.DictReader(payloads["h67_operator_transactions"].splitlines()))
    bottleneck_rows = [row for row in operator_rows if row["category"] == "bottleneck"]
    require(len(bottleneck_rows) == 4 and
            sum(int(row["activity_cycles_at_config_lanes"]) for row in bottleneck_rows)
            == 79630957, "M39-r2 bottleneck operator population drift")
    return {"rules": rules, "local": local, "motion": motion, "selected": selected,
            "balanced": balanced, "categories": categories,
            "bottleneck_rows": bottleneck_rows}


def category_ledger(categories):
    rows = []
    for name in ("bottleneck", "patch_embed", "ffn_expand", "downsample", "prediction"):
        source = categories[name]
        rows.append({"category": name, "total_cycles": source["cycles"],
                     "already_m4_eligible_cycles": source["eligible_cycles"],
                     "remaining_noneligible_cycles": source["cycles"] - source["eligible_cycles"]})
    require(sum(row["remaining_noneligible_cycles"] for row in rows) == 132987740,
            "M39-r2 noneligible reconciliation failed")
    return rows


def target_gates(fixed, ideal, population, replacement):
    rows = []
    for target in (Fraction(27, 10), Fraction(3, 1)):
        ceiling = Fraction(fixed, 1) / target
        saving = Fraction(ideal, 1) - ceiling
        maximum = Fraction(population, 1) - saving
        headroom = maximum - replacement
        rows.append({
            "target_speedup": fraction_json(target),
            "target_cycle_ceiling": fraction_json(ceiling),
            "saving_required_from_scope": fraction_json(saving),
            "maximum_scope_replacement_cycles": fraction_json(maximum),
            "modeled_replacement_overhead_headroom_cycles": fraction_json(headroom),
            "crosses_target_in_conditional_dse": Fraction(replacement, 1) <= maximum,
        })
    return rows


def scope_row(scope, line, ideal, before, event, late, control, implementation, fixed):
    replacement = event + late + control
    after = ideal - before + replacement
    require(after + before == ideal + replacement, "M39-r2 conservation failure")
    return {
        "scope": scope, "line": line, "late_scale_implementation": implementation,
        "before_cycles": before,
        "replacement": {
            "event_accumulation_cycles": event, "late_scale_cycles": late,
            "frontend_control_cycles": control, "overlap_credit_cycles": 0,
            "overlap_policy": "SERIAL_SUM_CONSERVATIVE_NO_CREDIT_BEFORE_INTEGRATED_PROOF",
            "total_cycles": replacement,
        },
        "savings_cycles": before - replacement,
        "minimum_50m_saving_pass": before - replacement >= 50000000,
        "m38_ideal_before_scope_substitution_cycles": ideal,
        "conditional_cycles_after_substitution": after,
        "conditional_speedup_vs_fixed_exact": fraction_json(Fraction(fixed, after)),
        "conditional_speedup_vs_fixed_decimal": fixed / float(after),
        "conservation_equation": "after = m38_ideal - before + event + late + control",
        "bucket_disjointness": "M38_T10_BUCKET_DISJOINT_FROM_M7_M25_NONELIGIBLE_SCOPE",
        "target_gates": target_gates(fixed, ideal, before, replacement),
    }


def build_dse(validated):
    rules = validated["rules"]
    fixed = rules["fixed_compute_cycles"]
    ideals = {
        "Local": rules["selected_m30_local_cycles"] - rules["m30_t10_cycles"]
        + rules["m38_conditional_t10_cycles"],
        "Motion": rules["selected_m30_motion_cycles"] - rules["m30_t10_cycles"]
        + rules["m38_conditional_t10_cycles"],
    }
    require(ideals == {"Local": 268455448, "Motion": 266785174},
            "M39-r2 ideal cycle drift")
    m25_rows = {"Local": validated["local"], "Motion": validated["motion"]}
    m32_rows = {"Local": validated["balanced"]["local"],
                "Motion": validated["balanced"]["motion"]}
    ten_rows, four_rows = [], []
    for line in ("Local", "Motion"):
        m32 = m32_rows[line]
        event = m32["event_accumulation_cycles_borrowed"]
        control = m32["proportional_frontend_control_cycles"]
        for implementation, outputs_per_cycle in (("M33_shared96", 4),
                                                    ("M35_zero_mul_sidecar", 8)):
            late = (rules["consumer_outputs_per_sample"] + outputs_per_cycle - 1) // outputs_per_cycle
            ten_rows.append(scope_row("ten_semantically_admitted_consumers", line,
                                      ideals[line], rules["consumer_population_cycles"],
                                      event, late, control, implementation, fixed))
        m25 = m25_rows[line]
        bottleneck_event = 13282495 if line == "Local" else 12836419
        bottleneck_control = (
            m25["m21_fifo4_phase1_incremental_cycles"]
            * rules["bottleneck_population_cycles"]
            + rules["m4_profiled_population_cycles"] - 1
        ) // rules["m4_profiled_population_cycles"]
        for implementation, outputs_per_cycle in (("M33_shared96", 4),
                                                    ("M35_zero_mul_sidecar", 8)):
            late = (rules["bottleneck_outputs_per_sample"] + outputs_per_cycle - 1) // outputs_per_cycle
            four_rows.append(scope_row("four_bottleneck_conv3x3", line, ideals[line],
                                       rules["bottleneck_population_cycles"],
                                       bottleneck_event, late, bottleneck_control,
                                       implementation, fixed))
    return {
        "selected_m30_anchor": {
            "name": rules["selected_m30_candidate"], "local_cycles": 305047198,
            "motion_cycles": 303376924,
            "dual384b_is_slower_cycles_per_line": 24,
        },
        "m38_conditional_ideal": {
            "substitution": "M30 cycles - 73183500 T10 + 36591750 T10",
            "conditional_t10_ii": 5, "local_cycles": 268455448,
            "motion_cycles": 266785174,
            "local_speedup_exact": fraction_json(Fraction(fixed, 268455448)),
            "motion_speedup_exact": fraction_json(Fraction(fixed, 266785174)),
            "system_speedup_admitted": False,
            "claim": "CONDITIONAL_THEORY_ONLY_NOT_INTEGRATED_RTL_OR_MEASURED_SYSTEM_SPEEDUP",
        },
        "scope_alternatives_not_additive": True,
        "four_bottleneck_rows": four_rows, "ten_consumer_rows": ten_rows,
    }


def build(contract_path=DEFAULT_CONTRACT):
    contract, payloads, hashes, paths = load_contract(contract_path)
    m38_audit = verify_m38(payloads["m38_contract"], payloads["m38_result"])
    m33_audit = verify_m33_receipt(payloads["m33_receipt"])
    m35_audit = verify_m35_receipt(payloads["m35_receipt"])
    validated = validate_system_inputs(contract, payloads, hashes)
    dse = build_dse(validated)
    categories = category_ledger(validated["categories"])
    qk = (validated["categories"]["attention_q_projection"]["cycles"]
          + validated["categories"]["attention_k_projection"]["cycles"])
    require(qk == 29072080, "M39-r2 Q/K reconciliation failed")
    noneligible = sum(row["remaining_noneligible_cycles"] for row in categories)
    require(noneligible + qk == 162059820, "M39-r2 noneligible+Q/K drift")
    bottleneck_census = []
    for row in sorted(validated["bottleneck_rows"], key=lambda item: item["name"]):
        bottleneck_census.append({
            "name": row["name"], "operator": row["operator"],
            "input_shape": json.loads(row["input_shape_first"]),
            "output_shape": json.loads(row["output_shape_first"]),
            "input_activity": float(row["input_activity"]),
            "baseline_activity_cycles": int(row["activity_cycles_at_config_lanes"]),
            "im2col_per_invocation": {"M": 3000, "K": 6912, "N": 768},
        })
    shared_rows = []
    for line, row, total in (("Local", validated["local"], 225815624),
                             ("Motion", validated["motion"], 224145350)):
        parts = {
            "accelerated_m4_cycles": row["accelerated_m4_cycles"],
            "noneligible_plus_qk_cycles": row["noneligible_plus_qk_cycles"],
            "m21_frontend_control_cycles": row["m21_fifo4_phase1_incremental_cycles"],
            "registered_bubble_cycles": row["m21_registered_result_bubble_cycles"],
            "h67_rqtb_attention_cycles": row["rqtb_attention_cycles"],
        }
        require(sum(parts.values()) == total, "M39-r2 shared ledger drift")
        shared_rows.append({"line": line, "shared_non_atlif_cycles": total,
                            "parts": parts, "system_speedup_admitted": False})
    m33_area = m33_audit["standalone_area_um2"]
    m35_area = m35_audit["standalone_area_um2"]
    result = {
        "schema": "m39_remaining_bottleneck_v2",
        "status": "BLOCKED_BY_STALE_M38_R2_EXPLORATORY_ONLY_REANCHOR_REQUIRED_AFTER_M38_R3",
        "identity": {"contract": str(Path(contract_path).resolve()),
                     "contract_sha256": sha256(contract_path),
                     "analyzer_sha256": sha256(Path(__file__).resolve()),
                     "verified_input_sha256": hashes,
                     "verified_input_paths": paths},
        "supersedes": contract["supersedes"],
        "reanchor_notice": contract["reanchor_notice"],
        "recursive_evidence_audit": {"m38_r2": m38_audit, "m33_flat_r2": m33_audit,
                                     "m35_r7": m35_audit},
        "attention_and_trace_completeness": {
            "h67": "120_ATTENTION_ROWS_ABSTRACT_PACKED1_COMPUTE_ANCHOR_NOT_PHYSICAL_TRAFFIC",
            "local5_ep44": "MISSING_UNKNOWN_NONZERO_AT_LEAST_120_CALLS",
            "local_motion_name_boundary":
            "LOCAL_AND_MOTION_ARE_MECHANISMS_ON_FROZEN_H67_PROFILE100_NOT_LOCAL5_EP44",
        },
        "remaining_cycle_ledger": {
            "fixed_compute_cycles": 620868243,
            "shared_non_atlif_by_line": shared_rows,
            "noneligible_plus_qk_decomposition": {
                "noneligible_operator_cycles": noneligible,
                "q_projection_cycles": 14536040, "k_projection_cycles": 14536040,
                "qk_cycles": qk, "total_cycles": noneligible + qk,
                "noneligible_categories": categories,
            },
            "independent_cycle_reduction_ceilings": [
                {"scope": "four_bottleneck_conv3x3", "cycles": 79630957,
                 "can_save_50m_alone": True,
                 "maximum_replacement_to_save_50m": 29630957},
                {"scope": "qk_plus_rqtb_attention", "cycles": 32162811,
                 "can_save_50m_alone": False},
                {"scope": "patch_embed_remaining", "cycles": 27099543,
                 "can_save_50m_alone": False},
                {"scope": "ffn_expand_remaining", "cycles": 17474490,
                 "can_save_50m_alone": False},
                {"scope": "downsample_remaining", "cycles": 8691053,
                 "can_save_50m_alone": False},
                {"scope": "prediction_remaining", "cycles": 91697,
                 "can_save_50m_alone": False},
            ],
            "bottleneck_operator_census": bottleneck_census,
            "bitplane_materialization": {
                "q24_output_bytes": 4383720000, "bitpack_output_bytes": 182655000,
                "output_payload_reduction_exact": fraction_json(Fraction(24, 1)),
                "boundary_payload_reduction_exact": fraction_json(Fraction(32, 9)),
                "cycle_credit_admitted": False,
            },
        },
        "conditional_dse": dse,
        "late_scale_architecture_alternatives": [
            {"name": "M33_shared96_generic_UQ0p24", "outputs_per_cycle": 4,
             "additional_int8_multipliers": 0, "shared_pool_lanes_used": 80,
             "standalone_flat_area_um2_at_2ns": m33_area,
             "formality": "PASS_655_0_0_SELF_CONTAINED_SNAPSHOT",
             "integrated_system_claim": False},
            {"name": "M35_parallel_complement_CSD_sidecar", "outputs_per_cycle": 8,
             "additional_int8_multipliers": 0, "maximum_csd_terms": 4,
             "frozen_h67_threshold_delta_range": [1, 588],
             "standalone_area_um2_at_2ns": m35_area,
             "standalone_throughput_density_vs_flat_m33_exact":
             fraction_json(Fraction(8, 4) / Fraction(str(m35_area)) * Fraction(str(m33_area))),
             "formality": "PASS_2333_0_0_SELF_CONTAINED_SNAPSHOT",
             "integrated_system_claim": False},
        ],
        "resource_bandwidth_sram_contract": {
            "sole_signed_int8_multiplier_lanes": 96, "sram_banks": 24,
            "sram_row_bytes": 96, "ports_per_bank": "1R1W",
            "fixed_resident_bytes": 52032, "preferred_total_sram_kib": 240,
            "hard_total_sram_kib": 408,
            "prosperity_probe_incremental_bytes": 106880,
            "prosperity_probe_with_fixed_resident_bytes": 158912,
        },
        "prosperity_phi_adapter_assessment": {
            "Prosperity": {
                "real_domain_evidence_authority": "M32_ONLY_EXACT_REAL_DOMAIN_NUMBERS",
                "fixed_point_and_accuracy": "UNADMITTED",
                "official_repository_commit": "6ee1c6f1cb419fcf942f2eda63db84ca28248f4b",
                "repository_retrieved_date": "2026-08-22",
                "repository_retrieval_method":
                "git ls-remote https://github.com/dubcyfor3/Prosperity HEAD",
                "repository_file_sha256": "NOT_MEASURED_DO_NOT_INFER",
                "four_bottleneck_replacement_gate_to_save_50m_cycles": 29630957,
                "blocking_observability":
                "EXACT_BINARY_IM2COL_ROWS_SUBSET_FOREST_PRODUCT_DENSITY_AND_METADATA_ABSENT",
            },
            "Phi": {"adapter": "UNIMPLEMENTED_UNADMITTED",
                    "accuracy_and_memory_traffic": "UNADMITTED"},
            "qk_plus_rqtb_cycles": 32162811,
            "qk_attention_can_save_50m_alone": False,
        },
        "go_no_go_matrix": {
            "cycle": "USE_EXACT_RATIONAL_2P7_AND_3X_GATES_NO_OVERLAP_CREDIT",
            "integration": "ONE_96_LANE_POOL_LOCAL_MOTION_TAG_STATE_AND_FULL_CONSUMER_MITER",
            "synopsys": "INTEGRATED_VCS_DC_STA_FORMALITY_REQUIRED",
            "energy": contract["resource_and_admission_gates"]["energy_gate"],
            "accuracy": "BIT_EXACT_PRIMARY_OTHERWISE_FUTURE_VALID825_DELTA_AEE_LE_2_OVER_100",
        },
        "admission": {
            "remaining_cycle_decomposition_admitted": True,
            "conditional_h67_compute_dse_admitted": False,
            "m38_r2_archival_artifact_hash_verified": True,
            "m38_r2_recursive_anchor_admitted": False,
            "m38_r2_current_rebuild_admitted": False,
            "current_dependency_chain_admitted": False,
            "m33_flat_r2_recursive_synopsys_evidence_admitted": True,
            "m35_r7_recursive_synopsys_evidence_admitted": True,
            "m32_h67_real_domain_semantics_admitted": True,
            "prosperity_fixed_point_accuracy_admitted": False,
            "integrated_rtl_admitted": False,
            "executable_integrated_cycles_admitted": False,
            "address_timed_memory_admitted": False, "accuracy_admitted": False,
            "power_energy_admitted": False, "local5_full_system_admitted": False,
            "system_speedup_admitted": False, "headline_admitted": False,
            "final_freeze_admitted": False,
        },
        "claim_boundary": contract["claim_boundary"],
    }
    return result


def write_output(path, payload):
    path = Path(path)
    if path.exists():
        raise ValueError("refusing to overwrite existing M39-r2 output")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.contract)
    write_output(args.output, result)
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                      "output_sha256": sha256(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
