#!/usr/bin/env python3
"""Validate the frozen M31-r4 static-phase VCS receipt fail closed."""

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
DEFAULT_RECEIPT = (
    HW_ROOT / "contracts/m31_output_receipt_r4_static_phase_20260822.json"
)
R3_RECEIPT = HW_ROOT / "contracts/m31_output_receipt_r3_20260822.json"
FORMALITY_TCL = HW_ROOT / "dc_handoff/scripts/run_formality.tcl"
FORMALITY_WRAPPER = HW_ROOT / "dc_handoff/run_formality.sh"

EXPECTED_RECEIPT_SHA256 = (
    "bae2f05e74ffa8863195bda9f222c22fc06364ade872e9cf83d3cd4106e5b77d"
)
EXPECTED_R3_RECEIPT_SHA256 = (
    "3785a36272845bb5ea240d9aa7eed5bdc934b6cf453ebf2a90f5a16131109577"
)
EXPECTED_CONTRACT_SHA256 = (
    "f98cdde7ad617ba0ceac14d9b145e3671403698ee40bae54c492010dc91997fd"
)
EXPECTED_SCHEMA = "m31_output_receipt_v4"
EXPECTED_STATUS = (
    "PASS_UNIFIED_T10_T2_STATIC_PHASE_EXACT_FIXED_POINT_SINGLE_SOURCE_"
    "MUL96_VCS_NO_DC_FORMALITY_PPA_OR_SYSTEM_CLAIM"
)

EXPECTED_FILES = {
    "multiplier_pool_rtl": (
        "hw_autoresearch_nts07/rtl_m31/qfit_signed_int8_mul96_pool.sv",
        "7872d25c01c112f07a7d8e3cfe728029eef1f68e0f7bf87bdf2a50416776ea18",
    ),
    "unified_core_rtl": (
        "hw_autoresearch_nts07/rtl_m31/"
        "qfit_atlif_unified_t10_t2_stream_core.sv",
        "c094849e88c0d9fc3a390d0cf6fc9adf10ff4dc31d77e265e425e5cf71b5ef15",
    ),
    "assertions": (
        "hw_autoresearch_nts07/verif_m31/"
        "qfit_atlif_unified_t10_t2_stream_assertions.sv",
        "695fd1923d0a9f6a2af40fb008e1c3ff4c1fec7aa88b6724cb7c7bac29e8f5da",
    ),
    "testbench": (
        "hw_autoresearch_nts07/tb_m31/"
        "tb_qfit_atlif_unified_t10_t2_stream_core.sv",
        "9d1a59b59e8711d137ac64be2f8b2e0314ea5b3fd08d75f44bbbf21d42ea7b79",
    ),
    "filelist": (
        "hw_autoresearch_nts07/dc_handoff/filelists/"
        "date_m31_unified_t10_t2_vcs.f",
        "435550cf64b2a71debefd69cf582f37adc0a30b49b886c46e4087d1b37cc94a9",
    ),
    "run_script": (
        "hw_autoresearch_nts07/dc_handoff/scripts/"
        "run_vcs_m31_unified_t10_t2_sva.sh",
        "a8469c5d4e61943339788134023c72474c24009448ab3d45f88762435d763d59",
    ),
}

TOP_KEYS = {
    "schema", "date", "status", "contract", "files", "source_revision",
    "vcs_run", "observed", "regression_against_r3", "supersedes",
    "claim_boundary", "headline_admitted", "independent_review_required",
}
VCS_RUN_KEYS = {
    "directory", "runner_exit_code", "input_sha256_manifest",
    "input_manifest_check_from_hw_root", "output_sha256_manifest",
    "output_manifest_check", "compile_log", "sim_log",
    "assertion_failure_signatures", "uncovered_SVA_cover_properties",
}
OBSERVED_KEYS = {
    "mode_sequence", "sole_source_multiplier_pool_instances",
    "source_multiplier_slots", "t10_tiles", "t10_arithmetic_cycles",
    "conditional_t10_no_stall_accept_ii", "t10_credit_wait_cycles",
    "t2_packets", "t2_nondegenerate_packets",
    "conditional_t2_no_stall_accept_ii", "conditional_t2_ii1_matches",
    "total_arithmetic_cycles", "maximum_result_fifo_occupancy",
    "fifo_full_cycles", "fifo_full_simultaneous_pop_push_cycles",
    "input_output_stall_cycles", "release_input_collisions",
    "t10_threshold_ties", "t10_intermediate_positive_negative_saturations",
    "t10_output_positive_negative_saturations",
    "threshold_equal_raw_just_below_cases",
    "t2_positive_negative_saturations", "t2_output_one_zero_bits",
}
PASS_TOKEN_KEYS = {
    "modes", "sole_mul_pool", "mul_slots", "t10_tiles", "t10_arithmetic",
    "t10_ii", "t10_credit_wait", "t2_packets", "t2_nondegenerate",
    "t2_ii", "t2_ii_matches", "total_arithmetic", "max_fifo",
    "full_cycles", "full_pop_push", "stalls", "release_input_collisions",
    "t10_ties", "t10_mid_sat", "t10_out_sat", "threshold_eq_below",
    "t2_sat", "t2_diversity",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_keys(mapping, expected, label):
    if not isinstance(mapping, dict) or set(mapping) != set(expected):
        raise ValueError("{} key population drift".format(label))


def read_json_no_duplicates(path):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: {}".format(key))
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=object_pairs)


def resolve_from_root(root, raw):
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def read_manifest(path):
    result = {}
    for number, raw in enumerate(
            Path(path).read_text(encoding="utf-8").splitlines(), 1):
        pieces = raw.strip().split(None, 1)
        if len(pieces) != 2 or not re.match(r"^[0-9a-f]{64}$", pieces[0]):
            raise ValueError("malformed manifest line {}".format(number))
        name = pieces[1].lstrip("*")
        if name in result:
            raise ValueError("duplicate manifest path")
        result[name] = pieces[0]
    if not result:
        raise ValueError("empty manifest")
    return result


def verify_manifest(entries, base):
    for raw, expected in entries.items():
        path = Path(raw)
        if not path.is_absolute():
            path = Path(base) / path
        if not path.is_file() or sha256(path) != expected:
            raise ValueError("manifest live content drift: {}".format(raw))


def parse_pass_line(sim_text):
    lines = [line for line in sim_text.splitlines()
             if line.startswith("M31_PASS ")]
    if len(lines) != 1:
        raise ValueError("M31_PASS line population drift")
    tokens = {}
    for item in lines[0].split()[1:]:
        if item.count("=") != 1:
            raise ValueError("malformed M31_PASS token")
        key, value = item.split("=", 1)
        if key in tokens:
            raise ValueError("duplicate M31_PASS token")
        tokens[key] = value
    exact_keys(tokens, PASS_TOKEN_KEYS, "M31_PASS tokens")
    return lines[0], tokens


def split_ints(value):
    return [int(item) for item in value.split("/")]


def observed_from_tokens(tokens):
    if tokens["modes"] != "T10_T2N_T2S_T10":
        raise ValueError("M31 mode sequence drift")
    scalar = {
        "sole_source_multiplier_pool_instances": "sole_mul_pool",
        "source_multiplier_slots": "mul_slots",
        "t10_tiles": "t10_tiles",
        "t10_arithmetic_cycles": "t10_arithmetic",
        "conditional_t10_no_stall_accept_ii": "t10_ii",
        "t10_credit_wait_cycles": "t10_credit_wait",
        "t2_packets": "t2_packets",
        "t2_nondegenerate_packets": "t2_nondegenerate",
        "conditional_t2_no_stall_accept_ii": "t2_ii",
        "conditional_t2_ii1_matches": "t2_ii_matches",
        "total_arithmetic_cycles": "total_arithmetic",
        "maximum_result_fifo_occupancy": "max_fifo",
        "fifo_full_cycles": "full_cycles",
        "fifo_full_simultaneous_pop_push_cycles": "full_pop_push",
        "release_input_collisions": "release_input_collisions",
        "t10_threshold_ties": "t10_ties",
    }
    vectors = {
        "input_output_stall_cycles": "stalls",
        "t10_intermediate_positive_negative_saturations": "t10_mid_sat",
        "t10_output_positive_negative_saturations": "t10_out_sat",
        "threshold_equal_raw_just_below_cases": "threshold_eq_below",
        "t2_positive_negative_saturations": "t2_sat",
        "t2_output_one_zero_bits": "t2_diversity",
    }
    observed = {
        "mode_sequence": ["T10", "T2_NONDEGENERATE", "T2_SATURATION", "T10"]
    }
    for field, token in scalar.items():
        observed[field] = int(tokens[token])
    for field, token in vectors.items():
        observed[field] = split_ints(tokens[token])
    exact_keys(observed, OBSERVED_KEYS, "derived M31 observed")
    return observed


def parse_cover_rows(sim_text):
    rows = re.findall(
        r'assertions\.sv",\s*(\d+):.*?,\s*(\d+)\s+attempts,\s*'
        r'(\d+)\s+match', sim_text,
    )
    parsed = {}
    for line, attempts, matches in rows:
        line = int(line)
        if line in parsed:
            raise ValueError("duplicate M31 cover report line")
        parsed[line] = [int(attempts), int(matches)]
    expected = {108: [527, 26], 110: [527, 85],
                111: [527, 32], 112: [527, 1]}
    if parsed != expected:
        raise ValueError("M31 exact cover population drift")
    return parsed


def scan_logs(compile_text, sim_text):
    combined = compile_text + "\n" + sim_text
    warning_lines = [line for line in combined.splitlines()
                     if re.search(r"^\s*Warning[:\-]", line, re.I)]
    failure_lines = [line for line in combined.splitlines() if re.search(
        r"^\s*(Error|Fatal)[:\-]|assertion.*(fail|error)|offending.*assert|"
        r"UVM_(ERROR|FATAL)", line, re.I)]
    if warning_lines:
        raise ValueError("M31 compile/sim warning population is nonzero")
    if failure_lines:
        raise ValueError("M31 compile/sim failure population is nonzero")
    for marker in ("M31_SVA_BOUND=1", "SIMULATOR=Synopsys VCS",
                   "ASSERTIONS=enabled"):
        if sim_text.splitlines().count(marker) != 1:
            raise ValueError("M31 exact simulator marker drift: {}".format(marker))
    return {"warning_count": 0, "failure_signature_count": 0}


def validate_static_source(core_path, pool_path):
    core = Path(core_path).read_text(encoding="utf-8")
    pool = Path(pool_path).read_text(encoding="utf-8")
    dynamic_phase_subscripts = [item for item in re.findall(
        r"\[[^\]]*\]", core, re.S) if "phase_cycle_q" in item]
    phase_loops = len(re.findall(
        r"for\s*\(int\s+phase\s*=\s*0;\s*phase\s*<\s*T10_PHASES", core))
    pool_instances = len(re.findall(
        r"^\s*qfit_signed_int8_mul96_pool\s*#", core, re.M))
    multiplier_templates = len(re.findall(
        r"assign\s+product\s*=\s*\$signed\(operand_a\)\s*\*\s*"
        r"\$signed\(operand_b\)", pool))
    if dynamic_phase_subscripts or phase_loops != 3:
        raise ValueError("M31 static phase source audit drift")
    if pool_instances != 1 or multiplier_templates != 1:
        raise ValueError("M31 source multiplier hierarchy drift")
    return {
        "dynamic_phase_indexed_t10_arrays": 0,
        "static_phase_for_loops": phase_loops,
        "source_multiplier_pool_instances": pool_instances,
        "source_multiplier_leaf_templates": multiplier_templates,
        "phase_5_through_7_fault_behavior_admitted": False,
    }


def validate_formality_filter(formality_tcl, formality_wrapper):
    tcl = Path(formality_tcl).read_text(encoding="utf-8")
    wrapper = Path(formality_wrapper).read_text(encoding="utf-8")
    if tcl.count("set_mismatch_message_filter -warn FMR_ELAB-147") != 1:
        raise ValueError("Formality mismatch filter population drift")
    expected_guard = (
        'if {$design_name eq "qfit_dual_line_descriptor_resident_engine"\n'
        '        || $design_name eq "qfit_dual_line_descriptor_stateful_engine"}'
    )
    if expected_guard not in tcl:
        raise ValueError("Formality descriptor-only filter guard drift")
    if ("qfit_atlif_unified_t10_t2_stream_core)" not in wrapper
            or "date_m31_unified_t10_t2_dc.f" not in wrapper):
        raise ValueError("Formality M31 launcher registration drift")
    return {
        "m31_fmr_elab_147_filter_applied": False,
        "formality_tcl_sha256": sha256(formality_tcl),
        "formality_wrapper_sha256": sha256(formality_wrapper),
        "formality_run_admitted": False,
        "binding_note": "current scripts are hashed by this admission, not r4 receipt",
    }


def build(receipt_path=DEFAULT_RECEIPT, enforce_receipt_sha=True,
          root=ROOT, formality_tcl=FORMALITY_TCL,
          formality_wrapper=FORMALITY_WRAPPER):
    receipt_path = Path(receipt_path).resolve()
    receipt_sha = sha256(receipt_path)
    if enforce_receipt_sha and receipt_sha != EXPECTED_RECEIPT_SHA256:
        raise ValueError("M31 r4 receipt identity drift")
    receipt = read_json_no_duplicates(receipt_path)
    exact_keys(receipt, TOP_KEYS, "M31 r4 receipt")
    if receipt["schema"] != EXPECTED_SCHEMA or receipt["status"] != EXPECTED_STATUS:
        raise ValueError("M31 r4 exact schema/status drift")
    if receipt["date"] != "2026-08-22":
        raise ValueError("M31 r4 receipt date drift")
    if receipt["contract"] != {
            "path": "hw_autoresearch_nts07/contracts/"
                    "m31_unified_t10_t2_vcs_contract_r2_20260822.json",
            "sha256": EXPECTED_CONTRACT_SHA256}:
        raise ValueError("M31 r4 contract identity drift")
    contract_path = resolve_from_root(root, receipt["contract"]["path"])
    if not contract_path.is_file() or sha256(contract_path) != EXPECTED_CONTRACT_SHA256:
        raise ValueError("M31 r4 live contract drift")
    contract = read_json_no_duplicates(contract_path)
    if contract.get("contract") != "m31_unified_t10_t2_vcs_contract_r2":
        raise ValueError("M31 r4 contract schema drift")

    exact_keys(receipt["files"], EXPECTED_FILES, "M31 r4 files")
    live_files = {}
    for name, expected in EXPECTED_FILES.items():
        item = receipt["files"][name]
        if not isinstance(item, list) or tuple(item) != expected:
            raise ValueError("M31 r4 exact source identity drift: {}".format(name))
        path = resolve_from_root(root, item[0])
        if not path.is_file() or sha256(path) != item[1]:
            raise ValueError("M31 r4 live source drift: {}".format(name))
        live_files[name] = path

    if receipt["source_revision"] != {
            "static_phase_array_indexing": True, "phase_count": 5,
            "dynamic_phase_indexed_t10_arrays": 0,
            "formality_mismatch_filter_permitted_for_m31": False}:
        raise ValueError("M31 r4 source revision schema drift")
    exact_keys(receipt["vcs_run"], VCS_RUN_KEYS, "M31 r4 VCS run")
    run_spec = receipt["vcs_run"]
    run = Path(run_spec["directory"]).resolve()
    if run.name != "m31_unified_t10_t2_vcs_r4_static_phase_20260822":
        raise ValueError("M31 r4 VCS run identity drift")
    if (run_spec["runner_exit_code"] != 0
            or run_spec["input_manifest_check_from_hw_root"] != "PASS_ALL_6"
            or run_spec["output_manifest_check"] != "PASS_ALL_2"
            or run_spec["assertion_failure_signatures"] != 0
            or run_spec["uncovered_SVA_cover_properties"] != 0):
        raise ValueError("M31 r4 wrapper admission drift")
    input_manifest = run / "input_sha256.txt"
    output_manifest = run / "output_sha256.txt"
    compile_log = run / "compile.log"
    sim_log = run / "sim.log"
    for path in (input_manifest, output_manifest, compile_log, sim_log):
        if not path.is_file():
            raise ValueError("M31 r4 run evidence is missing")
    if sha256(input_manifest) != run_spec["input_sha256_manifest"]:
        raise ValueError("M31 r4 input manifest hash drift")
    if sha256(output_manifest) != run_spec["output_sha256_manifest"]:
        raise ValueError("M31 r4 output manifest hash drift")
    if sha256(compile_log) != run_spec["compile_log"]:
        raise ValueError("M31 r4 compile log hash drift")
    if sha256(sim_log) != run_spec["sim_log"]:
        raise ValueError("M31 r4 simulation log hash drift")

    input_entries = read_manifest(input_manifest)
    expected_inputs = {}
    prefix = "hw_autoresearch_nts07/"
    for item in receipt["files"].values():
        if not item[0].startswith(prefix):
            raise ValueError("M31 r4 source root drift")
        expected_inputs[item[0][len(prefix):]] = item[1]
    if input_entries != expected_inputs:
        raise ValueError("M31 r4 input manifest population drift")
    verify_manifest(input_entries, Path(root) / "hw_autoresearch_nts07")
    output_entries = read_manifest(output_manifest)
    expected_outputs = {
        str(compile_log): run_spec["compile_log"],
        str(sim_log): run_spec["sim_log"],
    }
    if output_entries != expected_outputs:
        raise ValueError("M31 r4 output manifest population drift")
    verify_manifest(output_entries, run)

    compile_text = compile_log.read_text(encoding="utf-8")
    sim_text = sim_log.read_text(encoding="utf-8")
    log_audit = scan_logs(compile_text, sim_text)
    pass_line, pass_tokens = parse_pass_line(sim_text)
    observed = observed_from_tokens(pass_tokens)
    exact_keys(receipt["observed"], OBSERVED_KEYS, "M31 r4 observed receipt")
    if receipt["observed"] != observed:
        raise ValueError("M31_PASS observed receipt drift")
    covers = parse_cover_rows(sim_text)
    assertions_text = live_files["assertions"].read_text(encoding="utf-8")
    assert_count = len(re.findall(r"^\s*assert\s+property", assertions_text, re.M))
    cover_count = len(re.findall(r"^\s*cover\s+property", assertions_text, re.M))
    if (assert_count, cover_count) != (24, 4):
        raise ValueError("M31 r4 SVA property population drift")

    if not R3_RECEIPT.is_file() or sha256(R3_RECEIPT) != EXPECTED_R3_RECEIPT_SHA256:
        raise ValueError("M31 r3 regression receipt identity drift")
    r3 = read_json_no_duplicates(R3_RECEIPT)
    r3_sim = Path(r3["vcs_run"]["directory"]) / "sim.log"
    if not r3_sim.is_file() or sha256(r3_sim) != r3["vcs_run"]["sim_log"]:
        raise ValueError("M31 r3 regression simulation drift")
    r3_line, _ = parse_pass_line(r3_sim.read_text(encoding="utf-8"))
    r3_covers = parse_cover_rows(r3_sim.read_text(encoding="utf-8"))
    if r3_line != pass_line or r3_covers != covers:
        raise ValueError("M31 r4 functional or cover regression")
    if receipt["regression_against_r3"] != {
            "m31_pass_metric_line": "EXACT_MATCH",
            "SVA_cover_matches_r3": {
                "t10_ii10": 26, "t2_ii1": 85,
                "fifo_full": 32, "release_input_collision": 1},
            "functional_or_coverage_regression": False}:
        raise ValueError("M31 r4 regression receipt drift")
    if receipt["supersedes"] != {
            "path": "hw_autoresearch_nts07/contracts/"
                    "m31_output_receipt_r3_20260822.json",
            "state": "STALE_SUPERSEDED_DO_NOT_CITE_AS_CURRENT_SOURCE",
            "reason": "r3 predates the statically phase-bounded "
                      "T10 array-index revision"}:
        raise ValueError("M31 r4 supersession drift")
    if receipt["claim_boundary"] != {
            "permitted": "the statically phase-bounded current M31 source "
                         "passes the frozen nondegenerate and rail-saturation "
                         "T2 plus T10 exact fixed-point VCS/SVA workload with "
                         "the same measured behavior and cover hits as r3",
            "forbidden": "DC or Formality closure for this source revision, "
                         "post-synthesis resources or frequency, PPA, power or "
                         "energy, memory feasibility, full-network cycles or "
                         "speedup, trained accuracy, end-to-end bitplane "
                         "equivalence, comparison with prior accelerators, or "
                         "any DATE headline"}:
        raise ValueError("M31 r4 claim boundary drift")
    if receipt["headline_admitted"] is not False:
        raise ValueError("M31 r4 headline admission drift")

    source_audit = validate_static_source(
        live_files["unified_core_rtl"], live_files["multiplier_pool_rtl"])
    formality_filter = validate_formality_filter(
        formality_tcl, formality_wrapper)
    return {
        "schema": "m31_r4_static_phase_vcs_machine_admission_v1",
        "status": "PASS_EXACT_FROZEN_M31_R4_STATIC_PHASE_VCS_ONLY",
        "identity": {
            "receipt": str(receipt_path),
            "receipt_sha256": receipt_sha,
            "validator_sha256": sha256(Path(__file__).resolve()),
            "contract_sha256": EXPECTED_CONTRACT_SHA256,
            "r3_regression_receipt_sha256": EXPECTED_R3_RECEIPT_SHA256,
        },
        "manifest_audit": {
            "input_count": len(input_entries),
            "output_count": len(output_entries),
            "all_live_content_rehashed": True,
        },
        "log_audit": dict(log_audit, m31_pass_line_count=1,
                          assert_property_count=assert_count,
                          cover_property_count=cover_count,
                          exact_cover_rows={str(key): value
                                            for key, value in covers.items()}),
        "observed": observed,
        "r3_regression": {"m31_pass_exact": True, "cover_exact": True},
        "source_audit": source_audit,
        "current_formality_filter_audit": formality_filter,
        "admission": {
            "current_r4_vcs_source_admitted": True,
            "dc_sta_admitted": False,
            "formality_admitted": False,
            "phase_fault_recovery_admitted": False,
            "ppa_power_energy_admitted": False,
            "system_speedup_admitted": False,
            "headline_admitted": False,
        },
        "claim_boundary": receipt["claim_boundary"],
    }


def write_output(path, result):
    path = Path(path)
    if path.exists():
        raise ValueError("refusing to overwrite M31 r4 machine admission")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.receipt)
    write_output(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
