#!/usr/bin/env python3
"""Static fail-closed checker for the formal M1843 source package."""
from __future__ import print_function

import hashlib
import json
from pathlib import Path
import re
import sys


HW = Path(__file__).resolve().parents[2]
PATHS = {
    "runner": HW / "dc_handoff/scripts/run_m1843_c2_fresh_mapped_formality_dual_corner_pt_one_shot.py",
    "formality": HW / "dc_handoff/scripts/run_formality_m1843_m1809_c2_fresh_mapped_two_axis.tcl",
    "pt": HW / "dc_handoff/scripts/run_ptsta_m1843_m1809_c2_fresh_mapped_dual_corner.tcl",
    "checker": Path(__file__).resolve(),
    "test": HW / "system_simulator/tests/test_m1843_c2_fresh_mapped_formality_dual_corner_pt_source.py",
    "contract": HW / "contracts/m1843_m1834_c2_fresh_mapped_formality_dual_corner_pt_source_contract_r1_20260902.json",
}

BASE_TOP = "m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24"
M1811_MANIFEST = "695050260d54ca9b9d6f7b74d03021dd59afd642168981a13df0438e9fe12066"
M1811_OUTER = "04aa6bea4a06a8be3c441ddb984c68a046810a137fd2eca096adf513af0d324b"
M1830_REVIEW = "79e1885fad8ddac4ec0a6eee4d9034657761e778da384093fae5ab937f98f99b"
M1830_MANIFEST = "d0ef8172f33378e9b025aab18043da19335fd9f00d1cd8d240bfb620997c0d06"
M1830_OUTER = "0b9dc1915096db8df6702e3ab5027d267fb99a3178bc2288a8b5625e611e343d"
M1834_FAILED_REVIEW = "510133974d005a4259279966dff2f29205b077facfcb8ef798e608eddb4be33d"
M1834_FAILED_MANIFEST = "78e89c2022ecde5cfdb437ee4196c31a176992428649580be3f3866b786b56ea"
M1834_FAILED_OUTER = "126c85936ed69c8904f35823a11d292720f0a49de5ff5ea2ca28ab6b6b6247df"
LIVE_RTL = {
    "rtl_m1609/m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_registered_fault_successor.sv": "7ee28b3912ae34c99c795a48e80be29df2b59b363e5de2d2b359175ec9dda931",
    "rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv": "8295393bf91a9bfc64a2253aaff60db97df5df587ab9b77d56996afee82cb2a0",
    "rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv": "529e463802fec72716ac6592d31e7668104a5463ff92499a98ec7314c8e88267",
    "rtl_m218/m218_fc2_tagged_slice_service_island.sv": "f6537081977e9dc09e968fad800b333604b4573ee2e9361960483349fe1e8ad1",
    "rtl_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter.sv": "44f7df331af66ba62fadf5e336b9c0c00d00f809e215aa8e091e9de011c5627e",
    "rtl_m519/m519_fc2_k1_registered_release_service_island.sv": "3811998fc48d31e6519ecc6c6cfb8f5d38db6fc6dd070e09d73a5f70b7579871",
    "rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv": "010fe9e6786db1d3bbcad7759bda17a783ce5cfe15cae02c5b4c9ebf96e9950b",
    "rtl_m519/m519_fc2_k1_registered_release_8bank_raw4_acc24.sv": "6ea038ef935b1144d5424634e75446301270362c259341a8e7e7117523b25815",
    "rtl_m519/m519_fc2_k1x8_registered_release_raw4_acc24.sv": "11080d39c06672cebb64988e931c41e1d4c04134a312aeb8e250d01f0ac576ff",
    "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv": "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
    "rtl_m1801/m1801_c2_registered_public_fault_export.sv": "fcd002804f1086d90237ddb36ed2178213ef5992adde18d148f6c14ff11db18d",
    "rtl_m1801/m1801_m803_fc2_k8_registered_public_fault_8bank_raw4_acc24.sv": "f77ac9f343961ea37a277c106ebe099191cf7005c35dcbd8eb98e01b1eccb59c",
    "rtl_m1809/m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24.sv": "405ac73f401440245e3edaea7c6e23a222883c44e8ed77e732983df721664c66",
}
ARTIFACT_SHAS = {
    "K8": {
        "mapped_v_sha256": "63605469818c36574ce9719130877610e79cf0c3b7317c0e69848539afa6b792",
        "mapped_sdc_sha256": "af2fbde96a5046053aed137facc4fd2741b3f517eb678710c81eef9f7ed49018",
        "svf_sha256": "b5fe89b8c44e6edd9aa4e1a06e9d13234148f2dbd2b7b00cb8014bd838b65543",
    },
    "K1X8": {
        "mapped_v_sha256": "8698d227f3408b6e40c03bfe9282de458b0ba5cba4e22ec5f0c9bfd4ff16fc1b",
        "mapped_sdc_sha256": "1631f7d0cc3d0257439dea5f9ed2a2fc004556dc0f8f5657152a7d3f5f3e6c0a",
        "svf_sha256": "bcb2f9f974be2ee8d4927d41d99b4e06abac77635e8449e61d61163c6b05d2dc",
    },
}


class CheckFailure(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_json_text(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise CheckFailure("duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(text, object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           CheckFailure("nonfinite JSON: " + token)))
    if type(value) is not dict:
        raise CheckFailure("contract root must be object")
    return value


def source_map(overrides=None):
    overrides = {} if overrides is None else overrides
    result = {}
    for name, path in PATHS.items():
        if name in overrides:
            result[name] = overrides[name]
        elif name == "contract" and not path.exists():
            raise CheckFailure("formal contract absent")
        else:
            if not path.is_file() or path.is_symlink():
                raise CheckFailure("source absent/nonregular: " + str(path))
            result[name] = path.read_text()
    return result


def require(text, token, scope):
    if token not in text:
        raise CheckFailure(scope + " missing token: " + token)


def reject(text, token, scope):
    if token in text:
        raise CheckFailure(scope + " forbidden token: " + token)


def check_runner(text):
    required = (
        'M1811_MANIFEST_SHA = "' + M1811_MANIFEST + '"',
        'M1811_OUTER_SHA = "' + M1811_OUTER + '"',
        'M1830_REVIEW_SHA = "' + M1830_REVIEW + '"',
        'M1830_MANIFEST_SHA = "' + M1830_MANIFEST + '"',
        'M1830_OUTER_SHA = "' + M1830_OUTER + '"',
        'AXIS_ORDER = ("K8", "K1X8")',
        '"arch_mode": 0', '"arch_mode": 1',
        '"elab_parameters": "ARCH_MODE=0"',
        '"elab_parameters": "ARCH_MODE=1"',
        '"implementation_top": DESIGN + "_ARCH_MODE0"',
        '"implementation_top": DESIGN + "_ARCH_MODE1"',
        '"mapped_v_sha": "' + ARTIFACT_SHAS["K8"]["mapped_v_sha256"] + '"',
        '"mapped_sdc_sha": "' + ARTIFACT_SHAS["K8"]["mapped_sdc_sha256"] + '"',
        '"svf_sha": "' + ARTIFACT_SHAS["K8"]["svf_sha256"] + '"',
        '"mapped_v_sha": "' + ARTIFACT_SHAS["K1X8"]["mapped_v_sha256"] + '"',
        '"mapped_sdc_sha": "' + ARTIFACT_SHAS["K1X8"]["mapped_sdc_sha256"] + '"',
        '"svf_sha": "' + ARTIFACT_SHAS["K1X8"]["svf_sha256"] + '"',
        "verify_sealed_directory(M1811, M1811_MANIFEST_SHA, M1811_OUTER_SHA)",
        "verify_sealed_directory(M1830, M1830_MANIFEST_SHA, M1830_OUTER_SHA)",
        "exact_regular(M1830 / \"review.json\", M1830_REVIEW_SHA)",
        "def verify_live_rtl_identity(review):",
        "rows != list(sources.keys())",
        "exact_regular(M1811_INPUT_FILELIST, REFERENCE_FILELIST_SHA)",
        "M1811_INPUT_FILELIST.read_bytes() != REFERENCE_FILELIST.read_bytes()",
        "exact_regular(HW / rel, sources[rel])",
        "M1843_EXPECTED_RUNNER_SHA256",
        "M1843_EXPECTED_SOURCE_CONTRACT_SHA256",
        "M1843_EXPECTED_M1844_SOURCE_REVIEW_SHA256",
        "M1843_EXPECTED_M1844_SOURCE_REVIEW_MANIFEST_SHA256",
        "M1843_EXPECTED_M1844_SOURCE_REVIEW_OUTER_SHA256",
        "M1843_EXPECTED_M1846_LAUNCH_RELEASE_SHA256",
        '"m1844_source_review_manifest_sha256": review_manifest',
        '"m1844_source_review_outer_seal_file_sha256": review_outer',
        "AUTHORIZE_ONE_M1843_C2_FRESH_MAPPED_FORMALITY_DUAL_CORNER_PT_ATTEMPT",
        '"max_attempts": 1', '"formality_runs": 2', '"pt_runs": 2',
        '"automatic_retry": False',
        "for axis in AXIS_ORDER:",
        "run_tool(FM_SHELL, FM_TCL, axis, fm_dir, \"formality.log\")",
        "run_tool(PT_SHELL, PT_TCL, axis, pt_dir, \"pt.log\")",
        "write_attempt(release_sha)",
        "fcntl.flock(queue_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)",
        "license_gate()",
        "atomic_publish_no_replace(WORK, RESULT)",
        '"negative_slack_reported_not_hidden": True',
        '"hold_failure_blocks_result_publication": False',
        '"timing_exceptions_added": False',
        '"pt_eco": False',
        '"hold_closed": False',
        '"system_speedup": False',
        '"paper_ppa_ready": False',
        '"live_rtl_source_identity": live_rtl_identity',
        "summary.update(parse_pt_semantics(reports))",
        'reports / "exceptions.rpt"', 'reports / "design.rpt"',
        'reports / "wire_load.rpt"',
        'reports / "constraint_semantics_machine.txt"',
        'constraint_report = (reports / "constraint_violators.rpt").read_text(',
        'raw_constraint_violation_marker_count = len(re.findall(',
    )
    for token in required:
        require(text, token, "runner")
    for token in ("DRAFT_ONLY", "DraftOnly", "UNSEALED_SOURCE_DRAFT",
                  "m1833", "m1835", "m1845", "automatic_retry\": True",
                  "hold_failure_blocks_result_publication\": True"):
        reject(text, token, "runner")
    if text.count("run_tool(FM_SHELL, FM_TCL, axis") != 1:
        raise CheckFailure("runner Formality call site cardinality")
    if text.count("run_tool(PT_SHELL, PT_TCL, axis") != 1:
        raise CheckFailure("runner PT call site cardinality")
    if text.index("verify_authority()") > text.index("write_attempt(release_sha)"):
        raise CheckFailure("authority must precede attempt")
    if text.index("license_gate()") > text.index("write_attempt(release_sha)"):
        raise CheckFailure("license gate must precede attempt")
    if text.index("write_attempt(release_sha)") > text.index(
            "run_tool(FM_SHELL, FM_TCL, axis, fm_dir"):
        raise CheckFailure("attempt must precede first tool call")


def check_formality(text):
    for token in (
            "M1843_REF_ELAB_PARAMETERS",
            'set expected_elab_parameters "ARCH_MODE=0"',
            'set expected_elab_parameters "ARCH_MODE=1"',
            'set expected_implementation_top "${base_top}_ARCH_MODE0"',
            'set expected_implementation_top "${base_top}_ARCH_MODE1"',
            "[llength $rtl_files] != 13",
            "set_svf $implementation_svf",
            "read_sverilog -r $reference_files",
            "set_top r:/WORK/$reference_top -parameter $reference_elab_parameters",
            "read_verilog -i $implementation_netlist",
            "set_top i:/WORK/$implementation_top",
            "set verification_succeeded [verify]",
            "report_unmatched_points", "report_failing_points",
            "report_aborted_points", "report_unverified_points",
            "M1843_C2_FRESH_MAPPED_FORMALITY_INTERNAL_COMPLETE=PASS"):
        require(text, token, "formality")
    for token in ("source draft", "DRAFT", "ARCH_MODE=2"):
        reject(text, token, "formality")


def check_pt(text):
    for token in (
            'set expected_implementation_top "${base_top}_ARCH_MODE0"',
            'set expected_implementation_top "${base_top}_ARCH_MODE1"',
            "set_min_library $std_slow_db -min_version $std_fast_db",
            "set_operating_conditions -analysis_type on_chip_variation",
            "-max $slow_opcond -max_library $slow_lib_name",
            "-min $fast_opcond -min_library $fast_lib_name",
            "read_verilog $mapped_netlist", "read_sdc $mapped_sdc",
            "report_analysis_coverage -status_details untested",
            "report_timing -delay_type max", "timing_setup_slow.rpt",
            "report_timing -delay_type min", "timing_hold_fast.rpt",
            "report_constraint -all_violators", "constraint_violators.rpt",
            "constraint_semantics_machine.txt",
            "setup_violating_paths=[sizeof_collection $setup_violators]",
            "hold_violating_paths=[sizeof_collection $hold_violators]",
            "report_exceptions -ignored", "exceptions.rpt",
            "report_design", "design.rpt",
            "report_wire_load", "wire_load.rpt",
            "setup_corner=slow-max_ssg0p9v125c",
            "hold_corner=fast-min_ffg1p05vm40c",
            "parasitics=none_prelayout",
            "timing_exceptions_added=false", "pt_eco=false",
            "negative_slack_reported_not_hidden=true",
            'puts $summary_fp "setup_closed=[expr {$setup_slack >= 0.0}]"',
            'puts $summary_fp "hold_closed=[expr {$hold_slack >= 0.0}]"',
            "M1843_C2_FRESH_MAPPED_DUAL_CORNER_PT_INTERNAL_COMPLETE=PASS"):
        require(text, token, "pt")
    forbidden = re.compile(
        r"(^|\n)\s*(set_false_path|set_multicycle_path|set_min_delay|set_max_delay|set_disable_timing|set_case_analysis|fix_eco_timing)\b")
    if forbidden.search(text):
        raise CheckFailure("PT contains forbidden timing exception/ECO")
    for token in ("source draft", "DRAFT", "setup_slack < 0", "hold_slack < 0"):
        reject(text, token, "pt")


def check_contract(text, texts):
    value = strict_json_text(text)
    if (value.get("schema") != "m1843_m1834_c2_fresh_mapped_formality_dual_corner_pt_source_contract_r1_v1"
            or value.get("status") != "SOURCE_ONLY_M1843_C2_FRESH_MAPPED_FORMALITY_DUAL_CORNER_PT__NO_EDA_AUTHORIZED"):
        raise CheckFailure("formal contract identity/status")
    if value.get("authorization_now") != {
            "license_queries": 0, "attempts_created": 0,
            "formality_runs": 0, "pt_runs": 0, "all_other_eda_runs": 0,
            "results_created": 0, "releases_created": 0}:
        raise CheckFailure("author execution must be all zero")
    if value.get("future_execution_budget") != {
            "max_attempts": 1, "formality_runs_exact": 2,
            "pt_runs_exact": 2, "all_other_eda_runs": 0,
            "automatic_retry": False,
            "axis_order": ["K8", "K1X8"]}:
        raise CheckFailure("future execution budget drift")
    upstream = value.get("upstream_authority", {})
    for key, expected in (
            ("m1811_manifest_sha256", M1811_MANIFEST),
            ("m1811_outer_seal_file_sha256", M1811_OUTER),
            ("m1830_review_sha256", M1830_REVIEW),
            ("m1830_manifest_sha256", M1830_MANIFEST),
            ("m1830_outer_seal_file_sha256", M1830_OUTER)):
        if upstream.get(key) != expected:
            raise CheckFailure("upstream pin drift: " + key)
    failed_review = value.get("supersedes_failed_source_review", {})
    for key, expected in (
            ("m1834_review_sha256", M1834_FAILED_REVIEW),
            ("m1834_manifest_sha256", M1834_FAILED_MANIFEST),
            ("m1834_outer_seal_file_sha256", M1834_FAILED_OUTER)):
        if failed_review.get(key) != expected:
            raise CheckFailure("failed-review pin drift: " + key)
    reference = value.get("reference", {})
    if (reference.get("live_rtl_source_identity") != LIVE_RTL
            or reference.get("filelist_order") != list(LIVE_RTL)
            or reference.get("m1811_input_filelist_byte_exact") is not True):
        raise CheckFailure("live RTL/filelist identity contract drift")
    pt_semantics = value.get("prime_time_semantic_gate", {})
    if pt_semantics != {
            "required_reports": ["check_timing", "analysis_coverage",
                                 "constraint_violators", "exceptions",
                                 "design", "wire_load"],
            "parse_check_timing_success_and_warning_count": True,
            "parse_unconstrained_endpoint_count": True,
            "parse_setup_hold_all_checks_coverage": True,
            "parse_setup_hold_constraint_violation_counts": True,
            "publish_nonzero_counts_without_hiding": True}:
        raise CheckFailure("PrimeTime semantic gate drift")
    future = value.get("future_authority", {})
    for key in ("release_must_bind_m1844_review_sha256",
                "release_must_bind_m1844_manifest_sha256",
                "release_must_bind_m1844_outer_seal_file_sha256",
                "caller_environment_alone_is_not_authority"):
        if future.get(key) is not True:
            raise CheckFailure("future review/release binding drift: " + key)
    axes = value.get("axes", {})
    for axis in ("K8", "K1X8"):
        for key, digest in ARTIFACT_SHAS[axis].items():
            if axes.get(axis, {}).get(key) != digest:
                raise CheckFailure(axis + " artifact pin drift: " + key)
    expected_paths = {name: path for name, path in PATHS.items()
                      if name != "contract"}
    source_files = value.get("source_files", {})
    if set(source_files) != {path.relative_to(HW).as_posix()
                            for path in expected_paths.values()}:
        raise CheckFailure("formal source inventory drift")
    for path in expected_paths.values():
        rel = path.relative_to(HW).as_posix()
        expected = hashlib.sha256(texts[
            next(name for name, item in PATHS.items() if item == path)].encode()).hexdigest()
        if source_files[rel] != expected:
            raise CheckFailure("source SHA drift: " + rel)
    claims = value.get("claim_boundary", {})
    if claims.get("source_candidate") is not True:
        raise CheckFailure("source candidate missing")
    if any(v is not False for k, v in claims.items() if k != "source_candidate"):
        raise CheckFailure("source contract promotes claims")
    policy = value.get("timing_violation_policy", {})
    if policy != {
            "negative_setup_or_hold_is_reported": True,
            "negative_setup_or_hold_blocks_raw_result_publication": False,
            "timing_exceptions_added": False,
            "pt_eco": False,
            "hold_repair": False,
            "independent_result_review_required": True}:
        raise CheckFailure("timing violation policy drift")


def check_no_stale_draft(texts):
    joined = "\n".join(texts[name] for name in
                       ("runner", "formality", "pt", "contract"))
    for token in ("UNSEALED_SOURCE_DRAFT", "DRAFT_ONLY = True",
                  "source_files_unsealed", "m1833_", "m1835_", "m1845_"):
        reject(joined, token, "formal package stale-draft/legacy")


def check(overrides=None):
    texts = source_map(overrides)
    check_runner(texts["runner"])
    check_formality(texts["formality"])
    check_pt(texts["pt"])
    check_contract(texts["contract"], texts)
    check_no_stale_draft(texts)
    return {
        "status": "PASS_M1843_FORMAL_SOURCE_STATIC",
        "source_candidate": True,
        "launch_authorized": False,
        "eda_or_license_run": False,
        "future_formality_runs": 2,
        "future_pt_runs": 2,
        "negative_hold_reported_not_hidden": True,
    }


def main():
    try:
        result = check()
    except (CheckFailure, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
