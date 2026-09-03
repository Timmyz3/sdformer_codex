#!/usr/bin/env python3
"""Static fail-closed checker for the formal M1877 source package."""
from __future__ import print_function

import hashlib
import ast
import io
import json
from pathlib import Path
import re
import sys
import tokenize


HW = Path(__file__).resolve().parents[2]
PATHS = {
    "runner": HW / "dc_handoff/scripts/run_m1877_c2_fresh_mapped_formality_dual_corner_pt_one_shot.py",
    "formality": HW / "dc_handoff/scripts/run_formality_m1877_m1809_c2_fresh_mapped_two_axis.tcl",
    "pt": HW / "dc_handoff/scripts/run_ptsta_m1877_m1809_c2_fresh_mapped_dual_corner.tcl",
    "checker": Path(__file__).resolve(),
    "test": HW / "system_simulator/tests/test_m1877_c2_fresh_mapped_formality_dual_corner_pt_source.py",
    "contract": HW / "contracts/m1877_m1873_m1858_failure_c2_fresh_mapped_formality_dual_corner_pt_source_contract_r1_20260902.json",
}
M1858_ATTEMPT = HW / "dc_handoff/runs/.m1858_m1811_c2_fresh_mapped_formality_dual_corner_pt_attempt_consumed"
M1858_FAILURE = HW / "dc_handoff/runs/m1858_m1811_c2_fresh_mapped_formality_dual_corner_pt_r1_20260902.failed_or_incomplete.2511659.quarantine"
M1873_REVIEW_DIR = HW / "reviews/m1873_m1858_c2_formality_pt_failure_hammer_r1_20260902"

BASE_TOP = "m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24"
M1811_MANIFEST = "695050260d54ca9b9d6f7b74d03021dd59afd642168981a13df0438e9fe12066"
M1811_OUTER = "04aa6bea4a06a8be3c441ddb984c68a046810a137fd2eca096adf513af0d324b"
M1830_REVIEW = "79e1885fad8ddac4ec0a6eee4d9034657761e778da384093fae5ab937f98f99b"
M1830_MANIFEST = "d0ef8172f33378e9b025aab18043da19335fd9f00d1cd8d240bfb620997c0d06"
M1830_OUTER = "0b9dc1915096db8df6702e3ab5027d267fb99a3178bc2288a8b5625e611e343d"
M1873_REVIEW = "f3aa0562e4d131acb40da226110f74b4aad93712bc8d4c4235b0e13595925178"
M1873_MANIFEST = "b1a444f1e9ac035800f23582daedebcb39ef157dff6343244f79390c8a439ee4"
M1873_OUTER = "ecf806e0b43c82c63972252a1c5dffa5aa06ef771d4cfa484cbee61c08d9ad98"
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

RUNNER_SEMANTIC_TOKEN_SHA256 = {
    "verify_authority": "168a5ccf2a825476430af1248ebca33311de7a9f31162dc1bdae57ed2d8b39d4",
    "parse_formality_black_box_entries": "74029f95665d8d9d52b79e8107664aa483d0e47c855f19522695dbda8b72a9bc",
    "verify_formality_black_box_policy": "ecb05355318e1014e8ec8982705c54de2b7e42dc47ee985465fdb6fd861ebe94",
    "verify_formality": "b0590482235cd14e34b08d486c30bb7cd387efadca3fd5a9a180820821a9a225",
    "parse_pt_semantics": "a2cb9f8a40a2b529ce96fd5f872e22efb00cca4f163503e5cce9dccb5c3b1508",
    "verify_pt": "fdeef1a10414eb0be1901e6e569f10523147ee5d3b4c2e5afba1e87dc9e1f6f7",
    "write_attempt": "52edc53b4a8de73e9a10d60fd335ca966f3fa53ad6250caf9c817bc7f7121e84",
    "execute": "613c3a4469e3c0f1afdde8a61b6dd5cf2f56a1a31067fdac1db9b474fcccf9c7",
}


class CheckFailure(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_bound_failure_evidence():
    expected = (
        (M1858_ATTEMPT / "attempt.json", "fcca3129f572bbfa85ea7f8e33951497f40d20a93a534a80d3a3a782aea33487"),
        (M1858_ATTEMPT / "SHA256SUMS", "1899bc129ade7b16da92a5e9c2be43e0a7a96af3c9d0e9e7d9ed25ff9056320e"),
        (M1858_ATTEMPT / "SHA256SUMS.seal.sha256", "87124c075d7dad34c93bd472d612db47fac1e81dd53a4ca85646aa130b9bfbb2"),
        (M1858_FAILURE / "RUN_FAILED_OR_INCOMPLETE.txt", "117e58207a8983cd984cb7da09b1c9e79bd692f089dae1f4cca1241e4c20c279"),
        (M1858_FAILURE / "k8/formality/formality.log", "7423f0e04b8d48adab4bb8da6257f2555255a6816aa616dee830c7fdfc897d3a"),
        (M1858_FAILURE / "k8/formality/reports/formality_status.rpt", "c1422264617dcd5bf05a3a5c0157a3147415694c4040f751cfb8ea90e0fe5b72"),
        (M1858_FAILURE / "k8/formality/reports/formality_black_boxes.rpt", "936295aecbf6d13d33ffe47ef996c7485315665f222f1c585714c9fc4b54ebf0"),
        (M1858_FAILURE / "SHA256SUMS", "82c363a4869af160a4d7ec0a1f1c6d9d8587a583ae9e43fbf19d6eb3acba366d"),
        (M1858_FAILURE / "SHA256SUMS.seal.sha256", "3c47ed5d552c73e401c219bfc511f7f5830ac986cb8cbcd386e0dd24fcbd4bc3"),
        (M1873_REVIEW_DIR / "review.json", M1873_REVIEW),
        (M1873_REVIEW_DIR / "SHA256SUMS", M1873_MANIFEST),
        (M1873_REVIEW_DIR / "SHA256SUMS.seal.sha256", M1873_OUTER),
    )
    for path, digest in expected:
        if not path.is_file() or path.is_symlink() or sha(path) != digest:
            raise CheckFailure("bound M1858/M1873 evidence drift: " + str(path))
    if (M1858_ATTEMPT / "SHA256SUMS.seal.sha256").read_text() != (
            "1899bc129ade7b16da92a5e9c2be43e0a7a96af3c9d0e9e7d9ed25ff9056320e  SHA256SUMS\n"):
        raise CheckFailure("M1858 attempt outer semantics")
    if (M1858_FAILURE / "SHA256SUMS.seal.sha256").read_text() != (
            "82c363a4869af160a4d7ec0a1f1c6d9d8587a583ae9e43fbf19d6eb3acba366d  SHA256SUMS\n"):
        raise CheckFailure("M1858 failure outer semantics")
    if (M1873_REVIEW_DIR / "SHA256SUMS.seal.sha256").read_text() != (
            M1873_MANIFEST + "  SHA256SUMS\n"):
        raise CheckFailure("M1873 review outer semantics")


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


def function_source(text, name):
    match = re.search(r"(?ms)^def " + re.escape(name)
                      + r"\([^\n]*\):\n.*?(?=^(?:def |class |if __name__))", text)
    if match is None:
        raise CheckFailure("runner function absent: " + name)
    return match.group(0).rstrip() + "\n"


def semantic_token_sha(text, name):
    ignored = set((tokenize.COMMENT, tokenize.NL, tokenize.ENCODING,
                   tokenize.ENDMARKER))
    rows = []
    for token in tokenize.generate_tokens(
            io.StringIO(function_source(text, name)).readline):
        if token.type not in ignored:
            rows.append(tokenize.tok_name[token.type] + ":" + token.string)
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def validate_runner_semantics(text):
    tree = ast.parse(text)
    functions = dict((node.name, node) for node in tree.body
                     if isinstance(node, ast.FunctionDef))
    for name, digest in RUNNER_SEMANTIC_TOKEN_SHA256.items():
        if (name not in functions or digest == "PENDING"
                or semantic_token_sha(text, name) != digest):
            raise CheckFailure("runner semantic function drift: " + name)
    for node in ast.walk(tree):
        if isinstance(node, ast.BoolOp):
            for value in node.values:
                if ((isinstance(value, ast.NameConstant) and value.value is False)
                        or (hasattr(ast, "Constant")
                            and isinstance(value, ast.Constant)
                            and value.value is False)):
                    raise CheckFailure("constant-false semantic bypass")


def check_runner(text):
    validate_runner_semantics(text)
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
        'M1811 / "k8/netlist/m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24_mapped.v"',
        'M1811 / "k1x8/netlist/m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24_mapped.v"',
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
        "M1877_EXPECTED_RUNNER_SHA256",
        "M1877_EXPECTED_SOURCE_CONTRACT_SHA256",
        'M1858_ATTEMPT_MANIFEST_SHA = "1899bc129ade7b16da92a5e9c2be43e0a7a96af3c9d0e9e7d9ed25ff9056320e"',
        'M1858_FAILURE_MANIFEST_SHA = "82c363a4869af160a4d7ec0a1f1c6d9d8587a583ae9e43fbf19d6eb3acba366d"',
        'M1873_REVIEW_SHA = "f3aa0562e4d131acb40da226110f74b4aad93712bc8d4c4235b0e13595925178"',
        'M1873_MANIFEST_SHA = "b1a444f1e9ac035800f23582daedebcb39ef157dff6343244f79390c8a439ee4"',
        'M1873_OUTER_SHA = "ecf806e0b43c82c63972252a1c5dffa5aa06ef771d4cfa484cbee61c08d9ad98"',
        "M1877_EXPECTED_M1878_SOURCE_REVIEW_SHA256",
        "M1877_EXPECTED_M1878_SOURCE_REVIEW_MANIFEST_SHA256",
        "M1877_EXPECTED_M1878_SOURCE_REVIEW_OUTER_SHA256",
        "M1877_EXPECTED_M1879_LAUNCH_RELEASE_SHA256",
        '"m1878_source_review_manifest_sha256": review_manifest',
        '"m1878_source_review_outer_seal_file_sha256": review_outer',
        '"m1873_failure_review_sha256": M1873_REVIEW_SHA',
        '"m1873_failure_review_manifest_sha256": M1873_MANIFEST_SHA',
        '"m1873_failure_review_outer_seal_file_sha256": M1873_OUTER_SHA',
        '"m1858_attempt_manifest_sha256": M1858_ATTEMPT_MANIFEST_SHA',
        '"m1858_failure_manifest_sha256": M1858_FAILURE_MANIFEST_SHA',
        "AUTHORIZE_ONE_M1877_C2_FRESH_MAPPED_FORMALITY_DUAL_CORNER_PT_ATTEMPT",
        '"max_attempts": 1', '"formality_runs": 2', '"pt_runs": 2',
        '"automatic_retry": False',
        "for axis in AXIS_ORDER:",
        "run_tool(FM_SHELL, FM_TCL, axis, fm_dir, \"formality.log\")",
        "run_tool(PT_SHELL, PT_TCL, axis, pt_dir, \"pt.log\")",
        "passing = verify_formality(axis, fm_dir)",
        "timing = verify_pt(axis, pt_dir)",
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
        "def parse_formality_black_box_entries(text):",
        "def verify_formality_black_box_policy(black_boxes, status):",
        'exact symmetric dual-side TECH SNPS_BUSHOLD pair absent',
        '"TECH", "i:/TCBN28HPCPLUSBWP35P140SSG0P9V125C"',
        '"TECH", "r:/TCBN28HPCPLUSBWP35P140SSG0P9V125C"',
        'prefix + "BHDBWP35P140/C0"',
        'prefix + "BHDBWP35P140#PWR/C2"',
        '"Formality BBPin compare-point total nonzero: " + label',
        "verify_formality_black_box_policy(black_boxes, status)",
    )
    for token in required:
        require(text, token, "runner")
    for token in ("DRAFT_ONLY", "DraftOnly", "UNSEALED_SOURCE_DRAFT",
                  "m1833", "m1835", "m1845", "automatic_retry\": True",
                  "hold_failure_blocks_result_publication\": True"):
        reject(text, token, "runner")
    if text.count('M1811 / "k8/netlist/m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24_mapped.v"') != 1:
        raise CheckFailure("runner K8 mapped path cardinality")
    if text.count('M1811 / "k1x8/netlist/m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24_mapped.v"') != 1:
        raise CheckFailure("runner K1x8 mapped path cardinality")
    if text.count("run_tool(FM_SHELL, FM_TCL, axis") != 1:
        raise CheckFailure("runner Formality call site cardinality")
    if text.count("run_tool(PT_SHELL, PT_TCL, axis") != 1:
        raise CheckFailure("runner PT call site cardinality")
    if len(re.findall(r"= verify_authority\(\)", text)) != 2:
        raise CheckFailure("runner authority call cardinality")
    if text.count("            write_attempt(release_sha)") != 1:
        raise CheckFailure("runner attempt-consumption call cardinality")
    if text.count("passing = verify_formality(axis, fm_dir)") != 1:
        raise CheckFailure("runner Formality-result call cardinality")
    if text.count("timing = verify_pt(axis, pt_dir)") != 1:
        raise CheckFailure("runner PT-result call cardinality")
    if text.index("verify_authority()") > text.index("write_attempt(release_sha)"):
        raise CheckFailure("authority must precede attempt")
    if text.index("license_gate()") > text.index("write_attempt(release_sha)"):
        raise CheckFailure("license gate must precede attempt")
    if text.index("write_attempt(release_sha)") > text.index(
            "run_tool(FM_SHELL, FM_TCL, axis, fm_dir"):
        raise CheckFailure("attempt must precede first tool call")


def check_formality(text):
    for token in (
            "M1877_REF_ELAB_PARAMETERS",
            'set expected_elab_parameters "ARCH_MODE=0"',
            'set expected_elab_parameters "ARCH_MODE=1"',
            'set expected_implementation_top "${base_top}_ARCH_MODE0"',
            'set expected_implementation_top "${base_top}_ARCH_MODE1"',
            "[llength $rtl_files] != 13",
            "set_svf $implementation_svf",
            "read_sverilog -r $reference_files",
            "set_mismatch_message_filter -warn FMR_ELAB-147",
            "set_top r:/WORK/$reference_top -parameter $reference_elab_parameters",
            "read_verilog -i $implementation_netlist",
            "set_top i:/WORK/$implementation_top",
            "set verification_succeeded [verify]",
            "report_unmatched_points", "report_failing_points",
            "report_aborted_points", "report_unverified_points",
            "M1877_C2_FRESH_MAPPED_FORMALITY_INTERNAL_COMPLETE=PASS"):
        require(text, token, "formality")
    for token in ("source draft", "DRAFT", "ARCH_MODE=2"):
        reject(text, token, "formality")
    filter_rows = [row.strip() for row in text.splitlines()
                   if "set_mismatch_message_filter" in row]
    if filter_rows != ["set_mismatch_message_filter -warn FMR_ELAB-147"]:
        raise CheckFailure("Formality mismatch filter must be exactly one warn FMR_ELAB-147")
    expected_order = ("set_mismatch_message_filter -warn FMR_ELAB-147\n"
                      "set_top r:/WORK/$reference_top -parameter $reference_elab_parameters")
    if text.count(expected_order) != 1:
        raise CheckFailure("FMR_ELAB-147 warning filter must immediately precede reference set_top")
    if re.search(r"(?im)^\s*(?:suppress_message|set_message_info)\b", text):
        raise CheckFailure("Formality contains broad message suppression")
    for token in ("set_mismatch_message_filter -ignore",
                  "set_mismatch_message_filter -suppress"):
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
            'puts $summary_fp "setup_wns_ns=$setup_slack"',
            'puts $summary_fp "hold_wns_ns=$hold_slack"',
            "M1877_C2_FRESH_MAPPED_DUAL_CORNER_PT_INTERNAL_COMPLETE=PASS"):
        require(text, token, "pt")
    forbidden = re.compile(
        r"(^|\n)\s*(set_false_path|set_multicycle_path|set_min_delay|set_max_delay|set_disable_timing|set_case_analysis|fix_eco_timing)\b")
    if forbidden.search(text):
        raise CheckFailure("PT contains forbidden timing exception/ECO")
    for token in ("source draft", "DRAFT", "setup_slack < 0", "hold_slack < 0"):
        reject(text, token, "pt")


def check_contract(text, texts):
    value = strict_json_text(text)
    if (value.get("schema") != "m1877_m1873_m1858_failure_c2_fresh_mapped_formality_dual_corner_pt_source_contract_r1_v1"
            or value.get("status") != "SOURCE_ONLY_M1877_C2_FRESH_MAPPED_FORMALITY_DUAL_CORNER_PT__NO_EDA_AUTHORIZED"):
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
    failed_review = value.get("m1873_failure_review", {})
    if failed_review != {
            "directory": "reviews/m1873_m1858_c2_formality_pt_failure_hammer_r1_20260902",
            "review_sha256": M1873_REVIEW,
            "manifest_sha256": M1873_MANIFEST,
            "outer_seal_file_sha256": M1873_OUTER,
            "audit_status": "PASS",
            "production_admission": "FAIL_CLOSED",
            "k8_raw_formality_diagnostic_succeeded": True,
            "k8_raw_formality_paper_citable": False}:
        raise CheckFailure("M1873 failure-review contract drift")
    if value.get("supersedes_failed_execution") != {
            "attempt_directory": "dc_handoff/runs/.m1858_m1811_c2_fresh_mapped_formality_dual_corner_pt_attempt_consumed",
            "attempt_json_sha256": "fcca3129f572bbfa85ea7f8e33951497f40d20a93a534a80d3a3a782aea33487",
            "attempt_manifest_sha256": "1899bc129ade7b16da92a5e9c2be43e0a7a96af3c9d0e9e7d9ed25ff9056320e",
            "attempt_outer_seal_file_sha256": "87124c075d7dad34c93bd472d612db47fac1e81dd53a4ca85646aa130b9bfbb2",
            "failure_quarantine": "dc_handoff/runs/m1858_m1811_c2_fresh_mapped_formality_dual_corner_pt_r1_20260902.failed_or_incomplete.2511659.quarantine",
            "failure_manifest_sha256": "82c363a4869af160a4d7ec0a1f1c6d9d8587a583ae9e43fbf19d6eb3acba366d",
            "failure_outer_seal_file_sha256": "3c47ed5d552c73e401c219bfc511f7f5830ac986cb8cbcd386e0dd24fcbd4bc3",
            "failure_terminal_sha256": "117e58207a8983cd984cb7da09b1c9e79bd692f089dae1f4cca1241e4c20c279",
            "k8_formality_log_sha256": "7423f0e04b8d48adab4bb8da6257f2555255a6816aa616dee830c7fdfc897d3a",
            "k8_formality_status_sha256": "c1422264617dcd5bf05a3a5c0157a3147415694c4040f751cfb8ea90e0fe5b72",
            "k8_formality_black_boxes_sha256": "936295aecbf6d13d33ffe47ef996c7485315665f222f1c585714c9fc4b54ebf0",
            "observed_fmr_elab_147_count": 8,
            "effective_compare_reached": True,
            "k8_passing_compare_points": 33656,
            "k8_bbpin": 0,
            "k8_pt_runs": 0,
            "k1x8_formality_runs": 0,
            "k1x8_pt_runs": 0,
            "retry_same_m1858_identity": False}:
        raise CheckFailure("M1858 consumed attempt/failure binding drift")
    if value.get("formality_message_policy") != {
            "allowed_filter_command": "set_mismatch_message_filter -warn FMR_ELAB-147",
            "command_count_exact": 1,
            "must_immediately_precede_reference_set_top": True,
            "all_other_message_filters_or_suppressions_forbidden": True,
            "verify_match_compare_and_result_parsing_unchanged_from_m1858": True}:
        raise CheckFailure("M1877 Formality message policy drift")
    if value.get("formality_black_box_policy") != {
            "section_aware": True,
            "design_library_nonzero_u_e_star_instances_required": 0,
            "passing_bbpin_required": 0,
            "failing_bbpin_required": 0,
            "technology_type_m_is_macro_not_design_black_box": True,
            "generic_technology_u_e_star_nonzero_allowed": False,
            "only_allowed_nonzero_technology_e_design": "SNPS_BUSHOLD",
            "required_sides": [
                "i:/TCBN28HPCPLUSBWP35P140SSG0P9V125C",
                "r:/TCBN28HPCPLUSBWP35P140SSG0P9V125C"],
            "instances_per_side": 2,
            "instances_total_per_side": 2,
            "exact_paths_per_side": [
                "BHDBWP35P140/C0", "BHDBWP35P140#PWR/C2"],
            "missing_or_extra_side_fails": True,
            "name_count_or_path_drift_fails": True,
            "zero_instance_starred_entries_only_ignored": True}:
        raise CheckFailure("M1877 section-aware black-box policy drift")
    if value.get("failure_chain") != {
            "m1858_attempt_manifest_sha256": "1899bc129ade7b16da92a5e9c2be43e0a7a96af3c9d0e9e7d9ed25ff9056320e",
            "m1858_attempt_outer_file_sha256": "87124c075d7dad34c93bd472d612db47fac1e81dd53a4ca85646aa130b9bfbb2",
            "m1858_failure_manifest_sha256": "82c363a4869af160a4d7ec0a1f1c6d9d8587a583ae9e43fbf19d6eb3acba366d",
            "m1858_failure_outer_file_sha256": "3c47ed5d552c73e401c219bfc511f7f5830ac986cb8cbcd386e0dd24fcbd4bc3",
            "m1873_review_sha256": M1873_REVIEW,
            "m1873_manifest_sha256": M1873_MANIFEST,
            "m1873_outer_file_sha256": M1873_OUTER,
            "m1873_severity_counts": {"p0": 0, "p1": 1, "p2": 0},
            "m1858_must_remain_immutable_and_unretried": True}:
        raise CheckFailure("M1858/M1873 failure chain drift")
    mutation = value.get("semantic_mutation_contract", {})
    if (mutation.get("source_files_sha_synchronized_per_attack") is not True
            or mutation.get("m1858_inherited_semantic_attacks_retained") is not True
            or mutation.get("fmr_elab_147_filter_order_and_cardinality_attacks") != 6
            or mutation.get("other_message_suppression_attacks") != 4
            or set(mutation.get("section_aware_black_box_attacks", [])) != {
                "remove_one_side", "add_one_side", "change_instance_count",
                "rename_snps_bushold", "change_instance_path",
                "nonzero_design_e", "nonzero_passing_bbpin",
                "nonzero_failing_bbpin", "generic_nonzero_tech_e"}
            or mutation.get("required_rejection") !=
            "ALL_MATERIAL_ATTACKS_BOTH_CPYTHON_3_6_AND_3_12"):
        raise CheckFailure("semantic mutation contract drift")
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
    if (future.get("m1873_failure_review") !=
            "reviews/m1873_m1858_c2_formality_pt_failure_hammer_r1_20260902"
            or future.get("m1873_failure_review_present_now") is not True
            or future.get("m1873_failure_review_sha256") != M1873_REVIEW
            or future.get("m1873_failure_review_manifest_sha256") != M1873_MANIFEST
            or future.get("m1873_failure_review_outer_seal_file_sha256") != M1873_OUTER
            or future.get("release_must_bind_m1873_review_manifest_outer_triplet") is not True):
        raise CheckFailure("M1873 failure-review authority drift")
    for key in ("release_must_bind_m1878_review_sha256",
                "release_must_bind_m1878_manifest_sha256",
                "release_must_bind_m1878_outer_seal_file_sha256",
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
    check_bound_failure_evidence()
    check_runner(texts["runner"])
    check_formality(texts["formality"])
    check_pt(texts["pt"])
    check_contract(texts["contract"], texts)
    check_no_stale_draft(texts)
    return {
        "status": "PASS_M1877_FORMAL_SOURCE_STATIC",
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
