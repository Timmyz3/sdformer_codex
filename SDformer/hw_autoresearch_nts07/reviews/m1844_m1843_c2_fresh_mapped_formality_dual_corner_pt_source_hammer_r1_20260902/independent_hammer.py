#!/usr/bin/env python3
"""Independent read-only hammer for the sealed M1843 C2 FM/PT source."""
from __future__ import print_function

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CHECKER = HW / "system_simulator/scripts/check_m1843_c2_fresh_mapped_formality_dual_corner_pt_source.py"
SPEC = importlib.util.spec_from_file_location("m1843_checker_for_m1844", str(CHECKER))
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)

CONTRACT = HW / "contracts/m1843_m1834_c2_fresh_mapped_formality_dual_corner_pt_source_contract_r1_20260902.json"
CONTRACT_SIDECAR = Path(str(CONTRACT) + ".sha256")
CONTRACT_OUTER = Path(str(CONTRACT) + ".sha256.seal.sha256")
AUTHOR = HW / "reviews/m1843_m1834_c2_fresh_mapped_formality_dual_corner_pt_source_author_receipt_r1_20260902"
M1834 = HW / "reviews/m1834_m1832_c2_fresh_mapped_formality_dual_corner_pt_source_hammer_r1_20260902"
M1830 = HW / "reviews/m1830_m1811_c2_registered_fault_matched_two_axis_dc_result_hammer_r1_20260902"
M1811 = HW / "dc_handoff/runs/m1811_m1810_m1809_c2_registered_fault_matched_two_axis_dc_r1_20260902"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "contract": "c644edaa5a269f7f69f9d6dd76b556568bc9365071a2c1f04db0e3ca3d4cc9e9",
    "contract_sidecar": "df1c0fa8eb673261c605bfcfe66c758fe3890b5b17a1d8dcd10f771068ed7a49",
    "contract_outer": "54802ef449ca4a90974ce4e678365e4ea21e3a1d6e44c21c62b681cd16852ebc",
    "runner": "2c45ca351aaf364843865c348215d784b6a74d68a0f52aeb64aa396e397b5947",
    "formality": "2895d92c73170dcafc5520ff4ef0f0c6c1872ff4f10b2a306d1b416be4731315",
    "pt": "aa9d1391d4fb34be16a520c4bf6b9e7d5b801641b35449faf6a07003e248943d",
    "checker": "e4eda1455afb9612fc3cc3319a9a01685859f7bd9386a6f146265d11407a8233",
    "test": "b565dee770ca2b8159e6a8b11c0064a3f39e14eadb64d37d309b188b472462bf",
    "author_receipt": "3bcbb1c8508a6d1daa7bf8498175a955d8ebedce964f70d9cad736b98dc7c1c3",
    "author_manifest": "083684247336c6540515304db528d93ee3da05b1590ac0b2788b07baf952f004",
    "author_outer": "ccca9520c65770d0fce18949537d40d8515dae95c7040ced7260c6eab83881d1",
    "m1834_review": "510133974d005a4259279966dff2f29205b077facfcb8ef798e608eddb4be33d",
    "m1834_manifest": "78e89c2022ecde5cfdb437ee4196c31a176992428649580be3f3866b786b56ea",
    "m1834_outer": "126c85936ed69c8904f35823a11d292720f0a49de5ff5ea2ca28ab6b6b6247df",
    "m1830_review": "79e1885fad8ddac4ec0a6eee4d9034657761e778da384093fae5ab937f98f99b",
    "m1830_manifest": "d0ef8172f33378e9b025aab18043da19335fd9f00d1cd8d240bfb620997c0d06",
    "m1830_outer": "0b9dc1915096db8df6702e3ab5027d267fb99a3178bc2288a8b5625e611e343d",
    "m1811_manifest": "695050260d54ca9b9d6f7b74d03021dd59afd642168981a13df0438e9fe12066",
    "m1811_outer": "04aa6bea4a06a8be3c441ddb984c68a046810a137fd2eca096adf513af0d324b",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


class HammerFailure(RuntimeError):
    pass


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def exact(path, expected):
    path = Path(path)
    if (not path.is_file() or path.is_symlink()
            or not stat.S_ISREG(path.lstat().st_mode) or sha(path) != expected):
        raise HammerFailure("identity drift: " + str(path))


def sealed_directory(root, manifest_sha, outer_sha):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise HammerFailure("sealed directory absent/nonregular: " + str(root))
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    exact(manifest, manifest_sha)
    exact(outer, outer_sha)
    if outer.read_text().split() != [manifest_sha, "SHA256SUMS"]:
        raise HammerFailure("outer seal semantic drift: " + str(root))
    listed = {}
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        if len(fields) != 2 or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None:
            raise HammerFailure("manifest syntax: " + str(root))
        name = fields[1].lstrip("*")
        rel = Path(name)
        if name in listed or rel.is_absolute() or ".." in rel.parts:
            raise HammerFailure("unsafe/duplicate manifest member: " + name)
        exact(root / rel, fields[0])
        listed[name] = fields[0]
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise HammerFailure("symlink in sealed directory: " + str(path))
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            actual.add(path.relative_to(root).as_posix())
    if set(listed) != actual:
        raise HammerFailure("sealed population drift: " + str(root))
    return listed


def verify_file_double_seal():
    exact(CONTRACT, EXPECTED["contract"])
    exact(CONTRACT_SIDECAR, EXPECTED["contract_sidecar"])
    exact(CONTRACT_OUTER, EXPECTED["contract_outer"])
    if CONTRACT_SIDECAR.read_text() != EXPECTED["contract"] + "  " + CONTRACT.name + "\n":
        raise HammerFailure("contract sidecar semantic drift")
    if CONTRACT_OUTER.read_text() != EXPECTED["contract_sidecar"] + "  " + CONTRACT_SIDECAR.name + "\n":
        raise HammerFailure("contract outer seal semantic drift")


def strict(path):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise HammerFailure("duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          HammerFailure("nonfinite JSON: " + token)))


def verify_fixed_evidence():
    exact(DOCS359, EXPECTED["docs359"])
    sealed_directory(M1811, EXPECTED["m1811_manifest"], EXPECTED["m1811_outer"])
    m1830_members = sealed_directory(M1830, EXPECTED["m1830_manifest"], EXPECTED["m1830_outer"])
    if m1830_members.get("review.json") != EXPECTED["m1830_review"]:
        raise HammerFailure("M1830 review member drift")
    m1834_members = sealed_directory(M1834, EXPECTED["m1834_manifest"], EXPECTED["m1834_outer"])
    if m1834_members.get("review.json") != EXPECTED["m1834_review"]:
        raise HammerFailure("M1834 review member drift")
    failed = strict(M1834 / "review.json")
    if (failed.get("status") != "FAIL_M1834_M1832_C2_FRESH_MAPPED_FORMALITY_DUAL_CORNER_PT_SOURCE_HAMMER__P0_0_P1_2_P2_1__NO_AUTHORIZATION"
            or failed.get("p0_count") != 0 or failed.get("p1_count") != 2
            or failed.get("p2_count") != 1
            or {row.get("id") for row in failed.get("findings", [])} != {
                "P1-LIVE-RTL-IDENTITY", "P1-RELEASE-REVIEW-SEAL-BINDING",
                "P2-PT-REPORT-SEMANTIC-GATE"}):
        raise HammerFailure("M1834 failure diagnosis drift")


def verify_author_and_sources():
    members = sealed_directory(AUTHOR, EXPECTED["author_manifest"], EXPECTED["author_outer"])
    if members.get("author_receipt.json") != EXPECTED["author_receipt"]:
        raise HammerFailure("M1843 author receipt member drift")
    receipt = strict(AUTHOR / "author_receipt.json")
    if (receipt.get("status") != "PASS_SOURCE_AUTHORING_ONLY_M1843_C2_FRESH_MAPPED_FORMALITY_DUAL_CORNER_PT__NO_EDA_RUN"
            or receipt.get("static_validation", {}).get("negative_mutations_rejected_per_runtime") != 47
            or receipt.get("authorization", {}).get("all_eda_now") is not False
            or receipt.get("authorization", {}).get("paper_claim_now") is not False):
        raise HammerFailure("M1843 author receipt semantic drift")
    for name, key in (("runner", "runner"), ("formality", "formality"),
                      ("pt", "pt"), ("checker", "checker"), ("test", "test")):
        exact(C.PATHS[name], EXPECTED[key])


def verify_contract_and_live_identity():
    value = strict(CONTRACT)
    C.check()
    if value.get("authorization_now") != {
            "license_queries": 0, "attempts_created": 0, "formality_runs": 0,
            "pt_runs": 0, "all_other_eda_runs": 0, "results_created": 0,
            "releases_created": 0}:
        raise HammerFailure("source authoring executed work")
    if value.get("future_execution_budget") != {
            "max_attempts": 1, "formality_runs_exact": 2, "pt_runs_exact": 2,
            "all_other_eda_runs": 0, "automatic_retry": False,
            "axis_order": ["K8", "K1X8"]}:
        raise HammerFailure("future execution budget drift")
    future = value.get("future_authority", {})
    if (future.get("source_review_status") !=
            "PASS_M1844_M1843_C2_FRESH_MAPPED_FORMALITY_DUAL_CORNER_PT_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_ATTEMPT"
            or future.get("launch_release_status") !=
            "AUTHORIZE_ONE_M1843_C2_FRESH_MAPPED_FORMALITY_DUAL_CORNER_PT_ATTEMPT"
            or future.get("release_must_bind_m1844_review_sha256") is not True
            or future.get("release_must_bind_m1844_manifest_sha256") is not True
            or future.get("release_must_bind_m1844_outer_seal_file_sha256") is not True):
        raise HammerFailure("future review/release authority drift")
    review = strict(M1830 / "review.json")
    source_identity = review.get("source_identity", {})
    sources = source_identity.get("sources", {})
    rows = [line.strip() for line in (HW / value["reference"]["filelist"]).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if (len(rows) != 13 or len(set(rows)) != 13 or rows != list(sources)
            or rows != list(value["reference"]["live_rtl_source_identity"])):
        raise HammerFailure("13-row live RTL order/set drift")
    if (M1811 / "input_filelist.f").read_bytes() != (HW / value["reference"]["filelist"]).read_bytes():
        raise HammerFailure("M1811 filelist not byte exact")
    for rel in rows:
        if sources[rel] != value["reference"]["live_rtl_source_identity"][rel]:
            raise HammerFailure("M1830/M1843 source digest disagreement: " + rel)
        exact(HW / rel, sources[rel])
    artifact_paths = []
    for axis in ("K8", "K1X8"):
        row = value["axes"][axis]
        for path_key, sha_key in (("mapped_v", "mapped_v_sha256"),
                                  ("mapped_sdc", "mapped_sdc_sha256"),
                                  ("svf", "svf_sha256")):
            exact(HW / row[path_key], row[sha_key])
            artifact_paths.append((HW / row[path_key]).resolve())
    if len(set(artifact_paths)) != 6:
        raise HammerFailure("cross-axis mapped artifact sharing")


def synchronized_override(texts, name, old, new):
    if old not in texts[name]:
        raise HammerFailure("attack anchor absent: " + name + " / " + old[:48])
    changed = texts[name].replace(old, new, 1)
    contract = json.loads(texts["contract"])
    rel = C.PATHS[name].relative_to(HW).as_posix()
    contract["source_files"][rel] = text_sha(changed)
    return {name: changed, "contract": json.dumps(contract, sort_keys=True)}


def rejected(overrides):
    try:
        C.check(overrides)
    except C.CheckFailure:
        return True
    return False


def run_attacks():
    texts = C.source_map()
    results = []

    def source_attack(label, name, old, new):
        ok = rejected(synchronized_override(texts, name, old, new))
        results.append({"name": label, "result": "REJECTED" if ok else "ESCAPED"})

    def contract_attack(label, mutate):
        value = json.loads(texts["contract"])
        mutate(value)
        ok = rejected({"contract": json.dumps(value, sort_keys=True)})
        results.append({"name": label, "result": "REJECTED" if ok else "ESCAPED"})

    # Replay the exact three M1834 findings with source-inventory SHA updated,
    # so rejection cannot be credited merely to stale inventory hashes.
    source_attack("m1834_live_rtl_exact_check_removed", "runner",
                  "exact_regular(HW / rel, sources[rel])", "# exact live RTL check removed")
    source_attack("m1834_release_manifest_binding_removed", "runner",
                  '"m1844_source_review_manifest_sha256": review_manifest,',
                  "# review manifest binding removed")
    source_attack("m1834_release_outer_binding_removed", "runner",
                  '"m1844_source_review_outer_seal_file_sha256": review_outer,',
                  "# review outer binding removed")
    source_attack("m1834_pt_exceptions_report_removed", "runner",
                  'reports / "exceptions.rpt", reports / "design.rpt",',
                  'reports / "design.rpt",')
    source_attack("m1834_pt_design_report_removed", "runner",
                  'reports / "exceptions.rpt", reports / "design.rpt",',
                  'reports / "exceptions.rpt",')
    source_attack("m1834_pt_wireload_report_removed", "runner",
                  'reports / "wire_load.rpt",', "# wire load report removed")
    source_attack("m1834_pt_semantics_call_removed", "runner",
                  "summary.update(parse_pt_semantics(reports))", "# PT semantics removed")

    runner_mutations = [
        ("m1811_manifest", C.M1811_MANIFEST, "0" * 64),
        ("m1811_outer", C.M1811_OUTER, "0" * 64),
        ("m1830_review", C.M1830_REVIEW, "0" * 64),
        ("m1830_manifest", C.M1830_MANIFEST, "0" * 64),
        ("m1830_outer", C.M1830_OUTER, "0" * 64),
        ("filelist_order", "rows != list(sources.keys())", "False"),
        ("filelist_byte", "M1811_INPUT_FILELIST.read_bytes() != REFERENCE_FILELIST.read_bytes()", "False"),
        ("authority_before_attempt", "release_sha, live_rtl_identity = verify_authority()", "release_sha, live_rtl_identity = ('0'*64, {})"),
        ("second_authority", "current_release_sha, current_live_rtl_identity = verify_authority()", "current_release_sha, current_live_rtl_identity = (release_sha, live_rtl_identity)"),
        ("attempt_removed", "write_attempt(release_sha)", "# attempt consumption removed"),
        ("fm_call", 'run_tool(FM_SHELL, FM_TCL, axis, fm_dir, "formality.log")', "# FM removed"),
        ("pt_call", 'run_tool(PT_SHELL, PT_TCL, axis, pt_dir, "pt.log")', "# PT removed"),
        ("fm_verify", "passing = verify_formality(axis, fm_dir)", "passing = 1"),
        ("pt_verify", "timing = verify_pt(axis, pt_dir)", "timing = {}"),
        ("check_timing_unique", 'check_text.count("check_timing succeeded.") != 1', "False"),
        ("coverage_conservation", 'row["total"] != row["met"] + row["violated"] + row["untested"]', "False"),
        ("coverage_rows", 'set(coverage) != {"setup", "hold", "All Checks"}', "False"),
        ("constraint_machine_count", 're.fullmatch(r"\\d+", constraint_values.get(key, "")) is None', "False"),
        ("constraint_raw_visibility", 'and raw_constraint_violation_marker_count == 0', "and False"),
        ("negative_hold_hidden", '"negative_slack_reported_not_hidden": True', '"negative_slack_reported_not_hidden": False'),
        ("retry_enabled", '"automatic_retry": False', '"automatic_retry": True'),
        ("hold_blocks_raw", '"hold_failure_blocks_result_publication": False', '"hold_failure_blocks_result_publication": True'),
        ("system_claim", '"system_speedup": False', '"system_speedup": True'),
        ("paper_claim", '"paper_ppa_ready": False', '"paper_ppa_ready": True'),
    ]
    for label, old, new in runner_mutations:
        source_attack("runner_" + label, "runner", old, new)

    for axis in ("K8", "K1X8"):
        for key, digest in C.ARTIFACT_SHAS[axis].items():
            source_attack("runner_%s_%s" % (axis.lower(), key), "runner", digest, "0" * 64)

    formality_mutations = [
        ("verify", "set verification_succeeded [verify]", "set verification_succeeded true"),
        ("svf", "set_svf $implementation_svf", "# SVF removed"),
        ("fresh_reference", "read_sverilog -r $reference_files", "# fresh RTL removed"),
        ("reference_parameter", "set_top r:/WORK/$reference_top -parameter $reference_elab_parameters", "set_top r:/WORK/$reference_top"),
        ("implementation_top", "set_top i:/WORK/$implementation_top", "set_top i:/WORK/$base_top"),
        ("unmatched_report", "report_unmatched_points", "# unmatched report removed"),
        ("failing_report", "report_failing_points", "# failing report removed"),
        ("aborted_report", "report_aborted_points", "# aborted report removed"),
        ("unverified_report", "report_unverified_points", "# unverified report removed"),
    ]
    for label, old, new in formality_mutations:
        source_attack("formality_" + label, "formality", old, new)

    pt_mutations = [
        ("min_library", "set_min_library $std_slow_db -min_version $std_fast_db", "# min library removed"),
        ("ocv", "set_operating_conditions -analysis_type on_chip_variation", "set_operating_conditions"),
        ("setup_report", "report_timing -delay_type max", "report_timing"),
        ("hold_report", "report_timing -delay_type min", "report_timing"),
        ("coverage", "report_analysis_coverage -status_details untested", "# coverage removed"),
        ("constraint", "report_constraint -all_violators", "# constraints removed"),
        ("exceptions", "report_exceptions -ignored", "# exceptions removed"),
        ("design", "report_design", "# design removed"),
        ("wireload", "report_wire_load", "# wire load removed"),
        ("setup_count", 'puts $constraint_fp "setup_violating_paths=[sizeof_collection $setup_violators]"', 'puts $constraint_fp "setup_violating_paths=0"'),
        ("hold_count", 'puts $constraint_fp "hold_violating_paths=[sizeof_collection $hold_violators]"', 'puts $constraint_fp "hold_violating_paths=0"'),
        ("hold_slack", 'puts $summary_fp "hold_wns_ns=$hold_slack"', 'puts $summary_fp "hold_wns_ns=0.0"'),
        ("false_path", "quit", "set_false_path -from A -to B\nquit"),
        ("multicycle", "quit", "set_multicycle_path 2\nquit"),
        ("eco", "quit", "fix_eco_timing -type hold\nquit"),
    ]
    for label, old, new in pt_mutations:
        source_attack("pt_" + label, "pt", old, new)

    contract_attack("contract_author_eda", lambda v: v["authorization_now"].update(pt_runs=1))
    contract_attack("contract_third_pt", lambda v: v["future_execution_budget"].update(pt_runs_exact=3))
    contract_attack("contract_retry", lambda v: v["future_execution_budget"].update(automatic_retry=True))
    contract_attack("contract_claim_formality", lambda v: v["claim_boundary"].update(formality=True))
    contract_attack("contract_claim_system", lambda v: v["claim_boundary"].update(system_speedup=True))
    contract_attack("contract_hide_hold", lambda v: v["timing_violation_policy"].update(negative_setup_or_hold_is_reported=False))
    contract_attack("contract_release_manifest_false", lambda v: v["future_authority"].update(release_must_bind_m1844_manifest_sha256=False))
    contract_attack("contract_release_outer_false", lambda v: v["future_authority"].update(release_must_bind_m1844_outer_seal_file_sha256=False))
    contract_attack("contract_failed_review_drift", lambda v: v["supersedes_failed_source_review"].update(m1834_review_sha256="0" * 64))
    contract_attack("contract_live_rtl_drift", lambda v: v["reference"]["live_rtl_source_identity"].update({next(iter(v["reference"]["live_rtl_source_identity"])): "0" * 64}))

    escaped = [row for row in results if row["result"] != "REJECTED"]
    return results, escaped


def verify_namespace_boundary():
    # HERE is the expected in-progress review. No attempt, result, release,
    # launch lock, EDA or license action is authorized by this source review.
    for path in (HW / "dc_handoff/runs/.m1843_m1811_c2_fresh_mapped_formality_dual_corner_pt_attempt_consumed",
                 HW / "dc_handoff/runs/m1843_m1811_c2_fresh_mapped_formality_dual_corner_pt_r1_20260902",
                 HW / "dc_handoff/runs/.m1843_m1811_c2_fresh_mapped_formality_dual_corner_pt_launch_lock",
                 HW / "contracts/m1846_m1844_m1843_c2_fresh_mapped_formality_dual_corner_pt_launch_release_r1_20260902.json"):
        if os.path.lexists(str(path)):
            raise HammerFailure("unauthorized namespace exists: " + str(path))


def main():
    verify_file_double_seal()
    verify_fixed_evidence()
    verify_author_and_sources()
    verify_contract_and_live_identity()
    verify_namespace_boundary()
    results, escaped = run_attacks()
    output = {
        "status": ("PASS_M1844_INDEPENDENT_SOURCE_HAMMER" if not escaped
                   else "FAIL_M1844_INDEPENDENT_SOURCE_HAMMER_SEMANTIC_ESCAPES"),
        "attacks_total": len(results),
        "attacks_rejected": len(results) - len(escaped),
        "attacks_escaped": len(escaped),
        "m1834_findings_replayed": 3,
        "m1834_concrete_source_attacks": 7,
        "live_rtl_rows_exact": 13,
        "mapped_axis_artifacts_exact_and_distinct": 6,
        "license_or_eda_run": False,
        "attempt_or_result_created": False,
        "release_created": False,
    }
    print(json.dumps(output, sort_keys=True))
    if escaped:
        print(json.dumps({"escaped_attacks": [row["name"] for row in escaped]},
                         sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (HammerFailure, C.CheckFailure, OSError, ValueError,
            json.JSONDecodeError) as error:
        print(json.dumps({"status": "FAIL_M1844_INDEPENDENT_SOURCE_HAMMER",
                          "error": str(error)}, sort_keys=True))
        sys.exit(1)
