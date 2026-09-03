#!/usr/bin/env python3
"""Different-author, source-only hammer for the sealed M1858 FM/PT successor."""
from __future__ import print_function

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
CHECKER = HW / "system_simulator/scripts/check_m1858_c2_fresh_mapped_formality_dual_corner_pt_source.py"
SPEC = importlib.util.spec_from_file_location("m1858_checker_for_m1859", str(CHECKER))
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)

CONTRACT = HW / "contracts/m1858_m1857_m1850_failure_c2_fresh_mapped_formality_dual_corner_pt_source_contract_r1_20260902.json"
AUTHOR = HW / "reviews/m1858_m1857_m1850_failure_c2_fresh_mapped_formality_dual_corner_pt_source_author_receipt_r1_20260902"
M1811 = HW / "dc_handoff/runs/m1811_m1810_m1809_c2_registered_fault_matched_two_axis_dc_r1_20260902"
M1830 = HW / "reviews/m1830_m1811_c2_registered_fault_matched_two_axis_dc_result_hammer_r1_20260902"
M1850_ATTEMPT = HW / "dc_handoff/runs/.m1850_m1811_c2_fresh_mapped_formality_dual_corner_pt_attempt_consumed"
M1850_FAILURE = HW / "dc_handoff/runs/m1850_m1811_c2_fresh_mapped_formality_dual_corner_pt_r1_20260902.failed_or_incomplete.2329292.quarantine"
M1857 = HW / "reviews/m1857_m1850_c2_formality_pt_failure_hammer_r1_20260902"
M1850_RUNNER = HW / "dc_handoff/scripts/run_m1850_c2_fresh_mapped_formality_dual_corner_pt_one_shot.py"
M1850_FM = HW / "dc_handoff/scripts/run_formality_m1850_m1809_c2_fresh_mapped_two_axis.tcl"
M1850_PT = HW / "dc_handoff/scripts/run_ptsta_m1850_m1809_c2_fresh_mapped_dual_corner.tcl"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "contract": "35bfefaeaeb96c498a27bfb0c02bf30c1c575824498933d2292f24b101cb74b1",
    "contract_sidecar": "6f1e7245bebbf74ea44e3cb8d6c88dfd23a775915823f841aaf261bf97e6325f",
    "contract_outer": "89049c1d8d818a6b61fadf21b7f3eab4276bb1776200654782b4dba945374313",
    "runner": "b115d0483516c67eb4c39dfe22f29a19f4c13ce65a26bcacdc2fe3d5125d2dee",
    "formality": "e43bbff07c3814595c6647e7b255ba89fd07857ecb1b63a95edec193c39e2d84",
    "pt": "7c47825d9cee09e4cf0c717c7a7e1a7bd018509f105954db4d3ad62835267fcc",
    "checker": "cba190fdb2e9f2587bbb140946aec2cd1d3d8dddc2787e3e2d482c881522b5c3",
    "test": "8b8e69cb1a0dbd7e8f8bc1aa2abc5bc6cc2b8a58ae72389e8801699d3a51a447",
    "author_receipt": "38f84ef7773478da739df7f64a34ec9712f9600cc1e79207828c07ea2708dcbf",
    "author_manifest": "58b0306888c994852e154793d1fd37df683ccc7c2d8bfa2df16496c46f7935a9",
    "author_outer": "9f10c5ab8241e0c13947063fd89a0d7a62c3ca197028a5a70638f5226a04bb68",
    "m1811_manifest": "695050260d54ca9b9d6f7b74d03021dd59afd642168981a13df0438e9fe12066",
    "m1811_outer": "04aa6bea4a06a8be3c441ddb984c68a046810a137fd2eca096adf513af0d324b",
    "m1830_review": "79e1885fad8ddac4ec0a6eee4d9034657761e778da384093fae5ab937f98f99b",
    "m1830_manifest": "d0ef8172f33378e9b025aab18043da19335fd9f00d1cd8d240bfb620997c0d06",
    "m1830_outer": "0b9dc1915096db8df6702e3ab5027d267fb99a3178bc2288a8b5625e611e343d",
    "m1850_attempt_json": "b1aebc0cc1b36aed6282329df09a5157f37922c004c87a12258075a60502c164",
    "m1850_attempt_manifest": "6d4dea4b938405f0945a83627cc34ece27dc3da26423bfa2dba9611e04e3c3d2",
    "m1850_attempt_outer": "c7bde092beae58c3a7af18d9dc579254319cc69eddc212ba43e54f8342424f75",
    "m1850_failure_terminal": "8980bdc0edbf5531eb2d384d9a6b5b978e0c96a23f4eadd10b08394b69842671",
    "m1850_failure_log": "2968d764f8f652876aa45454600fc631c207043d1402c6de2ed4f10430d63ce1",
    "m1850_failure_manifest": "228735dc3e20947d21bb0333e3370aa4c83680f9896c616d3851f086c40b5757",
    "m1850_failure_outer": "9edc6f31517dbac35e3e4f9108a06450c33466b0df43ba46c0ea16eed14e93b9",
    "m1857_review": "90f68f526c17052a65433adcf5a3f79d91a938f8c290b1b239e5a117ce062a26",
    "m1857_manifest": "f4b022970901ce9c1fd55f21157fcab091c5a31178bea865fd47ae6e3b8000ce",
    "m1857_outer": "7a7827bbccf416804d9dcf6b9e450f3674bc000ac793db2f013c2a2176eedcbd",
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


def sealed_directory(root, manifest_sha, outer_sha):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise HammerFailure("sealed directory absent/nonregular: " + str(root))
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    exact(manifest, manifest_sha)
    exact(outer, outer_sha)
    if outer.read_text() != manifest_sha + "  SHA256SUMS\n":
        raise HammerFailure("outer seal semantic drift: " + str(root))
    listed = {}
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        if len(fields) != 2 or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None:
            raise HammerFailure("manifest syntax: " + str(root))
        name = fields[1].lstrip("*")
        rel = Path(name)
        if name in listed or rel.is_absolute() or ".." in rel.parts:
            raise HammerFailure("unsafe/duplicate member: " + name)
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


def verify_file_seal(path, file_sha, sidecar_sha, outer_sha):
    path = Path(path)
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    exact(path, file_sha)
    exact(sidecar, sidecar_sha)
    exact(outer, outer_sha)
    if sidecar.read_text() != file_sha + "  " + path.name + "\n":
        raise HammerFailure("file sidecar semantic drift")
    if outer.read_text() != sidecar_sha + "  " + sidecar.name + "\n":
        raise HammerFailure("file outer semantic drift")


def verify_source_and_author():
    verify_file_seal(CONTRACT, EXPECTED["contract"], EXPECTED["contract_sidecar"],
                     EXPECTED["contract_outer"])
    for name in ("runner", "formality", "pt", "checker", "test"):
        exact(C.PATHS[name], EXPECTED[name])
    members = sealed_directory(AUTHOR, EXPECTED["author_manifest"], EXPECTED["author_outer"])
    if members.get("author_receipt.json") != EXPECTED["author_receipt"]:
        raise HammerFailure("M1858 author receipt member drift")
    receipt = strict(AUTHOR / "author_receipt.json")
    if (not str(receipt.get("status", "")).startswith("PASS_SOURCE_AUTHORING_ONLY_M1858")
            or receipt.get("authorization") != {
                "license_queries": 0, "attempts_created": 0, "formality_runs": 0,
                "pt_runs": 0, "all_other_eda_runs": 0, "results_created": 0,
                "releases_created": 0, "paper_claim_now": False}):
        raise HammerFailure("M1858 author receipt semantics")
    C.check()


def verify_failure_chain():
    exact(DOCS359, EXPECTED["docs359"])
    sealed_directory(M1811, EXPECTED["m1811_manifest"], EXPECTED["m1811_outer"])
    members = sealed_directory(M1830, EXPECTED["m1830_manifest"], EXPECTED["m1830_outer"])
    if members.get("review.json") != EXPECTED["m1830_review"]:
        raise HammerFailure("M1830 review member drift")
    members = sealed_directory(M1850_ATTEMPT, EXPECTED["m1850_attempt_manifest"],
                               EXPECTED["m1850_attempt_outer"])
    if members.get("attempt.json") != EXPECTED["m1850_attempt_json"]:
        raise HammerFailure("M1850 attempt member drift")
    members = sealed_directory(M1850_FAILURE, EXPECTED["m1850_failure_manifest"],
                               EXPECTED["m1850_failure_outer"])
    if (members.get("RUN_FAILED_OR_INCOMPLETE.txt") != EXPECTED["m1850_failure_terminal"]
            or members.get("k8/formality/formality.log") != EXPECTED["m1850_failure_log"]):
        raise HammerFailure("M1850 failure member drift")
    members = sealed_directory(M1857, EXPECTED["m1857_manifest"], EXPECTED["m1857_outer"])
    if members.get("review.json") != EXPECTED["m1857_review"]:
        raise HammerFailure("M1857 review member drift")
    review = strict(M1857 / "review.json")
    if (review.get("status") !=
            "PASS_M1857_INDEPENDENT_FAILURE_AUDIT__M1850_FORMALITY_PT_FAIL_CLOSED__P0_0_P1_1_P2_0__NO_RETRY__NO_EQUIVALENCE_OR_PT"
            or review.get("severity_counts") != {"p0": 0, "p1": 1, "p2": 0}
            or review.get("classification", {}).get("valid_compare_pair_established") is not False
            or review.get("execution_audit", {}).get("formality_processes") != 1
            or review.get("execution_audit", {}).get("pt_processes") != 0):
        raise HammerFailure("M1857 failure-review semantics")


def verify_identity_and_warning_sites():
    contract = strict(CONTRACT)
    review = strict(M1830 / "review.json")
    sources = review.get("source_identity", {}).get("sources", {})
    reference = contract.get("reference", {})
    rows = [row.strip() for row in (HW / reference["filelist"]).read_text().splitlines()
            if row.strip() and not row.lstrip().startswith("#")]
    if (len(rows) != 13 or len(set(rows)) != 13 or rows != list(sources)
            or rows != reference.get("filelist_order")
            or rows != list(reference.get("live_rtl_source_identity", {}))):
        raise HammerFailure("13 live RTL order/set drift")
    if (M1811 / "input_filelist.f").read_bytes() != (HW / reference["filelist"]).read_bytes():
        raise HammerFailure("M1811 input filelist differs byte-for-byte")
    for rel in rows:
        if sources[rel] != reference["live_rtl_source_identity"][rel]:
            raise HammerFailure("M1830/M1858 RTL digest disagreement: " + rel)
        exact(HW / rel, sources[rel])
    paths = []
    for axis in ("K8", "K1X8"):
        row = contract["axes"][axis]
        for path_key, sha_key in (("mapped_v", "mapped_v_sha256"),
                                  ("mapped_sdc", "mapped_sdc_sha256"),
                                  ("svf", "svf_sha256")):
            exact(HW / row[path_key], row[sha_key])
            paths.append((HW / row[path_key]).resolve())
    if len(set(paths)) != 6:
        raise HammerFailure("mapped artifacts shared/crossed")

    log = (M1850_FAILURE / "k8/formality/formality.log").read_text(errors="replace")
    pattern = re.compile(
        r"Signal: ([^ ]+) Block: ([^ ]+) File: " + re.escape(str(HW))
        + r"/([^ ]+) Line: ([0-9]+)\).*\(FMR_ELAB-147\)")
    sites = {(signal, block, rel, int(line))
             for signal, block, rel, line in pattern.findall(log)}
    if len(pattern.findall(log)) != 8 or len(sites) != 8:
        raise HammerFailure("M1850 warning-site identity/count drift")
    runner_sites = getattr(__import__("builtins"), "set")()
    runner_text = C.PATHS["runner"].read_text()
    for signal, block, rel, line in pattern.findall(log):
        tuple_anchor = '("%s", "%s",\n     "%s", %d)' % (
            signal, block, rel, int(line))
        if tuple_anchor not in runner_text:
            raise HammerFailure("M1850 warning site absent from M1858 frozen set: " + signal)
        runner_sites.add((signal, block, rel, int(line)))
    if len(runner_sites) != 8:
        raise HammerFailure("M1850/M1858 warning-site cardinality drift")

    # PT is byte-identical after namespace substitution. Formality differs only
    # by the five-line explanation plus the one exact warning demotion.
    normalized_pt = (C.PATHS["pt"].read_text().replace("M1858", "M1850")
                     .replace("M1859", "M1851").replace("M1860", "M1852"))
    if normalized_pt != M1850_PT.read_text():
        raise HammerFailure("PT semantics changed beyond namespace")
    fm = (C.PATHS["formality"].read_text().replace("M1858", "M1850")
          .replace("M1859", "M1851").replace("M1860", "M1852")
          .replace("m1858", "m1850"))
    permitted = (
        "# FMR_ELAB-147 marks Formailty's conservative possible-out-of-bounds RTL\n"
        "# interpretation as a mismatch by default.  The frozen RTL remains unchanged;\n"
        "# downgrade only this exact diagnostic to warning before reference set_top so\n"
        "# Formality can build compare points and prove or disprove the real mapping.\n"
        "set_mismatch_message_filter -warn FMR_ELAB-147\n")
    if fm.count(permitted) != 1 or fm.replace(permitted, "") != M1850_FM.read_text():
        raise HammerFailure("Formality changed beyond one authorized filter")


def synchronized_override(texts, name, old, new):
    if old not in texts[name]:
        raise HammerFailure("attack anchor absent: " + name + " / " + old[:60])
    changed = texts[name].replace(old, new, 1)
    contract = json.loads(texts["contract"])
    rel = C.PATHS[name].relative_to(HW).as_posix()
    contract["source_files"][rel] = text_sha(changed)
    return {name: changed, "contract": json.dumps(contract, sort_keys=True)}


def rejected(overrides):
    try:
        C.check(overrides)
    except (C.CheckFailure, SyntaxError):
        return True
    return False


def run_attacks():
    texts = C.source_map()
    results = []

    def attack(label, name, old, new, group):
        ok = rejected(synchronized_override(texts, name, old, new))
        results.append({"name": label, "group": group,
                        "result": "REJECTED" if ok else "ESCAPED"})

    # M1844's eight material escapes, now with source-inventory SHA synchronized.
    attack("second_authority_bypass", "runner",
           "current_release_sha, current_live_rtl_identity = verify_authority()",
           "current_release_sha, current_live_rtl_identity = (release_sha, live_rtl_identity)", "m1844")
    attack("unique_attempt_removed", "runner", "            write_attempt(release_sha)",
           "            # attempt removed", "m1844")
    attack("check_timing_bypass", "runner",
           'check_text.count("check_timing succeeded.") != 1', "False", "m1844")
    attack("coverage_conservation_bypass", "runner",
           'row["total"] != row["met"] + row["violated"] + row["untested"]', "False", "m1844")
    attack("coverage_rows_bypass", "runner",
           'set(coverage) != {"setup", "hold", "All Checks"}', "False", "m1844")
    attack("constraint_count_bypass", "runner",
           're.fullmatch(r"\\d+", constraint_values.get(key, "")) is None', "False", "m1844")
    attack("raw_constraint_visibility_bypass", "runner",
           "and raw_constraint_violation_marker_count == 0", "and False", "m1844")
    attack("hold_wns_rewrite", "pt", 'puts $summary_fp "hold_wns_ns=$hold_slack"',
           'puts $summary_fp "hold_wns_ns=0.0"', "m1844")

    attack("verify_formality_bypass", "runner",
           "passing = verify_formality(axis, fm_dir)", "passing = 1", "result")
    attack("verify_pt_bypass", "runner", "timing = verify_pt(axis, pt_dir)",
           "timing = {}", "result")

    attack("filter_removed", "formality",
           "set_mismatch_message_filter -warn FMR_ELAB-147", "# removed", "filter")
    pair = ("set_mismatch_message_filter -warn FMR_ELAB-147\n"
            "set_top r:/WORK/$reference_top -parameter $reference_elab_parameters")
    attack("filter_moved_after_set_top", "formality", pair,
           "set_top r:/WORK/$reference_top -parameter $reference_elab_parameters\n"
           "set_mismatch_message_filter -warn FMR_ELAB-147", "filter")
    attack("filter_duplicated", "formality",
           "set_mismatch_message_filter -warn FMR_ELAB-147",
           "set_mismatch_message_filter -warn FMR_ELAB-147\n"
           "set_mismatch_message_filter -warn FMR_ELAB-147", "filter")
    attack("other_message_id", "formality",
           "set_mismatch_message_filter -warn FMR_ELAB-147",
           "set_mismatch_message_filter -warn FMR_ELAB-999", "filter")
    attack("ignore_filter", "formality",
           "set_mismatch_message_filter -warn FMR_ELAB-147",
           "set_mismatch_message_filter -ignore FMR_ELAB-147", "filter")
    attack("suppress_filter", "formality",
           "set_mismatch_message_filter -warn FMR_ELAB-147",
           "set_mismatch_message_filter -suppress FMR_ELAB-147", "filter")

    attack("suppress_message_added", "formality", "quit",
           "suppress_message FM-999\nquit", "suppression")
    attack("set_message_info_added", "formality", "quit",
           "set_message_info -id FM-999 -limit 0\nquit", "suppression")
    attack("second_warn_filter_added", "formality", "quit",
           "set_mismatch_message_filter -warn FM-999\nquit", "suppression")
    attack("ignore_other_filter_added", "formality", "quit",
           "set_mismatch_message_filter -ignore FM-999\nquit", "suppression")

    attack("valid_pair_status_bypass", "runner", "status.count(token) != 1",
           "False", "formal_result")
    attack("positive_passing_bypass", "runner",
           're.search(r"[1-9][0-9]*\\s+Passing compare points", status)',
           "True", "formal_result")
    attack("failing_total_bypass", "runner",
           "failing_row is None or int(failing_row.group(1)) != 0",
           "False", "formal_result")
    attack("unmatched_failing_aborted_unverified_loop_removed", "runner",
           'for name, forbidden in (("formality_unmatched.rpt", "unmatched"),',
           'for name, forbidden in (("formality_unmatched.rpt", "never"),', "formal_result")
    attack("blackbox_pair_bypass", "runner",
           '"Reference and implementation designs are not set" in black_boxes',
           "False", "formal_result")
    attack("blackbox_nonzero_bypass", "runner",
           're.search(r"(?m)^\\s*(?:u|e|\\*)\\s+\\S+[\\s\\S]{0,180}?Instances\\s*:\\s*[1-9][0-9]*", black_boxes)',
           "False", "formal_result")

    attack("m1857_semantic_bypass", "runner",
           'failure_review.get("audit_status") != "PASS"', "False", "authority")
    attack("m1857_release_binding_removed", "runner",
           '"m1857_failure_review_manifest_sha256": M1857_MANIFEST_SHA,',
           "# binding removed", "authority")

    escaped = [row for row in results if row["result"] != "REJECTED"]
    return results, escaped


def verify_namespace():
    for path in (
            HW / "dc_handoff/runs/.m1858_m1811_c2_fresh_mapped_formality_dual_corner_pt_attempt_consumed",
            HW / "dc_handoff/runs/m1858_m1811_c2_fresh_mapped_formality_dual_corner_pt_r1_20260902",
            HW / "dc_handoff/runs/.m1858_m1811_c2_fresh_mapped_formality_dual_corner_pt_launch_lock",
            HW / "contracts/m1860_m1859_m1858_c2_fresh_mapped_formality_dual_corner_pt_launch_release_r1_20260902.json"):
        if os.path.lexists(str(path)):
            raise HammerFailure("unauthorized namespace exists: " + str(path))


def main():
    verify_source_and_author()
    verify_failure_chain()
    verify_identity_and_warning_sites()
    verify_namespace()
    results, escaped = run_attacks()
    counts = {}
    for row in results:
        counts[row["group"]] = counts.get(row["group"], 0) + (row["result"] == "REJECTED")
    output = {
        "status": ("PASS_M1859_INDEPENDENT_SOURCE_HAMMER" if not escaped
                   else "FAIL_M1859_INDEPENDENT_SOURCE_HAMMER_SEMANTIC_ESCAPES"),
        "attacks_total": len(results),
        "attacks_rejected": len(results) - len(escaped),
        "attacks_escaped": len(escaped),
        "rejected_by_group": counts,
        "live_rtl_rows_exact": 13,
        "mapped_axis_artifacts_exact_and_distinct": 6,
        "m1850_warning_sites_exact": 8,
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
        print(json.dumps({"status": "FAIL_M1859_INDEPENDENT_SOURCE_HAMMER",
                          "error": str(error)}, sort_keys=True))
        sys.exit(1)
