#!/usr/bin/env python3
"""Different-author, source-only M1877 Formality/PT hammer.

This script is deliberately read-only.  It does not import or call execute(),
license_gate(), run_tool(), or any EDA executable.
"""
from __future__ import print_function

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import stat
import sys


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CHECKER = HW / "system_simulator/scripts/check_m1877_c2_fresh_mapped_formality_dual_corner_pt_source.py"
RUNNER = HW / "dc_handoff/scripts/run_m1877_c2_fresh_mapped_formality_dual_corner_pt_one_shot.py"
CONTRACT = HW / "contracts/m1877_m1873_m1858_failure_c2_fresh_mapped_formality_dual_corner_pt_source_contract_r1_20260902.json"
AUTHOR = HW / "reviews/m1877_m1873_m1858_failure_c2_fresh_mapped_formality_dual_corner_pt_source_author_receipt_r1_20260902"
M1858_ATTEMPT = HW / "dc_handoff/runs/.m1858_m1811_c2_fresh_mapped_formality_dual_corner_pt_attempt_consumed"
M1858_FAILURE = HW / "dc_handoff/runs/m1858_m1811_c2_fresh_mapped_formality_dual_corner_pt_r1_20260902.failed_or_incomplete.2511659.quarantine"
M1873 = HW / "reviews/m1873_m1858_c2_formality_pt_failure_hammer_r1_20260902"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "contract": "304f6eeceb7bd4a157efb15323d34e91d38ac52d9128eeb9418fe6910d0b2104",
    "contract_sidecar": "6ceb4bdac6db120fd5fc8932ff6cfca4ccec4f8cd640b33cb2a8253bd06e898d",
    "contract_outer": "58723fe9580397021317c9729e639dd65cfba021b5dd46f2b5c1b05fed3f8a5c",
    "runner": "e27a75adfd1febcfbbc32aa8def87ca785a225edaf861dcd0aa0c8a7d0822e87",
    "formality": "c47e5f8f5d5a68c32c47273ca9b82080c60e8e615e8ef3f5bc38a000e1a0741e",
    "pt": "1ffc1fc739faeafe75c64382794e14710f2ce40078099d2e5a1aa6cdba7426f2",
    "checker": "5b4c638fef7c6f82bc22e6f7bc03747228ea816bb02a362d762e859798dc9a40",
    "test": "b9e2bec19df25aa611ff31faa5e3975e2c50fe09b7292ae4cd1791c75f7ede13",
    "author_receipt": "972f51e94c6b0b60b900509c5016532fc8001571c19e4ab1e2771aed71ff0321",
    "author_manifest": "3e56a4710c5adefae4595f9c95160aa41d4ee2cc541a33896d3d6beb285accde",
    "author_outer": "3e8ca7afedf3f499b6d4e1d3f78ab2b524e11f268f33227273a33b66aeef08d0",
    "m1858_attempt_manifest": "1899bc129ade7b16da92a5e9c2be43e0a7a96af3c9d0e9e7d9ed25ff9056320e",
    "m1858_attempt_outer": "87124c075d7dad34c93bd472d612db47fac1e81dd53a4ca85646aa130b9bfbb2",
    "m1858_failure_manifest": "82c363a4869af160a4d7ec0a1f1c6d9d8587a583ae9e43fbf19d6eb3acba366d",
    "m1858_failure_outer": "3c47ed5d552c73e401c219bfc511f7f5830ac986cb8cbcd386e0dd24fcbd4bc3",
    "m1873_review": "f3aa0562e4d131acb40da226110f74b4aad93712bc8d4c4235b0e13595925178",
    "m1873_manifest": "b1a444f1e9ac035800f23582daedebcb39ef157dff6343244f79390c8a439ee4",
    "m1873_outer": "ecf806e0b43c82c63972252a1c5dffa5aa06ef771d4cfa484cbee61c08d9ad98",
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
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    if not root.is_dir() or root.is_symlink():
        raise HammerFailure("sealed directory absent/nonregular: " + str(root))
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


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = load(CHECKER, "m1877_checker_for_m1878")
R = load(RUNNER, "m1877_runner_for_m1878")


def verify_identity():
    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    exact(CONTRACT, EXPECTED["contract"])
    exact(sidecar, EXPECTED["contract_sidecar"])
    exact(outer, EXPECTED["contract_outer"])
    if sidecar.read_text() != EXPECTED["contract"] + "  " + CONTRACT.name + "\n":
        raise HammerFailure("contract sidecar semantics")
    if outer.read_text() != EXPECTED["contract_sidecar"] + "  " + sidecar.name + "\n":
        raise HammerFailure("contract outer semantics")
    for name in ("runner", "formality", "pt", "checker", "test"):
        exact(C.PATHS[name], EXPECTED[name])
    members = sealed_directory(AUTHOR, EXPECTED["author_manifest"], EXPECTED["author_outer"])
    if members.get("author_receipt.json") != EXPECTED["author_receipt"]:
        raise HammerFailure("author receipt member drift")
    members = sealed_directory(M1858_ATTEMPT, EXPECTED["m1858_attempt_manifest"],
                               EXPECTED["m1858_attempt_outer"])
    if members.get("attempt.json") != "fcca3129f572bbfa85ea7f8e33951497f40d20a93a534a80d3a3a782aea33487":
        raise HammerFailure("M1858 attempt identity drift")
    sealed_directory(M1858_FAILURE, EXPECTED["m1858_failure_manifest"],
                     EXPECTED["m1858_failure_outer"])
    members = sealed_directory(M1873, EXPECTED["m1873_manifest"], EXPECTED["m1873_outer"])
    if members.get("review.json") != EXPECTED["m1873_review"]:
        raise HammerFailure("M1873 review member drift")
    exact(DOCS359, EXPECTED["docs359"])
    review = strict(M1873 / "review.json")
    if (review.get("audit_status") != "PASS"
            or review.get("production_admission") != "FAIL_CLOSED"
            or review.get("severity_counts") != {"p0": 0, "p1": 1, "p2": 0}
            or review.get("classification", {}).get(
                "k8_raw_formality_verification_succeeded") is not True
            or review.get("classification", {}).get(
                "k8_raw_equivalence_paper_citable_or_production_admitted") is not False):
        raise HammerFailure("M1873 failure-review semantics")
    C.check()


def synchronized_override(texts, name, old, new):
    if old not in texts[name]:
        raise HammerFailure("attack anchor absent: " + name + " / " + old[:48])
    overrides = {name: texts[name].replace(old, new, 1)}
    if name in C.PATHS and name != "contract":
        contract = json.loads(texts["contract"])
        rel = C.PATHS[name].relative_to(HW).as_posix()
        contract["source_files"][rel] = hashlib.sha256(
            overrides[name].encode()).hexdigest()
        overrides["contract"] = json.dumps(contract, sort_keys=True)
    return overrides


def require_rejected(overrides, label):
    try:
        C.check(overrides)
    except (C.CheckFailure, SyntaxError):
        return
    raise HammerFailure("source mutation escaped: " + label)


def verify_source_mutations():
    texts = C.source_map()
    attacks = (
        ("runner", 'dangerous = {"u", "e", "*"}', 'dangerous = {"u", "*"}', "drop-tech/design-e"),
        ("runner", 'if row["section_kind"] == "DESIGN" and attrs.intersection(dangerous):',
         'if False:', "bypass-design-gate"),
        ("runner", 'if row["section_kind"] != "TECH":', 'if False:', "cross-section-gate"),
        ("runner", 'if len(bushold) != 2 or {row["library"] for row in bushold} != expected_libraries:',
         'if False:', "bypass-symmetric-pair"),
        ("runner", 'row["instances"] != 2 or row["instances_total"] != 2',
         'False', "bypass-instance-count"),
        ("runner", 'len(row["paths"]) != 2', 'False', "bypass-path-count"),
        ("runner", 'values[0] != 0', 'False', "bypass-bbpin-zero"),
        ("runner", 'verify_formality_black_box_policy(black_boxes, status)',
         '{}  # removed', "remove-blackbox-policy"),
        ("runner", 'AXIS_ORDER = ("K8", "K1X8")', 'AXIS_ORDER = ("K1X8", "K8")', "reverse-axis-order"),
        ("runner", 'run_tool(PT_SHELL, PT_TCL, axis, pt_dir, "pt.log")',
         '# removed', "remove-pt-axis"),
        ("runner", 'passing = verify_formality(axis, fm_dir)', 'passing = 1', "bypass-fm-result"),
        ("runner", 'timing = verify_pt(axis, pt_dir)', 'timing = {}', "bypass-pt-result"),
        ("runner", '            write_attempt(release_sha)', '            pass', "remove-attempt"),
        ("runner", 'current_release_sha, current_live_rtl_identity = verify_authority()',
         'current_release_sha, current_live_rtl_identity = (release_sha, live_rtl_identity)',
         "remove-pre-attempt-revalidation"),
        ("formality", 'set_mismatch_message_filter -warn FMR_ELAB-147',
         'set_mismatch_message_filter -ignore FMR_ELAB-147', "ignore-fmr147"),
        ("formality", 'set verification_succeeded [verify]',
         'set verification_succeeded true', "bypass-verify"),
        ("formality", 'report_unmatched_points > "$output_dir/reports/formality_unmatched.rpt"',
         '# unmatched report removed', "remove-unmatched-report"),
        ("pt", 'puts $summary_fp "hold_wns_ns=$hold_slack"',
         'puts $summary_fp "hold_wns_ns=0.0"', "hide-hold"),
        ("pt", 'set_min_library $std_slow_db -min_version $std_fast_db',
         '# removed', "remove-min-library"),
        ("pt", 'report_analysis_coverage -status_details untested',
         '# removed', "remove-coverage"),
        ("pt", 'M1877_C2_FRESH_MAPPED_DUAL_CORNER_PT_INTERNAL_COMPLETE=PASS',
         'M1877_C2_FRESH_MAPPED_DUAL_CORNER_PT_INTERNAL_COMPLETE=FAIL', "forge-pt-terminal"),
    )
    for name, old, new, label in attacks:
        require_rejected(synchronized_override(texts, name, old, new), label)
    return len(attacks)


def verify_black_box_fixtures():
    reports = M1858_FAILURE / "k8/formality/reports"
    black_boxes = (reports / "formality_black_boxes.rpt").read_text()
    status = (reports / "formality_status.rpt").read_text()
    base = R.verify_formality_black_box_policy(black_boxes, status)
    if base != {"parsed_entries": 138,
                "exact_symmetric_snps_bushold_entries": 2,
                "passing_bbpin": 0, "failing_bbpin": 0}:
        raise HammerFailure("actual M1858 black-box parse drift")
    parsed = R.parse_formality_black_box_entries(black_boxes)
    tech_m_nonzero = [row for row in parsed
                      if row["section_kind"] == "TECH"
                      and row["attributes"] == ["m"] and row["instances"] > 0]
    if len(tech_m_nonzero) != 2:
        raise HammerFailure("TECH type-m macro distinction drift")

    ref = ("e      SNPS_BUSHOLD\n\n       Instances : 2 of 2\n"
           "       ------------------------\n"
           "       r:/TCBN28HPCPLUSBWP35P140SSG0P9V125C/BHDBWP35P140/C0\n"
           "       r:/TCBN28HPCPLUSBWP35P140SSG0P9V125C/BHDBWP35P140#PWR/C2\n")
    imp = ref.replace("r:/", "i:/")
    marker = "####    DESIGN LIBRARY - r:/WORK\n"
    design_e = (marker + "##################################################################\n"
                "Type  Design Name\n----  ----------\n"
                "e      ATTACK_DESIGN\n\n       Instances : 1 of 1\n"
                "       ------------------------\n       r:/WORK/TOP/U_ATTACK\n\n")
    tech_e = ("e      GENERIC_TECH_EMPTY\n\n       Instances : 1 of 1\n"
              "       ------------------------\n"
              "       i:/TCBN28HPCPLUSBWP35P140SSG0P9V125C/GENERIC/X\n\n"
              "e      SNPS_BUSHOLD\n")
    fixtures = (
        (black_boxes.replace(ref, "", 1), status, "remove-one-side"),
        (black_boxes.replace(imp, imp + "\n" + imp, 1), status, "add-one-side"),
        (black_boxes.replace("Instances : 2 of 2", "Instances : 1 of 2", 1), status, "count"),
        (black_boxes.replace("e      SNPS_BUSHOLD", "e      SNPS_OTHER", 1), status, "rename"),
        (black_boxes.replace("BHDBWP35P140/C0", "BHDBWP35P140/C1", 1), status, "path"),
        (black_boxes.replace(marker, design_e, 1), status, "design-e"),
        (black_boxes, status.replace("Passing (equivalent)           0",
                                    "Passing (equivalent)           1", 1), "passing-bbpin"),
        (black_boxes, status.replace("Failing (not equivalent)       0",
                                    "Failing (not equivalent)       1", 1), "failing-bbpin"),
        (black_boxes.replace("e      SNPS_BUSHOLD\n", tech_e, 1), status, "generic-tech-e"),
    )
    for mutated_report, mutated_status, label in fixtures:
        try:
            R.verify_formality_black_box_policy(mutated_report, mutated_status)
        except R.M1877Error:
            continue
        raise HammerFailure("black-box fixture escaped: " + label)
    return len(fixtures), len(tech_m_nonzero)


def main():
    verify_identity()
    source_mutations = verify_source_mutations()
    fixtures, tech_m = verify_black_box_fixtures()
    result = {
        "status": "PASS_M1878_INDEPENDENT_HAMMER",
        "official_source_checker": "PASS_M1877_FORMAL_SOURCE_STATIC",
        "source_mutations_rejected": source_mutations,
        "source_mutations_escaped": 0,
        "black_box_negative_fixtures_rejected": fixtures,
        "black_box_negative_fixtures_escaped": 0,
        "actual_black_box_entries": 138,
        "exact_dual_side_snps_bushold_entries": 2,
        "nonzero_tech_type_m_macro_entries": tech_m,
        "passing_bbpin": 0,
        "failing_bbpin": 0,
        "docs359_sha256": EXPECTED["docs359"],
        "eda_or_license_run": False,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (HammerFailure, C.CheckFailure, R.M1877Error, OSError, ValueError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True))
        sys.exit(1)
