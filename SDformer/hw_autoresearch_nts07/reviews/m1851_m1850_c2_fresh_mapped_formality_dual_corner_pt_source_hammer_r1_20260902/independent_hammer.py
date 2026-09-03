#!/usr/bin/env python3
"""Different-author, read-only hammer for the sealed M1850 FM/PT source."""
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
CHECKER = HW / "system_simulator/scripts/check_m1850_c2_fresh_mapped_formality_dual_corner_pt_source.py"
SPEC = importlib.util.spec_from_file_location("m1850_checker_for_m1851", str(CHECKER))
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)

CONTRACT = HW / "contracts/m1850_m1844_m1843_m1834_m1832_c2_fresh_mapped_formality_dual_corner_pt_source_contract_r1_20260902.json"
AUTHOR = HW / "reviews/m1850_m1844_m1843_m1834_m1832_c2_fresh_mapped_formality_dual_corner_pt_source_author_receipt_r1_20260902"
M1811 = HW / "dc_handoff/runs/m1811_m1810_m1809_c2_registered_fault_matched_two_axis_dc_r1_20260902"
M1830 = HW / "reviews/m1830_m1811_c2_registered_fault_matched_two_axis_dc_result_hammer_r1_20260902"
M1832 = HW / "reviews/m1832_c2_fresh_mapped_formality_dual_corner_pt_source_author_receipt_r1_20260902"
M1834 = HW / "reviews/m1834_m1832_c2_fresh_mapped_formality_dual_corner_pt_source_hammer_r1_20260902"
M1843_CONTRACT = HW / "contracts/m1843_m1834_c2_fresh_mapped_formality_dual_corner_pt_source_contract_r1_20260902.json"
M1843_AUTHOR = HW / "reviews/m1843_m1834_c2_fresh_mapped_formality_dual_corner_pt_source_author_receipt_r1_20260902"
M1844 = HW / "reviews/m1844_m1843_c2_fresh_mapped_formality_dual_corner_pt_source_hammer_r1_20260902"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "contract": "844b008c8b8f548edbca8dbe7d715a33e1ff8ddb941a5e2e14ec2a59482c4a38",
    "contract_sidecar": "add03245ad42fe2b4ee2a1be853b0e832db3a4e10abf8d5de6b13c35513d697b",
    "contract_outer": "6ef2db1edc5106c412b04318ef3056db42c3e55182d2237553f5f6ea0763cc76",
    "runner": "448a085629b8b246b197b637787a916272d0dae7b4c476247cba48501837385e",
    "formality": "dca08813afd35bced464740073c6c7a810887e8fe512ad1a0af9ba9b9ebbb366",
    "pt": "4a249b393961ebd587f79910c6b21d954ea360b3cff1ad7cf91b8daa0173a06c",
    "checker": "bc68ba918aabb9bb0d1251f353cf773ce411411797c75c9bf9c45d8a0906cbdb",
    "test": "eed5e25843c161e915d261649e8af6c7b1b236733676bc3f1dd7c3666b09c38a",
    "author_receipt": "c21ed819dc2ef84f9890c708225d8dc88f522d4d818653087b95435b0820391f",
    "author_manifest": "25c78f66229e819ef5ce8ec2578091a3a38d9e3f697f0949d07102b16182ccb3",
    "author_outer": "de66393476ab032213fd821e20a0fc295008c701ceed95fe424f153090ac6b3d",
    "m1811_manifest": "695050260d54ca9b9d6f7b74d03021dd59afd642168981a13df0438e9fe12066",
    "m1811_outer": "04aa6bea4a06a8be3c441ddb984c68a046810a137fd2eca096adf513af0d324b",
    "m1830_review": "79e1885fad8ddac4ec0a6eee4d9034657761e778da384093fae5ab937f98f99b",
    "m1830_manifest": "d0ef8172f33378e9b025aab18043da19335fd9f00d1cd8d240bfb620997c0d06",
    "m1830_outer": "0b9dc1915096db8df6702e3ab5027d267fb99a3178bc2288a8b5625e611e343d",
    "m1832_receipt": "cdbd31376cfc6e0c72ca241bd27a82836487e3ff8e48c106d676ae8b8c9e4e89",
    "m1832_manifest": "39e5519d29d30dce536c3769984bd4b82a22d2da8c23b30b3767a4030c20a08b",
    "m1832_outer": "726820d1cafe55e53eb04d8d587eb0df69b4f64c8fba414f0bffe1026ca6d143",
    "m1834_review": "510133974d005a4259279966dff2f29205b077facfcb8ef798e608eddb4be33d",
    "m1834_manifest": "78e89c2022ecde5cfdb437ee4196c31a176992428649580be3f3866b786b56ea",
    "m1834_outer": "126c85936ed69c8904f35823a11d292720f0a49de5ff5ea2ca28ab6b6b6247df",
    "m1843_contract": "c644edaa5a269f7f69f9d6dd76b556568bc9365071a2c1f04db0e3ca3d4cc9e9",
    "m1843_sidecar": "df1c0fa8eb673261c605bfcfe66c758fe3890b5b17a1d8dcd10f771068ed7a49",
    "m1843_outer": "54802ef449ca4a90974ce4e678365e4ea21e3a1d6e44c21c62b681cd16852ebc",
    "m1843_receipt": "3bcbb1c8508a6d1daa7bf8498175a955d8ebedce964f70d9cad736b98dc7c1c3",
    "m1843_manifest": "083684247336c6540515304db528d93ee3da05b1590ac0b2788b07baf952f004",
    "m1843_author_outer": "ccca9520c65770d0fce18949537d40d8515dae95c7040ced7260c6eab83881d1",
    "m1844_review": "692aa653f2115287ab20c3b045c24805297c9ff0a58ba095993c0d273010181a",
    "m1844_manifest": "132fae48ca7c8118b21fbfedf600585240a5f3eb9325d7d2155c19f667a5f149",
    "m1844_outer": "a26ad87e35872a3f5eb70a4edc7d74e4823fae48ad2b60d5cb71170740e30b0c",
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


def verify_contract_and_sources():
    exact(CONTRACT, EXPECTED["contract"])
    exact(Path(str(CONTRACT) + ".sha256"), EXPECTED["contract_sidecar"])
    exact(Path(str(CONTRACT) + ".sha256.seal.sha256"), EXPECTED["contract_outer"])
    if Path(str(CONTRACT) + ".sha256").read_text() != EXPECTED["contract"] + "  " + CONTRACT.name + "\n":
        raise HammerFailure("M1850 contract sidecar semantics")
    if Path(str(CONTRACT) + ".sha256.seal.sha256").read_text() != EXPECTED["contract_sidecar"] + "  " + CONTRACT.name + ".sha256\n":
        raise HammerFailure("M1850 contract outer semantics")
    for name in ("runner", "formality", "pt", "checker", "test"):
        exact(C.PATHS[name], EXPECTED[name])
    members = sealed_directory(AUTHOR, EXPECTED["author_manifest"], EXPECTED["author_outer"])
    if members.get("author_receipt.json") != EXPECTED["author_receipt"]:
        raise HammerFailure("M1850 author receipt member drift")
    receipt = strict(AUTHOR / "author_receipt.json")
    if (receipt.get("status") != "PASS_SOURCE_AUTHORING_ONLY_M1850_C2_FRESH_MAPPED_FORMALITY_DUAL_CORNER_PT__M1851_REVIEW_M1852_RELEASE_REQUIRED__NO_EDA_RUN"
            or receipt.get("authorization", {}).get("formality_runs") != 0
            or receipt.get("authorization", {}).get("pt_runs") != 0
            or receipt.get("authorization", {}).get("attempts_created") != 0):
        raise HammerFailure("M1850 author receipt semantics")
    C.check()


def verify_failure_and_upstream_chain():
    exact(DOCS359, EXPECTED["docs359"])
    sealed_directory(M1811, EXPECTED["m1811_manifest"], EXPECTED["m1811_outer"])
    members = sealed_directory(M1830, EXPECTED["m1830_manifest"], EXPECTED["m1830_outer"])
    if members.get("review.json") != EXPECTED["m1830_review"]:
        raise HammerFailure("M1830 review member drift")
    members = sealed_directory(M1832, EXPECTED["m1832_manifest"], EXPECTED["m1832_outer"])
    if members.get("author_receipt.json") != EXPECTED["m1832_receipt"]:
        raise HammerFailure("M1832 receipt member drift")
    members = sealed_directory(M1834, EXPECTED["m1834_manifest"], EXPECTED["m1834_outer"])
    if members.get("review.json") != EXPECTED["m1834_review"]:
        raise HammerFailure("M1834 review member drift")
    failed1834 = strict(M1834 / "review.json")
    if (failed1834.get("p0_count"), failed1834.get("p1_count"), failed1834.get("p2_count")) != (0, 2, 1):
        raise HammerFailure("M1834 failure counts drift")
    exact(M1843_CONTRACT, EXPECTED["m1843_contract"])
    exact(Path(str(M1843_CONTRACT) + ".sha256"), EXPECTED["m1843_sidecar"])
    exact(Path(str(M1843_CONTRACT) + ".sha256.seal.sha256"), EXPECTED["m1843_outer"])
    members = sealed_directory(M1843_AUTHOR, EXPECTED["m1843_manifest"], EXPECTED["m1843_author_outer"])
    if members.get("author_receipt.json") != EXPECTED["m1843_receipt"]:
        raise HammerFailure("M1843 receipt member drift")
    members = sealed_directory(M1844, EXPECTED["m1844_manifest"], EXPECTED["m1844_outer"])
    if members.get("review.json") != EXPECTED["m1844_review"]:
        raise HammerFailure("M1844 review member drift")
    failed1844 = strict(M1844 / "review.json")
    if (failed1844.get("p0_count"), failed1844.get("p1_count"), failed1844.get("p2_count")) != (0, 1, 0):
        raise HammerFailure("M1844 failure counts drift")
    contract = strict(CONTRACT)
    if contract.get("failure_chain", {}).get("m1844_review_sha256") != EXPECTED["m1844_review"]:
        raise HammerFailure("M1850 does not bind M1844 failure")
    future = contract.get("future_authority", {})
    if (future.get("source_review_status") != "PASS_M1851_M1850_C2_FRESH_MAPPED_FORMALITY_DUAL_CORNER_PT_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_ATTEMPT"
            or future.get("launch_release_status") != "AUTHORIZE_ONE_M1850_C2_FRESH_MAPPED_FORMALITY_DUAL_CORNER_PT_ATTEMPT"
            or not all(future.get(key) is True for key in (
                "release_must_bind_m1851_review_sha256",
                "release_must_bind_m1851_manifest_sha256",
                "release_must_bind_m1851_outer_seal_file_sha256"))):
        raise HammerFailure("M1851/M1852 future authority drift")


def verify_live_identity():
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
        raise HammerFailure("M1811 filelist not byte exact")
    for rel in rows:
        if sources[rel] != reference["live_rtl_source_identity"][rel]:
            raise HammerFailure("M1830/M1850 RTL digest disagreement: " + rel)
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
        raise HammerFailure("mapped artifacts shared across axes")


def synchronized_override(texts, name, old, new):
    if old not in texts[name]:
        raise HammerFailure("attack anchor absent: " + name + " / " + old[:50])
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

    def attack(label, name, old, new, material=True):
        ok = rejected(synchronized_override(texts, name, old, new))
        results.append({"name": label, "material": material,
                        "result": "REJECTED" if ok else "ESCAPED"})

    # Exact eight material escapes diagnosed by M1844, replayed with the
    # source inventory SHA synchronized so inventory staleness cannot help.
    attack("m1844_second_authority_bypass", "runner",
           "current_release_sha, current_live_rtl_identity = verify_authority()",
           "current_release_sha, current_live_rtl_identity = (release_sha, live_rtl_identity)")
    attack("m1844_unique_attempt_removed", "runner",
           "            write_attempt(release_sha)",
           "            # attempt consumption removed")
    attack("m1844_check_timing_uniqueness_bypass", "runner",
           'check_text.count("check_timing succeeded.") != 1', "False")
    attack("m1844_coverage_conservation_bypass", "runner",
           'row["total"] != row["met"] + row["violated"] + row["untested"]', "False")
    attack("m1844_exact_coverage_rows_bypass", "runner",
           'set(coverage) != {"setup", "hold", "All Checks"}', "False")
    attack("m1844_constraint_machine_count_bypass", "runner",
           're.fullmatch(r"\\d+", constraint_values.get(key, "")) is None', "False")
    attack("m1844_raw_constraint_visibility_bypass", "runner",
           "and raw_constraint_violation_marker_count == 0", "and False")
    attack("m1844_verbatim_hold_wns_rewrite", "pt",
           'puts $summary_fp "hold_wns_ns=$hold_slack"',
           'puts $summary_fp "hold_wns_ns=0.0"')

    # The two additional result-binding attacks required by the M1850
    # mutation contract.
    attack("formality_result_verification_bypass", "runner",
           "passing = verify_formality(axis, fm_dir)", "passing = 1")
    attack("pt_result_verification_bypass", "runner",
           "timing = verify_pt(axis, pt_dir)", "timing = {}")

    # Additional ordering/raw-truth/source-authority attacks.
    attack("first_authority_bypass", "runner",
           "release_sha, live_rtl_identity = verify_authority()",
           "release_sha, live_rtl_identity = ('0'*64, {})")
    attack("live_rtl_exact_check_removed", "runner",
           "exact_regular(HW / rel, sources[rel])", "# live RTL exact check removed")
    attack("setup_wns_rewrite", "pt",
           'puts $summary_fp "setup_wns_ns=$setup_slack"',
           'puts $summary_fp "setup_wns_ns=0.0"')
    attack("pt_semantic_parser_call_removed", "runner",
           "summary.update(parse_pt_semantics(reports))", "# semantic parser removed")
    attack("pt_exception_report_gate_removed", "runner",
           'reports / "exceptions.rpt", reports / "design.rpt",',
           'reports / "design.rpt",')
    attack("release_review_manifest_binding_removed", "runner",
           '"m1851_source_review_manifest_sha256": review_manifest,',
           "# review manifest binding removed")
    attack("release_review_outer_binding_removed", "runner",
           '"m1851_source_review_outer_seal_file_sha256": review_outer,',
           "# review outer binding removed")
    attack("pt_false_path_added", "pt", "quit", "set_false_path -from A -to B\nquit")
    attack("pt_multicycle_added", "pt", "quit", "set_multicycle_path 2\nquit")
    attack("pt_eco_added", "pt", "quit", "fix_eco_timing -type hold\nquit")

    escaped = [row for row in results if row["result"] != "REJECTED"]
    return results, escaped


def verify_namespace():
    for path in (
            HW / "dc_handoff/runs/.m1850_m1811_c2_fresh_mapped_formality_dual_corner_pt_attempt_consumed",
            HW / "dc_handoff/runs/m1850_m1811_c2_fresh_mapped_formality_dual_corner_pt_r1_20260902",
            HW / "dc_handoff/runs/.m1850_m1811_c2_fresh_mapped_formality_dual_corner_pt_launch_lock",
            HW / "contracts/m1852_m1851_m1850_c2_fresh_mapped_formality_dual_corner_pt_launch_release_r1_20260902.json"):
        if os.path.lexists(str(path)):
            raise HammerFailure("unauthorized namespace exists: " + str(path))


def main():
    verify_contract_and_sources()
    verify_failure_and_upstream_chain()
    verify_live_identity()
    verify_namespace()
    results, escaped = run_attacks()
    output = {
        "status": ("PASS_M1851_INDEPENDENT_SOURCE_HAMMER" if not escaped
                   else "FAIL_M1851_INDEPENDENT_SOURCE_HAMMER_SEMANTIC_ESCAPES"),
        "attacks_total": len(results),
        "attacks_rejected": len(results) - len(escaped),
        "attacks_escaped": len(escaped),
        "m1844_material_attacks_rejected": sum(
            row["result"] == "REJECTED" for row in results[:8]),
        "additional_result_binding_attacks_rejected": sum(
            row["result"] == "REJECTED" for row in results[8:10]),
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
        print(json.dumps({"status": "FAIL_M1851_INDEPENDENT_SOURCE_HAMMER",
                          "error": str(error)}, sort_keys=True))
        sys.exit(1)
