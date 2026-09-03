#!/usr/bin/env python3
"""Read-only M1883 audit of the exact M1879 launch release.

This program deliberately imports M1877 only to call verify_authority().  It
never calls execute(), resource_gate(), license_gate(), or any EDA program.
All release mutations are in-memory and cannot alter the sealed authority.
"""
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


HW = Path(__file__).resolve().parents[2]
RUNNER = HW / "dc_handoff/scripts/run_m1877_c2_fresh_mapped_formality_dual_corner_pt_one_shot.py"
CONTRACT = HW / "contracts/m1877_m1873_m1858_failure_c2_fresh_mapped_formality_dual_corner_pt_source_contract_r1_20260902.json"
AUTHOR = HW / "reviews/m1877_m1873_m1858_failure_c2_fresh_mapped_formality_dual_corner_pt_source_author_receipt_r1_20260902"
REVIEW1878 = HW / "reviews/m1878_m1877_c2_fresh_mapped_formality_dual_corner_pt_source_hammer_r1_20260902"
REVIEW1873 = HW / "reviews/m1873_m1858_c2_formality_pt_failure_hammer_r1_20260902"
RELEASE = HW / "contracts/m1879_m1878_m1877_c2_fresh_mapped_formality_dual_corner_pt_launch_release_r1_20260902.json"
M1858_ATTEMPT = HW / "dc_handoff/runs/.m1858_m1811_c2_fresh_mapped_formality_dual_corner_pt_attempt_consumed"
M1858_FAILURE = HW / "dc_handoff/runs/m1858_m1811_c2_fresh_mapped_formality_dual_corner_pt_r1_20260902.failed_or_incomplete.2511659.quarantine"
M1811 = HW / "dc_handoff/runs/m1811_m1810_m1809_c2_registered_fault_matched_two_axis_dc_r1_20260902"
M1830 = HW / "reviews/m1830_m1811_c2_registered_fault_matched_two_axis_dc_result_hammer_r1_20260902"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
RUNS = HW / "dc_handoff/runs"

SHAS = {
    "runner": "e27a75adfd1febcfbbc32aa8def87ca785a225edaf861dcd0aa0c8a7d0822e87",
    "contract": "304f6eeceb7bd4a157efb15323d34e91d38ac52d9128eeb9418fe6910d0b2104",
    "contract_sidecar_file": "6ceb4bdac6db120fd5fc8932ff6cfca4ccec4f8cd640b33cb2a8253bd06e898d",
    "contract_outer_file": "58723fe9580397021317c9729e639dd65cfba021b5dd46f2b5c1b05fed3f8a5c",
    "author_receipt": "972f51e94c6b0b60b900509c5016532fc8001571c19e4ab1e2771aed71ff0321",
    "author_manifest": "3e56a4710c5adefae4595f9c95160aa41d4ee2cc541a33896d3d6beb285accde",
    "author_outer": "3e8ca7afedf3f499b6d4e1d3f78ab2b524e11f268f33227273a33b66aeef08d0",
    "m1878_review": "c2f16b3cadab1c0cd08047dbd8c8b1b7025d4af0cd627945e6c228df0fa5dbee",
    "m1878_manifest": "ff9289f5e9e98ee2a1560cc0cb754a09331c1770c878d9befd4dda71a3c856c7",
    "m1878_outer": "a5936fe5b99e16f3e7d44e0efd36ce11cae3aa4716b0b3da47214758a091cb27",
    "m1873_review": "f3aa0562e4d131acb40da226110f74b4aad93712bc8d4c4235b0e13595925178",
    "m1873_manifest": "b1a444f1e9ac035800f23582daedebcb39ef157dff6343244f79390c8a439ee4",
    "m1873_outer": "ecf806e0b43c82c63972252a1c5dffa5aa06ef771d4cfa484cbee61c08d9ad98",
    "m1858_attempt_manifest": "1899bc129ade7b16da92a5e9c2be43e0a7a96af3c9d0e9e7d9ed25ff9056320e",
    "m1858_attempt_outer": "87124c075d7dad34c93bd472d612db47fac1e81dd53a4ca85646aa130b9bfbb2",
    "m1858_attempt_json": "fcca3129f572bbfa85ea7f8e33951497f40d20a93a534a80d3a3a782aea33487",
    "m1858_failure_manifest": "82c363a4869af160a4d7ec0a1f1c6d9d8587a583ae9e43fbf19d6eb3acba366d",
    "m1858_failure_outer": "3c47ed5d552c73e401c219bfc511f7f5830ac986cb8cbcd386e0dd24fcbd4bc3",
    "m1858_failure_terminal": "117e58207a8983cd984cb7da09b1c9e79bd692f089dae1f4cca1241e4c20c279",
    "m1811_manifest": "695050260d54ca9b9d6f7b74d03021dd59afd642168981a13df0438e9fe12066",
    "m1811_outer": "04aa6bea4a06a8be3c441ddb984c68a046810a137fd2eca096adf513af0d324b",
    "m1830_review": "79e1885fad8ddac4ec0a6eee4d9034657761e778da384093fae5ab937f98f99b",
    "m1830_manifest": "d0ef8172f33378e9b025aab18043da19335fd9f00d1cd8d240bfb620997c0d06",
    "m1830_outer": "0b9dc1915096db8df6702e3ab5027d267fb99a3178bc2288a8b5625e611e343d",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "release": "4e8dc963c7c3527040be59338b380ff76d3ff561d0e5e9eb0f619456e3d25fd3",
    "release_sidecar_file": "b85016917eaa10f643abd5cdbfafc44610af26098dfef735f9415faba2553b62",
    "release_outer_file": "ca2beca79c3c44fdd732337ec1ed8a109c5ea39aed0e2dd13e8067dee8cdfd11",
}


class AuditError(RuntimeError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_regular(path, digest):
    path = Path(path)
    if (not path.is_file() or path.is_symlink()
            or not stat.S_ISREG(path.lstat().st_mode)
            or re.fullmatch(r"[0-9a-f]{64}", digest or "") is None
            or sha256(path) != digest):
        raise AuditError("identity mismatch: " + str(path))


def strict_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise AuditError("duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          AuditError("nonfinite JSON token: " + token)))


def parse_manifest(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise AuditError("malformed manifest row: " + line)
        rel = match.group(2)
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise AuditError("unsafe manifest path: " + rel)
        rows.append((match.group(1), rel))
    if not rows or len(rows) != len(set(rel for _, rel in rows)):
        raise AuditError("empty/duplicate manifest")
    return rows


def verify_dir(root, manifest_digest, outer_digest):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise AuditError("sealed directory invalid: " + str(root))
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    exact_regular(manifest, manifest_digest)
    exact_regular(outer, outer_digest)
    expected_outer = manifest_digest + "  SHA256SUMS\n"
    if outer.read_text() != expected_outer:
        raise AuditError("outer seal content mismatch: " + str(root))
    listed = set()
    for digest, rel in parse_manifest(manifest):
        if rel in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
            raise AuditError("recursive manifest member")
        exact_regular(root / rel, digest)
        listed.add(rel)
    actual = set(str(path.relative_to(root)) for path in root.rglob("*")
                 if path.is_file() and path.name not in
                 ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    if listed != actual:
        raise AuditError("manifest inventory mismatch: " + str(root))


def verify_file_double_seal(path, digest, sidecar_file_digest, outer_file_digest):
    path = Path(path)
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    exact_regular(path, digest)
    exact_regular(sidecar, sidecar_file_digest)
    exact_regular(outer, outer_file_digest)
    if sidecar.read_text() != digest + "  " + path.name + "\n":
        raise AuditError("file sidecar content mismatch")
    if outer.read_text() != sidecar_file_digest + "  " + sidecar.name + "\n":
        raise AuditError("file outer content mismatch")


EXPECTED_IDENTITY = {
    "runner_sha256": SHAS["runner"],
    "source_contract_sha256": SHAS["contract"],
    "author_receipt_sha256": SHAS["author_receipt"],
    "author_manifest_sha256": SHAS["author_manifest"],
    "author_outer_seal_file_sha256": SHAS["author_outer"],
    "m1858_attempt_manifest_sha256": SHAS["m1858_attempt_manifest"],
    "m1858_attempt_outer_seal_file_sha256": SHAS["m1858_attempt_outer"],
    "m1858_failure_manifest_sha256": SHAS["m1858_failure_manifest"],
    "m1858_failure_outer_seal_file_sha256": SHAS["m1858_failure_outer"],
    "m1873_failure_review_sha256": SHAS["m1873_review"],
    "m1873_failure_review_manifest_sha256": SHAS["m1873_manifest"],
    "m1873_failure_review_outer_seal_file_sha256": SHAS["m1873_outer"],
    "m1878_source_review_sha256": SHAS["m1878_review"],
    "m1878_source_review_manifest_sha256": SHAS["m1878_manifest"],
    "m1878_source_review_outer_seal_file_sha256": SHAS["m1878_outer"],
    "m1811_manifest_sha256": SHAS["m1811_manifest"],
    "m1811_outer_seal_file_sha256": SHAS["m1811_outer"],
    "m1830_review_sha256": SHAS["m1830_review"],
    "m1830_manifest_sha256": SHAS["m1830_manifest"],
    "m1830_outer_seal_file_sha256": SHAS["m1830_outer"],
}

EXPECTED_AUTH = {
    "max_attempts": 1, "formality_runs": 2, "pt_runs": 2,
    "dc_runs": 0, "vcs_runs": 0, "ptpx_runs": 0,
    "automatic_retry": False,
}
EXPECTED_FROZEN = {
    "docs359_sha256": SHAS["docs359"],
    "m1858_attempt_json_sha256": SHAS["m1858_attempt_json"],
    "m1858_failure_terminal_sha256": SHAS["m1858_failure_terminal"],
    "m1858_attempt_consumed": True,
    "m1858_retry_allowed": False,
    "m1858_raw_k8_formality_reusable_as_m1877_result": False,
    "m1858_raw_k8_formality_paper_citable": False,
}
EXPECTED_EXECUTION = {
    "axis_order": ["K8", "K1X8"],
    "exact_process_order": [
        "K8_FORMALITY", "K8_FORMALITY_RESULT_GATE",
        "K8_DUAL_CORNER_PRIMETIME", "K8_PRIMETIME_RESULT_GATE",
        "K1X8_FORMALITY", "K1X8_FORMALITY_RESULT_GATE",
        "K1X8_DUAL_CORNER_PRIMETIME", "K1X8_PRIMETIME_RESULT_GATE"],
    "fresh_complete_two_axis_campaign_required": True,
    "partial_axis_admission_allowed": False,
    "partial_process_admission_allowed": False,
    "m1858_raw_or_partial_result_reuse_allowed": False,
    "unique_attempt_consumed_before_first_eda": True,
    "negative_setup_or_hold_reported_not_hidden": True,
    "negative_setup_or_hold_blocks_raw_result_publication": False,
    "timing_exceptions_added": False,
    "pt_eco": False,
    "hold_repair": False,
}
EXPECTED_CLAIM = {
    "source_reviewed": True, "launch_authorized": True,
    "launch_executed": False, "formality": False, "prime_time": False,
    "setup_closed": False, "hold_closed": False, "power": False,
    "energy": False, "cycle_speedup": False, "system_speedup": False,
    "paper_ppa_ready": False, "paper_citable": False, "headline": False,
}
EXPECTED_POSTRUN = {
    "different_author_result_hammer_required": True,
    "raw_result_pending_hammer_not_citable": True,
    "both_axes_formality_and_prime_time_required_for_admission": True,
    "failed_or_incomplete_campaign_must_be_sealed_do_not_cite": True,
}


def verify_release_semantics(release, exact_prohibitions):
    if set(release) != set(("schema", "milestone", "date", "release_author",
                            "status", "authorization", "identity",
                            "frozen_authority", "execution_contract",
                            "claim_boundary", "postrun_requirement", "prohibitions")):
        raise AuditError("release top-level schema drift")
    if release.get("schema") != "m1879_m1878_m1877_c2_fresh_mapped_formality_dual_corner_pt_launch_release_r1_v1":
        raise AuditError("release schema drift")
    if release.get("milestone") != "M1879" or release.get("date") != "2026-09-02":
        raise AuditError("release milestone/date drift")
    if release.get("release_author") != "/root/m1878_c2_formality_source_review":
        raise AuditError("release author drift")
    if release.get("status") != "AUTHORIZE_ONE_M1877_C2_FRESH_MAPPED_FORMALITY_DUAL_CORNER_PT_ATTEMPT":
        raise AuditError("release status drift")
    if release.get("authorization") != EXPECTED_AUTH:
        raise AuditError("release authorization drift")
    if release.get("identity") != EXPECTED_IDENTITY:
        raise AuditError("release identity drift")
    if release.get("frozen_authority") != EXPECTED_FROZEN:
        raise AuditError("release frozen authority drift")
    if release.get("execution_contract") != EXPECTED_EXECUTION:
        raise AuditError("release execution contract drift")
    if release.get("claim_boundary") != EXPECTED_CLAIM:
        raise AuditError("release claim boundary drift")
    if release.get("postrun_requirement") != EXPECTED_POSTRUN:
        raise AuditError("release postrun requirement drift")
    if release.get("prohibitions") != exact_prohibitions:
        raise AuditError("release prohibitions drift")


def mutate_and_require_reject(canonical, path, value, exact_prohibitions):
    candidate = copy.deepcopy(canonical)
    cursor = candidate
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    try:
        verify_release_semantics(candidate, exact_prohibitions)
    except AuditError:
        return
    raise AuditError("semantic mutation escaped: " + ".".join(path))


def load_runner_and_verify_authority():
    os.environ["M1877_EXPECTED_RUNNER_SHA256"] = SHAS["runner"]
    os.environ["M1877_EXPECTED_SOURCE_CONTRACT_SHA256"] = SHAS["contract"]
    os.environ["M1877_EXPECTED_M1878_SOURCE_REVIEW_SHA256"] = SHAS["m1878_review"]
    os.environ["M1877_EXPECTED_M1878_SOURCE_REVIEW_MANIFEST_SHA256"] = SHAS["m1878_manifest"]
    os.environ["M1877_EXPECTED_M1878_SOURCE_REVIEW_OUTER_SHA256"] = SHAS["m1878_outer"]
    os.environ["M1877_EXPECTED_M1879_LAUNCH_RELEASE_SHA256"] = SHAS["release"]
    spec = importlib.util.spec_from_file_location("m1877_release_audit_runner", str(RUNNER))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    release_sha, live = module.verify_authority()
    if release_sha != SHAS["release"] or len(live) != 13 or len(set(live)) != 13:
        raise AuditError("runner verify_authority return drift")


def main():
    verify_file_double_seal(RELEASE, SHAS["release"],
                            SHAS["release_sidecar_file"], SHAS["release_outer_file"])
    verify_file_double_seal(CONTRACT, SHAS["contract"],
                            SHAS["contract_sidecar_file"], SHAS["contract_outer_file"])
    exact_regular(RUNNER, SHAS["runner"])
    verify_dir(AUTHOR, SHAS["author_manifest"], SHAS["author_outer"])
    exact_regular(AUTHOR / "author_receipt.json", SHAS["author_receipt"])
    verify_dir(REVIEW1878, SHAS["m1878_manifest"], SHAS["m1878_outer"])
    exact_regular(REVIEW1878 / "review.json", SHAS["m1878_review"])
    verify_dir(REVIEW1873, SHAS["m1873_manifest"], SHAS["m1873_outer"])
    exact_regular(REVIEW1873 / "review.json", SHAS["m1873_review"])
    verify_dir(M1858_ATTEMPT, SHAS["m1858_attempt_manifest"], SHAS["m1858_attempt_outer"])
    exact_regular(M1858_ATTEMPT / "attempt.json", SHAS["m1858_attempt_json"])
    verify_dir(M1858_FAILURE, SHAS["m1858_failure_manifest"], SHAS["m1858_failure_outer"])
    exact_regular(M1858_FAILURE / "RUN_FAILED_OR_INCOMPLETE.txt", SHAS["m1858_failure_terminal"])
    verify_dir(M1811, SHAS["m1811_manifest"], SHAS["m1811_outer"])
    verify_dir(M1830, SHAS["m1830_manifest"], SHAS["m1830_outer"])
    exact_regular(M1830 / "review.json", SHAS["m1830_review"])
    exact_regular(DOCS359, SHAS["docs359"])

    release = strict_json(RELEASE)
    exact_prohibitions = copy.deepcopy(release["prohibitions"])
    verify_release_semantics(release, exact_prohibitions)

    contract = strict_json(CONTRACT)
    author = strict_json(AUTHOR / "author_receipt.json")
    review1878 = strict_json(REVIEW1878 / "review.json")
    review1873 = strict_json(REVIEW1873 / "review.json")
    attempt = strict_json(M1858_ATTEMPT / "attempt.json")
    if contract.get("status") != "SOURCE_ONLY_M1877_C2_FRESH_MAPPED_FORMALITY_DUAL_CORNER_PT__NO_EDA_AUTHORIZED":
        raise AuditError("M1877 contract status drift")
    if author.get("status") != "PASS_SOURCE_AUTHORING_ONLY_M1877_C2_SECTION_AWARE_BLACK_BOX_SUCCESSOR__M1878_REVIEW_M1879_RELEASE_REQUIRED__NO_EDA_RUN":
        raise AuditError("M1877 author status drift")
    if review1878.get("status") != "PASS_M1878_M1877_C2_FRESH_MAPPED_FORMALITY_DUAL_CORNER_PT_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_ATTEMPT":
        raise AuditError("M1878 status drift")
    if review1878.get("severity_counts") != {"p0": 0, "p1": 0, "p2": 0}:
        raise AuditError("M1878 severity drift")
    if review1873.get("audit_status") != "PASS" or review1873.get("production_admission") != "FAIL_CLOSED":
        raise AuditError("M1873 fail-closed status drift")
    if review1873.get("severity_counts") != {"p0": 0, "p1": 1, "p2": 0}:
        raise AuditError("M1873 severity drift")
    if attempt.get("status") != "M1858_ATTEMPT_CONSUMED_BEFORE_FIRST_EDA":
        raise AuditError("M1858 attempt semantics drift")
    if M1858_FAILURE.joinpath("RUN_FAILED_OR_INCOMPLETE.txt").read_text() != (
            "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\n"
            "error=K8 unresolved/empty/unlinked black box nonzero\n"
            "retry=false\n"):
        raise AuditError("M1858 failure terminal drift")

    runner_text = RUNNER.read_text()
    ordered_tokens = [
        'AXIS_ORDER = ("K8", "K1X8")',
        "for axis in AXIS_ORDER:",
        "run_tool(FM_SHELL, FM_TCL, axis, fm_dir, \"formality.log\")",
        "passing = verify_formality(axis, fm_dir)",
        "run_tool(PT_SHELL, PT_TCL, axis, pt_dir, \"pt.log\")",
        "timing = verify_pt(axis, pt_dir)",
    ]
    positions = [runner_text.find(token) for token in ordered_tokens]
    if any(position < 0 for position in positions):
        raise AuditError("runner process-order token absent")
    loop_pos = runner_text.find("for axis in AXIS_ORDER:",
                                runner_text.find("metrics = {}"))
    process_positions = [runner_text.find(token, loop_pos) for token in ordered_tokens[2:]]
    if loop_pos < 0 or process_positions != sorted(process_positions) or any(p < loop_pos for p in process_positions):
        raise AuditError("runner K8/FM/PT/K1X8 order drift")
    if runner_text.count("\n            write_attempt(release_sha)\n") != 1:
        raise AuditError("unique attempt consumption drift")
    if runner_text.count("\n        license_gate()\n") != 1:
        raise AuditError("license gate cardinality drift")

    exact_result = RUNS / "m1877_m1811_c2_fresh_mapped_formality_dual_corner_pt_r1_20260902"
    exact_attempt = RUNS / ".m1877_m1811_c2_fresh_mapped_formality_dual_corner_pt_attempt_consumed"
    exact_lock = RUNS / ".m1877_m1811_c2_fresh_mapped_formality_dual_corner_pt_launch_lock"
    work_glob = list(RUNS.glob(".m1877_m1811_c2_fresh_mapped_formality_dual_corner_pt_work.*"))
    failure_glob = list(RUNS.glob("m1877_m1811_c2_fresh_mapped_formality_dual_corner_pt_r1_20260902.failed_or_incomplete.*"))
    if exact_result.exists() or exact_attempt.exists() or exact_lock.exists() or work_glob or failure_glob:
        raise AuditError("M1877 namespace not fresh at release audit")

    mutation_cases = [
        (("schema",), "wrong"), (("milestone",), "M1880"),
        (("release_author",), "/root/wrong"), (("status",), "PASS"),
        (("authorization", "max_attempts"), 2),
        (("authorization", "formality_runs"), 1),
        (("authorization", "pt_runs"), 1),
        (("authorization", "dc_runs"), 1),
        (("authorization", "vcs_runs"), 1),
        (("authorization", "ptpx_runs"), 1),
        (("authorization", "automatic_retry"), True),
        (("identity", "runner_sha256"), "0" * 64),
        (("identity", "source_contract_sha256"), "0" * 64),
        (("identity", "m1878_source_review_sha256"), "0" * 64),
        (("identity", "m1873_failure_review_sha256"), "0" * 64),
        (("identity", "m1858_attempt_manifest_sha256"), "0" * 64),
        (("identity", "m1811_manifest_sha256"), "0" * 64),
        (("identity", "m1830_review_sha256"), "0" * 64),
        (("frozen_authority", "docs359_sha256"), "0" * 64),
        (("frozen_authority", "m1858_attempt_consumed"), False),
        (("frozen_authority", "m1858_retry_allowed"), True),
        (("frozen_authority", "m1858_raw_k8_formality_reusable_as_m1877_result"), True),
        (("execution_contract", "axis_order"), ["K1X8", "K8"]),
        (("execution_contract", "exact_process_order"), list(reversed(EXPECTED_EXECUTION["exact_process_order"]))),
        (("execution_contract", "partial_axis_admission_allowed"), True),
        (("execution_contract", "partial_process_admission_allowed"), True),
        (("execution_contract", "m1858_raw_or_partial_result_reuse_allowed"), True),
        (("execution_contract", "unique_attempt_consumed_before_first_eda"), False),
        (("execution_contract", "timing_exceptions_added"), True),
        (("claim_boundary", "formality"), True),
        (("claim_boundary", "prime_time"), True),
        (("claim_boundary", "paper_citable"), True),
        (("postrun_requirement", "different_author_result_hammer_required"), False),
        (("postrun_requirement", "raw_result_pending_hammer_not_citable"), False),
    ]
    for path, value in mutation_cases:
        mutate_and_require_reject(release, path, value, exact_prohibitions)

    load_runner_and_verify_authority()
    print(json.dumps({
        "status": "PASS_M1883_RELEASE_AUDIT_RUNTIME",
        "python": sys.version.split()[0],
        "release_sha256": SHAS["release"],
        "negative_mutations_rejected": len(mutation_cases),
        "runner_verify_authority_calls": 1,
        "runner_execute_calls": 0,
        "license_queries": 0,
        "eda_runs": 0,
        "attempts_created": 0,
        "results_created": 0,
        "fresh_namespace": True,
    }, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print("ERROR_M1883: " + str(error), file=sys.stderr)
        raise SystemExit(1)
