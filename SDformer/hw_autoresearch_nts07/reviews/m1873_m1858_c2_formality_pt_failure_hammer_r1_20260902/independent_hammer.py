#!/usr/bin/env python3
"""Independent, read-only M1873 audit of the consumed M1858 failure.

This script never launches or queries EDA/licenses.  It verifies the immutable
attempt latch and failure quarantine, parses the raw K8 Formality reports, and
separates a tool-proven raw diagnostic fact from production admission of the
two-axis Formality/PT campaign.
"""
import hashlib
import json
from pathlib import Path
import re
import stat


OUT = Path(__file__).resolve().parent
HW = OUT.parents[1]
RUNS = HW / "dc_handoff/runs"
ATTEMPT = RUNS / ".m1858_m1811_c2_fresh_mapped_formality_dual_corner_pt_attempt_consumed"
FAILURE = RUNS / "m1858_m1811_c2_fresh_mapped_formality_dual_corner_pt_r1_20260902.failed_or_incomplete.2511659.quarantine"
CANONICAL = RUNS / "m1858_m1811_c2_fresh_mapped_formality_dual_corner_pt_r1_20260902"
RUNNER = HW / "dc_handoff/scripts/run_m1858_c2_fresh_mapped_formality_dual_corner_pt_one_shot.py"
RELEASE = HW / "contracts/m1860_m1859_m1858_c2_fresh_mapped_formality_dual_corner_pt_launch_release_r1_20260902.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
K8 = FAILURE / "k8/formality"
REPORTS = K8 / "reports"

EXPECTED = {
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "runner": "b115d0483516c67eb4c39dfe22f29a19f4c13ce65a26bcacdc2fe3d5125d2dee",
    "release": "86216f25fe73e51d0376bd9c856a6c2f3de13c78a433d9024f5312a9ea6496af",
    "attempt_json": "fcca3129f572bbfa85ea7f8e33951497f40d20a93a534a80d3a3a782aea33487",
    "attempt_manifest": "1899bc129ade7b16da92a5e9c2be43e0a7a96af3c9d0e9e7d9ed25ff9056320e",
    "attempt_outer": "87124c075d7dad34c93bd472d612db47fac1e81dd53a4ca85646aa130b9bfbb2",
    "failure_status": "117e58207a8983cd984cb7da09b1c9e79bd692f089dae1f4cca1241e4c20c279",
    "failure_input": "c7decca4bcacd61a6bc2b12da6469f161bba0cabbd50bd16f25a8445a2763d0c",
    "failure_manifest": "82c363a4869af160a4d7ec0a1f1c6d9d8587a583ae9e43fbf19d6eb3acba366d",
    "failure_outer": "3c47ed5d552c73e401c219bfc511f7f5830ac986cb8cbcd386e0dd24fcbd4bc3",
    "formality_log": "7423f0e04b8d48adab4bb8da6257f2555255a6816aa616dee830c7fdfc897d3a",
    "formality_status": "c1422264617dcd5bf05a3a5c0157a3147415694c4040f751cfb8ea90e0fe5b72",
    "formality_black_boxes": "936295aecbf6d13d33ffe47ef996c7485315665f222f1c585714c9fc4b54ebf0",
}


class AuditError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_regular(path: Path, expected: str) -> None:
    if (not path.is_file() or path.is_symlink()
            or not stat.S_ISREG(path.lstat().st_mode)
            or sha256(path) != expected):
        raise AuditError("identity mismatch: " + str(path))


def verify_sealed_directory(root: Path, manifest_sha: str, outer_sha: str) -> None:
    if not root.is_dir() or root.is_symlink():
        raise AuditError("sealed directory absent/invalid: " + str(root))
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    exact_regular(manifest, manifest_sha)
    exact_regular(outer, outer_sha)
    if outer.read_text().split() != [manifest_sha, "SHA256SUMS"]:
        raise AuditError("outer seal semantic mismatch: " + str(root))
    listed = set()
    for line in manifest.read_text().splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None:
            raise AuditError("manifest syntax: " + str(root))
        name = fields[1].lstrip("*")
        rel = Path(name)
        if name in listed or rel.is_absolute() or ".." in rel.parts:
            raise AuditError("unsafe/duplicate manifest path: " + name)
        exact_regular(root / rel, fields[0])
        listed.add(name)
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise AuditError("symlink in sealed directory: " + str(path))
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            actual.add(path.relative_to(root).as_posix())
    if listed != actual:
        raise AuditError("sealed population drift: " + str(root))


def strict_json(path: Path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise AuditError("duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          AuditError("nonfinite JSON: " + token)))


def parse_black_box_entries(text: str):
    """Parse report entries without conflating library class and attributes."""
    lines = text.splitlines()
    section_kind = None
    section_library = None
    entries = []
    index = 0
    header = re.compile(r"^####\s+(TECH|DESIGN) LIBRARY - (.+)$")
    entry = re.compile(r"^([a-zA-Z* ]{1,5})\s{2,}(\S.*)$")
    while index < len(lines):
        match = header.match(lines[index])
        if match:
            section_kind, section_library = match.group(1), match.group(2)
            index += 1
            continue
        candidate = entry.match(lines[index]) if section_kind else None
        if candidate and candidate.group(2) not in {"Design Name", "----------"}:
            attrs = candidate.group(1).split()
            design = candidate.group(2).strip()
            if attrs and all(token in {"s", "i", "u", "e", "*", "ut", "L", "cp", "ir", "f", "m"}
                             for token in attrs):
                stop = index + 1
                instance_count = None
                instance_total = None
                paths = []
                while stop < len(lines):
                    if header.match(lines[stop]):
                        break
                    next_entry = entry.match(lines[stop])
                    if (next_entry and next_entry.group(2) not in {"Design Name", "----------"}
                            and next_entry.group(1).split()
                            and all(token in {"s", "i", "u", "e", "*", "ut", "L", "cp", "ir", "f", "m"}
                                    for token in next_entry.group(1).split())):
                        break
                    count_match = re.match(r"^\s*Instances\s*:\s*(\d+)(?:\s+of\s+(\d+))?\s*$", lines[stop])
                    if count_match:
                        instance_count = int(count_match.group(1))
                        instance_total = int(count_match.group(2) or count_match.group(1))
                    elif re.match(r"^\s+[ri]:/", lines[stop]):
                        paths.append(lines[stop].strip())
                    stop += 1
                if instance_count is None:
                    raise AuditError("black-box entry lacks instance count: " + design)
                entries.append({
                    "section_kind": section_kind,
                    "library": section_library,
                    "attributes": attrs,
                    "design": design,
                    "instances": instance_count,
                    "instances_total": instance_total,
                    "paths": paths,
                })
                index = stop
                continue
        index += 1
    return entries


def seal_output() -> None:
    members = sorted(
        path for path in OUT.iterdir()
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    manifest = "".join(sha256(path) + "  " + path.name + "\n" for path in members)
    (OUT / "SHA256SUMS").write_text(manifest)
    (OUT / "SHA256SUMS.seal.sha256").write_text(
        sha256(OUT / "SHA256SUMS") + "  SHA256SUMS\n")


def main() -> None:
    for name in ("review.json", "review.md", "mechanical_checks.json",
                 "RUN_COMPLETE.txt", "SHA256SUMS", "SHA256SUMS.seal.sha256"):
        if (OUT / name).exists():
            raise AuditError("M1873 output namespace already consumed: " + name)

    exact_regular(DOCS359, EXPECTED["docs359"])
    exact_regular(RUNNER, EXPECTED["runner"])
    exact_regular(RELEASE, EXPECTED["release"])
    verify_sealed_directory(ATTEMPT, EXPECTED["attempt_manifest"], EXPECTED["attempt_outer"])
    verify_sealed_directory(FAILURE, EXPECTED["failure_manifest"], EXPECTED["failure_outer"])
    exact_regular(ATTEMPT / "attempt.json", EXPECTED["attempt_json"])
    exact_regular(FAILURE / "RUN_FAILED_OR_INCOMPLETE.txt", EXPECTED["failure_status"])
    exact_regular(FAILURE / "input_identity.json", EXPECTED["failure_input"])
    exact_regular(K8 / "formality.log", EXPECTED["formality_log"])
    exact_regular(REPORTS / "formality_status.rpt", EXPECTED["formality_status"])
    exact_regular(REPORTS / "formality_black_boxes.rpt", EXPECTED["formality_black_boxes"])

    attempt = strict_json(ATTEMPT / "attempt.json")
    identity = strict_json(FAILURE / "input_identity.json")
    if attempt != {
            "automatic_retry": False,
            "axes": ["K8", "K1X8"],
            "formality_runs": 2,
            "pt_runs": 2,
            "release_sha256": EXPECTED["release"],
            "schema": "m1858_c2_fresh_mapped_formality_dual_corner_pt_attempt_r1_v1",
            "status": "M1858_ATTEMPT_CONSUMED_BEFORE_FIRST_EDA"}:
        raise AuditError("attempt semantics drift")
    if identity.get("runner_sha256") != EXPECTED["runner"] or identity.get("m1860_release_sha256") != EXPECTED["release"]:
        raise AuditError("failure input identity drift")
    if (FAILURE / "RUN_FAILED_OR_INCOMPLETE.txt").read_text() != (
            "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\n"
            "error=K8 unresolved/empty/unlinked black box nonzero\n"
            "retry=false\n"):
        raise AuditError("failure terminal drift")

    namespace = list(RUNS.glob("m1858_m1811_c2_fresh_mapped_formality_dual_corner_pt_r1_20260902*"))
    if namespace != [FAILURE] or CANONICAL.exists():
        raise AuditError("M1858 canonical/failure namespace cardinality drift")
    if any(RUNS.glob(".m1858_m1811_c2_fresh_mapped_formality_dual_corner_pt_work.*")):
        raise AuditError("unsealed M1858 work namespace remains")

    if (K8 / "formality.rc").read_text() != "0\n":
        raise AuditError("K8 Formality process did not exit zero")
    marker = (K8 / "FORMALITY_INTERNAL_COMPLETE.txt").read_text()
    for token in (
            "M1858_C2_FRESH_MAPPED_FORMALITY_INTERNAL_COMPLETE=PASS",
            "axis=K8", "reference_elab_parameters=ARCH_MODE=0",
            "implementation_top=m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24_ARCH_MODE0"):
        if marker.count(token) != 1:
            raise AuditError("K8 internal marker drift: " + token)
    if (FAILURE / "k1x8").exists() or (FAILURE / "k8/pt").exists():
        raise AuditError("unexpected K1X8 or PT execution artifact")

    log = (K8 / "formality.log").read_text(errors="replace")
    status = (REPORTS / "formality_status.rpt").read_text(errors="replace")
    if log.count("Total               :       3798          0          0          0       3798") != 1:
        raise AuditError("SVF guidance total drift")
    if status.count("Verification SUCCEEDED") != 1 or status.count("33656 Passing compare points") != 1:
        raise AuditError("K8 proof result drift")
    if not re.search(r"(?m)^Passing \(equivalent\)\s+0\s+0\s+0\s+0\s+1228\s+32428\s+0\s+33656$", status):
        raise AuditError("K8 passing/BBPin row drift")
    if not re.search(r"(?m)^Failing \(not equivalent\)\s+(?:0\s+){7}0$", status):
        raise AuditError("K8 failing row drift")
    for report, token in (("formality_unmatched.rpt", "No unmatched points."),
                          ("formality_failing.rpt", "No failing compare points."),
                          ("formality_aborted.rpt", "No aborted compare points."),
                          ("formality_unverified.rpt", "No unverified compare points.")):
        if (REPORTS / report).read_text(errors="replace").count(token) != 1:
            raise AuditError("non-clean K8 report: " + report)
    if len(re.findall(r"\(FMR_ELAB-147\)", log)) != 8:
        raise AuditError("FMR_ELAB-147 warning count drift")

    black_text = (REPORTS / "formality_black_boxes.rpt").read_text(errors="replace")
    entries = parse_black_box_entries(black_text)
    old_regex = re.compile(r"(?m)^\s*(?:u|e|\*)\s+\S+[\s\S]{0,180}?Instances\s*:\s*[1-9][0-9]*")
    old_hits = old_regex.findall(black_text)
    if len(old_hits) != 2 or any("SNPS_BUSHOLD" not in hit for hit in old_hits):
        raise AuditError("old parser false-positive identity/count drift")

    design_nonzero = [row for row in entries
                      if row["section_kind"] == "DESIGN" and row["instances"] > 0
                      and any(attr in {"u", "e", "*"} for attr in row["attributes"])]
    tech_m_nonzero = [row for row in entries
                      if row["section_kind"] == "TECH" and "m" in row["attributes"]
                      and row["instances"] > 0]
    tech_e_nonzero = [row for row in entries
                      if row["section_kind"] == "TECH" and "e" in row["attributes"]
                      and row["instances"] > 0]
    design_zero_unlinked = [row for row in entries
                            if row["section_kind"] == "DESIGN" and row["instances"] == 0
                            and any(attr in {"u", "e", "*"} for attr in row["attributes"])]
    if design_nonzero:
        raise AuditError("nonzero unresolved/empty/unlinked design instance exists")
    if len(design_zero_unlinked) != 12:
        raise AuditError("zero-instance design-library entry count drift")
    if len(tech_m_nonzero) != 2 or any(row["design"] != "ANTENNABWP35P140#PWR_FM_BBOX"
                                      or row["instances"] != 1 for row in tech_m_nonzero):
        raise AuditError("technology macro instance identity/count drift")
    if len(tech_e_nonzero) != 2:
        raise AuditError("technology empty-module exception count drift")
    expected_libraries = {
        "i:/TCBN28HPCPLUSBWP35P140SSG0P9V125C",
        "r:/TCBN28HPCPLUSBWP35P140SSG0P9V125C",
    }
    if {row["library"] for row in tech_e_nonzero} != expected_libraries:
        raise AuditError("SNPS_BUSHOLD dual-side library symmetry drift")
    for row in tech_e_nonzero:
        prefix = row["library"].split(":", 1)[0] + ":/TCBN28HPCPLUSBWP35P140SSG0P9V125C/"
        expected_paths = {
            prefix + "BHDBWP35P140/C0",
            prefix + "BHDBWP35P140#PWR/C2",
        }
        if (row["attributes"] != ["e"] or row["design"] != "SNPS_BUSHOLD"
                or row["instances"] != 2 or row["instances_total"] != 2
                or set(row["paths"]) != expected_paths):
            raise AuditError("SNPS_BUSHOLD exception is not exact and symmetric")

    mechanical = {
        "schema": "m1873_m1858_c2_formality_pt_failure_mechanical_checks_r1_v1",
        "audit_pass": True,
        "docs359_sha256_ok": True,
        "attempt_double_seal_ok": True,
        "failure_double_seal_ok": True,
        "attempt_consumed": True,
        "automatic_retry": False,
        "canonical_result_count": 0,
        "failure_quarantine_count": 1,
        "k8_formality_artifact_count": 1,
        "k8_formality_rc": 0,
        "k8_valid_design_pair": True,
        "k8_verification_succeeded": True,
        "k8_passing_compare_points": 33656,
        "k8_failing_compare_points": 0,
        "k8_aborted_compare_points": 0,
        "k8_unverified_compare_points": 0,
        "k8_unmatched_compare_points": 0,
        "k8_bbpin_compare_points": 0,
        "svf_guidance_accepted": 3798,
        "svf_guidance_rejected": 0,
        "fmr_elab_147_warning_count": 8,
        "old_black_box_regex_false_positive_count": 2,
        "nonzero_design_unresolved_empty_unlinked_instances": 0,
        "zero_instance_design_empty_unlinked_entry_count": 12,
        "nonzero_technology_macro_entries": 2,
        "exact_dual_side_snps_bushold_entries": 2,
        "k8_pt_artifact_count": 0,
        "k1x8_formality_artifact_count": 0,
        "k1x8_pt_artifact_count": 0,
        "eda_or_license_operations_by_m1873": 0,
    }

    review = {
        "schema": "m1873_m1858_c2_formality_pt_failure_hammer_review_r1_v1",
        "milestone": "M1873",
        "date": "2026-09-02",
        "reviewer_identity": "/root/m1873_c2_formality_failure_review",
        "audit_status": "PASS",
        "production_admission": "FAIL_CLOSED",
        "status": "PASS_M1873_INDEPENDENT_FAILURE_AUDIT__M1858_CAMPAIGN_FAIL_CLOSED__K8_RAW_FORMALITY_DIAGNOSTIC_SUCCEEDED__P0_0_P1_1_P2_0__NO_RETRY__NO_PT",
        "score_over_100": 99,
        "severity_counts": {"p0": 0, "p1": 1, "p2": 0},
        "scope": "Different-author, read-only audit of the consumed M1858 C2 fresh-mapped Formality/dual-corner-PT attempt, its double-sealed attempt latch and failure quarantine, the sole K8 Formality process artifacts, and the runner post-proof black-box gate. No EDA, license query, retry, source mutation, docs/359 mutation, ucli.key access, or predecessor mutation was performed.",
        "verdict": "M1858 is correctly FAIL_CLOSED as a two-axis Formality/PT campaign, but its sealed K8 raw Formality result is a genuine diagnostic success: a valid ARCH_MODE=0 reference/implementation pair verified with 33,656 passing compare points, zero failing/aborted/unverified/unmatched points, BBPin=0, and 3,798 accepted/zero rejected SVF commands. The runner then falsely classified two symmetric TECH LIBRARY entries, each exactly `e SNPS_BUSHOLD / Instances: 2 of 2`, as actionable design black boxes. Technology-macro `m` entries are not unresolved design modules; all DESIGN LIBRARY `e *` entries have zero instances; and the sole nonzero `m` entry per side is the ANTENNA power-model technology macro. K8 PT and the complete K1X8 axis never ran. Therefore no full C2 equivalence/PT result is admitted, no paper result is citable, and the consumed namespace must not be retried.",
        "classification": {
            "primary": "POST_PROOF_BLACK_BOX_PARSER_FALSE_POSITIVE_ON_EXACT_DUAL_SIDE_TECH_LIBRARY_SNPS_BUSHOLD",
            "design_inequivalence_proven": False,
            "k8_valid_compare_pair_established": True,
            "k8_raw_formality_verification_succeeded": True,
            "k8_raw_equivalence_may_be_retained_as_sealed_diagnostic_fact": True,
            "k8_raw_equivalence_paper_citable_or_production_admitted": False,
            "campaign_formality_equivalence_complete": False,
            "campaign_pt_complete": False,
            "failure_is_parser_only_after_k8_proof": True,
            "actionable_design_black_box_instances": 0,
            "bbpin_compare_points": 0,
            "exact_allowed_technology_synthetic_case": "dual-side TECH LIBRARY only: type=e, design=SNPS_BUSHOLD, Instances=2 of 2, exact BHDBWP35P140/C0 and BHDBWP35P140#PWR/C2 paths",
        },
        "identity": {
            "docs359_sha256": EXPECTED["docs359"],
            "m1858_runner_sha256": EXPECTED["runner"],
            "m1860_release_sha256": EXPECTED["release"],
            "m1858_attempt_json_sha256": EXPECTED["attempt_json"],
            "m1858_attempt_manifest_sha256": EXPECTED["attempt_manifest"],
            "m1858_attempt_outer_seal_file_sha256": EXPECTED["attempt_outer"],
            "m1858_failure_status_sha256": EXPECTED["failure_status"],
            "m1858_failure_input_identity_sha256": EXPECTED["failure_input"],
            "m1858_failure_manifest_sha256": EXPECTED["failure_manifest"],
            "m1858_failure_outer_seal_file_sha256": EXPECTED["failure_outer"],
            "m1858_k8_formality_log_sha256": EXPECTED["formality_log"],
            "m1858_k8_formality_status_sha256": EXPECTED["formality_status"],
            "m1858_k8_formality_black_boxes_sha256": EXPECTED["formality_black_boxes"],
            "bound_attempt_and_failure_double_seals_valid": True,
            "input_identity_runner_and_release_match": True,
        },
        "execution_audit": {
            "attempt_consumed": True,
            "automatic_retry": False,
            "canonical_results": 0,
            "failure_quarantines": 1,
            "formality_process_artifacts": 1,
            "formality_axis": "K8",
            "formality_return_code": 0,
            "formality_internal_complete_markers": 1,
            "pt_process_artifacts": 0,
            "k1x8_process_artifacts": 0,
            "passing_compare_points": 33656,
            "svf_guidance_accepted": 3798,
            "svf_guidance_rejected": 0,
            "failure_error": "K8 unresolved/empty/unlinked black box nonzero",
            "retry": False,
        },
        "black_box_audit": {
            "old_regex_false_positive_matches": 2,
            "old_regex_match_identity": "TECH LIBRARY e SNPS_BUSHOLD, Instances 2 of 2, once under implementation tech library and once under reference tech library",
            "design_library_nonzero_u_e_star_entries": 0,
            "design_library_zero_instance_e_star_entries": 12,
            "technology_macro_nonzero_entries": 2,
            "technology_macro_nonzero_identity": "ANTENNABWP35P140#PWR_FM_BBOX, one pwrBB instance on each side",
            "technology_empty_nonzero_entries": 2,
            "technology_empty_nonzero_identity": "SNPS_BUSHOLD, exact symmetric two-instance synthetic hold-cell internals on each side",
            "general_ignore_of_technology_e_entries_permitted": False,
        },
        "claim_boundary": {
            "m1858_unique_failure_verified": True,
            "k8_raw_formality_diagnostic_fact": True,
            "k8_production_formality_admission": False,
            "k1x8_formality": False,
            "prime_time": False,
            "setup_closed": False,
            "hold_closed": False,
            "power": False,
            "energy": False,
            "performance": False,
            "speedup": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "paper_citable_new_result": False,
            "headline": False,
        },
        "findings": [{
            "id": "M1873-P1-01",
            "severity": "P1",
            "finding": "The runner's broad unresolved/empty/unlinked regex crossed library semantics and rejected two exact, symmetric TECH LIBRARY SNPS_BUSHOLD synthetic entries after K8 Formality had already succeeded. No actionable nonzero unresolved/empty/unlinked DESIGN LIBRARY instance and no BBPin compare point exists.",
            "impact": "The consumed campaign cannot admit K8 or K1X8 Formality/PT. K8's sealed raw proof may be retained only as a diagnostic fact; it cannot be cited as the C2 result because PT and the second axis never executed and the quarantine explicitly says DO_NOT_CITE.",
            "minimal_fix": "Create a new additive campaign namespace; M1858 itself remains immutable and consumed. Replace only the black-box parser/gate with a section-aware parser. Continue to fail on every nonzero DESIGN LIBRARY u/e/* instance and every unexpected TECH LIBRARY u/e/* instance. Permit only the exact dual-sided TECH LIBRARY `e SNPS_BUSHOLD / Instances: 2 of 2` case with the frozen BHDBWP35P140/C0 and BHDBWP35P140#PWR/C2 paths; recognize type m as a technology macro rather than an unresolved design module; ignore starred entries only when Instances=0; and require BBPin=0 plus zero unmatched/failing/aborted/unverified points. A new full two-axis Formality/PT run, with different-author source review and exact one-attempt release, is required for admission."
        }],
    }

    md = """# M1873｜M1858 C2 Formality/PT 唯一失败独立审阅

结论：**审计 PASS（99/100），M1858 整体 production admission 仍为 FAIL_CLOSED；P0=0、P1=1、P2=0。K8 原始 Formality 报告确实成功，但只能保留为封存的诊断事实，不得作为 C2 论文/生产准入结果。M1858 已消费、`retry=false`，不得重跑。**

## 唯一运行与双封

- attempt latch 与 PID 2511659 failure quarantine 的 manifest/外层 seal 均独立校验通过；M1860 release、runner 与 `docs/359` SHA 均精确一致。
- namespace 中 canonical result=0、failure quarantine=1，无遗留 work 目录。
- 只有 K8 Formality 产物：`formality.rc=0`且 internal-complete marker 存在。K8 PT=0，K1X8 Formality/PT=0。
- 本 M1873 审阅没有启动 EDA、license query、retry、GPU 或远程任务，也没有修改前序证据、RTL、runner 或 `docs/359`。

## K8 原始 Formality 是真的成功

K8 建立了有效的 `ARCH_MODE=0` reference/implementation pair，SVF guidance **3798 accepted / 0 rejected**。`report_status` 明确为 `Verification SUCCEEDED`：

- 33,656 passing compare points；
- failing=0、aborted=0、unverified=0、unmatched compare points=0；
- passing/failing 表的 BBPin 均为 0；
- 已冻结的 8 条 `FMR_ELAB-147` warning 数量不变。

因此这不是设计不等价，也不是 Formality 工具返回失败。失败发生在工具证明结束后的 Python black-box gate。

## black-box parser 为什么误报

旧 regex 只看行首 `u|e|*` 和后续非零 `Instances`，不区分 `TECH LIBRARY` 与 `DESIGN LIBRARY`。它精确命中两处：

- implementation tech library：`e SNPS_BUSHOLD`, `Instances: 2 of 2`；
- reference tech library：同样的 `e SNPS_BUSHOLD`, `Instances: 2 of 2`。

两侧都只是 TSMC library 中 `BHDBWP35P140/C0` 与 `BHDBWP35P140#PWR/C2` 的对称 synthetic hold-cell internal。同一报告还显示：

- `m` 是 **Technology Macro cell (.db)**，不是 unresolved design module；每侧唯一非零 `m` 条目是 `ANTENNABWP35P140#PWR_FM_BBOX / pwrBB`；
- DESIGN LIBRARY 的 12 个 `e *` 条目全是 `Instances: 0`；
- 真正非零的 DESIGN LIBRARY `u/e/*` 实例数为 0，BBPin=0。

不允许由此泛化为“忽略 tech-library e 黑盒”。只能为这一个精确、双侧对称的 `SNPS_BUSHOLD` case 建立白名单，任何其他非零 `u/e/*` 仍必须 fail closed。

## 最小合法 successor

M1858 本身不能修复或重跑。仅修 parser 代码虽然足以修正这次 K8 的误报，但不足以补出从未运行的 K8 PT 和 K1X8 整轴，所以**必须新建 additive campaign 并重走完整两轴 Formality/PT**。新 gate 必须：

1. 按 TECH/DESIGN library section 解析，不再跨语义匹配；
2. 任何非零 DESIGN LIBRARY `u/e/*` 立即失败；
3. TECH LIBRARY 中仅允许精确的双侧 `e SNPS_BUSHOLD / 2 of 2` 路径集，不得泛化；
4. `m` 按 technology macro 处理；带 `*` 条目只有 `Instances=0` 才能忽略；
5. 继续强制 BBPin=0，且 failing/aborted/unverified/unmatched=0、passing>0；
6. 经 different-author source review 与 exact one-attempt release 后才能执行。

## 论文边界

K8 的 33,656-point raw equivalence 可作为“封存诊断事实”保留，但 quarantine 明确标记 `FAILED_OR_INCOMPLETE_DO_NOT_CITE`，而且 K8 PT/K1X8 均缺失。因此不得在论文中声称 C2 Formality/PT 已闭合，也不得由 M1858 引出 setup/hold、PPA、功耗、能量或性能新数字。
"""

    (OUT / "mechanical_checks.json").write_text(
        json.dumps(mechanical, indent=2, sort_keys=True) + "\n")
    (OUT / "review.json").write_text(
        json.dumps(review, indent=2, ensure_ascii=False) + "\n")
    (OUT / "review.md").write_text(md)
    (OUT / "RUN_COMPLETE.txt").write_text(
        "M1873_INDEPENDENT_FAILURE_AUDIT=PASS\n"
        "PRODUCTION_ADMISSION=FAIL_CLOSED\n"
        "K8_RAW_FORMALITY_DIAGNOSTIC=SUCCEEDED_NOT_CITABLE\n"
        "P0=0 P1=1 P2=0\n"
        "NO_EDA_NO_LICENSE_NO_RETRY=TRUE\n")
    seal_output()


if __name__ == "__main__":
    main()
