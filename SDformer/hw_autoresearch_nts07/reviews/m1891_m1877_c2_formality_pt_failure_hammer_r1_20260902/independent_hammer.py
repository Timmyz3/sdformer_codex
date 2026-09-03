#!/usr/bin/env python3
"""Read-only M1891 audit of the consumed M1877 failure quarantine.

No EDA, license query, retry, or source mutation is performed.  The audit
separates the valid K8 Formality diagnostic, the complete-but-failing K8 PT
diagnostic, and the absent K1X8 axis from production admission.
"""
import hashlib
import json
from pathlib import Path
import re
import stat


OUT = Path(__file__).resolve().parent
HW = OUT.parents[1]
RUNS = HW / "dc_handoff/runs"
ATTEMPT = RUNS / ".m1877_m1811_c2_fresh_mapped_formality_dual_corner_pt_attempt_consumed"
FAILURE = RUNS / "m1877_m1811_c2_fresh_mapped_formality_dual_corner_pt_r1_20260902.failed_or_incomplete.3055288.quarantine"
CANONICAL = RUNS / "m1877_m1811_c2_fresh_mapped_formality_dual_corner_pt_r1_20260902"
RUNNER = HW / "dc_handoff/scripts/run_m1877_c2_fresh_mapped_formality_dual_corner_pt_one_shot.py"
RELEASE = HW / "contracts/m1879_m1878_m1877_c2_fresh_mapped_formality_dual_corner_pt_launch_release_r1_20260902.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "runner": "e27a75adfd1febcfbbc32aa8def87ca785a225edaf861dcd0aa0c8a7d0822e87",
    "release": "4e8dc963c7c3527040be59338b380ff76d3ff561d0e5e9eb0f619456e3d25fd3",
    "attempt_json": "0017ae02e0d4a59e8de302a74b0ac2a4ad111be0794ec7b711cd964eaa4587d5",
    "attempt_manifest": "018302c62688b66c4a8e08ac196ef2f5147955e43dade915c143330ef0b4dc79",
    "attempt_outer_file": "e8d0956c8352f0ac62d0e06ecdd26d5b2f26eff642750f9894fa75fc37a9fe35",
    "failure_status": "a6edca0500ceb7b17f17f836ab422cdd7bda1b798d9b2178fb66281420e61b37",
    "failure_input": "5b9bdaa0275a1a1bef599df72bbff4f2f9e1dc1eb6a5d41a1dd495c36fb995ee",
    "failure_manifest": "dde515f1349ca69d31c2ff36ddb7e71e28e64ae4ea0e1cae34f64e1eca4c2c8d",
    "failure_outer_file": "e49f517c498036974b4788ed53297d882910bb142d3c93982c5f112cb581bb66",
    "fm_log": "0e747547a2b9723bd6b95f8aadeb65693934b19755e49876cd9490457028f8cf",
    "fm_status": "698d744258c8904208fb07f066d2d12fbbd4d042e4354b6b9e600b93486ad2fa",
    "fm_black_boxes": "4872f595e29882115e524691e4e0a3e1cc06c7171213f722707cb9298897dcbd",
    "pt_log": "840a7929647a85113cce9c3aabeac7f585b53f4f470e161bb3995371f820dbeb",
    "pt_summary": "4a3c64fd8fb1c7c4c1b2a74fb924e11b2f2f5b814d2753d5f2158d67b618b287",
    "pt_constraints": "7a1961cc5e39446af72d265467a85d554eee2d49aa6d1c1b6acdf74bade80341",
    "pt_coverage": "88bbbdd309a0d720192bc4159917bf98dff2530c5a8e481a6685cae398fd848b",
    "pt_violators": "5eca10eb092f5ef9cabf5fffef97c175fb52a3e70534ae3ec0d278ef0a55b3fd",
}


class AuditError(RuntimeError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_regular(path, expected):
    if (not path.is_file() or path.is_symlink()
            or not stat.S_ISREG(path.lstat().st_mode)
            or sha256(path) != expected):
        raise AuditError("identity mismatch: " + str(path))


def verify_sealed_directory(root, manifest_sha, outer_file_sha):
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    if not root.is_dir() or root.is_symlink():
        raise AuditError("sealed directory absent/invalid: " + str(root))
    exact_regular(manifest, manifest_sha)
    exact_regular(outer, outer_file_sha)
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


def strict_json(path):
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


def main():
    exact_regular(DOCS359, EXPECTED["docs359"])
    exact_regular(RUNNER, EXPECTED["runner"])
    exact_regular(RELEASE, EXPECTED["release"])
    verify_sealed_directory(ATTEMPT, EXPECTED["attempt_manifest"],
                            EXPECTED["attempt_outer_file"])
    verify_sealed_directory(FAILURE, EXPECTED["failure_manifest"],
                            EXPECTED["failure_outer_file"])
    exact_regular(ATTEMPT / "attempt.json", EXPECTED["attempt_json"])
    exact_regular(FAILURE / "RUN_FAILED_OR_INCOMPLETE.txt", EXPECTED["failure_status"])
    exact_regular(FAILURE / "input_identity.json", EXPECTED["failure_input"])

    attempt = strict_json(ATTEMPT / "attempt.json")
    if attempt != {
            "automatic_retry": False,
            "axes": ["K8", "K1X8"],
            "formality_runs": 2,
            "pt_runs": 2,
            "release_sha256": EXPECTED["release"],
            "schema": "m1877_c2_fresh_mapped_formality_dual_corner_pt_attempt_r1_v1",
            "status": "M1877_ATTEMPT_CONSUMED_BEFORE_FIRST_EDA"}:
        raise AuditError("attempt semantics drift")
    if (FAILURE / "RUN_FAILED_OR_INCOMPLETE.txt").read_text() != (
            "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\n"
            "error=K8 pt_shell reported Error/Fatal\n"
            "retry=false\n"):
        raise AuditError("failure terminal drift")
    namespace = list(RUNS.glob(
        "m1877_m1811_c2_fresh_mapped_formality_dual_corner_pt_r1_20260902*"))
    if namespace != [FAILURE] or CANONICAL.exists():
        raise AuditError("canonical/failure namespace cardinality drift")
    if any(RUNS.glob(
            ".m1877_m1811_c2_fresh_mapped_formality_dual_corner_pt_work.*")):
        raise AuditError("unsealed work namespace remains")
    if (FAILURE / "k1x8").exists():
        raise AuditError("K1X8 unexpectedly ran")

    fm = FAILURE / "k8/formality"
    fm_reports = fm / "reports"
    exact_regular(fm / "formality.log", EXPECTED["fm_log"])
    exact_regular(fm_reports / "formality_status.rpt", EXPECTED["fm_status"])
    exact_regular(fm_reports / "formality_black_boxes.rpt",
                  EXPECTED["fm_black_boxes"])
    if (fm / "formality.rc").read_text() != "0\n":
        raise AuditError("K8 Formality rc drift")
    status = (fm_reports / "formality_status.rpt").read_text(errors="replace")
    if (status.count("Verification SUCCEEDED") != 1
            or status.count("33656 Passing compare points") != 1):
        raise AuditError("K8 Formality proof drift")
    if not re.search(
            r"(?m)^Passing \(equivalent\)\s+0\s+0\s+0\s+0\s+1228\s+32428\s+0\s+33656$",
            status):
        raise AuditError("K8 passing/BBPin row drift")
    if not re.search(r"(?m)^Failing \(not equivalent\)\s+(?:0\s+){7}0$", status):
        raise AuditError("K8 failing row drift")
    for report, token in (("formality_unmatched.rpt", "No unmatched points."),
                          ("formality_failing.rpt", "No failing compare points."),
                          ("formality_aborted.rpt", "No aborted compare points."),
                          ("formality_unverified.rpt", "No unverified compare points.")):
        if (fm_reports / report).read_text(errors="replace").count(token) != 1:
            raise AuditError("non-clean K8 Formality report: " + report)
    black = (fm_reports / "formality_black_boxes.rpt").read_text(errors="replace")
    if len(re.findall(r"(?m)^e\s+SNPS_BUSHOLD$", black)) != 2:
        raise AuditError("SNPS_BUSHOLD symmetry drift")
    if len(re.findall(r"(?m)^\s*Instances\s*:\s*2 of 2\s*$", black)) != 2:
        raise AuditError("SNPS_BUSHOLD counts drift")
    design = black.split("####    DESIGN LIBRARY - r:/WORK", 1)[1].split(
        "####    TECH LIBRARY - r:/", 1)[0]
    if re.search(r"(?ms)^[ue*].*?Instances\s*:\s*[1-9]", design):
        raise AuditError("nonzero design-library unresolved entry")

    pt = FAILURE / "k8/pt"
    reports = pt / "reports"
    exact_regular(pt / "pt.log", EXPECTED["pt_log"])
    exact_regular(reports / "timing_summary_machine.txt", EXPECTED["pt_summary"])
    exact_regular(reports / "constraint_semantics_machine.txt",
                  EXPECTED["pt_constraints"])
    exact_regular(reports / "analysis_coverage.rpt", EXPECTED["pt_coverage"])
    exact_regular(reports / "constraint_violators.rpt", EXPECTED["pt_violators"])
    if (pt / "pt.rc").read_text() != "0\n":
        raise AuditError("K8 PT rc drift")
    log = (pt / "pt.log").read_text(errors="replace")
    errors = re.findall(r"(?m)^(?:Error|Fatal):.*$", log)
    if errors != ["Error: Library Compiler executable path is not set. (PT-063)"]:
        raise AuditError("PT Error/Fatal set drift")
    if log.count("Diagnostics summary: 1 error, 5 warnings, 31 informationals") != 1:
        raise AuditError("PT diagnostics summary drift")
    summary = (reports / "timing_summary_machine.txt").read_text()
    for token in ("setup_wns_ns=0.001767", "hold_wns_ns=-0.023259",
                  "setup_closed=1", "hold_closed=0"):
        if summary.count(token) != 1:
            raise AuditError("PT summary drift: " + token)
    constraints = (reports / "constraint_semantics_machine.txt").read_text()
    if (constraints.count("setup_violating_paths=0") != 1
            or constraints.count("hold_violating_paths=30442") != 1):
        raise AuditError("PT constraint-count drift")
    if len(re.findall(r"slack \(VIOLATED\)",
                      (reports / "constraint_violators.rpt").read_text())) != 30442:
        raise AuditError("raw hold-violation marker count drift")
    coverage = (reports / "analysis_coverage.rpt").read_text()
    for pattern in (
            r"setup\s+32429\s+32429 \(100%\)\s+0 \(\s*0%\)\s+0 \(\s*0%\)",
            r"hold\s+32429\s+1987 \(\s*6%\)\s+30442 \(\s*94%\)\s+0 \(\s*0%\)",
            r"out_setup\s+1228\s+1088 \(\s*89%\)\s+0 \(\s*0%\)\s+140 \(\s*11%\)",
            r"out_hold\s+1228\s+1088 \(\s*89%\)\s+0 \(\s*0%\)\s+140 \(\s*11%\)"):
        if re.search(pattern, coverage) is None:
            raise AuditError("PT coverage semantic drift: " + pattern)

    print("PASS_M1891_READ_ONLY_FAILURE_AUDIT__M1877_FAIL_CLOSED__K8_FM_DIAGNOSTIC_PASS__K8_PT_COMPLETE_BUT_HOLD_FAIL__K1X8_NOT_RUN__P0_0_P1_2_P2_1")


if __name__ == "__main__":
    main()
