#!/usr/bin/env python3
"""Read-only M2128 hammer for the consumed/failed M2127 diagnostic.

This program invokes no EDA, license query, GPU process, M2125 runner, or
M2126 hammer and writes no file.  It prints its independent mechanical audit
to stdout.  Review artifacts are separately frozen by exhaustive double seal.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
REPO = HW.parent
RUNNER = HW / "dc_handoff/scripts/run_m2125_m2018_tsbg_rtl_saif_window_diagnostic_one_shot.py"
CONTRACT = HW / "contracts/m2125_m2018_tsbg_rtl_saif_window_diagnostic_source_contract_r1_20260904.json"
M2126 = HW / "reviews/m2126_m2125_m2018_tsbg_rtl_saif_window_diagnostic_source_hammer_r1_20260904"
ATTEMPT = HW / "results/.m2127_m2125_tsbg_rtl_saif_window_diagnostic_attempt_consumed"
FAILURE = HW / "results/m2127_m2125_m2018_tsbg_rtl_saif_window_diagnostic_r1_20260904.failed.1975613.quarantine"
CANONICAL = HW / "results/m2127_m2125_m2018_tsbg_rtl_saif_window_diagnostic_r1_20260904"
LOCK = HW / "results/.m2127_m2125_tsbg_rtl_saif_window_diagnostic_launch_lock"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "runner": "6021c4a9b4297e5527f09006f21dd3a06d98b2a7ad76ffc55ca259029e658815",
    "contract": "5fadf923093797c8734e1aa54044cd2292e745ec541983254cea7a4c4ce4457e",
    "m2126_review": "9949b7f7dabfb03eb0d1c6d64e8cea3339a0221fd21ec9b72afb90f14bcb910f",
    "m2126_manifest": "db8f8bd83ddc6a483baff88bd1460e8b829b51757ec524421399a45d84235bdc",
    "m2126_outer": "d3313574bf92184c6029d078dfa8010e733c0936519f76e790add24e8f6a87f7",
    "attempt_manifest": "4f210d30e864462c43129d1718cdf2ef14737d0355abe0060f114b26fa8b1d85",
    "attempt_outer": "d6d2f5eff2ce0b61d4ab68aa8a4aad287c2180d6b7bdb7213f423f593c0fcc14",
    "failure_manifest": "d4cd1a06601a3575c9e02f3b57a3ed62f4ee6d64fd66b4c8719e8a6223c3ab6f",
    "failure_outer": "412d86b9f2a8500b8eb5663283ef2adaaa8631c11e5d233d38e6278f4d07d11e",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: Path) -> dict:
    def pairs(items):
        value = {}
        for key, item in items:
            assert key not in value, "duplicate key " + key
            value[key] = item
        return value
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          AssertionError("nonfinite " + token)))


def verify_seal(root: Path, manifest_sha: str, outer_sha: str) -> dict[str, str]:
    assert root.is_dir() and not root.is_symlink()
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    assert manifest.is_file() and not manifest.is_symlink()
    assert outer.is_file() and not outer.is_symlink()
    assert sha(manifest) == manifest_sha and sha(outer) == outer_sha
    assert outer.read_text().split() == [sha(manifest), "SHA256SUMS"]
    rows = {}
    for line in manifest.read_text().splitlines():
        digest, raw = line.split(maxsplit=1)
        rel = Path(raw.lstrip("*"))
        assert not rel.is_absolute() and ".." not in rel.parts
        path = root / rel
        assert rel.as_posix() not in rows
        assert path.is_file() and not path.is_symlink() and sha(path) == digest
        rows[rel.as_posix()] = digest
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.name not in {
                  "SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    assert set(rows) == actual
    return rows


def get_compile_ast(runner: str) -> ast.List:
    tree = ast.parse(runner)
    rows = [node.value for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name)
                    and target.id == "compile_command" for target in node.targets)]
    assert len(rows) == 1 and isinstance(rows[0], ast.List)
    return rows[0]


def main() -> None:
    checks: dict[str, bool] = {}

    def need(value: bool, label: str) -> None:
        checks[label] = bool(value)
        assert value, label

    need(sha(RUNNER) == EXPECTED["runner"], "m2125_runner_sha_exact")
    need(sha(CONTRACT) == EXPECTED["contract"], "m2125_contract_sha_exact")
    need(sha(DOC359) == EXPECTED["docs359"], "docs359_sha_exact")

    m2126_members = verify_seal(
        M2126, EXPECTED["m2126_manifest"], EXPECTED["m2126_outer"])
    need(set(m2126_members) == {"RUN_COMPLETE.txt", "mechanical_checks.json",
         "mechanical_checks.py", "review.json", "review.md"},
         "m2126_exhaustive_member_set")
    need(sha(M2126 / "review.json") == EXPECTED["m2126_review"],
         "m2126_review_sha_exact")

    attempt_members = verify_seal(
        ATTEMPT, EXPECTED["attempt_manifest"], EXPECTED["attempt_outer"])
    need(set(attempt_members) == {"attempt.json"},
         "attempt_exhaustive_single_member")
    attempt = strict_json(ATTEMPT / "attempt.json")
    need(attempt["status"] == "M2127_ATTEMPT_CONSUMED",
         "m2127_attempt_consumed")
    need(attempt["automatic_retry"] is False,
         "attempt_automatic_retry_false")

    failure_members = verify_seal(
        FAILURE, EXPECTED["failure_manifest"], EXPECTED["failure_outer"])
    need(set(failure_members) == {"FAILED_DO_NOT_CITE.txt",
         "execution_commands.json", "execution_counts.json",
         "license_preflight.log"}, "failure_exhaustive_four_members")
    failure_text = (FAILURE / "FAILED_DO_NOT_CITE.txt").read_text()
    need(failure_text.splitlines() == ["status=FAILED_DO_NOT_CITE",
         "exception=Failure: timing contamination", "automatic_retry=false"],
         "unique_failure_exact_timing_contamination")
    counts = strict_json(FAILURE / "execution_counts.json")
    need(counts == {"license_queries": 1, "vcs_compiles": 0,
         "simv_runs": 0, "saif_files": 0, "dc_runs": 0, "ptpx_runs": 0},
         "execution_counts_exact")
    commands = strict_json(FAILURE / "execution_commands.json")
    need(commands == {"license_preflight": [
         "/opt/synopsys/scl/2025.03/linux64/bin/lmutil", "lmstat", "-a",
         "-c", "27030@ic.ismd-nemo"]}, "only_license_command_recorded")
    license_text = (FAILURE / "license_preflight.log").read_text(errors="replace")
    need("license server UP" in license_text and "snpslmd: UP" in license_text,
         "license_preflight_completed_before_failure")
    need(not CANONICAL.exists() and not LOCK.exists(),
         "canonical_absent_and_lock_released")
    need(len(list(HW.glob("results/m2127_m2125_m2018_tsbg_rtl_saif_window_diagnostic_r1_20260904.failed.*.quarantine"))) == 1,
         "single_failure_quarantine")

    runner = RUNNER.read_text()
    need('need(not any("UNIT_DELAY" in item or "sdf" in item.lower()' in runner,
         "faulting_predicate_exactly_present")
    need(runner.count('"timing contamination"') == 1,
         "unique_timing_failure_site")
    need(runner.index('counts["license_queries"] += 1')
         < runner.index('need(not any("UNIT_DELAY" in item or "sdf" in item.lower()')
         < runner.index('counts["vcs_compiles"] += 1'),
         "failure_order_explains_counts")

    compile_ast = get_compile_ast(runner)
    literal_tokens = [elt.value for elt in compile_ast.elts
                      if isinstance(elt, ast.Constant) and isinstance(elt.value, str)]
    need("+vcs+initreg+random" in literal_tokens
         and not any(token == "+define+UNIT_DELAY"
                     or token.lower().startswith(("-sdf", "+sdf"))
                     for token in literal_tokens),
         "literal_option_tokens_have_no_unit_delay_or_sdf")
    active_sources = [REPO / line.strip() for line in
        (HW / "dc_handoff/filelists/tcasii_m2125_m2018_tsbg_rtl_saif_window_diagnostic_vcs.f").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")]
    active_text = "\n".join(path.read_text(errors="replace") for path in active_sources)
    need("UNIT_DELAY" not in active_text
         and not re.search(r"\$sdf_annotate|(?:^|\s)-sdf(?:\s|$)",
                           active_text, flags=re.I),
         "active_sources_have_no_unit_delay_or_sdf_annotation")

    # Reconstruct the exact argument shapes using a representative absolute
    # work path.  The path prefix is forced by tempfile(dir=HW/results), and
    # the repository component "SDformer" contains the case-insensitive
    # substring "sdf".  Three non-option pathname arguments therefore trip
    # the broad predicate before VCS is launched.
    build = (HW / "results/.m2127_m2125_work.PROOF/vcs_build").resolve()
    representative = [
        "/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs", "-full64", "-sverilog",
        "+v2k", "-timescale=1ns/1ps", "+vcs+initreg+random",
        "-debug_access+r", "-lca", "+vcs+lic+wait",
        f"-Mdir={build / 'csrc'}", "-f", str(build / "sources.absolute.f"),
        "-top", "tb_m2125_m2018_tsbg_rtl_saif_window_diagnostic",
        "-o", str(build / "simv"),
    ]
    substring_hits = [index for index, item in enumerate(representative)
                      if "sdf" in item.lower()]
    need(substring_hits == [9, 11, 15],
         "path_substring_hits_are_mdir_filelist_and_simv")
    need(all("SDformer" in representative[index] for index in substring_hits),
         "all_false_hits_derive_from_repository_component")
    need(not any(representative[index].lower().startswith(("-sdf", "+sdf"))
                 for index in substring_hits), "false_hits_are_not_sdf_options")
    need(any("UNIT_DELAY" in item or "sdf" in item.lower()
             for item in representative), "old_predicate_reproduces_false_positive")

    m2126_source = (M2126 / "mechanical_checks.py").read_text()
    need("compile_constants = [elt.value" in m2126_source
         and "if isinstance(elt, ast.Constant)" in m2126_source,
         "m2126_checked_only_ast_literal_constants")
    need("for x in compile_constants + sim_constants" in m2126_source,
         "m2126_sdf_check_excluded_dynamic_paths")
    need("SDformer" not in (M2126 / "review.md").read_text(),
         "m2126_review_did_not_cover_path_collision")

    need(counts["vcs_compiles"] == 0 and counts["simv_runs"] == 0,
         "no_vcs_binary_or_simulation_evidence")
    need(counts["saif_files"] == 0 and counts["dc_runs"] == 0
         and counts["ptpx_runs"] == 0,
         "no_saif_dc_or_pt_evidence")

    output = {
        "schema": "m2128_m2127_m2125_failure_mechanical_checks_r1_v1",
        "status": "PASS_M2128_READ_ONLY_MECHANICAL_CHECKS__M2127_CONSUMED_NO_RETRY",
        "date_cst": "2026-09-04",
        "eda_invoked": False,
        "license_query_invoked": False,
        "gpu_invoked": False,
        "checks": checks,
        "check_count": len(checks),
        "identity": {
            "runner_sha256": sha(RUNNER),
            "contract_sha256": sha(CONTRACT),
            "m2126_review_sha256": sha(M2126 / "review.json"),
            "attempt_manifest_sha256": sha(ATTEMPT / "SHA256SUMS"),
            "failure_manifest_sha256": sha(FAILURE / "SHA256SUMS"),
            "docs359_sha256": sha(DOC359),
        },
        "root_cause": {
            "classification": "fail_closed_string_predicate_false_positive",
            "false_positive_argument_indices": substring_hits,
            "false_positive_argument_roles": ["Mdir", "filelist", "simv_output"],
            "repository_component": "SDformer",
            "actual_unit_delay_or_sdf_option": False,
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
