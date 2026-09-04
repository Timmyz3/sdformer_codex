#!/usr/bin/env python3
"""Read-only M2136 audit of the sealed M2135 ICC2 failure.

This script never invokes ICC2, a license utility, or a GPU process.  It only
hashes and parses the already-consumed attempt, its quarantine, the frozen
source, local installed documentation, and the escaped ICC2 transcript.
"""

import hashlib
import os
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
ATTEMPT = HW / "dc_handoff/runs/.m2135_m2029_m2018_matched_macrofree_icc2_pnr_attempt_consumed"
QUARANTINE = HW / "dc_handoff/runs/m2135_m2029_m2018_matched_macrofree_icc2_pnr_raw_r1_20260904.failed_or_incomplete.2100851.quarantine"
RUNNER = HW / "dc_handoff/scripts/run_m2133_m2134_m2029_m2018_matched_macrofree_icc2_pnr_one_shot.sh"
TCL = HW / "dc_handoff/scripts/run_icc2_m2133_m2029_m2018_matched_macrofree_axis.tcl"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
COLLATERAL = REPO / "icc2_output.txt"
DOCROOT = Path("/opt/synopsys/icc2/V-2023.12-SP3/doc")

EXPECTED = {
    "attempt_manifest": "41afca390cf4525ec02a06eb20a154704c6fe158350d6e0fca0dd36fa628f341",
    "attempt_outer": "00070fbfe144ed28892f76d165b7043897a16bf6061fd9e055e98b90334e230f",
    "quarantine_manifest": "57f1a6f1d1da388f01bc36311bea843317b6f94d5cbab003a34bf70b254ed752",
    "quarantine_outer": "76f5ad6c1f0b6b08bd1c3adea881d4914126f0973f1f74034dedffd5eaa6e526",
    "ordinary_log": "b03a336bc2a9d687e602dffa92ac4707252cc5bca6e97dfbb00c54a0605f8ab6",
    "license_log": "d362f1062074d77f4628d29318b7ce80eb114f3151f54f5d62b60f915f26d5f5",
    "runner": "3cde47d675728007782e34020356ff0196df2e82bdc9cefe456e2ed86ae542d8",
    "tcl": "0df08207da8c5601c0b23b21bff9ee84e73594101ec654ee4a7071a191ca1e5b",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "collateral": "0410c14052c0b18c0f1a92246ecec4f109a9e37130b8f95f5cb4587cbcf863d6",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_manifest(path):
    out = {}
    for line in path.read_text().splitlines():
        digest, rel = line.split(None, 1)
        rel = rel.lstrip("* ")
        assert rel not in out
        out[rel] = digest
    return out


def verify_exhaustive_double_seal(directory):
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    listed = parse_manifest(manifest)
    actual = {
        str(p.relative_to(directory))
        for p in directory.rglob("*")
        if p.is_file() and not p.is_symlink()
        and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    }
    assert set(listed) == actual, (directory, sorted(set(listed) ^ actual))
    for rel, digest in listed.items():
        p = directory / rel
        assert p.is_file() and not p.is_symlink() and sha(p) == digest
    outer_items = parse_manifest(outer)
    assert outer_items == {"SHA256SUMS": sha(manifest)}


checks = []


def need(condition, label):
    assert condition, label
    checks.append(label)


verify_exhaustive_double_seal(ATTEMPT)
need(sha(ATTEMPT / "SHA256SUMS") == EXPECTED["attempt_manifest"], "attempt_manifest_sha_exact")
need(sha(ATTEMPT / "SHA256SUMS.seal.sha256") == EXPECTED["attempt_outer"], "attempt_outer_seal_sha_exact")
verify_exhaustive_double_seal(QUARANTINE)
need(sha(QUARANTINE / "SHA256SUMS") == EXPECTED["quarantine_manifest"], "quarantine_manifest_sha_exact")
need(sha(QUARANTINE / "SHA256SUMS.seal.sha256") == EXPECTED["quarantine_outer"], "quarantine_outer_seal_sha_exact")

attempt_text = (ATTEMPT / "ATTEMPT_CONSUMED.txt").read_text()
need("status=M2133_ATTEMPT_CONSUMED" in attempt_text, "attempt_consumed")
need("retry=false" in attempt_text, "attempt_no_retry")
need("license_queries=1" in attempt_text and "icc2_shell_runs=2" in attempt_text,
     "attempt_marker_records_authorized_budget_not_observed_count")

failure = (QUARANTINE / "RUN_FAILED_OR_INCOMPLETE.txt").read_text()
need("status=FAILED_OR_INCOMPLETE_DO_NOT_CITE" in failure, "failure_do_not_cite")
need("exit_code=42" in failure and "retry=false" in failure, "failure_exit42_no_retry")

log_path = QUARANTINE / "ordinary_lru4.icc2.log"
license_path = QUARANTINE / "license_preflight.log"
need(sha(log_path) == EXPECTED["ordinary_log"], "ordinary_log_sha_exact")
need(sha(license_path) == EXPECTED["license_log"], "license_log_sha_exact")
log = log_path.read_text(errors="replace")
license_text = license_path.read_text(errors="replace")
need(log.count("IC Compiler II (TM)") == 1, "one_observed_icc2_session_banner")
need("Users of ICCompilerII:" in license_text and license_text.count("Flexible License Manager status") == 1,
     "one_observed_license_preflight_log")
need((QUARANTINE / "ordinary_lru4").is_dir(), "ordinary_axis_started")
need(not (QUARANTINE / "tsbg_b4").exists(), "tsbg_axis_not_started")
need(not (QUARANTINE / "tsbg_b4.icc2.log").exists(), "tsbg_icc2_log_absent")

for token in ("CMD-104", "LIB-117", "FILE-001", "LIB-027"):
    need(log.count(token) == 1, f"unique_runtime_{token.lower().replace('-', '_')}")
need("Variable 'lib.configuration.local_output_dir' is not an application variable. Using Tcl global variable." in log,
     "first_cause_wrong_setter_message")
need("Library configuration will not be performed: technology file not specified." in log,
     "library_configuration_skipped")
need("tcbn28hpcplusbwp35p140/lib.ndm' for reading; No such file or directory" in log,
     "milkyway_directory_misread_as_missing_ndm")
need("Reference library path" in log and "is not a valid library (LIB-027)" in log,
     "invalid_reference_library_cascade")
need("M2133_FATAL_FAIL_CLOSED: problem in create_lib" in log, "create_lib_fail_closed")

ordinary = QUARANTINE / "ordinary_lru4"
for sub in ("library_cache", "output", "raw_parasitics", "reports"):
    d = ordinary / sub
    need(d.is_dir() and not any(d.iterdir()), f"empty_before_design_{sub}")
design_names = {"machine_facts.txt", "RUN_COMPLETE.txt", "routed.v", "routed.sdc", "routed.def", "routed.spef"}
need(not any(p.name in design_names or p.suffix in {".def", ".spef"} for p in QUARANTINE.rglob("*")),
     "no_design_or_pnr_artifact")

need(sha(RUNNER) == EXPECTED["runner"], "runner_sha_exact")
need(sha(TCL) == EXPECTED["tcl"], "tcl_sha_exact")
need(sha(DOCS359) == EXPECTED["docs359"], "docs359_sha_exact")
runner = RUNNER.read_text()
need('cd -- "${REPO_ROOT}"' in runner, "runner_cwd_explains_repo_sidecar")
need('axis_names=(ordinary_lru4 tsbg_b4)' in runner, "runner_axis_order_ordinary_then_tsbg")
need('"${ICC2}" -f "${TCL}" >"${axis_dir}.icc2.log" 2>&1' in runner,
     "runner_sequential_icc2_site")

need(COLLATERAL.is_file() and not COLLATERAL.is_symlink(), "escaped_collateral_regular")
need(COLLATERAL.stat().st_size == 25324, "escaped_collateral_size_exact")
need(sha(COLLATERAL) == EXPECTED["collateral"], "escaped_collateral_sha_exact")
need(len(COLLATERAL.read_text(errors="replace").splitlines()) == 472, "escaped_collateral_line_count")
tracked = subprocess.run(
    ["git", "ls-files", "--error-unmatch", "icc2_output.txt"], cwd=REPO,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
).returncode == 0
need(not tracked, "escaped_collateral_untracked")
need("icc2_output.txt" not in parse_manifest(QUARANTINE / "SHA256SUMS"),
     "escaped_collateral_not_in_quarantine_manifest")
need("CMD-104" in COLLATERAL.read_text(errors="replace") and "LIB-027" in COLLATERAL.read_text(errors="replace"),
     "escaped_collateral_is_failure_transcript")

docs = {
    "set_app_options": DOCROOT / "ICC2/man/cat2/set_app_options.2",
    "get_app_option_value": DOCROOT / "ICC2/man/cat2/get_app_option_value.2",
    "local_output": DOCROOT / "ICC2/man/cat3/lib.configuration.local_output_dir.3",
    "create_lib": DOCROOT / "ICC2/man/cat2/create_lib.2",
    "generate_frame": DOCROOT / "ICC2/man/cat2/generate_frame_from_mw.2",
    "cmd104": DOCROOT / "snps_tcl/man/catn/CMD/CMD-104.n",
    "lib117": DOCROOT / "ICC2/man/catn/LIB/LIB-117.n",
    "lib027": DOCROOT / "ICC2/man/catn/LIB/LIB-027.n",
}
for label, path in docs.items():
    need(path.is_file() and path.stat().st_size > 0, f"official_doc_present_{label}")
need("-name" in docs["set_app_options"].read_text(errors="replace") and
     "-value" in docs["set_app_options"].read_text(errors="replace"),
     "official_set_app_options_name_value_syntax")
need("Returns  the  value" in docs["get_app_option_value"].read_text(errors="replace") and
     "-name" in docs["get_app_option_value"].read_text(errors="replace"),
     "official_get_app_option_value_query_syntax")
need("converted  frame libs will be put under" in docs["local_output"].read_text(errors="replace"),
     "official_local_output_dir_controls_mw_conversion")
need("Milkyway  libraries" in docs["create_lib"].read_text(errors="replace"),
     "official_create_lib_accepts_milkyway_physical_source")
need("-technology" in docs["create_lib"].read_text(errors="replace"),
     "official_create_lib_technology_option")
need("generate_frame_from_mw my_frame.ndm -mw_lib" in docs["generate_frame"].read_text(errors="replace"),
     "official_generate_frame_from_mw_preconversion")
need("will treat all  files  pro-" in docs["lib027"].read_text(errors="replace") and
     "full NDM reference libraries" in docs["lib027"].read_text(errors="replace"),
     "official_lib027_explains_missing_technology_fallback")

print("PASS_M2136_READ_ONLY_FAILURE_MECHANICAL_CHECKS")
print(f"checks_total={len(checks)}")
print(f"checks_passed={len(checks)}")
print("observed_license_queries=1")
print("observed_icc2_shell_runs=1")
print("observed_axes=ordinary_lru4:1,tsbg_b4:0")
print("design_or_pnr_results=0")
print("escaped_collateral=icc2_output.txt")
print("escaped_collateral_sha256=" + EXPECTED["collateral"])
print("eda_invoked_by_reviewer=false")
print("license_query_invoked_by_reviewer=false")
print("gpu_invoked_by_reviewer=false")
for label in checks:
    print("PASS " + label)
