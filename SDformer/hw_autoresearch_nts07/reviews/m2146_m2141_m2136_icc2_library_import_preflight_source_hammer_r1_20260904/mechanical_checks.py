#!/usr/bin/python3.12
"""Independent, read-only M2146 hammer of the M2141 ICC2 preflight source.

This script never invokes ICC2, lmutil, another license client, or a GPU.  It
hashes frozen inputs, re-derives the mapped-master union, checks the prior
failure evidence, and attacks the raw-result parser using synthetic temporary
fixtures.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
HERE = Path(__file__).resolve().parent

CONTRACT = HW / "contracts/m2141_m2136_icc2_library_import_preflight_source_contract_r1_20260904.json"
MASTER_LIST = HW / "dc_handoff/manifests/m2141_m2029_union94_mapped_master_names_r1_20260904.txt"
MW_MANIFEST = HW / "dc_handoff/manifests/m2133_tcbn28hpcplusbwp35p140_complete_milkyway_inventory_r1_20260904.sha256"
RUNNER = HW / "dc_handoff/scripts/run_m2141_m2136_icc2_library_import_preflight_one_shot.sh"
TCL = HW / "dc_handoff/scripts/run_icc2_m2141_library_import_preflight.tcl"
MONITOR = HW / "dc_handoff/scripts/monitor_m2141_icc2_process_tree.py"
CHECKER = HW / "system_simulator/scripts/check_m2141_icc2_library_import_preflight.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
M2136 = HW / "reviews/m2136_m2135_m2133_m2029_m2018_matched_macrofree_icc2_pnr_failure_hammer_r1_20260904"
AUTHOR_RECEIPT = HW / "reviews/m2141_m2136_icc2_library_import_preflight_source_author_receipt_r1_20260904"
M2135_ATTEMPT = HW / "dc_handoff/runs/.m2135_m2029_m2018_matched_macrofree_icc2_pnr_attempt_consumed"
M2135_QUAR = HW / "dc_handoff/runs/m2135_m2029_m2018_matched_macrofree_icc2_pnr_raw_r1_20260904.failed_or_incomplete.2100851.quarantine"
M2135_RUNNER = HW / "dc_handoff/scripts/run_m2133_m2134_m2029_m2018_matched_macrofree_icc2_pnr_one_shot.sh"
M2135_TCL = HW / "dc_handoff/scripts/run_icc2_m2133_m2029_m2018_matched_macrofree_axis.tcl"
COLLATERAL = REPO / "icc2_output.txt"

M2029 = HW / "dc_handoff/runs/m2029_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc_r1_20260902"
NETLISTS = [
    M2029 / "ordinary_lru4/netlist/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_mapped.v",
    M2029 / "tsbg_b4/netlist/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_mapped.v",
]

TECH_BASE = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital")
MW_REF = TECH_BASE / "Back_End/milkyway/tcbn28hpcplusbwp35p140_110a/frame_only_VHV_0d5_0/tcbn28hpcplusbwp35p140"
DB_BASE = TECH_BASE / "Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a"
TT_DB = DB_BASE / "tcbn28hpcplusbwp35p140tt0p9v25c.db"
SS_DB = DB_BASE / "tcbn28hpcplusbwp35p140ssg0p9v125c.db"
FF_DB = DB_BASE / "tcbn28hpcplusbwp35p140ffg1p05vm40c.db"
NXTGRD = Path("/opt/tech/tsmc28/RC_Extraction/starRC/typical/crn28hpc+_1p09m+ut-alrdl_6x1z1u_typical.nxtgrd")
LAYER_MAP = Path("/opt/tech/tsmc28/RC_Extraction/starRC/typical/Reference/MAP/star.map_icc_crn28hpc+_1p9m_6x1z1u_ut-alrdl")
ICC2_WRAPPER = Path("/opt/synopsys/icc2/V-2023.12-SP3/bin/icc2_shell")
ICC2_EXEC = Path("/opt/synopsys/icc2/V-2023.12-SP3/linux64/nwtn/bin/dgcom_exec")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
DOCROOT = Path("/opt/synopsys/icc2/V-2023.12-SP3/doc")

EXPECTED = {
    "contract": "89f38eabd33e7e4def2c377bf0ba3546d0e5de523dc10fab558bdacc0b5f9d73",
    "masters": "e6a8c7c500c587631715d5b1718cf928c253e1eb089a96b3b648b375faefa90b",
    "runner": "73dd2511bb4b7e94d99a39f7764a66876545cdfd1d970a164210a9c6eabd7276",
    "tcl": "f037a996e038e4e72094d272226032091219c94f6e37e00d8e4bb89a6e60d611",
    "monitor": "32399fde5cea3487439feee9b919322197bbbec3730d6d0298bce83a9456c268",
    "checker": "5324e0ac3a8cf53ff00ab9070f340bce24551ca6763081127bfc670b28ac40a0",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "m2136_review": "0207977db8f1d1ef5d10ca4af97c87b7eb074bb2000d0b7ca11c1a11da7ef552",
    "m2135_attempt_manifest": "41afca390cf4525ec02a06eb20a154704c6fe158350d6e0fca0dd36fa628f341",
    "m2135_attempt_outer": "00070fbfe144ed28892f76d165b7043897a16bf6061fd9e055e98b90334e230f",
    "m2135_quar_manifest": "57f1a6f1d1da388f01bc36311bea843317b6f94d5cbab003a34bf70b254ed752",
    "m2135_quar_outer": "76f5ad6c1f0b6b08bd1c3adea881d4914126f0973f1f74034dedffd5eaa6e526",
    "m2135_log": "b03a336bc2a9d687e602dffa92ac4707252cc5bca6e97dfbb00c54a0605f8ab6",
    "m2135_runner": "3cde47d675728007782e34020356ff0196df2e82bdc9cefe456e2ed86ae542d8",
    "m2135_tcl": "0df08207da8c5601c0b23b21bff9ee84e73594101ec654ee4a7071a191ca1e5b",
    "collateral": "0410c14052c0b18c0f1a92246ecec4f109a9e37130b8f95f5cb4587cbcf863d6",
    "mw_manifest": "7a50f23c8e5b164efe08b609409d43f781287c809e42a328bad10835fc1431d3",
    "ordinary_netlist": "f5847f355329a52511ab044ef458284a19ae424ac778418a4bc4778bb2d3a2b0",
    "tsbg_netlist": "739eb76dcb732ec0c66b75392c768cbe36027ecc5d458bd4b088f8488f67c9af",
    "tt": "d8975a427b9f5f6b6667ee5dbc7ff33eac15ab480a871d756af48cd9afa18070",
    "ss": "79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af",
    "ff": "a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a",
    "nxtgrd": "424477b89c352173da2c3adc1d723764e8ff68425289ef688793be364646fd02",
    "layer_map": "da6e70dae3b50cc8e7520e3576477f2f80c3ac55dbe2b61baad73eb36fe44ed3",
    "icc2_wrapper": "825f5d687e1a5f5ecf31d4439c867c50f1eef6fd33c967f2f17bf3ad6de6c2e4",
    "icc2_exec": "4b43acaeabd6243320e657daa4202b831bf11a60de53d6f82ac5e35092cccb1c",
    "lmutil": "e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        digest, raw = line.split(None, 1)
        rel = raw.lstrip("* ")
        assert rel not in result, (path, rel)
        result[rel] = digest
    return result


def exhaustive_double_seal(directory: Path) -> None:
    assert directory.is_dir() and not directory.is_symlink(), directory
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    listed = parse_manifest(manifest)
    actual = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and not path.is_symlink()
        and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    }
    assert set(listed) == actual, (directory, sorted(set(listed) ^ actual))
    assert not any(path.is_symlink() for path in directory.rglob("*")), directory
    for rel, digest in listed.items():
        assert sha(directory / rel) == digest, (directory, rel)
    assert parse_manifest(outer) == {"SHA256SUMS": sha(manifest)}, directory


def verify_listed_double_seal(directory: Path) -> set[str]:
    """Verify every listed file and return unlisted regular-file collateral."""
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    listed = parse_manifest(manifest)
    for rel, digest in listed.items():
        path = directory / rel
        assert path.is_file() and not path.is_symlink() and sha(path) == digest
    assert parse_manifest(outer) == {"SHA256SUMS": sha(manifest)}
    actual = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and not path.is_symlink()
        and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    }
    return actual - set(listed)


checks: list[str] = []


def need(condition: bool, label: str) -> None:
    assert condition, label
    checks.append(label)


def make_fake_work(root: Path) -> Path:
    work = root / "work"
    reports = work / "isolated_cwd/reports"
    reports.mkdir(parents=True)
    (work / "isolated_cwd/frame_output").mkdir()
    (work / "isolated_cwd/m2141_disposable_design.nlib").mkdir()
    (work / "prior_m2135_collateral").mkdir()
    shutil.copyfile(COLLATERAL, work / "prior_m2135_collateral/icc2_output.txt")
    terminal = "RAW_PASS_M2141_LIBRARY_IMPORT_PREFLIGHT_PENDING_M2148_INDEPENDENT_RESULT_HAMMER"
    log_lines = [f"M2141_GATE{n}_SYNTHETIC_PASS" for n in range(1, 6)] + [terminal]
    (work / "icc2_preflight.log").write_text("\n".join(log_lines) + "\n")
    (work / "icc2_preflight.rc").write_text("0\n")
    facts = {
        "status": "RAW_PASS_M2141_LIBRARY_IMPORT_PREFLIGHT_PENDING_M2148",
        "conversion_status": "1",
        "mapped_master_union_count": "94",
        "tt_master_coverage": "94",
        "ss_master_coverage": "94",
        "ff_master_coverage": "94",
        "physical_master_coverage": "94",
        "routing_layers": "M1,M2,M3,M4,M5,M6,M7,M8,M9",
        "via_layers": "VIA1,VIA2,VIA3,VIA4,VIA5,VIA6,VIA7,VIA8",
        "rc_technology_name": "crn28hpc+_1p09m+ut-alrdl_6x1z1u_typical",
        "rtl_imported": "false",
        "pnr_invoked": "false",
    }
    (reports / "machine_facts.txt").write_text("".join(f"{k}={v}\n" for k, v in facts.items()))
    rows = ["master\ttt\tss\tff\tphysical"]
    rows.extend(f"{name}\t1\t1\t1\t1" for name in MASTER_LIST.read_text().splitlines())
    (reports / "master_coverage.tsv").write_text("\n".join(rows) + "\n")
    (work / "process_tree.json").write_text(json.dumps({
        "root_seen": True,
        "unique_process_identity_count": 2,
        "tool_spawned_conversion_child_count": 0,
        "all_observed_processes": [],
    }) + "\n")
    (work / "repo_root_before.sha256").write_text("synthetic\n")
    (work / "repo_root_after.sha256").write_text("synthetic\n")
    return work


def checker_accepts(work: Path) -> bool:
    output = work / "receipt.json"
    proc = subprocess.run(
        [str(CHECKER), "--work", str(work), "--output", str(output)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    return proc.returncode == 0 and "PASS_RAW_M2147" in proc.stdout


# Exact M2141 identities.
for label, path in {
    "contract": CONTRACT, "masters": MASTER_LIST, "runner": RUNNER, "tcl": TCL,
    "monitor": MONITOR, "checker": CHECKER, "docs359": DOCS359,
}.items():
    need(path.is_file() and not path.is_symlink(), f"{label}_regular_nonsymlink")
    need(sha(path) == EXPECTED[label], f"{label}_sha_exact")

contract = json.loads(CONTRACT.read_text())
need(contract["status"] == "SOURCE_ONLY_PENDING_M2146_INDEPENDENT_HAMMER", "contract_source_only")
need(contract["exact_runtime_budget_after_m2146_pass"] == {
    "license_queries": 1,
    "top_level_icc2_shell_runs": 1,
    "pnr_runs": 0,
    "automatic_retry": False,
    "tool_spawned_conversion_children": "observed and counted, not budgeted as top-level shells",
}, "contract_exact_budget")

# M2135/M2136 failure lineage and exhaustive double seals.
exhaustive_double_seal(M2135_ATTEMPT)
exhaustive_double_seal(M2135_QUAR)
exhaustive_double_seal(M2136)
author_extras = verify_listed_double_seal(AUTHOR_RECEIPT)
need(author_extras == {
    "__pycache__/selfcheck.cpython-312.pyc",
    "__pycache__/tests.cpython-312.pyc",
}, "finding_author_receipt_has_two_unsealed_pycache_files")
for label, path in {
    "m2136_review": M2136 / "review.json",
    "m2135_attempt_manifest": M2135_ATTEMPT / "SHA256SUMS",
    "m2135_attempt_outer": M2135_ATTEMPT / "SHA256SUMS.seal.sha256",
    "m2135_quar_manifest": M2135_QUAR / "SHA256SUMS",
    "m2135_quar_outer": M2135_QUAR / "SHA256SUMS.seal.sha256",
    "m2135_log": M2135_QUAR / "ordinary_lru4.icc2.log",
    "m2135_runner": M2135_RUNNER,
    "m2135_tcl": M2135_TCL,
    "collateral": COLLATERAL,
}.items():
    need(sha(path) == EXPECTED[label], f"{label}_sha_exact")
m2135_log = (M2135_QUAR / "ordinary_lru4.icc2.log").read_text(errors="replace")
for token in ("CMD-104", "LIB-117", "FILE-001", "LIB-027"):
    need(m2135_log.count(token) == 1, f"m2135_unique_{token.lower().replace('-', '_')}")
need("M2133_FATAL_FAIL_CLOSED: problem in create_lib" in m2135_log, "m2135_create_lib_fail_closed")
need(not (M2135_QUAR / "tsbg_b4").exists(), "m2135_tsbg_never_started")

# Frozen technology, tools and official command references.
for label, path in {
    "tt": TT_DB, "ss": SS_DB, "ff": FF_DB, "nxtgrd": NXTGRD,
    "layer_map": LAYER_MAP, "icc2_wrapper": ICC2_WRAPPER,
    "icc2_exec": ICC2_EXEC, "lmutil": LMUTIL,
}.items():
    need(path.is_file() and not path.is_symlink(), f"{label}_regular_nonsymlink")
    need(sha(path) == EXPECTED[label], f"{label}_sha_exact")
official = {
    "set_app_options": (DOCROOT / "ICC2/man/cat2/set_app_options.2", "ae28a2f50dc5ed7457adad00428a0c0e7fa57cc4555866015d4ab4563e4ec0da"),
    "get_app_option_value": (DOCROOT / "ICC2/man/cat2/get_app_option_value.2", "f0d7b2b4334d00f90432c7fcdb319fe80668578633dfbda0bcdc644302e4e47a"),
    "local_output_dir": (DOCROOT / "ICC2/man/cat3/lib.configuration.local_output_dir.3", "5354ec5b5964e454395a8f8d8cfecd489470d5c6555ec78242213d5925c6d9ea"),
    "generate_frame_from_mw": (DOCROOT / "LM/man/cat2/generate_frame_from_mw.2", "f9424346c44d9d48cbae5a3839f26cadad46b4d85e405deb19354356cd232952"),
    "create_lib": (DOCROOT / "LM/man/cat2/create_lib.2", "c19f9fd04239f0be10b97816cb4913ba71868b2b02f7c760d443cebdd40d835b"),
    "icc2_shell": (DOCROOT / "ICC2/man/cat1/icc2_shell.1", "2662ac4bfae4515c12e4f08e172c9754f2894267bb2891ff3ecc0b4f4674ff26"),
    "read_parasitic_tech": (DOCROOT / "ICC2/man/cat2/read_parasitic_tech.2", "b55d4c3092acfaf0f94a37158882f884eef15bea0631a72d8b0e71963f7683ee"),
    "get_techs": (DOCROOT / "ICC2/man/cat2/get_techs.2", "28f7ad66b006c4c66583356c26ea2b66131f86116925443edd3c996374f1ddca"),
    "get_layers": (DOCROOT / "ICC2/man/cat2/get_layers.2", "2708ba1e09a4283067d53ad856bb50084719fde08e7637b674d8ae7fe9be71f5"),
    "get_site_defs": (DOCROOT / "ICC2/man/cat2/get_site_defs.2", "a5141d398d746058266a51c8909d804a200e9479f95e307502da29064ea896ff"),
}
for label, (path, digest) in official.items():
    need(path.is_file() and sha(path) == digest, f"official_{label}_sha_exact")
need("-no_init" in official["icc2_shell"][0].read_text(errors="replace"), "official_no_init_available")

# Exhaustive Milkyway identity, including path-set equality and no symlinks.
need(sha(MW_MANIFEST) == EXPECTED["mw_manifest"], "mw_manifest_sha_exact")
mw_listed = parse_manifest(MW_MANIFEST)
mw_actual = {
    str(path.relative_to(MW_REF))
    for path in MW_REF.rglob("*") if path.is_file() and not path.is_symlink()
}
need(set(mw_listed) == mw_actual and len(mw_actual) == 1051, "mw_inventory_exhaustive_1051")
need(not any(path.is_symlink() for path in MW_REF.rglob("*")), "mw_inventory_no_symlinks")
for rel, digest in mw_listed.items():
    need(sha(MW_REF / rel) == digest, f"mw_file_sha_{rel}")

# Re-derive the 94-master union with a broader independent instance grammar.
for label, path in zip(("ordinary_netlist", "tsbg_netlist"), NETLISTS):
    need(sha(path) == EXPECTED[label], f"{label}_sha_exact")
cell_re = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_$]*)\s+(?:\\\S+|[A-Za-z_][A-Za-z0-9_$]*)\s*\(", re.MULTILINE)
union: set[str] = set()
for path in NETLISTS:
    union.update(m.group(1) for m in cell_re.finditer(path.read_text()) if m.group(1).endswith("BWP35P140"))
masters = MASTER_LIST.read_text().splitlines()
need(len(masters) == 94 and masters == sorted(set(masters)), "master_list_94_sorted_unique")
need(masters == sorted(union), "master_list_equals_independent_netlist_union")

# Static command/budget checks.  These pass for the intended commands but also
# expose the startup-file and census-boundary defects recorded by the review.
runner = RUNNER.read_text()
tcl = TCL.read_text()
need(runner.count('"${LMUTIL}" lmstat') == 1, "one_license_query_site")
need(runner.count('"${ICC2}" -f "${TCL}"') == 1, "one_top_level_icc2_site")
need("place_opt" not in tcl and "clock_opt" not in tcl and "route_auto" not in tcl,
     "no_explicit_pnr_in_tcl")
need("read_verilog" not in tcl and "compile_fusion" not in tcl, "no_explicit_rtl_or_compile_in_tcl")
need("set_app_options -name lib.configuration.local_output_dir -value $cache" in tcl,
     "documented_option_setter")
need("get_app_option_value -name lib.configuration.local_output_dir" in tcl,
     "documented_option_query")
need("generate_frame_from_mw $frame_name -mw_lib $mw_ref" in tcl and "-overwrite" not in tcl,
     "single_nonoverwrite_frame_conversion")
need("create_lib -ref_libs [list $frame_ndm] $design_lib" in tcl,
     "create_lib_uses_converted_frame")
need("set_app_var link_library [list $tt_db $ss_db $ff_db]" in tcl,
     "create_lib_logic_inputs_exact_tt_ss_ff")

# Findings that prevent authorization of the exact M2141 source.
need('"${ICC2}" -no_init -f "${TCL}"' not in runner, "finding_icc2_no_init_missing")
need("HOME=" not in runner.partition("env -i")[2].partition('"${ICC2}"')[0],
     "finding_isolated_home_not_set")
need("find -P . -maxdepth 1 -type f" in runner,
     "finding_repo_root_snapshot_regular_files_only")
monitor_text = MONITOR.read_text()
need('"lm_shell_exec"' not in monitor_text and '"icc2_exec"' not in monitor_text,
     "finding_process_classifier_omits_actual_exec_names")
need("get_site_defs -quiet *core*" in tcl, "finding_core_site_gate_has_wildcard_fallback")
need(EXPECTED["icc2_exec"] not in contract["identity"].values(),
     "finding_actual_icc2_exec_not_contract_pinned")

# Independent fake-log/parser attacks.  The checker should reject forged gate
# semantics and changed coverage identity, but the exact M2141 checker accepts
# them.  These are recorded as findings, not as successes of the source.
with tempfile.TemporaryDirectory(prefix="m2146_fake_") as raw:
    root = Path(raw)
    baseline = make_fake_work(root / "baseline")
    need(checker_accepts(baseline), "synthetic_baseline_is_parser_accepted")

    forged_gate = make_fake_work(root / "forged_gate")
    p = forged_gate / "icc2_preflight.log"
    p.write_text(p.read_text().replace("M2141_GATE1_SYNTHETIC_PASS", "M2141_GATE1_FORGED_SEMANTICS"))
    need(checker_accepts(forged_gate), "finding_forged_gate_semantics_parser_accepted")

    wrong_master = make_fake_work(root / "wrong_master")
    p = wrong_master / "isolated_cwd/reports/master_coverage.tsv"
    lines = p.read_text().splitlines()
    lines[1] = "AAA_FAKE_MASTER\t1\t1\t1\t1"
    p.write_text("\n".join(lines) + "\n")
    need(checker_accepts(wrong_master), "finding_wrong_master_identity_parser_accepted")

    fake_process = make_fake_work(root / "fake_process")
    # The baseline already claims two process identities while listing zero.
    need(checker_accepts(fake_process), "finding_inconsistent_process_census_parser_accepted")

    empty_outputs = make_fake_work(root / "empty_outputs")
    # The empty frame and design-lib directories are sufficient for the checker.
    need(checker_accepts(empty_outputs), "finding_empty_frame_and_design_lib_parser_accepted")

print("FAIL_M2146_M2141_SOURCE_HAMMER__M2147_NOT_AUTHORIZED")
print(f"mechanical_checks_passed={len(checks)}")
print("p0=1 missing_-no_init_allows_unsealed_startup_commands_before_Tcl")
print("p1=5 parser_fake_gate_and_master_acceptance;incomplete_process_census;incomplete_root_snapshot;author_receipt_nonexhaustive")
print("p2=3 actual_exec_unpinned;core_wildcard_fallback;isolated_HOME_unset")
print("eda_invoked=false")
print("license_query_invoked=false")
print("gpu_invoked=false")
print("m2147_authorized=false")
