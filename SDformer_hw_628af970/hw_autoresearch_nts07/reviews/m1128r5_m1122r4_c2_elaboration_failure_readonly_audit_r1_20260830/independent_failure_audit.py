#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Read-only M1128r5 audit of the consumed M1122r4 failure.

No launcher, engine, simulator, synthesis, pgrep, lmstat, or source mutation is
performed.  The script reads sealed evidence and compares source/log structure.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import stat


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
ATTEMPT = HW / "results/.m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_attempt_consumed"
FAILURE = HW / "results/m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_r1_20260830.failed_or_incomplete.580027.quarantine"
R2_RTL = HW / "rtl_m1112r2/m1112r2_c2_k1_async_observation_shadow_wrapper.sv"
BASE_RTL = HW / "rtl_m1112/m1112_c2_k1_async_observation_shadow_wrapper.sv"
R2_TB = HW / "dc_handoff/tb/tb_m1112r2_c2_k1_async_observation_shadow_case0_short.sv"
BASE_TB = HW / "dc_handoff/tb/tb_m1112_c2_k1_async_observation_shadow_case0_short.sv"
FILELIST = HW / "dc_handoff/filelists/date_m1112r2_c2_k1_async_observation_shadow_logic_only_dc.f"
TCL = HW / "dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl"
ENGINE = HW / "dc_handoff/scripts/m1122r4_c2_dc_selector_async_observation_engine_source_r1.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
M872 = HW / "dc_handoff/runs/m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829"
M917 = HW / "dc_handoff/runs/m917_m518_r5_fixed_descendant_safe_setup_area_logic_only_dc_3p000ns_r1_20260829"
M903 = HW / "reviews/m903_m872_m803_c2_r16_three_axis_dc_result_hammer_r1_20260829"
M928 = HW / "reviews/m928_m917_m518_r5_fixed_dc_result_hammer_r1_20260829"
OUTPUT = HERE / "mechanical_checks.json"

EXPECTED = {
    ATTEMPT / "SHA256SUMS.seal.sha256": "8a012c8638c2e8a8da743cbf570a13f5c8bc8d85716b433882d03405e12e5e37",
    FAILURE / "SHA256SUMS.seal.sha256": "2f9173b1e988b1f639e6c3d683fdf720fa9debfeaca8caf27bf5845a36527f83",
    R2_RTL: "b1fccaa03b1e3c69205d440ed0e2af93beb0f6eca68e7f7291c67f56322e89f5",
    BASE_RTL: "95c31bc70a7617c6653eaca2f77a54388119f744b814dfc909c75edad1c39218",
    R2_TB: "134c4a430d1daa257d73403612cdf41a2bb75369a4f16026413304d38e828d9c",
    BASE_TB: "ff6bd371c3b1371c520b38680960ad0297a8c01eb92eb7b4a0f4d2e59fc861b6",
    FILELIST: "d7fce19161ea6b0411eb07b6036209d32eac40a30e2cacab9673b72d454a746d",
    TCL: "c9da61c9a483487b3d1157538481a6c940d7277534e2acef634c4b1a1ff7adbe",
    FAILURE / "dc/dc.log": "b20b3d163fbeb1d78ceb5da390d506fd4d1c5828eb98fb480a7ada5833075d4c",
    FAILURE / "dc/dc_selector_runtime_identity.json": "7cfdd39dd40c2a45cb2a26d66dfc7bd4d7b6967ef2e66ccd65b2c05926af29b4",
    M872 / "k1/dc.log": "8b99a6b9a578ec539d7a9b3ca96e06946304d4438bbf6283365647fadfab2cda",
    M917 / "fixed/dc.log": "dacbeddb401670818b6b0f302694ef2e1e6fca3c9268666fff7114cba4d2bc83",
    M872 / "SHA256SUMS.seal.sha256": "0c9da50fc21c97b66f192779e10a50de2319ddc77da51e236ed6ee786aafcd5e",
    M917 / "SHA256SUMS.seal.sha256": "e2f619c321218d78537528bb53d6de7b8817316008840198703103ff4c8c75b9",
    M903 / "SHA256SUMS.seal.sha256": "0394ce7e485c780355dbb841797f7fa518171bb00330ae07234a1a9a4e96316f",
    M928 / "SHA256SUMS.seal.sha256": "43e6cee08ed52c52d1e46d48afc8b6835fd735e74ce4320b671cd401cf9c17d3",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


class Reject(RuntimeError):
    pass


checks = 0


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise Reject(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_pairs(rows):
    result = {}
    for key, value in rows:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def load(path: Path):
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(), "direct regular JSON")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(Reject("nonfinite " + token)))


def manifest_rows(path: Path) -> dict[str, str]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]) is not None,
                "manifest row")
        name = fields[1].lstrip("*"); rel = Path(name)
        require(name and name not in rows and name == rel.as_posix() and not rel.is_absolute() and
                ".." not in rel.parts, "safe manifest member")
        rows[name] = fields[0]
    return rows


def verify_flat(directory: Path, expected_outer: str) -> dict:
    require(stat.S_ISDIR(directory.lstat().st_mode) and not directory.is_symlink(), "direct sealed directory")
    manifest = directory / "SHA256SUMS"; outer = directory / "SHA256SUMS.seal.sha256"
    require(sha(outer) == expected_outer and
            outer.read_text(encoding="utf-8").split() == [sha(manifest), "SHA256SUMS"], "outer seal")
    expected = manifest_rows(manifest); actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}: continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "live sealed symlink")
        if stat.S_ISREG(mode): actual.add(name)
        else: require(stat.S_ISDIR(mode), "sealed special member")
    require(actual == set(expected), "exact sealed member set")
    for name, digest in expected.items():
        require(sha(directory / name) == digest, "sealed member identity")
    return {"entries": len(expected), "manifest_sha256": sha(manifest), "outer_seal_file_sha256": sha(outer)}


def module_names(text: str) -> list[str]:
    return re.findall(r"(?m)^\s*module\s+([A-Za-z_][A-Za-z0-9_$]*)", text)


def main() -> int:
    for path, digest in EXPECTED.items():
        require(path.exists() and not path.is_symlink() and sha(path) == digest, "frozen identity " + str(path))
    attempt_seal = verify_flat(ATTEMPT, EXPECTED[ATTEMPT / "SHA256SUMS.seal.sha256"])
    failure_seal = verify_flat(FAILURE, EXPECTED[FAILURE / "SHA256SUMS.seal.sha256"])
    attempt = load(ATTEMPT / "attempt.json"); failure = load(FAILURE / "failure.json")
    selector = load(FAILURE / "dc/dc_selector_runtime_identity.json")
    require(attempt["status"] == "M1122R4_ATTEMPT_CONSUMED_AFTER_M1123R4_M1125R4" and
            attempt["dc_attempts"] == 1, "attempt permanently consumed")
    require(failure == {"m1112r3_retry": False, "m1122r4_retry": False, "message": "fresh DC failed",
                        "phase": "FRESH_DC_SELECTOR_M1122R4", "status": "FAILED_DIAGNOSTIC_DO_NOT_CITE"},
            "failure exact no-retry receipt")
    require(selector["status"] == "PASS_M1122R4_EXACT_DC_SELECTOR_RUNTIME_CAPTURE" and
            selector["exe"].endswith("/common_shell_exec") and selector["argv"] == [
                "/opt/synopsys/syn/V-2023.12-SP3/linux64/syn/bin/common_shell_exec",
                "-shell", "dc_shell", "-r", "/opt/synopsys/syn/V-2023.12-SP3", "-f", str(TCL)],
            "selector runtime pass remains valid")

    r2 = R2_RTL.read_text(encoding="utf-8"); base = BASE_RTL.read_text(encoding="utf-8")
    r2tb = R2_TB.read_text(encoding="utf-8"); basetb = BASE_TB.read_text(encoding="utf-8")
    filelist = FILELIST.read_text(encoding="utf-8").splitlines()
    log = (FAILURE / "dc/dc.log").read_text(encoding="utf-8", errors="replace")
    tcl = TCL.read_text(encoding="utf-8")
    old = "m1112_c2_k1_async_observation_shadow_wrapper"
    requested = "m1112r2_c2_k1_async_observation_shadow_wrapper"
    old_tb = "tb_m1112_c2_k1_async_observation_shadow_case0_short"
    requested_tb = "tb_m1112r2_c2_k1_async_observation_shadow_case0_short"

    require(module_names(r2) == [], "r2 shim declares no module")
    require(module_names(base) == [old], "included base declares only old module")
    require(f"`define {old} {requested}" in r2 and f'`include "rtl_m1112/{old}.sv"' in r2 and
            f"`{old}" not in r2, "RTL macro is defined but never invoked")
    require(not any(token in r2 + base for token in ("`ifdef", "`ifndef", "`elsif")),
            "RTL has no SYNTHESIS/conditional guard")
    require(filelist[-1] == f"rtl_m1112r2/{requested}.sv" and
            f"rtl_m1112/{old}.sv" not in filelist, "filelist selects only include shim")
    require(f"Compiling source file {R2_RTL}" in log and
            f"Opening include file rtl_m1112/{old}.sv" in log and
            "Presto compilation completed successfully." in log, "analyze and include succeeded")
    require(f"Cannot find the design '{requested}' in the library 'WORK'. (LBR-0)" in log and
            "Error: Current design is not defined. (UID-4)" in log and
            "status=FAIL_ELABORATION_NO_CURRENT_DESIGN" ==
                (FAILURE / "dc/TCL_EXPLICIT_FAILURE.txt").read_text(encoding="utf-8").strip(),
            "elaboration failed on absent renamed top")
    require("analyze -format sverilog -define SYNTHESIS $rtl_files" in tcl and
            "elaborate $design_name" in tcl and tcl.index("analyze -format") < tcl.index("elaborate $design_name"),
            "analyze and elaborate are separate gates")

    require(module_names(r2tb) == [] and module_names(basetb) == [old_tb], "r2 TB has same alias defect")
    require(f"`define {old_tb} {requested_tb}" in r2tb and f"`define {old} {requested}" in r2tb and
            f"`{old_tb}" not in r2tb and f"`{old}" not in r2tb, "TB macros defined but never invoked")
    require(re.search(rf"(?m)^\s*{old}\s+dut\s*\(", basetb) is not None,
            "base TB instantiates old DUT token")

    # In-memory only: prove the lowest-risk additive copy requires one RTL and two TB token replacements.
    r5 = "m1122r5_c2_k1_async_observation_shadow_wrapper"
    r5tb = "tb_m1122r5_c2_k1_async_observation_shadow_case0_short"
    rtl_candidate, rtl_count = re.subn(rf"(?m)^(\s*module\s+){old}(\s*#\s*\()",
                                       rf"\1{r5}\2", base, count=1)
    tb_candidate, top_count = re.subn(rf"(?m)^(\s*module\s+){old_tb}(\s*;)",
                                      rf"\1{r5tb}\2", basetb, count=1)
    tb_candidate, dut_count = re.subn(rf"(?m)^(\s*){old}(\s+dut\s*\()",
                                      rf"\1{r5}\2", tb_candidate, count=1)
    require((rtl_count, top_count, dut_count) == (1, 1, 1) and module_names(rtl_candidate) == [r5] and
            module_names(tb_candidate) == [r5tb] and old not in rtl_candidate and
            re.search(rf"(?m)^\s*{r5}\s+dut\s*\(", tb_candidate) is not None,
            "mechanical-copy r5 repair model")

    m872_log = (M872 / "k1/dc.log").read_text(encoding="utf-8", errors="replace")
    m917_log = (M917 / "fixed/dc.log").read_text(encoding="utf-8", errors="replace")
    m872_top = "m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24"
    m917_top = "m518_matched_fixed_t10_atlif"
    require(module_names((HW / "rtl_m803/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24.sv").read_text()) == [m872_top],
            "M872 direct top declaration")
    require(module_names((HW / "rtl_m518/m518_matched_fixed_t10_atlif.sv").read_text()) == [m917_top],
            "M917 direct top declaration")
    require("Elaborated 1 design." in m872_log and f"Current design is now '{m872_top}_ARCH_MODE0'." in m872_log and
            "Cannot find the design" not in m872_log, "M872 direct top elaborates")
    require("Elaborated 1 design." in m917_log and f"Current design is now '{m917_top}'." in m917_log and
            "Cannot find the design" not in m917_log, "M917 direct top elaborates")
    m903 = load(M903 / "review.json"); m928 = load(M928 / "review.json")
    require(m903["status"] == "PASS100_M872_M803_C2_R16_THREE_AXIS_LOGIC_ONLY_DC_RESULT_ADMITTED" and
            m903["identity"]["canonical_seal"]["outer_seal_file_sha256"] ==
                EXPECTED[M872 / "SHA256SUMS.seal.sha256"], "M872 admission remains valid")
    require(m928["status"] == "PASS_M928_M917_M518_R5_FIXED_LOGIC_ONLY_DC_RESULT_ADMITTED" and
            m928["identity"]["canonical_seal"]["outer_seal_file_sha256"] ==
                EXPECTED[M917 / "SHA256SUMS.seal.sha256"], "M917 admission remains valid")

    evidence = {
        "schema": "m1128r5_m1122r4_c2_elaboration_failure_mechanical_checks_r1_v1",
        "status": "PASS_M1128R5_READONLY_ROOT_CAUSE__R4_PERMANENT_NO_RETRY__ADDITIVE_R5_ONLY",
        "checks_passed": checks,
        "identity": {
            "attempt": attempt_seal, "failure_quarantine": failure_seal,
            "selector_runtime_identity_sha256": sha(FAILURE / "dc/dc_selector_runtime_identity.json"),
            "dc_log_sha256": sha(FAILURE / "dc/dc.log"),
            "r2_rtl_shim_sha256": sha(R2_RTL), "base_rtl_sha256": sha(BASE_RTL),
            "r2_tb_shim_sha256": sha(R2_TB), "base_tb_sha256": sha(BASE_TB),
            "filelist_sha256": sha(FILELIST), "tcl_sha256": sha(TCL),
            "m872_outer_seal_file_sha256": sha(M872 / "SHA256SUMS.seal.sha256"),
            "m917_outer_seal_file_sha256": sha(M917 / "SHA256SUMS.seal.sha256"),
            "docs359_sha256": sha(DOCS359),
        },
        "root_cause": {
            "confidence": "HIGH",
            "include_missing": False, "synthesis_guard_removed_top": False,
            "dc_or_library_selector_failure": False,
            "bare_macro_name_never_expands": True,
            "actual_work_module": old, "requested_missing_work_module": requested,
            "analyze_success_only_means_sources_compiled": True,
            "elaborate_is_first_requested_top_existence_check": True,
            "same_latent_defect_in_mapped_vcs_tb": True,
        },
        "repair_comparison": {
            "mechanical_copy_rename": {
                "recommended": True, "rtl_token_changes": 1, "tb_token_changes": 2,
                "connectivity_change": False, "old_source_modified": False,
                "reason": "direct declared tops match the proven M872/M917 pattern and preserve the exact body"
            },
            "true_wrapper_instantiating_old_module": {
                "recommended_now": False, "old_source_modified": False,
                "risks": ["complete parameter/port binding", "new hierarchy and flatten behavior",
                          "structural reset census naming", "TB still requires a real renamed top/DUT binding"]
            }
        },
        "execution_boundary": {
            "r4_retry": False, "r4_namespace_reuse": False, "eda_executed_by_audit": False,
            "vcs_executed_by_audit": False, "reviewed_source_modified": False,
            "r5_requires_fresh_source_filelist_engine_contract_hammers_and_namespaces": True,
        },
    }
    OUTPUT.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": evidence["status"], "checks_passed": checks}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
