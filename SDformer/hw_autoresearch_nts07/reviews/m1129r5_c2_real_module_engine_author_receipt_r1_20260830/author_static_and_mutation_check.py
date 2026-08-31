#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1129r5 real-module source-author check; static/mutation only, no EDA."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import stat
from pathlib import Path
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
ENGINE = HW / "dc_handoff/scripts/m1129r5_c2_real_module_async_observation_engine_source_r1.py"
CONTRACT = HW / "contracts/m1129r5_c2_dc_selector_async_observation_source_contract_r1_20260830.json"
RTL = HW / "rtl_m1129r5/m1129r5_c2_k1_async_observation_shadow_wrapper.sv"
BASE_RTL = HW / "rtl_m1112/m1112_c2_k1_async_observation_shadow_wrapper.sv"
TB = HW / "dc_handoff/tb/tb_m1129r5_c2_k1_async_observation_shadow_case0_short.sv"
BASE_TB = HW / "dc_handoff/tb/tb_m1112_c2_k1_async_observation_shadow_case0_short.sv"
FILELIST = HW / "dc_handoff/filelists/date_m1129r5_c2_k1_async_observation_shadow_logic_only_dc.f"
M1128 = HW / "reviews/m1128r5_m1122r4_c2_elaboration_failure_readonly_audit_r1_20260830"
R4_ATTEMPT = HW / "results/.m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_attempt_consumed"
R4_FAILURE = HW / "results/m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_r1_20260830.failed_or_incomplete.580027.quarantine"
R4_RESULT = HW / "results/m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_r1_20260830"
R5_ATTEMPT = HW / "results/.m1129r5_c2_dc_selector_async_observation_dc_mapped_vcs_attempt_consumed"
R5_RESULT = HW / "results/m1129r5_c2_dc_selector_async_observation_dc_mapped_vcs_r1_20260830"
R5_LOCK = Path("/tmp/m1129r5_c2_dc_selector_async_observation_eda.lock")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "engine": "c8fd3366ecf6c4377b62e5717d959348c08192ea8bdbd0afd3b0e566bd6fbd0b",
    "contract": "25cfbf9e2d75333e27a1162ab202b9b6a9b305876ee92ce6ed9f6d30513f370d",
    "contract_side": "d7b31831edf5ced6c9df04b12aa08ee8078e10da051bfab8be24bba9ab630a6a",
    "contract_outer": "b5a389b2b76a83f6449bfcbc928c416df877f611cfbd987d828552cb4bdf50cf",
    "rtl": "86df0f7be383e6ba8ee17c1e27fc25fd18eb6fecc01329c41a976cd836004dd0",
    "base_rtl": "95c31bc70a7617c6653eaca2f77a54388119f744b814dfc909c75edad1c39218",
    "tb": "c08d22d69c222b8c527bdb70cc5b49392c5467bc3142ebc22ec577da6918147b",
    "base_tb": "ff6bd371c3b1371c520b38680960ad0297a8c01eb92eb7b4a0f4d2e59fc861b6",
    "filelist": "1ac2715245cce259f3dcba37cbeecac0e9a2ab9b16a60463f6a53f668ff9e106",
    "m1128_outer": "9435b3e94b0053b296eccc95058b3799a1002e018d4b15b0c89058e8b68e8730",
    "r4_attempt_outer": "8a012c8638c2e8a8da743cbf570a13f5c8bc8d85716b433882d03405e12e5e37",
    "r4_failure_outer": "2f9173b1e988b1f639e6c3d683fdf720fa9debfeaca8caf27bf5845a36527f83",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

checks: list[str] = []
attacks: dict[str, str] = {}


def require(value: bool, label: str) -> None:
    if not value:
        raise RuntimeError(label)
    checks.append(label)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise RuntimeError("duplicate key " + key)
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite " + token)))


def regular(path: Path, expected: str, label: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(), label + " regular")
    require(sha(path) == expected, label + " SHA")


def verify_double(path: Path) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    regular(path, EXPECTED["contract"], "contract primary")
    regular(side, EXPECTED["contract_side"], "contract side")
    regular(outer, EXPECTED["contract_outer"], "contract outer")
    require(side.read_text(encoding="utf-8").split() ==
            [EXPECTED["contract"], path.relative_to(HW).as_posix()],
            "contract side content")
    require(outer.read_text(encoding="utf-8").split() ==
            [EXPECTED["contract_side"], side.relative_to(HW).as_posix()],
            "contract outer content")


def verify_flat(directory: Path, expected_outer: str) -> dict:
    require(stat.S_ISDIR(directory.lstat().st_mode) and not directory.is_symlink(),
            directory.name + " direct directory")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    regular(outer, expected_outer, directory.name + " outer")
    require(outer.read_text(encoding="utf-8").split() ==
            [sha(manifest), "SHA256SUMS"], directory.name + " outer content")
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]) is not None,
                directory.name + " manifest row")
        name = fields[1].lstrip("*"); rel = Path(name)
        require(name not in expected and name == rel.as_posix() and not rel.is_absolute()
                and ".." not in rel.parts, directory.name + " safe member")
        expected[name] = fields[0]
    actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), directory.name + " no symlink")
        if stat.S_ISREG(mode):
            actual.add(name)
        else:
            require(stat.S_ISDIR(mode), directory.name + " no special")
    require(actual == set(expected), directory.name + " exact members")
    for name, digest in expected.items():
        regular(directory / name, digest, directory.name + "/" + name)
    payloads = [name for name in ("review.json", "attempt.json", "failure.json")
                if (directory / name).exists()]
    require(len(payloads) == 1, directory.name + " one primary JSON")
    return strict_json(directory / payloads[0])


def rename_gate(rtl: str, tb: str, filelist: str) -> dict:
    design = "m1129r5_c2_k1_async_observation_shadow_wrapper"
    old_design = "m1112_c2_k1_async_observation_shadow_wrapper"
    tb_top = "tb_m1129r5_c2_k1_async_observation_shadow_case0_short"
    old_tb_top = "tb_m1112_c2_k1_async_observation_shadow_case0_short"
    base_rtl = BASE_RTL.read_text(encoding="utf-8")
    base_tb = BASE_TB.read_text(encoding="utf-8")
    rtl_modules = re.findall(r"(?m)^\s*module\s+(\w+)\b", rtl)
    tb_modules = re.findall(r"(?m)^\s*module\s+(\w+)\b", tb)
    require(rtl_modules == [design], "one real RTL declaration")
    require(tb_modules == [tb_top], "one real TB top")
    require(len(re.findall(r"(?m)^\s*"+re.escape(design)+r"\s+dut\s*\(", tb)) == 1,
            "one real TB DUT type")
    require(rtl.count(design) == 1 and
            rtl.replace(design, old_design, 1) == base_rtl, "RTL name-only diff")
    require(tb.count(tb_top) == 1 and tb.count(design) == 1 and
            tb.replace(tb_top, old_tb_top, 1).replace(design, old_design, 1) == base_tb,
            "TB top/DUT-only diff")
    lines = [line.strip() for line in filelist.splitlines() if line.strip()]
    require(lines[-1] == "rtl_m1129r5/m1129r5_c2_k1_async_observation_shadow_wrapper.sv"
            and lines.count(lines[-1]) == 1 and all("m1112r2" not in line for line in lines)
            and (chr(96)+"define") not in filelist and (chr(96)+"include") not in filelist,
            "filelist direct r5 RTL")
    return {"rtl_modules": len(rtl_modules), "tb_tops": len(tb_modules),
            "tb_dut_types": 1, "allowed_identifier_changes": 3}


def engine_gate(source: str, contract: dict) -> None:
    ast.parse(source)
    require("M1128R5_OUTER_SHA256 = \"9435b3e94b0053b296eccc95058b3799a1002e018d4b15b0c89058e8b68e8730\"" in source,
            "engine binds M1128r5 outer")
    require("OLD_M1122R4_ATTEMPT_OUTER_SHA256 = \"8a012c8638c2e8a8da743cbf570a13f5c8bc8d85716b433882d03405e12e5e37\"" in source and
            "OLD_M1122R4_FAILURE_OUTER_SHA256 = \"2f9173b1e988b1f639e6c3d683fdf720fa9debfeaca8caf27bf5845a36527f83\"" in source,
            "engine binds stopped r4 attempt/quarantine")
    require("def lexical_real_name_and_diff_gate()" in source and
            source.count("lexical_real_name_and_diff_gate()") == 2,
            "engine lexical gate defined and consumed")
    require('DC_SHELL = DC_INSTALL_ROOT / "bin/dc_shell"' in source and
            'DC_ACTUAL = DC_INSTALL_ROOT / "linux64/syn/bin/common_shell_exec"' in source and
            '[str(DC_SHELL), "-f", str(DC_TCL)]' in source and
            '"same_pid_exec_capture_required": true' in
                json.dumps(contract["dc_selector_contract"], sort_keys=True),
            "selector runtime capture inherited")
    require("SHADOW_REGISTER_BITS = 337" in source and
            "structural_reset_gate(netlist)" in source and
            'if sys.argv[1:] != ["--authorized-launch"]' in source,
            "337-bit reset provenance and fixed argv inherited")
    require("ATTEMPT.mkdir(); attempted = True" in source and
            '"m1129r5_retry": False' in source and
            contract["frozen_stopped_namespaces"]["m1129r5_maximum_attempts_after_all_hammers"] == 1 and
            contract["frozen_stopped_namespaces"]["automatic_retry"] is False,
            "one-shot no-retry inherited")
    require(contract["m1128r5_failure_authority"]["outer_seal_file_sha256"] ==
            EXPECTED["m1128_outer"] and
            contract["m1122r4_stopped_authority"]["retry_allowed"] is False,
            "contract exact repair authority")
    require(contract["claim_boundary"] == {
        "source_only": True, "mutation_selftest_only": True, "eda_executed": False,
        "attempt_consumed": False, "mapped_functionality": False,
        "paper_citable": False, "activity_or_power": False, "performance": False,
        "system_speedup": False, "paper_ppa_ready": False},
        "all claims false")


def rejected(label: str, action: Callable[[], Any]) -> None:
    before = len(checks)
    try:
        action()
    except Exception as error:
        attacks[label] = type(error).__name__ + ": " + str(error)
        del checks[before:]
        return
    raise RuntimeError("mutation accepted: " + label)


def main() -> None:
    regular(ENGINE, EXPECTED["engine"], "engine")
    verify_double(CONTRACT)
    regular(RTL, EXPECTED["rtl"], "r5 RTL")
    regular(BASE_RTL, EXPECTED["base_rtl"], "base RTL")
    regular(TB, EXPECTED["tb"], "r5 TB")
    regular(BASE_TB, EXPECTED["base_tb"], "base TB")
    regular(FILELIST, EXPECTED["filelist"], "r5 filelist")
    regular(DOCS359, EXPECTED["docs359"], "docs359")
    m1128 = verify_flat(M1128, EXPECTED["m1128_outer"])
    r4_attempt = verify_flat(R4_ATTEMPT, EXPECTED["r4_attempt_outer"])
    r4_failure = verify_flat(R4_FAILURE, EXPECTED["r4_failure_outer"])
    require(m1128["status"] ==
            "PASS_M1128R5_M1122R4_FAILURE_AUDIT__R4_PERMANENT_NO_RETRY__ADDITIVE_R5_ONLY",
            "M1128r5 GO exact")
    require(r4_attempt["dc_attempts"] == 1, "r4 exactly one DC attempt")
    require(r4_failure["status"] == "FAILED_DIAGNOSTIC_DO_NOT_CITE" and
            r4_failure["m1122r4_retry"] is False and not R4_RESULT.exists(),
            "r4 permanently quarantined")
    source = ENGINE.read_text(encoding="utf-8")
    contract = strict_json(CONTRACT)
    identity = rename_gate(RTL.read_text(encoding="utf-8"),
                           TB.read_text(encoding="utf-8"),
                           FILELIST.read_text(encoding="utf-8"))
    engine_gate(source, contract)
    require(not R5_ATTEMPT.exists() and not R5_ATTEMPT.is_symlink() and
            not R5_RESULT.exists() and not R5_RESULT.is_symlink() and
            not R5_LOCK.exists() and not R5_LOCK.is_symlink() and
            not any((HW / "results").glob(".m1129r5_c2_dc_selector_async_observation_dc_mapped_vcs_work.*")) and
            not any((HW / "results").glob("m1129r5_c2_dc_selector_async_observation_dc_mapped_vcs_r1_20260830.failed_or_incomplete.*")),
            "fresh r5 production namespace absent")

    rtl = RTL.read_text(encoding="utf-8")
    tb = TB.read_text(encoding="utf-8")
    filelist = FILELIST.read_text(encoding="utf-8")
    rejected("rtl_body_change", lambda: rename_gate(rtl.replace("obs_fault=", "obs_fault=1'b0|", 1), tb, filelist))
    rejected("rtl_old_module", lambda: rename_gate(rtl.replace("m1129r5_", "m1112_", 1), tb, filelist))
    rejected("tb_extra_change", lambda: rename_gate(rtl, tb.replace("wait_cycles=0;", "wait_cycles=1;", 1), filelist))
    rejected("tb_old_dut", lambda: rename_gate(rtl, tb.replace("m1129r5_c2_", "m1112_c2_", 1), filelist))
    rejected("filelist_r2_shim", lambda: rename_gate(rtl, tb, filelist.replace(
        "rtl_m1129r5/m1129r5_c2_k1_async_observation_shadow_wrapper.sv",
        "rtl_m1112r2/m1112r2_c2_k1_async_observation_shadow_wrapper.sv")))
    rejected("filelist_define", lambda: rename_gate(rtl, tb, filelist + chr(96)+"define BAD 1\n"))
    bad_contract = json.loads(json.dumps(contract)); bad_contract["claim_boundary"]["performance"] = True
    rejected("performance_claim", lambda: engine_gate(source, bad_contract))
    bad_contract = json.loads(json.dumps(contract)); bad_contract["m1122r4_stopped_authority"]["retry_allowed"] = True
    rejected("r4_retry", lambda: engine_gate(source, bad_contract))
    rejected("m1128_outer_drift", lambda: engine_gate(source.replace(
        EXPECTED["m1128_outer"], "0"*64), contract))
    rejected("selector_direct_backend", lambda: engine_gate(source.replace(
        '[str(DC_SHELL), "-f", str(DC_TCL)]',
        '[str(DC_ACTUAL), "-f", str(DC_TCL)]'), contract))
    rejected("shadow_census_336", lambda: engine_gate(source.replace(
        "SHADOW_REGISTER_BITS = 337", "SHADOW_REGISTER_BITS = 336"), contract))
    rejected("automatic_retry", lambda: engine_gate(source.replace(
        '"m1129r5_retry": False', '"m1129r5_retry": True'), contract))

    result = {
        "status": "PASS_M1129R5_REAL_MODULE_ENGINE_AUTHOR_SELFTEST__M1130R5_REQUIRED__NO_EDA",
        "checks_passed": len(checks),
        "attacks_rejected": len(attacks),
        "lexical_identity": identity,
        "attack_results": attacks,
        "engine_main_executed": False,
        "eda_executed": False,
        "attempt_created": False,
        "launch_authorized": False,
    }
    (HERE / "mechanical_checks.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
