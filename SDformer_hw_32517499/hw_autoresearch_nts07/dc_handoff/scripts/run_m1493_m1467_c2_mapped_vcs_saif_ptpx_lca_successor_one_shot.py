#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""One-shot C2 mapped SAIF/PTPX successor after the sealed M1467 failure.

M1467 crossed the previous ``-debug_access+r`` UCLI barrier, then VCS stopped
at 0 ps because SV-SAIF also requires ``-lca`` at compile time.  M1493 keeps
the frozen two-axis, five-case mapped campaign and adds only that option.

This file is inert without fresh M1494/M1495/M1496 authorities and exact
external SHA-256 pins.  Source checks never invoke EDA tools.
"""
from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from typing import Any


HW = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
OLD_RUNNER = HW / "dc_handoff/scripts/run_m1467_m1432_c2_mapped_vcs_saif_ptpx_debug_access_successor_one_shot.py"
SPEC = importlib.util.spec_from_file_location("m1493_frozen_m1467", OLD_RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("M1493 cannot load frozen M1467")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

SOURCE_CHECKER = HW / "verif_m1493_c2_lca_successor/check_m1493_c2_lca_successor_source.py"
SOURCE_TESTS = HW / "verif_m1493_c2_lca_successor/test_m1493_c2_lca_successor_source.py"
SOURCE_CONTRACT = HW / "contracts/m1493_m1467_c2_lca_successor_source_contract_r1_20260831.json"
M1484 = HW / "reviews/m1484_m1467_c2_second_production_failure_forensic_r1_20260831"
M1494 = HW / "reviews/m1494_m1493_c2_lca_successor_source_blind_hammer_r1_20260831"
M1495 = HW / "contracts/m1495_m1494_m1493_c2_lca_successor_launch_release_r1_20260831.json"
M1496 = HW / "reviews/m1496_m1495_m1493_c2_lca_successor_final_launch_hammer_r1_20260831"

OLD_ATTEMPT = HW / "results/.m1467_c2_mapped_vcs_saif_ptpx_attempt_consumed"
OLD_FAILURE = HW / "results/m1467_c2_mapped_vcs_saif_ptpx_r1_20260831.failed_or_incomplete.quarantine"
ATTEMPT = HW / "results/.m1493_c2_mapped_vcs_saif_ptpx_attempt_consumed"
RESULT = HW / "results/m1493_c2_mapped_vcs_saif_ptpx_r1_20260831"
FAILURE = Path(str(RESULT) + ".failed_or_incomplete.quarantine")
PRIVATE = Path(str(RESULT) + ".private_build.unsealed_do_not_cite")
WORK = HW / f"results/.m1493_c2_mapped_vcs_saif_ptpx_work.{os.getpid()}"
STAGE = HW / f"results/.m1493_c2_mapped_vcs_saif_ptpx_result_stage.{os.getpid()}"
FAIL_STAGE = HW / f"results/.m1493_c2_mapped_vcs_saif_ptpx_failure_stage.{os.getpid()}"
LOCK = Path("/tmp/m1493_c2_mapped_vcs_saif_ptpx.lock")

OLD_RUNNER_SHA = "120cb1a8abe3df1e537de6797b3962fe0a7496be78954ba3b31fd9c8627e9a8a"
OLD_ATTEMPT_SHA = {
    "payload": "a3eead113c10d0134dd83972aaa06c6b26256f7459a37d784f98c5eeb2c68f92",
    "manifest": "830d359dc80f2690913fb9b9f9a05b02073fd99e88639844412ac5f25138f526",
    "outer": "eba291930799326b00d5460ce66f32fe29fef0a8b9a379bd05a28794a0cd13dc",
}
OLD_FAILURE_SHA = {
    "payload": "39f3d5ffa39508db348cddf116584267e68e8796a008a7949bad88e02dd2c015",
    "manifest": "233067e03f011cb1c3b4bd9fb4160d4fa7225246fc2eab9159933cf3e8792dcd",
    "outer": "5503f1cc7db87e2cb1417f72167a5f11b6cc9fe86972c847b96f617357f80e82",
}
M1484_SHA = {
    "review": "d26f73469d3d9e131cb776d47c6ee12c2ddd9f546e47fae690f73d7f8186d826",
    "manifest": "d61787c9a4c25e8cfe6fe2b0980605b09cae9ffaf1d4c8406b28d93cd43618b3",
    "outer": "86c26e7109931199578e22cba7795aeea2673ea5e57f2524ed76790ce9d1487d",
}

ENV_PINS = (
    "M1493_EXPECTED_RUNNER_SHA256",
    "M1493_EXPECTED_SOURCE_CONTRACT_SHA256",
    "M1493_EXPECTED_M1494_REVIEW_SHA256",
    "M1493_EXPECTED_M1494_MANIFEST_SHA256",
    "M1493_EXPECTED_M1494_OUTER_FILE_SHA256",
    "M1493_EXPECTED_M1495_RELEASE_SHA256",
    "M1493_EXPECTED_M1496_REVIEW_SHA256",
    "M1493_EXPECTED_M1496_MANIFEST_SHA256",
    "M1493_EXPECTED_M1496_OUTER_FILE_SHA256",
)

CLAIMS = dict(BASE.CLAIMS)
COUNTS = {"vcs_compiles": 2, "simv_runs": 10,
          "saif_files": 10, "ptpx_runs": 10}
COMPILE_PREFIX = [str(BASE.BASE.VCS), "-full64", "-sverilog", "+v2k",
                  "-timescale=1ns/1ps", "-assert", "svaext",
                  "-debug_access+r", "-lca", "+vcs+lic+wait", "-Mdir=csrc"]


class Failure(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(path: Path, digest: str) -> None:
    if (not path.is_file() or path.is_symlink()
            or not stat.S_ISREG(path.lstat().st_mode) or sha(path) != digest):
        raise Failure("identity drift: " + str(path))


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        output = {}
        for key, value in items:
            if key in output:
                raise Failure("duplicate JSON key: " + key)
            output[key] = value
        return output
    if not path.is_file() or path.is_symlink():
        raise Failure("JSON absent/nonregular: " + str(path))
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           Failure("nonfinite JSON: " + token)))
    if type(value) is not dict:
        raise Failure("JSON root is not object")
    return value


def verify_predecessor_failure() -> None:
    exact(OLD_RUNNER, OLD_RUNNER_SHA)
    old_text = OLD_RUNNER.read_text()
    compile_region = old_text[old_text.index("COMPILE_PREFIX ="):
                              old_text.index("\n\n\nclass Failure")]
    if compile_region.count('"-debug_access+r"') != 1 or '"-lca"' in compile_region:
        raise Failure("M1467 compile-option premise drift")
    if BASE.verify_seal(OLD_ATTEMPT, OLD_ATTEMPT_SHA["manifest"],
                        OLD_ATTEMPT_SHA["outer"]) != {"attempt.json"}:
        raise Failure("M1467 attempt population drift")
    exact(OLD_ATTEMPT / "attempt.json", OLD_ATTEMPT_SHA["payload"])
    attempt = strict_json(OLD_ATTEMPT / "attempt.json")
    if (attempt.get("status") != "M1467_ATTEMPT_CONSUMED"
            or attempt.get("automatic_retry") is not False
            or attempt.get("sole_delta") != "vcs_compile_add_debug_access_r"):
        raise Failure("M1467 attempt semantic drift")
    if BASE.verify_seal(OLD_FAILURE, OLD_FAILURE_SHA["manifest"],
                        OLD_FAILURE_SHA["outer"]) != {"failure.json"}:
        raise Failure("M1467 failure population drift")
    exact(OLD_FAILURE / "failure.json", OLD_FAILURE_SHA["payload"])
    failure = strict_json(OLD_FAILURE / "failure.json")
    expected = {"vcs_compiles": 1, "simv_runs": 1,
                "saif_files": 0, "ptpx_runs": 0}
    if (failure.get("status") != "FAILED_OR_INCOMPLETE"
            or failure.get("phase") != "SIM_k8_0"
            or failure.get("counts") != expected
            or failure.get("attempt_consumed") is not True
            or failure.get("automatic_retry") is not False
            or failure.get("partial_axis_citable") is not False):
        raise Failure("M1467 failure semantic drift")
    if os.path.lexists(BASE.RESULT):
        raise Failure("M1467 canonical result unexpectedly exists")
    forensic = BASE.verify_authority(M1484, M1484_SHA["review"],
                                     M1484_SHA["manifest"], M1484_SHA["outer"])
    if (forensic.get("status") !=
            "FAIL_M1467_SECOND_PRODUCTION_ATTEMPT__DO_NOT_CITE__SUCCESSOR_SOURCE_OR_NO_GO_ONLY"
            or forensic.get("failure", {}).get("first_error_code") !=
            "Error-[LCA_FEATURES_NEED_OPTION]"
            or forensic.get("authorization", {}).get(
                "additive_successor_source_authoring_allowed") is not True
            or forensic.get("authorization", {}).get(
                "successor_execution_allowed_by_m1484") is not False):
        raise Failure("M1484 forensic semantic drift")


def verify_new_authority() -> None:
    pins = {name: os.environ.get(name, "") for name in ENV_PINS}
    if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in pins.values()):
        raise Failure("required exact SHA environment absent")
    exact(RUNNER, pins["M1493_EXPECTED_RUNNER_SHA256"])
    exact(SOURCE_CONTRACT, pins["M1493_EXPECTED_SOURCE_CONTRACT_SHA256"])
    BASE.verify_sidecars(SOURCE_CONTRACT)
    blind = BASE.verify_authority(
        M1494, pins["M1493_EXPECTED_M1494_REVIEW_SHA256"],
        pins["M1493_EXPECTED_M1494_MANIFEST_SHA256"],
        pins["M1493_EXPECTED_M1494_OUTER_FILE_SHA256"])
    exact(M1495, pins["M1493_EXPECTED_M1495_RELEASE_SHA256"])
    BASE.verify_sidecars(M1495)
    release = strict_json(M1495)
    final = BASE.verify_authority(
        M1496, pins["M1493_EXPECTED_M1496_REVIEW_SHA256"],
        pins["M1493_EXPECTED_M1496_MANIFEST_SHA256"],
        pins["M1493_EXPECTED_M1496_OUTER_FILE_SHA256"])
    if blind.get("status") != "PASS_M1494_M1493_C2_LCA_SUCCESSOR_SOURCE_ZERO_FALSE_NEGATIVE":
        raise Failure("M1494 status drift")
    if release.get("status") != "RELEASE_M1493_C2_LCA_SUCCESSOR__FRESH_M1496_REQUIRED__NO_LAUNCH":
        raise Failure("M1495 release status drift")
    expected_bindings = {
        "runner_sha256": sha(RUNNER),
        "source_contract_sha256": sha(SOURCE_CONTRACT),
        "m1494_review_sha256": sha(M1494 / "review.json"),
        "m1495_release_sha256": sha(M1495),
    }
    if (final.get("status") !=
            "PASS_M1496_AUTHORIZE_ONE_M1493_C2_MAPPED_VCS_SAIF_PTPX_LAUNCH"
            or final.get("authorization") !=
            {"launch": True, "campaigns": 1, "automatic_retry": False}
            or final.get("bindings") != expected_bindings
            or final.get("claim_boundary") != CLAIMS):
        raise Failure("M1496 final authority drift")


def verify_frozen_execution_inputs() -> None:
    verify_predecessor_failure()
    verify_new_authority()
    # Re-pin the original M1361/M1362/M1440 execution stack without requiring
    # the consumed M1467 launch environment a second time.
    BASE.BASE.verify_predecessor_failure()
    for path, digest in (
            (BASE.BASE.M1361_CHECKER, BASE.BASE.STATIC_SHA["m1361_checker"]),
            (BASE.BASE.M1361_TEST, BASE.BASE.STATIC_SHA["m1361_test"]),
            (BASE.BASE.M1361_CONTRACT, BASE.BASE.STATIC_SHA["m1361_contract"]),
            (BASE.BASE.SOURCE_CHECKER, BASE.BASE.STATIC_SHA["source_checker"]),
            (BASE.BASE.CELL_MODEL, BASE.BASE.STATIC_SHA["cell_model"]),
            (BASE.BASE.RESET_MEMORY_MODEL, BASE.BASE.STATIC_SHA["reset_memory_model"]),
            (BASE.BASE.CASE_TB, BASE.BASE.STATIC_SHA["case_tb"]),
            (BASE.BASE.ASSERTIONS, BASE.BASE.STATIC_SHA["assertions"]),
            (BASE.BASE.MAPPED_TB, BASE.BASE.STATIC_SHA["mapped_tb"]),
            (BASE.BASE.FILELIST["k8"], BASE.BASE.STATIC_SHA["filelist_k8"]),
            (BASE.BASE.FILELIST["k1x8"], BASE.BASE.STATIC_SHA["filelist_k1x8"]),
            (BASE.BASE.UCLI, BASE.BASE.STATIC_SHA["ucli"]),
            (BASE.BASE.PTPX_TCL, BASE.BASE.STATIC_SHA["ptpx_tcl"]),
            (BASE.BASE.VCS, BASE.BASE.STATIC_SHA["vcs"]),
            (BASE.BASE.PT, BASE.BASE.STATIC_SHA["pt"]),
            (BASE.BASE.LMUTIL, BASE.BASE.STATIC_SHA["lmutil"]),
            (BASE.BASE.PYTHON, BASE.BASE.STATIC_SHA["python"]),
            (BASE.BASE.LIB_DB, BASE.BASE.STATIC_SHA["lib_db"]),
            (BASE.BASE.DOCS359, BASE.BASE.STATIC_SHA["docs359"])):
        BASE.BASE.exact(path, digest)
    BASE.BASE.verify_dir(BASE.BASE.M1361_AUTHOR,
                         BASE.BASE.STATIC_SHA["m1361_review"],
                         BASE.BASE.STATIC_SHA["m1361_manifest"],
                         BASE.BASE.STATIC_SHA["m1361_outer"])
    BASE.BASE.verify_dir(BASE.BASE.M1362, BASE.BASE.STATIC_SHA["m1362_review"],
                         BASE.BASE.STATIC_SHA["m1362_manifest"],
                         BASE.BASE.STATIC_SHA["m1362_outer"])
    BASE.BASE.verify_dir(BASE.BASE.M1440, BASE.OLD_M1440_SHA["review"],
                         BASE.OLD_M1440_SHA["manifest"], BASE.OLD_M1440_SHA["outer"])
    for axis in ("k8", "k1x8"):
        netlist = BASE.BASE.M872 / axis / "netlist" / f"{BASE.BASE.DESIGN}_mapped.v"
        sdc = BASE.BASE.M872 / axis / "netlist" / f"{BASE.BASE.DESIGN}_mapped.sdc"
        BASE.BASE.exact(netlist, BASE.BASE.STATIC_SHA[axis + "_netlist"])
        BASE.BASE.exact(sdc, BASE.BASE.STATIC_SHA[axis + "_sdc"])


def identity() -> dict[str, str]:
    return {
        "m1467_runner_sha256": OLD_RUNNER_SHA,
        "m1467_attempt_payload_sha256": OLD_ATTEMPT_SHA["payload"],
        "m1467_failure_payload_sha256": OLD_FAILURE_SHA["payload"],
        "m1484_review_sha256": M1484_SHA["review"],
        "m1493_runner_sha256": sha(RUNNER),
        "m1493_source_contract_sha256": sha(SOURCE_CONTRACT),
        "m1494_review_sha256": sha(M1494 / "review.json"),
        "m1495_release_sha256": sha(M1495),
        "m1496_final_review_sha256": sha(M1496 / "review.json"),
        "mapped_tb_sha256": BASE.BASE.STATIC_SHA["mapped_tb"],
        "ucli_sha256": BASE.BASE.STATIC_SHA["ucli"],
        "ptpx_tcl_sha256": BASE.BASE.STATIC_SHA["ptpx_tcl"],
        "k8_netlist_sha256": BASE.BASE.STATIC_SHA["k8_netlist"],
        "k1x8_netlist_sha256": BASE.BASE.STATIC_SHA["k1x8_netlist"],
    }


def namespaces_fresh() -> None:
    for path in (ATTEMPT, RESULT, FAILURE, PRIVATE, WORK, STAGE, FAIL_STAGE):
        if os.path.lexists(path):
            raise Failure("namespace residue: " + str(path))
    for pattern in (".m1493_c2_mapped_vcs_saif_ptpx_work.*",
                    ".m1493_c2_mapped_vcs_saif_ptpx_result_stage.*",
                    ".m1493_c2_mapped_vcs_saif_ptpx_failure_stage.*"):
        if list((HW / "results").glob(pattern)):
            raise Failure("stale private namespace: " + pattern)


def publish_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint]
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    if len(sys.argv) != 1:
        raise Failure("M1493 accepts no arguments")
    state = {"phase": "SOURCE_CHAIN", "attempt": False, "complete": False,
             "vcs_compiles": 0, "simv_runs": 0, "saif_files": 0, "ptpx_runs": 0}
    lock_handle = LOCK.open("a+")
    try:
        verify_frozen_execution_inputs()
        state["identity"] = identity()
        namespaces_fresh()
        BASE.BASE.collision_gate()
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        BASE.BASE.collision_gate()
        BASE.BASE.resource_gate()
        namespaces_fresh()
        state["phase"] = "LICENSE_PREFLIGHT"
        if not BASE.BASE.LICENSE_FILE.is_file() or BASE.BASE.LICENSE_FILE.is_symlink():
            raise Failure("license file invalid")
        license_log = subprocess.run(
            [str(BASE.BASE.LMUTIL), "lmstat", "-a", "-c", BASE.BASE.LICENSE_SERVER],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=60, check=False)
        if license_log.returncode != 0:
            raise Failure("license preflight failed")
        state["phase"] = "ATTEMPT_CONSUME"
        ATTEMPT.mkdir()
        state["attempt"] = True
        write_json(ATTEMPT / "attempt.json", {
            "status": "M1493_ATTEMPT_CONSUMED", "identity": identity(),
            "campaigns": 1, "automatic_retry": False, "budget": COUNTS,
            "sole_delta": "vcs_compile_add_lca_after_debug_access_r"})
        BASE.BASE.seal_dir(ATTEMPT)
        WORK.mkdir()
        (WORK / "build").mkdir()
        (WORK / "candidate").mkdir()
        for axis in ("k8", "k1x8"):
            state["phase"] = "COMPILE_" + axis
            axis_dir = WORK / "build" / axis
            axis_dir.mkdir()
            state["vcs_compiles"] += 1
            command = COMPILE_PREFIX + ["-f", str(BASE.BASE.FILELIST[axis]),
                                        "-top", BASE.BASE.TB_TOP, "-o", "simv"]
            BASE.BASE.run(command, cwd=axis_dir,
                     env=BASE.BASE.clean_env({"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
                                         "VCS_ARCH_OVERRIDE": "linux"}), timeout=1800,
                     output=axis_dir / "compile.log")
            if not (axis_dir / "simv").is_file():
                raise Failure("simv absent: " + axis)
            for case in range(5):
                state["phase"] = f"SIM_{axis}_{case}"
                state["simv_runs"] += 1
                candidate = WORK / "candidate"
                saif = candidate / f"{axis}_case{case}.saif"
                log = candidate / f"{axis}_case{case}.log"
                report = candidate / f"{axis}_case{case}.assert.report"
                BASE.BASE.run(["./simv", "+M979_UCLI_SAIF", f"+M979_CASE={case}",
                          "-no_save", "-assert", f"report={report}", "-ucli",
                          "-i", str(BASE.BASE.UCLI)], cwd=axis_dir,
                         env=BASE.BASE.clean_env({
                             "VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
                             "VCS_ARCH_OVERRIDE": "linux",
                             "M1334_SAIF_FILE": str(saif)}), timeout=600, output=log)
                display = "K8" if axis == "k8" else "K1x8"
                cycles = BASE.BASE.CYCLES[axis][case]
                expected = (f"PASS M979 mapped replay axis={display} case={case} "
                            f"events={BASE.BASE.EVENTS[case]} cycles={cycles} "
                            f"saif_duration_ns={cycles*3} numeric_mismatches=0 "
                            "tuple_mismatches=0 weight_mismatches=0 accepted_unknowns=0 "
                            "protocol_errors=0")
                if expected not in log.read_text(errors="replace"):
                    raise Failure("PASS mismatch")
                check = candidate / f"{axis}_case{case}.saif_check.json"
                BASE.BASE.run([str(BASE.BASE.PYTHON), "-I", str(BASE.BASE.SOURCE_CHECKER),
                          "--saif", str(saif), "--axis", axis, "--case", str(case),
                          "--cycles", str(cycles)], cwd=HW, env=BASE.BASE.clean_env({}),
                         timeout=120, output=check)
                state["saif_files"] += 1
        if any(state[key] != COUNTS[key] for key in
               ("vcs_compiles", "simv_runs", "saif_files")):
            raise Failure("mapped VCS/SAIF campaign incomplete before PTPX")
        for axis in ("k8", "k1x8"):
            for case in range(5):
                candidate = WORK / "candidate"
                saif = candidate / f"{axis}_case{case}.saif"
                state["phase"] = f"PTPX_{axis}_{case}"
                state["ptpx_runs"] += 1
                pt_dir = candidate / f"{axis}_case{case}.ptpx"
                pt_dir.mkdir()
                netlist = BASE.BASE.M872 / axis / "netlist" / f"{BASE.BASE.DESIGN}_mapped.v"
                sdc = BASE.BASE.M872 / axis / "netlist" / f"{BASE.BASE.DESIGN}_mapped.sdc"
                BASE.BASE.run([str(BASE.BASE.PT), "-f", str(BASE.BASE.PTPX_TCL)], cwd=HW,
                         env=BASE.BASE.clean_env({"DESIGN_NAME": BASE.BASE.DESIGN,
                             "LIB_DB": str(BASE.BASE.LIB_DB), "MAPPED_NETLIST": str(netlist),
                             "MAPPED_SDC": str(sdc), "SAIF_FILE": str(saif),
                             "OUTPUT_DIR": str(pt_dir),
                             "OPERATING_CONDITION": "ssg0p9v125c",
                             "CORNER_ROLE": "slow_prelayout_power",
                             "SAIF_INSTANCE": BASE.BASE.SAIF_INSTANCE}), timeout=1800,
                         output=pt_dir / "ptpx.log")
                required = ("ptpx_check_power.rpt", "ptpx_power.rpt",
                            "ptpx_power_hierarchy.rpt", "ptpx_switching_summary.rpt")
                if any(not (pt_dir / "reports" / name).is_file()
                           or (pt_dir / "reports" / name).stat().st_size == 0
                           for name in required):
                    raise Failure("PTPX report absent")
        if any(state[key] != value for key, value in COUNTS.items()):
            raise Failure("execution count drift")
        state["phase"] = "SUCCESS_STAGE"
        STAGE.mkdir()
        shutil.copytree(WORK / "candidate", STAGE / "candidate")
        for axis in ("k8", "k1x8"):
            shutil.copy2(WORK / "build" / axis / "compile.log",
                         STAGE / f"{axis}.compile.log")
        write_json(STAGE / "m1493_receipt.json", {
            "schema": "m1493_c2_mapped_vcs_saif_ptpx_candidate_receipt_r1_v1",
            "status": "PASS_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "identity": identity(), "one_shot": {"attempt_consumed": True,
            **COUNTS, "automatic_retry": False}, "axes": ["k8", "k1x8"],
            "cases_per_axis": 5,
            "sole_delta": "vcs_compile_add_lca_after_debug_access_r",
            "claim_boundary": CLAIMS})
        (STAGE / "RUN_COMPLETE.txt").write_text(
            "PASS_M1493_C2_MAPPED_VCS_SAIF_PTPX_CANDIDATE_PENDING_RESULT_HAMMER\n")
        BASE.BASE.seal_dir(STAGE)
        publish_no_replace(WORK, PRIVATE)
        publish_no_replace(STAGE, RESULT)
        state["complete"] = True
        print("PASS_M1493_C2_MAPPED_VCS_SAIF_PTPX_CANDIDATE_PENDING_RESULT_HAMMER")
        return 0
    except BaseException as error:
        if not state["complete"]:
            try:
                FAIL_STAGE.mkdir(exist_ok=False)
                write_json(FAIL_STAGE / "failure.json", {
                    "status": "FAILED_OR_INCOMPLETE", "phase": state["phase"],
                    "error": type(error).__name__, "attempt_consumed": state["attempt"],
                    "identity": state.get("identity"),
                    "counts": {key: state[key] for key in COUNTS},
                    "automatic_retry": False, "canonical_result": False,
                    "partial_axis_citable": False})
                BASE.BASE.seal_dir(FAIL_STAGE)
                publish_no_replace(FAIL_STAGE, FAILURE)
            except BaseException:
                pass
            if WORK.is_dir() and not PRIVATE.exists():
                try:
                    publish_no_replace(WORK, PRIVATE)
                except BaseException:
                    pass
        raise
    finally:
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
