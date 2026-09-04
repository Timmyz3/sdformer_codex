#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""One-shot C2 successor for the sealed M1493 SOURCE_CHAIN failure.

M1493 failed before its attempt token because it called a method that does not
exist on the frozen M1432 module.  The following exact-pin loop already covers
that execution stack.  M1502 removes only the invalid call while preserving the
mapped campaign, ``-debug_access+r`` plus ``-lca``, and every EDA input.

This source is inert without fresh M1503/M1504/M1505 authorities.
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
OLD_RUNNER = HW / (
    "dc_handoff/scripts/run_m1493_m1467_c2_mapped_vcs_saif_ptpx_"
    "lca_successor_one_shot.py")
SPEC = importlib.util.spec_from_file_location("m1502_frozen_m1493", OLD_RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("M1502 cannot load frozen M1493")
OLD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OLD)
AUTH = OLD.BASE
EXEC = OLD.BASE.BASE

SOURCE_CHECKER = HW / (
    "verif_m1502_c2_source_chain_successor/"
    "check_m1502_c2_source_chain_successor_source.py")
SOURCE_TESTS = HW / (
    "verif_m1502_c2_source_chain_successor/"
    "test_m1502_c2_source_chain_successor_source.py")
SOURCE_CONTRACT = HW / (
    "contracts/m1502_m1493_c2_source_chain_successor_source_contract_"
    "r1_20260831.json")
M1494 = HW / (
    "reviews/m1494_m1493_c2_lca_successor_source_blind_hammer_"
    "r1_20260831")
M1495 = HW / (
    "contracts/m1495_m1494_m1493_c2_lca_successor_launch_release_"
    "r1_20260831.json")
M1496 = HW / (
    "reviews/m1496_m1495_m1493_c2_lca_successor_final_launch_hammer_"
    "r1_20260831")
M1503 = HW / (
    "reviews/m1503_m1502_c2_source_chain_successor_source_blind_hammer_"
    "r1_20260831")
M1504 = HW / (
    "contracts/m1504_m1503_m1502_c2_source_chain_successor_launch_release_"
    "r1_20260831.json")
M1505 = HW / (
    "reviews/m1505_m1504_m1502_c2_source_chain_successor_final_launch_hammer_"
    "r1_20260831")

OLD_FAILURE = HW / (
    "results/m1493_c2_mapped_vcs_saif_ptpx_r1_20260831."
    "failed_or_incomplete.quarantine")
ATTEMPT = HW / "results/.m1502_c2_mapped_vcs_saif_ptpx_attempt_consumed"
RESULT = HW / "results/m1502_c2_mapped_vcs_saif_ptpx_r1_20260831"
FAILURE = Path(str(RESULT) + ".failed_or_incomplete.quarantine")
PRIVATE = Path(str(RESULT) + ".private_build.unsealed_do_not_cite")
WORK = HW / f"results/.m1502_c2_mapped_vcs_saif_ptpx_work.{os.getpid()}"
STAGE = HW / f"results/.m1502_c2_mapped_vcs_saif_ptpx_result_stage.{os.getpid()}"
FAIL_STAGE = HW / f"results/.m1502_c2_mapped_vcs_saif_ptpx_failure_stage.{os.getpid()}"
LOCK = Path("/tmp/m1502_c2_mapped_vcs_saif_ptpx.lock")

OLD_RUNNER_SHA = "8d93d55ca600620eb903a7328f4cc38e0720ae45ce24d8128fac5924d2902677"
OLD_FAILURE_SHA = {
    "payload": "43497b8701400b6c7c5d3f0cc29a2a41955a135fff4be6720968cbeb736cc5e7",
    "manifest": "53e77670cd0f07ea457dc35f041e3885f7d73b304149c8d52e116fd06d6a5f88",
    "outer": "8cb2e41374f9b827c118b949e1a37b66baeec5bef578d81ee68a0d95a90d4a7e",
}
M1494_SHA = {
    "review": "65435aca804c486d50d8332774c70e87083d66d5c2e7acc30485dc84ba458340",
    "manifest": "b2ff59fd22bd0bd6463ae9ac9aa31ee82d77099d40ea4890fd99600255b9811b",
    "outer": "329ed4435761eb7d00be969d43ac05221c837cc3f79cedefd03d557034c432f7",
}
M1495_SHA = "838ea0f3714167c43c6f4e40829c2d1a59d1b84ee7468758798c82f21114eb94"
M1496_SHA = {
    "review": "ef0af9fbf0ab094f40052de8fc552b7b97e2519dd5db88c6f3c2bf7505acb810",
    "manifest": "72da922a5b652bf07eecc2ecc75ade847c7950c1c3a056299cca613bc1a19049",
    "outer": "2c8a99c7a9f0d2f56d6b77583f09cdc9ade265ba55b47c721e0ff44680d98e79",
}

ENV_PINS = (
    "M1502_EXPECTED_RUNNER_SHA256",
    "M1502_EXPECTED_SOURCE_CONTRACT_SHA256",
    "M1502_EXPECTED_M1503_REVIEW_SHA256",
    "M1502_EXPECTED_M1503_MANIFEST_SHA256",
    "M1502_EXPECTED_M1503_OUTER_FILE_SHA256",
    "M1502_EXPECTED_M1504_RELEASE_SHA256",
    "M1502_EXPECTED_M1505_REVIEW_SHA256",
    "M1502_EXPECTED_M1505_MANIFEST_SHA256",
    "M1502_EXPECTED_M1505_OUTER_FILE_SHA256",
)

CLAIMS = dict(OLD.CLAIMS)
COUNTS = dict(OLD.COUNTS)
COMPILE_PREFIX = list(OLD.COMPILE_PREFIX)


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
    if os.path.lexists(OLD.ATTEMPT):
        raise Failure("M1493 attempt unexpectedly exists")
    if os.path.lexists(OLD.RESULT) or os.path.lexists(OLD.PRIVATE):
        raise Failure("M1493 result/private unexpectedly exists")
    if AUTH.verify_seal(OLD_FAILURE, OLD_FAILURE_SHA["manifest"],
                        OLD_FAILURE_SHA["outer"]) != {"failure.json"}:
        raise Failure("M1493 failure population drift")
    exact(OLD_FAILURE / "failure.json", OLD_FAILURE_SHA["payload"])
    failure = strict_json(OLD_FAILURE / "failure.json")
    if failure != {
            "attempt_consumed": False,
            "automatic_retry": False,
            "canonical_result": False,
            "counts": {"ptpx_runs": 0, "saif_files": 0,
                       "simv_runs": 0, "vcs_compiles": 0},
            "error": "AttributeError", "identity": None,
            "partial_axis_citable": False, "phase": "SOURCE_CHAIN",
            "status": "FAILED_OR_INCOMPLETE"}:
        raise Failure("M1493 failure semantic drift")
    blind = AUTH.verify_authority(M1494, M1494_SHA["review"],
                                  M1494_SHA["manifest"], M1494_SHA["outer"])
    exact(M1495, M1495_SHA)
    AUTH.verify_sidecars(M1495)
    release = strict_json(M1495)
    final = AUTH.verify_authority(M1496, M1496_SHA["review"],
                                  M1496_SHA["manifest"], M1496_SHA["outer"])
    if (blind.get("status") !=
            "PASS_M1494_M1493_C2_LCA_SUCCESSOR_SOURCE_ZERO_FALSE_NEGATIVE"
            or release.get("status") !=
            "RELEASE_M1493_C2_LCA_SUCCESSOR__FRESH_M1496_REQUIRED__NO_LAUNCH"
            or final.get("status") !=
            "PASS_M1496_AUTHORIZE_ONE_M1493_C2_MAPPED_VCS_SAIF_PTPX_LAUNCH"
            or final.get("authorization") !=
            {"launch": True, "campaigns": 1, "automatic_retry": False}
            or final.get("bindings") != {
                "runner_sha256": OLD_RUNNER_SHA,
                "source_contract_sha256": sha(OLD.SOURCE_CONTRACT),
                "m1494_review_sha256": M1494_SHA["review"],
                "m1495_release_sha256": M1495_SHA}
            or final.get("claim_boundary") != CLAIMS):
        raise Failure("M1493 authority chain drift")


def verify_new_authority() -> None:
    pins = {name: os.environ.get(name, "") for name in ENV_PINS}
    if any(re.fullmatch(r"[0-9a-f]{64}", value) is None
           for value in pins.values()):
        raise Failure("M1502 authority absent: required exact SHA environment")
    exact(RUNNER, pins["M1502_EXPECTED_RUNNER_SHA256"])
    exact(SOURCE_CONTRACT, pins["M1502_EXPECTED_SOURCE_CONTRACT_SHA256"])
    AUTH.verify_sidecars(SOURCE_CONTRACT)
    blind = AUTH.verify_authority(
        M1503, pins["M1502_EXPECTED_M1503_REVIEW_SHA256"],
        pins["M1502_EXPECTED_M1503_MANIFEST_SHA256"],
        pins["M1502_EXPECTED_M1503_OUTER_FILE_SHA256"])
    exact(M1504, pins["M1502_EXPECTED_M1504_RELEASE_SHA256"])
    AUTH.verify_sidecars(M1504)
    release = strict_json(M1504)
    final = AUTH.verify_authority(
        M1505, pins["M1502_EXPECTED_M1505_REVIEW_SHA256"],
        pins["M1502_EXPECTED_M1505_MANIFEST_SHA256"],
        pins["M1502_EXPECTED_M1505_OUTER_FILE_SHA256"])
    if blind.get("status") != (
            "PASS_M1503_M1502_C2_SOURCE_CHAIN_SUCCESSOR_SOURCE_ZERO_FALSE_NEGATIVE"):
        raise Failure("M1503 status drift")
    if release.get("status") != (
            "RELEASE_M1502_C2_SOURCE_CHAIN_SUCCESSOR__FRESH_M1505_REQUIRED__NO_LAUNCH"):
        raise Failure("M1504 status drift")
    expected_bindings = {
        "runner_sha256": sha(RUNNER),
        "source_contract_sha256": sha(SOURCE_CONTRACT),
        "m1503_review_sha256": sha(M1503 / "review.json"),
        "m1504_release_sha256": sha(M1504),
    }
    if (final.get("status") !=
            "PASS_M1505_AUTHORIZE_ONE_M1502_C2_MAPPED_VCS_SAIF_PTPX_LAUNCH"
            or final.get("authorization") !=
            {"launch": True, "campaigns": 1, "automatic_retry": False}
            or final.get("bindings") != expected_bindings
            or final.get("claim_boundary") != CLAIMS):
        raise Failure("M1505 final authority drift")


def verify_frozen_execution_inputs() -> None:
    verify_predecessor_failure()
    verify_new_authority()
    # M1502 sole repair: the invalid predecessor-module method invocation is
    # absent.  The exact per-file and sealed-authority pins below cover the
    # original execution stack.
    for path, digest in (
            (EXEC.M1361_CHECKER, EXEC.STATIC_SHA["m1361_checker"]),
            (EXEC.M1361_TEST, EXEC.STATIC_SHA["m1361_test"]),
            (EXEC.M1361_CONTRACT, EXEC.STATIC_SHA["m1361_contract"]),
            (EXEC.SOURCE_CHECKER, EXEC.STATIC_SHA["source_checker"]),
            (EXEC.CELL_MODEL, EXEC.STATIC_SHA["cell_model"]),
            (EXEC.RESET_MEMORY_MODEL, EXEC.STATIC_SHA["reset_memory_model"]),
            (EXEC.CASE_TB, EXEC.STATIC_SHA["case_tb"]),
            (EXEC.ASSERTIONS, EXEC.STATIC_SHA["assertions"]),
            (EXEC.MAPPED_TB, EXEC.STATIC_SHA["mapped_tb"]),
            (EXEC.FILELIST["k8"], EXEC.STATIC_SHA["filelist_k8"]),
            (EXEC.FILELIST["k1x8"], EXEC.STATIC_SHA["filelist_k1x8"]),
            (EXEC.UCLI, EXEC.STATIC_SHA["ucli"]),
            (EXEC.PTPX_TCL, EXEC.STATIC_SHA["ptpx_tcl"]),
            (EXEC.VCS, EXEC.STATIC_SHA["vcs"]),
            (EXEC.PT, EXEC.STATIC_SHA["pt"]),
            (EXEC.LMUTIL, EXEC.STATIC_SHA["lmutil"]),
            (EXEC.PYTHON, EXEC.STATIC_SHA["python"]),
            (EXEC.LIB_DB, EXEC.STATIC_SHA["lib_db"]),
            (EXEC.DOCS359, EXEC.STATIC_SHA["docs359"])):
        EXEC.exact(path, digest)
    EXEC.verify_dir(EXEC.M1361_AUTHOR, EXEC.STATIC_SHA["m1361_review"],
                    EXEC.STATIC_SHA["m1361_manifest"],
                    EXEC.STATIC_SHA["m1361_outer"])
    EXEC.verify_dir(EXEC.M1362, EXEC.STATIC_SHA["m1362_review"],
                    EXEC.STATIC_SHA["m1362_manifest"],
                    EXEC.STATIC_SHA["m1362_outer"])
    EXEC.verify_dir(EXEC.M1440, OLD.BASE.OLD_M1440_SHA["review"],
                    OLD.BASE.OLD_M1440_SHA["manifest"],
                    OLD.BASE.OLD_M1440_SHA["outer"])
    for axis in ("k8", "k1x8"):
        netlist = EXEC.M872 / axis / "netlist" / f"{EXEC.DESIGN}_mapped.v"
        sdc = EXEC.M872 / axis / "netlist" / f"{EXEC.DESIGN}_mapped.sdc"
        EXEC.exact(netlist, EXEC.STATIC_SHA[axis + "_netlist"])
        EXEC.exact(sdc, EXEC.STATIC_SHA[axis + "_sdc"])


def identity() -> dict[str, str]:
    return {
        "m1493_runner_sha256": OLD_RUNNER_SHA,
        "m1493_failure_payload_sha256": OLD_FAILURE_SHA["payload"],
        "m1494_review_sha256": M1494_SHA["review"],
        "m1495_release_sha256": M1495_SHA,
        "m1496_review_sha256": M1496_SHA["review"],
        "m1502_runner_sha256": sha(RUNNER),
        "m1502_source_contract_sha256": sha(SOURCE_CONTRACT),
        "m1503_review_sha256": sha(M1503 / "review.json"),
        "m1504_release_sha256": sha(M1504),
        "m1505_final_review_sha256": sha(M1505 / "review.json"),
        "mapped_tb_sha256": EXEC.STATIC_SHA["mapped_tb"],
        "ucli_sha256": EXEC.STATIC_SHA["ucli"],
        "ptpx_tcl_sha256": EXEC.STATIC_SHA["ptpx_tcl"],
        "k8_netlist_sha256": EXEC.STATIC_SHA["k8_netlist"],
        "k1x8_netlist_sha256": EXEC.STATIC_SHA["k1x8_netlist"],
    }


def namespaces_fresh() -> None:
    for path in (ATTEMPT, RESULT, FAILURE, PRIVATE, WORK, STAGE, FAIL_STAGE):
        if os.path.lexists(path):
            raise Failure("namespace residue: " + str(path))
    for pattern in (".m1502_c2_mapped_vcs_saif_ptpx_work.*",
                    ".m1502_c2_mapped_vcs_saif_ptpx_result_stage.*",
                    ".m1502_c2_mapped_vcs_saif_ptpx_failure_stage.*"):
        if list((HW / "results").glob(pattern)):
            raise Failure("stale private namespace: " + pattern)


def publish_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint]
    if renameat2(-100, os.fsencode(source), -100,
                 os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    if len(sys.argv) != 1:
        raise Failure("M1502 accepts no arguments")
    state = {"phase": "SOURCE_CHAIN", "attempt": False, "complete": False,
             "vcs_compiles": 0, "simv_runs": 0,
             "saif_files": 0, "ptpx_runs": 0}
    lock_handle = LOCK.open("a+")
    try:
        verify_frozen_execution_inputs()
        state["identity"] = identity()
        namespaces_fresh()
        EXEC.collision_gate()
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        EXEC.collision_gate()
        EXEC.resource_gate()
        namespaces_fresh()
        state["phase"] = "LICENSE_PREFLIGHT"
        if not EXEC.LICENSE_FILE.is_file() or EXEC.LICENSE_FILE.is_symlink():
            raise Failure("license file invalid")
        license_log = subprocess.run(
            [str(EXEC.LMUTIL), "lmstat", "-a", "-c", EXEC.LICENSE_SERVER],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=60, check=False)
        if license_log.returncode != 0:
            raise Failure("license preflight failed")
        state["phase"] = "ATTEMPT_CONSUME"
        ATTEMPT.mkdir()
        state["attempt"] = True
        write_json(ATTEMPT / "attempt.json", {
            "status": "M1502_ATTEMPT_CONSUMED", "identity": identity(),
            "campaigns": 1, "automatic_retry": False, "budget": COUNTS,
            "sole_delta": "delete_invalid_source_chain_method_call"})
        EXEC.seal_dir(ATTEMPT)
        WORK.mkdir()
        (WORK / "build").mkdir()
        (WORK / "candidate").mkdir()
        for axis in ("k8", "k1x8"):
            state["phase"] = "COMPILE_" + axis
            axis_dir = WORK / "build" / axis
            axis_dir.mkdir()
            state["vcs_compiles"] += 1
            command = COMPILE_PREFIX + ["-f", str(EXEC.FILELIST[axis]),
                                        "-top", EXEC.TB_TOP, "-o", "simv"]
            EXEC.run(command, cwd=axis_dir,
                     env=EXEC.clean_env({
                         "VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
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
                EXEC.run(["./simv", "+M979_UCLI_SAIF", f"+M979_CASE={case}",
                          "-no_save", "-assert", f"report={report}", "-ucli",
                          "-i", str(EXEC.UCLI)], cwd=axis_dir,
                         env=EXEC.clean_env({
                             "VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
                             "VCS_ARCH_OVERRIDE": "linux",
                             "M1334_SAIF_FILE": str(saif)}), timeout=600,
                         output=log)
                display = "K8" if axis == "k8" else "K1x8"
                cycles = EXEC.CYCLES[axis][case]
                expected = (f"PASS M979 mapped replay axis={display} case={case} "
                            f"events={EXEC.EVENTS[case]} cycles={cycles} "
                            f"saif_duration_ns={cycles*3} numeric_mismatches=0 "
                            "tuple_mismatches=0 weight_mismatches=0 "
                            "accepted_unknowns=0 protocol_errors=0")
                if expected not in log.read_text(errors="replace"):
                    raise Failure("PASS mismatch")
                check = candidate / f"{axis}_case{case}.saif_check.json"
                EXEC.run([str(EXEC.PYTHON), "-I", str(EXEC.SOURCE_CHECKER),
                          "--saif", str(saif), "--axis", axis,
                          "--case", str(case), "--cycles", str(cycles)],
                         cwd=HW, env=EXEC.clean_env({}), timeout=120,
                         output=check)
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
                netlist = (EXEC.M872 / axis / "netlist" /
                           f"{EXEC.DESIGN}_mapped.v")
                sdc = (EXEC.M872 / axis / "netlist" /
                       f"{EXEC.DESIGN}_mapped.sdc")
                EXEC.run([str(EXEC.PT), "-f", str(EXEC.PTPX_TCL)], cwd=HW,
                         env=EXEC.clean_env({"DESIGN_NAME": EXEC.DESIGN,
                             "LIB_DB": str(EXEC.LIB_DB),
                             "MAPPED_NETLIST": str(netlist),
                             "MAPPED_SDC": str(sdc), "SAIF_FILE": str(saif),
                             "OUTPUT_DIR": str(pt_dir),
                             "OPERATING_CONDITION": "ssg0p9v125c",
                             "CORNER_ROLE": "slow_prelayout_power",
                             "SAIF_INSTANCE": EXEC.SAIF_INSTANCE}), timeout=1800,
                         output=pt_dir / "ptpx.log")
                required = ("ptpx_check_power.rpt", "ptpx_power.rpt",
                            "ptpx_power_hierarchy.rpt",
                            "ptpx_switching_summary.rpt")
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
        write_json(STAGE / "m1502_receipt.json", {
            "schema": "m1502_c2_mapped_vcs_saif_ptpx_candidate_receipt_r1_v1",
            "status": "PASS_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "identity": identity(), "one_shot": {"attempt_consumed": True,
            **COUNTS, "automatic_retry": False}, "axes": ["k8", "k1x8"],
            "cases_per_axis": 5,
            "sole_delta": "delete_invalid_source_chain_method_call",
            "claim_boundary": CLAIMS})
        (STAGE / "RUN_COMPLETE.txt").write_text(
            "PASS_M1502_C2_MAPPED_VCS_SAIF_PTPX_CANDIDATE_PENDING_RESULT_HAMMER\n")
        EXEC.seal_dir(STAGE)
        publish_no_replace(WORK, PRIVATE)
        publish_no_replace(STAGE, RESULT)
        state["complete"] = True
        print("PASS_M1502_C2_MAPPED_VCS_SAIF_PTPX_CANDIDATE_PENDING_RESULT_HAMMER")
        return 0
    except BaseException as error:
        if not state["complete"]:
            try:
                FAIL_STAGE.mkdir(exist_ok=False)
                write_json(FAIL_STAGE / "failure.json", {
                    "status": "FAILED_OR_INCOMPLETE", "phase": state["phase"],
                    "error": type(error).__name__,
                    "attempt_consumed": state["attempt"],
                    "identity": state.get("identity"),
                    "counts": {key: state[key] for key in COUNTS},
                    "automatic_retry": False, "canonical_result": False,
                    "partial_axis_citable": False})
                EXEC.seal_dir(FAIL_STAGE)
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
