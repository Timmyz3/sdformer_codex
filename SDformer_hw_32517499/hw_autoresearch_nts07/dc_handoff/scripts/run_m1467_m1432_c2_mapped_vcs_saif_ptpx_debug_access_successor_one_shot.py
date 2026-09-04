#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Additive one-shot C2 mapped SAIF/PTPX successor to consumed M1432.

M1432 failed in its first mapped simulation because its VCS compile omitted
``-debug_access+r`` while the frozen UCLI script requests gate-level power
activity.  M1467 preserves the exact workload, netlists, testbench, SAIF and
PTPX flow.  Its sole execution delta is the missing VCS observability flag.

This source is inert without fresh M1468/M1469/M1472 authorities and exact
external SHA-256 pins.  Authoring and source tests never invoke EDA tools.
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
OLD_RUNNER = HW / "dc_handoff/scripts/run_m1432_m1361_m1362_c2_mapped_vcs_saif_ptpx_one_shot.py"
SPEC = importlib.util.spec_from_file_location("m1467_frozen_m1432", OLD_RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("M1467 cannot load frozen M1432")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

SOURCE_CHECKER = HW / "verif_m1467_c2_debug_access_successor/check_m1467_c2_debug_access_successor_source.py"
SOURCE_TESTS = HW / "verif_m1467_c2_debug_access_successor/test_m1467_c2_debug_access_successor_source.py"
SOURCE_CONTRACT = HW / "contracts/m1467_m1432_c2_debug_access_successor_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1467_m1432_c2_debug_access_successor_source_author_r1_20260831"
M1468 = HW / "reviews/m1468_m1467_c2_debug_access_successor_source_blind_hammer_r1_20260831"
M1469 = HW / "contracts/m1469_m1468_m1467_c2_debug_access_successor_launch_release_r1_20260831.json"
M1472 = HW / "reviews/m1472_m1469_m1467_c2_debug_access_successor_final_launch_hammer_r1_20260831"

OLD_ATTEMPT = HW / "results/.m1432_c2_mapped_vcs_saif_ptpx_attempt_consumed"
OLD_FAILURE = HW / "results/m1432_c2_mapped_vcs_saif_ptpx_r1_20260831.failed_or_incomplete.quarantine"
ATTEMPT = HW / "results/.m1467_c2_mapped_vcs_saif_ptpx_attempt_consumed"
RESULT = HW / "results/m1467_c2_mapped_vcs_saif_ptpx_r1_20260831"
FAILURE = Path(str(RESULT) + ".failed_or_incomplete.quarantine")
PRIVATE = Path(str(RESULT) + ".private_build.unsealed_do_not_cite")
WORK = HW / f"results/.m1467_c2_mapped_vcs_saif_ptpx_work.{os.getpid()}"
STAGE = HW / f"results/.m1467_c2_mapped_vcs_saif_ptpx_result_stage.{os.getpid()}"
FAIL_STAGE = HW / f"results/.m1467_c2_mapped_vcs_saif_ptpx_failure_stage.{os.getpid()}"
LOCK = Path("/tmp/m1467_c2_mapped_vcs_saif_ptpx.lock")

OLD_RUNNER_SHA = "314be83304d4b62cf2c4b73feb394fa2ab20e60a89afb9c3dfc07622d25a7156"
OLD_ATTEMPT_SHA = {
    "payload": "3552c04045e19446fd9521e2a6145d6cf0c2090286f3cd5aa180a3074076f82f",
    "manifest": "9a50caa634e99c943677158babe9765b74ccab89b27e425d22a570ef5a9941f6",
    "outer": "ee66123a569c45de3aa0573a1db09af833428af300da1fe842f9e5c1b5be50f9",
}
OLD_FAILURE_SHA = {
    "payload": "4d21019bd0145b84646fad055de9b52fa66574144276027fa61598bd4e7607c5",
    "manifest": "2a2835af25d3947e6e445a8a268d3c254c986d8530267289fdc951fe917e7e97",
    "outer": "12ef0ad6c390ac343c68dc9f6936a8e4a1609427387d12dc4b63e412c5d401ec",
}
OLD_M1440_SHA = {
    "review": "a668f14217af0fec8ed0bfd300914ac06897b495cef1e08adf999d44841ff315",
    "manifest": "9d5311d013ae39c971f6e2a785bfe6006c93607d6a50048e60809b36f87a110b",
    "outer": "df917eeeba250df466f3cd2dd646bce5f7418050808a87b29c29a1a0a571233d",
}

ENV_PINS = (
    "M1467_EXPECTED_RUNNER_SHA256",
    "M1467_EXPECTED_SOURCE_CONTRACT_SHA256",
    "M1467_EXPECTED_AUTHOR_REVIEW_SHA256",
    "M1467_EXPECTED_AUTHOR_MANIFEST_SHA256",
    "M1467_EXPECTED_AUTHOR_OUTER_FILE_SHA256",
    "M1467_EXPECTED_M1468_REVIEW_SHA256",
    "M1467_EXPECTED_M1468_MANIFEST_SHA256",
    "M1467_EXPECTED_M1468_OUTER_FILE_SHA256",
    "M1467_EXPECTED_M1469_RELEASE_SHA256",
    "M1467_EXPECTED_M1472_REVIEW_SHA256",
    "M1467_EXPECTED_M1472_MANIFEST_SHA256",
    "M1467_EXPECTED_M1472_OUTER_FILE_SHA256",
)

CLAIMS = dict(BASE.CLAIMS)
COUNTS = {"vcs_compiles": 2, "simv_runs": 10,
          "saif_files": 10, "ptpx_runs": 10}
COMPILE_PREFIX = [str(BASE.VCS), "-full64", "-sverilog", "+v2k",
                  "-timescale=1ns/1ps", "-assert", "svaext",
                  "-debug_access+r", "+vcs+lic+wait", "-Mdir=csrc"]


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
    exact_type = path.is_file() and not path.is_symlink()
    if not exact_type:
        raise Failure("JSON absent/nonregular: " + str(path))
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           Failure("nonfinite JSON: " + token)))
    if type(value) is not dict:
        raise Failure("JSON root is not object")
    return value


def verify_seal(root: Path, manifest_sha: str, outer_sha: str) -> set[str]:
    if not root.is_dir() or root.is_symlink():
        raise Failure("sealed directory invalid: " + str(root))
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    exact(manifest, manifest_sha)
    exact(outer, outer_sha)
    if outer.read_text().split() != [manifest_sha, "SHA256SUMS"]:
        raise Failure("outer seal content drift")
    listed = set()
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        if len(fields) != 2:
            raise Failure("manifest row")
        digest, name = fields
        name = name.lstrip("*")
        rel = Path(name)
        if (not re.fullmatch(r"[0-9a-f]{64}", digest) or name in listed
                or rel.is_absolute() or ".." in rel.parts):
            raise Failure("manifest row")
        exact(root / rel, digest)
        listed.add(name)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name not in
              {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    if listed != actual:
        raise Failure("sealed population drift")
    return listed


def verify_authority(root: Path, review_sha: str, manifest_sha: str,
                     outer_sha: str) -> dict[str, Any]:
    verify_seal(root, manifest_sha, outer_sha)
    exact(root / "review.json", review_sha)
    return strict_json(root / "review.json")


def verify_sidecars(path: Path) -> None:
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    if sidecar.read_text().split() != [sha(path), path.name]:
        raise Failure("sidecar drift")
    if outer.read_text().split() != [sha(sidecar), sidecar.name]:
        raise Failure("outer sidecar drift")


def verify_predecessor_failure() -> None:
    exact(OLD_RUNNER, OLD_RUNNER_SHA)
    old_text = OLD_RUNNER.read_text()
    if '"-debug_access+r"' in old_text:
        raise Failure("M1432 root-cause premise drift")
    if "-gate_level all mda sv" not in BASE.UCLI.read_text():
        raise Failure("frozen UCLI power premise drift")
    if verify_seal(OLD_ATTEMPT, OLD_ATTEMPT_SHA["manifest"],
                   OLD_ATTEMPT_SHA["outer"]) != {"attempt.json"}:
        raise Failure("M1432 attempt population drift")
    exact(OLD_ATTEMPT / "attempt.json", OLD_ATTEMPT_SHA["payload"])
    attempt = strict_json(OLD_ATTEMPT / "attempt.json")
    if (attempt.get("status") != "M1432_ATTEMPT_CONSUMED"
            or attempt.get("automatic_retry") is not False):
        raise Failure("M1432 attempt semantic drift")
    if verify_seal(OLD_FAILURE, OLD_FAILURE_SHA["manifest"],
                   OLD_FAILURE_SHA["outer"]) != {"failure.json"}:
        raise Failure("M1432 failure population drift")
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
        raise Failure("M1432 failure semantic drift")
    if os.path.lexists(BASE.RESULT):
        raise Failure("M1432 canonical result unexpectedly exists")


def identity() -> dict[str, str]:
    return {
        "m1432_runner_sha256": OLD_RUNNER_SHA,
        "m1432_attempt_payload_sha256": OLD_ATTEMPT_SHA["payload"],
        "m1432_failure_payload_sha256": OLD_FAILURE_SHA["payload"],
        "m1467_runner_sha256": sha(RUNNER),
        "m1467_source_contract_sha256": sha(SOURCE_CONTRACT),
        "m1467_author_review_sha256": sha(AUTHOR / "review.json"),
        "m1468_review_sha256": sha(M1468 / "review.json"),
        "m1469_release_sha256": sha(M1469),
        "m1472_final_review_sha256": sha(M1472 / "review.json"),
        "mapped_tb_sha256": BASE.STATIC_SHA["mapped_tb"],
        "ucli_sha256": BASE.STATIC_SHA["ucli"],
        "ptpx_tcl_sha256": BASE.STATIC_SHA["ptpx_tcl"],
        "k8_netlist_sha256": BASE.STATIC_SHA["k8_netlist"],
        "k1x8_netlist_sha256": BASE.STATIC_SHA["k1x8_netlist"],
    }


def verify_new_authority() -> None:
    pins = {name: os.environ.get(name, "") for name in ENV_PINS}
    if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in pins.values()):
        raise Failure("required exact SHA environment absent")
    exact(RUNNER, pins["M1467_EXPECTED_RUNNER_SHA256"])
    exact(SOURCE_CONTRACT, pins["M1467_EXPECTED_SOURCE_CONTRACT_SHA256"])
    verify_sidecars(SOURCE_CONTRACT)
    author = verify_authority(
        AUTHOR, pins["M1467_EXPECTED_AUTHOR_REVIEW_SHA256"],
        pins["M1467_EXPECTED_AUTHOR_MANIFEST_SHA256"],
        pins["M1467_EXPECTED_AUTHOR_OUTER_FILE_SHA256"])
    blind = verify_authority(
        M1468, pins["M1467_EXPECTED_M1468_REVIEW_SHA256"],
        pins["M1467_EXPECTED_M1468_MANIFEST_SHA256"],
        pins["M1467_EXPECTED_M1468_OUTER_FILE_SHA256"])
    exact(M1469, pins["M1467_EXPECTED_M1469_RELEASE_SHA256"])
    verify_sidecars(M1469)
    release = strict_json(M1469)
    final = verify_authority(
        M1472, pins["M1467_EXPECTED_M1472_REVIEW_SHA256"],
        pins["M1467_EXPECTED_M1472_MANIFEST_SHA256"],
        pins["M1467_EXPECTED_M1472_OUTER_FILE_SHA256"])
    if author.get("status") != "PASS_M1467_C2_DEBUG_ACCESS_SUCCESSOR_SOURCE_AUTHOR__NO_EDA":
        raise Failure("M1467 author status drift")
    if blind.get("status") != "PASS_M1468_M1467_C2_DEBUG_ACCESS_SUCCESSOR_SOURCE_ZERO_FALSE_NEGATIVE":
        raise Failure("M1468 status drift")
    if release.get("status") != "RELEASE_M1467_C2_DEBUG_ACCESS_SUCCESSOR__FRESH_M1472_REQUIRED__NO_LAUNCH":
        raise Failure("M1469 release status drift")
    expected_bindings = {
        "runner_sha256": sha(RUNNER),
        "source_contract_sha256": sha(SOURCE_CONTRACT),
        "m1468_review_sha256": sha(M1468 / "review.json"),
        "m1469_release_sha256": sha(M1469),
    }
    if (final.get("status") !=
            "PASS_M1472_AUTHORIZE_ONE_M1467_C2_MAPPED_VCS_SAIF_PTPX_LAUNCH"
            or final.get("authorization") !=
            {"launch": True, "campaigns": 1, "automatic_retry": False}
            or final.get("bindings") != expected_bindings
            or final.get("claim_boundary") != CLAIMS):
        raise Failure("M1472 final authority drift")


def verify_frozen_execution_inputs() -> None:
    verify_predecessor_failure()
    verify_new_authority()
    # Exact-pin the original M1361/M1362/M1440 source and final authority chain.
    BASE.exact(BASE.M1361_CHECKER, BASE.STATIC_SHA["m1361_checker"])
    BASE.exact(BASE.M1361_TEST, BASE.STATIC_SHA["m1361_test"])
    BASE.exact(BASE.M1361_CONTRACT, BASE.STATIC_SHA["m1361_contract"])
    BASE.verify_dir(BASE.M1361_AUTHOR, BASE.STATIC_SHA["m1361_review"],
                    BASE.STATIC_SHA["m1361_manifest"], BASE.STATIC_SHA["m1361_outer"])
    BASE.verify_dir(BASE.M1362, BASE.STATIC_SHA["m1362_review"],
                    BASE.STATIC_SHA["m1362_manifest"], BASE.STATIC_SHA["m1362_outer"])
    BASE.verify_dir(BASE.M1440, OLD_M1440_SHA["review"], OLD_M1440_SHA["manifest"],
                    OLD_M1440_SHA["outer"])
    for path, digest in ((BASE.SOURCE_CHECKER, BASE.STATIC_SHA["source_checker"]),
                         (BASE.CELL_MODEL, BASE.STATIC_SHA["cell_model"]),
                         (BASE.RESET_MEMORY_MODEL, BASE.STATIC_SHA["reset_memory_model"]),
                         (BASE.CASE_TB, BASE.STATIC_SHA["case_tb"]),
                         (BASE.ASSERTIONS, BASE.STATIC_SHA["assertions"]),
                         (BASE.MAPPED_TB, BASE.STATIC_SHA["mapped_tb"]),
                         (BASE.FILELIST["k8"], BASE.STATIC_SHA["filelist_k8"]),
                         (BASE.FILELIST["k1x8"], BASE.STATIC_SHA["filelist_k1x8"]),
                         (BASE.UCLI, BASE.STATIC_SHA["ucli"]),
                         (BASE.PTPX_TCL, BASE.STATIC_SHA["ptpx_tcl"]),
                         (BASE.VCS, BASE.STATIC_SHA["vcs"]),
                         (BASE.PT, BASE.STATIC_SHA["pt"]),
                         (BASE.LMUTIL, BASE.STATIC_SHA["lmutil"]),
                         (BASE.PYTHON, BASE.STATIC_SHA["python"]),
                         (BASE.LIB_DB, BASE.STATIC_SHA["lib_db"]),
                         (BASE.DOCS359, BASE.STATIC_SHA["docs359"])):
        BASE.exact(path, digest)
    for axis in ("k8", "k1x8"):
        netlist = BASE.M872 / axis / "netlist" / f"{BASE.DESIGN}_mapped.v"
        sdc = BASE.M872 / axis / "netlist" / f"{BASE.DESIGN}_mapped.sdc"
        BASE.exact(netlist, BASE.STATIC_SHA[axis + "_netlist"])
        BASE.exact(sdc, BASE.STATIC_SHA[axis + "_sdc"])


def namespaces_fresh() -> None:
    for path in (ATTEMPT, RESULT, FAILURE, PRIVATE, WORK, STAGE, FAIL_STAGE):
        if os.path.lexists(path):
            raise Failure("namespace residue: " + str(path))
    for pattern in (".m1467_c2_mapped_vcs_saif_ptpx_work.*",
                    ".m1467_c2_mapped_vcs_saif_ptpx_result_stage.*",
                    ".m1467_c2_mapped_vcs_saif_ptpx_failure_stage.*"):
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
        raise Failure("M1467 accepts no arguments")
    state = {"phase": "SOURCE_CHAIN", "attempt": False, "complete": False,
             "vcs_compiles": 0, "simv_runs": 0, "saif_files": 0, "ptpx_runs": 0}
    lock_handle = LOCK.open("a+")
    try:
        verify_frozen_execution_inputs()
        state["identity"] = identity()
        namespaces_fresh()
        BASE.collision_gate()
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        BASE.collision_gate()
        BASE.resource_gate()
        namespaces_fresh()
        state["phase"] = "LICENSE_PREFLIGHT"
        if not BASE.LICENSE_FILE.is_file() or BASE.LICENSE_FILE.is_symlink():
            raise Failure("license file invalid")
        license_log = subprocess.run(
            [str(BASE.LMUTIL), "lmstat", "-a", "-c", BASE.LICENSE_SERVER],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=60, check=False)
        if license_log.returncode != 0:
            raise Failure("license preflight failed")
        state["phase"] = "ATTEMPT_CONSUME"
        ATTEMPT.mkdir()
        state["attempt"] = True
        write_json(ATTEMPT / "attempt.json", {
            "status": "M1467_ATTEMPT_CONSUMED", "identity": identity(),
            "campaigns": 1, "automatic_retry": False, "budget": COUNTS,
            "sole_delta": "vcs_compile_add_debug_access_r"})
        BASE.seal_dir(ATTEMPT)
        WORK.mkdir()
        (WORK / "build").mkdir()
        (WORK / "candidate").mkdir()
        for axis in ("k8", "k1x8"):
            state["phase"] = "COMPILE_" + axis
            axis_dir = WORK / "build" / axis
            axis_dir.mkdir()
            state["vcs_compiles"] += 1
            command = COMPILE_PREFIX + ["-f", str(BASE.FILELIST[axis]),
                                        "-top", BASE.TB_TOP, "-o", "simv"]
            BASE.run(command, cwd=axis_dir,
                     env=BASE.clean_env({"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
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
                BASE.run(["./simv", "+M979_UCLI_SAIF", f"+M979_CASE={case}",
                          "-no_save", "-assert", f"report={report}", "-ucli",
                          "-i", str(BASE.UCLI)], cwd=axis_dir,
                         env=BASE.clean_env({
                             "VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
                             "VCS_ARCH_OVERRIDE": "linux",
                             "M1334_SAIF_FILE": str(saif)}), timeout=600, output=log)
                display = "K8" if axis == "k8" else "K1x8"
                cycles = BASE.CYCLES[axis][case]
                expected = (f"PASS M979 mapped replay axis={display} case={case} "
                            f"events={BASE.EVENTS[case]} cycles={cycles} "
                            f"saif_duration_ns={cycles*3} numeric_mismatches=0 "
                            "tuple_mismatches=0 weight_mismatches=0 accepted_unknowns=0 "
                            "protocol_errors=0")
                if expected not in log.read_text(errors="replace"):
                    raise Failure("PASS mismatch")
                check = candidate / f"{axis}_case{case}.saif_check.json"
                BASE.run([str(BASE.PYTHON), "-I", str(BASE.SOURCE_CHECKER),
                          "--saif", str(saif), "--axis", axis, "--case", str(case),
                          "--cycles", str(cycles)], cwd=HW, env=BASE.clean_env({}),
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
                netlist = BASE.M872 / axis / "netlist" / f"{BASE.DESIGN}_mapped.v"
                sdc = BASE.M872 / axis / "netlist" / f"{BASE.DESIGN}_mapped.sdc"
                BASE.run([str(BASE.PT), "-f", str(BASE.PTPX_TCL)], cwd=HW,
                         env=BASE.clean_env({"DESIGN_NAME": BASE.DESIGN,
                             "LIB_DB": str(BASE.LIB_DB), "MAPPED_NETLIST": str(netlist),
                             "MAPPED_SDC": str(sdc), "SAIF_FILE": str(saif),
                             "OUTPUT_DIR": str(pt_dir),
                             "OPERATING_CONDITION": "ssg0p9v125c",
                             "CORNER_ROLE": "slow_prelayout_power",
                             "SAIF_INSTANCE": BASE.SAIF_INSTANCE}), timeout=1800,
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
        write_json(STAGE / "m1467_receipt.json", {
            "schema": "m1467_c2_mapped_vcs_saif_ptpx_candidate_receipt_r1_v1",
            "status": "PASS_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "identity": identity(), "one_shot": {"attempt_consumed": True,
            **COUNTS, "automatic_retry": False}, "axes": ["k8", "k1x8"],
            "cases_per_axis": 5, "sole_delta": "vcs_compile_add_debug_access_r",
            "claim_boundary": CLAIMS})
        (STAGE / "RUN_COMPLETE.txt").write_text(
            "PASS_M1467_C2_MAPPED_VCS_SAIF_PTPX_CANDIDATE_PENDING_RESULT_HAMMER\n")
        BASE.seal_dir(STAGE)
        publish_no_replace(WORK, PRIVATE)
        publish_no_replace(STAGE, RESULT)
        state["complete"] = True
        print("PASS_M1467_C2_MAPPED_VCS_SAIF_PTPX_CANDIDATE_PENDING_RESULT_HAMMER")
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
                BASE.seal_dir(FAIL_STAGE)
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
