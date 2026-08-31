#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""One-shot C1 functional successor with the corrected active-next oracle.

The M1270/R13 testbench, M1162 RTL, M1168/R3 SVA, and M1337/R15 witness
remain frozen.  M1497 compiles one additive testbench whose only semantic
delta is the active-next oracle.  VCS build products stay in RAW_BUILD; only
four regular evidence files enter CLEAN_RESULT_STAGE before recursive and
outer sealing.  This source is inert until a fresh M1498/M1499/M1500
different-author authority chain pins it exactly.
"""
from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
from datetime import datetime, timezone

if len(sys.argv) != 1:
    raise SystemExit("M1497: no arguments accepted")

SCRIPT_DIR = Path(__file__).resolve().parent
HW = SCRIPT_DIR.parents[1]
RUNNER = Path(__file__).resolve()
PREDECESSOR = SCRIPT_DIR / "run_vcs_m1459_m1433_c1_runtime_split_generic_seal_successor.py"
SPEC = importlib.util.spec_from_file_location("m1497_frozen_m1459", PREDECESSOR)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("M1497: cannot load frozen M1459 runner")
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)

TB_R13 = HW / "verif_m1270r13_c1_real_m935_protocol/tb_m1270r13_m1162_real_m935_protocol_unit_delay_r13.sv"
TB = HW / "verif_m1497_c1_active_next_oracle_successor/tb_m1497_m1270r13_m1162_real_m935_protocol_unit_delay.sv"
FILELIST = HW / "verif_m1497_c1_active_next_oracle_successor/m1497_unit_delay_filelist.f"
CHECKER = HW / "verif_m1497_c1_active_next_oracle_successor/check_m1497_source.py"
TESTS = HW / "verif_m1497_c1_active_next_oracle_successor/test_m1497_source.py"
CONTRACT = HW / "contracts/m1497_c1_active_next_oracle_clean_result_successor_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1497_c1_active_next_oracle_clean_result_successor_source_author_r1_20260831"
HAMMER = HW / "reviews/m1498_m1497_c1_active_next_oracle_source_blind_hammer_r1_20260831"
RELEASE = HW / "contracts/m1499_m1498_m1497_c1_active_next_oracle_vcs_launch_release_r1_20260831.json"
FINAL = HW / "reviews/m1500_m1499_m1497_c1_active_next_oracle_final_launch_hammer_r1_20260831"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

ATTEMPT = HW / "results/.m1497_c1_active_next_oracle_vcs_attempt_consumed"
RESULT = HW / "results/m1497_c1_active_next_oracle_unit_delay_vcs_r1_20260831"
QUARANTINE = Path(str(RESULT) + ".failed_or_incomplete.quarantine")
RAW_BUILD = HW / f"results/.m1497_c1_active_next_oracle_raw_build.{os.getpid()}"
CLEAN_RESULT_STAGE = HW / f"results/.m1497_c1_active_next_oracle_clean_result_stage.{os.getpid()}"
ATTEMPT_STAGE = HW / f"results/.m1497_c1_active_next_oracle_attempt_stage.{os.getpid()}"
FAILURE_STAGE = HW / f"results/.m1497_c1_active_next_oracle_failure_stage.{os.getpid()}"

SOURCE_STATUS = "M1497_C1_ACTIVE_NEXT_ORACLE_CLEAN_RESULT_SOURCE_READY__NO_LAUNCH"
AUTHOR_STATUS = "PASS_M1497_C1_ACTIVE_NEXT_ORACLE_CLEAN_RESULT_SOURCE__NO_VCS_NO_EDA"
HAMMER_STATUS = "PASS_M1498_M1497_C1_ACTIVE_NEXT_ORACLE_SOURCE__RELEASE_NOT_AUTHORED"
RELEASE_STATUS = "AUTHORIZE_ONE_M1497_C1_ACTIVE_NEXT_ORACLE_UNIT_DELAY_VCS_ATTEMPT"
FINAL_STATUS = "PASS_M1500_AUTHORIZE_ONE_M1497_C1_ACTIVE_NEXT_ORACLE_VCS_LAUNCH"
AUTHORIZATION = {"vcs_compiles": 1, "simv_runs": 1,
                 "all_other_eda_runs": 0, "automatic_retry": False}
CLAIMS = {"source_only": True, "functional_vcs": False,
          "timing_verified": False, "cycles_measured": False,
          "speedup": False, "ppa": False, "power": False, "energy": False,
          "system_speedup": False, "headline": False}
ENV_PINS = (
    "M1497_EXPECTED_RUNNER_SHA256",
    "M1497_EXPECTED_CONTRACT_SHA256",
    "M1497_EXPECTED_AUTHOR_REVIEW_SHA256",
    "M1497_EXPECTED_AUTHOR_MANIFEST_SHA256",
    "M1497_EXPECTED_AUTHOR_OUTER_SHA256",
    "M1497_EXPECTED_HAMMER_REVIEW_SHA256",
    "M1497_EXPECTED_HAMMER_MANIFEST_SHA256",
    "M1497_EXPECTED_HAMMER_OUTER_SHA256",
    "M1497_EXPECTED_RELEASE_SHA256",
    "M1497_EXPECTED_FINAL_REVIEW_SHA256",
    "M1497_EXPECTED_FINAL_MANIFEST_SHA256",
    "M1497_EXPECTED_FINAL_OUTER_SHA256",
)
COMPILE_COMMAND = [
    str(P.BASE.VCS), "-full64", "-sverilog", "-timescale=1ns/1ps",
    "-assert", "svaext", "+define+UNIT_DELAY", "+vcs+lic+wait",
    "-f", str(FILELIST), "-top", P.BASE.TOP, "-o", "simv",
]
SIM_COMMAND = ["./simv", "-no_save"]
CLEAN_PAYLOAD = {
    "compile.log", "sim.log",
    "m1497_c1_active_next_oracle_unit_delay_vcs_receipt_r1.json",
    "m1497_c1_active_next_oracle_identity_r1.json",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(path: Path, digest: str) -> None:
    mode = path.lstat().st_mode
    if path.is_symlink() or not stat.S_ISREG(mode) or sha(path) != digest:
        raise RuntimeError("identity drift: " + str(path))


def strict_json(path: Path) -> dict:
    return P.strict_json(path)


def publish_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint]
    if renameat2(-100, os.fsencode(source), -100,
                 os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def copy_regular(source: Path, destination: Path) -> None:
    mode = source.lstat().st_mode
    if source.is_symlink() or not stat.S_ISREG(mode):
        raise RuntimeError("nonregular raw evidence: " + str(source))
    if os.path.lexists(destination):
        raise RuntimeError("clean evidence collision: " + str(destination))
    shutil.copyfile(source, destination)
    if destination.is_symlink() or not stat.S_ISREG(destination.lstat().st_mode):
        raise RuntimeError("copied evidence not regular")


def seal_clean_result(root: Path) -> None:
    actual = {item.name for item in root.iterdir()}
    if actual != CLEAN_PAYLOAD:
        raise RuntimeError("clean result membership drift before seal")
    for item in root.iterdir():
        if item.is_symlink() or not stat.S_ISREG(item.lstat().st_mode):
            raise RuntimeError("clean result contains nonregular member")
    P.seal_dir_generic(root)
    P.verify_recursive_seal_generic(root)
    expected = CLEAN_PAYLOAD | {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    if {item.name for item in root.iterdir()} != expected:
        raise RuntimeError("clean result membership drift after seal")


def run_tool(command: list[str], log: Path, timeout: int,
             environment: dict[str, str]) -> int:
    process = subprocess.Popen(command, cwd=RAW_BUILD,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               start_new_session=True, env=environment)
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
        log.write_bytes(output or b"")
        raise RuntimeError("tool timeout")
    log.write_bytes(output or b"")
    return process.returncode


def namespace_gate() -> None:
    for item in (ATTEMPT, RESULT, QUARANTINE, RAW_BUILD,
                 CLEAN_RESULT_STAGE, ATTEMPT_STAGE, FAILURE_STAGE):
        if os.path.lexists(item):
            raise RuntimeError("namespace residue: " + str(item))
    patterns = (
        ".m1497_c1_active_next_oracle_raw_build.*",
        ".m1497_c1_active_next_oracle_clean_result_stage.*",
        ".m1497_c1_active_next_oracle_attempt_stage.*",
        ".m1497_c1_active_next_oracle_failure_stage.*",
    )
    if any(list((HW / "results").glob(pattern)) for pattern in patterns):
        raise RuntimeError("stale M1497 stage")


def validate_authority() -> None:
    exact(PREDECESSOR,
          "3c0028c41fbbd8f6d1ede4b284aece877dd926a2b82a67de26d71f5322a9e891")
    exact(TB_R13,
          "b749c7d635dc5b65669320aec7b7edb40cd5e2a5d781a9e474e3d28cbb054263")
    exact(TB,
          "e5604300f3e6cfcbdadfdafa8fae6a2faa6cdc1c18446fa8c48ba6ea10632526")
    exact(FILELIST,
          "de51bfdc95227ff7f8fbe2178465f1d088b9285067d9ed30770b357116a75e51")
    exact(DOCS359,
          "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")
    for name in ENV_PINS:
        if not re.fullmatch(r"[0-9a-f]{64}", os.environ.get(name, "")):
            raise RuntimeError("external digest absent/invalid: " + name)
    exact(RUNNER, os.environ["M1497_EXPECTED_RUNNER_SHA256"])
    P.verify_file_sidecar(CONTRACT)
    exact(CONTRACT, os.environ["M1497_EXPECTED_CONTRACT_SHA256"])
    author = P.verify_authority(
        AUTHOR, os.environ["M1497_EXPECTED_AUTHOR_REVIEW_SHA256"],
        os.environ["M1497_EXPECTED_AUTHOR_MANIFEST_SHA256"],
        os.environ["M1497_EXPECTED_AUTHOR_OUTER_SHA256"])
    hammer = P.verify_authority(
        HAMMER, os.environ["M1497_EXPECTED_HAMMER_REVIEW_SHA256"],
        os.environ["M1497_EXPECTED_HAMMER_MANIFEST_SHA256"],
        os.environ["M1497_EXPECTED_HAMMER_OUTER_SHA256"])
    P.verify_file_sidecar(RELEASE)
    exact(RELEASE, os.environ["M1497_EXPECTED_RELEASE_SHA256"])
    final = P.verify_authority(
        FINAL, os.environ["M1497_EXPECTED_FINAL_REVIEW_SHA256"],
        os.environ["M1497_EXPECTED_FINAL_MANIFEST_SHA256"],
        os.environ["M1497_EXPECTED_FINAL_OUTER_SHA256"])
    contract = strict_json(CONTRACT)
    release = strict_json(RELEASE)
    statuses = (contract.get("status"), author.get("status"),
                hammer.get("status"), release.get("status"),
                final.get("status"))
    if statuses != (SOURCE_STATUS, AUTHOR_STATUS, HAMMER_STATUS,
                    RELEASE_STATUS, FINAL_STATUS):
        raise RuntimeError("authority status drift")
    if release.get("authorization") != AUTHORIZATION:
        raise RuntimeError("authorization drift")
    if any(item.get("claim_boundary") != CLAIMS
           for item in (contract, author, hammer, release, final)):
        raise RuntimeError("claim boundary drift")


def make_clean_evidence(stage: Path, phase: str, exception: str | None,
                        compile_count: int, sim_count: int) -> None:
    stage.mkdir()
    for name in ("compile.log", "sim.log"):
        source = RAW_BUILD / name
        if source.exists():
            copy_regular(source, stage / name)
        else:
            (stage / name).write_text("")
    passed = exception is None
    receipt = {
        "schema": "m1497_c1_active_next_oracle_unit_delay_vcs_receipt_r1_v1",
        "status": ("PASS_FUNCTIONAL_VCS_REAL_M935_ACTIVE_NEXT_ORACLE"
                   if passed else "FAILED_OR_INCOMPLETE"),
        "phase": phase,
        "exception": exception,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "one_shot": {"attempt_consumed": True,
                     "vcs_compiles": compile_count, "simv_runs": sim_count,
                     "automatic_retry": False},
        "clean_result": {
            "raw_build_published": False,
            "regular_evidence_only": True,
            "symlink_policy_relaxed": False,
            "payload_before_seal": sorted(CLEAN_PAYLOAD),
        },
        "claim_boundary": {**CLAIMS, "source_only": not passed,
                           "functional_vcs": passed},
    }
    (stage / "m1497_c1_active_next_oracle_unit_delay_vcs_receipt_r1.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    identity = {
        "runner_sha256": sha(RUNNER), "testbench_sha256": sha(TB),
        "filelist_sha256": sha(FILELIST), "frozen_r13_sha256": sha(TB_R13),
        "contract_sha256": sha(CONTRACT),
    }
    (stage / "m1497_c1_active_next_oracle_identity_r1.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n")
    seal_clean_result(stage)


def main() -> int:
    os.umask(0o077)
    validate_authority()
    completed = subprocess.run(
        [str(P.BASE.PYTHON), "-I", str(CHECKER), "--mode", "runtime_present"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        timeout=120, check=False)
    if completed.returncode != 0:
        raise RuntimeError("runtime source gate failed: " + completed.stderr)
    namespace_gate()
    P.BASE.collision_gate()
    P.BASE.resource_gate()
    P.BASE.collision_gate()
    phase = "ATTEMPT_CONSUME"
    compile_count = 0
    sim_count = 0
    ATTEMPT_STAGE.mkdir()
    (ATTEMPT_STAGE / "attempt.json").write_text(json.dumps({
        "status": "M1497_ATTEMPT_CONSUMED", "automatic_retry": False,
        "maximum_vcs_compiles": 1, "maximum_simv_runs": 1,
    }, indent=2, sort_keys=True) + "\n")
    P.seal_dir_generic(ATTEMPT_STAGE)
    publish_no_replace(ATTEMPT_STAGE, ATTEMPT)
    RAW_BUILD.mkdir()
    environment = dict(os.environ)
    environment.update({
        "VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
        "VCS_ARCH_OVERRIDE": "linux",
        "SNPSLMD_LICENSE_FILE": "27030@ic.ismd-nemo",
        "LM_LICENSE_FILE": "/opt/synopsys/Synopsys.dat",
    })
    try:
        phase = "COMPILE"
        compile_count = 1
        if run_tool(COMPILE_COMMAND, RAW_BUILD / "compile.log",
                    P.BASE.COMPILE_TIMEOUT_SECONDS, environment) != 0:
            raise RuntimeError("compile failed")
        simv = RAW_BUILD / "simv"
        if simv.is_symlink() or not simv.is_file() or not os.access(simv, os.X_OK):
            raise RuntimeError("simv absent/nonregular")
        phase = "SIMULATE"
        sim_count = 1
        if run_tool(SIM_COMMAND, RAW_BUILD / "sim.log",
                    P.BASE.SIM_TIMEOUT_SECONDS, environment) != 0:
            raise RuntimeError("simulation failed")
        log = (RAW_BUILD / "sim.log").read_text(errors="replace")
        if (log.splitlines().count(P.BASE.R13_PASS) != 1
                or log.splitlines().count(P.BASE.R15_PASS) != 1):
            raise RuntimeError("pass token cardinality")
        phase = "SUCCESS_PUBLISH"
        make_clean_evidence(CLEAN_RESULT_STAGE, phase, None,
                            compile_count, sim_count)
        publish_no_replace(CLEAN_RESULT_STAGE, RESULT)
        print("PASS M1497 C1 active-next oracle functional VCS")
        return 0
    except BaseException as exc:
        if not RESULT.exists() and not QUARANTINE.exists():
            make_clean_evidence(FAILURE_STAGE, phase,
                                type(exc).__name__ + ": " + str(exc),
                                compile_count, sim_count)
            publish_no_replace(FAILURE_STAGE, QUARANTINE)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
