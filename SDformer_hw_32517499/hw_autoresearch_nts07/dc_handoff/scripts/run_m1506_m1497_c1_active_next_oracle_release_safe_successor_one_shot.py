#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Release-safe additive successor for the M1497 C1 active-next oracle.

The M1497 TB/oracle and raw/clean evidence split are preserved byte-for-byte.
M1506 adds exact live-input binding, strict log admission, and a failure guard
covering every operation after canonical attempt publication.  This source is
inert until fresh M1507/M1508/M1509 authorities exist and are externally
SHA-256 pinned.
"""
from __future__ import annotations

import ctypes
from datetime import datetime, timezone
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
from typing import Any


if len(sys.argv) != 1:
    raise SystemExit("M1506: no arguments accepted")

SCRIPT_DIR = Path(__file__).resolve().parent
HW = SCRIPT_DIR.parents[1]
RUNNER = Path(__file__).resolve()
M1497_RUNNER = SCRIPT_DIR / "run_m1497_m1459_c1_active_next_oracle_clean_result_successor_one_shot.py"
SPEC = importlib.util.spec_from_file_location("m1506_frozen_m1497", M1497_RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("M1506: cannot load frozen M1497 runner")
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)
BASE = P.P.BASE

CHECKER = HW / "verif_m1506_c1_active_next_oracle_release_safe_successor/check_m1506_source.py"
TESTS = HW / "verif_m1506_c1_active_next_oracle_release_safe_successor/test_m1506_source.py"
CONTRACT = HW / "contracts/m1506_c1_active_next_oracle_release_safe_successor_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1506_c1_active_next_oracle_release_safe_successor_source_author_r1_20260831"
M1498_FAIL = HW / "reviews/m1498_m1497_c1_active_next_oracle_source_blind_hammer_r1_20260831"
HAMMER = HW / "reviews/m1507_m1506_c1_active_next_oracle_release_safe_source_blind_hammer_r1_20260831"
RELEASE = HW / "contracts/m1508_m1507_m1506_c1_active_next_oracle_vcs_launch_release_r1_20260831.json"
FINAL = HW / "reviews/m1509_m1508_m1506_c1_active_next_oracle_final_launch_hammer_r1_20260831"

ATTEMPT = HW / "results/.m1506_c1_active_next_oracle_vcs_attempt_consumed"
RESULT = HW / "results/m1506_c1_active_next_oracle_unit_delay_vcs_r1_20260831"
QUARANTINE = Path(str(RESULT) + ".failed_or_incomplete.quarantine")
RAW_BUILD = HW / f"results/.m1506_c1_active_next_oracle_raw_build.{os.getpid()}"
CLEAN_RESULT_STAGE = HW / f"results/.m1506_c1_active_next_oracle_clean_result_stage.{os.getpid()}"
ATTEMPT_STAGE = HW / f"results/.m1506_c1_active_next_oracle_attempt_stage.{os.getpid()}"
FAILURE_STAGE = HW / f"results/.m1506_c1_active_next_oracle_failure_stage.{os.getpid()}"

SOURCE_STATUS = "M1506_C1_ACTIVE_NEXT_ORACLE_RELEASE_SAFE_SOURCE_READY__NO_LAUNCH"
AUTHOR_STATUS = "PASS_M1506_C1_ACTIVE_NEXT_ORACLE_RELEASE_SAFE_SOURCE__NO_VCS_NO_EDA"
HAMMER_STATUS = "PASS_M1507_M1506_C1_ACTIVE_NEXT_ORACLE_RELEASE_SAFE_SOURCE__RELEASE_NOT_AUTHORED"
RELEASE_STATUS = "AUTHORIZE_ONE_M1506_C1_ACTIVE_NEXT_ORACLE_UNIT_DELAY_VCS_ATTEMPT"
FINAL_STATUS = "PASS_M1509_AUTHORIZE_ONE_M1506_C1_ACTIVE_NEXT_ORACLE_VCS_LAUNCH"
M1498_STATUS = "FAIL_DO_NOT_CITE__M1497_ADDITIVE_SUCCESSOR_REQUIRED__NO_M1499"
AUTHORIZATION = {"vcs_compiles": 1, "simv_runs": 1,
                 "all_other_eda_runs": 0, "automatic_retry": False}
CLAIMS = {"source_only": True, "functional_vcs": False,
          "timing_verified": False, "cycles_measured": False,
          "speedup": False, "ppa": False, "power": False, "energy": False,
          "system_speedup": False, "headline": False}
ENV_PINS = (
    "M1506_EXPECTED_RUNNER_SHA256",
    "M1506_EXPECTED_CONTRACT_SHA256",
    "M1506_EXPECTED_AUTHOR_REVIEW_SHA256",
    "M1506_EXPECTED_AUTHOR_MANIFEST_SHA256",
    "M1506_EXPECTED_AUTHOR_OUTER_SHA256",
    "M1506_EXPECTED_HAMMER_REVIEW_SHA256",
    "M1506_EXPECTED_HAMMER_MANIFEST_SHA256",
    "M1506_EXPECTED_HAMMER_OUTER_SHA256",
    "M1506_EXPECTED_RELEASE_SHA256",
    "M1506_EXPECTED_FINAL_REVIEW_SHA256",
    "M1506_EXPECTED_FINAL_MANIFEST_SHA256",
    "M1506_EXPECTED_FINAL_OUTER_SHA256",
)
COMPILE_COMMAND = [
    str(BASE.VCS), "-full64", "-sverilog", "-timescale=1ns/1ps",
    "-assert", "svaext", "+define+UNIT_DELAY", "+vcs+lic+wait",
    "-f", str(P.FILELIST), "-top", BASE.TOP, "-o", "simv",
]
SIM_COMMAND = ["./simv", "-no_save"]
CLEAN_PAYLOAD = {
    "compile.log", "sim.log",
    "m1506_c1_active_next_oracle_unit_delay_vcs_receipt_r1.json",
    "m1506_c1_active_next_oracle_identity_r1.json",
}
M1497_PINS = {
    "runner": "db9154c1e8ab88afc209fefd39123ad812b2f6eeb566c031e7f1824d15ead708",
    "checker": "d29fd7c1fafa92ed572214b4ee2441bd5ec752adfb223f91632dd382489e74c0",
    "tests": "f15b007327e1394362ec818d48ee191656728c8fbebce75cd31c7b9dc2159110",
    "tb": "e5604300f3e6cfcbdadfdafa8fae6a2faa6cdc1c18446fa8c48ba6ea10632526",
    "filelist": "de51bfdc95227ff7f8fbe2178465f1d088b9285067d9ed30770b357116a75e51",
    "r13": "b749c7d635dc5b65669320aec7b7edb40cd5e2a5d781a9e474e3d28cbb054263",
    "contract": "c3531c4c8d55046cf7f5eee5717a3ed5a3a6c475cbbb115e8c84a4a80e308375",
    "contract_sidecar": "2f8eb776d3f655394cc655e6b6f07bf1319e3fa14574db3667edf6a5edc2dc7e",
    "contract_outer": "49506b3f2a8b8e09d839dab83bba157b26a5219666ba3c3a6cab9886b178a0d6",
    "author_review": "aa7689c0401fe212d006b5e0b32b3cdad2237c6f53c6054c197295f26ad55919",
    "author_manifest": "570101c7002765d75ead34e084412050c0418eb2d10c26f8f014ea7435d6cae0",
    "author_outer": "02ec9968b4e0edf719cd8793091b4ac096f0d96d8c651db9c3b86affb9e9db46",
}
M1498_PINS = {
    "review": "806cd6f629d17076e7f8bc1df0a633fb6d0a9cd68cf762d8f167123d3c7913b8",
    "manifest": "df0b581860be722c7c2e49bde4878dee317f72a5097d2b6e6c4e5c1861ddd300",
    "outer": "0e1d91e0dd700390abf78df87ab5a53fc3187eea1e4d53a8310ae77961eac2d4",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(path: Path, digest: str) -> None:
    mode = path.lstat().st_mode
    if path.is_symlink() or not stat.S_ISREG(mode) or sha(path) != digest:
        raise RuntimeError("identity drift: " + str(path))


def strict_json(path: Path) -> dict[str, Any]:
    return P.P.strict_json(path)


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
    if {item.name for item in root.iterdir()} != CLEAN_PAYLOAD:
        raise RuntimeError("clean result membership drift before seal")
    for item in root.iterdir():
        if item.is_symlink() or not stat.S_ISREG(item.lstat().st_mode):
            raise RuntimeError("clean result contains nonregular member")
    P.P.seal_dir_generic(root)
    P.P.verify_recursive_seal_generic(root)
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
    for pattern in (
        ".m1506_c1_active_next_oracle_raw_build.*",
        ".m1506_c1_active_next_oracle_clean_result_stage.*",
        ".m1506_c1_active_next_oracle_attempt_stage.*",
        ".m1506_c1_active_next_oracle_failure_stage.*",
    ):
        if list((HW / "results").glob(pattern)):
            raise RuntimeError("stale M1506 stage: " + pattern)


def validate_frozen_inputs(contract: dict[str, Any]) -> None:
    exact(M1497_RUNNER, M1497_PINS["runner"])
    exact(P.CHECKER, M1497_PINS["checker"])
    exact(P.TESTS, M1497_PINS["tests"])
    exact(P.TB, M1497_PINS["tb"])
    exact(P.FILELIST, M1497_PINS["filelist"])
    exact(P.TB_R13, M1497_PINS["r13"])
    exact(P.CONTRACT, M1497_PINS["contract"])
    exact(Path(str(P.CONTRACT) + ".sha256"), M1497_PINS["contract_sidecar"])
    exact(Path(str(P.CONTRACT) + ".sha256.seal.sha256"), M1497_PINS["contract_outer"])
    old_author = P.P.verify_authority(P.AUTHOR, M1497_PINS["author_review"],
                                      M1497_PINS["author_manifest"],
                                      M1497_PINS["author_outer"])
    if old_author.get("status") != P.AUTHOR_STATUS:
        raise RuntimeError("M1497 author status drift")
    old_failure = P.P.verify_authority(M1498_FAIL, M1498_PINS["review"],
                                       M1498_PINS["manifest"], M1498_PINS["outer"])
    if (old_failure.get("status") != M1498_STATUS
            or old_failure.get("authorization", {}).get(
                "m1499_release_authoring") is not False):
        raise RuntimeError("M1498 failure boundary drift")
    for path, digest in BASE.EXACT.items():
        exact(path, digest)
    identity = contract["identity"]
    exact(CHECKER, identity["checker_sha256"])
    exact(TESTS, identity["tests_sha256"])
    live = {
        "runner_sha256": sha(RUNNER),
        "checker_sha256": sha(CHECKER),
        "tests_sha256": sha(TESTS),
        "testbench_sha256": sha(P.TB),
        "filelist_sha256": sha(P.FILELIST),
        "parent_rtl_sha256": sha(BASE.PARENT),
        "m935_rtl_sha256": sha(BASE.M935),
        "wrapper_rtl_sha256": sha(BASE.WRAPPER),
        "sva_sha256": sha(BASE.SVA),
        "witness_sha256": sha(BASE.WITNESS),
        "foundry_model_sha256": sha(BASE.FOUNDRY),
        "vcs_binary_sha256": sha(BASE.VCS),
        "docs359_sha256": sha(BASE.DOCS359),
    }
    if any(identity.get(key) != value for key, value in live.items()):
        raise RuntimeError("M1506 live frozen-input binding drift")


def validate_authority() -> None:
    for name in ENV_PINS:
        if not re.fullmatch(r"[0-9a-f]{64}", os.environ.get(name, "")):
            raise RuntimeError("external digest absent/invalid: " + name)
    exact(RUNNER, os.environ["M1506_EXPECTED_RUNNER_SHA256"])
    P.P.verify_file_sidecar(CONTRACT)
    exact(CONTRACT, os.environ["M1506_EXPECTED_CONTRACT_SHA256"])
    contract = strict_json(CONTRACT)
    validate_frozen_inputs(contract)
    author = P.P.verify_authority(
        AUTHOR, os.environ["M1506_EXPECTED_AUTHOR_REVIEW_SHA256"],
        os.environ["M1506_EXPECTED_AUTHOR_MANIFEST_SHA256"],
        os.environ["M1506_EXPECTED_AUTHOR_OUTER_SHA256"])
    hammer = P.P.verify_authority(
        HAMMER, os.environ["M1506_EXPECTED_HAMMER_REVIEW_SHA256"],
        os.environ["M1506_EXPECTED_HAMMER_MANIFEST_SHA256"],
        os.environ["M1506_EXPECTED_HAMMER_OUTER_SHA256"])
    P.P.verify_file_sidecar(RELEASE)
    exact(RELEASE, os.environ["M1506_EXPECTED_RELEASE_SHA256"])
    release = strict_json(RELEASE)
    final = P.P.verify_authority(
        FINAL, os.environ["M1506_EXPECTED_FINAL_REVIEW_SHA256"],
        os.environ["M1506_EXPECTED_FINAL_MANIFEST_SHA256"],
        os.environ["M1506_EXPECTED_FINAL_OUTER_SHA256"])
    statuses = (contract.get("status"), author.get("status"),
                hammer.get("status"), release.get("status"),
                final.get("status"))
    if statuses != (SOURCE_STATUS, AUTHOR_STATUS, HAMMER_STATUS,
                    RELEASE_STATUS, FINAL_STATUS):
        raise RuntimeError("authority status drift")
    bindings = {
        "runner_sha256": sha(RUNNER), "checker_sha256": sha(CHECKER),
        "tests_sha256": sha(TESTS), "contract_sha256": sha(CONTRACT),
        "m1497_runner_sha256": M1497_PINS["runner"],
        "m1498_failure_review_sha256": M1498_PINS["review"],
    }
    if any(author.get("bindings", {}).get(key) != value
           for key, value in bindings.items()):
        raise RuntimeError("author binding drift")
    if any(hammer.get("bindings", {}).get(key) != value
           for key, value in bindings.items()):
        raise RuntimeError("hammer binding drift")
    if release.get("identity", {}).get("source_hammer_review_sha256") != sha(
            HAMMER / "review.json"):
        raise RuntimeError("release hammer binding drift")
    if final.get("bindings", {}).get("launch_release_sha256") != sha(RELEASE):
        raise RuntimeError("final release binding drift")
    if release.get("authorization") != AUTHORIZATION:
        raise RuntimeError("release authorization drift")
    if final.get("authorization") != AUTHORIZATION:
        raise RuntimeError("final authorization drift")
    if any(item.get("claim_boundary") != CLAIMS
           for item in (contract, author, hammer, release, final)):
        raise RuntimeError("claim boundary drift")


R13_ENTER = "PHASE_M1270R13_REAL_M935_INTEGRATED_ENTER"
R13_COMPLETE = "PHASE_M1270R13_REAL_M935_INTEGRATED_COMPLETE"
WITNESS_OPERANDS = (
    "M1337R15_WITNESS_OPERANDS pass=1 stage=7 weight_req=2 psum_req=1 "
    "responses=2 core_accepts=2 psum_commits=1 rows=1 tasks=1 "
    "design_issue=2 design_commit=1 design_rows=1 masks=0 faults=0"
)
COVERAGE_RE = re.compile(
    r"^COVERAGE_M1270R13_REAL_M935 first_beats=1 nonfirst_beats=1 "
    r"join_hold_cycles=2 issue_accepts=2 psum_reads=1 row_completions=1 "
    r"task_completions=1 response_cycle_gap=([2-9][0-9]*) "
    r"oracle_records=([8-9][0-9]*|[1-9][0-9]{2,}) "
    r"parent_issue_override=0 child_issue_override=0$"
)
CP_NONFIRST_RE = re.compile(
    r"^.*\.cp_nonfirst, [1-9][0-9]* attempts, [1-9][0-9]* match(?:es)?$"
)
CP_II2_RE = re.compile(
    r"^.*\.cp_ii2, [1-9][0-9]* attempts, [1-9][0-9]* match(?:es)?$"
)
FORBIDDEN_LOG_RE = re.compile(
    r"(^|[^A-Za-z0-9_])(Error|Fatal|\$error|\$fatal)([^A-Za-z0-9_]|$)|"
    r"assertion[^\n]*(fail|error)|(fail|error)[^\n]*assertion",
    re.IGNORECASE,
)


def validate_sim_log(log: str) -> dict[str, int]:
    lines = log.splitlines()
    required_exact = (BASE.R13_PASS, BASE.R15_PASS, R13_ENTER,
                      R13_COMPLETE, WITNESS_OPERANDS)
    if any(lines.count(token) != 1 for token in required_exact):
        raise RuntimeError("required log token cardinality")
    coverage = [COVERAGE_RE.fullmatch(line) for line in lines]
    coverage = [match for match in coverage if match is not None]
    if len(coverage) != 1:
        raise RuntimeError("coverage cardinality")
    nonfirst = [line for line in lines if CP_NONFIRST_RE.fullmatch(line)]
    ii2 = [line for line in lines if CP_II2_RE.fullmatch(line)]
    if len(nonfirst) != 1 or len(ii2) != 1:
        raise RuntimeError("cp_nonfirst/cp_ii2 coverage missing or duplicate")
    oracle_lines = [line for line in lines if line.startswith("ORACLE_M1270R13 ")]
    if len(oracle_lines) < 80 or any(" pass=1 " not in line for line in oracle_lines):
        raise RuntimeError("oracle record population/pass drift")
    if FORBIDDEN_LOG_RE.search(log):
        raise RuntimeError("error/fatal/assertion-failure line")
    if re.search(r"\b(?:boundary_fault|core_fault|m935_fault|faults?)=[xXzZ1-9]", log):
        raise RuntimeError("nonzero/unknown fault line")
    return {
        "weight_requests": 2, "psum_requests": 1, "responses": 2,
        "core_accepts": 2, "psum_commits": 1, "row_completions": 1,
        "task_completions": 1, "cp_nonfirst_matches_min": 1,
        "cp_ii2_matches_min": 1,
        "response_cycle_gap": int(coverage[0].group(1)),
        "oracle_records": len(oracle_lines), "assertion_failures": 0,
        "design_faults": 0,
    }


def make_clean_evidence(stage: Path, phase: str, exception: str | None,
                        compile_count: int, sim_count: int,
                        log_audit: dict[str, int] | None) -> None:
    passed = exception is None
    stage.mkdir()
    for name in ("compile.log", "sim.log"):
        source = RAW_BUILD / name
        if source.exists():
            mode = source.lstat().st_mode
            if passed or (not source.is_symlink() and stat.S_ISREG(mode)):
                copy_regular(source, stage / name)
            else:
                (stage / name).write_text(
                    "M1506 failure evidence: raw log was nonregular and was not followed.\n")
        else:
            (stage / name).write_text("")
    receipt = {
        "schema": "m1506_c1_active_next_oracle_unit_delay_vcs_receipt_r1_v1",
        "status": ("PASS_FUNCTIONAL_VCS_REAL_M935_RELEASE_SAFE_ORACLE"
                   if passed else "FAILED_OR_INCOMPLETE"),
        "phase": phase, "exception": exception,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "one_shot": {"attempt_consumed": True,
                     "vcs_compiles": compile_count, "simv_runs": sim_count,
                     "automatic_retry": False},
        "log_admission": log_audit,
        "clean_result": {"raw_build_published": False,
                         "regular_evidence_only": True,
                         "symlink_policy_relaxed": False,
                         "payload_before_seal": sorted(CLEAN_PAYLOAD)},
        "claim_boundary": {**CLAIMS, "source_only": not passed,
                           "functional_vcs": passed},
    }
    (stage / "m1506_c1_active_next_oracle_unit_delay_vcs_receipt_r1.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    identity = {
        "runner_sha256": sha(RUNNER), "checker_sha256": sha(CHECKER),
        "testbench_sha256": sha(P.TB), "filelist_sha256": sha(P.FILELIST),
        "frozen_r13_sha256": sha(P.TB_R13), "contract_sha256": sha(CONTRACT),
        "m1497_runner_sha256": M1497_PINS["runner"],
        "m1498_failure_review_sha256": M1498_PINS["review"],
    }
    (stage / "m1506_c1_active_next_oracle_identity_r1.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n")
    seal_clean_result(stage)


def main() -> int:
    os.umask(0o077)
    validate_authority()
    completed = subprocess.run(
        [str(BASE.PYTHON), "-I", str(CHECKER), "--mode", "runtime_present"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        timeout=120, check=False)
    if completed.returncode != 0:
        raise RuntimeError("runtime source gate failed: " + completed.stderr)
    namespace_gate()
    BASE.collision_gate()
    BASE.resource_gate()
    BASE.collision_gate()
    phase = "ATTEMPT_CONSUME"
    compile_count = 0
    sim_count = 0
    log_audit = None
    attempt_consumed = False
    try:
        def interrupted(signum, _frame):
            raise RuntimeError("interrupted by signal " + str(signum))
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, interrupted)
        ATTEMPT_STAGE.mkdir()
        (ATTEMPT_STAGE / "attempt.json").write_text(json.dumps({
            "status": "M1506_ATTEMPT_CONSUMED", "automatic_retry": False,
            "maximum_vcs_compiles": 1, "maximum_simv_runs": 1,
            "runner_sha256": sha(RUNNER), "contract_sha256": sha(CONTRACT),
        }, indent=2, sort_keys=True) + "\n")
        P.P.seal_dir_generic(ATTEMPT_STAGE)
        publish_no_replace(ATTEMPT_STAGE, ATTEMPT)
        attempt_consumed = True

        phase = "RAW_BUILD_CREATE"
        RAW_BUILD.mkdir()
        environment = dict(os.environ)
        environment.update({
            "VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
            "VCS_ARCH_OVERRIDE": "linux",
            "SNPSLMD_LICENSE_FILE": "27030@ic.ismd-nemo",
            "LM_LICENSE_FILE": "/opt/synopsys/Synopsys.dat",
        })
        phase = "COMPILE"
        compile_count = 1
        if run_tool(COMPILE_COMMAND, RAW_BUILD / "compile.log",
                    BASE.COMPILE_TIMEOUT_SECONDS, environment) != 0:
            raise RuntimeError("compile failed")
        simv = RAW_BUILD / "simv"
        if simv.is_symlink() or not simv.is_file() or not os.access(simv, os.X_OK):
            raise RuntimeError("simv absent/nonregular")
        phase = "SIMULATE"
        sim_count = 1
        if run_tool(SIM_COMMAND, RAW_BUILD / "sim.log",
                    BASE.SIM_TIMEOUT_SECONDS, environment) != 0:
            raise RuntimeError("simulation failed")
        phase = "LOG_ADMISSION"
        log_audit = validate_sim_log((RAW_BUILD / "sim.log").read_text(
            errors="replace"))
        phase = "SUCCESS_PUBLISH"
        make_clean_evidence(CLEAN_RESULT_STAGE, phase, None,
                            compile_count, sim_count, log_audit)
        publish_no_replace(CLEAN_RESULT_STAGE, RESULT)
        print("PASS M1506 C1 release-safe active-next oracle functional VCS")
        return 0
    except BaseException as exc:
        if attempt_consumed or ATTEMPT.exists():
            if not RESULT.exists() and not QUARANTINE.exists():
                make_clean_evidence(
                    FAILURE_STAGE, phase,
                    type(exc).__name__ + ": " + str(exc),
                    compile_count, sim_count, log_audit)
                publish_no_replace(FAILURE_STAGE, QUARANTINE)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
