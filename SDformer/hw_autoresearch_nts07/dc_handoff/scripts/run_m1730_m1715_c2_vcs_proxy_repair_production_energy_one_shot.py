#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1730 additive VCS-compile proxy repair for the exhausted M1715 run.

M1715 consumed exactly one VCS compile and was interrupted before producing a
simv, SAIF, or PTPX result.  M1730 never retries or rewrites M1715.  It binds
the sealed M1715 attempt/failure, keeps the same workload and execution budget,
and copies an exact, narrowly allowlisted launch-proxy tuple only into each VCS
compile environment.  The proxy is TCP-preflighted before the fresh attempt is
consumed.  Simulation, local checkers, lmutil, and PrimeTime receive no proxy
variables.  This source is inert until an independent M1731 review and M1732
release bind its exact hashes.
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
import socket
import stat
import subprocess
import sys
from typing import Any


HW = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
OLD_RUNNER = HW / "dc_handoff/scripts/run_m1715_m1710_m1684_m1661_c2_queue_order_repair_production_energy_one_shot.py"
SPEC = importlib.util.spec_from_file_location("m1715_runner_for_m1730", OLD_RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("M1715 runner import unavailable")
OLD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OLD)

CONTRACT = HW / "contracts/m1730_m1715_c2_vcs_proxy_repair_production_energy_source_contract_r1_20260901.json"
CHECKER = HW / "system_simulator/scripts/check_m1730_m1715_c2_vcs_proxy_repair_production_energy_source.py"
TEST = HW / "system_simulator/tests/test_m1730_m1715_c2_vcs_proxy_repair_production_energy_source.py"
M1731 = HW / "reviews/m1731_m1730_m1715_c2_vcs_proxy_repair_production_energy_source_hammer_r1_20260901"
M1732 = HW / "contracts/m1732_m1731_m1730_m1715_c2_vcs_proxy_repair_production_energy_launch_release_r1_20260901.json"

M1715_CHECKER = HW / "system_simulator/scripts/check_m1715_m1710_m1684_c2_queue_order_repair_production_energy_source.py"
M1715_TEST = HW / "system_simulator/tests/test_m1715_m1710_m1684_c2_queue_order_repair_production_energy_source.py"
M1715_CONTRACT = HW / "contracts/m1715_m1710_m1684_c2_queue_order_repair_production_energy_source_contract_r1_20260901.json"
M1715_AUTHOR = HW / "reviews/m1715_m1710_m1684_c2_queue_order_repair_production_energy_source_author_receipt_r1_20260901"
M1716 = HW / "reviews/m1716_m1715_m1710_m1684_c2_queue_order_repair_production_energy_source_hammer_r1_20260901"
M1717 = HW / "contracts/m1717_m1716_m1715_m1710_m1684_c2_queue_order_repair_production_energy_launch_release_r1_20260901.json"
M1715_ATTEMPT = HW / "results/.m1715_c2_queue_order_repair_production_energy_attempt_consumed"
M1715_FAILURE = HW / "results/m1715_c2_queue_order_repair_production_energy_r1_20260901.failed_or_incomplete.quarantine"
M1715_RESULT = HW / "results/m1715_c2_queue_order_repair_production_energy_r1_20260901"
M1715_PRIVATE = HW / "results/m1715_c2_queue_order_repair_production_energy_r1_20260901.private_build.unsealed_do_not_cite"

ATTEMPT = HW / "results/.m1730_c2_vcs_proxy_repair_production_energy_attempt_consumed"
RESULT = HW / "results/m1730_c2_vcs_proxy_repair_production_energy_r1_20260901"
FAILURE = HW / "results/m1730_c2_vcs_proxy_repair_production_energy_r1_20260901.failed_or_incomplete.quarantine"
PRIVATE = HW / "results/m1730_c2_vcs_proxy_repair_production_energy_r1_20260901.private_build.unsealed_do_not_cite"
WORK = HW / ("results/.m1730_c2_vcs_proxy_repair_production_energy_work." + str(os.getpid()))
STAGE = HW / ("results/.m1730_c2_vcs_proxy_repair_production_energy_stage." + str(os.getpid()))
FAIL_STAGE = HW / ("results/.m1730_c2_vcs_proxy_repair_production_energy_failure_stage." + str(os.getpid()))

LOCK = OLD.LOCK
VCS = OLD.VCS
PT = OLD.PT
PYTHON = OLD.PYTHON
LMUTIL = OLD.LMUTIL
LICENSE_SERVER = OLD.LICENSE_SERVER
LICENSE_FILE = OLD.LICENSE_FILE
FILELISTS = OLD.FILELISTS
UCLI = OLD.UCLI
PT_TCL = OLD.PT_TCL
BASE = OLD.BASE
NET_REL = OLD.NET_REL
SDC_REL = OLD.SDC_REL
DESIGN = OLD.DESIGN
TOP = OLD.TOP
SAIF_SCOPE = OLD.SAIF_SCOPE
TT_DB = OLD.TT_DB
SS_DB = OLD.SS_DB
CYCLES = OLD.CYCLES
EVENTS = OLD.EVENTS
COUNTS = {"vcs_compiles": 2, "simv_runs": 10,
          "saif_files": 10, "ptpx_runs": 10}
SOURCE_CLAIMS = dict((key, False) for key in (
    "vcs", "mapped_functionality", "production_saif", "ptpx", "power",
    "energy", "performance", "system_speedup", "paper_ppa_ready", "headline"))

PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
              "http_proxy", "https_proxy")
EXPECTED_PROXY = {
    "HTTP_PROXY": "http://127.0.0.1:7897",
    "HTTPS_PROXY": "http://127.0.0.1:7897",
    "ALL_PROXY": "http://127.0.0.1:7897",
    "NO_PROXY": "localhost,127.0.0.1,::1",
    "http_proxy": "http://127.0.0.1:7897",
    "https_proxy": "http://127.0.0.1:7897",
}
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 7897
PROXY_CONNECT_TIMEOUT_S = 2.0

FIXED_SHA = {
    "m1715_runner": "b5010f8d3d70ea2e029a3636b7b338d61b3b6f02ae8fa130dfbe279e736a2225",
    "m1715_checker": "1c2e917335be2e018c8df526e38265b94a7737241ac5ed1af3fbb02082538988",
    "m1715_test": "bf76d2242fca01cee07a12f243ed53c3a755e2c15e79274e92f0a4de5cede11c",
    "m1715_contract": "d775ca27bc3fa017751582aba34db8f3ebceac3aa9ecd8a550f413731ab68fc7",
    "m1715_author_receipt": "a584b48d3507320180fcfb3a5b66d5fe1ba426ae2434bf41f132c569153d6c2d",
    "m1715_author_manifest": "d7cc3a26ceb1418783270179d802f93182c6cc098f5580e9a2f9aef863261e2a",
    "m1715_author_outer": "1869f59abc911c43f422b6b6b052473c2c679fbba7846afcb7bb11d087106379",
    "m1716_review": "68c132bff9e52b2849c15e240e1cbc860706bdb24f4b3a1b42ddeb54423aeff2",
    "m1716_manifest": "72f7cfd0ea0569cb3fadf8009e47aad8420cd90936fcf51eba4359914b3f86d8",
    "m1716_outer": "67add3d29c18ed7e1f960078a2b358fd2f51ac09ca4d21d3e3bc19c577e686af",
    "m1717_release": "32ae066e43a0889852e6eed7aa7b3a7cedf1a594a3e60116a629912ab9c2b6c8",
    "m1717_release_sum": "8eaecbb4b87332e4afbb99819459127d06fce1442f611e7588324736817f8905",
    "m1717_release_outer": "501012d3f2d1ae666673e4591a1c92dcd398084d17dff3b816fc01d3163a86f5",
    "m1715_attempt_json": "9cdfebe0acd0bd81d1ff92f7423020b336cbe4c7ebc0dbcb0118d7ceab068c53",
    "m1715_attempt_manifest": "18657601c99e8eb4141e3b175976a84bb491e04d7d1e85c46e82d6b463f71cf3",
    "m1715_attempt_outer": "ab8f355009cb82e9c1211ff39a92295fcbe776ac26c856a804c2e52b903b7835",
    "m1715_failure_json": "e9e09df40e1e1b6e02064150b7d8752d2303bb1edabddc3f38085021cc6f3c02",
    "m1715_failure_manifest": "2d2a24c7499c31a2667cc097037ffa3fa9d270606f104b3d85aac145712d49fe",
    "m1715_failure_outer": "d09f54d3acc5f1dba9f0cc0387b9b003bf3a9ff136ff804f2ae198055cd8f971",
}


class Failure(RuntimeError):
    pass


sha = OLD.sha
exact = OLD.exact
strict_json = OLD.strict_json
verify_seal = OLD.verify_seal
seal_dir = OLD.seal_dir
publish_no_replace = OLD.publish_no_replace
write_json = OLD.write_json
collision_gate = OLD.collision_gate
resource_gate = OLD.resource_gate
runtime_bind_execution_sources = OLD.runtime_bind_execution_sources
forbidden_release_namespaces_absent = OLD.forbidden_release_namespaces_absent


def verify_contract_sources(contract: dict[str, Any]) -> None:
    if (contract.get("schema") !=
            "m1730_m1715_c2_vcs_proxy_repair_production_energy_source_contract_r1_v1"
            or contract.get("status") !=
            "SOURCE_ONLY__M1731_REVIEW_AND_M1732_RELEASE_REQUIRED__NO_EDA"
            or contract.get("claim_boundary") != SOURCE_CLAIMS):
        raise Failure("M1730 source contract semantic drift")
    rows = contract.get("source_files")
    if not isinstance(rows, list):
        raise Failure("source file inventory absent")
    seen = set()
    for row in rows:
        if type(row) is not dict or set(row) != {"path", "sha256"} or row["path"] in seen:
            raise Failure("source inventory malformed")
        exact(HW / row["path"], row["sha256"])
        seen.add(row["path"])
    required = {RUNNER.relative_to(HW).as_posix(), CHECKER.relative_to(HW).as_posix(),
                TEST.relative_to(HW).as_posix()}
    if seen != required:
        raise Failure("source inventory incomplete/excess")


def verify_authority() -> dict[str, Any]:
    pins = {name: os.environ.get(name, "") for name in (
        "M1730_EXPECTED_RUNNER_SHA256",
        "M1730_EXPECTED_SOURCE_CONTRACT_SHA256",
        "M1730_EXPECTED_M1731_REVIEW_SHA256",
        "M1730_EXPECTED_M1731_MANIFEST_SHA256",
        "M1730_EXPECTED_M1731_OUTER_FILE_SHA256",
        "M1730_EXPECTED_M1732_RELEASE_SHA256")}
    if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in pins.values()):
        raise Failure("fresh M1731/M1732 exact SHA authority absent")
    exact(RUNNER, pins["M1730_EXPECTED_RUNNER_SHA256"])
    exact(CONTRACT, pins["M1730_EXPECTED_SOURCE_CONTRACT_SHA256"])
    contract = strict_json(CONTRACT)
    verify_contract_sources(contract)
    verify_seal(M1731, pins["M1730_EXPECTED_M1731_MANIFEST_SHA256"],
                pins["M1730_EXPECTED_M1731_OUTER_FILE_SHA256"])
    exact(M1731 / "review.json", pins["M1730_EXPECTED_M1731_REVIEW_SHA256"])
    review = strict_json(M1731 / "review.json")
    exact(M1732, pins["M1730_EXPECTED_M1732_RELEASE_SHA256"])
    release_sum = Path(str(M1732) + ".sha256")
    release_outer = Path(str(M1732) + ".sha256.seal.sha256")
    if (release_sum.read_text() != sha(M1732) + "  " + M1732.name + "\n"
            or release_outer.read_text() != sha(release_sum) + "  " + release_sum.name + "\n"):
        raise Failure("M1732 double seal drift")
    release = strict_json(M1732)
    expected_budget = {"future_m1730_attempts": 1, "automatic_retry": False,
                       "vcs_compiles": 2, "simv_runs": 10,
                       "saif_files": 10, "ptpx_runs": 10}
    if review.get("status") != (
            "PASS_M1731_M1730_M1715_C2_VCS_PROXY_REPAIR_PRODUCTION_ENERGY_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_ATTEMPT"):
        raise Failure("M1731 status drift")
    if (release.get("status") !=
            "AUTHORIZE_ONE_M1730_C2_VCS_PROXY_REPAIR_PRODUCTION_ENERGY_ATTEMPT"
            or release.get("authorization") != expected_budget
            or release.get("identity") != {
                "runner_sha256": sha(RUNNER),
                "source_contract_sha256": sha(CONTRACT),
                "m1731_review_sha256": sha(M1731 / "review.json")}
            or release.get("claim_boundary") != SOURCE_CLAIMS):
        raise Failure("M1732 authority semantic drift")
    return contract


def verify_m1715_consumed_failure() -> dict[str, Any]:
    """Bind M1715's exhausted identity; its unsealed private tree is non-citable."""
    for path, key in ((OLD_RUNNER, "m1715_runner"),
                      (M1715_CHECKER, "m1715_checker"),
                      (M1715_TEST, "m1715_test"),
                      (M1715_CONTRACT, "m1715_contract")):
        exact(path, FIXED_SHA[key])
    verify_seal(M1715_AUTHOR, FIXED_SHA["m1715_author_manifest"],
                FIXED_SHA["m1715_author_outer"])
    exact(M1715_AUTHOR / "author_receipt.json", FIXED_SHA["m1715_author_receipt"])
    verify_seal(M1716, FIXED_SHA["m1716_manifest"], FIXED_SHA["m1716_outer"])
    exact(M1716 / "review.json", FIXED_SHA["m1716_review"])
    exact(M1717, FIXED_SHA["m1717_release"])
    exact(Path(str(M1717) + ".sha256"), FIXED_SHA["m1717_release_sum"])
    exact(Path(str(M1717) + ".sha256.seal.sha256"), FIXED_SHA["m1717_release_outer"])
    verify_seal(M1715_ATTEMPT, FIXED_SHA["m1715_attempt_manifest"],
                FIXED_SHA["m1715_attempt_outer"])
    exact(M1715_ATTEMPT / "attempt.json", FIXED_SHA["m1715_attempt_json"])
    verify_seal(M1715_FAILURE, FIXED_SHA["m1715_failure_manifest"],
                FIXED_SHA["m1715_failure_outer"])
    exact(M1715_FAILURE / "failure.json", FIXED_SHA["m1715_failure_json"])
    attempted = strict_json(M1715_ATTEMPT / "attempt.json")
    failed = strict_json(M1715_FAILURE / "failure.json")
    if (attempted.get("status") !=
            "M1715_C2_QUEUE_ORDER_REPAIR_PRODUCTION_ENERGY_ATTEMPT_CONSUMED"
            or attempted.get("budget") != COUNTS
            or attempted.get("automatic_retry") is not False):
        raise Failure("M1715 attempt semantic drift")
    if failed != {
            "attempt_consumed": True, "automatic_retry": False,
            "canonical_result": False,
            "counts": {"ptpx_runs": 0, "saif_files": 0,
                       "simv_runs": 0, "vcs_compiles": 1},
            "error": "KeyboardInterrupt", "partial_axis_citable": False,
            "phase": "COMPILE_k8", "status": "FAILED_OR_INCOMPLETE"}:
        raise Failure("M1715 failure semantic drift")
    if os.path.lexists(M1715_RESULT):
        raise Failure("M1715 canonical result unexpectedly exists")
    # M1715_PRIVATE is deliberately neither sealed nor inspected for claims.
    return failed


def verify_predecessors_and_inputs() -> None:
    verify_m1715_consumed_failure()
    OLD.verify_predecessors_and_inputs()


def namespaces_fresh() -> None:
    for path in (ATTEMPT, RESULT, FAILURE, PRIVATE, WORK, STAGE, FAIL_STAGE):
        if os.path.lexists(path):
            raise Failure("namespace residue: " + str(path))
    for pattern in (".m1730_c2_vcs_proxy_repair_production_energy_work.*",
                    ".m1730_c2_vcs_proxy_repair_production_energy_stage.*",
                    ".m1730_c2_vcs_proxy_repair_production_energy_failure_stage.*"):
        if next((HW / "results").glob(pattern), None) is not None:
            raise Failure("stale private namespace: " + pattern)


def capture_exact_compile_proxy_from_launch() -> dict[str, str]:
    """Copy only the six audited proxy variables from the launch environment."""
    captured = {}
    for name in PROXY_KEYS:
        value = os.environ.get(name)
        if value != EXPECTED_PROXY[name]:
            raise Failure("launch proxy identity drift: " + name)
        captured[name] = value
    return captured


def preflight_compile_proxy(proxy: dict[str, str]) -> None:
    if proxy != EXPECTED_PROXY:
        raise Failure("compile proxy tuple drift")
    try:
        connection = socket.create_connection(
            (PROXY_HOST, PROXY_PORT), timeout=PROXY_CONNECT_TIMEOUT_S)
        connection.close()
    except OSError as error:
        raise Failure("VCS compile proxy TCP preflight failed") from error


def clean_env(extra: dict[str, str], *,
              vcs_compile_proxy: dict[str, str] | None = None) -> dict[str, str]:
    value = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
             "SNPSLMD_LICENSE_FILE": LICENSE_SERVER,
             "LM_LICENSE_FILE": str(LICENSE_FILE)}
    value.update(extra)
    if vcs_compile_proxy is not None:
        if vcs_compile_proxy != EXPECTED_PROXY:
            raise Failure("non-exact proxy injection rejected")
        value.update(vcs_compile_proxy)
    return value


def run(command: list[str], *, cwd: Path, env: dict[str, str], timeout: int,
        output: Path) -> None:
    if Path(command[0]).name in {"vcs", "pt_shell"}:
        collision_gate()
    with output.open("wb") as stream:
        completed = subprocess.run(command, cwd=cwd, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   timeout=timeout, check=False)
    if completed.returncode != 0:
        raise Failure("tool failure: " + " ".join(command[:2]))


def result_identity() -> dict[str, str]:
    return {
        "runner_sha256": sha(RUNNER),
        "source_contract_sha256": sha(CONTRACT),
        "m1731_review_sha256": sha(M1731 / "review.json"),
        "m1732_release_sha256": sha(M1732),
        "m1715_attempt_json_sha256": FIXED_SHA["m1715_attempt_json"],
        "m1715_failure_json_sha256": FIXED_SHA["m1715_failure_json"],
        "m1715_failure_manifest_sha256": FIXED_SHA["m1715_failure_manifest"],
        "shared_eda_queue_path": str(LOCK),
        "m1609_rtl_sha256": OLD.STATIC_SHA["m1609"],
        "k8_mapped_netlist_sha256": OLD.STATIC_SHA["k8_net"],
        "k1x8_mapped_netlist_sha256": OLD.STATIC_SHA["k1x8_net"],
    }


def main() -> int:
    if len(sys.argv) != 1:
        raise Failure("M1730 accepts no arguments")
    state: dict[str, Any] = {"phase": "SOURCE_CHAIN", "attempt": False,
                             "complete": False, "vcs_compiles": 0,
                             "simv_runs": 0, "saif_files": 0,
                             "ptpx_runs": 0}
    lock_handle = LOCK.open("a+")
    try:
        verify_authority()
        verify_predecessors_and_inputs()
        namespaces_fresh()
        state["phase"] = "QUEUE_WAIT"
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        state["phase"] = "POST_LOCK_COLLISION"
        collision_gate()
        state["phase"] = "POST_LOCK_RUNTIME_REBIND"
        runtime_bind_execution_sources()
        forbidden_release_namespaces_absent()
        resource_gate()
        namespaces_fresh()
        state["phase"] = "LICENSE_PREFLIGHT"
        if not LICENSE_FILE.is_file() or LICENSE_FILE.is_symlink():
            raise Failure("license file invalid")
        license_check = subprocess.run(
            [str(LMUTIL), "lmstat", "-a", "-c", LICENSE_SERVER],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=clean_env({}),
            timeout=60, check=False)
        if license_check.returncode != 0:
            raise Failure("license preflight failed")

        state["phase"] = "PROXY_PREFLIGHT"
        compile_proxy = capture_exact_compile_proxy_from_launch()
        preflight_compile_proxy(compile_proxy)
        state["phase"] = "PRE_ATTEMPT_RUNTIME_REBIND"
        collision_gate()
        runtime_bind_execution_sources()
        forbidden_release_namespaces_absent()
        state["phase"] = "ATTEMPT_CONSUME"
        ATTEMPT.mkdir()
        state["attempt"] = True
        write_json(ATTEMPT / "attempt.json", {
            "status": "M1730_C2_VCS_PROXY_REPAIR_PRODUCTION_ENERGY_ATTEMPT_CONSUMED",
            "identity": result_identity(), "budget": COUNTS,
            "axes": ["k8", "k1x8"], "cases": [0, 1, 2, 3, 4],
            "proxy_policy": {"vcs_compile_only": True,
                             "proxy_host": PROXY_HOST, "proxy_port": PROXY_PORT,
                             "launch_values_exact": True},
            "automatic_retry": False})
        seal_dir(ATTEMPT)
        WORK.mkdir()
        (WORK / "build").mkdir()
        (WORK / "candidate").mkdir()

        for axis in ("k8", "k1x8"):
            state["phase"] = "COMPILE_" + axis
            axis_dir = WORK / "build" / axis
            axis_dir.mkdir()
            state["vcs_compiles"] += 1
            command = [str(VCS), "-full64", "-sverilog", "+v2k",
                       "-timescale=1ns/1ps", "-assert", "svaext",
                       "-debug_access+r", "-lca", "+vcs+lic+wait",
                       "-Mdir=csrc", "-f", str(FILELISTS[axis]),
                       "-top", TOP, "-o", "simv"]
            run(command, cwd=axis_dir,
                env=clean_env({"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
                               "VCS_ARCH_OVERRIDE": "linux"},
                              vcs_compile_proxy=compile_proxy),
                timeout=3600, output=axis_dir / "compile.log")
            if not (axis_dir / "simv").is_file():
                raise Failure("fresh simv absent: " + axis)
            for case_id in range(5):
                state["phase"] = "SIM_" + axis + "_" + str(case_id)
                state["simv_runs"] += 1
                candidate = WORK / "candidate"
                saif = candidate / (axis + "_case" + str(case_id) + ".saif")
                log = candidate / (axis + "_case" + str(case_id) + ".log")
                report = candidate / (axis + "_case" + str(case_id) + ".assert.report")
                run(["./simv", "-lca", "+M979_UCLI_SAIF",
                     "+M979_CASE=" + str(case_id), "-no_save", "-assert",
                     "report=" + str(report), "-ucli", "-i", str(UCLI)],
                    cwd=axis_dir,
                    env=clean_env({"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
                                   "VCS_ARCH_OVERRIDE": "linux",
                                   "M1684_SAIF_FILE": str(saif)}),
                    timeout=1200, output=log)
                check = candidate / (axis + "_case" + str(case_id) + ".saif_check.json")
                run([str(PYTHON), "-I", str(CHECKER), "--mode", "saif",
                     "--axis", axis, "--case", str(case_id),
                     "--cycles", str(CYCLES[axis][case_id]),
                     "--saif", str(saif), "--log", str(log)],
                    cwd=HW, env=clean_env({}), timeout=180, output=check)
                checked = strict_json(check)
                if (checked.get("status") !=
                        "PASS_M1684_BINARY_CLEAN_DUT_ONLY_PRODUCTION_SAIF"
                        or checked.get("accepted_sources") != EVENTS[case_id]):
                    raise Failure("mapped production/SAIF check failed")
                state["saif_files"] += 1

        if any(state[key] != COUNTS[key]
               for key in ("vcs_compiles", "simv_runs", "saif_files")):
            raise Failure("mapped VCS/SAIF campaign incomplete before PTPX")

        metric_rows = []
        for axis in ("k8", "k1x8"):
            for case_id in range(5):
                state["phase"] = "PTPX_" + axis + "_" + str(case_id)
                state["ptpx_runs"] += 1
                candidate = WORK / "candidate"
                saif = candidate / (axis + "_case" + str(case_id) + ".saif")
                pt_dir = candidate / (axis + "_case" + str(case_id) + ".ptpx")
                pt_dir.mkdir()
                run([str(PT), "-f", str(PT_TCL)], cwd=HW,
                    env=clean_env({
                        "DESIGN_NAME": DESIGN, "TT_LIB_DB": str(TT_DB),
                        "SDC_LIB_DB": str(SS_DB),
                        "MAPPED_NETLIST": str(BASE / axis / NET_REL),
                        "MAPPED_SDC": str(BASE / axis / SDC_REL),
                        "GATE_SAIF_FILE": str(saif),
                        "OUTPUT_DIR": str(pt_dir), "SAIF_INSTANCE": SAIF_SCOPE,
                        "SAIF_DURATION_NS": str(CYCLES[axis][case_id] * 3),
                        "MEASUREMENT_CYCLES": str(CYCLES[axis][case_id]),
                        "ACCEPTED_SOURCES": str(EVENTS[case_id]),
                        "AXIS": axis, "CASE_ID": str(case_id),
                        "FAULT_BINARY_CLEAN": "true",
                        "REGISTERED_FAULT_PUBLIC_ZERO": "true"}),
                    timeout=3600, output=pt_dir / "ptpx.log")
                marker = pt_dir / "PTPX_INTERNAL_COMPLETE.txt"
                if (not marker.is_file()
                        or "PASS_M1684_C2_M1609_FRESH_MAPPED_PRODUCTION_PTPX"
                        not in marker.read_text()):
                    raise Failure("PTPX completion marker absent")
                report = pt_dir / "reports/ptpx_power.rpt"
                power_check = pt_dir / "power_check.json"
                run([str(PYTHON), "-I", str(CHECKER), "--mode", "power",
                    "--power-report", str(report)], cwd=HW,
                    env=clean_env({}), timeout=120, output=power_check)
                power = strict_json(power_check)
                metric_rows.append({"axis": axis, "case": case_id,
                                    "cycles": CYCLES[axis][case_id],
                                    "accepted_sources": EVENTS[case_id], **power})

        if any(state[key] != value for key, value in COUNTS.items()):
            raise Failure("execution-count drift")
        metrics = OLD.CHECK.aggregate_metrics(metric_rows)
        metrics.update({
            "schema": "m1730_m1715_c2_vcs_proxy_repair_production_energy_metrics_r1_v1",
            "status": "CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER",
            "scope": "fresh_M1661_M1609_logic_only_premacro_TT0p9V25C_equal_bandwidth_five_directed_cases",
            "clock_period_ns": 3.0, "axes": metrics["axes"],
            "case_rows": metric_rows})

        state["phase"] = "SUCCESS_STAGE"
        STAGE.mkdir()
        shutil.copytree(WORK / "candidate", STAGE / "candidate")
        for axis in ("k8", "k1x8"):
            shutil.copy2(WORK / "build" / axis / "compile.log",
                         STAGE / (axis + ".compile.log"))
        write_json(STAGE / "metrics.json", metrics)
        write_json(STAGE / "receipt.json", {
            "schema": "m1730_m1715_c2_vcs_proxy_repair_production_energy_candidate_receipt_r1_v1",
            "status": "PASS_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "identity": result_identity(), "one_shot": {
                "attempt_consumed": True, **COUNTS, "automatic_retry": False},
            "axes": ["k8", "k1x8"], "cases_per_axis": 5,
            "same_workloads_clock_and_measurement_window": True,
            "accepted_sources_per_axis": sum(EVENTS),
            "fault_binary_clean_required": True,
            "registered_fault_public_zero_required": True,
            "compile_proxy": {"vcs_compile_only": True,
                              "exact_launch_tuple_required": True,
                              "tcp_preflight_before_attempt": True,
                              "not_forwarded_to_simv_checker_or_ptpx": True},
            "predecessor_failure": {
                "m1715_attempt_consumed": True,
                "m1715_execution_counts": {"vcs_compiles": 1,
                    "simv_runs": 0, "saif_files": 0, "ptpx_runs": 0},
                "m1715_retry_forbidden": True,
                "private_forensic_tree_citable": False},
            "claim_boundary": SOURCE_CLAIMS})
        (STAGE / "RUN_COMPLETE.txt").write_text(
            "PASS_M1730_M1715_C2_VCS_PROXY_REPAIR_PRODUCTION_ENERGY_CANDIDATE_PENDING_RESULT_HAMMER\n")
        seal_dir(STAGE)
        publish_no_replace(WORK, PRIVATE)
        publish_no_replace(STAGE, RESULT)
        state["complete"] = True
        print("PASS_M1730_M1715_C2_VCS_PROXY_REPAIR_PRODUCTION_ENERGY_CANDIDATE_PENDING_RESULT_HAMMER")
        return 0
    except BaseException as error:
        if not state["complete"]:
            try:
                FAIL_STAGE.mkdir(exist_ok=False)
                write_json(FAIL_STAGE / "failure.json", {
                    "status": "FAILED_OR_INCOMPLETE", "phase": state["phase"],
                    "error": type(error).__name__,
                    "attempt_consumed": state["attempt"],
                    "counts": {key: state[key] for key in COUNTS},
                    "automatic_retry": False, "canonical_result": False,
                    "partial_axis_citable": False})
                seal_dir(FAIL_STAGE)
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
