#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Source-only M1753 three-axis C2 mapped directed-component energy run.

This successor does not inherit the drifted, unreleased M1730 identity.  It
binds the sealed terminal M1715 failure and the admitted M1661/M1677 K1, K8,
and equal-bandwidth K1x8 mapped tops.  The same five frozen directed cases and
3 ns clock are replayed through public DUT ports.  SAIF is DUT-only and every
PrimeTime invocation reports the whole mapped component.  These 261 accepted
sources per axis are directed component evidence, never production/trace/system
energy.  No execution is authorized without an independent M1760 review and
an exact M1761 release.
"""
from __future__ import annotations

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
import subprocess
import sys
from typing import Any


HW = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
CHECKER = HW / "system_simulator/scripts/check_m1753_m1715_c2_three_axis_mapped_directed_component_energy_source.py"
TEST = HW / "system_simulator/tests/test_m1753_m1715_c2_three_axis_mapped_directed_component_energy_source.py"
CONTRACT = HW / "contracts/m1753_m1715_c2_three_axis_mapped_directed_component_energy_source_contract_r1_20260901.json"
M1760 = HW / "reviews/m1760_m1753_c2_three_axis_mapped_directed_component_energy_source_hammer_r1_20260901"
M1761 = HW / "contracts/m1761_m1760_m1753_c2_three_axis_mapped_directed_component_energy_launch_release_r1_20260901.json"

M1715_RUNNER = HW / "dc_handoff/scripts/run_m1715_m1710_m1684_m1661_c2_queue_order_repair_production_energy_one_shot.py"
M1715_ATTEMPT = HW / "results/.m1715_c2_queue_order_repair_production_energy_attempt_consumed"
M1715_FAILURE = HW / "results/m1715_c2_queue_order_repair_production_energy_r1_20260901.failed_or_incomplete.quarantine"
SPEC = importlib.util.spec_from_file_location("m1715_for_m1753", M1715_RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("M1715 import unavailable")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

CHECK_SPEC = importlib.util.spec_from_file_location("m1753_checker", CHECKER)
if CHECK_SPEC is None or CHECK_SPEC.loader is None:
    raise RuntimeError("M1753 checker import unavailable")
CHECK = importlib.util.module_from_spec(CHECK_SPEC)
CHECK_SPEC.loader.exec_module(CHECK)

DC_BASE = HW / "dc_handoff/runs/m1661_m1652_c2_resource_gate_successor_three_axis_logic_only_dc_3p000ns_r1_20260901"
DESIGN = BASE.DESIGN
NET_REL = BASE.NET_REL
SDC_REL = BASE.SDC_REL
TOP = BASE.TOP
SAIF_SCOPE = BASE.SAIF_SCOPE
CELL = BASE.CELL
TT_DB = BASE.TT_DB
SS_DB = BASE.SS_DB
VCS = BASE.VCS
PT = BASE.PT
LMUTIL = BASE.LMUTIL
PYTHON = BASE.PYTHON
LICENSE_SERVER = BASE.LICENSE_SERVER
LICENSE_FILE = BASE.LICENSE_FILE
LOCK = BASE.LOCK
UCLI = BASE.UCLI
PT_TCL = BASE.PT_TCL

FILELISTS = {
    "k1": HW / "dc_handoff/filelists/iscas_m1753_c2_m1609_k1_mapped_directed_energy.f",
    "k8": BASE.FILELISTS["k8"],
    "k1x8": BASE.FILELISTS["k1x8"],
}
CYCLES = {
    "k1": [259, 737, 3153, 7569, 14],
    "k8": [51, 131, 486, 1231, 14],
    "k1x8": [53, 133, 499, 1246, 14],
}
EVENTS = [20, 41, 90, 110, 0]
AREAS_UM2 = {
    "k1": 124546.967176,
    "k8": 130476.905184,
    "k1x8": 585534.971643,
}
AXES = ("k1", "k8", "k1x8")
CASES = tuple(range(5))
COUNTS = {"vcs_compiles": 3, "simv_runs": 15,
          "saif_files": 15, "ptpx_runs": 15}
CLAIMS = dict((key, False) for key in (
    "vcs", "mapped_functionality", "production_saif", "ptpx", "power",
    "energy", "performance", "system_speedup", "paper_ppa_ready", "headline"))

ATTEMPT = HW / "results/.m1753_c2_three_axis_mapped_directed_component_energy_attempt_consumed"
RESULT = HW / "results/m1753_c2_three_axis_mapped_directed_component_energy_r1_20260901"
FAILURE = HW / "results/m1753_c2_three_axis_mapped_directed_component_energy_r1_20260901.failed_or_incomplete.quarantine"
PRIVATE = HW / "results/m1753_c2_three_axis_mapped_directed_component_energy_r1_20260901.private_build.unsealed_do_not_cite"
WORK = HW / ("results/.m1753_c2_three_axis_mapped_directed_component_energy_work." + str(os.getpid()))
STAGE = HW / ("results/.m1753_c2_three_axis_mapped_directed_component_energy_stage." + str(os.getpid()))
FAIL_STAGE = HW / ("results/.m1753_c2_three_axis_mapped_directed_component_energy_failure_stage." + str(os.getpid()))

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


class Failure(RuntimeError):
    pass


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise Failure("duplicate JSON key")
            value[key] = item
        return value
    if not path.is_file() or path.is_symlink():
        raise Failure("JSON is not regular: " + str(path))
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           Failure("nonfinite JSON: " + token)))
    if type(value) is not dict:
        raise Failure("JSON root")
    return value


def exact(path: Path, digest: str) -> None:
    if not path.is_file() or path.is_symlink() or sha(path) != digest:
        raise Failure("exact identity drift: " + str(path))


def verify_seal(root: Path, manifest_sha: str, outer_sha: str) -> None:
    BASE.verify_seal(root, manifest_sha, outer_sha)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def source_mapping(contract: dict[str, Any]) -> dict[str, str]:
    rows = contract.get("execution_files")
    if not isinstance(rows, list):
        raise Failure("execution inventory absent")
    mapping: dict[str, str] = {}
    for row in rows:
        if (type(row) is not dict or set(row) != {"path", "sha256"}
                or row["path"] in mapping
                or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None):
            raise Failure("execution inventory malformed")
        mapping[row["path"]] = row["sha256"]
    return mapping


def verify_execution_sources() -> dict[str, Any]:
    contract = strict_json(CONTRACT)
    if (contract.get("schema") !=
            "m1753_m1715_c2_three_axis_mapped_directed_component_energy_source_contract_r1_v1"
            or contract.get("status") !=
            "SOURCE_ONLY__M1760_REVIEW_AND_M1761_RELEASE_REQUIRED__NO_EDA"
            or contract.get("claim_boundary") != CLAIMS):
        raise Failure("M1753 source contract semantics")
    mapping = source_mapping(contract)
    required = {RUNNER, CHECKER, TEST, UCLI, PT_TCL, *FILELISTS.values()}
    for path in required:
        rel = path.relative_to(HW).as_posix()
        if mapping.get(rel) is None:
            raise Failure("execution source absent: " + rel)
        exact(path, mapping[rel])
        if path.suffix in {".py", ".sv", ".tcl", ".f"}:
            text = path.read_text().lower()
            if "init" + "reg" in text:
                raise Failure("forbidden initialization token: " + rel)
            if BASE.CHECK.active_force_present(path):
                raise Failure("active force: " + rel)
    for axis in AXES:
        for relkey in ("netlist", "sdc"):
            row = contract["mapped_axes"][axis][relkey]
            path = HW / row["path"]
            exact(path, row["sha256"])
    return contract


def verify_authority() -> dict[str, Any]:
    pins = {name: os.environ.get(name, "") for name in (
        "M1753_EXPECTED_RUNNER_SHA256", "M1753_EXPECTED_SOURCE_CONTRACT_SHA256",
        "M1753_EXPECTED_M1760_REVIEW_SHA256", "M1753_EXPECTED_M1760_MANIFEST_SHA256",
        "M1753_EXPECTED_M1760_OUTER_FILE_SHA256", "M1753_EXPECTED_M1761_RELEASE_SHA256")}
    if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in pins.values()):
        raise Failure("fresh M1760/M1761 exact authority absent")
    exact(RUNNER, pins["M1753_EXPECTED_RUNNER_SHA256"])
    exact(CONTRACT, pins["M1753_EXPECTED_SOURCE_CONTRACT_SHA256"])
    contract = verify_execution_sources()
    verify_seal(M1760, pins["M1753_EXPECTED_M1760_MANIFEST_SHA256"],
                pins["M1753_EXPECTED_M1760_OUTER_FILE_SHA256"])
    exact(M1760 / "review.json", pins["M1753_EXPECTED_M1760_REVIEW_SHA256"])
    exact(M1761, pins["M1753_EXPECTED_M1761_RELEASE_SHA256"])
    release_sum = Path(str(M1761) + ".sha256")
    release_outer = Path(str(M1761) + ".sha256.seal.sha256")
    if (release_sum.read_text() != sha(M1761) + "  " + M1761.name + "\n"
            or release_outer.read_text() != sha(release_sum) + "  " + release_sum.name + "\n"):
        raise Failure("M1761 double seal drift")
    review = strict_json(M1760 / "review.json")
    release = strict_json(M1761)
    if (review.get("status") !=
            "PASS_M1760_M1753_C2_THREE_AXIS_MAPPED_DIRECTED_COMPONENT_ENERGY_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_ATTEMPT"
            or release.get("status") !=
            "AUTHORIZE_ONE_M1753_C2_THREE_AXIS_MAPPED_DIRECTED_COMPONENT_ENERGY_ATTEMPT"
            or release.get("authorization") != {
                "future_m1753_attempts": 1, "automatic_retry": False, **COUNTS}
            or release.get("identity") != {
                "runner_sha256": sha(RUNNER),
                "source_contract_sha256": sha(CONTRACT),
                "m1760_review_sha256": sha(M1760 / "review.json")}
            or release.get("claim_boundary") != CLAIMS):
        raise Failure("M1760/M1761 authority semantics")
    return contract


def verify_predecessors() -> None:
    exact(M1715_RUNNER, "b5010f8d3d70ea2e029a3636b7b338d61b3b6f02ae8fa130dfbe279e736a2225")
    BASE.verify_predecessors_and_inputs()
    verify_seal(M1715_ATTEMPT,
                "18657601c99e8eb4141e3b175976a84bb491e04d7d1e85c46e82d6b463f71cf3",
                "ab8f355009cb82e9c1211ff39a92295fcbe776ac26c856a804c2e52b903b7835")
    verify_seal(M1715_FAILURE,
                "2d2a24c7499c31a2667cc097037ffa3fa9d270606f104b3d85aac145712d49fe",
                "d09f54d3acc5f1dba9f0cc0387b9b003bf3a9ff136ff804f2ae198055cd8f971")
    exact(M1715_ATTEMPT / "attempt.json",
          "9cdfebe0acd0bd81d1ff92f7423020b336cbe4c7ebc0dbcb0118d7ceab068c53")
    exact(M1715_FAILURE / "failure.json",
          "e9e09df40e1e1b6e02064150b7d8752d2303bb1edabddc3f38085021cc6f3c02")
    attempted = strict_json(M1715_ATTEMPT / "attempt.json")
    failed = strict_json(M1715_FAILURE / "failure.json")
    if (attempted.get("status") !=
            "M1715_C2_QUEUE_ORDER_REPAIR_PRODUCTION_ENERGY_ATTEMPT_CONSUMED"
            or attempted.get("budget") != {"vcs_compiles": 2, "simv_runs": 10,
                                            "saif_files": 10, "ptpx_runs": 10}
            or attempted.get("automatic_retry") is not False
            or failed != {"attempt_consumed": True, "automatic_retry": False,
                "canonical_result": False,
                "counts": {"ptpx_runs": 0, "saif_files": 0,
                           "simv_runs": 0, "vcs_compiles": 1},
                "error": "KeyboardInterrupt", "partial_axis_citable": False,
                "phase": "COMPILE_k8", "status": "FAILED_OR_INCOMPLETE"}):
        raise Failure("M1715 sealed terminal failure semantics")
    if os.path.lexists(HW / "results/m1715_c2_queue_order_repair_production_energy_r1_20260901"):
        raise Failure("M1715 result unexpectedly exists")


def namespaces_fresh() -> None:
    for path in (ATTEMPT, RESULT, FAILURE, PRIVATE, WORK, STAGE, FAIL_STAGE):
        if os.path.lexists(path):
            raise Failure("namespace residue: " + str(path))
    for pattern in (".m1753_c2_three_axis_mapped_directed_component_energy_work.*",
                    ".m1753_c2_three_axis_mapped_directed_component_energy_stage.*",
                    ".m1753_c2_three_axis_mapped_directed_component_energy_failure_stage.*"):
        if next((HW / "results").glob(pattern), None) is not None:
            raise Failure("private namespace residue: " + pattern)


def capture_compile_proxy() -> dict[str, str]:
    captured = {name: os.environ.get(name, "") for name in PROXY_KEYS}
    if captured != EXPECTED_PROXY:
        raise Failure("compile-only proxy identity drift")
    try:
        connection = socket.create_connection(("127.0.0.1", 7897), timeout=2.0)
        connection.close()
    except OSError as error:
        raise Failure("VCS compile proxy preflight failed") from error
    return captured


def clean_env(extra: dict[str, str], proxy: dict[str, str] | None = None) -> dict[str, str]:
    value = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
             "SNPSLMD_LICENSE_FILE": LICENSE_SERVER, "LM_LICENSE_FILE": str(LICENSE_FILE)}
    value.update(extra)
    if proxy is not None:
        if proxy != EXPECTED_PROXY:
            raise Failure("non-exact proxy rejected")
        value.update(proxy)
    return value


def run(command: list[str], *, cwd: Path, env: dict[str, str], timeout: int,
        output: Path) -> None:
    verify_execution_sources()
    if Path(command[0]).name in {"vcs", "pt_shell"}:
        BASE.collision_gate()
    with output.open("wb") as stream:
        completed = subprocess.run(command, cwd=cwd, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   timeout=timeout, check=False)
    if completed.returncode != 0:
        raise Failure("tool failure: " + " ".join(command[:2]))


def identity() -> dict[str, str]:
    return {"runner_sha256": sha(RUNNER),
            "source_contract_sha256": sha(CONTRACT),
            "m1760_review_sha256": sha(M1760 / "review.json"),
            "m1761_release_sha256": sha(M1761),
            "m1715_runner_sha256": sha(M1715_RUNNER),
            "m1661_manifest_sha256": BASE.STATIC_SHA["m1661_manifest"]}


def main() -> int:
    if len(sys.argv) != 1:
        raise Failure("M1753 accepts no arguments")
    state: dict[str, Any] = {"phase": "SOURCE_CHAIN", "attempt": False,
                             "complete": False, **dict((key, 0) for key in COUNTS)}
    lock_handle = LOCK.open("a+")
    try:
        verify_authority()
        verify_predecessors()
        namespaces_fresh()
        state["phase"] = "QUEUE_WAIT"
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        BASE.collision_gate()
        BASE.resource_gate()
        verify_execution_sources()
        namespaces_fresh()
        state["phase"] = "LICENSE_PREFLIGHT"
        license_check = subprocess.run(
            [str(LMUTIL), "lmstat", "-a", "-c", LICENSE_SERVER],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=clean_env({}), timeout=60, check=False)
        if license_check.returncode != 0:
            raise Failure("license preflight failed")
        state["phase"] = "PROXY_PREFLIGHT"
        proxy = capture_compile_proxy()
        verify_execution_sources()
        namespaces_fresh()
        state["phase"] = "ATTEMPT_CONSUME"
        ATTEMPT.mkdir()
        state["attempt"] = True
        write_json(ATTEMPT / "attempt.json", {
            "status": "M1753_C2_THREE_AXIS_MAPPED_DIRECTED_COMPONENT_ENERGY_ATTEMPT_CONSUMED",
            "identity": identity(), "budget": COUNTS, "axes": list(AXES),
            "cases": list(CASES), "workload_class": "DIRECTED_COMPONENT_NOT_PRODUCTION",
            "accepted_sources_per_axis": sum(EVENTS), "automatic_retry": False})
        BASE.seal_dir(ATTEMPT)
        WORK.mkdir()
        (WORK / "build").mkdir()
        (WORK / "candidate").mkdir()

        for axis in AXES:
            state["phase"] = "COMPILE_" + axis
            axis_dir = WORK / "build" / axis
            axis_dir.mkdir()
            state["vcs_compiles"] += 1
            run([str(VCS), "-full64", "-sverilog", "+v2k", "-timescale=1ns/1ps",
                 "-assert", "svaext", "-debug_access+r", "-lca", "+vcs+lic+wait",
                 "-Mdir=csrc", "-f", str(FILELISTS[axis]), "-top", TOP, "-o", "simv"],
                cwd=axis_dir,
                env=clean_env({"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
                               "VCS_ARCH_OVERRIDE": "linux"}, proxy),
                timeout=7200, output=axis_dir / "compile.log")
            if not (axis_dir / "simv").is_file():
                raise Failure("fresh simv absent: " + axis)
            for case_id in CASES:
                state["phase"] = "SIM_" + axis + "_" + str(case_id)
                state["simv_runs"] += 1
                candidate = WORK / "candidate"
                saif = candidate / f"{axis}_case{case_id}.saif"
                log = candidate / f"{axis}_case{case_id}.log"
                assertion = candidate / f"{axis}_case{case_id}.assert.report"
                run(["./simv", "-lca", "+M979_UCLI_SAIF", f"+M979_CASE={case_id}",
                     "-no_save", "-assert", "report=" + str(assertion),
                     "-ucli", "-i", str(UCLI)], cwd=axis_dir,
                    env=clean_env({"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
                                   "VCS_ARCH_OVERRIDE": "linux",
                                   "M1684_SAIF_FILE": str(saif)}),
                    timeout=1800, output=log)
                check = candidate / f"{axis}_case{case_id}.saif_check.json"
                run([str(PYTHON), "-I", str(CHECKER), "--mode", "saif",
                     "--axis", axis, "--case", str(case_id), "--cycles",
                     str(CYCLES[axis][case_id]), "--saif", str(saif), "--log", str(log)],
                    cwd=HW, env=clean_env({}), timeout=240, output=check)
                checked = strict_json(check)
                if (checked.get("status") != "PASS_M1753_DIRECTED_COMPONENT_DUT_ONLY_SAIF"
                        or checked.get("accepted_sources") != EVENTS[case_id]):
                    raise Failure("mapped directed SAIF check failed")
                state["saif_files"] += 1
        if any(state[key] != COUNTS[key] for key in ("vcs_compiles", "simv_runs", "saif_files")):
            raise Failure("all fifteen SAIF coordinates required before PTPX")

        rows = []
        for axis in AXES:
            for case_id in CASES:
                state["phase"] = "PTPX_" + axis + "_" + str(case_id)
                state["ptpx_runs"] += 1
                candidate = WORK / "candidate"
                saif = candidate / f"{axis}_case{case_id}.saif"
                pt_dir = candidate / f"{axis}_case{case_id}.ptpx"
                pt_dir.mkdir()
                run([str(PT), "-f", str(PT_TCL)], cwd=HW,
                    env=clean_env({"DESIGN_NAME": DESIGN, "TT_LIB_DB": str(TT_DB),
                        "SDC_LIB_DB": str(SS_DB),
                        "MAPPED_NETLIST": str(DC_BASE / axis / NET_REL),
                        "MAPPED_SDC": str(DC_BASE / axis / SDC_REL),
                        "GATE_SAIF_FILE": str(saif), "OUTPUT_DIR": str(pt_dir),
                        "SAIF_INSTANCE": SAIF_SCOPE,
                        "SAIF_DURATION_NS": str(CYCLES[axis][case_id] * 3),
                        "MEASUREMENT_CYCLES": str(CYCLES[axis][case_id]),
                        "ACCEPTED_SOURCES": str(EVENTS[case_id]),
                        "AXIS": axis, "CASE_ID": str(case_id),
                        "FAULT_BINARY_CLEAN": "true",
                        "REGISTERED_FAULT_PUBLIC_ZERO": "true"}),
                    timeout=5400, output=pt_dir / "ptpx.log")
                marker = pt_dir / "PTPX_INTERNAL_COMPLETE.txt"
                if (not marker.is_file() or
                        "PASS_M1684_C2_M1609_FRESH_MAPPED_PRODUCTION_PTPX" not in marker.read_text()):
                    raise Failure("PTPX completion marker absent")
                power_check = pt_dir / "power_check.json"
                run([str(PYTHON), "-I", str(CHECKER), "--mode", "power",
                     "--power-report", str(pt_dir / "reports/ptpx_power.rpt")],
                    cwd=HW, env=clean_env({}), timeout=120, output=power_check)
                power = strict_json(power_check)
                rows.append({"axis": axis, "case": case_id,
                             "cycles": CYCLES[axis][case_id],
                             "accepted_sources": EVENTS[case_id], **power})
        if any(state[key] != value for key, value in COUNTS.items()):
            raise Failure("execution count drift")
        metrics = CHECK.aggregate_metrics(rows)
        metrics.update({"schema": "m1753_c2_three_axis_mapped_directed_component_energy_metrics_r1_v1",
                        "status": "CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER",
                        "workload_class": "DIRECTED_COMPONENT_NOT_PRODUCTION",
                        "scope": "whole_mapped_C2_component_logic_only_premacro_TT0p9V25C",
                        "clock_period_ns": 3.0, "case_rows": rows,
                        "external_exclusions": ["weight_sram", "testbench_memory_model",
                            "io_phy", "clock_tree", "postlayout_parasitics"]})
        STAGE.mkdir()
        shutil.copytree(WORK / "candidate", STAGE / "candidate")
        for axis in AXES:
            shutil.copy2(WORK / "build" / axis / "compile.log", STAGE / f"{axis}.compile.log")
        write_json(STAGE / "metrics.json", metrics)
        write_json(STAGE / "receipt.json", {
            "schema": "m1753_c2_three_axis_mapped_directed_component_energy_candidate_receipt_r1_v1",
            "status": "PASS_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "identity": identity(), "one_shot": {"attempt_consumed": True,
                **COUNTS, "automatic_retry": False}, "axes": list(AXES),
            "cases_per_axis": 5, "workload_class": "DIRECTED_COMPONENT_NOT_PRODUCTION",
            "same_workloads_and_clock": True, "whole_component_report_power": True,
            "equal_bandwidth_joint_disclosure_required": {
                "cycle_speedup_k8_vs_k1x8": 1945.0 / 1913.0,
                "throughput_per_mm2_k8_vs_k1x8":
                    (1945.0 * AREAS_UM2["k1x8"]) /
                    (1913.0 * AREAS_UM2["k8"]),
                "must_be_same_table_and_sentence": True},
            "claim_boundary": CLAIMS})
        (STAGE / "RUN_COMPLETE.txt").write_text(
            "PASS_M1753_C2_THREE_AXIS_MAPPED_DIRECTED_COMPONENT_ENERGY_CANDIDATE_PENDING_RESULT_HAMMER\n")
        BASE.seal_dir(STAGE)
        BASE.publish_no_replace(WORK, PRIVATE)
        BASE.publish_no_replace(STAGE, RESULT)
        state["complete"] = True
        print("PASS_M1753_C2_THREE_AXIS_MAPPED_DIRECTED_COMPONENT_ENERGY_CANDIDATE_PENDING_RESULT_HAMMER")
        return 0
    except BaseException as error:
        if not state["complete"]:
            try:
                FAIL_STAGE.mkdir(exist_ok=False)
                write_json(FAIL_STAGE / "failure.json", {
                    "status": "FAILED_OR_INCOMPLETE", "phase": state["phase"],
                    "error": type(error).__name__, "attempt_consumed": state["attempt"],
                    "counts": {key: state[key] for key in COUNTS},
                    "automatic_retry": False, "canonical_result": False,
                    "partial_axis_citable": False})
                BASE.seal_dir(FAIL_STAGE)
                BASE.publish_no_replace(FAIL_STAGE, FAILURE)
            except BaseException:
                pass
            if WORK.is_dir() and not PRIVATE.exists():
                try:
                    BASE.publish_no_replace(WORK, PRIVATE)
                except BaseException:
                    pass
        raise
    finally:
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
