#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""One-shot fresh M1798 C3 mapped-gate SAIF/PTPX campaign.

This additive successor is inert until a different-author M1799 source hammer
and a semantically bound, double-sealed M1800 release are supplied through
exact external SHA256 pins.  There is no retry and no reuse of any old EDA
artifact.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


HW = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
CHECKER = HW / "system_simulator/scripts/check_m1798_c3_m1454_fixed_t10_mapped_energy_source.py"
CHECK_SPEC = importlib.util.spec_from_file_location("m1798_checker", str(CHECKER))
if CHECK_SPEC is None or CHECK_SPEC.loader is None:
    raise RuntimeError("M1798 checker unavailable")
CHECK = importlib.util.module_from_spec(CHECK_SPEC)
CHECK_SPEC.loader.exec_module(CHECK)

BASE_RUNNER = HW / "dc_handoff/scripts/run_m1790_c3_m1454_fixed_t10_mapped_energy_one_shot.py"
BASE_SPEC = importlib.util.spec_from_file_location("m1790_runner_for_m1798",
                                                   str(BASE_RUNNER))
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise RuntimeError("M1790 runner helpers unavailable")
BASE = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(BASE)

CONTRACT = CHECK.CONTRACT
CONTRACT_SIDECAR = Path(str(CONTRACT) + ".sha256")
CONTRACT_OUTER = Path(str(CONTRACT) + ".sha256.seal.sha256")
NET = CHECK.NET
SDC = CHECK.SDC
FILELIST = CHECK.FILELIST
UCLI = CHECK.UCLI
PT_TCL = CHECK.PT_TCL
TT_DB = CHECK.TT_DB
TOP = CHECK.TOP
SAIF_SCOPE = CHECK.SAIF_SCOPE
DOC359 = CHECK.DOC359

M1799 = HW / "reviews/m1799_m1798_c3_m1454_fixed_t10_mapped_energy_source_hammer_r1_20260902"
M1800 = HW / "contracts/m1800_m1799_m1798_c3_m1454_fixed_t10_mapped_energy_launch_release_r1_20260902.json"
M1800_SIDECAR = Path(str(M1800) + ".sha256")
M1800_OUTER = Path(str(M1800) + ".sha256.seal.sha256")

VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
PT = Path("/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
LICENSE_SERVER = "27030@ic.ismd-nemo"
LICENSE_FILE = Path("/opt/synopsys/Synopsys.dat")

ATTEMPT = HW / "results/.m1798_c3_mapped_energy_attempt_consumed"
RESULT = HW / "results/m1798_c3_mapped_energy_r1_20260902"
FAILURE = HW / "results/m1798_c3_mapped_energy_r1_20260902.failed_or_incomplete.quarantine"
PRIVATE = HW / "results/m1798_c3_mapped_energy_r1_20260902.private_build.unsealed_do_not_cite"
WORK = HW / ("results/.m1798_c3_mapped_energy_work." + str(os.getpid()))
STAGE = HW / ("results/.m1798_c3_mapped_energy_stage." + str(os.getpid()))
FAIL_STAGE = HW / ("results/.m1798_c3_mapped_energy_failure_stage." + str(os.getpid()))
LOCK = Path("/tmp/m1798_c3_mapped_energy.lock")
SHARED_QUEUE = Path("/tmp/date_dual_synopsys_same_uid_eda_queue.lock")
COUNTS = {"vcs_compiles": 1, "simv_runs": 1, "saif_files": 1,
          "ptpx_runs": 1}

RELEASE_BOUNDARY = {
    "mapped_public_port_only": True,
    "directed_component_only": True,
    "checkpoint_capture": False,
    "timing_simulation": False,
    "macro_count": 0,
    "prelayout_logic_only": True,
}
ATTEMPT_UNIQUENESS = {
    "attempt_latch": str(ATTEMPT.relative_to(HW)),
    "canonical_result": str(RESULT.relative_to(HW)),
    "failure_result": str(FAILURE.relative_to(HW)),
    "private_build": str(PRIVATE.relative_to(HW)),
    "prelaunch_namespaces_required_absent": True,
    "no_replace_atomic_publish": True,
    "automatic_retry": False,
}


class Failure(RuntimeError):
    pass


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(path, digest):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or sha(path) != digest:
        raise Failure("identity drift " + str(path))


def strict_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise Failure("duplicate JSON key " + key)
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           Failure("nonfinite JSON " + token)))
    if type(value) is not dict:
        raise Failure("JSON root")
    return value


def authority_pin(name):
    value = os.environ.get(name, "")
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise Failure("exact authority absent " + name)
    return value


def verify_file_double_seal(path, sidecar, outer, file_sha, sidecar_sha,
                            outer_sha):
    exact(path, file_sha)
    exact(sidecar, sidecar_sha)
    exact(outer, outer_sha)
    if sidecar.read_text().split() != [sha(path), path.name]:
        raise Failure("release sidecar content")
    if outer.read_text().split() != [sha(sidecar), sidecar.name]:
        raise Failure("release outer seal content")


def verify_contract_double_seal():
    for path in (CONTRACT, CONTRACT_SIDECAR, CONTRACT_OUTER):
        if not path.is_file() or path.is_symlink():
            raise Failure("contract seal input")
    if CONTRACT_SIDECAR.read_text().split() != [sha(CONTRACT), CONTRACT.name]:
        raise Failure("contract sidecar content")
    if CONTRACT_OUTER.read_text().split() != [sha(CONTRACT_SIDECAR),
                                               CONTRACT_SIDECAR.name]:
        raise Failure("contract outer seal content")


def verify_review_member(root):
    mapping = {}
    for row in (Path(root) / "SHA256SUMS").read_text().splitlines():
        fields = row.split(maxsplit=1)
        if len(fields) != 2:
            raise Failure("review manifest syntax")
        name = Path(fields[1].lstrip("*")).as_posix()
        if name in mapping:
            raise Failure("review manifest duplicate")
        mapping[name] = fields[0]
    if mapping.get("review.json") != sha(Path(root) / "review.json"):
        raise Failure("review.json is not transitively sealed")


def verify_authority():
    exact(RUNNER, authority_pin("M1798_EXPECTED_RUNNER_SHA256"))
    exact(CONTRACT, authority_pin("M1798_EXPECTED_SOURCE_CONTRACT_SHA256"))
    verify_contract_double_seal()
    BASE.verify_seal(
        M1799,
        authority_pin("M1798_EXPECTED_M1799_MANIFEST_SHA256"),
        authority_pin("M1798_EXPECTED_M1799_OUTER_FILE_SHA256"))
    verify_review_member(M1799)
    exact(M1799 / "review.json",
          authority_pin("M1798_EXPECTED_M1799_REVIEW_SHA256"))
    verify_file_double_seal(M1800, M1800_SIDECAR, M1800_OUTER,
        authority_pin("M1798_EXPECTED_M1800_RELEASE_SHA256"),
        authority_pin("M1798_EXPECTED_M1800_SIDECAR_SHA256"),
        authority_pin("M1798_EXPECTED_M1800_OUTER_FILE_SHA256"))

    review = strict_json(M1799 / "review.json")
    release = strict_json(M1800)
    if review.get("status") != "PASS_M1799_M1798_C3_MAPPED_ENERGY_SOURCE_HAMMER__AUTHORIZE_ONE_FRESH_M1798_CAMPAIGN":
        raise Failure("M1799 status")
    if review.get("severity_counts") != {"p0": 0, "p1": 0, "p2": 0}:
        raise Failure("M1799 severity")
    if release.get("schema") != "m1800_m1799_m1798_c3_m1454_fixed_t10_mapped_energy_launch_release_r1_v1":
        raise Failure("M1800 schema")
    if release.get("status") != "AUTHORIZE_ONE_FRESH_M1798_C3_MAPPED_ENERGY_CAMPAIGN":
        raise Failure("M1800 status")

    expected_identity = {
        "runner_sha256": sha(RUNNER),
        "source_contract_sha256": sha(CONTRACT),
        "source_contract_sidecar_sha256": sha(CONTRACT_SIDECAR),
        "source_contract_outer_file_sha256": sha(CONTRACT_OUTER),
        "source_review_json_sha256": sha(M1799 / "review.json"),
        "source_review_manifest_sha256": sha(M1799 / "SHA256SUMS"),
        "source_review_outer_file_sha256": sha(
            M1799 / "SHA256SUMS.seal.sha256"),
        "docs359_sha256": sha(DOC359),
    }
    if release.get("identity") != expected_identity:
        raise Failure("M1800 transitive identity")
    if release.get("prelaunch_claim_boundary") != CHECK.CLAIMS:
        raise Failure("M1800 prelaunch claim boundary")
    if release.get("measurement_boundary") != RELEASE_BOUNDARY:
        raise Failure("M1800 measurement boundary")
    if release.get("attempt_uniqueness") != ATTEMPT_UNIQUENESS:
        raise Failure("M1800 attempt uniqueness")
    if release.get("fresh_execution_budget") != dict(
            COUNTS, automatic_retry=False,
            reuse_prior_simv_saif_ptpx=False):
        raise Failure("M1800 budget")
    if release.get("authorization") != {
            "launch_m1798_once": True,
            "automatic_retry": False,
            "publish_only_after_all_gates": True,
            "result_hammer_still_required": True}:
        raise Failure("M1800 authorization")


def namespaces_fresh():
    for path in (ATTEMPT, RESULT, FAILURE, PRIVATE, WORK, STAGE, FAIL_STAGE):
        if os.path.lexists(str(path)):
            raise Failure("namespace residue " + str(path))


def clean_env(extra):
    value = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
             "SNPSLMD_LICENSE_FILE": LICENSE_SERVER,
             "LM_LICENSE_FILE": str(LICENSE_FILE)}
    value.update(extra)
    return value


def run(command, cwd, env, timeout, output):
    CHECK.validate_sources()
    BASE.collision_gate()
    with Path(output).open("wb") as stream:
        completed = subprocess.run(command, cwd=cwd, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   timeout=timeout, check=False)
    if completed.returncode != 0:
        raise Failure("tool failure " + Path(command[0]).name)


def main():
    if len(sys.argv) != 1:
        raise Failure("M1798 accepts no arguments")
    state = {"phase": "SOURCE_CHAIN", "attempt": False, "complete": False,
             "vcs_compiles": 0, "simv_runs": 0, "saif_files": 0,
             "ptpx_runs": 0}
    queue_handle = SHARED_QUEUE.open("a+")
    lock_handle = LOCK.open("a+")
    try:
        verify_authority()
        CHECK.validate_sources()
        namespaces_fresh()
        fcntl.flock(queue_handle.fileno(), fcntl.LOCK_EX)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        BASE.collision_gate()
        BASE.resource_gate()
        namespaces_fresh()
        state["phase"] = "LICENSE_PREFLIGHT"
        probe = subprocess.run([str(LMUTIL), "lmstat", "-a", "-c", LICENSE_SERVER],
                               env=clean_env({}), stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=60, check=False)
        if probe.returncode != 0:
            raise Failure("license preflight")

        ATTEMPT.mkdir()
        state["attempt"] = True
        BASE.write_json(ATTEMPT / "attempt.json", {
            "status": "M1798_ATTEMPT_CONSUMED", "budget": COUNTS,
            "automatic_retry": False, "reuse_prior_simv_saif_ptpx": False,
            "workload": "one_directed_warmup_plus_eight_measured_fixed_t10_tiles",
            "attempt_uniqueness": ATTEMPT_UNIQUENESS})
        BASE.seal_dir(ATTEMPT)
        WORK.mkdir(); (WORK / "build").mkdir(); (WORK / "candidate").mkdir()
        build = WORK / "build"; candidate = WORK / "candidate"

        state["phase"] = "VCS_COMPILE"
        state["vcs_compiles"] += 1
        run([str(VCS), "-full64", "-sverilog", "+v2k", "-timescale=1ns/1ps",
             "+define+UNIT_DELAY", "-debug_access+r", "-lca", "+vcs+lic+wait",
             "-Mdir=csrc", "-f", str(FILELIST), "-top", TOP, "-o", "simv"],
            build, clean_env({"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
                              "VCS_ARCH_OVERRIDE": "linux"}), 7200,
            build / "compile.log")
        if not (build / "simv").is_file():
            raise Failure("simv absent")

        state["phase"] = "MAPPED_SIM_SAIF"
        state["simv_runs"] += 1
        saif = candidate / "m1798_c3_fixed_t10_component.saif"
        sim_log = candidate / "mapped_sim.log"
        run(["./simv", "-lca", "+M1790_UCLI_SAIF", "-no_save", "-ucli",
             "-i", str(UCLI)], build,
            clean_env({"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
                       "VCS_ARCH_OVERRIDE": "linux",
                       "M1798_SAIF_FILE": str(saif)}), 7200, sim_log)
        runtime = CHECK.validate_runtime(sim_log)
        saif_check = CHECK.validate_saif(saif, runtime["measurement_cycles"])
        state["saif_files"] += 1

        state["phase"] = "PTPX"
        state["ptpx_runs"] += 1
        pt_dir = candidate / "ptpx"; pt_dir.mkdir()
        run([str(PT), "-f", str(PT_TCL)], HW,
            clean_env({"M1798_TT_LIB_DB": str(TT_DB),
                       "M1798_MAPPED_NETLIST": str(NET),
                       "M1798_MAPPED_SDC": str(SDC),
                       "M1798_GATE_SAIF": str(saif),
                       "M1798_OUTPUT_DIR": str(pt_dir),
                       "M1798_SAIF_INSTANCE": SAIF_SCOPE,
                       "M1798_MEASUREMENT_CYCLES": str(
                           runtime["measurement_cycles"]),
                       "M1798_SAIF_DURATION_NS": str(
                           runtime["measurement_cycles"] * 3)}),
            7200, pt_dir / "ptpx.log")
        marker = pt_dir / "PTPX_INTERNAL_COMPLETE.txt"
        if (not marker.is_file() or marker.read_text().count(
                "PASS_M1790_C3_M1454_FIXED_T10_MAPPED_COMPONENT_PTPX_TOOL_COMPLETE") != 1):
            raise Failure("PTPX marker absent")
        metrics = CHECK.component_power(
            pt_dir / "reports/ptpx_whole_mapped_c3_logic.rpt",
            runtime["measurement_cycles"])
        if any(state[key] != value for key, value in COUNTS.items()):
            raise Failure("execution count drift")

        STAGE.mkdir()
        shutil.copytree(candidate, STAGE / "candidate")
        shutil.copy2(build / "compile.log", STAGE / "compile.log")
        BASE.write_json(STAGE / "runtime.json", runtime)
        BASE.write_json(STAGE / "metrics.json", metrics)
        BASE.write_json(STAGE / "receipt.json", {
            "schema": "m1798_c3_m1454_fixed_t10_mapped_energy_candidate_receipt_r1_v1",
            "status": "PASS_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "one_shot": dict(COUNTS, automatic_retry=False,
                              reuse_prior_simv_saif_ptpx=False),
            "identity": {"netlist_sha256": sha(NET), "sdc_sha256": sha(SDC),
                         "runner_sha256": sha(RUNNER),
                         "source_contract_sha256": sha(CONTRACT),
                         "source_review_json_sha256": sha(M1799 / "review.json"),
                         "launch_release_sha256": sha(M1800)},
            "workload": {"warmup_tiles_outside_saif": 1,
                         "measured_dense_tiles": 8,
                         "ordered_tile_done_tags_checked": 9,
                         "checkpoint_capture": False,
                         "public_port_only": True},
            "gate_simulation": {"mode": "UNIT_DELAY_functional",
                                "timing_simulation": False,
                                "independent_timing_authority": "M1456"},
            "timing_authority": {"clock_period_ns": 3.0,
                                 "pt_setup_wns_ns": 0.000299,
                                 "pt_hold_wns_ns": 0.030474},
            "saif_check": saif_check,
            "metrics": metrics,
            "claim_boundary": metrics["claim_boundary"]})
        (STAGE / "RUN_COMPLETE.txt").write_text(
            "PASS_M1798_C3_M1454_FIXED_T10_MAPPED_COMPONENT_ENERGY_CANDIDATE_PENDING_RESULT_HAMMER\n")
        BASE.seal_dir(STAGE)
        BASE.publish_no_replace(WORK, PRIVATE)
        BASE.publish_no_replace(STAGE, RESULT)
        state["complete"] = True
        print("PASS_M1798_C3_M1454_FIXED_T10_MAPPED_COMPONENT_ENERGY_CANDIDATE_PENDING_RESULT_HAMMER")
        return 0
    except BaseException as error:
        if not state["complete"]:
            try:
                FAIL_STAGE.mkdir(exist_ok=False)
                BASE.write_json(FAIL_STAGE / "failure.json", {
                    "status": "FAILED_OR_INCOMPLETE", "phase": state["phase"],
                    "error": type(error).__name__,
                    "attempt_consumed": state["attempt"],
                    "counts": dict((key, state[key]) for key in COUNTS),
                    "automatic_retry": False, "canonical_result": False})
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
        lock_handle.close(); queue_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
