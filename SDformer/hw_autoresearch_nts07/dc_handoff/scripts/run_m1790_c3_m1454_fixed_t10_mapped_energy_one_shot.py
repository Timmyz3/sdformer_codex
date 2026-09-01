#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""One-shot fresh M1790 C3 mapped-gate SAIF/PTPX campaign.

The source bundle is inert until a different-author M1791 source hammer and an
exact M1792 release are present and pinned through environment SHA256 values.
There is no automatic retry and no reuse of an old simv, SAIF, or PTPX report.
"""
from datetime import datetime, timezone
import ctypes
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
CHECKER = HW / "system_simulator/scripts/check_m1790_c3_m1454_fixed_t10_mapped_energy_source.py"
SPEC = importlib.util.spec_from_file_location("m1790_checker", str(CHECKER))
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("M1790 checker unavailable")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)

CONTRACT = CHECK.CONTRACT
NET = CHECK.NET
SDC = CHECK.SDC
FILELIST = CHECK.FILELIST
UCLI = CHECK.UCLI
PT_TCL = CHECK.PT_TCL
TT_DB = CHECK.TT_DB
TOP = CHECK.TOP
SAIF_SCOPE = CHECK.SAIF_SCOPE

M1791 = HW / "reviews/m1791_m1790_c3_m1454_fixed_t10_mapped_energy_source_hammer_r1_20260902"
M1792 = HW / "contracts/m1792_m1791_m1790_c3_m1454_fixed_t10_mapped_energy_launch_release_r1_20260902.json"

VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
PT = Path("/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
LICENSE_SERVER = "27030@ic.ismd-nemo"
LICENSE_FILE = Path("/opt/synopsys/Synopsys.dat")

ATTEMPT = HW / "results/.m1790_c3_mapped_energy_attempt_consumed"
RESULT = HW / "results/m1790_c3_mapped_energy_r1_20260902"
FAILURE = HW / "results/m1790_c3_mapped_energy_r1_20260902.failed_or_incomplete.quarantine"
PRIVATE = HW / "results/m1790_c3_mapped_energy_r1_20260902.private_build.unsealed_do_not_cite"
WORK = HW / ("results/.m1790_c3_mapped_energy_work." + str(os.getpid()))
STAGE = HW / ("results/.m1790_c3_mapped_energy_stage." + str(os.getpid()))
FAIL_STAGE = HW / ("results/.m1790_c3_mapped_energy_failure_stage." + str(os.getpid()))
LOCK = Path("/tmp/m1790_c3_mapped_energy.lock")
SHARED_QUEUE = Path("/tmp/date_dual_synopsys_same_uid_eda_queue.lock")
COUNTS = {"vcs_compiles": 1, "simv_runs": 1, "saif_files": 1,
          "ptpx_runs": 1}


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


def seal_dir(root):
    rows = []
    for path in Path(root).rglob("*"):
        if path.is_symlink():
            raise Failure("symlink in candidate")
        if path.is_file() and path.name not in {
                "SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            rows.append((path.relative_to(root).as_posix(), sha(path)))
    rows.sort()
    manifest = Path(root) / "SHA256SUMS"
    manifest.write_text("".join(digest + "  " + name + "\n"
                                for name, digest in rows))
    (Path(root) / "SHA256SUMS.seal.sha256").write_text(
        sha(manifest) + "  SHA256SUMS\n")


def verify_seal(root, manifest_sha, outer_sha):
    root = Path(root)
    exact(root / "SHA256SUMS", manifest_sha)
    exact(root / "SHA256SUMS.seal.sha256", outer_sha)
    if (root / "SHA256SUMS.seal.sha256").read_text().split() != [
            manifest_sha, "SHA256SUMS"]:
        raise Failure("outer seal content")
    listed = set()
    for row in (root / "SHA256SUMS").read_text().splitlines():
        fields = row.split(maxsplit=1)
        if len(fields) != 2:
            raise Failure("manifest syntax")
        rel = Path(fields[1].lstrip("*"))
        name = rel.as_posix()
        if rel.is_absolute() or ".." in rel.parts or name in listed:
            raise Failure("unsafe manifest")
        exact(root / rel, fields[0])
        listed.add(name)


def publish_no_replace(source, destination):
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p,
                          ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    if renameat2(-100, os.fsencode(source), -100,
                 os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n")


def authority_pin(name):
    value = os.environ.get(name, "")
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise Failure("exact authority absent " + name)
    return value


def verify_authority():
    exact(RUNNER, authority_pin("M1790_EXPECTED_RUNNER_SHA256"))
    exact(CONTRACT, authority_pin("M1790_EXPECTED_SOURCE_CONTRACT_SHA256"))
    verify_seal(M1791,
        authority_pin("M1790_EXPECTED_M1791_MANIFEST_SHA256"),
        authority_pin("M1790_EXPECTED_M1791_OUTER_FILE_SHA256"))
    exact(M1791 / "review.json",
          authority_pin("M1790_EXPECTED_M1791_REVIEW_SHA256"))
    exact(M1792, authority_pin("M1790_EXPECTED_M1792_RELEASE_SHA256"))
    review = strict_json(M1791 / "review.json")
    release = strict_json(M1792)
    if (review.get("status") !=
            "PASS_M1791_M1790_C3_MAPPED_ENERGY_SOURCE_HAMMER__AUTHORIZE_ONE_FRESH_CAMPAIGN"):
        raise Failure("M1791 status")
    if (release.get("status") !=
            "AUTHORIZE_ONE_FRESH_M1790_C3_MAPPED_ENERGY_CAMPAIGN"):
        raise Failure("M1792 status")
    if release.get("fresh_execution_budget") != dict(COUNTS,
            automatic_retry=False, reuse_prior_simv_saif_ptpx=False):
        raise Failure("M1792 budget")


def namespaces_fresh():
    for path in (ATTEMPT, RESULT, FAILURE, PRIVATE, WORK, STAGE, FAIL_STAGE):
        if os.path.lexists(str(path)):
            raise Failure("namespace residue " + str(path))


def collision_gate():
    blocked = {"vcs", "vcs1", "vlogan", "simv", "dc_shell", "dc_shell-t",
               "pt_shell", "fm_shell", "icc2_shell", "common_shell_exec",
               "common_shell_exe"}
    ancestry = set()
    pid = os.getpid()
    while pid > 1 and pid not in ancestry:
        ancestry.add(pid)
        try:
            pid = int((Path("/proc") / str(pid) / "stat").read_text().split()[3])
        except Exception:
            break
    hits = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit() or int(item.name) in ancestry:
            continue
        try:
            if item.stat().st_uid != os.getuid():
                continue
            comm = (item / "comm").read_text().strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if comm in blocked:
            hits.append((item.name, comm))
    if hits:
        raise Failure("same-UID EDA collision " + repr(hits))


def resource_gate():
    values = {}
    for row in Path("/proc/meminfo").read_text().splitlines():
        fields = row.replace(":", "").split()
        if fields and fields[0] in {"MemAvailable", "SwapFree",
                                    "CommitLimit", "Committed_AS"}:
            values[fields[0]] = int(fields[1])
    if values.get("MemAvailable", 0) < 24 * 1024 * 1024:
        raise Failure("MemAvailable below 24 GiB")
    if values.get("SwapFree", 0) < 8 * 1024 * 1024:
        raise Failure("SwapFree below 8 GiB")
    if values.get("CommitLimit", 0) - values.get("Committed_AS", 0) < 24 * 1024 * 1024:
        raise Failure("commit headroom below 24 GiB")
    if shutil.disk_usage(HW / "results").free < 16 * 1024 * 1024 * 1024:
        raise Failure("result disk free below 16 GiB")


def clean_env(extra):
    value = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
             "SNPSLMD_LICENSE_FILE": LICENSE_SERVER,
             "LM_LICENSE_FILE": str(LICENSE_FILE)}
    value.update(extra)
    return value


def run(command, cwd, env, timeout, output):
    CHECK.validate_sources()
    collision_gate()
    with Path(output).open("wb") as stream:
        completed = subprocess.run(command, cwd=cwd, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   timeout=timeout, check=False)
    if completed.returncode != 0:
        raise Failure("tool failure " + Path(command[0]).name)


def main():
    if len(sys.argv) != 1:
        raise Failure("M1790 accepts no arguments")
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
        collision_gate()
        resource_gate()
        namespaces_fresh()
        state["phase"] = "LICENSE_PREFLIGHT"
        probe = subprocess.run([str(LMUTIL), "lmstat", "-a", "-c", LICENSE_SERVER],
                               env=clean_env({}), stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=60, check=False)
        if probe.returncode != 0:
            raise Failure("license preflight")

        ATTEMPT.mkdir()
        state["attempt"] = True
        write_json(ATTEMPT / "attempt.json", {
            "status": "M1790_ATTEMPT_CONSUMED", "budget": COUNTS,
            "automatic_retry": False, "reuse_prior_simv_saif_ptpx": False,
            "workload": "one_directed_warmup_plus_eight_measured_fixed_t10_tiles"})
        seal_dir(ATTEMPT)
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
        saif = candidate / "m1790_c3_fixed_t10_component.saif"
        sim_log = candidate / "mapped_sim.log"
        run(["./simv", "-lca", "+M1790_UCLI_SAIF", "-no_save", "-ucli",
             "-i", str(UCLI)], build,
            clean_env({"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
                       "VCS_ARCH_OVERRIDE": "linux",
                       "M1790_SAIF_FILE": str(saif)}), 7200, sim_log)
        runtime = CHECK.validate_runtime(sim_log)
        saif_check = CHECK.validate_saif(saif, runtime["measurement_cycles"])
        state["saif_files"] += 1

        state["phase"] = "PTPX"
        state["ptpx_runs"] += 1
        pt_dir = candidate / "ptpx"; pt_dir.mkdir()
        run([str(PT), "-f", str(PT_TCL)], HW,
            clean_env({"M1790_TT_LIB_DB": str(TT_DB),
                       "M1790_MAPPED_NETLIST": str(NET),
                       "M1790_MAPPED_SDC": str(SDC),
                       "M1790_GATE_SAIF": str(saif),
                       "M1790_OUTPUT_DIR": str(pt_dir),
                       "M1790_SAIF_INSTANCE": SAIF_SCOPE,
                       "M1790_MEASUREMENT_CYCLES": str(runtime["measurement_cycles"]),
                       "M1790_SAIF_DURATION_NS": str(runtime["measurement_cycles"] * 3)}),
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
        write_json(STAGE / "runtime.json", runtime)
        write_json(STAGE / "metrics.json", metrics)
        write_json(STAGE / "receipt.json", {
            "schema": "m1790_c3_m1454_fixed_t10_mapped_energy_candidate_receipt_r1_v1",
            "status": "PASS_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "one_shot": dict(COUNTS, automatic_retry=False,
                              reuse_prior_simv_saif_ptpx=False),
            "identity": {"netlist_sha256": sha(NET), "sdc_sha256": sha(SDC),
                         "runner_sha256": sha(RUNNER),
                         "source_contract_sha256": sha(CONTRACT)},
            "workload": {"warmup_tiles_outside_saif": 1,
                         "measured_dense_tiles": 8,
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
            "PASS_M1790_C3_M1454_FIXED_T10_MAPPED_COMPONENT_ENERGY_CANDIDATE_PENDING_RESULT_HAMMER\n")
        seal_dir(STAGE)
        publish_no_replace(WORK, PRIVATE)
        publish_no_replace(STAGE, RESULT)
        state["complete"] = True
        print("PASS_M1790_C3_M1454_FIXED_T10_MAPPED_COMPONENT_ENERGY_CANDIDATE_PENDING_RESULT_HAMMER")
        return 0
    except BaseException as error:
        if not state["complete"]:
            try:
                FAIL_STAGE.mkdir(exist_ok=False)
                write_json(FAIL_STAGE / "failure.json", {
                    "status": "FAILED_OR_INCOMPLETE", "phase": state["phase"],
                    "error": type(error).__name__, "attempt_consumed": state["attempt"],
                    "counts": dict((key, state[key]) for key in COUNTS),
                    "automatic_retry": False, "canonical_result": False})
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
        lock_handle.close(); queue_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
