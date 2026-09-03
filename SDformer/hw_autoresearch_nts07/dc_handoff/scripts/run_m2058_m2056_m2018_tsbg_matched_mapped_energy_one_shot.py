#!/usr/bin/python3.12
"""Exactly-once, two-axis M2056 mapped VCS/SAIF/PTPX campaign.

This file is inert until the caller supplies exact M2058 source and M2059
review pins.  All EDA work is P1-serial under the shared same-UID queue.  A
failed campaign consumes the attempt and cannot retry.
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
PARSER_PATH = HW / "system_simulator/scripts/parse_m2058_m2056_m2018_tsbg_matched_mapped_energy_result.py"
SPEC = importlib.util.spec_from_file_location("m2058_parser", str(PARSER_PATH))
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("M2058 parser unavailable")
PARSER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARSER)

CONTRACT = PARSER.CONTRACT
CONTRACT_SIDECAR = Path(str(CONTRACT) + ".sha256")
CONTRACT_OUTER = Path(str(CONTRACT) + ".sha256.seal.sha256")
M2059_REVIEW = HW / "reviews/m2059_m2058_m2056_tsbg_matched_mapped_energy_runner_source_hammer_r1_20260903"

CELL = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/tcbn28hpcplusbwp35p140.v")
TT_DB = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140tt0p9v25c.db")
SSG_DB = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db")
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
PT = Path("/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
LICENSE_SERVER = "27030@ic.ismd-nemo"
LICENSE_FILE = Path("/opt/synopsys/Synopsys.dat")

TOOL_SHA256 = {
    CELL: "3ed0796ffa8a0eb1406860e07913b8457969bcec492c3cb15599ee8db964707a",
    TT_DB: "d8975a427b9f5f6b6667ee5dbc7ff33eac15ab480a871d756af48cd9afa18070",
    SSG_DB: "79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af",
    VCS: "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
    PT: "afdcfa7071f86d229ed3b4481f67adefd1c51bb0288b7e7d370213a43b70c9ef",
    LMUTIL: "e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07",
}

COUNTS = {"license_preflight_lmstat": 1, "vcs_compiles": 2,
          "simv_runs": 2, "saif_files": 2, "ptpx_runs": 2}
ATTEMPT = HW / "results/.m2058_m2056_tsbg_matched_mapped_energy_attempt_consumed"
RESULT = HW / "results/m2058_m2056_tsbg_matched_mapped_energy_r1_20260903"
FAILURE = HW / "results/m2058_m2056_tsbg_matched_mapped_energy_r1_20260903.failed_or_incomplete.quarantine"
PRIVATE = HW / "results/m2058_m2056_tsbg_matched_mapped_energy_r1_20260903.private_build.unsealed_do_not_cite"
WORK = HW / ("results/.m2058_m2056_tsbg_matched_mapped_energy_work." + str(os.getpid()))
STAGE = HW / ("results/.m2058_m2056_tsbg_matched_mapped_energy_stage." + str(os.getpid()))
FAIL_STAGE = HW / ("results/.m2058_m2056_tsbg_matched_mapped_energy_failure_stage." + str(os.getpid()))
LOCAL_LOCK = Path("/tmp/m2058_m2056_tsbg_matched_mapped_energy.lock")
SHARED_QUEUE = Path("/tmp/date_dual_synopsys_same_uid_eda_queue.lock")


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


def authority_pin(name):
    value = os.environ.get(name, "")
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise Failure("exact authority pin absent " + name)
    return value


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


def verify_sealed_directory(root, manifest_sha, outer_file_sha):
    root = Path(root)
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    exact(manifest, manifest_sha)
    exact(outer, outer_file_sha)
    if outer.read_text().split() != [sha(manifest), "SHA256SUMS"]:
        raise Failure("outer seal content " + str(root))
    mapping = {}
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        if len(fields) != 2:
            raise Failure("manifest syntax")
        rel = Path(fields[1].lstrip("*"))
        if rel.is_absolute() or ".." in rel.parts or rel.as_posix() in mapping:
            raise Failure("unsafe/duplicate manifest member")
        exact(root / rel, fields[0])
        mapping[rel.as_posix()] = fields[0]
    actual = set()
    for member in root.rglob("*"):
        if member.is_symlink():
            raise Failure("symlink in sealed directory")
        if member.is_file() and member.name not in {
                "SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            actual.add(member.relative_to(root).as_posix())
    if actual != set(mapping):
        raise Failure("non-exhaustive directory seal")
    return mapping


def verify_authority():
    exact(RUNNER, authority_pin("M2058_EXPECTED_RUNNER_SHA256"))
    exact(PARSER_PATH, authority_pin("M2058_EXPECTED_PARSER_SHA256"))
    exact(CONTRACT, authority_pin("M2058_EXPECTED_CONTRACT_SHA256"))
    exact(CONTRACT_SIDECAR, authority_pin("M2058_EXPECTED_CONTRACT_SIDECAR_SHA256"))
    exact(CONTRACT_OUTER, authority_pin("M2058_EXPECTED_CONTRACT_OUTER_FILE_SHA256"))
    if CONTRACT_SIDECAR.read_text().split() != [sha(CONTRACT), CONTRACT.name]:
        raise Failure("contract sidecar content")
    if CONTRACT_OUTER.read_text().split() != [sha(CONTRACT_SIDECAR),
                                               CONTRACT_SIDECAR.name]:
        raise Failure("contract outer content")
    review_map = verify_sealed_directory(
        M2059_REVIEW,
        authority_pin("M2058_EXPECTED_M2059_MANIFEST_SHA256"),
        authority_pin("M2058_EXPECTED_M2059_OUTER_FILE_SHA256"))
    exact(M2059_REVIEW / "review.json",
          authority_pin("M2058_EXPECTED_M2059_REVIEW_SHA256"))
    if review_map.get("review.json") != sha(M2059_REVIEW / "review.json"):
        raise Failure("M2059 review not sealed")
    review = strict_json(M2059_REVIEW / "review.json")
    if (review.get("status") !=
            "PASS_M2059_M2058_M2056_TSBG_MATCHED_MAPPED_ENERGY_RUNNER_SOURCE_HAMMER__AUTHORIZE_ONE_M2058_EXECUTION"
            or review.get("severity_counts") != {"p0": 0, "p1": 0, "p2": 1}
            or review.get("authorization") != {
                "execute_m2058_once": True, "license_preflight_lmstat": 1,
                "vcs_compiles": 2, "simv_runs": 2, "saif_files": 2,
                "ptpx_runs": 2, "p1_serial": True,
                "automatic_retry": False,
                "independent_result_hammer_required": True}):
        raise Failure("M2059 authorization")
    PARSER.validate_sources()
    for path, digest in TOOL_SHA256.items():
        exact(path, digest)


def namespaces_fresh():
    for path in (ATTEMPT, RESULT, FAILURE, PRIVATE, WORK, STAGE, FAIL_STAGE):
        if os.path.lexists(str(path)):
            raise Failure("namespace residue " + str(path))
    for pattern in (".m2058_m2056_tsbg_matched_mapped_energy_work.*",
                    ".m2058_m2056_tsbg_matched_mapped_energy_stage.*",
                    ".m2058_m2056_tsbg_matched_mapped_energy_failure_stage.*"):
        if next((HW / "results").glob(pattern), None) is not None:
            raise Failure("private namespace residue " + pattern)


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
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
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
        raise Failure("insufficient MemAvailable")
    if values.get("SwapFree", 0) < 8 * 1024 * 1024:
        raise Failure("insufficient SwapFree")
    if (values.get("CommitLimit", 0) - values.get("Committed_AS", 0)
            < 24 * 1024 * 1024):
        raise Failure("insufficient commit headroom")
    if shutil.disk_usage(HW / "results").free < 20 * 1024 * 1024 * 1024:
        raise Failure("insufficient result disk")


def clean_env(extra):
    value = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
             "SNPSLMD_LICENSE_FILE": LICENSE_SERVER,
             "LM_LICENSE_FILE": str(LICENSE_FILE)}
    value.update(extra)
    return value


def run_checked(command, cwd, env, timeout, output, record_command=False):
    verify_authority()
    collision_gate()
    with Path(output).open("wb") as stream:
        if record_command:
            stream.write(("M2058_COMMAND_JSON=" + json.dumps(
                list(command), separators=(",", ":"), ensure_ascii=True)
                + "\n").encode("ascii"))
            stream.flush()
        completed = subprocess.run(command, cwd=cwd, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   timeout=timeout, check=False)
    if completed.returncode != 0:
        raise Failure("tool failure " + Path(command[0]).name)


def write_json(path, value):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temp = Path(str(path) + ".tmp." + str(os.getpid()))
    with temp.open("x") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def seal_dir(root):
    root = Path(root)
    members = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise Failure("symlink before seal")
        if path.is_file() and path.name not in {
                "SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            members.append(path)
    manifest = root / "SHA256SUMS"
    with manifest.open("x") as handle:
        for path in members:
            handle.write(sha(path) + "  " + path.relative_to(root).as_posix() + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    outer = root / "SHA256SUMS.seal.sha256"
    with outer.open("x") as handle:
        handle.write(sha(manifest) + "  SHA256SUMS\n")
        handle.flush()
        os.fsync(handle.fileno())


def publish_no_replace(source, destination):
    source = Path(source)
    destination = Path(destination)
    if os.path.lexists(str(destination)):
        raise Failure("publish destination exists")
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.renameat2(-100, os.fsencode(source), -100,
                            os.fsencode(destination), 1)
    if result != 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), str(destination))


def make_attempt(state):
    ATTEMPT.mkdir()
    write_json(ATTEMPT / "attempt.json", {
        "schema": "m2058_m2056_tsbg_matched_mapped_energy_attempt_r1_v1",
        "status": "M2058_ATTEMPT_CONSUMED_NO_RETRY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "budget": dict(COUNTS, p1_serial=True, automatic_retry=False),
        "workload": "ep34_global_slot42_sample0_layer28_fc1_token0_g48",
        "required_simv_plusargs": ["+WORKLOAD_SLOT=42"]})
    seal_dir(ATTEMPT)
    state["attempt"] = True


def main():
    if len(sys.argv) != 1:
        raise Failure("M2058 accepts no command-line arguments")
    state = {"phase": "PRE_AUTHORITY", "attempt": False, "complete": False,
             "license_preflight_lmstat": 0, "vcs_compiles": 0,
             "simv_runs": 0, "saif_files": 0, "ptpx_runs": 0,
             "p1_serial": True, "automatic_retry": False}
    queue_handle = SHARED_QUEUE.open("a+")
    local_handle = LOCAL_LOCK.open("a+")
    try:
        verify_authority()
        namespaces_fresh()
        fcntl.flock(queue_handle.fileno(), fcntl.LOCK_EX)
        fcntl.flock(local_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        collision_gate()
        resource_gate()
        namespaces_fresh()
        make_attempt(state)

        WORK.mkdir()
        build_root = WORK / "build"
        compile_logs = WORK / "compile_logs"
        candidate = WORK / "candidate"
        build_root.mkdir(); compile_logs.mkdir(); candidate.mkdir()

        state["phase"] = "LICENSE_PREFLIGHT"
        verify_authority()
        state["license_preflight_lmstat"] += 1
        with (WORK / "license_preflight.log").open("wb") as stream:
            probe = subprocess.run(
                [str(LMUTIL), "lmstat", "-a", "-c", LICENSE_SERVER],
                env=clean_env({}), stdout=stream, stderr=subprocess.STDOUT,
                timeout=60, check=False)
        if probe.returncode != 0:
            raise Failure("single license preflight failed; attempt consumed")

        command_rows = []
        for axis in PARSER.AXIS_ORDER:
            cfg = PARSER.AXES[axis]
            build = build_root / axis
            axis_root = candidate / axis
            pt_root = axis_root / "ptpx"
            build.mkdir(); axis_root.mkdir(); pt_root.mkdir()

            state["phase"] = "VCS_COMPILE_" + axis
            compile_command = [
                str(VCS), "-full64", "-sverilog", "+v2k",
                "-timescale=1ns/1ps", "+define+UNIT_DELAY",
                "-debug_access+r", "-lca", "+vcs+lic+wait", "-Mdir=csrc",
                "-f", str(cfg["filelist"]), "-top", PARSER.TOP, "-o", "simv"]
            state["vcs_compiles"] += 1
            compile_log = compile_logs / (axis + ".compile.log")
            run_checked(compile_command, build,
                        clean_env({"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
                                   "VCS_ARCH_OVERRIDE": "linux"}),
                        21600, compile_log, record_command=True)
            if not (build / "simv").is_file() or (build / "simv").is_symlink():
                raise Failure("simv absent " + axis)
            PARSER.parse_command_log(compile_log, axis)

            state["phase"] = "SIMV_SAIF_" + axis
            sim_command = ["./simv", "-lca", "+WORKLOAD_SLOT=42",
                           "-no_save", "-ucli", "-i", str(cfg["ucli"])]
            if [arg for arg in sim_command[1:] if arg.startswith("+")] != [
                    "+WORKLOAD_SLOT=42"]:
                raise Failure("simv plusarg surface drift")
            state["simv_runs"] += 1
            saif = axis_root / "mapped_execute.saif"
            sim_log = axis_root / "mapped_sim.log"
            run_checked(sim_command, build,
                        clean_env({"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
                                   "VCS_ARCH_OVERRIDE": "linux",
                                   "M2056_SAIF_FILE": str(saif)}),
                        21600, sim_log)
            PARSER.parse_runtime(sim_log, axis)
            PARSER.parse_saif(saif, axis)
            state["saif_files"] += 1

            state["phase"] = "PTPX_" + axis
            state["ptpx_runs"] += 1
            pt_command = [str(PT), "-f", str(PARSER.PT_TCL)]
            run_checked(pt_command, HW,
                        clean_env({"M2056_AXIS": axis,
                                   "M2056_TT_LIB_DB": str(TT_DB),
                                   "M2056_SSG_LIB_DB": str(SSG_DB),
                                   "M2056_MAPPED_NETLIST": str(cfg["netlist"]),
                                   "M2056_MAPPED_SDC": str(cfg["sdc"]),
                                   "M2056_GATE_SAIF": str(saif),
                                   "M2056_OUTPUT_DIR": str(pt_root)}),
                        21600, pt_root / "ptpx.log")
            PARSER.parse_ptpx(pt_root, axis)
            command_rows.append({"axis": axis,
                                 "compile": compile_command,
                                 "simv": sim_command,
                                 "ptpx": pt_command})

        if any(state[key] != value for key, value in COUNTS.items()):
            raise Failure("execution count drift")
        verify_authority()
        receipt = PARSER.parse_candidate(candidate, compile_logs)
        receipt["created_utc"] = datetime.now(timezone.utc).isoformat()
        receipt["identity"] = {
            "runner_sha256": sha(RUNNER), "parser_sha256": sha(PARSER_PATH),
            "contract_sha256": sha(CONTRACT),
            "m2059_review_sha256": sha(M2059_REVIEW / "review.json"),
            "m2056_review_sha256": sha(PARSER.M2056_REVIEW / "review.json")}
        receipt["result_hammer_required"] = True

        STAGE.mkdir()
        os.rename(candidate, STAGE / "candidate")
        os.rename(compile_logs, STAGE / "compile_logs")
        shutil.copy2(WORK / "license_preflight.log", STAGE / "license_preflight.log")
        write_json(STAGE / "commands.json", command_rows)
        write_json(STAGE / "receipt.json", receipt)
        state["phase"] = "COMPLETE_PENDING_RESULT_HAMMER"
        state["complete"] = True
        write_json(STAGE / "execution_state.json", state)
        (STAGE / "RUN_COMPLETE.txt").write_text(
            "PASS_M2058_M2056_TSBG_MATCHED_MAPPED_ENERGY_PENDING_INDEPENDENT_RESULT_HAMMER\n")
        seal_dir(STAGE)
        publish_no_replace(STAGE, RESULT)
        publish_no_replace(WORK, PRIVATE)
        return 0
    except BaseException as exc:
        state["failure_type"] = type(exc).__name__
        state["failure"] = str(exc)
        if state["attempt"] and not os.path.lexists(str(FAILURE)):
            try:
                FAIL_STAGE.mkdir()
                write_json(FAIL_STAGE / "failure.json", state)
                if WORK.exists():
                    os.rename(WORK, FAIL_STAGE / "work")
                seal_dir(FAIL_STAGE)
                publish_no_replace(FAIL_STAGE, FAILURE)
            except BaseException:
                pass
        raise
    finally:
        local_handle.close()
        queue_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
