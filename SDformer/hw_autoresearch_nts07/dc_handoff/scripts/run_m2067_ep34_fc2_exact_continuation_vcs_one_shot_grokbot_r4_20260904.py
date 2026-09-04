#!/opt/anaconda3/bin/python
"""GROKBOT NEW FILE -- iscas_ssh / 2026-09-04.

Inert until an exact-pinned M2074 release authorizes one M2067 VCS run on the
r4 grokbot sticky-TB identity.  Does NOT overwrite the r3 runner.  Never targets
the quarantined ..._vcs_r1_20260903 result/failure paths.

One compile is shared by 960 serial per-slot simulations.  The attempt latch is
created before the only license query.  There is no retry, EDA, GPU, or CPU-to-
RTL ratio promotion path.
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
import shutil
import subprocess
import sys


HW = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
PARSER_PATH = HW / (
    "system_simulator/scripts/parse_m2067_ep34_fc2_exact_continuation_vcs_grokbot_r4_20260904.py"
)
SPEC = importlib.util.spec_from_file_location("m2067_parser", PARSER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("M2067 parser unavailable")
PARSER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARSER)

CONTRACT = PARSER.CONTRACT
M2074 = HW / (
    "reviews/m2074_m2073_m2067_ep34_fc2_exact_continuation_vcs_source_r4_hammer_r1_20260904"
)
FILELIST = HW / (
    "dc_handoff/filelists/iscas_m2067_ep34_fc2_exact_continuation_vcs_grokbot_sticky_r4_20260904.f"
)
TOP = "tb_m2067_ep34_fc2_exact_continuation_s960"
PYTHON = Path("/opt/anaconda3/bin/python3.12")
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
LICENSE_SERVER = "27030@ic.ismd-nemo"
LICENSE_FILE = Path("/opt/synopsys/Synopsys.dat")
TOOL_SHA256 = {
    VCS: "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
    LMUTIL: "e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07",
}
ATTEMPT = HW / "results/.m2067_ep34_fc2_exact_continuation_vcs_r4_grokbot_sticky_attempt_consumed"
RESULT = HW / "results/m2067_ep34_fc2_exact_continuation_vcs_r4_grokbot_sticky_20260904"
FAILURE = HW / (
    "results/m2067_ep34_fc2_exact_continuation_vcs_r4_grokbot_sticky_20260904."
    "failed_or_incomplete.quarantine"
)
WORK = HW / ("results/.m2067_ep34_fc2_exact_continuation_r4_grokbot_sticky_work." + str(os.getpid()))
STAGE = HW / ("results/.m2067_ep34_fc2_exact_continuation_r4_grokbot_sticky_stage." + str(os.getpid()))
FAIL_STAGE = HW / (
    "results/.m2067_ep34_fc2_exact_continuation_r4_grokbot_sticky_failure_stage." + str(os.getpid())
)
LOCK = Path("/tmp/m2067_ep34_fc2_exact_continuation_vcs_r4_grokbot_sticky.lock")
EDA_QUEUE = Path("/tmp/date_dual_synopsys_same_uid_eda_queue.lock")
RUN_STATE = {
    "phase": "PRE_AUTHORITY",
    "license_preflight_lmstat": 0,
    "vcs_compiles": 0,
    "simv_runs": 0,
    "completed_slots": 0,
    "current_slot": None,
}
LMSTAT_OUTPUT = b""
RESULT_PUBLISHED = False


class Failure(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(path: Path, digest: str) -> None:
    if not path.is_file() or path.is_symlink() or sha256(path) != digest:
        raise Failure("identity drift " + str(path))


def authority(name: str) -> str:
    value = os.environ.get(name, "")
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise Failure("authority pin absent " + name)
    return value


def strict_json(path: Path) -> dict:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise Failure("duplicate JSON key " + key)
            value[key] = item
        return value
    value = json.loads(
        path.read_text(), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            Failure("nonfinite JSON " + token)))
    if type(value) is not dict:
        raise Failure("JSON root")
    return value


def sealed_directory(root: Path, manifest_pin: str, outer_pin: str) -> dict:
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    exact(manifest, manifest_pin)
    exact(outer, outer_pin)
    if outer.read_text().split() != [sha256(manifest), "SHA256SUMS"]:
        raise Failure("outer seal content")
    mapping = {}
    for line in manifest.read_text().splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise Failure("manifest syntax")
        relative = Path(fields[1].lstrip("*"))
        if relative.is_absolute() or ".." in relative.parts:
            raise Failure("unsafe manifest member")
        exact(root / relative, fields[0])
        if relative.as_posix() in mapping:
            raise Failure("duplicate manifest member")
        mapping[relative.as_posix()] = fields[0]
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name not in {
                  "SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    if any(path.is_symlink() for path in root.rglob("*")):
        raise Failure("symlink in sealed directory")
    if actual != set(mapping):
        raise Failure("non-exhaustive sealed directory")
    return mapping


def verify_authority() -> None:
    exact(PYTHON, authority("M2067_EXPECTED_PYTHON_SHA256"))
    exact(RUNNER, authority("M2067_EXPECTED_RUNNER_SHA256"))
    exact(PARSER_PATH, authority("M2067_EXPECTED_PARSER_SHA256"))
    exact(CONTRACT, authority("M2067_EXPECTED_CONTRACT_SHA256"))
    review_map = sealed_directory(
        M2074, authority("M2067_EXPECTED_M2074_MANIFEST_SHA256"),
        authority("M2067_EXPECTED_M2074_OUTER_FILE_SHA256"))
    exact(M2074 / "review.json", authority("M2067_EXPECTED_M2074_REVIEW_SHA256"))
    if review_map.get("review.json") != sha256(M2074 / "review.json"):
        raise Failure("M2074 review not sealed")
    review = strict_json(M2074 / "review.json")
    identity = review.get("reviewed_source_identity", {})
    if (not review.get("status", "").startswith("PASS_M2074_")
            or review.get("severity_counts", {}).get("P0") != 0
            or review.get("severity_counts", {}).get("P1") != 0
            or identity.get("runner_sha256") != sha256(RUNNER)
            or identity.get("parser_sha256") != sha256(PARSER_PATH)
            or identity.get("contract_sha256") != sha256(CONTRACT)
            or review.get("authorization", {}).get("execute_once") is not True
            or review.get("authorization", {}).get("automatic_retry") is not False):
        raise Failure("M2074 source/release authorization")
    PARSER.validate_source()
    for path, digest in TOOL_SHA256.items():
        exact(path, digest)


def fresh_namespaces() -> None:
    for path in (ATTEMPT, RESULT, FAILURE, WORK, STAGE, FAIL_STAGE):
        if os.path.lexists(str(path)):
            raise Failure("namespace residue " + str(path))
    for pattern in (".m2067_ep34_fc2_exact_continuation_work.*",
                    ".m2067_ep34_fc2_exact_continuation_stage.*",
                    ".m2067_ep34_fc2_exact_continuation_failure_stage.*"):
        if next((HW / "results").glob(pattern), None) is not None:
            raise Failure("private namespace residue " + pattern)


def collision_gate() -> None:
    blocked = {"vcs", "vcs1", "vlogan", "simv", "dc_shell", "dc_shell-t",
               "pt_shell", "fm_shell", "icc2_shell", "common_shell_exec",
               "common_shell_exe"}
    ancestry = set()
    pid = os.getpid()
    while pid > 1 and pid not in ancestry:
        ancestry.add(pid)
        try:
            pid = int((Path("/proc") / str(pid) / "stat").read_text().split()[3])
        except (OSError, ValueError):
            break
    hits = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit() or int(item.name) in ancestry:
            continue
        try:
            if item.stat().st_uid == os.getuid():
                command = (item / "comm").read_text().strip()
                if command in blocked:
                    hits.append((item.name, command))
        except OSError:
            pass
    if hits:
        raise Failure("same-UID EDA collision " + repr(hits))


def clean_env() -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
        "VCS_ARCH_OVERRIDE": "linux", "SNPSLMD_LICENSE_FILE": LICENSE_SERVER,
        "LM_LICENSE_FILE": str(LICENSE_FILE),
    }


def run_checked(command: list[str], cwd: Path, output: Path,
                timeout: int) -> None:
    verify_authority()
    collision_gate()
    with output.open("xb") as stream:
        completed = subprocess.run(command, cwd=cwd, env=clean_env(),
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   timeout=timeout, check=False)
    if completed.returncode != 0:
        raise Failure("tool failure " + Path(command[0]).name)


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def seal_dir(root: Path) -> None:
    members = [path for path in sorted(root.rglob("*")) if path.is_file()]
    if any(path.is_symlink() for path in root.rglob("*")):
        raise Failure("symlink before seal")
    manifest = root / "SHA256SUMS"
    with manifest.open("x") as stream:
        for path in members:
            stream.write(sha256(path) + "  "
                         + path.relative_to(root).as_posix() + "\n")
    with (root / "SHA256SUMS.seal.sha256").open("x") as stream:
        stream.write(sha256(manifest) + "  SHA256SUMS\n")


def publish_no_replace(source: Path, destination: Path) -> None:
    if os.path.lexists(str(destination)):
        raise Failure("publish destination exists")
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.renameat2(-100, os.fsencode(source), -100,
                            os.fsencode(destination), 1)
    if result != 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), str(destination))


def raw_tree_fingerprint(label: str, root: Path) -> list[dict]:
    """Describe every entry without following symlinks or requiring a seal."""
    if not os.path.lexists(str(root)):
        return [{"root": label, "path": ".", "type": "absent"}]
    rows = [{"root": label, "path": ".", "type": "directory"}]
    if not root.is_dir() or root.is_symlink():
        return [{"root": label, "path": ".", "type": "unexpected_root"}]
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            rows.append({"root": label, "path": relative, "type": "symlink",
                         "target": os.readlink(path)})
        elif path.is_file():
            rows.append({"root": label, "path": relative, "type": "file",
                         "size": path.stat().st_size,
                         "sha256": sha256(path)})
        elif path.is_dir():
            rows.append({"root": label, "path": relative,
                         "type": "directory"})
        else:
            rows.append({"root": label, "path": relative,
                         "type": "other"})
    return rows


def copy_failure_evidence(label: str, root: Path, destination: Path) -> None:
    """Copy existing human-auditable logs/receipts, never links or tool trees."""
    if not root.is_dir() or root.is_symlink():
        return
    for path in sorted(root.rglob("*")):
        if (not path.is_file() or path.is_symlink()
                or path.suffix.lower() not in {".log", ".json", ".txt"}):
            continue
        target = destination / label / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def publish_failure_quarantine(exc: BaseException) -> None:
    """Atomically publish one sealed no-retry record after attempt creation."""
    if RESULT_PUBLISHED:
        return
    if not os.path.lexists(str(ATTEMPT)):
        return
    if os.path.lexists(str(FAILURE)) or os.path.lexists(str(FAIL_STAGE)):
        return
    fingerprints = []
    for label, root in (("attempt", ATTEMPT), ("work", WORK),
                        ("result_stage", STAGE)):
        fingerprints.extend(raw_tree_fingerprint(label, root))
    FAIL_STAGE.mkdir()
    evidence = FAIL_STAGE / "evidence"
    evidence.mkdir()
    for label, root in (("attempt", ATTEMPT), ("work", WORK),
                        ("result_stage", STAGE)):
        copy_failure_evidence(label, root, evidence)
    if LMSTAT_OUTPUT:
        with (evidence / "lmstat.log").open("xb") as stream:
            stream.write(LMSTAT_OUTPUT)
    def observed_identity(path: Path) -> str:
        if not path.is_file() or path.is_symlink():
            return "UNAVAILABLE_OR_NONREGULAR"
        return sha256(path)

    state = dict(RUN_STATE)
    state.update({
        "status": "FAILED_DO_NOT_CITE_NO_RETRY",
        "failed_utc": datetime.now(timezone.utc).isoformat(),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "runner_sha256": observed_identity(RUNNER),
        "parser_sha256": observed_identity(PARSER_PATH),
        "contract_sha256": observed_identity(CONTRACT),
        "attempt_consumed": True,
        "automatic_retry": False,
        "publish_no_replace": True,
    })
    write_json(FAIL_STAGE / "failure.json", state)
    write_json(FAIL_STAGE / "raw_tree_fingerprint.json", {
        "schema": "m2067_failed_raw_tree_fingerprint_r2_v1",
        "entries": fingerprints,
    })
    seal_dir(FAIL_STAGE)
    publish_no_replace(FAIL_STAGE, FAILURE)


def main() -> int:
    global LMSTAT_OUTPUT, RESULT_PUBLISHED
    if len(sys.argv) != 1:
        raise Failure("runner accepts no arguments")
    verify_authority()
    fresh_namespaces()
    LOCK.touch(exist_ok=True)
    EDA_QUEUE.touch(exist_ok=True)
    with LOCK.open("r+") as local_lock, EDA_QUEUE.open("r+") as queue_lock:
        fcntl.flock(local_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(queue_lock, fcntl.LOCK_EX)
        verify_authority()
        collision_gate()
        RUN_STATE["phase"] = "ATTEMPT_CREATE"
        ATTEMPT.mkdir()
        write_json(ATTEMPT / "attempt.json", {
            "schema": "m2067_ep34_fc2_exact_continuation_attempt_r4_grokbot_v1",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "runner_sha256": sha256(RUNNER),
            "parser_sha256": sha256(PARSER_PATH),
            "contract_sha256": sha256(CONTRACT),
            "vcs_compiles_budget": 1, "simv_runs_budget": 960,
            "automatic_retry": False,
        })
        seal_dir(ATTEMPT)
        RUN_STATE["phase"] = "LICENSE_PREFLIGHT"
        RUN_STATE["license_preflight_lmstat"] = 1
        lmstat_command = [str(LMUTIL), "lmstat", "-a", "-c", LICENSE_SERVER]
        RUN_STATE["command"] = lmstat_command
        try:
            lmstat = subprocess.run(
                lmstat_command, cwd=HW, env=clean_env(),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=120, check=False)
        except subprocess.TimeoutExpired as exc:
            LMSTAT_OUTPUT = exc.stdout or b""
            raise Failure("single license preflight timeout") from exc
        LMSTAT_OUTPUT = lmstat.stdout
        if lmstat.returncode != 0 or b"Users of VCSCompiler_Net" not in lmstat.stdout:
            raise Failure("single license preflight failed")
        WORK.mkdir()
        compile_log = WORK / "vcs_compile.log"
        compile_command = [
            str(VCS), "-full64", "-sverilog", "+v2k", "-timescale=1ns/1ps",
            "-debug_access+r", "-lca", "+vcs+lic+wait", "-Mdir=csrc",
            "-f", str(FILELIST), "-top", TOP, "-o", "simv",
        ]
        RUN_STATE["phase"] = "VCS_COMPILE"
        RUN_STATE["vcs_compiles"] = 1
        RUN_STATE["command"] = compile_command
        run_checked(compile_command, WORK, compile_log, 21600)
        if not (WORK / "simv").is_file() or (WORK / "simv").is_symlink():
            raise Failure("simv absent")
        STAGE.mkdir()
        logs = STAGE / "logs"
        logs.mkdir()
        rows = []
        PARSER.validate_source()
        metadata = PARSER.strict_json(PARSER.META)
        for slot in range(PARSER.WORKLOADS):
            RUN_STATE["phase"] = "SIMV_SLOT"
            RUN_STATE["current_slot"] = slot
            RUN_STATE["simv_runs"] = slot + 1
            log = logs / f"slot_{slot:04d}.log"
            sim_command = ["./simv", "-lca", f"+WORKLOAD_SLOT={slot}"]
            RUN_STATE["command"] = sim_command
            run_checked(sim_command, WORK, log, 21600)
            RUN_STATE["phase"] = "PARSE_SLOT"
            RUN_STATE["command"] = [
                str(PYTHON), str(PARSER_PATH), "--log", str(log)]
            parsed = PARSER.parse_log(log, validate_source_identity=False,
                                      metadata=metadata)
            parsed.pop("source_identity", None)
            rows.append(parsed)
            RUN_STATE["completed_slots"] = slot + 1
        RUN_STATE["phase"] = "RESULT_AGGREGATE"
        RUN_STATE["current_slot"] = None
        RUN_STATE["command"] = ["internal", "aggregate_960_exact_once"]
        if [row["workload_slot"] for row in rows] != list(range(960)):
            raise Failure("workload exact-once order")
        verify_authority()
        base_cycles = sum(row["base_cycles"] for row in rows)
        tsbg_cycles = sum(row["tsbg_cycles"] for row in rows)
        attempt_manifest = ATTEMPT / "SHA256SUMS"
        attempt_outer = ATTEMPT / "SHA256SUMS.seal.sha256"
        attempt_map = sealed_directory(
            ATTEMPT, sha256(attempt_manifest), sha256(attempt_outer))
        if attempt_map.get("attempt.json") != sha256(ATTEMPT / "attempt.json"):
            raise Failure("attempt identity not sealed")
        review_manifest = M2074 / "SHA256SUMS"
        review_outer = M2074 / "SHA256SUMS.seal.sha256"
        result = {
            "schema": "m2067_ep34_fc2_exact_continuation_vcs_result_r3_v1",
            "status": "PENDING_INDEPENDENT_RESULT_HAMMER_DO_NOT_CITE",
            "workloads": 960,
            "integer_checks_per_axis": sum(row["integer_checks"] for row in rows),
            "ordinary_cycles_observed": base_cycles,
            "tsbg_cycles_observed": tsbg_cycles,
            "rtl_cycle_ratio_observed": base_cycles / tsbg_cycles,
            "rows": rows,
            "source_and_authority_identity": {
                "runner_sha256": sha256(RUNNER),
                "parser_sha256": sha256(PARSER_PATH),
                "contract_sha256": sha256(CONTRACT),
                "m2074_review_sha256": sha256(M2074 / "review.json"),
                "m2074_manifest_sha256": sha256(review_manifest),
                "m2074_outer_file_sha256": sha256(review_outer),
            },
            "attempt_identity": {
                "attempt_json_sha256": sha256(ATTEMPT / "attempt.json"),
                "attempt_manifest_sha256": sha256(attempt_manifest),
                "attempt_outer_file_sha256": sha256(attempt_outer),
            },
            "claim_boundary": {
                "directed_weights": True, "component_workloads": True,
                "full_fc_wall_time": False, "system_speedup": False,
                "energy": False, "paper_admitted": False,
            },
        }
        RUN_STATE["phase"] = "RESULT_STAGE_WRITE"
        RUN_STATE["command"] = ["internal", "write_result_and_copy_compile_log"]
        write_json(STAGE / "result.json", result)
        shutil.copy2(compile_log, STAGE / "vcs_compile.log")
        verify_authority()
        RUN_STATE["phase"] = "RESULT_SEAL"
        RUN_STATE["command"] = ["internal", "seal_result_stage"]
        seal_dir(STAGE)
        RUN_STATE["phase"] = "WORK_CLEANUP"
        RUN_STATE["command"] = ["internal", "remove_private_work"]
        shutil.rmtree(WORK)
        RUN_STATE["phase"] = "RESULT_PUBLISH"
        RUN_STATE["command"] = ["internal", "atomic_publish_no_replace"]
        publish_no_replace(STAGE, RESULT)
        RESULT_PUBLISHED = True
        RUN_STATE["phase"] = "COMPLETE_PENDING_RESULT_HAMMER"
    print("PASS_M2067_ONE_SHOT_COMPLETE_PENDING_INDEPENDENT_RESULT_HAMMER")
    return 0


if __name__ == "__main__":
    try:
        return_code = main()
    except BaseException as exc:
        publish_failure_quarantine(exc)
        raise
    raise SystemExit(return_code)
