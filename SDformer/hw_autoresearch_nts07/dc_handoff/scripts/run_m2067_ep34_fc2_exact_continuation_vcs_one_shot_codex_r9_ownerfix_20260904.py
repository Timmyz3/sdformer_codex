#!/opt/anaconda3/bin/python3.12
"""M2067 R9 owner-safe full-960 Synopsys VCS one-shot.

This successor reruns slots 0..959 from scratch.  It inherits no R8 logs.  The
owner lock is acquired before namespace inspection and failure publication is
permitted only after this PID creates an owner-bound attempt marker.
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
import secrets
import shutil
import subprocess
import sys


HW = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
R8_RUNNER = HW / (
    "dc_handoff/scripts/run_m2067_ep34_fc2_exact_continuation_vcs_one_shot_"
    "codex_r8_20260904.py")
PARSER_PATH = HW / (
    "system_simulator/scripts/parse_m2067_ep34_fc2_exact_continuation_vcs_"
    "codex_r8_20260904.py")
SPEC = importlib.util.spec_from_file_location("m2067_r8_parser", PARSER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("M2067 parser unavailable")
PARSER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARSER)

CONTRACT = HW / (
    "contracts/m2067_ep34_fc2_exact_continuation_vcs_source_contract_r9_"
    "codex_ownerfix_20260904.json")
M2082 = HW / (
    "reviews/m2082_m2067_ep34_fc2_exact_continuation_vcs_source_r8_hammer_"
    "r1_20260904")
M2083 = HW / (
    "reviews/m2083_m2067_r8_external_interrupt_failure_hammer_r1_20260904")
M2084 = HW / (
    "reviews/m2084_m2067_ep34_fc2_exact_continuation_vcs_source_r9_hammer_"
    "r1_20260904")
FILELIST = HW / (
    "dc_handoff/filelists/iscas_m2067_ep34_fc2_exact_continuation_vcs_"
    "codex_zeroaware_r8_20260904.f")
TOP = "tb_m2067_ep34_fc2_exact_continuation_s960"
PYTHON = Path("/opt/anaconda3/bin/python3.12")
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
LICENSE_SERVER = "27030@ic.ismd-nemo"
LICENSE_FILE = Path("/opt/synopsys/Synopsys.dat")
TOOL_SHA256 = {
    PYTHON: "873a1168d6d2a7d1b406b85c2a1ea986a6f086041069ab1ee3f70b9217f10161",
    VCS: "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
    LMUTIL: "e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07",
}
R8_RUNNER_SHA256 = (
    "711d4e2d379d8af1bdd1408d504c3e7317c9d75e8c0c90d2581021371ed52783")
ATTEMPT = HW / (
    "results/.m2067_ep34_fc2_exact_continuation_vcs_r9_codex_ownerfix_"
    "attempt_consumed")
RESULT = HW / (
    "results/m2067_ep34_fc2_exact_continuation_vcs_r9_codex_ownerfix_"
    "20260904")
FAILURE = HW / (
    "results/m2067_ep34_fc2_exact_continuation_vcs_r9_codex_ownerfix_"
    "20260904.failed_or_incomplete.quarantine")
WORK = HW / (
    "results/.m2067_ep34_fc2_exact_continuation_r9_codex_ownerfix_work."
    + str(os.getpid()))
STAGE = HW / (
    "results/.m2067_ep34_fc2_exact_continuation_r9_codex_ownerfix_stage."
    + str(os.getpid()))
FAIL_STAGE = HW / (
    "results/.m2067_ep34_fc2_exact_continuation_r9_codex_ownerfix_failure."
    + str(os.getpid()))
OWNER_LOCK = Path(
    "/tmp/m2067_ep34_fc2_exact_continuation_vcs_r9_codex_ownerfix.lock")
EDA_QUEUE = Path("/tmp/date_dual_synopsys_same_uid_eda_queue.lock")
ATTEMPT_OWNED = False
OWNER_NONCE = ""
RESULT_PUBLISHED = False
LMSTAT_OUTPUT = b""
RUN_STATE = {
    "phase": "PRE_AUTHORITY", "license_preflight_lmstat": 0,
    "vcs_compiles": 0, "simv_runs": 0, "completed_slots": 0,
    "current_slot": None,
}


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
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
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
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           Failure("nonfinite JSON " + token)))
    if type(value) is not dict:
        raise Failure("JSON root " + str(path))
    return value


def sealed_directory(root: Path, manifest_pin: str, outer_pin: str) -> dict:
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    exact(manifest, manifest_pin)
    exact(outer, outer_pin)
    if outer.read_text().split() != [sha256(manifest), "SHA256SUMS"]:
        raise Failure("outer seal content " + str(root))
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
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.name not in {
                  "SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    if any(p.is_symlink() for p in root.rglob("*")) or actual != set(mapping):
        raise Failure("non-exhaustive or linked seal " + str(root))
    return mapping


def verify_review(root: Path, prefix: str, status_prefix: str) -> dict:
    mapping = sealed_directory(
        root, authority(prefix + "_MANIFEST_SHA256"),
        authority(prefix + "_OUTER_FILE_SHA256"))
    review_path = root / "review.json"
    exact(review_path, authority(prefix + "_REVIEW_SHA256"))
    if mapping.get("review.json") != sha256(review_path):
        raise Failure("review is not sealed " + str(root))
    review = strict_json(review_path)
    if not review.get("status", "").startswith(status_prefix):
        raise Failure("review status " + str(root))
    return review


def verify_authority() -> None:
    exact(RUNNER, authority("M2067_R9_EXPECTED_RUNNER_SHA256"))
    exact(R8_RUNNER, R8_RUNNER_SHA256)
    exact(PARSER_PATH, authority("M2067_R9_EXPECTED_PARSER_SHA256"))
    exact(CONTRACT, authority("M2067_R9_EXPECTED_CONTRACT_SHA256"))
    m2082 = verify_review(M2082, "M2067_R9_EXPECTED_M2082", "PASS_M2082_")
    r8_identity = m2082.get("reviewed_source_identity", {})
    if (r8_identity.get("runner_sha256") != R8_RUNNER_SHA256
            or r8_identity.get("parser_sha256") != sha256(PARSER_PATH)):
        raise Failure("M2082 R8 source identity")
    m2083 = verify_review(M2083, "M2067_R9_EXPECTED_M2083", "PASS_M2083_")
    if (m2083.get("observed", {}).get("paper_citable") is not False
            or m2083.get("observed", {}).get("completed_parser_valid_slots") != 210):
        raise Failure("M2083 failure lineage")
    m2084 = verify_review(M2084, "M2067_R9_EXPECTED_M2084", "PASS_M2084_")
    identity = m2084.get("reviewed_source_identity", {})
    if (identity.get("runner_sha256") != sha256(RUNNER)
            or identity.get("parser_sha256") != sha256(PARSER_PATH)
            or identity.get("contract_sha256") != sha256(CONTRACT)
            or m2084.get("authorization", {}).get("execute_once") is not True
            or m2084.get("authorization", {}).get("automatic_retry") is not False):
        raise Failure("M2084 R9 authorization")
    PARSER.validate_source()
    for path, digest in TOOL_SHA256.items():
        exact(path, digest)


def fresh_namespaces() -> None:
    for path in (ATTEMPT, RESULT, FAILURE, WORK, STAGE, FAIL_STAGE):
        if os.path.lexists(str(path)):
            raise Failure("namespace residue " + str(path))
    root = HW / "results"
    for pattern in (
        ".m2067_ep34_fc2_exact_continuation_r9_codex_ownerfix_work.*",
        ".m2067_ep34_fc2_exact_continuation_r9_codex_ownerfix_stage.*",
        ".m2067_ep34_fc2_exact_continuation_r9_codex_ownerfix_failure.*"):
        if next(root.glob(pattern), None) is not None:
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


def attempt_owned_by_this_process() -> bool:
    if not ATTEMPT_OWNED or not OWNER_NONCE:
        return False
    try:
        owner = strict_json(ATTEMPT / "owner.json")
    except Exception:
        return False
    return (owner.get("pid") == os.getpid()
            and owner.get("nonce") == OWNER_NONCE
            and owner.get("runner_sha256") == sha256(RUNNER))


def publish_failure_quarantine(exc: BaseException) -> None:
    if RESULT_PUBLISHED or not attempt_owned_by_this_process():
        return
    if os.path.lexists(str(RESULT)) or os.path.lexists(str(FAILURE)):
        return
    FAIL_STAGE.mkdir()
    state = dict(RUN_STATE)
    state.update({
        "status": "FAILED_DO_NOT_CITE_NO_RETRY",
        "failed_utc": datetime.now(timezone.utc).isoformat(),
        "error_type": type(exc).__name__, "error": str(exc),
        "runner_sha256": sha256(RUNNER), "parser_sha256": sha256(PARSER_PATH),
        "contract_sha256": sha256(CONTRACT), "attempt_owned": True,
        "owner_nonce": OWNER_NONCE, "automatic_retry": False,
    })
    write_json(FAIL_STAGE / "failure.json", state)
    evidence = FAIL_STAGE / "evidence"
    evidence.mkdir()
    if LMSTAT_OUTPUT:
        with (evidence / "lmstat.log").open("xb") as stream:
            stream.write(LMSTAT_OUTPUT)
    for root, label in ((WORK, "work"), (STAGE, "stage")):
        if root.is_dir() and not root.is_symlink():
            for path in sorted(root.rglob("*")):
                if (path.is_file() and not path.is_symlink()
                        and path.suffix.lower() in {".log", ".json", ".txt"}):
                    target = evidence / label / path.relative_to(root)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
    seal_dir(FAIL_STAGE)
    publish_no_replace(FAIL_STAGE, FAILURE)


def main() -> int:
    global ATTEMPT_OWNED, OWNER_NONCE, LMSTAT_OUTPUT, RESULT_PUBLISHED
    if len(sys.argv) != 1:
        raise Failure("runner accepts no arguments")
    verify_authority()
    OWNER_LOCK.touch(exist_ok=True)
    EDA_QUEUE.touch(exist_ok=True)
    with OWNER_LOCK.open("r+") as owner_lock, EDA_QUEUE.open("r+") as queue_lock:
        # A lock loser raises while ATTEMPT_OWNED is false and cannot publish
        # another process's failure namespace.
        fcntl.flock(owner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(queue_lock, fcntl.LOCK_EX)
        verify_authority()
        fresh_namespaces()
        collision_gate()
        RUN_STATE["phase"] = "ATTEMPT_CREATE"
        OWNER_NONCE = secrets.token_hex(16)
        ATTEMPT.mkdir()
        ATTEMPT_OWNED = True
        write_json(ATTEMPT / "owner.json", {
            "schema": "m2067_r9_attempt_owner_v1", "pid": os.getpid(),
            "nonce": OWNER_NONCE, "runner_sha256": sha256(RUNNER),
        })
        write_json(ATTEMPT / "attempt.json", {
            "schema": "m2067_ep34_fc2_exact_continuation_attempt_r9_v1",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "runner_sha256": sha256(RUNNER),
            "parser_sha256": sha256(PARSER_PATH),
            "contract_sha256": sha256(CONTRACT),
            "vcs_compiles_budget": 1, "simv_runs_budget": 960,
            "inherited_logs": 0, "automatic_retry": False,
        })
        seal_dir(ATTEMPT)

        RUN_STATE["phase"] = "LICENSE_PREFLIGHT"
        RUN_STATE["license_preflight_lmstat"] = 1
        lmstat_command = [str(LMUTIL), "lmstat", "-a", "-c", LICENSE_SERVER]
        RUN_STATE["command"] = lmstat_command
        completed = subprocess.run(
            lmstat_command, cwd=HW, env=clean_env(), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=120, check=False)
        LMSTAT_OUTPUT = completed.stdout
        if (completed.returncode != 0
                or b"Users of VCSCompiler_Net" not in LMSTAT_OUTPUT):
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
        metadata = PARSER.strict_json(PARSER.META)
        for slot in range(PARSER.WORKLOADS):
            RUN_STATE.update({"phase": "SIMV_SLOT", "current_slot": slot,
                              "simv_runs": slot + 1,
                              "command": ["./simv", "-lca",
                                          f"+WORKLOAD_SLOT={slot}"]})
            log = logs / f"slot_{slot:04d}.log"
            run_checked(RUN_STATE["command"], WORK, log, 21600)
            RUN_STATE["phase"] = "PARSE_SLOT"
            parsed = PARSER.parse_log(log, validate_source_identity=False,
                                      metadata=metadata)
            parsed.pop("source_identity", None)
            rows.append(parsed)
            RUN_STATE["completed_slots"] = slot + 1

        if [row["workload_slot"] for row in rows] != list(range(960)):
            raise Failure("workload exact-once order")
        verify_authority()
        base_cycles = sum(row["base_cycles"] for row in rows)
        tsbg_cycles = sum(row["tsbg_cycles"] for row in rows)
        result = {
            "schema": "m2067_ep34_fc2_exact_continuation_vcs_result_r9_v1",
            "status": "PENDING_INDEPENDENT_RESULT_HAMMER_DO_NOT_CITE",
            "workloads": 960, "inherited_logs": 0,
            "integer_checks_per_axis": sum(r["integer_checks"] for r in rows),
            "ordinary_cycles_observed": base_cycles,
            "tsbg_cycles_observed": tsbg_cycles,
            "rtl_cycle_ratio_observed": base_cycles / tsbg_cycles,
            "rows": rows,
            "source_and_authority_identity": {
                "runner_sha256": sha256(RUNNER),
                "parser_sha256": sha256(PARSER_PATH),
                "contract_sha256": sha256(CONTRACT),
                "m2082_review_sha256": sha256(M2082 / "review.json"),
                "m2083_review_sha256": sha256(M2083 / "review.json"),
                "m2084_review_sha256": sha256(M2084 / "review.json"),
            },
            "attempt_identity": {
                "owner_nonce": OWNER_NONCE,
                "attempt_json_sha256": sha256(ATTEMPT / "attempt.json"),
                "owner_json_sha256": sha256(ATTEMPT / "owner.json"),
                "manifest_sha256": sha256(ATTEMPT / "SHA256SUMS"),
                "outer_file_sha256": sha256(ATTEMPT / "SHA256SUMS.seal.sha256"),
            },
            "claim_boundary": {
                "directed_weights": True, "component_workloads": True,
                "full_fc_wall_time": False, "system_speedup": False,
                "energy": False, "paper_admitted": False,
            },
        }
        write_json(STAGE / "result.json", result)
        shutil.copy2(compile_log, STAGE / "vcs_compile.log")
        with (STAGE / "lmstat.log").open("xb") as stream:
            stream.write(LMSTAT_OUTPUT)
        verify_authority()
        seal_dir(STAGE)
        shutil.rmtree(WORK)
        if os.path.lexists(str(FAILURE)):
            raise Failure("contradictory failure exists before success publish")
        publish_no_replace(STAGE, RESULT)
        RESULT_PUBLISHED = True
    print("PASS_M2067_R9_COMPLETE_PENDING_INDEPENDENT_RESULT_HAMMER")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except BaseException as exc:
        publish_failure_quarantine(exc)
        raise
    raise SystemExit(code)
