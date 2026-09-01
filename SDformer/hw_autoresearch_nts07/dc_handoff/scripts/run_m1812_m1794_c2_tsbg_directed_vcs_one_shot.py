#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""One-shot M1812/M1794 directed VCS campaign; inert before M1813/M1814."""
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
CHECKER = HW / "system_simulator/scripts/check_m1812_m1794_c2_tsbg_production_campaign_source.py"
SPEC = importlib.util.spec_from_file_location("m1812_checker", str(CHECKER))
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("M1812 checker unavailable")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)

CONTRACT = CHECK.CONTRACT
CONTRACT_SIDECAR = Path(str(CONTRACT) + ".sha256")
CONTRACT_OUTER = Path(str(CONTRACT) + ".sha256.seal.sha256")
FILELIST = CHECK.FILELIST
DOC359 = CHECK.DOC359
M1794_CONTRACT = CHECK.M1794_CONTRACT
M1795_REVIEW = CHECK.M1795 / "review.json"
TOP = "tb_m1794_c2_tsbg_b8_real_channel_signed_frontend"

M1813 = HW / "reviews/m1813_m1812_m1794_c2_tsbg_production_campaign_source_hammer_r1_20260902"
M1814 = HW / "contracts/m1814_m1813_m1812_m1794_c2_tsbg_directed_vcs_launch_release_r1_20260902.json"
M1814_SIDECAR = Path(str(M1814) + ".sha256")
M1814_OUTER = Path(str(M1814) + ".sha256.seal.sha256")

VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
LICENSE_SERVER = "27030@ic.ismd-nemo"
LICENSE_FILE = Path("/opt/synopsys/Synopsys.dat")

ATTEMPT = HW / "results/.m1812_m1794_tsbg_directed_vcs_attempt_consumed"
RESULT = HW / "results/m1812_m1794_tsbg_directed_vcs_r1_20260902"
FAILURE = HW / "results/m1812_m1794_tsbg_directed_vcs_r1_20260902.failed_or_incomplete.quarantine"
PRIVATE = HW / "results/m1812_m1794_tsbg_directed_vcs_r1_20260902.private_build.unsealed_do_not_cite"
WORK = HW / ("results/.m1812_m1794_tsbg_directed_vcs_work." + str(os.getpid()))
STAGE = HW / ("results/.m1812_m1794_tsbg_directed_vcs_stage." + str(os.getpid()))
FAIL_STAGE = HW / ("results/.m1812_m1794_tsbg_directed_vcs_failure_stage." + str(os.getpid()))
LOCK = Path("/tmp/m1812_m1794_tsbg_directed_vcs.lock")
SHARED_QUEUE = Path("/tmp/date_dual_synopsys_same_uid_eda_queue.lock")

COUNTS = {"license_queries": 1, "vcs_compiles": 1, "simv_runs": 1}
PRELAUNCH_CLAIMS = dict(CHECK.CLAIMS)
RELEASE_BOUNDARY = {
    "behavioral_rtl_directed_only": True,
    "source_groups_elaborated": 12,
    "production_source_groups_proof": 48,
    "checkpoint_capture": False,
    "mapped_gate": False,
    "timing_simulation": False,
    "dc_or_ptpx": False,
}
ATTEMPT_UNIQUENESS = {
    "attempt_latch": str(ATTEMPT.relative_to(HW)),
    "canonical_result": str(RESULT.relative_to(HW)),
    "failure_result": str(FAILURE.relative_to(HW)),
    "private_build": str(PRIVATE.relative_to(HW)),
    "prelaunch_namespaces_required_absent": True,
    "no_replace_atomic_publish": True,
    "failure_quarantine_only_after_attempt_consumed": True,
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


def verify_directory_seal(root, manifest_sha, outer_sha):
    root = Path(root)
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    exact(manifest, manifest_sha)
    exact(outer, outer_sha)
    if outer.read_text().split() != [sha(manifest), "SHA256SUMS"]:
        raise Failure("review outer seal content")
    mapping = {}
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        if len(fields) != 2:
            raise Failure("review manifest syntax")
        rel = Path(fields[1].lstrip("*"))
        name = rel.as_posix()
        if rel.is_absolute() or ".." in rel.parts or name in mapping:
            raise Failure("unsafe review manifest")
        exact(root / rel, fields[0])
        mapping[name] = fields[0]
    if mapping.get("review.json") != sha(root / "review.json"):
        raise Failure("review.json not transitively sealed")


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


def verify_authority():
    exact(RUNNER, authority_pin("M1812_EXPECTED_RUNNER_SHA256"))
    exact(CONTRACT, authority_pin("M1812_EXPECTED_SOURCE_CONTRACT_SHA256"))
    verify_contract_double_seal()
    verify_directory_seal(
        M1813,
        authority_pin("M1812_EXPECTED_M1813_MANIFEST_SHA256"),
        authority_pin("M1812_EXPECTED_M1813_OUTER_FILE_SHA256"))
    exact(M1813 / "review.json",
          authority_pin("M1812_EXPECTED_M1813_REVIEW_SHA256"))
    verify_file_double_seal(
        M1814, M1814_SIDECAR, M1814_OUTER,
        authority_pin("M1812_EXPECTED_M1814_RELEASE_SHA256"),
        authority_pin("M1812_EXPECTED_M1814_SIDECAR_SHA256"),
        authority_pin("M1812_EXPECTED_M1814_OUTER_FILE_SHA256"))

    review = strict_json(M1813 / "review.json")
    release = strict_json(M1814)
    if review.get("status") != "PASS_M1813_M1812_TSBG_PRODUCTION_CAMPAIGN_SOURCE_HAMMER__AUTHORIZE_ONE_FRESH_DIRECTED_VCS":
        raise Failure("M1813 status")
    if review.get("severity_counts") != {"p0": 0, "p1": 0, "p2": 0}:
        raise Failure("M1813 severity")
    if release.get("schema") != "m1814_m1813_m1812_m1794_c2_tsbg_directed_vcs_launch_release_r1_v1":
        raise Failure("M1814 schema")
    if release.get("status") != "AUTHORIZE_ONE_FRESH_M1812_M1794_TSBG_DIRECTED_VCS_CAMPAIGN":
        raise Failure("M1814 status")
    expected_identity = {
        "runner_sha256": sha(RUNNER),
        "source_contract_sha256": sha(CONTRACT),
        "source_contract_sidecar_sha256": sha(CONTRACT_SIDECAR),
        "source_contract_outer_file_sha256": sha(CONTRACT_OUTER),
        "source_review_json_sha256": sha(M1813 / "review.json"),
        "source_review_manifest_sha256": sha(M1813 / "SHA256SUMS"),
        "source_review_outer_file_sha256": sha(
            M1813 / "SHA256SUMS.seal.sha256"),
        "m1794_source_contract_sha256": sha(M1794_CONTRACT),
        "m1795_review_sha256": sha(M1795_REVIEW),
        "docs359_sha256": sha(DOC359),
    }
    if release.get("identity") != expected_identity:
        raise Failure("M1814 transitive identity")
    if release.get("prelaunch_claim_boundary") != PRELAUNCH_CLAIMS:
        raise Failure("M1814 prelaunch claim boundary")
    if release.get("measurement_boundary") != RELEASE_BOUNDARY:
        raise Failure("M1814 measurement boundary")
    if release.get("attempt_uniqueness") != ATTEMPT_UNIQUENESS:
        raise Failure("M1814 attempt uniqueness")
    if release.get("fresh_execution_budget") != dict(
            COUNTS, automatic_retry=False, reuse_prior_simv=False):
        raise Failure("M1814 budget")
    if release.get("authorization") != {
            "launch_m1812_once": True,
            "automatic_retry": False,
            "publish_only_after_all_gates": True,
            "result_hammer_still_required": True}:
        raise Failure("M1814 authorization")


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
    if values.get("MemAvailable", 0) < 16 * 1024 * 1024:
        raise Failure("MemAvailable below 16 GiB")
    if values.get("SwapFree", 0) < 8 * 1024 * 1024:
        raise Failure("SwapFree below 8 GiB")
    if values.get("CommitLimit", 0) - values.get("Committed_AS", 0) < 16 * 1024 * 1024:
        raise Failure("commit headroom below 16 GiB")
    if shutil.disk_usage(HW / "results").free < 12 * 1024 * 1024 * 1024:
        raise Failure("result disk free below 12 GiB")


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


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n")


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


def publish_no_replace(source, destination):
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p,
                          ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    if renameat2(-100, os.fsencode(source), -100,
                 os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def main():
    if len(sys.argv) != 1:
        raise Failure("M1812 accepts no arguments")
    state = {"phase": "SOURCE_CHAIN", "attempt": False, "complete": False,
             "license_queries": 0, "vcs_compiles": 0, "simv_runs": 0}
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
        state["license_queries"] += 1
        probe = subprocess.run([str(LMUTIL), "lmstat", "-a", "-c", LICENSE_SERVER],
                               env=clean_env({}), stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=60, check=False)
        if probe.returncode != 0:
            raise Failure("license preflight")

        ATTEMPT.mkdir()
        state["attempt"] = True
        write_json(ATTEMPT / "attempt.json", {
            "status": "M1812_M1794_TSBG_DIRECTED_VCS_ATTEMPT_CONSUMED",
            "budget": COUNTS, "attempt_uniqueness": ATTEMPT_UNIQUENESS,
            "automatic_retry": False, "reuse_prior_simv": False})
        seal_dir(ATTEMPT)
        WORK.mkdir(); (WORK / "build").mkdir(); (WORK / "candidate").mkdir()
        build = WORK / "build"; candidate = WORK / "candidate"

        state["phase"] = "VCS_COMPILE"
        state["vcs_compiles"] += 1
        run([str(VCS), "-full64", "-sverilog", "+v2k", "-timescale=1ns/1ps",
             "-assert", "svaext", "-lca", "+vcs+lic+wait", "-Mdir=csrc",
             "-f", str(FILELIST), "-top", TOP, "-o", "simv"],
            build, clean_env({"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
                              "VCS_ARCH_OVERRIDE": "linux"}), 7200,
            build / "compile.log")
        if not (build / "simv").is_file():
            raise Failure("simv absent")

        state["phase"] = "DIRECTED_SIM"
        state["simv_runs"] += 1
        sim_log = candidate / "directed_sim.log"
        run(["./simv", "-lca"], build,
            clean_env({"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
                       "VCS_ARCH_OVERRIDE": "linux"}), 14400, sim_log)
        runtime = CHECK.validate_runtime(sim_log)
        if any(state[key] != value for key, value in COUNTS.items()):
            raise Failure("execution count drift")

        STAGE.mkdir()
        shutil.copy2(build / "compile.log", STAGE / "compile.log")
        shutil.copy2(sim_log, STAGE / "directed_sim.log")
        write_json(STAGE / "runtime.json", runtime)
        write_json(STAGE / "receipt.json", {
            "schema": "m1812_m1794_c2_tsbg_directed_vcs_candidate_receipt_r1_v1",
            "status": "PASS_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "one_shot": dict(COUNTS, automatic_retry=False,
                              reuse_prior_simv=False),
            "identity": {"runner_sha256": sha(RUNNER),
                         "source_contract_sha256": sha(CONTRACT),
                         "source_review_json_sha256": sha(M1813 / "review.json"),
                         "launch_release_sha256": sha(M1814),
                         "m1794_source_contract_sha256": sha(M1794_CONTRACT),
                         "m1795_review_sha256": sha(M1795_REVIEW)},
            "runtime": runtime,
            "claim_boundary": runtime["claim_boundary"]})
        (STAGE / "RUN_COMPLETE.txt").write_text(
            "PASS_M1812_M1794_TSBG_DIRECTED_VCS_CANDIDATE_PENDING_RESULT_HAMMER\n")
        seal_dir(STAGE)
        publish_no_replace(WORK, PRIVATE)
        publish_no_replace(STAGE, RESULT)
        state["complete"] = True
        print("PASS_M1812_M1794_TSBG_DIRECTED_VCS_CANDIDATE_PENDING_RESULT_HAMMER")
        return 0
    except BaseException as error:
        if state["attempt"] and not state["complete"]:
            try:
                FAIL_STAGE.mkdir(exist_ok=False)
                write_json(FAIL_STAGE / "failure.json", {
                    "status": "FAILED_OR_INCOMPLETE_DO_NOT_RETRY",
                    "phase": state["phase"], "error": type(error).__name__,
                    "attempt_consumed": True,
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
