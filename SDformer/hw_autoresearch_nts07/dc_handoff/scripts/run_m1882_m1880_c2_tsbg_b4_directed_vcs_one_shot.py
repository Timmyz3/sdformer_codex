#!/usr/bin/env python3
"""One-shot M1882/M1880 B4 TSBG VCS campaign; inert before M1884-86."""
from __future__ import print_function

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
ROOT = HW.parent
RUNNER = Path(__file__).resolve()
CHECKER = HW / "system_simulator/scripts/check_m1882_m1880_c2_tsbg_b4_campaign_source.py"
SPEC = importlib.util.spec_from_file_location("m1882_checker_runtime", str(CHECKER))
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("M1882 checker unavailable")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)

CONTRACT = CHECK.CONTRACT
CONTRACT_SIDECAR = Path(str(CONTRACT) + ".sha256")
CONTRACT_OUTER = Path(str(CONTRACT) + ".sha256.seal.sha256")
FILELIST = CHECK.FILELIST
TOP = "tb_m1880_c2_tsbg_b4_real_channel_signed_frontend"

M1884 = HW / "reviews/m1884_m1882_m1880_c2_tsbg_b4_campaign_source_hammer_r1_20260902"
M1885 = HW / "contracts/m1885_m1884_m1882_m1880_c2_tsbg_b4_directed_vcs_launch_release_r1_20260902.json"
M1885_SIDECAR = Path(str(M1885) + ".sha256")
M1885_OUTER = Path(str(M1885) + ".sha256.seal.sha256")
M1886 = HW / "reviews/m1886_m1885_c2_tsbg_b4_launch_release_audit_r1_20260902"

VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
LICENSE_SERVER = "27030@ic.ismd-nemo"
LICENSE_FILE = Path("/opt/synopsys/Synopsys.dat")

ATTEMPT = HW / "results/.m1882_m1880_c2_tsbg_b4_directed_vcs_attempt_consumed"
RESULT = HW / "results/m1882_m1880_c2_tsbg_b4_directed_vcs_r1_20260902"
FAILURE = HW / "results/m1882_m1880_c2_tsbg_b4_directed_vcs_r1_20260902.failed_or_incomplete.quarantine"
PRIVATE = HW / "results/m1882_m1880_c2_tsbg_b4_directed_vcs_r1_20260902.private_build.unsealed_do_not_cite"
WORK = HW / ("results/.m1882_m1880_c2_tsbg_b4_directed_vcs_work." + str(os.getpid()))
STAGE = HW / ("results/.m1882_m1880_c2_tsbg_b4_directed_vcs_stage." + str(os.getpid()))
FAIL_STAGE = HW / ("results/.m1882_m1880_c2_tsbg_b4_directed_vcs_failure_stage." + str(os.getpid()))
LOCK = Path("/tmp/m1882_m1880_c2_tsbg_b4_directed_vcs.lock")
SHARED_QUEUE = Path("/tmp/date_dual_synopsys_same_uid_eda_queue.lock")

COUNTS = {"license_queries": 1, "vcs_compiles": 1, "simv_runs": 1}
CLAIMS = dict(CHECK.CLAIMS)
MEASUREMENT_BOUNDARY = {
    "behavioral_rtl_directed_only": True,
    "source_groups_elaborated": 12,
    "production_source_groups_proof": 48,
    "checkpoint_capture": False,
    "mapped_gate": False,
    "timing_simulation": False,
    "dc_or_ptpx": False,
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
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       object_pairs_hook=pairs,
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


def verify_directory_seal(root, manifest_sha, outer_file_sha):
    root = Path(root)
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    exact(manifest, manifest_sha)
    exact(outer, outer_file_sha)
    if outer.read_text(encoding="ascii").split() != [sha(manifest), "SHA256SUMS"]:
        raise Failure("directory outer seal content")
    mapping = {}
    for row in manifest.read_text(encoding="ascii").splitlines():
        fields = row.split(maxsplit=1)
        if len(fields) != 2:
            raise Failure("manifest syntax")
        rel = Path(fields[1].lstrip("*"))
        name = rel.as_posix()
        if rel.is_absolute() or ".." in rel.parts or name in mapping:
            raise Failure("unsafe review manifest")
        exact(root / rel, fields[0])
        mapping[name] = fields[0]
    if mapping.get("review.json") != sha(root / "review.json"):
        raise Failure("review not transitively sealed")


def verify_file_double_seal(path, sidecar, outer, file_sha, sidecar_sha,
                            outer_file_sha):
    exact(path, file_sha)
    exact(sidecar, sidecar_sha)
    exact(outer, outer_file_sha)
    if sidecar.read_text(encoding="ascii").split() != [sha(path), path.name]:
        raise Failure("file sidecar content")
    if outer.read_text(encoding="ascii").split() != [sha(sidecar), sidecar.name]:
        raise Failure("file outer seal content")


def verify_contract_double_seal():
    verify_file_double_seal(
        CONTRACT, CONTRACT_SIDECAR, CONTRACT_OUTER,
        authority_pin("M1882_EXPECTED_SOURCE_CONTRACT_SHA256"),
        sha(CONTRACT_SIDECAR), sha(CONTRACT_OUTER))


def expected_release_identity():
    value = dict(CHECK.UPSTREAM_IDENTITY)
    value.update({
        "runner_sha256": sha(RUNNER),
        "source_contract_sha256": sha(CONTRACT),
        "source_contract_sidecar_sha256": sha(CONTRACT_SIDECAR),
        "source_contract_outer_file_sha256": sha(CONTRACT_OUTER),
        "source_review_json_sha256": sha(M1884 / "review.json"),
        "source_review_manifest_sha256": sha(M1884 / "SHA256SUMS"),
        "source_review_outer_file_sha256": sha(
            M1884 / "SHA256SUMS.seal.sha256"),
    })
    return value


def verify_authority():
    exact(RUNNER, authority_pin("M1882_EXPECTED_RUNNER_SHA256"))
    exact(CONTRACT, authority_pin("M1882_EXPECTED_SOURCE_CONTRACT_SHA256"))
    verify_contract_double_seal()
    verify_directory_seal(
        M1884,
        authority_pin("M1882_EXPECTED_M1884_MANIFEST_SHA256"),
        authority_pin("M1882_EXPECTED_M1884_OUTER_FILE_SHA256"))
    exact(M1884 / "review.json",
          authority_pin("M1882_EXPECTED_M1884_REVIEW_SHA256"))
    verify_file_double_seal(
        M1885, M1885_SIDECAR, M1885_OUTER,
        authority_pin("M1882_EXPECTED_M1885_RELEASE_SHA256"),
        authority_pin("M1882_EXPECTED_M1885_SIDECAR_SHA256"),
        authority_pin("M1882_EXPECTED_M1885_OUTER_FILE_SHA256"))
    verify_directory_seal(
        M1886,
        authority_pin("M1882_EXPECTED_M1886_MANIFEST_SHA256"),
        authority_pin("M1882_EXPECTED_M1886_OUTER_FILE_SHA256"))
    exact(M1886 / "review.json",
          authority_pin("M1882_EXPECTED_M1886_REVIEW_SHA256"))

    source_review = strict_json(M1884 / "review.json")
    release = strict_json(M1885)
    release_audit = strict_json(M1886 / "review.json")
    if source_review.get("status") != (
            "PASS_M1884_M1882_C2_TSBG_B4_CAMPAIGN_SOURCE_HAMMER__"
            "AUTHORIZE_RELEASE_SOURCE_ONLY"):
        raise Failure("M1884 status")
    if source_review.get("severity_counts") != {"p0": 0, "p1": 0, "p2": 0}:
        raise Failure("M1884 severity")
    if release.get("schema") != (
            "m1885_m1884_m1882_m1880_c2_tsbg_b4_directed_vcs_"
            "launch_release_r1_v1"):
        raise Failure("M1885 schema")
    if release.get("status") != (
            "AUTHORIZE_ONE_FRESH_M1882_M1880_C2_TSBG_B4_DIRECTED_VCS_CAMPAIGN"):
        raise Failure("M1885 status")
    if release.get("identity") != expected_release_identity():
        raise Failure("M1885 transitive identity")
    if release.get("prelaunch_claim_boundary") != CLAIMS:
        raise Failure("M1885 claim boundary")
    if release.get("measurement_boundary") != MEASUREMENT_BOUNDARY:
        raise Failure("M1885 measurement boundary")
    if release.get("fresh_execution_budget") != dict(
            COUNTS, automatic_retry=False, reuse_prior_simv=False):
        raise Failure("M1885 execution budget")
    if release.get("authorization") != {
            "launch_m1882_once_after_m1886_audit": True,
            "automatic_retry": False,
            "publish_only_after_all_gates": True,
            "result_hammer_still_required": True}:
        raise Failure("M1885 authorization")
    if release_audit.get("status") != (
            "PASS_M1886_M1885_C2_TSBG_B4_LAUNCH_RELEASE_AUDIT__"
            "AUTHORIZE_ONE_M1882_ATTEMPT"):
        raise Failure("M1886 status")
    if release_audit.get("severity_counts") != {"p0": 0, "p1": 0, "p2": 0}:
        raise Failure("M1886 severity")
    if release_audit.get("audited_identity") != {
            "runner_sha256": sha(RUNNER),
            "source_contract_sha256": sha(CONTRACT),
            "source_review_json_sha256": sha(M1884 / "review.json"),
            "release_sha256": sha(M1885),
            "release_sidecar_sha256": sha(M1885_SIDECAR),
            "release_outer_file_sha256": sha(M1885_OUTER)}:
        raise Failure("M1886 audited identity")


def exact_result_path(path):
    path = Path(path)
    resolved_parent = path.parent.resolve(strict=True)
    if resolved_parent != (HW / "results").resolve(strict=True):
        raise Failure("result path provenance " + str(path))
    if path.is_symlink():
        raise Failure("result path symlink " + str(path))


def namespaces_fresh():
    fixed = (ATTEMPT, RESULT, FAILURE, PRIVATE, WORK, STAGE, FAIL_STAGE)
    for path in fixed:
        exact_result_path(path)
        if os.path.lexists(str(path)):
            raise Failure("namespace residue " + str(path))
    prefixes = (
        ".m1882_m1880_c2_tsbg_b4_directed_vcs_work.*",
        ".m1882_m1880_c2_tsbg_b4_directed_vcs_stage.*",
        ".m1882_m1880_c2_tsbg_b4_directed_vcs_failure_stage.*",
    )
    for pattern in prefixes:
        if any((HW / "results").glob(pattern)):
            raise Failure("prior private build or simv namespace " + pattern)


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


def clean_env(extra=None):
    value = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
             "SNPSLMD_LICENSE_FILE": LICENSE_SERVER,
             "LM_LICENSE_FILE": str(LICENSE_FILE)}
    if extra:
        value.update(extra)
    return value


def run_tool(command, cwd, timeout, output):
    CHECK.validate_sources()
    collision_gate()
    with Path(output).open("wb") as stream:
        completed = subprocess.run(command, cwd=cwd, env=clean_env(),
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   timeout=timeout, check=False)
    if completed.returncode != 0:
        raise Failure("tool failure " + Path(command[0]).name)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n", encoding="utf-8")


def seal_dir(root):
    rows = []
    for path in Path(root).rglob("*"):
        if path.is_symlink():
            raise Failure("symlink in candidate")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            rows.append((path.relative_to(root).as_posix(), sha(path)))
    rows.sort()
    manifest = Path(root) / "SHA256SUMS"
    manifest.write_text("".join(digest + "  " + name + "\n"
                                for name, digest in rows), encoding="ascii")
    (Path(root) / "SHA256SUMS.seal.sha256").write_text(
        sha(manifest) + "  SHA256SUMS\n", encoding="ascii")


def publish_no_replace(source, destination):
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p,
                          ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    if renameat2(-100, os.fsencode(source), -100,
                 os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def attempt_terminal_gate(state):
    if not state["attempt"]:
        return
    success = RESULT.is_dir() and not RESULT.is_symlink()
    failure = FAILURE.is_dir() and not FAILURE.is_symlink()
    if success == failure:
        raise Failure("attempt must terminate in exactly one sealed namespace")


def main():
    if len(sys.argv) != 1:
        raise Failure("M1882 accepts no arguments")
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
        license_check = subprocess.run(
            [str(LMUTIL), "lmstat", "-a", "-c", LICENSE_SERVER],
            env=clean_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=120, check=False)
        if license_check.returncode != 0:
            raise Failure("single license preflight failed before attempt")
        if state["license_queries"] != COUNTS["license_queries"]:
            raise Failure("license query budget")

        state["phase"] = "ATTEMPT_CONSUMED"
        ATTEMPT.mkdir()
        state["attempt"] = True
        WORK.mkdir()
        STAGE.mkdir()
        (WORK / "license_preflight.log").write_bytes(license_check.stdout)

        state["phase"] = "VCS_COMPILE"
        state["vcs_compiles"] += 1
        simv = WORK / "simv"
        run_tool([
            str(VCS), "-full64", "-sverilog", "-assert", "svaext",
            "-top", TOP, "-f", str(FILELIST), "-o", str(simv),
            "-Mdir=" + str(WORK / "csrc")],
            ROOT, 3600, WORK / "vcs_compile.log")
        if state["vcs_compiles"] != COUNTS["vcs_compiles"]:
            raise Failure("VCS compile budget")

        state["phase"] = "SIMV"
        state["simv_runs"] += 1
        run_tool([str(simv)], ROOT, 3600, WORK / "simv.log")
        if state["simv_runs"] != COUNTS["simv_runs"]:
            raise Failure("simv budget")
        log = (WORK / "simv.log").read_text(encoding="utf-8", errors="replace")
        token = "PASS_M1880_C2_TSBG_B4_REAL_M803_TYPED_SIGNED_DIRECTED"
        if log.count(token) != 1:
            raise Failure("unique directed PASS absent")
        for forbidden in ("Assertion failed", "Error-", "$fatal", "Fatal:"):
            if forbidden in log:
                raise Failure("directed log failure token " + forbidden)

        state["phase"] = "SEAL"
        shutil.copy2(WORK / "license_preflight.log", STAGE / "license_preflight.log")
        shutil.copy2(WORK / "vcs_compile.log", STAGE / "vcs_compile.log")
        shutil.copy2(WORK / "simv.log", STAGE / "simv.log")
        write_json(STAGE / "receipt.json", {
            "schema": "m1882_m1880_c2_tsbg_b4_directed_vcs_receipt_r1_v1",
            "status": "RAW_PASS_AWAIT_DIFFERENT_AUTHOR_RESULT_HAMMER",
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "counts": dict(state),
            "source_sha256": dict(CHECK.SOURCE_SHA256),
            "upstream_identity": dict(CHECK.UPSTREAM_IDENTITY),
            "measurement_boundary": MEASUREMENT_BOUNDARY,
            "claim_boundary": CLAIMS,
            "result_hammer_required": True,
            "paper_admitted": False,
        })
        (STAGE / "RUN_COMPLETE.txt").write_text(
            "RAW_PASS_M1882_M1880_C2_TSBG_B4_DIRECTED_VCS__"
            "AWAIT_DIFFERENT_AUTHOR_RESULT_HAMMER\n", encoding="ascii")
        seal_dir(STAGE)
        publish_no_replace(STAGE, RESULT)
        state["complete"] = True
        state["phase"] = "RAW_PASS"
        attempt_terminal_gate(state)
        return 0
    except Exception as error:
        state["phase"] = "FAILED"
        if state["attempt"] and not state["complete"]:
            if os.path.lexists(str(FAILURE)):
                raise
            FAIL_STAGE.mkdir()
            write_json(FAIL_STAGE / "failure_receipt.json", {
                "schema": "m1882_m1880_c2_tsbg_b4_directed_vcs_failure_r1_v1",
                "status": "FAILED_OR_INCOMPLETE_DO_NOT_CITE_NO_RETRY",
                "date_utc": datetime.now(timezone.utc).isoformat(),
                "state": state,
                "error_type": type(error).__name__,
                "error": str(error),
                "claim_boundary": CLAIMS,
                "automatic_retry": False,
            })
            if WORK.exists():
                for name in ("license_preflight.log", "vcs_compile.log", "simv.log"):
                    source = WORK / name
                    if source.is_file() and not source.is_symlink():
                        shutil.copy2(source, FAIL_STAGE / name)
            seal_dir(FAIL_STAGE)
            publish_no_replace(FAIL_STAGE, FAILURE)
            attempt_terminal_gate(state)
        raise
    finally:
        for path in (WORK, STAGE, FAIL_STAGE):
            if path.exists() and not path.is_symlink():
                shutil.rmtree(path)
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            fcntl.flock(queue_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()
            queue_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
