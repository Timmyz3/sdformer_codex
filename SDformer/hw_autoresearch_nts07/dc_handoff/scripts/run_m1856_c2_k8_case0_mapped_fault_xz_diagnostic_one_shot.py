#!/usr/bin/env python3
"""One-shot M1856 diagnostic-only mapped X/Z localization.

This runner is inert until a different-author M1857 source review and exact
M1858 launch release exist.  Its sole future budget is one fresh K8 mapped VCS
compile and one M979 case-0 simulation.  It never enables UCLI, SAIF, PTPX,
power, performance, or production-functional admission.
"""
from __future__ import print_function

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
CHECKER = HW / "system_simulator/scripts/check_m1856_c2_k8_case0_mapped_fault_xz_diagnostic_source.py"
SPEC = importlib.util.spec_from_file_location("m1856_checker", str(CHECKER))
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("M1856 checker unavailable")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)

CONTRACT = CHECK.CONTRACT
REVIEW = HW / "reviews/m1857_m1856_c2_k8_case0_mapped_fault_xz_diagnostic_source_hammer_r1_20260902"
RELEASE = HW / "contracts/m1858_m1857_m1856_c2_k8_case0_mapped_fault_xz_diagnostic_launch_release_r1_20260902.json"
M1854 = HW / "reviews/m1854_m1845_c2_mapped_energy_failure_hammer_r1_20260902"
MAPPED = CHECK.MAPPED
FILELIST = CHECK.FILELIST
TOP = CHECK.TOP
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")

ATTEMPT = HW / "results/.m1856_c2_k8_case0_mapped_fault_xz_diagnostic_attempt_consumed"
RESULT = HW / "results/m1856_c2_k8_case0_mapped_fault_xz_diagnostic_r1_20260902"
FAILURE = HW / "results/m1856_c2_k8_case0_mapped_fault_xz_diagnostic_r1_20260902.failed_or_incomplete.quarantine"
WORK = HW / ("results/.m1856_c2_k8_case0_mapped_fault_xz_diagnostic_work." + str(os.getpid()))
STAGE = HW / ("results/.m1856_c2_k8_case0_mapped_fault_xz_diagnostic_stage." + str(os.getpid()))
FAIL_STAGE = HW / ("results/.m1856_c2_k8_case0_mapped_fault_xz_diagnostic_failure_stage." + str(os.getpid()))
QUEUE = Path("/tmp/date_dual_synopsys_same_uid_eda_queue.lock")
LOCAL_LOCK = Path("/tmp/m1856_c2_k8_case0_mapped_fault_xz_diagnostic.lock")

M1854_REVIEW_SHA256 = "9ef9f00091b145e438a03c9039123af198f28ef16f6d3180169361ca6470d0a6"
M1854_MANIFEST_SHA256 = "49176d9165cbfe449f243fe4f76b2e2ae3af1e388398d8556f843e75dbfd10b8"
M1854_OUTER_SHA256 = "28089b517fbe5a7f052dfef98b031cfc9887f8f288d610f917cd3234acc2c1f4"


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


def pin(name):
    value = os.environ.get(name, "")
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise Failure("authority pin absent " + name)
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


def verify_file_double_seal(path, file_sha, sidecar_sha, outer_sha):
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    exact(path, file_sha)
    exact(sidecar, sidecar_sha)
    exact(outer, outer_sha)
    if sidecar.read_text().split() != [sha(path), Path(path).name]:
        raise Failure("sidecar content")
    if outer.read_text().split() != [sha(sidecar), sidecar.name]:
        raise Failure("outer content")


def verify_directory(root, manifest_sha, outer_sha):
    root = Path(root)
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    exact(manifest, manifest_sha)
    exact(outer, outer_sha)
    if outer.read_text().split() != [sha(manifest), "SHA256SUMS"]:
        raise Failure("outer seal")
    mapping = {}
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        if len(fields) != 2:
            raise Failure("manifest syntax")
        rel = Path(fields[1].lstrip("*"))
        if rel.is_absolute() or ".." in rel.parts or rel.as_posix() in mapping:
            raise Failure("unsafe manifest")
        exact(root / rel, fields[0])
        mapping[rel.as_posix()] = fields[0]
    return mapping


def verify_authority():
    CHECK.validate_sources()
    exact(RUNNER, pin("M1856_EXPECTED_RUNNER_SHA256"))
    verify_file_double_seal(
        CONTRACT, pin("M1856_EXPECTED_SOURCE_CONTRACT_SHA256"),
        pin("M1856_EXPECTED_SOURCE_CONTRACT_SIDECAR_SHA256"),
        pin("M1856_EXPECTED_SOURCE_CONTRACT_OUTER_SHA256"))
    m1854_map = verify_directory(M1854, M1854_MANIFEST_SHA256, M1854_OUTER_SHA256)
    exact(M1854 / "review.json", M1854_REVIEW_SHA256)
    if m1854_map.get("review.json") != M1854_REVIEW_SHA256:
        raise Failure("M1854 review not sealed")
    failed = strict_json(M1854 / "review.json")
    if (failed.get("production_admission") != "FAIL_CLOSED"
            or failed.get("severity_counts") != {"p0": 0, "p1": 1, "p2": 0}
            or failed.get("execution_audit", {}).get("automatic_retry") is not False):
        raise Failure("M1854 failure boundary drift")
    review_map = verify_directory(
        REVIEW, pin("M1856_EXPECTED_M1857_MANIFEST_SHA256"),
        pin("M1856_EXPECTED_M1857_OUTER_SHA256"))
    exact(REVIEW / "review.json", pin("M1856_EXPECTED_M1857_REVIEW_SHA256"))
    if review_map.get("review.json") != sha(REVIEW / "review.json"):
        raise Failure("M1857 review not sealed")
    review = strict_json(REVIEW / "review.json")
    if (review.get("status") !=
            "PASS_M1857_M1856_C2_K8_CASE0_MAPPED_FAULT_XZ_DIAGNOSTIC_SOURCE__AUTHORIZE_M1858_RELEASE"
            or review.get("severity_counts") != {"p0": 0, "p1": 0, "p2": 0}):
        raise Failure("M1857 review status")
    verify_file_double_seal(
        RELEASE, pin("M1856_EXPECTED_M1858_RELEASE_SHA256"),
        pin("M1856_EXPECTED_M1858_RELEASE_SIDECAR_SHA256"),
        pin("M1856_EXPECTED_M1858_RELEASE_OUTER_SHA256"))
    release = strict_json(RELEASE)
    if (release.get("status") !=
            "AUTHORIZE_ONE_M1856_C2_K8_CASE0_MAPPED_FAULT_XZ_DIAGNOSTIC_ATTEMPT"
            or release.get("budget") != {
                "vcs_compiles": 1, "simv_runs": 1,
                "ucli_runs": 0, "saif_files": 0, "ptpx_runs": 0,
                "all_other_eda_runs": 0, "automatic_retry": False}):
        raise Failure("M1858 release status/budget")
    expected = {
        "runner_sha256": sha(RUNNER),
        "source_contract_sha256": sha(CONTRACT),
        "source_contract_sidecar_sha256": sha(Path(str(CONTRACT) + ".sha256")),
        "source_contract_outer_sha256": sha(Path(str(CONTRACT) + ".sha256.seal.sha256")),
        "m1857_review_sha256": sha(REVIEW / "review.json"),
        "m1857_manifest_sha256": sha(REVIEW / "SHA256SUMS"),
        "m1857_outer_sha256": sha(REVIEW / "SHA256SUMS.seal.sha256"),
        "mapped_netlist_sha256": CHECK.MAPPED_SHA256,
        "filelist_sha256": CHECK.FILELIST_SHA256,
    }
    if release.get("identity") != expected:
        raise Failure("M1858 transitive identity")
    return sha(RELEASE)


def collision_gate():
    blocked = {"vcs", "vcs1", "vlogan", "simv", "dc_shell", "dc_shell-t",
               "pt_shell", "fm_shell", "icc2_shell", "common_shell_exec",
               "common_shell_exe"}
    hits = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
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


def namespaces_fresh():
    for path in (ATTEMPT, RESULT, FAILURE, WORK, STAGE, FAIL_STAGE):
        if os.path.lexists(str(path)):
            raise Failure("namespace residue " + str(path))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n")


def seal_dir(root):
    rows = []
    for path in Path(root).rglob("*"):
        if path.is_symlink():
            raise Failure("symlink")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            rows.append((path.relative_to(root).as_posix(), sha(path)))
    rows.sort()
    manifest = Path(root) / "SHA256SUMS"
    manifest.write_text("".join(digest + "  " + name + "\n" for name, digest in rows))
    (Path(root) / "SHA256SUMS.seal.sha256").write_text(
        sha(manifest) + "  SHA256SUMS\n")


def publish_no_replace(source, destination):
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint]
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def compile_command():
    return [str(VCS), "-full64", "-sverilog", "+v2k", "-timescale=1ns/1ps",
            "-debug_access+r", "-lca", "+vcs+lic+wait", "-Mdir=csrc",
            str(MAPPED), "-f", str(FILELIST), "-top", TOP, "-o", "simv"]


def run(command, cwd, output, timeout):
    CHECK.validate_sources()
    collision_gate()
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
           "VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
           "VCS_ARCH_OVERRIDE": "linux",
           "SNPSLMD_LICENSE_FILE": "27030@ic.ismd-nemo",
           "LM_LICENSE_FILE": "/opt/synopsys/Synopsys.dat"}
    with Path(output).open("wb") as stream:
        completed = subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, timeout=timeout,
                                   check=False)
    if completed.returncode != 0:
        raise Failure("tool failure " + Path(command[0]).name)


def main():
    if len(sys.argv) != 1:
        raise Failure("M1856 accepts no arguments")
    state = {"phase": "SOURCE_CHAIN", "vcs_compiles": 0, "simv_runs": 0,
             "ucli_runs": 0, "saif_files": 0, "ptpx_runs": 0,
             "all_other_eda_runs": 0, "automatic_retry": False}
    queue_handle = QUEUE.open("a+")
    local_handle = LOCAL_LOCK.open("a+")
    try:
        release_sha = verify_authority()
        namespaces_fresh()
        fcntl.flock(queue_handle.fileno(), fcntl.LOCK_EX)
        fcntl.flock(local_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        collision_gate()
        namespaces_fresh()
        ATTEMPT.mkdir()
        write_json(ATTEMPT / "attempt.json", {
            "status": "M1856_DIAGNOSTIC_ATTEMPT_CONSUMED",
            "release_sha256": release_sha,
            "budget": {"vcs_compiles": 1, "simv_runs": 1,
                       "ucli_runs": 0, "saif_files": 0, "ptpx_runs": 0,
                       "all_other_eda_runs": 0, "automatic_retry": False}})
        seal_dir(ATTEMPT)
        WORK.mkdir()
        state["phase"] = "COMPILE_K8_CASE0_DIAGNOSTIC"
        state["vcs_compiles"] = 1
        run(compile_command(), WORK, WORK / "compile.log", 7200)
        if not (WORK / "simv").is_file():
            raise Failure("simv absent")
        state["phase"] = "SIM_K8_CASE0_DIAGNOSTIC"
        state["simv_runs"] = 1
        run(["./simv", "-lca", "+M979_CASE=0"], WORK,
            WORK / "diagnostic.log", 1800)
        result = CHECK.validate_diagnostic_log(WORK / "diagnostic.log")
        if state != {"phase": "SIM_K8_CASE0_DIAGNOSTIC", "vcs_compiles": 1,
                     "simv_runs": 1, "ucli_runs": 0, "saif_files": 0,
                     "ptpx_runs": 0, "all_other_eda_runs": 0,
                     "automatic_retry": False}:
            raise Failure("execution count drift")
        STAGE.mkdir()
        shutil.copy2(WORK / "compile.log", STAGE / "compile.log")
        shutil.copy2(WORK / "diagnostic.log", STAGE / "diagnostic.log")
        write_json(STAGE / "receipt.json", {
            "status": "M1856_DIAGNOSTIC_LOCALIZATION_COMPLETE_DO_NOT_CITE_AS_PRODUCTION",
            "execution": state, "localization": result,
            "claim_boundary": CHECK.CLAIMS})
        seal_dir(STAGE)
        publish_no_replace(STAGE, RESULT)
        shutil.rmtree(WORK)
        return 0
    except BaseException as error:
        if ATTEMPT.exists() and not FAILURE.exists():
            try:
                FAIL_STAGE.mkdir(exist_ok=False)
                write_json(FAIL_STAGE / "failure.json", {
                    "status": "M1856_DIAGNOSTIC_FAILED_OR_INCOMPLETE_DO_NOT_RETRY",
                    "error": type(error).__name__, "phase": state["phase"],
                    "execution": state, "automatic_retry": False})
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
