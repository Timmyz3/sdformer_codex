#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Additive one-shot successor for the M1433 C1 functional VCS runner.

M1433 correctly separated source-only tests from runtime-present tests, but its
generic stage sealer called an authority verifier which unconditionally opened
``review.json``.  Attempt and failure stages deliberately do not contain that
file, so the sole attempt stopped before VCS.  M1459 keeps the exact M1433
workload, runtime suite, tool bounds, and claim boundary while separating:

* recursive artifact verification (any sealed directory), and
* authority verification (a sealed directory which must contain review.json).

This file is inert without a fresh M1464/M1465/M1466 authority chain and exact
external SHA-256 pins.
"""
from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
from datetime import datetime, timezone


if len(sys.argv) != 1:
    raise SystemExit("M1459: no arguments accepted")

SCRIPT_DIR = Path(__file__).resolve().parent
HW = SCRIPT_DIR.parents[1]
RUNNER = Path(__file__).resolve()
M1433_RUNNER = SCRIPT_DIR / "run_vcs_m1433_m1337r15_m1162_c1_real_m935_runtime_witness_unit_delay_runtime_split_exact.py"
SPEC = importlib.util.spec_from_file_location("m1459_frozen_m1433", M1433_RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("M1459: cannot load frozen M1433 runner")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

SOURCE_CHECKER = HW / "verif_m1459_c1_generic_seal_successor/check_m1459_c1_generic_seal_successor_source.py"
SOURCE_TESTS = HW / "verif_m1459_c1_generic_seal_successor/test_m1459_c1_generic_seal_successor_source.py"
SOURCE_CONTRACT = HW / "contracts/m1459_m1433_c1_generic_seal_successor_source_contract_r1_20260831.json"
AUTHOR_DIR = HW / "reviews/m1459_m1433_c1_generic_seal_successor_source_author_r1_20260831"
SOURCE_HAMMER = HW / "reviews/m1464_m1459_c1_generic_seal_successor_source_blind_hammer_r1_20260831"
LAUNCH_RELEASE = HW / "contracts/m1465_m1464_m1459_c1_generic_seal_successor_vcs_launch_release_r1_20260831.json"
FINAL_HAMMER = HW / "reviews/m1466_m1465_m1459_c1_generic_seal_successor_final_launch_hammer_r1_20260831"

ATTEMPT = HW / "results/.m1459_c1_generic_seal_vcs_attempt_consumed"
RESULT = HW / "results/m1459_c1_real_m935_runtime_witness_unit_delay_vcs_r1_20260831"
QUARANTINE = Path(str(RESULT) + ".failed_or_incomplete.quarantine")
WORK = HW / f"results/.m1459_c1_generic_seal_vcs_work.{os.getpid()}"
ATTEMPT_STAGE = HW / f"results/.m1459_c1_generic_seal_vcs_attempt_stage.{os.getpid()}"
FAILURE_STAGE = HW / f"results/.m1459_c1_generic_seal_vcs_failure_stage.{os.getpid()}"

ENV_PINS = (
    "M1459_EXPECTED_RUNNER_SHA256",
    "M1459_EXPECTED_SOURCE_CONTRACT_SHA256",
    "M1459_EXPECTED_SOURCE_HAMMER_REVIEW_SHA256",
    "M1459_EXPECTED_SOURCE_HAMMER_MANIFEST_SHA256",
    "M1459_EXPECTED_SOURCE_HAMMER_OUTER_FILE_SHA256",
    "M1459_EXPECTED_LAUNCH_RELEASE_SHA256",
    "M1459_EXPECTED_FINAL_HAMMER_REVIEW_SHA256",
    "M1459_EXPECTED_FINAL_HAMMER_MANIFEST_SHA256",
    "M1459_EXPECTED_FINAL_HAMMER_OUTER_FILE_SHA256",
)

AUTHORIZATION = {"vcs_compiles": 1, "simv_runs": 1,
                 "all_other_eda_runs": 0, "automatic_retry": False}
CLAIMS = dict(BASE.CLAIMS)
COMPILE_COMMAND = list(BASE.COMPILE_COMMAND)
SIM_COMMAND = list(BASE.SIM_COMMAND)
M1433_CHAIN_PINS = {
    "source_contract": "eacc909123b18f9e2314cdb01bf4d2c5a98865a9754329c75a15568ae91c0379",
    "source_contract_sidecar": "958e03e20fa2ffc5a8dbaef8436ef710a63448d8bdc5314705f6caf2b81ed486",
    "source_contract_outer": "d427a2c80f3294ce57aba1283dd3533333c6a9781897589861193bb6a9472d91",
    "author_review": "b2207075e229e3b3a92135d5e950c51373225e2fe78ce26915165dff17ebc8fd",
    "author_manifest": "bfebacc92719ab2c42338dd2bfe254f9e7ae076eeb5e6697f98091dd6de168ee",
    "author_outer": "fa842bb5b43740e663f3c998a51c54f2295afab51698eed53b2fac4a891cfa1e",
    "source_hammer_review": "d5f5672c13f3dd3f6ce8927871d3959a538a74d9ad9d021e6026186d931a8716",
    "source_hammer_manifest": "6f8e2ce105a51d595bf2caa77b02964b44f447a6cfe710a0d8c19ea99fe67f4a",
    "source_hammer_outer": "bfef3978b151c1fd898b31341fc90914c3200e2e06b93226d0251c2ff207a256",
    "launch_release": "84bb5c0c6f1b808008c7fbc4adb637a183a759b348c9f08f2432aa5d8ac41f1a",
    "launch_release_sidecar": "5967bcc2af5ec8a1ace456d72d475c2432000bc3b5ac123ace7d94ad6b731265",
    "launch_release_outer": "6ace9edfac76553e8939037e221ef55baf88b41eb826c5f2dc939451cf7e118a",
    "final_hammer_review": "067e66554fcc67ff698b4ab8d58b3478ea9f12015d1e47590cf100efb3e6584c",
    "final_hammer_manifest": "ce3cd14001f091a4662ff45420556a2b4993f7aee13966fa0b2617f19daee792",
    "final_hammer_outer": "7e0093c3fcff7d691aa2429db0aa55f2fcd164f62a523bbcd37735d77c66157a",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(path: Path, digest: str) -> None:
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode) or path.is_symlink() or sha(path) != digest:
        raise RuntimeError("identity drift: " + str(path))


def strict_json(path: Path) -> dict:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise RuntimeError("duplicate JSON key")
            value[key] = item
        return value
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise RuntimeError("JSON not regular")
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(RuntimeError(token)))


def verify_recursive_seal_generic(root: Path, manifest_pin: str | None = None,
                                  outer_pin: str | None = None) -> None:
    """Verify a recursively sealed directory without assuming its payload type."""
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("sealed directory invalid")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    if manifest_pin is not None:
        exact(manifest, manifest_pin)
    if outer_pin is not None:
        exact(outer, outer_pin)
    if outer.read_text().split() != [sha(manifest), "SHA256SUMS"]:
        raise RuntimeError("outer seal drift")
    listed = set()
    for line in manifest.read_text().splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise RuntimeError("manifest row invalid")
        digest, name = fields
        name = name.lstrip("*")
        rel = Path(name)
        if (not re.fullmatch(r"[0-9a-f]{64}", digest) or name in listed
                or rel.is_absolute() or ".." in rel.parts):
            raise RuntimeError("manifest row invalid")
        exact(root / rel, digest)
        listed.add(name)
    actual = set()
    for base, dirs, files in os.walk(root, followlinks=False):
        base_path = Path(base)
        if any((base_path / name).is_symlink() for name in dirs + files):
            raise RuntimeError("sealed symlink")
        for name in files:
            rel = (base_path / name).relative_to(root).as_posix()
            if rel not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
                actual.add(rel)
    if listed != actual:
        raise RuntimeError("sealed membership drift")


def verify_authority(root: Path, review_pin: str | None = None,
                     manifest_pin: str | None = None,
                     outer_pin: str | None = None) -> dict:
    """Verify an authority seal, then require and parse its review payload."""
    verify_recursive_seal_generic(root, manifest_pin, outer_pin)
    review = root / "review.json"
    if review_pin is not None:
        exact(review, review_pin)
    return strict_json(review)


def seal_dir_generic(root: Path) -> None:
    rows = []
    for base, dirs, files in os.walk(root, followlinks=False):
        base_path = Path(base)
        if any((base_path / name).is_symlink() for name in dirs + files):
            raise RuntimeError("cannot seal symlink")
        for name in files:
            path = base_path / name
            rel = path.relative_to(root).as_posix()
            if rel in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
                continue
            if not stat.S_ISREG(path.lstat().st_mode):
                raise RuntimeError("nonregular result")
            rows.append((rel, sha(path)))
    rows.sort()
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join(f"{digest}  {name}\n" for name, digest in rows))
    (root / "SHA256SUMS.seal.sha256").write_text(
        f"{sha(manifest)}  SHA256SUMS\n")
    verify_recursive_seal_generic(root)


def verify_file_sidecar(path: Path) -> None:
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    for item in (path, sidecar, outer):
        if not item.is_file() or item.is_symlink():
            raise RuntimeError("sidecar absent")
    if sidecar.read_text().split() != [sha(path), path.name]:
        raise RuntimeError("sidecar mismatch")
    if outer.read_text().split() != [sha(sidecar), sidecar.name]:
        raise RuntimeError("outer mismatch")


def publish_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint]
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def namespace_gate() -> None:
    for path in (ATTEMPT, RESULT, QUARANTINE, WORK, ATTEMPT_STAGE, FAILURE_STAGE):
        if os.path.lexists(path):
            raise RuntimeError("namespace residue: " + str(path))
    for pattern in (".m1459_c1_generic_seal_vcs_work.*",
                    ".m1459_c1_generic_seal_vcs_attempt_stage.*",
                    ".m1459_c1_generic_seal_vcs_failure_stage.*"):
        if list((HW / "results").glob(pattern)):
            raise RuntimeError("stale stage: " + pattern)


def run_python_gate(path: Path, mode: str) -> None:
    completed = subprocess.run([str(BASE.PYTHON), "-I", str(path), "--mode", mode],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, timeout=120, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"python gate failed {path.name}: {completed.stderr}")


def run_tool(command, log: Path, timeout_seconds: int, environment: dict[str, str]) -> int:
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True,
                               env=environment, cwd=WORK)
    try:
        output, _ = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
        log.write_bytes(output or b"")
        raise RuntimeError("tool timeout")
    output = output or b""
    log.write_bytes(output)
    sys.stdout.buffer.write(output)
    sys.stdout.flush()
    return process.returncode


def validate_authorities() -> tuple[dict, dict, dict, dict, dict]:
    # Preserve every frozen M1433 input and predecessor authority.
    exact(M1433_RUNNER, "443ef3f2a2bc777095a5574da6b91aa2c97786505f86bff607fbc537adbae07a")
    for path, digest in BASE.EXACT.items():
        exact(path, digest)
    for root, pins in BASE.SEALED.items():
        verify_authority(root, pins[0], pins[1], pins[2])
    verify_file_sidecar(BASE.SOURCE_CONTRACT)
    exact(BASE.SOURCE_CONTRACT, M1433_CHAIN_PINS["source_contract"])
    exact(Path(str(BASE.SOURCE_CONTRACT) + ".sha256"),
          M1433_CHAIN_PINS["source_contract_sidecar"])
    exact(Path(str(BASE.SOURCE_CONTRACT) + ".sha256.seal.sha256"),
          M1433_CHAIN_PINS["source_contract_outer"])
    verify_authority(BASE.AUTHOR_DIR, M1433_CHAIN_PINS["author_review"],
                     M1433_CHAIN_PINS["author_manifest"],
                     M1433_CHAIN_PINS["author_outer"])
    verify_authority(BASE.SOURCE_HAMMER,
                     M1433_CHAIN_PINS["source_hammer_review"],
                     M1433_CHAIN_PINS["source_hammer_manifest"],
                     M1433_CHAIN_PINS["source_hammer_outer"])
    verify_file_sidecar(BASE.LAUNCH_RELEASE)
    exact(BASE.LAUNCH_RELEASE, M1433_CHAIN_PINS["launch_release"])
    exact(Path(str(BASE.LAUNCH_RELEASE) + ".sha256"),
          M1433_CHAIN_PINS["launch_release_sidecar"])
    exact(Path(str(BASE.LAUNCH_RELEASE) + ".sha256.seal.sha256"),
          M1433_CHAIN_PINS["launch_release_outer"])
    verify_authority(BASE.FINAL_HAMMER,
                     M1433_CHAIN_PINS["final_hammer_review"],
                     M1433_CHAIN_PINS["final_hammer_manifest"],
                     M1433_CHAIN_PINS["final_hammer_outer"])

    exact(RUNNER, os.environ["M1459_EXPECTED_RUNNER_SHA256"])
    verify_file_sidecar(SOURCE_CONTRACT)
    exact(SOURCE_CONTRACT, os.environ["M1459_EXPECTED_SOURCE_CONTRACT_SHA256"])
    author = verify_authority(AUTHOR_DIR)
    source_hammer = verify_authority(
        SOURCE_HAMMER,
        os.environ["M1459_EXPECTED_SOURCE_HAMMER_REVIEW_SHA256"],
        os.environ["M1459_EXPECTED_SOURCE_HAMMER_MANIFEST_SHA256"],
        os.environ["M1459_EXPECTED_SOURCE_HAMMER_OUTER_FILE_SHA256"],
    )
    verify_file_sidecar(LAUNCH_RELEASE)
    exact(LAUNCH_RELEASE, os.environ["M1459_EXPECTED_LAUNCH_RELEASE_SHA256"])
    final_hammer = verify_authority(
        FINAL_HAMMER,
        os.environ["M1459_EXPECTED_FINAL_HAMMER_REVIEW_SHA256"],
        os.environ["M1459_EXPECTED_FINAL_HAMMER_MANIFEST_SHA256"],
        os.environ["M1459_EXPECTED_FINAL_HAMMER_OUTER_FILE_SHA256"],
    )
    contract = strict_json(SOURCE_CONTRACT)
    release = strict_json(LAUNCH_RELEASE)
    return contract, author, source_hammer, release, final_hammer


def main() -> int:
    os.umask(0o077)
    for name in ENV_PINS:
        if not re.fullmatch(r"[0-9a-f]{64}", os.environ.get(name, "")):
            raise RuntimeError("external digest absent/invalid: " + name)
    contract, author, source_hammer, release, final_hammer = validate_authorities()
    bindings = {
        "runner_sha256": sha(RUNNER),
        "source_checker_sha256": sha(SOURCE_CHECKER),
        "source_tests_sha256": sha(SOURCE_TESTS),
        "runtime_tests_sha256": sha(BASE.RUNTIME_TESTS),
        "source_contract_sha256": sha(SOURCE_CONTRACT),
    }
    if contract.get("status") != "M1459_C1_GENERIC_SEAL_SUCCESSOR_SOURCE_READY__FRESH_M1464_REQUIRED__NO_LAUNCH":
        raise RuntimeError("contract status")
    if any(author.get("bindings", {}).get(key) != value for key, value in bindings.items()):
        raise RuntimeError("author binding")
    if any(source_hammer.get("bindings", {}).get(key) != value for key, value in bindings.items()):
        raise RuntimeError("source hammer binding")
    if release.get("status") != "AUTHORIZE_ONE_M1459_C1_GENERIC_SEAL_UNIT_DELAY_VCS_ATTEMPT":
        raise RuntimeError("release status")
    if release.get("identity", {}).get("source_hammer_review_sha256") != sha(SOURCE_HAMMER / "review.json"):
        raise RuntimeError("release source hammer")
    if final_hammer.get("status") != "PASS_M1466_AUTHORIZE_ONE_M1459_C1_GENERIC_SEAL_VCS_LAUNCH":
        raise RuntimeError("final status")
    if final_hammer.get("bindings", {}).get("launch_release_sha256") != sha(LAUNCH_RELEASE):
        raise RuntimeError("final release binding")
    if release.get("authorization") != AUTHORIZATION or final_hammer.get("authorization") != AUTHORIZATION:
        raise RuntimeError("authorization")
    if any(item.get("claim_boundary") != CLAIMS
           for item in (contract, author, source_hammer, release, final_hammer)):
        raise RuntimeError("claim boundary")

    # Preserve M1433's exact runtime-present gates; never invoke its source-only suite.
    run_python_gate(BASE.SOURCE_CHECKER, "runtime_present")
    run_python_gate(BASE.RUNTIME_TESTS, "runtime_present")
    namespace_gate()
    phase = "RESOURCE_PREFLIGHT"
    BASE.collision_gate()
    BASE.resource_gate()
    BASE.collision_gate()
    phase = "ATTEMPT_CONSUME"
    complete = False
    failure_armed = True
    compile_count = 0
    sim_count = 0
    try:
        def interrupted(signum, _frame):
            raise RuntimeError("interrupted by signal " + str(signum))
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, interrupted)
        ATTEMPT_STAGE.mkdir()
        (ATTEMPT_STAGE / "attempt.json").write_text(json.dumps({
            "status": "M1459_ATTEMPT_CONSUMED",
            "runner_sha256": sha(RUNNER),
            "source_contract_sha256": sha(SOURCE_CONTRACT),
            "source_hammer_review_sha256": sha(SOURCE_HAMMER / "review.json"),
            "launch_release_sha256": sha(LAUNCH_RELEASE),
            "final_hammer_review_sha256": sha(FINAL_HAMMER / "review.json"),
            "predecessor_m1433_compile_count": 0,
            "predecessor_m1433_sim_count": 0,
            "automatic_retry": False,
            "maximum_vcs_compiles": 1,
            "maximum_simv_runs": 1,
        }, indent=2, sort_keys=True) + "\n")
        seal_dir_generic(ATTEMPT_STAGE)
        publish_no_replace(ATTEMPT_STAGE, ATTEMPT)
        WORK.mkdir()
        environment = dict(os.environ)
        environment.update({
            "VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
            "VCS_ARCH_OVERRIDE": "linux",
            "SNPSLMD_LICENSE_FILE": "27030@ic.ismd-nemo",
            "LM_LICENSE_FILE": "/opt/synopsys/Synopsys.dat",
        })
        phase = "COMPILE"
        compile_count = 1
        if run_tool(COMPILE_COMMAND, WORK / "compile.log",
                    BASE.COMPILE_TIMEOUT_SECONDS, environment) != 0:
            raise RuntimeError("compile failed")
        simv = WORK / "simv"
        if not simv.is_file() or not os.access(simv, os.X_OK):
            raise RuntimeError("simv absent")
        phase = "SIMULATE"
        sim_count = 1
        if run_tool(SIM_COMMAND, WORK / "sim.log", BASE.SIM_TIMEOUT_SECONDS,
                    environment) != 0:
            raise RuntimeError("simulation failed")
        log = (WORK / "sim.log").read_text(errors="replace")
        if log.splitlines().count(BASE.R13_PASS) != 1 or log.splitlines().count(BASE.R15_PASS) != 1:
            raise RuntimeError("pass token cardinality")
        patterns = (
            r"^PHASE_M1270R13_REAL_M935_INTEGRATED_ENTER$",
            r"^PHASE_M1270R13_REAL_M935_INTEGRATED_COMPLETE$",
            r"^M1337R15_WITNESS_OPERANDS pass=1 ",
            r"^COVERAGE_M1270R13_REAL_M935 first_beats=1 nonfirst_beats=1 join_hold_cycles=2 issue_accepts=2 psum_reads=1 row_completions=1 task_completions=1 response_cycle_gap=[2-9][0-9]* oracle_records=[8-9][0-9]* parent_issue_override=0 child_issue_override=0$",
        )
        if any(len(re.findall(pattern, log, re.MULTILINE)) != 1 for pattern in patterns):
            raise RuntimeError("coverage cardinality")
        if re.search(r"(^|[^A-Za-z0-9_])(Error|Fatal|Assertion|\$error|\$fatal)([^A-Za-z0-9_]|$)",
                     log, re.IGNORECASE):
            raise RuntimeError("error/fatal/assertion line")
        receipt = {
            "schema": "m1459_c1_generic_seal_unit_delay_vcs_receipt_r1_v1",
            "status": "PASS_FUNCTIONAL_VCS_REAL_M935_RUNTIME_WITNESS",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "identity": {
                "runner_sha256": sha(RUNNER),
                "source_contract_sha256": sha(SOURCE_CONTRACT),
                "release_sha256": sha(LAUNCH_RELEASE),
                "final_hammer_review_sha256": sha(FINAL_HAMMER / "review.json"),
            },
            "seal_repair": {
                "generic_stage_verifier_requires_review_json": False,
                "authority_verifier_requires_review_json": True,
                "m1433_workload_and_runtime_suite_preserved": True,
            },
            "one_shot": {"attempt_consumed": True, "vcs_compiles": 1,
                         "simv_runs": 1, "automatic_retry": False,
                         "compile_timeout_seconds": BASE.COMPILE_TIMEOUT_SECONDS,
                         "sim_timeout_seconds": BASE.SIM_TIMEOUT_SECONDS},
            "claim_boundary": {**CLAIMS, "source_only": False,
                               "functional_vcs": True},
        }
        (WORK / "m1459_c1_generic_seal_unit_delay_vcs_receipt_r1.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        (WORK / "RUN_COMPLETE.txt").write_text(
            "PASS_FUNCTIONAL_VCS_REAL_M935_RUNTIME_WITNESS\n")
        phase = "SUCCESS_PUBLISH"
        seal_dir_generic(WORK)
        publish_no_replace(WORK, RESULT)
        complete = True
        print("PASS M1459 C1/R16 functional VCS result=" + str(RESULT))
        return 0
    except BaseException as exc:
        if failure_armed and not complete:
            FAILURE_STAGE.mkdir()
            if ATTEMPT_STAGE.is_dir() and not ATTEMPT_STAGE.is_symlink():
                os.rename(ATTEMPT_STAGE, FAILURE_STAGE / "private_attempt_stage")
            if WORK.is_dir() and not WORK.is_symlink():
                os.rename(WORK, FAILURE_STAGE / "private_build")
            (FAILURE_STAGE / "RUN_FAILED_OR_INCOMPLETE.json").write_text(json.dumps({
                "status": "FAILED_OR_INCOMPLETE",
                "phase": phase,
                "exception": type(exc).__name__ + ": " + str(exc),
                "compile_count": compile_count,
                "sim_count": sim_count,
                "automatic_retry": False,
                "functional_vcs": False,
                "timing_verified": False,
                "cycles_measured": False,
                "speedup": False,
                "ppa": False,
                "power": False,
                "energy": False,
                "system_speedup": False,
                "headline": False,
            }, indent=2, sort_keys=True) + "\n")
            seal_dir_generic(FAILURE_STAGE)
            publish_no_replace(FAILURE_STAGE, QUARANTINE)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
