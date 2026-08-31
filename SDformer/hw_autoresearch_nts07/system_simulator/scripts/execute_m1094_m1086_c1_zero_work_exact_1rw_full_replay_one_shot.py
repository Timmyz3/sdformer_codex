#!/usr/bin/env python3
"""M1094r2 source-only atomic library for frozen M1086 CPU-model replay.

Import, source validation and the CLI never invoke either production interface
and cannot consume an attempt.  A different-author M1095 successor must create
a new launch wrapper with authority identities hardcoded in that wrapper's
source.  Only that independently hammered wrapper may call the atomic functions
below.  Results remain raw CPU-model evidence pending a separate M1096 hammer.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
PYTHON_SHA = "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"
M1086_PATH = HERE / "run_m1086_c1_zero_work_exact_1rw_source.py"
M1086_SHA = "3925c97de922393786b4aa8ae6ca6b4942489e3cf10485f5d1b6cd423e797a51"
CONTRACT = HW / "contracts/m1094r2_m1087r3_m1086r2_c1_zero_work_full_replay_atomic_library_source_contract_r1_20260830.json"
CONTRACT_SHA = "5278c5fa03a74cf9e3364325865b1bd52a5f75f372de15d5172b0b38bda64be4"
CONTRACT_SIDECAR_SHA = "963315ed0cd04080eeeb7271dab2da0fa808891919d6aa119f4ed89d4b44fffa"
RELEASE = HW / "contracts/m1087r3_m1086r2_c1_m1094_runner_source_authoring_release_r1_20260830.json"
RELEASE_SHA = "331d82e47f5a315744272e8bde369d4a6c0cd49cb3d735240bbf3848e6b81345"
RELEASE_SIDECAR_SHA = "11696ee452b5acf3c5f0b90e7c14010aab11cab4b32611df925f60be0c8e307a"
M1087R3 = HW / "reviews/m1087r3_m1086r2_c1_zero_work_population_source_hammer_r1_20260830"
M1087R3_ID = (
    "a3b9e35079444a6272ee91040e0250f16d1284c00a3e62c8b5ebc462366d1974",
    "70a5641bc0ad8dde7cb921361e4cd9938737b9cd009747b4f5fcb128b164d1ca",
    "c8901ff70a8a22fa171f0fc47ae6ea40ee91c3af793c9dc5ca09670113369ae5",
)
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

RESULT = HW / "results/m1094_m1086_c1_zero_work_exact_1rw_full_replay_r1_20260830"
ATTEMPT = HW / "results/.m1094_m1086_c1_zero_work_exact_1rw_full_replay_attempt_consumed"
LOCK = HW / "results/.m1094_c1_zero_work_exact_1rw_full_replay.lock"
WORK_PREFIX = ".m1094_m1086_c1_zero_work_exact_1rw_full_replay_work."
FAILURE_PREFIX = RESULT.name + ".failed_or_incomplete."
PAYLOAD = "m1094_c1_zero_work_exact_1rw_full_replay_result_r1.json"
PREFLIGHT_RECEIPT = "m1094_work_domain_preflight_receipt_r1.json"
SEAL_DIR = ".m1094_atomic_seal"
MANIFEST = "SHA256SUMS"
OUTER = "SHA256SUMS.seal.sha256"
RESULT_SCHEMA = "m1094_c1_zero_work_exact_1rw_full_replay_result_r1_v1"
RESULT_STATUS = "PASS_M1094_RAW_CPU_MODEL_FULL_REPLAY_PENDING_M1096_RESULT_HAMMER"

TASKS = 812160
DESIGNS = ("candidate", "strongest_zero", "same_coordinate_bit")
VALUES = 2436480
SAMPLES = 10


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def exact_int(value: Any) -> bool:
    return type(value) is int


def lower_sha256(value: Any) -> bool:
    return (type(value) is str and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + token)))


def write_exclusive(path: Path, data: bytes) -> None:
    with Path(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    require(renameat2 is not None, "M1094 renameat2 unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p,
                          ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    rc = renameat2(-100, os.fsencode(Path(source)), -100,
                   os.fsencode(Path(destination)), 1)
    if rc:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise RuntimeError("M1094 atomic no-replace collision")
        raise OSError(code, os.strerror(code), str(destination))


def load_m1086():
    require(M1086_PATH.is_file() and not M1086_PATH.is_symlink() and
            sha256(M1086_PATH) == M1086_SHA, "M1086 source identity drift")
    spec = importlib.util.spec_from_file_location("m1094_frozen_m1086", M1086_PATH)
    require(spec is not None and spec.loader is not None, "cannot load M1086")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M1086 = load_m1086()


def verify_flat(directory: Path, identity: tuple[str, str, str]) -> None:
    review, manifest, outer = (directory / "review.json",
                               directory / MANIFEST, directory / OUTER)
    require(directory.is_dir() and not directory.is_symlink() and
            (sha256(review), sha256(manifest), sha256(outer)) == identity,
            "sealed authority identity drift")
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        member = directory / relative
        require(relative not in listed and member.is_file() and
                not member.is_symlink() and sha256(member) == expected,
                "sealed authority member drift")
        listed.add(relative)
    expected, relative = outer.read_text(encoding="utf-8").split()
    require(relative == MANIFEST and expected == sha256(manifest),
            "sealed authority outer drift")


def verify_double_seal(path: Path, file_sha: str, sidecar_sha: str) -> None:
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require(path.is_file() and not path.is_symlink() and sha256(path) == file_sha and
            sidecar.is_file() and not sidecar.is_symlink() and
            sha256(sidecar) == sidecar_sha, "double-sealed identity drift")
    expected, name = sidecar.read_text(encoding="utf-8").split()
    require(expected == file_sha and name == path.name, "sidecar content drift")
    expected, name = outer.read_text(encoding="utf-8").split()
    require(expected == sidecar_sha and name == sidecar.name,
            "outer sidecar content drift")


def verify_interpreter() -> None:
    executable = Path(sys.executable).resolve()
    require(executable == PYTHON and sha256(executable) == PYTHON_SHA and
            tuple(sys.version_info[:3]) == (3, 10, 18),
            "M1094 Python identity drift")


def validate_source_contract(runner: Path | None = None,
                             require_fresh: bool = True) -> dict[str, Any]:
    verify_interpreter()
    verify_double_seal(CONTRACT, CONTRACT_SHA, CONTRACT_SIDECAR_SHA)
    verify_double_seal(RELEASE, RELEASE_SHA, RELEASE_SIDECAR_SHA)
    verify_flat(M1087R3, M1087R3_ID)
    contract = strict_json(CONTRACT)
    release = strict_json(RELEASE)
    hammer = strict_json(M1087R3 / "review.json")
    require(contract.get("schema") ==
            "m1094r2_m1087r3_m1086r2_c1_zero_work_full_replay_atomic_library_source_contract_r1_v1" and
            contract.get("status") ==
            "PASS_M1094R2_ATOMIC_LIBRARY_SOURCE_CONTRACT__NO_EXECUTABLE_LAUNCH" and
            contract.get("launch_now") is False and
            contract.get("max_attempts_now") == 0,
            "M1094 contract content drift")
    population = contract.get("canonical_population", {})
    require(population.get("tasks") == TASKS and
            population.get("designs") == list(DESIGNS) and
            population.get("design_count") == len(DESIGNS) and
            population.get("task_design_work_values") == VALUES and
            population.get("required_preflight_values_checked") == VALUES,
            "M1094 population contract drift")
    require(release.get("status") ==
            "GO_M1094_ONE_SHOT_RUNNER_SOURCE_AUTHORING_ONLY__NO_EXECUTION" and
            release.get("launch_now") is False and
            hammer.get("status") ==
            "PASS_M1087R3_M1086R2_C1_ZERO_WORK_POPULATION_SOURCE_HAMMER",
            "M1094 authoring authority drift")
    require(sha256(M1086_PATH) == M1086_SHA and
            sha256(DOCS359) == DOCS359_SHA and
            len(inspect.signature(
                M1086.canonical_work_domain_preflight).parameters) == 0 and
            inspect.isgeneratorfunction(M1086.iter_canonical_full_replay_results) and
            len(inspect.signature(
                M1086.iter_canonical_full_replay_results).parameters) == 0,
            "M1086 production interface drift")
    if runner is not None:
        expected = HW / contract["source_topology"]["non_launch_stub_path"]
        require(Path(runner).resolve() == expected.resolve() and
                expected.is_file() and not expected.is_symlink(),
                "M1094 runner path drift")
    if require_fresh:
        require(not RESULT.exists() and not ATTEMPT.exists() and
                not any(RESULT.parent.glob(WORK_PREFIX + "*")) and
                not any(RESULT.parent.glob(FAILURE_PREFIX + "*")),
                "M1094 runtime namespace not fresh")
    return {"status": "PASS_M1094R2_ATOMIC_LIBRARY_IDENTITIES__NO_CANONICAL_PAYLOAD",
            "canonical_payload_opened_or_hashed": False,
            "tasks": TASKS, "design_count": len(DESIGNS),
            "task_design_work_values": VALUES}


def payload_files(directory: Path) -> list[Path]:
    files = []
    for item in sorted(Path(directory).rglob("*")):
        relative = item.relative_to(directory)
        if relative.parts and relative.parts[0] == SEAL_DIR:
            continue
        require(not item.is_symlink(), "M1094 seal refuses symlink")
        if item.is_file():
            files.append(item)
    return files


def verify_atomic_seal(directory: Path) -> dict[str, Any]:
    bundle = Path(directory) / SEAL_DIR
    manifest, outer = bundle / MANIFEST, bundle / OUTER
    require(bundle.is_dir() and not bundle.is_symlink() and
            manifest.is_file() and outer.is_file() and
            not manifest.is_symlink() and not outer.is_symlink(),
            "M1094 atomic seal absent")
    require(outer.read_text(encoding="utf-8") ==
            sha256(manifest) + "  " + MANIFEST + "\n", "M1094 outer drift")
    listed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        member = Path(directory) / relative
        require(relative not in listed and member.is_file() and
                not member.is_symlink() and sha256(member) == digest,
                "M1094 manifest member drift")
        listed[relative] = digest
    actual = {item.relative_to(directory).as_posix()
              for item in payload_files(directory)}
    require(set(listed) == actual, "M1094 manifest coverage drift")
    return {"manifest_sha256": sha256(manifest),
            "outer_seal_file_sha256": sha256(outer),
            "members": len(actual)}


def atomic_seal(directory: Path) -> dict[str, Any]:
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(),
            "M1094 seal target drift")
    if (directory / SEAL_DIR).exists():
        return verify_atomic_seal(directory)
    stages = sorted(directory.parent.glob(directory.name + ".m1094_seal_stage.*"))
    if stages:
        recovery = directory / "PARTIAL_SEAL_ATTEMPTS"
        recovery.mkdir(mode=0o700, exist_ok=True)
        for index, stage in enumerate(stages):
            rename_noreplace(stage, recovery / ("attempt_%03d" % index))
    members = payload_files(directory)
    require(members, "M1094 empty seal target")
    stage = directory.parent / (directory.name + ".m1094_seal_stage.%d.%d" %
                                (os.getpid(), time.time_ns()))
    stage.mkdir(mode=0o700)
    lines = [sha256(item) + "  " + item.relative_to(directory).as_posix()
             for item in members]
    write_exclusive(stage / MANIFEST, ("\n".join(lines) + "\n").encode())
    write_exclusive(stage / OUTER,
                    (sha256(stage / MANIFEST) + "  " + MANIFEST + "\n").encode())
    fsync_dir(stage)
    rename_noreplace(stage, directory / SEAL_DIR)
    fsync_dir(directory)
    return verify_atomic_seal(directory)


def safe_sibling(path: Path, prefix: str, parent: Path | None = None) -> None:
    root = RESULT.parent if parent is None else Path(parent)
    require(Path(path).parent.resolve() == root.resolve() and
            Path(path).name.startswith(prefix) and not Path(path).is_symlink(),
            "M1094 unsafe runtime sibling")


def consume_attempt(authority: Mapping[str, Any],
                    parent: Path | None = None) -> dict[str, Any]:
    require(type(authority) is dict and
            authority.get("status") ==
            "PASS_DIFFERENT_AUTHOR_HARDCODED_LAUNCH_AUTHORITY" and
            set(authority) == {
                "status", "m1095_review_sha256", "m1095_manifest_sha256",
                "m1095_outer_seal_file_sha256", "m1095_launch_wrapper_sha256",
                "m1094_engine_sha256", "m1094_contract_sha256",
                "m1086_source_sha256", "m1087r3_outer_seal_file_sha256"
            } and all(lower_sha256(authority[key]) for key in authority
                      if key != "status") and
            authority["m1094_engine_sha256"] == sha256(Path(__file__)) and
            authority["m1094_contract_sha256"] == CONTRACT_SHA and
            authority["m1086_source_sha256"] == M1086_SHA and
            authority["m1087r3_outer_seal_file_sha256"] == M1087R3_ID[2],
            "M1094 hardcoded launch authority shape/identity drift")
    root = ATTEMPT.parent if parent is None else Path(parent)
    final = ATTEMPT if parent is None else root / ATTEMPT.name
    safe_sibling(final, ATTEMPT.name, root)
    require(not final.exists() and (parent is not None or not RESULT.exists()),
            "M1094 attempt collision")
    try:
        final.mkdir(mode=0o700)
    except FileExistsError as error:
        raise RuntimeError("M1094 attempt collision") from error
    fsync_dir(root)
    receipt = {
        "schema": "m1094_c1_full_replay_attempt_r1_v1",
        "status": "CONSUMED_BEFORE_CANONICAL_PAYLOAD_ACCESS",
        "maximum_attempts": 1, "automatic_retry": False,
        "m1095_outer_seal_file_sha256": authority["m1095_outer_seal_file_sha256"],
        "m1086_source_sha256": M1086_SHA,
        "canonical_payload_opened_or_hashed_before_attempt": False,
    }
    write_exclusive(final / "attempt.json",
                    (json.dumps(receipt, sort_keys=True) + "\n").encode())
    return {"receipt": receipt, "seal": atomic_seal(final)}


def finalize_attempt(attempt: Path = ATTEMPT) -> dict[str, Any]:
    require(attempt.is_dir() and not attempt.is_symlink(),
            "M1094 interrupted attempt drift")
    if (attempt / SEAL_DIR).exists():
        return verify_atomic_seal(attempt)
    marker = attempt / "ATTEMPT_INTERRUPTED.json"
    if not marker.exists():
        write_exclusive(marker, (json.dumps({
            "status": "ATTEMPT_CONSUMED__NO_RETRY",
            "canonical_payload_may_have_opened": True,
            "automatic_retry": False}, sort_keys=True) + "\n").encode())
    return atomic_seal(attempt)


def validate_preflight(value: Any) -> dict[str, Any]:
    require(type(value) is dict and set(value) == {
                "schema", "status", "tasks", "designs", "values_checked",
                "domain", "counts", "task_design_work_digest_sha256",
                "row_work_execution_provenance_digest_sha256",
                "cycles_derived_or_exported", "caller_supplied_work"
            } and
            value.get("schema") == "m1086_canonical_work_domain_preflight_v1" and
            value.get("status") == "PASS_M1086_ALL_TASK_DESIGN_WORK_VALUES_DEFINED" and
            value.get("tasks") == TASKS and value.get("designs") == list(DESIGNS) and
            value.get("values_checked") == VALUES and
            value.get("cycles_derived_or_exported") is False and
            value.get("caller_supplied_work") is False,
            "M1094 exhaustive preflight population/content drift")
    counts = value.get("counts")
    require(type(counts) is dict and set(counts) == set(DESIGNS) and
            all(type(row) is dict and set(row) == {"zero", "positive"} and
                all(exact_int(row[key]) and row[key] >= 0 for key in row) and
                sum(row[key] for key in ("zero", "positive")) == TASKS
                for row in counts.values()), "M1094 preflight counts drift")
    for key in ("task_design_work_digest_sha256",
                "row_work_execution_provenance_digest_sha256"):
        require(lower_sha256(value.get(key)),
                "M1094 preflight digest drift")
    return value


def normalize_raw(raw: Any) -> dict[str, Any]:
    require(type(raw) is dict and raw.get("schema") ==
            "m1086_canonical_full_zero_work_exact_1rw_replay_result_v1" and
            raw.get("status") == "PASS_M1086_RAW_FULL_REPLAY_PENDING_RESULT_HAMMER",
            "M1094 raw result identity drift")
    samples, coverage, capacity = (raw.get("samples"), raw.get("coverage"),
                                   raw.get("capacity"))
    require(type(samples) is list and len(samples) == SAMPLES and
            type(coverage) is dict and
            coverage.get("schema") == "m1072_row_provenance_coverage_v1" and
            coverage.get("full_coverage_pass") is True and
            coverage.get("caller_supplied_coverage_or_digest") is False and
            type(coverage.get("checks")) is dict and
            coverage["checks"] and all(value is True for value in
                                        coverage["checks"].values()) and
            type(coverage.get("service_digests")) is dict and
            set(coverage["service_digests"]) == set(DESIGNS) and
            all(lower_sha256(value) for value in
                coverage["service_digests"].values()) and
            lower_sha256(coverage.get(
                "execution_provenance_digest_sha256")) and
            type(coverage.get("parent")) is dict and
            set(coverage["parent"]) == set(DESIGNS) and
            type(capacity) is dict and
            capacity.get("schema") == "m1064_frozen_physical_capacity_v1" and
            capacity.get("derived_total_bytes") == 214912 and
            capacity.get("budget_bytes") == 245760 and
            capacity.get("derived_margin_bytes") == 30848 and
            capacity.get("capacity_bytes_pass") is True and
            capacity.get("caller_supplied_capacity") is False and
            capacity.get("capacity_only_214912B_admitted") is False,
            "M1094 raw result population/capacity drift")
    aggregate = {name: {"cycles": 0, "delayed_accesses": 0,
                        "nominal_excess_accesses": 0} for name in DESIGNS}
    for sample, row in enumerate(samples):
        require(type(row) is dict and row.get("sample") == sample and
                row.get("first_task_id") == sample * M1086.M1072.TASKS_PER_SAMPLE and
                row.get("last_task_id") ==
                    (sample + 1) * M1086.M1072.TASKS_PER_SAMPLE - 1 and
                type(row.get("designs")) is dict and
                set(row["designs"]) == set(DESIGNS),
                "M1094 sample boundary drift")
        for name in DESIGNS:
            entry = row["designs"][name]
            require(type(entry) is dict and set(entry) == {
                "cycles_after_commit", "delayed_accesses",
                "nominal_excess_accesses"} and
                all(exact_int(entry[key]) and entry[key] >= 0 for key in entry),
                "M1094 sample cycle/stall drift")
            aggregate[name]["cycles"] += entry["cycles_after_commit"]
            aggregate[name]["delayed_accesses"] += entry["delayed_accesses"]
            aggregate[name]["nominal_excess_accesses"] += entry[
                "nominal_excess_accesses"]
    boundary = raw.get("claim_boundary", {})
    require(boundary.get("matched_cycles_admitted") is False and
            boundary.get("speedup_admitted") is False and
            boundary.get("rtl_cycles") is False and
            boundary.get("paper_ppa_ready") is False and
            boundary.get("independent_result_hammer_required") is True,
            "M1094 raw claim boundary drift")
    return {"samples": samples, "coverage": coverage, "capacity": capacity,
            "aggregate": aggregate}


def execute_full(authority: Mapping[str, Any], work: Path) -> dict[str, Any]:
    """The only production path: preflight(), then one zero-arg iterator call."""
    safe_sibling(work, WORK_PREFIX)
    require(type(authority) is dict and authority.get("status") ==
            "PASS_DIFFERENT_AUTHOR_HARDCODED_LAUNCH_AUTHORITY" and
            not work.exists() and ATTEMPT.is_dir() and not RESULT.exists(),
            "M1094 work/attempt state drift")
    work.mkdir(mode=0o700)
    try:
        # FIRST canonical payload operation after attempt consumption.
        preflight = validate_preflight(M1086.canonical_work_domain_preflight())
        write_exclusive(work / PREFLIGHT_RECEIPT,
                        (json.dumps(preflight, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n").encode())
        generator = M1086.iter_canonical_full_replay_results()
        raw = next(generator)
        try:
            next(generator)
        except StopIteration:
            pass
        else:
            raise RuntimeError("M1094 full iterator yielded more than once")
        normalized = normalize_raw(raw)
        result = {
            "schema": RESULT_SCHEMA, "status": RESULT_STATUS,
            "authority": authority,
            "work_domain_preflight": preflight,
            "raw_cpu_model": normalized,
            "claim_boundary": {
                "raw_cpu_model_full_replay_complete": True,
                "independent_m1096_result_hammer_required": True,
                "matched_cycles_admitted": False, "speedup_admitted": False,
                "rtl_cycles": False, "paper_citable": False,
                "paper_ppa_ready": False,
            },
        }
        write_exclusive(work / PAYLOAD,
                        (json.dumps(result, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n").encode())
        write_exclusive(work / "RUN_COMPLETE.txt",
                        b"M1094_RAW_CPU_MODEL_COMPLETE__M1096_RESULT_HAMMER_REQUIRED\n")
        seal = atomic_seal(work)
        return {"status": RESULT_STATUS, "seal": seal,
                "payload_sha256": sha256(work / PAYLOAD)}
    except BaseException:
        if not (work / "traceback.log").exists():
            write_exclusive(work / "traceback.log", traceback.format_exc().encode())
        raise


def publish_result(work: Path) -> dict[str, Any]:
    safe_sibling(work, WORK_PREFIX)
    seal = verify_atomic_seal(work)
    payload = strict_json(work / PAYLOAD)
    require(payload.get("schema") == RESULT_SCHEMA and
            payload.get("status") == RESULT_STATUS and
            payload.get("claim_boundary", {}).get(
                "independent_m1096_result_hammer_required") is True and
            payload.get("claim_boundary", {}).get("speedup_admitted") is False and
            not RESULT.exists(), "M1094 publish claim drift")
    rename_noreplace(work, RESULT)
    fsync_dir(RESULT.parent)
    require(verify_atomic_seal(RESULT) == seal, "M1094 publish identity drift")
    return {"status": RESULT_STATUS, "result": str(RESULT), "seal": seal}


def verify_published_result() -> dict[str, Any]:
    seal = verify_atomic_seal(RESULT)
    payload = strict_json(RESULT / PAYLOAD)
    require(payload.get("schema") == RESULT_SCHEMA and
            payload.get("status") == RESULT_STATUS and
            payload.get("work_domain_preflight", {}).get("values_checked") == VALUES and
            payload.get("claim_boundary", {}).get("speedup_admitted") is False,
            "M1094 published result drift")
    return {"status": RESULT_STATUS, "seal": seal}


def quarantine_work(work: Path, quarantine: Path, return_code: int,
                    phase: str, parent: Path | None = None) -> dict[str, Any]:
    root = RESULT.parent if parent is None else Path(parent)
    safe_sibling(work, WORK_PREFIX, root)
    safe_sibling(quarantine, FAILURE_PREFIX, root)
    if parent is None and ATTEMPT.exists():
        finalize_attempt(ATTEMPT)
    stage = root / (quarantine.name + ".stage")
    require(not quarantine.exists() and not stage.exists(),
            "M1094 quarantine collision")
    stage.mkdir(mode=0o700)
    if work.exists():
        require(work.is_dir() and not work.is_symlink(), "M1094 work drift")
        rename_noreplace(work, stage / "partial_result")
    partial = sorted(root.glob(work.name + ".m1094_seal_stage.*"))
    if partial:
        recovery = stage / "partial_result_seal_stages"
        recovery.mkdir(mode=0o700)
        for index, item in enumerate(partial):
            rename_noreplace(item, recovery / ("attempt_%03d" % index))
    write_exclusive(stage / "failure.json", (json.dumps({
        "schema": "m1094_failure_quarantine_r1_v1",
        "status": "FAILED_OR_INTERRUPTED__NO_RETRY",
        "return_code": int(return_code), "phase": str(phase),
        "attempt_consumed": True, "automatic_retry": False,
    }, sort_keys=True) + "\n").encode())
    seal = atomic_seal(stage)
    rename_noreplace(stage, quarantine)
    fsync_dir(root)
    require(verify_atomic_seal(quarantine) == seal,
            "M1094 quarantine publication drift")
    return {"status": "PASS_M1094_SEALED_FAILURE_QUARANTINE",
            "quarantine": str(quarantine), "seal": seal}


def synthetic_raw_result() -> dict[str, Any]:
    samples = []
    for sample in range(SAMPLES):
        samples.append({"sample": sample,
            "first_task_id": sample * M1086.M1072.TASKS_PER_SAMPLE,
            "last_task_id": (sample + 1) * M1086.M1072.TASKS_PER_SAMPLE - 1,
            "designs": {name: {"cycles_after_commit": 1000 + sample,
                "delayed_accesses": sample,
                "nominal_excess_accesses": sample + 1} for name in DESIGNS}})
    coverage = {"schema": "m1072_row_provenance_coverage_v1",
        "checks": {"exact_tasks": True, "exact_sample_commits": True,
                   "exact_raw_rows": True, "exact_services": True,
                   "exact_service_digest": True,
                   "candidate_parent_conservation": True,
                   "baseline_parent_accesses_zero": True,
                   "baseline_work_equal": True},
        "full_coverage_pass": True,
        "service_digests": {name: "a" * 64 for name in DESIGNS},
        "execution_provenance_digest_sha256": "b" * 64,
        "parent": {name: {} for name in DESIGNS},
        "caller_supplied_coverage_or_digest": False}
    capacity = {"schema": "m1064_frozen_physical_capacity_v1",
        "derived_total_bytes": 214912, "budget_bytes": 245760,
        "derived_margin_bytes": 30848, "capacity_bytes_pass": True,
        "caller_supplied_capacity": False,
        "capacity_only_214912B_admitted": False}
    return {"schema": "m1086_canonical_full_zero_work_exact_1rw_replay_result_v1",
        "status": "PASS_M1086_RAW_FULL_REPLAY_PENDING_RESULT_HAMMER",
        "samples": samples, "coverage": coverage, "capacity": capacity,
        "claim_boundary": {"matched_cycles_admitted": False,
            "speedup_admitted": False, "rtl_cycles": False,
            "paper_ppa_ready": False,
            "independent_result_hammer_required": True}}


def synthetic_preflight() -> dict[str, Any]:
    return {"schema": "m1086_canonical_work_domain_preflight_v1",
        "status": "PASS_M1086_ALL_TASK_DESIGN_WORK_VALUES_DEFINED",
        "tasks": TASKS, "designs": list(DESIGNS), "values_checked": VALUES,
        "domain": "exact_int && (work==0 || work>=15)",
        "counts": {name: {"zero": 1000, "positive": TASKS - 1000}
                   for name in DESIGNS},
        "task_design_work_digest_sha256": "c" * 64,
        "row_work_execution_provenance_digest_sha256": "d" * 64,
        "cycles_derived_or_exported": False, "caller_supplied_work": False}


def source_self_test() -> dict[str, Any]:
    source = validate_source_contract(require_fresh=True)
    normalized = normalize_raw(synthetic_raw_result())
    require(normalized["aggregate"]["candidate"]["cycles"] == 10045,
            "M1094 synthetic normalize drift")
    return {"status": "PASS_M1094R2_SOURCE_SELF_TEST__NO_ATTEMPT_NO_PAYLOAD",
            "source": source, "synthetic_normalization": True,
            "production_preflight_called": False,
            "production_iterator_called": False,
            "attempt_consumed": False, "full_replay_executed": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Deliberately read-only CLI.  M1095 must author a new launch wrapper with
    # exact authority identities compiled into that wrapper's source.
    for mode in ("self-test", "validate-source", "verify-published"):
        parser.add_argument("--" + mode, action="store_true")
    parser.add_argument("--runner", type=Path)
    args = parser.parse_args(argv)
    modes = (args.self_test, args.validate_source, args.verify_published)
    require(sum(bool(value) for value in modes) == 1,
            "M1094r2 requires exactly one read-only mode")
    if args.self_test:
        value = source_self_test()
    elif args.validate_source:
        value = validate_source_contract(args.runner)
    else:
        value = verify_published_result()
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
