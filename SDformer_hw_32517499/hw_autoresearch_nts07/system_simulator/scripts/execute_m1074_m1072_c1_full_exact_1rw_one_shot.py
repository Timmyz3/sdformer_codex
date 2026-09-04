#!/usr/bin/env python3
"""M1074 source-only atomic one-shot wrapper for the M1072 full iterator.

Importing, checking and self-testing this file never advances the M1072
generator and never opens the canonical row file.  A future independently
hammered runner may consume one attempt, then invoke ``execute_full`` exactly
once.  The resulting cycles remain raw CPU-model evidence pending a separate
result hammer; they are not RTL cycles or paper PPA.
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
import tempfile
import time
import traceback
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
REPO = HW.parent
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
PYTHON_SHA = "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"
M1072_PATH = HERE / "run_m1072_c1_row_provenance_exact_1rw_source.py"
M1072_SHA = "879712a59785acc79776990236884582431adea81103a222d5415905199a1e4c"
CONTRACT = HW / "contracts/m1074_m1073_m1072_c1_full_exact_1rw_one_shot_source_contract_r1_20260830.json"
CONTRACT_SIDECAR = Path(str(CONTRACT) + ".sha256")
CONTRACT_OUTER = Path(str(CONTRACT) + ".sha256.seal.sha256")
CONTRACT_SHA = "5d385afe4c0b5875568b19f903d1ed56a224d79790c206a62a28fdeefb967a67"
CONTRACT_SIDECAR_SHA = "259532aa54f20c02bfbb04c2e3722b9fb821ba82b4b9d025c45bc8b5fd3c348d"
M1073 = HW / "reviews/m1073_m1072_c1_row_provenance_exact_1rw_source_hammer_r1_20260830"
M1073_ID = (
    "c89662c9e1d46faba936a5b8eda80019780975dcfe9e71f301138286d2620fbb",
    "5ccaf435321a5ee58a499335ddff1ed28c560b09e5c536445bd6d021acaef7a9",
    "0a0457481fda030275205cb8c3b59938b66d86e1ce2cac63b0e2572b2de75e70",
)
M1075 = HW / "reviews/m1075_m1074_c1_full_exact_1rw_one_shot_source_hammer_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
RESULT = HW / "results/m1074_m1072_c1_full_exact_1rw_replay_r1_20260830"
ATTEMPT = HW / "results/.m1074_m1072_c1_full_exact_1rw_replay_attempt_consumed"
WORK_PREFIX = ".m1074_m1072_c1_full_exact_1rw_replay_work."
FAILURE_PREFIX = RESULT.name + ".failed_or_incomplete."
PAYLOAD = "m1074_c1_full_exact_1rw_replay_result_r1.json"
SEAL_DIR = ".m1074_atomic_seal"
SEAL_MANIFEST = "SHA256SUMS"
SEAL_OUTER = "SHA256SUMS.seal.sha256"
CONTRACT_SCHEMA = "m1074_m1073_m1072_c1_full_exact_1rw_one_shot_source_contract_v1"
RESULT_SCHEMA = "m1074_c1_full_exact_1rw_replay_result_v1"
RESULT_STATUS = "PASS_M1074_RAW_FULL_REPLAY_PENDING_INDEPENDENT_RESULT_HAMMER"
M1075_STATUS = "PASS_M1075_M1074_C1_FULL_EXACT_1RW_ONE_SHOT_SOURCE_HAMMER"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def exact_int(value: Any) -> bool:
    return type(value) is int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(
        Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            RuntimeError("nonfinite JSON: " + token)
        ),
    )


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
    """Linux atomic same-filesystem publication that never replaces evidence."""
    source = Path(source)
    destination = Path(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    require(renameat2 is not None, "M1074 renameat2 unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p,
                          ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100,
                       os.fsencode(destination), 1)
    if result != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise RuntimeError("M1074 atomic no-replace destination collision")
        raise OSError(code, os.strerror(code), str(destination))


def recover_partial_seal_stages(directory: Path) -> int:
    directory = Path(directory)
    prefix = directory.name + ".m1074_seal_stage."
    stages = sorted(item for item in directory.parent.iterdir()
                    if item.name.startswith(prefix))
    if not stages:
        return 0
    recovery = directory / "PARTIAL_SEAL_ATTEMPTS"
    recovery.mkdir(mode=0o700, exist_ok=True)
    for index, stage in enumerate(stages):
        destination = recovery / ("attempt_%03d" % index)
        require(not destination.exists(),
                "M1074 partial-seal recovery collision")
        rename_noreplace(stage, destination)
    fsync_dir(recovery)
    fsync_dir(directory)
    return len(stages)


def load_m1072():
    require(M1072_PATH.is_file() and not M1072_PATH.is_symlink() and
            sha256(M1072_PATH) == M1072_SHA, "M1072 source identity drift")
    spec = importlib.util.spec_from_file_location("m1074_frozen_m1072", M1072_PATH)
    require(spec is not None and spec.loader is not None, "cannot load M1072")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M1072 = load_m1072()


def verify_interpreter() -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    require(executable == PYTHON and sha256(executable) == PYTHON_SHA and
            tuple(sys.version_info[:3]) == (3, 10, 18),
            "M1074 interpreter identity drift")
    return {"path": str(executable), "sha256": PYTHON_SHA,
            "version": [3, 10, 18]}


def verify_flat(directory: Path, identity: tuple[str, str, str]) -> None:
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(directory.is_dir() and not directory.is_symlink() and
            (sha256(review), sha256(manifest), sha256(outer)) == identity,
            "sealed authority identity drift")
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        require(name not in listed and (directory / name).is_file() and
                not (directory / name).is_symlink() and
                sha256(directory / name) == expected,
                "sealed authority member drift")
        listed.add(name)
    expected, name = outer.read_text(encoding="utf-8").split()
    require(name == "SHA256SUMS" and expected == sha256(manifest),
            "sealed authority outer drift")


def validate_source_contract(runner: Path | None = None,
                             require_fresh: bool = True) -> dict[str, Any]:
    verify_interpreter()
    require(CONTRACT.is_file() and not CONTRACT.is_symlink() and
            sha256(CONTRACT) == CONTRACT_SHA and
            sha256(CONTRACT_SIDECAR) == CONTRACT_SIDECAR_SHA,
            "M1074 contract identity drift")
    expected, name = CONTRACT_SIDECAR.read_text(encoding="utf-8").split()
    require(expected == CONTRACT_SHA and name == CONTRACT.name,
            "M1074 contract sidecar drift")
    expected, name = CONTRACT_OUTER.read_text(encoding="utf-8").split()
    require(expected == CONTRACT_SIDECAR_SHA and name == CONTRACT_SIDECAR.name,
            "M1074 contract outer drift")
    contract = strict_json(CONTRACT)
    require(contract.get("schema") == CONTRACT_SCHEMA and
            contract.get("status") ==
            "PASS_M1074_ONE_SHOT_SOURCE_CONTRACT__M1075_REQUIRED_NO_LAUNCH" and
            contract.get("launch_now") is False and
            exact_int(contract.get("max_attempts_now")) and
            contract.get("max_attempts_now") == 0,
            "M1074 contract content drift")
    verify_flat(M1073, M1073_ID)
    m1073 = strict_json(M1073 / "review.json")
    require(m1073.get("status") ==
            "PASS_M1073_M1072_C1_ROW_PROVENANCE_EXACT_1RW_SOURCE_HAMMER" and
            m1073.get("authorized_next_step", {}).get(
                "m1074_full_replay_release_source_may_be_authored") is True and
            m1073.get("authorized_next_step", {}).get(
                "m1074_may_launch_now") is False,
            "M1073 source authority drift")
    require(sha256(M1072_PATH) == M1072_SHA and
            sha256(M1072.CONTRACT) == M1072.CONTRACT_SHA and
            sha256(DOCS359) == DOCS359_SHA,
            "M1072/docs359 source identity drift")
    require(inspect.isgeneratorfunction(
                M1072.iter_canonical_full_replay_results) and
            len(inspect.signature(
                M1072.iter_canonical_full_replay_results).parameters) == 0,
            "M1072 unique zero-argument iterator drift")
    if runner is not None:
        expected_path = HW / contract["source_identity"]["future_runner_path"]
        require(Path(runner).resolve() == expected_path.resolve() and
                expected_path.is_file() and not expected_path.is_symlink(),
                "M1074 runner path drift")
    if require_fresh:
        require(not RESULT.exists() and not ATTEMPT.exists() and
                not any(HW.joinpath("results").glob(WORK_PREFIX + "*")) and
                not any(HW.joinpath("results").glob(FAILURE_PREFIX + "*")),
                "M1074 canonical run namespace not fresh")
    # Deliberately do not stat, open, or hash M1072.ROWS here. Attempt must be
    # consumed before CanonicalRowReader performs any canonical-row operation.
    return {
        "status": "PASS_M1074_PRE_ATTEMPT_SOURCE_IDENTITY__ROWS_UNOPENED",
        "contract_sha256": CONTRACT_SHA,
        "m1072_source_sha256": M1072_SHA,
        "m1073_outer_seal_file_sha256": M1073_ID[2],
        "canonical_rows_opened_or_hashed": False,
    }


def validate_future_authority(
    runner: Path,
    m1075_identity: tuple[str, str, str],
) -> dict[str, Any]:
    source = validate_source_contract(runner, require_fresh=False)
    require(all(type(value) is str and len(value) == 64
                for value in m1075_identity), "M1075 identity tuple drift")
    verify_flat(M1075, m1075_identity)
    review = strict_json(M1075 / "review.json")
    require(review.get("status") == M1075_STATUS and
            review.get("verdict") == "GO_ONE_M1074_CPU_FULL_REPLAY_ONLY" and
            review.get("claim_boundary", {}).get("launch_now") is True and
            review.get("claim_boundary", {}).get("max_attempts") == 1 and
            review.get("claim_boundary", {}).get("automatic_retry") is False,
            "M1075 release hammer authority drift")
    identity = review.get("identity", {})
    require(identity.get("m1074_engine_sha256") == sha256(Path(__file__)) and
            identity.get("m1074_runner_sha256") == sha256(Path(runner)) and
            identity.get("m1074_contract_sha256") == CONTRACT_SHA and
            identity.get("m1072_source_sha256") == M1072_SHA and
            identity.get("m1073_outer_seal_file_sha256") == M1073_ID[2],
            "M1075 source binding drift")
    return {
        "status": "PASS_M1074_ONE_SHOT_AUTHORITY",
        "source": source,
        "m1075_review_sha256": m1075_identity[0],
        "m1075_manifest_sha256": m1075_identity[1],
        "m1075_outer_seal_file_sha256": m1075_identity[2],
    }


def payload_files(directory: Path) -> list[Path]:
    files = []
    for item in sorted(Path(directory).rglob("*")):
        relative = item.relative_to(directory)
        if relative.parts and relative.parts[0] == SEAL_DIR:
            continue
        require(not item.is_symlink(), "M1074 atomic seal refuses symlink")
        if item.is_file():
            files.append(item)
    return files


def verify_atomic_seal(directory: Path) -> dict[str, Any]:
    directory = Path(directory)
    bundle = directory / SEAL_DIR
    manifest = bundle / SEAL_MANIFEST
    outer = bundle / SEAL_OUTER
    require(bundle.is_dir() and not bundle.is_symlink() and
            manifest.is_file() and outer.is_file() and
            not manifest.is_symlink() and not outer.is_symlink(),
            "M1074 atomic seal absent or partial")
    require(outer.read_text(encoding="utf-8") ==
            sha256(manifest) + "  " + SEAL_MANIFEST + "\n",
            "M1074 atomic outer drift")
    listed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        require(relative not in listed, "M1074 duplicate manifest member")
        member = directory / relative
        require(member.resolve().is_relative_to(directory.resolve()) and
                member.is_file() and not member.is_symlink() and
                sha256(member) == digest, "M1074 manifest member drift")
        listed[relative] = digest
    actual = {item.relative_to(directory).as_posix()
              for item in payload_files(directory)}
    require(set(listed) == actual, "M1074 recursive manifest coverage drift")
    return {"manifest_sha256": sha256(manifest),
            "outer_seal_file_sha256": sha256(outer),
            "members": len(actual), "atomic_bundle": SEAL_DIR}


def atomic_seal(directory: Path, inject_fault: str = "") -> dict[str, Any]:
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(),
            "M1074 atomic seal target drift")
    if (directory / SEAL_DIR).exists():
        return verify_atomic_seal(directory)
    recover_partial_seal_stages(directory)
    members = payload_files(directory)
    require(members, "M1074 empty seal target")
    stage = directory.parent / (
        directory.name + ".m1074_seal_stage.%d.%d" % (os.getpid(), time.time_ns())
    )
    stage.mkdir(mode=0o700)
    lines = [sha256(item) + "  " + item.relative_to(directory).as_posix()
             for item in members]
    write_exclusive(stage / SEAL_MANIFEST, ("\n".join(lines) + "\n").encode())
    if inject_fault == "after_manifest":
        raise RuntimeError("M1074 injected interruption after manifest")
    write_exclusive(stage / SEAL_OUTER,
                    (sha256(stage / SEAL_MANIFEST) + "  " +
                     SEAL_MANIFEST + "\n").encode())
    fsync_dir(stage)
    if inject_fault == "before_rename":
        raise RuntimeError("M1074 injected interruption before atomic seal rename")
    rename_noreplace(stage, directory / SEAL_DIR)
    fsync_dir(directory)
    return verify_atomic_seal(directory)


def safe_result_sibling(path: Path, prefix: str,
                        allowed_parent: Path | None = None) -> None:
    path = Path(path)
    parent = RESULT.parent if allowed_parent is None else Path(allowed_parent)
    require(path.parent.resolve() == parent.resolve() and
            path.name.startswith(prefix) and not path.is_symlink(),
            "M1074 unsafe result sibling")


def consume_attempt(authority: Mapping[str, Any],
                    allowed_parent: Path | None = None) -> dict[str, Any]:
    parent = ATTEMPT.parent if allowed_parent is None else Path(allowed_parent)
    final = ATTEMPT if allowed_parent is None else parent / ATTEMPT.name
    safe_result_sibling(final, ATTEMPT.name, parent)
    require(not final.exists() and
            (allowed_parent is not None or not RESULT.exists()),
            "M1074 attempt namespace collision")
    # This exact canonical mkdir is the irreversible one-shot consumption
    # primitive. It precedes every CanonicalRowReader operation.
    try:
        final.mkdir(mode=0o700)
    except FileExistsError as error:
        raise RuntimeError("M1074 attempt namespace collision") from error
    fsync_dir(parent)
    receipt = {
        "schema": "m1074_full_exact_1rw_attempt_v1",
        "status": "CONSUMED_BEFORE_CANONICAL_ROWS_OPEN",
        "maximum_attempts": 1,
        "automatic_retry": False,
        "m1075_outer_seal_file_sha256":
            authority["m1075_outer_seal_file_sha256"],
        "m1072_source_sha256": M1072_SHA,
        "canonical_rows_opened_or_hashed_before_attempt": False,
    }
    write_exclusive(final / "attempt.json",
                    (json.dumps(receipt, sort_keys=True, allow_nan=False) +
                     "\n").encode())
    seal = atomic_seal(final)
    require(verify_atomic_seal(final) == seal,
            "M1074 attempt atomic publication drift")
    return {"receipt": receipt, "seal": seal}


def finalize_interrupted_attempt(attempt: Path = ATTEMPT) -> dict[str, Any]:
    """Best-effort durable closure after canonical mkdir consumed the attempt."""
    attempt = Path(attempt)
    require(attempt.is_dir() and not attempt.is_symlink(),
            "M1074 interrupted attempt root drift")
    if (attempt / SEAL_DIR).exists():
        return verify_atomic_seal(attempt)
    marker = attempt / "ATTEMPT_INTERRUPTED.json"
    if not marker.exists():
        write_exclusive(marker, (json.dumps({
            "schema": "m1074_interrupted_attempt_v1",
            "status": "ATTEMPT_CONSUMED__NO_RETRY",
            "canonical_rows_may_or_may_not_have_opened": True,
            "automatic_retry": False,
        }, sort_keys=True) + "\n").encode())
    return atomic_seal(attempt)


def validate_attempt(authority: Mapping[str, Any],
                     attempt: Path = ATTEMPT) -> dict[str, Any]:
    seal = verify_atomic_seal(attempt)
    receipt = strict_json(Path(attempt) / "attempt.json")
    require(receipt.get("status") == "CONSUMED_BEFORE_CANONICAL_ROWS_OPEN" and
            receipt.get("maximum_attempts") == 1 and
            receipt.get("automatic_retry") is False and
            receipt.get("canonical_rows_opened_or_hashed_before_attempt") is False and
            receipt.get("m1075_outer_seal_file_sha256") ==
                authority["m1075_outer_seal_file_sha256"],
            "M1074 attempt receipt drift")
    return {"receipt": receipt, "seal": seal}


def normalize_full_result(raw: Mapping[str, Any]) -> dict[str, Any]:
    require(type(raw) is dict and set(raw) == {
                "schema", "status", "samples", "coverage", "capacity",
                "claim_boundary"
            } and raw.get("schema") ==
            "m1072_canonical_full_exact_1rw_replay_result_v1" and
            raw.get("status") ==
            "PASS_M1072_CANONICAL_FULL_REPLAY_PENDING_RESULT_HAMMER",
            "M1072 raw full result identity drift")
    samples = raw.get("samples")
    coverage = raw.get("coverage")
    capacity = raw.get("capacity")
    require(type(samples) is list and len(samples) == M1072.SAMPLES and
            type(coverage) is dict and type(capacity) is dict,
            "M1072 raw full population drift")
    aggregate = {
        name: {"cycles": 0, "delayed_accesses": 0,
               "nominal_excess_accesses": 0}
        for name in M1072.DESIGNS
    }
    boundaries = []
    normalized_samples = []
    for sample, row in enumerate(samples):
        require(type(row) is dict and set(row) == {
                    "sample", "first_task_id", "last_task_id", "designs"
                } and row.get("sample") == sample and
                row.get("first_task_id") == sample * M1072.TASKS_PER_SAMPLE and
                row.get("last_task_id") ==
                    (sample + 1) * M1072.TASKS_PER_SAMPLE - 1 and
                type(row.get("designs")) is dict and
                set(row["designs"]) == set(M1072.DESIGNS),
                "M1072 sample boundary/order drift")
        boundaries.append({"sample": sample,
                           "first_task_id": row["first_task_id"],
                           "last_task_id": row["last_task_id"]})
        designs = {}
        for name in M1072.DESIGNS:
            item = row["designs"][name]
            require(type(item) is dict and set(item) == {
                "cycles_after_commit", "delayed_accesses",
                "nominal_excess_accesses"
            } and all(exact_int(value) and value >= 0
                      for value in item.values()) and
                    item["cycles_after_commit"] > 0,
                    "M1072 sample cycle/stall schema drift")
            designs[name] = dict(item)
            aggregate[name]["cycles"] += item["cycles_after_commit"]
            aggregate[name]["delayed_accesses"] += item["delayed_accesses"]
            aggregate[name]["nominal_excess_accesses"] += \
                item["nominal_excess_accesses"]
        normalized_samples.append({"sample": sample, "designs": designs})
    require(set(coverage) == {
                "schema", "checks", "full_coverage_pass", "service_digests",
                "execution_provenance_digest_sha256", "parent",
                "caller_supplied_coverage_or_digest"
            } and coverage.get("schema") ==
                "m1072_row_provenance_coverage_v1",
            "M1072 coverage schema drift")
    checks = coverage.get("checks")
    require(type(checks) is dict and set(checks) == {
                "exact_tasks", "exact_sample_commits", "exact_raw_rows",
                "exact_services", "exact_service_digest",
                "candidate_parent_conservation",
                "baseline_parent_accesses_zero", "baseline_work_equal"
            } and
            all(value is True for value in checks.values()) and
            coverage.get("full_coverage_pass") is True and
            coverage.get("caller_supplied_coverage_or_digest") is False,
            "M1072 full provenance coverage drift")
    service_digests = coverage.get("service_digests")
    require(type(service_digests) is dict and
            set(service_digests) == set(M1072.DESIGNS) and
            all(value == M1072.EXPECTED_SERVICE_DIGEST
                for value in service_digests.values()),
            "M1072 service digest drift")
    execution_digest = coverage.get("execution_provenance_digest_sha256")
    require(type(execution_digest) is str and len(execution_digest) == 64 and
            all(char in "0123456789abcdef" for char in execution_digest),
            "M1072 row-work provenance digest drift")
    parent = coverage.get("parent")
    require(type(parent) is dict and
            parent.get("candidate") == M1072.EXPECTED_CANDIDATE_PARENT and
            all(parent.get(name, {}).get(key, 0) == 0
                for name in ("strongest_zero", "same_coordinate_bit")
                for key in ("reads", "writes", "forwards")) and
            parent["strongest_zero"].get("work_cycles") ==
                parent["same_coordinate_bit"].get("work_cycles"),
            "M1072 parent conservation drift")
    require(capacity == M1072.M1064.derive_physical_capacity(),
            "M1072 capacity boundary drift")
    raw_boundary = raw["claim_boundary"]
    require(type(raw_boundary) is dict and
            raw_boundary == {
                "capacity_only_214912B_admitted": False,
                "matched_cycles_admitted": False,
                "speedup_admitted": False,
                "rtl_cycles": False,
                "paper_ppa_ready": False,
                "independent_result_hammer_required": True,
            }, "M1072 raw claim boundary drift")
    ratios = {
        name + "_over_candidate":
            aggregate[name]["cycles"] / aggregate["candidate"]["cycles"]
        for name in ("strongest_zero", "same_coordinate_bit")
    }
    return {
        "sample_boundaries": boundaries,
        "samples": normalized_samples,
        "aggregate": aggregate,
        "raw_cycle_ratios_pending_hammer": ratios,
        "capacity": capacity,
        "service_counts_per_design": {
            name: dict(M1072.EXPECTED_SERVICES) for name in M1072.DESIGNS
        },
        "service_digests": service_digests,
        "row_work_execution_provenance_digest_sha256": execution_digest,
        "coverage_checks": checks,
        "parent": parent,
    }


def execute_full(work: Path, authority: Mapping[str, Any]) -> dict[str, Any]:
    work = Path(work)
    safe_result_sibling(work, WORK_PREFIX)
    validate_attempt(authority)
    require(not work.exists() and not RESULT.exists(),
            "M1074 work/result collision")
    work.mkdir(mode=0o700)
    write_exclusive(work / "RUN_STARTED.json", (json.dumps({
        "status": "ATTEMPT_ALREADY_CONSUMED__OPEN_ROWS_NEXT",
        "m1072_source_sha256": M1072_SHA,
        "m1075_outer_seal_file_sha256":
            authority["m1075_outer_seal_file_sha256"],
    }, sort_keys=True) + "\n").encode())
    try:
        generator = M1072.iter_canonical_full_replay_results()
        raw = next(generator)
        try:
            next(generator)
        except StopIteration:
            exhausted = True
        else:
            exhausted = False
        require(exhausted, "M1072 production iterator yielded extra result")
        normalized = normalize_full_result(raw)
        result = {
            "schema": RESULT_SCHEMA,
            "status": RESULT_STATUS,
            "authority": {
                "m1072_source_sha256": M1072_SHA,
                "m1073_outer_seal_file_sha256": M1073_ID[2],
                "m1075_outer_seal_file_sha256":
                    authority["m1075_outer_seal_file_sha256"],
                "contract_sha256": CONTRACT_SHA,
            },
            "geometry": {"samples": M1072.SAMPLES, "tasks": M1072.TASKS,
                         "raw_rows": M1072.M1064.RAW_ROWS,
                         "designs": list(M1072.DESIGNS)},
            "full_replay": normalized,
            "claim_boundary": {
                "raw_full_replay_complete": True,
                "independent_result_hammer_required": True,
                "capacity_only_214912B_admitted": False,
                "full_trace_port_feasibility": False,
                "matched_cycles_admitted": False,
                "speedup_admitted": False,
                "rtl_cycles": False,
                "paper_ppa_ready": False,
            },
        }
        write_exclusive(work / PAYLOAD,
                        (json.dumps(result, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n").encode())
        write_exclusive(work / "RUN_COMPLETE.txt",
                        b"M1074_RAW_FULL_REPLAY_COMPLETE__RESULT_HAMMER_REQUIRED\n")
        seal = atomic_seal(work)
        return {"status": RESULT_STATUS, "seal": seal,
                "payload_sha256": sha256(work / PAYLOAD)}
    except BaseException:
        if not (work / "traceback.log").exists():
            write_exclusive(work / "traceback.log", traceback.format_exc().encode())
        raise


def publish_result(work: Path) -> dict[str, Any]:
    work = Path(work)
    safe_result_sibling(work, WORK_PREFIX)
    seal = verify_atomic_seal(work)
    payload = strict_json(work / PAYLOAD)
    require(payload.get("status") == RESULT_STATUS and
            payload.get("claim_boundary", {}).get(
                "raw_full_replay_complete") is True and
            payload.get("claim_boundary", {}).get(
                "independent_result_hammer_required") is True and
            payload.get("claim_boundary", {}).get("speedup_admitted") is False and
            not RESULT.exists(), "M1074 raw result publication drift")
    rename_noreplace(work, RESULT)
    fsync_dir(RESULT.parent)
    require(verify_atomic_seal(RESULT) == seal,
            "M1074 atomic result publication identity drift")
    return {"status": RESULT_STATUS, "result": str(RESULT), "seal": seal}


def verify_published_result() -> dict[str, Any]:
    seal = verify_atomic_seal(RESULT)
    payload = strict_json(RESULT / PAYLOAD)
    require(payload.get("schema") == RESULT_SCHEMA and
            payload.get("status") == RESULT_STATUS and
            payload.get("claim_boundary", {}).get(
                "raw_full_replay_complete") is True and
            payload.get("claim_boundary", {}).get(
                "independent_result_hammer_required") is True and
            payload.get("claim_boundary", {}).get("speedup_admitted") is False,
            "M1074 published result claim drift")
    return {"status": RESULT_STATUS, "result": str(RESULT), "seal": seal}


def quarantine_work(work: Path, quarantine: Path, return_code: int,
                    phase: str, allowed_parent: Path | None = None) -> dict[str, Any]:
    work = Path(work)
    quarantine = Path(quarantine)
    parent = RESULT.parent if allowed_parent is None else Path(allowed_parent)
    safe_result_sibling(work, WORK_PREFIX, parent)
    safe_result_sibling(quarantine, FAILURE_PREFIX, parent)
    if allowed_parent is None and ATTEMPT.exists():
        finalize_interrupted_attempt(ATTEMPT)
    stage = parent / (quarantine.name + ".stage")
    require(not quarantine.exists() and not stage.exists(),
            "M1074 failure quarantine collision")
    stage.mkdir(mode=0o700)
    if work.exists():
        require(work.is_dir() and not work.is_symlink(),
                "M1074 failed work root drift")
        rename_noreplace(work, stage / "partial_result")
    partial_stages = sorted(
        item for item in parent.iterdir()
        if item.name.startswith(work.name + ".m1074_seal_stage.")
    )
    if partial_stages:
        recovery = stage / "partial_result_seal_stages"
        recovery.mkdir(mode=0o700)
        for index, partial in enumerate(partial_stages):
            rename_noreplace(partial, recovery / ("attempt_%03d" % index))
    write_exclusive(stage / "failure.json", (json.dumps({
        "schema": "m1074_failure_quarantine_v1",
        "status": "FAILED_OR_INTERRUPTED__NO_RETRY",
        "return_code": int(return_code),
        "phase": str(phase),
        "attempt_consumed": True,
        "automatic_retry": False,
    }, sort_keys=True, allow_nan=False) + "\n").encode())
    seal = atomic_seal(stage)
    rename_noreplace(stage, quarantine)
    fsync_dir(parent)
    require(verify_atomic_seal(quarantine) == seal,
            "M1074 failure quarantine publication drift")
    return {"status": "PASS_M1074_SEALED_FAILURE_QUARANTINE",
            "quarantine": str(quarantine), "seal": seal}


def synthetic_raw_result() -> dict[str, Any]:
    samples = []
    for sample in range(M1072.SAMPLES):
        samples.append({
            "sample": sample,
            "first_task_id": sample * M1072.TASKS_PER_SAMPLE,
            "last_task_id": (sample + 1) * M1072.TASKS_PER_SAMPLE - 1,
            "designs": {
                name: {"cycles_after_commit": 1000 + sample,
                       "delayed_accesses": sample,
                       "nominal_excess_accesses": sample + 1}
                for name in M1072.DESIGNS
            },
        })
    coverage = {
        "schema": "m1072_row_provenance_coverage_v1",
        "checks": {"exact_tasks": True, "exact_raw_rows": True,
                   "exact_sample_commits": True,
                   "exact_services": True, "exact_service_digest": True,
                   "candidate_parent_conservation": True,
                   "baseline_parent_accesses_zero": True,
                   "baseline_work_equal": True},
        "full_coverage_pass": True,
        "service_digests": {name: M1072.EXPECTED_SERVICE_DIGEST
                            for name in M1072.DESIGNS},
        "execution_provenance_digest_sha256": "a" * 64,
        "parent": {
            "candidate": dict(M1072.EXPECTED_CANDIDATE_PARENT),
            "strongest_zero": {"reads": 0, "writes": 0, "forwards": 0,
                               "work_cycles": 55},
            "same_coordinate_bit": {"reads": 0, "writes": 0, "forwards": 0,
                                    "work_cycles": 55},
        },
        "caller_supplied_coverage_or_digest": False,
    }
    return {
        "schema": "m1072_canonical_full_exact_1rw_replay_result_v1",
        "status": "PASS_M1072_CANONICAL_FULL_REPLAY_PENDING_RESULT_HAMMER",
        "samples": samples,
        "coverage": coverage,
        "capacity": M1072.M1064.derive_physical_capacity(),
        "claim_boundary": {
            "capacity_only_214912B_admitted": False,
            "matched_cycles_admitted": False,
            "speedup_admitted": False,
            "rtl_cycles": False,
            "paper_ppa_ready": False,
            "independent_result_hammer_required": True,
        },
    }


def source_self_test() -> dict[str, Any]:
    source = validate_source_contract(require_fresh=True)
    normalized = normalize_full_result(synthetic_raw_result())
    require(len(normalized["sample_boundaries"]) == 10 and
            normalized["aggregate"]["candidate"]["cycles"] == 10_045 and
            normalized["row_work_execution_provenance_digest_sha256"] == "a" * 64,
            "M1074 synthetic normalization drift")
    with tempfile.TemporaryDirectory(prefix="m1074_source_") as temp:
        parent = Path(temp)
        payload = parent / "payload"
        payload.mkdir()
        write_exclusive(payload / "data.json", b"{}\n")
        seal = atomic_seal(payload)
        require(verify_atomic_seal(payload) == seal,
                "M1074 atomic seal self-test drift")
        work = parent / (WORK_PREFIX + "failure")
        work.mkdir()
        write_exclusive(work / "partial", b"partial\n")
        quarantine = parent / (FAILURE_PREFIX + "failure")
        failure = quarantine_work(work, quarantine, 143, "INJECTED", parent)
        require(not work.exists() and quarantine.exists() and
                failure["status"] == "PASS_M1074_SEALED_FAILURE_QUARANTINE",
                "M1074 quarantine self-test drift")
    return {
        "status": "PASS_M1074_SOURCE_SELF_TEST__NO_FULL_REPLAY_NO_ATTEMPT",
        "source": source,
        "synthetic_ten_sample_normalization": True,
        "atomic_complete_seal": True,
        "failure_quarantine": True,
        "m1072_generator_advanced": False,
        "canonical_rows_opened_or_hashed": False,
        "attempt_consumed": False,
        "full_replay_executed": False,
        "eda_gpu_remote_used": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate-source", action="store_true")
    parser.add_argument("--validate-authority", action="store_true")
    parser.add_argument("--consume-attempt", action="store_true")
    parser.add_argument("--execute-full", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--verify-published", action="store_true")
    parser.add_argument("--quarantine-work", action="store_true")
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--quarantine", type=Path)
    parser.add_argument("--return-code", type=int, default=1)
    parser.add_argument("--phase", default="UNKNOWN")
    parser.add_argument("--expected-m1075-review-sha", default="")
    parser.add_argument("--expected-m1075-manifest-sha", default="")
    parser.add_argument("--expected-m1075-outer-sha", default="")
    args = parser.parse_args(argv)
    modes = (args.self_test, args.validate_source, args.validate_authority,
             args.consume_attempt, args.execute_full, args.publish,
             args.verify_published,
             args.quarantine_work)
    require(sum(bool(mode) for mode in modes) == 1,
            "M1074 requires exactly one explicit mode")
    if args.self_test:
        value = source_self_test()
    elif args.validate_source:
        value = validate_source_contract(args.runner)
    elif args.quarantine_work:
        require(args.work is not None and args.quarantine is not None,
                "M1074 quarantine paths required")
        value = quarantine_work(args.work, args.quarantine,
                                args.return_code, args.phase)
    elif args.verify_published:
        value = verify_published_result()
    else:
        require(args.runner is not None, "M1074 runner required")
        identity = (args.expected_m1075_review_sha,
                    args.expected_m1075_manifest_sha,
                    args.expected_m1075_outer_sha)
        authority = validate_future_authority(args.runner, identity)
        if args.validate_authority:
            value = authority
        elif args.consume_attempt:
            value = consume_attempt(authority)
        else:
            validate_attempt(authority)
            if args.execute_full:
                require(args.work is not None, "M1074 work required")
                value = execute_full(args.work, authority)
            else:
                require(args.work is not None, "M1074 work required")
                value = publish_result(args.work)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
