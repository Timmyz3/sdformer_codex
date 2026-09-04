#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1102 source-only atomic library for the legal-work8 C1 replay.

The CLI is read-only.  No launcher is included.  A different author must pin
this library, the M1102 source contract and an independent source hammer in a
zero-argument wrapper before exactly one production attempt may be consumed.
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

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HERE / "run_m1102_c1_work8_exact_1rw_source.py"
SOURCE_SHA = "95bd50aebcc473ab69cdea6ccf27d54743c89926c5e0f31199dc469ced9bf7cc"
CONTRACT = HW / "contracts/m1102_c1_legal_work8_exact_1rw_additive_source_contract_r1_20260830.json"
CONTRACT_SHA = "fad9c381fc1e55fc78d6cf4b95ad0959b5a7089989a7acce1ccfafa73714db6e"
CONTRACT_SIDECAR_SHA = "e6754574c804a7ed2cfd39e5a99c991db38402389901fef570359decf43e3607"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
RESULT = HW / "results/m1102_c1_work8_exact_1rw_full_replay_r1_20260830"
ATTEMPT = HW / "results/.m1102_c1_work8_exact_1rw_full_replay_attempt_consumed"
LOCK = HW / "results/.m1102_c1_work8_exact_1rw_full_replay.lock"
WORK_PREFIX = ".m1102_c1_work8_exact_1rw_full_replay_work."
FAILURE_PREFIX = RESULT.name + ".failed_or_incomplete."
PAYLOAD = "m1102_c1_work8_exact_1rw_full_replay_result_r1.json"
PREFLIGHT = "m1102_work8_domain_preflight_receipt_r1.json"
SEAL_DIR = ".m1102_atomic_seal"
MANIFEST = "SHA256SUMS"
OUTER = "SHA256SUMS.seal.sha256"
TASKS = 812160
VALUES = 2436480
SAMPLES = 10
DESIGNS = ("candidate", "strongest_zero", "same_coordinate_bit")
WORK_DIGEST = "480c6fe7ea316279bd662ff34cf4cecc1aaee1196dc9d82fc76517d8c7fb3d83"
PROVENANCE_DIGEST = "e7a84f88706b27f9c8ba0ade1f4d80c111b4dd93ac11e1b96a589bbad28f0b11"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


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


def lower_sha(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value)


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
    function = getattr(libc, "renameat2", None)
    require(function is not None, "M1102 renameat2 unavailable")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p,
                         ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise RuntimeError("M1102 atomic no-replace collision")
        raise OSError(code, os.strerror(code), str(destination))


def verify_double_seal(path: Path, file_sha: str, sidecar_sha: str) -> None:
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require(path.is_file() and not path.is_symlink() and sha256(path) == file_sha and
            sidecar.is_file() and not sidecar.is_symlink() and
            sha256(sidecar) == sidecar_sha,
            "M1102 contract identity drift")
    expected, name = sidecar.read_text(encoding="utf-8").split()
    require(expected == file_sha and name == path.name,
            "M1102 contract sidecar drift")
    expected, name = outer.read_text(encoding="utf-8").split()
    require(expected == sidecar_sha and name == sidecar.name,
            "M1102 contract outer drift")


def load_source():
    require(SOURCE.is_file() and not SOURCE.is_symlink() and
            sha256(SOURCE) == SOURCE_SHA, "M1102 semantic source drift")
    spec = importlib.util.spec_from_file_location("m1102_atomic_semantics", SOURCE)
    require(spec is not None and spec.loader is not None,
            "cannot load M1102 semantic source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M1102 = load_source()


def validate_source_contract(require_fresh: bool = True) -> dict[str, Any]:
    verify_double_seal(CONTRACT, CONTRACT_SHA, CONTRACT_SIDECAR_SHA)
    contract = strict_json(CONTRACT)
    require(contract.get("status") ==
            "PASS_M1102_ADDITIVE_SOURCE_CONTRACT__NO_LAUNCH_NO_ATTEMPT" and
            contract.get("authority", {}).get("m1102_source_sha256") == SOURCE_SHA and
            contract.get("canonical_population", {}).get(
                "task_design_work_values") == VALUES and
            contract.get("canonical_population", {}).get(
                "task_design_work_digest_sha256") == WORK_DIGEST and
            contract.get("canonical_population", {}).get(
                "row_work_execution_provenance_digest_sha256") == PROVENANCE_DIGEST and
            contract.get("claim_boundary", {}).get("launch_now") is False and
            sha256(DOCS359) == DOCS359_SHA and
            len(inspect.signature(M1102.canonical_work_domain_preflight).parameters) == 0 and
            inspect.isgeneratorfunction(M1102.iter_canonical_full_replay_results) and
            len(inspect.signature(M1102.iter_canonical_full_replay_results).parameters) == 0,
            "M1102 source contract content drift")
    if require_fresh:
        require(not RESULT.exists() and not ATTEMPT.exists() and not LOCK.exists() and
                not any(RESULT.parent.glob(WORK_PREFIX + "*")) and
                not any(RESULT.parent.glob(FAILURE_PREFIX + "*")),
                "M1102 production namespace not fresh")
    return {
        "status": "PASS_M1102_ATOMIC_LIBRARY_IDENTITIES__NO_CANONICAL_PAYLOAD",
        "source_sha256": SOURCE_SHA,
        "contract_sha256": CONTRACT_SHA,
        "canonical_payload_opened_or_hashed": False,
        "attempt_created": False,
    }


def payload_files(directory: Path) -> list[Path]:
    files = []
    for item in sorted(Path(directory).rglob("*")):
        relative = item.relative_to(directory)
        if relative.parts and relative.parts[0] == SEAL_DIR:
            continue
        require(not item.is_symlink(), "M1102 seal refuses symlink")
        if item.is_file():
            files.append(item)
    return files


def verify_atomic_seal(directory: Path) -> dict[str, Any]:
    bundle = Path(directory) / SEAL_DIR
    manifest, outer = bundle / MANIFEST, bundle / OUTER
    require(bundle.is_dir() and not bundle.is_symlink() and
            manifest.is_file() and not manifest.is_symlink() and
            outer.is_file() and not outer.is_symlink(),
            "M1102 atomic seal absent")
    require(outer.read_text(encoding="utf-8") ==
            sha256(manifest) + "  " + MANIFEST + "\n",
            "M1102 outer seal drift")
    listed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        member = Path(directory) / relative
        require(relative not in listed and member.is_file() and
                not member.is_symlink() and sha256(member) == digest,
                "M1102 manifest member drift")
        listed[relative] = digest
    actual = {item.relative_to(directory).as_posix()
              for item in payload_files(directory)}
    require(set(listed) == actual, "M1102 manifest coverage drift")
    return {"manifest_sha256": sha256(manifest),
            "outer_seal_file_sha256": sha256(outer),
            "members": len(actual)}


def atomic_seal(directory: Path) -> dict[str, Any]:
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(),
            "M1102 seal target drift")
    require(not (directory / SEAL_DIR).exists(), "M1102 duplicate seal")
    members = payload_files(directory)
    require(members, "M1102 empty seal target")
    stage = directory.parent / (directory.name + ".m1102_seal_stage.%d.%d" %
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


def _safe_sibling(path: Path, prefix: str, root: Path) -> None:
    require(path.parent.resolve() == root.resolve() and
            path.name.startswith(prefix) and not path.is_symlink(),
            "M1102 unsafe runtime sibling")


def _validate_launch_authority(authority: Mapping[str, Any]) -> None:
    expected = {
        "status", "launch_wrapper_sha256", "launch_hammer_outer_seal_file_sha256",
        "m1102_atomic_library_sha256", "m1102_semantic_source_sha256",
        "m1102_contract_sha256", "m1100_outer_seal_file_sha256",
        "m1101_outer_seal_file_sha256",
    }
    require(type(authority) is dict and set(authority) == expected and
            authority.get("status") ==
                "PASS_DIFFERENT_AUTHOR_M1102_HARDCODED_LAUNCH_AUTHORITY" and
            all(lower_sha(authority[key]) for key in expected - {"status"}) and
            authority["m1102_atomic_library_sha256"] == sha256(Path(__file__)) and
            authority["m1102_semantic_source_sha256"] == SOURCE_SHA and
            authority["m1102_contract_sha256"] == CONTRACT_SHA and
            authority["m1100_outer_seal_file_sha256"] == M1102.M1100_ID[2] and
            authority["m1101_outer_seal_file_sha256"] == M1102.M1101_ID[2],
            "M1102 hardcoded launch authority drift")


def consume_attempt(authority: Mapping[str, Any],
                    parent: Path | None = None) -> dict[str, Any]:
    _validate_launch_authority(authority)
    root = ATTEMPT.parent if parent is None else Path(parent)
    final = ATTEMPT if parent is None else root / ATTEMPT.name
    _safe_sibling(final, ATTEMPT.name, root)
    require(not final.exists() and (parent is not None or not RESULT.exists()),
            "M1102 attempt collision")
    try:
        final.mkdir(mode=0o700)
    except FileExistsError as error:
        raise RuntimeError("M1102 attempt collision") from error
    fsync_dir(root)
    receipt = {
        "schema": "m1102_c1_work8_full_replay_attempt_r1_v1",
        "status": "CONSUMED_BEFORE_CANONICAL_PAYLOAD_ACCESS",
        "maximum_attempts": 1,
        "automatic_retry": False,
        "canonical_payload_opened_or_hashed_before_attempt": False,
        "m1102_semantic_source_sha256": SOURCE_SHA,
        "launch_hammer_outer_seal_file_sha256":
            authority["launch_hammer_outer_seal_file_sha256"],
    }
    write_exclusive(final / "attempt.json",
                    (json.dumps(receipt, sort_keys=True) + "\n").encode())
    return {"receipt": receipt, "seal": atomic_seal(final)}


def validate_preflight(value: Any) -> dict[str, Any]:
    require(type(value) is dict and value.get("schema") ==
            "m1102_c1_work8_domain_and_geometry_preflight_v1" and
            value.get("status") ==
            "PASS_M1102_EXHAUSTIVE_812160X3_AND_12522_WORK8_REGRESSION" and
            value.get("tasks") == TASKS and value.get("values_checked") == VALUES and
            value.get("designs") == list(DESIGNS) and
            value.get("work8_occurrences_total") == 12522 and
            value.get("task_design_work_digest_sha256") == WORK_DIGEST and
            value.get("row_work_execution_provenance_digest_sha256") == PROVENANCE_DIGEST and
            value.get("full_coverage_pass") is True and
            value.get("production_full_cycle_iterator_called") is False and
            value.get("attempt_created") is False and
            value.get("cycles_or_speedup_admitted") is False,
            "M1102 exhaustive preflight drift")
    expected_counts = {"zero": 74106, "work8": 4174,
                       "positive_ge16": 733880}
    require(value.get("counts") == {name: expected_counts for name in DESIGNS} and
            all(row == {"occurrences": 4174, "fresh_pass": 4174,
                        "delayed_raw_pass": 4174,
                        "raw_dependencies_pass": 4174,
                        "minimum_dependency_delay": 0}
                for row in value.get("work8_geometry", {}).values()) and
            set(value.get("work8_geometry", {})) == set(DESIGNS),
            "M1102 preflight count/geometry drift")
    return value


def normalize_raw(raw: Any) -> dict[str, Any]:
    require(type(raw) is dict and raw.get("schema") ==
            "m1102_canonical_full_work8_exact_1rw_replay_result_v1" and
            raw.get("status") ==
            "PASS_M1102_RAW_CPU_MODEL_FULL_REPLAY_PENDING_RESULT_HAMMER",
            "M1102 raw result identity drift")
    samples = raw.get("samples")
    coverage = raw.get("coverage")
    capacity = raw.get("capacity")
    require(type(samples) is list and len(samples) == SAMPLES and
            type(coverage) is dict and coverage.get("full_coverage_pass") is True and
            coverage.get("execution_provenance_digest_sha256") == PROVENANCE_DIGEST and
            type(capacity) is dict and capacity.get("derived_total_bytes") == 214912 and
            capacity.get("budget_bytes") == 245760 and
            capacity.get("capacity_bytes_pass") is True,
            "M1102 raw coverage/capacity drift")
    aggregate = {name: {"cycles": 0, "delayed_accesses": 0,
                        "nominal_excess_accesses": 0} for name in DESIGNS}
    for sample, row in enumerate(samples):
        require(row.get("sample") == sample and
                row.get("first_task_id") == sample * M1102.M1072.TASKS_PER_SAMPLE and
                row.get("last_task_id") ==
                    (sample + 1) * M1102.M1072.TASKS_PER_SAMPLE - 1 and
                set(row.get("designs", {})) == set(DESIGNS),
                "M1102 sample boundary drift")
        for name in DESIGNS:
            entry = row["designs"][name]
            require(set(entry) == {"cycles_after_commit", "delayed_accesses",
                                   "nominal_excess_accesses"} and
                    all(type(entry[key]) is int and entry[key] >= 0 for key in entry),
                    "M1102 sample metric drift")
            aggregate[name]["cycles"] += entry["cycles_after_commit"]
            aggregate[name]["delayed_accesses"] += entry["delayed_accesses"]
            aggregate[name]["nominal_excess_accesses"] += entry[
                "nominal_excess_accesses"]
    boundary = raw.get("claim_boundary", {})
    require(boundary.get("matched_cycles_admitted") is False and
            boundary.get("speedup_admitted") is False and
            boundary.get("independent_result_hammer_required") is True,
            "M1102 raw claim boundary drift")
    return {"samples": samples, "coverage": coverage,
            "capacity": capacity, "aggregate": aggregate}


def execute_full(authority: Mapping[str, Any], work: Path) -> dict[str, Any]:
    """Only production path: consumed attempt, exhaustive preflight, one iterator."""
    _validate_launch_authority(authority)
    _safe_sibling(work, WORK_PREFIX, RESULT.parent)
    require(not work.exists() and ATTEMPT.is_dir() and not RESULT.exists(),
            "M1102 work/attempt state drift")
    work.mkdir(mode=0o700)
    try:
        preflight = validate_preflight(M1102.canonical_work_domain_preflight())
        write_exclusive(work / PREFLIGHT,
                        (json.dumps(preflight, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n").encode())
        generator = M1102.iter_canonical_full_replay_results()
        raw = next(generator)
        try:
            next(generator)
        except StopIteration:
            pass
        else:
            raise RuntimeError("M1102 iterator yielded more than once")
        normalized = normalize_raw(raw)
        result = {
            "schema": "m1102_c1_work8_exact_1rw_full_replay_result_r1_v1",
            "status": "PASS_M1102_RAW_CPU_MODEL_FULL_REPLAY_PENDING_RESULT_HAMMER",
            "authority": dict(authority),
            "work_domain_preflight": preflight,
            "raw_cpu_model": normalized,
            "claim_boundary": {
                "raw_cpu_model_full_replay_complete": True,
                "independent_result_hammer_required": True,
                "matched_cycles_admitted": False,
                "speedup_admitted": False,
                "rtl_cycles": False,
                "paper_citable": False,
                "paper_ppa_ready": False,
            },
        }
        write_exclusive(work / PAYLOAD,
                        (json.dumps(result, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n").encode())
        write_exclusive(work / "RUN_COMPLETE.txt",
                        b"M1102_RAW_CPU_MODEL_COMPLETE__RESULT_HAMMER_REQUIRED\n")
        return {"status": result["status"], "seal": atomic_seal(work),
                "payload_sha256": sha256(work / PAYLOAD)}
    except BaseException:
        if not (work / "traceback.log").exists():
            write_exclusive(work / "traceback.log", traceback.format_exc().encode())
        raise


def publish_result(work: Path) -> dict[str, Any]:
    _safe_sibling(work, WORK_PREFIX, RESULT.parent)
    seal = verify_atomic_seal(work)
    payload = strict_json(work / PAYLOAD)
    require(payload.get("status") ==
            "PASS_M1102_RAW_CPU_MODEL_FULL_REPLAY_PENDING_RESULT_HAMMER" and
            payload.get("claim_boundary", {}).get("speedup_admitted") is False and
            not RESULT.exists(), "M1102 publish boundary drift")
    rename_noreplace(work, RESULT)
    fsync_dir(RESULT.parent)
    require(verify_atomic_seal(RESULT) == seal, "M1102 publish identity drift")
    return {"status": payload["status"], "result": str(RESULT), "seal": seal}


def quarantine_work(work: Path, quarantine: Path, return_code: int,
                    phase: str) -> dict[str, Any]:
    _safe_sibling(work, WORK_PREFIX, RESULT.parent)
    _safe_sibling(quarantine, FAILURE_PREFIX, RESULT.parent)
    require(not quarantine.exists(), "M1102 quarantine collision")
    stage = Path(str(quarantine) + ".stage")
    require(not stage.exists(), "M1102 quarantine stage collision")
    stage.mkdir(mode=0o700)
    if work.exists():
        rename_noreplace(work, stage / "partial_result")
    write_exclusive(stage / "failure.json", (json.dumps({
        "schema": "m1102_failure_quarantine_r1_v1",
        "status": "FAILED_OR_INTERRUPTED__NO_RETRY",
        "return_code": int(return_code),
        "phase": str(phase),
        "attempt_consumed": True,
        "automatic_retry": False,
    }, sort_keys=True) + "\n").encode())
    seal = atomic_seal(stage)
    rename_noreplace(stage, quarantine)
    fsync_dir(RESULT.parent)
    return {"status": "PASS_M1102_SEALED_FAILURE_QUARANTINE",
            "quarantine": str(quarantine), "seal": seal}


def verify_published_result() -> dict[str, Any]:
    seal = verify_atomic_seal(RESULT)
    payload = strict_json(RESULT / PAYLOAD)
    require(payload.get("status") ==
            "PASS_M1102_RAW_CPU_MODEL_FULL_REPLAY_PENDING_RESULT_HAMMER" and
            payload.get("work_domain_preflight", {}).get("values_checked") == VALUES and
            payload.get("claim_boundary", {}).get("speedup_admitted") is False,
            "M1102 published result drift")
    return {"status": payload["status"], "seal": seal}


def source_self_test() -> dict[str, Any]:
    identities = validate_source_contract(require_fresh=True)
    oracle = M1102.source_small_oracle()
    require(oracle.get("status") == "PASS_M1102_WORK8_SOURCE_SMALL_ORACLE" and
            oracle.get("attempt_created") is False,
            "M1102 semantic oracle drift")
    return {
        "status": "PASS_M1102_ATOMIC_LIBRARY_SOURCE_SELF_TEST__NO_ATTEMPT",
        "identities": identities,
        "semantic_oracle": oracle,
        "production_preflight_called": False,
        "production_full_cycle_iterator_called": False,
        "attempt_created": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate-source", action="store_true")
    parser.add_argument("--verify-published", action="store_true")
    args = parser.parse_args(argv)
    modes = (args.self_test, args.validate_source, args.verify_published)
    require(sum(bool(mode) for mode in modes) == 1,
            "select exactly one read-only mode")
    if args.self_test:
        result = source_self_test()
    elif args.validate_source:
        result = validate_source_contract()
    else:
        result = verify_published_result()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
