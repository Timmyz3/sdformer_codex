#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1202 different-author bounded source hammer for M1199.

This checker never opens the sealed 836 MB M1141 production JSONL and never
launches M1199 production.  It pins the full admitted dependency chain, runs
the source tests under an audit hook, independently checks the II=2 recurrence,
attacks bounded streams, and exercises success/failure atomicity only after all
production paths have been rebound to temporary six-record fixtures.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import re
import stat
import sys
import tempfile
import unittest
from typing import Any, Iterator

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent

SOURCE = HW / "system_simulator/scripts/run_m1199_c1_ii2_service_aware_production_consumer_one_shot_source.py"
TESTS = HW / "system_simulator/tests/test_m1199_c1_ii2_service_aware_production_consumer_source.py"
CONTRACT = HW / "contracts/m1199_c1_ii2_service_aware_production_consumer_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1199_c1_ii2_service_aware_production_consumer_source_author_receipt_r1_20260830"
M1161 = HW / "results/m1161ca_c1_production_real_replay_r1_20260830"
M1196 = HW / "reviews/m1196_m1161ca_c1_production_real_replay_result_hammer_r1_20260830"
M1169 = HW / "system_simulator/scripts/build_m1169_c1_ii2_service_aware_interval_replay_source.py"
M1170 = HW / "reviews/m1170_m1169_c1_ii2_service_aware_interval_replay_source_hammer_r1_20260830"
M1141 = HW / "results/m1141ca_c1_production_schedule_release_r1_20260830"
PRODUCTION_JSONL = M1141 / "m1141ca_per_task_schedule_records.jsonl"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

SOURCE_SHA = "b77bde3c15e74e6320e39ea2b0f4066ff3d8cbc7af945d88d77324f148a24768"
TESTS_SHA = "781e45dded3a17df8d098e584ef1ecc06778d4dd4f00c3a3ab0bc241e858ab64"
CONTRACT_ID = (
    "0f277f7f8f9437ce0692d5e4ce8c167d288894be6486cadef968455a2eae3ecb",
    "69eb75bf0a471316945cbdc67927d9f99467c8e245fe053284983189f7cd46d4",
    "bba87f25ace5739f780cccf0de98a101928a051d678fbb98da5f5a2b8d539015",
)
AUTHOR_ID = (
    "3a9bbe91eec177064aa64ea79389023b18e3a425369f0e143f7ee2cfa1da9935",
    "46cfdfdeab1d687da591a6c000049868e5ceeffdbf3489a5e35ac522fd56c0e4",
)
M1161_ID = (
    "b6c2be64d8cb32fcf0c31ae44070b5efdcb10d0db2661dddb0ec2c4cc3733198",
    "7bb4ff9dc40a9764d9312c1639a022756305c0170c483854a84c02d2a6cf5b5c",
)
M1196_ID = (
    "7b1a8b4fa8f1e2a6c361817c65ba198f76e332f5ed09a5199b96c699e241a65e",
    "174dee393c022db03dc315266e0d90f4ba45892147d4d69b01b970ffb1f16092",
    "8b919a0ad6e6ba6638ba6c21a5fbe993dfde0097fddc327001b5c4c5543a8dd0",
)
M1169_SHA = "bd243ca34760757cadbf9c1104049480197f1fb77bf6ad6ec1071870250ebc4f"
M1169_CONTRACT_SHA = "275214c40e1a53b922c1db448dcedff8792f5232124fc1ea5d474360ded861dc"
M1170_ID = (
    "c52c7bb2086e2ad638b7b91656c9c21c1fe517d81fa032a158973a2867f57f16",
    "5a3d7a821190c39d4b1213517e81f240ec2cd8e1a1e557832d6c404c74291af0",
    "0e1cf625aee653b734b2e949a459fe9d8ac3c9b95d830c772a9682b5e7c3bebd",
)
M1141_ID = (
    "4c4d264a9ac1e084c8c0acf0a6d150140f95ee96ee967b038ea4c1eefcc2b58c",
    "852b48c0d8098ef69a619925f82a8e1a308e87f2faf9ea76becabf51d52caace",
    "0b6549ce38a62bcb22e8a97d0c038860f5698fabc0d9bff162dc6af95d4f043a",
    "4d4e0e6396ac1061aca7ada142bc2761bf12a785e5373640a28503e3d73a0a81",
    836_268_740,
)
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
EXPECTED_UID = 1913
AXES = ("candidate", "strongest_zero", "same_coordinate_bit")
MANIFEST = "SHA256SUMS"
OUTER = "SHA256SUMS.seal.sha256"


class Failure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise Failure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_bytes(payload: bytes) -> Any:
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    return json.loads(payload.decode("utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("nonfinite JSON: " + token)))


def strict_json(path: Path) -> Any:
    return strict_bytes(path.read_bytes())


def regular(path: Path, expected: str, uid: int | None = EXPECTED_UID) -> None:
    value = path.lstat()
    require(stat.S_ISREG(value.st_mode) and not path.is_symlink() and
            sha256(path) == expected and (uid is None or value.st_uid == uid),
            "identity/owner drift: " + str(path))


def sealed_tree(directory: Path, manifest_sha: str,
                outer_file_sha: str) -> dict[str, str]:
    manifest = directory / MANIFEST
    outer = directory / OUTER
    regular(manifest, manifest_sha)
    regular(outer, outer_file_sha)
    require(outer.read_text(encoding="ascii").split() == [manifest_sha, MANIFEST],
            "outer content drift: " + str(directory))
    rows: dict[str, str] = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]),
                "manifest syntax drift")
        name = fields[1].lstrip("*")
        rel = Path(name)
        require(name not in rows and name == rel.as_posix() and
                not rel.is_absolute() and ".." not in rel.parts,
                "manifest member drift")
        rows[name] = fields[0]
    actual: set[str] = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {MANIFEST, OUTER}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "sealed symlink")
        if stat.S_ISREG(mode):
            actual.add(name)
        else:
            require(stat.S_ISDIR(mode), "sealed special member")
    require(actual == set(rows), "sealed exact member set drift")
    for name, expected in rows.items():
        regular(directory / name, expected)
    return rows


def double_file(path: Path, identity: tuple[str, str, str]) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    regular(path, identity[0]); regular(side, identity[1]); regular(outer, identity[2])
    require(side.read_text(encoding="ascii").split() == [identity[0], path.name],
            "side seal content drift")
    require(outer.read_text(encoding="ascii").split() == [identity[1], side.name],
            "double seal content drift")


def verify_chain_without_production_jsonl_open() -> dict[str, Any]:
    regular(SOURCE, SOURCE_SHA)
    regular(TESTS, TESTS_SHA)
    double_file(CONTRACT, CONTRACT_ID)
    author_rows = sealed_tree(AUTHOR, AUTHOR_ID[0], AUTHOR_ID[1])
    author = strict_json(AUTHOR / "review.json")
    require(author["identity"]["source_sha256"] == SOURCE_SHA and
            author["identity"]["tests_sha256"] == TESTS_SHA and
            author["identity"]["contract_sha256"] == CONTRACT_ID[0] and
            author["status"].startswith("PASS_M1199_ONE_SHOT"),
            "M1199 author admission drift")
    require(author_rows.get("review.json") == sha256(AUTHOR / "review.json"),
            "author review manifest drift")

    m1161_rows = sealed_tree(M1161, M1161_ID[0], M1161_ID[1])
    require(m1161_rows.get("producer_replay_terminal.json") ==
            "e681c65f25a42b7960b2a68f0709fff2b4c2bfe7d4ac7e69cccf689b9723add8" and
            m1161_rows.get("receipt.json") ==
            "2e6d5ae223f4057e66916ee46c483b523ec233d4a621a070e1438e50b559c751",
            "M1161 member identity drift")
    m1161 = strict_json(M1161 / "producer_replay_terminal.json")
    require(m1161["status"].startswith("PASS_REAL_M1137") and
            m1161["sealed_schedule"]["sha256"] == M1141_ID[3],
            "M1161 terminal drift")

    m1196_rows = sealed_tree(M1196, M1196_ID[1], M1196_ID[2])
    require(m1196_rows.get("review.json") == M1196_ID[0], "M1196 review identity")
    m1196 = strict_json(M1196 / "review.json")
    require(m1196["status"].startswith("PASS_M1196_M1161CA") and
            m1196["score"] == 99 and
            m1196["sealed_chain"]["result_outer_seal_file_sha256"] == M1161_ID[1] and
            m1196["production_evidence"]["rows_per_axis"] == 70_853_184,
            "M1196 admission drift")

    regular(M1169, M1169_SHA)
    regular(HW / "contracts/m1169_c1_ii2_service_aware_interval_replay_source_contract_r1_20260830.json",
            M1169_CONTRACT_SHA)
    m1170_rows = sealed_tree(M1170, M1170_ID[1], M1170_ID[2])
    require(m1170_rows.get("hammer_result.json") == M1170_ID[0],
            "M1170 result identity")
    m1170 = strict_json(M1170 / "hammer_result.json")
    require(m1170["status"].startswith("PASS_M1170_M1169") and
            m1170["identity"]["m1169_source_sha256"] == M1169_SHA and
            m1170["production_geometry_proof"]["beats_per_axis"] == 70_853_184,
            "M1170 admission drift")

    release = M1141 / "m1141ca_schedule_release.json"
    manifest = M1141 / MANIFEST
    outer = M1141 / OUTER
    regular(release, M1141_ID[0]); regular(manifest, M1141_ID[1]); regular(outer, M1141_ID[2])
    require(outer.read_text(encoding="ascii").split() == [M1141_ID[1], MANIFEST],
            "M1141 outer content drift")
    rows = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2, "M1141 manifest syntax")
        rows[fields[1].lstrip("*")] = fields[0]
    require(rows.get(PRODUCTION_JSONL.name) == M1141_ID[3] and
            rows.get(release.name) == M1141_ID[0], "M1141 identity rows drift")
    value = PRODUCTION_JSONL.lstat()
    require(stat.S_ISREG(value.st_mode) and not PRODUCTION_JSONL.is_symlink() and
            value.st_uid == EXPECTED_UID and value.st_size == M1141_ID[4],
            "M1141 JSONL metadata drift")
    release_json = strict_json(release)
    require(release_json["records"]["count"] == 2_436_480 and
            release_json["records"]["sha256"] == M1141_ID[3] and
            release_json["geometry"]["tasks"] == 812_160 and
            tuple(release_json["geometry"]["axes"]) == AXES,
            "M1141 release geometry drift")
    regular(DOCS359, DOCS359_SHA)
    return {
        "m1161_outer_seal_file_sha256": M1161_ID[1],
        "m1196_review_sha256": M1196_ID[0],
        "m1196_outer_seal_file_sha256": M1196_ID[2],
        "m1169_source_sha256": M1169_SHA,
        "m1170_outer_seal_file_sha256": M1170_ID[2],
        "m1141_records_sha256_pinned_without_open": M1141_ID[3],
    }


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "cannot load " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def independent_recurrence_checks(module: Any) -> dict[str, int]:
    rng = random.Random(0x1202_1199)
    scalar = 0
    for _ in range(20_000):
        requested = rng.randrange(0, 200)
        beats = rng.randrange(1, 80)
        previous = None if rng.randrange(4) == 0 else rng.randrange(0, 250)
        completed = []
        prior = previous
        for ordinal in range(beats):
            eligible = requested + ordinal
            accept = eligible if prior is None else max(eligible, prior + 1)
            prior = accept + 1
            completed.append(prior)
        got = module._M1169.advance_zero_stall_ii2(previous, requested, beats)
        independent_delay = sum(
            cycle - (requested + ordinal + 1)
            for ordinal, cycle in enumerate(completed))
        require((got.first_completed_cycle, got.last_completed_cycle,
                 got.aggregate_queue_delay_cycles) ==
                (completed[0], completed[-1], independent_delay),
                "independent II=2 scalar recurrence mismatch")
        scalar += 1

    quota_cases = 0
    for tasks in range(1, 80):
        for beats in range(tasks, tasks + 100):
            counts = [module._M1169.floor_quota(task, tasks, beats)[1] -
                      module._M1169.floor_quota(task, tasks, beats)[0]
                      for task in range(tasks)]
            require(sum(counts) == beats and min(counts) >= 1 and
                    max(counts) - min(counts) <= 1,
                    "floor quota conservation mismatch")
            quota_cases += 1
    return {"independent_scalar_cases": scalar, "floor_quota_geometries": quota_cases}


def stream_attacks(module: Any) -> dict[str, Any]:
    payload = module._bounded_payload(module._M1169)
    lines = payload.splitlines(keepends=True)
    require(len(lines) == 6, "fixture row drift")

    def execute(data: bytes, records: int = 6, byte_count: int | None = None,
                digest: str | None = None) -> None:
        replay = module._M1169.IntervalReplay(2, 7, AXES)
        module._parse_and_replay(
            io.BytesIO(data), records, len(data) if byte_count is None else byte_count,
            hashlib.sha256(data).hexdigest() if digest is None else digest, replay)

    execute(payload)
    mutations: dict[str, tuple[bytes, int, int | None, str | None]] = {}
    base = strict_bytes(lines[0][:-1])

    def replace_first(mapping: dict[str, Any]) -> bytes:
        new = (json.dumps(mapping, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
        return new + b"".join(lines[1:])

    extra = dict(base); extra["extra"] = 1
    missing = dict(base); missing.pop("chunk")
    boolean = dict(base); boolean["requested_cycle_first"] = False
    bad_provenance = dict(base); bad_provenance["schedule_record_provenance_sha256"] = "0" * 64
    wrong_task = dict(base); wrong_task["task_sequence_ordinal"] = 1
    mutations["extra_field"] = (replace_first(extra), 6, None, None)
    mutations["missing_field"] = (replace_first(missing), 6, None, None)
    mutations["boolean_integer"] = (replace_first(boolean), 6, None, None)
    mutations["bad_provenance"] = (replace_first(bad_provenance), 6, None, None)
    mutations["wrong_task"] = (replace_first(wrong_task), 6, None, None)
    duplicate = lines[0].replace(b'"axis":"candidate"',
                                 b'"axis":"candidate","axis":"candidate"', 1)
    mutations["duplicate_key"] = (duplicate + b"".join(lines[1:]), 6, None, None)
    nonfinite = lines[0].replace(b'"chunk":0', b'"chunk":NaN', 1)
    mutations["nonfinite"] = (nonfinite + b"".join(lines[1:]), 6, None, None)
    swapped = list(lines); swapped[0], swapped[1] = swapped[1], swapped[0]
    mutations["reorder"] = (b"".join(swapped), 6, None, None)
    mutations["drop"] = (b"".join(lines[:-1]), 6, None, None)
    mutations["duplicate_row"] = (payload + lines[-1], 6, None, None)
    mutations["crlf"] = (payload.replace(b"\n", b"\r\n", 1), 6, None, None)
    mutations["unterminated"] = (payload[:-1], 6, None, None)
    mutations["oversize"] = (lines[0][:-1] + b" " * 65_536 + b"\n" +
                              b"".join(lines[1:]), 6, None, None)
    mutations["record_count"] = (payload, 5, None, None)
    mutations["byte_count"] = (payload, 6, len(payload) + 1, None)
    mutations["terminal_sha"] = (payload, 6, None, "0" * 64)

    rejected = []
    for name, (data, records, byte_count, digest) in mutations.items():
        try:
            execute(data, records, byte_count, digest)
        except (module.Failure, module._M1169.Failure, Failure,
                json.JSONDecodeError, UnicodeDecodeError):
            rejected.append(name)
    require(set(rejected) == set(mutations), "bounded stream mutation escaped")
    return {"valid_fixture_records": 6, "attacks_rejected": len(rejected),
            "attack_names": sorted(rejected)}


def static_guards() -> dict[str, Any]:
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    production = next(node for node in tree.body
                      if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and
                      node.name == "production_main")
    attempt_lines = []
    schedule_open_lines = []
    for node in ast.walk(production):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (isinstance(node.func.value, ast.Name) and
                    node.func.value.id == "ATTEMPT" and node.func.attr == "mkdir"):
                attempt_lines.append(node.lineno)
            if (isinstance(node.func.value, ast.Name) and
                    node.func.value.id == "os" and node.func.attr == "open" and
                    node.args and isinstance(node.args[0], ast.Name) and
                    node.args[0].id == "M1141_RECORDS"):
                schedule_open_lines.append(node.lineno)
    require(len(attempt_lines) == 1 and len(schedule_open_lines) == 1 and
            attempt_lines[0] < schedule_open_lines[0],
            "attempt-before-schedule-open AST proof failed")
    required = (
        "M1199_SINGLE_ATTEMPT_CONSUMED__NO_AUTOMATIC_RETRY",
        "fcntl.LOCK_EX | fcntl.LOCK_NB", "MIN_COMMIT_HEADROOM",
        "same_uid_conflicts", "renameat2", "failure/result mutual exclusion drift",
        '"component_weight_service_schedule_only": True',
        '"rtl_cycles_or_system_speedup": False',
        '"traffic_energy_or_paper_ppa": False',
        '"retained_schedule_record_or_event_history": False',
    )
    require(all(token in text for token in required), "static one-shot/claim guard absent")
    main_start = text.index("def main()")
    require(text.index('require(len(sys.argv) == 1', main_start) <
            text.index("production_main()", main_start),
            "zero-argument guard order drift")
    return {"attempt_mkdir_line": attempt_lines[0],
            "production_schedule_os_open_line": schedule_open_lines[0],
            "attempt_precedes_schedule_open": True,
            "absolute_command_required_for_same_uid_token_guard": True,
            "persistent_attempt_and_no_retry_tokens": True,
            "resource_guard_tokens": True,
            "strict_claim_boundary_tokens": True}


@contextmanager
def rebound_fixture(module: Any, root: Path, payload: bytes) -> Iterator[None]:
    names = ("RESULTS", "RESULT", "ATTEMPT", "LOCK", "M1141_RECORDS",
             "M1141_RECORDS_BYTES", "M1141_RECORDS_SHA", "TASKS",
             "EVENTS_PER_AXIS", "EXPECTED_RECORDS", "source_preflight",
             "resource_preflight", "_parse_and_replay")
    saved = {name: getattr(module, name) for name in names}
    results = root / "results"; results.mkdir()
    schedule = root / "bounded_schedule.jsonl"; schedule.write_bytes(payload)
    module.RESULTS = results
    module.RESULT = results / "bounded_result"
    module.ATTEMPT = results / ".bounded_attempt"
    module.LOCK = root / "bounded.lock"
    module.M1141_RECORDS = schedule
    module.M1141_RECORDS_BYTES = len(payload)
    module.M1141_RECORDS_SHA = hashlib.sha256(payload).hexdigest()
    module.TASKS = 2
    module.EVENTS_PER_AXIS = 7
    module.EXPECTED_RECORDS = 6
    module.source_preflight = lambda require_fresh_namespace=True: {
        "status": "BOUNDED_REBOUND_PREFLIGHT", "production_schedule_opened": False}
    module.resource_preflight = lambda: {
        "cpus": 4, "mem_available_bytes": 1 << 40,
        "commit_headroom_bytes": 1 << 40, "disk_free_bytes": 1 << 40,
        "same_uid_conflicts": 0}
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(module, name, value)


def bounded_atomicity(module: Any) -> dict[str, Any]:
    payload = module._bounded_payload(module._M1169)
    with tempfile.TemporaryDirectory(prefix="m1202_success_") as temporary:
        root = Path(temporary)
        with rebound_fixture(module, root, payload):
            result = module.production_main()
            require(result["status"].startswith("PASS_M1199") and
                    module.RESULT.is_dir() and module.ATTEMPT.is_dir() and
                    not module.LOCK.exists() and
                    not tuple(module.RESULTS.glob(module.WORK_PREFIX + "*")) and
                    not tuple(module.RESULTS.glob(module.FAILURE_PREFIX + "*")),
                    "bounded success atomic publication drift")
            receipt = strict_json(module.RESULT / "receipt.json")
            require(receipt["attempt_consumed"] is True and
                    receipt["automatic_retry"] is False and
                    receipt["component_schedule_only"] is True and
                    receipt["rtl_or_system_speedup"] is False,
                    "bounded success claim drift")

    with tempfile.TemporaryDirectory(prefix="m1202_failure_") as temporary:
        root = Path(temporary)
        with rebound_fixture(module, root, payload):
            original_parse = module._parse_and_replay
            module._parse_and_replay = lambda *args, **kwargs: (_ for _ in ()).throw(
                module.Failure("M1202 injected bounded stream failure"))
            try:
                try:
                    module.production_main()
                except module.Failure as error:
                    require("M1202 injected" in str(error), "wrong injected failure")
                else:
                    raise Failure("bounded failure injection escaped")
                failures = tuple(module.RESULTS.glob(module.FAILURE_PREFIX + "*"))
                require(module.ATTEMPT.is_dir() and len(failures) == 1 and
                        not module.RESULT.exists() and not module.LOCK.exists() and
                        not tuple(module.RESULTS.glob(module.WORK_PREFIX + "*")),
                        "bounded failure quarantine/cleanup drift")
                failure = strict_json(failures[0] / "failure.json")
                require(failure["status"] == "FAILED_OR_INCOMPLETE_DO_NOT_CITE" and
                        failure["attempt_consumed"] is True and
                        failure["automatic_retry"] is False,
                        "bounded failure receipt drift")
            finally:
                module._parse_and_replay = original_parse
    return {"bounded_success_atomic_publish": True,
            "bounded_failure_quarantine": True,
            "attempt_persists_success_and_failure": True,
            "lock_and_work_cleanup": True,
            "result_failure_mutual_exclusion": True}


def run_unittests() -> dict[str, int]:
    module = load(TESTS, "m1202_loaded_m1199_tests")
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    require(result.wasSuccessful() and result.testsRun == 7,
            "M1199 source unittest failure:\n" + stream.getvalue())
    return {"tests_run": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors)}


def write_exclusive(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    finally:
        os.close(fd)


def write_json(path: Path, value: Any) -> None:
    write_exclusive(path, (json.dumps(value, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n").encode())


def main() -> int:
    require(os.getuid() == EXPECTED_UID, "hammer uid drift")
    production_abs = os.path.realpath(os.fspath(PRODUCTION_JSONL))
    production_open_attempts: list[str] = []

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event != "open" or not args:
            return
        path = args[0]
        if isinstance(path, (str, bytes, os.PathLike)):
            candidate = os.path.realpath(os.fsdecode(os.fspath(path)))
            if candidate == production_abs:
                production_open_attempts.append(candidate)
                raise Failure("M1202 forbids opening production M1141 JSONL")

    sys.addaudithook(audit)
    before_namespace = tuple(sorted(
        str(path) for path in (HW / "results").glob("*m1199*")))
    require(before_namespace == (), "M1199 namespace not fresh before hammer")

    chain = verify_chain_without_production_jsonl_open()
    static = static_guards()
    module = load(SOURCE, "m1202_loaded_m1199_source")
    preflight = module.source_preflight(True)
    require(preflight["production_schedule_opened"] is False and
            preflight["production_records_consumed"] == 0,
            "source preflight scope drift")
    unit = run_unittests()
    bounded = module.bounded_source_self_test()
    require(bounded["production_schedule_opened"] is False and
            bounded["production_namespace_mutated"] is False and
            bounded["attacks_rejected"] == 5,
            "bounded source oracle drift")
    recurrence = independent_recurrence_checks(module)
    attacks = stream_attacks(module)
    atomicity = bounded_atomicity(module)

    after_namespace = tuple(sorted(
        str(path) for path in (HW / "results").glob("*m1199*")))
    require(before_namespace == after_namespace == () and
            production_open_attempts == [],
            "production namespace/open boundary violated")
    regular(SOURCE, SOURCE_SHA); regular(TESTS, TESTS_SHA); double_file(CONTRACT, CONTRACT_ID)
    regular(DOCS359, DOCS359_SHA)

    exact_command = (
        "/opt/anaconda3/envs/pytorch310/bin/python3.10 "
        "/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/"
        "system_simulator/scripts/"
        "run_m1199_c1_ii2_service_aware_production_consumer_one_shot_source.py"
    )
    review = {
        "schema": "m1202_m1199_c1_ii2_production_consumer_source_hammer_r1_v1",
        "status": "PASS_M1202_M1199_SOURCE_HAMMER__AUTHORIZE_EXACTLY_ONE_ABSOLUTE_ZERO_ARGUMENT_PRODUCTION_LAUNCH",
        "date": "2026-08-30", "score": 100, "p0": [], "p1": [],
        "identity": {
            "source_sha256": SOURCE_SHA, "tests_sha256": TESTS_SHA,
            "contract_sha256": CONTRACT_ID[0],
            "contract_outer_seal_file_sha256": CONTRACT_ID[2],
            "author_manifest_sha256": AUTHOR_ID[0],
            "author_outer_seal_file_sha256": AUTHOR_ID[1],
            "docs359_sha256": DOCS359_SHA,
        },
        "sealed_admission_chain": chain,
        "static_one_shot_guards": static,
        "bounded_evidence": {
            "source_unittests": unit,
            "author_bounded_oracle": {
                "records": bounded["records"],
                "beats_per_axis": bounded["beats_per_axis"],
                "attacks_rejected": bounded["attacks_rejected"],
                "expanded_beats": bounded["terminal"]["expanded_beats"],
            },
            "independent_recurrence": recurrence,
            "independent_stream_attacks": attacks,
            "atomicity": atomicity,
        },
        "scope": {
            "production_schedule_open_attempts": len(production_open_attempts),
            "production_schedule_opened": False,
            "production_records_consumed": 0,
            "production_namespace_mutated": False,
            "production_execution": False,
            "gpu_vcs_dc_pt_formality_ptpx_remote": False,
            "docs359_modified": False,
        },
        "authorization": {
            "exactly_one_zero_argument_production_launch": True,
            "absolute_command_required": True,
            "exact_command": exact_command,
            "automatic_retry": False,
            "fresh_different_author_result_hammer_after_execution": True,
        },
        "claim_boundary": {
            "component_weight_service_schedule_only": True,
            "rtl_cycles_or_system_speedup": False,
            "traffic_energy_or_paper_ppa": False,
            "paper_citable_before_result_hammer": False,
        },
    }
    mechanical = {
        "status": "PASS_M1202_MECHANICAL_CHECKS",
        "source_hash": True, "test_hash": True, "contract_double_seal": True,
        "author_tree_exact": True, "dependency_chain_exact": True,
        "production_jsonl_open_attempts": 0, "production_namespace_unchanged": True,
        "docs359_sha256": DOCS359_SHA,
    }
    markdown = "# M1202 independent source hammer\n\n"
    markdown += "Verdict: **PASS (100/100)**.\n\n"
    markdown += (
        "The M1199 source pins the admitted M1161/M1196 and M1169/M1170 chains, "
        "checks M1141 identity by sealed metadata without opening its 836 MB JSONL, "
        "persists the one-shot attempt before the sole production schedule open, "
        "and emits only O(axes) terminal state. Independent bounded tests covered "
        f"{recurrence['independent_scalar_cases']} II=2 scalar cases, "
        f"{recurrence['floor_quota_geometries']} quota geometries, "
        f"{attacks['attacks_rejected']} stream attacks, and isolated atomic "
        "success/failure paths.\n\n"
        "Only the exact absolute zero-argument command in `review.json` is "
        "authorized, once, with no retry. A fresh different-author result hammer "
        "is mandatory. Outputs remain component weight-service schedule metrics, "
        "not RTL cycles, system speedup, traffic, energy, or paper PPA.\n"
    )
    write_json(HERE / "review.json", review)
    write_json(HERE / "mechanical_checks.json", mechanical)
    write_json(HERE / "bounded_adversarial_checks.json", {
        "recurrence": recurrence, "stream_attacks": attacks,
        "atomicity": atomicity, "production_schedule_opened": False,
        "production_records_consumed": 0})
    write_exclusive(HERE / "review.md", markdown.encode())
    write_exclusive(HERE / "AUTHORIZE_EXACTLY_ONE_ABSOLUTE_ZERO_ARGUMENT_PRODUCTION_LAUNCH_NO_RETRY_RESULT_HAMMER_REQUIRED.txt",
                    (exact_command + "\n").encode())
    write_exclusive(HERE / "BOUNDED_READONLY_NO_M1141_JSONL_OPEN_NO_PRODUCTION_NO_EDA.txt",
                    b"M1202 opened zero production schedule records and launched no production/GPU/VCS/EDA.\n")
    write_exclusive(HERE / "RUN_COMPLETE.txt",
                    b"PASS_M1202_M1199_SOURCE_HAMMER__ONE_ABSOLUTE_ZERO_ARGUMENT_LAUNCH_AUTHORIZED\n")

    members = sorted(path for path in HERE.iterdir()
                     if path.is_file() and path.name not in {MANIFEST, OUTER})
    lines = [f"{sha256(path)}  {path.name}" for path in members]
    write_exclusive(HERE / MANIFEST, ("\n".join(lines) + "\n").encode())
    manifest_sha = sha256(HERE / MANIFEST)
    write_exclusive(HERE / OUTER, f"{manifest_sha}  {MANIFEST}\n".encode())
    print(json.dumps({"status": review["status"], "score": 100,
                      "review_sha256": sha256(HERE / "review.json"),
                      "manifest_sha256": manifest_sha,
                      "outer_seal_file_sha256": sha256(HERE / OUTER),
                      "exact_command": exact_command},
                     sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
