#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1145CA independent O(1)-state hammer for M1143CA/M1141CA results.

This is read-only over sealed production results.  It does not open M410,
compile digests, run the full replay, or invoke EDA.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import stat
import struct
import sys
import tempfile
from typing import Any, BinaryIO, Callable

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RESULTS = HW / "results"
LAUNCH = RESULTS / "m1143ca_c1_production_schedule_one_shot_launch_r1_20260830"
CHILD = RESULTS / "m1141ca_c1_production_schedule_release_r1_20260830"
ATTEMPT = RESULTS / ".m1143ca_c1_production_schedule_one_shot_attempt_consumed"
RECORDS_NAME = "m1141ca_per_task_schedule_records.jsonl"
RELEASE_NAME = "m1141ca_schedule_release.json"
RECORDS = CHILD / RECORDS_NAME
RELEASE = CHILD / RELEASE_NAME
LAUNCH_RECEIPT = LAUNCH / "receipt.json"
ATTEMPT_JSON = ATTEMPT / "attempt.json"
M1143_SOURCE = HW / "system_simulator/scripts/run_m1143ca_c1_production_schedule_one_shot_launcher_source.py"
M1141_SOURCE = HW / "system_simulator/scripts/run_m1141ca_c1_production_schedule_release_source.py"
M1139_SOURCE = HW / "system_simulator/scripts/build_m1139ca_c1_independent_per_task_schedule_authority_source.py"
M1140 = HW / "reviews/m1140ca_m1139ca_c1_independent_per_task_schedule_authority_hammer_r1_20260830"
M1144 = HW / "reviews/m1144ca_m1143ca_c1_final_launcher_hammer_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    "launch_outer": "759d5de768ee606d7080c6788f1e85ad3c3ae0815461d611eac806799dd1f05d",
    "launch_manifest": "fdf94a118ff880d4af036da983e123297be7aa0af12bfcba1e04b01596be6058",
    "child_outer": "0b6549ce38a62bcb22e8a97d0c038860f5698fabc0d9bff162dc6af95d4f043a",
    "child_manifest": "852b48c0d8098ef69a619925f82a8e1a308e87f2faf9ea76becabf51d52caace",
    "attempt_outer": "93a31db726b3c4b8a4e8cc65d30ae47b2084f378b82885c193eee59203e8722d",
    "attempt_manifest": "0010996d163c3f2f3fa7383396349aadd070c45b92a2731814a2be2f633e948d",
    "records": "4d4e0e6396ac1061aca7ada142bc2761bf12a785e5373640a28503e3d73a0a81",
    "release": "4c4d264a9ac1e084c8c0acf0a6d150140f95ee96ee967b038ea4c1eefcc2b58c",
    "launch_receipt": "e57b4b7c06c190cbf1a39428538b776fa842ad984b9d59a7d68806cc382d25bb",
    "m1143_source": "184528ee978f3260e7e52d1048d96ecd99a3488d516c85ee3dbc0bcdd2d56be7",
    "m1141_source": "e2f5d4e0bab472b3a5c7ec5259a805641b800efd3c0e82884e81152eb41cb611",
    "m1139_source": "d18137661517538a8273b696b5f2ada09aff9847c16da0d3a723037e901153a9",
    "m1140_outer": "f73cafa73ed047abd59730749bf48fcb3f463fca77609aec6085f5b3389fa352",
    "m1144_outer": "f3940f001ef05434513b469a33a60087dda524615e659b130aef69b52d04013c",
    "m410": "6e03352b89eff1955825334b4dedd991db8c975a9ef6662fe0317e73ccfa8334",
    "m1016": "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "schedule": "d8289ede1ec668cd86b9ea2c561c76f62738cbd5aa361d9c21642f900e3fa1b9",
}
AXES = ("candidate", "strongest_zero", "same_coordinate_bit")
KEYS = frozenset(("axis", "chunk", "operator", "partition",
                  "requested_cycle_first", "sample",
                  "schedule_record_provenance_sha256",
                  "source_task_provenance_sha256", "task_sequence_ordinal"))
SAMPLES = 10
OPERATORS = 4
CHUNKS = 47
PARTITIONS = 432
TASKS = 812_160
RECORD_COUNT = 2_436_480
EXPECTED_FIRST = (
    {"axis": "candidate", "chunk": 0, "operator": 0, "partition": 0,
     "requested_cycle_first": 0, "sample": 0,
     "schedule_record_provenance_sha256": "9ddf9284c28ce70dfb0ebd562e815e9ee5ed8f4d2780798d22b3f5e547ee4f2e",
     "source_task_provenance_sha256": "dbc91b4928ce32aca93c83a535468347863b9e5eac73ee2799a72524b628f289",
     "task_sequence_ordinal": 0},
    {"axis": "strongest_zero", "chunk": 0, "operator": 0, "partition": 0,
     "requested_cycle_first": 0, "sample": 0,
     "schedule_record_provenance_sha256": "bf8851d97946942bfdacdc3c0eb2b67f61f37aa777e14e1b4cac6fe3971d42dc",
     "source_task_provenance_sha256": "dbc91b4928ce32aca93c83a535468347863b9e5eac73ee2799a72524b628f289",
     "task_sequence_ordinal": 0},
    {"axis": "same_coordinate_bit", "chunk": 0, "operator": 0, "partition": 0,
     "requested_cycle_first": 0, "sample": 0,
     "schedule_record_provenance_sha256": "84127b7d64f053218e07638a0341ee2acd9adca83f25e9fcffb2afa76f253d28",
     "source_task_provenance_sha256": "dbc91b4928ce32aca93c83a535468347863b9e5eac73ee2799a72524b628f289",
     "task_sequence_ordinal": 0},
)
EXPECTED_LAST = (
    {"axis": "candidate", "chunk": 46, "operator": 3, "partition": 431,
     "requested_cycle_first": 434146693, "sample": 9,
     "schedule_record_provenance_sha256": "d7e70ed7aa79cdb1a4e1ebdcd17cbcd3bd9882a7ae2806026331ceb55ba2e02f",
     "source_task_provenance_sha256": "d4980d3a47400d02462805b9bd279be198675ead03927e5bf59dddb3673fd30c",
     "task_sequence_ordinal": 812159},
    {"axis": "strongest_zero", "chunk": 46, "operator": 3, "partition": 431,
     "requested_cycle_first": 752971230, "sample": 9,
     "schedule_record_provenance_sha256": "310d87085ef92b60966a5c121aa977daacc7996541dafb9cc5698ec695fde023",
     "source_task_provenance_sha256": "d4980d3a47400d02462805b9bd279be198675ead03927e5bf59dddb3673fd30c",
     "task_sequence_ordinal": 812159},
    {"axis": "same_coordinate_bit", "chunk": 46, "operator": 3, "partition": 431,
     "requested_cycle_first": 752971230, "sample": 9,
     "schedule_record_provenance_sha256": "831e2a0dc7776e334d7366e2be3ac4878a422fdee60ef1d1aa4acfdabf8f6fc1",
     "source_task_provenance_sha256": "d4980d3a47400d02462805b9bd279be198675ead03927e5bf59dddb3673fd30c",
     "task_sequence_ordinal": 812159},
)
checks = 0
attacks: dict[str, str] = {}


class Failure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise Failure(message)


def rejected(label: str, action: Callable[[], Any], contains: str | None = None) -> None:
    try:
        action()
    except BaseException as error:
        if contains is not None:
            require(contains in str(error), label + " wrong rejection: " + str(error))
        attacks[label] = type(error).__name__ + ": " + str(error)
        return
    raise Failure("attack accepted: " + label)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, expected: str, owner: int | None = 1913) -> None:
    value = path.lstat()
    require(stat.S_ISREG(value.st_mode) and not path.is_symlink() and
            (owner is None or value.st_uid == owner) and sha(path) == expected,
            "identity/owner drift: " + str(path))


def strict_loads(payload: str) -> Any:
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    return json.loads(payload, object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("nonfinite JSON: " + token)))


def strict_json(path: Path) -> Any:
    return strict_loads(path.read_text(encoding="utf-8"))


def inspect_tree(directory: Path, expected_outer: str, expected_manifest: str,
                 exact_members: set[str]) -> dict[str, str]:
    require(directory.is_dir() and not directory.is_symlink() and
            directory.stat().st_uid == 1913, "result tree owner/type drift")
    manifest = directory / "SHA256SUMS"; outer = directory / "SHA256SUMS.seal.sha256"
    regular(manifest, expected_manifest); regular(outer, expected_outer)
    require(outer.read_text(encoding="utf-8").split() ==
            [expected_manifest, "SHA256SUMS"], "outer seal content drift")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*"); rel = Path(name)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None and
                name not in expected and name == rel.as_posix() and
                not rel.is_absolute() and ".." not in rel.parts,
                "manifest member unsafe/duplicate")
        expected[name] = digest
    actual = set()
    for member in directory.iterdir():
        if member.name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}: continue
        mode = member.lstat().st_mode
        require(stat.S_ISREG(mode) and not member.is_symlink() and member.stat().st_uid == 1913,
                "nonregular/symlink/owner member")
        actual.add(member.name)
    require(actual == exact_members == set(expected), "exact tree member census drift")
    return expected


def verify_small_members(directory: Path, manifest: dict[str, str],
                         skip: set[str] = frozenset()) -> None:
    for name, digest in manifest.items():
        if name not in skip: regular(directory / name, digest)


def u64(value: int) -> bytes:
    require(type(value) is int and 0 <= value < (1 << 64), "u64 value drift")
    return struct.pack(">Q", value)


def record_provenance(value: dict[str, Any]) -> str:
    payload = b"".join((
        b"M1139CA_SCHEDULE_RECORD\x00\x01", bytes.fromhex(EXPECTED["m1016"]),
        bytes.fromhex("95bd50aebcc473ab69cdea6ccf27d54743c89926c5e0f31199dc469ced9bf7cc"),
        bytes.fromhex("9ec640ae8c9fa75f9cbf706e15d2d26a4233def77e5be4d67e94c084347b20a6"),
        struct.pack(">B", AXES.index(value["axis"])),
        u64(value["task_sequence_ordinal"]), u64(value["sample"]),
        u64(value["operator"]), u64(value["chunk"]), u64(value["partition"]),
        u64(value["requested_cycle_first"]),
        bytes.fromhex(value["source_task_provenance_sha256"]),
    ))
    return hashlib.sha256(payload).hexdigest()


def coords(task: int, samples: int, operators: int,
           chunks: int, partitions: int) -> tuple[int, int, int, int]:
    partition = task % partitions; quotient = task // partitions
    chunk = quotient % chunks; quotient //= chunks
    operator = quotient % operators; sample = quotient // operators
    require(0 <= sample < samples, "task coordinate overflow")
    return sample, operator, chunk, partition


def scan_stream(stream: BinaryIO, *, samples: int, operators: int, chunks: int,
                partitions: int, tasks: int, expected_records: int,
                expected_file_sha: str | None = None,
                expected_schedule_sha: str | None = None,
                expected_first: tuple[dict[str, Any], ...] | None = None,
                expected_last: tuple[dict[str, Any], ...] | None = None) -> dict[str, Any]:
    file_digest = hashlib.sha256(); schedule_digest = hashlib.sha256()
    axis_counts = {axis: 0 for axis in AXES}
    last_requested: dict[str, int | None] = {axis: None for axis in AXES}
    first: list[dict[str, Any]] = []; last: list[dict[str, Any]] = []
    current_task = -1; current_source: str | None = None; count = 0
    for raw in stream:
        file_digest.update(raw)
        require(raw.endswith(b"\n") and b"\x00" not in raw, "record framing drift")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise Failure("record UTF-8 drift") from error
        value = strict_loads(text)
        require(type(value) is dict and set(value) == KEYS, "record exact key set drift")
        expected_task = count // len(AXES); expected_axis = AXES[count % len(AXES)]
        require(expected_task < tasks, "extra record")
        expected_coords = coords(expected_task, samples, operators, chunks, partitions)
        require(value["axis"] == expected_axis and
                type(value["task_sequence_ordinal"]) is int and
                value["task_sequence_ordinal"] == expected_task and
                all(type(value[name]) is int for name in
                    ("sample", "operator", "chunk", "partition", "requested_cycle_first")) and
                tuple(value[name] for name in ("sample", "operator", "chunk", "partition")) ==
                    expected_coords and value["requested_cycle_first"] >= 0,
                "missing, duplicate, reordered, or coordinate drift")
        require(type(value["source_task_provenance_sha256"]) is str and
                type(value["schedule_record_provenance_sha256"]) is str and
                re.fullmatch(r"[0-9a-f]{64}", value["source_task_provenance_sha256"]) is not None and
                re.fullmatch(r"[0-9a-f]{64}", value["schedule_record_provenance_sha256"]) is not None,
                "provenance encoding drift")
        if expected_task != current_task:
            current_task = expected_task; current_source = value["source_task_provenance_sha256"]
        require(value["source_task_provenance_sha256"] == current_source,
                "source-task provenance differs within task")
        axis = value["axis"]; requested = value["requested_cycle_first"]
        require(last_requested[axis] is None or requested >= last_requested[axis],
                "requested cycle regressed")
        last_requested[axis] = requested
        computed = record_provenance(value)
        require(computed == value["schedule_record_provenance_sha256"],
                "schedule-record provenance mismatch")
        canonical = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                                allow_nan=False) + "\n").encode()
        require(canonical == raw, "record noncanonical encoding")
        schedule_digest.update(bytes.fromhex(computed)); axis_counts[axis] += 1
        if count < 3: first.append(value)
        if len(last) == 3: last.pop(0)
        last.append(value); count += 1
    require(count == expected_records == tasks * len(AXES),
            "terminal missing/duplicate/count drift")
    result = {"records": count, "tasks": tasks, "axis_counts": axis_counts,
              "last_requested_cycle_by_axis": last_requested,
              "records_sha256": file_digest.hexdigest(),
              "schedule_provenance_sha256": schedule_digest.hexdigest(),
              "first_records": first, "last_records": last,
              "retained_history": False,
              "state_complexity": "O(axes) plus first/last three records"}
    if expected_file_sha is not None:
        require(result["records_sha256"] == expected_file_sha, "records SHA mismatch")
    if expected_schedule_sha is not None:
        require(result["schedule_provenance_sha256"] == expected_schedule_sha,
                "schedule provenance aggregate mismatch")
    if expected_first is not None: require(tuple(first) == expected_first, "first records drift")
    if expected_last is not None: require(tuple(last) == expected_last, "last records drift")
    return result


def encode_fixture(records: list[dict[str, Any]]) -> bytes:
    for value in records:
        value["schedule_record_provenance_sha256"] = record_provenance(value)
    return b"".join((json.dumps(value, sort_keys=True, separators=(",", ":"),
                                allow_nan=False) + "\n").encode() for value in records)


def fixture_records() -> list[dict[str, Any]]:
    values = []
    for task in range(2):
        source = hashlib.sha256(("fixture-%d" % task).encode()).hexdigest()
        for index, axis in enumerate(AXES):
            values.append({"axis": axis, "chunk": 0, "operator": 0,
                           "partition": task, "requested_cycle_first": task * 10 + index,
                           "sample": 0, "schedule_record_provenance_sha256": "0" * 64,
                           "source_task_provenance_sha256": source,
                           "task_sequence_ordinal": task})
    return values


def scan_fixture(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return scan_stream(stream, samples=1, operators=1, chunks=1,
                           partitions=2, tasks=2, expected_records=6)


def controlled_attacks(root: Path) -> dict[str, Any]:
    base = fixture_records(); good = root / "good.jsonl"; good.write_bytes(encode_fixture(base))
    positive = scan_fixture(good)
    require(positive["records"] == 6 and positive["tasks"] == 2, "fixture positive")
    variants: dict[str, bytes] = {}
    variants["missing_record"] = encode_fixture(fixture_records()[:-1])
    duplicate = fixture_records(); duplicate.insert(2, dict(duplicate[1]))
    variants["duplicate_record"] = encode_fixture(duplicate)
    reordered = fixture_records(); reordered[0], reordered[1] = reordered[1], reordered[0]
    variants["reordered_record"] = encode_fixture(reordered)
    extra = fixture_records(); extra[0]["extra_attack"] = 1
    variants["extra_key"] = encode_fixture(extra)
    corrupt = fixture_records(); payload = encode_fixture(corrupt).replace(
        corrupt[0]["schedule_record_provenance_sha256"].encode(), b"0" * 64, 1)
    variants["bad_provenance"] = payload
    regression = fixture_records(); regression[0]["requested_cycle_first"] = 10
    regression[3]["requested_cycle_first"] = 0
    variants["requested_cycle_regression"] = encode_fixture(regression)
    variants["nonfinite"] = encode_fixture(fixture_records()).replace(
        b'"requested_cycle_first":0', b'"requested_cycle_first":NaN', 1)
    for label, payload in variants.items():
        target = root / (label + ".jsonl"); target.write_bytes(payload)
        rejected(label, lambda target=target: scan_fixture(target))
    duplicate_json = (b'{"axis":"candidate","axis":"candidate","chunk":0,'
                      b'"operator":0,"partition":0,"requested_cycle_first":0,'
                      b'"sample":0,"schedule_record_provenance_sha256":"' + b"0" * 64 +
                      b'","source_task_provenance_sha256":"' + b"0" * 64 +
                      b'","task_sequence_ordinal":0}\n')
    target = root / "duplicate_key.jsonl"; target.write_bytes(duplicate_json)
    rejected("duplicate_json_key", lambda: scan_fixture(target), "duplicate JSON key")
    return {"positive_records": 6, "attacks": 8,
            "missing_duplicate_reorder_nonfinite_extra_key_detected": True}


def verify_namespace() -> dict[str, Any]:
    require(sum(path == ATTEMPT for path in RESULTS.glob(
                ".m1143ca_c1_production_schedule_one_shot_attempt_consumed")) == 1,
            "attempt not exactly one")
    launcher_failure = tuple(RESULTS.glob(
        "m1143ca_c1_production_schedule_one_shot_launch_r1_20260830.failed_or_incomplete.*"))
    launcher_work = tuple(RESULTS.glob(".m1143ca_c1_production_schedule_one_shot_work.*"))
    child_failure = tuple(RESULTS.glob(
        "m1141ca_c1_production_schedule_release_r1_20260830.failed_or_incomplete.*"))
    child_work = tuple(RESULTS.glob(".m1141ca_c1_production_schedule_release_work.*"))
    require(launcher_failure == launcher_work == child_failure == child_work == (),
            "production result/failure/work mutual exclusion drift")
    require(not Path("/tmp/m1143ca_c1_production_schedule_one_shot.lock").exists(),
            "launcher lock remains")
    return {"attempt_directories": 1, "launcher_failures": 0, "launcher_works": 0,
            "child_failures": 0, "child_works": 0, "launcher_locks": 0}


def verify_semantics(release: dict[str, Any], receipt: dict[str, Any],
                     attempt: dict[str, Any], scan: dict[str, Any]) -> None:
    require(set(release) == {"authority", "claim_boundary", "geometry", "records",
                            "retained_record_or_key_history", "schema", "source_rows",
                            "state_complexity", "status", "terminal"},
            "release exact top-level keys")
    require(release["schema"] == "m1141ca_c1_production_schedule_release_r1_v1" and
            release["status"] == "PASS_EXACT_PRODUCTION_SCHEDULE_RELEASE__DIFFERENT_AUTHOR_RESULT_HAMMER_REQUIRED" and
            release["geometry"] == {"axes": list(AXES), "chunks": CHUNKS,
                "operators": OPERATORS, "partitions": PARTITIONS,
                "records": RECORD_COUNT, "samples": SAMPLES, "tasks": TASKS} and
            release["records"]["axis_order_within_each_task"] == list(AXES) and
            release["records"]["axis_counts"] == scan["axis_counts"] and
            release["records"]["count"] == scan["records"] and
            release["records"]["sha256"] == scan["records_sha256"] and
            release["records"]["schedule_provenance_sha256"] ==
                scan["schedule_provenance_sha256"] and
            release["terminal"]["tasks"] == scan["tasks"] and
            release["terminal"]["records_by_axis"] == scan["axis_counts"] and
            release["terminal"]["last_requested_cycle_by_axis"] ==
                scan["last_requested_cycle_by_axis"], "release/stream semantic drift")
    require(release["source_rows"] == {"bytes": 466560000,
                "identity_reverified_after_stream": True, "no_follow_single_fd": True,
                "sha256": EXPECTED["m410"]} and
            release["authority"] == {"m1016_source_sha256": EXPECTED["m1016"],
                "m1139ca_source_sha256": EXPECTED["m1139_source"],
                "m1140ca_outer_seal_file_sha256": EXPECTED["m1140_outer"]},
            "raw M410 or schedule authority binding drift")
    require(release["claim_boundary"] == {"digest_compiler": False, "eda": False,
                "full_replay": False, "paper_citable": False, "real_driver": False,
                "traffic_cycles_energy_speedup": False} and
            release["retained_record_or_key_history"] is False,
            "release claim/state boundary drift")
    require(receipt["schema"] == "m1143ca_c1_production_schedule_one_shot_launch_receipt_r1_v1" and
            receipt["status"] == "PASS_M1143CA_ONE_SHOT_CHILD_COMPLETE__RESULT_HAMMER_REQUIRED" and
            receipt["source_sha256"] == EXPECTED["m1143_source"] and
            receipt["attempt_consumed"] is True and receipt["automatic_retry"] is False and
            receipt["child"]["source_sha256"] == EXPECTED["m1141_source"] and
            receipt["child"]["processes"] == 1 and receipt["child"]["arguments"] == 0 and
            receipt["child"]["returncode"] == 0 and
            receipt["child"]["result"] == {"manifest_sha256": EXPECTED["child_manifest"],
                "outer_seal_file_sha256": EXPECTED["child_outer"],
                "records": RECORD_COUNT, "records_sha256": EXPECTED["records"]} and
            receipt["claim_boundary"] == {"paper_citable": False,
                "result_hammer_required": True, "traffic_cycles_energy_speedup": False},
            "launcher receipt semantic drift")
    require(attempt == {"automatic_retry": False, "expected_child_processes": 1,
                "m1141ca_author_outer_seal_file_sha256": "b5602b120cc7c02769a54e67c78588c481776af9f40f3d3359a2938bf2f8b825",
                "m1141ca_contract_outer_seal_file_sha256": "6e5561e52fab6b4ae3018f8995f4b71f4c8eaeaf02c83ea192421081b5af8184",
                "m1141ca_source_sha256": EXPECTED["m1141_source"],
                "m1142ca_outer_seal_file_sha256": "7a8f8da04bb81a0097d819f98a3bed6e9e40b86a32aef055134f3306bb1850e8",
                "schema": "m1143ca_c1_production_schedule_one_shot_attempt_r1_v1",
                "status": "M1143CA_SINGLE_ATTEMPT_CONSUMED__NO_AUTO_RETRY"},
            "exactly-one attempt semantic drift")


def main() -> None:
    require(len(sys.argv) == 1, "M1145CA accepts zero arguments")
    docs_before = sha(DOCS359); namespace = verify_namespace()
    regular(M1143_SOURCE, EXPECTED["m1143_source"])
    regular(M1141_SOURCE, EXPECTED["m1141_source"])
    regular(M1139_SOURCE, EXPECTED["m1139_source"])
    regular(M1140 / "SHA256SUMS.seal.sha256", EXPECTED["m1140_outer"])
    regular(M1144 / "SHA256SUMS.seal.sha256", EXPECTED["m1144_outer"])
    regular(DOCS359, EXPECTED["docs359"])
    launch_manifest = inspect_tree(LAUNCH, EXPECTED["launch_outer"],
        EXPECTED["launch_manifest"], {"RUN_COMPLETE.txt", "child.stderr.log",
        "child.stdout.log", "receipt.json"})
    child_manifest = inspect_tree(CHILD, EXPECTED["child_outer"],
        EXPECTED["child_manifest"], {RECORDS_NAME, RELEASE_NAME})
    attempt_manifest = inspect_tree(ATTEMPT, EXPECTED["attempt_outer"],
        EXPECTED["attempt_manifest"], {"attempt.json"})
    verify_small_members(LAUNCH, launch_manifest)
    verify_small_members(CHILD, child_manifest, {RECORDS_NAME})
    verify_small_members(ATTEMPT, attempt_manifest)
    require(child_manifest[RECORDS_NAME] == EXPECTED["records"] and
            child_manifest[RELEASE_NAME] == EXPECTED["release"] and
            launch_manifest["receipt.json"] == EXPECTED["launch_receipt"],
            "sealed manifest expected result identities drift")
    release = strict_json(RELEASE); receipt = strict_json(LAUNCH_RECEIPT)
    attempt = strict_json(ATTEMPT_JSON)
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with RECORDS.open("rb", buffering=1 << 20) as stream:
        scan = scan_stream(stream, samples=SAMPLES, operators=OPERATORS,
            chunks=CHUNKS, partitions=PARTITIONS, tasks=TASKS,
            expected_records=RECORD_COUNT, expected_file_sha=EXPECTED["records"],
            expected_schedule_sha=EXPECTED["schedule"],
            expected_first=EXPECTED_FIRST, expected_last=EXPECTED_LAST)
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    verify_semantics(release, receipt, attempt, scan)
    with tempfile.TemporaryDirectory(prefix="m1145ca_controlled.", dir="/tmp") as name:
        controlled = controlled_attacks(Path(name))
    require(sha(DOCS359) == docs_before == EXPECTED["docs359"], "docs359 changed")
    result = {
        "schema": "m1145ca_m1143ca_c1_production_result_hammer_mechanical_r1_v1",
        "status": "PASS_M1145CA_INDEPENDENT_PRODUCTION_RESULT_HAMMER__DIGEST_COMPILER_SOURCE_AUTHORING_ONLY_NEXT",
        "checks": checks, "attacks_rejected": attacks,
        "sealed_results": {"launcher_outer_seal_file_sha256": EXPECTED["launch_outer"],
                           "child_outer_seal_file_sha256": EXPECTED["child_outer"],
                           "attempt_outer_seal_file_sha256": EXPECTED["attempt_outer"]},
        "namespace": namespace, "stream": scan,
        "memory": {"algorithmic_state": "O(axes) plus first/last three records",
                   "retained_record_or_key_history": False,
                   "inherited_ru_maxrss_before_kib": rss_before,
                   "ru_maxrss_after_kib": rss_after,
                   "ru_maxrss_increment_kib": max(0, rss_after - rss_before)},
        "controlled": controlled,
        "raw_authority": {"m410_sha256": EXPECTED["m410"],
                          "m410_bytes": 466560000,
                          "bound_by_frozen_m1141_source_and_sealed_release": True,
                          "m410_reopened_by_hammer": False},
        "authorization": {"digest_compiler_source_authoring_only_next": True,
                          "digest_compiler_execution": False,
                          "full_replay": False, "eda": False},
        "claim_boundary": {"schedule_release_citable_as_input_authority": True,
                           "traffic_cycles_energy_speedup": False,
                           "paper_performance_citable": False,
                           "paper_ppa_ready": False},
        "docs359_sha256": EXPECTED["docs359"],
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
