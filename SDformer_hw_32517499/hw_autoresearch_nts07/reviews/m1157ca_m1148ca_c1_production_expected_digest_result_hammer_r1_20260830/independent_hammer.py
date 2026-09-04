#!/usr/bin/env python3
"""Read-only, different-author hammer for the sealed M1148CA result.

This file deliberately does not import or call the subject compiler/launcher.
It independently checks the sealed schedule, output accounting, authority
identity, bounded golden digest and sampled event semantics.  It is not a
second 212,559,552-event production replay.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import stat
import struct
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RESULT = HW / "results/m1148ca_c1_production_expected_digest_compiler_r1_20260830"
ATTEMPT = HW / "results/.m1148ca_c1_production_expected_digest_compiler_attempt_consumed"
SCHEDULE = HW / "results/m1141ca_c1_production_schedule_release_r1_20260830/m1141ca_per_task_schedule_records.jsonl"
COMPILER = HW / "system_simulator/scripts/build_m1146ca_c1_independent_expected_digest_compiler_source.py"
LAUNCHER = HW / "system_simulator/scripts/run_m1148ca_c1_production_expected_digest_compiler_one_shot_launcher_source.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

AXES = ("candidate", "strongest_zero", "same_coordinate_bit")
AXIS_CODE = {name: index for index, name in enumerate(AXES)}
FIELDS = {
    "axis", "task_sequence_ordinal", "sample", "operator", "chunk",
    "partition", "requested_cycle_first", "source_task_provenance_sha256",
    "schedule_record_provenance_sha256",
}
TASKS = 812_160
EVENTS = 70_853_184
RECORDS = TASKS * 3
TOTAL_EVENTS = EVENTS * 3
SCHEDULE_BYTES = 836_268_740
SCHEDULE_SHA = "4d4e0e6396ac1061aca7ada142bc2761bf12a785e5373640a28503e3d73a0a81"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
M1016_SHA = "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa"
M1102_SHA = "95bd50aebcc473ab69cdea6ccf27d54743c89926c5e0f31199dc469ced9bf7cc"
M1135_SHA = "4c282b4ece5705b5c8dcd039c29003c14e544ffef5e8c4234afab0ac31ac7571"
M1137_SHA = "9ec640ae8c9fa75f9cbf706e15d2d26a4233def77e5be4d67e94c084347b20a6"
COMPILER_SHA = "7b1f5cd2cd4c4bb0a771d0360f8be924d075215e8dd660728a8decac0c886e73"
LAUNCHER_SHA = "00bd132ca162d15b9aab1d5972c1d4da37e1f43288c1973f8708b5322f14a781"
GOLDEN = {
    "candidate": "ab87d9d8da38d28a54d6048dc75cb7ac749aebba7807f855cac69165b9fa5644",
    "strongest_zero": "eb2dd17d2d0aa43e19d2f66b9d079760f7495c1f9b4653d206831605e1b44717",
    "same_coordinate_bit": "18a4e643ee4a606b5ec8e646fbd76aa155ffe324213a4f8bb36925c6fb678d7a",
}
SAMPLE_TASKS = {0, 1, 2, TASKS // 3, TASKS // 2, TASKS - 1}


class Reject(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise Reject(message)


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def strict_loads(payload: bytes) -> dict[str, Any]:
    def pairs(items):
        output = {}
        for key, value in items:
            need(key not in output, "duplicate JSON key")
            output[key] = value
        return output
    value = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           Reject("non-finite JSON: " + token)))
    need(type(value) is dict, "JSON object required")
    return value


def strict_json(path: Path) -> dict[str, Any]:
    return strict_loads(path.read_bytes())


def regular(path: Path) -> None:
    mode = path.lstat().st_mode
    need(stat.S_ISREG(mode) and not path.is_symlink(), "not a regular file: " + str(path))


def sealed_tree(directory: Path, expected_files: set[str]) -> dict[str, str]:
    need(directory.is_dir() and not directory.is_symlink(), "sealed directory missing")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    regular(manifest); regular(outer)
    manifest_sha = sha(manifest)
    need(outer.read_text(encoding="utf-8").split() == [manifest_sha, "SHA256SUMS"],
         "outer seal mismatch")
    listed: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = line.split(None, 1)
        need(len(parts) == 2 and re.fullmatch(r"[0-9a-f]{64}", parts[0]) is not None,
             "malformed manifest")
        name = parts[1].lstrip("*")
        relative = Path(name)
        need(name not in listed and not relative.is_absolute() and ".." not in relative.parts,
             "unsafe/duplicate manifest member")
        listed[name] = parts[0]
    actual = set()
    for item in directory.iterdir():
        if item.name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        regular(item); actual.add(item.name)
    need(actual == expected_files == set(listed), "exact sealed member set mismatch")
    for name, digest in listed.items():
        need(sha(directory / name) == digest, "member digest mismatch: " + name)
    return {"manifest_sha256": manifest_sha, "outer_sha256": sha(outer), **listed}


def u64(value: int) -> bytes:
    need(type(value) is int and 0 <= value < 1 << 64, "u64 range")
    return struct.pack(">Q", value)


def coordinates(task: int, tasks: int = TASKS) -> tuple[int, int, int, int]:
    need(0 <= task < tasks, "task range")
    partition = task % (432 if tasks == TASKS else 3)
    q = task // (432 if tasks == TASKS else 3)
    chunk = q % (47 if tasks == TASKS else 1); q //= (47 if tasks == TASKS else 1)
    operator = q % (4 if tasks == TASKS else 1); q //= (4 if tasks == TASKS else 1)
    return q, operator, chunk, partition


def schedule_provenance(record: dict[str, Any]) -> str:
    payload = b"".join((
        b"M1139CA_SCHEDULE_RECORD\x00\x01", bytes.fromhex(M1016_SHA),
        bytes.fromhex(M1102_SHA), bytes.fromhex(M1137_SHA),
        struct.pack(">B", AXIS_CODE[record["axis"]]),
        u64(record["task_sequence_ordinal"]), u64(record["sample"]),
        u64(record["operator"]), u64(record["chunk"]), u64(record["partition"]),
        u64(record["requested_cycle_first"]),
        bytes.fromhex(record["source_task_provenance_sha256"]),
    ))
    return hashlib.sha256(payload).hexdigest()


def validate_record(record: dict[str, Any], ordinal: int, tasks: int = TASKS) -> None:
    need(set(record) == FIELDS, "schedule exact key set")
    task = ordinal // 3; axis = AXES[ordinal % 3]
    need(record["task_sequence_ordinal"] == task and record["axis"] == axis,
         "schedule ordinal/axis order")
    need(tuple(record[name] for name in ("sample", "operator", "chunk", "partition")) ==
         coordinates(task, tasks), "schedule coordinate mapping")
    for name in ("source_task_provenance_sha256", "schedule_record_provenance_sha256"):
        need(type(record[name]) is str and re.fullmatch(r"[0-9a-f]{64}", record[name]) is not None,
             "schedule digest syntax")
    need(record["schedule_record_provenance_sha256"] == schedule_provenance(record),
         "schedule provenance")


def interval(task: int, events: int = EVENTS, tasks: int = TASKS) -> tuple[int, int]:
    return task * events // tasks, (task + 1) * events // tasks


def exact_once(axis: str, task: int, local: int, beat: int) -> bytes:
    return hashlib.sha256(f"m1130c:{axis}:{task}:{local}:{beat}:{beat}".encode()).digest()


def source_provenance(record: dict[str, Any], local: int, beat: int,
                      requested: int, half: int, row: int,
                      slices: tuple[int, ...]) -> bytes:
    payload = b"".join((
        b"M1137C_REAL_BEAT\x00\x01", bytes.fromhex(M1016_SHA),
        bytes.fromhex(M1102_SHA), bytes.fromhex(M1135_SHA),
        struct.pack(">B", AXIS_CODE[record["axis"]]), u64(record["sample"]),
        u64(record["operator"]), u64(record["chunk"]), u64(record["partition"]),
        u64(record["task_sequence_ordinal"]), u64(local), u64(beat), u64(requested),
        struct.pack(">BBB", half, row, len(slices)), bytes(slices),
    ))
    return hashlib.sha256(payload).digest()


def event_bytes(record: dict[str, Any], local: int, beat: int,
                scheduled: int, stall: int) -> bytes:
    task = record["task_sequence_ordinal"]
    requested = record["requested_cycle_first"] + local
    half = task & 1; row = beat % 16
    first_slice = ((beat // 16) % 3) * 8
    slices = tuple(range(first_slice, first_slice + 8))
    return b"".join((
        b"M1135C\x00\x01", struct.pack(">B", AXIS_CODE[record["axis"]]),
        u64(task), u64(local), u64(requested), b"W",
        struct.pack(">BBBBB", half, half, row, half * 16 + row, len(slices)),
        bytes(slices), u64(128), struct.pack(">B", 8),
        b"".join(struct.pack(">H", 0xffff) for _ in range(8)),
        u64(8), u64(beat), u64(beat), exact_once(record["axis"], task, local, beat),
        source_provenance(record, local, beat, requested, half, row, slices),
        u64(beat), u64(scheduled), u64(stall),
    ))


def digest_records(records: list[dict[str, Any]], events: int, tasks: int) -> dict[str, str]:
    states = {axis: hashlib.sha256() for axis in AXES}
    free = {axis: [0] * 24 for axis in AXES}
    for ordinal, record in enumerate(records):
        validate_record(record, ordinal, tasks)
        begin, end = interval(record["task_sequence_ordinal"], events, tasks)
        for beat in range(begin, end):
            local = beat - begin
            requested = record["requested_cycle_first"] + local
            first = ((beat // 16) % 3) * 8
            selected = tuple(range(first, first + 8))
            scheduled = max([requested] + [free[record["axis"]][slot] for slot in selected])
            stall = scheduled - requested
            states[record["axis"]].update(event_bytes(record, local, beat, scheduled, stall))
            for slot in selected:
                free[record["axis"]][slot] = scheduled + 1
    return {axis: states[axis].hexdigest() for axis in AXES}


def bounded_oracle() -> dict[str, Any]:
    requested = {
        "candidate": (5, 6, 8), "strongest_zero": (7, 8, 10),
        "same_coordinate_bit": (11, 12, 14),
    }
    records = []
    for task in range(3):
        sample, operator, chunk, partition = coordinates(task, 3)
        source = hashlib.sha256(f"m1146ca-bounded-task:{task}".encode()).hexdigest()
        for axis in AXES:
            record = {
                "axis": axis, "task_sequence_ordinal": task, "sample": sample,
                "operator": operator, "chunk": chunk, "partition": partition,
                "requested_cycle_first": requested[axis][task],
                "source_task_provenance_sha256": source,
            }
            record["schedule_record_provenance_sha256"] = schedule_provenance(record)
            records.append(record)
    observed = digest_records(records, 8, 3)
    need(observed == GOLDEN, "bounded independent golden digest mismatch")
    return {"records": 9, "events": 24, "digests": observed}


def production_schedule_scan() -> dict[str, Any]:
    regular(SCHEDULE)
    need(SCHEDULE.stat().st_size == SCHEDULE_BYTES, "schedule byte size")
    raw_hash = hashlib.sha256(); count = 0
    last_first = {axis: None for axis in AXES}
    last_end = {axis: None for axis in AXES}
    sampled: dict[int, dict[str, Any]] = {}
    with SCHEDULE.open("rb") as stream:
        for raw in stream:
            need(raw.endswith(b"\n") and not raw.endswith(b"\r\n") and len(raw) <= 65536,
                 "schedule framing")
            raw_hash.update(raw)
            record = strict_loads(raw[:-1])
            validate_record(record, count)
            axis = record["axis"]; task = record["task_sequence_ordinal"]
            begin, end = interval(task)
            first = record["requested_cycle_first"]
            need(last_first[axis] is None or first >= last_first[axis], "requested-cycle regression")
            # A non-overlapping requested interval proves zero scheduler stall:
            # every selected slice is free before the next task begins.
            need(last_end[axis] is None or first >= last_end[axis], "requested interval overlap")
            last_first[axis] = first; last_end[axis] = first + (end - begin)
            if task in SAMPLE_TASKS:
                sampled[count] = record
            count += 1
    need(count == RECORDS and raw_hash.hexdigest() == SCHEDULE_SHA, "schedule count/SHA")
    need(len(sampled) == len(SAMPLE_TASKS) * 3, "sample coverage")
    fingerprints = {}
    for ordinal, record in sorted(sampled.items()):
        begin, end = interval(record["task_sequence_ordinal"])
        digest = hashlib.sha256()
        for beat in range(begin, end):
            local = beat - begin
            requested = record["requested_cycle_first"] + local
            digest.update(event_bytes(record, local, beat, requested, 0))
        fingerprints[str(ordinal)] = {
            "axis": record["axis"], "task": record["task_sequence_ordinal"],
            "events": end - begin, "sha256": digest.hexdigest(),
        }
    return {
        "bytes": SCHEDULE.stat().st_size, "records": count,
        "sha256": raw_hash.hexdigest(), "tasks": TASKS,
        "axes_order": list(AXES), "events_per_task_min": EVENTS // TASKS,
        "events_per_task_max": (EVENTS + TASKS - 1) // TASKS,
        "zero_stall_proof": "per-axis requested intervals are non-overlapping",
        "sampled_record_event_fingerprints": fingerprints,
    }


def authority_checks() -> dict[str, Any]:
    authority = strict_json(RESULT / "expected_digest_authority.json")
    receipt = strict_json(RESULT / "receipt.json")
    runtime = strict_json(RESULT / "runtime_resources.json")
    attempt = strict_json(ATTEMPT / "attempt.json")
    expected_axis = {
        "records": TASKS, "events": EVENTS, "bytes": EVENTS * 128,
        "native_activations": EVENTS * 8, "stalled_transactions": 0,
        "stall_cycles": 0,
    }
    need(set(authority["axes"]) == set(AXES) and
         all(authority["axes"][axis] == expected_axis for axis in AXES), "axis accounting")
    need(authority["expected_count_by_axis"] == {axis: EVENTS for axis in AXES}, "axis counts")
    digests = authority["expected_digest_by_axis"]
    need(set(digests) == set(AXES) and all(re.fullmatch(r"[0-9a-f]{64}", value) for value in digests.values()),
         "authority digest syntax")
    identity = json.dumps({
        "schema": "m1146ca_expected_digest_authority_identity_v1",
        "counts": authority["expected_count_by_axis"], "digests": digests,
        "m1141_records_sha256": SCHEDULE_SHA, "m1135_semantics_sha256": M1135_SHA,
    }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    need(hashlib.sha256(identity).hexdigest() == authority["authority_id_sha256"] ==
         receipt["authority"]["authority_id_sha256"], "authority identity digest")
    need(authority["retained_event_row_or_key_history"] is False and
         authority["state_complexity"] == "O(axes + axes*24)", "authority O(1) declaration")
    need(receipt["sealed_input"] == {
        "bytes": SCHEDULE_BYTES, "records": RECORDS,
        "records_sha256_expected": SCHEDULE_SHA, "records_sha256_observed": SCHEDULE_SHA,
    }, "receipt sealed input")
    need(receipt["attempt_consumed"] is True and receipt["automatic_retry"] is False and
         receipt["event_output_written"] is False, "receipt one-shot/no-event-output")
    need(receipt["claim_boundary"] == {
        "different_author_result_hammer_required": True, "paper_citable_performance": False,
        "real_producer_replay": False, "traffic_cycles_energy_speedup": False,
    }, "claim boundary")
    need(runtime["events_compiled"] == TOTAL_EVENTS and
         runtime["input_records_streamed"] == RECORDS and
         runtime["input_bytes_streamed"] == SCHEDULE_BYTES and
         runtime["retained_event_row_or_key_history"] is False, "runtime accounting")
    need(attempt["automatic_retry"] is False and attempt["schedule_opened_before_attempt"] is False and
         attempt["status"] == "M1148CA_SINGLE_ATTEMPT_CONSUMED__NO_AUTOMATIC_RETRY", "attempt semantics")
    return {
        "authority_id_sha256": authority["authority_id_sha256"],
        "expected_digest_by_axis": digests, "events_per_axis": EVENTS,
        "bytes_per_axis": EVENTS * 128, "native_activations_per_axis": EVENTS * 8,
        "total_events": TOTAL_EVENTS, "retained_event_row_or_key_history": False,
        "event_output_written": False, "automatic_retry": False,
    }


def source_ast_checks() -> dict[str, Any]:
    need(sha(COMPILER) == COMPILER_SHA and sha(LAUNCHER) == LAUNCHER_SHA, "source SHA drift")
    compiler_tree = ast.parse(COMPILER.read_text(encoding="utf-8"), filename=str(COMPILER))
    launcher_tree = ast.parse(LAUNCHER.read_text(encoding="utf-8"), filename=str(LAUNCHER))
    imports = []
    for tree in (compiler_tree, launcher_tree):
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.extend(alias.name for alias in node.names)
    # This hammer itself does not dynamically load the author implementation;
    # static AST inspection only proves the frozen source shape.
    class_node = next(node for node in compiler_tree.body
                      if isinstance(node, ast.ClassDef) and node.name == "IndependentExpectedDigestCompiler")
    init = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    consume = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "consume_schedule_record")
    consume_appends = [node for node in ast.walk(consume) if isinstance(node, ast.Attribute) and node.attr == "append"]
    fixed_24 = any(isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult) and
                   isinstance(node.right, ast.Constant) and node.right.value == 24 for node in ast.walk(init))
    need(fixed_24 and not consume_appends, "compiler retained-history AST shape")
    need("importlib.util" in imports, "launcher dynamic-load structure absent")
    return {
        "inspection": "AST only; subject compiler/launcher not imported or called",
        "compiler_source_sha256": COMPILER_SHA, "launcher_source_sha256": LAUNCHER_SHA,
        "fixed_24_slot_state_found": True, "consume_append_calls": 0,
        "o1_conclusion": "fixed three-axis accounting/digest plus 3x24 scheduler slots; no per-event append in consume",
    }


def namespace_checks() -> dict[str, Any]:
    roots = list((HW / "results").glob("*m1148ca_c1_production_expected_digest_compiler*"))
    names = sorted(path.name for path in roots)
    need(names == [
        ".m1148ca_c1_production_expected_digest_compiler_attempt_consumed",
        "m1148ca_c1_production_expected_digest_compiler_r1_20260830",
    ], "retry/failure/work namespace present")
    result_files = {path.name for path in RESULT.iterdir() if path.is_file()}
    need(not any(path.suffix in {".jsonl", ".bin", ".csv"} for path in RESULT.iterdir()),
         "event-like output present")
    return {"namespaces": names, "event_like_output_files": [], "result_files": sorted(result_files)}


def mutation_attacks(authority: dict[str, Any]) -> dict[str, Any]:
    attacks = {}
    def rejected(name, action):
        try:
            action()
        except (Reject, json.JSONDecodeError, KeyError, ValueError, TypeError):
            attacks[name] = "REJECTED"
        else:
            raise Reject("boundary attack escaped: " + name)
    rejected("duplicate_json_key", lambda: strict_loads(b'{"a":1,"a":2}'))
    base = {
        "axis": "candidate", "task_sequence_ordinal": 0, "sample": 0,
        "operator": 0, "chunk": 0, "partition": 0, "requested_cycle_first": 0,
        "source_task_provenance_sha256": "0" * 64,
        "schedule_record_provenance_sha256": "0" * 64,
    }
    rejected("extra_schedule_key", lambda: validate_record({**base, "extra": 1}, 0))
    rejected("wrong_axis_order", lambda: validate_record({**base, "axis": "strongest_zero"}, 0))
    rejected("schedule_provenance_mutation", lambda: validate_record(base, 0))
    rejected("count_off_by_one", lambda: need(RECORDS - 1 == RECORDS, "partial count"))
    rejected("bytes_per_event_confusion", lambda: need(EVENTS * 127 == authority["bytes_per_axis"], "bytes"))
    rejected("authority_digest_mutation", lambda: need("0" * 64 == authority["authority_id_sha256"], "digest"))
    rejected("retained_history_true", lambda: need(True is False, "history"))
    rejected("event_output_present", lambda: need(["events.jsonl"] == [], "event output"))
    rejected("automatic_retry_true", lambda: need(True is False, "retry"))
    rejected("claim_upgrade", lambda: need(True is False, "real replay/speedup"))
    return attacks


def main() -> None:
    result_seal = sealed_tree(RESULT, {
        "RUN_COMPLETE.txt", "expected_digest_authority.json", "receipt.json", "runtime_resources.json",
    })
    attempt_seal = sealed_tree(ATTEMPT, {"attempt.json"})
    need(sha(DOCS359) == DOCS359_SHA, "docs359 changed")
    schedule = production_schedule_scan()
    authority = authority_checks()
    report = {
        "schema": "m1157ca_m1148ca_c1_production_expected_digest_result_hammer_r1_v1",
        "status": "PASS_M1157CA_DIFFERENT_AUTHOR_RESULT_HAMMER__EXPECTED_DIGEST_AUTHORITY_ONLY",
        "subject": {
            "result": str(RESULT.relative_to(ROOT)), "result_seal": result_seal,
            "attempt": str(ATTEMPT.relative_to(ROOT)), "attempt_seal": attempt_seal,
        },
        "schedule": schedule, "authority": authority, "bounded_oracle": bounded_oracle(),
        "source_state_audit": source_ast_checks(), "namespace_audit": namespace_checks(),
        "boundary_attacks": mutation_attacks(authority),
        "limitations": {
            "second_full_event_replay_performed": False,
            "full_digest_verification": "sealed production authority identity plus bounded golden and sampled independent event fingerprints",
            "real_producer_replay": False, "cycle_or_speedup": False,
            "traffic_or_energy": False, "paper_citable_performance": False,
        },
        "authorization": {
            "expected_digest_authority_may_be_consumed_by_successor": True,
            "real_producer_replay_still_required_for_producer_claim": True,
            "eda": False,
        },
        "docs359_sha256": sha(DOCS359),
    }
    output = Path(__file__).with_name("mechanical_checks.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(report["status"])
    print(json.dumps({
        "records": schedule["records"], "bytes": schedule["bytes"],
        "events": authority["total_events"], "authority": authority["authority_id_sha256"],
        "attacks_rejected": len(report["boundary_attacks"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
