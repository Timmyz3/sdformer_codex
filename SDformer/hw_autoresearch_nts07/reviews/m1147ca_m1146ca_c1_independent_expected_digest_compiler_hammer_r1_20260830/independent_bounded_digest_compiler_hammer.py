#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author bounded hammer for the frozen M1146CA compiler source.

This script never opens the production M1141CA JSONL and never invokes the
production compiler, a driver, a full replay, or EDA.  It independently
reconstructs the bounded 17-field events, M1137C provenance, M1135C byte
serialization, and the three 24-slot schedulers before authorizing only a
successor production-compiler launcher/source.
"""
from __future__ import annotations

import ast
import builtins
from dataclasses import fields, replace
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import re
import stat
import struct
import sys
from typing import Any, Callable, Mapping
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/build_m1146ca_c1_independent_expected_digest_compiler_source.py"
CONTRACT = HW / "contracts/m1146ca_c1_independent_expected_digest_compiler_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1146ca_c1_independent_expected_digest_compiler_author_receipt_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
OUT = HERE

EXPECTED = {
    "source": "7b1f5cd2cd4c4bb0a771d0360f8be924d075215e8dd660728a8decac0c886e73",
    "contract": "5f36b42c088e0143ab61b90098d55610c7e4bb555f318b416968759c45a33a2f",
    "contract_side": "854d2d1c0b1162e8618198d44c2fe1f7bb272672735474e6a7489e405f4bc02c",
    "contract_outer": "60fae54229f5c6f802127ea10906ed7bec3c42e8d2b6bb50b6b323e8c4e42b13",
    "author_outer": "9aa612c53b3d4064f4fb80ac057f936459624cc7a211373664a9fd04c3650414",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
AXES = ("candidate", "strongest_zero", "same_coordinate_bit")
AXIS_CODE = {axis: index for index, axis in enumerate(AXES)}
EVENT_FIELDS = (
    "axis", "task_id", "source_local_ordinal", "requested_cycle", "op",
    "logical_bank", "half_slot", "logical_row", "local_row", "native_slices",
    "bytes", "byte_enable_per_slice", "native_macro_activations",
    "service_beat_ordinal", "store_transaction_ordinal",
    "service_event_exact_once_id", "source_row_provenance_sha256",
)
SOURCE_HASHES = {
    "m1016": "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa",
    "m1102": "95bd50aebcc473ab69cdea6ccf27d54743c89926c5e0f31199dc469ced9bf7cc",
    "m1135": "4c282b4ece5705b5c8dcd039c29003c14e544ffef5e8c4234afab0ac31ac7571",
    "m1137": "9ec640ae8c9fa75f9cbf706e15d2d26a4233def77e5be4d67e94c084347b20a6",
}
GOLDEN = {
    "candidate": "ab87d9d8da38d28a54d6048dc75cb7ac749aebba7807f855cac69165b9fa5644",
    "strongest_zero": "eb2dd17d2d0aa43e19d2f66b9d079760f7495c1f9b4653d206831605e1b44717",
    "same_coordinate_bit": "18a4e643ee4a606b5ec8e646fbd76aa155ffe324213a4f8bb36925c6fb678d7a",
}
checks = 0
attacks: dict[str, str] = {}


class HammerFailure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise HammerFailure(message)


def rejected(label: str, action: Callable[[], Any], contains: str | None = None) -> None:
    try:
        action()
    except Exception as error:
        if contains is not None:
            require(contains.lower() in str(error).lower(),
                    f"{label}: wrong rejection: {error}")
        attacks[label] = f"{type(error).__name__}: {error}"
        return
    raise HammerFailure("attack accepted: " + label)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_regular(path: Path, expected: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and sha256(path) == expected,
            "identity drift: " + str(path))


def strict_json(path: Path) -> Any:
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          HammerFailure("nonfinite JSON: " + token)))


def verify_flat(directory: Path, expected_outer: str) -> dict[str, Any]:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    verify_regular(outer, expected_outer)
    outer_parts = outer.read_text(encoding="utf-8").split()
    require(len(outer_parts) == 2 and outer_parts[1] == "SHA256SUMS" and
            re.fullmatch(r"[0-9a-f]{64}", outer_parts[0]) is not None,
            "author outer syntax")
    verify_regular(manifest, outer_parts[0])
    listed: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, raw_name = line.split(maxsplit=1)
        name = raw_name.lstrip("*")
        relative = Path(name)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None and
                name not in listed and name == relative.as_posix() and
                not relative.is_absolute() and ".." not in relative.parts,
                "author manifest row")
        listed[name] = digest
    actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "author symlink")
        if stat.S_ISREG(mode):
            actual.add(name)
        else:
            require(stat.S_ISDIR(mode), "author special member")
    require(actual == set(listed), "author exact member set")
    for name, digest in listed.items():
        verify_regular(directory / name, digest)
    return strict_json(directory / "review.json")


def verify_identities() -> dict[str, Any]:
    verify_regular(SOURCE, EXPECTED["source"])
    verify_regular(CONTRACT, EXPECTED["contract"])
    side = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    verify_regular(side, EXPECTED["contract_side"])
    verify_regular(outer, EXPECTED["contract_outer"])
    require(side.read_text(encoding="utf-8").split() ==
            [EXPECTED["contract"], CONTRACT.name], "contract side content")
    require(outer.read_text(encoding="utf-8").split() ==
            [EXPECTED["contract_side"], side.name], "contract outer content")
    verify_regular(DOCS359, EXPECTED["docs359"])
    author = verify_flat(AUTHOR, EXPECTED["author_outer"])
    require(author["status"] ==
            "PASS_M1146CA_INDEPENDENT_EXPECTED_DIGEST_COMPILER_AUTHOR__BOUNDED_ONLY_PRODUCTION_STOP" and
            author["subject"]["source_sha256"] == EXPECTED["source"] and
            tuple(author["subject"]["contract_identity"]) ==
                (EXPECTED["contract"], EXPECTED["contract_side"], EXPECTED["contract_outer"]) and
            author["authorization"]["production_digest_compiler_execution"] is False,
            "author receipt schema/authorization drift")
    return author


def load_subject():
    spec = importlib.util.spec_from_file_location("m1147ca_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "subject import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def u64(value: int) -> bytes:
    require(type(value) is int and 0 <= value < 1 << 64, "oracle u64")
    return struct.pack(">Q", value)


def independent_schedule_provenance(axis: str, task: int, sample: int,
                                    operator: int, chunk: int, partition: int,
                                    requested: int, source_task: str) -> str:
    payload = b"".join((
        b"M1139CA_SCHEDULE_RECORD\x00\x01", bytes.fromhex(SOURCE_HASHES["m1016"]),
        bytes.fromhex(SOURCE_HASHES["m1102"]), bytes.fromhex(SOURCE_HASHES["m1137"]),
        struct.pack(">B", AXIS_CODE[axis]), u64(task), u64(sample), u64(operator),
        u64(chunk), u64(partition), u64(requested), bytes.fromhex(source_task),
    ))
    return hashlib.sha256(payload).hexdigest()


def independent_event(axis: str, task: int, requested_first: int,
                      global_beat: int, begin: int) -> dict[str, Any]:
    local = global_beat - begin
    requested = requested_first + local
    half = task & 1
    row = global_beat % 16
    base = ((global_beat // 16) % 3) * 8
    slices = tuple(range(base, base + 8))
    exact = hashlib.sha256(
        f"m1130c:{axis}:{task}:{local}:{global_beat}:{global_beat}".encode("utf-8")
    ).hexdigest()
    provenance = hashlib.sha256(b"".join((
        b"M1137C_REAL_BEAT\x00\x01", bytes.fromhex(SOURCE_HASHES["m1016"]),
        bytes.fromhex(SOURCE_HASHES["m1102"]), bytes.fromhex(SOURCE_HASHES["m1135"]),
        struct.pack(">B", AXIS_CODE[axis]), u64(0), u64(0), u64(0), u64(task),
        u64(task), u64(local), u64(global_beat), u64(requested),
        struct.pack(">B", half), struct.pack(">B", row), struct.pack(">B", 8),
        bytes(slices),
    ))).hexdigest()
    return {
        "axis": axis, "task_id": task, "source_local_ordinal": local,
        "requested_cycle": requested, "op": "WRITE", "logical_bank": half,
        "half_slot": half, "logical_row": row, "local_row": half * 16 + row,
        "native_slices": slices, "bytes": 128,
        "byte_enable_per_slice": (0xffff,) * 8, "native_macro_activations": 8,
        "service_beat_ordinal": global_beat,
        "store_transaction_ordinal": global_beat,
        "service_event_exact_once_id": exact,
        "source_row_provenance_sha256": provenance,
    }


def independent_bytes(event: Mapping[str, Any], sequence: int,
                      scheduled: int, stall: int) -> bytes:
    return b"".join((
        b"M1135C\x00\x01", struct.pack(">B", AXIS_CODE[event["axis"]]),
        u64(event["task_id"]), u64(event["source_local_ordinal"]),
        u64(event["requested_cycle"]), b"W", struct.pack(">B", event["logical_bank"]),
        struct.pack(">B", event["half_slot"]), struct.pack(">B", event["logical_row"]),
        struct.pack(">B", event["local_row"]),
        struct.pack(">B", len(event["native_slices"])), bytes(event["native_slices"]),
        u64(event["bytes"]), struct.pack(">B", len(event["byte_enable_per_slice"])),
        b"".join(struct.pack(">H", item) for item in event["byte_enable_per_slice"]),
        u64(event["native_macro_activations"]), u64(event["service_beat_ordinal"]),
        u64(event["store_transaction_ordinal"]),
        bytes.fromhex(event["service_event_exact_once_id"]),
        bytes.fromhex(event["source_row_provenance_sha256"]),
        u64(sequence), u64(scheduled), u64(stall),
    ))


def independent_oracle(module) -> tuple[dict[str, str], dict[str, Any]]:
    records = list(module.bounded_schedule_records())
    require(len(records) == 9, "bounded record count")
    next_free = {axis: [0] * 24 for axis in AXES}
    digests = {axis: hashlib.sha256() for axis in AXES}
    stalls = {axis: [0, 0] for axis in AXES}
    event_checks = 0
    for ordinal, record in enumerate(records):
        task = ordinal // 3
        axis = AXES[ordinal % 3]
        require(record.axis == axis and record.task_sequence_ordinal == task,
                "bounded task/axis order")
        require(record.schedule_record_provenance_sha256 == independent_schedule_provenance(
            axis, task, record.sample, record.operator, record.chunk, record.partition,
            record.requested_cycle_first, record.source_task_provenance_sha256),
            "independent M1139 provenance")
        begin = task * 8 // 3
        end = (task + 1) * 8 // 3
        for beat in range(begin, end):
            expected = independent_event(axis, task, record.requested_cycle_first, beat, begin)
            actual = module.reconstruct_event(record, beat, begin)
            require(tuple(field.name for field in fields(actual)) == EVENT_FIELDS,
                    "17-field declaration/order drift")
            for field in EVENT_FIELDS:
                require(getattr(actual, field) == expected[field],
                        "independent event mismatch: " + field)
                event_checks += 1
            scheduled = max([expected["requested_cycle"]] +
                            [next_free[axis][slot] for slot in expected["native_slices"]])
            stall = scheduled - expected["requested_cycle"]
            subject_bytes = module.canonical_event_bytes(actual, beat, scheduled, stall)
            oracle_bytes = independent_bytes(expected, beat, scheduled, stall)
            require(subject_bytes == oracle_bytes, "M1135 fixed-endian byte mismatch")
            require(len(subject_bytes) == 184, "M1135 serialized length drift")
            digests[axis].update(oracle_bytes)
            for slot in expected["native_slices"]:
                next_free[axis][slot] = scheduled + 1
            stalls[axis][0] += int(stall > 0)
            stalls[axis][1] += stall
    answer = {axis: digest.hexdigest() for axis, digest in digests.items()}
    require(answer == GOLDEN == module.BOUNDED_GOLDEN_DIGESTS,
            "independent bounded golden mismatch")
    require(all(stalls[axis] == [6, 9] for axis in AXES), "scheduler stall oracle")
    return answer, {"event_field_checks": event_checks,
                    "serializer_byte_checks": 24 * 184,
                    "scheduler_state_entries": 72,
                    "stalls": stalls}


def compile_records(module, records):
    compiler = module.IndependentExpectedDigestCompiler(module.BOUNDED_GEOMETRY)
    beats = [compiler.consume_schedule_record(record) for record in records]
    accepted = []
    authority = compiler.finalize(accepted.append)
    require(len(accepted) == 1, "sink cardinality")
    return compiler, beats, authority


def assert_golden(module, records) -> None:
    authority = compile_records(module, records)[2]
    require(authority["expected_digest_by_axis"] == GOLDEN, "golden mismatch")


def static_and_positive(module) -> dict[str, Any]:
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    imports = [alias.name.lower() for node in ast.walk(tree)
               if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names]
    require(not any(any(subject in name for subject in
                        ("m1137", "m1135", "m1130", "m1132")) for name in imports),
            "subject runtime import")
    require("importlib" not in imports, "dynamic import surface")
    compiler_class = next(node for node in tree.body
                          if isinstance(node, ast.ClassDef) and
                          node.name == "IndependentExpectedDigestCompiler")
    mutators = [node.func.attr for node in ast.walk(compiler_class)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                node.func.attr in {"append", "extend", "add", "insert"}]
    require(mutators == [], "event/key history retained")
    constructor = inspect.signature(module.IndependentExpectedDigestCompiler)
    require(tuple(constructor.parameters) == ("geometry",), "caller secret parameter")
    require(module.PRODUCTION_COMPILER_EXECUTION_AUTHORIZATION_SHA256 is None,
            "production authorization unexpectedly populated")
    production = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and
                      node.name == "compile_production_expected_digest_authority")
    ptext = ast.unparse(production)
    require("M1141_RECORDS" not in ptext and "open(" not in ptext and
            ptext.index("PRODUCTION_COMPILER_EXECUTION_AUTHORIZATION_SHA256") <
            ptext.index("raise Failure"), "production gate/opener drift")
    compiler = module.IndependentExpectedDigestCompiler(module.BOUNDED_GEOMETRY)
    require(len(compiler._axis) == 3 and
            sum(len(state.next_free_cycle) for state in compiler._axis.values()) == 72,
            "3 axes x 24 scheduler drift")

    record_opens = 0
    original_path_open = Path.open
    original_builtin_open = builtins.open
    target = module.M1141_RECORDS.resolve()
    def watched_path_open(path, *args, **kwargs):
        nonlocal record_opens
        if Path(path).resolve() == target:
            record_opens += 1
            raise HammerFailure("production M1141 JSONL opened")
        return original_path_open(path, *args, **kwargs)
    def watched_builtin_open(file, *args, **kwargs):
        nonlocal record_opens
        try:
            resolved = Path(file).resolve()
        except TypeError:
            resolved = None
        if resolved == target:
            record_opens += 1
            raise HammerFailure("production M1141 JSONL opened")
        return original_builtin_open(file, *args, **kwargs)
    with patch.object(Path, "open", watched_path_open), patch.object(builtins, "open", watched_builtin_open):
        small = module.source_small_oracle()
    require(record_opens == 0 and small["production_events_compiled"] == 0 and
            small["production_schedule_records_opened"] is False,
            "production path touched")
    records = list(module.bounded_schedule_records())
    compiled, beats, authority = compile_records(module, records)
    require(beats == [2, 2, 2, 3, 3, 3, 3, 3, 3], "variable beat partition")
    require(authority["expected_count_by_axis"] == {axis: 8 for axis in AXES} and
            authority["expected_digest_by_axis"] == GOLDEN and
            authority["retained_event_row_or_key_history"] is False and
            compiled.snapshot()["state_complexity"] == "O(axes + axes*24)",
            "terminal bounded authority")
    return {"imports": imports, "history_mutators": mutators,
            "scheduler_state_entries": 72, "production_jsonl_opens": record_opens,
            "production_events_compiled": 0, "beats": beats,
            "authority_id_sha256": authority["authority_id_sha256"]}


def attacks_fail_closed(module) -> None:
    records = list(module.bounded_schedule_records())
    partial = module.IndependentExpectedDigestCompiler(module.BOUNDED_GEOMETRY)
    for record in records[:-1]:
        partial.consume_schedule_record(record)
    rejected("partial_finalize", lambda: partial.finalize(lambda _: None), "partial")
    rejected("missing_record", lambda: compile_records(module, records[:-1]), "partial")
    duplicate = records[:1] + records
    rejected("duplicate_record", lambda: compile_records(module, duplicate), "schedule")
    reordered = records.copy(); reordered[0], reordered[1] = reordered[1], reordered[0]
    rejected("record_reorder", lambda: compile_records(module, reordered), "schedule")
    bad_task = records.copy(); bad_task[0] = replace(bad_task[0], task_sequence_ordinal=1)
    rejected("task_coordinate", lambda: compile_records(module, bad_task), "coordinate")
    bad_axis = records.copy(); item = bad_axis[0]
    bad_axis[0] = replace(item, axis=AXES[1],
        schedule_record_provenance_sha256=module.schedule_record_provenance(
            AXES[1], item.task_sequence_ordinal, item.sample, item.operator, item.chunk,
            item.partition, item.requested_cycle_first, item.source_task_provenance_sha256))
    rejected("axis_order", lambda: compile_records(module, bad_axis), "axis")
    bad_prov = records.copy(); bad_prov[0] = replace(
        bad_prov[0], schedule_record_provenance_sha256="0" * 64)
    rejected("schedule_provenance", lambda: compile_records(module, bad_prov), "provenance")
    bad_source = records.copy(); item = bad_source[0]
    bad_source[0] = replace(item, source_task_provenance_sha256="1" * 64)
    rejected("source_task_provenance", lambda: compile_records(module, bad_source), "provenance")
    regressed = records.copy(); item = regressed[3]
    regressed[3] = replace(item, requested_cycle_first=4,
        schedule_record_provenance_sha256=module.schedule_record_provenance(
            item.axis, item.task_sequence_ordinal, item.sample, item.operator,
            item.chunk, item.partition, 4, item.source_task_provenance_sha256))
    rejected("requested_cycle_regression", lambda: compile_records(module, regressed), "regression")

    original_reconstruct = module.reconstruct_event
    mutations = {
        "axis": lambda e: replace(e, axis="invalid"),
        "task_id": lambda e: replace(e, task_id=-1),
        "source_local_ordinal": lambda e: replace(e, source_local_ordinal=-1),
        "requested_cycle": lambda e: replace(e, requested_cycle=-1),
        "op": lambda e: replace(e, op="READ"),
        "logical_bank": lambda e: replace(e, logical_bank=1 - e.logical_bank),
        "half_slot": lambda e: replace(e, half_slot=2),
        "logical_row": lambda e: replace(e, logical_row=16),
        "local_row": lambda e: replace(e, local_row=e.local_row + 1),
        "native_slices": lambda e: replace(e, native_slices=(0, 1, 2, 3, 4, 5, 6, 8)),
        "bytes": lambda e: replace(e, bytes=64),
        "byte_enable_per_slice": lambda e: replace(e, byte_enable_per_slice=(0xffff,) * 7),
        "native_macro_activations": lambda e: replace(e, native_macro_activations=7),
        "service_beat_ordinal": lambda e: replace(e, service_beat_ordinal=-1),
        "store_transaction_ordinal": lambda e: replace(e, store_transaction_ordinal=-1),
        "service_event_exact_once_id": lambda e: replace(e, service_event_exact_once_id="0" * 64),
        "source_row_provenance_sha256": lambda e: replace(e, source_row_provenance_sha256="0" * 64),
    }
    for field_name, mutate in mutations.items():
        calls = 0
        def changed(record, beat, begin, mutate=mutate):
            nonlocal calls
            event = original_reconstruct(record, beat, begin)
            calls += 1
            return mutate(event) if calls == 1 else event
        with patch.object(module, "reconstruct_event", changed):
            if field_name == "source_row_provenance_sha256":
                rejected("field_" + field_name, lambda: assert_golden(module, records), "golden")
            else:
                rejected("field_" + field_name, lambda: assert_golden(module, records))

    original_serializer = module.canonical_event_bytes
    serializer_mutations = {
        "serializer_extra_field": lambda payload: payload + b"\x00",
        "serializer_order": lambda payload: payload[::-1],
        "serializer_endianness": lambda payload: payload[:-24] +
            struct.pack("<QQQ", *struct.unpack(">QQQ", payload[-24:])),
    }
    for label, mutate in serializer_mutations.items():
        with patch.object(module, "canonical_event_bytes",
                          lambda event, seq, cycle, stall, mutate=mutate:
                          mutate(original_serializer(event, seq, cycle, stall))):
            rejected(label, lambda: assert_golden(module, records), "golden")

    with patch.object(module, "canonical_event_bytes",
                      lambda event, seq, cycle, stall:
                      original_serializer(event, seq, cycle + 1, stall + 1)):
        rejected("scheduled_cycle", lambda: assert_golden(module, records), "golden")
    original_id = module.exact_once_id
    with patch.object(module, "exact_once_id", lambda *args:
                      hashlib.sha256((original_id(*args) + "drift").encode()).hexdigest()):
        rejected("exact_id_algorithm", lambda: assert_golden(module, records), "golden")
    with patch.object(module, "source_row_provenance", lambda *args: "f" * 64):
        rejected("source_provenance_algorithm", lambda: assert_golden(module, records), "golden")

    rejected("caller_digest_secret", lambda:
             module.IndependentExpectedDigestCompiler(module.BOUNDED_GEOMETRY,
                                                       expected_digest_by_axis=GOLDEN),
             "unexpected")
    complete = module.IndependentExpectedDigestCompiler(module.BOUNDED_GEOMETRY)
    for record in records:
        complete.consume_schedule_record(record)
    before = complete.snapshot()
    rejected("terminal_sink_failure", lambda: complete.finalize(
        lambda _: (_ for _ in ()).throw(RuntimeError("controlled sink failure"))), "controlled")
    require(complete.snapshot() == before and not complete.snapshot()["finalized"],
            "sink failure committed state")
    accepted = []
    complete.finalize(accepted.append)
    require(len(accepted) == 1, "sink retry cardinality")

    atomic = module.IndependentExpectedDigestCompiler(module.BOUNDED_GEOMETRY)
    before = atomic.snapshot(); calls = 0
    def failing_serializer(event, sequence, scheduled, stall):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("controlled serializer failure")
        return original_serializer(event, sequence, scheduled, stall)
    with patch.object(module, "canonical_event_bytes", failing_serializer):
        rejected("record_atomicity", lambda: atomic.consume_schedule_record(records[0]), "controlled")
    require(atomic.snapshot() == before, "failed record partially committed")
    rejected("production_gate", module.compile_production_expected_digest_authority, "not authorized")


def write_receipt(author: Mapping[str, Any], static: Mapping[str, Any],
                  golden: Mapping[str, str], oracle: Mapping[str, Any]) -> None:
    review = {
        "schema": "m1147ca_m1146ca_c1_independent_expected_digest_compiler_hammer_r1_v1",
        "status": "PASS_M1147CA_DIFFERENT_AUTHOR_BOUNDED_DIGEST_COMPILER_HAMMER__PRODUCTION_LAUNCHER_SOURCE_ONLY_NEXT",
        "date": "2026-08-30",
        "subject": {
            "source": str(SOURCE.relative_to(HW)), "source_sha256": EXPECTED["source"],
            "contract_identity": [EXPECTED["contract"], EXPECTED["contract_side"],
                                  EXPECTED["contract_outer"]],
            "author_receipt_outer_sha256": EXPECTED["author_outer"],
        },
        "verdict": {
            "independent_17_field_reconstruction": True,
            "independent_exact_id": True, "independent_m1137_provenance": True,
            "independent_m1135_fixed_endian_serialization": True,
            "independent_three_axis_24_slot_scheduler": True,
            "bounded_golden_match": True, "all_attacks_rejected": True,
        },
        "evidence": {
            "checks": checks, "attacks_rejected": len(attacks),
            "attack_labels": sorted(attacks), "tasks": 3, "axes": 3,
            "events": 24, "event_fields": 17,
            "event_field_checks": oracle["event_field_checks"],
            "serializer_byte_checks": oracle["serializer_byte_checks"],
            "scheduler_state_entries": oracle["scheduler_state_entries"],
            "state_complexity": "O(axes + axes*24)",
            "retained_event_row_or_key_history": False,
            "expected_digest_by_axis": dict(golden),
            "production_schedule_jsonl_opens": static["production_jsonl_opens"],
            "production_events_compiled": 0, "production_target_events": 212559552,
        },
        "authorization": {
            "one_shot_production_digest_compiler_launcher_source_next": True,
            "production_digest_compiler_execution_by_this_hammer": False,
            "real_producer_replay": False, "full_replay": False, "eda": False,
        },
        "claim_boundary": {
            "source_and_bounded_synthetic_only": True,
            "production_expected_digest_authority": False,
            "traffic_cycles_energy_speedup": False,
            "paper_citable_performance": False, "paper_ppa_ready": False,
        },
        "docs359_sha256": EXPECTED["docs359"],
    }
    mechanical = {
        "schema": "m1147ca_mechanical_checks_r1_v1", "checks": checks,
        "attacks_rejected": attacks, "static": dict(static),
        "independent_oracle": dict(oracle), "author_status": author["status"],
    }
    markdown = "\n".join((
        "# M1147CA different-author bounded hammer",
        "",
        f"- Verdict: `{review['status']}`",
        f"- Checks: {checks}; attacks rejected: {len(attacks)}.",
        "- Independently matched all 17 event fields, exact-ID, M1137 provenance, "
        "M1135 fixed-endian bytes, and three 24-slot schedulers.",
        "- Production M1141 JSONL opens: 0; production events compiled: 0.",
        "- Authorization: a successor may author a one-shot production compiler "
        "launcher/source. This hammer did not authorize or execute production.",
        "- No driver, full replay, EDA, performance, energy, or paper-PPA claim.",
        "",
    ))
    marker = "\n".join((
        "M1147CA PASS is bounded-source evidence only.",
        "NO PRODUCTION M1141 JSONL OPEN.",
        "NO PRODUCTION DIGEST COMPILATION OR REAL PRODUCER REPLAY.",
        "NO FULL REPLAY OR EDA.",
        "NEXT: ONE-SHOT PRODUCTION DIGEST COMPILER LAUNCHER/SOURCE AUTHORING ONLY.",
        "",
    ))
    (OUT / "review.json").write_text(json.dumps(review, indent=2, sort_keys=True,
                                                  allow_nan=False) + "\n", encoding="utf-8")
    (OUT / "mechanical_checks.json").write_text(
        json.dumps(mechanical, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    (OUT / "review.md").write_text(markdown, encoding="utf-8")
    (OUT / "BOUNDED_ONLY_NO_PRODUCTION_JSONL_NO_COMPILER_NO_REPLAY_NO_EDA.txt").write_text(
        marker, encoding="utf-8")
    (OUT / "RUN_COMPLETE.txt").write_text(review["status"] + "\n", encoding="utf-8")
    members = sorted(path for path in OUT.iterdir()
                     if path.is_file() and path.name not in
                     {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    lines = [f"{sha256(path)}  {path.name}\n" for path in members]
    (OUT / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")
    outer_digest = sha256(OUT / "SHA256SUMS")
    (OUT / "SHA256SUMS.seal.sha256").write_text(
        f"{outer_digest}  SHA256SUMS\n", encoding="utf-8")


def main() -> None:
    require(sys.argv[1:] == [], "zero-argument bounded hammer only")
    author = verify_identities()
    module = load_subject()
    static = static_and_positive(module)
    golden, oracle = independent_oracle(module)
    attacks_fail_closed(module)
    require(sha256(DOCS359) == EXPECTED["docs359"], "docs359 post-hammer drift")
    write_receipt(author, static, golden, oracle)
    print(json.dumps({"status": "PASS", "checks": checks,
                      "attacks_rejected": len(attacks),
                      "outer_seal_sha256": sha256(OUT / "SHA256SUMS.seal.sha256")},
                     sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
