#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1146CA author check: bounded digest compiler only, production STOP."""
from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import struct
import sys
from typing import Any, Callable
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/build_m1146ca_c1_independent_expected_digest_compiler_source.py"
CONTRACT = HW / "contracts/m1146ca_c1_independent_expected_digest_compiler_source_contract_r1_20260830.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    "source": "7b1f5cd2cd4c4bb0a771d0360f8be924d075215e8dd660728a8decac0c886e73",
    "contract": "5f36b42c088e0143ab61b90098d55610c7e4bb555f318b416968759c45a33a2f",
    "contract_side": "854d2d1c0b1162e8618198d44c2fe1f7bb272672735474e6a7489e405f4bc02c",
    "contract_outer": "60fae54229f5c6f802127ea10906ed7bec3c42e8d2b6bb50b6b323e8c4e42b13",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
AXES = ("candidate", "strongest_zero", "same_coordinate_bit")
GOLDEN = {
    "candidate": "ab87d9d8da38d28a54d6048dc75cb7ac749aebba7807f855cac69165b9fa5644",
    "strongest_zero": "eb2dd17d2d0aa43e19d2f66b9d079760f7495c1f9b4653d206831605e1b44717",
    "same_coordinate_bit": "18a4e643ee4a606b5ec8e646fbd76aa155ffe324213a4f8bb36925c6fb678d7a",
}
checks = 0
attacks: dict[str, str] = {}


class CheckFailure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise CheckFailure(message)


def rejected(label: str, action: Callable[[], Any], contains: str | None = None) -> None:
    try:
        action()
    except Exception as error:
        if contains is not None:
            require(contains in str(error), label + " wrong rejection: " + str(error))
        attacks[label] = type(error).__name__ + ": " + str(error)
        return
    raise CheckFailure("attack accepted: " + label)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_regular(path: Path, expected: str) -> None:
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink() and
            sha(path) == expected, "identity drift: " + str(path))


def verify_frozen() -> None:
    verify_regular(SOURCE, EXPECTED["source"]); verify_regular(CONTRACT, EXPECTED["contract"])
    side = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    verify_regular(side, EXPECTED["contract_side"]); verify_regular(outer, EXPECTED["contract_outer"])
    require(side.read_text(encoding="utf-8").split() == [EXPECTED["contract"], CONTRACT.name] and
            outer.read_text(encoding="utf-8").split() == [EXPECTED["contract_side"], side.name],
            "contract double seal")
    verify_regular(DOCS359, EXPECTED["docs359"])


def load_subject():
    spec = importlib.util.spec_from_file_location("m1146ca_author_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "subject module spec")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def static_checks(module) -> dict[str, Any]:
    text = SOURCE.read_text(encoding="utf-8"); tree = ast.parse(text)
    imported = [alias.name.lower() for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names]
    require(not any(any(token in name for token in ("m1137", "m1135", "m1130", "m1132"))
                    for name in imported) and "importlib" not in imported,
            "forbidden subject runtime import")
    require(not any(token in text for token in (
        "load_m1137", "load_m1135", "load_m1130", "load_m1132",
        "producer_output", "producer_result")), "forbidden subject call/output read")
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and
                 node.name == "IndependentExpectedDigestCompiler")
    history_calls = []
    for node in ast.walk(klass):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                node.func.attr in {"append", "extend", "add"}):
            history_calls.append(node.func.attr)
    require(history_calls == [], "compiler retained event/row/key history")
    production = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and
                      node.name == "compile_production_expected_digest_authority")
    production_text = ast.unparse(production)
    require(production_text.index("require(PRODUCTION_COMPILER_EXECUTION_AUTHORIZATION_SHA256") <
            production_text.index("raise Failure") and "open(" not in production_text,
            "production gate/open drift")
    compiler = module.IndependentExpectedDigestCompiler(module.BOUNDED_GEOMETRY)
    require(len(compiler._axis) == 3 and
            all(len(state.next_free_cycle) == 24 for state in compiler._axis.values()),
            "scheduler is not exact 3x24 state")
    return {
        "subject_runtime_imports_or_calls": 0,
        "producer_output_reads": 0,
        "compiler_history_mutators": history_calls,
        "scheduler_state_entries": 72,
        "state_complexity": "O(axes + axes*24)",
        "production_gate_before_record_open": True,
    }


def independent_golden() -> dict[str, str]:
    """Separate implementation: no subject helper, event class, or serializer."""
    axis_code = {axis: index for index, axis in enumerate(AXES)}
    requested = {
        "candidate": (5, 6, 8), "strongest_zero": (7, 8, 10),
        "same_coordinate_bit": (11, 12, 14),
    }
    h1016 = "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa"
    h1102 = "95bd50aebcc473ab69cdea6ccf27d54743c89926c5e0f31199dc469ced9bf7cc"
    h1135 = "4c282b4ece5705b5c8dcd039c29003c14e544ffef5e8c4234afab0ac31ac7571"
    u64 = lambda value: struct.pack(">Q", value)
    answer = {}
    for axis in AXES:
        digest = hashlib.sha256(); next_free = [0] * 24
        for task_id in range(3):
            begin = task_id * 8 // 3; end = (task_id + 1) * 8 // 3
            for global_beat in range(begin, end):
                local = global_beat - begin; req = requested[axis][task_id] + local
                half = task_id & 1; row = global_beat % 16
                base = ((global_beat // 16) % 3) * 8
                slices = tuple(range(base, base + 8))
                cycle = max([req] + [next_free[item] for item in slices]); stall = cycle - req
                exact = hashlib.sha256(
                    f"m1130c:{axis}:{task_id}:{local}:{global_beat}:{global_beat}".encode()).digest()
                provenance = hashlib.sha256(b"".join((
                    b"M1137C_REAL_BEAT\x00\x01", bytes.fromhex(h1016), bytes.fromhex(h1102),
                    bytes.fromhex(h1135), struct.pack(">B", axis_code[axis]),
                    u64(0), u64(0), u64(0), u64(task_id), u64(task_id), u64(local),
                    u64(global_beat), u64(req), struct.pack(">B", half),
                    struct.pack(">B", row), struct.pack(">B", 8), bytes(slices)))).digest()
                payload = b"".join((
                    b"M1135C\x00\x01", struct.pack(">B", axis_code[axis]), u64(task_id),
                    u64(local), u64(req), b"W", struct.pack(">B", half),
                    struct.pack(">B", half), struct.pack(">B", row),
                    struct.pack(">B", half * 16 + row), struct.pack(">B", 8), bytes(slices),
                    u64(128), struct.pack(">B", 8),
                    b"".join(struct.pack(">H", 0xffff) for _ in range(8)), u64(8),
                    u64(global_beat), u64(global_beat), exact, provenance,
                    u64(global_beat), u64(cycle), u64(stall)))
                digest.update(payload)
                for item in slices:
                    next_free[item] = cycle + 1
        answer[axis] = digest.hexdigest()
    return answer


def compile_records(module, records):
    compiler = module.IndependentExpectedDigestCompiler(module.BOUNDED_GEOMETRY)
    beats = [compiler.consume_schedule_record(record) for record in records]
    accepted = []
    authority = compiler.finalize(accepted.append)
    require(len(accepted) == 1, "authority sink cardinality")
    return compiler, beats, authority


def bounded_positive(module) -> dict[str, Any]:
    row_opens = 0
    original_open = Path.open
    def watched_open(path, *args, **kwargs):
        nonlocal row_opens
        if Path(path) == module.M1141_RECORDS:
            row_opens += 1
            raise CheckFailure("production records opened")
        return original_open(path, *args, **kwargs)
    with patch.object(Path, "open", watched_open):
        preflight_oracle = module.source_small_oracle()
    records = list(module.bounded_schedule_records())
    compiler, beats, authority = compile_records(module, records)
    reference = independent_golden()
    require(reference == GOLDEN == module.BOUNDED_GOLDEN_DIGESTS ==
            authority["expected_digest_by_axis"], "independent golden mismatch")
    require(beats == [2, 2, 2, 3, 3, 3, 3, 3, 3] and
            all(authority["axes"][axis]["stalled_transactions"] == 6 and
                authority["axes"][axis]["stall_cycles"] == 9 for axis in AXES),
            "bounded variable beat/scheduler coverage")
    require(row_opens == 0 and preflight_oracle["production_events_compiled"] == 0,
            "production was touched")
    return {
        "tasks": 3, "axes": 3, "schedule_records": 9, "events": 24,
        "beats_per_task": [2, 3, 3], "independent_golden": reference,
        "authority_id_sha256": authority["authority_id_sha256"],
        "stalled_transactions_per_axis": 6, "stall_cycles_per_axis": 9,
        "scheduler_state_entries": 72,
        "production_schedule_record_opens": row_opens,
        "production_events_compiled": 0,
        "retained_event_row_or_key_history": False,
    }


def fail_closed_attacks(module) -> None:
    records = list(module.bounded_schedule_records())
    partial = module.IndependentExpectedDigestCompiler(module.BOUNDED_GEOMETRY)
    for record in records[:-1]:
        partial.consume_schedule_record(record)
    rejected("partial_output", lambda: partial.finalize(lambda value: None), "partial")

    reordered = records.copy(); reordered[0], reordered[1] = reordered[1], reordered[0]
    rejected("schedule_reorder", lambda: compile_records(module, reordered), "schedule")
    wrong_task = records.copy(); wrong_task[0] = replace(wrong_task[0], task_sequence_ordinal=1)
    rejected("schedule_task", lambda: compile_records(module, wrong_task), "coordinate")
    wrong_axis = records.copy()
    item = wrong_axis[0]
    wrong_axis[0] = replace(item, axis="strongest_zero",
        schedule_record_provenance_sha256=module.schedule_record_provenance(
            "strongest_zero", item.task_sequence_ordinal, item.sample, item.operator,
            item.chunk, item.partition, item.requested_cycle_first,
            item.source_task_provenance_sha256))
    rejected("schedule_axis", lambda: compile_records(module, wrong_axis), "axis")
    bad_prov = records.copy(); bad_prov[0] = replace(bad_prov[0],
        schedule_record_provenance_sha256="0" * 64)
    rejected("schedule_provenance", lambda: compile_records(module, bad_prov), "provenance")
    regressed = records.copy(); item = regressed[3]
    regressed[3] = replace(item, requested_cycle_first=4,
        schedule_record_provenance_sha256=module.schedule_record_provenance(
            item.axis, item.task_sequence_ordinal, item.sample, item.operator,
            item.chunk, item.partition, 4, item.source_task_provenance_sha256))
    rejected("schedule_cycle_regression", lambda: compile_records(module, regressed), "regression")
    rejected("schedule_missing", lambda: compile_records(module, records[:-1]), "partial")

    original_serializer = module.canonical_event_bytes
    def digest_must_match():
        authority = compile_records(module, records)[2]
        require(authority["expected_digest_by_axis"] == GOLDEN, "golden digest mismatch")
    with patch.object(module, "canonical_event_bytes",
                      lambda event, seq, cycle, stall:
                          original_serializer(event, seq, cycle, stall) + b"\x00"):
        rejected("serializer_field_drift", digest_must_match, "golden")
    with patch.object(module, "canonical_event_bytes",
                      lambda event, seq, cycle, stall:
                          original_serializer(event, seq, cycle, stall)[::-1]):
        rejected("serializer_order_drift", digest_must_match, "golden")
    def little_endian_tail(event, seq, cycle, stall):
        payload = original_serializer(event, seq, cycle, stall)
        return payload[:-24] + struct.pack("<QQQ", seq, cycle, stall)
    with patch.object(module, "canonical_event_bytes", little_endian_tail):
        rejected("serializer_endianness_drift", digest_must_match, "golden")

    original_id = module.exact_once_id
    with patch.object(module, "exact_once_id",
                      lambda axis, task, local, beat, transaction:
                          hashlib.sha256((original_id(axis, task, local, beat, transaction) +
                                          "id-drift").encode()).hexdigest()):
        rejected("exact_id_drift", digest_must_match, "golden")
    with patch.object(module, "source_row_provenance", lambda *args: "0" * 64):
        rejected("source_provenance_drift", digest_must_match, "golden")
    with patch.object(module, "canonical_event_bytes",
                      lambda event, seq, cycle, stall:
                          original_serializer(event, seq, cycle + 1, stall + 1)):
        rejected("scheduled_cycle_drift", digest_must_match, "golden")

    rejected("caller_expected_digest_secret", lambda:
             module.IndependentExpectedDigestCompiler(
                 module.BOUNDED_GEOMETRY, expected_digest_by_axis=GOLDEN), "unexpected")

    compiler = module.IndependentExpectedDigestCompiler(module.BOUNDED_GEOMETRY)
    for record in records:
        compiler.consume_schedule_record(record)
    before = compiler.snapshot()
    rejected("terminal_sink_failure", lambda: compiler.finalize(
        lambda value: (_ for _ in ()).throw(RuntimeError("controlled sink failure"))),
        "controlled")
    require(compiler.snapshot() == before and compiler.snapshot()["finalized"] is False,
            "failed terminal sink committed compiler")
    accepted = []
    compiler.finalize(accepted.append)
    require(len(accepted) == 1, "terminal retry did not commit exactly once")

    atomic = module.IndependentExpectedDigestCompiler(module.BOUNDED_GEOMETRY)
    before = atomic.snapshot(); calls = 0
    def failing_serializer(event, seq, cycle, stall):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("controlled serializer failure")
        return original_serializer(event, seq, cycle, stall)
    with patch.object(module, "canonical_event_bytes", failing_serializer):
        rejected("record_atomic_serializer_failure", lambda:
                 atomic.consume_schedule_record(records[0]), "controlled")
    require(atomic.snapshot() == before, "failed variable-beat record partially committed")


def main() -> None:
    verify_frozen(); module = load_subject(); static = static_checks(module)
    bounded = bounded_positive(module); fail_closed_attacks(module)
    output = {
        "schema": "m1146ca_c1_independent_expected_digest_compiler_author_check_r1_v1",
        "status": "PASS_M1146CA_AUTHOR_BOUNDED_DIGEST_COMPILER__PRODUCTION_STOP",
        "checks": checks, "static": static, "bounded": bounded,
        "attacks_rejected": attacks,
        "authorization": {
            "different_author_hammer_next": True,
            "production_digest_compiler_execution": False,
            "real_driver_full_replay_eda": False,
        },
        "docs359_sha256": EXPECTED["docs359"],
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
