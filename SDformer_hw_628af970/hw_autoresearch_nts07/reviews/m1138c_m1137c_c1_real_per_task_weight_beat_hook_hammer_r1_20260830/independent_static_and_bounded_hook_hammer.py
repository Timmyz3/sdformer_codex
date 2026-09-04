#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1138C different-author static/bounded hook hammer; no production/full/EDA."""
from __future__ import annotations

import ast
from dataclasses import fields
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import stat
import struct
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HW / "system_simulator/scripts/build_m1137c_c1_real_per_task_weight_beat_hook_source.py"
SOURCE_SHA = "9ec640ae8c9fa75f9cbf706e15d2d26a4233def77e5be4d67e94c084347b20a6"
CONTRACT = HW / "contracts/m1137c_c1_real_per_task_weight_beat_hook_source_contract_r1_20260830.json"
CONTRACT_ID = (
    "51e9370e43830ba10075c994d73da665e8b7d559697f54ebb38ad93a13128acc",
    "01c888e5477133d716ad0db499107ff77eb21b2b1e17688784df3a2716e45e61",
    "865dac0d7bf89f1a57777f5eafbc6b6fef8b8cbc78403c1822ba5191adfc349d",
)
AUTHOR = HW / "reviews/m1137c_c1_real_per_task_weight_beat_hook_author_receipt_r1_20260830"
AUTHOR_ID = (
    "fbdf785267f3052a3c15edb94c882a174cf6ddc5c87be62d2a36151ab6303e71",
    "5deaa8e187f0969cc7d4fdfefa3cffa09710d90267a6fc19b84f21c20ccb7ecf",
    "c3accde9f308c800b5d211c5f6e76eeb3e39912b9115536080801bac3c0fb81e",
)
M1016 = HW / "system_simulator/scripts/run_m1016_c1_full_matched_address_replay.py"
M1016_SHA = "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa"
M1102 = HW / "system_simulator/scripts/run_m1102_c1_work8_exact_1rw_source.py"
M1102_SHA = "95bd50aebcc473ab69cdea6ccf27d54743c89926c5e0f31199dc469ced9bf7cc"
M1132 = HW / "system_simulator/scripts/build_m1132c_c1_upstream_weight_event_producer_source.py"
M1132_SHA = "d6b077fc71d7433f194d497834babd530e0939ca1166dab9376546c670bbdc5f"
M1135 = HW / "system_simulator/scripts/build_m1135c_c1_oaxes_streaming_weight_validator_sink_source.py"
M1135_SHA = "4c282b4ece5705b5c8dcd039c29003c14e544ffef5e8c4234afab0ac31ac7571"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
AXES = ("candidate", "strongest_zero", "same_coordinate_bit")
EVENT_FIELDS = (
    "axis", "task_id", "source_local_ordinal", "requested_cycle", "op",
    "logical_bank", "half_slot", "logical_row", "local_row", "native_slices",
    "bytes", "byte_enable_per_slice", "native_macro_activations",
    "service_beat_ordinal", "store_transaction_ordinal",
    "service_event_exact_once_id", "source_row_provenance_sha256",
)
EXPECTED_DIGESTS = {
    "candidate": "49facfeb00bb3b388d1ac1145a9a099602f54a625875ed34d14cfa5125edc749",
    "strongest_zero": "47950bf0e7f5187e3516aa9fd87e605e75789972663bb1772522fc298aecad4b",
    "same_coordinate_bit": "605be1f2dfc3443850bf4f2a7bee0f7e8c7fb2d992862d50f5a8c143fd0a63d9",
}
EXPECTED_ROW_DIGEST = "1c4a870df979adec71b3b10fc725f3ea84e7bc174b0e907e2088717f5641a063"
checks = 0
attacks: dict[str, str] = {}


def require(value: bool, message: str) -> None:
    global checks
    if not value:
        raise RuntimeError(message)
    checks += 1


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, expected: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and sha(path) == expected,
            "regular identity drift: " + str(path))


def strict_json(path: Path) -> dict:
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite: " + token)))


def double_seal() -> None:
    side = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    regular(CONTRACT, CONTRACT_ID[0]); regular(side, CONTRACT_ID[1]); regular(outer, CONTRACT_ID[2])
    require(side.read_text(encoding="utf-8").split() == [CONTRACT_ID[0], CONTRACT.name],
            "contract side content")
    require(outer.read_text(encoding="utf-8").split() == [CONTRACT_ID[1], side.name],
            "contract outer content")


def exact_flat(directory: Path, identity: tuple[str, str, str]) -> dict:
    review = directory / "review.json"; manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    regular(review, identity[0]); regular(manifest, identity[1]); regular(outer, identity[2])
    require(directory.is_dir() and not directory.is_symlink() and
            outer.read_text(encoding="utf-8").split() == [identity[1], "SHA256SUMS"],
            "author outer content")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1); name = name.lstrip("*"); relative = Path(name)
        require(name not in expected and relative.as_posix() == name and
                not relative.is_absolute() and ".." not in relative.parts,
                "author manifest member")
        expected[name] = digest
    actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "sealed author symlink")
        if stat.S_ISREG(mode):
            actual.add(name)
        else:
            require(stat.S_ISDIR(mode), "sealed author special member")
    require(actual == set(expected), "author exact member census")
    for name, digest in expected.items():
        regular(directory / name, digest)
    return strict_json(review)


def load_subject():
    spec = importlib.util.spec_from_file_location("m1138c_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "subject import spec")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rejected(label: str, action, contains: str | None = None) -> None:
    try:
        action()
    except Exception as error:
        if contains is not None:
            require(contains in str(error), label + " wrong rejection: " + str(error))
        attacks[label] = type(error).__name__ + ": " + str(error)
        return
    raise RuntimeError(label + " did not fail closed")


def class_and_loop_policy(text: str) -> list[str]:
    tree = ast.parse(text)
    subject = next(node for node in tree.body if isinstance(node, ast.ClassDef) and
                   node.name == "M1016SuccessorPerTaskWeightBeatHook")
    violations = []
    assigned_self = set()
    for node in ast.walk(subject):
        if isinstance(node, (ast.Set, ast.SetComp)):
            violations.append("history_set")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {"set", "list"}:
                violations.append("history_container")
            if isinstance(node.func, ast.Attribute) and node.func.attr in {"add", "append", "extend"}:
                violations.append("history_mutator")
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and
                        target.value.id == "self"):
                    assigned_self.add(target.attr)
    if assigned_self != {"_authority_scope", "_validator", "_cursor"}:
        violations.append("unexpected_hook_state")
    forbidden = (
        "common_service_receipt", "weight_beat_first", "total_weight_count",
        "task_weight_count", "instrument_real_event_inputs", "schedule_native_one_rw",
        "validate_exact_once_and_conflicts", "PerBeatAddressedWeightRefillProducer",
    )
    if any(token in text for token in forbidden):
        violations.append("posthoc_or_batch_adapter")
    interval = next(node for node in subject.body if isinstance(node, ast.FunctionDef) and
                    node.name == "_stream_task_interval")
    loops = [node for node in ast.walk(interval) if isinstance(node, ast.While)]
    if len(loops) != 1:
        violations.append("live_beat_loop_census")
        return violations
    loop = loops[0]
    constructors = [node for node in ast.walk(loop) if isinstance(node, ast.Call) and
                    isinstance(node.func, ast.Attribute) and
                    node.func.attr == "InternalWeightServiceRefillEvent"]
    if len(constructors) != 1 or len(constructors[0].args) != 17 or constructors[0].keywords:
        violations.append("17_field_live_constructor")
    loop_text = ast.get_source_segment(text, loop) or ast.unparse(loop)
    ordered = ("successor_exact_once_id", "beat_provenance",
               "InternalWeightServiceRefillEvent", "event.validate()",
               "self._validator(event)", "current += 1", "state.emitted += 1")
    try:
        positions = [loop_text.index(token) for token in ordered]
        if positions != sorted(positions):
            violations.append("create_validate_immediate_sink_commit_order")
    except ValueError:
        violations.append("create_validate_immediate_sink_commit_order")
    before_sink = loop_text[:loop_text.find("self._validator(event)")]
    if ("state.emitted +=" in before_sink or "state.next_task_id +=" in before_sink or
            "state.active_signature = signature" in before_sink or
            "state.next_global_beat = current" in before_sink):
        violations.append("successor_commit_before_sink")
    iterator = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and
                    node.name == "iter_canonical_real_per_task_weight_beats")
    iterator_text = ast.get_source_segment(text, iterator) or ast.unparse(iterator)
    if ("if False" not in iterator_text or
            any(token in iterator_text for token in ("open(", "read_", "yield from"))):
        violations.append("canonical_open_or_real_driver")
    return violations


def enforce_policy(text: str) -> None:
    violations = class_and_loop_policy(text)
    if violations:
        raise RuntimeError("hook policy violation: " + ",".join(sorted(set(violations))))


def static_hammer(module) -> dict[str, Any]:
    text = SOURCE.read_text(encoding="utf-8")
    violations = class_and_loop_policy(text)
    require(violations == [], "subject hook policy violation: " + repr(violations))
    contract = strict_json(CONTRACT)
    require(tuple(contract["creation_point_contract"]["fields"]) == EVENT_FIELDS and
            contract["creation_point_contract"]["all_17_fields_created_before_sink"] is True and
            contract["production_geometry"]["posthoc_receipt_expansion"] is False and
            contract["streaming_state"]["complexity"] == "O(axes + axes*24)" and
            contract["production_fail_closed"]["expected_digest_authority_available"] is False and
            contract["production_fail_closed"]["real_production_driver_integrated"] is False and
            contract["production_fail_closed"]["canonical_rows"] == 0 and
            contract["production_fail_closed"]["canonical_events"] == 0 and
            contract["authorization"]["full_replay_now"] is False and
            contract["authorization"]["eda_rtl_gpu_remote_now"] is False,
            "contract hook/boundary drift")
    m1130 = module.load_m1135().load_m1130()
    require(tuple(item.name for item in fields(m1130.InternalWeightServiceRefillEvent)) ==
            EVENT_FIELDS == module.EVENT_FIELDS, "runtime exact 17-field schema")
    cursor_fields = tuple(item.name for item in fields(module._TaskCursor))
    require(cursor_fields == ("next_task_id", "active_signature", "next_global_beat", "emitted"),
            "fixed per-axis cursor schema")
    return {"policy_violations": violations, "live_loop_event_fields": 17,
            "exact_id_before_event": True, "provenance_before_event": True,
            "validate_then_immediate_m1135c_sink_then_commit": True,
            "hook_self_state_fields": 3, "cursor_fields_per_axis": 4,
            "posthoc_or_batch_adapter_tokens": 0,
            "canonical_iterator_closed_stub": True}


def source_mutation_attacks() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    mutations = {
        "posthoc_aggregate_adapter_mutation": text.replace(
            "        self._authority_scope = authority.scope\n",
            "        self._authority_scope = authority.scope\n"
            "        common_service_receipt = []\n", 1),
        "first_count_adapter_mutation": text.replace(
            "        self._authority_scope = authority.scope\n",
            "        self._authority_scope = authority.scope\n"
            "        weight_beat_first = total_weight_count = 0\n", 1),
        "m1132_set_producer_mutation": text.replace(
            "        self._cursor = {axis: _TaskCursor() for axis in AXES}\n",
            "        self._cursor = {axis: _TaskCursor() for axis in AXES}\n"
            "        self._seen = set()  # PerBeatAddressedWeightRefillProducer\n", 1),
        "m1130_batch_mutation": text.replace(
            "            self._validator(event)\n",
            "            instrument_real_event_inputs([event])\n"
            "            self._validator(event)\n", 1),
        "delayed_sink_mutation": text.replace(
            "            self._validator(event)\n            current += 1\n",
            "            current += 1\n            self._validator(event)\n", 1),
        "canonical_open_mutation": text.replace(
            "    if False:  # pragma: no cover\n        yield None\n",
            "    with open('canonical.jsonl', 'rb') as stream:\n"
            "        yield from stream\n", 1),
    }
    require(all(value != text for value in mutations.values()), "mutation insertion anchors")
    for label, mutated in mutations.items():
        rejected(label, lambda mutated=mutated: enforce_policy(mutated),
                 "hook policy violation")


def exact_id(axis: str, task: int, local: int, beat: int, transaction: int) -> str:
    return hashlib.sha256(
        f"m1130c:{axis}:{task}:{local}:{beat}:{transaction}".encode()).hexdigest()


def u64(value: int) -> bytes:
    return struct.pack(">Q", value)


def provenance(item: Any) -> str:
    task = item.task_id; local = item.source_local_ordinal
    global_beat = task * 2 + local
    slices = tuple(range(((global_beat // 16) % 3) * 8,
                         ((global_beat // 16) % 3) * 8 + 8))
    payload = b"".join((
        b"M1137C_REAL_BEAT\x00\x01", bytes.fromhex(M1016_SHA),
        bytes.fromhex(M1102_SHA), bytes.fromhex(M1135_SHA),
        struct.pack(">B", AXES.index(item.axis)), u64(0), u64(0), u64(0),
        u64(task), u64(task), u64(local), u64(global_beat),
        u64(item.requested_cycle), struct.pack(">B", task & 1),
        struct.pack(">B", global_beat % 16), struct.pack(">B", len(slices)),
        bytes(slices),
    ))
    return hashlib.sha256(payload).hexdigest()


def m1135_digest_bytes(item: Any, sequence: int, cycle: int, stalls: int) -> bytes:
    return b"".join([
        b"M1135C\x00\x01", struct.pack(">B", AXES.index(item.axis)),
        u64(item.task_id), u64(item.source_local_ordinal), u64(item.requested_cycle), b"W",
        struct.pack(">B", item.logical_bank), struct.pack(">B", item.half_slot),
        struct.pack(">B", item.logical_row), struct.pack(">B", item.local_row),
        struct.pack(">B", len(item.native_slices)), bytes(item.native_slices),
        u64(item.bytes), struct.pack(">B", len(item.byte_enable_per_slice)),
        b"".join(struct.pack(">H", value) for value in item.byte_enable_per_slice),
        u64(item.native_macro_activations), u64(item.service_beat_ordinal),
        u64(item.store_transaction_ordinal), bytes.fromhex(item.service_event_exact_once_id),
        bytes.fromhex(item.source_row_provenance_sha256),
        u64(sequence), u64(cycle), u64(stalls),
    ])


class IndependentOnlineObserver:
    def __init__(self):
        self.count = {axis: 0 for axis in AXES}
        self.next_free = {axis: [0] * 24 for axis in AXES}
        self.digest = {axis: hashlib.sha256() for axis in AXES}

    def before(self, item: Any) -> tuple[int, int]:
        axis = item.axis; sequence = self.count[axis]
        require(tuple(getattr(item, name) for name in EVENT_FIELDS) and
                len(EVENT_FIELDS) == 17, "live 17 fields")
        require(item.service_beat_ordinal == sequence and
                item.store_transaction_ordinal == sequence,
                "live beat/transaction ordinal")
        require(item.service_event_exact_once_id == exact_id(
            axis, item.task_id, item.source_local_ordinal, sequence, sequence),
            "live independent exact ID")
        require(item.source_row_provenance_sha256 == provenance(item),
                "live independent provenance")
        cycle = max([item.requested_cycle] +
                    [self.next_free[axis][part] for part in item.native_slices])
        return cycle, cycle - item.requested_cycle

    def commit(self, item: Any, cycle: int, stalls: int) -> None:
        axis = item.axis; sequence = self.count[axis]
        self.digest[axis].update(m1135_digest_bytes(item, sequence, cycle, stalls))
        self.count[axis] += 1
        for part in item.native_slices:
            self.next_free[axis][part] = cycle + 1


class RowSink:
    def __init__(self, fail_at: int | None = None):
        self.calls = 0; self.accepted = 0; self.fail_at = fail_at
        self.digest = hashlib.sha256()

    def __call__(self, row: Any) -> None:
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("controlled sink exception")
        row.validate(); self.accepted += 1
        self.digest.update(json.dumps({
            "axis": row.axis, "requested_cycle": row.requested_cycle,
            "cycle": row.cycle, "stall_cycles": row.stall_cycles,
            "logical_bank": row.logical_bank, "logical_row": row.logical_row,
            "native_slices": list(row.native_slices), "bytes": row.bytes,
            "service_beat_ordinal": row.service_beat_ordinal,
            "store_transaction_ordinal": row.store_transaction_ordinal,
            "task_id": row.source_task_id,
            "source_local_ordinal": row.source_local_ordinal,
            "source_row_provenance_sha256": row.source_row_provenance_sha256,
        }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def structural_state_bytes(hook) -> int:
    size = sys.getsizeof(hook._cursor) + sys.getsizeof(hook._validator._state)
    size += sys.getsizeof(hook._validator._next_free_cycle)
    for axis in AXES:
        cursor = hook._cursor[axis]
        state = hook._validator._state[axis]
        size += sys.getsizeof(cursor) + sys.getsizeof(vars(cursor))
        if cursor.active_signature is not None:
            size += sys.getsizeof(cursor.active_signature)
        size += sys.getsizeof(state) + sys.getsizeof(vars(state))
        size += sys.getsizeof(state.digest)
        if state.last_scheduler_key is not None:
            size += sys.getsizeof(state.last_scheduler_key)
        size += sys.getsizeof(hook._validator._next_free_cycle[axis])
    return size


def bounded_reproduction(module) -> dict[str, Any]:
    m1135 = module.load_m1135(); original = m1135.OAxesStreamingWeightValidatorSink.__call__
    observer = IndependentOnlineObserver(); sink = RowSink()

    def inspect_then_call(instance, item):
        cycle, stalls = observer.before(item)
        row = original(instance, item)
        require((row.cycle, row.stall_cycles) == (cycle, stalls),
                "live independent scheduler")
        observer.commit(item, cycle, stalls)
        return row

    m1135.OAxesStreamingWeightValidatorSink.__call__ = inspect_then_call
    try:
        hook = module.M1016SuccessorPerTaskWeightBeatHook(module.bounded_authority(), sink)
        initial_keys = tuple(sorted(hook.__dict__))
        initial_bytes = structural_state_bytes(hook)
        for axis in AXES:
            require(hook.stream_bounded_task(axis=axis, task_id=0,
                                             requested_cycle_first=5) == 2,
                    "task0 emits two live beats")
            require(hook.stream_bounded_task(axis=axis, task_id=1,
                                             requested_cycle_first=6) == 2,
                    "task1 emits two live beats")
        final_bytes = structural_state_bytes(hook)
        terminal = hook.finalize()
    finally:
        m1135.OAxesStreamingWeightValidatorSink.__call__ = original
    one_task_hook = module.M1016SuccessorPerTaskWeightBeatHook(
        module.bounded_authority(), RowSink())
    for axis in AXES:
        one_task_hook.stream_bounded_task(axis=axis, task_id=0,
                                          requested_cycle_first=5)
    one_task_bytes = structural_state_bytes(one_task_hook)
    digests = {axis: observer.digest[axis].hexdigest() for axis in AXES}
    require(observer.count == {axis: 4 for axis in AXES} and sink.accepted == 12,
            "2task x 2beat x 3axis conservation")
    require(digests == EXPECTED_DIGESTS and
            {axis: terminal["m1135c_terminal"]["axes"][axis]["digest"]
             for axis in AXES} == EXPECTED_DIGESTS,
            "independent and M1135 digest match")
    require(sink.digest.hexdigest() == EXPECTED_ROW_DIGEST, "row sink digest")
    require(tuple(sorted(hook.__dict__)) == initial_keys and len(hook._cursor) == 3 and
            all(len(hook._validator._next_free_cycle[axis]) == 24 for axis in AXES) and
            one_task_bytes == final_bytes,
            "O(axes) retained structural state")
    return {"tasks_per_axis": 2, "beats_per_task": 2,
            "events_per_axis": 4, "total_events": 12,
            "independent_live_event_counts": observer.count,
            "independent_digests": digests,
            "row_sink_digest": sink.digest.hexdigest(),
            "initial_structural_state_bytes": initial_bytes,
            "one_task_per_axis_structural_state_bytes": one_task_bytes,
            "final_structural_state_bytes": final_bytes,
            "successor_axis_states": 3, "validator_next_free_values": 72,
            "retained_rows_events_or_key_history": False}


def clean_task_snapshot(module, axis: str) -> dict:
    hook = module.M1016SuccessorPerTaskWeightBeatHook(module.bounded_authority(), RowSink())
    hook.stream_bounded_task(axis=axis, task_id=0, requested_cycle_first=5)
    return hook.snapshot()


def atomicity(module) -> dict[str, Any]:
    results = {}
    for axis in AXES:
        first_sink = RowSink(fail_at=1)
        first = module.M1016SuccessorPerTaskWeightBeatHook(
            module.bounded_authority(), first_sink)
        before = first.snapshot(); before_bytes = structural_state_bytes(first)
        rejected("first_sink_exception_" + axis, lambda axis=axis:
                 first.stream_bounded_task(axis=axis, task_id=0,
                                           requested_cycle_first=5), "controlled sink")
        require(first.snapshot() == before and structural_state_bytes(first) == before_bytes,
                "first sink exception committed state: " + axis)
        first._validator._sink = RowSink()
        require(first.stream_bounded_task(axis=axis, task_id=0,
                                          requested_cycle_first=5) == 2 and
                first.snapshot() == clean_task_snapshot(module, axis),
                "first sink retry mismatch: " + axis)

        middle_sink = RowSink(fail_at=2)
        middle = module.M1016SuccessorPerTaskWeightBeatHook(
            module.bounded_authority(), middle_sink)
        rejected("middle_sink_exception_" + axis, lambda axis=axis:
                 middle.stream_bounded_task(axis=axis, task_id=0,
                                            requested_cycle_first=5), "controlled sink")
        paused = middle.snapshot()
        require(paused["successor"][axis]["emitted"] == 1 and
                paused["successor"][axis]["next_global_beat"] == 1 and
                paused["validator"][axis]["event_count"] == 1 and
                paused["validator"][axis]["next_beat"] == 1 and
                paused["validator"][axis]["next_transaction"] == 1,
                "middle failed beat was not zero-commit: " + axis)
        middle._validator._sink = RowSink()
        require(middle.stream_bounded_task(axis=axis, task_id=0,
                                           requested_cycle_first=5) == 1 and
                middle.snapshot() == clean_task_snapshot(module, axis),
                "middle sink resume mismatch: " + axis)
        results[axis] = {"first_snapshot_equal": True,
                         "first_retry_matches_clean": True,
                         "middle_prior_commits": 1,
                         "middle_resume_events": 1,
                         "middle_resume_matches_clean": True}
    return results


def production_stop(module) -> None:
    m1135 = module.load_m1135()
    production = m1135.ExpectedDigestAuthority(
        "production", "1" * 64,
        {axis: 70_853_184 for axis in AXES}, {axis: "2" * 64 for axis in AXES})
    rejected("production_authority_stop", lambda:
             module.M1016SuccessorPerTaskWeightBeatHook(production, RowSink()),
             "sealed production digest authority is absent")
    bounded = module.M1016SuccessorPerTaskWeightBeatHook(
        module.bounded_authority(), RowSink())
    rejected("production_driver_stop", lambda: bounded.stream_production_task(
        axis="candidate", sample=0, operator=0, chunk=0, partition=0,
        requested_cycle_first=0), "production task requires")
    rejected("canonical_open_stop", lambda:
             next(module.iter_canonical_real_per_task_weight_beats()), "STOP")


def main() -> None:
    frozen = (SOURCE, CONTRACT, M1016, M1102, M1132, M1135, DOCS359)
    before = {path: sha(path) for path in frozen}
    for path, expected in ((SOURCE, SOURCE_SHA), (M1016, M1016_SHA),
                           (M1102, M1102_SHA), (M1132, M1132_SHA),
                           (M1135, M1135_SHA), (DOCS359, DOCS359_SHA)):
        regular(path, expected)
    double_seal(); author = exact_flat(AUTHOR, AUTHOR_ID)
    require(author["status"] ==
            "PASS_M1137C_REAL_PER_TASK_WEIGHT_BEAT_HOOK_AUTHOR_RECEIPT__DIFFERENT_AUTHOR_HAMMER_REQUIRED" and
            author["authorization"]["different_author_m1138c_static_and_bounded_hook_hammer"] is True,
            "M1137C author authority")
    module = load_subject(); preflight = module.source_preflight()
    require(preflight["production_expected_digest_authority_integrated"] is False and
            preflight["real_production_driver_integrated"] is False and
            preflight["canonical_rows"] == preflight["canonical_events"] == 0,
            "production preflight not STOP")
    static = static_hammer(module); source_mutation_attacks()
    bounded = bounded_reproduction(module)
    exceptions = atomicity(module); production_stop(module)
    oracle = module.source_small_oracle()
    require(oracle["status"] ==
            "PASS_BOUNDED_2_TASK_3_AXIS_REAL_CREATION_HOOK__CANONICAL_STOP" and
            oracle["row_sink_count"] == 12 and oracle["canonical_rows"] == 0 and
            oracle["canonical_events"] == 0, "subject bounded oracle")
    require({path: sha(path) for path in frozen} == before,
            "frozen subject/authorities changed")
    result = {
        "schema": "m1138c_m1137c_independent_static_bounded_hook_hammer_r1_v1",
        "status": "PASS_M1138C_M1137C_REAL_PER_TASK_BEAT_HOOK_HAMMER__AUTHOR_PRODUCTION_EXPECTED_DIGEST_AUTHORITY_CAPTURE_SOURCE_ONLY",
        "checks_passed": checks, "attacks_rejected": len(attacks),
        "attacks": attacks, "static": static, "bounded": bounded,
        "sink_exception_resume": exceptions,
        "identity": {"source_sha256": SOURCE_SHA,
                     "contract_sha256": CONTRACT_ID[0],
                     "contract_outer_seal_file_sha256": CONTRACT_ID[2],
                     "author_receipt_outer_seal_file_sha256": AUTHOR_ID[2],
                     "docs359_sha256": DOCS359_SHA},
        "execution": {"bounded_total_events": 12, "production_authority": False,
                      "real_driver": False, "canonical_open": False,
                      "full_replay": False, "eda": False, "gpu": False,
                      "remote": False},
        "authorization": {
            "production_expected_digest_authority_capture_source_only": True,
            "production_authority_execution": False, "real_driver": False,
            "canonical_open": False, "full_replay": False,
            "eda_gpu_remote": False},
        "claim_boundary": {"source_and_bounded_hook_only": True,
                           "real_h67_transactions": False,
                           "production_scalability_measured": False,
                           "traffic_cycles_energy_speedup": False,
                           "paper_citable_performance": False},
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
