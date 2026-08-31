#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1135C author static/mutation check; bounded synthetic only."""
from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import stat
import struct
import sys
from typing import Any, Callable

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/build_m1135c_c1_oaxes_streaming_weight_validator_sink_source.py"
CONTRACT = HW / "contracts/m1135c_c1_oaxes_streaming_weight_validator_sink_source_contract_r1_20260830.json"
M1130 = HW / "system_simulator/scripts/build_m1130c_c1_internal_weight_service_refill_instrumentation_source.py"
M1132 = HW / "system_simulator/scripts/build_m1132c_c1_upstream_weight_event_producer_source.py"
M1134 = HW / "reviews/m1134c_m1132c_production_scale_first_principles_audit_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
OUTPUT = HERE / "mechanical_checks.json"
ORACLE = HERE / "small_synthetic_oracle.json"

SOURCE_SHA = "4c282b4ece5705b5c8dcd039c29003c14e544ffef5e8c4234afab0ac31ac7571"
CONTRACT_ID = (
    "6d6fcdcd414e020c6aa456d4e162a63e85d4f70cd37d849abbe292bc7ce9c41f",
    "8532cbf2f9d69852593536d1900ab95a225d2484bd46de8f82c050e34cd5a67b",
    "310608b91bc36f48fd7a82024ef84e2843f802878cf9ed6e11ee799823bda0d6",
)
M1130_SHA = "ce157e7b4b8b9507ba71948fd4b7fcef4145fb24e3252097b5e50b68cf519eaf"
M1132_SHA = "d6b077fc71d7433f194d497834babd530e0939ca1166dab9376546c670bbdc5f"
M1134_OUTER = "8522bc2b5b271a1b9e55a420ac4d82c221c8455175910a803919243de9ffdf11"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
AXES = ("candidate", "strongest_zero", "same_coordinate_bit")
EXPECTED_DIGESTS = {
    "candidate": "f4e5a19127c3310ecfe1b538c9f1cc295a5a8b6f83488e28fbfcb44acae891c7",
    "strongest_zero": "f502037e7b6de7dd55105f5db435e6ab60312962a8ebbe7b864e6c3b3c06e8a3",
    "same_coordinate_bit": "4c05281142fa41fe9c2bf98024862658bebe5480c03ffc4de6bca68bd662e435",
}


class Reject(RuntimeError):
    pass


checks = 0
attacks: dict[str, str] = {}


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise Reject(message)


def rejected(label: str, action: Callable[[], Any], contains: str | None = None) -> None:
    try:
        action()
    except Exception as error:
        if contains is not None:
            require(contains in str(error), label + " wrong rejection: " + str(error))
        attacks[label] = type(error).__name__ + ": " + str(error)
        return
    raise Reject("attack accepted: " + label)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, expected: str) -> None:
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink() and
            sha(path) == expected, "identity drift: " + str(path))


def double_seal() -> None:
    side = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    for path, expected in zip((CONTRACT, side, outer), CONTRACT_ID):
        regular(path, expected)
    require(side.read_text(encoding="utf-8").split() ==
            [CONTRACT_ID[0], CONTRACT.name] and
            outer.read_text(encoding="utf-8").split() ==
            [CONTRACT_ID[1], side.name], "contract double seal content")


def flat_outer(directory: Path, expected: str) -> None:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    regular(outer, expected)
    require(outer.read_text(encoding="utf-8").split() ==
            [sha(manifest), "SHA256SUMS"], "M1134 outer content")


def strict_pairs(rows):
    value = {}
    for key, item in rows:
        require(key not in value, "duplicate JSON key")
        value[key] = item
    return value


def strict_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=strict_pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Reject("nonfinite JSON: " + token)))


def load_subject():
    spec = importlib.util.spec_from_file_location("m1135c_author_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "subject module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def authority(module, count: int = 2, digests=None, scope="bounded_synthetic"):
    if digests is None:
        digests = {axis: "0" * 64 for axis in AXES}
    return module.ExpectedDigestAuthority(
        scope, hashlib.sha256(b"m1135c-author-test-authority").hexdigest(),
        {axis: count for axis in AXES}, digests)


def event(module, axis="candidate", ordinal=0, cycle=5, task=0, source=None,
          **changes):
    if source is None:
        source = ordinal
    m1130 = module.load_m1130()
    values = {
        "axis": axis, "task_id": task, "source_local_ordinal": source,
        "requested_cycle": cycle, "op": "WRITE", "logical_bank": 0,
        "half_slot": 0, "logical_row": AXES.index(axis),
        "local_row": AXES.index(axis), "native_slices": tuple(range(8)),
        "bytes": 128, "byte_enable_per_slice": (0xffff,) * 8,
        "native_macro_activations": 8, "service_beat_ordinal": ordinal,
        "store_transaction_ordinal": ordinal,
        "source_row_provenance_sha256": hashlib.sha256(
            f"m1135c-author:{axis}:{ordinal}".encode()).hexdigest(),
    }
    values.update(changes)
    values.setdefault("service_event_exact_once_id", module.recompute_exact_once_id(
        values["axis"], values["task_id"], values["source_local_ordinal"],
        values["service_beat_ordinal"], values["store_transaction_ordinal"]))
    return m1130.InternalWeightServiceRefillEvent(**values)


def u64(value: int) -> bytes:
    return struct.pack(">Q", value)


def reference_bytes(item, sequence: int, scheduled: int, stalls: int) -> bytes:
    axis_code = AXES.index(item.axis)
    parts = [
        b"M1135C\x00\x01", struct.pack(">B", axis_code), u64(item.task_id),
        u64(item.source_local_ordinal), u64(item.requested_cycle), b"W",
        struct.pack(">B", item.logical_bank), struct.pack(">B", item.half_slot),
        struct.pack(">B", item.logical_row), struct.pack(">B", item.local_row),
        struct.pack(">B", len(item.native_slices)), bytes(item.native_slices),
        u64(item.bytes), struct.pack(">B", len(item.byte_enable_per_slice)),
        b"".join(struct.pack(">H", value) for value in item.byte_enable_per_slice),
        u64(item.native_macro_activations), u64(item.service_beat_ordinal),
        u64(item.store_transaction_ordinal),
        bytes.fromhex(item.service_event_exact_once_id),
        bytes.fromhex(item.source_row_provenance_sha256),
        u64(sequence), u64(scheduled), u64(stalls),
    ]
    return b"".join(parts)


class CountingSink:
    def __init__(self):
        self.calls = 0
        self.last = None

    def __call__(self, row):
        self.calls += 1
        self.last = row


def static_checks(module) -> dict[str, Any]:
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and
               node.name == "OAxesStreamingWeightValidatorSink"]
    require(len(classes) == 1, "one streaming validator class")
    subject = classes[0]
    forbidden_calls = []
    for node in ast.walk(subject):
        if isinstance(node, (ast.Set, ast.SetComp)):
            forbidden_calls.append(type(node).__name__)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "set":
                forbidden_calls.append("set()")
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                    "append", "extend", "add"}:
                forbidden_calls.append(node.func.attr)
    require(forbidden_calls == [], "event/key history primitive in production class")
    require("instrument_real_event_inputs" not in ast.get_source_segment(text, subject) and
            "schedule_native_one_rw" not in ast.get_source_segment(text, subject) and
            "validate_exact_once_and_conflicts" not in ast.get_source_segment(text, subject),
            "batch helper entered production class")
    contract = strict_json(CONTRACT)
    require(contract["frozen_production_scale"] == {
                "axes": 3, "weight_events_per_axis": 70853184,
                "total_weight_events": 212559552}, "production scale contract")
    require(len(contract["input_contract"]["event_fields"]) == 17 and
            tuple(contract["input_contract"]["event_fields"]) == module.EVENT_FIELDS,
            "exact 17-field contract")
    return {
        "production_class_forbidden_history_primitives": forbidden_calls,
        "production_batch_helpers": 0,
        "production_events_per_axis": 70853184,
        "event_fields": 17,
    }


def authority_attacks(module) -> None:
    good_counts = {axis: 2 for axis in AXES}
    good_digests = {axis: "0" * 64 for axis in AXES}
    make = module.ExpectedDigestAuthority
    rejected("authority_bad_scope", lambda: make(
        "other", "0" * 64, good_counts, good_digests).validate())
    rejected("authority_bad_id", lambda: make(
        "bounded_synthetic", "0" * 63, good_counts, good_digests).validate())
    rejected("authority_axis_order", lambda: make(
        "bounded_synthetic", "0" * 64,
        {axis: 2 for axis in reversed(AXES)}, good_digests).validate())
    rejected("authority_bad_digest", lambda: make(
        "bounded_synthetic", "0" * 64, good_counts,
        {**good_digests, "candidate": "g" * 64}).validate())
    rejected("bounded_authority_over_64", lambda: authority(module, 65).validate())
    rejected("production_count_not_exact", lambda: authority(
        module, 70853183, scope="production").validate())
    production = authority(module, 70853184, scope="production")
    production.validate()
    require(all(production.expected_count_by_axis[axis] == 70853184
                for axis in AXES), "production count authority accepted exactly")


def streaming_attacks(module) -> None:
    # Exact type and validation occur before the sink.
    observed = CountingSink()
    validator = module.OAxesStreamingWeightValidatorSink(authority(module), observed)
    rejected("exact_type", lambda: validator(object()), "exact M1130C event type")
    invalid = replace(event(module), bytes=127)
    rejected("17_field_validation_before_sink", lambda: validator(invalid))
    require(observed.calls == 0, "invalid event never enters sink")

    for label, changes in (
        ("bad_mapping", {"logical_bank": 1}),
        ("bad_native_slice", {"native_slices": (0, 1, 2, 3, 4, 5, 6, 6)}),
        ("non_write", {"op": "READ"}),
        ("bad_provenance", {"source_row_provenance_sha256": "g" * 64}),
    ):
        rejected(label, lambda changes=changes:
                 module.OAxesStreamingWeightValidatorSink(authority(module), CountingSink())(
                     replace(event(module), **changes)))

    def after_first(second):
        value = module.OAxesStreamingWeightValidatorSink(authority(module), CountingSink())
        value(event(module, ordinal=0))
        value(second)

    rejected("beat_gap", lambda: after_first(event(module, ordinal=2,
             store_transaction_ordinal=1)), "service beat")
    rejected("transaction_gap", lambda: after_first(event(module, ordinal=1,
             store_transaction_ordinal=2)), "transaction")
    rejected("scheduler_key_regression", lambda: after_first(
        event(module, ordinal=1, cycle=4)), "scheduler key")
    bad_id = replace(event(module), service_event_exact_once_id="0" * 64)
    rejected("exact_id_mismatch", lambda:
             module.OAxesStreamingWeightValidatorSink(authority(module), CountingSink())(bad_id))

    one = authority(module, 1)
    value = module.OAxesStreamingWeightValidatorSink(one, CountingSink())
    value(event(module, ordinal=0))
    rejected("event_overflow", lambda: value(event(module, ordinal=1)),
             "event count exceeds")
    rejected("early_finalize", lambda:
             module.OAxesStreamingWeightValidatorSink(authority(module), CountingSink()).finalize(),
             "terminal per-axis")

    wrong = module.OAxesStreamingWeightValidatorSink(authority(module), CountingSink())
    for item in module.iter_bounded_synthetic_events():
        wrong(item)
    rejected("wrong_terminal_digest", wrong.finalize, "digest mismatch")


def sink_atomicity(module) -> None:
    class SinkError(Exception):
        pass

    calls = {"count": 0}
    def bad_sink(_row):
        calls["count"] += 1
        raise SinkError("controlled sink failure")

    value = module.OAxesStreamingWeightValidatorSink(authority(module), bad_sink)
    before = value.snapshot()
    first = event(module)
    rejected("sink_exception_propagates", lambda: value(first), "controlled sink failure")
    require(calls["count"] == 1 and value.snapshot() == before,
            "sink failure commits zero validator/scheduler/digest state")
    good = CountingSink()
    value._sink = good
    value(first)
    require(good.calls == 1 and value.snapshot()["candidate"]["event_count"] == 1,
            "same event retry succeeds after zero-commit failure")


def bounded_and_state(module) -> dict[str, Any]:
    oracle = module.source_small_oracle()
    require(oracle["status"] ==
            "PASS_BOUNDED_O_AXES_STREAMING__CANONICAL_ZERO_STOP" and
            oracle["bounded_rows"] == 6 and oracle["canonical_rows"] == 0 and
            oracle["canonical_events"] == 0, "bounded source oracle")
    require({axis: oracle["terminal"]["axes"][axis]["digest"] for axis in AXES} ==
            EXPECTED_DIGESTS, "bounded frozen digests")

    # Independent serializer and scheduler reconstruction.
    digests = {axis: hashlib.sha256() for axis in AXES}
    next_free = {axis: [0] * 24 for axis in AXES}
    sequence = {axis: 0 for axis in AXES}
    for item in module.iter_bounded_synthetic_events():
        axis = item.axis
        scheduled = max([item.requested_cycle] +
                        [next_free[axis][part] for part in item.native_slices])
        stalls = scheduled - item.requested_cycle
        digests[axis].update(reference_bytes(item, sequence[axis], scheduled, stalls))
        sequence[axis] += 1
        for part in item.native_slices:
            next_free[axis][part] = scheduled + 1
    require({axis: digests[axis].hexdigest() for axis in AXES} == EXPECTED_DIGESTS,
            "independent serialization/digest agrees")

    sink = CountingSink()
    validator = module.OAxesStreamingWeightValidatorSink(
        authority(module, 2, EXPECTED_DIGESTS), sink)
    initial_shape = (len(validator._state),
                     tuple(len(value) for value in validator._next_free_cycle.values()),
                     tuple(sorted(validator.__dict__)))
    for item in module.iter_bounded_synthetic_events():
        validator(item)
    final_shape = (len(validator._state),
                   tuple(len(value) for value in validator._next_free_cycle.values()),
                   tuple(sorted(validator.__dict__)))
    terminal = validator.finalize()
    require(initial_shape == final_shape and initial_shape[:2] == (3, (24, 24, 24)),
            "runtime state cardinality fixed at axes plus axes*24")
    require(sink.calls == 6 and terminal["state_complexity"] ==
            "O(axes + axes*24)", "one sink call/event and O(axes) terminal")
    rejected("post_finalize_event", lambda: validator(event(module)),
             "already finalized")
    rejected("post_finalize_finalize", validator.finalize, "already finalized")
    rejected("canonical_iterator_without_authority", lambda:
             next(module.iter_canonical_oaxes_streaming_weight_events()), "STOP")
    return {
        "bounded_events": 6, "sink_calls": sink.calls,
        "independent_digests": {axis: digests[axis].hexdigest() for axis in AXES},
        "initial_state_shape": initial_shape,
        "final_state_shape": final_shape,
        "event_or_key_history_entries": 0,
        "canonical_rows": 0, "canonical_events": 0,
    }


def main() -> None:
    before = {path.as_posix(): sha(path) for path in
              (SOURCE, CONTRACT, M1130, M1132, DOCS359)}
    regular(SOURCE, SOURCE_SHA); regular(M1130, M1130_SHA)
    regular(M1132, M1132_SHA); regular(DOCS359, DOCS359_SHA)
    double_seal(); flat_outer(M1134, M1134_OUTER)
    module = load_subject()
    preflight = module.source_preflight()
    require(preflight["canonical_rows"] == preflight["canonical_events"] == 0 and
            not preflight["real_hook_integrated"] and
            not preflight["production_expected_digest_authority_integrated"],
            "fail-closed preflight")
    static = static_checks(module)
    authority_attacks(module)
    streaming_attacks(module)
    sink_atomicity(module)
    bounded = bounded_and_state(module)
    after = {path.as_posix(): sha(path) for path in
             (SOURCE, CONTRACT, M1130, M1132, DOCS359)}
    require(before == after, "frozen source/contract/authorities changed")
    result = {
        "schema": "m1135c_author_static_mutation_check_v1",
        "status": "PASS_SOURCE_AND_BOUNDED_O_AXES_STREAMING__DIFFERENT_AUTHOR_HAMMER_REQUIRED",
        "checks": checks, "attacks": attacks, "attack_count": len(attacks),
        "static": static, "bounded": bounded,
        "production_state_complexity": "O(axes + axes*24)",
        "production_event_or_key_history": False,
        "real_hook": False, "full_replay": False, "eda_gpu_remote": False,
        "canonical_rows": 0, "canonical_events": 0,
        "source_sha256": SOURCE_SHA, "contract_identity": list(CONTRACT_ID),
        "docs359_sha256": DOCS359_SHA,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n", encoding="utf-8")
    ORACLE.write_text(json.dumps(module.source_small_oracle(), indent=2,
                                 sort_keys=True, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
