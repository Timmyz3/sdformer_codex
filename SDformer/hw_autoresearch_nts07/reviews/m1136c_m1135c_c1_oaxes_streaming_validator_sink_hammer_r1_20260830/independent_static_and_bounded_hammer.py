#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author M1136C hammer; bounded synthetic only, no real hook/full/EDA."""
from __future__ import annotations

import ast
from dataclasses import fields
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import struct
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HW / "system_simulator/scripts/build_m1135c_c1_oaxes_streaming_weight_validator_sink_source.py"
SOURCE_SHA = "4c282b4ece5705b5c8dcd039c29003c14e544ffef5e8c4234afab0ac31ac7571"
CONTRACT = HW / "contracts/m1135c_c1_oaxes_streaming_weight_validator_sink_source_contract_r1_20260830.json"
CONTRACT_ID = (
    "6d6fcdcd414e020c6aa456d4e162a63e85d4f70cd37d849abbe292bc7ce9c41f",
    "8532cbf2f9d69852593536d1900ab95a225d2484bd46de8f82c050e34cd5a67b",
    "310608b91bc36f48fd7a82024ef84e2843f802878cf9ed6e11ee799823bda0d6",
)
AUTHOR = HW / "reviews/m1135c_c1_oaxes_streaming_weight_validator_sink_author_receipt_r1_20260830"
AUTHOR_ID = (
    "3fff01a5c2c9599dfd0e80cc3b7e3c36d1756307083817f93375be744037355e",
    "07c743ac204d0cb1cfc5246edd78d139ab6469e0e1303319eb074b93bf15b8cc",
    "8226da4cee7f019f79e83c5d2351615c2503a49c8a75f32f8ce671bb99d2f045",
)
M1130 = HW / "system_simulator/scripts/build_m1130c_c1_internal_weight_service_refill_instrumentation_source.py"
M1130_SHA = "ce157e7b4b8b9507ba71948fd4b7fcef4145fb24e3252097b5e50b68cf519eaf"
M1132 = HW / "system_simulator/scripts/build_m1132c_c1_upstream_weight_event_producer_source.py"
M1132_SHA = "d6b077fc71d7433f194d497834babd530e0939ca1166dab9376546c670bbdc5f"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
AXES = ("candidate", "strongest_zero", "same_coordinate_bit")
FIELDS = (
    "axis", "task_id", "source_local_ordinal", "requested_cycle", "op",
    "logical_bank", "half_slot", "logical_row", "local_row", "native_slices",
    "bytes", "byte_enable_per_slice", "native_macro_activations",
    "service_beat_ordinal", "store_transaction_ordinal",
    "service_event_exact_once_id", "source_row_provenance_sha256",
)
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
            require(key not in value, "duplicate key: " + key)
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


def exact_flat(directory: Path, identity: tuple[str, str, str]) -> None:
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    regular(review, identity[0]); regular(manifest, identity[1]); regular(outer, identity[2])
    require(directory.is_dir() and not directory.is_symlink() and
            outer.read_text(encoding="utf-8").split() == [identity[1], "SHA256SUMS"],
            "author outer content")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        name = name.lstrip("*")
        require(name not in expected and not Path(name).is_absolute() and
                ".." not in Path(name).parts, "manifest member")
        expected[name] = digest
    actual = set()
    for path in directory.rglob("*"):
        relative = path.relative_to(directory).as_posix()
        if relative in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        require(not path.is_symlink(), "sealed symlink")
        if path.is_file():
            actual.add(relative)
    require(actual == set(expected), "author exact member census")
    for name, digest in expected.items():
        regular(directory / name, digest)


def load_subject():
    spec = importlib.util.spec_from_file_location("m1136c_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "subject import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def exact_id(axis: str, task: int, ordinal: int,
             beat: int | None = None, transaction: int | None = None) -> str:
    beat_value = ordinal if beat is None else beat
    transaction_value = ordinal if transaction is None else transaction
    return hashlib.sha256(
        f"m1130c:{axis}:{task}:{ordinal}:{beat_value}:{transaction_value}".encode()).hexdigest()


def event(module, axis: str, ordinal: int, requested: int | None = None):
    axis_index = AXES.index(axis)
    requested_cycle = ordinal // 4 if requested is None else requested
    provenance = hashlib.sha256(f"m1136c:{axis}:{ordinal}".encode()).hexdigest()
    return module.load_m1130().InternalWeightServiceRefillEvent(
        axis, axis_index, ordinal, requested_cycle, "WRITE", 0, 0,
        axis_index, axis_index, tuple(range(8)), 128, (0xffff,) * 8, 8,
        ordinal, ordinal, exact_id(axis, axis_index, ordinal), provenance)


def u64(value: int) -> bytes:
    require(type(value) is int and 0 <= value < (1 << 64), "reference u64")
    return struct.pack(">Q", value)


def reference_bytes(item: Any, sequence: int, scheduled: int, stalls: int) -> bytes:
    axis_code = AXES.index(item.axis)
    return b"".join([
        b"M1135C\x00\x01", struct.pack(">B", axis_code),
        u64(item.task_id), u64(item.source_local_ordinal), u64(item.requested_cycle), b"W",
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
    ])


def independent_authority(module, count: int):
    digests = {axis: hashlib.sha256() for axis in AXES}
    next_free = {axis: [0] * 24 for axis in AXES}
    expected_schedule = {}
    for ordinal in range(count):
        for axis in AXES:
            item = event(module, axis, ordinal)
            scheduled = max([item.requested_cycle] +
                            [next_free[axis][part] for part in item.native_slices])
            stalls = scheduled - item.requested_cycle
            digests[axis].update(reference_bytes(item, ordinal, scheduled, stalls))
            expected_schedule[(axis, ordinal)] = (scheduled, stalls)
            for part in item.native_slices:
                next_free[axis][part] = scheduled + 1
    expected = {axis: digests[axis].hexdigest() for axis in AXES}
    authority = module.ExpectedDigestAuthority(
        "bounded_synthetic", hashlib.sha256(
            f"m1136c-authority:{count}".encode()).hexdigest(),
        {axis: count for axis in AXES}, expected)
    return authority, expected_schedule, expected


class CountSink:
    def __init__(self):
        self.calls = 0

    def __call__(self, _row):
        self.calls += 1


def deep_size(value: Any, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        return size + sum(deep_size(key, seen) + deep_size(item, seen)
                          for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return size + sum(deep_size(item, seen) for item in value)
    if hasattr(value, "__dict__") and not callable(value):
        return size + deep_size(vars(value), seen)
    return size


def state_footprint(validator) -> int:
    return deep_size((validator._state, validator._next_free_cycle,
                      validator._finalized))


def structural_state_bytes(validator) -> int:
    """Measure retained container/state-object capacity, excluding scalar values.

    Python scalar aliasing differs for zero versus later ordinals, so recursive
    object bytes are diagnostic only.  Container capacity is the correct test
    for retained event history: it must be identical after 1 and 64 events.
    """
    size = sys.getsizeof(validator._state) + sys.getsizeof(validator._next_free_cycle)
    for axis in AXES:
        state = validator._state[axis]
        size += sys.getsizeof(state) + sys.getsizeof(vars(state))
        size += sys.getsizeof(state.digest)
        if state.last_scheduler_key is not None:
            size += sys.getsizeof(state.last_scheduler_key)
        size += sys.getsizeof(validator._next_free_cycle[axis])
    return size


def rejected(label: str, function, contains: str | None = None) -> None:
    try:
        function()
    except Exception as error:  # the exact fail-closed exception is recorded
        if contains is not None:
            require(contains in str(error), label + " wrong failure")
        attacks[label] = type(error).__name__ + ": " + str(error)
        return
    raise RuntimeError(label + " did not fail closed")


def source_policy_violations(text: str) -> list[str]:
    tree = ast.parse(text)
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and
                 node.name == "OAxesStreamingWeightValidatorSink")
    class_text = ast.get_source_segment(text, klass) or ast.unparse(klass)
    violations = []
    assigned_self = set()
    for node in ast.walk(klass):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (isinstance(target, ast.Attribute) and
                        isinstance(target.value, ast.Name) and target.value.id == "self"):
                    assigned_self.add(target.attr)
        if isinstance(node, (ast.Set, ast.SetComp)):
            violations.append(type(node).__name__)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {"set", "list"}:
                violations.append(node.func.id)
            if isinstance(node.func, ast.Attribute) and node.func.attr in {"add", "append", "extend"}:
                violations.append(node.func.attr)
    expected_self = {"_authority", "_sink", "_state", "_next_free_cycle", "_finalized"}
    if assigned_self != expected_self:
        violations.append("self_state_fields")
    if any(token in class_text for token in
           ("instrument_real_event_inputs", "schedule_native_one_rw",
            "validate_exact_once_and_conflicts", "run_full", "rows.append")):
        violations.append("batch_helper")
    iterator = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and
                    node.name == "iter_canonical_oaxes_streaming_weight_events")
    iterator_text = ast.get_source_segment(text, iterator) or ast.unparse(iterator)
    if ("canonical_ready" not in iterator_text or "if False" not in iterator_text or
            any(token in iterator_text for token in
                ("open(", "read_", "instrument_real_event_inputs", "yield from"))):
        violations.append("canonical_open_or_real_hook")
    return violations


def enforce_source_policy(text: str) -> None:
    violations = source_policy_violations(text)
    if violations:
        raise RuntimeError("source policy violation: " + ",".join(sorted(violations)))


def static_hammer(module) -> dict[str, Any]:
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and
                 node.name == "OAxesStreamingWeightValidatorSink")
    violations = source_policy_violations(text)
    require(violations == [], "source policy violations: " + repr(violations))
    serializer = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and
                      node.name == "canonical_event_bytes")
    referenced = {node.attr for node in ast.walk(serializer)
                  if isinstance(node, ast.Attribute) and
                  isinstance(node.value, ast.Name) and node.value.id == "event"}
    require(referenced == set(FIELDS), "digest serializer does not cover exact 17 fields")
    contract = strict_json(CONTRACT)
    require(tuple(contract["input_contract"]["event_fields"]) == FIELDS and
            contract["state_contract"]["complexity"] == "O(axes + axes*native_slices)" and
            contract["canonical_fail_closed"]["canonical_rows"] == 0 and
            contract["canonical_fail_closed"]["canonical_events"] == 0 and
            contract["authorization"]["full_51840000_replay_now"] is False and
            contract["authorization"]["eda_rtl_gpu_remote_now"] is False,
            "contract boundary drift")
    event_type_fields = tuple(item.name for item in fields(
        module.load_m1130().InternalWeightServiceRefillEvent))
    require(event_type_fields == FIELDS and module.EVENT_FIELDS == FIELDS,
            "17-field runtime type drift")
    return {"policy_violations": violations, "history_primitives": [], "batch_helpers": 0,
            "digest_event_fields_referenced": sorted(referenced),
            "canonical_iterator_closed_stub": True, "runtime_event_fields": 17}


def source_mutation_attacks() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    mutations = {
        "history_set_source_mutation": text.replace(
            "        self._finalized = False\n",
            "        self._finalized = False\n        self._history = set()\n", 1),
        "history_list_source_mutation": text.replace(
            "        self._finalized = False\n",
            "        self._finalized = False\n        self._history = []\n", 1),
        "batch_consumer_source_mutation": text.replace(
            "        require(not self._finalized, \"stream already finalized\")\n",
            "        instrument_real_event_inputs([event])\n"
            "        require(not self._finalized, \"stream already finalized\")\n", 1),
        "real_hook_canonical_open_source_mutation": text.replace(
            "    if False:  # pragma: no cover\n        yield None\n",
            "    with open('canonical.jsonl', 'rb') as stream:\n"
            "        yield from stream\n", 1),
    }
    require(all(value != text for value in mutations.values()), "mutation insertion anchors")
    for label, mutated in mutations.items():
        rejected(label, lambda mutated=mutated: enforce_source_policy(mutated),
                 "source policy violation")


def run_stream(module, count: int):
    authority, schedule, expected_digests = independent_authority(module, count)
    sink = CountSink()
    validator = module.OAxesStreamingWeightValidatorSink(authority, sink)
    initial_shape = (tuple(sorted(validator.__dict__)), len(validator._state),
                     tuple(len(validator._next_free_cycle[axis]) for axis in AXES))
    initial_bytes = state_footprint(validator)
    initial_structural_bytes = structural_state_bytes(validator)
    last = {axis: (-1, -1) for axis in AXES}
    for ordinal in range(count):
        for axis in AXES:
            item = event(module, axis, ordinal)
            require(item.service_event_exact_once_id == exact_id(
                axis, AXES.index(axis), ordinal), "independent exact ID")
            row = validator(item)
            require((row.cycle, row.stall_cycles) == schedule[(axis, ordinal)],
                    "independent 1RW scheduler")
            snap = validator.snapshot()[axis]
            require((snap["last_beat"], snap["last_transaction"]) == (ordinal, ordinal) and
                    snap["next_beat"] == ordinal + 1 and
                    snap["next_transaction"] == ordinal + 1 and
                    (ordinal, ordinal) > last[axis], "three-axis beat/txn monotonicity")
            last[axis] = (ordinal, ordinal)
    final_shape = (tuple(sorted(validator.__dict__)), len(validator._state),
                   tuple(len(validator._next_free_cycle[axis]) for axis in AXES))
    final_bytes = state_footprint(validator)
    final_structural_bytes = structural_state_bytes(validator)
    terminal = validator.finalize()
    require(initial_shape == final_shape == (
        ("_authority", "_finalized", "_next_free_cycle", "_sink", "_state"),
        3, (24, 24, 24)), "validator state shape independent of event count")
    require(sink.calls == count * 3, "one sink call per event")
    require(all(terminal["axes"][axis]["events"] == count and
                terminal["axes"][axis]["first_beat"] == 0 and
                terminal["axes"][axis]["last_beat"] == count - 1 and
                terminal["axes"][axis]["first_transaction"] == 0 and
                terminal["axes"][axis]["last_transaction"] == count - 1 and
                terminal["axes"][axis]["digest"] == expected_digests[axis]
                for axis in AXES), "terminal count/ordinal/digest authority")
    require(terminal["authority_id_sha256"] == authority.authority_id_sha256,
            "terminal authority ID")
    return {"events_per_axis": count, "total_events": count * 3,
            "initial_deep_python_bytes": initial_bytes,
            "final_deep_python_bytes": final_bytes,
            "initial_structural_state_bytes": initial_structural_bytes,
            "final_structural_state_bytes": final_structural_bytes,
            "state_shape": initial_shape, "sink_calls": sink.calls,
            "digests": expected_digests}


def semantic_attacks(module) -> None:
    authority, _, digests = independent_authority(module, 2)
    sink = CountSink()
    value = module.OAxesStreamingWeightValidatorSink(authority, sink)
    rejected("exact_type", lambda: value(object()), "exact M1130C")
    bad = event(module, "candidate", 0)
    bad = type(bad)(*([getattr(bad, name) for name in FIELDS[:10]] + [127] +
                      [getattr(bad, name) for name in FIELDS[11:]]))
    rejected("17_field_validate_before_sink", lambda: value(bad), "schema/mapping")
    require(sink.calls == 0, "invalid event never reaches sink")
    wrong_id = event(module, "candidate", 0)
    wrong_id = type(wrong_id)(*([getattr(wrong_id, name) for name in FIELDS[:15]] +
                                 ["0" * 64, wrong_id.source_row_provenance_sha256]))
    rejected("independent_id_mismatch", lambda: value(wrong_id), "exact-once")
    value(event(module, "candidate", 0, requested=10))
    rejected("beat_gap", lambda: value(event(module, "candidate", 2, requested=10)),
             "service beat")
    transaction_gap = event(module, "candidate", 1, requested=10)
    transaction_gap = type(transaction_gap)(
        *([getattr(transaction_gap, name) for name in FIELDS[:14]] +
          [2, exact_id("candidate", 0, 1, beat=1, transaction=2),
           transaction_gap.source_row_provenance_sha256]))
    rejected("transaction_gap", lambda: value(transaction_gap), "transaction")
    rejected("scheduler_key_regression", lambda: value(
        event(module, "candidate", 1, requested=9)), "scheduler key")

    early = module.OAxesStreamingWeightValidatorSink(authority, CountSink())
    rejected("early_final_count", early.finalize, "count/ordinal")
    wrong_authority = module.ExpectedDigestAuthority(
        "bounded_synthetic", authority.authority_id_sha256,
        {axis: 2 for axis in AXES}, {**digests, "candidate": "0" * 64})
    wrong = module.OAxesStreamingWeightValidatorSink(wrong_authority, CountSink())
    for ordinal in range(2):
        for axis in AXES:
            wrong(event(module, axis, ordinal))
    rejected("wrong_final_digest", wrong.finalize, "digest mismatch")
    production = module.ExpectedDigestAuthority(
        "production", "1" * 64, {axis: 70_853_184 for axis in AXES},
        {axis: "2" * 64 for axis in AXES})
    production.validate()
    require(all(production.expected_count_by_axis[axis] == 70_853_184 for axis in AXES),
            "production final count authority exact")
    rejected("canonical_open", lambda: next(
        module.iter_canonical_oaxes_streaming_weight_events()), "STOP")


def sink_atomicity(module) -> dict[str, Any]:
    class ControlledSinkError(Exception):
        pass

    class ToggleSink:
        def __init__(self):
            self.fail = False
            self.calls = 0

        def __call__(self, _row):
            self.calls += 1
            if self.fail:
                raise ControlledSinkError("controlled sink exception")

    per_axis = {}
    for axis in AXES:
        authority, _, _ = independent_authority(module, 2)
        sink = ToggleSink()
        value = module.OAxesStreamingWeightValidatorSink(authority, sink)
        value(event(module, axis, 0))
        before = value.snapshot()
        sink.fail = True
        rejected("sink_exception_zero_commit_" + axis,
                 lambda axis=axis: value(event(module, axis, 1)), "controlled sink")
        require(value.snapshot() == before, "sink exception changed any state: " + axis)
        sink.fail = False
        retried = value(event(module, axis, 1))
        require(retried.store_transaction_ordinal == 1 and
                value.snapshot()[axis]["event_count"] == 2,
                "same event retry failed: " + axis)
        per_axis[axis] = {"snapshot_equal_after_exception": True,
                          "retry_event_count": 2, "sink_calls": sink.calls}
    return per_axis


def main() -> None:
    before = {path.as_posix(): sha(path) for path in (SOURCE, CONTRACT, M1130, M1132, DOCS359)}
    regular(SOURCE, SOURCE_SHA); regular(M1130, M1130_SHA); regular(M1132, M1132_SHA)
    regular(DOCS359, DOCS359_SHA); double_seal(); exact_flat(AUTHOR, AUTHOR_ID)
    author = strict_json(AUTHOR / "review.json")
    require(author["status"] ==
            "PASS_M1135C_O_AXES_STREAMING_SOURCE_AUTHOR_RECEIPT__DIFFERENT_AUTHOR_M1136C_HAMMER_REQUIRED",
            "M1135C author status")
    module = load_subject()
    preflight = module.source_preflight()
    require(preflight["canonical_rows"] == preflight["canonical_events"] == 0 and
            preflight["canonical_ready"] is False and
            preflight["real_hook_integrated"] is False and
            preflight["production_expected_digest_authority_integrated"] is False,
            "production preflight not fail closed")
    static = static_hammer(module)
    source_mutation_attacks()
    one = run_stream(module, 1)
    sixty_four = run_stream(module, 64)
    require(one["initial_structural_state_bytes"] ==
                sixty_four["initial_structural_state_bytes"] and
            one["final_structural_state_bytes"] ==
                sixty_four["final_structural_state_bytes"],
            "retained structural state depends on stream length")
    semantic_attacks(module)
    atomicity = sink_atomicity(module)
    after = {path.as_posix(): sha(path) for path in (SOURCE, CONTRACT, M1130, M1132, DOCS359)}
    require(before == after, "frozen subject/authority changed")
    result = {
        "schema": "m1136c_m1135c_independent_static_bounded_hammer_r1_v1",
        "status": "PASS_M1136C_M1135C_O_AXES_STREAMING_HAMMER__AUTHOR_ADDITIVE_REAL_PRODUCER_HOOK_SOURCE_ONLY",
        "checks_passed": checks, "attacks_rejected": len(attacks),
        "attacks": attacks, "static": static,
        "state_scaling": {"one": one, "sixty_four": sixty_four,
                          "retained_structural_bytes_equal": True,
                          "deep_python_scalar_bytes_are_diagnostic_only": True,
                          "event_or_key_history_entries": 0},
        "sink_atomicity": atomicity,
        "identity": {"source_sha256": SOURCE_SHA,
                     "contract_sha256": CONTRACT_ID[0],
                     "contract_outer_seal_file_sha256": CONTRACT_ID[2],
                     "author_receipt_outer_seal_file_sha256": AUTHOR_ID[2],
                     "docs359_sha256": DOCS359_SHA},
        "execution": {"bounded_synthetic_events_max_per_axis": 64,
                      "real_hook": False, "canonical_open": False,
                      "full_replay": False, "eda": False, "gpu": False,
                      "remote": False},
        "authorization": {"additive_real_producer_hook_source_only": True,
                          "real_hook_execution": False,
                          "production_digest_authority": False,
                          "canonical_open": False, "full_replay": False,
                          "eda_gpu_remote": False},
        "claim_boundary": {"source_and_bounded_synthetic_only": True,
                           "production_scalability_measured": False,
                           "canonical_weight_ledger": False,
                           "traffic_cycles_energy_speedup": False,
                           "paper_citable_performance": False},
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
