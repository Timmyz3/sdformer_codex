#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1133C different-author static/controlled-synthetic hammer; no real hook/full/EDA."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import re
import stat
import sys
from typing import Any, Callable

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/build_m1132c_c1_upstream_weight_event_producer_source.py"
CONTRACT = HW / "contracts/m1132c_c1_upstream_weight_event_producer_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1132c_c1_upstream_weight_event_producer_author_receipt_r1_20260830"
M1130 = HW / "system_simulator/scripts/build_m1130c_c1_internal_weight_service_refill_instrumentation_source.py"
M1016 = HW / "system_simulator/scripts/run_m1016_c1_full_matched_address_replay.py"
M1102 = HW / "system_simulator/scripts/run_m1102_c1_work8_exact_1rw_source.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
OUTPUT = HERE / "mechanical_checks.json"
ORACLE = HERE / "controlled_synthetic_oracle.json"

SOURCE_SHA = "d6b077fc71d7433f194d497834babd530e0939ca1166dab9376546c670bbdc5f"
CONTRACT_ID = (
    "8218699210c481a5a8d2ddfc7b2fe1091b24ef36b004716dc530d9b193acec91",
    "be85e9a08684691c964c78f0b441a85a43a61c69a3d4014ae608a7c123526b4f",
    "9592d136ea18b86c722fb69af3422ef8106d5d5d628d8badbf1e5b079f8d9f07",
)
AUTHOR_OUTER = "4ee223cd45dca8a677a0796e56ac9a8b4f653ec036db3b79c9ff4bdc265d13ba"
M1130_SHA = "ce157e7b4b8b9507ba71948fd4b7fcef4145fb24e3252097b5e50b68cf519eaf"
M1016_SHA = "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa"
M1102_SHA = "95bd50aebcc473ab69cdea6ccf27d54743c89926c5e0f31199dc469ced9bf7cc"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
FIELDS = (
    "axis", "task_id", "source_local_ordinal", "requested_cycle", "op",
    "logical_bank", "half_slot", "logical_row", "local_row", "native_slices",
    "bytes", "byte_enable_per_slice", "native_macro_activations",
    "service_beat_ordinal", "store_transaction_ordinal",
    "service_event_exact_once_id", "source_row_provenance_sha256",
)
AXES = ("candidate", "strongest_zero", "same_coordinate_bit")


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
            require(contains in str(error), label + " wrong rejection")
        attacks[label] = type(error).__name__ + ": " + str(error)
        return
    raise Reject("attack accepted: " + label)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_pairs(rows):
    value = {}
    for key, item in rows:
        require(key not in value, "duplicate JSON key")
        value[key] = item
    return value


def load_json(path: Path):
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(),
            "direct regular JSON " + path.name)
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Reject("nonfinite JSON " + token)))


def manifest_rows(path: Path) -> dict[str, str]:
    value = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]) is not None,
                "manifest row")
        name = fields[1].lstrip("*"); rel = Path(name)
        require(name not in value and name == rel.as_posix() and not rel.is_absolute()
                and ".." not in rel.parts, "safe manifest member")
        value[name] = fields[0]
    return value


def verify_flat(directory: Path, expected_outer: str):
    require(stat.S_ISDIR(directory.lstat().st_mode) and not directory.is_symlink(),
            "direct sealed directory")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(stat.S_ISREG(manifest.lstat().st_mode) and not manifest.is_symlink() and
            stat.S_ISREG(outer.lstat().st_mode) and not outer.is_symlink() and
            sha(outer) == expected_outer and
            outer.read_text(encoding="utf-8").split() == [sha(manifest), "SHA256SUMS"],
            "author outer")
    expected = manifest_rows(manifest); actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "sealed symlink")
        if stat.S_ISREG(mode):
            actual.add(name)
        else:
            require(stat.S_ISDIR(mode), "sealed special member")
    require(actual == set(expected), "sealed exact member set")
    for name, digest in expected.items():
        member = directory / name
        require(stat.S_ISREG(member.lstat().st_mode) and not member.is_symlink()
                and sha(member) == digest, "sealed member " + name)
    return load_json(directory / "review.json")


def verify_double() -> None:
    side = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    for path, expected in zip((CONTRACT, side, outer), CONTRACT_ID):
        require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
                and sha(path) == expected, "contract identity")
    require(side.read_text(encoding="utf-8").split() ==
            [CONTRACT_ID[0], CONTRACT.name] and
            outer.read_text(encoding="utf-8").split() ==
            [CONTRACT_ID[1], side.name], "contract double seal content")


def load_subject():
    spec = importlib.util.spec_from_file_location("m1133c_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "subject spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def event_args(module, *, axis="candidate", task=0, source=0, cycle=7,
               bank=0, row=0, beat=0, transaction=0) -> dict[str, Any]:
    return {
        "axis": axis, "task_id": task, "source_local_ordinal": source,
        "requested_cycle": cycle, "op": "WRITE", "logical_bank": bank,
        "half_slot": bank, "logical_row": row,
        "local_row": bank * 16 + row, "native_slices": tuple(range(8)),
        "bytes": 128, "byte_enable_per_slice": (0xffff,) * 8,
        "native_macro_activations": 8, "service_beat_ordinal": beat,
        "store_transaction_ordinal": transaction,
        "service_event_exact_once_id": module.load_m1130().exact_once_id(
            axis, task, source, beat, transaction),
        "source_row_provenance_sha256": hashlib.sha256(
            f"m1133c:{axis}:{task}:{source}".encode()).hexdigest(),
    }


def run_controlled(module) -> dict[str, Any]:
    m1130 = module.load_m1130()
    events = []
    producer = module.PerBeatAddressedWeightRefillProducer(events.append)
    per_call_sizes = []
    expected = []
    for axis_index, axis in enumerate(AXES):
        for local in range(2):
            values = event_args(
                module, axis=axis, task=axis_index, source=local, cycle=11,
                bank=0, row=axis_index, beat=axis_index * 10 + local,
                transaction=axis_index * 10 + local)
            before = len(events)
            event = producer.emit_refill_event(**values)
            per_call_sizes.append(len(events) - before)
            expected.append(values)
            require(type(event) is m1130.InternalWeightServiceRefillEvent and
                    event is events[-1], "exact M1130C event object returned/sunk")
            require({name: getattr(event, name) for name in FIELDS} == values,
                    "exact transaction/beat/event field identity")
    require(per_call_sizes == [1] * 6 and producer.emitted == len(events) == 6,
            "one call exactly one event")
    require(len({(event.axis, event.service_event_exact_once_id) for event in events}) == 6 and
            len({(event.axis, event.service_beat_ordinal) for event in events}) == 6 and
            len({(event.axis, event.task_id, event.source_local_ordinal,
                  event.store_transaction_ordinal) for event in events}) == 6,
            "all exact identities unique")
    scheduled = m1130.instrument_real_event_inputs(events)
    stalled = sum(row.stall_cycles > 0 for row in scheduled)
    require(len(scheduled) == 6 and stalled == 3, "three controlled 1RW stalls")
    # Re-validation includes the final zero-conflict gate inside M1130C.
    require(all(row.cycle >= row.requested_cycle and
                row.stall_cycles == row.cycle - row.requested_cycle
                for row in scheduled),
            "scheduled cycles bounded after request")
    return {
        "events": 6, "one_call_deltas": per_call_sizes,
        "unique_exact_ids": 6, "unique_beats": 6, "unique_transactions": 6,
        "stalled_transactions": stalled, "post_schedule_native_1rw_conflicts": 0,
        "axes": list(AXES),
    }


def duplicate_and_bounds(module) -> None:
    base = event_args(module)
    producer = module.PerBeatAddressedWeightRefillProducer(lambda _event: None)
    producer.emit_refill_event(**base)
    rejected("duplicate_exact_once_id", lambda: producer.emit_refill_event(**base),
             "duplicate producer exact-once ID")

    beat_producer = module.PerBeatAddressedWeightRefillProducer(lambda _event: None)
    beat_producer.emit_refill_event(**base)
    same_beat = event_args(module, task=1, source=1, beat=0, transaction=1)
    rejected("duplicate_service_beat", lambda: beat_producer.emit_refill_event(**same_beat),
             "duplicate producer service beat")

    transaction_producer = module.PerBeatAddressedWeightRefillProducer(lambda _event: None)
    transaction_producer.emit_refill_event(**base)
    same_transaction = event_args(module, task=0, source=0, beat=1, transaction=0)
    rejected("duplicate_transaction", lambda: transaction_producer.emit_refill_event(
        **same_transaction), "duplicate producer transaction identity")

    # Numeric ordinals are axis scoped; the same numbers on another axis remain legal.
    axis_producer = module.PerBeatAddressedWeightRefillProducer(lambda _event: None)
    axis_producer.emit_refill_event(**base)
    other_axis = event_args(module, axis="strongest_zero")
    axis_producer.emit_refill_event(**other_axis)
    require(axis_producer.emitted == 2, "axis-scoped identities do not false-collide")

    invalid = (
        ("negative_task", {"task_id": -1}),
        ("negative_source", {"source_local_ordinal": -1}),
        ("negative_cycle", {"requested_cycle": -1}),
        ("bank_upper_bound", {"logical_bank": 2, "half_slot": 2, "local_row": 32}),
        ("bank_half_mismatch", {"logical_bank": 1}),
        ("logical_row_negative", {"logical_row": -1, "local_row": -1}),
        ("logical_row_upper_bound", {"logical_row": 16, "local_row": 16}),
        ("local_row_mismatch", {"local_row": 1}),
        ("slice_negative", {"native_slices": (-1,) + tuple(range(1, 8))}),
        ("slice_upper_bound", {"native_slices": tuple(range(7)) + (24,)}),
        ("slice_duplicate", {"native_slices": (0, 1, 2, 3, 4, 5, 6, 6)}),
        ("slice_noncontiguous", {"native_slices": (0, 1, 2, 3, 4, 5, 6, 8)}),
        ("bytes_mismatch", {"bytes": 127}),
        ("byte_enable_mismatch", {"byte_enable_per_slice": (0xffff,) * 7 + (0xfffe,)}),
        ("activation_mismatch", {"native_macro_activations": 7}),
        ("beat_negative", {"service_beat_ordinal": -1}),
        ("transaction_negative", {"store_transaction_ordinal": -1}),
        ("exact_id_wrong", {"service_event_exact_once_id": "0" * 64}),
        ("provenance_short", {"source_row_provenance_sha256": "0" * 63}),
        ("provenance_nonhex", {"source_row_provenance_sha256": "g" * 64}),
        ("non_write", {"op": "READ"}),
    )
    for label, changes in invalid:
        values = dict(base); values.update(changes)
        rejected(label, lambda values=values:
                 module.PerBeatAddressedWeightRefillProducer(lambda _event: None)
                 .emit_refill_event(**values))


def sink_and_signature(module) -> None:
    base = event_args(module)
    rejected("noncallable_sink", lambda: module.PerBeatAddressedWeightRefillProducer(None),
             "event sink must be callable")
    observed = []
    invalid = dict(base); invalid["bytes"] = 127
    rejected("validation_before_sink", lambda:
             module.PerBeatAddressedWeightRefillProducer(observed.append)
             .emit_refill_event(**invalid))
    require(observed == [], "invalid event never reaches sink")

    class SinkError(Exception):
        pass
    def bad_sink(_event):
        raise SinkError("controlled sink failure")
    producer = module.PerBeatAddressedWeightRefillProducer(bad_sink)
    rejected("sink_exception_propagation",
             lambda: producer.emit_refill_event(**base), "controlled sink failure")
    require(producer.emitted == 0 and producer._exact_ids == set() and
            producer._beats == set() and producer._transactions == set(),
            "sink failure does not admit or poison identities")
    accepted = []
    producer._sink = accepted.append
    producer.emit_refill_event(**base)
    require(producer.emitted == len(accepted) == 1,
            "same event retry is legal only because failed sink committed nothing")

    signature = inspect.signature(
        module.PerBeatAddressedWeightRefillProducer.emit_refill_event)
    parameters = list(signature.parameters.values())
    require(parameters[0].name == "self" and
            [item.name for item in parameters[1:]] == list(FIELDS) and
            all(item.kind is inspect.Parameter.KEYWORD_ONLY for item in parameters[1:]),
            "exact 17 keyword-only signature")
    rejected("positional_fields", lambda:
             module.PerBeatAddressedWeightRefillProducer(lambda _event: None)
             .emit_refill_event(*[base[name] for name in FIELDS]))
    for name in FIELDS:
        missing = dict(base); missing.pop(name)
        rejected("missing_" + name, lambda missing=missing:
                 module.PerBeatAddressedWeightRefillProducer(lambda _event: None)
                 .emit_refill_event(**missing))
    for name in ("count", "weight_beat_first", "start", "beats", "capacity",
                 "aggregate", "geometry"):
        extra = dict(base); extra[name] = 1
        rejected("aggregate_" + name, lambda extra=extra:
                 module.PerBeatAddressedWeightRefillProducer(lambda _event: None)
                 .emit_refill_event(**extra))


def static_gate(text: str, contract: dict, author: dict) -> None:
    tree = ast.parse(text)
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and
                 node.name == "PerBeatAddressedWeightRefillProducer")
    emit = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and
                node.name == "emit_refill_event")
    iterator = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and
                    node.name == "iter_canonical_upstream_weight_refill_events")
    emit_text = ast.get_source_segment(text, emit) or ast.unparse(emit)
    iterator_text = ast.get_source_segment(text, iterator) or ast.unparse(iterator)
    require([arg.arg for arg in emit.args.kwonlyargs] == list(FIELDS) and
            emit.args.args[1:] == [] and emit.args.vararg is None and
            emit.args.kwarg is None, "AST exact 17 keyword-only fields")
    require(emit_text.count("InternalWeightServiceRefillEvent(") == 1 and
            emit_text.count("self._sink(event)") == 1 and
            emit_text.index("event.validate()") < emit_text.index("self._sink(event)") <
            emit_text.index("self._exact_ids.add(exact_key)"),
            "AST one construction/one sink/commit ordering")
    emit_identifiers = {node.id for node in ast.walk(emit) if isinstance(node, ast.Name)}
    require(not (emit_identifiers & {
                "count", "weight_beat_first", "start", "beats", "capacity",
                "aggregate", "geometry"}) and "range(bytes)" not in emit_text,
            "AST no aggregate fallback identifiers")
    require('audit["canonical_ready"] is True' in iterator_text and
            iterator_text.index('audit["canonical_ready"] is True') <
            iterator_text.index("yield None"), "canonical STOP before yield")
    require(contract["producer_supplied_event_fields"] == list(FIELDS) and
            contract["emission_contract"]["one_call_one_event"] is True and
            contract["emission_contract"]["aggregate_or_geometry_inference"] is False and
            contract["canonical_fail_closed"]["canonical_rows"] == 0 and
            contract["canonical_fail_closed"]["canonical_events"] == 0 and
            contract["authorization"]["integrate_canonical_hook_now"] is False,
            "contract exact interface/STOP")
    require(author["status"] ==
            "PASS_M1132C_ADDITIVE_UPSTREAM_WEIGHT_EVENT_PRODUCER_AUTHOR_RECEIPT__DIFFERENT_AUTHOR_HAMMER_REQUIRED" and
            author["identity"]["source_sha256"] == SOURCE_SHA and
            author["identity"]["contract_outer_seal_file_sha256"] == CONTRACT_ID[2] and
            author["canonical_boundary"]["canonical_rows"] == 0 and
            author["canonical_boundary"]["canonical_events"] == 0,
            "author receipt exact identity/STOP")


def main() -> int:
    before = {path: sha(path) for path in
              (SOURCE, CONTRACT, M1130, M1016, M1102, DOCS359)}
    require(before == {
        SOURCE: SOURCE_SHA, CONTRACT: CONTRACT_ID[0], M1130: M1130_SHA,
        M1016: M1016_SHA, M1102: M1102_SHA, DOCS359: DOCS359_SHA,
    }, "all frozen primary identities")
    verify_double()
    author = verify_flat(AUTHOR, AUTHOR_OUTER)
    contract = load_json(CONTRACT)
    text = SOURCE.read_text(encoding="utf-8")
    static_gate(text, contract, author)
    module = load_subject()
    preflight = module.source_preflight()
    require(preflight == {
        "status": "STOP_CANONICAL_HOOK_NOT_INTEGRATED__ADDITIVE_PRODUCER_SOURCE_ONLY",
        "producer_source_exists": True, "real_callsite_integrated": False,
        "canonical_ready": False, "canonical_rows": 0, "canonical_events": 0,
    }, "live preflight canonical zero STOP")
    synthetic = run_controlled(module)
    duplicate_and_bounds(module)
    sink_and_signature(module)
    rejected("canonical_iterator_before_hook",
             lambda: next(module.iter_canonical_upstream_weight_refill_events()),
             "STOP: additive producer exists")
    require(before == {path: sha(path) for path in before},
            "M1132/M1130/M1016/M1102/docs359 unchanged")

    oracle = {
        "schema": "m1133c_controlled_upstream_weight_event_producer_oracle_r1_v1",
        "status": "PASS_M1133C_CONTROLLED_SYNTHETIC__CANONICAL_ZERO_STOP",
        "synthetic": synthetic,
        "canonical": {"ready": False, "rows": 0, "events": 0, "verdict": "STOP"},
        "full_51840000": False, "real_hook_integrated": False,
        "eda": False, "gpu": False, "remote": False,
    }
    result = {
        "schema": "m1133c_m1132c_upstream_weight_event_producer_hammer_mechanical_r1_v1",
        "status": "PASS_M1133C_M1132C_DIFFERENT_AUTHOR_STATIC_SYNTHETIC_HAMMER__AUTHOR_REAL_HOOK_SOURCE_ONLY",
        "checks_passed": checks, "attacks_rejected": len(attacks),
        "attack_results": attacks, "synthetic": synthetic,
        "identity": {
            "source_sha256": sha(SOURCE), "contract_sha256": sha(CONTRACT),
            "contract_outer_seal_file_sha256": sha(Path(str(CONTRACT) + ".sha256.seal.sha256")),
            "author_receipt_outer_seal_file_sha256": sha(AUTHOR / "SHA256SUMS.seal.sha256"),
            "m1130c_source_sha256": sha(M1130), "m1016_source_sha256": sha(M1016),
            "m1102_source_sha256": sha(M1102), "docs359_sha256": sha(DOCS359),
        },
        "canonical": {"real_hook_integrated": False, "rows": 0, "events": 0,
                      "aggregate_fallback": False, "verdict": "STOP"},
        "execution": {"full_51840000": False, "eda": False, "gpu": False,
                      "remote": False, "real_hook": False, "subject_modified": False},
        "authorization": {"additive_real_producer_hook_source_only_next": True,
                          "integrate_or_execute_hook_now": False},
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    ORACLE.write_text(json.dumps(oracle, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": checks,
                      "attacks": len(attacks)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
