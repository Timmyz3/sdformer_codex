#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1137C author check; bounded live-hook oracle only, no full/EDA/GPU/remote."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import stat
import struct
import sys
from typing import Any, Callable
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/build_m1137c_c1_real_per_task_weight_beat_hook_source.py"
CONTRACT = HW / "contracts/m1137c_c1_real_per_task_weight_beat_hook_source_contract_r1_20260830.json"
M1016 = HW / "system_simulator/scripts/run_m1016_c1_full_matched_address_replay.py"
M1102 = HW / "system_simulator/scripts/run_m1102_c1_work8_exact_1rw_source.py"
M1132 = HW / "system_simulator/scripts/build_m1132c_c1_upstream_weight_event_producer_source.py"
M1135 = HW / "system_simulator/scripts/build_m1135c_c1_oaxes_streaming_weight_validator_sink_source.py"
M1136 = HW / "reviews/m1136c_m1135c_c1_oaxes_streaming_validator_sink_hammer_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
OUTPUT = HERE / "mechanical_checks.json"
ORACLE = HERE / "bounded_hook_oracle.json"

EXPECTED = {
    "source": "9ec640ae8c9fa75f9cbf706e15d2d26a4233def77e5be4d67e94c084347b20a6",
    "contract": "51e9370e43830ba10075c994d73da665e8b7d559697f54ebb38ad93a13128acc",
    "contract_side": "01c888e5477133d716ad0db499107ff77eb21b2b1e17688784df3a2716e45e61",
    "contract_outer": "865dac0d7bf89f1a57777f5eafbc6b6fef8b8cbc78403c1822ba5191adfc349d",
    "m1016": "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa",
    "m1102": "95bd50aebcc473ab69cdea6ccf27d54743c89926c5e0f31199dc469ced9bf7cc",
    "m1132": "d6b077fc71d7433f194d497834babd530e0939ca1166dab9376546c670bbdc5f",
    "m1135": "4c282b4ece5705b5c8dcd039c29003c14e544ffef5e8c4234afab0ac31ac7571",
    "m1136_review": "35559fdec20ddee27f29ef6f2cf1841c55258f067c8cbc8dbc16b2159548cb81",
    "m1136_manifest": "45056ee2a2e2e79eebfd2b438899c64bef98bece0238ce1c93a8a4ee1a8d74f0",
    "m1136_outer": "fe766b8810c74489f058f0cc38275951e335c9e369ef096e608e3fe82d1a198d",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
AXES = ("candidate", "strongest_zero", "same_coordinate_bit")
EXPECTED_DIGESTS = {
    "candidate": "49facfeb00bb3b388d1ac1145a9a099602f54a625875ed34d14cfa5125edc749",
    "strongest_zero": "47950bf0e7f5187e3516aa9fd87e605e75789972663bb1772522fc298aecad4b",
    "same_coordinate_bit": "605be1f2dfc3443850bf4f2a7bee0f7e8c7fb2d992862d50f5a8c143fd0a63d9",
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


def strict_json(path: Path):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          CheckFailure("nonfinite JSON: " + token)))


def verify_double() -> None:
    side = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    for path, expected in zip((CONTRACT, side, outer),
                              (EXPECTED["contract"], EXPECTED["contract_side"],
                               EXPECTED["contract_outer"])):
        verify_regular(path, expected)
    require(side.read_text(encoding="utf-8").split() ==
            [EXPECTED["contract"], CONTRACT.name] and
            outer.read_text(encoding="utf-8").split() ==
            [EXPECTED["contract_side"], side.name], "contract double seal")


def verify_m1136() -> dict[str, Any]:
    review = M1136 / "review.json"; manifest = M1136 / "SHA256SUMS"
    outer = M1136 / "SHA256SUMS.seal.sha256"
    verify_regular(review, EXPECTED["m1136_review"])
    verify_regular(manifest, EXPECTED["m1136_manifest"])
    verify_regular(outer, EXPECTED["m1136_outer"])
    require(outer.read_text(encoding="utf-8").split() ==
            [EXPECTED["m1136_manifest"], "SHA256SUMS"], "M1136 outer content")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*"); rel = Path(name)
        require(name not in expected and name == rel.as_posix() and
                not rel.is_absolute() and ".." not in rel.parts,
                "M1136 safe manifest")
        expected[name] = digest
    actual = set()
    for member in M1136.rglob("*"):
        name = member.relative_to(M1136).as_posix()
        if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}: continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "M1136 symlink")
        if stat.S_ISREG(mode): actual.add(name)
        else: require(stat.S_ISDIR(mode), "M1136 special member")
    require(actual == set(expected), "M1136 exact member set")
    for name, digest in expected.items(): verify_regular(M1136 / name, digest)
    return strict_json(review)


def load_subject():
    spec = importlib.util.spec_from_file_location("m1137c_author_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "subject module spec")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def function_node(tree: ast.Module, name: str):
    value = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == name), None)
    require(value is not None, "missing function: " + name)
    return value


def method_node(subject: ast.ClassDef, name: str):
    value = next((node for node in subject.body if isinstance(node, ast.FunctionDef)
                  and node.name == name), None)
    require(value is not None, "missing method: " + name)
    return value


def static_checks(module) -> dict[str, Any]:
    text = SOURCE.read_text(encoding="utf-8"); tree = ast.parse(text)
    contract = strict_json(CONTRACT)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and
               node.name == "M1016SuccessorPerTaskWeightBeatHook"]
    require(len(classes) == 1, "one successor hook class")
    subject = classes[0]
    forbidden_history = []
    for node in ast.walk(subject):
        if isinstance(node, (ast.Set, ast.SetComp)):
            forbidden_history.append(type(node).__name__)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "set":
                forbidden_history.append("set()")
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                    "append", "extend", "add"}:
                forbidden_history.append(node.func.attr)
    require(forbidden_history == [], "production class retains history primitive")
    forbidden_tokens = (
        "common" + "_receipt",
        "weight" + "_beat_first",
        "instrument" + "_real_event_inputs",
        "schedule" + "_native_one_rw",
        "PerBeat" + "AddressedWeightRefillProducer",
    )
    require(not any(token in text for token in forbidden_tokens),
            "post-hoc/batch/O(N) adapter entered source")
    production = method_node(subject, "stream_production_task")
    production_text = ast.unparse(production)
    require("m1016.task_index" in production_text and
            "task_id * PRODUCTION_EVENTS_PER_AXIS" in production_text and
            "(task_id + 1) * PRODUCTION_EVENTS_PER_AXIS" in production_text and
            "_stream_task_interval" in production_text,
            "direct production task beat interval creation drift")
    loop = method_node(subject, "_stream_task_interval")
    calls = [node for node in ast.walk(loop) if isinstance(node, ast.Call) and
             isinstance(node.func, ast.Attribute) and
             node.func.attr == "InternalWeightServiceRefillEvent"]
    require(len(calls) == 1 and len(calls[0].args) == 17 and not calls[0].keywords,
            "exact 17-field event created in loop")
    loop_text = ast.unparse(loop)
    require("event.validate()" in loop_text and "self._validator(event)" in loop_text and
            loop_text.index("event.validate()") < loop_text.index("self._validator(event)") and
            loop_text.index("self._validator(event)") < loop_text.index("state.emitted += 1"),
            "creation/validation/sink/commit order drift")
    require(contract["creation_point_contract"]["all_17_fields_created_before_sink"] is True and
            len(contract["creation_point_contract"]["fields"]) == 17 and
            contract["production_geometry"]["events_per_axis"] == 70853184 and
            contract["production_geometry"]["tasks"] == 812160 and
            contract["production_geometry"]["posthoc_receipt_expansion"] is False,
            "contract creation/scale drift")
    return {"exact_event_fields_at_creation": 17,
            "posthoc_or_batch_adapter_tokens": 0,
            "production_class_history_primitives": forbidden_history,
            "direct_task_interval_loop": True,
            "validation_sink_commit_order": True}


def reference_exact_id(axis: str, task: int, local: int, beat: int, transaction: int) -> str:
    return hashlib.sha256(
        f"m1130c:{axis}:{task}:{local}:{beat}:{transaction}".encode()).hexdigest()


def u64(value: int) -> bytes:
    return struct.pack(">Q", value)


def reference_provenance(module, event) -> str:
    task = event.task_id; local = event.source_local_ordinal
    global_beat = task * 2 + local
    half = task & 1; row = global_beat % 16
    slices = tuple(range(((global_beat // 16) % 3) * 8,
                         ((global_beat // 16) % 3) * 8 + 8))
    payload = b"".join((
        b"M1137C_REAL_BEAT\x00\x01", bytes.fromhex(EXPECTED["m1016"]),
        bytes.fromhex(EXPECTED["m1102"]), bytes.fromhex(EXPECTED["m1135"]),
        struct.pack(">B", AXES.index(event.axis)), u64(0), u64(0), u64(0),
        u64(task), u64(task), u64(local), u64(global_beat),
        u64(event.requested_cycle), struct.pack(">B", half),
        struct.pack(">B", row), struct.pack(">B", len(slices)), bytes(slices),
    ))
    return hashlib.sha256(payload).hexdigest()


class RowSink:
    def __init__(self, fail_at: int | None = None):
        self.calls = 0
        self.accepted = 0
        self.fail_at = fail_at
        self.digest = hashlib.sha256()

    def __call__(self, row):
        self.calls += 1
        if self.fail_at == self.calls:
            raise RuntimeError("controlled row sink failure")
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
        }, sort_keys=True, separators=(",", ":")).encode())


def bounded_live_event_check(module) -> dict[str, Any]:
    m1135 = module.load_m1135()
    sink = RowSink(); inspected = {axis: 0 for axis in AXES}
    original = m1135.OAxesStreamingWeightValidatorSink.__call__

    def inspect_and_call(self, event):
        event.validate()
        expected_beat = inspected[event.axis]
        require(event.service_beat_ordinal == expected_beat and
                event.store_transaction_ordinal == expected_beat,
                "live event global ordinal")
        require(event.service_event_exact_once_id == reference_exact_id(
                    event.axis, event.task_id, event.source_local_ordinal,
                    expected_beat, expected_beat), "live exact ID independent match")
        require(event.source_row_provenance_sha256 == reference_provenance(module, event),
                "live provenance independent match")
        require(len([getattr(event, name) for name in module.EVENT_FIELDS]) == 17,
                "live 17 fields present")
        inspected[event.axis] += 1
        return original(self, event)

    with patch.object(m1135.OAxesStreamingWeightValidatorSink, "__call__", inspect_and_call):
        hook = module.M1016SuccessorPerTaskWeightBeatHook(module.bounded_authority(), sink)
        initial_keys = tuple(sorted(hook.__dict__))
        for axis in AXES:
            require(hook.stream_bounded_task(
                axis=axis, task_id=0, requested_cycle_first=5) == 2,
                "bounded task0 emits two")
            require(hook.stream_bounded_task(
                axis=axis, task_id=1, requested_cycle_first=6) == 2,
                "bounded task1 emits two")
        snapshot = hook.snapshot(); terminal = hook.finalize()
    require(inspected == {axis: 4 for axis in AXES} and sink.accepted == 12 and
            sink.digest.hexdigest() ==
                "1c4a870df979adec71b3b10fc725f3ea84e7bc174b0e907e2088717f5641a063",
            "bounded live event/row conservation")
    require({axis: terminal["m1135c_terminal"]["axes"][axis]["digest"]
             for axis in AXES} == EXPECTED_DIGESTS, "bounded terminal digests")
    require(tuple(sorted(hook.__dict__)) == initial_keys and
            len(hook._cursor) == 3 and
            all(len(hook._validator._next_free_cycle[axis]) == 24 for axis in AXES),
            "O(axes + axes*24) state cardinality")
    return {"tasks_per_axis": 2, "events_per_axis": 4, "total_events": 12,
            "live_events_inspected": inspected, "row_sink_digest": sink.digest.hexdigest(),
            "terminal_digests": EXPECTED_DIGESTS,
            "successor_axis_states": len(hook._cursor),
            "validator_next_free_values": 72,
            "retained_rows_events_or_key_history": False,
            "canonical_rows": 0, "canonical_events": 0}


def fail_closed_and_atomicity(module) -> None:
    m1135 = module.load_m1135()
    production = m1135.ExpectedDigestAuthority(
        "production", "1" * 64,
        {axis: 70853184 for axis in AXES}, {axis: "0" * 64 for axis in AXES})
    rejected("production_authority_absent", lambda:
             module.M1016SuccessorPerTaskWeightBeatHook(production, RowSink()),
             "sealed production digest authority is absent")
    rejected("wrong_authority_type", lambda:
             module.M1016SuccessorPerTaskWeightBeatHook(object(), RowSink()))
    rejected("noncallable_row_sink", lambda:
             module.M1016SuccessorPerTaskWeightBeatHook(module.bounded_authority(), None))

    failing = RowSink(fail_at=1)
    hook = module.M1016SuccessorPerTaskWeightBeatHook(module.bounded_authority(), failing)
    before = hook.snapshot()
    rejected("first_beat_sink_failure", lambda: hook.stream_bounded_task(
        axis="candidate", task_id=0, requested_cycle_first=5), "controlled row sink failure")
    require(hook.snapshot() == before, "first failed beat commits zero complete state")
    replacement = RowSink(); hook._validator._sink = replacement
    require(hook.stream_bounded_task(axis="candidate", task_id=0,
                                     requested_cycle_first=5) == 2 and
            replacement.accepted == 2, "failed first beat retries exact task")

    middle_sink = RowSink(fail_at=2)
    middle = module.M1016SuccessorPerTaskWeightBeatHook(
        module.bounded_authority(), middle_sink)
    rejected("middle_beat_sink_failure", lambda: middle.stream_bounded_task(
        axis="candidate", task_id=0, requested_cycle_first=5),
        "controlled row sink failure")
    paused = middle.snapshot()
    require(paused["successor"]["candidate"]["emitted"] == 1 and
            paused["successor"]["candidate"]["next_global_beat"] == 1 and
            paused["validator"]["candidate"]["event_count"] == 1,
            "middle failure retains only prior committed beat and resume cursor")
    resumed_sink = RowSink(); middle._validator._sink = resumed_sink
    require(middle.stream_bounded_task(axis="candidate", task_id=0,
                                       requested_cycle_first=5) == 1 and
            resumed_sink.accepted == 1 and
            middle.snapshot()["successor"]["candidate"]["next_task_id"] == 1,
            "middle failure resumes failed beat without replay")
    rejected("wrong_next_task", lambda: middle.stream_bounded_task(
        axis="candidate", task_id=0, requested_cycle_first=5), "task id")
    rejected("task_gap", lambda:
        module.M1016SuccessorPerTaskWeightBeatHook(module.bounded_authority(), RowSink())
        .stream_bounded_task(axis="candidate", task_id=1, requested_cycle_first=5),
        "task id")
    rejected("negative_cycle", lambda:
        module.M1016SuccessorPerTaskWeightBeatHook(module.bounded_authority(), RowSink())
        .stream_bounded_task(axis="candidate", task_id=0, requested_cycle_first=-1))
    rejected("canonical_without_authority_driver", lambda:
             next(module.iter_canonical_real_per_task_weight_beats()), "STOP")


def main() -> None:
    frozen_paths = (SOURCE, CONTRACT, M1016, M1102, M1132, M1135, DOCS359)
    before = {path: sha(path) for path in frozen_paths}
    for path, key in ((SOURCE, "source"), (M1016, "m1016"), (M1102, "m1102"),
                      (M1132, "m1132"), (M1135, "m1135"), (DOCS359, "docs359")):
        verify_regular(path, EXPECTED[key])
    verify_double(); m1136 = verify_m1136()
    require(m1136["status"] ==
            "PASS_M1136C_M1135C_O_AXES_STREAMING_HAMMER__AUTHOR_ADDITIVE_REAL_PRODUCER_HOOK_SOURCE_ONLY",
            "M1136 GO status")
    module = load_subject(); preflight = module.source_preflight()
    require(preflight["canonical_rows"] == preflight["canonical_events"] == 0 and
            preflight["production_expected_digest_authority_integrated"] is False and
            preflight["real_production_driver_integrated"] is False,
            "source fail-closed preflight")
    static = static_checks(module)
    bounded = bounded_live_event_check(module)
    fail_closed_and_atomicity(module)
    oracle = module.source_small_oracle()
    require(oracle["status"] ==
            "PASS_BOUNDED_2_TASK_3_AXIS_REAL_CREATION_HOOK__CANONICAL_STOP" and
            oracle["row_sink_count"] == 12 and oracle["canonical_rows"] == 0 and
            oracle["canonical_events"] == 0, "source bounded oracle")
    require({path: sha(path) for path in frozen_paths} == before,
            "source/contract/frozen authorities modified")
    result = {
        "schema": "m1137c_author_static_bounded_hook_check_v1",
        "status": "PASS_M1137C_REAL_PER_TASK_WEIGHT_BEAT_HOOK_AUTHOR_TESTS__DIFFERENT_AUTHOR_HAMMER_REQUIRED",
        "checks": checks, "attacks_rejected": len(attacks), "attacks": attacks,
        "preflight": preflight, "static": static, "bounded": bounded,
        "production_expected_digest_authority": False,
        "real_production_driver": False, "full_replay": False,
        "eda_gpu_remote": False, "canonical_rows": 0, "canonical_events": 0,
        "source_sha256": EXPECTED["source"],
        "contract_identity": [EXPECTED["contract"], EXPECTED["contract_side"],
                              EXPECTED["contract_outer"]],
        "docs359_sha256": EXPECTED["docs359"],
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n", encoding="utf-8")
    ORACLE.write_text(json.dumps(oracle, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
