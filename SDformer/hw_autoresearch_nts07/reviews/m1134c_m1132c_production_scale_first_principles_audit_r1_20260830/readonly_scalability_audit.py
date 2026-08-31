#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1134C read-only production-scale audit; bounded microbenchmark only."""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import stat
import struct
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
M1132 = HW / "system_simulator/scripts/build_m1132c_c1_upstream_weight_event_producer_source.py"
M1130 = HW / "system_simulator/scripts/build_m1130c_c1_internal_weight_service_refill_instrumentation_source.py"
M1016 = HW / "system_simulator/scripts/run_m1016_c1_full_matched_address_replay.py"
M1102 = HW / "system_simulator/scripts/run_m1102_c1_work8_exact_1rw_source.py"
M1102_RESULT = HW / "results/m1102_c1_work8_exact_1rw_full_replay_r1_20260830/m1102_c1_work8_exact_1rw_full_replay_result_r1.json"
M1133 = HW / "reviews/m1133c_m1132c_c1_upstream_weight_event_producer_static_hammer_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
OUTPUT = HERE / "scale_metrics.json"

EXPECTED = {
    M1132: "d6b077fc71d7433f194d497834babd530e0939ca1166dab9376546c670bbdc5f",
    M1130: "ce157e7b4b8b9507ba71948fd4b7fcef4145fb24e3252097b5e50b68cf519eaf",
    M1016: "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa",
    M1102: "95bd50aebcc473ab69cdea6ccf27d54743c89926c5e0f31199dc469ced9bf7cc",
    M1102_RESULT: "a229c21b1469f2482ade412a8965e66018db1e4aaa5d434329994a0572587d91",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
M1133_OUTER = "099882fff10e587c9f7a7580297cd06c35f49639f6e4ef845317ed2cc019a057"
AXES = 3
EVENTS_PER_AXIS = 70_853_184
EVENTS = AXES * EVENTS_PER_AXIS
BENCHMARK_EVENTS = 100_000


class Failure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise Failure(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_regular(path: Path, expected: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and sha(path) == expected,
            "frozen identity drift: " + str(path))


def verify_m1133() -> None:
    manifest = M1133 / "SHA256SUMS"
    outer = M1133 / "SHA256SUMS.seal.sha256"
    verify_regular(outer, M1133_OUTER)
    require(outer.read_text(encoding="utf-8").split() ==
            [sha(manifest), "SHA256SUMS"], "M1133 outer content")


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def meminfo() -> dict[str, int]:
    result = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        if key in {"MemAvailable", "CommitLimit", "Committed_AS"}:
            result[key + "_kib"] = int(raw.split()[0])
    require(set(result) == {"MemAvailable_kib", "CommitLimit_kib", "Committed_AS_kib"},
            "meminfo schema")
    result["CommitHeadroom_kib"] = result["CommitLimit_kib"] - result["Committed_AS_kib"]
    require(result["CommitHeadroom_kib"] > 0, "commit already over limit")
    return result


def next_set_capacity(count: int) -> int:
    """Minimum power-of-two slots satisfying CPython set's <=3/5 fill gate."""
    capacity = 8
    while capacity * 3 // 5 <= count:
        capacity *= 2
    return capacity


def object_census(m1130) -> dict[str, int]:
    event = m1130.InternalWeightServiceRefillEvent(
        "candidate", 0, 0, 0, "WRITE", 0, 0, 0, 0, tuple(range(8)), 128,
        (0xffff,) * 8, 8, 0, 0, m1130.exact_once_id("candidate", 0, 0, 0, 0),
        "0" * 64)
    row = m1130.instrument_real_event_inputs([event])[0]
    sample_set = set(range(1_000_000))
    census = {
        "pointer_bytes": struct.calcsize("P"),
        "empty_set_bytes": sys.getsizeof(set()),
        "set_1m_bytes": sys.getsizeof(sample_set),
        "tuple2_bytes": sys.getsizeof((None, None)),
        "tuple4_bytes": sys.getsizeof((None, None, None, None)),
        "large_int_bytes": sys.getsizeof(EVENTS),
        "sha64_hex_str_bytes": sys.getsizeof("0" * 64),
        "event_shell_bytes": sys.getsizeof(event),
        "event_dict_bytes": sys.getsizeof(event.__dict__),
        "row_shell_bytes": sys.getsizeof(row),
        "row_dict_bytes": sys.getsizeof(row.__dict__),
        "list_reference_bytes": struct.calcsize("P"),
    }
    del sample_set
    return census


def bounded_benchmark(m1132) -> dict[str, float | int]:
    m1130 = m1132.load_m1130()
    producer = m1132.PerBeatAddressedWeightRefillProducer(lambda _event: None)
    provenance = "0" * 64
    slices = tuple(range(8)); enables = (0xffff,) * 8
    gc_was_enabled = gc.isenabled()
    gc.disable()
    started = time.perf_counter()
    try:
        for index in range(BENCHMARK_EVENTS):
            producer.emit_refill_event(
                axis="candidate", task_id=0, source_local_ordinal=index,
                requested_cycle=index, op="WRITE", logical_bank=0, half_slot=0,
                logical_row=0, local_row=0, native_slices=slices, bytes=128,
                byte_enable_per_slice=enables, native_macro_activations=8,
                service_beat_ordinal=index, store_transaction_ordinal=index,
                service_event_exact_once_id=m1130.exact_once_id(
                    "candidate", 0, index, index, index),
                source_row_provenance_sha256=provenance)
    finally:
        elapsed = time.perf_counter() - started
        if gc_was_enabled:
            gc.enable()
    require(producer.emitted == BENCHMARK_EVENTS and elapsed > 0,
            "bounded benchmark conservation")
    throughput = BENCHMARK_EVENTS / elapsed
    return {
        "events": BENCHMARK_EVENTS,
        "elapsed_seconds": elapsed,
        "events_per_second": throughput,
        "projected_producer_only_seconds": EVENTS / throughput,
    }


def main() -> int:
    before = {path: sha(path) for path in EXPECTED}
    for path, expected in EXPECTED.items():
        verify_regular(path, expected)
    verify_m1133()
    m1016 = load(M1016, "m1134c_m1016_readonly")
    require(m1016.EXPECTED_SERVICE_COUNTS["weight"] == EVENTS_PER_AXIS,
            "frozen weight count drift")
    m1130 = load(M1130, "m1134c_m1130_readonly")
    m1132 = load(M1132, "m1134c_m1132_readonly")
    info = meminfo()
    census = object_census(m1130)
    sample_capacity = next_set_capacity(1_000_000)
    require(sample_capacity * (2 * census["pointer_bytes"]) +
            census["empty_set_bytes"] == census["set_1m_bytes"],
            "CPython set-capacity model disagrees with bounded 1M census")
    capacity = next_set_capacity(EVENTS)
    set_entry_bytes = 2 * census["pointer_bytes"]
    per_set_table = capacity * set_entry_bytes + census["empty_set_bytes"]
    all_set_tables = 3 * per_set_table
    tuple_bytes_per_event = 2 * census["tuple2_bytes"] + census["tuple4_bytes"]
    all_key_tuples = EVENTS * tuple_bytes_per_event
    structural_set_floor = all_set_tables + all_key_tuples
    unique_exact_id_floor = EVENTS * census["sha64_hex_str_bytes"]
    # Beat values repeat across axes and transaction may alias beat: count only
    # one retained int object per distinct per-axis ordinal for a strict floor.
    ordinal_object_floor = EVENTS_PER_AXIS * census["large_int_bytes"]
    retained_producer_floor = (structural_set_floor + unique_exact_id_floor +
                               ordinal_object_floor)
    one_row_shell = (census["row_shell_bytes"] + census["row_dict_bytes"] +
                     census["list_reference_bytes"])
    one_m1130_rows_floor = EVENTS * one_row_shell
    current_chain_floor = retained_producer_floor + one_m1130_rows_floor
    commit_headroom = info["CommitHeadroom_kib"] * 1024
    benchmark = bounded_benchmark(m1132)
    m1102_result = json.loads(M1102_RESULT.read_text(encoding="utf-8"))
    base_runtime = float(m1102_result["work_domain_preflight"]["runtime_seconds"])
    benchmark["frozen_m1102_runtime_seconds"] = base_runtime
    benchmark["optimistic_additive_total_seconds"] = (
        base_runtime + float(benchmark["projected_producer_only_seconds"]))
    benchmark["interpretation"] = (
        "Optimistic same-interpreter proxy: bounded source-only producer with no "
        "downstream work, GC disabled, and small sets. Full-scale cache/resize, "
        "row construction, scheduling, validation, I/O, and sealing only increase it.")

    metrics = {
        "schema": "m1134c_production_scale_first_principles_metrics_r1_v1",
        "status": "STOP_M1132C_AND_BATCH_M1130C_AT_PRODUCTION_SCALE__GO_O_AXES_STREAMING_SUCCESSOR_SOURCE_ONLY",
        "inputs": {
            "weight_events_per_axis": EVENTS_PER_AXIS, "axes": AXES,
            "total_weight_events": EVENTS, "memory_snapshot": info,
            "python": sys.version, "full_replay_executed": False,
        },
        "cpython_object_census": census,
        "set_projection": {
            "set_count": 3, "entries_per_set": EVENTS,
            "minimum_slots_per_set_at_3_over_5_fill": capacity,
            "set_entry_bytes": set_entry_bytes,
            "table_bytes_per_set": per_set_table,
            "three_table_bytes": all_set_tables,
            "key_tuple_bytes_per_event": tuple_bytes_per_event,
            "all_key_tuple_bytes": all_key_tuples,
            "structural_set_floor_bytes": structural_set_floor,
            "unique_exact_id_floor_bytes": unique_exact_id_floor,
            "minimum_distinct_ordinal_object_bytes": ordinal_object_floor,
            "retained_producer_floor_bytes": retained_producer_floor,
            "capacity_model_validated_against_1m_live_set": True,
        },
        "batch_chain_projection": {
            "one_m1130_row_shell_dict_listref_bytes_per_event": one_row_shell,
            "one_m1130_rows_floor_bytes": one_m1130_rows_floor,
            "producer_plus_one_rows_floor_bytes": current_chain_floor,
            "commit_headroom_bytes": commit_headroom,
            "floor_minus_commit_headroom_bytes": current_chain_floor - commit_headroom,
            "fits_commit_headroom": current_chain_floor <= commit_headroom,
            "excluded_from_floor": [
                "second scheduled row list", "M1130 exact/beat/transaction sets",
                "native occupied conflict set (up to 8 keys/write)",
                "event list and event objects", "payload tuples/strings beyond strict minima",
                "allocator fragmentation", "source traversal state", "output and seal buffers",
            ],
        },
        "time_projection": benchmark,
        "operation_lower_bound": {
            "producer_calls": EVENTS, "event_validations": EVENTS,
            "exact_id_recomputations_inside_validate": EVENTS,
            "upstream_exact_id_constructions_required": EVENTS,
            "set_membership_checks": 3 * EVENTS, "set_insertions": 3 * EVENTS,
            "sink_calls": EVENTS,
        },
        "successor_feasibility": {
            "o_axes_exact_once_possible": True,
            "conditional_on": [
                "per-axis service beat ordinal is contiguous and strictly equals next_beat[axis]",
                "per-axis store transaction ordinal is globally contiguous and strictly equals next_txn[axis]",
                "producer order is nondecreasing in the frozen scheduler key within each axis",
                "exact-once ID is recomputed from axis/task/source/beat/transaction and matched",
                "all 17-field schema/address checks remain unchanged",
                "final per-axis count equals 70,853,184",
                "canonical binary streaming digest matches an independently sealed expected digest",
                "sink exception commits no counter, ordinal, digest, or scheduler state",
            ],
            "state": [
                "next_beat[3]", "next_transaction[3]", "event_count[3]",
                "byte/native-activation/stall conservation counters[3]",
                "SHA-256 context[3]", "next_free_cycle[3][24 native slices]",
            ],
            "memory_complexity": "O(axes + axes*native_slices), independent of event count",
            "failure_if_conditions_unavailable": (
                "STOP: use external-memory sort/bitmap or preserve O(N) exact sets; "
                "a digest without a sealed expected digest is evidence, not duplicate detection."
            ),
        },
        "verdict": {
            "m1132c_current_sets_at_full_scale": "STOP",
            "m1130c_batch_instrumentation_at_full_scale": "STOP",
            "real_hook_now": "STOP",
            "full_replay_now": "STOP",
            "minimum_next_step": "AUTHOR_ADDITIVE_O_AXES_STREAMING_VALIDATOR_AND_SINK_SOURCE_ONLY",
        },
        "execution": {"real_hook": False, "full_replay": False, "eda": False,
                      "gpu": False, "remote": False, "subjects_modified": False},
    }
    require(metrics["batch_chain_projection"]["fits_commit_headroom"] is False,
            "unexpected production memory fit")
    require(before == {path: sha(path) for path in before},
            "frozen source/result/docs mutation")
    OUTPUT.write_text(json.dumps(metrics, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": metrics["status"],
        "producer_floor_gib": retained_producer_floor / 2**30,
        "chain_floor_gib": current_chain_floor / 2**30,
        "commit_headroom_gib": commit_headroom / 2**30,
        "producer_only_proxy_seconds":
            benchmark["projected_producer_only_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
