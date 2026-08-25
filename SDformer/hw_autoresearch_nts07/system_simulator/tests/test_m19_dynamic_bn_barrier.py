#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_m19_dynamic_bn_barrier import (
    batchnorm_contract,
    index_unique_atlif_enters,
    movement_cycle_variants,
    trace_unique_direct_path,
)
from analyze_h67_atlif_dependency_dag import build_output_index


def ref(identity: int, shape: list[int], *, version: int = 0) -> dict:
    stride = []
    value = 1
    for size in reversed(shape):
        stride.append(value)
        value *= size
    return {
        "python_id": identity,
        "storage_cdata": identity * 100,
        "storage_data_ptr": identity * 1000,
        "storage_offset": 0,
        "shape": shape,
        "stride": list(reversed(stride)),
        "dtype": "torch.float32",
        "version": version,
    }


def event(index, kind, name, inputs, outputs, module_type=None, call=0):
    return {
        "event_index": index, "kind": kind, "name": name,
        "module_type": module_type, "module_call_index": call,
        "sample_id": 0, "sequence_key": "seq", "inputs": inputs, "outputs": outputs,
    }


def main() -> int:
    raw = ref(1, [10, 1, 96, 4, 8])
    normalized = ref(2, [10, 1, 96, 4, 8])
    events = [
        event(0, "leaf_module_exit", "conv", [], [raw], "Conv2d"),
        event(1, "leaf_module_exit", "bn", [raw], [normalized], "BatchNorm2d"),
        event(2, "leaf_module_enter", "atlif", [normalized], [], "ATLIFTernaryPSN"),
    ]
    boundary = {
        "producer": "conv", "producer_event_index": 0,
        "edge_enter_event_index": 2,
    }
    enters = index_unique_atlif_enters(events)
    enter = enters[(0, "seq", "atlif", 0)]
    path = trace_unique_direct_path(events, build_output_index(events), enter, boundary)
    assert [item["name"] for item in path] == ["bn", "conv"]
    bn = batchnorm_contract(path, {item["event_index"]: item for item in events}, 32)
    assert bn["channels"] == 96
    assert bn["reduction_population_per_channel"] == 320
    assert bn["elements"] == 30720
    assert bn["barrier_class"].startswith("BN_BLOCKED")

    permuted = ref(3, [10, 1, 384, 2, 3])
    normalized_linear = ref(4, [10, 1, 384, 2, 3])
    restored = ref(5, [10, 1, 2, 3, 384])
    linear_raw = ref(6, [10, 1, 2, 3, 384])
    linear_events = [
        event(0, "leaf_module_exit", "linear", [], [linear_raw], "Linear"),
        event(1, "functional_op", "aten.permute.default", [linear_raw], [permuted]),
        event(2, "leaf_module_exit", "bn_linear", [permuted], [normalized_linear], "BatchNorm2d"),
        event(3, "functional_op", "aten.permute.default", [normalized_linear], [restored]),
        event(4, "leaf_module_enter", "atlif_linear", [restored], [], "ATLIFTernaryPSN"),
    ]
    linear_enter = index_unique_atlif_enters(linear_events)[(0, "seq", "atlif_linear", 0)]
    linear_path = trace_unique_direct_path(
        linear_events, build_output_index(linear_events), linear_enter,
        {"producer": "linear", "producer_event_index": 0},
    )
    assert [item["name"] for item in linear_path] == [
        "aten.permute.default", "bn_linear", "aten.permute.default", "linear"
    ]
    linear_bn = batchnorm_contract(
        linear_path, {item["event_index"]: item for item in linear_events}, 32
    )
    assert linear_bn["channels"] == 384
    assert linear_bn["reduction_population_per_channel"] == 60

    duplicate = events + [dict(events[-1], event_index=3)]
    try:
        index_unique_atlif_enters(duplicate)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate ATLIF enter was admitted")

    without_bn = [events[0], dict(events[2], inputs=[raw], event_index=1)]
    no_bn_enter = index_unique_atlif_enters(without_bn)[(0, "seq", "atlif", 0)]
    no_bn_path = trace_unique_direct_path(
        without_bn, build_output_index(without_bn), no_bn_enter,
        {"producer": "conv", "producer_event_index": 0},
    )
    try:
        batchnorm_contract(no_bn_path, {item["event_index"]: item for item in without_bn}, 32)
    except ValueError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("BN-free path was admitted as dynamic BN")

    compute_bound = movement_cycle_variants(
        source_cycles=100, moment_serialization_cycles=20,
        moment_update_cycles=30, consumer_cycles=40,
        materialized_bytes=1000, memory_bytes_per_cycle=100.0,
    )
    assert compute_bound == {
        "unfused_five_movement": 210,
        "online_only_four_movement": 200,
        "fusion_only_three_movement": 170,
        "online_plus_fusion_two_movement": 160,
        "proposed_two_movement": 160,
    }
    memory_bound = movement_cycle_variants(
        source_cycles=100, moment_serialization_cycles=20,
        moment_update_cycles=30, consumer_cycles=40,
        materialized_bytes=1000, memory_bytes_per_cycle=10.0,
    )
    assert memory_bound == {
        "unfused_five_movement": 500,
        "online_only_four_movement": 420,
        "fusion_only_three_movement": 300,
        "online_plus_fusion_two_movement": 220,
        "proposed_two_movement": 220,
    }
    print("PASS m19-dynamic-bn-barrier 6/6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
