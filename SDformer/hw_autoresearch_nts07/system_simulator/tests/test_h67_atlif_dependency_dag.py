#!/usr/bin/env python3
import csv
import json
import tempfile
from pathlib import Path

from analyze_h67_atlif_dependency_dag import (
    analyze, atlif_service_cycles, build_output_index, find_latest_producer,
    trace_atlif,
)


def ref(identity, offset=0, shape=None):
    shape = shape or [1, 16]
    return {
        "python_id": identity, "storage_cdata": identity * 100,
        "storage_data_ptr": identity * 1000, "storage_nbytes": 4096,
        "storage_offset": offset, "shape": shape, "stride": [16, 1],
        "dtype": "torch.float32", "device": "cuda:0", "version": 0,
    }


def main():
    a, b, c, d, e = (ref(value) for value in range(1, 6))
    events = [
        {"event_index": 0, "kind": "leaf_module_exit", "name": "conv", "module_type": "Conv2d", "inputs": [a], "outputs": [b]},
        {"event_index": 1, "kind": "functional_op", "name": "aten.view.default", "inputs": [b], "outputs": [c]},
        {"event_index": 2, "kind": "leaf_module_enter", "name": "atlif.direct", "module_type": "ATLIFTernaryPSN", "module_call_index": 0, "inputs": [c], "outputs": []},
        {"event_index": 3, "kind": "leaf_module_exit", "name": "left", "module_type": "Linear", "inputs": [a], "outputs": [d]},
        {"event_index": 4, "kind": "leaf_module_exit", "name": "right", "module_type": "Linear", "inputs": [a], "outputs": [e]},
        {"event_index": 5, "kind": "functional_op", "name": "aten.add.Tensor", "inputs": [d, e], "outputs": [a]},
        {"event_index": 6, "kind": "leaf_module_enter", "name": "atlif.join", "module_type": "ATLIFTernaryPSN", "module_call_index": 0, "inputs": [a], "outputs": []},
    ]
    index = build_output_index(events)
    direct = trace_atlif(events[2], index)
    assert direct["category"] == "direct" and direct["producers"] == ["conv"]
    joined = trace_atlif(events[6], index)
    assert joined["category"] == "join" and joined["producers"] == ["left", "right"]
    assert len(joined["join_operands"]) == 1
    assert len(joined["join_operands"][0]["operands"]) == 2
    assert {
        endpoint["name"]
        for operand in joined["join_operands"][0]["operands"]
        for endpoint in operand["nearest_boundaries"]
    } == {"left", "right"}
    persistent = ref(6)
    persistent_view = dict(persistent, python_id=7)
    joined_value = ref(8)
    persistent_events = events + [
        {"event_index": 7, "kind": "persistent_tensor", "name": "block.positional_encoding", "inputs": [], "outputs": [persistent]},
        {"event_index": 8, "kind": "functional_op", "name": "aten.view.default", "inputs": [persistent], "outputs": [persistent_view]},
        {"event_index": 9, "kind": "functional_op", "name": "aten.add.Tensor", "inputs": [d, persistent_view], "outputs": [joined_value]},
        {"event_index": 10, "kind": "leaf_module_enter", "name": "atlif.persistent_join", "module_type": "ATLIFTernaryPSN", "module_call_index": 0, "inputs": [joined_value], "outputs": []},
    ]
    persistent_join = trace_atlif(persistent_events[-1], build_output_index(persistent_events))
    persistent_boundaries = [
        endpoint
        for operand in persistent_join["join_operands"][0]["operands"]
        for endpoint in operand["nearest_boundaries"]
        if endpoint["kind"] == "persistent"
    ]
    assert len(persistent_boundaries) == 1
    assert persistent_boundaries[0]["name"] == "block.positional_encoding"
    assert persistent_boundaries[0]["match_quality"] == "exact_tensor_version"
    reused_ptr = dict(c, storage_cdata=999999)
    assert find_latest_producer(reused_ptr, 3, index, -1, "") is None
    older_version = dict(c, version=-1)
    assert find_latest_producer(older_version, 3, index, -1, "") is None
    partitioned = [dict(events[1], sample_id=1, sequence_key="other")]
    assert find_latest_producer(c, 3, build_output_index(partitioned), 0, "tiny") is None
    assert atlif_service_cycles(160, 10, 1, lanes=16) == 10
    assert atlif_service_cycles(800, 2, 1, lanes=16) == 10
    ordered = [{"kind": "atlif", "name": "atlif.direct"}, {"kind": "atlif", "name": "atlif.join"}]
    ledger = {
        "atlif.direct": {"live": True, "temporal_steps": 10, "service_cycles": 10},
        "atlif.join": {"live": True, "temporal_steps": 10, "service_cycles": 10},
    }
    rows, summary = analyze(events, ordered, {"conv"}, ledger, lanes=16)
    categories = {row["name"]: row["category"] for row in rows}
    assert categories == {"atlif.direct": "direct_m4", "atlif.join": "join_non_m4"}
    assert summary["live_service_cycles_l16"] == 20
    assert summary["all_calls_with_unmatched_refs"] == 0
    assert summary["all_calls_with_uncertain_matches"] == 0
    assert not summary["ordered_names_missing_from_dependency"]
    print("PASS_H67_ATLIF_DEPENDENCY_DAG_CLASSIFIER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
