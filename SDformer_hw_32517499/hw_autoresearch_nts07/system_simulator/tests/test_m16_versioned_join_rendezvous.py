#!/usr/bin/env python3
"""Synthetic fail-closed checks for M16 versioned join classification."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_m16_versioned_join_rendezvous.py"
SPEC = importlib.util.spec_from_file_location("m16", SCRIPT)
M16 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M16)


def boundary(name: str, event: int, *, exact: bool = True) -> dict:
    return {
        "event_index": event,
        "kind": "module",
        "name": name,
        "module_call_index": event % 3,
        "match_quality": "exact_tensor_version" if exact else "storage_overlap_only",
    }


def operand(index: int, shape: list[int], endpoint: dict) -> dict:
    return {
        "operand_index": index,
        "input_tensor": {
            "shape": shape, "stride": [6144, 6144, 64, 8, 1],
            "storage_offset": 0, "version": 7, "dtype": "torch.float32",
        },
        "nearest_boundaries": [endpoint],
    }


def row() -> dict:
    shape = [10, 1, 96, 8, 8]
    return {
        "name": "atlif0",
        "sample_id": 0,
        "sequence_key": "seq",
        "module_call_index": 4,
        "service_cycles_l16": 3840,
        "category": "join_with_m4",
        "live": True,
        "transformed_seen": False,
        "functional_ops": ["aten.add.Tensor"],
        "join_operands": [{
            "join_event_index": 30,
            "join_name": "aten.add.Tensor",
            "output_tensor": {
                "shape": shape, "stride": [6144, 6144, 64, 8, 1],
                "storage_offset": 0, "version": 8, "dtype": "torch.float32",
            },
            "operands": [
                operand(0, shape, boundary("resident", 10)),
                operand(1, shape, boundary("m4", 20)),
            ],
        }],
    }


def test_classification() -> None:
    m4 = {(0, "seq", "m4", 2)}
    good = M16.classify_join_edge(row(), m4)
    assert good["candidate"] is True
    assert good["stream_operands"][0]["name"] == "m4"
    assert good["logical_resident_operand_payload_bytes_at_1B_per_element"] == 10 * 96 * 8 * 8

    broadcast = row()
    broadcast["join_operands"][0]["operands"][0]["input_tensor"]["shape"] = [1]
    assert "BROADCAST_OR_SHAPE_CHANGE" in M16.classify_join_edge(broadcast, m4)["reasons"]

    transformed = row()
    transformed["join_operands"][0]["join_name"] = "aten.cat.default"
    assert "NOT_POINTWISE_ADD" in M16.classify_join_edge(transformed, m4)["reasons"]

    old_m4 = row()
    old_m4["join_operands"][0]["operands"] = [
        operand(0, [10, 1, 96, 8, 8], boundary("m4", 10)),
        operand(1, [10, 1, 96, 8, 8], boundary("resident", 20)),
    ]
    assert "M4_NOT_LAST_BOUNDARY_IN_SOFTWARE_TRACE" in M16.classify_join_edge(old_m4, m4)["reasons"]

    inexact = row()
    inexact["join_operands"][0]["operands"][1]["nearest_boundaries"] = [
        boundary("m4", 20, exact=False)
    ]
    assert "NON_EXACT_OPERAND_VERSION" in M16.classify_join_edge(inexact, m4)["reasons"]

    future = row()
    future["join_operands"][0]["operands"][1]["nearest_boundaries"] = [boundary("m4", 31)]
    assert "OPERAND_NOT_AVAILABLE_BEFORE_JOIN" in M16.classify_join_edge(future, m4)["reasons"]

    not_binary = row()
    not_binary["join_operands"][0]["operands"] = not_binary["join_operands"][0]["operands"][:1]
    assert "NOT_BINARY_ADD" in M16.classify_join_edge(not_binary, m4)["reasons"]

    wrong_call = M16.classify_join_edge(row(), {(0, "seq", "m4", 1)})
    assert "NO_M4_STREAM_OPERAND" in wrong_call["reasons"]

    view = row()
    view["join_operands"][0]["operands"][1]["input_tensor"]["stride"][-1] = 2
    assert "NON_POINTWISE_VIEW_LAYOUT" in M16.classify_join_edge(view, m4)["reasons"]

    downstream = row()
    downstream["transformed_seen"] = True
    assert "DOWNSTREAM_OR_JOIN_TRANSFORM_NOT_EXCLUDED" in M16.classify_join_edge(downstream, m4)["reasons"]


if __name__ == "__main__":
    test_classification()
    print("PASS m16-versioned-join 10/10")
