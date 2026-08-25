#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from extract_m18_direct_edge_boundaries import (
    HARDWARE_TAG_FIELD_ORDER,
    HardwareTagLifecycle,
    bind_hardware_tag_ledger_to_m17,
    build_hardware_tag_ledger,
    extract_boundaries,
)


def ref(identity: int, shape: list[int], stride: list[int], *, version: int = 0,
        data_ptr: int | None = None) -> dict:
    return {
        "python_id": identity * 10 + 1,
        "storage_cdata": identity * 100,
        "storage_data_ptr": data_ptr if data_ptr is not None else identity * 1000,
        "storage_offset": 0, "shape": shape, "stride": stride,
        "dtype": "torch.float32", "version": version,
    }


def make_case() -> tuple[list[dict], dict]:
    source = ref(1, [2, 3], [3, 1])
    produced = ref(2, [2, 3, 96], [288, 96, 1])
    produced_view = dict(produced, python_id=22, shape=[2, 96, 3], stride=[288, 1, 96])
    normalized = ref(3, [2, 96, 3], [288, 3, 1])
    normalized_view = dict(normalized, python_id=32, shape=[2, 3, 96], stride=[288, 96, 1])
    events = [
        {
            "event_index": 0, "kind": "leaf_module_exit", "name": "linear",
            "module_type": "Linear", "module_call_index": 2,
            "sample_id": 0, "sequence_key": "seq", "inputs": [source],
            "outputs": [produced],
        },
        {
            "event_index": 1, "kind": "functional_op", "name": "aten.permute.default",
            "sample_id": 0, "sequence_key": "seq", "inputs": [produced],
            "outputs": [produced_view],
        },
        {
            "event_index": 2, "kind": "leaf_module_exit", "name": "bn",
            "module_type": "BatchNorm2d", "module_call_index": 0,
            "sample_id": 0, "sequence_key": "seq", "inputs": [produced_view],
            "outputs": [normalized],
        },
        {
            "event_index": 3, "kind": "functional_op", "name": "aten.permute.default",
            "sample_id": 0, "sequence_key": "seq", "inputs": [normalized],
            "outputs": [normalized_view],
        },
        {
            "event_index": 4, "kind": "leaf_module_enter", "name": "atlif",
            "module_type": "ATLIFTernaryPSN", "module_call_index": 1,
            "sample_id": 0, "sequence_key": "seq", "inputs": [normalized_view],
            "outputs": [],
        },
    ]
    audit = {
        "rows": [{
            "admitted_for_overlap": True, "sample_id": 0, "sequence_key": "seq",
            "category": "direct_m4", "live": True,
            "name": "atlif", "module_call_index": 1, "enter_event_index": 4,
            "producers": ["linear"], "temporal_steps": 2, "service_cycles_l16": 7,
        }],
        "summary": {
            "admitted_direct_m4_calls": 1,
            "admitted_direct_m4_service_cycles": 7,
            "categories": {
                "direct_m4": {"calls": 1, "live_calls": 1, "service_cycles": 7},
            },
        },
    }
    return events, audit


def must_fail(events: list[dict], audit: dict, expected: str) -> None:
    try:
        extract_boundaries(events, audit)
    except ValueError as exc:
        assert expected in str(exc), (expected, str(exc))
    else:
        raise AssertionError("fail-close case was admitted: " + expected)


def main() -> int:
    tests = 0
    events, audit = make_case()
    rows = extract_boundaries(events, audit)
    assert len(rows) == 1
    row = rows[0]
    assert [hop["name"] for hop in row["path_certificate"]["producer_to_atlif_path"]] == [
        "linear", "aten.permute.default", "bn", "aten.permute.default",
    ]
    assert len(row["path_certificate_sha256"]) == 64
    assert row["causal_classification"] == "BN_BLOCKED"
    assert row["readiness_boundary"] == "GLOBAL_REDUCTION_STATISTICS_BARRIER"
    assert row["m15_admitted"] is False and len(row["bn_barriers"]) == 1
    tests += 1

    duplicate_enter = deepcopy(events)
    duplicate_enter.append(dict(duplicate_enter[-1], event_index=5))
    must_fail(duplicate_enter, audit, "duplicate ATLIF enter")
    tests += 1

    stale_audit = deepcopy(audit)
    stale_audit["rows"][0]["enter_event_index"] = 3
    must_fail(events, stale_audit, "enter_event_index")
    tests += 1

    version_drift = deepcopy(events)
    version_drift[-1]["inputs"] = [dict(version_drift[-1]["inputs"][0], version=1)]
    must_fail(version_drift, audit, "non-exact match quality")
    tests += 1

    storage_reuse = deepcopy(events)
    reused_pointer = storage_reuse[3]["outputs"][0]["storage_data_ptr"]
    unrelated = {
        "event_index": 3, "kind": "functional_op", "name": "aten.clone.default",
        "sample_id": 0, "sequence_key": "seq", "inputs": [ref(8, [1], [1])],
        "outputs": [ref(9, [2, 3, 96], [288, 96, 1], data_ptr=reused_pointer)],
    }
    for event in storage_reuse[3:]:
        event["event_index"] += 1
    storage_reuse.insert(3, unrelated)
    storage_audit = deepcopy(audit)
    storage_audit["rows"][0]["enter_event_index"] = 5
    storage_rows = extract_boundaries(storage_reuse, storage_audit)
    assert [hop["event_index"] for hop in storage_rows[0]["path_certificate"]["producer_to_atlif_path"]] == [0, 1, 2, 4]
    tests += 1

    no_bn = [events[0], events[-1]]
    direct_input = deepcopy(no_bn[-1])
    direct_input["event_index"] = 1
    direct_input["inputs"] = [deepcopy(events[0]["outputs"][0])]
    direct_audit = deepcopy(audit)
    direct_audit["rows"][0]["enter_event_index"] = 1
    must_fail([deepcopy(events[0]), direct_input], direct_audit, "not explicitly BN-blocked")
    tests += 1

    broken_total = deepcopy(audit)
    broken_total["summary"]["admitted_direct_m4_service_cycles"] = 8
    must_fail(events, broken_total, "service total")
    tests += 1

    category_mismatch = deepcopy(audit)
    category_mismatch["rows"][0]["category"] = "join_with_m4"
    must_fail(events, category_mismatch, "live direct_m4")
    tests += 1

    dead_direct = deepcopy(audit)
    dead_direct["rows"][0]["live"] = False
    must_fail(events, dead_direct, "live direct_m4")
    tests += 1

    category_summary_mismatch = deepcopy(audit)
    category_summary_mismatch["summary"]["categories"]["direct_m4"]["calls"] = 2
    must_fail(events, category_summary_mismatch, "categories.direct_m4")
    tests += 1

    hidden_live_direct = deepcopy(audit)
    hidden_row = deepcopy(hidden_live_direct["rows"][0])
    hidden_row["name"] = "hidden_atlif"
    hidden_row["module_call_index"] = 2
    hidden_row["admitted_for_overlap"] = False
    hidden_live_direct["rows"].append(hidden_row)
    must_fail(events, hidden_live_direct, "live direct_m4 population")
    tests += 1

    duplicate_audit = deepcopy(audit)
    duplicate_audit["rows"].append(deepcopy(duplicate_audit["rows"][0]))
    duplicate_audit["summary"]["admitted_direct_m4_calls"] = 2
    duplicate_audit["summary"]["admitted_direct_m4_service_cycles"] = 14
    duplicate_audit["summary"]["categories"]["direct_m4"] = {
        "calls": 2, "live_calls": 2, "service_cycles": 14,
    }
    must_fail(events, duplicate_audit, "duplicate admitted")
    tests += 1

    ledger_rows = extract_boundaries(events, audit)
    ledger = build_hardware_tag_ledger(
        ledger_rows, sample_epoch_bits=1, population_cluster_bits=2,
        contexts=4, producer_lanes=96,
    )
    assert ledger["field_order"] == list(HARDWARE_TAG_FIELD_ORDER)
    assert set(ledger["fields"]) == set(HARDWARE_TAG_FIELD_ORDER)
    assert "trace_sha256" in ledger["excluded_from_hardware_tag"]
    assert all("sha" not in name for name in ledger["fields"])
    assert ledger_rows[0]["static_edge_id"] == 0
    tests += 1

    with tempfile.TemporaryDirectory() as directory:
        manifest_path = Path(directory) / "manifest.json"
        stream_path = Path(directory) / "ordered_stream.npz"
        prototypes_path = Path(directory) / "prototypes.json"
        np.savez_compressed(
            stream_path,
            operator_index=np.asarray([0, 0, 0, 0, 0], dtype=np.uint16),
            population_cluster_id=np.asarray([0, 1, 2, 3, 4], dtype=np.uint64),
            prototype_id=np.asarray([0, 0, 0, 0, 0], dtype=np.uint32),
        )
        stream_bytes = stream_path.read_bytes()
        prototypes_path.write_text(json.dumps([{"contexts": 1}]), encoding="utf-8")
        prototype_bytes = prototypes_path.read_bytes()
        fake_manifest = {
            "status": "PASS_EXACT_FULL_SPATIAL_C4_SUFFICIENT_STATISTICS_NOT_SYSTEM_SPEEDUP",
            "ordered_clusters": 5,
            "files": {"ordered_stream.npz": {
                "sha256": hashlib.sha256(stream_bytes).hexdigest(),
                "bytes": len(stream_bytes),
            }, "prototypes.json": {
                "sha256": hashlib.sha256(prototype_bytes).hexdigest(),
                "bytes": len(prototype_bytes),
            }},
            "operators": [{
                "operator_index": 0,
                "sample_id": 0, "sequence_key": "seq", "name": "linear",
                "operator_call_index": 2, "temporal_steps": 2,
                "fanout": 96, "lane_tiles": 1,
                "row_count": 5, "population_clusters": 5,
                "motion_selected_rows_by_t": [0, 1],
            }],
        }
        manifest_path.write_text(json.dumps(fake_manifest), encoding="utf-8")
        try:
            bind_hardware_tag_ledger_to_m17(ledger, ledger_rows, manifest_path)
        except ValueError as exc:
            assert "population_cluster_id" in str(exc)
        else:
            raise AssertionError("an undersized population field passed the M17 bound check")
        wider_ledger = build_hardware_tag_ledger(
            ledger_rows, sample_epoch_bits=1, population_cluster_bits=3,
            contexts=4, producer_lanes=96,
        )
        bind_hardware_tag_ledger_to_m17(wider_ledger, ledger_rows, manifest_path)
        assert wider_ledger["observed_m17_bounds"]["population_cluster_id"] == 4
        assert wider_ledger["m17_binding"]["bound_check"].startswith("PASS")
        fake_manifest["operators"][0]["temporal_steps"] = 100
        manifest_path.write_text(json.dumps(fake_manifest), encoding="utf-8")
        try:
            bind_hardware_tag_ledger_to_m17(wider_ledger, ledger_rows, manifest_path)
        except ValueError as exc:
            assert "temporal/fanout/lane geometry" in str(exc)
        else:
            raise AssertionError("mismatched M17 temporal geometry passed the tag binder")

        fake_manifest["operators"][0]["temporal_steps"] = 2
        fake_manifest["operators"][0]["population_clusters"] = 999
        manifest_path.write_text(json.dumps(fake_manifest), encoding="utf-8")
        try:
            bind_hardware_tag_ledger_to_m17(wider_ledger, ledger_rows, manifest_path)
        except ValueError as exc:
            assert "population_clusters" in str(exc)
        else:
            raise AssertionError("mismatched population_clusters passed the tag binder")

        fake_manifest["operators"][0]["population_clusters"] = 5
        fake_manifest["operators"][0]["row_count"] = 6
        manifest_path.write_text(json.dumps(fake_manifest), encoding="utf-8")
        try:
            bind_hardware_tag_ledger_to_m17(wider_ledger, ledger_rows, manifest_path)
        except ValueError as exc:
            assert "row_count" in str(exc)
        else:
            raise AssertionError("mismatched row_count passed the tag binder")

        fake_manifest["operators"][0]["row_count"] = 5
        for invalid_motion in ([-1, 1], [1, 1], [0, 6]):
            fake_manifest["operators"][0]["motion_selected_rows_by_t"] = invalid_motion
            manifest_path.write_text(json.dumps(fake_manifest), encoding="utf-8")
            try:
                bind_hardware_tag_ledger_to_m17(wider_ledger, ledger_rows, manifest_path)
            except ValueError as exc:
                assert "Motion selector" in str(exc)
            else:
                raise AssertionError("invalid Motion timeline passed the tag binder")
    tests += 1

    lifecycle = HardwareTagLifecycle(ledger)
    lifecycle.begin_epoch(0)
    epoch0 = (0, 0, 1, 0, 0, 0, 0)
    lifecycle.issue(epoch0)
    try:
        lifecycle.issue(epoch0)
    except ValueError as exc:
        assert "duplicate hardware tag" in str(exc)
    else:
        raise AssertionError("duplicate token was admitted")
    tests += 1

    # One population owner may switch Local/Motion decisions across temporal
    # steps while several tokens remain resident in the same context.
    same_owner_other_token = (0, 0, 1, 1, 1, 0, 0)
    lifecycle.issue(same_owner_other_token)
    population_conflict = (0, 0, 2, 0, 0, 0, 0)
    try:
        lifecycle.issue(population_conflict)
    except ValueError as exc:
        assert "context already owned" in str(exc)
    else:
        raise AssertionError("a second population stole an active context")
    line_conflict = (0, 0, 1, 0, 1, 0, 0)
    try:
        lifecycle.issue(line_conflict)
    except ValueError as exc:
        assert "source line already selected" in str(exc)
    else:
        raise AssertionError("the same population timestep executed both source lines")
    try:
        lifecycle.consumer_finish(population_conflict)
    except ValueError as exc:
        assert "stale or duplicate" in str(exc)
    else:
        raise AssertionError("a wrong owner released an active context")
    assert lifecycle.context_owners[0] == (0, 0, 1)
    try:
        lifecycle.begin_epoch(1)
    except ValueError as exc:
        assert "live context owners" in str(exc)
    else:
        raise AssertionError("context was released before consumer finish")
    lifecycle.consumer_finish(epoch0)
    assert 0 in lifecycle.context_owners
    lifecycle.consumer_finish(same_owner_other_token)
    assert 0 not in lifecycle.context_owners
    tests += 1

    lifecycle.begin_epoch(1)
    tests += 1

    try:
        lifecycle.issue(epoch0)
    except ValueError as exc:
        assert "stale or future" in str(exc)
    else:
        raise AssertionError("stale epoch tag was admitted")
    tests += 1

    replay = (0, 1, 1, 0, 0, 0, 0)
    lifecycle.issue(replay)
    lifecycle.consumer_finish(replay)
    try:
        lifecycle.begin_epoch(1)
    except ValueError as exc:
        assert "next epoch" in str(exc)
    else:
        raise AssertionError("sample replay reused its epoch")
    tests += 1

    try:
        lifecycle.begin_epoch(0)
    except ValueError as exc:
        assert "wrap requires" in str(exc)
    else:
        raise AssertionError("sample_epoch wrapped without a fence")
    lifecycle.begin_epoch(0, wrap_fence=True)
    tests += 1

    print("PASS m18-bn-blocked-boundaries {}/{}".format(tests, tests))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
