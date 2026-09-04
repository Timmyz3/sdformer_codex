#!/usr/bin/env python3
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from analyze_m14_stratified_c4_wall_cycles import build_clusters, cluster_metrics, load_wall


def make_records() -> list[dict[str, str]]:
    records = []
    for row_id in range(4):
        for temporal_step in range(2):
            records.append({
                "sample_id": "0", "sample_key": "sample", "sequence_key": "sequence",
                "name": "linear", "operator": "Linear", "operator_call_index": "0",
                "weight_group": "0", "row_id": str(row_id), "chunk_index": "0",
                "chunks_per_row": "1", "source_width": "4", "valid_bits": "4",
                "output_channel_fanout": "4", "temporal_step": str(temporal_step),
                "sample_cluster_id": "3", "sample_cluster_lane": str(row_id),
                "sample_cluster_rows": "4", "sampling_stratum": "flat",
                "stratum_population_clusters": "10", "stratum_sample_clusters": "2",
                "cluster_inverse_probability_weight": "5.0", "row_use_motion": "false",
            })
    return records


def main() -> int:
    wall = load_wall()
    records = make_records()
    clusters = build_clusters(records, wall)
    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster["row_ids"] == [0, 1, 2, 3]
    assert cluster["weight"] == 5.0
    bits = np.zeros((len(records), 256), dtype=bool)
    bits[:, 0] = True
    metrics = cluster_metrics(
        cluster, records, bits, wall, issue_width=2, reduce_slots=1, output_lanes=4
    )
    assert metrics == {
        "descriptor_load_cycles": 8,
        "selected_sources": 8,
        "selected_product_terms": 32,
        "same_width_dense_issue_cycles": 16,
        "compact_issue_cycles": 8,
        "chunk_control_cycles": 4,
        "output_cycles": 8,
    }
    broken = copy.deepcopy(records)
    for row in broken:
        if row["sample_cluster_lane"] == "3":
            row["row_id"] = "9"
    try:
        build_clusters(broken, wall)
    except ValueError as exc:
        assert "non-adjacent" in str(exc)
    else:
        raise AssertionError("non-adjacent C4 was admitted")
    print("PASS_M14_STRATIFIED_C4_WALL_CYCLES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
