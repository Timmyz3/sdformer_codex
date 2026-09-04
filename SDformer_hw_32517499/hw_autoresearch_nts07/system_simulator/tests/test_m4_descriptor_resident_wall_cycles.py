from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/analyze_m4_descriptor_resident_wall_cycles.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("m4_wall_cycles", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compact_scheduler_enforces_context_reducer_slots() -> None:
    module = load_module()
    counts = np.zeros((2, 4), dtype=np.int64)
    counts[0, :] = 1
    assert module.compact_issue_cycles(counts, reduce_slots=4) == 1
    assert module.compact_issue_cycles(counts, reduce_slots=2) == 2


def test_compact_scheduler_fills_idle_banks_from_another_context() -> None:
    module = load_module()
    counts = np.asarray([[2, 0, 0, 0], [0, 2, 0, 0]], dtype=np.int64)
    assert module.compact_issue_cycles(counts, reduce_slots=1) == 2


def test_wall_cycle_model_charges_load_control_issue_and_output() -> None:
    module = load_module()
    common = {
        "sample_id": "0",
        "sequence_key": "seq",
        "name": "proj",
        "operator": "Linear",
        "operator_call_index": "0",
        "row_id": "0",
        "temporal_step": "0",
        "weight_group": "0",
        "source_width": "8",
        "chunks_per_row": "1",
        "chunk_index": "0",
        "valid_bits": "8",
        "output_lane_tile_count_96": "2",
        "output_channel_fanout": "192",
        "row_use_motion": "false",
    }
    records = [common, {**common, "row_id": "1"}]
    current = np.zeros((2, 1), dtype=np.uint8)
    previous = np.zeros_like(current)
    current[0, 0] = 0b0000_0101
    current[1, 0] = 0b0000_1010
    result = module.analyze_identity(
        records,
        current,
        previous,
        line="local",
        issue_width=4,
        contexts=2,
        reduce_slots=1,
    )
    # load=2; each of two lane tiles has issue=2, PREP+DRAIN=2, output=2.
    assert result["descriptor_load_cycles"] == 2
    assert result["compact_issue_cycles"] == 4
    assert result["chunk_control_cycles"] == 4
    assert result["output_cycles"] == 4
    assert result["m4_wall_cycles"] == 14
    assert result["p1_sparse_wall_cycles"] == 18


def test_sample_boundary_is_a_batch_fence() -> None:
    module = load_module()
    common = {
        "sequence_key": "seq",
        "name": "proj",
        "operator": "Linear",
        "operator_call_index": "0",
        "row_id": "0",
        "temporal_step": "0",
        "weight_group": "0",
        "source_width": "8",
        "chunks_per_row": "1",
        "chunk_index": "0",
        "valid_bits": "8",
        "output_lane_tile_count_96": "1",
        "output_channel_fanout": "96",
        "row_use_motion": "false",
    }
    records = [{**common, "sample_id": "0"}, {**common, "sample_id": "1"}]
    current = np.asarray([[1], [2]], dtype=np.uint8)
    previous = np.zeros_like(current)
    result = module.analyze_identity(
        records,
        current,
        previous,
        line="local",
        issue_width=4,
        contexts=2,
        reduce_slots=1,
    )
    assert result["batches"] == 2
    assert result["cross_sample_contexts"] is False


def test_temporal_fence_removes_cross_step_context_fill() -> None:
    module = load_module()
    common = {
        "sample_id": "0",
        "sequence_key": "seq",
        "name": "proj",
        "operator": "Linear",
        "operator_call_index": "7",
        "row_id": "0",
        "weight_group": "0",
        "source_width": "8",
        "chunks_per_row": "1",
        "chunk_index": "0",
        "valid_bits": "8",
        "output_lane_tile_count_96": "1",
        "output_channel_fanout": "96",
        "row_use_motion": "false",
    }
    records = [
        {**common, "temporal_step": str(step), "operator_call_index": str(7 + step)}
        for step in range(4)
    ]
    current = np.ones((4, 1), dtype=np.uint8)
    previous = np.zeros_like(current)
    materialized = module.analyze_identity(
        records,
        current,
        previous,
        line="local",
        issue_width=4,
        contexts=4,
        reduce_slots=1,
        availability_mode="layer_materialized_greedy",
    )
    fenced = module.analyze_identity(
        records,
        current,
        previous,
        line="local",
        issue_width=4,
        contexts=4,
        reduce_slots=1,
        availability_mode="temporal_fenced",
    )
    assert materialized["batches"] == 1
    assert materialized["cross_temporal_batches"] == 1
    assert fenced["batches"] == 4
    assert fenced["cross_temporal_batches"] == 0
    assert fenced["m4_wall_cycles"] > materialized["m4_wall_cycles"]
