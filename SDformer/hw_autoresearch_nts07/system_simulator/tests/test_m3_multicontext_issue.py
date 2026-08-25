from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_m3_multicontext_issue.py"


def load_module():
    spec = importlib.util.spec_from_file_location("m3_multicontext_issue", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contexts_fill_complementary_banks() -> None:
    module = load_module()
    counts = np.asarray([[2, 0], [0, 2]], dtype=np.int64)
    assert module.schedule_object_bank_counts(counts, 1) == 4
    assert module.schedule_object_bank_counts(counts, 2) == 2


def test_context_batch_boundary_is_fail_closed() -> None:
    module = load_module()
    counts = np.asarray([[1, 0], [0, 1], [1, 0]], dtype=np.int64)
    assert module.schedule_object_bank_counts(counts, 2) == 2
    assert module.schedule_object_bank_counts(counts, 3) == 2


def test_weight_object_separates_group_and_lane_tile() -> None:
    module = load_module()
    row = {
        "name": "conv",
        "operator": "Conv2d",
        "weight_group": "3",
        "source_base": "256",
        "source_width": "432",
        "chunk_index": "1",
    }
    assert module.weight_object(row, 0) != module.weight_object({**row, "weight_group": "4"}, 0)
    assert module.weight_object(row, 0) != module.weight_object(row, 1)


def test_analyzer_never_fills_banks_across_sample_boundary() -> None:
    module = load_module()
    common = {
        "name": "proj",
        "operator": "Linear",
        "weight_group": "0",
        "source_base": "0",
        "source_width": "256",
        "chunk_index": "0",
        "output_lane_tile_count_96": "1",
        "valid_bits": "256",
        "chunks_per_row": "1",
        "row_use_motion": "false",
    }
    records = [{**common, "sample_id": "0"}, {**common, "sample_id": "1"}]
    current = np.zeros((2, 32), dtype=np.uint8)
    previous = np.zeros_like(current)
    current[0, 0] = 0b0000_0101  # two sources in bank 0
    current[1, 0] = 0b0000_1010  # two sources in bank 1
    result = module.analyze_identity(
        records, current, previous, line="local", issue_width=2, contexts=2
    )
    assert result["cross_sample_contexts"] is False
    assert result["issue_cycles"] == 4
    assert result["speedup_vs_p1_source_cycles"] == 1.0


def test_serialized_command_cost_and_descriptor_reuse_are_separate() -> None:
    module = load_module()
    common = {
        "name": "proj",
        "operator": "Linear",
        "weight_group": "0",
        "source_base": "0",
        "source_width": "256",
        "chunk_index": "0",
        "output_lane_tile_count_96": "2",
        "valid_bits": "256",
        "chunks_per_row": "1",
        "row_use_motion": "false",
        "sample_id": "0",
    }
    records = [common.copy(), common.copy()]
    current = np.zeros((2, 32), dtype=np.uint8)
    previous = np.zeros_like(current)
    current[0, 0] = 0b0000_0101  # two sources in bank 0
    current[1, 0] = 0b0000_1010  # two sources in bank 1
    result = module.analyze_identity(
        records, current, previous, line="local", issue_width=2, contexts=2
    )
    assert result["selected_sources"] == 8
    assert result["issue_cycles"] == 4
    assert result["lane_expanded_transactions"] == 4
    assert result["serialized_service_lower_bound_cycles"] == 8
    assert result["speedup_vs_p1_serialized_service_lower_bound"] == 1.5
    assert result["descriptor_load_cycles_if_reused_across_output_lanes"] == 2
    assert result["descriptor_residency_optimistic_cycles"] == 6
    assert result["speedup_vs_p1_descriptor_residency_optimistic_bound"] == 2.0
