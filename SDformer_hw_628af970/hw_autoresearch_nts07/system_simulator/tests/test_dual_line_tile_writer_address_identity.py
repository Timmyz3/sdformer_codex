from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
WRITER_PATH = (
    ROOT
    / "neuron_experiments/H9_bipolar_self_attention/entrypoints/h67_dual_line_tile_trace.py"
)


def load_writer():
    spec = importlib.util.spec_from_file_location("dual_line_tile_writer", WRITER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DualLineTileTraceWriter


def context(name: str) -> dict[str, object]:
    return {
        "name": name,
        "sample_id": 0,
        "sample_key": "sample",
        "sequence_key": "sequence",
        "operator_call_index": 0,
        "temporal_steps": 2,
    }


def test_linear_records_output_lane_tile_count(tmp_path: Path) -> None:
    writer = load_writer()(tmp_path / "linear", pairs_per_call=1)
    module = torch.nn.Linear(4, 200, bias=False)
    value = torch.tensor([[[0.0, 1.0, 0.0, 1.0]], [[1.0, 1.0, 0.0, 0.0]]])
    writer.record_operator(module, value, **context("linear"))
    writer.close()
    rows = list(csv.DictReader((tmp_path / "linear/tile_records.csv").open()))
    assert len(rows) == 2
    assert {row["weight_group"] for row in rows} == {"0"}
    assert {row["output_lane_tile_count_96"] for row in rows} == {"3"}


def test_grouped_conv_records_exact_weight_group(tmp_path: Path) -> None:
    writer = load_writer()(tmp_path / "conv", pairs_per_call=1)
    module = torch.nn.Conv2d(4, 4, kernel_size=1, groups=4, bias=False)
    value = torch.tensor(
        [
            [[[[0.0]], [[1.0]], [[0.0]], [[1.0]]]],
            [[[[1.0]], [[1.0]], [[0.0]], [[0.0]]]],
        ]
    )
    writer.record_operator(module, value, **context("depthwise"))
    writer.close()
    rows = list(csv.DictReader((tmp_path / "conv/tile_records.csv").open()))
    assert len(rows) == 2
    # One deterministic row is selected from four depthwise groups.  Its group
    # identity must survive into the address contract instead of aliasing group 0.
    assert {row["weight_group"] for row in rows} == {"2"}
    assert {row["output_channel_fanout"] for row in rows} == {"1"}
    assert {row["output_lane_tile_count_96"] for row in rows} == {"1"}


def test_conv_p64_includes_corners_and_both_axes(tmp_path: Path) -> None:
    writer = load_writer()(tmp_path / "conv_p64", pairs_per_call=64)
    module = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
    value = torch.ones(2, 1, 1, 8, 12)
    writer.record_operator(module, value, **context("conv_p64"))
    writer.close()
    rows = list(csv.DictReader((tmp_path / "conv_p64/tile_records.csv").open()))
    row_ids = {int(row["row_id"]) for row in rows}
    assert len(row_ids) == 64
    xy = {(row_id // 12, row_id % 12) for row_id in row_ids}
    assert {(0, 0), (0, 11), (7, 0), (7, 11)} <= xy
    assert {0, 7} <= {y for y, _x in xy}
    assert {0, 11} <= {x for _y, x in xy}
    by_cluster: dict[int, set[int]] = {}
    for row in rows:
        by_cluster.setdefault(int(row["sample_cluster_id"]), set()).add(int(row["row_id"]))
    assert len(by_cluster) == 16
    for cluster_rows in by_cluster.values():
        assert len(cluster_rows) == 4
        assert sorted(cluster_rows) == list(range(min(cluster_rows), min(cluster_rows) + 4))
    manifest = json.loads((tmp_path / "conv_p64/manifest.json").read_text())
    assert manifest["schema"] == "dual_line_real_tile_trace_v2"
    assert manifest["cluster_contexts"] == 4
    coverage = manifest["conv_sampling"][0]
    assert coverage["sampled_rows"] == 64
    assert coverage["distinct_y"] >= 4
    assert coverage["distinct_x"] == 12
    assert coverage["corner_rows"] == 4
    assert coverage["padding_halo_rows"] > 0
    assert coverage["interior_rows"] > 0
    clusters = manifest["cluster_sampling"][0]
    assert clusters["population_clusters"] == 24
    assert clusters["sampled_clusters"] == 16
    assert sum(clusters["sample_clusters_by_stratum"].values()) == 16
