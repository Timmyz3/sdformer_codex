from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_m6_motion_granularity.py"


def load_module():
    spec = importlib.util.spec_from_file_location("m6_motion", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chunk_upper_bound_and_state_cost(tmp_path: Path) -> None:
    module = load_module()
    fields = [
        "sample_id", "sequence_key", "name", "operator", "operator_call_index",
        "row_id", "weight_group", "chunk_index", "chunks_per_row",
        "output_channel_fanout", "state_valid", "row_use_motion",
        "tile_current_count", "tile_positive_count", "tile_negative_count",
    ]
    rows = [
        dict(zip(fields, map(str, [0, "s", "n", "Linear", 0, 4, 0, 0, 2, 8,
                                   True, False, 10, 1, 1]))),
        dict(zip(fields, map(str, [0, "s", "n", "Linear", 0, 4, 0, 1, 2, 8,
                                   True, False, 1, 5, 5]))),
    ]
    with (tmp_path / "tile_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (tmp_path / "packed_tiles.npz").write_bytes(b"npz")
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    result = module.analyze_identity("x", tmp_path)
    assert result["source_work"]["local"] == 11
    assert result["source_work"]["row_hybrid"] == 11
    assert result["source_work"]["chunk_hybrid"] == 3
    assert result["max_state"]["row_destination_state_bits"] == 256
    assert result["max_state"]["chunk_partial_destination_state_bits"] == 512
