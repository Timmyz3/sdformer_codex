from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/summarize_m5_lane_reducer_dse.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("m5_lane_dse", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def payload(lanes: int, slots: int, cycles: int) -> dict:
    identity = {
        "availability_mode": "temporal_fenced",
        "cross_temporal_batches": 0,
        "cross_operator_call_batches": 0,
        "cross_sequence_batches": 0,
    }
    line = {
        "m4_wall_cycles": cycles,
        "speedup_vs_p1_sparse_wall": 4.0,
        "speedup_vs_same_width_dense_wall": 3.0,
        "same_width_dense_sample_speedup_min": 2.5,
        "per_identity": {"x": identity},
    }
    return {
        "architecture": {
            "availability_mode": "temporal_fenced",
            "output_lanes": lanes,
            "reduce_slots_per_context": slots,
            "weight_response_width_bits": 16 * lanes * 8,
            "accumulator_output_width_bits": lanes * 32,
            "accumulator_state_bits": 4 * lanes * 32,
            "shared_reducer_signed_adders": 4 * slots * lanes,
        },
        "identities": {"x": {"sha": "same"}},
        "variants": {"local": dict(line), "hybrid": dict(line)},
    }


def test_summary_normalizes_against_l96_r4() -> None:
    module = load_module()
    result = module.summarize([payload(16, 2, 600), payload(96, 4, 100)])
    reference = result["candidates"][1]
    narrow = result["candidates"][0]
    assert reference["local"]["throughput_vs_l96_r4"] == 1.0
    assert narrow["local"]["throughput_vs_l96_r4"] == 1 / 6
    assert narrow["local"]["throughput_per_adder_vs_l96_r4"] == 2.0


def test_summary_rejects_cross_temporal_candidate() -> None:
    module = load_module()
    bad = payload(16, 2, 600)
    bad["variants"]["local"]["per_identity"]["x"]["cross_temporal_batches"] = 1
    try:
        module.summarize([bad, payload(96, 4, 100)])
    except ValueError as error:
        assert "availability fence" in str(error)
    else:
        raise AssertionError("cross-temporal candidate was admitted")
