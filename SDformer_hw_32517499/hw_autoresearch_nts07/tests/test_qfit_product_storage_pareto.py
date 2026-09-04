import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "pareto",
    ROOT / "scripts/analyze_qfit_product_storage_pareto.py",
)
assert SPEC and SPEC.loader
pareto = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pareto)


def test_product_cache_scales_with_output_width():
    narrow = pareto.cache_bits(
        lanes=32, ways=4, product_bits=128, gate_bits=9
    )
    wide = pareto.cache_bits(
        lanes=32, ways=4, product_bits=1024, gate_bits=9
    )
    assert narrow["data_bits"] == 16_384
    assert wide["data_bits"] == 131_072
    assert wide["total_bits"] > narrow["total_bits"]


def test_dqfs_term_storage_does_not_scale_with_output_width():
    narrow = pareto.dqfs_bits(capacity=128, ways=6, product_bits=128)
    wide = pareto.dqfs_bits(capacity=128, ways=6, product_bits=1024)
    assert wide["term_bits"] == narrow["term_bits"]
    assert wide["directory_bits"] == narrow["directory_bits"]
    assert wide["total_bits"] - narrow["total_bits"] == 896


def test_actual_trace_gate_slot_bound_and_pareto():
    result = pareto.evaluate(4, product_bits=68)
    assert result["terms"] == 1494
    assert result["max_distinct_gates_per_lane"] == 7
    slot4 = next(
        row
        for row in result["candidates"]
        if row["name"] == "cross_stage_gate_slot_4"
    )
    slot6 = next(
        row
        for row in result["candidates"]
        if row["name"] == "cross_stage_gate_slot_6"
    )
    assert slot4["product_computes"] == 397
    assert slot4["overflow_computes"] == 301
    assert slot6["product_computes"] == 165
    assert slot6["overflow_computes"] == 21
    frozen4 = next(
        row
        for row in result["candidates"]
        if row["name"] == "profile_frozen_gate_codebook_4"
    )
    assert frozen4["codebook"] == [15, 29, 31, 32]
    assert frozen4["product_computes"] == 262
    assert frozen4["total_bits"] == 8_900
    dqfs = [
        row
        for row in result["candidates"]
        if row["kind"] == "narrow_term_reorder"
    ]
    assert all(row["dominated"] for row in dqfs)
