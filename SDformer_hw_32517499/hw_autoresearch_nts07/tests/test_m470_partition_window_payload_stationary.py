#!/usr/bin/env python3
"""Small synthetic invariants for the M470 CPU DSE; no frozen trace read."""

import importlib.util
from pathlib import Path


SCRIPT = (Path(__file__).resolve().parents[1] / "system_simulator" /
          "scripts" / "analyze_m470_h67_partition_window_payload_stationary.py")
SPEC = importlib.util.spec_from_file_location("m470_analyzer", str(SCRIPT))
M470 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M470)


def synthetic_phases():
    result = []
    for partition in range(432):
        result.append({
            "active_rows": 1,
            "used_center_runs": [(0, 1)],
            "m430_used_pwp_patterns": 2,
            "used_center_population_sum": 5,
            "used_center_add_source_sum": 3,
            "k1_m430_generator_cycles_per_half": 24,
            "k2_m430_generator_cycles_per_half": 16,
            "k4_m430_generator_cycles_per_half": 12,
            "k8_m430_generator_cycles_per_half": 12,
        })
    return result


def cycle_contract():
    return {
        "dma_command_setup_cycles": 32,
        "psum_bytes_per_direction_per_boundary": 3000 * 8 * 228,
        "task_drain_cycles": 2,
        "popcount_filter_pipeline_cycles_per_task": 5,
        "config_select_cycles_per_task": 1,
        "descriptor_sram_latency_cycles_per_nonempty_task": 8,
    }


def aggregate():
    return {
        "source_rows": 432 * 3000,
        "active_rows": 432,
        "pwp_rows": 432,
        "correction_ops_per_block": 864,
        "bit_sparse_vector_ops_per_block": 1728,
        "early_extra": 432,
    }


def test_stored_vs_lazy_macro_width_and_depth_rounding():
    stored = M470.capacity_breakdown(1, 4, "stored_pwp", 1, 32, 16384)
    lazy = M470.capacity_breakdown(1, 4, "lazy_pwp", 1, 32, 16384)
    assert stored["macro_rounded_items"]["window_stored_pwp"] == (
        9 * 18 * 128)
    assert lazy["macro_rounded_items"]["window_lazy_generated_pwp"] == (
        8 * 18 * 128)
    assert stored["macro_rounded_items"]["window_stored_pwp"] >= (
        stored["logical_items"]["window_stored_pwp"])
    assert lazy["macro_rounded_items"]["window_lazy_generated_pwp"] >= (
        lazy["logical_items"]["window_lazy_generated_pwp"])


def test_full_operator_spill_is_charged_both_directions():
    phases = synthetic_phases()
    window = {
        "total_operator_windows": 108,
        "total_operator_window_boundaries": 107,
    }
    point = M470.compute_point(
        "stored_pwp", 4, 8, 3000, 128, phases, aggregate(), 432,
        window, cycle_contract())
    one_direction = 107 * 3000 * 8 * 228
    assert point["psum_spill_write_bytes"] == one_direction
    assert point["psum_reload_read_bytes"] == one_direction
    assert point["spill_reload_dram_bytes"] == 2 * one_direction
    assert point["spill_dma_commands"] == 214
    assert point["dram_bytes"] == (
        point["payload_fill_bytes"] + 2 * one_direction)


def test_same_window_spill_for_strong_zero_and_stored_pwp():
    phases = synthetic_phases()
    window = {
        "total_operator_windows": 54,
        "total_operator_window_boundaries": 53,
    }
    stored = M470.compute_point(
        "stored_pwp", 8, 4, 3000, 32, phases, aggregate(), 432,
        window, cycle_contract())
    zero = M470.compute_point(
        "strong_zero", 8, 4, 3000, 32, phases, aggregate(), 432,
        window, cycle_contract())
    assert stored["psum_spill_write_bytes"] == zero["psum_spill_write_bytes"]
    assert stored["psum_reload_read_bytes"] == zero["psum_reload_read_bytes"]
    assert stored["spill_reload_cycles"] == zero["spill_reload_cycles"]
    assert stored["passes_per_task"] == 2
    assert zero["passes_per_task"] == 2
    assert stored["source_sram_bytes"] == aggregate()["source_rows"] * 4


def test_csv_field_union_accepts_candidate_only_comparison_fields():
    fields = M470.csv_field_union([
        {"mode": "strong_zero", "total_cycles": 10},
        {"mode": "stored_pwp", "total_cycles": 8,
         "speedup_vs_same_resource_optimized_strong_zero": 1.25},
    ])
    assert fields == [
        "mode", "speedup_vs_same_resource_optimized_strong_zero",
        "total_cycles"]


if __name__ == "__main__":
    test_stored_vs_lazy_macro_width_and_depth_rounding()
    test_full_operator_spill_is_charged_both_directions()
    test_same_window_spill_for_strong_zero_and_stored_pwp()
    test_csv_field_union_accepts_candidate_only_comparison_fields()
    print("PASS M470 synthetic invariants=4")
