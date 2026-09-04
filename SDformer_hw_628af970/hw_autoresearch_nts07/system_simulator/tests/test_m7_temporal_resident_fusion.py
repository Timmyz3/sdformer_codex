from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_m7_temporal_resident_fusion.py"


def load_module():
    spec = importlib.util.spec_from_file_location("m7_fusion", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_system_envelope_charges_slots_parallel_stream() -> None:
    module = load_module()
    ledger = {
        "status": "PASS_TRANSACTION_LEDGER_MODEL_NOT_CYCLE_ACCURATE",
        "cycles_per_frame_model": {
            "fixed_total": 1300, "operator_activity_weighted": 1000,
            "atlif_non_dead": 200,
        },
        "attention": {"fixed_cycles_per_frame": 120, "rqtb_cycles_per_frame": 100},
        "config": {"atlif_lanes": 100},
    }
    contract = {"coverage": {
        "eligible_cycles": 600,
        "categories": {
            "attention_k_projection": {"eligible_cycles": 50},
            "attention_q_projection": {"eligible_cycles": 50},
        },
    }}
    item = {"speedup_vs_p1_sparse_wall": 3.0, "m4_wall_cycles": 90}
    m4 = {"variants": {
        "local": {"per_identity": {"H67": item}},
        "hybrid": {"per_identity": {"H67": {**item, "m4_wall_cycles": 81}}},
    }}
    transactions = [{
        "name": "t10", "deployment_dead_result": "False", "temporal_steps": "10",
        "elements_per_frame": "200", "dense_macs_per_frame": "2000",
    }, {
        "name": "t2", "deployment_dead_result": "False", "temporal_steps": "2",
        "elements_per_frame": "100", "dense_macs_per_frame": "200",
    }]
    result = module.system_envelope(
        ledger, contract, m4, transactions, slots=10, stream_lanes=[10]
    )
    local = result["variants"]["local"]
    assert local["optimized_operator_cycles"] == 667
    point = local["stream_points"][0]
    assert point["atlif_stream_service_cycles"] == 22
    assert point["atlif_equal_resource_service_cycles"] == 22
    assert point["atlif_ideal_compute_occupancy_no_bank_stalls"] == 1.0
    assert point["no_overlap_cycles"] == 789
    assert point["atlif_mac_matched_fixed_operator_rqtb_cycles"] == 1122
    assert point["packing_compute_speedup_same_m4_rqtb"] == 1.0
    assert point["temporal_residency_cycle_gain_same_m4_rqtb"] == "UNMODELED"


def test_t2_packing_rounds_per_invocation() -> None:
    module = load_module()
    rows = [{
        "name": "short_t2", "deployment_dead_result": "False", "temporal_steps": "2",
        "elements_per_frame": "12", "dense_macs_per_frame": "24",
    }]
    result = module.packed_atlif_service(rows, lanes=1, slots=10)
    assert result["packed_service_cycles"] == 4
    assert result["equal_resource_service_cycles"] == 3
    assert result["slot_packing_utilization"] == 0.75
