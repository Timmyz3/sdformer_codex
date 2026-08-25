#!/usr/bin/env python3
"""Focused deterministic contracts for the M45-r2 producer."""

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / (
    "system_simulator/scripts/analyze_m45_r2_context8_primary_schedule.py")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def must_raise_value_error(function, *args):
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError("expected fail-closed ValueError")


def test_m45_r2_contract_and_capacity_are_pinned():
    module = load_module(SCRIPT, "m45_r2_test_contract")
    contract = module.validate_contract()
    capacity = contract["capacity_model"]
    assert capacity["context_bytes_per_entry"] == 352
    assert capacity["eight_context_bytes"] == 2816
    assert capacity["extra_state_bytes_vs_four_contexts"] == 1408
    assert capacity["combined_local_capacity_bytes"] == 150656
    assert capacity["local_capacity_headroom_bytes"] == 43072
    assert capacity["partial_frame_spill_permitted"] is False
    assert capacity["external_accumulator_backing_permitted"] is False


def test_m45_r2_configuration_roles_are_fail_closed():
    module = load_module(SCRIPT, "m45_r2_test_roles")
    contract = module.validate_contract()
    configs = {item["name"]: item for item in contract["configurations"]}
    assert configs["K1_CTX4_REPRODUCTION"]["destination_fanout_k"] == 1
    assert configs["K2_CTX8_PRIMARY"]["resident_contexts"] == 8
    assert configs["K2_CTX4_CAPACITY_ABLATION"]["resident_contexts"] == 4
    assert "KILLED" in configs["K4_CTX4_KILLED_ABLATION"]["name"]
    assert contract["capacity_model"][
        "context_increment_must_receive_later_rtl_area_gate"] is True


def test_m45_r2_uses_response_ready_release_and_complete_vector_fifo():
    module = load_module(SCRIPT, "m45_r2_test_physical")
    contract = module.validate_contract()
    schedule = contract["frozen_schedule"]
    capacity = contract["capacity_model"]
    assert schedule["context_release"].startswith("response_ready")
    assert capacity["complete_fifo_vector_storage_bytes_per_entry"] == 288
    assert capacity["complete_fifo_tag_control_bytes_per_entry"] == 16
    assert capacity["complete_fifo_storage_bytes"] == 16 * (288 + 16)


def test_m45_r1_sample0_diagnostic_remains_no_go_not_all10():
    module = load_module(SCRIPT, "m45_r2_test_diagnostic")
    contract = module.validate_contract()
    item = contract["inputs"]["m45_r1_sample0_diagnostic"]
    diagnostic = json.loads((module.HW_ROOT / item["path"]).read_text())
    assert diagnostic["status"] == "NO_GO_R1_SAMPLE0_ONLY_NOT_ALL10"
    assert diagnostic["scope"]["all10_result"] is False
    assert diagnostic["r1_kill_gate_diagnostic"][
        "k2_ctx4_per_sample_overhead_le_10pct"] is False
    assert diagnostic["r1_kill_gate_diagnostic"][
        "ctx8_improvement_over_ctx4_le_3pct_saturation"] is False


def test_m47_is_only_a_deferred_non_claiming_interface():
    module = load_module(SCRIPT, "m45_r2_test_deferred")
    contract = module.validate_contract()
    deferred = contract["deferred_interface"]
    assert deferred["milestone"] == "M47_NOT_PART_OF_M45_R2"
    assert "68400-byte packed frame" in deferred["required_future_audit"]
    assert contract["frozen_schedule"]["weight_replays_per_sample"] == 10


def test_m45_r2_negative_contract_identity_and_geometry_fail_closed():
    module = load_module(SCRIPT, "m45_r2_test_negative")
    expected = module.EXPECTED_CONTRACT_SHA256
    module.EXPECTED_CONTRACT_SHA256 = "0" * 64
    must_raise_value_error(module.validate_contract)
    module.EXPECTED_CONTRACT_SHA256 = expected
    r1 = module.load_r1()
    m43 = r1.load_m43_module()
    must_raise_value_error(r1.delayed_pair_cycles, m43, 0, 0, 0)
    must_raise_value_error(r1.schedule_tile_timestep,
                           m43, [], [], 9, 8, 0, 0, 0)
