from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/analyze_m4_stateful_wall_cycles.py"
)
SPEC = spec_from_file_location("m4_stateful_wall_cycles", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeWall:
    @staticmethod
    def ordered_row_bundles(records):
        return [[index] for index in range(len(records))]


def test_count_state_outputs_charges_lane_tiles_and_motion():
    records = [
        {"sample_id": "0", "output_channel_fanout": "96", "row_use_motion": "false"},
        {"sample_id": "0", "output_channel_fanout": "192", "row_use_motion": "true"},
        {"sample_id": "1", "output_channel_fanout": "97", "row_use_motion": "true"},
    ]
    local = MODULE.count_state_outputs(FakeWall, records, line="local", output_lanes=96)
    hybrid = MODULE.count_state_outputs(
        FakeWall, records, line="hybrid", output_lanes=96
    )
    assert local["local_outputs"] == 5
    assert local["motion_outputs"] == 0
    assert hybrid["local_outputs"] == 1
    assert hybrid["motion_outputs"] == 4
    assert hybrid["per_sample"]["0"]["motion_outputs"] == 2
    assert hybrid["per_sample"]["1"]["motion_outputs"] == 2


def test_compose_state_cost_adds_exactly_one_cycle_per_motion_output():
    kernel = {
        "output_cycles": 5,
        "m4_wall_cycles": 20,
        "p1_sparse_wall_cycles": 80,
        "same_width_dense_wall_cycles": 60,
        "per_sample": {
            "0": {
                "output_cycles": 3,
                "m4_wall_cycles": 12,
                "p1_sparse_wall_cycles": 48,
                "same_width_dense_wall_cycles": 36,
            },
            "1": {
                "output_cycles": 2,
                "m4_wall_cycles": 8,
                "p1_sparse_wall_cycles": 32,
                "same_width_dense_wall_cycles": 24,
            },
        },
    }
    counts = {
        "outputs": 5,
        "local_outputs": 1,
        "motion_outputs": 4,
        "per_sample": {
            "0": {"local_outputs": 1, "motion_outputs": 2},
            "1": {"local_outputs": 0, "motion_outputs": 2},
        },
    }
    result = MODULE.compose_state_cost(kernel, counts)
    assert result["stateful_nonoverlap_cycles_upper_bound"] == 24
    assert result["state_bank_reads"] == 24
    assert result["state_bank_writes"] == 30
    assert result["per_sample"]["0"]["stateful_nonoverlap_cycles_upper_bound"] == 14
    assert result["per_sample"]["1"]["stateful_nonoverlap_cycles_upper_bound"] == 10
