#!/usr/bin/env python3
from __future__ import annotations

import copy

from simulate_m15_finite_credit_retirement import simulate_event_stream


STATE_HASH = "1" * 64
COST_HASH = "2" * 64


def pattern(cluster: int, sample: int = 0, contexts: int = 4, fanout: int = 96):
    return {
        "sample_id": sample,
        "sequence_key": "seq{}".format(sample),
        "producer": "conv",
        "edge": "atlif",
        "edge_kind": "direct_m4",
        "admitted_for_overlap": True,
        "producer_call_index": 0,
        "edge_call_index": 0,
        "version": 0,
        "version_identity_sha256": "3" * 64,
        "sample_cluster_id": cluster,
        "population_cluster_id": cluster,
        "cost_basis": "synthetic_exact",
        "scheduler_sufficient_statistics_sha256": STATE_HASH,
        "cost_source_sha256": COST_HASH,
        "fanout": fanout,
        "lane_tiles": (fanout + 95) // 96,
        "chunks": 1,
        "steps": [
            {"temporal_step": 0, "descriptor_cycles": contexts, "lane_compute_cycles": 8, "contexts": contexts},
            {"temporal_step": 1, "descriptor_cycles": contexts, "lane_compute_cycles": 8, "contexts": contexts},
        ],
    }


def main() -> int:
    c1 = simulate_event_stream(
        [pattern(10, contexts=1)], variant="full_context", context_slots=1, fifo_depth=2
    )
    assert (c1["producer_work_cycles"], c1["atlif_service_cycles"], c1["fused_finish_cycles"], c1["hidden_cycles"]) == (20, 12, 26, 6)
    assert (c1["p_done_tokens"], c1["max_fifo_occupancy"]) == (2, 1)
    c2 = simulate_event_stream(
        [pattern(11, contexts=2)], variant="full_context", context_slots=1, fifo_depth=2
    )
    assert (c2["producer_work_cycles"], c2["atlif_service_cycles"], c2["fused_finish_cycles"], c2["hidden_cycles"]) == (24, 24, 35, 13)
    assert c2["p_done_tokens"] == c2["consumer_started_tags"] == c2["consumer_finished_tags"] == 4

    groups = [pattern(0), pattern(1)]
    one_slot = simulate_event_stream(
        groups, variant="full_context", context_slots=1, fifo_depth=32
    )
    two_slots = simulate_event_stream(
        groups, variant="full_context", context_slots=2, fifo_depth=32
    )
    assert one_slot["p_done_tokens"] == one_slot["unique_tags"] == 16
    assert one_slot["atlif_service_cycles"] == 96
    assert one_slot["context_credit_stall_cycles"] > 0
    assert two_slots["context_credit_stall_cycles"] == 0
    assert two_slots["fused_finish_cycles"] < one_slot["fused_finish_cycles"]
    assert two_slots["hidden_cycles"] > one_slot["hidden_cycles"]
    shallow_fifo = simulate_event_stream(
        groups, variant="full_context", context_slots=2, fifo_depth=1
    )
    assert shallow_fifo["fifo_backpressure_stall_cycles"] > 0
    assert shallow_fifo["fused_finish_cycles"] >= two_slots["fused_finish_cycles"]
    assert shallow_fifo["max_fifo_occupancy"] == 1

    partial_tail = simulate_event_stream(
        [pattern(2, contexts=3, fanout=100)],
        variant="lane_replay", context_slots=1, fifo_depth=8,
    )
    assert partial_tail["p_done_tokens"] == 12
    assert partial_tail["atlif_service_cycles"] == 42
    assert partial_tail["state"]["atlif_state_bits"] == 3 * 96 * 2 * 24

    fenced = simulate_event_stream(
        [pattern(0, sample=0), pattern(0, sample=1), pattern(1, sample=0)],
        variant="full_context", context_slots=2, fifo_depth=32,
    )
    assert fenced["sample_fences"] == 2
    assert fenced["unique_tags"] == 24

    broken = copy.deepcopy(pattern(0))
    broken["steps"][1]["temporal_step"] = 2
    try:
        simulate_event_stream([broken], variant="full_context", context_slots=1, fifo_depth=8)
    except ValueError as exc:
        assert "temporal" in str(exc)
    else:
        raise AssertionError("missing timestep was admitted")
    duplicate = [pattern(0), pattern(0)]
    try:
        simulate_event_stream(duplicate, variant="full_context", context_slots=1, fifo_depth=8)
    except ValueError as exc:
        assert "duplicate population cluster" in str(exc)
    else:
        raise AssertionError("duplicate population cluster was admitted")
    early = pattern(0)
    early["steps"][0]["descriptor_cycles"] = 0
    try:
        simulate_event_stream([early], variant="full_context", context_slots=1, fifo_depth=8)
    except ValueError as exc:
        assert "descriptor completion" in str(exc)
    else:
        raise AssertionError("early P_DONE contract was admitted")
    early_lane = pattern(0)
    early_lane["steps"][0]["lane_compute_cycles"] = 1
    try:
        simulate_event_stream([early_lane], variant="full_context", context_slots=1, fifo_depth=8)
    except ValueError as exc:
        assert "PREP/DRAIN" in str(exc)
    else:
        raise AssertionError("early lane completion was admitted")
    join = pattern(0)
    join["edge_kind"] = "join_with_m4"
    try:
        simulate_event_stream([join], variant="full_context", context_slots=1, fifo_depth=8)
    except ValueError as exc:
        assert "direct_m4" in str(exc)
    else:
        raise AssertionError("join edge was admitted")
    try:
        simulate_event_stream([], variant="full_context", context_slots=1, fifo_depth=8)
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("empty event stream was admitted")
    try:
        simulate_event_stream([pattern(0)], variant="full_context", context_slots=1, fifo_depth=8, output_lanes=64)
    except ValueError as exc:
        assert "96 lanes" in str(exc)
    else:
        raise AssertionError("non-96 output width was admitted")
    same_prototype = pattern(20)
    second_population = copy.deepcopy(same_prototype)
    second_population["population_cluster_id"] = 21
    second_population["fanout"] = 192
    second_population["lane_tiles"] = 2
    try:
        simulate_event_stream(
            [same_prototype, second_population], variant="full_context",
            context_slots=2, fifo_depth=8,
        )
    except ValueError as exc:
        assert "prototype ID" in str(exc)
    else:
        raise AssertionError("prototype ID drift was admitted")
    lane_cache = simulate_event_stream(
        [pattern(30)], variant="lane_cache", context_slots=1, fifo_depth=32
    )
    lane_replay = simulate_event_stream(
        [pattern(30)], variant="lane_replay", context_slots=1, fifo_depth=32
    )
    expected_descriptors = sum(step["descriptor_cycles"] for step in pattern(30)["steps"])
    expected_lane_work = sum(step["lane_compute_cycles"] for step in pattern(30)["steps"])
    expected_output_work = lane_cache["p_done_tokens"]
    assert lane_cache["producer_work_cycles"] == expected_descriptors + expected_lane_work + expected_output_work
    assert lane_replay["producer_work_cycles"] == expected_descriptors + expected_lane_work + expected_output_work
    assert lane_cache["tag_encoding"]["required_bits"] <= lane_cache["tag_encoding"]["configured_bits"]
    print("PASS_M15_FINITE_CREDIT_RETIREMENT 18/18")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
