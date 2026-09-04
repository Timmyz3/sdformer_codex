#!/usr/bin/env python3
from simulate_m13_temporal_partial_retirement import simulate_patterns


def main() -> int:
    patterns = [{
        "temporal_steps": 2,
        "fanout": 96,
        "lane_tiles": 1,
        "chunks": 1,
        "steps": [
            {"descriptor_cycles": 4, "lane_compute_cycles": 8, "contexts": 4},
            {"descriptor_cycles": 4, "lane_compute_cycles": 8, "contexts": 4},
        ],
    }]
    full = simulate_patterns(patterns, group_count=2, scale=1.0, variant="full_context")
    cache = simulate_patterns(patterns, group_count=2, scale=1.0, variant="lane_cache")
    replay = simulate_patterns(patterns, group_count=2, scale=1.0, variant="lane_replay")
    assert full["atlif_tasks"] == cache["atlif_tasks"] == replay["atlif_tasks"] == 96
    assert full["descriptor_raw_cycles"] == cache["descriptor_raw_cycles"] == 16
    assert replay["descriptor_raw_cycles"] == 16  # one lane: replay has no amplification
    assert full["fused_finish_cycles"] <= cache["fused_finish_cycles"]
    assert cache["fused_finish_cycles"] > 0 and replay["fused_finish_cycles"] > 0
    two_lanes = [dict(patterns[0], fanout=192, lane_tiles=2)]
    replay_two = simulate_patterns(two_lanes, group_count=1, scale=1.0, variant="lane_replay")
    cache_two = simulate_patterns(two_lanes, group_count=1, scale=1.0, variant="lane_cache")
    assert replay_two["descriptor_raw_cycles"] == 16
    assert cache_two["descriptor_raw_cycles"] == 8
    assert replay_two["fused_finish_cycles"] >= cache_two["fused_finish_cycles"]
    print("PASS_M13_TEMPORAL_PARTIAL_RETIREMENT_EVENT_MODEL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
