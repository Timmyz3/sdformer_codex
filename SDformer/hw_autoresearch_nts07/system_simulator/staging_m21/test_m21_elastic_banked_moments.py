#!/usr/bin/env python3
"""Focused arithmetic tests for M21 elastic FIFO scheduling."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_m21_elastic_banked_moments import (
    advance_idle, emit_packet, fair_serialized_region_cycles,
)


def new_state() -> dict[str, int]:
    return {
        "source_cycles": 0, "producer_stall_cycles": 0,
        "moment_work": 0, "packets": 0, "moment_service_cycles": 0,
        "maximum_resident_packets": 0,
    }


def main() -> int:
    tests = 0

    state = new_state()
    for _ in range(4):
        emit_packet(state, service_cycles=6, resident_packet_capacity=None)
    assert state["source_cycles"] == 4
    assert state["moment_work"] == 21
    assert state["maximum_resident_packets"] == 4
    assert state["moment_service_cycles"] == 24
    assert state["source_cycles"] + state["moment_work"] == 25
    tests += 1

    row = {
        "batchnorm": {"elements": 960},
        "source_output_cycles_at_96_lanes": 10,
        "source_cycles": {"local": 100},
        "one_way_memory_cycles": 20,
        "consumer_cycles": 60,
    }
    assert fair_serialized_region_cycles(
        [row], arithmetic_tiles=1, variant="local",
    ) == 210  # 100 + (60-10) + 60
    assert fair_serialized_region_cycles(
        [row], arithmetic_tiles=3, variant="local",
    ) == 170  # 100 + (20-10) + 60
    assert fair_serialized_region_cycles(
        [row], arithmetic_tiles=6, variant="local",
    ) == 160  # 100 + (10-10) + 60
    tests += 1

    state = new_state()
    for _ in range(4):
        emit_packet(state, service_cycles=6, resident_packet_capacity=1)
    assert state["producer_stall_cycles"] == 15
    assert state["source_cycles"] == 19
    assert state["moment_work"] == 6
    assert state["maximum_resident_packets"] == 1
    assert state["source_cycles"] + state["moment_work"] == 25
    tests += 1

    state = new_state()
    emit_packet(state, service_cycles=6, resident_packet_capacity=2)
    advance_idle(state, 20)
    assert state["moment_work"] == 0
    emit_packet(state, service_cycles=6, resident_packet_capacity=2)
    assert state["producer_stall_cycles"] == 0
    assert state["maximum_resident_packets"] == 1
    tests += 1

    state = new_state()
    for _ in range(4):
        emit_packet(state, service_cycles=1, resident_packet_capacity=1)
    assert state["producer_stall_cycles"] == 0
    assert state["moment_work"] == 1
    assert state["source_cycles"] + state["moment_work"] == 5
    tests += 1

    try:
        emit_packet(new_state(), service_cycles=0, resident_packet_capacity=1)
    except ValueError as exc:
        assert "service" in str(exc)
    else:
        raise AssertionError("zero-cycle moment service was admitted")
    tests += 1

    print("PASS m21-elastic-banked-moments {}/{}".format(tests, tests))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
