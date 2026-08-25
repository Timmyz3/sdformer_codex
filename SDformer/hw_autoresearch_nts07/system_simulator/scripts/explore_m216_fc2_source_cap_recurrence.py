#!/usr/bin/env python3
"""Opportunity recurrence for source-cap matched K1/K8 replay.

The model deliberately follows pre-edge combinational decisions and the RTL's
nonblocking-update priorities.  It is first admitted only against a Synopsys
VCS continuous-source sweep; frozen-H67 replay is a separate evidence step.
"""

import argparse
import collections
import json
import re
from pathlib import Path


GEOMETRY = {1: (4, 2), 2: (8, 4), 4: (16, 8), 8: (32, 8)}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def event_pattern(beat, seed):
    value = 0
    b0, r0 = (beat * 5 + seed) % 8, (beat * 3 + seed * 2) % 12
    value |= 1 << (r0 * 8 + b0)
    b1, r1 = (beat * 3 + seed + 1) % 8, (beat * 7 + seed + 3) % 12
    if (beat + seed) % 2 == 0:
        value |= 1 << (r1 * 8 + b1)
    b2, r2 = (beat + seed * 3 + 2) % 8, (beat * 11 + seed + 5) % 12
    if (beat + 2 * seed) % 5 == 0:
        value |= 1 << (r2 * 8 + b2)
    return value


def sweep_payload(blocks, mode, seed):
    extent, _ = GEOMETRY[blocks]
    result = []
    for beat in range(extent):
        active = (mode == 0
                  or (mode == 1 and (beat * 7 + seed) % 5 < 2)
                  or (mode == 2 and beat < seed % (extent + 1)))
        result.append(event_pattern(beat, seed)
                      if active and mode != 3 else 0)
    return result


def bank_load(bitmap):
    result = [0] * 8
    for position in range(96):
        if bitmap & (1 << position):
            result[position % 8] += 1
    return result


def simulate_m216_bank_loads(payload_loads, depth, output_blocks,
                             queue_depth=8, source_cap=1):
    """Simulate one token from per-beat eight-bank event-count vectors.

    A zero beat is represented by ``None``.  Using already reduced bank loads
    keeps the frozen replay proportional to controller cycles rather than
    repeating a 96-bit scan inside every token.
    """
    extent = len(payload_loads)
    require((extent, depth) == GEOMETRY[output_blocks], "geometry drift")
    require(source_cap in (1, 8), "SOURCE_CAP must be exactly 1 or 8")

    # M216 terminal-hint compactor state after joint header acceptance.
    raw_position = 0
    window_fill = 0
    descriptor_total = 0
    queue = []
    raw_done = False
    compact_done_valid = False

    # M216 paired sink registered state after the same header edge.
    buffers = [
        {"closed": False, "entries": 0, "banks": [0] * 8},
        {"closed": False, "entries": 0, "banks": [0] * 8},
    ]
    fill_select = 0
    drain_select = 0
    upstream_done_seen = False
    accepted_descriptors = 0
    group = None

    cycles = 0
    maximum_queue = 0
    maximum_descriptor_hold = 0
    descriptor_hold_run = 0
    terminal_collapses = 0
    terminal_partial_closes = 0
    same_cycle_done_loads = 0

    while True:
        require(cycles < 100000, "M216 recurrence watchdog")
        cycles += 1

        # Terminal-hint compactor analysis of the held four-beat source.
        raw_valid = raw_position < extent
        packet = []
        next_window_fill = window_fill
        if raw_valid:
            for beat in range(raw_position, min(extent, raw_position + 4)):
                load = payload_loads[beat]
                if load is not None:
                    next_window_fill += 1
                    window_last = next_window_fill == depth
                    packet.append((load, beat, window_last))
                    if window_last:
                        next_window_fill = 0
        incoming_count = len(packet)
        fresh_mode = not queue and raw_valid and incoming_count != 0
        source = packet if fresh_mode else queue
        descriptor_count = min(4, len(source))
        for lane in range(descriptor_count):
            if source[lane][2]:
                descriptor_count = lane + 1
                break
        descriptors = list(source[:descriptor_count])
        descriptor_valid = bool(source)
        # M216 opportunity: upstream done is authoritative on its accepted
        # edge.  A previously terminal-hint-closed lone window may therefore
        # become load-eligible without first registering done_seen.
        same_cycle_done_fence = (compact_done_valid
                                 and not upstream_done_seen
                                 and not descriptor_valid)

        # M216 drain-side combinational view from pre-edge state.
        pair_first = drain_select
        pair_second = 1 - drain_select
        pair_has_two = (output_blocks != 1
                        and buffers[pair_first]["closed"]
                        and buffers[pair_second]["closed"])
        old_pair_available = (pair_has_two
                              or (output_blocks == 1
                                  and buffers[pair_first]["closed"])
                              or (upstream_done_seen
                                  and buffers[pair_first]["closed"]))
        pair_available = (old_pair_available
                          or (same_cycle_done_fence
                              and buffers[pair_first]["closed"]))
        candidate_banks = [
            buffers[pair_first]["banks"][bank]
            + (buffers[pair_second]["banks"][bank]
               if pair_has_two else 0)
            for bank in range(8)
        ]
        candidate_total = sum(candidate_banks)
        if source_cap == 8:
            candidate_selected = [
                (bank, pair_first
                 if buffers[pair_first]["banks"][bank] else pair_second)
                for bank, count in enumerate(candidate_banks) if count]
        else:
            candidate_selected = [
                (bank, pair_first) for bank in range(8)
                if buffers[pair_first]["banks"][bank]][:1]
            if not candidate_selected and pair_has_two:
                candidate_selected = [
                    (bank, pair_second) for bank in range(8)
                    if buffers[pair_second]["banks"][bank]][:1]
        candidate_count = len(candidate_selected)
        candidate_pair_last = candidate_count != 0 \
            and candidate_total == candidate_count
        handoff_banks = list(buffers[pair_second]["banks"])
        handoff_total = sum(handoff_banks)
        handoff_selected = [
            (bank, pair_second) for bank, count in enumerate(handoff_banks)
            if count]
        if source_cap == 1:
            handoff_selected = handoff_selected[:1]
        handoff_count = len(handoff_selected)
        handoff_pair_last = handoff_count != 0 \
            and handoff_total == handoff_count

        group_valid = group is not None
        group_final_accept = (group_valid
                              and group["output_block"] + 1
                              == output_blocks)
        pair_release = group_final_accept and group["pair_last"]
        terminal_pair_release = (pair_release and upstream_done_seen
                                 and (group["pair_has_two"]
                                      or (not buffers[pair_second]["closed"]
                                          and buffers[pair_second]["entries"]
                                          == 0)))
        candidate_load = (pair_available and candidate_count != 0
                          and (not group_valid
                               or (group_final_accept
                                   and not group["pair_last"])))
        stage0_handoff_load = (pair_release and output_blocks == 1
                               and buffers[pair_second]["closed"]
                               and handoff_count != 0)
        fill_window_releasing = pair_release \
            and fill_select == pair_first

        fill_entries_effective = (0 if fill_window_releasing
                                  else buffers[fill_select]["entries"])
        fill_blocked = (buffers[fill_select]["closed"]
                        and not fill_window_releasing)
        descriptor_packet_last = (descriptor_count != 0
                                  and descriptors[-1][2])
        raw_packet_last = (raw_valid
                           and raw_position
                           + min(4, extent - raw_position) == extent)
        descriptor_drains_visible_source = descriptor_count == (
            incoming_count if fresh_mode else len(queue))
        descriptor_token_last = (descriptor_valid
                                 and descriptor_drains_visible_source
                                 and (raw_done
                                      or (raw_packet_last
                                          and (fresh_mode
                                               or incoming_count == 0))))
        descriptor_storage_legal = (not fill_blocked
                                    and fill_entries_effective
                                    + descriptor_count <= 8
                                    and not (fill_entries_effective
                                             + descriptor_count == 8
                                             and not (descriptor_packet_last
                                                      or descriptor_token_last)))
        descriptor_ready = (descriptor_valid and not upstream_done_seen
                            and descriptor_storage_legal)
        descriptor_accept = descriptor_valid and descriptor_ready
        if descriptor_valid and not descriptor_ready:
            descriptor_hold_run += 1
            maximum_descriptor_hold = max(maximum_descriptor_hold,
                                          descriptor_hold_run)
        else:
            descriptor_hold_run = 0

        queue_pop = descriptor_count if descriptor_accept \
            and not fresh_mode else 0
        fresh_pop = descriptor_count if descriptor_accept \
            and fresh_mode else 0
        queue_available = queue_depth - len(queue) + queue_pop
        raw_accept = (raw_valid
                      and not (descriptor_valid and not descriptor_ready)
                      and incoming_count - fresh_pop <= queue_available)

        # Queue transition is evaluated even when only one side fires.
        next_queue = list(queue[queue_pop:])
        if raw_accept:
            next_queue.extend(packet[fresh_pop if fresh_mode else 0:])
        require(len(next_queue) <= queue_depth, "queue overflow")

        compact_done_accept = same_cycle_done_fence
        sink_done_empty = (upstream_done_seen
                           and not buffers[0]["closed"]
                           and not buffers[1]["closed"]
                           and buffers[0]["entries"] == 0
                           and buffers[1]["entries"] == 0
                           and not group_valid)
        token_done_accept = sink_done_empty or terminal_pair_release

        # Sequential update: pair release, then refill/close, done-close, and
        # candidate load follow the RTL's nonblocking assignment priorities.
        next_buffers = [
            {"closed": item["closed"], "entries": item["entries"],
             "banks": list(item["banks"])} for item in buffers
        ]
        next_fill_select = fill_select
        next_drain_select = drain_select
        next_upstream_done_seen = upstream_done_seen
        next_accepted_descriptors = accepted_descriptors
        next_group = None if group is None else dict(group)

        if pair_release:
            next_buffers[pair_first] = {
                "closed": False, "entries": 0, "banks": [0] * 8}
            if group["pair_has_two"]:
                next_buffers[pair_second] = {
                    "closed": False, "entries": 0, "banks": [0] * 8}
            else:
                next_drain_select = 1 - drain_select

        if descriptor_accept:
            target = fill_select
            base_entries = 0 if fill_window_releasing \
                else buffers[target]["entries"]
            base_banks = ([0] * 8 if fill_window_releasing
                          else list(buffers[target]["banks"]))
            for load, _beat, _last in descriptors:
                base_banks = [left + right
                              for left, right in zip(base_banks, load)]
            next_buffers[target]["entries"] = base_entries + descriptor_count
            next_buffers[target]["banks"] = base_banks
            next_accepted_descriptors += descriptor_count
            if descriptor_packet_last or descriptor_token_last:
                next_buffers[target]["closed"] = True
                next_fill_select = 1 - fill_select
            if descriptor_token_last and not descriptor_packet_last:
                terminal_partial_closes += 1

        if compact_done_accept:
            next_upstream_done_seen = True
            if (buffers[fill_select]["entries"] != 0
                    and not buffers[fill_select]["closed"]):
                next_buffers[fill_select]["closed"] = True
                next_fill_select = 1 - fill_select

        if candidate_load or stage0_handoff_load:
            load_selected = handoff_selected if stage0_handoff_load \
                else candidate_selected
            next_group = {
                "output_block": 0,
                "pair_last": (handoff_pair_last if stage0_handoff_load
                              else candidate_pair_last),
                "pair_has_two": False if stage0_handoff_load
                                  else pair_has_two,
            }
            # K8 takes one source per active bank.  K1 takes the first active
            # bank in the older window, then considers the successor.
            for bank, selected in load_selected:
                next_buffers[selected]["banks"][bank] -= 1
            # Count only loads caused by the new done-fence edge.  Stage-0 or
            # complete-pair loads that M212 already allowed are not M216 wins,
            # even when they happen to coincide with compact_done_accept.
            if (candidate_load and same_cycle_done_fence
                    and not old_pair_available):
                same_cycle_done_loads += 1
        elif group_final_accept:
            next_group = None
        elif group_valid:
            next_group["output_block"] += 1

        if terminal_pair_release:
            terminal_collapses += 1

        # Terminal-hint compactor registered update.
        queue = next_queue
        if raw_accept:
            raw_position += min(4, extent - raw_position)
            window_fill = next_window_fill
            descriptor_total += incoming_count
            if raw_position == extent:
                raw_done = True
        if ((raw_done or (raw_accept and raw_position == extent))
                and not next_queue):
            compact_done_valid = True
        if compact_done_accept:
            raw_done = False
            compact_done_valid = False
            descriptor_total = 0
            window_fill = 0
        maximum_queue = max(maximum_queue, len(queue))

        buffers = next_buffers
        fill_select = next_fill_select
        drain_select = next_drain_select
        upstream_done_seen = next_upstream_done_seen
        accepted_descriptors = next_accepted_descriptors
        group = next_group

        if token_done_accept:
            require(accepted_descriptors == sum(
                value is not None for value in payload_loads),
                    "descriptor conservation failure")
            return {
                "cycles": cycles,
                "maximum_queue": maximum_queue,
                "maximum_descriptor_hold": maximum_descriptor_hold,
                "terminal_collapses": terminal_collapses,
                "terminal_partial_closes": terminal_partial_closes,
                "same_cycle_done_loads": same_cycle_done_loads,
            }


def simulate_m216(payload, depth, output_blocks, queue_depth=8,
                  source_cap=1):
    """Return isolated-token cycles from integer 96-bit payload beats."""
    payload_loads = [bank_load(int(value)) if value else None
                     for value in payload]
    return simulate_m216_bank_loads(
        payload_loads, depth, output_blocks, queue_depth=queue_depth,
        source_cap=source_cap)


def validate_sweep(log_path):
    pattern = re.compile(
        r"M216TAIL source_cap=(\d+) blocks=(\d+) mode=(\d+) seed=(\d+) "
        r"descriptors=(\d+) measured=(\d+)")
    records = []
    mismatch = []
    by_shape = collections.Counter()
    for match in pattern.finditer(Path(log_path).read_text()):
        source_cap, blocks, mode, seed, descriptors, measured = map(
            int, match.groups())
        payload = sweep_payload(blocks, mode, seed)
        _, depth = GEOMETRY[blocks]
        modeled = simulate_m216(
            payload, depth, blocks, source_cap=source_cap)
        record = {
            "source_cap": source_cap, "blocks": blocks,
            "mode": mode, "seed": seed,
            "descriptors": descriptors, "vcs_cycles": measured,
            "modeled_cycles": modeled["cycles"],
            "maximum_queue": modeled["maximum_queue"],
            "maximum_descriptor_hold": modeled["maximum_descriptor_hold"],
            "terminal_collapses": modeled["terminal_collapses"],
            "terminal_partial_closes": modeled["terminal_partial_closes"],
            "same_cycle_done_loads": modeled["same_cycle_done_loads"],
        }
        records.append(record)
        by_shape[str(blocks)] += 1
        if measured != modeled["cycles"]:
            mismatch.append(record)
    require(len(records) == 256, "VCS sweep extent drift")
    return {
        "schema": "m216_rtl_control_recurrence_v1",
        "status": "PASS_EXACT_256_CASE_VCS" if not mismatch
                  else "FAIL_VCS_MISMATCH",
        "cases": len(records),
        "mismatches": len(mismatch),
        "source_caps": sorted({item["source_cap"] for item in records}),
        "cases_by_output_blocks": dict(sorted(by_shape.items())),
        "maximum_queue": max(item["maximum_queue"] for item in records),
        "maximum_descriptor_hold": max(
            item["maximum_descriptor_hold"] for item in records),
        "all_records": records,
        "mismatch_records": mismatch,
        "claim_boundary": {
            "synopsys_vcs_calibrated": not mismatch,
            "scope_matched_k1_k8": True,
            "frozen_h67_replayed": False,
            "complete_fc2": False,
            "physical_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite output")
    result = validate_sweep(args.sweep_log)
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "cases", "mismatches", "maximum_queue",
        "maximum_descriptor_hold")}, sort_keys=True))
    require(result["mismatches"] == 0, "M216 recurrence mismatch")


if __name__ == "__main__":
    main()
