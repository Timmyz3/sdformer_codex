#!/usr/bin/env python3
"""Synthetic event-order proof for the M863 held-final TB observation."""

import json


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(events):
    forced = False
    settled = False
    accepted = 0
    released = False
    psum = 0
    rows = 0
    preedge_observed = False
    post_accept_ready_sampled = False
    for event in events:
        if event == "force_authoritative_slot_and_sinks":
            require(not forced and accepted == 0, "force order")
            forced = True
        elif event == "combinational_settle":
            require(forced and not settled and accepted == 0, "settle order")
            settled = True
        elif event == "observe_valid_ready_preedge":
            require(forced and settled and accepted == 0, "preedge observation order")
            preedge_observed = True
        elif event == "posedge_accept":
            require(forced and preedge_observed and not released, "accept order")
            accepted += 1
            psum += 1
            rows += 1
        elif event == "sample_ready_post_accept":
            post_accept_ready_sampled = True
        elif event == "negedge_release":
            require(accepted == 1 and forced and not released, "release order")
            released = True
            forced = False
        elif event == "check_exact_deltas_and_emit_cover":
            require(released and accepted == 1 and psum == 1 and rows == 1,
                    "terminal exact-delta gate")
            require(not post_accept_ready_sampled, "post-handshake ready resample")
        else:
            raise RuntimeError("unknown event " + event)
    return accepted, psum, rows


def expect_fail(events):
    try:
        run(events)
    except RuntimeError:
        return True
    return False


def main():
    canonical = [
        "force_authoritative_slot_and_sinks",
        "combinational_settle",
        "observe_valid_ready_preedge",
        "posedge_accept",
        "negedge_release",
        "check_exact_deltas_and_emit_cover",
    ]
    require(run(canonical) == (1, 1, 1), "canonical synthetic order")
    negatives = {
        "old_post_accept_ready_resample": canonical[:4] + [
            "sample_ready_post_accept", "negedge_release",
            "check_exact_deltas_and_emit_cover"],
        "release_before_accept": canonical[:3] + [
            "negedge_release", "posedge_accept",
            "check_exact_deltas_and_emit_cover"],
        "double_accept": canonical[:4] + [
            "posedge_accept", "negedge_release",
            "check_exact_deltas_and_emit_cover"],
        "accept_without_preedge_observation": [
            "force_authoritative_slot_and_sinks", "combinational_settle",
            "posedge_accept", "negedge_release",
            "check_exact_deltas_and_emit_cover"],
    }
    require(all(expect_fail(events) for events in negatives.values()),
            "one or more synthetic negative mutations escaped")
    print(json.dumps({
        "schema": "m863_m533_tb_r10_held_final_event_order_model_v1",
        "status": "PASS_SYNTHETIC_PREEDGE_HANDSHAKE_ONE_ACCEPT_INACTIVE_RELEASE",
        "canonical_events": canonical,
        "accepted_edges": 1,
        "psum_delta": 1,
        "row_delta": 1,
        "negative_mutations_rejected": sorted(negatives),
        "vcs_or_simv_executed": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
