#!/usr/bin/env python3
"""Independent bounded semantic attacks for the exact M861 source identity.

This reviewer-owned harness never enters the full-row or production paths.
"""

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (ROOT / "system_simulator/scripts/"
          "analyze_m861_decoder_streaming_event_sweep.py")
EXPECTED = "f72ed3b820051d624699152b784c05fa674106556ab73f452a2cf96a9f72d7a4"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


assert sha256(SCRIPT) == EXPECTED
spec = importlib.util.spec_from_file_location("m865_exact_m861", SCRIPT)
assert spec is not None and spec.loader is not None
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def scheduled(index, earliest, dependency, issue, returned, reason):
    return M.ScheduledRequest(
        request_id="independent{}".format(index),
        transaction_id="independent",
        population_id="M865",
        config="TYPED_SIGNED_K8",
        kind="compute",
        addresses=(index,), banks=(0,), width_bytes=1,
        dependency_tokens=(), earliest_issue_cycle=earliest,
        dependency_ready_cycle=dependency, issue_cycle=issue,
        return_cycle=returned, commit_cycle=returned,
        wait_reason=reason, produces_token="")


def reference_cycle_classes(rows, total_cycles):
    result = {name: 0 for name in M.CYCLE_CLASS_ORDER}
    issue_cycles = {int(row.issue_cycle) for row in rows}
    for cycle in range(total_cycles):
        if cycle in issue_cycles:
            result["active_service"] += 1
            continue
        waiting = [row for row in rows
                   if row.earliest_issue_cycle <= cycle < row.issue_cycle]
        inflight = [row for row in rows
                    if row.issue_cycle <= cycle < row.return_cycle]
        dependency = (
            bool(inflight) or
            any(row.dependency_ready_cycle > cycle for row in waiting) or
            any(row.wait_reason == "dependency_completion" for row in waiting)
        )
        if dependency:
            result["dependency_completion"] += 1
        elif any(row.wait_reason == "weight_bank" for row in waiting):
            result["weight_bank"] += 1
        elif any(row.wait_reason == "psum_bank" for row in waiting):
            result["psum_bank"] += 1
        elif any(row.wait_reason == "memory" for row in waiting):
            result["memory"] += 1
        else:
            result["compute"] += 1
    return result


def attack_manual_priority():
    rows = [
        scheduled(0, 0, 2, 2, 4, "dependency_completion"),
        scheduled(1, 4, 4, 6, 6, "weight_bank"),
        scheduled(2, 7, 7, 9, 9, "psum_bank"),
        scheduled(3, 10, 10, 12, 12, "memory"),
        scheduled(4, 14, 14, 14, 14, "none"),
    ]
    sweep = M.ExactCycleClassSweep()
    for row in reversed(rows):
        sweep.observe(row)
    observed = sweep.finalize(15)
    expected = reference_cycle_classes(rows, 15)
    assert observed == expected
    assert all(observed[name] > 0 for name in M.CYCLE_CLASS_ORDER)
    assert sum(observed.values()) == 15
    return observed


def attack_interval_union():
    rng = random.Random(86501)
    checked = 0
    out_of_order = 0
    for _ in range(128):
        pairs = []
        covered = set()
        for _ in range(80):
            start = rng.randrange(0, 256)
            end = min(280, start + rng.randrange(0, 25))
            pairs.append((start, end))
            covered.update(range(start, end))
        rng.shuffle(pairs)
        union = M.IntervalUnion()
        for start, end in pairs:
            union.add(start, end)
        expanded = set()
        for start, end in union.intervals:
            expanded.update(range(start, end))
        assert expanded == covered
        assert union.cardinality == len(covered)
        out_of_order += union.out_of_order_insertions
        checked += len(pairs)
    assert out_of_order > 0
    return {"intervals_checked": checked,
            "out_of_order_insertions": out_of_order}


def attack_random_endpoint_sweeps():
    rng = random.Random(86502)
    comparisons = 0
    for trial in range(256):
        rows = []
        for index in range(rng.randrange(1, 65)):
            earliest = rng.randrange(0, 80)
            issue = earliest + rng.randrange(0, 16)
            dependency = rng.randrange(earliest, issue + 1)
            returned = issue + rng.randrange(0, 12)
            reason = rng.choice(("none", "compute", "dependency_completion",
                                 "weight_bank", "psum_bank", "memory"))
            rows.append(scheduled(index, earliest, dependency, issue,
                                  returned, reason))
        insertion = list(rows)
        rng.shuffle(insertion)
        total = max(row.return_cycle for row in rows) + 1
        sweep = M.ExactCycleClassSweep()
        for row in insertion:
            sweep.observe(row)
        assert sweep.finalize(total) == reference_cycle_classes(rows, total)
        comparisons += total
    return {"trials": 256, "cycles_compared": comparisons}


def attack_random_dag():
    requests = M.deterministic_random_dag(512, seed=86503)
    old_scheduler = M.M785.AddressTimedScheduler(M._synthetic_resource())
    new_scheduler = M.StreamingAddressTimedScheduler(M._synthetic_resource())
    old = old_scheduler.schedule(list(requests))
    new = new_scheduler.schedule(iter(requests), retain_details=True)
    equal = {}
    for field in M.M768_RESULT_FIELDS:
        equal[field] = old[field] == new[field]
        assert equal[field]
    assert old_scheduler.token_ready == new_scheduler.token_ready
    return {
        "requests": len(requests),
        "all_11_fields_equal": all(equal.values()),
        "field_equal": equal,
        "token_ready_entries": len(new_scheduler.token_ready),
        "token_ready_equal": True,
        "token_ready_sha256": M.M785.canonical_sha256(
            new_scheduler.token_ready),
    }


def attack_ports_and_response_reuse():
    base = M._synthetic_resource()
    rows = [
        M._request(0, "weight_read", earliest=0),
        M._request(8, "weight_write", earliest=0),
        M._request(6, "psum_read", earliest=0),
        M._request(12, "psum_write", earliest=0),
    ]
    old = M.M785.AddressTimedScheduler(base).schedule(rows)
    new = M.StreamingAddressTimedScheduler(base).schedule(
        iter(rows), retain_details=True)
    assert all(old[field] == new[field] for field in M.M768_RESULT_FIELDS)
    by_id = {row["request_id"]: row for row in new["scheduled_requests"]}
    assert by_id["r0"]["issue_cycle"] == by_id["r8"]["issue_cycle"]
    assert by_id["r12"]["issue_cycle"] > by_id["r6"]["issue_cycle"]

    external = M.M785.PortSpec(1, "1RW", 192, 2, 2, 1, 1)
    one = M.M785.CommonResource(
        lanes=base.lanes, accumulator_bits=base.accumulator_bits,
        clock_ns=base.clock_ns,
        external_bytes_per_cycle=base.external_bytes_per_cycle,
        onchip_budget_bytes_macro_rounded=
            base.onchip_budget_bytes_macro_rounded,
        macro_round_bytes=base.macro_round_bytes,
        weight_bytes_logical=base.weight_bytes_logical,
        psum_bytes_logical=base.psum_bytes_logical,
        descriptor_control_bytes_logical=base.descriptor_control_bytes_logical,
        reserved_unallocated_bytes=base.reserved_unallocated_bytes,
        weight=base.weight, psum=base.psum, external=external,
        compute=base.compute)
    exrows = [M._request(index + 100, "external_read", earliest=0)
              for index in range(4)]
    old_scheduler = M.M785.AddressTimedScheduler(one)
    new_scheduler = M.StreamingAddressTimedScheduler(one)
    old = old_scheduler.schedule(exrows)
    new = new_scheduler.schedule(iter(exrows), retain_details=True)
    assert all(old[field] == new[field] for field in M.M768_RESULT_FIELDS)
    issued = [row["issue_cycle"] for row in new["scheduled_requests"]]
    returned = [row["return_cycle"] for row in new["scheduled_requests"]]
    assert issued[1:] == returned[:-1]
    return {
        "weight_1r1w_parallel_issue": True,
        "psum_1rw_serialized": True,
        "outstanding_one_issue_cycles": issued,
        "outstanding_one_return_cycles": returned,
        "same_cycle_slot_reuse": True,
    }


class OneShotRequests:
    def __init__(self, count):
        self.count = count
        self.started = False

    def __iter__(self):
        assert not self.started, "request iterable consumed more than once"
        self.started = True
        return M.synthetic_prefix_requests(self.count)


def attack_no_detail_retention():
    scheduler = M.StreamingAddressTimedScheduler(M._synthetic_resource())
    result = scheduler.schedule(OneShotRequests(10000), retain_details=False)
    assert result["detail_retained"] is False
    assert "scheduled_requests" not in result
    assert "compressed_schedule" not in result
    assert not hasattr(scheduler, "scheduled_requests")
    assert not hasattr(scheduler, "compressed_schedule")
    source = inspect.getsource(M.StreamingAddressTimedScheduler.schedule)
    assert "list(requests)" not in source
    assert "rows = list" not in source
    assert "M785.M777.M768.compress_scheduled_rows" in source
    assert source.count("if detailed is not None") >= 2
    return {
        "requests": 10000,
        "one_shot_iterable": True,
        "result_scheduled_requests_absent": True,
        "result_compressed_schedule_absent": True,
        "scheduler_detail_attributes_absent": True,
        "source_rows_list_absent": True,
    }


def main():
    result = {
        "schema": "m865_independent_bounded_hammer_v1",
        "source_sha256": sha256(SCRIPT),
        "manual_priority": attack_manual_priority(),
        "interval_union": attack_interval_union(),
        "random_endpoint_sweep": attack_random_endpoint_sweeps(),
        "random_dag": attack_random_dag(),
        "ports": attack_ports_and_response_reuse(),
        "no_detail_retention": attack_no_detail_retention(),
        "full_first_row_invoked": False,
        "full_population_invoked": False,
        "production_invoked": False,
        "status": "PASS_INDEPENDENT_BOUNDED_ATTACKS",
    }
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
