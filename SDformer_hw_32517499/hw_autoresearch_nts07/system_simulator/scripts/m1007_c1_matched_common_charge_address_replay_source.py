#!/usr/bin/env python3
"""M1007 source-only C1 matched-charge/address-timed replay primitives.

This module never turns the frozen M528 CPU ratios into RTL cycles.  It freezes
the M505 dead-write-only parent recurrence, exposes a streaming per-cycle 1RW
parent trace, and provides generic common-charge, conflict, and lifetime gates
for parent/psum/weight/source/DMA/commit evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
M505_PATH = HERE / "analyze_m505_h67_liveness_aware_single_port_parent_scratch.py"
M505_SHA = "9d55d960d237a1940fb8e9efaa4e227a4ec1025489f80804d1c677e12bc9aced"
M504_PATH = HERE / "analyze_m504_h67_single_port_parent_scratch.py"
M504_SHA = "9a7586b096e5ffa47867a8c20f32f49a607a5724f5df835827b7a28f9d230a5e"
M528_RESULT = HW / "results/m528_h67_single_port_same_ledger_recompute_r4_20260827/m528_h67_single_port_same_ledger_recompute_result_r1.json"
M528_RESULT_SHA = "778c8e1bed6a19852c14bc61e00761f798008d67042b7a74efbaaffdde4b3de1"
M505_RESULT = HW / "results/m505_h67_liveness_aware_single_port_parent_scratch_r1_20260827/m505_h67_liveness_aware_single_port_parent_scratch_result_r1.json"
M505_RESULT_SHA = "b8a29f2fafc0e7d051d66ed206cd5c25efb866d4a1ab02082aa71bad4b14eb61"
ROWS = HW / "results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/m410r2_h67_q32_runtime_rows_32.memh"
ROWS_SHA = "6e03352b89eff1955825334b4dedd991db8c975a9ef6662fe0317e73ccfa8334"
M1000 = HW / "reviews/m1000_c1_same_ledger_storage_physical_closure_first_principles_r1_20260829"
M1000_ID = (
    "475dace8e8b8d7e3c40e6c252c2eea5e4f1ae228d7789bac26ea482fb58c6944",
    "5424a5a5c60d7040327cfcfca40e16f3eb28aa6de9504fed8b98c12304d05eac",
    "fd700b7f9e1497fb4ed7fda5f1c725c5408233a84238da6787a871e69892f4d5",
)
CONTRACT = HW / "contracts/m1007_m1000_c1_matched_common_charge_address_replay_source_contract_r1_20260829.json"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

SAMPLES, OPERATORS, PARTITIONS = 10, 4, 432
ROWS_PER_PHASE, ROW_TILE, BLOCKS = 3000, 64, 8
BYTES_PER_LINE = 9
COMMON_RESOURCES = ("psum", "weight", "source", "dma", "commit")
DESIGNS = ("candidate", "strongest_zero", "same_coordinate_bit")


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(path: Path) -> Any:
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key: " + key); out[key] = value
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + x)))


def verify_flat(directory: Path, identity: tuple[str, str, str]) -> None:
    review, manifest, outer = directory / "review.json", directory / "SHA256SUMS", directory / "SHA256SUMS.seal.sha256"
    require(sha256(review) == identity[0] and sha256(manifest) == identity[1] and
            sha256(outer) == identity[2], "M1000 identity drift")
    for line in manifest.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        require(sha256(directory / name) == expected, "M1000 member drift")
    expected, name = outer.read_text().split()
    require(name == "SHA256SUMS" and sha256(manifest) == expected, "M1000 outer drift")


def load_m505():
    require(sha256(M505_PATH) == M505_SHA and sha256(M504_PATH) == M504_SHA,
            "frozen M504/M505 analyzer drift")
    spec = importlib.util.spec_from_file_location("m1007_frozen_m505", M505_PATH)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module); return module


M505 = load_m505()


def validate_frozen(contract: Path = CONTRACT) -> dict[str, Any]:
    require(sha256(M528_RESULT) == M528_RESULT_SHA and
            sha256(M505_RESULT) == M505_RESULT_SHA and sha256(ROWS) == ROWS_SHA,
            "M528/M505/M410 frozen evidence drift")
    verify_flat(M1000, M1000_ID)
    value = strict_json(contract)
    require(value.get("status") == "PASS_M1007_SOURCE_ONLY__NO_FULL_REPLAY_NO_EDA" and
            value.get("launch_now") is False, "M1007 contract drift")
    require(sha256(HW / "docs/359_DATE终局冻结_20260813.md") == DOCS359_SHA,
            "docs359 drift")
    return {"status": "PASS_M1007_FROZEN_INPUTS", "contract_sha256": sha256(contract)}


def parent_cycle_trace(masks: Sequence[int]) -> Iterator[dict[str, Any]]:
    """Yield every cycle of the exact frozen M505 dead-write-only 1RW task."""
    masks_a = np.asarray(masks, dtype=np.uint16)
    residual, parent = M505.M504.cleanroom_subset(masks_a)
    original_pc = M505.M504.POPCOUNT[masks_a].astype(np.int32)
    residual_pc = M505.M504.POPCOUNT[residual].astype(np.int32)
    order = np.lexsort((np.arange(masks_a.size, dtype=np.int32), original_pc))
    active = [int(row) for row in order if int(masks_a[row]) != 0]
    requirements = [int(parent[row]) for row in active if int(parent[row]) >= 0]
    consumers = [i for i, row in enumerate(active) if int(parent[row]) >= 0]
    use_count = np.bincount(np.asarray(requirements, dtype=np.int64),
                            minlength=masks_a.size).astype(np.int32)
    remaining = use_count.copy()
    queue: list[int] = []; pending: int | None = None
    next_req = row_cursor = beat = cycle = 0
    written = np.zeros(masks_a.size, dtype=np.bool_)
    while row_cursor < len(active):
        row = active[row_cursor]; parent_id = int(parent[row])
        work = int(residual_pc[row])
        if parent_id >= 0 and work == 0: work = 1
        require(work > 0, "active row has zero work")
        parent_ready = parent_id < 0 or bool(queue and queue[0] == parent_id)
        final_if_issued = bool(parent_ready and beat + 1 == work)
        reserved = len(queue) + int(pending is not None)
        request_exists = next_req < len(requirements)
        requested_parent = requirements[next_req] if request_exists else -1
        requested_consumer = consumers[next_req] if request_exists else -1
        has_capacity = reserved < 2
        producer_ready = bool(request_exists and written[requested_parent])
        predicted_forward = bool(final_if_issued and request_exists and
                                 has_capacity and requested_parent == row)
        predicted_write = bool(final_if_issued and int(use_count[row]) > 0)
        hold = bool(predicted_write and request_exists and has_capacity and
                    producer_ready and requested_parent != row and
                    requested_consumer == row_cursor + 1)
        issue = bool(parent_ready and not hold)
        last = bool(issue and beat + 1 == work)
        forward = bool(last and request_exists and has_capacity and requested_parent == row)
        write = bool(last and int(use_count[row]) > 0)
        read = bool((not write) and (not forward) and request_exists and
                    has_capacity and written[requested_parent])
        require(not (read and write), "parent 1RW collision")
        consumed = bool(last and parent_id >= 0)
        queue_before = tuple(queue); pending_before = pending
        free_addr = None
        if consumed:
            require(queue and queue[0] == parent_id, "parent FIFO order drift")
            queue.pop(0); remaining[parent_id] -= 1
            require(remaining[parent_id] >= 0, "parent refcount underflow")
            if remaining[parent_id] == 0 and written[parent_id]: free_addr = parent_id
        if pending is not None:
            require(len(queue) < 2, "parent response overflow"); queue.append(pending)
        if forward:
            require(len(queue) < 2, "forward overflow")
            queue.append(requested_parent); next_req += 1
        next_pending = None
        if read:
            next_pending = requested_parent; next_req += 1
        pending = next_pending
        if write: written[row] = True
        op = "WRITE" if write else "READ" if read else "IDLE"
        address = row if write else requested_parent if read else None
        yield {"cycle": cycle, "resource": "parent", "op": op,
               "address": address, "issue": issue, "issue_row": row,
               "issue_beat": beat, "issue_last": last, "parent_id": parent_id,
               "forward": forward, "forward_address": requested_parent if forward else None,
               "free_address": free_addr, "queue_before": queue_before,
               "queue_after": tuple(queue), "pending_before": pending_before,
               "pending_after": pending, "hold": hold}
        if issue:
            if last: row_cursor += 1; beat = 0
            else: beat += 1
        cycle += 1
        require(cycle <= int(residual_pc.sum()) + len(active) + len(requirements) + 8,
                "parent trace failed bounded progress")
    require(next_req == len(requirements) and pending is None and not queue,
            "parent trace terminal drift")


def parent_summary(events: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    ops = Counter(event["op"] for event in events)
    return {"cycles": len(events), "macro_reads": ops["READ"],
            "macro_writes": ops["WRITE"],
            "idle_cycles": ops["IDLE"],
            "forwarded_reads": sum(bool(e["forward"]) for e in events),
            "issue_cycles": sum(bool(e["issue"]) for e in events),
            "stall_cycles": sum(not bool(e["issue"]) for e in events)}


def stream_parent_memh(rows_path: Path = ROWS) -> Iterator[dict[str, Any]]:
    """Memory-bounded full-ledger-capable stream; calling it does no work."""
    require(sha256(rows_path) == ROWS_SHA, "row ledger drift")
    fd = os.open(rows_path, os.O_RDONLY)
    try:
        for sample in range(SAMPLES):
            for operator in range(OPERATORS):
                for chunk in range((ROWS_PER_PHASE + ROW_TILE - 1) // ROW_TILE):
                    count = min(ROW_TILE, ROWS_PER_PHASE - chunk * ROW_TILE)
                    for partition in range(PARTITIONS):
                        phase = (sample * OPERATORS + operator) * PARTITIONS + partition
                        offset = (phase * ROWS_PER_PHASE + chunk * ROW_TILE) * BYTES_PER_LINE
                        raw = os.pread(fd, count * BYTES_PER_LINE, offset)
                        require(len(raw) == count * BYTES_PER_LINE, "short M410 tile read")
                        masks = [int(line, 16) & 0xffff for line in raw.splitlines()]
                        require(len(masks) == count, "M410 tile line drift")
                        tile_trace = list(parent_cycle_trace(masks))
                        tile_cycles = len(tile_trace)
                        for block in range(BLOCKS):
                            for event in tile_trace:
                                yield {**event, "sample": sample, "operator": operator,
                                       "chunk": chunk, "partition": partition,
                                       "block": block, "task_local_cycle":
                                           block * tile_cycles + event["cycle"]}
    finally:
        os.close(fd)


def logical_access_key(event: Mapping[str, Any]) -> tuple[Any, ...]:
    return (event["resource"], event["op"], event.get("bank"),
            event.get("address"), event.get("bytes"), event.get("transaction"))


def verify_matched_common_charge(per_design: Mapping[str, Sequence[Mapping[str, Any]]],
                                 policy: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    require(set(per_design) == set(DESIGNS), "three-design population drift")
    rows = {}
    for resource in COMMON_RESOURCES:
        require(resource in policy and policy[resource].get("mode") in
                ("include_both", "exclude_both"), "common-charge policy drift")
        specs = [policy[resource].get(key) for key in
                 ("capacity_bytes", "ports", "latency_cycles")]
        require(all(value is not None for value in specs), "common service spec incomplete")
        counters = {}
        for design in DESIGNS:
            counters[design] = Counter(logical_access_key(e) for e in per_design[design]
                                       if e["resource"] == resource)
        require(counters[DESIGNS[0]] == counters[DESIGNS[1]] == counters[DESIGNS[2]],
                "asymmetric common charge: " + resource)
        rows[resource] = {"mode": policy[resource]["mode"],
                          "logical_accesses": sum(counters[DESIGNS[0]].values()),
                          "identical_multiset": True,
                          "capacity_bytes": specs[0], "ports": specs[1],
                          "latency_cycles": specs[2]}
    return {"status": "PASS_M1007_MATCHED_COMMON_CHARGE", "resources": rows,
            "cycle_merge_pending": True,
            "note": "Logical access equality is necessary but not a measured total cycle."}


def analyze_1rw(events: Sequence[Mapping[str, Any]], group_fn,
                 initial_valid: bool = False) -> dict[str, Any]:
    by_cycle_group: dict[tuple[int, Any], list[Mapping[str, Any]]] = defaultdict(list)
    live: dict[tuple[Any, Any], int] = {}; peak = 0; max_lifetime = 0
    conflicts = Counter(); read_before_write = overwrite_live = 0
    banks_per_conflict = []
    for event in events:
        if event["op"] not in ("READ", "WRITE"): continue
        group = group_fn(event); key = (group, event.get("address"))
        by_cycle_group[(int(event["cycle"]), group)].append(event)
        if event["op"] == "WRITE":
            if key in live: overwrite_live += 1
            live[key] = int(event["cycle"]); peak = max(peak, len(live))
        else:
            if key not in live and not initial_valid: read_before_write += 1
            if key in live: max_lifetime = max(max_lifetime, int(event["cycle"]) - live[key])
        free = event.get("free_address")
        if free is not None: live.pop((group, free), None)
    for (_, _), group_events in by_cycle_group.items():
        if len(group_events) > 1:
            kinds = "".join(sorted(e["op"][0] for e in group_events))
            conflicts[kinds] += 1
            banks_per_conflict.append(sorted({e.get("bank") for e in group_events}))
    return {"accesses": sum(len(x) for x in by_cycle_group.values()),
            "conflict_cycles": sum(conflicts.values()),
            "conflict_types": dict(conflicts), "conflict_banks": banks_per_conflict,
            "read_before_write": read_before_write,
            "overwrite_while_live": overwrite_live,
            "peak_live_entries": peak, "maximum_observed_lifetime_cycles": max_lifetime,
            "terminal_live_entries": len(live), "one_rw_legal": not conflicts}


def packing_summary(psum_events: Sequence[Mapping[str, Any]],
                    weight_events: Sequence[Mapping[str, Any]],
                    coverage_complete: bool) -> dict[str, Any]:
    psum = analyze_1rw(psum_events, lambda e: int(e["bank"]) // 2,
                       initial_valid=True)
    weight = analyze_1rw(weight_events, lambda e: 0, initial_valid=True)
    half_slot_overlap = len({int(e["cycle"]) for e in weight_events if e["op"] in
                             ("READ", "WRITE") and int(e.get("bank", 0)) == 0} &
                            {int(e["cycle"]) for e in weight_events if e["op"] in
                             ("READ", "WRITE") and int(e.get("bank", 0)) == 1})
    admitted = bool(coverage_complete and psum["one_rw_legal"] and
                    weight["one_rw_legal"] and half_slot_overlap == 0)
    return {"schema": "m1007_214912B_packing_gate_v1",
            "coverage_complete": coverage_complete,
            "psum_depth_packed_pair": psum, "weight_single_group": weight,
            "weight_half_slot_overlap_cycles": half_slot_overlap,
            "capacity_only_214912B_admitted": admitted,
            "reason_if_blocked": None if admitted else
                "requires complete frozen trace and zero 1RW conflicts/half-slot overlap"}


def small_oracle() -> dict[str, Any]:
    cases = ([1, 3, 5], [1, 3, 7, 15], [3, 3, 3, 3],
             [1, 2, 3, 4, 5, 7, 15, 0])
    checked = []
    for masks in cases:
        events = list(parent_cycle_trace(masks)); summary = parent_summary(events)
        frozen = M505.simulate_liveness_task(np.asarray(masks, dtype=np.uint16), False)
        for ours, theirs in (("cycles", "liveness_cycles"),
                             ("macro_reads", "macro_reads"),
                             ("macro_writes", "macro_writes"),
                             ("forwarded_reads", "forwarded_reads"),
                             ("issue_cycles", "ideal_1r1w_issue_cycles"),
                             ("stall_cycles", "liveness_stall_cycles")):
            require(summary[ours] == frozen[theirs], "parent oracle drift: " + ours)
        require(all(sum(e["op"] == op for op in ("READ", "WRITE")) <= 1
                    for e in events), "parent 1RW double operation")
        checked.append(summary)
    common = [
        {"resource": "source", "op": "READ", "bank": 0, "address": 7,
         "bytes": 2, "transaction": "s0", "cycle": 0},
        {"resource": "weight", "op": "READ", "bank": 0, "address": 3,
         "bytes": 128, "transaction": "w0", "cycle": 1},
        {"resource": "psum", "op": "READ", "bank": 0, "address": 2,
         "bytes": 228, "transaction": "p0", "cycle": 2},
        {"resource": "dma", "op": "WRITE", "bank": 0, "address": 1,
         "bytes": 6144, "transaction": "d0", "cycle": 3},
        {"resource": "commit", "op": "WRITE", "bank": 0, "address": 9,
         "bytes": 228, "transaction": "c0", "cycle": 4},
    ]
    per = {name: [dict(e, cycle=e["cycle"] + i * 10) for e in common]
           for i, name in enumerate(DESIGNS)}
    policy = {r: {"mode": "include_both", "capacity_bytes": 1,
                  "ports": "1RW", "latency_cycles": 1} for r in COMMON_RESOURCES}
    matched = verify_matched_common_charge(per, policy)
    bad = {name: list(events) for name, events in per.items()}
    bad["strongest_zero"] = bad["strongest_zero"][:-1]
    try: verify_matched_common_charge(bad, policy)
    except RuntimeError: asymmetric_rejected = True
    else: asymmetric_rejected = False
    require(asymmetric_rejected, "asymmetric charge accepted")
    psum = [
        {"cycle": 5, "op": "READ", "bank": 0, "address": 1},
        {"cycle": 5, "op": "WRITE", "bank": 1, "address": 65},
    ]
    weight = [
        {"cycle": 7, "op": "READ", "bank": 0, "address": 2},
        {"cycle": 7, "op": "WRITE", "bank": 1, "address": 18},
    ]
    packing = packing_summary(psum, weight, coverage_complete=False)
    require(packing["psum_depth_packed_pair"]["conflict_cycles"] == 1 and
            packing["weight_single_group"]["conflict_cycles"] == 1 and
            not packing["capacity_only_214912B_admitted"], "packing oracle drift")
    return {"status": "PASS_M1007_SMALL_ORACLE__NO_FULL_REPLAY",
            "parent_cases": checked, "matched_common_charge": matched,
            "asymmetric_charge_rejected": asymmetric_rejected,
            "packing_negative_oracle": packing,
            "full_51840000_replayed": False, "eda_gpu_remote_used": False}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(); p.add_argument("--self-test", action="store_true")
    p.add_argument("--validate-source", action="store_true")
    p.add_argument("--contract", type=Path, default=CONTRACT)
    a = p.parse_args(argv)
    require(a.self_test ^ a.validate_source, "select exactly one source-safe mode")
    value = small_oracle() if a.self_test else validate_frozen(a.contract)
    print(json.dumps(value, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
