#!/usr/bin/env python3
"""Read-only, cycle-by-cycle payload regression for the M2260 service model.

Run with a modern Python and NumPy, for example:
  PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python \
    hw_autoresearch_nts07/system_simulator/scripts/m2260_hot_parent_payload_regression.py

The production function supplies arbitration events, not payloads. This checker
independently owns physical hot-slot arrays, backing words, response snapshots,
consumer identities, reference counts, FIFO age, and signed arithmetic. It checks
every completed row against a dense oracle and reconciles every service counter.
An AST-located trace hook observes the point before production mutates its queue;
it does not change the production function or depend on a hard-coded line number.

This is an event-to-payload refinement check, NOT an independent arbitration
oracle, RTL simulation, timing/area result, or sink-backpressure test. The model
permits two old-value hot reads plus a write on one edge. Snapshot/consumer-corruption
negative controls prove that the payload checker is not merely comparing tags.
No ledger, output file, contract, or hash is written.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import dataclass
import inspect
import json
import sys
import textwrap

import numpy as np

import m2260_c1_hot_parent_probe as production


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def encode(value):
    require(np.all((-2048 <= value) & (value <= 2047)), "12-bit overflow")
    return (value & 4095).astype(np.uint16)


def decode(words):
    unsigned = words.astype(np.int64)
    return np.where(unsigned & 2048, unsigned - 4096, unsigned)


@dataclass
class Response:
    parent: int
    consumer: int
    ordinal: int
    payload: np.ndarray
    issued: int


@dataclass
class HotSlot:
    payload: np.ndarray
    row: int = -1
    birth: int = -1


def event_line():
    """Fail closed if the production pre-mutation observation point changes."""
    source, start = inspect.getsourcelines(production.serve_hot)
    tree = ast.parse(textwrap.dedent("".join(source)))
    target = ast.dump(ast.parse("last and p >= 0", mode="eval").body)
    matches = [node for node in ast.walk(tree)
               if isinstance(node, ast.If) and ast.dump(node.test) == target
               and any(isinstance(child, ast.Call)
                       and isinstance(child.func, ast.Attribute)
                       and isinstance(child.func.value, ast.Name)
                       and child.func.value.id == "queue"
                       and child.func.attr == "pop"
                       for child in ast.walk(node))]
    require(len(matches) == 1, "production queue-update hook is ambiguous/missing")
    return start + matches[0].lineno - 1


class PayloadReplay:
    def __init__(self, masks, residual, parent, order, slots, weights,
                 mutation=None):
        self.masks, self.parent, self.order = masks, parent, list(order)
        self.bits = {r: [b for b in range(16) if int(residual[r]) >> b & 1]
                     for r in order}
        self.weights = weights
        self.dense = {r: sum((weights[b] for b in range(16)
                             if int(masks[r]) >> b & 1),
                            np.zeros(96, dtype=np.int64)) for r in order}
        self.edges = [(int(parent[r]), r) for r in order if parent[r] >= 0]
        self.remaining = Counter(p for p, _ in self.edges)
        self.slots = [HotSlot(np.zeros(96, dtype=np.uint16))
                      for _ in range(slots)]
        self.backing = {}
        self.queue = []
        self.pending = None
        self.accumulator = np.zeros(96, dtype=np.int64)
        self.request = self.consumed = self.completed = 0
        self.count = Counter()
        self.coverage = Counter()
        self.mutation, self.injected = mutation, False

    def hot(self):
        return {slot.row: slot for slot in self.slots if slot.row >= 0}

    def check_response(self, response):
        require((response.parent, response.consumer) == self.edges[response.ordinal],
                "response consumer/parent/ordinal mismatch")
        require(np.array_equal(decode(response.payload), self.dense[response.parent]),
                "response snapshot payload mismatch")
        for slot in self.slots:
            require(not np.shares_memory(response.payload, slot.payload),
                    "response aliases a reusable hot slot")
        for data in self.backing.values():
            require(not np.shares_memory(response.payload, data),
                    "response aliases backing storage")

    def response(self, words):
        p, consumer = self.edges[self.request]
        result = Response(p, consumer, self.request, words.copy(),
                          self.count["cycles"])
        self.check_response(result)
        return result

    def step(self, state):
        f = state
        hot = self.hot()
        require(set(hot) == set(f["hot"]), "hot-tag state drift")
        require(set(self.backing) == set(f["written"]), "backing-valid state drift")
        require(self.remaining == f["remaining"], "reference-count state drift")
        require([r.parent for r in self.queue] == f["queue"], "queue state drift")
        require((None if self.pending is None else self.pending.parent) == f["pending"],
                "pending-tag state drift")
        require(self.request == f["request"], "request prefix drift")
        require(self.completed == f["cursor"], "completion prefix drift")
        for response in self.queue:
            self.check_response(response)
        if self.pending is not None:
            if self.mutation == "pending_payload" and not self.injected:
                self.pending.payload[0] ^= 1
                self.injected = True
            self.check_response(self.pending)
            require(self.pending.issued + 1 == self.count["cycles"],
                    "SRAM response is not exactly one cycle after read")
            self.coverage["pending_snapshot_returns"] += 1

        row, p = f["row"], f["p"]
        if self.request < len(self.edges):
            require(self.edges[self.request] == (f["asked"], self.order[f["consumer"]]),
                    "production request consumer differs from independent edge list")
        ready = p < 0 or bool(self.queue and self.queue[0].parent == p
                             and self.queue[0].consumer == row)
        require(ready == f["ready"], "parent readiness mismatch")
        require(not (f["read"] and f["write"]), "1RW collision")
        require(sum(bool(f[name]) for name in ("read", "forward", "hot_read")) <= 1,
                "multiple new prefetch requests on one cycle")

        value = None
        if f["issue"]:
            if f["beat"] == 0:
                self.accumulator.fill(0)
            if self.bits[row]:
                self.accumulator += self.weights[self.bits[row][f["beat"]]]
        if f["last"]:
            value = self.accumulator.copy()
            if p >= 0:
                require(self.queue[0].ordinal == self.consumed,
                        "response consumed out of order")
                value += decode(self.queue[0].payload)
            require(np.array_equal(value, self.dense[row]), "dense row payload mismatch")
            value = encode(value)
            self.coverage["completed_rows"] += 1
            self.coverage["scalar_comparisons"] += 96
            self.coverage["negative_limit_lanes"] += int(np.count_nonzero(decode(value) == -2048))
            self.coverage["positive_int8_sum_limit_lanes"] += int(np.count_nonzero(decode(value) == 2032))

        # Capture ALL reads from pre-edge storage, before any slot is reused.
        next_pending = self.response(self.backing[f["asked"]]) if f["read"] else None
        fresh = None
        if f["forward"]:
            fresh = self.response(value)
        elif f["hot_read"]:
            fresh = self.response(hot[f["asked"]].payload)

        retiring = p if f["last"] and p >= 0 and self.remaining[p] == 1 else -1
        admit = bool(f["last"] and self.remaining[row] > 0)
        survivors = [slot for slot in self.slots
                     if slot.row >= 0 and slot.row != retiring]
        victim = min(survivors, key=lambda slot: slot.birth) if (
            admit and len(survivors) == len(self.slots)) else None
        if f["last"]:
            require((victim.row if victim else -1) == f["victim"], "FIFO victim drift")
        expected_write = victim is not None and victim.row not in self.backing
        require(bool(f["write"]) == expected_write, "spill decision mismatch")
        if expected_write:
            self.backing[victim.row] = victim.payload.copy()
            self.coverage["spill_writes"] += 1
            if f["hot_read"]:
                self.coverage["spill_plus_hot_read"] += 1
                kind = "same_slot" if victim.row == f["asked"] else "different_slots"
                self.coverage["spill_plus_hot_read_" + kind] += 1

        prior_pending = self.pending
        if f["last"] and p >= 0:
            self.queue.pop(0)
            self.consumed += 1
        if prior_pending is not None:
            self.queue.append(prior_pending)
        if fresh is not None:
            self.queue.append(fresh)
        if prior_pending is not None and fresh is not None:
            self.coverage["dual_enqueue"] += 1
        if next_pending is not None or fresh is not None:
            self.request += 1
        self.pending = next_pending
        require(len(self.queue) + int(self.pending is not None) <= 2,
                "physical response queue over-reserved")

        if f["last"]:
            if p >= 0:
                self.remaining[p] -= 1
            if retiring in hot:
                hot[retiring].row = -1
                self.count["hot_releases"] += 1
            if admit:
                target = victim if victim is not None else next(
                    slot for slot in self.slots if slot.row < 0)
                old_row = target.row
                target.payload[:] = value  # In-place physical reuse, not dict rebinding.
                target.row, target.birth = row, self.count["cycles"]
                self.count["hot_writes"] += 1
                if old_row >= 0 and any(r.parent == old_row for r in self.queue):
                    self.coverage["slot_reuse_with_queued_snapshot"] += 1
                    if self.mutation == "queued_payload" and not self.injected:
                        next(r for r in self.queue if r.parent == old_row).payload[0] ^= 1
                        self.injected = True
            self.completed += 1
        if self.mutation == "consumer" and self.queue and not self.injected:
            self.queue[0].consumer = -1
            self.injected = True
        # Recheck snapshots after in-place hot-slot overwrite on the SAME edge.
        for response in self.queue:
            self.check_response(response)
        if self.pending is not None:
            self.check_response(self.pending)
        self.count.update(cycles=1, issues=int(f["issue"]), stalls=int(not f["issue"]),
                          reads=int(f["read"]), writes=int(f["write"]),
                          forwards=int(f["forward"]), deadline_holds=int(f["hold"]),
                          hot_reads=int(f["hot_read"]), spill_writes=int(f["write"]),
                          hot_spill_reads=int(f["write"]))

    def run(self, masks, residual, parent, order, slots):
        line = event_line()

        def trace(frame, event, arg):
            if frame.f_code is not production.serve_hot.__code__:
                return None
            if event == "line" and frame.f_lineno == line:
                self.step(frame.f_locals)
            return trace

        previous_trace = sys.gettrace()
        require(previous_trace is None, "run outside debugger/coverage tracing")
        sys.settrace(trace)
        try:
            actual = production.serve_hot(masks, residual, parent, order, slots)
        finally:
            sys.settrace(previous_trace)
        require(not self.hot() and not self.queue and self.pending is None,
                "payload storage did not drain")
        require(not any(self.remaining.values()), "remaining references did not drain")
        require(self.completed == len(order) and self.consumed == len(self.edges)
                and self.request == len(self.edges), "incomplete row/edge coverage")
        keys = set(actual) | set(self.count)
        require(all(actual.get(k, 0) == self.count[k] for k in keys),
                "service counter drift: " + repr((actual, dict(self.count))))
        return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=9502260)
    parser.add_argument("--random-cases", type=int, default=180)
    args = parser.parse_args()
    require(args.random_cases >= 0, "random-cases must be nonnegative")
    rng = np.random.default_rng(args.seed)
    cases = [[0] * 64, [3] * 64, [65535] * 64,
             [(1 << bits) - 1 for bits in range(1, 17)],
             [1, 3, 5, 9, 17, 33, 65, 129],
             [1, 2, 3, 4, 5, 7, 15, 0]]
    for _ in range(args.random_cases):
        width = int(rng.integers(2, 17))
        cases.append(rng.integers(0, 1 << width, size=64, dtype=np.uint16))
    counters = Counter()
    coverage = Counter()
    killed = {}
    replays = 0
    for index, raw in enumerate(cases):
        masks = np.asarray(raw, dtype=np.uint16)
        residual, parent = production.base.old.M504.cleanroom_subset(masks)
        stable, _ = production.base.forest_orders(masks, parent)
        threaded, _ = production.threaded_order(masks, parent)
        weights = rng.integers(-128, 128, size=(16, 96), dtype=np.int64)
        weights[:, :3] = [-128, 127, 0]  # Exact lower bound, upper sum, and zero lanes.
        for label, order in (("stable", stable), ("threaded", threaded)):
            replay = PayloadReplay(masks, residual, parent, order, 2, weights)
            try:
                counters.update(replay.run(masks, residual, parent, order, 2))
            except AssertionError as error:
                raise AssertionError(
                    f"seed={args.seed}, case={index}, order={label}: {error}") from error
            coverage.update(replay.coverage)
            replays += 1
            for mutation in ("pending_payload", "queued_payload", "consumer"):
                if mutation in killed:
                    continue
                negative = PayloadReplay(masks, residual, parent, order, 2, weights, mutation)
                try:
                    negative.run(masks, residual, parent, order, 2)
                except AssertionError as error:
                    require(negative.injected, "negative control failed before injection")
                    killed[mutation] = dict(case=index, order=label, error=str(error))
                else:
                    require(not negative.injected, "injected payload fault escaped checker")
    required = ("dual_enqueue", "spill_plus_hot_read_different_slots",
                "pending_snapshot_returns", "slot_reuse_with_queued_snapshot",
                "negative_limit_lanes", "positive_int8_sum_limit_lanes")
    require(all(coverage[k] > 0 for k in required),
            "vacuous required coverage: " + repr({k: coverage[k] for k in required}))
    require(len(killed) == 3, "not all negative-control faults were exercised")
    print(json.dumps(dict(status="PASS", seed=args.seed, slots=2,
                          mask_cases=len(cases), payload_replays=replays,
                          scalar_mismatches=0, counters=dict(counters),
                          coverage=dict(coverage), negative_controls=killed,
                          scope="Production event refinement with independent 96x12-bit "
                                "payload/consumer state; not RTL or independent arbitration"),
                     indent=2))


if __name__ == "__main__":
    main()
