#!/usr/bin/env python3
"""M1056 source-only exact packed-psum 1RW arbitration replay.

This additive source repairs the M1016 packing/cycle disconnect.  It keeps the
same task/pipeline event geometry, assigns every psum access a physical packed
macro group and address, and passes requests through a deterministic FIFO per
1RW group.  A delayed request changes task completion, the following task's
work start, and sample commit.  Capacity bytes and port feasibility are
separate gates.

No full frozen-row replay, EDA, GPU, remote work, or paper claim is authorized
by this source milestone.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
M1016 = HW / "system_simulator/scripts/run_m1016_c1_full_matched_address_replay.py"
M1051 = HW / "reviews/m1051_m1040_m1016_c1_full_replay_result_hammer_r1_20260829"
CONTRACT = HW / "contracts/m1056_m1051_c1_exact_1rw_arbitration_replay_source_contract_r1_20260829.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

M1016_SHA = "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa"
M1051_ID = (
    "e74974a15b6ad888af9675d6feee276a805840d93235e0c1ee9eff0f877e051f",
    "f87f501bc50073bb946786ce8a23d8413d6b68dd166effc10e78ff7b926f0b69",
    "15e0c98654db25599520025bc43448e1e38c58fe9002ed5ad8a5f71b9eef4b0f",
)
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

BLOCKS = 8
PACKED_GROUPS = 4
ROWS_PER_LOGICAL_BANK = 64
CAPACITY_BUDGET_BYTES = 240 * 1024
CAPACITY_HYPOTHESIS_BYTES = 214_912
DESIGNS = ("candidate", "strongest_zero", "same_coordinate_bit")
COMMON_RESOURCES = ("psum", "weight", "source", "dma", "commit")


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            RuntimeError("nonfinite JSON: " + token)
        ),
    )


def verify_flat(directory: Path, identity: tuple[str, str, str]) -> None:
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require((sha256(review), sha256(manifest), sha256(outer)) == identity,
            "M1051 identity drift")
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        require(name not in listed and sha256(directory / name) == expected,
                "M1051 sealed member drift")
        listed.add(name)
    expected, name = outer.read_text(encoding="utf-8").split()
    require(name == "SHA256SUMS" and expected == sha256(manifest),
            "M1051 outer seal drift")


@dataclass(frozen=True)
class ArbiterConfig:
    groups: int = PACKED_GROUPS
    ports_per_group: int = 1
    port_mode: str = "1RW"
    request_fifo: str = "arrival_cycle_then_program_order"
    read_after_write_latency: int = 1
    initial_psum_valid: bool = True

    def validate(self) -> None:
        require(self.groups == PACKED_GROUPS and self.ports_per_group == 1 and
                self.port_mode == "1RW" and
                self.request_fifo == "arrival_cycle_then_program_order" and
                self.read_after_write_latency == 1 and self.initial_psum_valid,
                "arbiter coordinate drift")


@dataclass(frozen=True)
class Dependency:
    event_id: str
    delay_cycles: int = 0

    def validate(self) -> None:
        require(bool(self.event_id) and self.delay_cycles >= 0,
                "invalid event dependency")


@dataclass(frozen=True)
class PortEvent:
    event_id: str
    task_id: int
    program_order: int
    group: int
    logical_bank: int
    address: int
    op: str
    base_ready_cycle: int
    dependencies: tuple[Dependency, ...] = ()
    external_ready_cycle: int = 0

    def validate(self) -> None:
        require(bool(self.event_id) and self.task_id >= 0 and
                self.program_order >= 0 and 0 <= self.group < PACKED_GROUPS and
                0 <= self.logical_bank < BLOCKS and
                self.group == self.logical_bank // 2 and
                0 <= self.address < 2 * ROWS_PER_LOGICAL_BANK and
                self.op in ("READ", "WRITE") and
                self.base_ready_cycle >= 0 and self.external_ready_cycle >= 0,
                "invalid port event")
        for dependency in self.dependencies:
            dependency.validate()


@dataclass(frozen=True)
class Grant:
    event_id: str
    cycle: int
    group: int
    address: int
    op: str
    arrival_cycle: int
    program_order: int


@dataclass
class GroupArbitration:
    group: int
    grants: dict[str, Grant]
    grant_order: list[str]
    queue_peak: int
    port_busy_cycles: int
    first_cycle: int | None
    last_cycle: int | None


def arbitrate_group(events: Sequence[PortEvent], group: int,
                    config: ArbiterConfig = ArbiterConfig()) -> GroupArbitration:
    """Schedule one group with a fixed arrival-order FIFO and one 1RW port.

    Input sequence order is irrelevant.  Events enter the FIFO when both their
    base/external readiness and all dependency delays are satisfied.  Events
    becoming ready in the same cycle are appended by ``program_order``.
    Exactly one FIFO head is granted per cycle.
    """
    config.validate()
    require(0 <= group < config.groups, "group outside configuration")
    require(events, "empty group arbitration")
    by_id: dict[str, PortEvent] = {}
    orders = set()
    for event in events:
        event.validate()
        require(event.group == group and event.event_id not in by_id and
                event.program_order not in orders, "group event identity drift")
        by_id[event.event_id] = event
        orders.add(event.program_order)
    for event in events:
        for dependency in event.dependencies:
            require(dependency.event_id in by_id,
                    "dependency must be inside the same group arbitration")

    pending = set(by_id)
    queued = set()
    fifo: deque[tuple[str, int]] = deque()
    grants: dict[str, Grant] = {}
    order: list[str] = []
    queue_peak = 0
    cycle = min(max(event.base_ready_cycle, event.external_ready_cycle)
                for event in events)
    guard = 0
    while pending or fifo:
        guard += 1
        require(guard <= 1_000_000, "arbiter failed bounded progress")
        arrivals: list[tuple[int, PortEvent]] = []
        future_cycles = []
        for event_id in sorted(pending, key=lambda item: by_id[item].program_order):
            event = by_id[event_id]
            if not all(dep.event_id in grants for dep in event.dependencies):
                continue
            release = max(
                event.base_ready_cycle,
                event.external_ready_cycle,
                *(grants[dep.event_id].cycle + dep.delay_cycles
                  for dep in event.dependencies),
            )
            if release <= cycle:
                arrivals.append((event.program_order, event))
            else:
                future_cycles.append(release)
        for _, event in sorted(arrivals):
            fifo.append((event.event_id, cycle))
            queued.add(event.event_id)
            pending.remove(event.event_id)
        queue_peak = max(queue_peak, len(fifo))
        if fifo:
            event_id, arrival = fifo.popleft()
            event = by_id[event_id]
            grant = Grant(event_id, cycle, group, event.address, event.op,
                          arrival, event.program_order)
            grants[event_id] = grant
            order.append(event_id)
            cycle += 1
            continue
        require(future_cycles, "arbiter deadlock")
        cycle = min(future_cycles)

    require(queued == set(by_id) and len(grants) == len(by_id),
            "arbiter terminal conservation drift")
    grant_cycles = [grant.cycle for grant in grants.values()]
    require(len(grant_cycles) == len(set(grant_cycles)), "1RW double grant")
    return GroupArbitration(
        group=group,
        grants=grants,
        grant_order=order,
        queue_peak=queue_peak,
        port_busy_cycles=len(grants),
        first_cycle=min(grant_cycles) if grant_cycles else None,
        last_cycle=max(grant_cycles) if grant_cycles else None,
    )


@dataclass(frozen=True)
class TaskPlan:
    task_id: int
    preprocess_cycles: int
    work_cycles: int
    psum_row: int

    def validate(self) -> None:
        require(self.task_id >= 0 and self.preprocess_cycles >= 0 and
                self.work_cycles >= 0 and
                0 <= self.psum_row < ROWS_PER_LOGICAL_BANK,
                "invalid task plan")


@dataclass
class TaskResult:
    task_id: int
    work_start: int
    nominal_work_end: int
    effective_work_end: int
    events: list[PortEvent]
    grants: dict[str, Grant]
    queue_peak: int
    nominal_excess_accesses: int
    delayed_accesses: int
    maximum_read_write_lifetime: int
    raw_dependencies_pass: bool


def packed_address(logical_bank: int, row: int) -> int:
    require(0 <= logical_bank < BLOCKS and
            0 <= row < ROWS_PER_LOGICAL_BANK, "packed address coordinate drift")
    return (logical_bank & 1) * ROWS_PER_LOGICAL_BANK + row


def nominal_task_events(plan: TaskPlan, work_start: int,
                        last_write_cycle: Mapping[tuple[int, int], int]) -> list[PortEvent]:
    """Generate the exact M1016 task-local psum event geometry with addresses."""
    plan.validate()
    require(work_start >= 0, "negative work start")
    work_end = work_start + plan.work_cycles
    span = max(1, plan.work_cycles // BLOCKS)
    events: list[PortEvent] = []
    for bank in range(BLOCKS):
        group = bank // 2
        address = packed_address(bank, plan.psum_row)
        read_cycle = work_start + bank * span
        write_cycle = min(work_end, read_cycle + span - 1)
        read_id = f"t{plan.task_id}:b{bank}:R"
        write_id = f"t{plan.task_id}:b{bank}:W"
        predecessor = last_write_cycle.get((group, address), -1)
        events.append(PortEvent(
            event_id=read_id,
            task_id=plan.task_id,
            program_order=plan.task_id * 32 + bank * 2,
            group=group,
            logical_bank=bank,
            address=address,
            op="READ",
            base_ready_cycle=read_cycle,
            external_ready_cycle=predecessor + 1,
        ))
        events.append(PortEvent(
            event_id=write_id,
            task_id=plan.task_id,
            program_order=plan.task_id * 32 + bank * 2 + 1,
            group=group,
            logical_bank=bank,
            address=address,
            op="WRITE",
            base_ready_cycle=write_cycle,
            dependencies=(Dependency(read_id, write_cycle - read_cycle),),
        ))
    return events


def schedule_task(plan: TaskPlan, work_start: int,
                  last_write_cycle: dict[tuple[int, int], int],
                  config: ArbiterConfig = ArbiterConfig()) -> TaskResult:
    events = nominal_task_events(plan, work_start, last_write_cycle)
    grants: dict[str, Grant] = {}
    queue_peak = 0
    for group in range(PACKED_GROUPS):
        result = arbitrate_group([event for event in events if event.group == group],
                                 group, config)
        grants.update(result.grants)
        queue_peak = max(queue_peak, result.queue_peak)
    require(len(grants) == len(events), "task port-event conservation drift")

    nominal_slots = Counter((event.group, event.base_ready_cycle) for event in events)
    nominal_excess = sum(count - 1 for count in nominal_slots.values() if count > 1)
    delayed = sum(grants[event.event_id].cycle > event.base_ready_cycle for event in events)
    maximum_lifetime = 0
    raw_pass = True
    write_cycles = []
    for bank in range(BLOCKS):
        read = grants[f"t{plan.task_id}:b{bank}:R"]
        write = grants[f"t{plan.task_id}:b{bank}:W"]
        require(write.cycle >= read.cycle, "write granted before current read")
        group = bank // 2
        address = packed_address(bank, plan.psum_row)
        predecessor = last_write_cycle.get((group, address), -1)
        raw_pass &= read.cycle >= predecessor + 1
        maximum_lifetime = max(maximum_lifetime, write.cycle - read.cycle)
        last_write_cycle[(group, address)] = write.cycle
        write_cycles.append(write.cycle)
    effective_end = max(work_start + plan.work_cycles, max(write_cycles))
    return TaskResult(
        task_id=plan.task_id,
        work_start=work_start,
        nominal_work_end=work_start + plan.work_cycles,
        effective_work_end=effective_end,
        events=events,
        grants=grants,
        queue_peak=queue_peak,
        nominal_excess_accesses=nominal_excess,
        delayed_accesses=delayed,
        maximum_read_write_lifetime=maximum_lifetime,
        raw_dependencies_pass=bool(raw_pass),
    )


@dataclass
class SequenceResult:
    tasks: list[TaskResult]
    sample_cycles_before_commit: int
    sample_cycles_after_commit: int
    total_nominal_excess_accesses: int
    total_delayed_accesses: int
    capacity_bytes_pass: bool
    port_feasibility_pass: bool


def replay_task_sequence(plans: Sequence[TaskPlan], commit_cycles: int = 0,
                         capacity_bytes: int = CAPACITY_HYPOTHESIS_BYTES,
                         config: ArbiterConfig = ArbiterConfig()) -> SequenceResult:
    """Replay one sample, propagating 1RW delay through the M1016 pipeline."""
    require(plans and commit_cycles >= 0 and capacity_bytes >= 0,
            "invalid sequence coordinate")
    require([plan.task_id for plan in plans] == sorted(plan.task_id for plan in plans) and
            len({plan.task_id for plan in plans}) == len(plans),
            "task sequence must have unique increasing program order")
    last_write: dict[tuple[int, int], int] = {}
    task_results = []
    previous_start = None
    previous_effective_end = None
    for plan in plans:
        plan.validate()
        if previous_start is None:
            start = plan.preprocess_cycles
        else:
            require(previous_effective_end is not None, "pipeline state drift")
            preprocess_ready = previous_start + plan.preprocess_cycles
            start = max(previous_effective_end, preprocess_ready) + 2
        result = schedule_task(plan, start, last_write, config)
        task_results.append(result)
        previous_start = start
        previous_effective_end = result.effective_work_end
    before_commit = int(previous_effective_end) + 2
    port_pass = all(result.raw_dependencies_pass for result in task_results)
    return SequenceResult(
        tasks=task_results,
        sample_cycles_before_commit=before_commit,
        sample_cycles_after_commit=before_commit + commit_cycles,
        total_nominal_excess_accesses=sum(
            result.nominal_excess_accesses for result in task_results
        ),
        total_delayed_accesses=sum(result.delayed_accesses for result in task_results),
        capacity_bytes_pass=capacity_bytes <= CAPACITY_BUDGET_BYTES,
        port_feasibility_pass=port_pass,
    )


def common_service_digest(receipts: Iterable[Mapping[str, Any]]) -> tuple[Counter, str]:
    counts: Counter = Counter()
    digest = hashlib.sha256()
    for receipt in receipts:
        require(set(receipt.get("counts", {})) == set(COMMON_RESOURCES),
                "common-service receipt resource drift")
        for resource, count in receipt["counts"].items():
            require(isinstance(count, int) and count >= 0,
                    "common-service count invalid")
            counts[resource] += count
        digest.update(json.dumps(receipt, sort_keys=True,
                                 separators=(",", ":")).encode())
    return counts, digest.hexdigest()


def validate_three_design_common_coordinate(
    receipts: Mapping[str, Sequence[Mapping[str, Any]]],
    configs: Mapping[str, ArbiterConfig],
) -> dict[str, Any]:
    require(set(receipts) == set(DESIGNS) and set(configs) == set(DESIGNS),
            "three-design population drift")
    rows = {name: common_service_digest(receipts[name]) for name in DESIGNS}
    for config in configs.values():
        config.validate()
    require(len({tuple(sorted(counts.items())) for counts, _ in rows.values()}) == 1 and
            len({digest for _, digest in rows.values()}) == 1 and
            len({configs[name] for name in DESIGNS}) == 1,
            "asymmetric common service or arbiter coordinate")
    return {
        "status": "PASS_M1056_THREE_DESIGN_COMMON_COORDINATE",
        "service_counts_equal": True,
        "service_digests_equal": True,
        "arbiter_resources_equal": True,
    }


def replay_three_design_sequences(
    plans: Mapping[str, Sequence[TaskPlan]],
    receipts: Mapping[str, Sequence[Mapping[str, Any]]],
    commit_cycles: int = 0,
    capacity_bytes: int = CAPACITY_HYPOTHESIS_BYTES,
) -> dict[str, Any]:
    """Apply one frozen arbiter coordinate to all matched comparison designs."""
    require(set(plans) == set(DESIGNS), "three-design plan population drift")
    configs = {name: ArbiterConfig() for name in DESIGNS}
    common = validate_three_design_common_coordinate(receipts, configs)
    results = {
        name: replay_task_sequence(plans[name], commit_cycles, capacity_bytes,
                                   configs[name])
        for name in DESIGNS
    }
    require(all(result.capacity_bytes_pass ==
                (capacity_bytes <= CAPACITY_BUDGET_BYTES)
                for result in results.values()), "capacity gate asymmetry")
    require(all(result.port_feasibility_pass for result in results.values()),
            "one design failed deterministic port arbitration")
    return {
        "status": "PASS_M1056_THREE_DESIGN_EXACT_1RW_REPLAY",
        "common_coordinate": common,
        "cycles_after_commit": {
            name: results[name].sample_cycles_after_commit for name in DESIGNS
        },
        "capacity_bytes": capacity_bytes,
        "capacity_bytes_pass": capacity_bytes <= CAPACITY_BUDGET_BYTES,
        "port_feasibility_pass": True,
        "results": results,
    }


def nominal_m1016_sequence_cycles(plans: Sequence[TaskPlan], commit_cycles: int = 0) -> int:
    """Independent closed form of M1016 Pipeline for small-oracle comparison."""
    start = None
    previous_work = 0
    total = 0
    for plan in plans:
        if start is None:
            start = plan.preprocess_cycles
        else:
            start = start + max(previous_work, plan.preprocess_cycles) + 2
        previous_work = plan.work_cycles
        total = start + plan.work_cycles + 2
    return total + commit_cycles


def direct_event(event_id: str, order: int, ready: int, address: int,
                 dependencies: tuple[Dependency, ...] = ()) -> PortEvent:
    return PortEvent(event_id, 0, order, 0, 0, address, "READ" if not dependencies
                     else "WRITE", ready, dependencies)


def small_oracle() -> dict[str, Any]:
    verify_flat(M1051, M1051_ID)
    require(sha256(M1016) == M1016_SHA and sha256(DOCS359) == DOCS359_SHA,
            "frozen authority drift")

    # No-conflict coordinate: span=2, so each read/write occupies a unique slot.
    clean_plan = [TaskPlan(0, 3, 16, 7)]
    clean = replay_task_sequence(clean_plan)
    require(clean.sample_cycles_after_commit ==
            nominal_m1016_sequence_cycles(clean_plan) and
            clean.total_nominal_excess_accesses == 0 and
            clean.capacity_bytes_pass and clean.port_feasibility_pass,
            "no-conflict identity drift")

    # Different addresses still collide on the one shared group port.
    different_addresses = arbitrate_group([
        direct_event("a", 0, 4, 1),
        direct_event("b", 1, 4, 65),
    ], 0)
    require([different_addresses.grants[key].cycle for key in ("a", "b")] == [4, 5],
            "different-address port conflict escaped")

    # Multiplicity three and reversed input prove global ready/order arbitration.
    triple = arbitrate_group([
        direct_event("c", 2, 9, 2),
        direct_event("a", 0, 9, 0),
        direct_event("b", 1, 9, 1),
    ], 0)
    require(triple.grant_order == ["a", "b", "c"] and
            [triple.grants[key].cycle for key in triple.grant_order] == [9, 10, 11] and
            triple.queue_peak == 3, "multiplicity-three FIFO drift")

    # Cross-task/list disorder: arrival cycle wins, not append order.
    unordered = arbitrate_group([
        PortEvent("late_t0", 0, 0, 0, 0, 3, "READ", 8),
        PortEvent("early_t1", 1, 32, 0, 0, 4, "READ", 2),
    ], 0)
    require(unordered.grant_order == ["early_t1", "late_t0"] and
            unordered.grants["early_t1"].cycle == 2 and
            unordered.grants["late_t0"].cycle == 8,
            "cross-task ready order drift")

    # Same-address RAW: the early nominal read waits for its predecessor write.
    raw = arbitrate_group([
        PortEvent("new_read", 1, 2, 0, 0, 11, "READ", 0,
                  (Dependency("old_write", 1),)),
        PortEvent("old_write", 0, 1, 0, 0, 11, "WRITE", 5),
    ], 0)
    require(raw.grant_order == ["old_write", "new_read"] and
            raw.grants["old_write"].cycle == 5 and
            raw.grants["new_read"].cycle == 6,
            "same-address RAW dependency escaped")

    # Two span=1 tasks expose serialization and feedback into the next start.
    cascade_plans = [TaskPlan(0, 0, 8, 3), TaskPlan(1, 0, 8, 3)]
    nominal = nominal_m1016_sequence_cycles(cascade_plans)
    cascade = replay_task_sequence(cascade_plans)
    require(cascade.tasks[0].effective_work_end >
            cascade.tasks[0].nominal_work_end and
            cascade.tasks[1].work_start >
            cascade.tasks[0].work_start + cascade_plans[0].work_cycles + 2 and
            cascade.sample_cycles_after_commit > nominal and
            cascade.sample_cycles_after_commit - nominal !=
            cascade.total_nominal_excess_accesses,
            "serialization delay did not cascade or became naive +conflicts")

    receipt = {"task": 0, "counts": {
        "psum": 16, "weight": 7, "source": 64, "dma": 1, "commit": 0,
    }}
    common = validate_three_design_common_coordinate(
        {name: [receipt] for name in DESIGNS},
        {name: ArbiterConfig() for name in DESIGNS},
    )
    try:
        validate_three_design_common_coordinate(
            {**{name: [receipt] for name in DESIGNS},
             "candidate": [{"task": 0, "counts": {
                 "psum": 15, "weight": 7, "source": 64, "dma": 1, "commit": 0,
             }}]},
            {name: ArbiterConfig() for name in DESIGNS},
        )
    except RuntimeError:
        asymmetric_rejected = True
    else:
        asymmetric_rejected = False
    require(asymmetric_rejected, "asymmetric common service admitted")

    three_design = replay_three_design_sequences(
        {
            "candidate": [TaskPlan(0, 0, 8, 0)],
            "strongest_zero": [TaskPlan(0, 0, 16, 0)],
            "same_coordinate_bit": [TaskPlan(0, 0, 16, 0)],
        },
        {name: [receipt] for name in DESIGNS},
        commit_cycles=7,
    )
    require(three_design["status"] ==
            "PASS_M1056_THREE_DESIGN_EXACT_1RW_REPLAY" and
            three_design["capacity_bytes_pass"] and
            three_design["port_feasibility_pass"],
            "three-design executable coordinate drift")

    return {
        "schema": "m1056_c1_exact_1rw_arbitration_small_oracle_v1",
        "status": "PASS_M1056_SMALL_ORACLE__NO_FULL_REPLAY_NO_EDA",
        "no_conflict_m1016_identity": True,
        "multiplicity_2_serialized": True,
        "multiplicity_3_serialized": True,
        "cross_task_input_order_independent": True,
        "different_address_same_port_conflict": True,
        "same_address_raw_enforced": True,
        "delay_cascades_to_next_task_and_commit": True,
        "naive_cycles_plus_conflicts_rejected": True,
        "capacity_bytes_pass": cascade.capacity_bytes_pass,
        "port_feasibility_pass": cascade.port_feasibility_pass,
        "capacity_and_port_gates_separate": True,
        "three_design_common_coordinate": common,
        "three_design_same_arbiter_replay": {
            "status": three_design["status"],
            "cycles_after_commit": three_design["cycles_after_commit"],
            "capacity_bytes_pass": three_design["capacity_bytes_pass"],
            "port_feasibility_pass": three_design["port_feasibility_pass"],
        },
        "cascade": {
            "nominal_cycles": nominal,
            "arbitrated_cycles": cascade.sample_cycles_after_commit,
            "nominal_excess_accesses": cascade.total_nominal_excess_accesses,
            "delayed_accesses": cascade.total_delayed_accesses,
            "task_starts": [task.work_start for task in cascade.tasks],
            "task_nominal_ends": [task.nominal_work_end for task in cascade.tasks],
            "task_effective_ends": [task.effective_work_end for task in cascade.tasks],
        },
        "claim_boundary": {
            "source_only": True,
            "full_51840000_replay": False,
            "capacity_only_214912B_admitted": False,
            "matched_cycles_admitted": False,
            "speedup_admitted": False,
            "rtl_cycles": False,
            "paper_ppa_ready": False,
        },
    }


def validate_source_contract(contract_path: Path = CONTRACT) -> dict[str, Any]:
    verify_flat(M1051, M1051_ID)
    require(sha256(M1016) == M1016_SHA and sha256(DOCS359) == DOCS359_SHA,
            "frozen authority drift")
    contract = strict_json(contract_path)
    require(contract.get("status") ==
            "PASS_M1056_SOURCE_ONLY__M1057_REQUIRED_NO_FULL_REPLAY" and
            contract.get("launch_now") is False and
            contract.get("max_attempts_now") == 0,
            "M1056 source contract drift")
    return {
        "status": "PASS_M1056_SOURCE_CONTRACT_PREFLIGHT__NO_FULL_REPLAY",
        "launch_now": False,
        "full_replay": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate-contract", action="store_true")
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    args = parser.parse_args(argv)
    require(args.self_test ^ args.validate_contract,
            "select exactly one source-only action")
    result = small_oracle() if args.self_test else validate_source_contract(args.contract)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
