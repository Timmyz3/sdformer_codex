#!/usr/bin/env python3
"""Finite-credit producer-to-ATLIF discrete-event retirement model.

This module intentionally models explicit P_DONE tokens, bounded FIFO credit,
context-slot reuse, temporal order, and sample fences.  It does not infer join
readiness; join edges remain disabled until a versioned rendezvous trace exists.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
from pathlib import Path
from typing import Any


VARIANTS = {"full_context", "lane_cache", "lane_replay"}


def is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def unsigned_width(maximum: int) -> int:
    if maximum < 0:
        raise ValueError("tag fields must be unsigned")
    return max(1, maximum.bit_length())


def tag_encoding_ledger(patterns: list[dict[str, Any]]) -> dict[str, Any]:
    categorical = {
        "sequence_key": sorted({str(item["sequence_key"]) for item in patterns}),
        "producer": sorted({str(item["producer"]) for item in patterns}),
        "edge": sorted({str(item["edge"]) for item in patterns}),
    }
    numeric_values = {
        "sample_id": [int(item["sample_id"]) for item in patterns],
        "producer_call_index": [int(item["producer_call_index"]) for item in patterns],
        "edge_call_index": [int(item["edge_call_index"]) for item in patterns],
        "version": [int(item["version"]) for item in patterns],
        "sample_cluster_id": [int(item["sample_cluster_id"]) for item in patterns],
        "population_cluster_id": [int(item["population_cluster_id"]) for item in patterns],
        "context": [max(int(step["contexts"]) for item in patterns for step in item["steps"]) - 1],
        "lane_tile": [max(int(item["lane_tiles"]) for item in patterns) - 1],
        "temporal_step": [max(len(item["steps"]) for item in patterns) - 1],
    }
    fields: dict[str, dict[str, Any]] = {}
    for name, values in categorical.items():
        fields[name] = {
            "encoding": "sorted_dictionary_id",
            "cardinality": len(values),
            "bits": unsigned_width(len(values) - 1),
            "dictionary_sha256": hashlib.sha256(
                ("\n".join(values) + "\n").encode("utf-8")
            ).hexdigest(),
        }
    for name, values in numeric_values.items():
        maximum = max(values)
        minimum = min(values)
        if minimum < 0:
            raise ValueError(f"negative tag field: {name}")
        fields[name] = {
            "encoding": "unsigned_binary", "minimum": minimum,
            "maximum": maximum, "bits": unsigned_width(maximum),
        }
    return {
        "fields": fields,
        "required_bits": sum(int(item["bits"]) for item in fields.values()),
        "collision_contract": "categorical dictionaries and unsigned fields are exact within this stream",
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_pattern(pattern: dict[str, Any]) -> None:
    required = (
        "sample_id", "sequence_key", "producer", "edge", "edge_kind",
        "admitted_for_overlap", "producer_call_index",
        "edge_call_index", "version", "sample_cluster_id", "population_cluster_id",
        "version_identity_sha256", "cost_basis",
        "scheduler_sufficient_statistics_sha256", "cost_source_sha256",
        "fanout", "lane_tiles", "chunks", "steps",
    )
    missing = [field for field in required if field not in pattern]
    if missing:
        raise ValueError("pattern identity is incomplete: " + ",".join(missing))
    if not pattern["steps"] or int(pattern["fanout"]) <= 0 or int(pattern["chunks"]) <= 0:
        raise ValueError("pattern geometry is empty")
    if pattern["edge_kind"] != "direct_m4" or pattern["admitted_for_overlap"] is not True:
        raise ValueError("M15 accepts admitted direct_m4 edges only")
    if pattern["cost_basis"] not in {"synthetic_exact", "exact_full_population"}:
        raise ValueError("producer cost is not based on an admitted exact population")
    if not is_sha256(pattern["scheduler_sufficient_statistics_sha256"]):
        raise ValueError("scheduler sufficient-statistics hash is invalid")
    if not is_sha256(pattern["cost_source_sha256"]):
        raise ValueError("lane-cost source hash is invalid")
    if not is_sha256(pattern["version_identity_sha256"]):
        raise ValueError("immutable version identity hash is invalid")
    temporal_steps = len(pattern["steps"])
    context_counts = set()
    for timestep, step in enumerate(pattern["steps"]):
        if int(step.get("temporal_step", -1)) != timestep:
            raise ValueError("pattern temporal steps are not dense from t0")
        contexts = int(step.get("contexts", 0))
        context_counts.add(contexts)
        if not 0 < contexts <= 4:
            raise ValueError("pattern has an invalid C1-C4 context count")
        for field in ("descriptor_cycles", "lane_compute_cycles"):
            if int(step.get(field, -1)) < 0:
                raise ValueError("negative producer event cost")
        if int(step["descriptor_cycles"]) != contexts * int(pattern["chunks"]):
            raise ValueError("descriptor completion does not cover every context/chunk")
        if int(step["lane_compute_cycles"]) < 2 * int(pattern["chunks"]):
            raise ValueError("lane completion is earlier than per-chunk PREP/DRAIN")
    if len(context_counts) != 1:
        raise ValueError("context membership changed across temporal steps")
    if int(pattern["lane_tiles"]) != math.ceil(int(pattern["fanout"]) / 96):
        raise ValueError("96-lane tile geometry mismatch")


def state_bits(
    patterns: list[dict[str, Any]], *, variant: str, context_slots: int,
    fifo_depth: int, output_lanes: int, atlif_lanes: int,
    value_bits: int, tag_bits: int,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError("unknown state variant")
    peak = {
        "atlif_state_bits": 0,
        "accumulator_bits": 0,
        "descriptor_bits": 0,
        "fifo_payload_bits": fifo_depth * output_lanes * value_bits,
        "fifo_tag_bits": fifo_depth * tag_bits,
        "context_credit_valid_bits": context_slots,
        "context_owner_tag_bits": context_slots * tag_bits,
        "output_holding_register_bits": output_lanes * value_bits,
        "atlif_consumer_register_bits": atlif_lanes * 24,
    }
    for pattern in patterns:
        contexts = max(int(step["contexts"]) for step in pattern["steps"])
        fanout = int(pattern["fanout"])
        chunks = int(pattern["chunks"])
        temporal_steps = len(pattern["steps"])
        resident_fanout = fanout if variant == "full_context" else min(output_lanes, fanout)
        peak["atlif_state_bits"] = max(
            peak["atlif_state_bits"],
            context_slots * contexts * resident_fanout * temporal_steps * 24,
        )
        peak["accumulator_bits"] = max(
            peak["accumulator_bits"], contexts * output_lanes * 32,
        )
        descriptor_steps = temporal_steps if variant == "lane_cache" else 1
        peak["descriptor_bits"] = max(
            peak["descriptor_bits"], descriptor_steps * contexts * chunks * 2 * 256,
        )
        peak["context_completion_bits"] = max(
            peak.get("context_completion_bits", 0),
            context_slots * contexts * int(pattern["lane_tiles"]) * temporal_steps,
        )
    peak["total_bits"] = sum(peak.values())
    peak["contract"] = {
        "atlif_state_bits": "slot-resident per-temporal-position 24b state; includes max T",
        "atlif_consumer_register_bits": (
            "one independent active DP-TME lane register per ATLIF lane; current temporal "
            "position retires before the register is reused, so this field intentionally excludes T"
        ),
        "fifo_credit": "payload and collision-free immutable version tag retained through service finish",
    }
    return peak


def simulate_event_stream(
    patterns: list[dict[str, Any]], *, variant: str, context_slots: int,
    fifo_depth: int, output_lanes: int = 96, atlif_lanes: int = 16,
    value_bits: int = 16, tag_bits: int = 96,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported variant: {variant}")
    if context_slots <= 0 or fifo_depth <= 0 or output_lanes <= 0 or atlif_lanes <= 0:
        raise ValueError("credits and lane counts must be positive")
    if not patterns:
        raise ValueError("empty event stream")
    if output_lanes != 96:
        raise ValueError("M15 v1 freezes the producer output width at 96 lanes")
    for pattern in patterns:
        validate_pattern(pattern)
    prototype_signatures: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    for pattern in patterns:
        prototype_id = (
            pattern["sample_id"], pattern["sequence_key"], pattern["producer"],
            pattern["producer_call_index"], pattern["sample_cluster_id"],
        )
        signature = (
            int(pattern["fanout"]), int(pattern["lane_tiles"]), int(pattern["chunks"]),
            pattern["cost_basis"], pattern["scheduler_sufficient_statistics_sha256"],
            pattern["cost_source_sha256"],
            json.dumps(pattern["steps"], sort_keys=True, separators=(",", ":")),
        )
        previous_signature = prototype_signatures.setdefault(prototype_id, signature)
        if previous_signature != signature:
            raise ValueError("prototype ID has inconsistent geometry, costs, or bit-pattern hash")
    pattern_identities = [
        (
            pattern["sample_id"], pattern["sequence_key"], pattern["producer"],
            pattern["producer_call_index"], pattern["edge"], pattern["edge_call_index"],
            pattern["population_cluster_id"],
        )
        for pattern in patterns
    ]
    if len(pattern_identities) != len(set(pattern_identities)):
        raise ValueError("duplicate population cluster identity")
    version_ids: dict[int, str] = {}
    version_hashes: dict[str, int] = {}
    for pattern in patterns:
        version = int(pattern["version"])
        version_hash = pattern["version_identity_sha256"]
        if version in version_ids and version_ids[version] != version_hash:
            raise ValueError("one hardware version ID maps to multiple immutable identities")
        if version_hash in version_hashes and version_hashes[version_hash] != version:
            raise ValueError("one immutable version identity maps to multiple hardware IDs")
        version_ids[version] = version_hash
        version_hashes[version_hash] = version
    tag_ledger = tag_encoding_ledger(patterns)
    if tag_bits < int(tag_ledger["required_bits"]):
        raise ValueError("configured tag width truncates the event-stream identity")

    producer_time = 0
    atlif_time = 0
    producer_work = 0
    atlif_service = 0
    context_stall = 0
    fifo_stall = 0
    sample_fence_cycles = 0
    slot_release = [0] * context_slots
    fifo_completions: list[tuple[int, int, tuple[Any, ...]]] = []
    max_fifo_occupancy = 0
    tags_seen = set()
    consumer_started_tags = set()
    consumer_finished_tags = set()
    last_timestep: dict[tuple[Any, ...], int] = {}
    event_hash = hashlib.sha256()
    tokens = 0
    sample_fences = 0
    previous_sample = None

    def drain_completed(now: int) -> None:
        while fifo_completions and fifo_completions[0][0] <= now:
            _, _, completed_tag = heapq.heappop(fifo_completions)
            if completed_tag in consumer_finished_tags:
                raise ValueError("consumer completed a P_DONE tag more than once")
            consumer_finished_tags.add(completed_tag)

    def producer_delay(cycles: int) -> None:
        nonlocal producer_time, producer_work
        producer_time += cycles
        producer_work += cycles

    def emit_token(
        pattern: dict[str, Any], population_group: int, slot: int,
        timestep: int, context: int, lane_tile: int,
    ) -> int:
        nonlocal producer_time, atlif_time, atlif_service, fifo_stall
        nonlocal max_fifo_occupancy, tokens
        temporal_key = (
            pattern["sample_id"], pattern["sequence_key"], pattern["producer"],
            pattern["producer_call_index"], pattern["edge"], pattern["edge_call_index"],
            pattern["population_cluster_id"], context, lane_tile, pattern["version"],
        )
        expected = last_timestep.get(temporal_key, -1) + 1
        if timestep != expected:
            raise ValueError(f"ATLIF temporal order violation: got t{timestep}, expected t{expected}")
        last_timestep[temporal_key] = timestep
        tag = temporal_key + (
            timestep, pattern["sample_cluster_id"], population_group,
        )
        if tag in tags_seen:
            raise ValueError("duplicate P_DONE tag")
        tags_seen.add(tag)
        # Requant/P_DONE consumes the output cycle and is the earliest legal
        # downstream visibility point after all chunks and Acc32 finish.
        producer_delay(1)
        drain_completed(producer_time)
        if len(fifo_completions) >= fifo_depth:
            release, _, released_tag = heapq.heappop(fifo_completions)
            if released_tag in consumer_finished_tags:
                raise ValueError("FIFO released an already-completed P_DONE tag")
            if release > producer_time:
                fifo_stall += release - producer_time
                producer_time = release
            consumer_finished_tags.add(released_tag)
            drain_completed(producer_time)
        valid_lanes = min(output_lanes, int(pattern["fanout"]) - lane_tile * output_lanes)
        service = math.ceil(valid_lanes / atlif_lanes)
        start = max(atlif_time, producer_time)
        finish = start + service
        atlif_time = finish
        atlif_service += service
        if tag in consumer_started_tags:
            raise ValueError("consumer started a P_DONE tag more than once")
        consumer_started_tags.add(tag)
        heapq.heappush(fifo_completions, (finish, tokens, tag))
        max_fifo_occupancy = max(max_fifo_occupancy, len(fifo_completions))
        event_hash.update((repr(tag) + f"@{producer_time}:{start}:{finish}\n").encode("utf-8"))
        tokens += 1
        return finish

    for population_group, pattern in enumerate(patterns):
        sample = (pattern["sample_id"], pattern["sequence_key"])
        if previous_sample is not None and sample != previous_sample:
            fence_start = producer_time
            producer_time = max(producer_time, atlif_time)
            sample_fence_cycles += producer_time - fence_start
            sample_fences += 1
            drain_completed(producer_time)
            if fifo_completions:
                raise ValueError("sample fence did not retire all FIFO tags")
            slot_release = [producer_time] * context_slots
        previous_sample = sample
        slot = population_group % context_slots
        if slot_release[slot] > producer_time:
            context_stall += slot_release[slot] - producer_time
            producer_time = slot_release[slot]
            drain_completed(producer_time)
        group_finish = producer_time
        if variant == "full_context":
            for timestep, step in enumerate(pattern["steps"]):
                producer_delay(int(step["descriptor_cycles"]))
                for lane_tile in range(int(pattern["lane_tiles"])):
                    producer_delay(int(step["lane_compute_cycles"]))
                    for context in range(int(step["contexts"])):
                        group_finish = max(
                            group_finish,
                            emit_token(pattern, population_group, slot, timestep, context, lane_tile),
                        )
        else:
            if variant == "lane_cache":
                producer_delay(sum(int(step["descriptor_cycles"]) for step in pattern["steps"]))
            lane_release = producer_time
            for lane_tile in range(int(pattern["lane_tiles"])):
                if lane_release > producer_time:
                    context_stall += lane_release - producer_time
                    producer_time = lane_release
                    drain_completed(producer_time)
                for timestep, step in enumerate(pattern["steps"]):
                    if variant == "lane_replay":
                        producer_delay(int(step["descriptor_cycles"]))
                    producer_delay(int(step["lane_compute_cycles"]))
                    for context in range(int(step["contexts"])):
                        lane_release = max(
                            lane_release,
                            emit_token(pattern, population_group, slot, timestep, context, lane_tile),
                        )
                group_finish = max(group_finish, lane_release)
        slot_release[slot] = group_finish

    finish = max(producer_time, atlif_time)
    drain_completed(finish)
    if tags_seen != consumer_started_tags or tags_seen != consumer_finished_tags:
        raise ValueError("P_DONE tag did not retire exactly once")
    expected_temporal_keys = sum(
        max(int(step["contexts"]) for step in pattern["steps"]) * int(pattern["lane_tiles"])
        for pattern in patterns
    )
    if len(last_timestep) != expected_temporal_keys:
        raise ValueError("missing temporal context identities")
    for temporal_key, last in last_timestep.items():
        population_cluster_id = temporal_key[6]
        group = next(
            index for index, item in enumerate(patterns)
            if item["population_cluster_id"] == population_cluster_id
            and item["sample_id"] == temporal_key[0]
            and item["sequence_key"] == temporal_key[1]
            and item["producer"] == temporal_key[2]
            and item["producer_call_index"] == temporal_key[3]
            and item["edge"] == temporal_key[4]
            and item["edge_call_index"] == temporal_key[5]
        )
        if last + 1 != len(patterns[group]["steps"]):
            raise ValueError("incomplete ATLIF temporal chain")
    serial = producer_work + atlif_service
    hidden = serial - finish
    if not 0 <= hidden <= min(producer_work, atlif_service):
        raise ValueError("overlap conservation failed")
    return {
        "variant": variant,
        "patterns": len(patterns),
        "p_done_tokens": tokens,
        "unique_tags": len(tags_seen),
        "consumer_started_tags": len(consumer_started_tags),
        "consumer_finished_tags": len(consumer_finished_tags),
        "producer_work_cycles": producer_work,
        "atlif_service_cycles": atlif_service,
        "producer_finish_cycles": producer_time,
        "atlif_finish_cycles": atlif_time,
        "fused_finish_cycles": finish,
        "hidden_cycles": hidden,
        "context_credit_stall_cycles": context_stall,
        "fifo_backpressure_stall_cycles": fifo_stall,
        "sample_fences": sample_fences,
        "sample_fence_cycles": sample_fence_cycles,
        "max_fifo_occupancy": max_fifo_occupancy,
        "event_stream_sha256": event_hash.hexdigest(),
        "tag_encoding": {**tag_ledger, "configured_bits": tag_bits},
        "version_encoding": {
            "entries": len(version_ids),
            "ledger_sha256": hashlib.sha256(
                json.dumps(version_ids, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        },
        "state": state_bits(
            patterns, variant=variant, context_slots=context_slots, fifo_depth=fifo_depth,
            output_lanes=output_lanes, atlif_lanes=atlif_lanes,
            value_bits=value_bits, tag_bits=tag_bits,
        ),
        "claim_boundary": (
            "Integer producer/P_DONE/ATLIF events with finite FIFO and context credits; "
            "FIFO entries retain credit until ATLIF service finish (conservative versus dequeue-at-start); "
            "direct edges only, no join admission, memory-port timing, RTL, or PPA."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patterns", type=Path, required=True)
    parser.add_argument("--context-slots", type=int, default=1)
    parser.add_argument("--fifo-depth", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    patterns = json.loads(args.patterns.read_text(encoding="utf-8"))
    variants = {
        variant: simulate_event_stream(
            patterns, variant=variant, context_slots=args.context_slots,
            fifo_depth=args.fifo_depth,
        )
        for variant in sorted(VARIANTS)
    }
    payload = {
        "schema": "m15_finite_credit_retirement_event_model_v1",
        "status": "PASS_INTEGER_FINITE_CREDIT_DIRECT_EDGE_EVENTS_NOT_SYSTEM_SPEEDUP",
        "variants": variants,
        "identities": {
            "patterns_sha256": sha256(args.patterns),
            "source_sha256": sha256(Path(__file__).resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("PASS_M15_FINITE_CREDIT_EVENT_MODEL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
