#!/usr/bin/env python3
"""Schedule exact M17 output bursts through elastic banked raw moments.

M19 conservatively serialized one 16-lane moment update behind every 96-lane
source result.  M21 instead evaluates a concrete decoupling contract: output
packets are simultaneously materialized and tapped into a bounded FIFO; one or
more 16-lane arithmetic slices time-multiplex across channel-resident moment
state banks while M4 continues descriptor/compute work.  Every source segment
and output burst is reconstructed from the exact ordered M17 prototypes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


M17_STATUS = "PASS_EXACT_FULL_SPATIAL_C4_SUFFICIENT_STATISTICS_NOT_SYSTEM_SPEEDUP"
M19_STATUS = "PASS_EXACT_DYNAMIC_BN_BARRIER_CENSUS_BOUNDED_SOURCE_BN_ATLIF_MODEL"
M21_STATUS = "PASS_EXACT_M17_ORDERED_ELASTIC_BANKED_MOMENT_DSE_NOT_SYSTEM_SPEEDUP"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return (int(numerator) + int(denominator) - 1) // int(denominator)


def exact_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(label + " must be a positive integer")
    return int(value)


def advance_idle(state: dict[str, int], cycles: int) -> None:
    if cycles < 0:
        raise ValueError("negative producer segment")
    state["source_cycles"] += cycles
    state["moment_work"] = max(0, state["moment_work"] - cycles)


def emit_packet(
    state: dict[str, int], *, service_cycles: int,
    resident_packet_capacity: int | None,
) -> None:
    """Emit one packet with same-cycle dequeue/enqueue at a ready-valid FIFO.

    ``moment_work`` is remaining arithmetic service.  Every admitted M17 packet
    is exactly 96 channels, so all packets in one DSE point have equal service.
    Capacity includes the packet currently in service, which is conservative
    when translated into a physical FIFO entry count.
    """
    if service_cycles < 1:
        raise ValueError("packet service must be positive")
    if resident_packet_capacity is not None:
        if resident_packet_capacity < 1:
            raise ValueError("resident packet capacity must be positive")
        after_same_cycle_service = max(0, state["moment_work"] - 1)
        threshold = service_cycles * (resident_packet_capacity - 1)
        stalls = max(0, after_same_cycle_service - threshold)
        state["source_cycles"] += stalls
        state["producer_stall_cycles"] += stalls
        state["moment_work"] = max(0, state["moment_work"] - stalls)
    state["source_cycles"] += 1
    state["moment_work"] = max(0, state["moment_work"] - 1) + service_cycles
    state["packets"] += 1
    state["moment_service_cycles"] += service_cycles
    resident = ceil_div(state["moment_work"], service_cycles)
    if resident_packet_capacity is not None and resident > resident_packet_capacity:
        raise ValueError("finite FIFO admission exceeded its resident packet capacity")
    state["maximum_resident_packets"] = max(
        state["maximum_resident_packets"], resident,
    )


def dense_lane_compute_cycles(operator: dict[str, Any], contexts: int) -> int:
    source_width = exact_positive_int(operator.get("source_width"), "source width")
    chunks = exact_positive_int(operator.get("chunks"), "chunks")
    dense_issue = sum(
        ceil_div(min(256, source_width - 256 * chunk), 16)
        for chunk in range(chunks)
    )
    return 2 * chunks + contexts * dense_issue


def simulate_operator(
    operator: dict[str, Any], prototype_ids: np.ndarray,
    prototypes: list[dict[str, Any]], *, variant: str,
    arithmetic_tiles: int, resident_packet_capacity: int | None,
) -> dict[str, int]:
    if variant not in {"local", "hybrid", "same_width_dense"}:
        raise ValueError("unsupported M21 source variant")
    if arithmetic_tiles not in {1, 2, 3, 6}:
        raise ValueError("M21 arithmetic tile count must divide the 96-lane packet")
    temporal_steps = exact_positive_int(operator.get("temporal_steps"), "temporal steps")
    lane_tiles = exact_positive_int(operator.get("lane_tiles"), "lane tiles")
    population_clusters = exact_positive_int(
        operator.get("population_clusters"), "population clusters",
    )
    fanout = exact_positive_int(operator.get("fanout"), "fanout")
    if fanout != lane_tiles * 96:
        raise ValueError("M21 currently requires exact full 96-lane output packets")
    if len(prototype_ids) != population_clusters:
        raise ValueError("operator prototype stream cardinality mismatch")
    service_cycles = ceil_div(96, 16 * arithmetic_tiles)
    state = {
        "source_cycles": 0,
        "producer_stall_cycles": 0,
        "moment_work": 0,
        "packets": 0,
        "moment_service_cycles": 0,
        "maximum_resident_packets": 0,
    }
    for raw_prototype_id in prototype_ids:
        prototype_id = int(raw_prototype_id)
        if not 0 <= prototype_id < len(prototypes):
            raise ValueError("ordered stream references an invalid prototype")
        prototype = prototypes[prototype_id]
        contexts = exact_positive_int(prototype.get("contexts"), "prototype contexts")
        if not (
            exact_positive_int(prototype.get("chunks"), "prototype chunks")
            == exact_positive_int(operator.get("chunks"), "operator chunks")
            and exact_positive_int(prototype.get("fanout"), "prototype fanout") == fanout
            and exact_positive_int(prototype.get("lane_tiles"), "prototype lane tiles")
            == lane_tiles
        ):
            raise ValueError("prototype/operator geometry mismatch")
        descriptors = prototype.get("descriptor_cycles_by_t")
        local_compute = prototype.get("local_lane_compute_cycles_by_t")
        hybrid_compute = prototype.get("hybrid_lane_compute_cycles_by_t")
        if not all(
            isinstance(values, list) and len(values) == temporal_steps
            for values in (descriptors, local_compute, hybrid_compute)
        ):
            raise ValueError("prototype temporal schedule is incomplete")
        dense_compute = dense_lane_compute_cycles(operator, contexts)
        for timestep in range(temporal_steps):
            advance_idle(state, exact_positive_int(
                descriptors[timestep], "descriptor cycles",
            ))
            if variant == "local":
                lane_compute = exact_positive_int(
                    local_compute[timestep], "Local lane compute cycles",
                )
            elif variant == "hybrid":
                lane_compute = exact_positive_int(
                    hybrid_compute[timestep], "Hybrid lane compute cycles",
                )
            else:
                lane_compute = dense_compute
            for _lane_tile in range(lane_tiles):
                advance_idle(state, lane_compute)
                for _context in range(contexts):
                    emit_packet(
                        state, service_cycles=service_cycles,
                        resident_packet_capacity=resident_packet_capacity,
                    )
    source_without_stalls = state["source_cycles"] - state["producer_stall_cycles"]
    drain_cycles = state["moment_work"]
    makespan = state["source_cycles"] + drain_cycles
    return {
        "source_cycles_without_stalls": source_without_stalls,
        "producer_stall_cycles": state["producer_stall_cycles"],
        "source_completion_cycles": state["source_cycles"],
        "barrier_drain_cycles": drain_cycles,
        "source_plus_moment_makespan_cycles": makespan,
        "output_packets": state["packets"],
        "moment_service_cycles": state["moment_service_cycles"],
        "maximum_resident_packets": state["maximum_resident_packets"],
    }


def load_exact_m17(m17_manifest_path: Path) -> tuple[
    dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray
]:
    manifest = json.loads(m17_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != M17_STATUS:
        raise ValueError("M17 manifest is not admitted")
    root = m17_manifest_path.parent
    files = manifest.get("files", {})
    for name in ("prototypes.json", "ordered_stream.npz"):
        entry = files.get(name)
        path = root / name
        if not isinstance(entry, dict) or not path.is_file():
            raise ValueError("M17 file identity is incomplete: " + name)
        if sha256(path) != entry.get("sha256") or path.stat().st_size != int(entry.get("bytes", -1)):
            raise ValueError("M17 file identity mismatch: " + name)
    prototypes = json.loads((root / "prototypes.json").read_text(encoding="utf-8"))
    with np.load(root / "ordered_stream.npz", allow_pickle=False) as stream:
        if set(stream.files) != {"operator_index", "population_cluster_id", "prototype_id"}:
            raise ValueError("M17 ordered stream field set mismatch")
        operator_index = stream["operator_index"].copy()
        population = stream["population_cluster_id"].copy()
        prototype_id = stream["prototype_id"].copy()
    ordered_clusters = exact_positive_int(manifest.get("ordered_clusters"), "ordered clusters")
    if not (
        len(operator_index) == len(population) == len(prototype_id) == ordered_clusters
    ):
        raise ValueError("M17 ordered stream cardinality mismatch")
    return manifest, prototypes, operator_index, population, prototype_id


def fair_serialized_region_cycles(
    m19_rows: list[dict[str, Any]], *, arithmetic_tiles: int, variant: str,
) -> int:
    """Recompute M19 serialization with the same moment arithmetic width.

    Comparing a 2/3/6-slice elastic scheduler with M19's fixed one-slice
    denominator incorrectly attributes replicated arithmetic to scheduling.
    This is the exact M19 phase equation with moment_lanes=16*tiles.
    """
    total = 0
    for row in m19_rows:
        elements = int(row["batchnorm"]["elements"])
        moment_cycles = ceil_div(elements, 16 * arithmetic_tiles)
        serialization = moment_cycles - int(row["source_output_cycles_at_96_lanes"])
        if serialization < 0:
            raise ValueError("fair M19 moment serialization became negative")
        phase_1 = max(
            int(row["source_cycles"][variant]) + serialization,
            int(row["one_way_memory_cycles"]),
        )
        phase_2 = max(
            int(row["consumer_cycles"]), int(row["one_way_memory_cycles"]),
        )
        total += phase_1 + phase_2
    return total


def require_schedule_equivalence(
    finite: dict[str, int | float], unbounded: dict[str, int | float],
    *, variants: tuple[str, ...],
) -> None:
    """Fail closed when a claimed sufficient FIFO changes any schedule field."""
    schedule_suffixes = (
        "source_cycles_without_stalls",
        "producer_stall_cycles",
        "source_completion_cycles",
        "barrier_drain_cycles",
        "source_plus_moment_makespan_cycles",
        "output_packets",
        "moment_service_cycles",
        "maximum_resident_packets",
        "payload_only_region_cycles",
    )
    for variant in variants:
        for suffix in schedule_suffixes:
            key = variant + "_" + suffix
            if finite.get(key) != unbounded.get(key):
                raise ValueError(
                    "finite FIFO schedule differs from unbounded: " + key
                )


def analyze(
    m17_manifest_path: Path, m17_reconciliation_path: Path,
    m19_path: Path,
) -> dict[str, Any]:
    m17, prototypes, operator_index, population, prototype_id = load_exact_m17(
        m17_manifest_path
    )
    m17_reconciliation = json.loads(m17_reconciliation_path.read_text(encoding="utf-8"))
    m19 = json.loads(m19_path.read_text(encoding="utf-8"))
    if m19.get("status") != M19_STATUS:
        raise ValueError("M19 artifact is not admitted")
    if m17_reconciliation.get("status") != "PASS_SAME_SAMPLE_EXACT_C4_SOURCE_AND_ORDERED_WALL_RECONCILIATION":
        raise ValueError("M17 reconciliation is not admitted")
    if m17_reconciliation["identities"]["oracle_manifest_sha256"] != sha256(m17_manifest_path):
        raise ValueError("M17 reconciliation identity mismatch")
    if (
        m19.get("identities", {}).get("m17_manifest_sha256") != sha256(m17_manifest_path)
        or m19.get("identities", {}).get("m17_reconciliation_sha256")
        != sha256(m17_reconciliation_path)
    ):
        raise ValueError("M19 is not identity-bound to the exact M17 inputs")
    expected_m19_resources = {
        "consumer_lanes": 16,
        "moment_lanes": 16,
        "source_output_lanes": 96,
        "input_bits": 32,
    }
    if any(
        int(m19.get("resource_model", {}).get(field, -1)) != expected
        for field, expected in expected_m19_resources.items()
    ):
        raise ValueError("M19 lane/width resource contract differs from M21")
    if not math.isclose(
        float(m19["resource_model"].get("frequency_mhz", -1.0)),
        333.333333333, rel_tol=0.0, abs_tol=1e-9,
    ) or not math.isclose(
        float(m19["resource_model"].get("dram_bandwidth_gbps", -1.0)),
        64.0, rel_tol=0.0, abs_tol=1e-12,
    ):
        raise ValueError("M19 frequency/payload bandwidth contract differs from M21")
    m19_by_operator = {
        (str(row["producer"]), int(row["producer_call_index"])): row
        for row in m19["rows"]
    }
    if len(m19_by_operator) != len(m19["rows"]):
        raise ValueError("duplicate M19 producer identity")

    variants = ("local", "hybrid", "same_width_dense")
    fifo_capacities = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    configurations: list[tuple[int, int | None]] = [
        (1, capacity) for capacity in fifo_capacities
    ] + [
        (2, 40), (2, 256), (3, 4), (3, 16), (3, 40),
    ] + [(tiles, None) for tiles in (1, 2, 3, 6)]
    rows = []
    totals: dict[str, dict[str, int]] = {}
    for arithmetic_tiles, capacity in configurations:
        label = "tiles{}_fifo{}".format(
            arithmetic_tiles, "unbounded" if capacity is None else capacity,
        )
        totals[label] = {}
        for variant in variants:
            aggregate = {
                "source_cycles_without_stalls": 0,
                "producer_stall_cycles": 0,
                "source_completion_cycles": 0,
                "barrier_drain_cycles": 0,
                "source_plus_moment_makespan_cycles": 0,
                "output_packets": 0,
                "moment_service_cycles": 0,
                "maximum_resident_packets": 0,
                "payload_only_region_cycles": 0,
            }
            for expected_index, operator in enumerate(m17["operators"]):
                if int(operator.get("operator_index", -1)) != expected_index:
                    raise ValueError("M17 operator indices are not dense")
                positions = np.flatnonzero(operator_index == expected_index)
                operator_population = operator.get("population_clusters")
                if len(positions) != operator_population:
                    raise ValueError("M17 operator population differs from ordered stream")
                if not np.array_equal(
                    population[positions], np.arange(len(positions), dtype=population.dtype)
                ):
                    raise ValueError("M17 population IDs are not dense and ordered per operator")
                key = (str(operator["name"]), int(operator["operator_call_index"]))
                if key not in m19_by_operator:
                    raise ValueError("M17 operator is absent from M19")
                m19_row = m19_by_operator[key]
                result = simulate_operator(
                    operator, prototype_id[positions], prototypes,
                    variant=variant, arithmetic_tiles=arithmetic_tiles,
                    resident_packet_capacity=capacity,
                )
                expected_source = int(m19_row["source_cycles"][variant])
                if result["source_cycles_without_stalls"] != expected_source:
                    raise ValueError("M21 reconstructed source cycles differ from M17/M19")
                if result["output_packets"] * 96 != int(m19_row["batchnorm"]["elements"]):
                    raise ValueError("M21 output packets do not cover the BN tensor")
                expected_moment = ceil_div(
                    int(m19_row["batchnorm"]["elements"]), 16 * arithmetic_tiles,
                )
                if result["moment_service_cycles"] != expected_moment:
                    raise ValueError("M21 moment service does not close to the BN tensor")
                phase_1 = max(
                    result["source_plus_moment_makespan_cycles"],
                    int(m19_row["one_way_memory_cycles"]),
                )
                phase_2 = max(
                    int(m19_row["consumer_cycles"]),
                    int(m19_row["one_way_memory_cycles"]),
                )
                result["payload_only_phase1_cycles"] = phase_1
                result["payload_only_phase2_cycles"] = phase_2
                result["payload_only_region_cycles"] = phase_1 + phase_2
                for field in aggregate:
                    if field == "maximum_resident_packets":
                        aggregate[field] = max(aggregate[field], result[field])
                    else:
                        aggregate[field] += result[field]
                if variant == "local" and arithmetic_tiles == 1 and capacity in {1, 4, 8, 32, 256, None}:
                    rows.append({
                        "operator": operator["name"],
                        "operator_call_index": int(operator["operator_call_index"]),
                        "configuration": label,
                        **result,
                    })
            for field, value in aggregate.items():
                totals[label][variant + "_" + field] = value

    exact_packets = sum(int(row["batchnorm"]["elements"]) for row in m19["rows"]) // 96
    serial = m19["summary"]["bounded_proposed_two_pass_cycles"]
    fair_serial = {
        tiles: {
            variant: fair_serialized_region_cycles(
                m19["rows"], arithmetic_tiles=tiles, variant=variant,
            )
            for variant in variants
        }
        for tiles in (1, 2, 3, 6)
    }
    if any(fair_serial[1][variant] != int(serial[variant]) for variant in variants):
        raise ValueError("one-tile fair serialization does not reproduce M19")
    for label, values in totals.items():
        if values["local_output_packets"] != exact_packets:
            raise ValueError("aggregate M21 packet population mismatch")
        arithmetic_tiles = int(label.split("_", 1)[0].removeprefix("tiles"))
        values["fair_m19_serialized_local_cycles"] = fair_serial[arithmetic_tiles]["local"]
        values["fair_m19_serialized_hybrid_cycles"] = fair_serial[arithmetic_tiles]["hybrid"]
        values["local_speedup_vs_same_tile_count_m19_serialized_moment"] = (
            fair_serial[arithmetic_tiles]["local"]
            / values["local_payload_only_region_cycles"]
        )
        values["hybrid_speedup_vs_same_tile_count_m19_serialized_moment"] = (
            fair_serial[arithmetic_tiles]["hybrid"]
            / values["hybrid_payload_only_region_cycles"]
        )
        values["local_speedup_vs_same_width_dense_shadow"] = (
            values["same_width_dense_payload_only_region_cycles"]
            / values["local_payload_only_region_cycles"]
        )
        values["hybrid_speedup_vs_same_width_dense_shadow"] = (
            values["same_width_dense_payload_only_region_cycles"]
            / values["hybrid_payload_only_region_cycles"]
        )
        if "unbounded" in label:
            configured_capacity_by_variant = {
                variant: values[variant + "_maximum_resident_packets"]
                for variant in variants
            }
        else:
            configured_capacity = int(label.rsplit("fifo", 1)[1])
            configured_capacity_by_variant = {
                variant: configured_capacity for variant in variants
            }
        for variant, configured_capacity in configured_capacity_by_variant.items():
            values[
                variant + "_required_fifo_payload_bits_at_configured_or_observed_maximum"
            ] = configured_capacity * 96 * 32
        dual_line_capacity = max(
            configured_capacity_by_variant["local"],
            configured_capacity_by_variant["hybrid"],
        )
        values["dual_line_required_fifo_payload_bits_at_configured_or_observed_maximum"] = (
            dual_line_capacity * 96 * 32
        )
        # Retain the r2 field name, but make it conservatively mean a hardware
        # FIFO that admits either deployed Local or Hybrid traffic.
        values["required_fifo_payload_bits_at_configured_or_observed_maximum"] = (
            values[
                "dual_line_required_fifo_payload_bits_at_configured_or_observed_maximum"
            ]
        )

    require_schedule_equivalence(
        totals["tiles3_fifo40"], totals["tiles3_fifounbounded"],
        variants=variants,
    )

    one_tile_unbounded = totals["tiles1_fifounbounded"]
    six_tile_unbounded = totals["tiles6_fifounbounded"]
    return {
        "schema": "m21_elastic_banked_moments_v1",
        "revision": 3,
        "status": M21_STATUS,
        "summary": {
            "operators": len(m17["operators"]),
            "ordered_population_clusters": len(operator_index),
            "exact_output_packets_96lane": exact_packets,
            "exact_moment_updates_16lane": exact_packets * 6,
            "m19_serialized_local_cycles": int(serial["local"]),
            "m19_serialized_hybrid_cycles": int(serial["hybrid"]),
            "one_tile_unbounded_local_cycles": one_tile_unbounded["local_payload_only_region_cycles"],
            "one_tile_unbounded_hybrid_cycles": one_tile_unbounded["hybrid_payload_only_region_cycles"],
            "one_tile_unbounded_local_speedup_vs_m19_serialized": one_tile_unbounded["local_speedup_vs_same_tile_count_m19_serialized_moment"],
            "one_tile_unbounded_local_speedup_vs_dense_shadow": one_tile_unbounded["local_speedup_vs_same_width_dense_shadow"],
            "six_tile_unbounded_local_cycles": six_tile_unbounded["local_payload_only_region_cycles"],
            "fair_m19_serialized_cycles_by_arithmetic_tiles": {
                str(tiles): fair_serial[tiles]["local"] for tiles in (1, 2, 3, 6)
            },
            "elastic_speedup_vs_fair_serialized_by_arithmetic_tiles": {
                str(tiles): (
                    fair_serial[tiles]["local"]
                    / totals["tiles{}_fifounbounded".format(tiles)]["local_payload_only_region_cycles"]
                )
                for tiles in (1, 2, 3, 6)
            },
            "three_tile_fifo40_local_cycles": totals["tiles3_fifo40"]["local_payload_only_region_cycles"],
            "three_tile_fifo40_maximum_resident_packets": totals["tiles3_fifo40"]["local_maximum_resident_packets"],
            "three_tile_fifo40_required_payload_bits": totals["tiles3_fifo40"]["required_fifo_payload_bits_at_configured_or_observed_maximum"],
            "three_tile_fifo40_matches_unbounded_all_variants": True,
            "one_tile_cycle_gap_vs_six_tile_unbounded": (
                one_tile_unbounded["local_payload_only_region_cycles"]
                - six_tile_unbounded["local_payload_only_region_cycles"]
            ),
            "maximum_moment_state_bits": max(
                int(row["batchnorm"]["moment_state_bits"]) for row in m19["rows"]
            ),
        },
        "configurations": totals,
        "selected_operator_rows": rows,
        "architecture_contract": {
            "source_packet": "one exact 96-channel M4 output vector",
            "moment_arithmetic_slice": "16 signed sum plus 16 unsigned square/sumsq updates",
            "state_organization": (
                "channel-resident banked raw-moment state; arithmetic slices rotate across "
                "six 16-channel subtiles per 96-channel packet"
            ),
            "decoupling": (
                "ready-valid output tap and bounded packet FIFO; producer descriptor/compute "
                "cycles drain moment work before the dynamic-BN barrier"
            ),
            "finite_fifo_capacity_includes_in_service_packet": True,
            "same_cycle_dequeue_enqueue_when_full": True,
            "fifo_bit_accounting": (
                "96x32 payload only; lane-tile address, operator/epoch/tag, first/last, "
                "and channel-bank sideband are explicitly excluded pending RTL; Local, "
                "Hybrid, and conservative dual-line capacities are reported separately"
            ),
            "barrier": "each operator drains all moment work before coefficient generation",
            "retained_materialization": "producer output write plus fused BN-ATLIF replay read",
        },
        "identities": {
            "m17_manifest_sha256": sha256(m17_manifest_path),
            "m17_prototypes_sha256": sha256(m17_manifest_path.parent / "prototypes.json"),
            "m17_ordered_stream_sha256": sha256(m17_manifest_path.parent / "ordered_stream.npz"),
            "m17_reconciliation_sha256": sha256(m17_reconciliation_path),
            "m19_sha256": sha256(m19_path),
            "source_sha256": sha256(Path(__file__).resolve()),
        },
        "claim_boundary": (
            "Exact ordered producer/moment scheduling DSE for the frozen H67 ep35 single "
            "sample and 13 direct M4-to-dynamic-BN-to-ATLIF regions. Region cycles retain "
            "M19's payload-only 64 GB/s and perfect compute/payload-overlap assumptions. "
            "The model is not full-network timing, measured DRAM, RTL proof of the banked "
            "scheduler, accuracy, energy, FPS, physical PPA, or speedup over an equally "
            "resourced implementation of the same elastic dataflow."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m17-manifest", type=Path, required=True)
    parser.add_argument("--m17-reconciliation", type=Path, required=True)
    parser.add_argument("--m19", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(
        args.m17_manifest.resolve(), args.m17_reconciliation.resolve(), args.m19.resolve()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = payload["summary"]
    print(
        "PASS_M21_ELASTIC_BANKED_MOMENTS local_cycles={} serial_speedup={:.6f} "
        "dense_shadow_speedup={:.6f}".format(
            summary["one_tile_unbounded_local_cycles"],
            summary["one_tile_unbounded_local_speedup_vs_m19_serialized"],
            summary["one_tile_unbounded_local_speedup_vs_dense_shadow"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
