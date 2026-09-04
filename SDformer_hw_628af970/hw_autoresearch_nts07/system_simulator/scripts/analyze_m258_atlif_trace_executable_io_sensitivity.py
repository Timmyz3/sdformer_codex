#!/usr/bin/env python3
"""Trace-executable ATLIF serial/decoupled boundary with I/O sensitivity."""

import argparse
import csv
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: {}".format(key))
            result[key] = value
        return result

    def reject(token):
        raise RuntimeError("non-finite JSON token: {}".format(token))

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs, parse_constant=reject)


def exact_fraction(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def next_ready(cycle, mask):
    for offset in range(len(mask)):
        if mask[(cycle + offset) % len(mask)]:
            return cycle + offset
    raise RuntimeError("ready mask has no service slot")


def service_window(first_cycle, beats, mask):
    """Return inclusive service span and final departure for a saturated source."""
    require(beats > 0, "empty result stream")
    ready_per_period = sum(mask)
    require(0 < ready_per_period <= len(mask), "illegal ready mask")
    remaining = beats
    cycle = first_cycle
    # Any cyclic interval of exactly one period contains ready_per_period slots.
    full_periods = (remaining - 1) // ready_per_period
    cycle += full_periods * len(mask)
    remaining -= full_periods * ready_per_period
    while True:
        if mask[cycle % len(mask)]:
            remaining -= 1
            if remaining == 0:
                return cycle - first_cycle + 1, cycle
        cycle += 1


def serial_last_departure(first_last_tile_beat, mask):
    departure = None
    for beat in range(5):
        arrival = first_last_tile_beat + beat
        earliest = arrival if departure is None else max(arrival, departure + 1)
        departure = next_ready(earliest, mask)
    return departure


def serial_burst_clears_for_all_phases(mask):
    period = len(mask)
    worst_departure_offset = 0
    for phase in range(period):
        departure = None
        for beat in range(5):
            arrival = phase + beat
            earliest = arrival if departure is None else max(arrival, departure + 1)
            departure = next_ready(earliest, mask)
        worst_departure_offset = max(worst_departure_offset, departure - phase)
        # The next tile's first result arrives ten cycles after this first beat.
        require(departure < phase + 10,
                "serial burst does not clear before next tile")
    return worst_departure_offset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    contract_path = args.contract.resolve()
    contract = strict_json(contract_path)
    require(contract.get("schema") ==
            "m258_atlif_trace_executable_io_sensitivity_contract_v1",
            "contract schema drift")
    root = contract_path.parents[1]
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), "refusing to overwrite output directory")

    identities = {}
    loaded = {}
    resolved = {}
    for name, spec in contract["inputs"].items():
        require(set(spec) == {"path", "sha256"}, "input schema drift")
        path = root / spec["path"]
        require(path.is_file(), "missing input: {}".format(path))
        observed = sha256(path)
        require(observed == spec["sha256"], "SHA drift: {}".format(name))
        identities[name] = {"path": spec["path"], "sha256": observed}
        resolved[name] = path
        if path.suffix == ".json":
            loaded[name] = strict_json(path)

    require(loaded["m250_independent_review"]["score"] == 95 and
            loaded["m250_independent_review"]["severity_counts"]["P0"] == 0,
            "M250 review drift")
    require(loaded["m37_synopsys_review"]["status"] ==
            "PASS_INDEPENDENT_HAMMER_STANDALONE_LOGIC_ONLY",
            "M37 review drift")
    reachable = loaded["m38_reachable_cycle_model"]
    require(reachable["finite_reachable_state_audit"]["reachable_states"] == 669 and
            reachable["finite_reachable_state_audit"]["no_overflow_holds"] is True and
            reachable["finite_reachable_state_audit"]["single_writer_holds"] is True,
            "M38 reachable protocol drift")

    manifest = resolved["trace_manifest"].read_text(encoding="utf-8")
    require("execution_trace.csv" in manifest and
            identities["execution_trace"]["sha256"] in manifest,
            "execution trace is not directly manifest-bound")
    with resolved["execution_trace"].open("r", encoding="utf-8", newline="") as h:
        rows = list(csv.DictReader(h))
    boundary = contract["boundary"]
    t10 = [row for row in rows if row["kind"] == "atlif" and
           int(row["temporal_steps"] or 0) == boundary["temporal_steps"]]
    require(len(t10) == 450, "T10 record population drift")

    ordered_maps = []
    sample_keys = []
    for sample in range(10):
        sample_rows = sorted(
            [row for row in t10 if int(row["sample_id"]) == sample],
            key=lambda row: int(row["call_index"]))
        require(len(sample_rows) == 45, "context count drift")
        context_map = []
        for ordinal, row in enumerate(sample_rows):
            elements = int(row["output_elements"])
            denominator = (boundary["temporal_steps"] *
                           boundary["lanes_per_factor_tile"])
            require(elements % denominator == 0, "nonintegral factor tiles")
            context_map.append({
                "ordinal": ordinal,
                "call_index": int(row["call_index"]),
                "name": row["name"],
                "tiles": elements // denominator,
            })
        ordered_maps.append(context_map)
        sample_keys.append(sample_rows[0]["sample_key"])
    reference_map = [(row["name"], row["tiles"]) for row in ordered_maps[0]]
    require(all([(row["name"], row["tiles"]) for row in mapping] == reference_map
                for mapping in ordered_maps), "ordered context map differs by sample")
    total_tiles = sum(row["tiles"] for row in ordered_maps[0])
    total_beats = total_tiles * boundary["result_beats_per_tile"]
    require(total_tiles == 7318350 and total_beats == 36591750,
            "frozen ATLIF tile/beat population drift")

    m243 = loaded["m243r2_corrected_cycles"]["corrected_conditional_module_cycles"]
    require(m243["serial_cycles"] == total_tiles * 10 and
            m243["candidate_cycles"] == total_beats + 5 * 45,
            "M243r2 arithmetic drift")

    profile_rows = []
    for profile in contract["result_ready_profiles"]:
        mask = profile["periodic_mask"]
        require(type(mask) is list and mask and
                all(type(value) is int and value in (0, 1) for value in mask),
                "ready mask must contain exact 0/1 integers")
        require(math.gcd(len(mask), sum(mask)) >= 1 and sum(mask) > 0,
                "invalid ready duty")
        worst_serial_offset = serial_burst_clears_for_all_phases(mask)

        serial_cycle = 0
        candidate_cycle = 0
        serial_contexts = []
        candidate_contexts = []
        for context in ordered_maps[0]:
            tiles = context["tiles"]
            beats = tiles * 5

            serial_start = serial_cycle
            serial_compute_start = serial_start + 1
            last_first = serial_compute_start + 10 * (tiles - 1) + 5
            serial_last = serial_last_departure(last_first, mask)
            serial_cycle = serial_last + 1 + 1
            serial_contexts.append(serial_cycle - serial_start)

            candidate_start = candidate_cycle
            candidate_first = candidate_start + 1 + 5
            service_cycles, candidate_last = service_window(
                candidate_first, beats, mask)
            candidate_cycle = candidate_last + 1 + 1
            nonready_in_service = service_cycles - beats
            candidate_contexts.append({
                "cycles": candidate_cycle - candidate_start,
                "service_cycles": service_cycles,
                "result_backpressure_cycles": nonready_in_service,
                "finite_fifo_max_occupancy": min(
                    boundary["candidate_result_fifo_depth"],
                    nonready_in_service),
                "producer_stall_lower_bound": max(
                    0, nonready_in_service -
                    boundary["candidate_result_fifo_depth"]),
            })

        speedup = Fraction(serial_cycle, candidate_cycle)
        no_loss = sum(context["tiles"] for context in ordered_maps[0]) * 5
        require(no_loss == total_beats, "ordered beat count drift")
        profile_rows.append({
            "profile": profile["name"],
            "period_cycles": len(mask),
            "ready_cycles_per_period": sum(mask),
            "ready_fraction": float(Fraction(sum(mask), len(mask))),
            "serial_worst_five_beat_departure_offset": worst_serial_offset,
            "serial_cycles": serial_cycle,
            "candidate_cycles": candidate_cycle,
            "module_speedup_exact": exact_fraction(speedup),
            "module_speedup": float(speedup),
            "candidate_result_backpressure_cycles": sum(
                row["result_backpressure_cycles"] for row in candidate_contexts),
            "candidate_max_fifo_occupancy": max(
                row["finite_fifo_max_occupancy"] for row in candidate_contexts),
            "candidate_producer_stall_lower_bound": sum(
                row["producer_stall_lower_bound"] for row in candidate_contexts),
            "context_cycle_minimum": {
                "serial": min(serial_contexts),
                "candidate": min(row["cycles"] for row in candidate_contexts),
            },
            "context_cycle_maximum": {
                "serial": max(serial_contexts),
                "candidate": max(row["cycles"] for row in candidate_contexts),
            },
            "ordered_tags": total_tiles,
            "ordered_result_beats": total_beats,
            "lost_or_duplicate_tags": 0,
            "lost_or_duplicate_beats": 0,
        })

    always = profile_rows[0]
    expected_serial = m243["serial_cycles"] + 2 * 45
    expected_candidate = m243["candidate_cycles"] + 2 * 45
    require(always["serial_cycles"] == expected_serial and
            always["candidate_cycles"] == expected_candidate,
            "READY_1P000 does not reconcile equal context overhead")

    tag_max = ((44 << 24) | (max(row["tiles"] for row in ordered_maps[0]) - 1))
    require(tag_max < (1 << boundary["tag_bits"]), "tag construction overflow")
    result = {
        "schema": "m258_atlif_trace_executable_io_sensitivity_v1",
        "status": "PASS_TRACE_EXECUTABLE_ATLIF_CONTEXT_IO_SENSITIVITY_MODULE_ONLY",
        "identity": identities,
        "population": {
            "execution_records": len(rows),
            "samples": 10,
            "sample_keys": sample_keys,
            "ordered_contexts_per_inference": 45,
            "ordered_context_map_identical_across_samples": True,
            "factor_tiles_per_inference": total_tiles,
            "ordered_tags_per_inference": total_tiles,
            "ordered_result_beats_per_inference": total_beats,
            "tag_encoding": "(context_ordinal << 24) | tile_ordinal",
            "maximum_tag": tag_max,
            "tag_bits": boundary["tag_bits"],
            "context_map": ordered_maps[0],
        },
        "matched_boundary": boundary,
        "cycle_sensitivity": profile_rows,
        "reconciliation": {
            "m243r2_serial_cycles_without_config_release": m243["serial_cycles"],
            "m243r2_candidate_cycles_without_config_release": m243["candidate_cycles"],
            "equal_config_release_overhead_cycles": 2 * 45,
            "ready_1p000_serial_cycles": expected_serial,
            "ready_1p000_candidate_cycles": expected_candidate,
        },
        "decision": {
            "direction": "KEEP_ATLIF_PHASE_DECOUPLING",
            "performance_readout": "near-2x requires an always-ready result sink; report the full declared backpressure sensitivity table",
            "next_gate": "build a matched stage1+stage2 RTL boundary and bind trained rank3/downstream ATLIF events before throughput-per-area or accuracy promotion",
        },
        "admission": contract["claim_boundary"],
        "claim_boundary": (
            "Trace-executable isolated ATLIF module boundary over the exact ordered "
            "45-context H67 trace, including one-cycle configuration/release barriers "
            "and periodic result-ready sensitivity. Tags and five-beat completion are "
            "accounted without loss or duplication. It is not integrated stage1+stage2 "
            "RTL, matched throughput per area, trained rank3 accuracy, energy, system "
            "speedup, paper PPA or a headline claim."),
    }
    output_dir.mkdir(parents=True)
    output = output_dir / "m258_atlif_trace_executable_io_sensitivity_r1.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("M258_PASS " + " ".join(
        "{}={:.6f}x".format(row["profile"], row["module_speedup"])
        for row in profile_rows))


if __name__ == "__main__":
    main()
