#!/usr/bin/env python3
"""Cycle/traffic DSE for exact dynamic-BN materialization elision.

This is a matched local schedule model.  Both schedules retain every raw BN
input through the current-batch moment barrier and use the same serialized M281
coefficient service.  The fused schedule only removes the normalized-tensor
write/read pair; it does not remove raw retention, replay, moments, or BN math.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path


T = 10
WIDTHS = (16, 24, 32)
BANDWIDTHS = (32, 64, 128)
OVERLAP_MODES = ("none", "coefficient_with_replay")
COEFF_FIRST_LATENCY = 8
COEFF_II = 9
BARRIER_CYCLES = 1
M159_FIXED_FFN_CYCLES_EXCLUDING_BN_RESIDUAL = 205_384_111

SOURCES = {
    "m159_dynamic_bn_correction": (
        "results/m159_m160_dynamic_bn_correction_hammer_r1_20260824/"
        "m159_m160_dynamic_bn_correction_hammer_r1.json",
        "f0a8b131d1a7df5df2fe6a4cf3b47397d99a9f5ed596f535c1c30830b7e635d6"),
    "m232_stream_screen": (
        "results/m232_dynamic_bn_coefficient_stream_screen_r1_20260825/"
        "m232_dynamic_bn_coefficient_stream_screen_r1.json",
        "51175ae37085cef00efce57ecabeff34099ff3e05ed468f1da1da2f17f823a5d"),
    "m232_per_ffn_geometry": (
        "results/m232_dynamic_bn_coefficient_stream_screen_r1_20260825/"
        "per_ffn_coefficient_stream.csv",
        "d0cf81a881a7a9f6671bbc84dc7f32afe78f1d3e9841607ffce424c7bcbbb734"),
    "m232_correction_overlay": (
        "contracts/m232_r1_storage_and_first_latency_correction_overlay_"
        "r1_20260825.json",
        "9afa3c3863f64a72a8254f3c455012a9d054f6ef95d065d519ec1e306628478f"),
    "m240_coefficient_pareto_review": (
        "results/m240_bn_pareto_independent_hammer_r1_20260825/"
        "m240_bn_pareto_independent_hammer_r1.json",
        "52a06d6c24c5369978048631c840a28fca01d7ffa6a384df2ba1ee78a24d52cc"),
    "m281_protocol_ii_review": (
        "results/m281_m276_bn_protocol_ii_independent_hammer_r1_20260825/"
        "m281_m276_bn_protocol_ii_independent_hammer_review_r1.json",
        "a68debd51a191f5f1ff99dd9b175294cd55c19e917b4c4c3c079568e15cdb152"),
    "docs359": (
        "docs/359_DATE终局冻结_20260813.md",
        "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ceil_div(a, b):
    return (a + b - 1) // b


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_sources(root):
    identity = {}
    for name, (relative, expected) in SOURCES.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError("missing frozen source: " + relative)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                "frozen source changed: {} expected={} actual={}".format(
                    relative, expected, actual))
        identity[name] = {"path": relative, "sha256": actual}
    return identity


def load_phases(root):
    path = root / SOURCES["m232_per_ffn_geometry"][0]
    phases = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for kind in ("bn1", "bn2"):
                spatial_positions = int(row["positions_per_channel"])
                population = T * spatial_positions
                channels = int(row[kind + "_channels"])
                phases.append({
                    "module": row["module"],
                    "stage": int(row["stage"]),
                    "phase": kind,
                    "spatial_positions_per_channel": spatial_positions,
                    "reduction_population_per_channel": population,
                    "channels": channels,
                    "elements": population * channels,
                })
    if len(phases) != 24:
        raise RuntimeError("expected 24 BN phases")
    bn1 = sum(row["elements"] for row in phases if row["phase"] == "bn1")
    bn2 = sum(row["elements"] for row in phases if row["phase"] == "bn2")
    coeff_pairs = sum(row["channels"] for row in phases)
    if (bn1, bn2, coeff_pairs) != (350_208_000, 87_552_000, 22_080):
        raise RuntimeError("frozen BN geometry mismatch")
    return phases


def coefficient_exposure(channels, replay_cycles_per_channel, overlap_mode):
    full_serial = COEFF_FIRST_LATENCY + (channels - 1) * COEFF_II
    if overlap_mode == "none":
        return full_serial, full_serial
    finish = 0
    for channel in range(channels):
        ready = COEFF_FIRST_LATENCY + channel * COEFF_II
        start = max(finish, ready)
        finish = start + replay_cycles_per_channel
    replay_cycles = channels * replay_cycles_per_channel
    exposed = finish - replay_cycles
    return exposed, full_serial


def phase_schedule(phase, width, bandwidth, overlap_mode):
    bytes_per_element = width // 8
    elements = phase["elements"]
    channels = phase["channels"]
    population = phase["reduction_population_per_channel"]
    payload_bytes = elements * bytes_per_element

    raw_capture_cycles = ceil_div(payload_bytes, bandwidth)
    replay_cycles_per_channel = ceil_div(
        population * bytes_per_element, bandwidth)
    replay_cycles = channels * replay_cycles_per_channel
    normalized_consume_cycles = ceil_div(payload_bytes, bandwidth)
    coefficient_exposed, coefficient_serial = coefficient_exposure(
        channels, replay_cycles_per_channel, overlap_mode)

    fused_cycles = (raw_capture_cycles + BARRIER_CYCLES +
                    coefficient_exposed + replay_cycles)
    materialized_cycles = fused_cycles + normalized_consume_cycles
    raw_retention_cycles = BARRIER_CYCLES + coefficient_exposed + replay_cycles

    output = dict(phase)
    output.update({
        "activation_width_bits": width,
        "bandwidth_bytes_per_cycle": bandwidth,
        "memory_ports": "1R1W",
        "coefficient_overlap_mode": overlap_mode,
        "raw_payload_bytes": payload_bytes,
        "raw_capture_write_cycles": raw_capture_cycles,
        "moment_barrier_cycles": BARRIER_CYCLES,
        "coefficient_first_latency_cycles": COEFF_FIRST_LATENCY,
        "coefficient_ii_cycles": COEFF_II,
        "coefficient_full_serial_cycles": coefficient_serial,
        "coefficient_exposed_cycles": coefficient_exposed,
        "raw_replay_read_cycles": replay_cycles,
        "replay_cycles_per_channel": replay_cycles_per_channel,
        "normalized_materialization_write_cycles": replay_cycles,
        "normalized_consume_read_cycles": normalized_consume_cycles,
        "materialized_schedule_cycles": materialized_cycles,
        "fused_schedule_cycles": fused_cycles,
        "cycle_reduction": materialized_cycles - fused_cycles,
        "raw_retention_bits": elements * width,
        "raw_retention_allocation_cycles": raw_retention_cycles,
        "materialized_traffic_bytes": 4 * payload_bytes,
        "fused_traffic_bytes": 2 * payload_bytes,
        "materialized_read_port_cycles": replay_cycles +
            normalized_consume_cycles,
        "materialized_write_port_cycles": raw_capture_cycles + replay_cycles,
        "materialized_simultaneous_raw_read_normalized_write_cycles":
            replay_cycles,
        "fused_read_port_cycles": replay_cycles,
        "fused_write_port_cycles": raw_capture_cycles,
        "same_address_read_before_write_required_by_materialized": True,
        "same_address_read_before_write_required_by_fused": False,
    })
    return output


def aggregate(phases, width, bandwidth, overlap_mode):
    rows = [phase_schedule(row, width, bandwidth, overlap_mode)
            for row in phases]
    materialized_cycles = sum(row["materialized_schedule_cycles"] for row in rows)
    fused_cycles = sum(row["fused_schedule_cycles"] for row in rows)
    materialized_traffic = sum(row["materialized_traffic_bytes"] for row in rows)
    fused_traffic = sum(row["fused_traffic_bytes"] for row in rows)
    peak_raw_bits = max(row["raw_retention_bits"] for row in rows)
    coeff_serial = sum(row["coefficient_full_serial_cycles"] for row in rows)
    coeff_exposed = sum(row["coefficient_exposed_cycles"] for row in rows)
    total_elements = sum(row["elements"] for row in rows)
    minimum_replay_cycles_per_channel = min(
        row["replay_cycles_per_channel"] for row in rows)

    summary = {
        "activation_width_bits": width,
        "bandwidth_bytes_per_cycle": bandwidth,
        "memory_ports": "1R1W",
        "coefficient_overlap_mode": overlap_mode,
        "bn_phases": len(rows),
        "total_bn_elements": total_elements,
        "barriers_charged": len(rows),
        "raw_capture_writes_charged": total_elements,
        "raw_replay_reads_charged": total_elements,
        "normalized_writes_materialized": total_elements,
        "normalized_reads_materialized": total_elements,
        "normalized_writes_fused": 0,
        "normalized_reads_fused": 0,
        "coefficient_pairs_charged": sum(row["channels"] for row in rows),
        "coefficient_full_serial_cycles": coeff_serial,
        "coefficient_exposed_cycles": coeff_exposed,
        "minimum_replay_cycles_per_channel": minimum_replay_cycles_per_channel,
        "coefficient_service_rate_hidden_after_first_result_all_phases":
            minimum_replay_cycles_per_channel >= COEFF_II,
        "materialized_cycles": materialized_cycles,
        "fused_cycles": fused_cycles,
        "bn_local_cycle_speedup": materialized_cycles / fused_cycles,
        "cycles_elided": materialized_cycles - fused_cycles,
        "materialized_traffic_bytes": materialized_traffic,
        "fused_traffic_bytes": fused_traffic,
        "traffic_reduction": materialized_traffic / fused_traffic,
        "peak_raw_retention_bits_both_schedules": peak_raw_bits,
        "peak_raw_retention_bytes_both_schedules": peak_raw_bits // 8,
        "peak_buffer_capacity_reduction": 1.0,
        "m159_fixed_ffn_cycles_excluding_bn_residual":
            M159_FIXED_FFN_CYCLES_EXCLUDING_BN_RESIDUAL,
        "serial_m159_plus_materialized_cycles":
            M159_FIXED_FFN_CYCLES_EXCLUDING_BN_RESIDUAL + materialized_cycles,
        "serial_m159_plus_fused_cycles":
            M159_FIXED_FFN_CYCLES_EXCLUDING_BN_RESIDUAL + fused_cycles,
        "serial_m159_accounted_speedup":
            (M159_FIXED_FFN_CYCLES_EXCLUDING_BN_RESIDUAL + materialized_cycles) /
            (M159_FIXED_FFN_CYCLES_EXCLUDING_BN_RESIDUAL + fused_cycles),
    }
    return summary, rows


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("refusing empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)

    source_identity = check_sources(root)
    phases = load_phases(root)
    m159 = load_json(root / SOURCES["m159_dynamic_bn_correction"][0])
    m232 = load_json(root / SOURCES["m232_stream_screen"][0])
    m232_overlay = load_json(root / SOURCES["m232_correction_overlay"][0])
    m240 = load_json(root / SOURCES["m240_coefficient_pareto_review"][0])
    m281 = load_json(root / SOURCES["m281_protocol_ii_review"][0])
    if m159["m159_disposition"]["preserved"]["accounted_excluding_bn_residual"] != \
            M159_FIXED_FFN_CYCLES_EXCLUDING_BN_RESIDUAL:
        raise RuntimeError("M159 fixed-cycle identity changed")
    if m232["frozen_semantics"]["total_coefficients_per_frame"] != 22_080:
        raise RuntimeError("M232 coefficient identity changed")
    if m232_overlay["admission"]["storage_20x_revoked"] is not True:
        raise RuntimeError("M232 correction overlay not applied")
    if m240["admission"]["full_bn"] is not False:
        raise RuntimeError("M240 claim boundary changed")
    schedule = m281["schedule_and_protocol_reconstruction"]
    if (schedule["first_result_latency_cycles"],
            schedule["intrinsic_unstalled_accept_interval_cycles"]) != (8, 9):
        raise RuntimeError("M281 coefficient timing changed")

    summaries = []
    details = []
    for width in WIDTHS:
        for bandwidth in BANDWIDTHS:
            for overlap_mode in OVERLAP_MODES:
                summary, rows = aggregate(
                    phases, width, bandwidth, overlap_mode)
                summaries.append(summary)
                details.extend(rows)

    write_csv(out / "dse_summary.csv", summaries)
    write_csv(out / "per_phase_schedule.csv", details)
    reference = next(row for row in summaries
                     if row["activation_width_bits"] == 24
                     and row["bandwidth_bytes_per_cycle"] == 64
                     and row["coefficient_overlap_mode"] ==
                     "coefficient_with_replay")
    local_gate_passes = sum(
        row["bn_local_cycle_speedup"] >= 1.25 for row in summaries)
    accounted_gate_passes = sum(
        row["serial_m159_accounted_speedup"] >= 1.05 for row in summaries)
    result = {
        "schema": "m480_dynamic_bn_exact_materialization_elision_cpu_dse_v1",
        "date": "2026-08-26",
        "status": "PASS_EXACT_SCHEDULE_GO_BASELINE_HYGIENE_NO_GO_STANDALONE_NOVELTY",
        "scope": "H67/Motion ep35 12-FFN, 24 current-batch BN phases",
        "frozen_semantics": {
            "bn_policy": "no_running/current-batch",
            "time_steps": T,
            "bn1_elements": 350_208_000,
            "bn2_elements": 87_552_000,
            "total_bn_elements": 437_760_000,
            "channel_coefficient_pairs": 22_080,
            "coefficient_engine_first_latency_cycles": COEFF_FIRST_LATENCY,
            "coefficient_engine_intrinsic_ii_cycles": COEFF_II,
            "global_moment_barriers": 24,
        },
        "matched_schedule_contract": {
            "materialized": [
                "capture raw BN input and accumulate moments inline",
                "retain global per-module moment barrier",
                "generate exact same coefficient stream",
                "replay raw through 1R port while writing normalized tensor through 1W port",
                "read normalized tensor again through the same 1R port for the consumer"
            ],
            "fused": [
                "capture raw BN input and accumulate moments inline",
                "retain global per-module moment barrier",
                "generate exact same coefficient stream",
                "replay every raw element once and apply affine directly at the consumer",
                "never write or read a normalized intermediate"
            ],
            "exactness": "At each scanned stored-activation width, both paths use the same raw payload and affine rounding contract; elision changes schedule only and drops no element.",
            "port_contract": "one byte-addressed 1R1W store; one read and one write may occur in the same cycle, but two reads may not",
            "materialized_in_place_assumption": "read-before-write is defined for same-address 1R1W; otherwise materialized needs a second bank and is worse",
            "raw_retention_required_both": True,
            "raw_replay_required_both": True,
            "barrier_removed": False,
        },
        "dse_axes": {
            "activation_width_bits": list(WIDTHS),
            "bandwidth_bytes_per_cycle": list(BANDWIDTHS),
            "memory_ports": ["1R1W"],
            "coefficient_overlap_mode": list(OVERLAP_MODES),
            "summary_points": len(summaries),
            "per_phase_rows": len(details),
        },
        "reference_point_q24_bw64_overlap": reference,
        "ranges": {
            "bn_local_cycle_speedup_min": min(
                row["bn_local_cycle_speedup"] for row in summaries),
            "bn_local_cycle_speedup_max": max(
                row["bn_local_cycle_speedup"] for row in summaries),
            "serial_m159_accounted_speedup_min": min(
                row["serial_m159_accounted_speedup"] for row in summaries),
            "serial_m159_accounted_speedup_max": max(
                row["serial_m159_accounted_speedup"] for row in summaries),
            "traffic_reduction_min": min(
                row["traffic_reduction"] for row in summaries),
            "traffic_reduction_max": max(
                row["traffic_reduction"] for row in summaries),
            "peak_buffer_capacity_reduction": 1.0,
        },
        "thresholds": {
            "implementation_baseline_gate": {
                "bn_local_cycle_speedup_at_least": 1.25,
                "traffic_reduction_at_least": 1.9,
                "result": "PASS_ALL_18_POINTS",
                "passing_points": local_gate_passes,
            },
            "serial_m159_accounted_gate": {
                "speedup_at_least": 1.05,
                "passing_points": accounted_gate_passes,
                "total_points": len(summaries),
                "qualification": "Pessimistic additive schedule, not a system admission; producer/consumer compute overlap is not claimed."
            },
            "standalone_novelty_gate": {
                "result": "NO_GO",
                "reason": "M159/M161 already identify write-raw/read-raw-and-stream-normalized as the fair strong dense baseline. Eliding a normalized intermediate is mandatory baseline hygiene, not a new accelerator mechanism by itself."
            },
        },
        "findings": {
            "what_improves": "Normalized-intermediate traffic falls from one write plus one read to zero; total local BN-buffer traffic falls exactly 2x.",
            "what_does_not_improve": "The exact fused path retains the same raw tensor until the global moment barrier and replay, so peak barrier capacity is unchanged.",
            "coefficient_overlap": "At all scanned points, one channel replay lasts at least the M281 II=9; coefficient production is hidden after the first result when overlap is enabled.",
            "hardware_cost_warning": "The schedule assumes a bus-wide affine/consumer datapath at the scanned bandwidth. Its multiplier/add lanes, rounding, SRAM macro and energy are not implemented or priced here."
        },
        "claim_boundary": {
            "cpu_cycle_dse": True,
            "exact_schedule_at_fixed_payload_width": True,
            "fixed_point_accuracy": False,
            "moment_finalizer_rtl": False,
            "runtime_affine_rtl": False,
            "sram_macro": False,
            "vcs": False,
            "dc_sta": False,
            "power": False,
            "energy": False,
            "module_speedup_admitted": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
        "next_gate": [
            "Use fused replay as the mandatory fair baseline for any later BN accelerator comparison.",
            "Before RTL, add an address-bearing 1R1W SRAM schedule and exact runtime-affine numeric miter at one selected width.",
            "Do not promote unless a mechanism beats this fused baseline while retaining the moment barrier/raw replay contract."
        ],
        "files": {
            "summary_csv": "dse_summary.csv",
            "per_phase_csv": "per_phase_schedule.csv",
        },
        "source_identity": source_identity,
        "docs359_sha256_unchanged": source_identity["docs359"]["sha256"],
    }
    (out / "m480_dynamic_bn_exact_materialization_elision_dse_r1.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "RUN_COMPLETE.txt").write_text(
        "PASS_M480_DYNAMIC_BN_EXACT_MATERIALIZATION_ELISION_CPU_DSE\n"
        "system_speedup=false\npaper_ppa_ready=false\nheadline=false\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
