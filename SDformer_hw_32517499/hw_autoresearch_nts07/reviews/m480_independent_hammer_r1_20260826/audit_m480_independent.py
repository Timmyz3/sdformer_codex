#!/opt/anaconda3/envs/pytorch310/bin/python
"""Receipt-blind independent audit of the sealed M480 dynamic-BN DSE.

The independent expectations are fully reconstructed from the frozen upstream
geometry before this program opens any M480 producer CSV or JSON receipt.  The
producer analyzer is never imported or executed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


T = 10
WIDTHS = (16, 24, 32)
BANDWIDTHS = (32, 64, 128)
OVERLAP_MODES = ("none", "coefficient_with_replay")
COEFF_FIRST_LATENCY = 8
COEFF_II = 9
BARRIER_CYCLES = 1
M159_FIXED_CYCLES = 205_384_111
EXPECTED_ANALYZER_SHA = "b5f99fc8517f9a89c58ff2502e2560759bf9ef906735ed16568f69aa8f3bbb40"
EXPECTED_DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

UPSTREAM = {
    "m159": (
        "results/m159_m160_dynamic_bn_correction_hammer_r1_20260824/"
        "m159_m160_dynamic_bn_correction_hammer_r1.json",
        "f0a8b131d1a7df5df2fe6a4cf3b47397d99a9f5ed596f535c1c30830b7e635d6",
    ),
    "geometry": (
        "results/m232_dynamic_bn_coefficient_stream_screen_r1_20260825/"
        "per_ffn_coefficient_stream.csv",
        "d0cf81a881a7a9f6671bbc84dc7f32afe78f1d3e9841607ffce424c7bcbbb734",
    ),
    "m281": (
        "results/m281_m276_bn_protocol_ii_independent_hammer_r1_20260825/"
        "m281_m276_bn_protocol_ii_independent_hammer_review_r1.json",
        "a68debd51a191f5f1ff99dd9b175294cd55c19e917b4c4c3c079568e15cdb152",
    ),
    "m161_fair_overlay": (
        "contracts/m161_r1_width_and_fair_baseline_correction_overlay_r1_20260824.json",
        "f59a44685bfc1bdf283a6e6e4a94f1189d3ad384fe211cc381611edec334a6cd",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ceil_div(left: int, right: int) -> int:
    return (left + right - 1) // right


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def require(condition: bool, label: str, failures: list[str], checks: Counter[str]) -> None:
    checks["performed"] += 1
    if not condition:
        failures.append(label)


def coefficient_exposure(channels: int, replay_per_channel: int, overlap: str) -> tuple[int, int]:
    full_serial = COEFF_FIRST_LATENCY + (channels - 1) * COEFF_II
    if overlap == "none":
        return full_serial, full_serial
    finish = 0
    for channel in range(channels):
        ready = COEFF_FIRST_LATENCY + channel * COEFF_II
        finish = max(finish, ready) + replay_per_channel
    return finish - channels * replay_per_channel, full_serial


def expected_phase(phase: dict[str, object], width: int, bandwidth: int, overlap: str) -> dict[str, object]:
    bytes_per_element = width // 8
    elements = int(phase["elements"])
    channels = int(phase["channels"])
    population = int(phase["reduction_population_per_channel"])
    payload = elements * bytes_per_element
    capture = ceil_div(payload, bandwidth)
    replay_per_channel = ceil_div(population * bytes_per_element, bandwidth)
    replay = channels * replay_per_channel
    consume = ceil_div(payload, bandwidth)
    exposed, full_serial = coefficient_exposure(channels, replay_per_channel, overlap)
    fused = capture + BARRIER_CYCLES + exposed + replay
    materialized = fused + consume
    return {
        **phase,
        "activation_width_bits": width,
        "bandwidth_bytes_per_cycle": bandwidth,
        "memory_ports": "1R1W",
        "coefficient_overlap_mode": overlap,
        "raw_payload_bytes": payload,
        "raw_capture_write_cycles": capture,
        "moment_barrier_cycles": BARRIER_CYCLES,
        "coefficient_first_latency_cycles": COEFF_FIRST_LATENCY,
        "coefficient_ii_cycles": COEFF_II,
        "coefficient_full_serial_cycles": full_serial,
        "coefficient_exposed_cycles": exposed,
        "raw_replay_read_cycles": replay,
        "replay_cycles_per_channel": replay_per_channel,
        "normalized_materialization_write_cycles": replay,
        "normalized_consume_read_cycles": consume,
        "materialized_schedule_cycles": materialized,
        "fused_schedule_cycles": fused,
        "cycle_reduction": materialized - fused,
        "raw_retention_bits": elements * width,
        "raw_retention_allocation_cycles": BARRIER_CYCLES + exposed + replay,
        "materialized_traffic_bytes": 4 * payload,
        "fused_traffic_bytes": 2 * payload,
        "materialized_read_port_cycles": replay + consume,
        "materialized_write_port_cycles": capture + replay,
        "materialized_simultaneous_raw_read_normalized_write_cycles": replay,
        "fused_read_port_cycles": replay,
        "fused_write_port_cycles": capture,
        "same_address_read_before_write_required_by_materialized": True,
        "same_address_read_before_write_required_by_fused": False,
    }


def expected_summary(rows: list[dict[str, object]], width: int, bandwidth: int, overlap: str) -> dict[str, object]:
    materialized = sum(int(row["materialized_schedule_cycles"]) for row in rows)
    fused = sum(int(row["fused_schedule_cycles"]) for row in rows)
    materialized_traffic = sum(int(row["materialized_traffic_bytes"]) for row in rows)
    fused_traffic = sum(int(row["fused_traffic_bytes"]) for row in rows)
    elements = sum(int(row["elements"]) for row in rows)
    min_replay = min(int(row["replay_cycles_per_channel"]) for row in rows)
    return {
        "activation_width_bits": width,
        "bandwidth_bytes_per_cycle": bandwidth,
        "memory_ports": "1R1W",
        "coefficient_overlap_mode": overlap,
        "bn_phases": len(rows),
        "total_bn_elements": elements,
        "barriers_charged": len(rows),
        "raw_capture_writes_charged": elements,
        "raw_replay_reads_charged": elements,
        "normalized_writes_materialized": elements,
        "normalized_reads_materialized": elements,
        "normalized_writes_fused": 0,
        "normalized_reads_fused": 0,
        "coefficient_pairs_charged": sum(int(row["channels"]) for row in rows),
        "coefficient_full_serial_cycles": sum(int(row["coefficient_full_serial_cycles"]) for row in rows),
        "coefficient_exposed_cycles": sum(int(row["coefficient_exposed_cycles"]) for row in rows),
        "minimum_replay_cycles_per_channel": min_replay,
        "coefficient_service_rate_hidden_after_first_result_all_phases": min_replay >= COEFF_II,
        "materialized_cycles": materialized,
        "fused_cycles": fused,
        "bn_local_cycle_speedup": materialized / fused,
        "cycles_elided": materialized - fused,
        "materialized_traffic_bytes": materialized_traffic,
        "fused_traffic_bytes": fused_traffic,
        "traffic_reduction": materialized_traffic / fused_traffic,
        "peak_raw_retention_bits_both_schedules": max(int(row["raw_retention_bits"]) for row in rows),
        "peak_raw_retention_bytes_both_schedules": max(int(row["raw_retention_bits"]) for row in rows) // 8,
        "peak_buffer_capacity_reduction": 1.0,
        "m159_fixed_ffn_cycles_excluding_bn_residual": M159_FIXED_CYCLES,
        "serial_m159_plus_materialized_cycles": M159_FIXED_CYCLES + materialized,
        "serial_m159_plus_fused_cycles": M159_FIXED_CYCLES + fused,
        "serial_m159_accounted_speedup": (M159_FIXED_CYCLES + materialized) / (M159_FIXED_CYCLES + fused),
    }


def csv_value_matches(actual: str, expected: object) -> bool:
    if isinstance(expected, bool):
        return actual == str(expected)
    if isinstance(expected, int):
        return int(actual) == expected
    if isinstance(expected, float):
        return close(float(actual), expected)
    return actual == str(expected)


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    hw = repo / "hw_autoresearch_nts07"
    producer = hw / "results/m480_dynamic_bn_exact_materialization_elision_dse_r1_20260826"
    failures: list[str] = []
    checks: Counter[str] = Counter()

    # Phase 1: receipt-blind reconstruction from frozen upstream evidence.
    upstream_identity = {}
    for name, (relative, expected_hash) in UPSTREAM.items():
        path = hw / relative
        actual_hash = sha256(path)
        require(actual_hash == expected_hash, f"upstream SHA mismatch: {name}", failures, checks)
        upstream_identity[name] = {"path": relative, "sha256": actual_hash}

    geometry_path = hw / UPSTREAM["geometry"][0]
    with geometry_path.open(newline="", encoding="utf-8") as stream:
        geometry_rows = list(csv.DictReader(stream))
    require(len(geometry_rows) == 12, "geometry does not contain 12 FFNs", failures, checks)
    require(len({row["module"] for row in geometry_rows}) == 12, "duplicate FFN module", failures, checks)
    require(Counter(int(row["stage"]) for row in geometry_rows) == Counter({0: 2, 1: 2, 2: 6, 3: 2}), "stage population drift", failures, checks)

    phases = []
    for source in geometry_rows:
        positions = int(source["positions_per_channel"])
        for phase_name in ("bn1", "bn2"):
            channels = int(source[phase_name + "_channels"])
            phases.append({
                "module": source["module"],
                "stage": int(source["stage"]),
                "phase": phase_name,
                "spatial_positions_per_channel": positions,
                "reduction_population_per_channel": T * positions,
                "channels": channels,
                "elements": T * positions * channels,
            })
    require(len(phases) == 24, "phase count is not 24", failures, checks)
    bn1_elements = sum(int(row["elements"]) for row in phases if row["phase"] == "bn1")
    bn2_elements = sum(int(row["elements"]) for row in phases if row["phase"] == "bn2")
    coefficient_pairs = sum(int(row["channels"]) for row in phases)
    require(bn1_elements == 350_208_000, "BN1 extent drift", failures, checks)
    require(bn2_elements == 87_552_000, "BN2 extent drift", failures, checks)
    require(coefficient_pairs == 22_080, "coefficient-pair extent drift", failures, checks)

    m159 = json.loads((hw / UPSTREAM["m159"][0]).read_text())
    direct = m159["direct_model_audit"]
    semantics = m159["bn_reduction_semantics"]
    require(direct["config_bn_policy"] == "no_running", "BN policy is not no_running", failures, checks)
    require(direct["ffn_bn_modules"] == 24 and direct["ffn_bn_step_mode_m"] == 24, "24 FFN dynamic BN modules not proven", failures, checks)
    require(direct["bn_after_track_running_false"] == 78, "running-stat tracking remains", failures, checks)
    require(direct["bn_after_running_buffers_none"] == 78, "running buffers remain", failures, checks)
    require(semantics["moment_axes"] == ["time", "batch", "height", "width"], "current-batch reduction axes drift", failures, checks)
    require(semantics["per_channel_population"] == "M=T*B*H*W", "current-batch population drift", failures, checks)

    m281 = json.loads((hw / UPSTREAM["m281"][0]).read_text())
    protocol = m281["schedule_and_protocol_reconstruction"]
    require(protocol["first_result_latency_cycles"] == COEFF_FIRST_LATENCY, "M281 latency drift", failures, checks)
    require(protocol["intrinsic_unstalled_accept_interval_cycles"] == COEFF_II, "M281 II drift", failures, checks)

    m161 = json.loads((hw / UPSTREAM["m161_fair_overlay"][0]).read_text())
    fair = m161["fair_movement_correction"]
    require("write raw once" in fair["qualification"] or fair["fair_existing_online_moment_plus_fused_read_bn1_bits"] == 10_395_648_000, "M161 fair baseline missing", failures, checks)
    require(m159["m161_fair_baseline_recompute"]["strong_dense_streaming_baseline_contract"].startswith("Update moments inline while writing raw x once"), "M159 strong baseline contract drift", failures, checks)

    expected_details = []
    expected_summaries = []
    for width in WIDTHS:
        for bandwidth in BANDWIDTHS:
            for overlap in OVERLAP_MODES:
                detail_rows = [expected_phase(phase, width, bandwidth, overlap) for phase in phases]
                expected_details.extend(detail_rows)
                expected_summaries.append(expected_summary(detail_rows, width, bandwidth, overlap))
    require(len(expected_details) == 432, "independent phase-DSE count is not 432", failures, checks)
    require(len(expected_summaries) == 18, "independent summary-DSE count is not 18", failures, checks)
    require({int(row["activation_width_bits"]) for row in expected_summaries} == set(WIDTHS), "width axis incomplete", failures, checks)
    require({int(row["bandwidth_bytes_per_cycle"]) for row in expected_summaries} == set(BANDWIDTHS), "bandwidth axis incomplete", failures, checks)
    require({str(row["coefficient_overlap_mode"]) for row in expected_summaries} == set(OVERLAP_MODES), "overlap axis incomplete", failures, checks)

    # Phase 2: only now open the M480 producer artifacts and compare all rows.
    analyzer_path = hw / "system_simulator/scripts/analyze_m480_dynamic_bn_materialization_elision.py"
    require(sha256(analyzer_path) == EXPECTED_ANALYZER_SHA, "M480 analyzer SHA drift", failures, checks)
    require(sha256(hw / "docs/359_DATE终局冻结_20260813.md") == EXPECTED_DOCS359_SHA, "docs359 changed", failures, checks)

    sums_path = producer / "SHA256SUMS"
    seal_path = producer / "SHA256SUMS.seal.sha256"
    for line in sums_path.read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        require(sha256(producer / relative) == digest, f"producer manifest mismatch: {relative}", failures, checks)
    seal_digest, seal_relative = seal_path.read_text().strip().split(maxsplit=1)
    require(seal_relative.lstrip("* ") == "SHA256SUMS", "unexpected producer seal target", failures, checks)
    require(sha256(sums_path) == seal_digest, "producer outer seal mismatch", failures, checks)

    with (producer / "per_phase_schedule.csv").open(newline="", encoding="utf-8") as stream:
        actual_details = list(csv.DictReader(stream))
    with (producer / "dse_summary.csv").open(newline="", encoding="utf-8") as stream:
        actual_summaries = list(csv.DictReader(stream))
    require(len(actual_details) == 432, "producer phase row count is not 432", failures, checks)
    require(len(actual_summaries) == 18, "producer summary row count is not 18", failures, checks)

    detail_key_fields = ("module", "stage", "phase", "activation_width_bits", "bandwidth_bytes_per_cycle", "coefficient_overlap_mode")
    def detail_key(row: dict[str, object]) -> tuple[str, ...]:
        return tuple(str(row[field]) for field in detail_key_fields)
    actual_detail_map = {detail_key(row): row for row in actual_details}
    expected_detail_map = {detail_key(row): row for row in expected_details}
    require(len(actual_detail_map) == 432, "producer phase key duplicates", failures, checks)
    require(set(actual_detail_map) == set(expected_detail_map), "producer/independent phase grains differ", failures, checks)
    for key, expected in expected_detail_map.items():
        actual = actual_detail_map.get(key, {})
        for field, expected_value in expected.items():
            require(field in actual and csv_value_matches(actual[field], expected_value), f"phase mismatch {key} field={field}", failures, checks)

    summary_key_fields = ("activation_width_bits", "bandwidth_bytes_per_cycle", "coefficient_overlap_mode")
    def summary_key(row: dict[str, object]) -> tuple[str, ...]:
        return tuple(str(row[field]) for field in summary_key_fields)
    actual_summary_map = {summary_key(row): row for row in actual_summaries}
    expected_summary_map = {summary_key(row): row for row in expected_summaries}
    require(len(actual_summary_map) == 18, "producer summary key duplicates", failures, checks)
    require(set(actual_summary_map) == set(expected_summary_map), "producer/independent summary grains differ", failures, checks)
    for key, expected in expected_summary_map.items():
        actual = actual_summary_map.get(key, {})
        for field, expected_value in expected.items():
            require(field in actual and csv_value_matches(actual[field], expected_value), f"summary mismatch {key} field={field}", failures, checks)

    # Schedule invariants independent of CSV equality.
    require(all(int(row["moment_barrier_cycles"]) == 1 for row in expected_details), "not every phase retains its barrier", failures, checks)
    require(all(int(row["raw_replay_read_cycles"]) > 0 for row in expected_details), "raw replay missing", failures, checks)
    require(all(int(row["raw_retention_bits"]) > 0 for row in expected_details), "raw retention missing", failures, checks)
    require(all(row["memory_ports"] == "1R1W" for row in expected_details), "port contract drift", failures, checks)
    require(all(row["same_address_read_before_write_required_by_materialized"] is True for row in expected_details), "materialized read-before-write contract missing", failures, checks)
    require(all(row["same_address_read_before_write_required_by_fused"] is False for row in expected_details), "fused incorrectly needs read-before-write", failures, checks)
    overlap_rows = [row for row in expected_details if row["coefficient_overlap_mode"] == "coefficient_with_replay"]
    require(all(int(row["replay_cycles_per_channel"]) >= COEFF_II for row in overlap_rows), "coefficient service not hidden by replay", failures, checks)
    require(all(int(row["coefficient_exposed_cycles"]) == COEFF_FIRST_LATENCY for row in overlap_rows), "overlap exposure is not first latency only", failures, checks)

    # Load the producer receipt last and constrain its claims.
    receipt_path = producer / "m480_dynamic_bn_exact_materialization_elision_dse_r1.json"
    receipt = json.loads(receipt_path.read_text())
    require(receipt["dse_axes"]["summary_points"] == 18, "receipt summary count mismatch", failures, checks)
    require(receipt["dse_axes"]["per_phase_rows"] == 432, "receipt phase count mismatch", failures, checks)
    require(receipt["frozen_semantics"]["global_moment_barriers"] == 24, "receipt barrier count mismatch", failures, checks)
    require(receipt["matched_schedule_contract"]["raw_retention_required_both"] is True, "receipt drops raw retention", failures, checks)
    require(receipt["matched_schedule_contract"]["raw_replay_required_both"] is True, "receipt drops raw replay", failures, checks)
    require(receipt["matched_schedule_contract"]["barrier_removed"] is False, "receipt removes barrier", failures, checks)
    require(receipt["thresholds"]["standalone_novelty_gate"]["result"] == "NO_GO", "receipt novelty gate not NO_GO", failures, checks)
    for field in ("fixed_point_accuracy", "moment_finalizer_rtl", "runtime_affine_rtl", "sram_macro", "vcs", "dc_sta", "power", "energy", "module_speedup_admitted", "system_speedup", "paper_ppa_ready", "headline"):
        require(receipt["claim_boundary"][field] is False, f"receipt overclaims {field}", failures, checks)

    reference = expected_summary_map[("24", "64", "coefficient_with_replay")]
    peak_bytes = {
        str(width): max(int(row["peak_raw_retention_bytes_both_schedules"]) for row in expected_summaries if int(row["activation_width_bits"]) == width)
        for width in WIDTHS
    }
    local_speedups = [float(row["bn_local_cycle_speedup"]) for row in expected_summaries]
    accounted_speedups = [float(row["serial_m159_accounted_speedup"]) for row in expected_summaries]
    fair_strong_baseline_semantic_ratio = 1.0
    audit = {
        "schema": "m480_independent_hammer_receipt_v1",
        "status": "PASS" if not failures else "FAIL_CLOSED",
        "receipt_blind_order": "Independent expectations were reconstructed before M480 CSV/JSON artifacts were opened.",
        "checks_performed": checks["performed"],
        "failure_count": len(failures),
        "failures": failures,
        "identity": {
            "producer_result_sha256": sha256(receipt_path),
            "producer_manifest_sha256": sha256(sums_path),
            "producer_seal_file_sha256": sha256(seal_path),
            "analyzer_sha256": sha256(analyzer_path),
            "docs359_sha256": sha256(hw / "docs/359_DATE终局冻结_20260813.md"),
            "upstream": upstream_identity,
        },
        "recomputed": {
            "summary_points": len(expected_summaries),
            "phase_rows": len(expected_details),
            "physical_bn_phases_per_configuration": len(phases),
            "barriers_per_configuration": 24,
            "bn1_elements": bn1_elements,
            "bn2_elements": bn2_elements,
            "total_bn_elements": bn1_elements + bn2_elements,
            "coefficient_pairs": coefficient_pairs,
            "m281_first_latency_cycles": COEFF_FIRST_LATENCY,
            "m281_intrinsic_ii_cycles": COEFF_II,
            "overlap_exposed_cycles_per_phase": COEFF_FIRST_LATENCY,
            "reference_q24_bw64_overlap": {
                "materialized_cycles": reference["materialized_cycles"],
                "fused_cycles": reference["fused_cycles"],
                "bn_local_cycle_speedup_vs_explicit_materialized": reference["bn_local_cycle_speedup"],
                "cycles_elided": reference["cycles_elided"],
                "materialized_traffic_bytes": reference["materialized_traffic_bytes"],
                "fused_traffic_bytes": reference["fused_traffic_bytes"],
                "serial_m159_accounted_speedup_vs_explicit_materialized": reference["serial_m159_accounted_speedup"],
                "minimum_replay_cycles_per_channel": reference["minimum_replay_cycles_per_channel"],
                "coefficient_exposed_cycles_24_phases": reference["coefficient_exposed_cycles"],
            },
            "local_speedup_range_vs_explicit_materialized": [min(local_speedups), max(local_speedups)],
            "serial_m159_accounted_range_vs_explicit_materialized": [min(accounted_speedups), max(accounted_speedups)],
            "traffic_ratio_vs_explicit_materialized": 2.0,
            "fair_strong_baseline_semantic_speedup": fair_strong_baseline_semantic_ratio,
            "peak_raw_retention_bytes": peak_bytes,
            "peak_raw_retention_mib": {width: value / (1024 * 1024) for width, value in peak_bytes.items()},
        },
        "score": {
            "out_of_100": 72 if not failures else 0,
            "data_identity_and_grain": 20,
            "arithmetic_and_schedule_recompute": 25,
            "protocol_semantics": 14,
            "baseline_fairness": 8,
            "implementation_evidence": 5,
            "novelty": 0,
        },
        "findings": {
            "p0": [
                {
                    "id": "P0-1",
                    "severity": "critical",
                    "finding": "M480 fused is the already-required M159/M161 fair strong dense baseline: inline moments, one raw write, barrier, one raw replay, and direct normalized consumption.",
                    "impact": "The 2x traffic and local-cycle ratios versus explicit normalized materialization are baseline-hygiene diagnostics, not an accelerator speedup or novelty claim.",
                },
                {
                    "id": "P0-2",
                    "severity": "high",
                    "finding": "Peak raw retention is 140.625/210.9375/281.25 MiB for 16/24/32-bit payloads and is unchanged by elision.",
                    "impact": "Without an address-bearing SRAM/DRAM schedule, macro capacity, and memory energy/latency, no implementable BN performance or PPA claim is supported.",
                },
            ],
            "p1": [
                {
                    "id": "P1-1",
                    "finding": "Exactness is schedule-only at an assumed stored width; no fixed-point runtime-affine or downstream ATLIF numeric miter exists.",
                },
                {
                    "id": "P1-2",
                    "finding": "The materialized comparator depends on same-address 1R1W read-before-write behavior; no selected SRAM macro proves that semantic, and 24-bit packed accesses are not implemented.",
                },
                {
                    "id": "P1-3",
                    "finding": "Coefficient/replay overlap arithmetic is valid, but the model assumes an unpriced bus-wide affine and consumer datapath: up to 64 elements/cycle at 16-bit and 128 B/cycle.",
                },
            ],
        },
        "decision": {
            "baseline_hygiene": "GO",
            "standalone_novelty": "NO_GO",
            "rtl_nominated": False,
            "performance_admitted": False,
            "paper_contribution": False,
            "verdict": "KEEP_FUSED_REPLAY_AS_MANDATORY_FAIR_BASELINE__DO_NOT_BUILD_OR_CLAIM_M480_AS_A_STANDALONE_MECHANISM",
        },
        "claim_boundary": {
            "dynamic_current_batch_bn_geometry": True,
            "cpu_cycle_schedule_recompute": True,
            "exact_fixed_point": False,
            "rtl": False,
            "synopsys": False,
            "sram_macro": False,
            "power": False,
            "energy": False,
            "module_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    print(json.dumps(audit, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
