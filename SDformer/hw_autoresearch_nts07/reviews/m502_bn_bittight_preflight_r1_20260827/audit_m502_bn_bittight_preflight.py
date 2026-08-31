#!/usr/bin/env python3
"""Receipt-blind M502 preflight: bit-tight current-batch BN raw replay.

This script does not import an M502 producer.  It rebuilds the opportunity
from the independently reviewed M161 analytic widths and the independently
reviewed M480 strong fused-replay baseline.  Results are opportunities only:
M161 did not admit the INT8 bridge/Q24 numerical path, while M480 did not
admit a physical raw store, bus-wide consumer, cycle speedup, or energy.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]

SOURCES = {
    "m161_independent_recompute": (
        "results/m161_independent_hammer_review_r1_20260824/"
        "independent_recompute.json",
        "99507f266ac9226ee6866bb7a827914d21929ada2d60303f81cc09b7d9fa78d9"),
    "m161_fairness_overlay": (
        "contracts/m161_r1_width_and_fair_baseline_correction_overlay_"
        "r1_20260824.json",
        "f59a44685bfc1bdf283a6e6e4a94f1189d3ad384fe211cc381611edec334a6cd"),
    "m161_admission_overlay": (
        "contracts/m161_r2_independent_review_admission_overlay_"
        "r1_20260824.json",
        "458e5eb7f2332ba040110afad423bbbd32701dfdd71564bca8ec381d20580502"),
    "m480_phases": (
        "results/m480_dynamic_bn_exact_materialization_elision_dse_"
        "r1_20260826/per_phase_schedule.csv",
        "a56c4ef18bddb90515d671b37f5750a164d4dd42650fb695036a527c2f75dd92"),
    "m480_summary": (
        "results/m480_dynamic_bn_exact_materialization_elision_dse_"
        "r1_20260826/dse_summary.csv",
        "43ee37bc662e01fba4355b65992f60597932f9c639da1f2d770d0e433a898a88"),
    "m480_independent_receipt": (
        "reviews/m480_independent_hammer_r1_20260826/"
        "m480_independent_hammer_receipt_r1.json",
        "0a8a7db2d017f66735516ff051dbef94c5e0ba32a766cee682b8a029121a10b9"),
    "m160_parameter_census": (
        "results/m160_h67_ffn_bn_atlif_fusion_r1_20260824/"
        "per_ffn_bn_atlif_fusion.csv",
        "309a5d802c7e49d432285f09ff43b9d1ec797db815b949cd34798c0a94f4f464"),
    "m292_compute_envelope": (
        "contracts/m292_m287_scope_corrected_amdahl_overlay_contract_"
        "r1_20260825.json",
        "df5bfe7318b776f24646b303927e73d2e729a6e00b4abbb0c205806c375ba471"),
    "cicc26_local_pdf": (
        "docs/Zhang 等 - 2026 - A 28-nm Optical Flow Estimation Accelerator "
        "with Redundancy Speculation, Bit-Width-Aware Compression.pdf",
        "b7e40d1c2d28f4f6b12e0c8e70fbf68819d12c451c890c81a27c44eb17a4bd09"),
    "docs359": (
        "docs/359_DATE终局冻结_20260813.md",
        "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"),
}

M159_FIXED = 205_384_111
COEFF_FIRST = 8
COEFF_II = 9
BARRIER = 1


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    with path.open(encoding="utf-8") as handle:
        return json.load(
            handle,
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                RuntimeError("non-finite JSON: " + value)))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, "CSV missing header")
        require(len(reader.fieldnames) == len(set(reader.fieldnames)),
                "duplicate CSV header")
        return list(reader)


def ceil_div(a, b):
    return (int(a) + int(b) - 1) // int(b)


def pin_sources():
    identity = {}
    for name, (relative, expected) in SOURCES.items():
        path = HW / relative
        require(path.is_file(), "missing source: " + relative)
        actual = sha256(path)
        require(actual == expected,
                "source drift {} expected={} actual={}".format(
                    relative, expected, actual))
        identity[name] = {"path": relative, "sha256": actual}
    return identity


def validate_inputs():
    recompute = strict_json(HW / SOURCES["m161_independent_recompute"][0])
    fairness = strict_json(HW / SOURCES["m161_fairness_overlay"][0])
    admission = strict_json(HW / SOURCES["m161_admission_overlay"][0])
    m480_receipt = strict_json(HW / SOURCES["m480_independent_receipt"][0])
    envelope = strict_json(HW / SOURCES["m292_compute_envelope"][0])
    rows = recompute["geometry_width_and_movement_rows"]

    require(recompute["status"] ==
            "FAIL_HARDWARE_DSE_CLAIMS_RETAIN_REAL_ALGEBRA_AND_RAW_COUNTS",
            "M161 independent status drift")
    require(fairness["admission"]["q8_training_accuracy"] is False and
            fairness["admission"]["raw_width_rank3_implementation"] is False,
            "M161 fairness admission drift")
    require(admission["admission"]["hardware_dse"] is False and
            admission["admission"]["cycle_speedup"] is False,
            "M161 fail-closed admission drift")
    require(m480_receipt["decision"]["baseline_hygiene"] == "GO" and
            m480_receipt["decision"]["performance_admitted"] is False and
            m480_receipt["decision"]["rtl_nominated"] is False,
            "M480 admission drift")
    require(len(rows) == 12, "expected twelve FFN modules")

    fc1_widths = Counter(int(row["raw_accumulator_signed_bits"])
                         for row in rows)
    fc2_widths = Counter(int(row["fc2_raw_accumulator_signed_bits"])
                         for row in rows)
    require(fc1_widths == Counter({14: 2, 15: 4, 16: 6}),
            "FC1 analytic width distribution drift")
    require(fc2_widths == Counter({15: 2, 16: 2, 17: 6, 18: 2}),
            "FC2 analytic width distribution drift")
    require(sum(int(row["bn1_elements"]) for row in rows) == 350_208_000,
            "BN1 element total drift")
    require(sum(int(row["bn2_elements"]) for row in rows) == 87_552_000,
            "BN2 element total drift")

    phase_rows = read_csv(HW / SOURCES["m480_phases"][0])
    q24 = [row for row in phase_rows
           if int(row["activation_width_bits"]) == 24
           and int(row["bandwidth_bytes_per_cycle"]) == 64
           and row["coefficient_overlap_mode"] == "coefficient_with_replay"]
    require(len(q24) == 24, "M480 selected phase population drift")
    require(sum(int(row["fused_schedule_cycles"]) for row in q24) == 41_048_856,
            "M480 Q24/BW64/overlap fused cycles drift")
    require(sum(int(row["fused_traffic_bytes"]) for row in q24) ==
            2_626_560_000, "M480 Q24 fused useful traffic drift")
    require(all(row["memory_ports"] == "1R1W" for row in q24),
            "M480 memory-port drift")
    require(envelope["scope_partition"]["frozen_compute_envelope_cycles"] ==
            620_302_905, "M292 envelope drift")
    return rows, q24, envelope


def mapped_width(raw_width, scheme):
    if scheme == "analytic_exact":
        return raw_width
    if scheme == "even_14_16_18":
        if raw_width <= 14:
            return 14
        if raw_width <= 16:
            return 16
        return 18
    if scheme == "nibble_16_20":
        return 16 if raw_width <= 16 else 20
    if scheme == "byte_16_24":
        return 16 if raw_width <= 16 else 24
    if scheme == "q24":
        return 24
    raise RuntimeError("unknown width scheme: " + scheme)


def overlap_exposure(channels, replay_cycles_per_channel):
    finish = 0
    for channel in range(channels):
        ready = COEFF_FIRST + channel * COEFF_II
        finish = max(finish, ready) + replay_cycles_per_channel
    return finish - channels * replay_cycles_per_channel


def schedule(rows, scheme, bus_bits, lane_cap):
    phases = []
    for row in rows:
        population = int(row["dynamic_bn_reduction_population_per_channel"])
        for phase, width_key in (
                ("bn1", "raw_accumulator_signed_bits"),
                ("bn2", "fc2_raw_accumulator_signed_bits")):
            channels = (int(row["expanded_channels"]) if phase == "bn1"
                        else int(row["expanded_channels"]) // 4)
            elements = int(row[phase + "_elements"])
            width = mapped_width(int(row[width_key]), scheme)
            capture_bus = ceil_div(elements * width, bus_bits)
            replay_bus_per_channel = ceil_div(population * width, bus_bits)
            if lane_cap:
                capture = max(capture_bus, ceil_div(elements, lane_cap))
                replay_per_channel = max(
                    replay_bus_per_channel, ceil_div(population, lane_cap))
            else:
                capture = capture_bus
                replay_per_channel = replay_bus_per_channel
            replay = channels * replay_per_channel
            exposed = overlap_exposure(channels, replay_per_channel)
            fused = capture + BARRIER + exposed + replay
            phases.append({
                "module": row["module"],
                "stage": int(row["stage"]),
                "phase": phase,
                "analytic_width_bits": int(row[width_key]),
                "stored_width_bits": width,
                "elements": elements,
                "population_per_channel": population,
                "channels": channels,
                "capture_cycles": capture,
                "replay_cycles": replay,
                "coefficient_exposed_cycles": exposed,
                "fused_cycles": fused,
                "useful_write_plus_read_bits": 2 * elements * width,
                "occupied_bus_cycle_capacity_bits":
                    (capture + replay) * bus_bits,
                "maximum_values_completed_in_one_bus_cycle":
                    ceil_div(bus_bits, width),
            })
    useful_bits = sum(row["useful_write_plus_read_bits"] for row in phases)
    cycles = sum(row["fused_cycles"] for row in phases)
    return {
        "scheme": scheme,
        "bus_bits": bus_bits,
        "lane_cap": lane_cap,
        "phases": 24,
        "fused_schedule_cycles": cycles,
        "useful_write_plus_read_bits": useful_bits,
        "useful_write_plus_read_bytes": useful_bits // 8,
        "occupied_bus_cycle_capacity_bits": sum(
            row["occupied_bus_cycle_capacity_bits"] for row in phases),
        "peak_raw_retention_bits": max(
            row["elements"] * row["stored_width_bits"] for row in phases),
        "maximum_values_completed_in_one_bus_cycle": max(
            row["maximum_values_completed_in_one_bus_cycle"] for row in phases),
        "phase_rows": phases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=HERE / "m502_bn_bittight_preflight_r1.json")
    args = parser.parse_args()

    identity = pin_sources()
    rows, m480_q24, envelope = validate_inputs()
    schemes = ("analytic_exact", "even_14_16_18", "nibble_16_20",
               "byte_16_24")
    bus_widths = (128, 512)
    lane_caps = (0, 16, 21, 24, 28, 32, 36, 64)
    q24 = {bus: schedule(rows, "q24", bus, 0) for bus in bus_widths}
    require(q24[512]["fused_schedule_cycles"] == 41_048_856,
            "independent M480 recurrence mismatch")
    require(q24[512]["useful_write_plus_read_bytes"] == 2_626_560_000,
            "independent M480 useful-traffic mismatch")

    dse = []
    for bus in bus_widths:
        for scheme in schemes:
            for lanes in lane_caps:
                candidate = schedule(rows, scheme, bus, lanes)
                baseline = schedule(rows, "q24", bus, lanes)
                candidate["q24_same_bus_same_lane_cycles"] = baseline[
                    "fused_schedule_cycles"]
                candidate["local_schedule_opportunity_vs_q24"] = (
                    baseline["fused_schedule_cycles"] /
                    candidate["fused_schedule_cycles"])
                candidate["useful_traffic_opportunity_vs_q24"] = (
                    baseline["useful_write_plus_read_bits"] /
                    candidate["useful_write_plus_read_bits"])
                candidate["m159_serial_opportunity_vs_q24"] = (
                    (M159_FIXED + baseline["fused_schedule_cycles"]) /
                    (M159_FIXED + candidate["fused_schedule_cycles"]))
                # Per-phase rows are deterministic implementation detail.  Keep
                # the sealed artifact compact; this script independently
                # regenerates them on every run.
                candidate.pop("phase_rows")
                dse.append(candidate)

    selected = next(row for row in dse
                    if row["bus_bits"] == 512
                    and row["scheme"] == "analytic_exact"
                    and row["lane_cap"] == 0)
    lane32 = next(row for row in dse
                  if row["bus_bits"] == 512
                  and row["scheme"] == "analytic_exact"
                  and row["lane_cap"] == 32)
    byte_mode = next(row for row in dse
                     if row["bus_bits"] == 512
                     and row["scheme"] == "byte_16_24"
                     and row["lane_cap"] == 32)
    full_envelope = int(
        envelope["scope_partition"]["frozen_compute_envelope_cycles"])
    full_envelope_overlay = full_envelope / (
        full_envelope - q24[512]["fused_schedule_cycles"]
        + selected["fused_schedule_cycles"])

    widths = {
        "fc1_modules": dict(sorted(Counter(
            int(row["raw_accumulator_signed_bits"])
            for row in rows).items())),
        "fc2_modules": dict(sorted(Counter(
            int(row["fc2_raw_accumulator_signed_bits"])
            for row in rows).items())),
    }
    payload = {
        "schema": "m502_bn_bittight_preflight_v1",
        "status": "PASS_PRELIMINARY_ARITHMETIC__GO_OFFLINE_AUDIT_ONLY",
        "verdict": "GO_OFFLINE_AUDIT_ONLY",
        "scope": (
            "H67/Motion twelve-FFN current-batch/no-running BN1+BN2 raw "
            "retention and replay; same 1R1W/barrier/coefficient-overlap "
            "recurrence as the M480 strong fused baseline"),
        "identity": identity,
        "validated_grain": {
            "ffn_modules": len(rows),
            "bn_phases": len(m480_q24),
            "bn1_elements": sum(int(row["bn1_elements"]) for row in rows),
            "bn2_elements": sum(int(row["bn2_elements"]) for row in rows),
            "analytic_width_distribution": widths,
            "m480_q24_bw64_overlap_fused_cycles": 41_048_856,
            "m480_q24_bw64_useful_traffic_bytes": 2_626_560_000,
        },
        "selected_unpriced_opportunity": {
            "bus_bits": 512,
            "scheme": "analytic_exact_14_to_18_bits",
            "downstream_lane_cap": "none__unpriced",
            "candidate_fused_cycles": selected["fused_schedule_cycles"],
            "q24_fused_cycles": q24[512]["fused_schedule_cycles"],
            "local_schedule_opportunity":
                selected["local_schedule_opportunity_vs_q24"],
            "useful_traffic_opportunity":
                selected["useful_traffic_opportunity_vs_q24"],
            "useful_bytes_saved_per_frame":
                q24[512]["useful_write_plus_read_bytes"]
                - selected["useful_write_plus_read_bytes"],
            "m159_serial_opportunity":
                selected["m159_serial_opportunity_vs_q24"],
            "full_620302905_cycle_envelope_substitution_sensitivity":
                full_envelope_overlay,
            "full_envelope_qualification": (
                "Illustrative zero-overlap substitution only; M480 BN cycles "
                "and the M292 compute envelope are not an integrated system "
                "schedule, so this is not system speedup."),
            "peak_retention_mib_q24":
                q24[512]["peak_raw_retention_bits"] / 8 / 1024 / 1024,
            "peak_retention_mib_candidate":
                selected["peak_raw_retention_bits"] / 8 / 1024 / 1024,
            "maximum_unpacked_values_in_one_512b_cycle":
                selected["maximum_values_completed_in_one_bus_cycle"],
        },
        "implementation_sensitivities": {
            "analytic_exact_with_32_output_lanes": {
                "cycles": lane32["fused_schedule_cycles"],
                "local_opportunity":
                    lane32["local_schedule_opportunity_vs_q24"],
                "m159_serial_opportunity":
                    lane32["m159_serial_opportunity_vs_q24"],
            },
            "byte_16_24_with_32_output_lanes": {
                "cycles": byte_mode["fused_schedule_cycles"],
                "local_opportunity":
                    byte_mode["local_schedule_opportunity_vs_q24"],
                "useful_traffic_opportunity":
                    byte_mode["useful_traffic_opportunity_vs_q24"],
                "m159_serial_opportunity":
                    byte_mode["m159_serial_opportunity_vs_q24"],
                "recommendation": (
                    "First RTL candidate only after trace admission: static "
                    "per-phase 16/24-bit modes on a compile-time BUS_W, not a "
                    "runtime-general 14..18-bit dual-bus barrel network."),
            },
        },
        "data_quality_findings": {
            "critical": [
                "M161 explicitly does not admit an FFN INT8 activation bridge, Q24 sufficiency, fixed binary point, or overflow proof.",
                "The 14-18-bit values are checkpoint-weight sumabs analytic bounds under a binary-input convention, not measured frozen-H67 raw accumulator traces.",
                "M480 does not price a physical raw store or the bus-wide affine/consumer lanes; a packer alone can save traffic but cannot realize the ideal cycle ratio when the consumer is lane-limited.",
            ],
            "high": [
                "CICC'26 BWAC already uses group minimum width plus online decompression, while Stripes/Loom/Bit Fusion establish broad precision-adaptive acceleration prior art; generic bit-width packing is not standalone novelty.",
                "Exactness can only mean sign-extension equivalence to the hypothetical M480 Q24 container after an integer bridge is proved; it cannot currently mean end-to-end accuracy equivalence.",
                "The 620302905-cycle substitution is not an integrated schedule and must not be labeled system speedup.",
            ],
        },
        "hard_gates_before_rtl": [
            "Freeze the H67 ep35 integer bridge: activation codebook/scales, INT8 weight codes, bias policy, exact dot order, saturation/rounding, and BN raw boundary for all 24 phases.",
            "Prove every raw value fits its declared signed width by analytic domain proof and trace it on S10; capture min/max, signed-width histogram, overflow count=0, and SHA-bound stratified raw vectors including all extrema.",
            "Generate address-timed 1R1W raw write/replay transactions with barrier, coefficient overlap, tails, padding, and downstream ready/backpressure; no useful-bit divided by bus-width shortcut.",
            "At 512 bits/cycle, retain at least 1.25x local schedule opportunity versus Q24 after the same downstream lane cap and all pack/unpack stalls; otherwise traffic/energy-only.",
            "A 16/24-bit two-mode point must retain at least 1.35x useful-traffic reduction and 1.20x local schedule opportunity before any arbitrary-width RTL is considered.",
            "Only after trace gates pass: VCS exact round-trip under arbitrary stalls and cross-beat tails, then 3 ns DC/STA; pack/unpack plus control area <=15000 um^2 and matched PTPX must show at least 20% net raw-path energy reduction after SRAM/memory savings.",
        ],
        "required_exact_trace": {
            "identity": [
                "checkpoint/config/data-list SHA256",
                "bn_policy=no_running/current-batch and eval_batch_size=1",
                "module/stage/block/phase/sample/sequence and exact tensor shape",
            ],
            "integer_bridge": [
                "input event/amplitude code and scale",
                "per-output-channel INT8 weight code and scale",
                "bias absence/presence",
                "dot accumulation order and accumulator binary point",
                "round/saturation points before moments and replay",
            ],
            "streamed_statistics": [
                "per-phase signed min/max and exact required-bit histogram",
                "analytic-bound margin and overflow_count=0",
                "raw element count and rolling/zlib/raw SHA",
                "all extrema plus stratified boundary-crossing raw vectors",
            ],
            "memory_schedule": [
                "raw address, cycle, write/read, width mode, bus beat and valid-bit mask",
                "barrier and coefficient-ready timestamps",
                "consumer ready/backpressure and actual values accepted per cycle",
                "tail/padding/metadata bytes and 1R1W conflicts",
            ],
        },
        "admission": {
            "source_identity_and_recompute": True,
            "analytic_opportunity": True,
            "integer_bridge": False,
            "q24_numeric_sufficiency": False,
            "trace_width_fit": False,
            "exact_pack_roundtrip": False,
            "address_timed_schedule": False,
            "physical_memory": False,
            "rtl_nominated": False,
            "vcs": False,
            "synopsys_ppa": False,
            "cycle_speedup": False,
            "energy": False,
            "system_speedup": False,
            "standalone_novelty": False,
            "headline": False,
        },
        "dse": dse,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("PASS_M502_PRELIMINARY_ARITHMETIC")
    print("verdict=GO_OFFLINE_AUDIT_ONLY rtl_nominated=false")
    print("q24_cycles=41048856 analytic_candidate_cycles={}".format(
        selected["fused_schedule_cycles"]))
    print("local_opportunity={:.9f} m159_serial={:.9f}".format(
        selected["local_schedule_opportunity_vs_q24"],
        selected["m159_serial_opportunity_vs_q24"]))
    print("lane32_local_opportunity={:.9f} byte16_24_local={:.9f}".format(
        lane32["local_schedule_opportunity_vs_q24"],
        byte_mode["local_schedule_opportunity_vs_q24"]))


if __name__ == "__main__":
    main()
