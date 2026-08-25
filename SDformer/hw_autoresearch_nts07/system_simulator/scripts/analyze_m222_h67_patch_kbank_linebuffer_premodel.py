#!/usr/bin/env python3
"""Exact M51-s10 patch Conv3x3 bank-coissue and line-buffer screen.

This model is deliberately output-centric.  For every frozen binary patch
input bit, it enumerates the valid 3x3 receptive-field destinations and maps
the (input-channel, kernel-tap) weight row to a source-owned bank.  It then
compares a row-striped K1/D96 reference with several K/D coissue points while
preserving every source x destination product.

The result is a standalone service premodel.  It does not model dynamic-BN
barriers, real SRAM macros, weight-load energy, RTL timing, or system cycles.
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


EXPECTED_MANIFEST_SHA256 = (
    "2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e"
)
EXPECTED_DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
PATCH_PREFIX = "sttmultires_unet.encoders.swin3d.patch_embed."
SELECTED_MODULE_INDICES = tuple(range(6))
OUTPUT_CHANNELS = 96
KERNEL = 3
PADDING = 1
MAX_BANKS = 8
GLOBAL_ENVELOPE_CYCLES = 620302905
PATCH_LEDGER_CYCLES = 199420620
PATCH_BINARY_ELIGIBLE_LEDGER_CYCLES = 172321077
PATCH_NONBINARY_LEDGER_CYCLES = 27099543

MODEL_POINTS = (
    ("K1_D96_ROW_STRIPED", 1, 96),
    ("K2_D48_EQUAL96", 2, 48),
    ("K4_D24_EQUAL96", 4, 24),
    ("K8_D12_EQUAL96", 8, 12),
    ("K4_D32_W128", 4, 32),
    ("K8_D16_W128_M218_LIKE", 8, 16),
    ("K8_D32_W256", 8, 32),
    ("K8_D48_W384", 8, 48),
    ("K8_D96_W768", 8, 96),
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def decode_record(record, payload_root):
    path = payload_root / record["relative_path"]
    require(path.is_file(), "missing payload: {}".format(path))
    require(sha256(path) == record["file_sha256"],
            "payload SHA drift: {}".format(path))
    shape = tuple(int(value) for value in record["input_shape"])
    element_count = int(np.prod(shape))
    packed = np.fromfile(str(path), dtype=np.uint8)
    require(packed.size == int(record["packed_bytes"]),
            "packed byte count mismatch: {}".format(path))
    bits = np.unpackbits(packed, bitorder="little")[:element_count]
    require(int(bits.sum()) == int(record["active_elements"]),
            "active element mismatch: {}".format(path))
    return bits.reshape(shape).astype(np.uint8, copy=False)


def build_occ8(bits, output_shape):
    if bits.ndim == 4:
        bits = bits[:, np.newaxis, :, :, :]
    require(bits.ndim == 5, "expected T(B)CHW input")
    t_count, batch, channels, height, width = bits.shape
    if len(output_shape) == 4:
        output_shape = [output_shape[0], 1] + list(output_shape[1:])
    require(len(output_shape) == 5, "expected T(B)CHW output")
    out_t, out_b, out_channels, out_height, out_width = (
        int(value) for value in output_shape)
    require((t_count, batch, out_channels) ==
            (out_t, out_b, OUTPUT_CHANNELS), "output shape mismatch")
    require(height % out_height == 0 and width % out_width == 0,
            "non-integral patch stride")
    stride_h = height // out_height
    stride_w = width // out_width
    require(stride_h == stride_w and stride_h in (1, 2),
            "unsupported stride")
    stride = stride_h

    padded = np.pad(
        bits,
        ((0, 0), (0, 0), (0, 0), (PADDING, PADDING),
         (PADDING, PADDING)),
        mode="constant",
    )
    occ8 = np.zeros(
        (t_count, batch, out_height, out_width, MAX_BANKS),
        dtype=np.uint16,
    )
    for kernel_y in range(KERNEL):
        for kernel_x in range(KERNEL):
            tap = kernel_y * KERNEL + kernel_x
            sampled = padded[
                :, :, :,
                kernel_y:kernel_y + stride * out_height:stride,
                kernel_x:kernel_x + stride * out_width:stride,
            ]
            require(sampled.shape[3:] == (out_height, out_width),
                    "sampled geometry mismatch")
            # Flattened source key is channel*9+tap.  Since 9 mod 8 is 1,
            # source_key mod 8 is (channel+tap) mod 8.
            for bank in range(MAX_BANKS):
                first_channel = (bank - tap) % MAX_BANKS
                bank_sum = sampled[:, :, first_channel::MAX_BANKS, :, :].sum(
                    axis=2, dtype=np.uint16
                )
                occ8[..., bank] += bank_sum

    source_contributions = int(occ8.sum(dtype=np.uint64))
    output_tokens = int(t_count * batch * out_height * out_width)
    input_vectors = int(t_count * batch * height * width)
    return occ8, source_contributions, input_vectors, output_tokens, stride


def fold_banks(occ8, banks):
    require(banks in (1, 2, 4, 8), "unsupported bank count")
    if banks == 8:
        return occ8
    pieces = []
    for bank in range(banks):
        piece = np.zeros(occ8.shape[:-1], dtype=np.uint16)
        for source_bank in range(bank, MAX_BANKS, banks):
            piece += occ8[..., source_bank]
        pieces.append(piece)
    return np.stack(pieces, axis=-1)


def point_metrics(occ8, source_contributions, input_vectors, output_tokens,
                  name, banks, destination_lanes):
    occupancy = fold_banks(occ8, banks)
    group_depth = occupancy.max(axis=-1)
    groups = int(group_depth.sum(dtype=np.uint64))
    slices = ceil_div(OUTPUT_CHANNELS, destination_lanes)
    service_cycles = int(groups * slices)
    serial_cycles = int(input_vectors + service_cycles + output_tokens)
    pipeline_lower_bound = int(max(input_vectors, service_cycles,
                                   output_tokens))
    active_tokens = int(np.count_nonzero(group_depth))
    zero_tokens = int(group_depth.size - active_tokens)
    coissue = (float(source_contributions) / float(groups)
               if groups else 0.0)
    product_lanes = int(banks * destination_lanes)
    weight_bits_per_issue = int(product_lanes * 8)
    banks_128b_required = int(banks * ceil_div(destination_lanes, 16))
    return {
        "name": name,
        "source_banks": banks,
        "destination_lanes_per_source": destination_lanes,
        "product_lanes": product_lanes,
        "destination_slices": slices,
        "source_owned_groups": groups,
        "average_sources_per_nonempty_group": coissue,
        "service_cycles": service_cycles,
        "linebuffer_scan_cycles": input_vectors,
        "output_commit_cycles": output_tokens,
        "conservative_serial_cycles": serial_cycles,
        "optimistic_full_overlap_lower_bound_cycles": pipeline_lower_bound,
        "active_output_tokens": active_tokens,
        "zero_output_tokens": zero_tokens,
        "maximum_group_depth": int(group_depth.max()) if group_depth.size else 0,
        "weight_bits_per_issue": weight_bits_per_issue,
        "banks_128b_required": banks_128b_required,
        "fits_eight_128b_banks": banks_128b_required <= 8,
        "product_updates": int(source_contributions * OUTPUT_CHANNELS),
    }


def load_ledger(path):
    rows = {}
    with path.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["name"]] = row
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--m55", required=True, type=Path)
    parser.add_argument("--docs359", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    require(sha256(args.manifest) == EXPECTED_MANIFEST_SHA256,
            "M51 manifest drift")
    require(sha256(args.docs359) == EXPECTED_DOCS359_SHA256,
            "docs/359 drift")
    manifest = json.loads(args.manifest.read_text())
    m55 = json.loads(args.m55.read_text())
    ledger = load_ledger(args.ledger)
    records = [
        row for row in manifest["records"]
        if int(row["module_index"]) in SELECTED_MODULE_INDICES
    ]
    require(len(records) == 60, "expected 60 patch binary records")
    require(sorted(set(int(row["module_index"]) for row in records)) ==
            list(SELECTED_MODULE_INDICES), "module population mismatch")
    require(all(row["name"].startswith(PATCH_PREFIX) for row in records),
            "non-patch record selected")

    per_record = []
    aggregate_occ8 = None
    aggregate_source = 0
    aggregate_input_vectors = 0
    aggregate_output_tokens = 0
    stride_histogram = {}
    for ordinal, record in enumerate(sorted(
            records, key=lambda row: (int(row["sample_id"]),
                                      int(row["module_index"])))):
        bits = decode_record(record, args.payload_root)
        occ8, source_count, input_vectors, output_tokens, stride = build_occ8(
            bits, record["output_shape"]
        )
        # Group counts, rather than the full occupancy tensors, are additive.
        metrics = {}
        for name, banks, lanes in MODEL_POINTS:
            metrics[name] = point_metrics(
                occ8, source_count, input_vectors, output_tokens,
                name, banks, lanes
            )
        per_record.append({
            "ordinal": ordinal,
            "sample_id": int(record["sample_id"]),
            "module_index": int(record["module_index"]),
            "name": record["name"],
            "input_shape": record["input_shape"],
            "output_shape": record["output_shape"],
            "stride": stride,
            "active_input_elements": int(record["active_elements"]),
            "valid_receptive_field_source_contributions": source_count,
            "input_vectors": input_vectors,
            "output_tokens": output_tokens,
            "model_points": metrics,
        })
        aggregate_source += source_count
        aggregate_input_vectors += input_vectors
        aggregate_output_tokens += output_tokens
        stride_histogram[str(stride)] = stride_histogram.get(str(stride), 0) + 1

    aggregate_points = {}
    for name, banks, lanes in MODEL_POINTS:
        sums = {
            "source_owned_groups": 0,
            "service_cycles": 0,
            "conservative_serial_cycles": 0,
            "optimistic_full_overlap_lower_bound_cycles": 0,
            "product_updates": 0,
            "active_output_tokens": 0,
            "zero_output_tokens": 0,
        }
        max_depth = 0
        for row in per_record:
            point = row["model_points"][name]
            for key in sums:
                sums[key] += int(point[key])
            max_depth = max(max_depth, int(point["maximum_group_depth"]))
        template = per_record[0]["model_points"][name]
        sums.update({
            "name": name,
            "source_banks": banks,
            "destination_lanes_per_source": lanes,
            "product_lanes": int(banks * lanes),
            "destination_slices": ceil_div(OUTPUT_CHANNELS, lanes),
            "average_sources_per_nonempty_group": (
                float(aggregate_source) / float(sums["source_owned_groups"])
            ),
            "linebuffer_scan_cycles": aggregate_input_vectors,
            "output_commit_cycles": aggregate_output_tokens,
            "maximum_group_depth": max_depth,
            "weight_bits_per_issue": template["weight_bits_per_issue"],
            "banks_128b_required": template["banks_128b_required"],
            "fits_eight_128b_banks": template["fits_eight_128b_banks"],
        })
        aggregate_points[name] = sums

    baseline = aggregate_points["K1_D96_ROW_STRIPED"]
    for point in aggregate_points.values():
        point["service_speedup_vs_k1_d96"] = (
            float(baseline["service_cycles"]) / float(point["service_cycles"])
        )
        point["serial_speedup_vs_k1_d96"] = (
            float(baseline["conservative_serial_cycles"]) /
            float(point["conservative_serial_cycles"])
        )
        scaled_binary_cycles = (
            float(PATCH_BINARY_ELIGIBLE_LEDGER_CYCLES) /
            point["serial_speedup_vs_k1_d96"]
        )
        scaled_patch_cycles = scaled_binary_cycles + PATCH_NONBINARY_LEDGER_CYCLES
        point["profile100_patch_cycles_sensitivity"] = scaled_patch_cycles
        point["profile100_patch_speedup_sensitivity"] = (
            float(PATCH_LEDGER_CYCLES) / scaled_patch_cycles
        )
        point["profile100_compute_envelope_speedup_sensitivity"] = (
            float(GLOBAL_ENVELOPE_CYCLES) /
            float(GLOBAL_ENVELOPE_CYCLES - PATCH_LEDGER_CYCLES +
                  scaled_patch_cycles)
        )

    ledger_selected = []
    for module in sorted(m55["per_module"], key=lambda row: row["module_index"]):
        if int(module["module_index"]) not in SELECTED_MODULE_INDICES:
            continue
        row = ledger[module["name"]]
        ledger_selected.append({
            "module_index": int(module["module_index"]),
            "name": module["name"],
            "profile100_activity_cycles": int(row["activity_cycles_at_config_lanes"]),
            "profile100_activity_weighted_macs": int(
                row["activity_weighted_macs_per_frame"]
            ),
            "m51_s10_zero_source_bits": int(module["zero_source_bits"]),
            "m51_s10_local_source_bits": int(module["local_source_bits"]),
            "m51_s10_dual_source_bits": int(module["dual_source_bits"]),
        })
    require(sum(row["profile100_activity_cycles"] for row in ledger_selected) ==
            PATCH_BINARY_ELIGIBLE_LEDGER_CYCLES,
            "binary patch ledger partition mismatch")
    require(PATCH_BINARY_ELIGIBLE_LEDGER_CYCLES +
            PATCH_NONBINARY_LEDGER_CYCLES == PATCH_LEDGER_CYCLES,
            "patch ledger partition mismatch")

    result = {
        "schema": "m222_h67_patch_kbank_linebuffer_premodel_v1",
        "status": "PASS_EXACT_M51_S10_PATCH_RECEPTIVE_FIELD_SCREEN_NO_SPEEDUP_ADMISSION",
        "scope": (
            "six frozen binary-eligible patch Conv3x3 inputs, exact valid "
            "receptive-field source work, source-key-mod-K banking, ideal "
            "line-buffer scan and output commit"
        ),
        "identity": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "ledger": str(args.ledger.resolve()),
            "ledger_sha256": sha256(args.ledger),
            "m55": str(args.m55.resolve()),
            "m55_sha256": sha256(args.m55),
            "analyzer_sha256": sha256(Path(__file__)),
            "docs359_sha256": sha256(args.docs359),
        },
        "population": {
            "samples": 10,
            "binary_patch_modules": 6,
            "records": len(records),
            "stride_histogram": stride_histogram,
            "valid_receptive_field_source_contributions": aggregate_source,
            "product_updates": aggregate_source * OUTPUT_CHANNELS,
            "input_vectors": aggregate_input_vectors,
            "output_tokens": aggregate_output_tokens,
        },
        "bank_mapping": (
            "flattened source key ((input_channel*3+kernel_y)*3+kernel_x) "
            "mod K; at most one source row per bank per coissue group"
        ),
        "work_conservation": {
            "all_points_preserve_source_x_destination_products": True,
            "destination_channels": OUTPUT_CHANNELS,
            "zero_or_delta_work_removed": False,
            "weight_rows_or_bytes_skipped_beyond_zero_input": False,
        },
        "aggregate_model_points": aggregate_points,
        "ledger_partition": {
            "profile100_patch_total_cycles": PATCH_LEDGER_CYCLES,
            "profile100_binary_eligible_cycles": PATCH_BINARY_ELIGIBLE_LEDGER_CYCLES,
            "profile100_nonbinary_head_plus_ped_shortcut_cycles": PATCH_NONBINARY_LEDGER_CYCLES,
            "selected_modules": ledger_selected,
            "sensitivity_qualification": (
                "profile100 ledger cycles are scaled only by the exact M51-s10 "
                "serial service ratios; these fields are cross-population DSE, "
                "not admitted cycle or system speedup"
            ),
        },
        "fairness_attack": {
            "strong_k1_reference": (
                "K1/D96 stripes one 96-byte weight row over six of the same "
                "eight 128-bit banks and completes one source contribution/cycle"
            ),
            "equal_96_product_lane_points_cannot_beat_k1_by_parallelism": True,
            "m218_like_k8_d16_has_128_product_lanes_and_1024_weight_bits_per_issue": True,
            "wide_points_trade_weight_bandwidth_and_add_lanes_for_cycles": True,
            "required_next_evidence": (
                "matched DC for 96 INT8 MAC reference versus selected add-only "
                "KxD datapath, plus real SRAM width/energy and accumulator ports"
            ),
        },
        "per_record": per_record,
        "claim_boundary": {
            "exact_s10_source_work_and_bank_groups": True,
            "exact_product_conservation": True,
            "linebuffer_scan_and_commit_accounted_as_ideal_cycles": True,
            "dynamic_bn_barrier_cycles": False,
            "real_sram_ports_or_energy": False,
            "rtl_or_vcs_calibration": False,
            "dc_area_or_timing": False,
            "profile100_speedup": False,
            "complete_patch_cycles": False,
            "system_speedup": False,
            "headline": False,
        },
        "next_gate": (
            "Select only a point that exceeds 1.5x conservative serial service "
            "against K1/D96, then synthesize a matched add-only datapath and "
            "96-MAC reference.  Otherwise pivot to FC1 Acc19 K-bank service."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_json = args.output_dir / "m222_h67_patch_kbank_linebuffer_premodel_r1.json"
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    output_csv = args.output_dir / "m222_model_points.csv"
    with output_csv.open("w", newline="") as handle:
        keys = [
            "name", "source_banks", "destination_lanes_per_source",
            "product_lanes", "weight_bits_per_issue", "banks_128b_required",
            "fits_eight_128b_banks", "source_owned_groups",
            "average_sources_per_nonempty_group", "service_cycles",
            "conservative_serial_cycles", "service_speedup_vs_k1_d96",
            "serial_speedup_vs_k1_d96", "profile100_patch_speedup_sensitivity",
            "profile100_compute_envelope_speedup_sensitivity",
        ]
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for name, _, _ in MODEL_POINTS:
            writer.writerow({key: aggregate_points[name][key] for key in keys})
    print("PASS M222 records={} source={} products={} output={}".format(
        len(records), aggregate_source, aggregate_source * OUTPUT_CHANNELS,
        output_json
    ))


if __name__ == "__main__":
    main()
