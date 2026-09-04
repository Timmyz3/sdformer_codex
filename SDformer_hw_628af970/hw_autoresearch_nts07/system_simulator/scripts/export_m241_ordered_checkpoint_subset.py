#!/usr/bin/env python3
"""Export a real signed-INT8 ordered Conv subset for the M241 exact miter."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M158_SCRIPT = HW / "system_simulator/scripts/prove_m158_source_major_acc19_reorder_exactness.py"
M158_RESULT = HW / "results/m158_source_major_acc19_reorder_exactness_r2_20260824/m158_source_major_acc19_reorder_exactness.json"
M158_MANIFEST = HW / "results/m158_source_major_acc19_reorder_exactness_r2_20260824/manifest.sha256"
M150_AUDIT = HW / "results/m150_independent_hammer_review_r1_20260824/audit_m150_independent.py"
M41_RESULT = HW / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/m41_h67_ep35_bottleneck_int8_bridge.json"
WEIGHT = HW / "results/m41_h67_ep35_bottleneck_int8_bridge_r1_20260823/o0_weight_i_ky_kx_o_s8.bin"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
PATHS = {
    "m158_script": M158_SCRIPT,
    "m158_result": M158_RESULT,
    "m158_manifest": M158_MANIFEST,
    "m150_audit": M150_AUDIT,
    "m41_result": M41_RESULT,
    "weight_o0": WEIGHT,
    "docs359": DOCS359,
}
EXPECTED = {
    "m158_script": "be024d2b8f7a674e5ab6dbad8c4a43cbea31320e8e9ab193fdf9e8dd5b9f0a3e",
    "m158_result": "c7c6738f66f6b6eb00455c6edc17983bccf3a40d09d8e1f2b9e0752a27b644b2",
    "m158_manifest": "22067f8a3bffaae0b00e200a8c3950467c17ee57692862d66d823689ecc14f1e",
    "m150_audit": "3f975e74178a51d709f502a135792cebdac7f8422a61ff8277acf401c8433d9a",
    "m41_result": "20d745559612c828674a89a417b5ff94512d4bf2553f37b03e8d7b1575f636fb",
    "weight_o0": "1197b961e08f4ca8f156c301280e7e3c630aea3b3bf68b0e78ee0f701e2e9f31",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
LANES = 8
ROWS = 384
FEATURES = 768 * 3 * 3
OUTPUTS = 768


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(label, path):
    spec = importlib.util.spec_from_file_location(label, path)
    require(spec is not None and spec.loader is not None,
            "cannot import " + label)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_descriptors(event_masks, negative_masks, partition, start, end):
    descriptors = []
    previous = None
    for source in range(16):
        active = ((event_masks[partition, start:end, :] >> source)
                  & 1).astype(np.uint8, copy=False)
        negative = ((negative_masks[partition, start:end, :] >> source)
                    & 1).astype(np.uint8, copy=False)
        destination_masks = np.zeros(end - start, dtype=np.uint8)
        negative_masks8 = np.zeros(end - start, dtype=np.uint8)
        for destination in range(8):
            destination_masks |= active[:, destination] << destination
            negative_masks8 |= negative[:, destination] << destination

        for half in (0, 1):
            phase_masks = ((destination_masks >> (4 * half))
                           & np.uint8(0x0f))
            indices = np.flatnonzero(phase_masks).tolist()
            if not indices:
                continue
            first = 0
            if previous is not None:
                prev_half, prev_row, prev_banks = previous
                hazard = (prev_half == half and prev_row == indices[0]
                          and bool(prev_banks & int(phase_masks[indices[0]])))
                if hazard and len(indices) > 1:
                    first = next(index for index, row in enumerate(indices)
                                 if row != prev_row)
            ordered = [indices[first]] + [row for index, row in enumerate(indices)
                                          if index != first]
            for local_row in ordered:
                mask = int(phase_masks[local_row])
                destinations = [bank + 4 * half for bank in range(4)
                                if mask & (1 << bank)]
                negates = [int(bool(int(negative_masks8[local_row])
                                        & (1 << destination)))
                           for destination in destinations]
                descriptors.append({
                    "row": int(local_row),
                    "source": source,
                    "half": half,
                    "bank_mask": mask,
                    "destinations": destinations,
                    "negates": negates,
                })
                previous = (half, int(local_row), mask)
    return descriptors


def context_score(descriptors):
    count = len(descriptors)
    full4 = sum(len(row["destinations"]) == 4 for row in descriptors)
    tails = sum(len(row["destinations"]) < 4 for row in descriptors)
    negated = sum(sum(row["negates"]) for row in descriptors)
    lows = sum(row["half"] == 0 for row in descriptors)
    highs = count - lows
    source_keys = len({(row["source"], row["half"])
                       for row in descriptors})
    eligible = (16 <= count <= 128 and full4 > 0
                and negated > 0 and lows > 0 and highs > 0
                and source_keys >= 3)
    return eligible, (min(count, 96), full4, tails, negated, source_keys)


def pack_descriptor(order, row):
    destinations = row["destinations"] + [0] * (4 - len(row["destinations"]))
    negates = row["negates"] + [0] * (4 - len(row["negates"]))
    value = int(order)
    value |= int(row["row"]) << 16
    value |= int(row["source"]) << 25
    value |= ((1 << len(row["destinations"])) - 1) << 29
    for index, destination in enumerate(destinations):
        value |= int(destination) << (33 + 3 * index)
    for index, negate in enumerate(negates):
        value |= int(negate) << (45 + index)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), "refusing to overwrite M241 vectors")
    script_start = sha256(Path(__file__).resolve())
    observed = {label: sha256(path) for label, path in PATHS.items()}
    require(observed == EXPECTED, "M241 frozen direct-input identity drift")

    m158 = load_module("m241_frozen_m158", M158_SCRIPT)
    m157 = load_module("m241_frozen_m157", m158.M157_SCRIPT)
    m152 = load_module("m241_frozen_m152", m157.M152_SCRIPT)
    m150 = load_module("m241_frozen_m150", m152.M150_SCRIPT)
    m147 = load_module("m241_frozen_m147", m150.M147_SCRIPT)
    inherited = {label: sha256(path) for label, path in m147.PATHS.items()}
    require(inherited == m147.EXPECTED,
            "M241 inherited ordered-trace identity drift")
    audit = load_module("m241_frozen_audit", m147.PATHS["m141_audit"])
    m132 = load_module("m241_frozen_m132", m147.PATHS["m132_script"])
    m105 = load_module("m241_frozen_m105", m132.M105_SCRIPT)
    m150_audit = load_module("m241_frozen_m150_audit", M150_AUDIT)

    manifest = audit.strict_json(m147.PATHS["m40_manifest"])
    m72 = audit.strict_json(m147.PATHS["m72_result"])
    m41 = audit.strict_json(m147.PATHS["m41_result"])
    heldout = sorted((row for row in manifest["records"]
                      if row["sample_id"] in range(5, 10)),
                     key=lambda row: (row["sample_id"],
                                      row["operator_index"]))
    popcount = np.fromiter((bin(value).count("1")
                            for value in range(1 << 16)),
                           dtype=np.uint8, count=1 << 16)
    centers = m105.centers_array(m72)
    widths, _, _ = m105.build_width_catalog(m72, m41)

    selected = None
    best_score = None
    for record in heldout:
        if int(record["operator_index"]) != 0:
            continue
        masks = m105.decode_natural_partition_masks(record, popcount)
        event_masks, negative_masks, _, _ = m150_audit.select_events(
            masks, record["operator_index"], centers, widths, popcount)
        for partition in range(m147.PARTITIONS):
            for window, start in enumerate(range(0, m147.ROWS, ROWS)):
                end = min(start + ROWS, m147.ROWS)
                descriptors = build_descriptors(
                    event_masks, negative_masks, partition, start, end)
                eligible, score = context_score(descriptors)
                if eligible and (best_score is None or score > best_score):
                    selected = (record, partition, window, start, end,
                                descriptors)
                    best_score = score
        if selected is not None:
            break
    require(selected is not None, "no bounded M241 coverage context found")
    record, partition, window, start, end, descriptors = selected
    require(len(descriptors) < (1 << 16), "M241 order width overflow")

    for index, row in enumerate(descriptors):
        row["order"] = index
        row["last"] = index == len(descriptors) - 1
    packed = [pack_descriptor(index, row)
              | (int(row["last"]) << 49)
              for index, row in enumerate(descriptors)]

    weight = np.fromfile(str(WEIGHT), dtype=np.int8)
    require(weight.size == FEATURES * OUTPUTS, "M241 weight size drift")
    weight = weight.reshape(FEATURES, OUTPUTS)
    weight_lines = []
    for bank in range(4):
        for address in range(32):
            half = address >> 4
            source = address & 15
            feature = partition * 16 + source
            destination = bank + 4 * half
            for lane in range(LANES):
                weight_lines.append(int(weight[
                    feature, destination * 96 + lane]) & 0xff)

    sequence = 0x24100000 | (int(record["sample_id"]) << 8)
    epoch = int(EXPECTED["weight_o0"][:4], 16)
    negative_descriptors = sum(any(row["negates"]) for row in descriptors)
    negative_tuples = sum(sum(row["negates"]) for row in descriptors)
    full4 = sum(len(row["destinations"]) == 4 for row in descriptors)
    tail = len(descriptors) - full4
    low = sum(row["half"] == 0 for row in descriptors)
    high = len(descriptors) - low
    meta = [len(descriptors), sequence, int(record["operator_index"]),
            partition, epoch, window, int(record["sample_id"]), LANES,
            ROWS, negative_descriptors, negative_tuples, full4, tail,
            low, high, start]

    output.mkdir(parents=True)
    (output / "descriptor.mem").write_text(
        "".join(f"{value:016x}\n" for value in packed),
        encoding="ascii")
    (output / "weight.mem").write_text(
        "".join(f"{value:02x}\n" for value in weight_lines),
        encoding="ascii")
    (output / "meta.mem").write_text(
        "".join(f"{value & 0xffffffff:08x}\n" for value in meta),
        encoding="ascii")

    descriptor_digest = hashlib.sha256(
        b"".join(int(value).to_bytes(8, "little") for value in packed)
    ).hexdigest()
    payload = {
        "schema": "m241_ordered_checkpoint_subset_v1",
        "status": "PASS_EXACT_SHA_REAL_ORDERED_SIGNED_INT8_SUBSET_EXPORT",
        "identity": {
            "exporter_start_end_sha256": script_start,
            "direct_inputs_sha256": observed,
            "inherited_m147_inputs_sha256": inherited,
        },
        "selection": {
            "lineage": "H67/Motion ep35 heldout",
            "sample_id": int(record["sample_id"]),
            "operator_index": int(record["operator_index"]),
            "operator": "sttmultires_unet.resblocks.0.conv1.0",
            "partition": partition,
            "window": window,
            "raw_row_start": start,
            "raw_row_end_exclusive": end,
            "local_rows": ROWS,
            "lanes_per_destination_subset": LANES,
            "sequence": sequence,
            "weight_epoch": epoch,
        },
        "coverage": {
            "descriptors": len(descriptors),
            "full4_descriptors": full4,
            "tail_descriptors": tail,
            "low_half_descriptors": low,
            "high_half_descriptors": high,
            "negative_descriptors": negative_descriptors,
            "negative_tuples": negative_tuples,
            "source_half_keys": len({(row["source"], row["half"])
                                     for row in descriptors}),
            "ordered_descriptor_digest_sha256": descriptor_digest,
        },
        "layout": {
            "weight": "bank_then_addr_half_source_then_lane_s8",
            "descriptor": "u64_order_row_source_prefix_valid_d0_d1_d2_d3_negate_last",
            "accumulator_address": "dense_half_times_384_plus_local_row",
        },
        "admission": {
            "real_checkpoint_payload": True,
            "real_ordered_signed_event_subset": True,
            "complete_heldout_trace": False,
            "physical_speedup": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
    }
    require(sha256(Path(__file__).resolve()) == script_start,
            "M241 exporter changed during execution")
    (output / "m241_ordered_checkpoint_subset.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print("PASS M241 export descriptors={} full4={} tail={} negative={} "
          "low={} high={} sample={} op={} partition={} window={} "
          "complete_trace=false physical_speedup=false system_speedup=false "
          "headline=false".format(
              len(descriptors), full4, tail, negative_tuples, low, high,
              record["sample_id"], record["operator_index"], partition,
              window), flush=True)


if __name__ == "__main__":
    main()
