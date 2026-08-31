#!/usr/bin/env python3
"""Synthetic-only independent reconstruction for the M513 analyzer.

This script imports the analyzer functions but never opens a capture, checkpoint,
contract, payload-verifier result, or any other production data artifact.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


HW_ROOT = Path(__file__).resolve().parents[2]
ANALYZER = HW_ROOT / "system_simulator/scripts/analyze_m513_h67_decoder_pgpr_tdr_fastkill.py"
EXPECTED_ANALYZER_SHA256 = (
    "303863453d56bf6472ecaf55315b2a5e895494eb019ef70e0f25e13233d089be"
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_analyzer():
    assert sha256(ANALYZER) == EXPECTED_ANALYZER_SHA256
    spec = importlib.util.spec_from_file_location("m513_static_only", str(ANALYZER))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_scatter(counts):
    timesteps, height, width = counts.shape
    result = np.zeros((timesteps, 2 * height, 2 * width), dtype=np.int64)
    for timestep in range(timesteps):
        for source_y in range(height):
            for source_x in range(width):
                for kernel_y in range(3):
                    output_y = 2 * source_y - 1 + kernel_y
                    for kernel_x in range(3):
                        output_x = 2 * source_x - 1 + kernel_x
                        if 0 <= output_y < 2 * height and \
                                0 <= output_x < 2 * width:
                            result[timestep, output_y, output_x] += \
                                int(counts[timestep, source_y, source_x])
    return result


def reference_record(bits, channels_out):
    timesteps, _batch, channels_in, height, width = bits.shape
    previous = np.zeros((channels_in, height, width), dtype=np.bool_)
    current_vectors = delta_vectors = rise_vectors = fall_vectors = 0
    nonempty_destinations = 0
    for timestep in range(timesteps):
        destination = np.zeros((2 * height, 2 * width), dtype=np.int64)
        for channel in range(channels_in):
            for source_y in range(height):
                for source_x in range(width):
                    current = bool(bits[timestep, 0, channel, source_y, source_x])
                    prior = bool(previous[channel, source_y, source_x])
                    changed = current != prior
                    rise = current and not prior
                    fall = prior and not current
                    for kernel_y in range(3):
                        output_y = 2 * source_y - 1 + kernel_y
                        for kernel_x in range(3):
                            output_x = 2 * source_x - 1 + kernel_x
                            if 0 <= output_y < 2 * height and \
                                    0 <= output_x < 2 * width:
                                current_vectors += int(current)
                                delta_vectors += int(changed)
                                rise_vectors += int(rise)
                                fall_vectors += int(fall)
                                if current:
                                    destination[output_y, output_x] += 1
        nonempty_destinations += int(np.count_nonzero(destination))
        previous = bits[timestep, 0].copy()
    slices = channels_out // 96
    return {
        "current_vectors": current_vectors,
        "delta_vectors": delta_vectors,
        "rise_vectors": rise_vectors,
        "fall_vectors": fall_vectors,
        "nonempty_destinations": nonempty_destinations,
        "a1_products": current_vectors * channels_out,
        "a1_cycles": current_vectors * slices,
        "tdr_products": delta_vectors * channels_out,
        "tdr_cycles": delta_vectors * slices,
    }


def main():
    analyzer = load_analyzer()
    random = np.random.RandomState(513)
    scatter_cases = 0
    for height in range(1, 6):
        for width in range(1, 6):
            counts = random.randint(
                0, 8, size=(3, height, width), dtype=np.int32)
            observed = analyzer.scatter_destination_contributors(counts)
            expected = reference_scatter(counts)
            assert np.array_equal(observed, expected)
            scatter_cases += 1

    record_cases = 0
    for channels_in, height, width, channels_out in (
            (1, 1, 1, 96), (3, 2, 4, 192), (5, 4, 3, 384)):
        for _trial in range(8):
            bits = random.randint(
                0, 2, size=(10, 1, channels_in, height, width),
                dtype=np.uint8).astype(np.bool_)
            observed = analyzer.analyze_record(bits, {
                "in_channels": channels_in,
                "out_channels": channels_out,
            })
            expected = reference_record(bits, channels_out)
            assert observed["a1_source_tap_vectors"] == expected["current_vectors"]
            assert observed["destination_contributor_sum"] == expected["current_vectors"]
            assert observed["delta_source_tap_vectors"] == expected["delta_vectors"]
            assert observed["rise_source_tap_vectors"] == expected["rise_vectors"]
            assert observed["fall_source_tap_vectors"] == expected["fall_vectors"]
            assert observed["nonempty_destinations"] == expected["nonempty_destinations"]
            assert observed["a1_products"] == expected["a1_products"]
            assert observed["a1_product_issue_cycles"] == expected["a1_cycles"]
            assert observed["tdr_products"] == expected["tdr_products"]
            assert observed["tdr_product_issue_cycles"] == expected["tdr_cycles"]
            assert expected["delta_vectors"] == \
                expected["rise_vectors"] + expected["fall_vectors"]
            assert observed["a1_product_issue_cycles"] == \
                observed["a1_products"] // 96
            assert observed["tdr_product_issue_cycles"] == \
                observed["tdr_products"] // 96
            record_cases += 1

    input_shapes = ((1536, 15, 20), (770, 30, 40),
                    (386, 60, 80), (194, 120, 160))
    output_shapes = ((384, 30, 40), (192, 60, 80),
                     (96, 120, 160), (96, 240, 320))
    previous_input_bitmap_bytes = sum(
        channels * height * width // 8
        for channels, height, width in input_shapes)
    previous_output_elements = sum(
        channels * height * width
        for channels, height, width in output_shapes)
    assert previous_input_bitmap_bytes == 870300
    assert previous_output_elements == 10598400

    print(json.dumps({
        "schema": "m513_synthetic_math_reconstruction_v1",
        "status": "PASS_SYNTHETIC_ONLY_NO_PRODUCTION_DATA",
        "analyzer_sha256": EXPECTED_ANALYZER_SHA256,
        "scatter_reference_cases": scatter_cases,
        "record_reference_cases": record_cases,
        "top_left_only_fanout_set": [4, 6, 9],
        "cout_full_slice_values": [96, 192, 384],
        "previous_input_bitmap_bytes": previous_input_bitmap_bytes,
        "previous_output_elements": previous_output_elements,
        "previous_output_int16_bytes": previous_output_elements * 2,
        "previous_output_acc24_bytes": previous_output_elements * 3,
        "production_data_opened": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
