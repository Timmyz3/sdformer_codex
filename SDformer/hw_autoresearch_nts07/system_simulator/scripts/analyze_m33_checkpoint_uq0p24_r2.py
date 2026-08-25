#!/usr/bin/env python3
"""Audit checkpoint-exact UQ0.24 and its 5x4 signed-digit product identity."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import struct


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACT = (
    ROOT / "hw_autoresearch_nts07/contracts/"
    "m33_checkpoint_uq0p24_input_contract_r2_20260822.json"
)
REGRESSION_SEED = 0x4D333202
RANDOM_CASES = 10000


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(raw):
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_contract(contract_path):
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    if (
        contract.get("schema") != "m33_checkpoint_uq0p24_input_contract_v2"
        or contract.get("status")
        != "FROZEN_H67_EP35_UQ0P24_CROSS_PRODUCT_PROOF"
    ):
        raise ValueError("unexpected M33 UQ0.24 r2 contract")
    paths = {}
    hashes = {}
    for name, spec in sorted(contract["inputs"].items()):
        path = resolve_path(spec["path"])
        if not path.is_file():
            raise ValueError("missing M33 UQ input {}".format(name))
        actual = sha256(path)
        if actual != spec["sha256"]:
            raise ValueError("M33 UQ input hash drift for {}".format(name))
        paths[name] = path
        hashes[name] = actual
    return paths, hashes


def float32_raw_to_exact_uq0p24(raw_bytes):
    """Decode IEEE binary32 bits directly and reject non-exact UQ0.24."""
    if len(raw_bytes) != 4:
        raise ValueError("binary32 input must contain four bytes")
    word = struct.unpack("<I", raw_bytes)[0]
    sign = (word >> 31) & 1
    exponent = (word >> 23) & 0xff
    fraction = word & 0x7fffff
    if sign or exponent == 0xff:
        raise ValueError("binary32 value is negative or non-finite")
    if exponent == 0:
        significand = fraction
        shift = -125
    else:
        significand = (1 << 23) | fraction
        shift = exponent - 126
    if shift >= 0:
        uq_raw = significand << shift
    else:
        divisor = 1 << (-shift)
        if significand % divisor:
            raise ValueError("binary32 value is not exact UQ0.24")
        uq_raw = significand // divisor
    if uq_raw < 0 or uq_raw >= (1 << 24):
        raise ValueError("binary32 value is outside UQ0.24 range")
    return word, uq_raw


def balanced_radix128(value, digits, minimum, maximum):
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("balanced-radix input must be an integer")
    if value < minimum or value > maximum:
        raise ValueError("balanced-radix input outside declared range")
    residual = value
    result = []
    for _unused in range(digits):
        digit = residual % 128
        if digit >= 64:
            digit -= 128
        result.append(digit)
        residual = (residual - digit) // 128
    if residual != 0:
        raise ValueError("balanced-radix residual is nonzero")
    if any(digit < -64 or digit > 63 for digit in result):
        raise ValueError("balanced-radix digit exceeds signed INT8 contract")
    if sum(digit << (7 * index) for index, digit in enumerate(result)) != value:
        raise ValueError("balanced-radix reconstruction mismatch")
    return result


def acc32_digits(value):
    return balanced_radix128(value, 5, -(1 << 31), (1 << 31) - 1)


def uq24_digits(value):
    result = balanced_radix128(value, 4, 0, (1 << 24) - 1)
    if result[3] < 0 or result[3] > 8:
        raise ValueError("UQ24 high digit outside constructive bound")
    return result


def cross_product(accumulator, threshold_uq):
    left = acc32_digits(accumulator)
    right = uq24_digits(threshold_uq)
    recombined = 0
    for left_index, left_digit in enumerate(left):
        for right_index, right_digit in enumerate(right):
            # Both operands are signed INT8 digits, including negative low
            # UQ digits.  Each signed16 product is widened before a positive
            # power-of-two shift and signed64 addition.
            recombined += (
                left_digit * right_digit
            ) << (7 * (left_index + right_index))
    return recombined


def build_cross_product_regression():
    accumulator_edges = [
        -(1 << 31), -(1 << 31) + 1, -129, -128, -65, -64, -1,
        0, 1, 63, 64, 127, 128, (1 << 31) - 2, (1 << 31) - 1,
    ]
    threshold_edges = [
        0, 1, 63, 64, 127, 128, (1 << 23) - 1, 1 << 23,
        (1 << 24) - 2, (1 << 24) - 1,
    ]
    cases = [(acc, threshold) for acc in accumulator_edges
             for threshold in threshold_edges]
    generator = random.Random(REGRESSION_SEED)
    for _unused in range(RANDOM_CASES):
        cases.append((
            generator.randint(-(1 << 31), (1 << 31) - 1),
            generator.randint(0, (1 << 24) - 1),
        ))
    digest = hashlib.sha256()
    mismatches = 0
    for accumulator, threshold in cases:
        observed = cross_product(accumulator, threshold)
        expected = accumulator * threshold
        digest.update(struct.pack(">iIqq", accumulator, threshold,
                                  expected, observed))
        if observed != expected:
            mismatches += 1
    if mismatches:
        raise ValueError("M33 UQ cross-product regression mismatch")
    return {
        "seed_hex": "0x{:08x}".format(REGRESSION_SEED),
        "edge_cases": len(accumulator_edges) * len(threshold_edges),
        "random_cases": RANDOM_CASES,
        "total_cases": len(cases),
        "mismatches": mismatches,
        "vector_and_result_sha256": digest.hexdigest(),
    }


def build_report(contract_path=DEFAULT_CONTRACT):
    paths, hashes = load_contract(contract_path)
    m32 = json.loads(paths["m32_semantic_closure"].read_text(encoding="utf-8"))
    thresholds = json.loads(
        paths["threshold_manifest"].read_text(encoding="utf-8")
    )
    if (
        m32.get("semantic_admission") is not True
        or m32.get("headline_admitted") is not False
        or thresholds.get("schema") != "m32_h67_checkpoint_threshold_manifest_v1"
        or len(thresholds.get("producers", [])) != 10
    ):
        raise ValueError("M33 UQ semantic/threshold population drift")

    rows = []
    for source in thresholds["producers"]:
        raw_bytes = bytes.fromhex(source["value_raw_le_hex"])
        raw_word, uq_raw = float32_raw_to_exact_uq0p24(raw_bytes)
        float_value = struct.unpack("<f", raw_bytes)[0]
        if float_value != float(source["value_float32"]):
            raise ValueError("M33 threshold float32 raw identity drift")
        digits = uq24_digits(uq_raw)
        rows.append({
            "producer": source["producer"],
            "float32_value": float_value,
            "float32_raw_le_hex": source["value_raw_le_hex"],
            "float32_raw_word_hex": "{:08x}".format(raw_word),
            "uq0p24_raw": uq_raw,
            "uq0p24_raw_hex": "{:06x}".format(uq_raw),
            "balanced_radix128_signed_int8_digits_lsd_first": digits,
            "exact_roundtrip": True,
        })

    signed56_minimum = -(1 << 55)
    signed56_maximum = (1 << 55) - 1
    product_minimum = -(1 << 31) * ((1 << 24) - 1)
    product_maximum = ((1 << 31) - 1) * ((1 << 24) - 1)
    if product_minimum < signed56_minimum or product_maximum > signed56_maximum:
        raise ValueError("M33 Acc32-by-UQ0.24 does not fit signed56")

    return {
        "schema": "m33_checkpoint_uq0p24_cross_product_audit_v2",
        "status": "PASS_EXACT_UQ0P24_AND_SIGNED_DIGIT_CROSS_PRODUCT_IDENTITY",
        "identity": {
            "input_contract": str(Path(contract_path).resolve()),
            "input_contract_sha256": sha256(contract_path),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
            "verified_input_sha256": hashes,
            "checkpoint_sha256": m32["identity"]["checkpoint_sha256"],
        },
        "format": {
            "name": "UQ0.24",
            "stored_bits": 24,
            "fractional_bits": 24,
            "decode": "IEEE-754 raw-bit integer decode; no float rounding path",
            "checkpoint_representation_exact": True,
            "rounding_required_for_threshold_load": False,
        },
        "thresholds": rows,
        "constructive_full_domain_identity": {
            "accumulator_domain": "signed Acc32",
            "threshold_domain": "unsigned UQ24 raw",
            "accumulator_digits": 5,
            "threshold_digits": 4,
            "products_per_scalar_output": 20,
            "digit_encoding": "two's-complement signed INT8 [-64,63]",
            "multiplier_semantics": "both digit operands sign-extended; signed8xsigned8 -> signed16",
            "recombination_semantics": "signed16 product widened to signed64 before left shift by 7*(i+j), then signed64 sum",
            "proof": "exact digit reconstruction plus integer distributivity proves sum_ij(a_i*b_j*128^(i+j)) = Acc32*UQ24 for the full declared domains",
            "ideal_arithmetic_slot_packing_upper_bound_per_96_lanes": 4,
            "active_slots_at_upper_bound": 80,
            "unassigned_slots_at_upper_bound": 16,
        },
        "cross_product_regression": build_cross_product_regression(),
        "signed56_range_proof": {
            "minimum_product_formula": "-2^31*(2^24-1)",
            "maximum_product_formula": "(2^31-1)*(2^24-1)",
            "minimum_product": product_minimum,
            "maximum_product": product_maximum,
            "signed56_minimum": signed56_minimum,
            "signed56_maximum": signed56_maximum,
            "minimum_headroom": product_minimum - signed56_minimum,
            "maximum_headroom": signed56_maximum - product_maximum,
            "fits": True,
        },
        "admission": {
            "threshold_representation_admitted": True,
            "integer_cross_product_identity_admitted": True,
            "rtl_admitted": False,
            "feed_and_recombine_timing_admitted": False,
            "full_fixed_point_pipeline_admitted": False,
            "rne_saturation_bias_admitted": False,
            "system_cycle_performance_admitted": False,
            "ppa_admitted": False,
            "power_energy_admitted": False,
            "headline_admitted": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite M33 UQ r2 report")
    report = build_report(args.contract.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
