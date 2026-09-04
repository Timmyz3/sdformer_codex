#!/usr/bin/env python3
"""Prove exact UQ0.24 representation for frozen H67 M32 thresholds."""

import argparse
import hashlib
import json
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACT = (
    ROOT / "hw_autoresearch_nts07/contracts/"
    "m33_checkpoint_uq0p24_input_contract_r1_20260822.json"
)


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
        contract.get("schema") != "m33_checkpoint_uq0p24_input_contract_v1"
        or contract.get("status")
        != "FROZEN_H67_EP35_TEN_THRESHOLD_EXACT_REPRESENTATION"
    ):
        raise ValueError("unexpected M33 UQ0.24 contract")
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
    return contract, paths, hashes


def balanced_radix128_unsigned24(value):
    if int(value) < 0 or int(value) >= (1 << 24):
        raise ValueError("value outside unsigned24 range")
    residual = int(value)
    digits = []
    for _unused in range(4):
        digit = residual % 128
        if digit >= 64:
            digit -= 128
        digits.append(digit)
        residual = (residual - digit) // 128
    if residual != 0:
        raise ValueError("unsigned24 balanced radix residual is nonzero")
    if any(digit < -64 or digit > 63 for digit in digits[:3]):
        raise ValueError("unsigned24 low digit range drift")
    if digits[3] < 0 or digits[3] > 8:
        raise ValueError("unsigned24 high digit range drift")
    reconstructed = sum(
        digit * (128 ** index) for index, digit in enumerate(digits)
    )
    if reconstructed != int(value):
        raise ValueError("unsigned24 digit reconstruction mismatch")
    return digits


def build_report(contract_path=DEFAULT_CONTRACT):
    _contract, paths, hashes = load_contract(contract_path)
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
        value = struct.unpack("<f", raw_bytes)[0]
        raw_word = struct.unpack("<I", raw_bytes)[0]
        if value != float(source["value_float32"]):
            raise ValueError("M33 threshold float32 raw identity drift")
        scaled = value * float(1 << 24)
        uq_raw = int(scaled)
        if (
            value < 0.0 or value >= 1.0
            or scaled != float(uq_raw)
            or float(uq_raw) / float(1 << 24) != value
            or uq_raw < 0 or uq_raw >= (1 << 24)
        ):
            raise ValueError("M33 threshold is not exact UQ0.24")
        digits = balanced_radix128_unsigned24(uq_raw)
        rows.append({
            "producer": source["producer"],
            "float32_value": value,
            "float32_raw_le_hex": source["value_raw_le_hex"],
            "float32_raw_word_hex": "{:08x}".format(raw_word),
            "uq0p24_raw": uq_raw,
            "uq0p24_raw_hex": "{:06x}".format(uq_raw),
            "balanced_radix128_digits_lsd_first": digits,
            "exact_roundtrip": True,
        })

    signed56_minimum = -(1 << 55)
    signed56_maximum = (1 << 55) - 1
    acc32_minimum = -(1 << 31)
    acc32_maximum = (1 << 31) - 1
    uq24_maximum = (1 << 24) - 1
    product_minimum = acc32_minimum * uq24_maximum
    product_maximum = acc32_maximum * uq24_maximum
    if (
        product_minimum < signed56_minimum
        or product_maximum > signed56_maximum
    ):
        raise ValueError("M33 Acc32-by-UQ0.24 does not fit signed56")

    return {
        "schema": "m33_checkpoint_uq0p24_audit_v1",
        "status": (
            "PASS_TEN_H67_FLOAT32_THRESHOLDS_EXACT_UQ0P24_"
            "FOUR_BALANCED_RADIX128_DIGITS_SIGNED56_PRODUCT_RANGE"
        ),
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
            "range": "0 <= raw <= 2^24-1; value=raw/2^24",
            "checkpoint_representation_exact": True,
            "rounding_required_for_threshold_load": False,
        },
        "thresholds": rows,
        "radix_schedule": {
            "radix": 128,
            "acc32_digits": 5,
            "uq0p24_digits": 4,
            "signed_int8_products_per_output": 20,
            "outputs_per_96_lane_cycle_floor": 4,
            "active_multiplier_lanes_full_packet": 80,
            "spare_multiplier_lanes_full_packet": 16,
        },
        "signed56_range_proof": {
            "acc32_minimum": acc32_minimum,
            "acc32_maximum": acc32_maximum,
            "uq0p24_maximum": uq24_maximum,
            "minimum_product": product_minimum,
            "maximum_product": product_maximum,
            "signed56_minimum": signed56_minimum,
            "signed56_maximum": signed56_maximum,
            "fits": True,
        },
        "admission": {
            "threshold_representation_admitted": True,
            "full_fixed_point_pipeline_admitted": False,
            "rne_saturation_bias_admitted": False,
            "rtl_admitted": False,
            "cycle_performance_admitted": False,
            "headline_admitted": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite M33 UQ report")
    report = build_report(args.contract.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
