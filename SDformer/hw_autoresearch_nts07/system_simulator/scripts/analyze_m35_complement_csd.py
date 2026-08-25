#!/usr/bin/env python3
"""Prove multiplier-free complement/CSD scaling for frozen H67 thresholds."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import struct


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACT = (
    ROOT / "hw_autoresearch_nts07/contracts/"
    "m35_complement_csd_input_contract_r3_20260822.json"
)
REGRESSION_SEED = 0x4D350001


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_signed_digit(value):
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("CSD input must be an integer")
    if value < 0:
        raise ValueError("CSD complement must be nonnegative")
    residual = value
    bit = 0
    terms = []
    while residual:
        if residual & 1:
            sign = 2 - (residual & 3)
            terms.append({"shift": bit, "coefficient": sign})
            residual -= sign
        residual //= 2
        bit += 1
    if sum(term["coefficient"] << term["shift"] for term in terms) != value:
        raise ValueError("CSD reconstruction mismatch")
    if any(term["coefficient"] not in (-1, 1) for term in terms):
        raise ValueError("CSD coefficient drift")
    return terms


def csd_multiply(accumulator, terms):
    return sum(
        term["coefficient"] * (accumulator << term["shift"])
        for term in terms
    )


def minimum_signed_power_terms(value, maximum_shift=12):
    """Exhaustively prove the minimum unique signed-power term count."""
    for term_count in range(maximum_shift + 2):
        for shifts in itertools.combinations(
                range(maximum_shift + 1), term_count):
            for coefficients in itertools.product((-1, 1), repeat=term_count):
                if sum(coefficient << shift for shift, coefficient
                       in zip(shifts, coefficients)) == value:
                    return term_count
    raise ValueError("no signed-power representation inside declared bound")


def complement_product(accumulator, delta, terms=None):
    if accumulator < -(1 << 31) or accumulator > (1 << 31) - 1:
        raise ValueError("accumulator outside signed32")
    if delta < 0 or delta >= (1 << 24):
        raise ValueError("delta outside complement domain")
    terms = canonical_signed_digit(delta) if terms is None else terms
    correction = csd_multiply(accumulator, terms)
    return (accumulator << 24) - correction


def load_input(contract_path):
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    if (
        contract.get("schema") != "m35_complement_csd_input_contract_v3"
        or contract.get("status")
        != "FROZEN_M33_R2_COMPLEMENT_CSD_SIGNED42_GLOBAL_MINIMUM_SEARCH"
    ):
        raise ValueError("unexpected M35 contract")
    spec = contract["inputs"]["m33_uq_cross_product"]
    path = Path(spec["path"])
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if not path.is_file() or sha256(path) != spec["sha256"]:
        raise ValueError("M35 M33 input missing or hash drift")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("status")
        != "PASS_EXACT_UQ0P24_AND_SIGNED_DIGIT_CROSS_PRODUCT_IDENTITY"
        or report["admission"].get("threshold_representation_admitted") is not True
        or report["admission"].get("integer_cross_product_identity_admitted") is not True
        or len(report.get("thresholds", [])) != 10
    ):
        raise ValueError("M35 M33 semantic admission drift")
    return report, path


def build_regression(rows):
    accumulators = [
        -(1 << 31), -(1 << 31) + 1, -129, -128, -65, -64, -1,
        0, 1, 63, 64, 127, 128, (1 << 31) - 2, (1 << 31) - 1,
    ]
    cases = []
    for row in rows:
        for accumulator in accumulators:
            cases.append((accumulator, row["threshold_uq0p24_raw"], row["delta"], row["csd_terms"]))
    generator = random.Random(REGRESSION_SEED)
    for _unused in range(10000):
        row = rows[generator.randrange(len(rows))]
        cases.append((
            generator.randint(-(1 << 31), (1 << 31) - 1),
            row["threshold_uq0p24_raw"], row["delta"], row["csd_terms"],
        ))
    digest = hashlib.sha256()
    mismatches = 0
    for accumulator, threshold, delta, terms in cases:
        observed = complement_product(accumulator, delta, terms)
        expected = accumulator * threshold
        digest.update(struct.pack(">iIIqq", accumulator, threshold, delta,
                                  expected, observed))
        if observed != expected:
            mismatches += 1
    if mismatches:
        raise ValueError("M35 complement/CSD regression mismatch")
    return {
        "seed_hex": "0x{:08x}".format(REGRESSION_SEED),
        "edge_cases": 150,
        "random_cases": 10000,
        "total_cases": len(cases),
        "mismatches": mismatches,
        "vector_and_result_sha256": digest.hexdigest(),
    }


def build_report(contract_path=DEFAULT_CONTRACT):
    source, source_path = load_input(contract_path)
    rows = []
    for threshold in source["thresholds"]:
        raw = int(threshold["uq0p24_raw"])
        delta = (1 << 24) - raw
        terms = canonical_signed_digit(delta)
        minimum_terms = minimum_signed_power_terms(delta)
        if len(terms) != minimum_terms:
            raise ValueError("canonical CSD is not minimum term count")
        if delta <= 0 or delta > 1023 or len(terms) > 4:
            raise ValueError("M35 checkpoint complement outside 10-bit/four-term bound")
        rows.append({
            "producer": threshold["producer"],
            "threshold_uq0p24_raw": raw,
            "threshold_uq0p24_raw_hex": "{:06x}".format(raw),
            "delta": delta,
            "delta_hex": "{:03x}".format(delta),
            "csd_terms": terms,
            "csd_nonzero_terms": len(terms),
            "minimum_signed_power_terms": minimum_terms,
            "minimum_term_count_exhaustively_proven": True,
            "maximum_shift": max(term["shift"] for term in terms),
            "exact_identity": "Acc*(2^24-delta)=(Acc<<24)-sum_k(sign_k*(Acc<<shift_k))",
        })
    regression = build_regression(rows)
    correction_abs_bound = (1 << 31) * max(row["delta"] for row in rows)
    signed42_minimum = -(1 << 41)
    signed42_maximum = (1 << 41) - 1
    correction_minimum = -(1 << 31) * max(row["delta"] for row in rows)
    correction_maximum = ((1 << 31) - 1) * max(
        row["delta"] for row in rows
    )
    if (
        correction_minimum < signed42_minimum
        or correction_maximum > signed42_maximum
    ):
        raise ValueError("M35 correction exceeds signed42 bound")
    return {
        "schema": "m35_complement_csd_audit_v3",
        "status": "PASS_TEN_CHECKPOINT_THRESHOLDS_EXACT_UP_TO_FOUR_TERM_COMPLEMENT_CSD_SIGNED42",
        "identity": {
            "input_contract": str(Path(contract_path).resolve()),
            "input_contract_sha256": sha256(contract_path),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
            "m33_source": str(source_path),
            "m33_source_sha256": sha256(source_path),
            "checkpoint_sha256": source["identity"]["checkpoint_sha256"],
        },
        "thresholds": rows,
        "architecture_bound": {
            "threshold_form": "UQ0.24 raw = 2^24 - unsigned delta",
            "delta_bits": 10,
            "maximum_delta": max(row["delta"] for row in rows),
            "maximum_csd_nonzero_terms": max(row["csd_nonzero_terms"] for row in rows),
            "csd_minimum_term_counts_exhaustively_proven": True,
            "minimum_term_search_maximum_shift": 12,
            "minimum_term_search_sufficiency_lemma": "for at most four distinct signed powers with highest shift H>=13, the smallest nonzero absolute sum is 2^(H-3)>=1024, which exceeds every admitted delta<=588; therefore exhaustive shifts 0..12 prove the global minimum among representations with no more terms than the admitted four-term construction",
            "maximum_shift": max(row["maximum_shift"] for row in rows),
            "runtime_integer_multiplier_products_per_output": 0,
            "signed_shift_terms_per_output_upper_bound": 4,
            "raw_product_width": 56,
            "correction_absolute_bound": correction_abs_bound,
            "correction_minimum": correction_minimum,
            "correction_maximum": correction_maximum,
            "signed42_minimum": signed42_minimum,
            "signed42_maximum": signed42_maximum,
            "correction_fits_signed42": True,
            "base_shift_fits_signed56": True,
            "final_product_fits_signed56": source["signed56_range_proof"]["fits"],
            "unverified_rtl_design_target_outputs_per_cycle": 8,
            "design_target_has_no_math_resource_or_timing_admission": True,
        },
        "regression": regression,
        "admission": {
            "checkpoint_complement_bound_admitted": True,
            "integer_csd_identity_admitted": True,
            "rtl_admitted": False,
            "pipeline_ii_admitted": False,
            "timing_area_admitted": False,
            "system_overlap_admitted": False,
            "system_cycle_performance_admitted": False,
            "fixed_point_rounding_admitted": False,
            "accuracy_admitted": False,
            "ppa_power_energy_admitted": False,
            "headline_admitted": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite M35 report")
    report = build_report(args.contract.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
