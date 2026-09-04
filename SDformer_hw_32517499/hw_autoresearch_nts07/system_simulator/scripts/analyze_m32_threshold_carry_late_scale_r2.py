#!/usr/bin/env python3
"""Harden M32 with checkpoint thresholds, a signed-product oracle, and control cost."""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import struct


ROOT = Path(__file__).resolve().parents[3]
R1_SCRIPT = Path(__file__).resolve().with_name(
    "analyze_m32_threshold_carry_late_scale.py"
)
R1_SPEC = importlib.util.spec_from_file_location("m32_r1", str(R1_SCRIPT))
R1 = importlib.util.module_from_spec(R1_SPEC)
R1_SPEC.loader.exec_module(R1)
DEFAULT_CONTRACT = (
    ROOT / "hw_autoresearch_nts07/contracts/"
    "m32_threshold_carry_input_contract_r2_20260822.json"
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
    if contract.get("schema") != "m32_threshold_carry_input_contract_v2":
        raise ValueError("unexpected M32 r2 input contract")
    paths = {}
    hashes = {}
    for name, spec in sorted(contract["inputs"].items()):
        path = resolve_path(spec["path"])
        if not path.is_file():
            raise ValueError("missing M32 r2 input {}: {}".format(name, path))
        actual = sha256(path)
        if actual != spec["sha256"]:
            raise ValueError(
                "M32 r2 input hash drift for {}: {} != {}".format(
                    name, actual, spec["sha256"]
                )
            )
        paths[name] = path
        hashes[name] = actual
    return contract, paths, hashes


def balanced_radix128_digits(value, bit_width, digit_count):
    minimum = -(1 << (int(bit_width) - 1))
    maximum = (1 << (int(bit_width) - 1)) - 1
    if int(value) < minimum or int(value) > maximum:
        raise ValueError("value outside signed{} range".format(bit_width))
    remaining = int(value)
    digits = []
    for _unused in range(int(digit_count)):
        digit = remaining % 128
        if digit >= 64:
            digit -= 128
        if digit < -128 or digit > 127:
            raise ValueError("balanced radix digit is not signed INT8")
        digits.append(digit)
        remaining = (remaining - digit) // 128
    if remaining != 0:
        raise ValueError("insufficient balanced radix digits")
    reconstructed = sum(
        digit * (128 ** index) for index, digit in enumerate(digits)
    )
    if reconstructed != int(value):
        raise ValueError("balanced radix reconstruction mismatch")
    return digits


def balanced_product_acc32_q24(accumulator, threshold_raw):
    acc_digits = balanced_radix128_digits(accumulator, 32, 5)
    threshold_digits = balanced_radix128_digits(threshold_raw, 24, 4)
    product = 0
    product_count = 0
    for acc_index, acc_digit in enumerate(acc_digits):
        for threshold_index, threshold_digit in enumerate(threshold_digits):
            product += (
                acc_digit * threshold_digit
                * (128 ** (acc_index + threshold_index))
            )
            product_count += 1
    if product_count != 20:
        raise ValueError("balanced Acc32-by-Q24 product count drift")
    return product, acc_digits, threshold_digits


def build_balanced_product_oracle():
    accumulator_edges = [
        -(1 << 31), -(1 << 31) + 1, -129, -1, 0, 1, 127, (1 << 31) - 1,
    ]
    threshold_edges = [-(1 << 23), -1, 0, 1, 127, 128, (1 << 23) - 1]
    cases = [(acc, threshold) for acc in accumulator_edges for threshold in threshold_edges]
    generator = random.Random(32024)
    for _unused in range(4096):
        cases.append((
            generator.randint(-(1 << 31), (1 << 31) - 1),
            generator.randint(-(1 << 23), (1 << 23) - 1),
        ))
    digest = hashlib.sha256()
    mismatches = 0
    maximum_abs_digit = 0
    for accumulator, threshold in cases:
        product, acc_digits, threshold_digits = balanced_product_acc32_q24(
            accumulator, threshold
        )
        reference = accumulator * threshold
        if product != reference:
            mismatches += 1
        maximum_abs_digit = max(
            [maximum_abs_digit]
            + [abs(value) for value in acc_digits]
            + [abs(value) for value in threshold_digits]
        )
        digest.update(
            ("{}:{}:{}:{}:{}\n".format(
                accumulator, threshold, reference, acc_digits, threshold_digits
            )).encode("ascii")
        )
    if mismatches:
        raise ValueError("balanced product oracle mismatch")
    return {
        "status": "PASS_SIGNED_INTEGER_PRODUCT_IDENTITY_SCHEDULE_PENDING",
        "radix": 128,
        "accumulator_bits": 32,
        "accumulator_digits": 5,
        "threshold_raw_bits": 24,
        "threshold_digits": 4,
        "signed_int8_products_per_output": 20,
        "cases": len(cases),
        "mismatches": mismatches,
        "maximum_abs_digit": maximum_abs_digit,
        "case_digest_sha256": digest.hexdigest(),
        "scope": (
            "proves exact signed integer multiplication only; threshold Q-format, "
            "rounding, saturation, bias, feed, recombination pipeline, and II remain open"
        ),
    }


def verify_threshold_manifest(manifest, candidate_rows, extractor_sha):
    if (
        manifest.get("schema") != "m32_h67_checkpoint_threshold_manifest_v1"
        or "PASS_FROZEN_SCALAR_THRESHOLDS" not in manifest.get("status", "")
        or manifest["checkpoint"]["sha256"]
        != "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158"
        or int(manifest["checkpoint"]["scalar_threshold_population"]) != 105
        or manifest["extractor"]["sha256"] != extractor_sha
        or not manifest["inference_profile"]["model_eval_token_present"]
        or not manifest["inference_profile"]["threshold_update_call_absent"]
    ):
        raise ValueError("M32 checkpoint threshold manifest identity drift")
    rows = {}
    for row in manifest["producers"]:
        producer = row["producer"]
        if producer in rows:
            raise ValueError("duplicate M32 threshold producer")
        rows[producer] = row
    expected = set(row["producer"] for row in candidate_rows)
    if set(rows) != expected:
        raise ValueError("M32 threshold producer population drift")
    for candidate in candidate_rows:
        row = rows[candidate["producer"]]
        if row["shape"] != [] or row["dtype"] != "torch.float32":
            raise ValueError("M32 checkpoint threshold is not scalar float32")
        value = float(row["value_float32"])
        if value <= 0.0:
            raise ValueError("M32 checkpoint threshold is not positive")
        if row["value_raw_le_hex"] != struct.pack("<f", value).hex():
            raise ValueError("M32 checkpoint threshold raw-byte drift")
        if value != float(candidate["observed_first_sample_threshold_amplitude"]):
            raise ValueError("M32 checkpoint/runtime threshold mismatch")
    return [rows[name] for name in sorted(rows)]


def audit_dual_trace_population(dual_rows, candidate_rows, continuous_rows, samples):
    names = [row["name"] for row in candidate_rows + continuous_rows]
    expected_samples = set(str(index) for index in range(int(samples)))
    for name in names:
        rows = [row for row in dual_rows if row["name"] == name]
        if len(rows) != int(samples):
            raise ValueError("M32 dual trace population drift for {}".format(name))
        if (
            set(row["status"] for row in rows) != {"NON_BINARY_BYPASS"}
            or set(row["temporal_step"] for row in rows) != {"-1"}
            or set(row["sample_id"] for row in rows) != expected_samples
            or set(row["operator_call_index"] for row in rows) != expected_samples
        ):
            raise ValueError("M32 dual trace identity drift for {}".format(name))
    return {
        "operators": len(names),
        "records": len(names) * int(samples),
        "samples": int(samples),
        "status": "PASS_FROZEN_NON_BINARY_BYPASS_POPULATION_IDENTITY",
    }


def ceil_div(numerator, denominator):
    return (int(numerator) + int(denominator) - 1) // int(denominator)


def build_control_charged_rows(r1_report, m25):
    event_cycles = int(r1_report["census"]["factorable_bypass_cycles"])
    outputs = int(r1_report["census"]["factorable_outputs_per_sample"])
    fixed = int(r1_report["cycle_sensitivity"]["fixed_compute_cycles"])
    arithmetic_rows = {
        (row["line"], row["variant"]): row
        for row in r1_report["cycle_sensitivity"]["rows"]
    }
    variants = [
        ("byte12_arithmetic_lower_bound", "byte12_arithmetic_lower_bound", 12,
         "UNSIGNED_LIMB_AND_RECOMBINATION_LOWER_BOUND_ONLY"),
        ("balanced_radix20_exact_product", "radix24_provisional", 20,
         "SIGNED_PRODUCT_IDENTITY_PROVEN_FEED_RECOMBINATION_AND_II_PENDING"),
        ("stress48", "stress48", 48,
         "STRESS_ENVELOPE_NOT_AN_IMPLEMENTED_SCHEDULE"),
    ]
    line_contract = {
        "local": m25["compute_envelopes"]["local"]["10"],
        "motion": m25["compute_envelopes"]["hybrid"]["10"],
    }
    rows = []
    for line in ("local", "motion"):
        anchor = line_contract[line]
        base_population = int(anchor["m4_profiled_eligible_cycles"])
        base_increment = int(anchor["m21_fifo4_phase1_incremental_cycles"])
        control_cycles = int(math.ceil(
            event_cycles * base_increment / float(base_population)
        ))
        for variant, r1_variant, products, status in variants:
            source = arithmetic_rows[(line, r1_variant)]
            outputs_per_cycle = 96 // products
            late_cycles = ceil_div(outputs, outputs_per_cycle)
            if late_cycles != int(source["late_scale_cycles"]):
                raise ValueError("M32 r2 late-scale mapping drift")
            arithmetic_cycles = int(source["proposal_compute_cycles_sensitivity"])
            proposal_cycles = arithmetic_cycles + control_cycles
            rows.append({
                "line": line,
                "variant": variant,
                "status": status,
                "signed_int8_product_slots": 96,
                "products_per_output": products,
                "outputs_per_cycle_floor": outputs_per_cycle,
                "event_accumulation_cycles_borrowed": int(
                    source["borrowed_event_accumulation_cycles"]
                ),
                "late_scale_cycles_arithmetic": late_cycles,
                "arithmetic_only_proposal_cycles": arithmetic_cycles,
                "frontend_control_anchor_population_cycles": base_population,
                "frontend_control_anchor_increment_cycles": base_increment,
                "proportional_frontend_control_cycles": control_cycles,
                "control_charged_proposal_cycles_sensitivity": proposal_cycles,
                "control_charged_speedup_vs_fixed_sensitivity": (
                    fixed / float(proposal_cycles)
                ),
                "crosses_2p5x_sensitivity": proposal_cycles * 2.5 < fixed,
                "crosses_2p75x_sensitivity": proposal_cycles * 2.75 < fixed,
                "crosses_3x_sensitivity": proposal_cycles * 3 < fixed,
                "cycles_margin_to_3x": fixed / 3.0 - proposal_cycles,
            })
    return rows


def build_report(contract_path=DEFAULT_CONTRACT):
    contract, paths, hashes = load_contract(contract_path)
    r1_file = json.loads(paths["r1_report"].read_text(encoding="utf-8"))
    r1_built = R1.build_report(paths["r1_contract"])
    if r1_file != r1_built:
        raise ValueError("M32 r1 report is not reproducible from frozen inputs")
    _base_contract, base_paths, _base_hashes = R1.load_and_verify_inputs(
        paths["r1_contract"]
    )
    threshold_manifest = json.loads(
        paths["threshold_manifest"].read_text(encoding="utf-8")
    )
    threshold_rows = verify_threshold_manifest(
        threshold_manifest,
        r1_file["census"]["factorable"],
        hashes["threshold_extractor"],
    )
    dual_audit = audit_dual_trace_population(
        R1.read_csv(base_paths["dual_line_operator_trace"]),
        r1_file["census"]["factorable"],
        r1_file["census"]["continuous_preserved"],
        int(r1_file["identity"]["samples"]),
    )
    m25 = json.loads(base_paths["m25_cycle_model"].read_text(encoding="utf-8"))
    product_oracle = build_balanced_product_oracle()
    sensitivity = build_control_charged_rows(r1_file, m25)

    candidates = []
    threshold_by_producer = {row["producer"]: row for row in threshold_rows}
    for row in r1_file["census"]["factorable"]:
        candidate = dict(row)
        candidate["candidate_status"] = (
            "CANDIDATE_SCALAR_FACTORABLE_DATAFLOW_DIGEST_PENDING"
        )
        candidate["semantic_admission"] = False
        candidate["checkpoint_threshold"] = threshold_by_producer[row["producer"]]
        candidate.pop("semantic_status", None)
        candidates.append(candidate)

    return {
        "schema": "m32_threshold_carry_late_scale_audit_v2",
        "status": (
            "PASS_CANDIDATE_CENSUS_THRESHOLD_MANIFEST_SIGNED_PRODUCT_ORACLE_"
            "CONTROL_CHARGED_SENSITIVITY_NO_SEMANTIC_OR_HEADLINE_ADMISSION"
        ),
        "identity": {
            "input_contract": str(Path(contract_path).resolve()),
            "input_contract_sha256": sha256(contract_path),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
            "verified_supplemental_sha256": hashes,
            "base_report_sha256": hashes["r1_report"],
            "checkpoint_sha256": r1_file["identity"]["checkpoint_sha256"],
            "samples": int(r1_file["identity"]["samples"]),
        },
        "candidate_census": {
            "candidate_factorable_operators": len(candidates),
            "candidate_factorable_cycles": int(
                r1_file["census"]["factorable_bypass_cycles"]
            ),
            "candidate_factorable_outputs_per_sample": int(
                r1_file["census"]["factorable_outputs_per_sample"]
            ),
            "continuous_preserved_operators": int(
                r1_file["census"]["continuous_bypass_operators"]
            ),
            "continuous_preserved_cycles": int(
                r1_file["census"]["continuous_bypass_cycles"]
            ),
            "candidates": candidates,
            "continuous_preserved": r1_file["census"]["continuous_preserved"],
        },
        "checkpoint_threshold_audit": {
            "manifest_sha256": hashes["threshold_manifest"],
            "producer_thresholds": threshold_rows,
            "inference_profile": threshold_manifest["inference_profile"],
            "status": "PASS_TEN_SCALAR_FLOAT32_VALUES_FROZEN_NOT_QFORMAT_ADMITTED",
        },
        "dual_trace_crosscheck": dual_audit,
        "signed_product_oracle": product_oracle,
        "control_charged_cycle_sensitivity": {
            "fixed_compute_cycles": int(
                r1_file["cycle_sensitivity"]["fixed_compute_cycles"]
            ),
            "rows": sensitivity,
            "unmodeled_nonzero_costs": [
                "producer-output versus consumer-prehook value/dtype/raw-byte digest",
                "new-operator row/source trace and exact M21 phase/FIFO schedule",
                "balanced-radix digit conversion and operand-feed bandwidth",
                "wide shift/recombination pipeline startup, tail, and physical area",
                "Q-format conversion, RNE, saturation, bias stage, and SRAM traffic",
            ],
            "interpretation": (
                "optimistic compute sensitivity after proportional M21 frontend/control "
                "charge; not executable cycles and not measured system performance"
            ),
        },
        "pending_dataflow_identity_contract": {
            "available": False,
            "required_per_call_fields": [
                "sample_id", "producer", "consumer", "producer_call_index",
                "consumer_call_index", "dtype", "shape", "numel",
                "producer_raw_value_sha256", "consumer_raw_value_sha256",
                "same_storage_pointer", "same_value_digest",
            ],
            "admission_rule": (
                "all ten samples and every candidate call must match dtype, shape, "
                "numel, raw-value digest, and tensor storage identity"
            ),
        },
        "claim_boundary": {
            "permitted": [
                "ten candidate consumers with frozen scalar checkpoint thresholds",
                "exact signed Acc32-by-signed-Q24 integer product decomposition identity",
                "control-charged optimistic compute sensitivity",
            ],
            "forbidden": [
                "calling any candidate semantically admitted before dataflow digests",
                "using near-one threshold values as an approximation argument",
                "claiming threshold Q-format, rounding, saturation, or bias bit exactness",
                "claiming the balanced-radix product oracle is an executable pipeline",
                "claiming any 2.5x, 2.75x, or 3x row as measured performance",
                "claiming PPA, energy, FPS, DRAMsim3 timing, or DATE comparison",
            ],
        },
        "semantic_admission": False,
        "headline_admitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite M32 r2 report: {}".format(args.output))
    report = build_report(args.contract.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
