#!/usr/bin/env python3
"""Independent manifest-level audit of M370 bottleneck magnitude gating."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
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

    def reject(token):
        raise RuntimeError("non-standard JSON number: " + token)

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=reject)


def float32_from_hex(word):
    require(isinstance(word, str) and len(word) == 8,
            "invalid float32 bit word")
    return struct.unpack(">f", bytes.fromhex(word))[0]


def next_float32_bits(word):
    value = int(word, 16)
    require(0 < value < 0x7f800000, "expected finite positive float32")
    return "{:08x}".format(value + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M371 output overwrite")

    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m371_m370_independent_codebook_hammer_contract_v1",
            "M371 contract schema drift")
    require(contract.get("status") == "FROZEN_BEFORE_M371_EXECUTION",
            "M371 contract is not frozen")
    root = args.contract.resolve().parents[1]
    theta_grid = contract["mechanism"]["theta_grid"]
    require(theta_grid == [0.0, 0.015625, 0.03125, 0.0625, 0.125],
            "theta grid drift")

    identities = {}
    for name, identity in contract["read_only_identities"].items():
        path = root / identity["path"]
        require(path.is_file(), "missing frozen input: " + str(path))
        observed = sha256(path)
        require(observed == identity["sha256"], "SHA drift: " + name)
        identities[name] = {"path": identity["path"], "sha256": observed}

    record_ledger = []
    cohort_summaries = []
    global_operator_bits = {}
    global_operator_active = {}
    total_records = 0
    total_values = 0
    total_active = 0
    total_grid_dropped = 0

    for cohort, spec in contract["manifest_inputs"].items():
        path = root / spec["path"]
        manifest = strict_json(path)
        records = manifest.get("records")
        require(isinstance(records, list), "records missing: " + cohort)
        require(len(records) == spec["expected_records"],
                "record population drift: " + cohort)
        seen_record_ids = set()
        cohort_operator = {}
        cohort_values = 0
        cohort_active = 0

        for record_index, record in enumerate(records):
            record_id = (record.get("sample_id"), record.get("operator_index"))
            require(record_id not in seen_record_ids,
                    "duplicate sample/operator record: " + cohort)
            seen_record_ids.add(record_id)
            operator = record.get("operator")
            require(operator in contract["expected_operators"],
                    "unexpected operator: " + str(operator))
            population = record["value_bit_pattern_population"]
            require(population.get("full_codebook_in_manifest") is True,
                    "codebook is not complete")
            require(population.get("unique_float32_bit_patterns") == 2,
                    "unique bit-pattern count is not two")
            codebook = population.get("codebook")
            require(isinstance(codebook, list) and len(codebook) == 2,
                    "codebook is not exactly two entries")
            bit_rows = {}
            for row in codebook:
                bits = row.get("float32_bits_hex")
                count = row.get("count")
                require(bits not in bit_rows, "duplicate codebook bits")
                require(isinstance(count, int) and count > 0,
                        "non-positive codebook count")
                bit_rows[bits] = count
            require("00000000" in bit_rows and len(bit_rows) == 2,
                    "record codebook is not strict {+0,a}")
            active_bits = next(bits for bits in bit_rows if bits != "00000000")
            active_amplitude = float32_from_hex(active_bits)
            require(math.isfinite(active_amplitude) and active_amplitude > 0.0,
                    "active amplitude is not finite positive")
            zero_count = bit_rows["00000000"]
            active_count = bit_rows[active_bits]
            elements = record.get("elements")
            require(zero_count + active_count == elements,
                    "codebook count does not equal elements")
            require(record.get("nonzero_count") == active_count,
                    "nonzero_count disagrees with active codebook")
            require(record.get("positive_count") == active_count and
                    record.get("negative_count") == 0,
                    "sign counts disagree with strict {0,+a}")
            value_audit = record.get("value_audit", {})
            require(value_audit.get("minimum") == 0.0,
                    "value-audit minimum is not zero")
            require(value_audit.get("maximum") == active_amplitude,
                    "value-audit maximum disagrees with active amplitude")

            operator_row = cohort_operator.setdefault(operator, {
                "active_bits_hex": active_bits,
                "active_amplitude": active_amplitude,
                "records": 0,
                "values": 0,
                "zero_sources": 0,
                "active_sources": 0,
                "theta_dropped_active_sources": {str(theta): 0
                                                   for theta in theta_grid},
            })
            require(operator_row["active_bits_hex"] == active_bits,
                    "within-cohort operator amplitude drift")
            global_operator_bits.setdefault(operator, set()).add(active_bits)
            global_operator_active[operator] = (
                global_operator_active.get(operator, 0) + active_count)

            theta_drops = []
            for theta in theta_grid:
                dropped = active_count if active_amplitude < theta else 0
                theta_drops.append(dropped)
                operator_row["theta_dropped_active_sources"][str(theta)] += dropped
                total_grid_dropped += dropped
            operator_row["records"] += 1
            operator_row["values"] += elements
            operator_row["zero_sources"] += zero_count
            operator_row["active_sources"] += active_count
            cohort_values += elements
            cohort_active += active_count
            record_ledger.append({
                "cohort": cohort,
                "record_index": record_index,
                "sample_id": record.get("sample_id"),
                "sample_key": record.get("sample_key"),
                "operator_index": record.get("operator_index"),
                "operator": operator,
                "elements": elements,
                "zero_count": zero_count,
                "active_count": active_count,
                "active_bits_hex": active_bits,
                "active_amplitude": active_amplitude,
                "theta_0_drop": theta_drops[0],
                "theta_1_64_drop": theta_drops[1],
                "theta_1_32_drop": theta_drops[2],
                "theta_1_16_drop": theta_drops[3],
                "theta_1_8_drop": theta_drops[4],
            })

        require(set(cohort_operator) == set(contract["expected_operators"]),
                "cohort does not contain exactly four operators")
        for row in cohort_operator.values():
            require(row["records"] == spec["expected_records_per_operator"],
                    "operator record count drift")
        cohort_summaries.append({
            "cohort": cohort,
            "records": len(records),
            "values": cohort_values,
            "active_sources": cohort_active,
            "operators": cohort_operator,
        })
        total_records += len(records)
        total_values += cohort_values
        total_active += cohort_active

    require(total_records == 248, "total record count is not 248")
    require(all(len(bits) == 1 for bits in global_operator_bits.values()),
            "per-operator amplitude differs across cohorts")
    require(total_grid_dropped == 0,
            "frozen theta grid unexpectedly drops an active source")

    cliff_rows = []
    for operator in contract["expected_operators"]:
        active_bits = next(iter(global_operator_bits[operator]))
        active_amplitude = float32_from_hex(active_bits)
        next_bits = next_float32_bits(active_bits)
        next_theta = float32_from_hex(next_bits)
        require(not (active_amplitude < active_amplitude),
                "strict threshold unexpectedly drops at theta=a")
        require(active_amplitude < next_theta,
                "next float32 threshold does not cross amplitude")
        cliff_rows.append({
            "operator": operator,
            "active_bits_hex": active_bits,
            "active_amplitude": active_amplitude,
            "theta_equal_a_drop_fraction_of_active": 0.0,
            "next_float32_theta_bits_hex": next_bits,
            "next_float32_theta": next_theta,
            "next_float32_theta_dropped_active_sources":
                global_operator_active[operator],
            "next_float32_theta_drop_fraction_of_active": 1.0,
        })

    args.output_dir.mkdir(parents=True, exist_ok=False)
    ledger_path = args.output_dir / "m371_all_248_record_codebook_audit.csv"
    with ledger_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record_ledger[0]))
        writer.writeheader()
        writer.writerows(record_ledger)

    result = {
        "schema": "m371_m370_independent_codebook_hammer_v1",
        "status": "PASS_M371_INDEPENDENT_MANIFEST_CODEBOOK_AUDIT",
        "identity": identities,
        "independence": {
            "m370_result_used_for_computation": False,
            "m370_analyzer_imported_or_called": False,
            "all_manifest_records_reparsed": True,
            "all_codebook_counts_recomputed": True,
        },
        "totals": {
            "cohorts": 4,
            "records": total_records,
            "values": total_values,
            "active_sources": total_active,
            "grid_dropped_active_sources": total_grid_dropped,
        },
        "cohorts": cohort_summaries,
        "operator_cliffs": cliff_rows,
        "decision": {
            "g7_bottleneck_scalar_magnitude_gate":
                "FAST_KILL_NO_NONTRIVIAL_PARTIAL_DROP_POINT",
            "sufficient_reason":
                "For strict drop iff abs(x)<theta and exact {0,a_l} inputs, "
                "theta<=a_l drops no active source and theta>a_l drops every "
                "active source in that layer. The frozen grid is below every "
                "a_l, so it adds zero sparsity.",
            "accuracy_collapse_proven": False,
            "rtl_or_a800_job_warranted_for_this_g7": False,
            "g11_weight_product_budget_negated": False,
            "fc_patch_attention_inference_supported": False,
        },
        "claim_boundary": {
            "four_bottleneck_conv_inputs_only": True,
            "manifest_codebook_audit": True,
            "raw_payload_reconstruction": False,
            "accuracy": False,
            "cycle_speedup": False,
            "rtl": False,
            "system_speedup": False,
            "date_headline": False,
        },
    }
    result_path = args.output_dir / (
        "m371_m370_independent_codebook_hammer_r1.json")
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("M371_PASS records={} values={} active={} grid_drop={} operators=4".
          format(total_records, total_values, total_active,
                 total_grid_dropped), flush=True)


if __name__ == "__main__":
    main()
