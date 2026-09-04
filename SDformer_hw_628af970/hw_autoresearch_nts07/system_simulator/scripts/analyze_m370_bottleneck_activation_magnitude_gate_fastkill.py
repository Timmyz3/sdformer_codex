#!/usr/bin/env python3
"""Fast-kill G7 magnitude gating from frozen source-value codebooks."""

from __future__ import division

import argparse
import hashlib
import json
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


def float_from_hex(word):
    return struct.unpack(">f", bytes.fromhex(word))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M370 output overwrite")
    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m370_bottleneck_activation_magnitude_gate_fastkill_contract_v1",
            "M370 contract schema drift")
    require(contract.get("status") == "FROZEN_BEFORE_M370_EXECUTION",
            "M370 contract not frozen")
    root = args.contract.resolve().parents[1]
    identities = {}
    manifests = {}
    for name, identity in contract["inputs"].items():
        path = root / identity["path"]
        require(path.is_file(), "missing input: " + str(path))
        observed = sha256(path)
        require(observed == identity["sha256"], "SHA drift for " + name)
        identities[name] = {"path": identity["path"], "sha256": observed}
        if name not in ("analyzer", "docs359"):
            manifests[name] = strict_json(path)

    theta_grid = contract["mechanism"]["theta_grid"]
    require(theta_grid == [0.0, 0.015625, 0.03125, 0.0625, 0.125],
            "theta grid drift")
    cohort_rows = []
    operator_amplitudes = {}
    for cohort, manifest in manifests.items():
        require(len(manifest["records"]) in (40, 128),
                "unexpected record population for " + cohort)
        operator_rows = {}
        total_nonzero = 0
        for record in manifest["records"]:
            codebook = record["value_bit_pattern_population"]["codebook"]
            require(len(codebook) == 2, "non-binary-amplitude codebook")
            zero = [row for row in codebook
                    if row["float32_bits_hex"] == "00000000"]
            active = [row for row in codebook
                      if row["float32_bits_hex"] != "00000000"]
            require(len(zero) == 1 and len(active) == 1,
                    "codebook is not exactly {0,a}")
            amplitude = float_from_hex(active[0]["float32_bits_hex"])
            require(amplitude > 0.0, "active amplitude must be positive")
            operator = record["operator"]
            if operator in operator_rows:
                require(operator_rows[operator]["active_bits_hex"] ==
                        active[0]["float32_bits_hex"],
                        "per-operator amplitude drift")
            else:
                operator_rows[operator] = {
                    "active_bits_hex": active[0]["float32_bits_hex"],
                    "active_amplitude": amplitude,
                    "active_sources": 0,
                    "total_sources": 0,
                }
            operator_rows[operator]["active_sources"] += active[0]["count"]
            operator_rows[operator]["total_sources"] += sum(
                row["count"] for row in codebook)
            total_nonzero += active[0]["count"]
        require(len(operator_rows) == 4, "expected four bottleneck operators")
        for operator, row in operator_rows.items():
            operator_amplitudes.setdefault(operator, set()).add(
                row["active_bits_hex"])
            row["theta_points"] = [{
                "theta": theta,
                "dropped_active_sources": (row["active_sources"]
                                             if row["active_amplitude"] < theta
                                             else 0),
                "incremental_drop_fraction_of_active":
                    (1.0 if row["active_amplitude"] < theta else 0.0),
            } for theta in theta_grid]
        cohort_rows.append({
            "cohort": cohort,
            "records": len(manifest["records"]),
            "active_sources": total_nonzero,
            "operators": operator_rows,
        })

    require(all(len(bits) == 1 for bits in operator_amplitudes.values()),
            "operator amplitude differs between identities")
    minimum = min(float_from_hex(next(iter(bits)))
                  for bits in operator_amplitudes.values())
    grid_drops = sum(
        point["dropped_active_sources"]
        for cohort in cohort_rows
        for row in cohort["operators"].values()
        for point in row["theta_points"])
    require(grid_drops == 0, "frozen grid unexpectedly drops active sources")
    payload = {
        "schema": "m370_bottleneck_activation_magnitude_gate_fastkill_v1",
        "status": "PASS_M370_G7_BOTTLENECK_MAGNITUDE_GRID_ZERO_GAIN_FASTKILL",
        "identity": identities,
        "observation": {
            "all_records_exact_zero_plus_one_fixed_amplitude": True,
            "per_operator_amplitude_stable_across_checkpoint_identities": True,
            "minimum_nonzero_amplitude": minimum,
            "maximum_tested_theta": max(theta_grid),
            "grid_dropped_active_sources": grid_drops,
            "intermediate_drop_fraction_observed": False,
            "transition_above_operator_amplitude":
                "0% active dropped to 100% active dropped",
        },
        "cohorts": cohort_rows,
        "decision": {
            "g7_bottleneck_conv": "NO_GO_RTL_AND_A800_ACCURACY",
            "reason": "The proposed grid is identically lossless and yields zero new sparsity; crossing a producer amplitude drops every active source for that operator, so no measured intermediate magnitude Pareto exists.",
            "g11_weight_product_budget_unaffected": True,
            "other_operator_classes_require_separate_traces": True,
        },
        "admission": {
            "trace_codebook_audit": True,
            "accuracy": False,
            "cycle_speedup": False,
            "rtl": False,
            "system_speedup": False,
            "date_headline": False,
        },
        "claim_boundary": "M370 fast-kills only a source-amplitude threshold on the four frozen bottleneck Conv inputs. It does not assess weight-product budgeting, FC, patch embed, attention, accuracy or cycles.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / (
        "m370_bottleneck_activation_magnitude_gate_fastkill_r1.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("M370_PASS grid_drop=0 min_active={:.9f} G7_bottleneck=NO_GO".
          format(minimum), flush=True)


if __name__ == "__main__":
    main()
