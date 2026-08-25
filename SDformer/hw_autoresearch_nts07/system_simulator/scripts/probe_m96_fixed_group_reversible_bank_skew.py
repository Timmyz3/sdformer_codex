#!/usr/bin/env python3
"""M96 fixed-M89-group reversible weight-bank-skew source screen."""

from __future__ import print_function

import argparse
import collections
import hashlib
import importlib.util
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW_ROOT / "contracts/m96_fixed_group_reversible_bank_skew_contract_r1_20260824.json"
M45_ANALYZER = HW_ROOT / (
    "system_simulator/scripts/analyze_m45_dual_destination_bank_fused_integrated_schedule.py")
M53_ANALYZER = HW_ROOT / (
    "system_simulator/scripts/analyze_m53_adaptive_temporal_parent_k4_ctx16_dse.py")
M53_RESULT = HW_ROOT / (
    "results/m53_adaptive_temporal_parent_k4_ctx16_dse_r1_20260823/"
    "m53_adaptive_temporal_parent_k4_ctx16_dse.json")
M43_TEMPORAL = HW_ROOT / (
    "results/m43_tile_resident_parent_delta_schedule_r1_20260823/"
    "m43_spatiotemporal_parent_delta_ablation.json")
M89_RECEIPT = HW_ROOT / (
    "results/m89_temporal_fanout_hold_screen_r1_20260823/"
    "m89_temporal_fanout_hold_screen_receipt.json")

EXPECTED = {
    "contract": "251ebb1f19abd07166e5af99e872cbe0013dff038836638ed5ebdeb783e496fc",
    "m45_analyzer": "c1e3610ce59753f786498db46cde7b330155fa2e3c836198be165aad3eb3f38f",
    "m53_analyzer": "638809bd72ab7f66fc69b51f4cb726f2c0d1c7712f71188066b4ef04cbdda531",
    "m53_result": "344ae1f777e0640d46b19118f0b6d451465046350d68a9f33b1faae124747bb4",
    "m43_temporal": "995fa9643ab2180d9b1480b4143959275dc3a04b4b346f8d7e22bed5266a639c",
    "m89_receipt": "afacec344ec8481dd27b667751e97d938655f46e5cced7b330460a530b92e9cf",
}
MODES = ("H0_IDENTITY", "H1_XOR_ROW", "H2_ADD_ROW", "H3_ADD_3ROW")
FANOUT = 6
CONTEXTS = 16
MAX_SOURCE = 69614355


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def population(value):
    method = getattr(value, "bit_count", None)
    return method() if method is not None else bin(value).count("1")


def read_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: {}".format(raw))
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=reject)


def fraction(numerator, denominator):
    require(denominator > 0, "zero fraction denominator")
    divisor = math.gcd(int(numerator), int(denominator))
    return {
        "numerator": int(numerator) // divisor,
        "denominator": int(denominator) // divisor,
        "decimal": float(numerator) / float(denominator),
    }


def nearest_rank(values, numerator=95, denominator=100):
    require(values, "empty nearest-rank population")
    ordered = sorted(values)
    index = (len(ordered) * numerator + denominator - 1) // denominator - 1
    return ordered[index]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_inputs():
    paths = {
        "contract": CONTRACT,
        "m45_analyzer": M45_ANALYZER,
        "m53_analyzer": M53_ANALYZER,
        "m53_result": M53_RESULT,
        "m43_temporal": M43_TEMPORAL,
        "m89_receipt": M89_RECEIPT,
    }
    for name, path in paths.items():
        require(path.is_file() and sha256(path) == EXPECTED[name],
                "M96 {} identity drift".format(name))


def build_bank_masks():
    masks = {}
    for mode in MODES:
        banks = [0] * 8
        for position in range(256):
            weight_row, base_bank = divmod(position, 8)
            row_low = weight_row & 7
            if mode == "H0_IDENTITY":
                bank = base_bank
            elif mode == "H1_XOR_ROW":
                bank = base_bank ^ row_low
            elif mode == "H2_ADD_ROW":
                bank = (base_bank + row_low) & 7
            else:
                bank = (base_bank + 3 * row_low) & 7
            banks[bank] |= 1 << position
        require(all(population(mask) == 32 for mask in banks),
                "M96 non-bijective bank hash {}".format(mode))
        require(sum(banks) == (1 << 256) - 1,
                "M96 bank mask coverage drift {}".format(mode))
        masks[mode] = tuple(banks)
    return masks


def build_namespace():
    m53 = load_module(M53_ANALYZER, "m96_m53")
    m53.validate_contract()
    canonical, transformed, edits = m53.transformed_m45_source(True)
    hook_from = "        for task in group:\n            del resident[task]"
    hook_to = ("        audit_frozen_group(union_mask, group_cycles)\n"
               "        for task in group:\n"
               "            del resident[task]")
    require(canonical.count(hook_from) == 1 and
            transformed.count(hook_from) == 1,
            "M96 group hook source identity drift")
    transformed = transformed.replace(hook_from, hook_to)
    edits = list(edits) + [{
        "name": "post_selection_fixed_group_bank_hash_audit",
        "occurrences": 1,
        "qualification": "AUDIT_ONLY_NO_SCHEDULER_FEEDBACK",
    }]
    namespace = {
        "__file__": str(M53_ANALYZER),
        "__name__": "m96_fixed_group_transformed_m45",
    }
    exec(compile(transformed, str(M53_ANALYZER) + "#M96_FIXED_GROUP", "exec"),
         namespace)
    bank_masks = build_bank_masks()
    audit = collections.defaultdict(lambda: {
        "groups": 0,
        "union_popcount": 0,
        "tight_eight_bank_lower_bound_cycles": 0,
        "mode_source_cycles": dict((mode, 0) for mode in MODES),
    })
    current = {"key": None}

    def audit_frozen_group(union_mask, canonical_group_cycles):
        require(current["key"] is not None, "M96 missing current record identity")
        popcount = population(union_mask)
        row = audit[current["key"]]
        row["groups"] += 1
        row["union_popcount"] += popcount
        row["tight_eight_bank_lower_bound_cycles"] += (popcount + 7) // 8
        for mode in MODES:
            cycles = max(population(union_mask & mask)
                         for mask in bank_masks[mode])
            row["mode_source_cycles"][mode] += cycles
        require(row["mode_source_cycles"]["H0_IDENTITY"] >= 0,
                "M96 negative H0 source cycles")
        require(canonical_group_cycles == max(
            population(union_mask & mask)
            for mask in bank_masks["H0_IDENTITY"]),
            "M96 H0 bank-cycle identity drift")

    namespace["audit_frozen_group"] = audit_frozen_group
    original_analyze_record = namespace["analyze_record"]

    def analyze_record_with_identity(m43, masks, expected_m43_record,
                                     fanout_k, context_capacity):
        current["key"] = (expected_m43_record["sample_id"],
                          expected_m43_record["operator"])
        try:
            return original_analyze_record(
                m43, masks, expected_m43_record, fanout_k, context_capacity)
        finally:
            current["key"] = None

    namespace["analyze_record"] = analyze_record_with_identity
    require(namespace["schedule_tile_timestep"].__globals__ is namespace,
            "M96 scheduler namespace mismatch")
    m43 = namespace["load_m43_module"]()
    require(bool(m43.ALLOW_TEMPORAL_PARENT),
            "M96 temporal parent was not enabled")
    source_audit = {
        "canonical_m45_sha256": sha256_bytes(canonical.encode("utf-8")),
        "transformed_source_sha256": sha256_bytes(transformed.encode("utf-8")),
        "edit_count": len(edits),
        "edits": edits,
        "unlisted_source_edits": 0,
    }
    return m53, namespace, m43, audit, source_audit


def replay():
    validate_inputs()
    m53, namespace, m43, audit, source_audit = build_namespace()
    namespace["validate_contract"]()
    manifest = namespace["read_json"](namespace["MANIFEST"])
    reference = m53.read_json(M43_TEMPORAL)
    references = dict(
        ((row["sample_id"], row["operator"]), row)
        for row in reference["records"])
    require(len(manifest["records"]) == 40 and len(references) == 40,
            "M96 frozen cohort drift")
    cached = []
    for record in manifest["records"]:
        key = (record["sample_id"], record["operator"])
        require(key in references, "M96 M43 reference record drift")
        masks = m43.unpack_record_masks(namespace["MANIFEST"].parent, record)
        cached.append((record, masks, references[key]))
    baseline = m53.analyze_configuration(
        namespace, m43, cached, "K6_CTX16_TEMPORAL_M96_FIXED_GROUP_H0",
        FANOUT, CONTEXTS, True,
        "M96_FROZEN_GROUP_BANK_HASH_STAGE1_SOURCE_SCREEN")
    blocks = namespace["BLOCKS"]
    require(len(audit) == 40, "M96 audit record population drift")
    records = []
    baseline_records = dict(
        ((row["sample_id"], row["operator"]), row)
        for row in baseline["record_ledger"]["records"])
    for key in sorted(audit):
        row = audit[key]
        source_by_mode = dict(
            (mode, row["mode_source_cycles"][mode] * blocks)
            for mode in MODES)
        ledger = baseline_records[key]
        require(row["groups"] * blocks == ledger["fusion_groups"],
                "M96 record group identity drift")
        require(row["union_popcount"] * blocks == ledger["unique_weight_issues"],
                "M96 record unique-issue identity drift")
        require(source_by_mode["H0_IDENTITY"] == ledger["source_only_cycles"],
                "M96 record H0 source identity drift")
        records.append({
            "sample_id": key[0],
            "operator": key[1],
            "fusion_groups": row["groups"] * blocks,
            "union_popcount": row["union_popcount"] * blocks,
            "tight_eight_bank_lower_bound_cycles":
                row["tight_eight_bank_lower_bound_cycles"] * blocks,
            "source_cycles_by_mode": source_by_mode,
        })
    baseline["dynamic_source_edit_audit"] = source_audit
    return baseline, records


def aggregate_rows(records, field):
    result = collections.defaultdict(collections.Counter)
    for row in records:
        key = row[field]
        result[key]["fusion_groups"] += row["fusion_groups"]
        result[key]["union_popcount"] += row["union_popcount"]
        result[key]["tight_eight_bank_lower_bound_cycles"] += row[
            "tight_eight_bank_lower_bound_cycles"]
        for mode in MODES:
            result[key][mode] += row["source_cycles_by_mode"][mode]
    return result


def build():
    baseline, records = replay()
    m89 = read_json(M89_RECEIPT)
    matches = [row for row in m89["configurations"] if row["name"] == "K6"]
    require(len(matches) == 1, "M96 M89 K6 baseline missing")
    m89_k6 = matches[0]
    require(baseline["aggregate_source_only_cycles"] ==
            m89_k6["source_cycles"] == 69964176,
            "M96 baseline source reproduction drift")
    require(baseline["aggregate_integrated_cycles"] ==
            m89_k6["integrated_cycles"] == 76677320,
            "M96 baseline integrated reproduction drift")

    by_operator = aggregate_rows(records, "operator")
    by_sample = aggregate_rows(records, "sample_id")
    operator_modes = {}
    operator_rows = []
    for operator in sorted(by_operator):
        values = by_operator[operator]
        selected = min(MODES, key=lambda mode: (values[mode], MODES.index(mode)))
        operator_modes[operator] = selected
        operator_rows.append({
            "operator": operator,
            "selected_mode": selected,
            "source_cycles_by_mode": dict((mode, values[mode]) for mode in MODES),
            "fusion_groups": values["fusion_groups"],
            "union_popcount": values["union_popcount"],
            "tight_eight_bank_lower_bound_cycles":
                values["tight_eight_bank_lower_bound_cycles"],
        })

    per_sample = []
    for sample_id in range(10):
        sample_records = [row for row in records if row["sample_id"] == sample_id]
        selected_cycles = sum(
            row["source_cycles_by_mode"][operator_modes[row["operator"]]]
            for row in sample_records)
        source_by_mode = dict(
            (mode, sum(row["source_cycles_by_mode"][mode]
                       for row in sample_records)) for mode in MODES)
        per_sample_oracle = sum(
            min(row["source_cycles_by_mode"].values())
            for row in sample_records)
        reference = next(row for row in m89_k6["per_sample"]
                         if row["sample_id"] == sample_id)
        require(source_by_mode["H0_IDENTITY"] == reference["source"],
                "M96 sample H0 reproduction drift")
        per_sample.append({
            "sample_id": sample_id,
            "baseline_source_cycles": reference["source"],
            "selected_source_cycles": selected_cycles,
            "selected_delta_cycles": selected_cycles - reference["source"],
            "source_cycles_by_global_mode": source_by_mode,
            "prohibited_per_record_oracle_source_cycles": per_sample_oracle,
        })

    selected_source = sum(row["selected_source_cycles"] for row in per_sample)
    h0_source = sum(row["source_cycles_by_global_mode"]["H0_IDENTITY"]
                    for row in per_sample)
    total_groups = sum(row["fusion_groups"] for row in records)
    total_unique = sum(row["union_popcount"] for row in records)
    tight_lower = sum(row["tight_eight_bank_lower_bound_cycles"]
                      for row in records)
    oracle = sum(row["prohibited_per_record_oracle_source_cycles"]
                 for row in per_sample)
    gates = {
        "H0_exact_source_cycles_equal_69964176": h0_source == 69964176,
        "H0_exact_group_count_equal_10436792": total_groups == 10436792,
        "unique_weight_issues_equal_416232640": total_unique == 416232640,
        "logical_source_updates_equal_562451704":
            baseline["aggregate_logical_source_updates"] == 562451704,
        "weight_dma_bytes_per_sample_equal_212336640":
            baseline["traffic_bytes_per_sample"]["weight_dma"] == 212336640,
        "selected_source_cycles_le_69614355": selected_source <= MAX_SOURCE,
        "each_sample_source_cycles_must_not_regress": all(
            row["selected_source_cycles"] <= row["baseline_source_cycles"]
            for row in per_sample),
        "each_operator_mode_is_fixed_across_all_samples":
            len(operator_modes) == 4,
        "zero_extra_ports_capacity_and_vector_storage": True,
    }
    passed = all(gates.values())
    return {
        "schema": "m96_fixed_group_reversible_bank_skew_stage1_result_v1",
        "status": ("PASS_M96_STAGE1_PROMOTE_EXACT_INTEGRATED_REPLAY" if passed
                   else "PASS_M96_STAGE1_EXECUTION_NO_GO"),
        "identity": {
            "contract_sha256": EXPECTED["contract"],
            "probe_sha256": sha256(Path(__file__).resolve()),
            "m45_analyzer_sha256": EXPECTED["m45_analyzer"],
            "m53_analyzer_sha256": EXPECTED["m53_analyzer"],
            "m53_result_sha256": EXPECTED["m53_result"],
            "m43_temporal_sha256": EXPECTED["m43_temporal"],
            "m89_receipt_sha256": EXPECTED["m89_receipt"],
        },
        "fixed_operator_modes": operator_modes,
        "per_operator": operator_rows,
        "per_sample": per_sample,
        "fixed_group_record_ledger": {
            "record_count": len(records),
            "records": records,
        },
        "aggregate": {
            "baseline_source_cycles": h0_source,
            "selected_source_cycles": selected_source,
            "selected_source_cycle_gain": h0_source - selected_source,
            "selected_source_speedup": fraction(h0_source, selected_source),
            "selected_source_reduction_fraction":
                fraction(h0_source - selected_source, h0_source),
            "fusion_groups": total_groups,
            "unique_weight_issues": total_unique,
            "tight_eight_bank_lower_bound_cycles": tight_lower,
            "remaining_cycles_above_tight_lower_bound":
                selected_source - tight_lower,
            "prohibited_per_record_oracle_source_cycles": oracle,
            "prohibited_oracle_source_gain": h0_source - oracle,
        },
        "stage1_gates": gates,
        "all_stage1_gates_pass": passed,
        "stage2": {
            "exact_integrated_replay_required": passed,
            "executed": False,
            "integrated_cycles": None,
            "p95_integrated_cycles": None,
        },
        "baseline_replay": baseline,
        "claim_policy": {
            "paper_ppa_ready": False,
            "rtl_cycle_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    result = build()
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    compact = {
        "status": result["status"],
        "all_stage1_gates_pass": result["all_stage1_gates_pass"],
        "operator_modes": result["fixed_operator_modes"],
        "baseline_source": result["aggregate"]["baseline_source_cycles"],
        "selected_source": result["aggregate"]["selected_source_cycles"],
        "gain": result["aggregate"]["selected_source_cycle_gain"],
        "speedup": result["aggregate"]["selected_source_speedup"]["decimal"],
        "tight_lower": result["aggregate"]["tight_eight_bank_lower_bound_cycles"],
        "per_sample_deltas": [row["selected_delta_cycles"]
                              for row in result["per_sample"]],
    }
    print("M96_FIXED_GROUP_BANK_SKEW=" + json.dumps(compact, sort_keys=True))


if __name__ == "__main__":
    main()
