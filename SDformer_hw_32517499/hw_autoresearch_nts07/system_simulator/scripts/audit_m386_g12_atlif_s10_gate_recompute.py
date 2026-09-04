#!/usr/bin/env python3
"""Independently recompute M366 G12 S10 counters and promotion gates."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


HW = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: {}".format(key))
        result[key] = value
    return result


def strict_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=no_duplicates)


def resolve(path_text):
    path = Path(path_text)
    return path if path.is_absolute() else HW / path


def close(a, b, tolerance=1e-15):
    return math.isclose(float(a), float(b), rel_tol=tolerance,
                        abs_tol=tolerance)


SCALARS = (
    "calls", "samples", "spatial_lanes", "output_lanes", "input_values",
    "input_nonfinite", "signed_q8_range_violations",
    "symmetric_q8_range_violations", "term_total", "term_skipped",
    "bound_violations", "integer_early_mismatches",
    "integer_vs_float_event_mismatches", "unresolved_after_t",
    "baseline_issue_cycles", "lane_compaction_issue_cycles",
)

VECTORS = (
    "resolved_at_k", "positive_resolved_at_k", "zero_resolved_at_k",
    "tile_resolved_at_k", "lane_compaction_need_by_step",
)


def empty_counter(temporal):
    result = {key: 0 for key in SCALARS}
    result.update({
        "unclamped_code_min": None,
        "unclamped_code_max": None,
        "resolved_at_k": [0] * (temporal + 1),
        "positive_resolved_at_k": [0] * (temporal + 1),
        "zero_resolved_at_k": [0] * (temporal + 1),
        "tile_resolved_at_k": [0] * (temporal + 1),
        "lane_compaction_need_by_step": [0] * temporal,
    })
    return result


def add_row(destination, row):
    for key in SCALARS:
        destination[key] += int(row[key])
    for key in VECTORS:
        require(len(destination[key]) == len(row[key]),
                "vector length drift {}".format(key))
        destination[key] = [a + int(b) for a, b in zip(
            destination[key], row[key])]
    value = int(row["unclamped_code_min"])
    destination["unclamped_code_min"] = (
        value if destination["unclamped_code_min"] is None else
        min(destination["unclamped_code_min"], value))
    value = int(row["unclamped_code_max"])
    destination["unclamped_code_max"] = (
        value if destination["unclamped_code_max"] is None else
        max(destination["unclamped_code_max"], value))


def finalize(counter):
    result = dict(counter)
    result["term_skip_ratio"] = (
        float(result["term_skipped"]) / result["term_total"])
    result["lane_compaction_issue_cycle_reduction"] = (
        1.0 - float(result["lane_compaction_issue_cycles"]) /
        result["baseline_issue_cycles"])
    return result


def compare_counter(observed, expected, label):
    for key in SCALARS + VECTORS + (
            "unclamped_code_min", "unclamped_code_max"):
        require(observed[key] == expected[key],
                "{} counter mismatch {}".format(label, key))
    require(close(observed["term_skip_ratio"], expected["term_skip_ratio"]),
            "{} term ratio mismatch".format(label))
    require(close(observed["lane_compaction_issue_cycle_reduction"],
                  expected["lane_compaction_issue_cycle_reduction"]),
            "{} cycle ratio mismatch".format(label))


def combine(aggregates, names):
    temporal = None
    result = None
    for name in names:
        item = aggregates[name]
        width = len(item["lane_compaction_need_by_step"])
        if temporal is None:
            temporal = width
            result = empty_counter(width)
        require(width == temporal, "mixed temporal group")
        add_row(result, item)
    return finalize(result)


def combined_projection(counter):
    return {
        "sites": None,
        "calls": counter["calls"],
        "spatial_lanes": counter["spatial_lanes"],
        "output_lanes": counter["output_lanes"],
        "term_total": counter["term_total"],
        "term_skipped": counter["term_skipped"],
        "term_skip_ratio": counter["term_skip_ratio"],
        "baseline_issue_cycles": counter["baseline_issue_cycles"],
        "lane_compaction_issue_cycles": counter[
            "lane_compaction_issue_cycles"],
        "lane_compaction_issue_cycle_reduction": counter[
            "lane_compaction_issue_cycle_reduction"],
        "signed_q8_range_violations": counter[
            "signed_q8_range_violations"],
        "symmetric_q8_range_violations": counter[
            "symmetric_q8_range_violations"],
        "input_nonfinite": counter["input_nonfinite"],
        "bound_violations": counter["bound_violations"],
        "integer_early_mismatches": counter[
            "integer_early_mismatches"],
        "integer_vs_float_event_mismatches": counter[
            "integer_vs_float_event_mismatches"],
    }


def compare_combined(observed, recomputed, sites, label):
    projected = combined_projection(recomputed)
    projected["sites"] = sites
    for key, value in projected.items():
        if isinstance(value, float):
            require(close(observed[key], value),
                    "{} float mismatch {}".format(label, key))
        else:
            require(observed[key] == value,
                    "{} mismatch {}".format(label, key))


def parse_receipt(path):
    receipt = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            require(key not in receipt, "duplicate receipt key")
            receipt[key] = value
    return receipt


def validate_identity(contract_path):
    contract = strict_json(contract_path)
    require(contract["schema"] ==
            "m386_g12_atlif_s10_gate_recompute_contract_v1",
            "M386 contract schema drift")
    observed = {}
    for name, record in contract["identity"].items():
        path = resolve(record["path"]).resolve()
        require(path.is_file(), "missing M386 input {}".format(name))
        actual = sha256(path)
        require(actual == record["sha256"],
                "M386 identity drift {}".format(name))
        observed[name] = {"path": str(path), "sha256": actual}
    return contract, observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite M386 output")
    contract, inputs = validate_identity(args.contract.resolve())
    manifest_path = Path(inputs["m366_manifest"]["path"])
    manifest = strict_json(manifest_path)
    require(manifest["schema"] ==
            "m366_h67_ep35_atlif_remaining_budget_s10_capture_v1",
            "M366 manifest schema drift")
    require(manifest["status"] == "PASS_M366_CAPTURE__NO_GO_G12_RTL",
            "M366 status drift")
    require(manifest["identity"]["contract_sha256"] ==
            inputs["m366_contract"]["sha256"],
            "M366 contract identity mismatch")
    require(manifest["identity"]["capture_script_sha256"] ==
            inputs["m366_capture_script"]["sha256"],
            "M366 script identity mismatch")
    load = manifest["identity"]["checkpoint_load_audit"]
    require(load["missing_count"] == 0 and load["unexpected_count"] == 0 and
            load["checkpoint_overlay_keys"] == load["model_overlay_keys"] == 210,
            "M366 checkpoint load mismatch")
    require(manifest["identity"]["source_config_bn_policy"] == "no_running" and
            manifest["identity"]["capture_bn_policy"] == "no_running",
            "M366 BN identity mismatch")

    receipt = parse_receipt(inputs["m366_queue_receipt"]["path"])
    require(receipt["status"] ==
            "PASS_M366_H67_EP35_ATLIF_REMAINING_BUDGET_S10",
            "M366 receipt status mismatch")
    require(receipt["manifest_sha256"] == inputs["m366_manifest"]["sha256"] and
            receipt["contract_sha256"] == inputs["m366_contract"]["sha256"] and
            receipt["capture_script_sha256"] ==
            inputs["m366_capture_script"]["sha256"],
            "M366 receipt identity mismatch")
    require(receipt["rtl_decision"] == "NO_GO_RTL" and
            receipt["system_speedup"] == "false" and
            receipt["headline"] == "false", "M366 receipt claim drift")

    rows = manifest["sample_site_rows"]
    sites = manifest["static_site_codes"]
    sample_keys = manifest["population"]["sample_keys"]
    require(len(sites) == 81 and len(rows) == 810 and len(sample_keys) == 10,
            "M366 population drift")
    require(len(manifest["site_aggregate"]) == 81 and
            len(manifest["witnesses"]) == 81,
            "M366 site-key coverage drift")
    require(set(sites) == set(manifest["site_aggregate"]) ==
            set(manifest["witnesses"]), "M366 site set mismatch")

    by_sample = defaultdict(set)
    sample_site_count = Counter()
    recomputed = {
        name: empty_counter(int(site["temporal_steps"]))
        for name, site in sites.items()
    }
    row_invariant_checks = 0
    range_rows = []
    for row in rows:
        name = row["name"]
        require(name in sites, "unknown row site")
        temporal = int(sites[name]["temporal_steps"])
        require(row["temporal_steps"] == temporal, "row T mismatch")
        sample_id = int(row["sample_id"])
        require(0 <= sample_id < 10 and
                row["sample_key"] == sample_keys[sample_id],
                "row sample identity mismatch")
        require(name not in by_sample[sample_id], "duplicate sample-site row")
        by_sample[sample_id].add(name)
        sample_site_count[name] += 1
        spatial = int(row["spatial_lanes"])
        output = int(row["output_lanes"])
        require(row["calls"] == row["samples"] == 1,
                "row call/sample mismatch")
        require(output == spatial * temporal and
                row["input_values"] == output and
                row["term_total"] == output * temporal,
                "row work identity mismatch")
        require(sum(row["resolved_at_k"]) == output and
                all(p + z == r for p, z, r in zip(
                    row["positive_resolved_at_k"],
                    row["zero_resolved_at_k"], row["resolved_at_k"])),
                "row resolution histogram mismatch")
        expected_skip = sum((temporal - k) * int(value)
                            for k, value in enumerate(row["resolved_at_k"]))
        require(expected_skip == row["term_skipped"],
                "row term skip mismatch")
        needs = row["lane_compaction_need_by_step"]
        require(len(needs) == temporal and
                all(0 <= int(value) <= spatial for value in needs) and
                all(needs[k] >= needs[k + 1]
                    for k in range(temporal - 1)),
                "row compaction need invalid")
        require(row["baseline_issue_cycles"] ==
                int(math.ceil(float(spatial) / 32.0)) * temporal and
                row["lane_compaction_issue_cycles"] ==
                sum(int(math.ceil(float(value) / 32.0)) for value in needs),
                "row cycle formula mismatch")
        require(sum(row["tile_resolved_at_k"]) ==
                int(math.ceil(float(spatial) / 32.0)),
                "row tile population mismatch")
        require(row["unresolved_after_t"] == 0 and
                row["bound_violations"] == 0 and
                row["integer_early_mismatches"] == 0,
                "row exactness mismatch")
        if row["signed_q8_range_violations"]:
            range_rows.append({
                "sample_id": sample_id,
                "sample_key": row["sample_key"],
                "name": name,
                "temporal_steps": temporal,
                "violations": row["signed_q8_range_violations"],
                "unclamped_code_min": row["unclamped_code_min"],
                "unclamped_code_max": row["unclamped_code_max"],
            })
        add_row(recomputed[name], row)
        row_invariant_checks += 14

    require(set(by_sample) == set(range(10)) and
            all(names == set(sites) for names in by_sample.values()) and
            all(count == 10 for count in sample_site_count.values()),
            "sample-site coverage mismatch")
    recomputed = {name: finalize(value)
                  for name, value in recomputed.items()}
    for name in sorted(sites):
        compare_counter(manifest["site_aggregate"][name],
                        recomputed[name], name)

    t10_names = sorted(name for name, site in sites.items()
                       if site["temporal_steps"] == 10)
    t2_names = sorted(name for name, site in sites.items()
                      if site["temporal_steps"] == 2)
    require(len(t10_names) == 45 and len(t2_names) == 36,
            "temporal site population mismatch")
    t10 = combine(recomputed, t10_names)
    t2 = combine(recomputed, t2_names)
    compare_combined(manifest["t10_nonattention_main"], t10, 45, "T10")
    compare_combined(manifest["t2_attention_diagnostic"], t2, 36, "T2")

    context = contract["performance_context"]
    fixed_reference = int(context["fixed_compute_reference_cycles"])
    dense_t10 = int(context["dense_t10_atlif_cycles"])
    fixed_speedup = fixed_reference / (
        fixed_reference - dense_t10 *
        t10["lane_compaction_issue_cycle_reduction"])
    require(close(fixed_speedup, manifest["fixed_compute_projection"][
        "conditional_fixed_context_speedup"]),
        "fixed-context projection mismatch")
    gates = contract["promotion_gates"]
    recomputed_gates = {
        "zero_mismatch": t10["integer_early_mismatches"] == 0,
        "zero_bound_violation": t10["bound_violations"] == 0,
        "zero_range_violation": (
            t10["signed_q8_range_violations"] == 0 and
            t10["input_nonfinite"] == 0),
        "term_skip": t10["term_skip_ratio"] >=
            gates["min_term_skip_ratio"],
        "executable_issue_cycle":
            t10["lane_compaction_issue_cycle_reduction"] >=
            gates["min_executable_issue_cycle_reduction"],
        "fixed_context": fixed_speedup >=
            gates["min_fixed_context_speedup"],
    }
    recomputed_gates["metric_gates_pass"] = all(recomputed_gates.values())
    recomputed_gates["metadata_and_compare_net_energy_positive"] = False
    recomputed_gates["all_pass"] = False
    require(recomputed_gates == manifest["promotion_gates"]["observed"],
            "promotion gate mismatch")

    m360 = strict_json(inputs["m360_predesign_hammer"]["path"])
    curated = m360["curated_integer_vector_recompute"][
        "issue_orders"]["per_site_descending_sum_abs_weight"][
            "term_skip_ratio"]
    site_rows = []
    for name in t10_names:
        item = recomputed[name]
        site_rows.append({
            "name": name,
            "spatial_lanes": item["spatial_lanes"],
            "term_skip_ratio": item["term_skip_ratio"],
            "issue_cycle_reduction": item[
                "lane_compaction_issue_cycle_reduction"],
            "range_violations": item["signed_q8_range_violations"],
            "integer_vs_float_event_mismatch_ratio":
                float(item["integer_vs_float_event_mismatches"]) /
                item["output_lanes"],
        })
    top_term = sorted(site_rows, key=lambda row: row["term_skip_ratio"],
                      reverse=True)[:10]
    top_cycle = sorted(site_rows, key=lambda row: row["issue_cycle_reduction"],
                       reverse=True)[:10]
    result = {
        "schema": "m386_g12_atlif_s10_gate_recompute_independent_hammer_v1",
        "milestone": "M386",
        "status": "PASS_INDEPENDENT_RECOMPUTE__KILL_G12_RTL",
        "identity": {
            "contract_path": str(args.contract.resolve()),
            "contract_sha256": sha256(args.contract.resolve()),
            "inputs": inputs,
            "m366_receipt": receipt,
            "checkpoint_load_audit": load,
            "paper_identity": "H67 ep35/no_running",
            "paft_ep4_mixed": False,
        },
        "independent_recompute": {
            "sample_site_rows": len(rows),
            "row_invariant_checks": row_invariant_checks,
            "sample_site_coverage_checks": 810,
            "site_aggregate_counter_mismatches": 0,
            "combined_counter_mismatches": 0,
            "promotion_gate_mismatches": 0,
            "fixed_projection_mismatches": 0,
        },
        "t10_nonattention_main": {
            **combined_projection(t10),
            "sites": 45,
            "integer_vs_float_event_mismatch_ratio":
                float(t10["integer_vs_float_event_mismatches"]) /
                t10["output_lanes"],
            "signed_q8_range_violation_ratio":
                float(t10["signed_q8_range_violations"]) /
                t10["input_values"],
        },
        "t2_attention_diagnostic": {
            **combined_projection(t2),
            "sites": 36,
            "integer_vs_float_event_mismatch_ratio":
                float(t2["integer_vs_float_event_mismatches"]) /
                t2["output_lanes"],
            "signed_q8_range_violation_ratio":
                float(t2["signed_q8_range_violations"]) /
                t2["input_values"],
        },
        "range_audit": {
            "affected_sample_site_rows": len(range_rows),
            "signed_q8_violations_all_sites": sum(
                row["violations"] for row in range_rows),
            "t10_signed_q8_violations": t10[
                "signed_q8_range_violations"],
            "t2_signed_q8_violations": t2[
                "signed_q8_range_violations"],
            "rows": range_rows,
        },
        "site_screen": {
            "t10_sites_meeting_35pct_term_gate": sum(
                row["term_skip_ratio"] >= gates["min_term_skip_ratio"]
                for row in site_rows),
            "t10_sites_meeting_25pct_cycle_gate": sum(
                row["issue_cycle_reduction"] >=
                gates["min_executable_issue_cycle_reduction"]
                for row in site_rows),
            "maximum_t10_site_term_skip_ratio": top_term[0][
                "term_skip_ratio"],
            "maximum_t10_site_issue_cycle_reduction": top_cycle[0][
                "issue_cycle_reduction"],
            "top_t10_by_term_skip": top_term,
            "top_t10_by_issue_cycle_reduction": top_cycle,
        },
        "m360_to_m366_generalization": {
            "curated_sample0_term_skip_ratio": curated,
            "representative_s10_term_skip_ratio": t10["term_skip_ratio"],
            "absolute_change": t10["term_skip_ratio"] - curated,
            "relative_change": t10["term_skip_ratio"] / curated - 1.0,
            "meaning": "The S10 result is weaker than the curated screen; the curated result did not hide a representative uplift."
        },
        "fixed_compute_projection": {
            "fixed_compute_reference_cycles": fixed_reference,
            "dense_t10_atlif_cycles": dense_t10,
            "recomputed_issue_cycle_reduction": t10[
                "lane_compaction_issue_cycle_reduction"],
            "conditional_fixed_context_speedup": fixed_speedup,
            "system_speedup_admitted": False,
        },
        "promotion_gates": {
            "thresholds": gates,
            "recomputed": recomputed_gates,
            "margins": {
                "term_skip_ratio_minus_threshold":
                    t10["term_skip_ratio"] - gates["min_term_skip_ratio"],
                "issue_cycle_reduction_minus_threshold":
                    t10["lane_compaction_issue_cycle_reduction"] -
                    gates["min_executable_issue_cycle_reduction"],
                "fixed_context_speedup_minus_threshold":
                    fixed_speedup - gates["min_fixed_context_speedup"],
            },
            "rtl_decision": "KILL_G12_RTL",
        },
        "hardware_and_algorithm_feedback": {
            "hardware": [
                "Do not build the dense remaining-budget G12 RTL: only 0.0676% of executable 32-lane issue cycles are removed.",
                "Per-lane term skips do not align across the lane group; the best individual T10 site removes only about 1.12% of issue cycles.",
                "Do not add the G12 opportunity to rank-3 ATLIF or any system headline; M360 already forbids additive bookkeeping and M366 fails independently."
            ],
            "algorithm": [
                "The sample0-frozen input scales have 35 T10 and 19 T2 S10 overflow events; representative per-site calibration or QAT is required for any future integer ATLIF deployment.",
                "T10 integer-versus-float event flips are 0.7496% over 11.709B events; paired valid825 accuracy is required before promoting the integer bridge.",
                "Scale/QAT repair cannot rescue G12 performance because the executable issue-cycle gate misses by roughly 24.93 percentage points."
            ],
        },
        "disposition": {
            "g12_dense_remaining_budget": "STOP_AFTER_PROOF_AND_NEGATIVE_S10_RESULT",
            "rtl": False,
            "vcs": False,
            "synopsys": False,
            "energy": False,
            "system_speedup": False,
            "headline": False,
            "paper_role": "negative design-space evidence or omit; not a contribution headline",
        },
        "protected_docs359": {
            "path": "docs/359_DATE终局冻结_20260813.md",
            "sha256": inputs["protected_docs359"]["sha256"],
            "modified": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
