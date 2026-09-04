#!/usr/bin/env python3
"""Combine independently audited FC1 and patch-Conv bounded-group screens.

This is deliberately an Amdahl sensitivity overlay.  It does not turn ideal
task compaction into executable hardware cycles.
"""

from __future__ import division

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


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


def resolve_identity(contract_path, contract):
    hw = contract_path.parents[1]
    repo = hw.parent
    paths = {}
    identity = {
        "contract": {
            "path": str(contract_path.relative_to(hw)),
            "sha256": sha256(contract_path),
        }
    }
    for label, spec in contract["identity"].items():
        base = repo if spec.get("relative_to") == "repo" else hw
        path = (base / spec["path"]).resolve()
        require(path.is_file(), "missing M300 input: " + str(path))
        digest = sha256(path)
        require(digest == spec["sha256"],
                "M300 input SHA drift for {}: {}".format(label, digest))
        paths[label] = path
        identity[label] = {"path": spec["path"], "sha256": digest}
    return hw, repo, paths, identity


def read_cycle_ledger(path):
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None and
                "name" in reader.fieldnames and
                "activity_cycles_at_config_lanes" in reader.fieldnames,
                "M300 operator ledger schema drift")
        for row in reader:
            name = row["name"]
            require(name not in rows, "duplicate operator ledger row: " + name)
            rows[name] = int(row["activity_cycles_at_config_lanes"])
    require(len(rows) == 79, "M300 frozen operator ledger row-count drift")
    return rows


def keyed_grid(rows):
    result = {}
    for row in rows:
        key = (int(row["destination_group_size"]),
               int(row["maximum_absolute_int8_weight"]))
        require(key not in result, "duplicate grid point: {}".format(key))
        result[key] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source_path = Path(__file__).resolve()
    source_start = sha256(source_path)
    contract_path = args.contract.resolve()
    contract = strict_json(contract_path)
    require(contract.get("schema") ==
            "m300_fc1_patch_shared_group_combined_dse_contract_v1",
            "M300 contract schema drift")
    require(source_start == contract["analyzer"]["sha256"],
            "M300 analyzer SHA drift")
    hw, _repo, paths, identity = resolve_identity(contract_path, contract)
    identity["analyzer"] = {
        "path": str(source_path.relative_to(hw.parent)),
        "sha256": source_start,
    }

    fc = strict_json(paths["m288_fc1_independent_recompute"])
    conv = strict_json(paths["m293_patch_result"])
    conv_independent = strict_json(
        paths["m295_patch_independent_recompute"])
    envelope_doc = strict_json(paths["m221_envelope"])
    fc_review = strict_json(paths["m299_fc1_independent_review"])
    conv_review = strict_json(paths["m295_patch_independent_review"])
    ledger = read_cycle_ledger(paths["operator_cycle_ledger"])

    require(fc["status"] ==
            "PASS_RAW_DSE_REBUILD__P0_FULL_FC1_AMDAHL_DENOMINATOR_REJECTED" and
            fc_review["severity_counts"]["P0"] == 0,
            "M300 FC1 independent review not admitted")
    require(conv["status"] ==
            "PASS_OPPORTUNITY_SCREEN_NOT_CYCLE_OR_ACCURACY_ADMISSION",
            "M300 patch source status drift")
    require(conv_independent["status"] ==
            "PASS_RAW_DSE__P1_AMDAHL_METHOD_CORRECTION__NO_HARDWARE_PROMOTION" and
            conv_review["status"] ==
            "RAW_DSE_PASS__P1_SCOPE_METHOD_CORRECTION__NO_ACCURACY_OR_HARDWARE_PROMOTION" and
            int(conv_review["scores"]["evidence_quality"]["score"]) >= 90,
            "M300 patch independent review not admitted")

    envelope = int(contract["frozen_scope"]["compute_envelope_cycles"])
    fc_cycles = int(contract["frozen_scope"]["eligible_fc1_cycles"])
    conv_cycles = int(contract["frozen_scope"]["eligible_patch_conv_cycles"])
    require(int(envelope_doc["frozen_h67_compute_envelope"]
                             ["cycles_per_frame"]) == envelope,
            "M300 envelope drift")

    fc_grid = {}
    for group, rows in fc["amdahl"]["module_cycle_weighted_scope_grid"].items():
        for row in rows:
            key = (int(group), int(row["maximum_absolute_int8_weight"]))
            require(key not in fc_grid, "duplicate M300 FC grid point")
            fc_grid[key] = row
    groups = [int(value) for value in contract["dse"]["destination_group_sizes"]]
    betas = [int(value) for value in
             contract["dse"]["maximum_absolute_int8_weight_per_group"]]
    require(groups == [4, 8, 16, 32, 96] and
            betas == [0, 8, 16, 24, 32, 48, 64, 80, 96],
            "M300 grid drift")
    require(set(fc_grid) == set((group, beta) for group in groups
                                for beta in betas),
            "M300 FC grid coverage drift")

    conv_modules = {}
    for module in conv["per_module"]:
        name = module["module"]
        require(name in ledger, "M300 patch module absent from cycle ledger")
        require(name not in conv_modules, "M300 duplicate patch module")
        conv_modules[name] = module
    require(len(conv_modules) == 6, "M300 expected six patch modules")
    conv_ledger_cycles = sum(ledger[name] for name in conv_modules)
    require(conv_ledger_cycles == conv_cycles,
            "M300 six-module patch cycle partition drift")

    combined_grid = {}
    first_crossing = {}
    target = float(contract["dse"]["target_full_envelope_speedup"])
    for group in groups:
        rows = []
        for beta in betas:
            fc_row = fc_grid[(group, beta)]
            projected_fc = float(fc_row["projected_eligible_fc1_cycles"])
            projected_conv = 0.0
            conv_baseline_tasks = 0
            conv_kept_tasks = 0
            conv_static_pairs = 0
            conv_static_pairs_removed = 0
            conv_module_rows = []
            for name in sorted(conv_modules):
                module = conv_modules[name]
                point = module["groups"][str(group)][str(beta)]
                baseline = int(point["baseline_group_tasks"])
                kept = int(point["kept_group_tasks"])
                require(baseline > 0 and kept > 0,
                        "M300 zero task denominator")
                ratio = float(baseline) / float(kept)
                cycles = int(ledger[name])
                projected = float(cycles) / ratio
                projected_conv += projected
                conv_baseline_tasks += baseline
                conv_kept_tasks += kept
                conv_static_pairs += int(point["static_source_group_pairs"])
                conv_static_pairs_removed += int(
                    point["static_source_group_pairs_removed"])
                conv_module_rows.append({
                    "module": name,
                    "baseline_cycles": cycles,
                    "baseline_group_tasks": baseline,
                    "kept_group_tasks": kept,
                    "ideal_task_compaction_speedup": ratio,
                    "projected_cycles": projected,
                })
            independent_conv_points = [
                point for point in
                conv_independent["scope_corrected_5x9_grid"][str(group)]
                if int(point["beta"]) == beta
            ]
            require(len(independent_conv_points) == 1,
                    "M300 independent patch point uniqueness drift")
            independent_conv = independent_conv_points[0]
            independent_projected_conv = float(
                independent_conv["projected_eligible_cycles_per_module_exact"]
                                ["decimal"])
            require(abs(projected_conv - independent_projected_conv) <=
                    1.0e-7,
                    "M300 patch per-module cycle projection mismatch")
            require(conv_baseline_tasks ==
                    int(independent_conv["baseline_group_tasks"]) and
                    conv_kept_tasks ==
                    int(independent_conv["kept_group_tasks"]),
                    "M300 patch task ledger mismatch")
            denominator = (float(envelope) - float(fc_cycles) -
                           float(conv_cycles) + projected_fc + projected_conv)
            speedup = float(envelope) / denominator
            denominator_floor = (envelope - fc_cycles - conv_cycles +
                                 math.floor(projected_fc) +
                                 math.floor(projected_conv))
            denominator_ceil = (envelope - fc_cycles - conv_cycles +
                                math.ceil(projected_fc) +
                                math.ceil(projected_conv))
            row = {
                "destination_group_size": group,
                "maximum_absolute_int8_weight": beta,
                "baseline_eligible_fc1_cycles": fc_cycles,
                "projected_eligible_fc1_cycles": projected_fc,
                "baseline_eligible_patch_conv_cycles": conv_cycles,
                "projected_eligible_patch_conv_cycles": projected_conv,
                "combined_projected_compute_envelope_cycles": denominator,
                "combined_full_compute_envelope_sensitivity": speedup,
                "integer_rounding_attack": {
                    "floor_both_projected_cycle_terms_speedup":
                        float(envelope) / float(denominator_floor),
                    "ceil_both_projected_cycle_terms_speedup":
                        float(envelope) / float(denominator_ceil),
                },
                "fc1_weighted_group_task_fraction_removed":
                    float(fc["independent_aggregate_dse"][str(group)]
                            [betas.index(beta)]
                            ["weighted_group_task_fraction_removed"]),
                "fc1_static_source_group_pair_fraction_removed":
                    float(fc["independent_aggregate_dse"][str(group)]
                            [betas.index(beta)]
                            ["static_source_group_fraction_removed"]),
                "patch_conv_weighted_group_task_fraction_removed":
                    float(conv_baseline_tasks - conv_kept_tasks) /
                    float(conv_baseline_tasks),
                "patch_conv_static_source_group_pair_fraction_removed":
                    float(conv_static_pairs_removed) / float(conv_static_pairs),
                "m295_exact_patch_projection_reproduced": True,
                "crosses_target": speedup >= target,
                "conv_module_cycle_projection": conv_module_rows,
                "scope_warning": "ideal source/destination-group task compaction sensitivity; no router, bank conflicts, scan/commit overhead, SRAM timing, or executable schedule",
            }
            rows.append(row)
        combined_grid[str(group)] = rows
        candidates = [row for row in rows if row["crosses_target"]]
        first_crossing[str(group)] = candidates[0] if candidates else None

    selected_group = int(contract["paired_s10_candidate"]["destination_group_size"])
    selected_beta = int(contract["paired_s10_candidate"]
                        ["maximum_absolute_int8_weight"])
    selected = [row for row in combined_grid[str(selected_group)]
                if row["maximum_absolute_int8_weight"] == selected_beta]
    require(len(selected) == 1, "M300 selected candidate uniqueness drift")
    selected = selected[0]
    require(selected["combined_full_compute_envelope_sensitivity"] >= target,
            "M300 selected candidate fell below target")

    result = {
        "schema": "m300_fc1_patch_shared_group_combined_dse_v1",
        "status": "PASS_COMBINED_SENSITIVITY_ELIGIBLE_FOR_PAIRED_S10_ONLY",
        "identity": identity,
        "frozen_scope": contract["frozen_scope"],
        "mechanism": {
            "name": "shared bounded source-by-destination-group task elision",
            "shared_hardware_metadata": "one source bitmap plus per-source destination-group keep mask serves FC1 and 3x3 patch Conv",
            "zero_budget_behavior": "beta=0 preserves every source/group task and is the exact engine subset",
            "deterministic_integer_bound": "each omitted source/group task perturbs each covered destination INT8 accumulator by at most beta times source magnitude; published bound remains raw integer-domain only",
        },
        "population_caveat": {
            "fc1": "M51 detailed task fractions are mapped onto M221 module cycles; aggregate source population differs by 0.45336% and the maximum module delta is 3.26616%",
            "patch_conv": "M51 detailed receptive-field source fractions are mapped onto M221 module cycles; maximum module input-population delta is 4.61322%, so this is not a same-population executable cycle replay",
            "consequence": "all values in this result remain Amdahl sensitivities, including points robust to integer rounding",
        },
        "combined_grid": combined_grid,
        "first_beta_crossing_combined_full_envelope_1p15": first_crossing,
        "paired_s10_candidate": selected,
        "decision_policy": {
            "paired_protocol": "same frozen checkpoint, same ordered S10 sample IDs, no_running BN, beta=0 baseline before beta=48 candidate",
            "maximum_absolute_aee_increase": 0.02,
            "stop_if_s10_fails": True,
            "valid825_only_if_s10_passes": True,
            "rtl_only_after_accuracy_and_executable_cycle_adapter": True,
        },
        "admission": {
            "checkpoint_trace_weight_opportunity": True,
            "scope_correct_amdahl_sensitivity": True,
            "shared_fc1_patch_candidate_for_paired_s10": True,
            "modified_forward_accuracy": False,
            "hardware_cycles": False,
            "rtl": False,
            "dc": False,
            "power": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    target_path = args.output_dir / "m300_fc1_patch_shared_group_combined_dse_r1.json"
    target_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    require(sha256(source_path) == source_start,
            "M300 analyzer changed during execution")
    print("PASS M300 g{} beta{} sensitivity={:.12f} output={}".format(
        selected_group, selected_beta,
        selected["combined_full_compute_envelope_sensitivity"], target_path))


if __name__ == "__main__":
    main()
