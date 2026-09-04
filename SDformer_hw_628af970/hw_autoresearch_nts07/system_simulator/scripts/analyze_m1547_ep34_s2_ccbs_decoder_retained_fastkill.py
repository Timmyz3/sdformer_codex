#!/opt/anaconda3/envs/pytorch310_cpu/bin/python
"""M1547: retained ep34 decoder screen for S2 CCBS.

This is a deliberately local, CPU-only screen.  It uses the sealed M1521
binary decoder planes and the exact ep34 FP32 decoder weights to ask whether
the CCBS bound is already too coarse or its directory is already too large.
It does not evaluate AEE and cannot admit cycles, traffic, energy or RTL.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
CHECKPOINT = HW / "system_handoff/incoming/motion_c12_ep34_live93_checkpoint_epoch34.pth"
M1521_ROOT = HW / "results/m1521_ep34_decoder_positive_planes_s30_c120_r1_20260831"
M1521_MANIFEST = M1521_ROOT / "manifest.json"
M1521_SUMS = M1521_ROOT / "SHA256SUMS"
M1521_OUTER = M1521_ROOT / "SHA256SUMS.seal.sha256"
M1535 = HW / "reviews/m1535_ep34_lossy_sparse_candidate_mine_r1_20260831/review.json"
M1545 = HW / "reviews/m1545_ep34_sparse_candidate_priority_first_principles_review_r1_20260831/review.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
CONTRACT = HW / "contracts/m1547_ep34_s2_ccbs_decoder_retained_fastkill_contract_r1_20260831.json"
DEFAULT_OUTPUT = HW / "results/m1547_ep34_s2_ccbs_decoder_retained_fastkill_r1_20260831"

EXPECTED = {
    CHECKPOINT: "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48",
    M1521_MANIFEST: "969b786bf66323174bc734630384ae03abab5b81a4fc59000b113e0b7a5d8304",
    M1521_SUMS: "985b7089560b77b09dc0e5327780da1d81e24f03670ee2658433cae3f7603efa",
    M1521_OUTER: "60a172e5cd041bcdd0ca38db87250090c48c66e655364b332868fb40a1b182f2",
    M1535: "a0c6d1bc2f0fd03db472781418bd113f87b02f2a3dcbba2ad089ade95ba9e0e8",
    M1545: "8fb823b3cf1a325df6404a985a898b0e057d73aa99e2210e83864dbdf9b00d7f",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

CHECKPOINT_WEIGHT_SHA256 = (
    "cb1a90a4ff33622024b43ee6b15a3409e2567ea1e7b626715f40cf8a4fbfd83b",
    "35a9214e9fbc2e4e271beea74c4f329c12d6c072cda9252eaae350dd404a51cb",
    "75f9921f3cd9786ece78247115dd07bdda425b4f6e068d43936c884c611d3ef7",
    "6a42dabae358d0048aa46c609c9cb633f1e8d0479e4628e4f85c21e00835ea4e",
)
BLOCK_CONFIGS = ((8, 16), (16, 16), (32, 16))
EPSILON_GRID = (0.0, 0.05, 0.10, 0.20, 0.30)
SELECTED_LOCAL_SAMPLE_POSITIONS = (0, 4, 9)
SITES_PER_CALL = 64
EXACT_SITES_PER_CALL = 2
EXACT_G_BLOCKS_PER_LAYER = 16
EXACT_O_BLOCKS_PER_LAYER = 8
BOUND_MEDIAN_RATIO_MAX = 4.0
BOUND_P90_RATIO_MAX = 12.0
BOUND_FALSE_ZERO_COLLISION_MAX = 0.01
METADATA_WEIGHT_RATIO_MAX = 0.02
METADATA_REDUCTION_MIN = 8.0
DYNAMIC_WITNESS_EPSILON = 0.10

CLAIM_BOUNDARY = {
    "retained_decoder_local_screen": True,
    "checkpoint_bound": True,
    "m1521_positive_plane_bound": True,
    "aee": False,
    "accuracy_admission": False,
    "cycles": False,
    "speedup": False,
    "system_speedup": False,
    "traffic": False,
    "energy": False,
    "rtl": False,
    "vcs": False,
    "eda": False,
    "paper_headline": False,
}


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
        output = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON: " + token)))
    require(type(value) is dict, "JSON root is not object")
    return value


def ceil_div(value, divisor):
    require(type(value) is int and type(divisor) is int and value >= 0 and divisor > 0,
            "ceil_div arguments invalid")
    return (value + divisor - 1) // divisor


def metadata_account(cin, cout, kernel_elements, group_size, output_tile):
    """Charge one uint16 M per (G,O), versus uint16 beta per source/O."""
    g_blocks = ceil_div(cin, group_size)
    o_blocks = ceil_div(cout, output_tile)
    metadata_bytes = 2 * g_blocks * o_blocks
    old_g11_bytes = 2 * cin * o_blocks
    int8_weight_bytes = cin * cout * kernel_elements
    return {
        "g_blocks": g_blocks,
        "o_blocks": o_blocks,
        "metadata_bytes": metadata_bytes,
        "metadata_reads_per_source_site": g_blocks * o_blocks,
        "old_g11_per_source_metadata_bytes": old_g11_bytes,
        "metadata_to_int8_weight_bytes": float(metadata_bytes) / float(int8_weight_bytes),
        "reduction_vs_old_g11": float(old_g11_bytes) / float(metadata_bytes),
    }


def fixed_order_drop(bounds, epsilon, reference):
    """Fixed source-group order; return drop mask and accumulated local debt."""
    require(epsilon >= 0.0 and reference > 0.0, "drop arguments invalid")
    debt = 0.0
    mask = []
    limit = epsilon * reference
    for value in bounds:
        value = float(value)
        require(value >= 0.0 and math.isfinite(value), "invalid bound")
        drop = (value == 0.0) or (debt + value <= limit)
        mask.append(bool(drop))
        if drop:
            debt += value
    return mask, debt


def dynamic_witness_count(states):
    """states maps a static block identity to bit 1=drop, 2=keep."""
    return sum(1 for value in states.values() if value == 3)


def validate_block_configs(configs):
    require(tuple(tuple(row) for row in configs) == BLOCK_CONFIGS,
            "block config/order drift")


def verify_inputs():
    for path, expected in EXPECTED.items():
        require(path.is_file(), "missing input: " + str(path))
        require(sha256(path) == expected, "SHA mismatch: " + str(path))
    require(M1521_OUTER.read_text().split() == [EXPECTED[M1521_SUMS], "SHA256SUMS"],
            "M1521 outer seal content drift")
    manifest = strict_json(M1521_MANIFEST)
    require(manifest.get("schema") == "m1521_ep34_decoder_positive_plane_materialization_r1_v1",
            "M1521 schema drift")
    require(manifest.get("capture", {}).get("checkpoint_sha256") == EXPECTED[CHECKPOINT],
            "M1521 checkpoint identity drift")
    require(manifest.get("population", {}).get("calls") == 120 and
            manifest.get("population", {}).get("samples") == 30 and
            len(manifest.get("records", [])) == 120,
            "M1521 population drift")
    require(strict_json(M1535).get("status") ==
            "PASS_READONLY_CANDIDATE_MINE__S1_S2_CPU_FORWARD_FASTKILL_ONLY__NO_RTL",
            "M1535 status drift")
    require(strict_json(M1545).get("status") ==
            "PASS_PRIORITY_REVIEW__DUAL_PUSH_S2_AND_TSBG__S1_PIGGYBACK_ONLY__NO_RTL_AUTHORIZED",
            "M1545 status drift")
    validate_block_configs(BLOCK_CONFIGS)
    return manifest


def select_records(manifest):
    records = manifest["records"]
    sequences = sorted(set(row["sequence"] for row in records))
    require(sequences == ["interlaken_01_a", "thun_01_b", "zurich_city_12_a"],
            "sequence population drift")
    selected = []
    selected_samples = []
    for sequence in sequences:
        samples = sorted(set((row["replay_sample_ordinal"], row["global_sample_id"])
                             for row in records if row["sequence"] == sequence))
        require(len(samples) == 10, "sequence sample count drift")
        for local_position in SELECTED_LOCAL_SAMPLE_POSITIONS:
            sample = samples[local_position]
            rows = [row for row in records
                    if row["sequence"] == sequence and
                    row["replay_sample_ordinal"] == sample[0]]
            rows.sort(key=lambda row: row["module_ordinal"])
            require([row["module_ordinal"] for row in rows] == [0, 1, 2, 3],
                    "selected layer/order drift")
            selected.extend(rows)
            selected_samples.append({
                "sequence": sequence,
                "replay_sample_ordinal": sample[0],
                "global_sample_id": sample[1],
            })
    require(len(selected) == 36 and len(selected_samples) == 9,
            "selected population drift")
    return selected, selected_samples


def unpack_selected_sites(path, record, sites_per_call):
    import numpy as np
    require(sha256(path) == record["positive_output_sha256"], "payload SHA drift")
    raw = np.fromfile(str(path), dtype=np.uint8)
    require(int(raw.size) == int(record["plane_bytes"]), "payload byte extent drift")
    lookup = np.array([[((value >> bit) & 1) for bit in range(8)]
                       for value in range(256)], dtype=np.uint8)
    bits = lookup[raw].reshape(-1)[:int(record["elements"])]
    shape = tuple(int(value) for value in record["shape"])
    require(int(bits.size) == int(record["elements"]) and shape[1] == 1,
            "payload element/shape drift")
    plane = bits.reshape(shape)
    time_steps, _, channels, height, width = shape
    sites = time_steps * height * width
    indices = np.linspace(0, sites - 1, sites_per_call, dtype=np.int64)
    output = np.empty((sites_per_call, channels), dtype=np.uint8)
    for ordinal, flat in enumerate(indices.tolist()):
        time_index = flat // (height * width)
        spatial = flat % (height * width)
        output[ordinal, :] = plane[time_index, 0, :,
                                   spatial // width, spatial % width]
    require(set(np.unique(output).tolist()).issubset(set([0, 1])),
            "selected payload is not binary")
    return output


def quantiles(values):
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    require(values.size > 0 and bool(np.isfinite(values).all()), "empty/nonfinite quantiles")
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def sampled_indices(count, wanted):
    import numpy as np
    require(count > 0 and wanted > 0, "sample index arguments invalid")
    return sorted(set(int(value) for value in
                      np.linspace(0, count - 1, min(count, wanted), dtype=np.int64)))


def analyze_config(weights, activities, group_size, output_tile):
    import numpy as np
    aggregate_bounds = []
    exact_ratios = []
    false_zero_collisions = 0
    exact_positive_bounds = 0
    epsilon_totals = dict((str(value), {"blocks": 0, "dropped": 0,
                                        "normalized_debt": []})
                          for value in EPSILON_GRID)
    witness_states = {}
    layer_rows = []
    aggregate_metadata = {"metadata_bytes": 0, "int8_weight_bytes": 0,
                          "old_g11_bytes": 0, "metadata_reads": 0}

    for layer_ordinal in range(4):
        weight = weights[layer_ordinal]
        obs = activities[layer_ordinal]
        cin, cout, kh, kw = weight.shape
        require(obs.shape[1] == cin and kh == 3 and kw == 3,
                "weight/activity shape mismatch")
        account = metadata_account(cin, cout, kh * kw, group_size, output_tile)
        aggregate_metadata["metadata_bytes"] += account["metadata_bytes"]
        aggregate_metadata["int8_weight_bytes"] += cin * cout * kh * kw
        aggregate_metadata["old_g11_bytes"] += account["old_g11_per_source_metadata_bytes"]
        aggregate_metadata["metadata_reads"] += account["metadata_reads_per_source_site"] * int(obs.shape[0])
        g_blocks = account["g_blocks"]
        o_blocks = account["o_blocks"]
        max_weight = float(np.abs(weight).max())
        require(max_weight > 0.0, "zero layer max weight")
        metadata = np.zeros((g_blocks, o_blocks), dtype=np.float64)
        for gb in range(g_blocks):
            gs = gb * group_size
            ge = min(cin, gs + group_size)
            for ob in range(o_blocks):
                os_ = ob * output_tile
                oe = min(cout, os_ + output_tile)
                metadata[gb, ob] = float(np.abs(weight[gs:ge, os_:oe, :, :]).max())
        padded = np.zeros((int(obs.shape[0]), g_blocks * group_size), dtype=np.uint8)
        padded[:, :cin] = obs
        activity = padded.reshape(int(obs.shape[0]), g_blocks, group_size).sum(axis=2)
        bounds = activity[:, :, None].astype(np.float64) * metadata[None, :, :]
        normalized_block = bounds / (max_weight * float(group_size))
        aggregate_bounds.append(normalized_block.reshape(-1))
        reference = max_weight * float(cin)

        for eps in EPSILON_GRID:
            totals = epsilon_totals[str(eps)]
            for observation in range(int(obs.shape[0])):
                for ob in range(o_blocks):
                    mask, debt = fixed_order_drop(bounds[observation, :, ob].tolist(),
                                                  eps, reference)
                    totals["blocks"] += len(mask)
                    totals["dropped"] += sum(1 for value in mask if value)
                    totals["normalized_debt"].append(debt / reference)
                    if eps == DYNAMIC_WITNESS_EPSILON:
                        for gb, dropped in enumerate(mask):
                            key = (layer_ordinal, gb, ob)
                            witness_states[key] = witness_states.get(key, 0) | (1 if dropped else 2)

        exact_observations = sampled_indices(int(obs.shape[0]), 18)
        exact_g_blocks = sampled_indices(g_blocks, EXACT_G_BLOCKS_PER_LAYER)
        exact_o_blocks = sampled_indices(o_blocks, EXACT_O_BLOCKS_PER_LAYER)
        for observation in exact_observations:
            for gb in exact_g_blocks:
                gs = gb * group_size
                ge = min(cin, gs + group_size)
                active = obs[observation, gs:ge].astype(bool)
                if not bool(active.any()):
                    continue
                for ob in exact_o_blocks:
                    os_ = ob * output_tile
                    oe = min(cout, os_ + output_tile)
                    bound = float(bounds[observation, gb, ob])
                    require(bound > 0.0, "active exact sample has zero bound")
                    contribution = weight[gs:ge, os_:oe, :, :][active, :, :, :].sum(axis=0)
                    exact_value = float(np.abs(contribution).max())
                    exact_positive_bounds += 1
                    if exact_value <= 1.0e-12:
                        false_zero_collisions += 1
                    else:
                        exact_ratios.append(bound / exact_value)

        layer_rows.append({
            "module_ordinal": layer_ordinal,
            "cin": cin,
            "cout": cout,
            "selected_observations": int(obs.shape[0]),
            "metadata": account,
            "layer_max_abs_fp32_weight": max_weight,
            "bound_observations": int(bounds.size),
            "exact_ratio_sample_policy": {
                "observations": len(exact_observations),
                "g_blocks": len(exact_g_blocks),
                "o_blocks": len(exact_o_blocks),
            },
        })

    bound_stats = quantiles(np.concatenate(aggregate_bounds))
    ratio_stats = quantiles(exact_ratios)
    false_zero_fraction = (float(false_zero_collisions) /
                           float(exact_positive_bounds))
    metadata_ratio = (float(aggregate_metadata["metadata_bytes"]) /
                      float(aggregate_metadata["int8_weight_bytes"]))
    metadata_reduction = (float(aggregate_metadata["old_g11_bytes"]) /
                          float(aggregate_metadata["metadata_bytes"]))
    epsilon_rows = []
    for eps in EPSILON_GRID:
        totals = epsilon_totals[str(eps)]
        epsilon_rows.append({
            "epsilon_normalized": eps,
            "block_decisions": totals["blocks"],
            "drop_fraction": float(totals["dropped"]) / float(totals["blocks"]),
            "normalized_debt": quantiles(totals["normalized_debt"]),
            "is_aee_budget": False,
        })
    dynamic_count = dynamic_witness_count(witness_states)
    gates = {
        "bound_median_ratio_le_4": ratio_stats["median"] <= BOUND_MEDIAN_RATIO_MAX,
        "bound_p90_ratio_le_12": ratio_stats["p90"] <= BOUND_P90_RATIO_MAX,
        "false_zero_collision_le_1pct": false_zero_fraction <= BOUND_FALSE_ZERO_COLLISION_MAX,
        "metadata_le_2pct_int8_weight_bytes": metadata_ratio <= METADATA_WEIGHT_RATIO_MAX,
        "metadata_reduction_vs_old_g11_ge_8x": metadata_reduction >= METADATA_REDUCTION_MIN,
        "dynamic_same_block_keep_drop_witness": dynamic_count > 0,
        "zero_epsilon_only_zero_bound": epsilon_rows[0]["normalized_debt"]["max"] == 0.0,
        "fetch_gate_precedes_weight_fetch_structurally": True,
    }
    return {
        "block": {"source_group": group_size, "output_tile": output_tile},
        "weight_domain": "exact ep34 FP32 for local bound; no INT8 numeric authority",
        "metadata_charge_domain": "uint16 M directory versus hypothetical packed INT8 weight bytes",
        "metadata_aggregate": {
            "metadata_bytes": aggregate_metadata["metadata_bytes"],
            "int8_weight_bytes": aggregate_metadata["int8_weight_bytes"],
            "metadata_to_int8_weight_bytes": metadata_ratio,
            "old_g11_per_source_metadata_bytes": aggregate_metadata["old_g11_bytes"],
            "reduction_vs_old_g11": metadata_reduction,
            "metadata_reads_over_selected_sites": aggregate_metadata["metadata_reads"],
        },
        "bound_normalized_to_layer_max_times_group_capacity": bound_stats,
        "bound_to_exact_local_contribution_ratio_sample": ratio_stats,
        "positive_bound_exact_zero_collision_fraction": false_zero_fraction,
        "dynamic_witness": {
            "epsilon_normalized": DYNAMIC_WITNESS_EPSILON,
            "static_blocks_observed": len(witness_states),
            "blocks_with_both_keep_and_drop": dynamic_count,
            "fraction": float(dynamic_count) / float(len(witness_states)),
        },
        "epsilon_diagnostics": epsilon_rows,
        "layers": layer_rows,
        "gates": gates,
        "passes_local_screen": all(gates.values()),
    }


def load_weights():
    import numpy as np
    import torch
    wrapper = torch.load(str(CHECKPOINT), map_location="cpu")
    require(type(wrapper) is dict and set(wrapper) == set(["model_state_dict"]),
            "checkpoint wrapper drift")
    state = wrapper["model_state_dict"]
    require(len(state) == 921, "checkpoint state population drift")
    weights = []
    for ordinal in range(4):
        key = "sttmultires_unet.decoders.{}.deconv.0.weight".format(ordinal)
        require(key in state and torch.is_tensor(state[key]), "decoder weight missing")
        value = state[key].detach().cpu().contiguous().numpy()
        require(value.dtype == np.float32 and value.ndim == 4, "decoder weight dtype/rank drift")
        require(hashlib.sha256(value.tobytes(order="C")).hexdigest() ==
                CHECKPOINT_WEIGHT_SHA256[ordinal], "decoder weight content SHA drift")
        weights.append(value)
    return weights


def write_sealed_result(output, result):
    require(not output.exists(), "output already exists")
    output.mkdir(parents=True)
    result_path = output / "m1547_ep34_s2_ccbs_decoder_retained_fastkill_r1.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n")
    report_path = output / "m1547_REPORT.md"
    passing = ["{}x{}".format(row["block"]["source_group"],
                               row["block"]["output_tile"])
               for row in result["block_results"] if row["passes_local_screen"]]
    report_path.write_text(
        "# M1547 S2 CCBS retained decoder fast-kill\n\n"
        "Status: **{}**.\n\n"
        "This CPU-only screen binds ep34 checkpoint `{}` and 36 sealed M1521 decoder "
        "calls (three fixed samples per DSEC sequence, all four layers). Passing block "
        "configurations: `{}`. Decoder is only a local binary screen: no FC/patch "
        "coverage, AEE, cycles, speedup, traffic, energy or RTL is admitted. Epsilon "
        "and debt values are unitless local diagnostics, not an AEE budget.\n".format(
            result["status"], EXPECTED[CHECKPOINT], ", ".join(passing) or "none"))
    complete_path = output / "RUN_COMPLETE.txt"
    complete_path.write_text(result["status"] + "\n")
    member_paths = [result_path, report_path, complete_path]
    sums_path = output / "SHA256SUMS"
    sums_path.write_text("".join("{}  {}\n".format(sha256(path), path.name)
                                 for path in member_paths))
    (output / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(sha256(sums_path)))


def run(output):
    import numpy as np
    manifest = verify_inputs()
    contract = strict_json(CONTRACT)
    require(contract.get("block_configs") == [list(row) for row in BLOCK_CONFIGS] and
            contract.get("epsilon_grid") == list(EPSILON_GRID) and
            contract.get("claim_boundary") == CLAIM_BOUNDARY,
            "contract policy drift")
    selected, selected_samples = select_records(manifest)
    activities = dict((ordinal, []) for ordinal in range(4))
    selected_calls = []
    for record in selected:
        path = M1521_ROOT / record["positive_output"]
        values = unpack_selected_sites(path, record, SITES_PER_CALL)
        activities[record["module_ordinal"]].append(values)
        selected_calls.append({
            "global_call_ordinal": record["global_call_ordinal"],
            "sequence": record["sequence"],
            "replay_sample_ordinal": record["replay_sample_ordinal"],
            "module_ordinal": record["module_ordinal"],
            "positive_output_sha256": record["positive_output_sha256"],
        })
    activities = dict((ordinal, np.concatenate(rows, axis=0))
                      for ordinal, rows in activities.items())
    require(all(values.shape[0] == 9 * SITES_PER_CALL
                for values in activities.values()), "activity population drift")
    weights = load_weights()
    block_results = [analyze_config(weights, activities, group, output)
                     for group, output in BLOCK_CONFIGS]
    passing = [row for row in block_results if row["passes_local_screen"]]
    status = ("PASS_LOCAL_SCREEN__REQUEST_INCREMENTAL_FC_PATCH_CAPTURE_ONLY__NO_RTL_AEE_OR_PERFORMANCE"
              if passing else
              "NO_GO_S2_CCBS_RETAINED_DECODER_SCREEN__NO_RTL_AEE_OR_PERFORMANCE")
    result = {
        "schema": "m1547_ep34_s2_ccbs_decoder_retained_fastkill_r1_v1",
        "status": status,
        "identity": {
            "checkpoint_sha256": EXPECTED[CHECKPOINT],
            "m1521_manifest_sha256": EXPECTED[M1521_MANIFEST],
            "m1521_sha256s_sha256": EXPECTED[M1521_SUMS],
            "m1521_outer_file_sha256": EXPECTED[M1521_OUTER],
            "m1535_review_sha256": EXPECTED[M1535],
            "m1545_review_sha256": EXPECTED[M1545],
            "docs359_sha256": EXPECTED[DOCS359],
            "contract_sha256": sha256(CONTRACT),
        },
        "population": {
            "selection": "per sequence local sample positions 0,4,9; all decoder layers",
            "selected_samples": selected_samples,
            "selected_calls": len(selected_calls),
            "selected_sites_per_call": SITES_PER_CALL,
            "selected_observations_per_layer": 9 * SITES_PER_CALL,
            "calls": selected_calls,
        },
        "schedule_screen": {
            "decision_input": "A(G) from source-group popcount plus one static M(G,O) directory read",
            "decision_before_weight_fetch": True,
            "weight_payload_needed_for_decision": False,
            "runtime_sorter": False,
            "fixed_source_group_order": True,
            "decoder_scope_only": True,
        },
        "block_results": block_results,
        "decision": {
            "passing_configs": [row["block"] for row in passing],
            "retained_decoder_screen": "PASS" if passing else "NO_GO",
            "next_authorized_request": ("compact incremental FC/patch capture only"
                                        if passing else "none"),
            "rtl_authorized": False,
            "aee_authorized_by_this_result": False,
            "performance_claim_authorized": False,
            "reason": ("at least one frozen block clears predeclared bound, metadata, exact-zero and dynamic-witness gates"
                       if passing else
                       "all frozen blocks fail at least one predeclared local gate"),
        },
        "limitations": [
            "M1521 decoder planes are binary and are not the final patch/FC S2 scope",
            "FP32 exact weights are used for local bounds; uint16 metadata and INT8 weight bytes are capacity diagnostics only",
            "no paired forward or AEE was run",
            "no address-timed replay charges cycles, traffic, bank conflicts or energy",
            "normalized epsilon/debt is dimensionless and must not be called an AEE budget",
        ],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_sealed_result(output, result)
    print(status)
    return result


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.preflight:
        manifest = verify_inputs()
        selected, _ = select_records(manifest)
        require(CONTRACT.is_file(), "missing contract")
        print("PASS_M1547_PREFLIGHT calls={} gpu=false ssh=false rtl=false".format(len(selected)))
        return
    run(args.output)


if __name__ == "__main__":
    main()
