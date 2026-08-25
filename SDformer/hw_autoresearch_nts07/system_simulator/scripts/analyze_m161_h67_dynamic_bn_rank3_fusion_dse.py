#!/usr/bin/env python3
"""DSE the corrected no-running BN1 plus rank-3 ATLIF fusion for H67 FFNs.

The frozen evaluation protocol uses current-batch statistics.  This analyzer
therefore rejects static BN folding and instead derives a streaming moment +
right-projection schedule.  Counts are exact geometry/value/bit-movement
counts under explicit widths; none are admitted cycle or accuracy claims.
"""

import argparse
import ast
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
PATHS = {
    "correction": HW / "contracts/m159_m160_dynamic_bn_semantic_correction_overlay_r1_20260824.json",
    "m159_review": HW / "results/m159_independent_hammer_review_r1_20260824/m159_independent_hammer_review_r1.json",
    "m160_parameter_csv": HW / "results/m160_h67_ffn_bn_atlif_fusion_r1_20260824/per_ffn_bn_atlif_fusion.csv",
    "execution_trace": HW / "results/h67_ep35_full_network_ordered_trace_s10_20260821/execution_trace.csv",
    "m31_rtl": HW / "rtl_m31/qfit_atlif_unified_t10_t2_stream_core.sv",
    "m30_contract": HW / "contracts/m30a_rank3_resident_stream_vcs_contract_r3_20260822.json",
    "profile": ROOT / "neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_nts11_hardware_p0.py",
    "config": ROOT / "neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml",
    "docs359": HW / "docs/359_DATE终局冻结_20260813.md",
}
EXPECTED = {
    "correction": "450b9e5af6f1d2ffc763b65e637d5857e2c433d957a16fe12e74b6d1c82addc5",
    "m159_review": "904fc3737c24ae5e7030abb1f4f4e1f017176ae256976011d78a7f38b29c9410",
    "m160_parameter_csv": "309a5d802c7e49d432285f09ff43b9d1ec797db815b949cd34798c0a94f4f464",
    "execution_trace": "ad8d1f286c0936ce7cf42324068cfd074aeef3cf77af62890e0598b663b91bfd",
    "m31_rtl": "c094849e88c0d9fc3a390d0cf6fc9adf10ff4dc31d77e265e425e5cf71b5ef15",
    "m30_contract": "665d5d803588c006dc85b5c841a0fa8fb6cc7656d00dd38a9a7ace77f44b6b5a",
    "profile": "04f692c5bda6d1f88cdc932ce48f012767f22a2bb1ca161378971232f99c0684",
    "config": "8be3f7bbffd75c4356d3abf5935679d80e15c1caefd307c19a727729659e6c49",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
T = 10
RANK = 3
M31_LANES = 16
M31_PRODUCTS = 96
Q8_BITS = 8
Q24_BITS = 24
DENSE_SN2_CYCLES = 36_480_000


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
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(
            handle, object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                RuntimeError("non-finite JSON: " + value)))


def ceil_div(numerator, denominator):
    return (int(numerator) + int(denominator) - 1) // int(denominator)


def signed_bits_for_magnitude(magnitude):
    magnitude = int(magnitude)
    require(magnitude >= 0, "negative magnitude")
    bits = 1
    while magnitude > (1 << (bits - 1)) - 1:
        bits += 1
    return bits


def unsigned_bits(maximum):
    maximum = int(maximum)
    require(maximum >= 0, "negative unsigned maximum")
    return max(1, maximum.bit_length())


def dynamic_bn_rank_miter():
    rng = np.random.RandomState(161)
    maximum = 0.0
    for _ in range(100):
        positions = int(rng.randint(3, 41))
        x = rng.normal(size=(T, positions)).astype(np.float64)
        right = rng.normal(size=(RANK, T)).astype(np.float64)
        gamma = float(rng.uniform(0.1, 4.0))
        beta = float(rng.uniform(-2.0, 2.0))
        epsilon = float(rng.choice([1.0e-5, 1.0e-4]))
        mean = float(np.mean(x))
        variance = float(np.var(x))
        alpha = gamma / math.sqrt(variance + epsilon)
        offset = beta - alpha * mean
        reference = np.matmul(right, alpha * x + offset)
        transformed = alpha * np.matmul(right, x) + offset * np.sum(
            right, axis=1)[:, None]
        maximum = max(maximum, float(np.max(np.abs(reference - transformed))))
    require(maximum <= 2.0e-14, "dynamic BN/rank algebra miter failed")
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    require(not output.exists(), "refusing to overwrite M161 output")
    script_path = Path(__file__).resolve()
    script_start = sha256(script_path)
    observed = {label: sha256(path) for label, path in PATHS.items()}
    require(observed == EXPECTED, "M161 frozen input identity drift")

    correction = strict_json(PATHS["correction"])
    review = strict_json(PATHS["m159_review"])
    config = yaml.safe_load(PATHS["config"].read_text(encoding="utf-8"))
    require(correction["frozen_protocol"]["bn_policy"] == "no_running",
            "correction BN policy drift")
    require(review["corrected_topology_and_protocol"]["frozen_trace_bn_semantics"].startswith(
        "no_running/"), "independent review BN policy drift")
    require(config["test"]["bn_policy"] == "no_running" and
            int(config["test"]["eval_batch_size"]) == 1,
            "config BN protocol drift")
    profile_text = PATHS["profile"].read_text(encoding="utf-8")
    for fragment in (
        "configure_batch_norm_evaluation(model, bn_policy)",
        "module.track_running_stats = False",
        "module.running_mean = None",
        "module.running_var = None",
    ):
        require(fragment in profile_text, "profile BN protocol source drift")
    m31_text = PATHS["m31_rtl"].read_text(encoding="utf-8")
    for fragment in (
        "T10_RANK = 3", "T10_LANES = 16", "MULTIPLIERS = 96",
        "t10_intermediate_q", "rne_sat_q24_to_q8",
    ):
        require(fragment in m31_text, "M31 resource/source drift")
    m30 = strict_json(PATHS["m30_contract"])
    require(m30["resources"]["signed_int8_multiplier_slots"] == 96 and
            m30["resources"]["rank"] == RANK and
            m30["resources"]["lanes"] == M31_LANES,
            "M30 resource contract drift")

    parameter_rows = {}
    with PATHS["m160_parameter_csv"].open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parameter_rows[row["module"]] = row
    require(len(parameter_rows) == 12, "M160 checkpoint parameter population drift")

    trace_rows = []
    all_shapes = defaultdict(set)
    with PATHS["execution_trace"].open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if ".mlp.fc1" not in row["name"]:
                continue
            prefix = row["name"][:-len(".fc1")]
            shape = tuple(int(value) for value in ast.literal_eval(row["output_shape"]))
            all_shapes[prefix].add(shape)
            if int(row["sample_id"]) == 0:
                trace_rows.append((prefix, row, shape))
    require(len(all_shapes) == 12 and all(len(values) == 1 for values in all_shapes.values()),
            "FFN geometry changed across ten samples")
    require(len(trace_rows) == 12 and set(parameter_rows) == set(all_shapes),
            "trace/checkpoint FFN identity mismatch")

    module_rows = []
    stage_rows = defaultdict(lambda: {
        "blocks": 0, "bn1_elements": 0, "columns": 0,
        "baseline_bit_movements": 0,
        "exact_rank3_bit_movements": 0,
        "q8_rank3_bit_movements": 0,
        "baseline_barrier_bits": 0,
        "exact_rank3_barrier_bits": 0,
        "q8_rank3_barrier_bits": 0,
        "moment_state_bits": 0,
    })
    total_projection_stage1_cycles = 0
    total_bn1_elements = 0

    for prefix, trace, shape in trace_rows:
        require(len(shape) == 5 and shape[0] == T and shape[1] == 1,
                "FFN BN1 logical shape drift")
        stage = int(prefix.split(".layers.")[1].split(".")[0])
        expanded = int(shape[-1])
        positions = int(shape[1] * shape[2] * shape[3])
        reduction_population = T * positions
        elements = int(trace["output_elements"])
        columns = positions * expanded
        require(elements == T * columns, "BN1 element/column arithmetic drift")
        parameter = parameter_rows[prefix]
        require(int(parameter["expanded_channels"]) == expanded,
                "checkpoint/trace hidden geometry drift")
        dot_bound = int(parameter["fc1_int8_sumabs_max"])
        raw_bits = signed_bits_for_magnitude(dot_bound)
        sum_bound = reduction_population * dot_bound
        sumsq_bound = reduction_population * dot_bound * dot_bound
        sum_bits = signed_bits_for_magnitude(sum_bound)
        sumsq_bits = unsigned_bits(sumsq_bound)
        moment_bits = expanded * (sum_bits + sumsq_bits)

        # Baseline dynamic-BN movement model per spatial/hidden column:
        # raw write + statistics read + normalize read at raw accumulator width,
        # then normalized write + ATLIF read at Q8.
        baseline_bits = columns * (3 * T * raw_bits + 2 * T * Q8_BITS)
        # Corrected fusion stores the R-wide right-projection state across the
        # global moment barrier, then reads it once after alpha/beta are ready.
        exact_rank_bits = columns * 2 * RANK * Q24_BITS
        q8_rank_bits = columns * 2 * RANK * Q8_BITS
        baseline_barrier = columns * T * raw_bits
        exact_rank_barrier = columns * RANK * Q24_BITS
        q8_rank_barrier = columns * RANK * Q8_BITS
        stage1_cycles = ceil_div(columns, M31_LANES) * (T // 2)
        total_projection_stage1_cycles += stage1_cycles
        total_bn1_elements += elements

        row = {
            "module": prefix,
            "stage": stage,
            "shape": list(shape),
            "expanded_channels": expanded,
            "positions_per_channel": positions,
            "dynamic_bn_reduction_population_per_channel": reduction_population,
            "bn1_elements": elements,
            "spatial_hidden_columns": columns,
            "fc1_int8_dot_sumabs_bound": dot_bound,
            "raw_accumulator_signed_bits": raw_bits,
            "moment_signed_sum_bits": sum_bits,
            "moment_unsigned_sumsq_bits": sumsq_bits,
            "moment_state_bits": moment_bits,
            "baseline_dynamic_bn_bit_movements": baseline_bits,
            "exact_q24_rank3_bit_movements": exact_rank_bits,
            "train_required_q8_rank3_bit_movements": q8_rank_bits,
            "baseline_raw_barrier_storage_bits": baseline_barrier,
            "exact_q24_rank3_barrier_storage_bits": exact_rank_barrier,
            "train_required_q8_rank3_barrier_storage_bits": q8_rank_barrier,
            "rank3_right_projection_cycles_m31_geometry": stage1_cycles,
        }
        module_rows.append(row)
        bucket = stage_rows[stage]
        bucket["blocks"] += 1
        bucket["bn1_elements"] += elements
        bucket["columns"] += columns
        bucket["baseline_bit_movements"] += baseline_bits
        bucket["exact_rank3_bit_movements"] += exact_rank_bits
        bucket["q8_rank3_bit_movements"] += q8_rank_bits
        bucket["baseline_barrier_bits"] += baseline_barrier
        bucket["exact_rank3_barrier_bits"] += exact_rank_barrier
        bucket["q8_rank3_barrier_bits"] += q8_rank_barrier
        bucket["moment_state_bits"] += moment_bits

    require(total_bn1_elements == 350_208_000, "M161 BN1 extent drift")
    require(total_projection_stage1_cycles == 10_944_000,
            "M31 rank3 right-projection issue arithmetic drift")
    baseline_movement = sum(row["baseline_dynamic_bn_bit_movements"] for row in module_rows)
    exact_movement = sum(row["exact_q24_rank3_bit_movements"] for row in module_rows)
    q8_movement = sum(row["train_required_q8_rank3_bit_movements"] for row in module_rows)
    baseline_storage = sum(row["baseline_raw_barrier_storage_bits"] for row in module_rows)
    exact_storage = sum(row["exact_q24_rank3_barrier_storage_bits"] for row in module_rows)
    q8_storage = sum(row["train_required_q8_rank3_barrier_storage_bits"] for row in module_rows)
    maximum_block_baseline = max(row["baseline_raw_barrier_storage_bits"] for row in module_rows)
    maximum_block_exact = max(row["exact_q24_rank3_barrier_storage_bits"] for row in module_rows)
    maximum_block_q8 = max(row["train_required_q8_rank3_barrier_storage_bits"] for row in module_rows)

    moment_dse = []
    for lanes in (8, 16, 32, 48, 96):
        cycles = sum(ceil_div(row["bn1_elements"], lanes) for row in module_rows)
        moment_dse.append({
            "square_sum_lanes": lanes,
            "ideal_vector_issues": cycles,
            "relative_to_rank3_right_projection_issues": (
                float(cycles) / total_projection_stage1_cycles),
            "can_fit_under_rank3_right_projection_issue_count":
                cycles <= total_projection_stage1_cycles,
        })
    require(next(row for row in moment_dse if row["square_sum_lanes"] == 32)[
        "ideal_vector_issues"] == total_projection_stage1_cycles,
        "32-lane balanced moment geometry drift")

    miter_error = dynamic_bn_rank_miter()
    stage_payload = []
    for stage in range(4):
        row = dict(stage_rows[stage])
        row["stage"] = stage
        row["exact_movement_reduction"] = (
            float(row["baseline_bit_movements"]) /
            row["exact_rank3_bit_movements"])
        row["q8_movement_reduction_train_required"] = (
            float(row["baseline_bit_movements"]) /
            row["q8_rank3_bit_movements"])
        stage_payload.append(row)

    output.mkdir(parents=True)
    csv_path = output / "per_ffn_dynamic_bn_rank3_dse.csv"
    fields = [
        "module", "stage", "shape", "expanded_channels",
        "positions_per_channel", "dynamic_bn_reduction_population_per_channel",
        "bn1_elements", "spatial_hidden_columns", "fc1_int8_dot_sumabs_bound",
        "raw_accumulator_signed_bits", "moment_signed_sum_bits",
        "moment_unsigned_sumsq_bits", "moment_state_bits",
        "baseline_dynamic_bn_bit_movements", "exact_q24_rank3_bit_movements",
        "train_required_q8_rank3_bit_movements",
        "baseline_raw_barrier_storage_bits",
        "exact_q24_rank3_barrier_storage_bits",
        "train_required_q8_rank3_barrier_storage_bits",
        "rank3_right_projection_cycles_m31_geometry",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in module_rows:
            encoded = dict(row)
            encoded["shape"] = json.dumps(encoded["shape"], separators=(",", ":"))
            writer.writerow(encoded)

    payload = {
        "schema": "m161_h67_dynamic_bn_rank3_fusion_dse_v1",
        "status": "PASS_DYNAMIC_BN_RANK3_ALGEBRA_AND_BIT_MOVEMENT_DSE",
        "identity": {
            "analyzer_start_end_sha256": script_start,
            "inputs_sha256": observed,
            "checkpoint_parameter_reuse_qualification": (
                "Only fc1 checkpoint INT8 sumabs bounds are reused from M160 r1; "
                "all running-stat BN fold fields remain revoked."
            ),
        },
        "frozen_semantics": {
            "bn_policy": "no_running/current-batch",
            "eval_batch_size": 1,
            "ffn_bn1_elements_per_frame": total_bn1_elements,
            "ffn_bn2_elements_per_frame": 87_552_000,
            "global_moment_barrier_per_module": True,
            "static_bn_fold": False,
        },
        "exact_transform": {
            "equations": [
                "mu[j]=mean_{t,p}(x[t,p,j])",
                "alpha[j]=gamma[j]/sqrt(var[j]+eps)",
                "offset[j]=beta[j]-alpha[j]*mu[j]",
                "v[r,p,j]=sum_t R[r,t]*x[t,p,j]",
                "v_bn[r,p,j]=alpha[j]*v[r,p,j]+offset[j]*sum_t R[r,t]",
                "h[t,p,j]=sum_r L[t,r]*v_bn[r,p,j]+bias[t]",
            ],
            "float64_random_miter_trials": 100,
            "maximum_abs_error": miter_error,
            "scope": (
                "Exact real-valued algebra for a rank-factored temporal matrix. "
                "It does not prove rank-3 model accuracy or fixed-point order equivalence."
            ),
        },
        "balanced_streaming_candidate": {
            "right_projection": {
                "T": T, "rank": RANK, "lanes": M31_LANES,
                "shared_int8_product_slots": M31_PRODUCTS,
                "values_consumed_per_issue": 32,
                "issues_per_16_column_tile": 5,
                "issues_per_frame": total_projection_stage1_cycles,
            },
            "moment_lane_dse": moment_dse,
            "balanced_choice": (
                "32 square+sum lanes consume the same 32 raw fc1 values/cycle as "
                "the 96-slot rank-3 right projection, so moment updates have no "
                "additional ideal issue count. Area, timing, input readiness and the "
                "global barrier remain unproved."
            ),
            "rank3_full_projection_ideal_issues": 2 * total_projection_stage1_cycles,
            "dense_sn2_issue_cycles": DENSE_SN2_CYCLES,
            "conditional_sn2_arithmetic_speedup": (
                float(DENSE_SN2_CYCLES) / (2 * total_projection_stage1_cycles)),
        },
        "bit_movement_dse": {
            "baseline_contract": (
                "Per column: three T-wide raw-accumulator movements (write, stats "
                "read, normalize read) plus two T-wide Q8 movements (normalized "
                "write, ATLIF read)."
            ),
            "exact_rank3_contract": (
                "Per column: one Q24 rank-R write before the moment barrier and one "
                "Q24 rank-R read after alpha/offset commit."
            ),
            "train_required_q8_contract": (
                "Same two rank-R movements at Q8; early requantization changes "
                "rounding order and requires PAFT/validation."
            ),
            "baseline_bits_per_frame": baseline_movement,
            "exact_q24_rank3_bits_per_frame": exact_movement,
            "q8_rank3_bits_per_frame_train_required": q8_movement,
            "exact_bit_movement_reduction": float(baseline_movement) / exact_movement,
            "q8_bit_movement_reduction_train_required": float(baseline_movement) / q8_movement,
            "baseline_barrier_storage_bits_sum_of_12_modules": baseline_storage,
            "exact_q24_rank3_barrier_storage_bits_sum_of_12_modules": exact_storage,
            "q8_rank3_barrier_storage_bits_sum_train_required": q8_storage,
            "baseline_peak_single_block_storage_bits": maximum_block_baseline,
            "exact_q24_peak_single_block_storage_bits": maximum_block_exact,
            "q8_peak_single_block_storage_bits_train_required": maximum_block_q8,
            "exact_peak_storage_reduction": float(maximum_block_baseline) / maximum_block_exact,
            "q8_peak_storage_reduction_train_required": float(maximum_block_baseline) / maximum_block_q8,
            "qualification": (
                "These are local intermediate-buffer bit counts, not SRAM/DRAM "
                "transactions, cycles, energy, or system speedup."
            ),
        },
        "stage_rows": stage_payload,
        "bn2_remaining_problem": {
            "elements_per_frame": 87_552_000,
            "reason": (
                "BN2 is also current-batch and has no following rank contraction. "
                "fc2 output must be buffered/replayed for normalize+residual commit, "
                "recomputed, or removed by an admitted deployment-normalization change."
            ),
        },
        "algorithm_feedback": {
            "preferred": (
                "Fine-tune/calibrate a deployment checkpoint with frozen running BN, "
                "then validate against the no-running baseline; successful conversion "
                "unlocks true static BN1/BN2 folding and removes both global barriers."
            ),
            "fallback": (
                "Train the rank-3 path with dynamic BN correction applied to Q24 right "
                "projection state; optionally train Q8 early-requantization for the "
                "larger movement reduction."
            ),
            "tile_norm_option": (
                "A hardware-tile normalization policy can bound the barrier/storage, "
                "but changes network semantics and requires explicit training."
            ),
        },
        "admission": {
            "dynamic_bn_rank3_real_algebra": True,
            "geometry_and_bit_counts": True,
            "moment_lane_issue_sensitivity": True,
            "rank3_trained_accuracy": False,
            "fixed_point_equivalence": False,
            "buffer_address_schedule": False,
            "rtl": False,
            "vcs": False,
            "cycle_speedup": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
    }
    result_path = output / "m161_h67_dynamic_bn_rank3_fusion_dse.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    require(sha256(script_path) == script_start, "M161 analyzer changed during run")
    print(json.dumps({
        "status": payload["status"],
        "bn1_elements": total_bn1_elements,
        "right_projection_issues": total_projection_stage1_cycles,
        "balanced_moment_lanes": 32,
        "exact_bit_movement_reduction": payload["bit_movement_dse"]["exact_bit_movement_reduction"],
        "q8_bit_movement_reduction_train_required": payload["bit_movement_dse"]["q8_bit_movement_reduction_train_required"],
        "conditional_sn2_speedup": payload["balanced_streaming_candidate"]["conditional_sn2_arithmetic_speedup"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
