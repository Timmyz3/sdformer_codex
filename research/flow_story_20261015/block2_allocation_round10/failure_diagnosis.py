#!/usr/bin/env python3
"""Post-result descriptive algebra on frozen R10; no fit, search or model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PINS = {"collect_v1/results.json": "c31d18f5eca5d4343c02f347d0fbed0c9604120dfb874524f8caf6404b1b005d",
        "analysis_v1/predictions_before_check_evaluation.json": "886ef2829882395c99909688a83bff50b79d9bf48e951b8a75a50a02196e4558",
        "analysis_v1/analysis.json": "c9db420865f867be97457fa3a7dd04e3a954a50ccf0ba05e3a1fdba45e8d38a2"}
ARMS = tuple("fine_r{}_c{}".format(r, c) for r in range(3) for c in range(4))
FAMILIES = ("B5", "rich37", "PCA11", "flow11")
FLOW_NAMES = ("mean_magnitude", "max_magnitude", "std_magnitude", "mean_neighbour_variation",
              "max_neighbour_variation", "norm_mean_vector_over_mean_magnitude")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def ranks(x):
    x = np.asarray(x, dtype=np.float64)
    return np.asarray([np.sum(x < v)+(np.sum(x == v)-1)/2 for v in x])


def rho(x, y):
    a, b = ranks(x), ranks(y)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None


def aee(frame, arm):
    return frame["arms"][arm]["metrics"]["regions"]["all_valid"]["AEE"]


def frozen_score(model, x):
    # Exact frozen FP64-fold -> FP32 inference recipe, no solve/fit occurs here.
    weight = np.asarray(model["beta"])/np.asarray(model["std"])
    bias = model["intercept"]-np.asarray(model["mean"])@weight
    return np.asarray(x, dtype=np.float32)@weight.astype(np.float32)+np.asarray(bias, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--output", type=Path, default=HERE/"failure_diagnosis.json")
    args = parser.parse_args()
    require(not args.output.exists(), "Refuse diagnosis overwrite")
    data = {}
    for name, pin in PINS.items():
        path = args.root/name
        require(sha(path) == pin, "Frozen source differs: "+name)
        data[name] = json.loads(path.read_text())
    raw = data["collect_v1/results.json"]
    pred = data["analysis_v1/predictions_before_check_evaluation.json"]
    analysis = data["analysis_v1/analysis.json"]
    require(raw["status"] == "ROUND10_BLOCK2_SINGLETON_COLLECTION_COMPLETE" and raw["full_forward_count"] == 366,
            "Complete R10 required")
    require(analysis["advancement"]["passed"] is False and pred["input_sha256"] == PINS["collect_v1/results.json"],
            "Frozen negative result/provenance differs")
    result = {"schema": "flow_round10_post_result_failure_diagnosis_v1", "script_sha256": sha(__file__),
        "source_sha256": PINS, "NumPy": np.__version__, "new_models_fitted": 0, "network_forwards": 0,
        "frozen_artifacts_modified": False, "original_advancement_passed": False,
        "role": "R10 implementer post-result diagnosis, not independent review",
        "method": {"R2": "1 - sum((frozen_prediction-target)^2)/sum((target-global_mean_target)^2), all candidates in each group",
                   "rho": "Mean of per-frame Spearman correlations with average tied ranks; frame is analysis unit",
                   "task_target": "Signed relative all-valid gain (CC_AEE-singleton_AEE)/CC_AEE",
                   "MSE_target": "Actual same-u fine/coarse branch MSE",
                   "scores": "Reevaluate frozen coefficients with original folded FP32 inference; verify exact equality to archived check scores",
                   "fit_scope": "Selector training-in diagnostics, despite out-of-fold head features; NOT selector cross-validation or unbiased generalization",
                   "flow_ablation": "FP64 algebra of existing standardized coefficients: zero six flow contribution terms without refitting. Descriptive sensitivity, never a new deployed policy"},
        "group_score_diagnostics": {}, "flow_contribution_diagnostics": {}, "check_frames": []}
    groups = {g: [f for f in raw["frames"] if f["group"] == g] for g in ("fit", "check")}
    for group, frames in groups.items():
        features = pred[group+"_features"]
        task = np.asarray([[(aee(f, "coarse_stage02")-aee(f, arm))/aee(f, "coarse_stage02") for arm in ARMS] for f in frames])
        mse = np.asarray([[{tuple(v["tile"]): v["MSE"] for v in f["offline_fine_coarse_branch_mse"]}[(i//4, i%4)] for i in range(12)] for f in frames])
        summary = {"frames": len(frames), "candidate_rows_not_independent_frames": int(task.size),
            "positive_task_gains": int((task > 0).sum()), "task_target_std": float(task.std()),
            "between_frame_target_mean_std": float(task.mean(1).std()), "mean_within_frame_target_std": float(task.std(1).mean()),
            "CC_AEE_range": [min(aee(f, "coarse_stage02") for f in frames), max(aee(f, "coarse_stage02") for f in frames)], "models": {}}
        for family in FAMILIES:
            for target, truth in (("task", task), ("MSE", mse)):
                name = family+"_"+target
                scores = np.asarray([frozen_score(pred["selector_models"][name], features[f["target_id"]][family]) for f in frames])
                if group == "check":
                    for f, values in zip(frames, scores):
                        require(np.array_equal(values, np.asarray(pred["scores"][name][f["target_id"]], dtype=np.float32)),
                                "Frozen check score replay differs: "+name)
                r2 = 1-float(np.square(scores-truth).sum()/np.square(truth-truth.mean()).sum())
                per_frame = [{"target_id": f["target_id"], "own_target_rho": rho(s, y), "task_rho": rho(s, t)}
                             for f, s, y, t in zip(frames, scores, truth, task)]
                summary["models"][name] = {"own_target_R2": r2,
                    "mean_own_target_rho": float(np.mean([v["own_target_rho"] for v in per_frame])),
                    "mean_task_rho": float(np.mean([v["task_rho"] for v in per_frame])), "frames": per_frame}
        result["group_score_diagnostics"][group] = summary
        model = pred["selector_models"]["flow11_task"]
        beta, mean, std = (np.asarray(model[key]) for key in ("beta", "mean", "std"))
        per_frame = []
        for f in frames:
            z = (np.asarray(features[f["target_id"]]["flow11"])-mean)/std
            base, flow = z[:, :5]@beta[:5], z[:, 5:]@beta[5:]
            full = base+flow
            order_base, order_full = np.argsort(np.argsort(base)), np.argsort(np.argsort(full))
            per_frame.append({"target_id": f["target_id"], "full_top1": ARMS[int(np.argmax(full))],
                "fixed_joint_B5_terms_only_top1": ARMS[int(np.argmax(base))],
                "same_top1": bool(np.argmax(base) == np.argmax(full)),
                "flow_terms_std_over_B5_terms_std": float(flow.std()/base.std()),
                "changed_rank_positions_of12": int((order_base != order_full).sum()),
                "base_vs_full_rho": rho(base, full)})
        ratios = [f["flow_terms_std_over_B5_terms_std"] for f in per_frame]
        result["flow_contribution_diagnostics"][group] = {"unchanged_top1": sum(f["same_top1"] for f in per_frame),
            "frames_count": len(per_frame), "std_ratio_median": float(np.median(ratios)), "std_ratio_range": [min(ratios), max(ratios)],
            "frames": per_frame}
    result["flow_standardized_coefficients"] = dict(zip(FLOW_NAMES, pred["selector_models"]["flow11_task"]["beta"][5:]))
    checks = groups["check"]
    gain = np.asarray([aee(f, "coarse_stage02")-aee(f, pred["selected_arms"]["flow11_task"][f["target_id"]]) for f in checks])
    coarse = np.asarray([aee(f, "coarse_stage02") for f in checks])
    for f, g, c in zip(checks, gain, coarse):
        tid = f["target_id"]
        vals = {name: aee(f, pred["selected_arms"][name][tid]) for name in ("flow11_task", "B5_task", "B5_small4_task", "psn1_activity")}
        singles = [aee(f, arm) for arm in ARMS]
        result["check_frames"].append({"target_id": tid, "CC_AEE": float(c), "policy_AEE": vals,
            "best_actual_k1_AEE": min(singles), "best_actual_k0_or_k1_AEE": min([c]+singles),
            "all12_singleton_mean_AEE": float(np.mean(singles)), "flow11_signed_gain": float(g),
            "fraction_total_CC_AEE": float(c/coarse.sum()), "fraction_total_flow11_signed_gain": float(g/gain.sum()),
            "flow11_B5_small4_same_choice": len({pred["selected_arms"][n][tid] for n in ("flow11_task", "B5_task", "B5_small4_task")}) == 1})
    activity = float(np.mean([f["policy_AEE"]["psn1_activity"] for f in result["check_frames"]]))
    oracle = float(np.mean([f["best_actual_k1_AEE"] for f in result["check_frames"]]))
    oracle0 = float(np.mean([f["best_actual_k0_or_k1_AEE"] for f in result["check_frames"]]))
    result["same_action_upper_bound"] = {"ordinary_activity_AEE": activity, "best_actual_k1_AEE": oracle,
        "k1_relative_improvement_over_activity": (activity-oracle)/activity,
        "best_actual_k0_or_k1_AEE": oracle0, "k0_or_k1_relative_improvement_over_activity": (activity-oracle0)/activity,
        "scope": "Posthoc empirical ceilings over already executed arms on these eight frames only; no generalized impossibility or new selector"}
    reduced = [f for f in result["check_frames"] if f["target_id"] != "zurich_city_05_a_0229"]
    result["influence_only_omit_05a0229_not_primary"] = {"frames": len(reduced),
        "CC_AEE": float(np.mean([f["CC_AEE"] for f in reduced])),
        "flow11_task_AEE": float(np.mean([f["policy_AEE"]["flow11_task"] for f in reduced])),
        "psn1_activity_AEE": float(np.mean([f["policy_AEE"]["psn1_activity"] for f in reduced]))}
    result["cell_flow_head_quality_original"] = analysis["head_quality"]
    result["limitations"] = ["Post-result descriptive computation; no significance or independent review", "Fit selector diagnostics are training-in",
        "Nonzero flow terms changing ranks but not top1 do not establish universal information redundancy",
        "Cell-flow head AEE is not dense final-flow AEE; no matched zero/constant head reference here",
        "Task gain is actual action-aligned supervision; weak predictability does not prove the target is intrinsically wrong",
        "No current native/FF/full-layer result is read by this script", "Different R6/R10 members prevent cross-round causal attribution"]
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output), "new_models_fitted": 0,
                      "network_forwards": 0, "original_advancement_passed": False}))


if __name__ == "__main__":
    main()
