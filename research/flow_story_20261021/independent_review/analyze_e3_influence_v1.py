"""Descriptive influence review of the frozen E3 all-GT oracle, no new policy."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "E3_ACTUAL_RAW_QUALITY_V1.json"
d = json.loads(SOURCE.read_text())
frames = d["frames"]
regions = list(frames[0]["states"]["background_Q4"])
sequences = sorted({f["sequence"] for f in frames})


def pooled(fs, state, region):
    n = sum(f["states"][state][region]["valid_pixels"] for f in fs)
    e = sum(f["states"][state][region]["error_sum"] for f in fs)
    return {"valid_pixels": n, "error_sum": e, "AEE": e / n if n else None}


def compare(fs, region):
    a = pooled(fs, "single_action_oracle", region)
    b = pooled(fs, "background_Q4", region)
    return {"oracle": a, "Q4": b,
            "delta_AEE": a["AEE"] - b["AEE"] if a["AEE"] is not None else None,
            "relative_change": a["AEE"] / b["AEE"] - 1 if a["AEE"] is not None and b["AEE"] else None,
            "error_sum_gain": b["error_sum"] - a["error_sum"]}


total_gain = compare(frames, "all_valid")["error_sum_gain"]
seq = {}
leave_one = {}
for s in sequences:
    fs = [f for f in frames if f["sequence"] == s]
    seq[s] = {r: compare(fs, r) for r in ["all_valid", "B", "L"]}
    seq[s]["allGT_error_gain_share"] = seq[s]["all_valid"]["error_sum_gain"] / total_gain
    leave_one[s] = {r: compare([f for f in frames if f["sequence"] != s], r)
                    for r in ["all_valid", "B", "L"]}

per_frame = []
for f in frames:
    t = f["target_id"]
    c = d["selected_action_coverage"][t]["single_action_oracle"]
    per_frame.append({"target_id": t, "sequence": f["sequence"],
                      "action": d["policy_choices"][t]["single_action_oracle"],
                      "oracle_vs_Q4": {r: compare([f], r) for r in ["all_valid", "B", "L"]},
                      "selected_coverage": c})

coverage = {}
for k in ["image_pixels", "valid_GT_pixels", "voxel_active_pixels", "voxel_zero_pixels",
          "nonzero_voxel_bins", "valid_GT_voxel_active_pixels", "valid_GT_voxel_zero_pixels"]:
    vals = [x["selected_coverage"][k] for x in per_frame]
    coverage[k] = {"sum": sum(vals), "min": min(vals), "max": max(vals)}
coverage["valid_GT_fraction_of_selected_pooled"] = coverage["valid_GT_pixels"]["sum"] / coverage["image_pixels"]["sum"]
coverage["voxel_active_fraction_of_selected_pooled"] = coverage["voxel_active_pixels"]["sum"] / coverage["image_pixels"]["sum"]
coverage["valid_GT_fraction_of_frame_min_max"] = [min(x["selected_coverage"]["valid_GT_fraction_of_frame"] for x in per_frame), max(x["selected_coverage"]["valid_GT_fraction_of_frame"] for x in per_frame)]

means = {s: pooled(frames, s, "all_valid")["AEE"] for s in ["background_Q4", "single_action_oracle", "fit_fold_static", "in_sample_static_diagnostic", "uniform_upgrade_expectation", "endpoint_M", "endpoint_Q8"]}
oracle = means["single_action_oracle"]
quality_budget = {"max_additional_selector_regret_to_tie_Q4_px": means["background_Q4"] - oracle,
                  "max_additional_selector_regret_to_tie_M_px": means["endpoint_M"] - oracle,
                  "max_additional_selector_regret_to_tie_best_static_px": means["in_sample_static_diagnostic"] - oracle,
                  "uniform_to_oracle_gap_px": means["uniform_upgrade_expectation"] - oracle,
                  "fraction_uniform_to_oracle_gap_required_to_tie_Q4": (means["uniform_upgrade_expectation"] - means["background_Q4"]) / (means["uniform_upgrade_expectation"] - oracle),
                  "interpretation": "Descriptive exact quality margins on these exposed 16 identities, not a prespecified acceptance threshold or an estimate of generalization; no runtime cost is discounted."}
out = {"schema": "flow_e3_influence_review_v1", "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
       "method": "Frozen all-GT oracle; no change of action or objective. Leave-one results only remove each sequence from aggregation without refitting/reselecting.",
       "all_regions_oracle_vs_Q4": {r: compare(frames, r) for r in regions},
       "per_sequence": seq, "leave_one_sequence_out_aggregation": leave_one,
       "frames": per_frame, "oracle_action_counts": dict(collections.Counter(x["action"] for x in per_frame)),
       "frame_allGT_improve_count": sum(x["oracle_vs_Q4"]["all_valid"]["delta_AEE"] < 0 for x in per_frame),
       "frame_allGT_worse_count": sum(x["oracle_vs_Q4"]["all_valid"]["delta_AEE"] > 0 for x in per_frame),
       "selected_coverage": coverage, "quality_budget": quality_budget,
       "uniform_vs_Q4_nonzero_tail_frames": [f for f in d["uniform_action_tail_risk"]["background_Q4"]["frames"] if f["probability_any_main_region_tail"] > 0]}
(ROOT / "E3_INFLUENCE_REVIEW_V1.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({k: out[k] for k in ["frame_allGT_improve_count", "frame_allGT_worse_count", "oracle_action_counts", "selected_coverage", "quality_budget", "uniform_vs_Q4_nonzero_tail_frames"]}, indent=2))
print("sequence / allGT delta / gain share / leave-one relative / B delta / L delta")
for s in sequences:
    print(s, seq[s]["all_valid"]["delta_AEE"], seq[s]["allGT_error_gain_share"], leave_one[s]["all_valid"]["relative_change"], seq[s]["B"]["delta_AEE"], seq[s]["L"]["delta_AEE"])
print("region / support / delta / relative")
for r, v in out["all_regions_oracle_vs_Q4"].items():
    print(r, v["Q4"]["valid_pixels"], v["delta_AEE"], v["relative_change"])
