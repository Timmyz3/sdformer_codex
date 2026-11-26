"""Independent NumPy reconstruction of all frozen E3 response diagnostic choices."""
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(HERE))
import reduce_e1_quality_v1 as prior


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def near(a, b, tol=2e-12):
    assert np.allclose(a, b, rtol=tol, atol=tol), (a, b)


def means_by_slices(v, valid, g):
    """Explicit spatial slices, independently of implementation's reshape axes."""
    h, w = valid.shape
    counts = np.zeros((g, g), dtype=np.int64)
    values = np.zeros((2, g, g), dtype=np.float64)
    for y in range(g):
        for x in range(g):
            sy, sx = slice(y*h//g, (y+1)*h//g), slice(x*w//g, (x+1)*w//g)
            m = valid[sy, sx]
            counts[y, x] = int(m.sum())
            if counts[y, x]:
                values[:, y, x] = v[:, sy, sx][:, m].mean(axis=1)
    return counts, values


def average_rank(v):
    v = np.asarray(v)
    return np.array([1 + int((v < x).sum()) + (int((v == x).sum()) - 1)/2 for x in v])


def ranks_review(scores, truth):
    a, b = average_rank(scores), average_rank(truth)
    rho = float(np.corrcoef(a, b)[0, 1]) if np.ptp(a) and np.ptp(b) else None
    pairs = []
    pt = at = bt = 0
    for i in range(6):
        for j in range(i+1, 6):
            s, t = np.sign(scores[i]-scores[j]), np.sign(truth[i]-truth[j])
            pt += int(s == 0); at += int(t == 0); bt += int(s == t == 0)
            pairs.append(1 if s == t else .5 if s == 0 or t == 0 else 0)
    choice, best = int(np.argmin(scores)), int(np.argmin(truth))
    return {"spearman": rho, "pairwise_agreement": sum(pairs)/15,
            "predicted_tie_pairs": pt, "actual_tie_pairs": at, "both_tie_pairs": bt,
            "picked_action": choice, "exact_oracle_action": best,
            "hit_oracle_action": choice == best, "hit_best_true_value": truth[choice] == min(truth),
            "selected_actual_delta_AEE": truth[choice], "regret_AEE": truth[choice]-truth[best],
            "predicted_order": sorted(range(6), key=lambda i: (scores[i], i)),
            "actual_order": sorted(range(6), key=lambda i: (truth[i], i))}


folder = BASE / "e3_response_diagnostic_v1"
bindings = read(folder / "SOURCE_BINDINGS_PRE_RUN_V1.json")
for rel, expected in bindings["files"].items():
    assert sha(ROOT/rel) == expected
d = read(folder / "RESULTS_V1.json")
assert d["status"] == "COMPLETE" and d["model_forwards"] == 0 and not d["GPU_used"] and not d["head_fitting"]
assert d["sources"]["pre_run_bindings_sha256"] == sha(folder / "SOURCE_BINDINGS_PRE_RUN_V1.json")
run = BASE/"actual/e3_rne_probe_v1"
result = read(run/"results.json")
ids = result["selected_ids"]
acts = [f"upgrade_{i}" for i in range(6)]
policies = ["response_norm", "signed_full", "signed_grid4", "signed_grid1", "activity_max_bins"]
members = read(prior.R27/"independent_review/membership_v1.json")
records = {r["target_id"]: r for r in members["records"]}
pred_paths, source_paths = {}, {}
for state in ["background_Q4"] + acts:
    for row in result["evaluations"][state]["frames"]:
        pred_paths[state, row["target_id"]] = prior.prediction(run, row)
for tid in ids:
    source_paths[tid] = {}
    for x in records[tid]["source_files"]:
        p = prior.R27/"dev64_source_snapshot"/x["relative_path"]
        assert sha(p) == x["sha256"] and p.stat().st_size == int(x["bytes"])
        source_paths[tid][x["category"]] = p
# All input and output bytes bound before any GT array is loaded.
regions_path = ROOT/"research/flow_story_20261016/coverage_recovery_round13/preparation/regions_v1.json"
assert sha(regions_path) == prior.REGIONS_SHA
regions = read(regions_path)
details = {x["target_id"]: x for x in d["action_diagnostics"]}
diagframes = {x["target_id"]: x for x in d["frames"]}
measured_frames = []
max_score_difference = 0.0
for tid in ids:
    gt = np.load(source_paths[tid]["gt"], allow_pickle=False)
    valid = np.load(source_paths[tid]["mask"], allow_pickle=False)
    events = np.load(source_paths[tid]["input"], allow_pickle=False)
    pred = {}
    for state in ["background_Q4"] + acts:
        with np.load(pred_paths[state, tid], allow_pickle=False) as z:
            pred[state] = z["head3"].astype(np.float64)
    f0 = pred["background_Q4"]
    err = f0 - gt.astype(np.float64)
    length = np.hypot(err[0], err[1])
    unit = err / np.maximum(length, 1e-12)[None]
    n = int(valid.sum())
    truth, scores = [], {p: [] for p in policies}
    region_masks = prior.raw.region_masks(gt, valid, events, regions["boundary_threshold_pixels"], regions["large_displacement_threshold_pixels"])
    metrics = {}
    for state in ["background_Q4"] + acts:
        error = pred[state] - gt.astype(np.float64)
        norm = np.hypot(error[0], error[1])
        metrics[state] = {}
        for r, mask in region_masks.items():
            count = int(mask.sum()); total = float(norm[mask].sum())
            metrics[state][r] = {"valid_pixels": count, "error_sum": total, "AEE": total/count if count else None}
        if state == "background_Q4":
            continue
        i = acts.index(state)
        truth.append(metrics[state]["all_valid"]["AEE"]-metrics["background_Q4"]["all_valid"]["AEE"])
        delta = pred[state] - f0
        scores["response_norm"].append(float(np.hypot(delta[0], delta[1])[valid].mean()))
        scores["signed_full"].append(float((unit[:, valid]*delta[:, valid]).sum(axis=0).mean()))
        for g in [4, 1]:
            counts, u = means_by_slices(unit, valid, g)
            counts2, delta_mean = means_by_slices(delta, valid, g)
            assert np.array_equal(counts, counts2)
            value = math.fsum(int(counts[y, x]) * float(np.dot(u[:, y, x], delta_mean[:, y, x])) for y in range(g) for x in range(g))/n
            scores[f"signed_grid{g}"].append(value)
            saved = details[tid][f"grid{g}"]
            assert np.array_equal(counts, saved["support_counts"])
            near(u, saved["unit_direction_means"])
            near(delta_mean, saved["response_means_by_action"][i])
        # Tile slicing directly in input resolution. No GT mask/validity involved.
        first = result["action_contract"]["family_targets"][0]
        bins = 0
        for y, x in result["action_contract"]["groups"][i]["tiles"][first]:
            bins += int(np.count_nonzero(events[:, 18*y:18*(y+1), 18*x:18*(x+1)]))
        scores["activity_max_bins"].append(-bins)
    near(truth, details[tid]["true_delta_AEE"])
    policy_rows = {}
    for policy in policies:
        near(scores[policy], details[tid]["scores"][policy])
        max_score_difference = max(max_score_difference, float(np.max(np.abs(np.array(scores[policy])-details[tid]["scores"][policy]))))
        rank = ranks_review(scores[policy], truth)
        for k, v in rank.items():
            wanted = details[tid]["rank_metrics"][policy][k]
            if isinstance(v, float):
                near(v, wanted)
            else:
                assert v == wanted, (tid, policy, k, v, wanted)
        m = metrics[acts[rank["picked_action"]]]
        for r in m:
            target = diagframes[tid]["states"][policy][r]
            assert m[r]["valid_pixels"] == target["valid_pixels"]
            near(m[r]["error_sum"], target["error_sum"])
            if m[r]["AEE"] is not None:
                near(m[r]["AEE"], target["AEE"])
        policy_rows[policy] = {"chosen_action": acts[rank["picked_action"]], "rank": rank, "regions": m}
    measured_frames.append({"target_id": tid, "sequence": diagframes[tid]["sequence"], "policies": policy_rows})
    print(tid, "PASS", flush=True)

# Synthetic implementation checks use explicit independent expectations, including
# a covariance counterexample that distinguishes pool-then-contract from its reverse.
spec = importlib.util.spec_from_file_location("diagnostic_implementation", folder/"analyze.py")
impl = importlib.util.module_from_spec(spec); spec.loader.exec_module(impl)
v = np.zeros((2, 8, 8)); v[0] = np.where(np.indices((8, 8)).sum(axis=0) % 2, -1., 1.)
mask = np.ones((8, 8), dtype=bool)
c, mean = impl.grid_labels(v, mask, 4)
assert c.sum() == 64 and np.count_nonzero(mean) == 0
assert impl.contraction(c, mean, mean) == 0.0
assert float(np.mean(np.sum(v*v, axis=0))) == 1.0
mask[:2, :2] = False
v[:, :2, :2] = 1e8
c, mean = impl.grid_labels(v, mask, 4)
assert c[0, 0] == 0 and np.count_nonzero(mean[:, 0, 0]) == 0
expected_count, expected_mean = means_by_slices(v, mask, 4)
near(mean, expected_mean); assert np.array_equal(c, expected_count)
near(impl.ranks([1, 1, 1, 4, 2, 2]), [2, 2, 2, 6, 4.5, 4.5])
tie = impl.rank_metrics([0]*6, [0, 1, 2, 3, 4, 5])
assert tie["spearman"] is None and tie["picked_action"] == 0 and tie["pairwise_agreement"] == .5
tie2 = impl.rank_metrics([0]*6, [0]*6)
assert tie2["spearman"] is None and tie2["pairwise_agreement"] == 1

summary = {}
for policy in policies:
    summary[policy] = {}
    for scope in ["overall"] + sorted({f["sequence"] for f in measured_frames}):
        fs = [f for f in measured_frames if scope == "overall" or f["sequence"] == scope]
        by_region = {}
        for region in prior.mathcore.REGIONS:
            rows = [f["policies"][policy]["regions"][region] for f in fs]
            n = sum(r["valid_pixels"] for r in rows)
            total = math.fsum(r["error_sum"] for r in rows)
            avg = total/n if n else None
            target = d["summaries"][policy][scope][region]
            assert n == target["valid_pixels"]
            if avg is not None:
                near(avg, target["pixel_weighted_AEE"])
            by_region[region] = {"valid_pixels": n, "pixel_weighted_AEE": avg}
        regs = [f["policies"][policy]["rank"] for f in fs]
        supports = [f["policies"][policy]["regions"]["all_valid"]["valid_pixels"] for f in fs]
        regret = math.fsum(n*r["regret_AEE"] for n, r in zip(supports, regs))/sum(supports)
        near(regret, d["rank_summary"][policy][scope]["regret_pixel_weighted_AEE"])
        hits = sum(r["hit_oracle_action"] for r in regs)
        assert hits == d["rank_summary"][policy][scope]["exact_oracle_action_hits"]
        summary[policy][scope] = {"regions": by_region, "regret_pixel_weighted_AEE": regret, "oracle_hits": hits}

out = {"schema": "flow_e3_response_independent_audit_v1", "status": "PASS",
       "diagnostic_result_sha256": sha(folder/"RESULTS_V1.json"), "diagnostic_prerun_bindings_sha256": sha(folder/"SOURCE_BINDINGS_PRE_RUN_V1.json"),
       "auditor_sha256": sha(__file__), "new_model_forwards": 0, "GPU_used": False,
       "independently_read_original_predictions": 112, "independently_checked_source_arrays": 48,
       "checked_action_choices": 80, "checked_frames": 16, "checked_regions_per_policy": 16,
       "score_and_rank_reconstruction": "Independent explicit spatial slices, np.hypot norms, direct event tile counts, average ranks by counting ties; production contraction/score functions not called on real data.",
       "max_absolute_score_difference": max_score_difference,
       "synthetic_checks": ["separate pooling covariance counterexample", "invalid cell exclusion and zero-cell means", "tie average ranks", "constant-score Spearman null", "one-sided ties half agreement", "both ties full agreement"],
       "runtime_scope": "All response scores require six counterfactual outputs; signed scores additionally require GT values; norm needs GT valid support. Activity score uses full footprint event bins without GT gating, but no online cost is modeled.",
       "summaries": summary, "frames": measured_frames}
(HERE/"E3_RESPONSE_INDEPENDENT_AUDIT_V1.json").write_text(json.dumps(out, indent=2, allow_nan=False)+"\n")
print(json.dumps({p: summary[p]["overall"] for p in policies}, indent=2))
