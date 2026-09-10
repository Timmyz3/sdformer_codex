#!/opt/anaconda3/bin/python3.12
"""Depth-4 regression-tree core: raw theta*g SSE and distinct terminal IDs.

This supersedes the Gini/majority-label probe as the stronger tree-core
comparison; its files remain intact.  Each complete depth-4 tree has 16
terminal identities.  Splits minimize weighted within-child SSE of the
original 16 theta*g coordinates.  Fixed trees share one j per depth;
candidate trees choose j at each node.  With binary gates the only
nonconstant threshold is theta/2, implemented as a bit test.  Features do
not repeat on a path.  Both axes use exactly four tests, including nodes
whose training subset has zero variance or is empty; ties choose the
lowest unused j.  Empty leaves have zero centroid and are reported.

Centroid reconstruction and FC2 centroid-table errors are only proxies.
The terminal IDs, not rounded centroids or majority labels, are exported
for an independent identical-ridge code-table fit.  No ridge fitting,
fine-tuning, AEE, producer scheduling, or RTL is performed in this script.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import time
from pathlib import Path
import numpy as np

import screen_hidden_queries as common

ROOT = Path(__file__).resolve().parent
MODELS = common.MODELS[:2]
GROUP = 16
DEPTH = 4
K = 16
CHUNK = 4096


def sse_from_moments(count, ones):
    return float(np.sum(ones) - ones @ ones / count) if count else 0.0


def split_costs(bits, counts, members, theta):
    b = bits[members]
    n = float(counts[members].sum())
    if not n:
        return np.zeros(GROUP)
    weighted = b * counts[members, None]
    ones = weighted.sum(0)
    joint = b.T @ weighted
    return np.asarray([sse_from_moments(ones[j], joint[j])
                       + sse_from_moments(n - ones[j], ones - joint[j])
                       for j in range(GROUP)]) * (theta * theta)


def fit_tree(words, counts, theta, fixed):
    bits = ((words[:, None] >> np.arange(GROUP)) & 1).astype(np.float64)
    features = np.full(31, -1, np.int8)
    codes = np.zeros(31, np.uint8)
    codes[15:] = np.arange(K, dtype=np.uint8)
    current = [(0, np.arange(len(words)), ())]
    common_indices = []
    levels = []
    for depth in range(DEPTH):
        costs = [split_costs(bits, counts, members, theta)
                 for _, members, _ in current]
        if fixed:
            total = np.sum(costs, axis=0)
            total[list(current[0][2])] = np.inf
            shared_j = int(np.argmin(total))
            common_indices.append(shared_j)
        following = []
        chosen = []
        for (node, members, used), cost in zip(current, costs):
            cost[list(used)] = np.inf
            j = shared_j if fixed else int(np.argmin(cost))
            features[node] = j
            chosen.append({"node": node, "feature": j,
                           "child_SSE": float(cost[j]),
                           "training_examples": int(counts[members].sum())})
            branch = bits[members, j].astype(bool)
            for side in (0, 1):
                following.append((2 * node + 1 + side,
                                  members[branch == side], (*used, j)))
        levels.append(chosen)
        current = following
    centroids = np.zeros((K, GROUP), np.float64)
    leaf_counts = np.zeros(K, np.int64)
    training_sse = 0.0
    for node, members, _ in current:
        leaf = node - 15
        n = int(counts[members].sum())
        leaf_counts[leaf] = n
        if n:
            ones = (bits[members] * counts[members, None]).sum(0)
            centroids[leaf] = theta * ones / n
            training_sse += theta * theta * sse_from_moments(n, ones)
    return {"features_heap": features.tolist(),
            "leaf_index_heap": [-1] * 15 + list(range(K)),
            "leaf_or_fallback_labels": codes.tolist(),
            "leaf_id_rule": "after four tests, terminal heap index minus 15",
            "common_indices_by_level": common_indices if fixed else None,
            "distinct_possible_hidden_indices": sorted(set(int(j) for j in features if j >= 0)),
            "centroids_theta_g": centroids.tolist(), "leaf_training_counts": leaf_counts.tolist(),
            "empty_leaf_count": int(np.count_nonzero(leaf_counts == 0)),
            "training_SSE": training_sse,
            "training_mse_per_hidden_element": training_sse / (counts.sum() * GROUP),
            "split_levels": levels}


def validate_split_reference(tree, words, counts, theta, fixed):
    """Separate small raw-SSE calculation checks the moment objective."""
    bits = ((words[:, None] >> np.arange(GROUP)) & 1).astype(np.float64) * theta
    root_costs = []
    for j in range(GROUP):
        sse = 0.0
        for side in (0, 1):
            mask = ((words >> j) & 1) == side
            w = counts[mask]
            x = bits[mask]
            if w.sum():
                mean = (x * w[:, None]).sum(0) / w.sum()
                sse += float(np.sum((x - mean) ** 2 * w[:, None]))
        root_costs.append(sse)
    chosen = tree["features_heap"][0]
    assert root_costs[chosen] <= min(root_costs) + 1e-8
    np.testing.assert_allclose(root_costs[chosen], tree["split_levels"][0][0]["child_SSE"], rtol=0, atol=1e-8)
    if fixed:
        for depth in range(DEPTH):
            assert len(set(tree["features_heap"][(1 << depth) - 1:(1 << (depth + 1)) - 1])) == 1


def evaluate_proxy(flat, groups, code_lookups, weight, theta):
    """All validation positions; complete 96-dimensional FC2 contribution."""
    M = len(flat)
    partials, inverses, targets = [], [], []
    reconstruction_tables = {m: [] for m in MODELS}
    fc2_tables = {m: [] for m in MODELS}
    result = {m: {"hidden_centroid_squared_error": 0.0,
                  "hidden_teacher_squared_norm": 0.0,
                  "hidden_elements": M * 384,
                  "empty_training_leaf_hits": 0, "group_words": M * 24,
                  "fc2_squared_error": 0.0, "fc2_absolute_error": 0.0,
                  "fc2_teacher_squared_norm": 0.0,
                  "fc2_max_absolute_error": 0.0,
                  "fc2_elements": M * 96} for m in MODELS}
    for group in range(24):
        unique, inverse, counts = np.unique(flat[:, group], return_inverse=True, return_counts=True)
        truth = ((unique[:, None] >> np.arange(GROUP)) & 1).astype(np.float64) * theta
        w = np.asarray(weight[:, group * GROUP:(group + 1) * GROUP], np.float64).T
        partials.append(truth @ w)
        inverses.append(inverse.astype(np.int32))
        targets.append(truth)
        for model in MODELS:
            tree = groups[group]["trees"][model]
            centroids = np.asarray(tree["centroids_theta_g"], np.float64)
            leaf_n = np.asarray(tree["leaf_training_counts"])
            code = code_lookups[model][group][unique]
            errors = centroids[code] - truth
            result[model]["hidden_centroid_squared_error"] += float(np.sum(errors * errors * counts[:, None]))
            result[model]["hidden_teacher_squared_norm"] += float(np.sum(truth * truth * counts[:, None]))
            result[model]["empty_training_leaf_hits"] += int(np.sum(counts[leaf_n[code] == 0]))
            reconstruction_tables[model].append(centroids)
            fc2_tables[model].append(centroids @ w)
    for start in range(0, M, CHUNK):
        stop = min(M, start + CHUNK)
        teacher = np.zeros((stop - start, 96), np.float64)
        errors = {model: np.zeros_like(teacher) for model in MODELS}
        for group in range(24):
            truth = partials[group][inverses[group][start:stop]]
            teacher += truth
            for model in MODELS:
                code = code_lookups[model][group][flat[start:stop, group]]
                errors[model] += fc2_tables[model][group][code] - truth
        if start == 0:
            direct_bits = ((flat[:4, :, None] >> np.arange(GROUP)) & 1).reshape(4, 384)
            direct_teacher = direct_bits.astype(np.float64) @ weight.T.astype(np.float64) * theta
            np.testing.assert_allclose(teacher[:4], direct_teacher, rtol=0, atol=1e-12)
            for model in MODELS:
                reconstructed = np.concatenate([reconstruction_tables[model][g][code_lookups[model][g][flat[:4, g]]]
                                                for g in range(24)], axis=1)
                direct_error = reconstructed @ weight.T.astype(np.float64) - direct_teacher
                np.testing.assert_allclose(errors[model][:4], direct_error, rtol=0, atol=1e-12)
        energy = float(np.sum(teacher * teacher))
        for model in MODELS:
            error = errors[model]
            result[model]["fc2_squared_error"] += float(np.sum(error * error))
            result[model]["fc2_absolute_error"] += float(np.sum(np.abs(error)))
            result[model]["fc2_teacher_squared_norm"] += energy
            result[model]["fc2_max_absolute_error"] = max(result[model]["fc2_max_absolute_error"], float(np.max(np.abs(error))))
    return result


def normalize_proxy(stats):
    stats["hidden_centroid_mse"] = stats["hidden_centroid_squared_error"] / stats["hidden_elements"]
    stats["hidden_centroid_relative_l2"] = float(np.sqrt(stats["hidden_centroid_squared_error"] / stats["hidden_teacher_squared_norm"]))
    stats["fc2_mse"] = stats["fc2_squared_error"] / stats["fc2_elements"]
    stats["fc2_relative_l2"] = float(np.sqrt(stats["fc2_squared_error"] / stats["fc2_teacher_squared_norm"]))
    stats["empty_training_leaf_hit_fraction"] = stats["empty_training_leaf_hits"] / stats["group_words"]
    return stats


def main():
    start = time.monotonic()
    run = json.loads((ROOT / "run.json").read_text())
    theta = float(run["theta"])
    train = common.read_words(ROOT / "train_hidden.npz").reshape(-1, 24)
    weight = np.load(ROOT / "fc2_weight.npy", allow_pickle=False)
    groups = []
    lookups = {model: [] for model in MODELS}
    qlookups = {model: [] for model in MODELS}
    for group in range(24):
        unique, counts = np.unique(train[:, group], return_counts=True)
        record = {"group": group, "global_hidden_start": group * GROUP,
                  "training_unique_words": len(unique), "trees": {}}
        for model, fixed in zip(MODELS, (True, False)):
            tree = fit_tree(unique, counts, theta, fixed)
            validate_split_reference(tree, unique, counts, theta, fixed)
            lookup, q = common.tree_lookup(tree)
            lookups[model].append(lookup)
            qlookups[model].append(q)
            record["trees"][model] = tree
        groups.append(record)
    description = {"module": run["module"], "theta": theta,
       "numeric_path": run["numeric_path"], "groups": groups,
       "configuration": {"depth": 4, "group_size": 16, "groups": 24, "leaves_per_group": 16,
          "training_examples_per_group": len(train), "training_files": run["training_files"],
          "validation_files": run["validation_files"],
          "objective": "weighted child SSE of the original 16 theta*g values; no class labels or preselected frequency codebook",
          "fixed_constraint": "all nodes at a depth share a single feature index",
          "candidate_change": "node-specific feature index",
          "depth_policy": "complete depth four; no early stopping; all 16 terminal IDs remain distinct",
          "ties": "lowest unused feature index; features do not repeat on a path",
          "empty_leaf": "zero centroid; identity retained and unseen validation hits counted",
          "inference": "node=0; repeat four times: j=features_heap[node]; node=2*node+1+((word>>j)&1); code=node-15",
          "lookup_layout": "sse_tree_lookups.npz keys fixed_level_index/node_specific_index; shape[24,65536], uint8 leaf code",
          "scope": "MADDNESS binary regression-tree core migration; ridge code-table optimization is a separate common downstream comparator",
          "prior": "https://proceedings.mlr.press/v139/blalock21a/blalock21a.pdf"}}
    (ROOT / "hidden_sse_trees.json").write_text(json.dumps(description, ensure_ascii=False, indent=2) + "\n")
    np.savez_compressed(ROOT / "sse_tree_lookups.npz", **{model: np.asarray(lookups[model]) for model in MODELS})
    print("TREES_READY", ROOT / "hidden_sse_trees.json", "seconds", time.monotonic() - start, flush=True)
    result = {"configuration": description["configuration"],
              "scope": "full ten-frame CPU reconstruction/request probe; centroid tables only, no ridge or AEE",
              "module": run["module"], "theta": theta,
              "frames": [], "aggregate": {model: {"proxy": {}, "queries": {}} for model in MODELS}}
    for name in run["validation_files"]:
        words = common.read_words(ROOT / (Path(name).stem + "_hidden.npz"))
        flat = words.reshape(-1, 24)
        proxy = evaluate_proxy(flat, groups, lookups, weight, theta)
        frame = {"file": name, "models": {}}
        for model in MODELS:
            q = common.query_stats(words, qlookups[model])
            common.merge_queries(result["aggregate"][model]["queries"], q)
            target = result["aggregate"][model]["proxy"]
            for key, value in proxy[model].items():
                target[key] = max(target.get(key, 0), value) if key.endswith("max_absolute_error") else target.get(key, 0) + value
            frame["models"][model] = {"proxy": normalize_proxy(proxy[model]),
                                      "queries": common.finalize_queries(q)}
        result["frames"].append(frame)
        print(json.dumps({"file": name, "elapsed_s": round(time.monotonic() - start, 2),
              "models": {model: {"fc2_relative_l2": frame["models"][model]["proxy"]["fc2_relative_l2"],
                         "hidden_centroid_mse": frame["models"][model]["proxy"]["hidden_centroid_mse"],
                         "mean_h": frame["models"][model]["queries"]["union_h_per_spatial_position"]["mean"]}
                         for model in MODELS}}), flush=True)
    for model in MODELS:
        normalize_proxy(result["aggregate"][model]["proxy"])
        common.finalize_queries(result["aggregate"][model]["queries"])
        result["aggregate"][model]["unconditional_all_possible_tree_h"] = sum(len(group["trees"][model]["distinct_possible_hidden_indices"]) for group in groups)
        result["aggregate"][model]["training_mse_per_hidden_element"] = sum(group["trees"][model]["training_SSE"] for group in groups) / (len(train) * 384)
    result["elapsed_seconds"] = time.monotonic() - start
    (ROOT / "sse_tree_query_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("WROTE", ROOT / "sse_tree_query_result.json", flush=True)
    print(json.dumps({model: {"proxy": r["proxy"], "mean_h": r["queries"]["union_h_per_spatial_position"]["mean"],
                     "producer_fraction": r["queries"]["producer_fraction_of_full_384"],
                     "unconditional_h": r["unconditional_all_possible_tree_h"]}
                     for model, r in result["aggregate"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
