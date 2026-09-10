#!/opt/anaconda3/bin/python3.12
"""Bounded depth-4 binary hidden-query tree transfer, using existing captures.

This is not a full MADDNESS reproduction.  It preserves MADDNESS's common
feature index at each tree level in the fixed-index comparator.  Both axes
use the same 16 most frequent training words, nearest-Hamming training
labels, weighted Gini splits, and majority-label leaves.  Pure-label nodes
stop early in both axes.  The candidate permits a different feature at each
node.  No validation data select codewords, splits, or tie-breaking rules.

A first request for (spatial position, hidden channel) produces all T gates;
later requests consume the cached response.  Frontiers are modeled by tree
depth, merging same-depth requests before production.  This is an optimistic
request/state model, not a schedule or a cycle count.  All 24 groups share no
hidden channels.  The oracle computes every hidden channel and then chooses
the nearest codeword in the identical training dictionary.

FC2 error is measured on the complete 96-component linear contribution,
sum_g(theta * reconstructed_bits_g @ W_g.T), relative to captured gates.
Float64 evaluation of the supplied float32 W is a numerical error proxy;
BN2, residual addition, flow heads, and AEE are not evaluated.  Neither
conditional narrow-FFN training nor weight-bank service is implemented.
The fixed tree's four feature indices are known before any response, so its
96 producer requests can be issued without waiting for four tree frontiers.
Per-depth counts must not be read as a compulsory fixed-baseline schedule.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DEPTH = 4
GROUP = 16
K = 16
CHUNK = 4096
ALL_WORDS = np.arange(65536, dtype=np.uint16)
POPCOUNT = np.unpackbits(ALL_WORDS.astype("<u2").view(np.uint8).reshape(-1, 2),
                        axis=1).sum(1).astype(np.uint8)
MODELS = ("fixed_level_index", "node_specific_index", "forced_nearest_oracle")


def read_words(path):
    with np.load(path, allow_pickle=False) as z:
        packed = z["gate_bits"]
        shape = tuple(int(v) for v in z["shape"])
    assert shape[2] == 384 and shape[0] == 10
    assert packed.shape == (shape[0], shape[1], 48)
    return (packed[..., 0::2].astype(np.uint16)
            | (packed[..., 1::2].astype(np.uint16) << 8))


def histogram(labels, counts):
    return np.bincount(labels, weights=counts, minlength=K)


def gini_cost(hist):
    total = hist.sum()
    return float(total - hist @ hist / total) if total else 0.0


def fit_tree(words, counts, labels, fixed):
    """The only difference between axes is common-level versus node-local j."""
    features = np.full(31, -1, np.int8)
    predictions = np.zeros(31, np.uint8)
    bits = ((words[:, None] >> np.arange(GROUP)) & 1).astype(bool)
    current = [(0, np.arange(len(words)), ())]
    level_indices = []
    for depth in range(DEPTH):
        eligible = []
        for node, members, used in current:
            hist = histogram(labels[members], counts[members])
            predictions[node] = int(np.argmax(hist))
            if np.count_nonzero(hist) > 1:
                eligible.append((node, members, used))
        if not eligible:
            break
        shared_feature = None
        if fixed:
            best = None
            used = eligible[0][2]
            for feature in range(GROUP):
                if feature in used:
                    continue
                score = 0.0
                any_split = False
                for _, members, _ in eligible:
                    branch = bits[members, feature]
                    any_split |= bool(branch.any() and not branch.all())
                    for side in (False, True):
                        subset = members[branch == side]
                        score += gini_cost(histogram(labels[subset], counts[subset]))
                if any_split and (best is None or score < best[0] - 1e-10):
                    best = (score, feature)
            assert best is not None
            shared_feature = best[1]
            level_indices.append(int(shared_feature))
        following = []
        for node, members, used in eligible:
            feature = shared_feature
            if not fixed:
                best = None
                for j in range(GROUP):
                    if j in used:
                        continue
                    branch = bits[members, j]
                    if not branch.any() or branch.all():
                        continue
                    score = sum(gini_cost(histogram(labels[members[branch == side]],
                                                    counts[members[branch == side]]))
                                for side in (False, True))
                    if best is None or score < best[0] - 1e-10:
                        best = (score, j)
                assert best is not None
                feature = best[1]
            features[node] = feature
            branch = bits[members, feature]
            for side in (0, 1):
                child = 2 * node + 1 + side
                subset = members[branch == side]
                predictions[child] = predictions[node]
                if len(subset):
                    predictions[child] = int(np.argmax(histogram(labels[subset], counts[subset])))
                    following.append((child, subset, (*used, feature)))
        current = following
    return {"features_heap": features.tolist(), "leaf_or_fallback_labels": predictions.tolist(),
            "common_indices_by_level": level_indices if fixed else None,
            "nonterminal_nodes": int(np.count_nonzero(features >= 0))}


def tree_lookup(tree):
    features = np.asarray(tree["features_heap"], dtype=np.int8)
    predictions = np.asarray(tree["leaf_or_fallback_labels"], dtype=np.uint8)
    node = np.zeros(65536, np.int16)
    queries = np.full((DEPTH, 65536), -1, np.int8)
    for depth in range(DEPTH):
        feature = features[node]
        active = feature >= 0
        queries[depth, active] = feature[active]
        node[active] = 2 * node[active] + 1 + ((ALL_WORDS[active] >> feature[active]) & 1)
    return predictions[node], queries


def bit_errors(words, reconstructed, weights=None):
    xor = POPCOUNT[words ^ reconstructed].astype(np.int64)
    fp = POPCOUNT[reconstructed & np.bitwise_not(words)].astype(np.int64)
    fn = POPCOUNT[words & np.bitwise_not(reconstructed)].astype(np.int64)
    if weights is None:
        weights = np.ones(words.shape, np.int64)
    return {"bits": int(np.sum(weights) * GROUP),
            "teacher_active_bits": int(POPCOUNT[words].astype(np.int64) @ weights),
            "reconstructed_active_bits": int(POPCOUNT[reconstructed].astype(np.int64) @ weights),
            "mismatched_bits": int(xor @ weights),
            "false_positive_bits": int(fp @ weights),
            "false_negative_bits": int(fn @ weights),
            "words": int(np.sum(weights)),
            "exact_words": int((xor == 0) @ weights)}


def add_fields(dst, src):
    for key, value in src.items():
        dst[key] = dst.get(key, 0) + value


def hist_summary(hist):
    hist = np.asarray(hist, dtype=np.int64)
    count = int(hist.sum())
    cdf = np.cumsum(hist)
    def quantile(q):
        return int(np.searchsorted(cdf, max(1, int(np.ceil(q * count)))))
    return {"count": count, "mean": float(hist @ np.arange(len(hist)) / count),
            "p50": quantile(.5), "p90": quantile(.9), "p95": quantile(.95),
            "p99": quantile(.99), "max": int(np.flatnonzero(hist)[-1]),
            "histogram": hist.tolist()}


def query_stats(words, query_lookups):
    T, N, groups = words.shape
    per_position = np.zeros(N, np.int16)
    group_hist = np.zeros(17, np.int64)
    levels = [{"raw_gate_queries": 0, "previous_frontier_cache_hits": 0,
               "same_frontier_coalesced_queries": 0, "first_fullT_requests": 0}
              for _ in range(DEPTH)]
    frontier_new = np.zeros((DEPTH, N), np.int16)
    frontier_distinct = np.zeros((DEPTH, N), np.int16)
    for group in range(groups):
        seen = np.zeros(N, np.uint16)
        for depth in range(DEPTH):
            q = query_lookups[group][depth, words[:, :, group]]
            valid = q >= 0
            mask = np.left_shift(np.uint16(1), np.maximum(q, 0).astype(np.uint16))
            mask[~valid] = 0
            requested = np.bitwise_or.reduce(mask, axis=0)
            new = requested & np.bitwise_not(seen)
            new_count = POPCOUNT[new]
            distinct_count = POPCOUNT[requested]
            raw = int(valid.sum())
            previous = int(np.count_nonzero(mask & seen[None, :]))
            first = int(new_count.sum())
            add_fields(levels[depth], {"raw_gate_queries": raw,
                       "previous_frontier_cache_hits": previous,
                       "same_frontier_coalesced_queries": raw - previous - first,
                       "first_fullT_requests": first})
            frontier_new[depth] += new_count
            frontier_distinct[depth] += distinct_count
            seen |= requested
        union_count = POPCOUNT[seen]
        per_position += union_count
        group_hist += np.bincount(union_count, minlength=17)
    for depth, level in enumerate(levels):
        level["new_request_histogram_per_spatial_position"] = np.bincount(frontier_new[depth], minlength=385).tolist()
        level["distinct_h_histogram_per_spatial_position"] = np.bincount(frontier_distinct[depth], minlength=385).tolist()
    first = int(per_position.sum())
    raw = sum(level["raw_gate_queries"] for level in levels)
    assert first == sum(level["first_fullT_requests"] for level in levels)
    assert raw - first == sum(level["previous_frontier_cache_hits"] + level["same_frontier_coalesced_queries"] for level in levels)
    return {"spatial_positions": N, "raw_gate_queries": raw, "first_fullT_requests": first,
            "cache_hits_including_same_frontier_merge": raw - first,
            "full_hidden_producer_requests": N * 384,
            "full_16_hidden_group_instances": int(group_hist[16]),
            "group_instances": N * groups,
            "union_h_histogram_per_spatial_position": np.bincount(per_position, minlength=385).tolist(),
            "union_h_histogram_per_group": group_hist.tolist(), "levels": levels}


def merge_queries(total, frame):
    for key, value in frame.items():
        if key == "levels":
            if key not in total:
                total[key] = [{} for _ in range(DEPTH)]
            for a, b in zip(total[key], value):
                for subkey, subvalue in b.items():
                    a[subkey] = ((np.asarray(a.get(subkey, np.zeros(len(subvalue), np.int64))) + subvalue).tolist()
                                if isinstance(subvalue, list) else a.get(subkey, 0) + subvalue)
        elif isinstance(value, list):
            total[key] = (np.asarray(total.get(key, np.zeros(len(value), np.int64))) + value).tolist()
        else:
            total[key] = total.get(key, 0) + value


def finalize_queries(q):
    q["union_h_per_spatial_position"] = hist_summary(q.pop("union_h_histogram_per_spatial_position"))
    q["union_h_per_group"] = hist_summary(q.pop("union_h_histogram_per_group"))
    q["producer_fraction_of_full_384"] = q["first_fullT_requests"] / q["full_hidden_producer_requests"]
    q["full_16_hidden_group_fraction"] = q["full_16_hidden_group_instances"] / q["group_instances"]
    q["cache_hit_fraction_of_gate_queries"] = q["cache_hits_including_same_frontier_merge"] / q["raw_gate_queries"]
    for level in q["levels"]:
        level["new_parallel_h_per_spatial_position"] = hist_summary(level.pop("new_request_histogram_per_spatial_position"))
        level["distinct_h_per_spatial_position"] = hist_summary(level.pop("distinct_h_histogram_per_spatial_position"))
    return q


def fc2_error(words_flat, dictionaries, lookups, weight, theta):
    """Full validation, complete output-vector error; no validation subsampling."""
    M, groups = words_flat.shape
    partials, inverses, code_partials = [], [], []
    for group in range(groups):
        unique, inverse = np.unique(words_flat[:, group], return_inverse=True)
        bits = ((unique[:, None] >> np.arange(GROUP)) & 1).astype(np.float64)
        code_bits = ((dictionaries[group][:, None] >> np.arange(GROUP)) & 1).astype(np.float64)
        local_w = np.asarray(weight[:, group * GROUP:(group + 1) * GROUP], dtype=np.float64).T * theta
        partials.append(bits @ local_w)
        inverses.append(inverse.astype(np.int32))
        code_partials.append(code_bits @ local_w)
    totals = {model: {"elements": M * weight.shape[0], "squared_error": 0.0,
                      "absolute_error": 0.0, "max_absolute_error": 0.0,
                      "teacher_squared_norm": 0.0} for model in MODELS}
    for start in range(0, M, CHUNK):
        stop = min(M, start + CHUNK)
        teacher = np.zeros((stop - start, weight.shape[0]), np.float64)
        errors = {model: np.zeros_like(teacher) for model in MODELS}
        for group in range(groups):
            true_partial = partials[group][inverses[group][start:stop]]
            teacher += true_partial
            words = words_flat[start:stop, group]
            for model in MODELS:
                code_indices = lookups[model][group][words]
                errors[model] += code_partials[group][code_indices] - true_partial
        if start == 0:
            direct_bits = ((words_flat[:4, :, None] >> np.arange(GROUP)) & 1).reshape(4, 384)
            direct_teacher = direct_bits.astype(np.float64) @ np.asarray(weight.T, np.float64) * theta
            np.testing.assert_allclose(teacher[:4], direct_teacher, rtol=0, atol=1e-12)
            for model in MODELS:
                reconstructed = np.stack([dictionaries[g][lookups[model][g][words_flat[:4, g]]]
                                          for g in range(groups)], axis=1)
                bits = ((reconstructed[:, :, None] >> np.arange(GROUP)) & 1).reshape(4, 384)
                direct_error = bits.astype(np.float64) @ np.asarray(weight.T, np.float64) * theta - direct_teacher
                np.testing.assert_allclose(errors[model][:4], direct_error, rtol=0, atol=1e-12)
        energy = float(np.sum(teacher * teacher))
        for model in MODELS:
            err = errors[model]
            totals[model]["squared_error"] += float(np.sum(err * err))
            totals[model]["absolute_error"] += float(np.sum(np.abs(err)))
            totals[model]["max_absolute_error"] = max(totals[model]["max_absolute_error"], float(np.max(np.abs(err))))
            totals[model]["teacher_squared_norm"] += energy
    return totals


def normalize_errors(stats):
    if "bits" in stats:
        stats["bit_mismatch_fraction"] = stats["mismatched_bits"] / stats["bits"]
        stats["exact_word_fraction"] = stats["exact_words"] / stats["words"]
        stats["agreement_with_forced_nearest_fraction"] = stats["agreement_with_forced_nearest_words"] / stats["words"]
        stats["teacher_active_fraction"] = stats["teacher_active_bits"] / stats["bits"]
        stats["false_negative_fraction_of_teacher_active"] = stats["false_negative_bits"] / stats["teacher_active_bits"]
    else:
        stats["mse"] = stats["squared_error"] / stats["elements"]
        stats["rmse"] = float(np.sqrt(stats["mse"]))
        stats["mae"] = stats["absolute_error"] / stats["elements"]
        stats["relative_l2_error"] = float(np.sqrt(stats["squared_error"] / stats["teacher_squared_norm"]))
    return stats


def main():
    started = time.monotonic()
    run = json.loads((ROOT / "run.json").read_text())
    assert not set(run["training_files"]).intersection(run["validation_files"])
    train = read_words(ROOT / "train_hidden.npz").reshape(-1, 24)
    weight = np.load(ROOT / "fc2_weight.npy", allow_pickle=False)
    assert weight.shape == (96, 384)
    dictionaries, fitted_groups = [], []
    lookups = {model: [] for model in MODELS}
    query_lookups = {model: [] for model in MODELS[:2]}
    for group in range(24):
        unique, counts = np.unique(train[:, group], return_counts=True)
        order = np.lexsort((unique, -counts))
        dictionary = unique[order[:K]]
        assert len(dictionary) == K
        dictionaries.append(dictionary)
        oracle = np.argmin(POPCOUNT[ALL_WORDS[:, None] ^ dictionary[None, :]], axis=1).astype(np.uint8)
        lookups[MODELS[2]].append(oracle)
        labels = oracle[unique]
        record = {"group": group, "training_unique_words": len(unique),
                  "dictionary_words": dictionary.tolist(),
                  "dictionary_training_counts": counts[order[:K]].tolist(),
                  "dictionary_training_exact_fraction": float(counts[order[:K]].sum() / counts.sum()),
                  "trees": {}}
        for model, fixed in zip(MODELS[:2], (True, False)):
            tree = fit_tree(unique, counts, labels, fixed)
            tree["distinct_possible_hidden_indices"] = sorted(set(j for j in tree["features_heap"] if j >= 0))
            lookup, q = tree_lookup(tree)
            lookups[model].append(lookup)
            query_lookups[model].append(q)
            tree["training_reconstruction"] = bit_errors(unique, dictionary[lookup[unique]], counts)
            tree["training_reconstruction"]["agreement_with_forced_nearest_words"] = int(np.sum(counts[lookup[unique] == labels]))
            normalize_errors(tree["training_reconstruction"])
            record["trees"][model] = tree
        fitted_groups.append(record)
        print(f"fit group {group + 1}/24 unique={len(unique)}", flush=True)
    result = {"status": "completed_CPU_transfer_probe", "module": run["module"],
              "numeric_path": run["numeric_path"], "theta": run["theta"],
              "configuration": {"T": 10, "H": 384, "groups": 24, "group_size": 16,
                 "depth": DEPTH, "dictionary_size": K, "training_spatial_positions": 16384,
                 "training_examples_per_group": len(train), "training_files": run["training_files"],
                 "validation_files": run["validation_files"],
                 "dictionary": "16 most frequent training words; count descending, word ascending ties",
                 "labels": "nearest Hamming dictionary word; dictionary order breaks ties",
                 "split": "weighted Gini impurity; lower feature index breaks ties; no repeated path feature",
                 "leaves": "weighted majority training label; pure-label early stop in both models",
                 "fixed_constraint": "one shared feature index across all nonterminal nodes at a depth",
                 "candidate_change": "feature index may differ at each node",
                 "oracle": "nearest Hamming in same dictionary, after generating every hidden channel",
                 "fc2_proxy": "complete 96-component theta*g*W contribution, float64 evaluation of supplied float32 W",
                 "request_model": "first (p,h) request generates complete T10; per-depth frontier coalesces same h",
                 "fixed_baseline_preissue": "all four common indices are known in advance; all 96 hidden channels can be requested before traversing the tree",
                 "verification": "per-frame first four outputs checked against direct 384x96 float64 dot for teacher and all three reconstructions; zero train/validation filename overlap",
                 "explicit_limits": ["not a complete MADDNESS implementation", "not AEE or BN2/residual evaluation",
                    "not cycles, RTL, SRAM traffic, area, or energy", "no held-out tuning of dictionaries or trees",
                    "no jointly trained fixed 4/8-channel narrow FFN", "frontier concurrency is an optimistic request count",
                    "Hamming oracle does not minimize FC2 error"]},
              "fitted_groups": fitted_groups, "frames": [],
              "aggregate": {model: {"hidden": {}, "fc2": {}, "queries": {}} for model in MODELS}}
    for name in run["validation_files"]:
        words = read_words(ROOT / (Path(name).stem + "_hidden.npz"))
        flat = words.reshape(-1, 24)
        frame = {"file": name, "shape": [10, words.shape[1], 384], "models": {}}
        for model in MODELS:
            hidden = {}
            per_group = []
            for group in range(24):
                source = flat[:, group]
                reconstructed = dictionaries[group][lookups[model][group][source]]
                stats = bit_errors(source, reconstructed)
                stats["agreement_with_forced_nearest_words"] = int(np.count_nonzero(lookups[model][group][source] == lookups[MODELS[2]][group][source]))
                add_fields(hidden, stats)
                per_group.append(normalize_errors(stats))
            add_fields(result["aggregate"][model]["hidden"], hidden)
            frame["models"][model] = {"hidden": normalize_errors(hidden), "hidden_per_group": per_group}
            if model != MODELS[2]:
                q = query_stats(words, query_lookups[model])
                merge_queries(result["aggregate"][model]["queries"], q)
                frame["models"][model]["queries"] = finalize_queries(q)
            else:
                q = {"first_fullT_requests": words.shape[1] * 384,
                     "full_hidden_producer_requests": words.shape[1] * 384,
                     "spatial_positions": words.shape[1]}
                add_fields(result["aggregate"][model]["queries"], q)
                frame["models"][model]["queries"] = {**q, "producer_fraction_of_full_384": 1.0}
        errors = fc2_error(flat, dictionaries, lookups, weight, float(run["theta"]))
        for model, error in errors.items():
            target = result["aggregate"][model]["fc2"]
            for key, value in error.items():
                target[key] = max(target.get(key, 0), value) if key == "max_absolute_error" else target.get(key, 0) + value
            frame["models"][model]["fc2"] = normalize_errors(error)
        result["frames"].append(frame)
        print(json.dumps({"file": name, "elapsed_s": round(time.monotonic() - started, 1),
              "models": {model: {"bit_error": frame["models"][model]["hidden"]["bit_mismatch_fraction"],
                          "fc2_relative_l2": frame["models"][model]["fc2"]["relative_l2_error"],
                          "producer_fraction": frame["models"][model]["queries"]["producer_fraction_of_full_384"]}
                         for model in MODELS}}, ensure_ascii=False), flush=True)
    for model, aggregate in result["aggregate"].items():
        normalize_errors(aggregate["hidden"])
        normalize_errors(aggregate["fc2"])
        if model != MODELS[2]:
            finalize_queries(aggregate["queries"])
        else:
            aggregate["queries"]["producer_fraction_of_full_384"] = 1.0
    result["elapsed_seconds"] = time.monotonic() - started
    destination = ROOT / "tree_query_result.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("WROTE", destination, flush=True)
    print(json.dumps({m: {"hidden": a["hidden"], "fc2": a["fc2"],
               "producer_fraction": a["queries"]["producer_fraction_of_full_384"],
               "mean_h": a["queries"].get("union_h_per_spatial_position", {}).get("mean", 384)}
           for m, a in result["aggregate"].items()}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
