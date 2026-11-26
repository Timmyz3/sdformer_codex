"""Independently bind all E2 actions, then measure raw GT and finite-action headroom.

No fitting of runtime predictor. LOSO static selection is head-development
crossfit within exposed scenes; family/layout selection is not held out.
"""
import argparse
import itertools
import json
import math
import sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reduce_e1_quality_v1 as base
ROOT, R27, raw, mc = base.ROOT, base.R27, base.raw, base.mathcore
E2 = ROOT / "research/flow_story_20261021/e2_group_damage"
sha, read, child, prediction = base.sha, base.read, base.child, base.prediction
ACTIONS = [()] + [(i,) for i in range(6)] + list(itertools.combinations(range(6), 2))
PAIRS = ACTIONS[7:]


def label(action):
    return "background_Q6" if not action else ("single_" if len(action) == 1 else "pair_") + "_".join(map(str, action))


def masks_and_contract(contract):
    """Independent NumPy construction; no production materialize() call."""
    assert contract["status"] == "FROZEN_PATCH6_REACHABILITY_V1"
    assert contract["source_state_sha256"] == base.STATE_SHA
    assert len(contract["geometries"]) == 28 and len(contract["family_targets"]) == 4
    names = contract["family_targets"]
    assert all(".patch_embed.residual_encoding." in n for n in names)
    assert [g["id"] for g in contract["groups"]] == list(range(6))
    groups = []
    for i, group in enumerate(contract["groups"]):
        assert set(group["tiles"]) == set(names)
        cells = [[z // 35, z % 35] for z in range(i * 151, (i + 1) * 151)]
        assert all(group["tiles"][n] == cells for n in names)
        assert group["cancelled_kernel_ticks"] == 41096160
        grid = np.zeros((240, 320), dtype=np.bool_)
        for y, x in cells:
            grid[y * 9:y * 9 + 9, x * 9:x * 9 + 9] = True
        assert grid.sum() == 12231
        groups.append(grid)
    assert np.max(np.sum(groups, axis=0)) == 1
    assert contract["single_cancelled_kernel_ticks"] == 41096160
    assert contract["pair_cancelled_kernel_ticks"] == 82192320
    assert contract["formal_quality_forwards"] == 352 and contract["engineering_dense_reference_forwards"] == 2
    layouts = {}
    for action in ACTIONS:
        layouts[label(action)] = {
            n: (np.logical_or.reduce([groups[i] for i in action]) if n in names and action
                else np.zeros((g["H"], g["W"]), dtype=np.bool_))
            for n, g in contract["geometries"].items()
        }
    return groups, layouts


def check_work(work, geometries, masks, *, reference=False):
    """Reconstruct every row domain and actually issued integer plane call."""
    assert set(work) == set(geometries)
    cancelled = 0
    for name, g in geometries.items():
        w, selected = work[name], masks[name]
        T, H, W, K, C = (g[k] for k in ("T", "H", "W", "K", "Cout"))
        assert T == 10
        tail = np.ones((H, W), dtype=np.bool_)
        tail[:H//9*9, :W//9*9] = False
        q6 = ~tail & ~selected
        domains = {k: np.broadcast_to(v, (T, 1, H, W)).reshape(-1).copy()
                   for k, v in (("tail", tail), ("Q6", q6), ("Q4", selected))}
        rows = T*H*W
        counts = {k: int(v.sum()) for k, v in domains.items()}
        for k, n in counts.items():
            assert w[k + "_rows"] == n
            assert w[k + "_rows_sha256"] == raw.payload_sha(domains[k])
        assert w["geometric_shape"] == [T, 1, H, W]
        assert w["rows"] == rows and w["Cin"] == K and w["Cout"] == C
        assert w["complete_rows"] == counts["Q6"] + counts["Q4"]
        assert w["selected_spatial_mask_sha256"] == raw.payload_sha(selected)
        cells = [[y, x] for y in range(H//9) for x in range(W//9) if selected[y*9, x*9]]
        assert w["selected_complete_tiles"] == cells
        expected_shape = [T, 1, K//9, H, W] if ".patch_embed." in name else [T, 1, H, W, K]
        assert w["actual_input_shape"] == expected_shape
        assert w["mode"] == "Q6_background_Q4_selected_Q8_true_tail"
        assert w["reference"] is reference and w["scale_applications"] == 1
        for flag in ("binary_verified", "int32_accumulation", "shared_q8_scale",
                     "actual_low2_plane_cancellation", "canonical_scatter_before_BN",
                     "spatial_rows_retained", "full_T_and_channels"):
            assert w[flag] is True
        assert w["low_bit_rule"] == "tail_drop0_complete_drop2_selected_complete_drop4"
        limit = max(24, min(65536, (8*1024*1024)//C))//8*8
        calls = []
        if reference:
            for scope, n in (("tail8", counts["tail"]), ("complete6", counts["Q6"]),
                             ("selected4", counts["Q4"])):
                for start in range(0, n, limit):
                    nr = min(limit, n-start)
                    calls.append(dict(kind="dense_integer_reference", bit=None, scope=scope,
                                      rows=nr, padding_rows=max(24, (nr+7)//8*8)-nr, coefficient=None))
        else:
            for bit in range(8):
                n = counts["tail"] if bit < 2 else rows-counts["Q4"] if bit < 4 else rows
                scope = "tail8" if bit < 2 else "non_q4" if bit < 4 else "all"
                for start in range(0, n, limit):
                    nr = min(limit, n-start)
                    calls.append(dict(kind="binary_weight_plane", bit=bit, scope=scope,
                                      rows=nr, padding_rows=max(24, (nr+7)//8*8)-nr,
                                      coefficient=-128 if bit == 7 else 1 << bit))
        assert w["kernel_calls"] == calls
        assert w["actual_integer_kernel_calls"] == len(calls)
        assert w["explicit_padded_rows"] == sum(x["padding_rows"] for x in calls)
        assert w["logical_row_plane_work"] == (None if reference else 8*counts["tail"] + 6*counts["Q6"] + 4*counts["Q4"])
        cancelled += 2*counts["Q4"]*math.ceil(K/128)*math.ceil(C/16)
    return cancelled


def bind(run):
    result = read(run / "results.json")
    assert result["schema"] == "flow_e2_action_collection_v1"
    assert result["status"] == "E2_352_ACTION_LABELS_AND2_DENSE_REFERENCES_COMPLETE"
    assert (result["model_forwards"], result["formal_model_forwards"], result["engineering_model_forwards"]) == (354, 352, 2)
    assert result["expected_quality_predictions"] == 352 and result["completed_optimizer_steps"] == 0
    for flag in ("training_performed", "development_GT_arrays_read", "quality_computed", "quality_selection"):
        assert result[flag] is False
    assert result["background_replay_bitwise_equal_all16"] is True
    assert result["source_state_sha256"] == result["final_state_sha256"] == base.STATE_SHA
    assert result["runtime"]["AMP"] is False and result["runtime"]["TF32"] is False
    source_paths = {
        "runner": E2/"collect_actions.py", "core": E2/"group_precision_core.py",
        "group_contract": E2/"group_contract.py", "action_contract": E2/"PATCH6_CONTRACT_V1.json",
        "selections": E2.parent/"selections_v1.json",
        "old_smoke": R27/"implementation/run_smoke.py", "old_helper": R27/"implementation/run_recovery.py",
        "old_Q6_core": R27/"implementation/recovery_integer_core.py",
        "members": R27/"independent_review/membership_v1.json",
        "old_result": R27/"actual_v1/formal_Q6_v1/results.json",
    }
    for k, p in source_paths.items():
        assert result["sources"][k] == sha(p), k
    assert result["sources"]["weights"] == base.WEIGHT_SHA
    roles, members = read(source_paths["selections"]), read(source_paths["members"])
    contract = read(source_paths["action_contract"])
    assert result["action_contract"] == contract
    groups, layouts = masks_and_contract(contract)
    ids = roles["e2_reserved_ids"]
    assert ids == result["selected_ids"] == contract["target_ids"] and len(set(ids)) == 16
    assert not set(ids) & set(roles["localization_ids"] + roles["confirmation_reserved_ids"])
    records = {r["target_id"]: r for r in members["records"]}
    assert not set(ids) & set(members["sealed_ids"] + members["H8_historical_ids"])
    regions_path = ROOT/"research/flow_story_20261016/coverage_recovery_round13/preparation/regions_v1.json"
    assert sha(regions_path) == base.REGIONS_SHA
    regions = read(regions_path)
    assert not set(ids) & set(regions["fit_target_ids"])
    sources, evidence = {}, {}
    for tid in ids:
        sources[tid], evidence[tid] = {}, {}
        for rec in records[tid]["source_files"]:
            k = {"input": "events", "gt": "gt", "mask": "mask"}[rec["category"]]
            p = child(R27/"dev64_source_snapshot", rec["relative_path"])
            assert sha(p) == rec["sha256"] and p.stat().st_size == int(rec["bytes"])
            sources[tid][k] = p
            evidence[tid][k] = dict(relative_path=rec["relative_path"], sha256=rec["sha256"])
    q6_old = read(source_paths["old_result"])
    oldframes = {r["target_id"]: r for r in q6_old["evaluations"]["endpoint"]["frames"]}
    # R27 ledger did not save weight representations. Bind these across all new
    # actions to the exact old-Q6 replay instead of inventing absent old hashes.
    weight_reference = result["evaluations"]["background_Q6"]["frames"][0]["work"]
    bound = {}
    assert set(result["evaluations"]) == {label(a) for a in ACTIONS}
    for action in ACTIONS:
        mode = label(action)
        ev = result["evaluations"][mode]
        assert ev["action"] == list(action) and ev["model_forwards"] == 16 and ev["GT_arrays_read"] is False
        assert ev["state_sha256"] == base.STATE_SHA
        layout = ev["mask_layout"]
        assert layout["action"] == list(action) and layout["cancelled_kernel_ticks"] == len(action)*41096160
        assert layout["selected_mask_sha256"] == {n: raw.payload_sha(m) for n, m in layouts[mode].items()}
        assert layout["selected_complete_tiles"] == {n: int(m.sum())//81 for n, m in layouts[mode].items()}
        assert [r["target_id"] for r in ev["frames"]] == ids
        for row in ev["frames"]:
            tid = row["target_id"]
            assert row["action"] == list(action) and row["sequence"] == records[tid]["sequence"]
            assert row["input_evidence"] == evidence[tid]
            assert row["actual_BN_calls"] == oldframes[tid]["actual_BN_calls"]
            assert len(row["actual_BN_calls"]) == 78 and set(row["actual_BN_calls"].values()) == {1}
            assert set(row["actual_affine_call_counts"]) == set(contract["geometries"])
            assert set(row["actual_affine_call_counts"].values()) == {1}
            assert check_work(row["work"], contract["geometries"], layouts[mode]) == len(action)*41096160
            for n, w in row["work"].items():
                for k in ("weight_q8_sha256", "weight_scale_sha256"):
                    assert w[k] == weight_reference[n][k] and len(w[k]) == 64
            if not action:
                assert row["heads"] == oldframes[tid]["heads"]
            bound[(mode, tid)] = prediction(run, row)
    assert set(result["parity"]) == {"single_0", "pair_0_1"}
    for mode, p in result["parity"].items():
        assert p["target_id"] == ids[0] and p["comparison"]["pass"]
        checks = p["comparison"]["checks"]
        assert len(checks) == 137
        assert sum(k.startswith("integer::") for k in checks) == 28
        assert sum(k.startswith("PSN::") for k in checks) == 105
        assert sum(k.startswith("head") for k in checks) == 4
        assert all(v["bitwise_equal"] and v["different_bytes"] == 0 and v["actual_sha256"] == v["reference_sha256"] for v in checks.values())
        check_work(p["actual_record"]["work"], contract["geometries"], layouts[mode])
        check_work(p["dense_record"]["work"], contract["geometries"], layouts[mode], reference=True)
        path = child(run, p["heads_file"])
        assert sha(path) == p["heads_sha256"]
        with np.load(path, allow_pickle=False) as data:
            assert set(data.files) == {"head0", "head1", "head2", "head3"}
            for key in data.files:
                assert raw.payload_sha(data[key]) == checks[key]["actual_sha256"]
    native_run = R27/"actual_v1/formal_native_v1"
    native = read(native_run/"results.json")
    old_audit = read(R27/"actual_execution_audit_v1.json")
    assert old_audit["status"] == "PASS" and old_audit["quality_analysis_allowed"]
    assert old_audit["source_results"]["native"] == sha(native_run/"results.json")
    assert old_audit["source_results"]["Q6"] == sha(source_paths["old_result"])
    for stage, state in (("initial", "original_native"), ("endpoint", "recovered_native")):
        rows = {r["target_id"]: r for r in native["evaluations"][stage]["frames"]}
        for tid in ids:
            assert rows[tid]["input_evidence"] == evidence[tid]
            bound[(state, tid)] = prediction(native_run, rows[tid])
    audit = dict(status="PASS", result_sha256=sha(run/"results.json"), complete_formal_predictions=352,
                 engineering_references=2, old_native_references=32, optimizer_steps=0,
                 Q6_background_all16_four_heads_exact=True, same_W_state=base.STATE_SHA,
                 source_byte_binding_before_GT_values=True, original_arrays=48,
                 full_graph_references=2, action_contract_sha256=sha(source_paths["action_contract"]),
                 source_paths={k: dict(path=str(p), sha256=sha(p)) for k,p in source_paths.items()})
    return result, roles, regions, sources, bound, groups, audit


def uniform_metrics(metrics):
    out = {}
    for region in mc.REGIONS:
        n = metrics[0][region]["valid_pixels"]
        assert all(x[region]["valid_pixels"] == n for x in metrics)
        err = math.fsum(x[region]["error_sum"] for x in metrics)/len(metrics)
        out[region] = dict(valid_pixels=n, error_sum=err, AEE=err/n if n else None)
    return out


def choose_policies(frames):
    """Policy labels fixed before E2 arrays. All selections optimize all-valid AEE."""
    seqs = list(dict.fromkeys(f["sequence"] for f in frames))
    assert len(frames) == 16 and len(seqs) == 8 and all(sum(f["sequence"] == s for f in frames) == 2 for s in seqs)
    pair_names = [label(a) for a in PAIRS]
    score = lambda f, a: f["states"][a]["all_valid"]["error_sum"]
    # Identical support across actions means minimizing sum errors equals pooled AEE.
    global_best = min(pair_names, key=lambda a: (math.fsum(score(f,a) for f in frames), a))
    fit_static = {
        seq: min(pair_names, key=lambda a: (math.fsum(score(f,a) for f in frames if f["sequence"] != seq), a))
        for seq in seqs
    }
    choices, interactions = {}, []
    for f in frames:
        ranked_singles = sorted(range(6), key=lambda i: (score(f, label((i,))), i))
        single_pair = tuple(sorted(ranked_singles[:2]))
        joint = min(PAIRS, key=lambda a: (score(f, label(a)), a))
        chosen = dict(fit_fold_static=fit_static[f["sequence"]],
                      singleton_oracle=label(single_pair), joint_pair_oracle=label(joint),
                      in_sample_best_static_diagnostic=global_best)
        choices[f["target_id"]] = chosen
        for policy, action in chosen.items():
            f["states"][policy] = f["states"][action]
        f["states"]["uniform_pair_expectation"] = uniform_metrics([f["states"][a] for a in pair_names])
        ref = f["states"]["background_Q6"]["all_valid"]["AEE"]
        for a in PAIRS:
            actual = f["states"][label(a)]["all_valid"]["AEE"]-ref
            additive = math.fsum(f["states"][label((i,))]["all_valid"]["AEE"]-ref for i in a)
            interactions.append(dict(target_id=f["target_id"], sequence=f["sequence"], pair=label(a),
                                     actual_delta_AEE=actual, singleton_sum_delta_AEE=additive,
                                     interaction_AEE=actual-additive))
    return choices, fit_static, global_best, interactions


def coverage(arrays, masks, groups):
    """Geometric footprint only, never assumed to bound actual downstream influence."""
    valid = arrays["mask"]
    bins = np.count_nonzero(arrays["events"], axis=0)
    active = bins > 0
    out = {}
    for action in ACTIONS[1:]:
        patch = np.logical_or.reduce([groups[i] for i in action])
        footprint = np.repeat(np.repeat(patch, 2, axis=0), 2, axis=1)
        assert footprint.shape == valid.shape
        n = int(footprint.sum())
        vp = int(np.count_nonzero(footprint & valid))
        out[label(action)] = dict(
            image_pixels=n, valid_GT_pixels=vp, valid_GT_fraction_of_selected=vp/n,
            valid_GT_fraction_of_frame=vp/int(valid.sum()),
            voxel_active_pixels=int(np.count_nonzero(footprint & active)),
            voxel_zero_pixels=int(np.count_nonzero(footprint & ~active)),
            nonzero_voxel_bins=int(bins[footprint].sum()),
            valid_GT_voxel_active_pixels=int(np.count_nonzero(footprint & valid & active)),
            valid_GT_voxel_zero_pixels=int(np.count_nonzero(footprint & valid & ~active)),
            valid_GT_nonzero_voxel_bins=int(bins[footprint & valid].sum()),
            region_support={r: int(np.count_nonzero(footprint & m)) for r,m in masks.items()})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    assert not a.output.exists()
    result, roles, regions, sources, bound, groups, audit = bind(a.run)
    action_states = [label(x) for x in ACTIONS]
    ref_states = ["original_native", "recovered_native"]
    frames, covered = [], {}
    for tid in roles["e2_reserved_ids"]:
        arrays = {k: np.load(p, allow_pickle=False) for k,p in sources[tid].items()}
        masks = raw.region_masks(arrays["gt"], arrays["mask"], arrays["events"],
                                 regions["boundary_threshold_pixels"], regions["large_displacement_threshold_pixels"])
        metrics = {}
        for state in action_states + ref_states:
            with np.load(bound[(state,tid)], allow_pickle=False) as d:
                metrics[state] = raw.measure(d["head3"], arrays["gt"], masks)
        frames.append(dict(target_id=tid, sequence=tid.rsplit("_", 1)[0], states=metrics))
        covered[tid] = coverage(arrays, masks, groups)
    choices, fit_static, global_best, interactions = choose_policies(frames)
    policies = ["fit_fold_static", "singleton_oracle", "joint_pair_oracle",
                "in_sample_best_static_diagnostic", "uniform_pair_expectation"]
    states = action_states + ref_states + policies
    seqs = list(dict.fromkeys(f["sequence"] for f in frames))
    summaries = {state: {scope: {r: mc.aggregate([f["states"][state][r] for f in frames
                                                if scope == "overall" or f["sequence"] == scope])
                                for r in mc.REGIONS}
                         for scope in ["overall"] + seqs} for state in states}
    pairs = [(c, r) for c in action_states[1:] + policies
             for r in ("background_Q6", "original_native", "recovered_native")]
    pairs += [(c, r) for c,r in (
        ("joint_pair_oracle", "fit_fold_static"), ("singleton_oracle", "fit_fold_static"),
        ("joint_pair_oracle", "singleton_oracle"), ("fit_fold_static", "uniform_pair_expectation"),
        ("background_Q6", "recovered_native"), ("background_Q6", "original_native"))]
    comparisons = {c+"__vs__"+r: mc.comparison(frames, summaries, c, r, seqs) for c,r in pairs}
    selected_coverage = {tid: {policy: covered[tid][name] for policy,name in names.items()}
                         for tid,names in choices.items()}
    output = dict(schema="flow_e2_independent_raw_quality_v1", status="COMPLETE", execution_review=audit,
                  states=states, frames=frames, summaries=summaries, comparisons=comparisons,
                  policy_choices=choices, fit_fold_static=fit_static,
                  in_sample_best_static_diagnostic=global_best,
                  action_coverage=covered, selected_action_coverage=selected_coverage,
                  interaction_rows=interactions,
                  selector_learning_demonstrated=False, runtime_GT_available=False,
                  selection_objective="Minimum all-valid pixel-pooled AEE; lexicographic tie-break. Static fit excludes target sequence.",
                  uniform_definition="Expected per-pixel error over 15 complete actual pair forwards; not flow averaging or a realized tail bound.",
                  oracle_definition="GT-using reachability upper bounds, not runtime algorithms; singleton-selected pair evaluated by actual joint forward.",
                  coverage_definition="Patch 9x9 output-tile footprint upsampled 2x to input image; not causal affected-output support. Voxels are nonzero signed bins, not raw event counts.",
                  scope=contract_scope(result),
                  cost=dict(pair_extra_cancelled_ticks=82192320, native_arithmetic=3652955059,
                            pair_extra_fraction_native=82192320/3652955059,
                            net_cycles=False, selector_supply_control_costs_deducted=False),
                  legacy_lines="3% allGT/5% B,L are legacy diagnostic references, not an automatic new experiment gate.",
                  reducer_sha256=sha(__file__), raw_math_sha256=sha(raw.__file__),
                  scalar_math_sha256=sha(mc.__file__))
    a.output.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    chosen_states = ["background_Q6", "recovered_native"] + policies
    table = {s: {r: summaries[s]["overall"][r]["pixel_weighted_AEE"] for r in ("all_valid","B","L")}
             for s in chosen_states}
    print(json.dumps(dict(status="COMPLETE", output=str(a.output), pixel_AEE=table), indent=2))


def contract_scope(result):
    return dict(data=result["selection_scope"], independent_pipeline_confirmation=False,
                qualification="E1 family selection used the same eight scenes; these 16 E2 identities differ from E1, but all belong to previously exposed development64. LOSO is head-development crossfit only.")


if __name__ == "__main__":
    main()
