#!/usr/bin/env python3
"""Fixed stage02 coarse baseline, twelve same-cost singleton fine interventions.

Offline counterfactual screen only. Ordinary features/masks do not use GT;
the explicitly posthoc task-gain top2 mask does use measured GT intervention loss.
No training or deployable-selector claim. Default preflight; root starts GPU.
"""
import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "structured_merge"
SCOPE_RUNNER = BASE / "scope_extension_v1/run_scope_extension.py"
SCOPE_RUNNER_SHA = "109f535f29fa6cbad5e37258a44ec3b7a1f0626303a85f4afdcd47ace4665402"
SCOPE_RESULTS_SHA = "e43b7e06b12ac0360a4b8799a89ee94f020e6c71b791534d98949b4280f133fb"
TARGET = "sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.0.mlp"
BASE_ARM = "coarse_stage02"
CANDIDATES = tuple((r, c) for r in range(3) for c in range(4))
ARMS = ((BASE_ARM, ()),) + tuple((f"fine_r{r}_c{c}", ((r, c),)) for r, c in CANDIDATES)
COMBINATION_ARMS = ("k2_posthoc_singleton_task_gain", "k2_amplitude", "k2_pooling_residual",
                    "k2_fixed_uniform", "k2_offline_branch_mse")
SCORE_NAMES = ("amplitude_abs_mean", "feature_rms", "pooling_residual_mse",
               "spatial_neighbor_abs_difference", "temporal_neighbor_abs_difference",
               "psn1_activity", "psn1_spatial_neighbor_hamming")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def load_scope_runner():
    require(sha(SCOPE_RUNNER) == SCOPE_RUNNER_SHA, "Frozen scope runner differs")
    spec = importlib.util.spec_from_file_location("_damage_frozen_scope", SCOPE_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def full_tiles(shape):
    return tuple((r, c) for r in range(shape[2]//9) for c in range(shape[3]//9))


def tile_mask(shape, selected, torch):
    require(len(selected) == len(set(selected)), "Duplicate fine tile")
    require(set(selected) <= set(full_tiles(shape)), "Fine candidate must be a complete 9x9 tile")
    mask = torch.zeros((math.ceil(shape[2]/9), math.ceil(shape[3]/9)), dtype=torch.bool)
    for r, c in selected:
        mask[r, c] = True
    return mask


@contextlib.contextmanager
def wrap(model, targets, shapes, selected, mode, merge, torch):
    wrappers = {}
    with contextlib.ExitStack() as stack:
        for name in targets:
            if not any(f".layers.{s}." in name for s in (0, 2)):
                continue
            mask = tile_mask(shapes[name], selected if name == TARGET else (), torch)
            wrappers[name] = stack.enter_context(merge.replace_mlp(model, name, mask, 3, mode,
                                                                   expected_shape=shapes[name]))
        require(TARGET in wrappers and len(wrappers) == 8, "Exactly fixed stage02 required")
        yield wrappers


def forward_probe(model, functional, targets, shapes, chunk, selected, merge, torch,
                  mode="actual", capture=False, capture_u=False, expected_u_sha=None,
                  capture_tiles=(), capture_bn=False, deadline=lambda: None):
    deadline(); functional.reset_net(model)
    observed = {name: {"fc1_rows": 0, "fc2_rows": 0} for name in targets}
    branches, captured = {}, {}
    bn_inputs = {}
    handles = []
    begin = time.monotonic()
    try:
        if capture_bn:
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    bn_inputs[name] = {"track_running_stats": module.track_running_stats,
                                       "training": module.training,
                                       "running_mean_exists": module.running_mean is not None,
                                       "input_calls": []}
                    def capture_bn_input(module, inputs, layer=name):
                        x = inputs[0]
                        bn_inputs[layer]["input_calls"].append({"shape": list(x.shape), "stride": list(x.stride()),
                                                               "dtype": str(x.dtype), "device": str(x.device)})
                    handles.append(module.register_forward_pre_hook(capture_bn_input))
        for name, module in targets.items():
            for fc in ("fc1", "fc2"):
                def count(module, inputs, layer=name, key=fc+"_rows"):
                    observed[layer][key] += inputs[0].numel()//inputs[0].shape[-1]
                handles.append(getattr(module, fc).register_forward_pre_hook(count))
        with wrap(model, targets, shapes, selected, mode, merge, torch) as wrappers:
            if capture_u:
                def store_u(module, inputs):
                    require("u" not in captured, "Duplicate target input capture")
                    captured["u"] = inputs[0].detach().clone()
                handles.append(wrappers[TARGET].register_forward_pre_hook(store_u))
            if capture_tiles:
                def store_tiles(module, inputs, output):
                    require("tile_branches" not in captured, "Duplicate tile output capture")
                    captured["tile_branches"] = {(r, c): output[:, :, r*9:(r+1)*9, c*9:(c+1)*9].detach().cpu().clone()
                                                 for r, c in capture_tiles}
                handles.append(wrappers[TARGET].register_forward_hook(store_tiles))
            if capture:
                for name, wrapper in wrappers.items():
                    def store(module, inputs, output, layer=name):
                        branches[layer] = output.detach().cpu().clone()
                    handles.append(wrapper.register_forward_hook(store))
            with torch.inference_mode():
                output = model(chunk)
            require(isinstance(output["flow"], (list, tuple)) and len(output["flow"]) > 0, "Native flow list required")
            require(all(bool(torch.isfinite(x).all()) for x in output["flow"]), "Nonfinite flow")
            flows = [x.detach().cpu().clone() for x in output["flow"]]
            require(all(len(w.records) == 1 for w in wrappers.values()), "One execution per wrapped layer required")
            records = {name: w.records[0] for name, w in wrappers.items()}
        if expected_u_sha is not None:
            require(records[TARGET]["input_sha256"] == expected_u_sha, "Intervention changed predecision input")
        for name, counts in observed.items():
            rows = records[name]["executed_fc_rows_per_layer"] if name in records else math.prod(shapes[name][:-1])
            require(counts["fc1_rows"] == counts["fc2_rows"] == rows, "Independent FC audit differs: " + name)
        return flows, {"layers": records, "all12_observed_fc_rows": observed,
                       "summed_fc1_rows": sum(x["fc1_rows"] for x in observed.values()),
                       "summed_fc2_rows": sum(x["fc2_rows"] for x in observed.values()),
                       "batch_norm_inputs": bn_inputs,
                       "wall_seconds_diagnostic_only": time.monotonic()-begin}, branches, captured
    finally:
        for handle in handles:
            handle.remove()


def feature_scores(u, native, merge, torch):
    """Only pre-target input; PSN1 is explicitly extra computation, never free."""
    require(not torch.is_grad_enabled(), "Feature audit is inference-only")
    require(tuple(u.shape[:2]) == (10, 1), "Complete T10 required")
    features = []
    for r, c in full_tiles(u.shape):
        x = u[:, :, 9*r:9*(r+1), 9*c:9*(c+1)].contiguous()
        representative, spans = merge.spatial_mean(x, 3)
        restored = merge.repeat_branch(representative, spans, 9, 9)
        spikes = native.sn1(x)
        require(spikes.shape == x.shape and bool(((spikes == 0) | (spikes == 1)).all()), "PSN1 must return binary spikes")
        features.append({"tile": [r, c], "amplitude_abs_mean": float(x.abs().mean()),
                         "feature_rms": float(x.square().mean().sqrt()),
                         "pooling_residual_mse": float((x-restored).square().mean()),
                         "spatial_neighbor_abs_difference": float(((x[:, :, 1:]-x[:, :, :-1]).abs().mean() +
                                                                    (x[:, :, :, 1:]-x[:, :, :, :-1]).abs().mean())/2),
                         "temporal_neighbor_abs_difference": float((x[1:]-x[:-1]).abs().mean()),
                         "psn1_activity": float(spikes.mean()),
                         "psn1_spatial_neighbor_hamming": float(((spikes[:, :, 1:] != spikes[:, :, :-1]).float().mean()+
                                                                   (spikes[:, :, :, 1:] != spikes[:, :, :, :-1]).float().mean())/2)})
    n = len(features); t, b, _, _, channels = u.shape
    cost = {"diagnostic_only_extra_psn1_calls": n,
            "diagnostic_only_extra_psn1_elements": n*81*t*b*channels,
            "diagnostic_only_extra_psn1_temporal_MACs_nominal": n*81*t*t*b*channels,
            "extra_fc_calls": 0,
            "score_reductions_and_feature_reads_counted_in_cycles": False,
            "scope": "Input scores and PSN1 scores are ordinary controls, not official ADAPT implementation; PSN1 scores require full-tile neuron computation before selection and are not free"}
    return features, cost


def top2(scores):
    require(set(scores) == set(CANDIDATES) and all(math.isfinite(x) for x in scores.values()), "Twelve finite candidate scores required")
    return tuple(sorted(sorted(scores, key=lambda tile: (-scores[tile], tile))[:2]))


def combination_masks(frame):
    base_aee = frame["arms"][BASE_ARM]["metrics"]["regions"]["all_valid"]["AEE"]
    task = {(r, c): base_aee-frame["arms"][f"fine_r{r}_c{c}"]["metrics"]["regions"]["all_valid"]["AEE"] for r, c in CANDIDATES}
    ordinary = {tuple(x["tile"]): x for x in frame["ordinary_scores"]}
    branch = {tuple(x["tile"]): x["MSE"] for x in frame["offline_fine_coarse_branch_mse"]}
    return {"k2_posthoc_singleton_task_gain": top2(task),
            "k2_amplitude": top2({tile: x["amplitude_abs_mean"] for tile, x in ordinary.items()}),
            "k2_pooling_residual": top2({tile: x["pooling_residual_mse"] for tile, x in ordinary.items()}),
            "k2_fixed_uniform": (CANDIDATES[3], CANDIDATES[8]),
            "k2_offline_branch_mse": top2(branch)}


def prepare(a, parent):
    require(a.prior_scope_results_sha256 == SCOPE_RESULTS_SHA == sha(a.prior_scope_results), "Fixed ROUND4 result differs")
    own_check = parent.checked_cpu_receipt(a.runner_checks, [Path(__file__), HERE/"test_tile_interventions_cpu.py", BASE/"structured_merge_mlp.py"])
    inherited = argparse.Namespace(**vars(a))
    inherited.runner_checks = a.scope_runner_checks
    prepared = parent.prepare(inherited)
    c, b, context, base, scope, loader, dev, inherited_binding = prepared
    prior = json.loads(a.prior_scope_results.read_text())
    require(prior.get("schema") == "flow_structured_merge_scope_extension_v1" and
            prior.get("status") == "STRUCTURED_MERGE_SCOPE_EXTENSION_DEV8_COMPLETE" and
            prior.get("completed_targets") == 8 and prior.get("full_forward_count") == 59,
            "Completed ROUND4 scope result required")
    require(prior["binding"]["runner_sha256"] == SCOPE_RUNNER_SHA and
            prior["binding"]["module_sha256"] == parent.MODULE_SHA and
            prior["binding"]["checkpoint_sha256"] == parent.CHECKPOINT_SHA and
            prior["binding"]["dev_ids"] == inherited_binding["dev_ids"], "ROUND4 numerical source/order differs")
    require(prior["original_model_unchanged"] and prior["BN_buffers_unchanged"], "ROUND4 state audit missing")
    for frame in prior["frames"]:
        require(sha(a.prior_scope_results.parent/frame["prediction_file"]) == frame["prediction_file_sha256"], "ROUND4 prediction source differs")
    require(full_tiles(parent.STAGE_SHAPES[2]) == CANDIDATES, "Fixed twelve candidate tiles differ")
    binding = {"inherited_data_and_numerics": inherited_binding, "runner_sha256": sha(__file__),
               "runner_checks_sha256": own_check, "scope_runner_sha256": SCOPE_RUNNER_SHA,
               "module_sha256": parent.MODULE_SHA, "checkpoint_sha256": parent.CHECKPOINT_SHA,
               "prior_scope_results_sha256": SCOPE_RESULTS_SHA, "target": TARGET,
               "dev_ids": inherited_binding["dev_ids"], "arms": ARMS, "candidate_tiles": CANDIDATES,
               "all_fine_candidates_same_81_spatial_points": True,
               "tail_rule": "All tails remain coarse; evaluation still includes every valid GT region",
               "full_forward_budget_maximum": 146, "singleton_and_dual_forwards": 106,
               "combination_arms": COMBINATION_ARMS, "fixed_uniform_candidate_indices": [3, 8],
               "combination_k": 2, "tie_rule": "score descending, lexicographic tile ascending",
               "combination_protocol": "Fixed before reading this intervention's GT results; five actual k2 arms, identical masks deduplicated per frame",
               "extra_psn1_score_calls": 8*12,
               "GPU_lock": str(parent.GPU_LOCK), "wall_minutes": a.wall_minutes,
               "scope": "Offline singleton interventions on reused dev8; no GT in ordinary scores; no deployable predictor, additive joint-gain, or net hardware gain claim"}
    return c, b, context, base, scope, loader, dev, binding


def execute(a, prepared, result, parent):
    import numpy as np
    import torch
    merge = parent.load_frozen_merge()
    c, b, context, base, scope, loader, dev, binding = prepared
    require(not torch.cuda.is_initialized(), "CUDA initialized before lock")
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(20261014)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0"); torch.cuda.set_device(device); torch.cuda.manual_seed_all(20261014)
    started = time.monotonic()
    def deadline():
        require(time.monotonic()-started < a.wall_minutes*60, "Fixed wall budget exhausted")
    native, trainer, metrics, dev32, hashes, config, saved, side, regions, plan = context
    model, functional = loader.load_model(b, context, base, torch, device)
    bn_info = scope.apply_scope(model, "MLP_CMA176", b.bn_payload, torch)
    model.eval().requires_grad_(False)
    targets, shapes = parent.target_layers(model)
    before, bn_before = parent.source_digest(model), base.bn_signature(model, torch)
    result.update(status="RUNNING", runtime={"torch": str(torch.__version__), "cuda": torch.version.cuda,
                  "cudnn": torch.backends.cudnn.version(), "device_name": torch.cuda.get_device_name(device),
                  "device": str(device), "TF32": False, "AMP": False, "threads": 2},
                  original_state_sha256=before, BN_scope=bn_info, frames=[], first_frame_pairs={}, full_forward_count=0)
    meters = {arm: metrics.MetricAccumulator() for arm in [x[0] for x in ARMS] + list(COMBINATION_ARMS)}
    prior_frames = {x["target_id"]: x for x in json.loads(a.prior_scope_results.read_text())["frames"]}
    with torch.inference_mode():
        for index, row in enumerate(dev):
            deadline(); expected = {}; evidence = {}
            for key, relative in native.target_files(row).items():
                native.checked_path(b.data_root, relative)
                expected["events" if key == "input" else key] = hashes[relative]["sha256"]
            arrays, _ = metrics.load_arrays(b.data_root/"saved_flow_data", row, evidence=evidence, expected_hashes=expected)
            masks = metrics.partitions(arrays["gt"], arrays["mask"], arrays["events"], regions)
            chunk = base.prepare_input(arrays["events"], device, torch)
            frame = {"target_id": row["target_id"], "sequence": row["sequence"], "input_evidence": evidence, "arms": {}}
            predictions, input_hash, baseline_tiles, fine_tiles = {}, None, None, {}
            frame["offline_fine_coarse_branch_mse"] = []
            for arm, selected in ARMS:
                pair_check = index == 0 and arm in (BASE_ARM, "fine_r0_c0")
                flows, execution, branches, captured = forward_probe(model, functional, targets, shapes, chunk, selected,
                    merge, torch, capture=pair_check, capture_u=arm == BASE_ARM, expected_u_sha=input_hash,
                    capture_tiles=CANDIDATES if arm == BASE_ARM else selected,
                    capture_bn=index == 0 and arm == BASE_ARM, deadline=deadline)
                result["full_forward_count"] += 1
                pred = flows[-1][0].numpy().copy(); predictions[arm] = pred
                metric = meters[arm].update(row["target_id"], row["sequence"], pred, arrays["gt"], masks)
                frame["arms"][arm] = {"metrics": metric, "execution": execution,
                                      "prediction_sha256": hashlib.sha256(pred.tobytes()).hexdigest()}
                if arm == BASE_ARM:
                    u = captured["u"]
                    baseline_tiles = captured["tile_branches"]
                    input_hash = execution["layers"][TARGET]["input_sha256"]
                    require(merge.tensor_sha(u) == input_hash, "Captured predecision input differs")
                    frame["ordinary_scores"], frame["ordinary_score_cost"] = feature_scores(u, targets[TARGET], merge, torch)
                    input_path = a.output_dir/("input_"+row["target_id"]+".npz")
                    np.savez_compressed(input_path, predecision_u=u.cpu().numpy(),
                                        feature_names=np.asarray(SCORE_NAMES), candidate_tiles=np.asarray(CANDIDATES),
                                        no_GT_feature_vectors=np.asarray([[v[name] for name in SCORE_NAMES] for v in frame["ordinary_scores"]]))
                    frame["predecision_input_file"] = input_path.name
                    frame["predecision_input_file_sha256"] = sha(input_path)
                    frame["predecision_u_sha256"] = input_hash
                    old = prior_frames[row["target_id"]]
                    with np.load(a.prior_scope_results.parent/old["prediction_file"], allow_pickle=False) as prior_arrays:
                        require(np.array_equal(pred, prior_arrays[BASE_ARM]), "ROUND4 coarse baseline prediction changed")
                    require(metric == old["arms"][BASE_ARM]["metrics"], "ROUND4 coarse baseline metrics changed")
                    frame["ROUND4_coarse_baseline_prediction_and_metrics_exact"] = True
                    if index == 0:
                        require(len(execution["batch_norm_inputs"]) == 78 and all(x["input_calls"] for x in execution["batch_norm_inputs"].values()),
                                "All 78 BatchNorm modules must execute in first coarse baseline")
                    del u
                else:
                    baseline_rows = frame["arms"][BASE_ARM]["execution"]["summed_fc1_rows"]
                    require(execution["summed_fc1_rows"] - baseline_rows == 720, "Singletons do not have equal selected FC service")
                    tile = selected[0]
                    fine_tiles[tile] = captured["tile_branches"][tile]
                    frame["offline_fine_coarse_branch_mse"].append({"tile": list(tile),
                        "MSE": float((fine_tiles[tile].double()-baseline_tiles[tile].double()).square().mean()),
                        "scope": "Offline counterfactual branch-output score consuming fine and coarse producers, not a free predecision feature"})
                if pair_check:
                    dual, dual_record, dual_branches, _ = forward_probe(model, functional, targets, shapes, chunk, selected,
                        merge, torch, mode="dual_reference", capture=True, expected_u_sha=input_hash, deadline=deadline)
                    result["full_forward_count"] += 1
                    pair = {"flow_comparisons": [loader.compare(x, y, torch) for x, y in zip(flows, dual)],
                            "layers": {name: {"input_sha_equal": execution["layers"][name]["input_sha256"] == dual_record["layers"][name]["input_sha256"],
                                              "branch_comparison": loader.compare(branches[name], dual_branches[name], torch)} for name in branches},
                            "actual_execution": execution, "dual_execution": dual_record}
                    result["first_frame_pairs"][arm] = pair
                    require(len(flows) == len(dual) and all(x["numerically_equal"] for x in pair["flow_comparisons"]) and
                            all(x["input_sha_equal"] and x["branch_comparison"]["numerically_equal"] for x in pair["layers"].values()),
                            "Actual/dual comparison failed")
                    del dual, dual_branches
                del flows, branches
                del captured
            branch_path = a.output_dir/("branches_"+row["target_id"]+".npz")
            np.savez_compressed(branch_path, candidate_tiles=np.asarray(CANDIDATES),
                                coarse=np.stack([baseline_tiles[x].numpy() for x in CANDIDATES]),
                                singleton_fine=np.stack([fine_tiles[x].numpy() for x in CANDIDATES]))
            frame["offline_branch_file"] = branch_path.name; frame["offline_branch_file_sha256"] = sha(branch_path)
            del baseline_tiles, fine_tiles
            masks_by_arm = combination_masks(frame)
            frame["combination_masks"] = masks_by_arm
            executed_masks = {}
            for arm in COMBINATION_ARMS:
                selected = masks_by_arm[arm]
                if selected in executed_masks:
                    alias = executed_masks[selected]
                    pred = predictions[alias]
                    execution = frame["arms"][alias]["execution"]
                else:
                    flows, execution, _, _ = forward_probe(model, functional, targets, shapes, chunk, selected,
                        merge, torch, expected_u_sha=input_hash, deadline=deadline)
                    result["full_forward_count"] += 1
                    pred = flows[-1][0].numpy().copy()
                    executed_masks[selected] = arm
                    alias = None
                    require(execution["summed_fc1_rows"]-frame["arms"][BASE_ARM]["execution"]["summed_fc1_rows"] == 1440,
                            "k2 masks have different selected FC service")
                    del flows
                predictions[arm] = pred
                frame["arms"][arm] = {"metrics": meters[arm].update(row["target_id"], row["sequence"], pred, arrays["gt"], masks),
                                      "execution": execution, "prediction_sha256": hashlib.sha256(pred.tobytes()).hexdigest(),
                                      "identical_mask_execution_alias": alias}
            frame["combination_unique_forward_count"] = len(executed_masks)
            output_path = a.output_dir/("dev_"+row["target_id"]+".npz")
            np.savez_compressed(output_path, **predictions)
            frame["prediction_file"] = output_path.name; frame["prediction_file_sha256"] = sha(output_path)
            result["frames"].append(frame)
            result["summaries"] = {arm: meter.summary() for arm, meter in meters.items()}
            parent.write(a.output_dir/"results.json", result)
            print(json.dumps({"target_id": row["target_id"], "completed_targets": index+1, "full_forwards": result["full_forward_count"]}), flush=True)
            del chunk, predictions
    require(parent.source_digest(model) == before and base.bn_signature(model, torch) == bn_before, "Native state/BN changed")
    require(all(model.get_submodule(name) is m for name, m in targets.items()), "Native modules not restored")
    require(result["full_forward_count"] == 106+sum(x["combination_unique_forward_count"] for x in result["frames"]) and
            result["full_forward_count"] <= 146, "Full forward budget differs")
    result.update(status="TILE_INTERVENTIONS_DEV8_COMPLETE", completed_targets=8,
                  original_model_unchanged=True, BN_buffers_unchanged=True, original_modules_restored=True,
                  wall_seconds=time.monotonic()-started, hardware_benefit_claim=False,
                  scope="Same dev8, fixed stage02 base; real same-cost singleton interventions and five actual k2 policies with identical-mask deduplication. Task-gain reference uses offline GT singleton ranking and is not a jointly optimal oracle; gains are never obtained by summing singleton gains")


def main(argv=None):
    parent = load_scope_runner()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-contract", "identity-bank", "prior-screen-results", "prior-scope-results"):
        parser.add_argument("--"+name, type=Path, required=True)
        parser.add_argument("--"+name+"-sha256", required=True)
    parser.add_argument("--module-checks", type=Path, default=BASE/"cpu_results.json")
    parser.add_argument("--scope-runner-checks", type=Path, default=BASE/"scope_extension_v1/runner_cpu_results.json")
    parser.add_argument("--runner-checks", type=Path, default=HERE/"runner_cpu_results.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--wall-minutes", type=float, default=30)
    parser.add_argument("--execute", action="store_true")
    a = parser.parse_args(argv)
    prepared = prepare(a, parent)
    result = {"schema": "flow_tile_intervention_probe_v1", "status": "PREFLIGHT_PASS_NO_INFERENCE",
              "binding": prepared[-1], "native_preflight": prepared[2][-1],
              "inference_executed": False, "training_performed": False}
    if not a.execute:
        require(a.preflight_output and not a.preflight_output.exists(), "New preflight output required")
        parent.write(a.preflight_output, result); print(result["status"]); return
    require(parent.GPU_LOCK.parent.is_dir(), "Existing shared GPU lock directory required")
    with parent.GPU_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a.output_dir.mkdir(parents=True, exist_ok=False)
        previous = signal.getsignal(signal.SIGALRM)
        def timeout(signum, frame):
            raise TimeoutError("Tile intervention probe reached its fixed wall budget")
        signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, a.wall_minutes*60)
        try:
            result["inference_executed"] = True
            execute(a, prepared, result, parent)
        except BaseException as exc:
            result.update(status="INCOMPLETE_NO_SCIENTIFIC_CONCLUSION", error=type(exc).__name__+": "+str(exc), traceback=traceback.format_exc())
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)
            parent.write(a.output_dir/"results.json", result)


if __name__ == "__main__":
    main()
