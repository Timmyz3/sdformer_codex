#!/usr/bin/env python3
"""Predeclared ordinary SNN score controls, at most 24 new full forwards.

TP-Spikformer official SDT score formula on continuous pre-MLP u is a named
MLP/tile mechanism adaptation, not a full-method reproduction. All three masks
are GT-free. Equal masks may reuse immutable parent predictions, independently
of which parent policy happened to execute that mask.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROBE_SHA = "68ca81dc2dada3c6ce17f1d7edd7b762e992808c6f2fc359f5fe104cc20e214f"
TP_SHA = "7d0db322ac276cfaa459cb6ece4b41722b85714bd25feb4f53ddd4529093108a"
ARMS = ("k2_tp_official_sdt_tile", "k2_psn1_activity", "k2_psn1_spatial_hamming")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(data)
    return h.hexdigest()


def pinned_import(path, pin, name):
    require(sha(path) == pin, "Pinned source differs: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_dependencies():
    probe = pinned_import(HERE/"run_tile_interventions.py", PROBE_SHA, "_supplement_frozen_probe")
    tp = pinned_import(HERE/"frozen_tp_score.py", TP_SHA, "_supplement_frozen_tp")
    return probe, probe.load_scope_runner(), tp


def logical_score_cost(base_cost):
    """Count semantic score volumes, explicitly not GPU kernels or PE cycles."""
    t, b, c, h, w = base_cost["input_shape_TBCHW"]
    n, spatial = b*h*w, h*w
    return {"official_formula_cost": base_cost,
            "semantic_reduction_and_nonlinear_volumes": {
                "channel_mean_additions": t*n*(c-1), "channel_mean_reciprocal_multiplications": t*n,
                "temporal_difference_and_absolute_each": (t-1)*n,
                "temporal_change_mean_additions": (t-2)*n, "temporal_change_reciprocal_multiplications": n,
                "time_mean_additions": (t-1)*n*c, "time_mean_reciprocal_multiplications": n*c,
                "cosine_dot_products": n, "cosine_dot_vector_length": c,
                "cosine_norms": 2*n, "cosine_norm_vector_length": c,
                "cosine_logical_square_roots": 2*n, "cosine_norm_eps_clamps": 2*n,
                "softmax_calls": 2, "softmax_full_map_elements": 2*n,
                "softmax_stable_max_reduction_comparisons": 2*b*(spatial-1),
                "softmax_centering_subtractions": 2*n, "softmax_exponentials": 2*n,
                "softmax_sum_reduction_additions": 2*b*(spatial-1), "softmax_normalization_divisions": 2*n,
                "temporal_plus_spatial_additions": n,
                "complete_tile_mean_additions": 12*(81-1), "complete_tile_mean_reciprocal_multiplications": 12},
            "unpriced": ["cosine kernel arithmetic implementation", "top2 comparisons and metadata",
                         "layout/feature/score traffic", "validation copies and synchronization", "PE scheduling"],
            "scope": "Full official dense channel-mixing 3x3 convolution and two whole-map softmaxes retained; semantic volumes only, not cycle/RTL/traffic measurements"}


def masks_and_scores(u, ordinary, probe, tp, torch):
    """No GT/prediction arguments: full-map TP followed by complete-tile means."""
    require(tuple(u.shape) == (10, 1, 30, 40, 384) and u.dtype == torch.float32, "Fixed continuous FP32 u shape required")
    scores = tp.official_sdt_scores(u.permute(0, 1, 4, 2, 3).contiguous())
    masks, records = masks_from_score_maps(scores, ordinary, probe)
    records[ARMS[0]]["cost"] = logical_score_cost(scores["cost"])
    return masks, records, scores


def masks_from_score_maps(scores, ordinary, probe):
    total = scores["total"]
    require(tuple(total.shape) == (1, 30, 40), "Fixed full-layer TP score shape required")
    ordinary_by_tile = {tuple(x["tile"]): x for x in ordinary}
    require(set(ordinary_by_tile) == set(probe.CANDIDATES), "Ordinary score candidates differ")
    tp_scores = {(r, c): float(total[0, r*9:(r+1)*9, c*9:(c+1)*9].mean()) for r, c in probe.CANDIDATES}
    activity = {tile: x["psn1_activity"] for tile, x in ordinary_by_tile.items()}
    hamming = {tile: x["psn1_spatial_neighbor_hamming"] for tile, x in ordinary_by_tile.items()}
    values = dict(zip(ARMS, (tp_scores, activity, hamming)))
    masks = {name: probe.top2(value) for name, value in values.items()}
    records = {name: {"unit_scores": [{"tile": list(tile), "score": value[tile]} for tile in probe.CANDIDATES],
                      "GT_used_for_selection": False,
                      "selection_rule": "Complete tile score descending, lexicographic tile tie; fixed k2"}
               for name, value in values.items()}
    records[ARMS[0]].update(input_domain="Continuous signed pre-MLP u, not assumed binary spikes",
                           mechanism="TP-Spikformer official no-previous-mask SDT formula followed by complete 9x9 tile mean; MLP/tile adaptation, not full-method reproduction",
                           source_sha256=TP_SHA)
    for name in ARMS[1:]:
        records[name].update(input_domain="True PSN1 outputs on each complete 9x9 tile from pinned parent diagnostic",
                            cost_scope="Precomputes full candidate PSN1 neurons; parent extra cost retained, not free runtime features")
    return masks, records


def parent_alias(parent_frame, selected):
    """Identity-only lookup; never inspect quality or the parent selection score."""
    matches = [name for name, mask in parent_frame["combination_masks"].items()
               if tuple(tuple(x) for x in mask) == tuple(selected)]
    return sorted(matches)[0] if matches else None


def prepare(a, probe, parent):
    require(a.parent_probe_results_sha256 == sha(a.parent_probe_results), "Explicit parent probe result hash differs")
    checks = parent.checked_cpu_receipt(a.runner_checks, [Path(__file__), HERE/"test_supplemental_snn_scores_cpu.py", HERE/"frozen_tp_score.py"])
    inherited = argparse.Namespace(**vars(a)); inherited.runner_checks = a.probe_runner_checks
    prepared = probe.prepare(inherited, parent)
    c, b, context, base, scope, loader, dev, probe_binding = prepared
    source = json.loads(a.parent_probe_results.read_text())
    require(source.get("schema") == "flow_tile_intervention_probe_v1" and source.get("status") == "TILE_INTERVENTIONS_DEV8_COMPLETE" and
            source.get("completed_targets") == 8 and 106 <= source.get("full_forward_count", 0) <= 146,
            "Completed pinned main intervention probe required")
    require(source["binding"]["runner_sha256"] == PROBE_SHA and source["binding"]["dev_ids"] == probe_binding["dev_ids"] and
            source["binding"]["checkpoint_sha256"] == parent.CHECKPOINT_SHA and source["binding"]["module_sha256"] == parent.MODULE_SHA,
            "Main probe source/identities differ")
    require(source["original_model_unchanged"] and source["BN_buffers_unchanged"] and source["original_modules_restored"], "Parent state audit missing")
    require([frame["target_id"] for frame in source["frames"]] == probe_binding["dev_ids"], "Parent frame order differs")
    for frame in source["frames"]:
        require(frame["ROUND4_coarse_baseline_prediction_and_metrics_exact"], "Parent baseline reproduction missing")
        for key in ("prediction_file", "predecision_input_file"):
            require(sha(a.parent_probe_results.parent/frame[key]) == frame[key+"_sha256"], "Parent artifact differs: "+key)
    binding = {"runner_sha256": sha(__file__), "runner_checks_sha256": checks, "probe_runner_sha256": PROBE_SHA,
               "tp_adaptation_sha256": TP_SHA, "parent_probe_results_sha256": a.parent_probe_results_sha256,
               "inherited_data_numerics_and_geometry": probe_binding,
               "checkpoint_sha256": parent.CHECKPOINT_SHA, "module_sha256": parent.MODULE_SHA,
               "dev_ids": probe_binding["dev_ids"], "arms": ARMS, "k": 2,
               "maximum_new_full_forwards": 24, "GT_used_for_selection": False,
               "freeze_rule": "These three formulas and k2 fixed before reading main probe GT results; no score formula or candidate search",
               "score_directions": "higher score keeps fine", "GPU_lock": str(parent.GPU_LOCK), "wall_minutes": a.wall_minutes}
    return c, b, context, base, scope, loader, dev, binding, source


def execute(a, prepared, result, probe, parent, tp):
    import numpy as np
    import torch
    c, b, context, base, scope, loader, dev, binding, source = prepared
    merge = parent.load_frozen_merge()
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
    require(before == source["original_state_sha256"], "Model state differs from main probe")
    runtime = {"torch": str(torch.__version__), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
               "device_name": torch.cuda.get_device_name(device), "device": str(device), "TF32": False, "AMP": False, "threads": 2}
    require(runtime == source["runtime"], "Runtime differs from pinned main probe")
    result.update(status="RUNNING", runtime=runtime, original_state_sha256=before, BN_scope=bn_info,
                  frames=[], new_full_forward_count=0, TP_full_map_score_calls=0)
    meters = {name: metrics.MetricAccumulator() for name in (probe.BASE_ARM,)+ARMS}
    with torch.inference_mode():
        for row, prior in zip(dev, source["frames"]):
            deadline()
            with np.load(a.parent_probe_results.parent/prior["predecision_input_file"], allow_pickle=False) as cached:
                u = torch.from_numpy(cached["predecision_u"].copy()).to(device)
                require(tuple(cached["feature_names"].tolist()) == probe.SCORE_NAMES and
                        np.array_equal(cached["candidate_tiles"], np.asarray(probe.CANDIDATES)), "Parent feature coordinates differ")
                require(np.array_equal(cached["no_GT_feature_vectors"], np.asarray([[v[name] for name in probe.SCORE_NAMES] for v in prior["ordinary_scores"]])),
                        "Parent GT-free feature cache differs from JSON")
            require(merge.tensor_sha(u) == prior["predecision_u_sha256"], "Parent captured input hash differs")
            # Masks are completely fixed here, before loading GT arrays for evaluation.
            selected_masks, score_records, score_maps = masks_and_scores(u, prior["ordinary_scores"], probe, tp, torch)
            result["TP_full_map_score_calls"] += 1
            score_path = a.output_dir/("tp_scores_"+row["target_id"]+".npz")
            np.savez_compressed(score_path, **{key: score_maps[key].cpu().numpy() for key in ("temporal", "spatial", "total")})
            del u, score_maps
            frame = {"target_id": row["target_id"], "sequence": row["sequence"], "selected_masks": selected_masks,
                     "scores": score_records, "TP_score_file": score_path.name, "TP_score_file_sha256": sha(score_path),
                     "parent_predecision_u_sha256": prior["predecision_u_sha256"],
                     "PSN_control_score_cost": prior["ordinary_score_cost"], "arms": {}}
            expected, evidence = {}, {}
            for key, relative in native.target_files(row).items():
                native.checked_path(b.data_root, relative)
                expected["events" if key == "input" else key] = hashes[relative]["sha256"]
            arrays, _ = metrics.load_arrays(b.data_root/"saved_flow_data", row, evidence=evidence, expected_hashes=expected)
            masks = metrics.partitions(arrays["gt"], arrays["mask"], arrays["events"], regions)
            frame["input_evidence"] = evidence
            with np.load(a.parent_probe_results.parent/prior["prediction_file"], allow_pickle=False) as saved_predictions:
                cached_predictions = {name: saved_predictions[name].copy() for name in saved_predictions.files}
            predictions = {probe.BASE_ARM: cached_predictions[probe.BASE_ARM]}
            base_metric = meters[probe.BASE_ARM].update(row["target_id"], row["sequence"], predictions[probe.BASE_ARM], arrays["gt"], masks)
            require(base_metric == prior["arms"][probe.BASE_ARM]["metrics"], "Parent baseline metric replay differs")
            frame["arms"][probe.BASE_ARM] = {"metrics": base_metric, "execution_source": "pinned_parent", "parent_arm": probe.BASE_ARM}
            chunk, executed = None, {}
            for arm in ARMS:
                selected = selected_masks[arm]
                alias = parent_alias(prior, selected)
                if alias is not None:
                    pred = cached_predictions[alias]
                    execution = prior["arms"][alias]["execution"]
                    execution_source, source_arm = "pinned_parent_identical_mask", alias
                elif selected in executed:
                    source_arm = executed[selected]
                    pred = predictions[source_arm]
                    execution = frame["arms"][source_arm]["execution"]
                    execution_source = "supplement_identical_mask"
                else:
                    if chunk is None:
                        chunk = base.prepare_input(arrays["events"], device, torch)
                    flows, execution, _, _ = probe.forward_probe(model, functional, targets, shapes, chunk, selected, merge, torch,
                        expected_u_sha=prior["predecision_u_sha256"], deadline=deadline)
                    result["new_full_forward_count"] += 1
                    pred = flows[-1][0].numpy().copy(); del flows
                    executed[selected] = arm
                    execution_source, source_arm = "new_actual_full_forward", None
                require(execution["summed_fc1_rows"]-prior["arms"][probe.BASE_ARM]["execution"]["summed_fc1_rows"] == 1440,
                        "Supplement selected producer service differs from main k2")
                target_record = execution["layers"][probe.TARGET]
                require(target_record["input_sha256"] == prior["predecision_u_sha256"], "Supplement execution input hash differs")
                executed_tiles = tuple((r, c) for r, values in enumerate(target_record["fine_mask"]) for c, fine in enumerate(values) if fine)
                require(executed_tiles == selected, "Supplement selected mask differs from execution or alias")
                predictions[arm] = pred
                frame["arms"][arm] = {"metrics": meters[arm].update(row["target_id"], row["sequence"], pred, arrays["gt"], masks),
                                      "prediction_sha256": hashlib.sha256(pred.tobytes()).hexdigest(), "execution": execution,
                                      "execution_source": execution_source, "source_arm": source_arm}
            output_path = a.output_dir/("dev_"+row["target_id"]+".npz")
            np.savez_compressed(output_path, **predictions)
            frame["prediction_file"] = output_path.name; frame["prediction_file_sha256"] = sha(output_path)
            frame["new_unique_forward_count"] = len(executed)
            result["frames"].append(frame); result["summaries"] = {name: meter.summary() for name, meter in meters.items()}
            parent.write(a.output_dir/"results.json", result)
            print(json.dumps({"target_id": row["target_id"], "completed_targets": len(result["frames"]),
                              "new_full_forwards": result["new_full_forward_count"]}), flush=True)
            del chunk, cached_predictions, predictions
    require(parent.source_digest(model) == before and base.bn_signature(model, torch) == bn_before, "Original model/BN changed")
    require(all(model.get_submodule(name) is original for name, original in targets.items()), "Original modules not restored")
    require(result["new_full_forward_count"] == sum(x["new_unique_forward_count"] for x in result["frames"]) <= 24 and
            result["TP_full_map_score_calls"] == 8, "Fixed supplement execution budget differs")
    result.update(status="SUPPLEMENTAL_SNN_SCORES_DEV8_COMPLETE", completed_targets=8,
                  original_model_unchanged=True, BN_buffers_unchanged=True, original_modules_restored=True,
                  wall_seconds=time.monotonic()-started, hardware_benefit_claim=False,
                  scope="Three predeclared GT-free ordinary SNN score controls on same dev8 and same k2 producer cost. TP retains official score formula as MLP/tile adaptation; costs differ and net cycle gains are unmeasured")


def main(argv=None):
    probe, parent, tp = load_dependencies()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-contract", "identity-bank", "prior-screen-results", "prior-scope-results", "parent-probe-results"):
        parser.add_argument("--"+name, type=Path, required=True)
        parser.add_argument("--"+name+"-sha256", required=True)
    parser.add_argument("--module-checks", type=Path, default=probe.BASE/"cpu_results.json")
    parser.add_argument("--scope-runner-checks", type=Path, default=probe.BASE/"scope_extension_v1/runner_cpu_results.json")
    parser.add_argument("--probe-runner-checks", type=Path, default=HERE/"runner_cpu_results.json")
    parser.add_argument("--runner-checks", type=Path, default=HERE/"supplemental_cpu_results.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--wall-minutes", type=float, default=30)
    parser.add_argument("--execute", action="store_true")
    a = parser.parse_args(argv)
    prepared = prepare(a, probe, parent)
    result = {"schema": "flow_supplemental_snn_scores_v1", "status": "PREFLIGHT_PASS_NO_INFERENCE",
              "binding": prepared[-2], "native_preflight": prepared[2][-1], "training_performed": False, "inference_executed": False}
    if not a.execute:
        require(a.preflight_output and not a.preflight_output.exists(), "New preflight output required")
        parent.write(a.preflight_output, result); print(result["status"]); return
    require(parent.GPU_LOCK.parent.is_dir(), "Existing GPU lock directory required")
    with parent.GPU_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a.output_dir.mkdir(parents=True, exist_ok=False)
        previous = signal.getsignal(signal.SIGALRM)
        def timeout(signum, frame):
            raise TimeoutError("Supplemental SNN controls reached fixed wall budget")
        signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, a.wall_minutes*60)
        try:
            result["inference_executed"] = True
            execute(a, prepared, result, probe, parent, tp)
        except BaseException as exc:
            result.update(status="INCOMPLETE_NO_SCIENTIFIC_CONCLUSION", error=type(exc).__name__+": "+str(exc), traceback=traceback.format_exc())
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)
            parent.write(a.output_dir/"results.json", result)


if __name__ == "__main__":
    main()
