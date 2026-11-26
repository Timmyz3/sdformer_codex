#!/usr/bin/env python3
"""New-member stage02 singleton damage cache, inherited frozen ROUND5 execution.

Default preflight, explicit execute and shared GPU lock, no model optimization.
GT is retained only as offline outcomes and coarse-head training labels.
"""
import argparse
import csv
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
PROBE_PATH = HERE.parent / "damage_allocation_probe/run_tile_interventions.py"
PROBE_SHA = "68ca81dc2dada3c6ce17f1d7edd7b762e992808c6f2fc359f5fe104cc20e214f"
FIT_SHA = "29ec7ba9c9faca74c62c3e170c9d3c70855a4605a7eab31f9331c519f56f8915"
CHECK64_SHA = "310d7c55fe5dc625d7b974bd03482513d46227254105501c088c17e580997a92"
CHECKPOINT_SHA = "d5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30"
TIMESTAMP_SHA = "533726df5e50f31b576a0d5972fe171fca0c55d0500b99a5a8514b6c0e143304"
COARSE_HEAD_SHA = "eb0b5e3c3e675c4466a42b73c77237735460913fd79452b9eb9a662226f42ef5"
FLOW_CONTRACT = {"direction": "forward_t_start_to_t_end_rectified_left_event_camera",
                 "time_interval": "same-target forward GT pair; per-target timestamps bound separately",
                 "units": "original_image_pixels"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pinned(path, pin, name):
    require(sha(path) == pin, "Pinned source differs: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_probe():
    return load_pinned(PROBE_PATH, PROBE_SHA, "_round6_frozen_probe")


def select_rows(manifest, fit, canonical, roles, excluded_ids, requested_group="all"):
    """Identity-only selection; no image, event, feature, prediction or GT input."""
    require(manifest.get("schema") == "flow_damage_round6_selection_v1", "Unknown selection manifest")
    require(manifest.get("checkpoint_sha256") == fit.get("checkpoint_sha256") == CHECKPOINT_SHA,
            "Selection checkpoint differs")
    require(manifest.get("source_sha256", {}).get("fit20") == FIT_SHA, "Fit membership binding differs")
    groups = manifest.get("groups")
    require(isinstance(groups, dict) and set(groups) == {"fit", "check"}, "Exactly fit/check groups required")
    require(requested_group in ("fit", "check", "all"), "Unknown requested group")
    by_id = {row["target_id"]: row for row in canonical}
    require(len(by_id) == len(canonical), "Duplicate canonical identity")
    fit_ids = fit["target_ids"]
    require(len(fit_ids) == len(set(fit_ids)) == 20, "Fixed fit20 identities differ")
    require(groups["fit"] == fit_ids, "Fit group must use all fixed fit20 in original order")
    require(len(groups["check"]) == 8, "Fixed check8 required")
    require(set(manifest.get("excluded_ids", [])) == set(excluded_ids).union(fit_ids), "Prior-member exclusion list differs")
    sequences = {}
    seen = set()
    for group in ("fit", "check"):
        ids = groups[group]
        require(isinstance(ids, list) and ids and all(isinstance(x, str) for x in ids), "Nonempty identity list required")
        require(len(ids) == len(set(ids)) and not seen.intersection(ids), "Group identity overlap/duplicate")
        require(set(ids) <= set(by_id), "Member outside canonical")
        if group == "check":
            require(not set(ids).intersection(set(excluded_ids).union(fit_ids)), "Prior member overlap prohibited")
        rows = [by_id[x] for x in ids]
        expected_role = "gate_fit" if group == "fit" else "gate_development"
        require(all(row["protocol_split"] == "train" and roles.get(row["target_id"]) == expected_role for row in rows),
                "Group auxiliary role or train split differs")
        sequences[group] = {row["sequence"] for row in rows}
        seen.update(ids)
    require(not sequences["fit"].intersection(sequences["check"]), "Fit/check sequence overlap prohibited")
    selected_groups = ("fit", "check") if requested_group == "all" else (requested_group,)
    chosen = [(group, by_id[target]) for group in selected_groups for target in groups[group]]
    require(len(chosen) <= 28, "At most fixed fit20/check8 members")
    return chosen


def cell_targets(gt, valid_mask, *, feature_hw, feature_stride_pixels=(16, 16),
                 image_origin_xy=(0., 0.), cell_size=3):
    """Generalize frozen coarse_head.make_tile_targets 9-cell support to 3 cells.

    Same half-open origin0 image support and pixel centers; never rescale dx/dy.
    No events argument exists: valid zero-event GT is included automatically.
    """
    import torch
    require(type(cell_size) is int and cell_size > 0, "Positive integer cell_size required")
    require(len(feature_hw) == 2 and all(type(x) is int and x > 0 for x in feature_hw), "Positive feature shape required")
    require(len(feature_stride_pixels) == 2 and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in feature_stride_pixels),
            "Positive feature strides required")
    require(len(image_origin_xy) == 2 and all(type(x) in (int, float) and math.isfinite(x) for x in image_origin_xy),
            "Finite origin required")
    require(isinstance(gt, torch.Tensor) and gt.ndim == 3 and gt.shape[0] == 2 and gt.is_floating_point(), "GT must be floating 2HW")
    require(isinstance(valid_mask, torch.Tensor) and valid_mask.shape == gt.shape[1:] and valid_mask.device == gt.device,
            "GT mask shape/device differs")
    require(bool(((valid_mask == 0) | (valid_mask == 1)).all()), "Valid mask must be binary")
    valid = valid_mask.bool()
    require(bool(torch.isfinite(gt[:, valid]).all()), "Valid GT is nonfinite")
    h, w = feature_hw; sx, sy = feature_stride_pixels; ox, oy = image_origin_xy
    gh, gw = gt.shape[1:]; ny, nx = math.ceil(h/cell_size), math.ceil(w/cell_size)
    flow = gt.new_full((1, ny, nx, 2), float("nan"))
    count = torch.zeros((1, ny, nx), dtype=torch.long, device=gt.device)
    support = torch.zeros_like(count); bounds = []
    with torch.no_grad():
        for y in range(ny):
            row = []
            for x in range(nx):
                left, right = ox+x*cell_size*sx, ox+min((x+1)*cell_size, w)*sx
                top, bottom = oy+y*cell_size*sy, oy+min((y+1)*cell_size, h)*sy
                x0, x1 = max(0, min(gw, math.ceil(left-.5))), max(0, min(gw, math.ceil(right-.5)))
                y0, y1 = max(0, min(gh, math.ceil(top-.5))), max(0, min(gh, math.ceil(bottom-.5)))
                row.append([left, top, right, bottom])
                part = valid[y0:y1, x0:x1]; n = int(part.sum())
                count[0, y, x] = n; support[0, y, x] = (y1-y0)*(x1-x0)
                if n:
                    flow[0, y, x] = gt[:, y0:y1, x0:x1][:, part].mean(dim=1)
            bounds.append(row)
    return {"flow": flow, "valid": count > 0, "valid_pixel_count": count,
            "support_pixel_count": support, "physical_bounds_xyxy": bounds,
            "flow_contract": FLOW_CONTRACT, "cell_size_feature_yx": [cell_size, cell_size],
            "feature_hw": list(feature_hw), "feature_stride_pixels_xy": list(feature_stride_pixels),
            "image_origin_xy": list(image_origin_xy),
            "label_aggregation": "valid-only arithmetic mean; no flow-unit rescaling",
            "support_convention": "half-open image support; pixel centers x+.5,y+.5"}


def cell_descriptors(u):
    import torch
    require(tuple(u.shape) == (10, 1, 30, 40, 384), "Fixed stage2 complete T10 input required")
    require(u.is_floating_point() and bool(torch.isfinite(u).all()), "Finite continuous input required")
    # Fixed merge.spatial_mean reduction shape/order, then full-T mean. Do not
    # silently commute the reductions: FP32 results need not stay bit equal.
    pooled_rows = []
    for y in range(0, 30, 3):
        pooled_rows.append(torch.stack([u[:, :, y:y+3, x:x+3].sum(dim=(2, 3))/(3*min(3, 40-x))
                                        for x in range(0, 40, 3)], dim=2))
    spatial = torch.stack(pooled_rows, dim=2)
    descriptor = spatial.mean(dim=0)
    require(tuple(descriptor.shape) == (1, 10, 14, 384), "Cell descriptor shape differs")
    return descriptor


def verify_cell_partition(targets, gt, mask, old_head):
    """Exact support/count conservation; valid-count aggregation checks old labels."""
    import torch
    require(int(targets["valid_pixel_count"].sum()) == int(mask.sum()), "Valid GT lost in cell labels")
    require(int(targets["support_pixel_count"].sum()) == gt.shape[1]*gt.shape[2], "Image support lost in cell labels")
    old = old_head.make_tile_targets(gt, mask, feature_hw=(30, 40), feature_stride_pixels=(16, 16),
                                     flow_contract=FLOW_CONTRACT, image_origin_xy=(0., 0.))
    max_error = 0.
    for r in range(4):
        for c in range(5):
            counts = targets["valid_pixel_count"][0, 3*r:3*r+3, 3*c:3*c+3]
            values = targets["flow"][0, 3*r:3*r+3, 3*c:3*c+3]
            n = int(counts.sum())
            require(n == int(old["valid_pixel_count"][0, r, c]), "Subcell/tile GT counts differ")
            require(int(targets["support_pixel_count"][0, 3*r:3*r+3, 3*c:3*c+3].sum()) == int(old["support_pixel_count"][0, r, c]),
                    "Subcell/tile support differs")
            if n:
                weighted = (values[counts > 0].double()*counts[counts > 0].unsqueeze(-1)).sum(dim=0)/n
                ref = old["flow"][0, r, c].double()
                max_error = max(max_error, float((weighted-ref).abs().max()))
                require(torch.allclose(weighted, ref, atol=2e-5, rtol=2e-5), "Subcell weighted means differ from old tile support")
    return {"all_valid_GT_included": True, "full_image_support_pixels": int(targets["support_pixel_count"].sum()),
            "valid_GT_pixels": int(targets["valid_pixel_count"].sum()), "old_9x9_count_support_exact": True,
            "weighted_3x3_to_old_9x9_mean_max_abs": max_error,
            "mean_tolerance_atol_rtol": [2e-5, 2e-5], "GT_units_rescaled": False}


def prepare(a, probe, parent):
    inherited = argparse.Namespace(**vars(a)); inherited.runner_checks = a.probe_runner_checks
    prepared = probe.prepare(inherited, parent)
    c, b, context, base, scope, loader, old_dev, old_binding = prepared
    own_check = parent.checked_cpu_receipt(a.runner_checks, [Path(__file__), HERE/"test_collect_singletons_cpu.py"])
    require(a.fit_membership_sha256 == FIT_SHA == sha(a.fit_membership), "Fixed fit20 membership source differs")
    require(a.prior_check_membership_sha256 == CHECK64_SHA == sha(a.prior_check_membership), "Fixed check64 membership source differs")
    require(a.selection_manifest_sha256 == sha(a.selection_manifest), "Selection manifest SHA differs")
    fit = json.loads(a.fit_membership.read_text()); manifest = json.loads(a.selection_manifest.read_text())
    require(fit["canonical_sha256"] == sha(b.canonical) and fit["roles_sha256"] == sha(b.roles), "Fixed fit20 canonical/roles differs")
    native, trainer, metrics, dev32, hashes, config, saved, side, regions, plan = context
    canonical = metrics.canonical_rows(b.canonical); roles = metrics.auxiliary_roles(b.roles, canonical)
    check64 = json.loads(a.prior_check_membership.read_text())
    require(manifest["source_sha256"]["canonical"] == sha(b.canonical) and manifest["source_sha256"]["roles"] == sha(b.roles)
            and manifest["source_sha256"]["check64"] == CHECK64_SHA, "Selection source binding differs")
    excluded = {r["target_id"] for r in dev32}.union(check64["target_ids"])
    chosen = select_rows(manifest, fit, canonical, roles, excluded, a.group)
    motion = Path(c["motion_root"])
    timing = load_pinned(motion/"collect_coarse_cache.py", TIMESTAMP_SHA, "_round6_frozen_timestamps")
    old_head = load_pinned(motion/"coarse_head.py", COARSE_HEAD_SHA, "_round6_frozen_label_reference")
    acceptance = json.loads(b.acceptance.read_text()); inventory_path = b.acceptance.parent/"native_deployment_manifest.csv"
    require(sha(inventory_path) == acceptance["deployment_csv_sha256"], "Native manifest differs")
    with inventory_path.open(newline="") as stream:
        inventory = {row["relative_path"]: row for row in csv.DictReader(stream)}
    data_audit = []
    for group, row in chosen:
        paths = native.target_files(row)
        for relative in paths.values():
            path = native.checked_path(b.data_root, relative)
            require(relative in hashes and sha(path) == hashes[relative]["sha256"], "New-member accepted array differs")
        data_audit.append({"target_id": row["target_id"], "group": group,
            "timestamps": timing.timestamp_binding(b.data_root, row, paths, hashes, inventory, acceptance)})
    binding = {"inherited_numerical_and_source_contract": old_binding,
        "inherited_dev8_use": "Numerical/source receipts only; no dev8 predictions, features or damage labels used for collection",
        "runner_sha256": sha(__file__), "runner_checks_sha256": own_check,
        "frozen_R5_probe_sha256": PROBE_SHA, "checkpoint_sha256": CHECKPOINT_SHA,
        "fit_membership_sha256": FIT_SHA, "prior_check_membership_sha256": CHECK64_SHA,
        "selection_manifest_sha256": sha(a.selection_manifest),
        "selection_manifest": manifest, "requested_group": a.group,
        "target_ids": [row["target_id"] for group, row in chosen], "groups": [group for group, row in chosen],
        "group_sequence_disjoint": True, "new_member_data_preflight": data_audit,
        "coarse_head_label_reference_sha256": COARSE_HEAD_SHA, "timestamp_helper_sha256": TIMESTAMP_SHA,
        "flow_contract": FLOW_CONTRACT, "cell_feature_support": {"shape_BHWC": [1, 10, 14, 384],
            "spatial_cell": [3, 3], "complete_T": 10, "feature_stride_pixels_xy": [16, 16],
            "image_origin_xy": [0, 0], "right_tail_real_feature_columns": 1,
            "aggregation_order": "3x3 real-point spatial mean per T including tail, then complete T10 mean"},
        "arms": probe.ARMS, "target": probe.TARGET, "candidate_tiles": probe.CANDIDATES,
        "full_forward_budget": len(chosen)*13+2, "extra_PSN1_score_calls": len(chosen)*12,
        "GPU_lock": str(parent.GPU_LOCK), "wall_minutes": a.wall_minutes,
        "scope": "Train-side backbone-seen new singleton labels; no k2, model optimization or deployed selector; check member separation does not remove historical sequence exposure or make final validation"}
    return c, b, context, base, scope, loader, chosen, binding, old_head


def execute(a, prepared, result, probe, parent):
    import numpy as np
    import torch
    c, b, context, base, scope, loader, chosen, binding, old_head = prepared
    require(not torch.cuda.is_initialized(), "CUDA initialized before GPU lock")
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
    targets, shapes = parent.target_layers(model); merge = parent.load_frozen_merge()
    before, bn_before = parent.source_digest(model), base.bn_signature(model, torch)
    result.update(status="RUNNING", runtime={"torch": str(torch.__version__), "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "device_name": torch.cuda.get_device_name(device),
        "device": str(device), "TF32": False, "AMP": False, "threads": 2},
        original_state_sha256=before, BN_scope=bn_info, frames=[], first_frame_pairs={}, full_forward_count=0)
    meters = {arm: metrics.MetricAccumulator() for arm, _ in probe.ARMS}
    timestamp_by_id = {x["target_id"]: x["timestamps"] for x in binding["new_member_data_preflight"]}
    with torch.inference_mode():
        for index, (group, row) in enumerate(chosen):
            deadline(); evidence = {}; expected = {}
            for key, relative in native.target_files(row).items():
                native.checked_path(b.data_root, relative)
                expected["events" if key == "input" else key] = hashes[relative]["sha256"]
            arrays, _ = metrics.load_arrays(b.data_root/"saved_flow_data", row, evidence=evidence, expected_hashes=expected)
            masks = metrics.partitions(arrays["gt"], arrays["mask"], arrays["events"], regions)
            chunk = base.prepare_input(arrays["events"], device, torch)
            frame = {"target_id": row["target_id"], "sequence": row["sequence"], "group": group,
                     "input_evidence": evidence, "timestamps": timestamp_by_id[row["target_id"]], "arms": {},
                     "offline_fine_coarse_branch_mse": []}
            predictions, input_hash, baseline_tiles, fine_tiles = {}, None, None, {}
            for arm, selected in probe.ARMS:
                pair_check = index == 0 and arm in (probe.BASE_ARM, "fine_r0_c0")
                flows, execution, branches, captured = probe.forward_probe(model, functional, targets, shapes, chunk, selected,
                    merge, torch, capture=pair_check, capture_u=arm == probe.BASE_ARM, expected_u_sha=input_hash,
                    capture_tiles=probe.CANDIDATES if arm == probe.BASE_ARM else selected,
                    capture_bn=index == 0 and arm == probe.BASE_ARM, deadline=deadline)
                result["full_forward_count"] += 1
                pred = flows[-1][0].numpy().copy(); predictions[arm] = pred
                metric = meters[arm].update(row["target_id"], row["sequence"], pred, arrays["gt"], masks)
                frame["arms"][arm] = {"metrics": metric, "execution": execution,
                                      "prediction_sha256": hashlib.sha256(pred.tobytes()).hexdigest()}
                if arm == probe.BASE_ARM:
                    u = captured["u"]; baseline_tiles = captured["tile_branches"]
                    input_hash = execution["layers"][probe.TARGET]["input_sha256"]
                    require(merge.tensor_sha(u) == input_hash, "Captured predecision input differs")
                    frame["ordinary_scores"], frame["ordinary_score_cost"] = probe.feature_scores(u, targets[probe.TARGET], merge, torch)
                    descriptor = cell_descriptors(u).cpu()
                    gt, mask = torch.from_numpy(arrays["gt"]), torch.from_numpy(arrays["mask"])
                    labels = cell_targets(gt, mask, feature_hw=(30, 40))
                    label_audit = verify_cell_partition(labels, gt, mask, old_head)
                    zero = cell_targets(gt, torch.from_numpy(masks["voxel_zero"]), feature_hw=(30, 40))
                    require(int(zero["valid_pixel_count"].sum()) == int(masks["voxel_zero"].sum()), "Zero-event valid GT lost")
                    label_audit["voxel_zero_valid_GT_pixels"] = int(zero["valid_pixel_count"].sum())
                    input_path = a.output_dir/("input_"+row["target_id"]+".npz")
                    np.savez_compressed(input_path, predecision_u=u.cpu().numpy(),
                        feature_names=np.asarray(probe.SCORE_NAMES), candidate_tiles=np.asarray(probe.CANDIDATES),
                        no_GT_feature_vectors=np.asarray([[v[name] for name in probe.SCORE_NAMES] for v in frame["ordinary_scores"]]),
                        no_GT_cell_descriptor=descriptor.numpy(), cell_GT_flow=labels["flow"].numpy(),
                        cell_GT_valid=labels["valid"].numpy(), cell_GT_valid_pixel_count=labels["valid_pixel_count"].numpy(),
                        cell_support_pixel_count=labels["support_pixel_count"].numpy(),
                        cell_voxel_zero_valid_pixel_count=zero["valid_pixel_count"].numpy(),
                        cell_physical_bounds_xyxy=np.asarray(labels["physical_bounds_xyxy"]))
                    frame.update(predecision_input_file=input_path.name, predecision_input_file_sha256=sha(input_path),
                                 predecision_u_sha256=input_hash, cell_label_audit=label_audit)
                    if index == 0:
                        require(len(execution["batch_norm_inputs"]) == 78 and all(x["input_calls"] for x in execution["batch_norm_inputs"].values()),
                                "All 78 BN modules must execute in first baseline")
                    del u, descriptor, labels, zero
                else:
                    baseline_rows = frame["arms"][probe.BASE_ARM]["execution"]["summed_fc1_rows"]
                    require(execution["summed_fc1_rows"]-baseline_rows == 720, "Singleton actual FC service differs")
                    tile = selected[0]; fine_tiles[tile] = captured["tile_branches"][tile]
                    frame["offline_fine_coarse_branch_mse"].append({"tile": list(tile),
                        "MSE": float((fine_tiles[tile].double()-baseline_tiles[tile].double()).square().mean()),
                        "scope": "Offline fine/coarse counterfactual branch score, not free predecision information"})
                if pair_check:
                    dual, dual_record, dual_branches, _ = probe.forward_probe(model, functional, targets, shapes, chunk, selected,
                        merge, torch, mode="dual_reference", capture=True, expected_u_sha=input_hash, deadline=deadline)
                    result["full_forward_count"] += 1
                    pair = {"flow_comparisons": [loader.compare(x, y, torch) for x, y in zip(flows, dual)],
                        "layers": {name: {"input_sha_equal": execution["layers"][name]["input_sha256"] == dual_record["layers"][name]["input_sha256"],
                            "branch_comparison": loader.compare(branches[name], dual_branches[name], torch)} for name in branches},
                        "actual_execution": execution, "dual_execution": dual_record}
                    result["first_frame_pairs"][arm] = pair
                    require(len(flows) == len(dual) and all(x["numerically_equal"] for x in pair["flow_comparisons"]) and
                        all(x["input_sha_equal"] and x["branch_comparison"]["numerically_equal"] for x in pair["layers"].values()), "Actual/dual differs")
                    del dual, dual_branches
                del flows, branches, captured
            branch_path = a.output_dir/("branches_"+row["target_id"]+".npz")
            np.savez_compressed(branch_path, candidate_tiles=np.asarray(probe.CANDIDATES),
                coarse=np.stack([baseline_tiles[x].numpy() for x in probe.CANDIDATES]),
                singleton_fine=np.stack([fine_tiles[x].numpy() for x in probe.CANDIDATES]))
            frame["offline_branch_file"] = branch_path.name; frame["offline_branch_file_sha256"] = sha(branch_path)
            output_path = a.output_dir/("predictions_"+row["target_id"]+".npz")
            np.savez_compressed(output_path, **predictions)
            frame["prediction_file"] = output_path.name; frame["prediction_file_sha256"] = sha(output_path)
            result["frames"].append(frame)
            result["summaries"] = {arm: meter.summary() for arm, meter in meters.items()}
            result["summaries_by_group"] = {g: {arm: metrics.MetricAccumulator.summarize_frames(
                [f["arms"][arm]["metrics"] for f in result["frames"] if f["group"] == g]) for arm, _ in probe.ARMS}
                for g in dict.fromkeys(f["group"] for f in result["frames"])}
            parent.write(a.output_dir/"results.json", result)
            print(json.dumps({"target_id": row["target_id"], "group": group, "completed_targets": index+1,
                              "full_forwards": result["full_forward_count"]}), flush=True)
            del chunk, predictions, baseline_tiles, fine_tiles
    require(parent.source_digest(model) == before and base.bn_signature(model, torch) == bn_before, "Native/BN state changed")
    require(all(model.get_submodule(name) is m for name, m in targets.items()), "Native modules not restored")
    require(result["full_forward_count"] == binding["full_forward_budget"], "Full forward count differs")
    result.update(status="ROUND6_SINGLETON_COLLECTION_COMPLETE", completed_targets=len(chosen),
        original_model_unchanged=True, BN_buffers_unchanged=True, original_modules_restored=True,
        wall_seconds=time.monotonic()-started, hardware_benefit_claim=False,
        scope="Fresh stage02 coarse-prefix u and same-u singleton final-flow losses; all valid GT plus every partition; fit/check train-side groups, no k2, training, selector claim or net hardware gain")


def main(argv=None):
    probe = load_probe(); parent = probe.load_scope_runner()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-contract", "identity-bank", "prior-screen-results", "prior-scope-results", "selection-manifest", "fit-membership", "prior-check-membership"):
        parser.add_argument("--"+name, type=Path, required=True); parser.add_argument("--"+name+"-sha256", required=True)
    parser.add_argument("--group", choices=("fit", "check", "all"), default="all")
    parser.add_argument("--module-checks", type=Path, default=probe.BASE/"cpu_results.json")
    parser.add_argument("--scope-runner-checks", type=Path, default=probe.BASE/"scope_extension_v1/runner_cpu_results.json")
    parser.add_argument("--probe-runner-checks", type=Path, default=PROBE_PATH.parent/"runner_cpu_results.json")
    parser.add_argument("--runner-checks", type=Path, default=HERE/"collector_cpu_results.json")
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--wall-minutes", type=float, default=30); parser.add_argument("--execute", action="store_true")
    a = parser.parse_args(argv); prepared = prepare(a, probe, parent)
    result = {"schema": "flow_damage_round6_singletons_v1", "status": "PREFLIGHT_PASS_NO_INFERENCE",
        "binding": prepared[-2], "native_preflight": prepared[2][-1],
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
            raise TimeoutError("Round6 collection reached its fixed wall budget")
        signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, a.wall_minutes*60)
        try:
            result["inference_executed"] = True; execute(a, prepared, result, probe, parent)
        except BaseException as exc:
            result.update(status="INCOMPLETE_NO_SCIENTIFIC_CONCLUSION", error=type(exc).__name__+": "+str(exc), traceback=traceback.format_exc())
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)
            parent.write(a.output_dir/"results.json", result)


if __name__ == "__main__":
    main()
