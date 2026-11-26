#!/usr/bin/env python3
"""Post-result mixed stage0/2 diagnosis, exact20 full forwards, no training.
Adapted from R6 matched-fine runner and R4 instrumentation; both stages tiled.
"""
import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import sys
import time
import traceback
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
COLLECTOR_PATH = HERE.parent/"damage_predictor_round6/collect_singletons.py"
COLLECTOR_SHA = "ade75ced3543accad2f3f7222c2a02a6606df7b95035317fa1f47aff4f05f991"
ARMS = (("C0_F2", (0,)), ("F0_C2", (2,)))
BUDGET = 20


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def load_collector():
    require(sha(COLLECTOR_PATH) == COLLECTOR_SHA, "Frozen new-member collector differs")
    spec = importlib.util.spec_from_file_location("_round6_native_frozen_collector", COLLECTOR_PATH)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def mixed_forward(parent, model, functional, targets, shapes, chunk, coarse_stages, merge, torch,
                  mode="actual", capture=False, deadline=lambda: None):
    require(coarse_stages in ((), (0,), (2,), (0, 2)), "Fixed stage02 states only")
    original_wrapper = parent.wrapped_model
    @contextlib.contextmanager
    def wrapped(model, targets, shapes, kind, stages, mode, merge, torch):
        require(stages == (0, 2), "All arms use the same stage02 tiled support")
        wrappers = {}
        with contextlib.ExitStack() as stack:
            for name, original in targets.items():
                stage = next(s for s in parent.DEPTHS if f".layers.{s}." in name)
                if stage not in (0, 2):
                    continue
                shape = shapes[name]
                mask = torch.full((math.ceil(shape[2]/9), math.ceil(shape[3]/9)),
                                  stage not in coarse_stages, dtype=torch.bool)
                wrappers[name] = stack.enter_context(merge.replace_mlp(model, name, mask, 3, mode,
                                                                       expected_shape=shape))
            yield wrappers
    # This private imported parent's adapter is restored even on failure. Its
    # source and old experiment files stay immutable; execution instrumentation
    # is reused unchanged and every selected stage has a tiled wrapper.
    try:
        parent.wrapped_model = wrapped
        return parent.forward_probe(model, functional, targets, shapes, chunk, "coarse", (0, 2),
                                    merge, torch, mode=mode, capture=capture, deadline=deadline)
    finally:
        parent.wrapped_model = original_wrapper


def prepare(a, collector, probe, parent):
    require(0 < a.wall_minutes <= 10, "Native check anchor at most 10 minutes")
    inherited = argparse.Namespace(**vars(a)); inherited.runner_checks = a.collector_runner_checks
    inherited.group = "check"
    prepared = collector.prepare(inherited, probe, parent)
    require(len(prepared[6]) == 8 and all(group == "check" for group, row in prepared[6]), "Exactly check8 required")
    check = parent.checked_cpu_receipt(a.runner_checks, [Path(__file__), HERE/"test_repair_scope_cpu.py"])
    binding = {"inherited_data_and_numerics": prepared[7], "runner_sha256": sha(__file__),
               "runner_checks_sha256": check, "collector_source_sha256": COLLECTOR_SHA,
               "native_forward_source_sha256": probe.SCOPE_RUNNER_SHA,
               "checkpoint_sha256": collector.CHECKPOINT_SHA, "BN_scope": "MLP_CMA176",
               "target_ids": [row["target_id"] for group, row in prepared[6]],
               "arms": ARMS, "full_forward_budget": BUDGET,
               "GPU_lock": str(parent.GPU_LOCK), "wall_minutes": a.wall_minutes,
               "scope": "Post-result stage0/stage2 factorial diagnostic on reused R6check8. Fixed mixed arms, same allstage02 tiled arithmetic, no training or policy claim"}
    references = {}
    for key in ("reference_coarse", "reference_fine"):
        path = getattr(a, key)
        require(sha(path) == getattr(a, key+"_sha256"), "Reference SHA differs")
        reference = json.loads(path.read_text())
        refs = {f["target_id"]: f for f in reference["frames"] if f.get("group") == "check"}
        require(list(refs) == binding["target_ids"], "Reference check membership/order differs")
        references[key] = refs
        binding[key+"_sha256"] = sha(path)
    binding["extra_forwards"] = "First member two archived anchor replays plus both mixed dual references"
    return (prepared, references), binding


def execute(a, prepared, result, parent):
    import numpy as np
    import torch
    inherited_and_refs, binding = prepared
    inherited, references = inherited_and_refs
    c, b, context, base, scope, loader, chosen, inherited_binding, old_head = inherited
    require(not torch.cuda.is_initialized(), "CUDA initialized before GPU lock")
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(20261014)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0"); torch.cuda.set_device(device); torch.cuda.manual_seed_all(20261014)
    started = time.monotonic()
    def deadline():
        require(time.monotonic()-started < a.wall_minutes*60, "Fixed diagnostic wall budget exhausted")
    native, trainer, metrics, dev32, hashes, config, saved, side, regions, plan = context
    model, functional = loader.load_model(b, context, base, torch, device)
    bn_info = scope.apply_scope(model, "MLP_CMA176", b.bn_payload, torch)
    bn_info["execution"] = "Eight stage02 tiled wrappers in every arm; fixed complete-stage fine/coarse, no training"
    model.eval().requires_grad_(False)
    targets, shapes = parent.target_layers(model); merge = parent.load_frozen_merge()
    before, bn_before = parent.source_digest(model), base.bn_signature(model, torch)
    result.update(status="RUNNING", runtime={"torch": str(torch.__version__), "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "device_name": torch.cuda.get_device_name(device),
        "device": str(device), "TF32": False, "AMP": False, "threads": 2},
        original_state_sha256=before, BN_scope=bn_info, frames=[], full_forward_count=0,
        anchor_replays={}, dual_checks={})
    meters = {arm: metrics.MetricAccumulator() for arm, _ in ARMS}
    timestamp_by_id = {x["target_id"]: x["timestamps"] for x in inherited_binding["new_member_data_preflight"]}
    def forward(chunk, coarse, **kwargs):
        out = mixed_forward(parent, model, functional, targets, shapes, chunk, coarse, merge, torch,
                            deadline=deadline, **kwargs)
        result["full_forward_count"] += 1
        require(result["full_forward_count"] <= BUDGET, "Forward budget exceeded")
        return out
    with torch.inference_mode():
        for index, (group, row) in enumerate(chosen):
            deadline(); expected = {}; evidence = {}; tid = row["target_id"]
            for key, relative in native.target_files(row).items():
                native.checked_path(b.data_root, relative)
                expected["events" if key == "input" else key] = hashes[relative]["sha256"]
            arrays, _ = metrics.load_arrays(b.data_root/"saved_flow_data", row, evidence=evidence, expected_hashes=expected)
            for ref in references.values():
                require(ref[tid]["input_evidence"] == evidence and ref[tid]["timestamps"] == timestamp_by_id[tid],
                        "Archived input/timestamp evidence differs")
            masks = metrics.partitions(arrays["gt"], arrays["mask"], arrays["events"], regions)
            chunk = base.prepare_input(arrays["events"], device, torch)
            if index == 0:
                for key, prior_arm, coarse in (("reference_coarse", "coarse_stage02", (0, 2)),
                                               ("reference_fine", "fine_stage02", ())):
                    flows, execution, _ = forward(chunk, coarse)
                    prediction = flows[-1][0].numpy().copy()
                    digest = hashlib.sha256(prediction.tobytes()).hexdigest()
                    require(digest == references[key][tid]["arms"][prior_arm]["prediction_sha256"],
                            "Archived anchor replay not exact; stop before interpreting mixed quality")
                    result["anchor_replays"][prior_arm] = {"target_id": tid, "prediction_sha256": digest,
                                                          "exact_archived_prediction": True, "execution": execution}
            frame = {"target_id": tid, "sequence": row["sequence"], "group": "post_result_development",
                     "input_evidence": evidence, "timestamps": timestamp_by_id[tid], "arms": {}}
            predictions = {}
            for arm, coarse in ARMS:
                flows, execution, branches = forward(chunk, coarse, capture=index == 0)
                first0 = "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp"
                first2 = "sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.0.mlp"
                prefix_key = "reference_coarse" if 0 in coarse else "reference_fine"
                prefix_arm = "coarse_stage02" if 0 in coarse else "fine_stage02"
                prior_layers = references[prefix_key][tid]["arms"][prefix_arm]["execution"]["layers"]
                require(all(execution["layers"][name]["input_sha256"] == prior_layers[name]["input_sha256"]
                            for name in (first0, first2)), "Same-prefix first MLP input differs")
                prediction = flows[-1][0].numpy().copy(); predictions[arm] = prediction
                frame["arms"][arm] = {"metrics": meters[arm].update(tid, row["sequence"], prediction, arrays["gt"], masks),
                    "execution": execution, "same_prefix_input_matches": prefix_arm,
                    "prediction_sha256": hashlib.sha256(prediction.tobytes()).hexdigest()}
                if index == 0:
                    dual, dex, dbranch = forward(chunk, coarse, mode="dual_reference", capture=True)
                    require(all(torch.equal(x, y) for x, y in zip(flows, dual)), "Mixed actual/dual flow differs")
                    require(branches.keys() == dbranch.keys() and all(torch.equal(branches[k], dbranch[k]) for k in branches),
                            "Mixed actual/dual branches differ")
                    result["dual_checks"][arm] = {"flow_exact": True, "all8_branches_exact": True,
                                                 "actual_execution": execution, "dual_execution": dex}
            path = a.output_dir/(tid+".npz"); np.savez_compressed(path, **predictions)
            frame.update(prediction_file=path.name, prediction_file_sha256=sha(path))
            result["frames"].append(frame)
            result["summaries"] = {arm: meter.summary() for arm, meter in meters.items()}
            parent.write(a.output_dir/"results.json", result)
            print(json.dumps({"target_id": tid, "completed_targets": len(result["frames"]),
                              "full_forwards": result["full_forward_count"]}), flush=True)
            del chunk, flows, prediction
    require(parent.source_digest(model) == before and base.bn_signature(model, torch) == bn_before, "Model/BN state changed")
    require(all(model.get_submodule(name) is m for name, m in targets.items()), "Original modules not restored")
    require(result["full_forward_count"] == BUDGET and len(result["frames"]) == 8, "Fixed budget/membership differs")
    result.update(status="ROUND7_MIXED_SCOPE_DIAGNOSTIC_COMPLETE", completed_targets=8, original_model_unchanged=True,
                  BN_buffers_unchanged=True, original_modules_restored=True, wall_seconds=time.monotonic()-started,
                  hardware_benefit_claim=False, scope="Post-result diagnosis on all reused R6check8, no fitted selector or training. Fullstage rescues are diagnostic endpoints with different costs, not a dynamic policy or new validation")


def main(argv=None):
    collector = load_collector(); probe = collector.load_probe(); parent = probe.load_scope_runner()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-contract", "identity-bank", "prior-screen-results", "prior-scope-results", "selection-manifest", "fit-membership", "prior-check-membership"):
        parser.add_argument("--"+name, type=Path, required=True); parser.add_argument("--"+name+"-sha256", required=True)
    parser.add_argument("--module-checks", type=Path, default=probe.BASE/"cpu_results.json")
    parser.add_argument("--scope-runner-checks", type=Path, default=probe.BASE/"scope_extension_v1/runner_cpu_results.json")
    parser.add_argument("--probe-runner-checks", type=Path, default=probe.HERE/"runner_cpu_results.json")
    parser.add_argument("--collector-runner-checks", type=Path, default=HERE.parent/"damage_predictor_round6/collector_cpu_results.json")
    parser.add_argument("--runner-checks", type=Path, default=HERE/"runner_cpu_results.json")
    for name in ("reference-coarse", "reference-fine"):
        parser.add_argument("--"+name, type=Path, required=True); parser.add_argument("--"+name+"-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--wall-minutes", type=float, default=10); parser.add_argument("--execute", action="store_true")
    a = parser.parse_args(argv); prepared = prepare(a, collector, probe, parent)
    result = {"schema": "flow_round7_scope_repair_v1", "status": "PREFLIGHT_PASS_NO_INFERENCE",
              "binding": prepared[1], "native_preflight": prepared[0][0][2][-1],
              "inference_executed": False, "training_performed": False}
    if not a.execute:
        require(a.preflight_output and not a.preflight_output.exists(), "New native preflight output required")
        parent.write(a.preflight_output, result); print(result["status"]); return
    require(parent.GPU_LOCK.parent.is_dir(), "Existing shared GPU lock directory required")
    with parent.GPU_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a.output_dir.mkdir(parents=True, exist_ok=False)
        previous = signal.getsignal(signal.SIGALRM)
        def timeout(signum, frame):
            raise TimeoutError("Native check anchor reached its fixed wall budget")
        signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, a.wall_minutes*60)
        try:
            result["inference_executed"] = True; execute(a, prepared, result, parent)
        except BaseException as exc:
            result.update(status="INCOMPLETE_NO_SCIENTIFIC_CONCLUSION", error=type(exc).__name__+": "+str(exc), traceback=traceback.format_exc())
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)
            parent.write(a.output_dir/"results.json", result)


if __name__ == "__main__":
    main()
