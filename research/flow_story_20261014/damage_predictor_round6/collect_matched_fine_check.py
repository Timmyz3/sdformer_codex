#!/usr/bin/env python3
"""Result-triggered matched all-fine stage02 diagnostic on fixed check8; no fitting.

Keeps the same checkpoint and BN scope. Native arithmetic is not claimed equal
to any tile execution graph. Default preflight; root executes after main run.
"""
import argparse
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
COLLECTOR_PATH = HERE/"collect_singletons.py"
COLLECTOR_SHA = "ade75ced3543accad2f3f7222c2a02a6606df7b95035317fa1f47aff4f05f991"
ARM = "fine_stage02"


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


def native_forward(parent, model, functional, targets, shapes, chunk, merge, torch, deadline=lambda: None):
    flows, execution, branches = parent.forward_probe(model, functional, targets, shapes, chunk,
                                                      "fine", (0, 2), merge, torch, deadline=deadline)
    require(len(execution["layers"]) == 8 and branches == {}, "Exactly eight all-fine stage02 wrappers required")
    expected = sum(math.prod(shape[:-1]) for shape in shapes.values())
    require(execution["summed_fc1_rows"] == execution["summed_fc2_rows"] == expected,
            "Native full-spatial FC row count differs")
    return flows, execution


def prepare(a, collector, probe, parent):
    require(0 < a.wall_minutes <= 10, "Native check anchor at most 10 minutes")
    inherited = argparse.Namespace(**vars(a)); inherited.runner_checks = a.collector_runner_checks
    inherited.group = "check"
    prepared = collector.prepare(inherited, probe, parent)
    require(len(prepared[6]) == 8 and all(group == "check" for group, row in prepared[6]), "Exactly check8 required")
    check = parent.checked_cpu_receipt(a.runner_checks, [Path(__file__), HERE/"test_matched_fine_check_cpu.py"])
    binding = {"inherited_data_and_numerics": prepared[7], "runner_sha256": sha(__file__),
               "runner_checks_sha256": check, "collector_source_sha256": COLLECTOR_SHA,
               "native_forward_source_sha256": probe.SCOPE_RUNNER_SHA,
               "checkpoint_sha256": collector.CHECKPOINT_SHA, "BN_scope": "MLP_CMA176",
               "target_ids": [row["target_id"] for group, row in prepared[6]],
               "arms": [ARM], "full_forward_budget": 8,
               "GPU_lock": str(parent.GPU_LOCK), "wall_minutes": a.wall_minutes,
               "scope": "Result-triggered matched tiled all-fine stage02 quality diagnostic on fixed check8. Not consumed by heads, selectors, masks or fitting; not numerically equivalent to tiled arithmetic; no hardware gain claim"}
    return prepared, binding


def execute(a, prepared, result, parent):
    import numpy as np
    import torch
    inherited, binding = prepared
    c, b, context, base, scope, loader, chosen, inherited_binding, old_head = inherited
    require(not torch.cuda.is_initialized(), "CUDA initialized before native-anchor lock")
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(20261014)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0"); torch.cuda.set_device(device); torch.cuda.manual_seed_all(20261014)
    started = time.monotonic()
    def deadline():
        require(time.monotonic()-started < a.wall_minutes*60, "Native anchor wall budget exhausted")
    native, trainer, metrics, dev32, hashes, config, saved, side, regions, plan = context
    model, functional = loader.load_model(b, context, base, torch, device)
    bn_info = scope.apply_scope(model, "MLP_CMA176", b.bn_payload, torch)
    model.eval().requires_grad_(False)
    targets, shapes = parent.target_layers(model); merge = parent.load_frozen_merge()
    before, bn_before = parent.source_digest(model), base.bn_signature(model, torch)
    require(sum(math.prod(s[:-1]) for s in shapes.values()) == 558000, "Declared native FC service differs")
    result.update(status="RUNNING", runtime={"torch": str(torch.__version__), "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "device_name": torch.cuda.get_device_name(device),
        "device": str(device), "TF32": False, "AMP": False, "threads": 2},
        original_state_sha256=before, BN_scope=bn_info, frames=[], full_forward_count=0)
    meter = metrics.MetricAccumulator()
    timestamp_by_id = {x["target_id"]: x["timestamps"] for x in inherited_binding["new_member_data_preflight"]}
    with torch.inference_mode():
        for group, row in chosen:
            deadline(); expected = {}; evidence = {}
            for key, relative in native.target_files(row).items():
                native.checked_path(b.data_root, relative)
                expected["events" if key == "input" else key] = hashes[relative]["sha256"]
            arrays, _ = metrics.load_arrays(b.data_root/"saved_flow_data", row, evidence=evidence, expected_hashes=expected)
            masks = metrics.partitions(arrays["gt"], arrays["mask"], arrays["events"], regions)
            chunk = base.prepare_input(arrays["events"], device, torch)
            flows, execution = native_forward(parent, model, functional, targets, shapes, chunk, merge, torch, deadline)
            result["full_forward_count"] += 1
            prediction = flows[-1][0].numpy().copy()
            metric = meter.update(row["target_id"], row["sequence"], prediction, arrays["gt"], masks)
            path = a.output_dir/("fine_stage02_"+row["target_id"]+".npz")
            np.savez_compressed(path, **{ARM: prediction})
            frame = {"target_id": row["target_id"], "sequence": row["sequence"], "group": "check",
                     "input_evidence": evidence, "timestamps": timestamp_by_id[row["target_id"]],
                     "arms": {ARM: {"metrics": metric, "execution": execution,
                                    "prediction_sha256": hashlib.sha256(prediction.tobytes()).hexdigest()}},
                     "prediction_file": path.name, "prediction_file_sha256": sha(path)}
            result["frames"].append(frame); result["summaries"] = {ARM: meter.summary()}
            parent.write(a.output_dir/"results.json", result)
            print(json.dumps({"target_id": row["target_id"], "completed_targets": len(result["frames"]),
                              "full_forwards": result["full_forward_count"]}), flush=True)
            del chunk, flows, prediction
    require(parent.source_digest(model) == before and base.bn_signature(model, torch) == bn_before, "Native/BN state changed")
    require(all(model.get_submodule(name) is m for name, m in targets.items()), "Native module identity changed")
    require(result["full_forward_count"] == len(result["frames"]) == 8, "Native check8 forward budget differs")
    result.update(status="ROUND6_MATCHED_FINE_CHECK8_COMPLETE", completed_targets=8, original_model_unchanged=True,
                  BN_buffers_unchanged=True, original_modules_restored=True, wall_seconds=time.monotonic()-started,
                  hardware_benefit_claim=False,
                  scope="Matched all-fine tiled stage02 diagnostic; all valid GT and partitions. Not used by frozen ROUND6 heads, selectors or ranking; native and tiled arithmetic not declared numerically equal")


def main(argv=None):
    collector = load_collector(); probe = collector.load_probe(); parent = probe.load_scope_runner()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-contract", "identity-bank", "prior-screen-results", "prior-scope-results", "selection-manifest", "fit-membership", "prior-check-membership"):
        parser.add_argument("--"+name, type=Path, required=True); parser.add_argument("--"+name+"-sha256", required=True)
    parser.add_argument("--module-checks", type=Path, default=probe.BASE/"cpu_results.json")
    parser.add_argument("--scope-runner-checks", type=Path, default=probe.BASE/"scope_extension_v1/runner_cpu_results.json")
    parser.add_argument("--probe-runner-checks", type=Path, default=probe.HERE/"runner_cpu_results.json")
    parser.add_argument("--collector-runner-checks", type=Path, default=HERE/"collector_cpu_results.json")
    parser.add_argument("--runner-checks", type=Path, default=HERE/"matched_fine_cpu_results.json")
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--wall-minutes", type=float, default=10); parser.add_argument("--execute", action="store_true")
    a = parser.parse_args(argv); prepared = prepare(a, collector, probe, parent)
    result = {"schema": "flow_damage_round6_matched_fine_check_v1", "status": "PREFLIGHT_PASS_NO_INFERENCE",
              "binding": prepared[1], "native_preflight": prepared[0][2][-1],
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
