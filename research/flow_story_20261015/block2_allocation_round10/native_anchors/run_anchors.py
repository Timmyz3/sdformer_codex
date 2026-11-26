#!/usr/bin/env python3
"""Outcome-triggered same-check native/FF/full-block2 anchors; exactly 27 forwards."""
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
R10 = HERE.parent
sys.path.insert(0, str(R10))
ARMS = ("native", "FF", "fine_s2b2")
ROWS = {"native": 558000, "FF": 558000, "CC": 153600, "fine_s2b2": 164200}
BUDGET = 27


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def pins():
    return json.loads((HERE/"source_pins.json").read_text())


def source_path(key):
    pin = pins()["sources"][key]
    path = R10/pin["relative_to_round10"]
    require(sha(path) == pin["sha256"], "Anchor dependency differs: "+key)
    return path


def load(key, name):
    spec = importlib.util.spec_from_file_location(name, source_path(key))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_runtime():
    for key in pins()["sources"]:
        source_path(key)
    r10 = load("round10_collector", "_anchors_frozen_r10")
    collector, probe, parent = r10.load_runtime()
    r9 = load("round9_runner", "_anchors_frozen_r9")
    r8 = r9.load_r8()
    require(sha(r8.__file__) == pins()["sources"]["round8_runner"]["sha256"], "R8 helper differs")
    return r10, collector, probe, parent, r9, r8


def expected_rows(shapes, arm, mode="actual"):
    require(arm in ARMS+("CC",), "Unknown anchor arm")
    rows = 0
    for name, (t, b, h, w, c) in shapes.items():
        if arm == "native" or not any(".layers.{}.".format(s) in name for s in (0, 2)):
            rows += t*b*h*w
        else:
            fine = arm == "FF" or (arm == "fine_s2b2" and ".layers.2.swin_blocks.2." in name)
            reps = sum(math.ceil(min(9, h-y)/3)*math.ceil(min(9, w-x)/3)
                       for y in range(0, h, 9) for x in range(0, w, 9))
            points = h*w+reps if mode == "dual_reference" else (h*w if fine else reps)
            rows += t*b*points
    return rows


def forward_arm(r9, r8, parent, model, functional, targets, shapes, chunk, arm, merge, torch, **kwargs):
    mode = kwargs.get("mode", "actual")
    out = r9.forward_arm(r8, parent, model, functional, targets, shapes, chunk, arm, merge, torch, **kwargs)
    execution = out[1]
    require(execution["summed_fc1_rows"] == execution["summed_fc2_rows"] == expected_rows(shapes, arm, mode),
            "Independent all12 FC service differs")
    names = set() if arm == "native" else {n for n in targets if any(".layers.{}.".format(s) in n for s in (0, 2))}
    require(set(execution["layers"]) == names and len(names) == (0 if arm == "native" else 8), "Wrapper scope differs")
    for name, layer in execution["layers"].items():
        fine = arm == "FF" or (arm == "fine_s2b2" and ".layers.2.swin_blocks.2." in name)
        h, w = shapes[name][2:4]
        require(layer["fine_mask"] == [[fine]*math.ceil(w/9) for _ in range(math.ceil(h/9))],
                "Whole-layer mask including true tails differs")
    return out


def check_prefix(execution, archived, prefix):
    layers = execution["layers"]
    old = archived["arms"]["coarse_stage02"]["execution"]["layers"]
    require(all(layers[n]["input_sha256"] == old[n]["input_sha256"] for n in prefix), "Archived CC prefix input differs")
    require(all(layers[n]["output_sha256"] == old[n]["output_sha256"] for n in prefix[:-1]), "Archived earlier prefix output differs")


def check_pair(actual, dual, torch):
    flows, ex, branches = actual
    dflows, dex, dbranches = dual
    require(len(flows) == len(dflows) and len(flows) > 0 and all(torch.equal(x, y) for x, y in zip(flows, dflows)),
            "Actual/dual complete flow list differs")
    require(set(branches) == set(dbranches) == set(ex["layers"]) == set(dex["layers"]) and len(branches) == 8,
            "Actual/dual branch coverage differs")
    require(all(torch.equal(branches[n], dbranches[n]) and ex["layers"][n]["input_sha256"] == dex["layers"][n]["input_sha256"]
                for n in branches), "Actual/dual branch or input differs")
    return {"flow_list_length": len(flows), "all_flows_exact": True, "all8_inputs_exact": True,
            "all8_branches_exact": True, "actual_execution": ex, "dual_execution": dex}


def prepare(a, r10, collector, probe, parent):
    require(0 < a.wall_minutes <= 10, "Anchor wall cap is 10 minutes")
    inherited = argparse.Namespace(**vars(a))
    inherited.group = "all"
    inherited.runner_checks = a.round10_runner_checks
    prepared = list(r10.prepare(inherited, collector, probe, parent))
    require(sha(a.reference_round10) == a.reference_round10_sha256 == pins()["round10_result_sha256"], "Complete R10 result differs")
    reference = json.loads(a.reference_round10.read_text())
    require(reference["status"] == r10.STATUS and reference["full_forward_count"] == 366 and reference["completed_targets"] == 28,
            "Complete R10 parent required")
    require(reference["original_model_unchanged"] and reference["BN_buffers_unchanged"] and reference["original_modules_restored"],
            "R10 parent closure differs")
    require(reference["binding"]["r10_membership_sha256"] == sha(source_path("membership")) and
            reference["binding"]["checkpoint_sha256"] == collector.CHECKPOINT_SHA and
            reference["binding"]["target"] == r10.TARGET, "Parent membership/checkpoint/target differs")
    chosen = [(group, row) for group, row in prepared[6] if group == "check"]
    refs = {f["target_id"]: f for f in reference["frames"] if f["group"] == "check"}
    member = json.loads(source_path("membership").read_text())
    require([row["target_id"] for _, row in chosen] == list(refs) == member["groups"]["check"] and len(chosen) == 8,
            "Exactly original R10 check8 in frozen order required")
    own = parent.checked_cpu_receipt(a.runner_checks, [Path(__file__), HERE/"test_anchors_cpu.py"])
    prepared[6] = chosen
    binding = {"schema": "flow_round10_same_check_anchor_binding_v1", "runner_sha256": sha(__file__),
        "runner_checks_sha256": own, "source_pins_sha256": sha(HERE/"source_pins.json"),
        "reference_round10_sha256": sha(a.reference_round10), "membership_sha256": sha(source_path("membership")),
        "inherited_data_and_numerical_contract": prepared[7], "checkpoint_sha256": collector.CHECKPOINT_SHA,
        "BN_scope": "MLP_CMA176", "target_ids": list(refs), "arms": ARMS, "full_forward_budget": BUDGET,
        "actual_anchor_forwards": 24, "extra_forwards": "First-check CC replay plus FF/fine_s2b2 dual references: 3",
        "GPU_lock": str(parent.GPU_LOCK), "wall_minutes": a.wall_minutes,
        "scope": "Outcome-triggered same-check diagnostic after failed R10 advancement. No selector change, new validation, native pass assumption or hardware benefit."}
    return tuple(prepared), reference, refs, binding


def execute(a, prepared, result, r10, r9, r8, parent):
    import numpy as np
    import torch
    inherited, reference, refs, binding = prepared
    c, b, context, base, scope, loader, chosen, oldbinding, oldhead = inherited
    require(not torch.cuda.is_initialized(), "CUDA initialized before lock")
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(20261014)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0"); torch.cuda.set_device(device); torch.cuda.manual_seed_all(20261014)
    started = time.monotonic()
    def deadline():
        require(time.monotonic()-started < a.wall_minutes*60, "Anchor wall budget exhausted")
    native, trainer, metrics, dev32, hashes, config, saved, side, regions, plan = context
    model, functional = loader.load_model(b, context, base, torch, device)
    bn_info = scope.apply_scope(model, "MLP_CMA176", b.bn_payload, torch)
    bn_info["execution"] = "Native arm has no MLP tile wrappers; FF and fine_s2b2/CC arms keep all eight stage02 wrappers including real tails"
    model.eval().requires_grad_(False)
    targets, shapes = parent.target_layers(model); merge = parent.load_frozen_merge()
    before, bn_before = parent.source_digest(model), base.bn_signature(model, torch)
    require(before == reference["original_state_sha256"], "R10 model numerical state differs")
    result.update(status="RUNNING", original_state_sha256=before, BN_scope=bn_info, frames=[], full_forward_count=0,
        anchor_replays={}, dual_checks={}, runtime={"torch": str(torch.__version__), "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device), "TF32": False, "AMP": False, "threads": 2})
    meters = {arm: metrics.MetricAccumulator() for arm in ARMS}
    times = {x["target_id"]: x["timestamps"] for x in oldbinding["new_member_data_preflight"]}
    def forward(chunk, arm, **kwargs):
        out = forward_arm(r9, r8, parent, model, functional, targets, shapes, chunk, arm, merge, torch,
                          deadline=deadline, **kwargs)
        result["full_forward_count"] += 1
        require(result["full_forward_count"] <= BUDGET, "Anchor forward cap exceeded")
        expected = 609600 if kwargs.get("mode") == "dual_reference" else ROWS[arm]
        require(out[1]["summed_fc1_rows"] == out[1]["summed_fc2_rows"] == expected, "Production all12 rows differ")
        return out
    with torch.inference_mode():
        for index, (_, row) in enumerate(chosen):
            deadline(); tid = row["target_id"]; expected = {}; evidence = {}
            for key, relative in native.target_files(row).items():
                native.checked_path(b.data_root, relative)
                expected["events" if key == "input" else key] = hashes[relative]["sha256"]
            arrays, _ = metrics.load_arrays(b.data_root/"saved_flow_data", row, evidence=evidence, expected_hashes=expected)
            require(evidence == refs[tid]["input_evidence"] and times[tid] == refs[tid]["timestamps"], "R10 input/timestamps differ")
            masks = metrics.partitions(arrays["gt"], arrays["mask"], arrays["events"], regions)
            chunk = base.prepare_input(arrays["events"], device, torch)
            if index == 0:
                flows, ex, _ = forward(chunk, "CC")
                digest = hashlib.sha256(flows[-1][0].numpy().tobytes()).hexdigest()
                old = refs[tid]["arms"]["coarse_stage02"]
                require(digest == old["prediction_sha256"], "R10 archived CC prediction not exact")
                require(set(ex["layers"]) == set(old["execution"]["layers"]) and
                        all(ex["layers"][n][k] == old["execution"]["layers"][n][k]
                            for n in ex["layers"] for k in ("input_sha256", "output_sha256")), "R10 archived CC layer hashes differ")
                result["anchor_replays"]["CC"] = {"target_id": tid, "prediction_sha256": digest,
                    "exact_archived_prediction": True, "all8_input_output_SHA_exact": True, "execution": ex}
            frame = {"target_id": tid, "sequence": row["sequence"], "group": "post_result_same_check_diagnostic",
                     "input_evidence": evidence, "timestamps": times[tid], "arms": {}}
            predictions = {}
            for arm in ARMS:
                actual = forward(chunk, arm, capture=index == 0 and arm != "native")
                flows, ex, branches = actual
                if arm == "fine_s2b2":
                    check_prefix(ex, refs[tid], r10.PREFIX_NAMES)
                    ex["prefix_input_SHA_exact_to_archived_CC"] = list(r10.PREFIX_NAMES)
                    ex["earlier_prefix_output_SHA_exact_to_archived_CC"] = list(r10.PREFIX_NAMES[:-1])
                prediction = flows[-1][0].numpy().copy(); predictions[arm] = prediction
                frame["arms"][arm] = {"metrics": meters[arm].update(tid, row["sequence"], prediction, arrays["gt"], masks),
                    "prediction_sha256": hashlib.sha256(prediction.tobytes()).hexdigest(), "execution": ex}
                if index == 0 and arm != "native":
                    dual = forward(chunk, arm, mode="dual_reference", capture=True)
                    result["dual_checks"][arm] = check_pair(actual, dual, torch)
                    del dual
            path = a.output_dir/(tid+".npz"); np.savez_compressed(path, **predictions)
            frame.update(prediction_file=path.name, prediction_file_sha256=sha(path)); result["frames"].append(frame)
            result["summaries"] = {arm: meter.summary() for arm, meter in meters.items()}
            parent.write(a.output_dir/"results.json", result)
            print(json.dumps({"target_id": tid, "completed_targets": len(result["frames"]),
                              "full_forwards": result["full_forward_count"]}), flush=True)
            del chunk, flows, actual, branches, predictions, prediction
    require(parent.source_digest(model) == before and base.bn_signature(model, torch) == bn_before, "Model/BN state changed")
    require(all(model.get_submodule(name) is module for name, module in targets.items()), "Original modules not restored")
    require(all(not module._forward_pre_hooks and not module._forward_hooks for module in targets.values()), "Audit hooks remain")
    require(result["full_forward_count"] == BUDGET and len(result["frames"]) == 8, "Fixed member/forward count differs")
    result.update(status="ROUND10_SAME_CHECK_NATIVE_ANCHORS_COMPLETE", completed_targets=8, original_model_unchanged=True,
        BN_buffers_unchanged=True, original_modules_restored=True, wall_seconds=time.monotonic()-started,
        hardware_benefit_claim=False, scope=binding["scope"])


def main(argv=None):
    r10, collector, probe, parent, r9, r8 = load_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-contract", "identity-bank", "prior-screen-results", "prior-scope-results",
                 "selection-manifest", "fit-membership", "prior-check-membership", "reference-round10"):
        parser.add_argument("--"+name, type=Path, required=True)
        parser.add_argument("--"+name+"-sha256", required=True)
    for name, default in (("module-checks", probe.BASE/"cpu_results.json"),
                          ("scope-runner-checks", probe.BASE/"scope_extension_v1/runner_cpu_results.json"),
                          ("probe-runner-checks", probe.HERE/"runner_cpu_results.json"),
                          ("collector-runner-checks", r10.source_path("collector_receipt")),
                          ("round10-runner-checks", R10/"collector_cpu_results.json"),
                          ("runner-checks", HERE/"runner_cpu_results.json")):
        parser.add_argument("--"+name, type=Path, default=default)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--wall-minutes", type=float, default=10)
    parser.add_argument("--execute", action="store_true")
    a = parser.parse_args(argv)
    prepared = prepare(a, r10, collector, probe, parent)
    result = {"schema": "flow_round10_same_check_native_anchors_v1", "status": "PREFLIGHT_PASS_NO_INFERENCE",
        "binding": prepared[3], "inference_executed": False, "training_performed": False}
    if not a.execute:
        require(a.preflight_output and not a.preflight_output.exists(), "Fresh preflight output required")
        parent.write(a.preflight_output, result); print(result["status"]); return
    require(parent.GPU_LOCK.parent.is_dir(), "Shared lock directory required")
    with parent.GPU_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a.output_dir.mkdir(parents=True, exist_ok=False)
        previous = signal.getsignal(signal.SIGALRM)
        def timeout(signum, frame):
            raise TimeoutError("Anchor fixed wall budget exhausted")
        signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, a.wall_minutes*60)
        try:
            result["inference_executed"] = True
            execute(a, prepared, result, r10, r9, r8, parent)
        except BaseException as exc:
            result.update(status="INCOMPLETE_NO_SCIENTIFIC_CONCLUSION", error=type(exc).__name__+": "+str(exc),
                          traceback=traceback.format_exc())
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)
            parent.write(a.output_dir/"results.json", result)


if __name__ == "__main__":
    main()
