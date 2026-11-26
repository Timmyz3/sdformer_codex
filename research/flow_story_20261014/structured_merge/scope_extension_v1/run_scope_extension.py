#!/usr/bin/env python3
"""Post-ROUND3 fixed dev8 scope extension; exploratory static control, no training.

Fork of run_structured_merge_screen.py SHA f23667f986ce6f83d6a51a7056ff25e82432d05def3bbd174530f17afc2551bf.
Default preflight; --execute needs shared GPU lock.

No training, learned gate, motion input, or inherited per-frame MLP mask/input hash.
The old mask bank supplies ONLY the fixed dev8 identity/order. Writes a new run.
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
BASE = HERE.parent
PARENT_RUNNER_SHA = "f23667f986ce6f83d6a51a7056ff25e82432d05def3bbd174530f17afc2551bf"
MODULE_SHA = "26454de4009e20d45edc186eaf255ef1df918b2250f270a239713efb90fe72f0"
ROUND3_SHA = "70208e3c0378bed2127d41e47745a373f1c02aad49b07da2a79af0ad9be1e9fb"
PARENT_SHA = "564d70a02b65d00c1aec40eacf069babb38f600c5a553a5ceb2a263630f3c389"
BANK_SHA = "2d831c7a3a77ffd4bc22efeb256a9d1b672f7ef8edf2300cac413c603d7fdfdf"
CHECKPOINT_SHA = "d5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30"
GPU_LOCK = Path("/data/flow_a/runs/.flow_a_gpu0.lock")
STAGE_SHAPES = {0: (10, 1, 120, 160, 96), 1: (10, 1, 60, 80, 192),
                2: (10, 1, 30, 40, 384), 3: (10, 1, 15, 20, 768)}
DEPTHS = {0: 2, 1: 2, 2: 6, 3: 2}
ARMS = (("native_dense", "native", ()),
        ("fine_stage02", "fine", (0, 2)), ("coarse_stage02", "coarse", (0, 2)),
        ("fine_stage23", "fine", (2, 3)), ("coarse_stage23", "coarse", (2, 3)),
        ("fine_stage023", "fine", (0, 2, 3)), ("coarse_stage023", "coarse", (0, 2, 3)))
MATCHED = {"coarse_stage02": "fine_stage02", "coarse_stage23": "fine_stage23",
           "coarse_stage023": "fine_stage023"}
FINE_ARMS = tuple(name for name, kind, stages in ARMS if kind == "fine")
FINE_PAIRS = (("fine_stage02", "fine_stage23"), ("fine_stage02", "fine_stage023"),
              ("fine_stage23", "fine_stage023"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    os.replace(temporary, path)


def source_digest(model):
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        result.update(name.encode()); result.update(str(value.dtype).encode()); result.update(str(tuple(value.shape)).encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def parse_parent_args(parent_args, output, extended):
    arguments = list(parent_args)
    require(arguments.count("--output") == 1, "Parent output missing/ambiguous")
    arguments[arguments.index("--output")+1] = str(output)
    return extended.parse_extended(arguments)


def fixed_dev_rows(bank, dev32, roles):
    require(bank.get("schema") == "flow_fast_recovery_mask_bank_v1" and bank.get("status") == "MASK_BANK_COMPLETE",
            "Completed fixed mask bank identity source required")
    require(bank["binding"]["checkpoint_sha256"] == CHECKPOINT_SHA, "Identity bank checkpoint differs")
    ids = bank["dev_ids"]
    require(len(ids) == len(set(ids)) == 8 and digest(ids) == bank["binding"]["dev_ids_sha256"], "Fixed dev8 order differs")
    by_id = {row["target_id"]: row for row in dev32}
    require(set(ids) <= set(by_id), "Fixed dev8 is outside pinned dev32")
    rows = [by_id[target] for target in ids]
    require(all(row["protocol_split"] == "train" and roles.get(row["target_id"]) == "gate_development" for row in rows),
            "Identity source is outside gate development")
    return rows


def checked_cpu_receipt(path, expected_sources):
    value = json.loads(Path(path).read_text())
    require(value.get("passed") is True and value.get("GPU_initialized") is False, "CPU checks failed or used GPU")
    pins = value.get("source_sha256", {})
    for source in expected_sources:
        matches = [pin for name, pin in pins.items() if Path(name).name == source.name]
        require(matches == [sha(source)], "CPU checks do not bind current source: " + source.name)
    return sha(path)


def load_frozen_merge():
    path = BASE/"structured_merge_mlp.py"
    require(sha(path) == MODULE_SHA, "Frozen merge module differs")
    spec = importlib.util.spec_from_file_location("_scope_extension_frozen_merge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_round3(receipt, dev_ids):
    require(receipt.get("schema") == "flow_structured_merge_screen_v1" and
            receipt.get("status") == "STRUCTURED_MERGE_DEV8_COMPLETE" and
            receipt.get("completed_targets") == 8 and receipt.get("full_forward_count") == 67,
            "Completed fixed ROUND3 receipt required")
    binding = receipt["binding"]
    require(binding["module_sha256"] == MODULE_SHA and binding["runner_sha256"] == PARENT_RUNNER_SHA and
            binding["checkpoint_sha256"] == CHECKPOINT_SHA and binding["identity_bank_sha256"] == BANK_SHA,
            "ROUND3 source/numerical graph differs")
    require(binding["dev_ids"] == dev_ids, "ROUND3 dev8 identity/order differs")
    require(receipt.get("original_model_unchanged") is True and receipt.get("BN_buffers_unchanged") is True,
            "ROUND3 unchanged-native checks missing")


def prepare(a):
    require(a.parent_contract_sha256 == PARENT_SHA == sha(a.parent_contract), "Pinned parent contract differs")
    require(a.identity_bank_sha256 == BANK_SHA == sha(a.identity_bank), "Pinned dev8 identity bank differs")
    require(not a.output_dir.exists(), "New output directory required")
    require(0 < a.wall_minutes <= 30, "Wall budget must be within 30 minutes")
    require(sha(BASE/"run_structured_merge_screen.py") == PARENT_RUNNER_SHA, "Frozen parent runner differs")
    require(sha(BASE/"structured_merge_mlp.py") == MODULE_SHA, "Frozen merge module differs")
    require(a.prior_screen_results_sha256 == ROUND3_SHA == sha(a.prior_screen_results), "Fixed ROUND3 result differs")
    module_check = checked_cpu_receipt(a.module_checks, [BASE/"structured_merge_mlp.py", BASE/"test_structured_merge_cpu.py"])
    runner_check = checked_cpu_receipt(a.runner_checks, [Path(__file__), HERE/"test_scope_extension_cpu.py", BASE/"structured_merge_mlp.py"])
    c = json.loads(a.parent_contract.read_text())
    require(c["schema"] == "flow_fast_runtime_contract_v1" and c["target_count"] == 4, "Parent contract scope differs")
    for path, pin in c["source_pins"].items():
        require(sha(path) == pin, "Pinned source differs: " + path)
    visible_before = os.environ.get("CUDA_VISIBLE_DEVICES")
    sys.path[:0] = [c["breadth_root"], c["motion_root"]]
    import evaluate_breadth as base
    import evaluate_breadth_extended as extended
    import run_bn_scope as scope
    import run_same_shape_reference as loader
    b = parse_parent_args(c["base_args"], a.output_dir/"results.json", extended)
    require(b.geometry == "full9" and b.checkpoint_sha256 == CHECKPOINT_SHA and not b.run_inference,
            "Fixed full9 interim checkpoint required")
    b.device = "cuda:0"; b.wall_minutes = a.wall_minutes
    context = extended.extended_preflight(b)
    native, trainer, metrics, dev32, hashes, config, saved, side, regions, plan = context
    require(side["completed_optimizer_steps"] == 135930, "Interim checkpoint step differs")
    roles = metrics.auxiliary_roles(b.roles, metrics.canonical_rows(b.canonical))
    bank = json.loads(a.identity_bank.read_text())
    dev = fixed_dev_rows(bank, dev32, roles)
    validate_round3(json.loads(a.prior_screen_results.read_text()), [row["target_id"] for row in dev])
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == visible_before, "Loader changed CUDA visibility")
    import torch
    require(not torch.cuda.is_initialized(), "Preflight initialized CUDA before lock")
    binding = {"parent_contract_sha256": PARENT_SHA, "identity_bank_sha256": BANK_SHA,
               "identity_bank_usage": "dev_ids only; prior masks/u hashes/head/fast weights not consumed",
               "checkpoint_sha256": CHECKPOINT_SHA, "BN_scope": "MLP_CMA176",
               "module_sha256": MODULE_SHA, "runner_sha256": sha(__file__),
               "forked_parent_runner_sha256": PARENT_RUNNER_SHA, "prior_round3_results_sha256": ROUND3_SHA,
               "selection_scope": "Post-ROUND3 exploratory static scope extension; same dev8, not independent confirmation or A dynamic contribution",
               "module_checks_sha256": module_check, "runner_checks_sha256": runner_check,
               "dev_ids": [row["target_id"] for row in dev], "dev_ids_sha256": digest([row["target_id"] for row in dev]),
               "arms": ARMS, "matched_fine_baselines": MATCHED, "merge_stride": 3,
               "GPU_lock": str(GPU_LOCK), "wall_minutes": a.wall_minutes}
    return c, b, context, base, scope, loader, dev, binding


def target_layers(model):
    targets = {name: module for name, module in model.named_modules() if type(module).__name__ == "MS_Spiking_Mlp"}
    expected = {f"sttmultires_unet.encoders.swin3d.layers.{stage}.swin_blocks.{block}.mlp": STAGE_SHAPES[stage]
                for stage, depth in DEPTHS.items() for block in range(depth)}
    require(set(targets) == set(expected), "Declared twelve native MLP targets differ")
    return targets, expected


@contextlib.contextmanager
def wrapped_model(model, targets, shapes, kind, stages, mode, merge, torch):
    wrappers = {}
    with contextlib.ExitStack() as stack:
        for name, original in targets.items():
            if not any(f".layers.{stage}." in name for stage in stages):
                continue
            shape = shapes[name]
            mask = torch.full((math.ceil(shape[2]/9), math.ceil(shape[3]/9)), kind == "fine", dtype=torch.bool)
            require(kind in ("fine", "coarse"), "Only declared fine/coarse scope arms are supported")
            wrapper = stack.enter_context(merge.replace_mlp(model, name, mask, 3, mode, expected_shape=shape))
            wrappers[name] = wrapper
        yield wrappers


def forward_probe(model, functional, targets, shapes, chunk, kind, stages, merge, torch,
                  mode="actual", capture=False, deadline=lambda: None):
    deadline(); functional.reset_net(model)
    observed = {name: {"fc1_rows": 0, "fc2_rows": 0} for name in targets}
    branch_values = {}
    handles = []
    begin = time.monotonic()
    try:
        for name, module in targets.items():
            for fc in ("fc1", "fc2"):
                def count(module, inputs, layer=name, key=fc+"_rows"):
                    observed[layer][key] += inputs[0].numel()//inputs[0].shape[-1]
                handles.append(getattr(module, fc).register_forward_pre_hook(count))
        with wrapped_model(model, targets, shapes, kind, stages, mode, merge, torch) as wrappers:
            if capture:
                for name, wrapper in wrappers.items():
                    def store(module, inputs, output, layer=name):
                        branch_values[layer] = output.detach().cpu().clone()
                    handles.append(wrapper.register_forward_hook(store))
            with torch.inference_mode():
                output = model(chunk)
            require(isinstance(output["flow"], (list, tuple)) and bool(output["flow"]), "Native flow list required")
            require(all(bool(torch.isfinite(value).all()) for value in output["flow"]), "Nonfinite flow")
            flows = [value.detach().cpu().clone() for value in output["flow"]]
            records = {}
            for name, wrapper in wrappers.items():
                require(len(wrapper.records) == 1, "Missing/duplicate wrapper execution: " + name)
                records[name] = wrapper.records[0]
        for name, counts in observed.items():
            expected = records[name]["executed_fc_rows_per_layer"] if name in records else math.prod(shapes[name][:-1])
            require(counts["fc1_rows"] == counts["fc2_rows"] == expected, "Independent FC row audit differs: " + name)
        return flows, {"layers": records, "all12_observed_fc_rows": observed,
                       "summed_fc1_rows": sum(v["fc1_rows"] for v in observed.values()),
                       "summed_fc2_rows": sum(v["fc2_rows"] for v in observed.values()),
                       "wall_seconds_diagnostic_only": time.monotonic()-begin}, branch_values
    finally:
        for handle in handles:
            handle.remove()


def execute(a, prepared, result):
    import numpy as np
    import torch
    merge = load_frozen_merge()
    c, b, context, base, scope, loader, dev, binding = prepared
    require(not torch.cuda.is_initialized(), "CUDA initialized before lock")
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(20261014)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0"); torch.cuda.set_device(device); torch.cuda.manual_seed_all(20261014)
    started = time.monotonic()
    def deadline():
        require(time.monotonic()-started < a.wall_minutes*60, "30-minute maximum phase budget exhausted")
    native, trainer, metrics, dev32, hashes, config, saved, side, regions, plan = context
    model, functional = loader.load_model(b, context, base, torch, device)
    bn_info = scope.apply_scope(model, "MLP_CMA176", b.bn_payload, torch)
    model.eval().requires_grad_(False)
    targets, shapes = target_layers(model)
    before, bn_before = source_digest(model), base.bn_signature(model, torch)
    result.update(status="RUNNING", runtime={"torch": str(torch.__version__), "cuda": torch.version.cuda,
                  "cudnn": torch.backends.cudnn.version(), "device_name": torch.cuda.get_device_name(device),
                  "device": str(device), "TF32": False, "AMP": False, "threads": 2},
                  original_state_sha256=before, BN_scope=bn_info, frames=[], first_frame_pairs={})
    meters = {arm: metrics.MetricAccumulator() for arm, _, _ in ARMS}

    def forward(chunk, kind, stages, mode="actual", capture=False):
        return forward_probe(model, functional, targets, shapes, chunk, kind, stages, merge, torch,
                             mode=mode, capture=capture, deadline=deadline)

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
            predictions = {}; flow_cache = {}
            for arm, kind, stages in ARMS:
                flows, execution, branches = forward(chunk, kind, stages, capture=index == 0 and kind == "coarse")
                flow_cache[arm] = flows
                pred = flows[-1][0].numpy().copy(); predictions[arm] = pred
                frame["arms"][arm] = {"metrics": meters[arm].update(row["target_id"], row["sequence"], pred, arrays["gt"], masks),
                                      "execution": execution, "prediction_sha256": hashlib.sha256(pred.tobytes()).hexdigest()}
                if index == 0 and kind == "coarse":
                    dual, dual_execution, dual_branches = forward(chunk, kind, stages, "dual_reference", capture=True)
                    per_layer = {name: {"input_sha_equal": execution["layers"][name]["input_sha256"] == dual_execution["layers"][name]["input_sha256"],
                                         "branch_comparison": loader.compare(branches[name], dual_branches[name], torch)} for name in branches}
                    require(len(flows) == len(dual), "Actual/dual flow count differs")
                    pair = {"flow_comparisons": [loader.compare(x, y, torch) for x, y in zip(flows, dual)],
                            "layers": per_layer, "actual_execution": execution, "dual_execution": dual_execution}
                    result["first_frame_pairs"][arm] = pair
                    write(a.output_dir/"results.json", result)
                    require(all(v["numerically_equal"] for v in pair["flow_comparisons"]) and
                            all(v["input_sha_equal"] and v["branch_comparison"]["numerically_equal"] for v in per_layer.values()),
                            "Actual/dual selected producers differ; no quality conclusion")
                    del dual, dual_branches
                del branches
            frame["native_to_fine_arithmetic_drift"] = {arm: [loader.compare(x, y, torch) for x, y in zip(flow_cache["native_dense"], flow_cache[arm])]
                                                         for arm in FINE_ARMS}
            frame["matched_scope_fine_pairwise_drift"] = {left+"_to_"+right: [loader.compare(x, y, torch) for x, y in zip(flow_cache[left], flow_cache[right])]
                                                         for left, right in FINE_PAIRS}
            frame["matched_fine_flow_differences"] = {arm: [loader.compare(x, y, torch) for x, y in zip(flow_cache[baseline], flow_cache[arm])]
                                                       for arm, baseline in MATCHED.items()}
            prediction_path = a.output_dir/("dev_"+row["target_id"]+".npz")
            np.savez_compressed(prediction_path, **predictions)
            frame["prediction_file"] = prediction_path.name; frame["prediction_file_sha256"] = sha(prediction_path)
            result["frames"].append(frame)
            result["summaries"] = {arm: meter.summary() for arm, meter in meters.items()}
            write(a.output_dir/"results.json", result)
            print(json.dumps({"target_id": row["target_id"], "completed_targets": index+1, "arms": len(ARMS)}), flush=True)
            del chunk, flows, flow_cache, predictions
    require(source_digest(model) == before and base.bn_signature(model, torch) == bn_before, "Native weights/BN changed")
    require(all(model.get_submodule(name) is original for name, original in targets.items()), "Native MLP restoration failed")
    result.update(status="STRUCTURED_MERGE_SCOPE_EXTENSION_DEV8_COMPLETE", completed_targets=8, full_forward_count=8*len(ARMS)+3,
                  original_model_unchanged=True, BN_buffers_unchanged=True, original_modules_restored=True,
                  wall_seconds=time.monotonic()-started, hardware_benefit_claim=False,
                  scope="Post-ROUND3 exploratory static scope extension on same backbone-seen training-side dev8; no independent confirmation, recovery, gate, A dynamic contribution or hardware claim")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-contract", type=Path, required=True)
    parser.add_argument("--parent-contract-sha256", required=True)
    parser.add_argument("--identity-bank", type=Path, required=True)
    parser.add_argument("--identity-bank-sha256", required=True)
    parser.add_argument("--prior-screen-results", type=Path, required=True)
    parser.add_argument("--prior-screen-results-sha256", required=True)
    parser.add_argument("--module-checks", type=Path, default=BASE/"cpu_results.json")
    parser.add_argument("--runner-checks", type=Path, default=HERE/"runner_cpu_results.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--wall-minutes", type=float, default=30)
    parser.add_argument("--execute", action="store_true")
    a = parser.parse_args(argv)
    prepared = prepare(a)
    result = {"schema": "flow_structured_merge_scope_extension_v1", "status": "PREFLIGHT_PASS_NO_INFERENCE",
              "binding": prepared[-1], "native_preflight": prepared[2][-1], "inference_executed": False,
              "training_performed": False, "same_budget_claim": False}
    if not a.execute:
        require(a.preflight_output and not a.preflight_output.exists(), "New preflight output required")
        write(a.preflight_output, result); print(result["status"]); return
    require(GPU_LOCK.parent.is_dir(), "Existing shared GPU lock directory required")
    with GPU_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a.output_dir.mkdir(parents=True, exist_ok=False)
        previous = signal.getsignal(signal.SIGALRM)
        def timeout(signum, frame):
            raise TimeoutError("Screen reached its fixed wall budget")
        signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, a.wall_minutes*60)
        try:
            result["inference_executed"] = True
            execute(a, prepared, result)
        except BaseException as exc:
            result.update(status="INCOMPLETE_NO_SCIENTIFIC_CONCLUSION", error=type(exc).__name__+": "+str(exc), traceback=traceback.format_exc())
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
            write(a.output_dir/"results.json", result)


if __name__ == "__main__":
    main()
