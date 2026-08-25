"""M51-r2 GPU producer with synchronized, bounded input tracing.

This entry point captures inputs only.  It emits no operator output payload and
does not calculate cycles, speedup, accuracy, PPA, power, or energy.
"""

from __future__ import print_function

import argparse
import hashlib
import os
from pathlib import Path
import sys

import numpy as np
import torch

import profile_nts11_hardware_p0 as profile
from h67_binary_input_trace_r2 import (
    EXPECTED_TARGET_PLAN_SHA256, ExactBinaryInputTraceWriterR2,
    require_c_order_contiguous, sha256_path, strict_json,
)


ROOT = Path(__file__).resolve().parents[3]
RUNNER_R2_CONTRACT = ROOT / (
    "hw_autoresearch_nts07/contracts/"
    "m51_h67_ep35_binary_input_trace_runner_r2_contract_r1_20260823.json")
EXPECTED_RUNNER_R2_CONTRACT_SHA256 = (
    "571b54e28e6778c4920baa16b44ebc7b76dc00cb158bcc676921539cfd302f5e")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parameter_payload(tensor):
    if tensor is None:
        return None
    value = tensor.detach().to(device="cpu").contiguous()
    array = value.numpy()
    payload = array.tobytes(order="C")
    return {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "content_bytes": len(payload),
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "byte_order": sys.byteorder,
        "layout": "C_ORDER_CONTIGUOUS",
    }


def module_identities(model, plan):
    named = dict(model.named_modules())
    identities = {}
    for target in plan["modules"]:
        name = target["name"]
        require(name in named, "target module missing: {}".format(name))
        module = named[name]
        operator = module.__class__.__name__
        require(operator == target["operator"] and
                isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)),
                "target module type mismatch: {}".format(name))
        identity = {
            "operator": operator,
            "weight": parameter_payload(module.weight),
            "bias": parameter_payload(module.bias),
        }
        require(int(module.weight.numel()) == target["expected_weight_elements"],
                "target weight element mismatch: {}".format(name))
        identities[name] = identity
    return identities


def stream_torch_binary_r2(tensor, chunk_elements, handle, digest):
    """Reject non-C layouts, then validate/pack bounded flat slices."""
    require(torch.is_tensor(tensor), "hook input is not a tensor")
    require_c_order_contiguous(tensor)
    require(chunk_elements > 0 and chunk_elements % 8 == 0,
            "chunk-elements must be a positive multiple of eight")
    # `view` is metadata-only after the strict default-contiguous gate above.
    # No whole-call activation copy or materialization is permitted here.
    flat = tensor.detach().view(-1)
    elements = int(flat.numel())
    active = packed_bytes = 0
    for start in range(0, elements, chunk_elements):
        chunk = flat[start:min(elements, start + chunk_elements)]
        exact = torch.logical_or(chunk == 0, chunk == 1)
        if not bool(torch.all(exact).item()):
            invalid = int(torch.count_nonzero(torch.logical_not(exact)).item())
            raise ValueError(
                "non-binary raw hook input rejected: {} invalid values in chunk "
                "starting at element {}".format(invalid, start))
        byte_values = chunk.to(device="cpu", dtype=torch.uint8).numpy()
        active += int(byte_values.sum(dtype=np.uint64))
        payload = np.packbits(byte_values, bitorder="little").tobytes(order="C")
        require(len(payload) == (int(chunk.numel()) + 7) // 8,
                "packed chunk length mismatch")
        handle.write(payload)
        digest.update(payload)
        packed_bytes += len(payload)
    return {
        "elements": elements,
        "active": active,
        "packed_bytes": packed_bytes,
        "tail_used_bits": elements % 8 or 8,
    }


def validate_frozen_protocol(plan, config, bn_modules_changed, module_counts,
                             num_workers):
    frozen = plan["frozen_run_identity"]
    protocol = frozen["eval_protocol"]
    observed = {
        "resolution": list(config["loader"]["resolution"]),
        "crop": config["loader"].get("crop"),
        "window_size": list(config["swin_transformer"]["window_size"]),
        "pretrained_window_size": config["swin_transformer"].get(
            "pretrained_window_size"),
        "tokens_per_window": int(np.prod(config["swin_transformer"]["window_size"])),
        "remap": config["loader"].get("remap"),
        "bn_policy": config.get("test", {}).get("bn_policy", "running"),
        "bn_modules_changed": bn_modules_changed,
        "eval_batch_size": 1,
        "num_workers": num_workers,
    }
    require(observed == protocol, "frozen evaluation protocol mismatch")
    require(module_counts == frozen["module_counts"],
            "frozen H67 module-count mismatch")


def sample_identity(dataset, sample_id):
    file_row = dataset.files[sample_id]
    file_names = (list(file_row) if isinstance(file_row, (list, tuple))
                  else [str(file_row)])
    sample_key = "|".join(str(item) for item in file_names)
    sequence_names = ["_".join(Path(str(item)).stem.split("_")[:-1])
                      for item in file_names]
    return sample_key, "|".join(sequence_names)


def cuda_memory_snapshot(device, phase):
    return {
        "phase": phase,
        "cuda_available": bool(torch.cuda.is_available()),
        "capture_device_type": torch.device(device).type,
        "memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(
            torch.cuda.max_memory_reserved(device)),
    }


def failure_memory_snapshot(device):
    try:
        return cuda_memory_snapshot(device, "FAILURE_BEFORE_PASS_MANIFEST")
    except BaseException as error:  # best-effort evidence must not mask cause
        return {
            "phase": "FAILURE_MEMORY_SNAPSHOT_UNAVAILABLE",
            "error": "{}: {}".format(type(error).__name__, error),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chunk-elements", type=int, default=8 * 1024 * 1024)
    args = parser.parse_args()
    require(sys.byteorder == "little", "M51-r2 requires a little-endian producer")
    require(sha256_path(RUNNER_R2_CONTRACT) ==
            EXPECTED_RUNNER_R2_CONTRACT_SHA256,
            "M51-r2 runner contract SHA mismatch")
    require(args.samples == 10 and args.num_workers == 0,
            "M51-r2 requires exactly ten samples and num-workers=0")
    require(args.chunk_elements > 0 and args.chunk_elements % 8 == 0,
            "chunk-elements must be a positive multiple of eight")
    require(not args.output_dir.resolve().exists(),
            "refusing existing M51-r2 output directory")
    plan_path = args.target_plan.resolve()
    require(sha256_path(plan_path) == EXPECTED_TARGET_PLAN_SHA256,
            "M51-r2 target plan SHA mismatch")
    plan = strict_json(plan_path)
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    require(config_path.is_file() and checkpoint_path.is_file(),
            "missing M51-r2 config/checkpoint")
    require(sha256_path(config_path) ==
            plan["frozen_run_identity"]["config_sha256"],
            "M51-r2 config SHA mismatch")
    require(checkpoint_path.name == plan["checkpoint_identity"]["basename"] and
            checkpoint_path.stat().st_size ==
            plan["checkpoint_identity"]["expected_size_bytes"] and
            sha256_path(checkpoint_path) ==
            plan["checkpoint_identity"]["expected_sha256"],
            "M51-r2 checkpoint identity mismatch")

    config, device = profile.load_config(config_path)
    capture_device = torch.device(device)
    require(torch.cuda.is_available() and capture_device.type == "cuda",
            "M51-r2 capture requires an available CUDA device")
    dataset = profile.DSECDatasetLite(
        config, file_list="valid", stereo=False,
        scale_factor=config.get("test", {}).get("scale_factor", 1))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False,
        pin_memory=False, num_workers=args.num_workers)
    transform_valid = None
    if config["loader"].get("crop") is not None:
        transform_valid = profile.Compose([profile.CenterCrop((
            config["loader"]["crop"][0], config["loader"]["crop"][1]))])
    model = profile.build_model(config, checkpoint_path, device)
    checkpoint_load_audit = profile.validate_h9_load_audit(model, config)
    require(checkpoint_load_audit is not None and
            int(checkpoint_load_audit.get("missing_count", 0)) == 0 and
            int(checkpoint_load_audit.get("unexpected_count", 0)) == 0,
            "M51-r2 checkpoint load audit is not exact")
    counts = profile.h9_module_counts(model)
    bn_policy = config.get("test", {}).get("bn_policy", "running")
    bn_changed = profile.configure_batch_norm_evaluation(model, bn_policy)
    validate_frozen_protocol(plan, config, bn_changed, counts, args.num_workers)

    identities = module_identities(model, plan)
    writer = ExactBinaryInputTraceWriterR2(
        plan_path, args.output_dir.resolve(), EXPECTED_TARGET_PLAN_SHA256)
    handles = []
    completed = False
    synchronization_counts = {
        "before_capture": 0,
        "per_sample_post_forward": 0,
        "final_pre_manifest": 0,
    }
    try:
        # Drain model/checkpoint setup, then make capture-only peaks comparable.
        torch.cuda.synchronize(device)
        synchronization_counts["before_capture"] += 1
        torch.cuda.reset_peak_memory_stats(device)
        memory_before = cuda_memory_snapshot(device, "BEFORE_CAPTURE")

        writer.bind_module_identities(identities)
        entrypoint_dir = Path(__file__).resolve().parent
        writer.bind_run_context({
            "runner_revision": "M51_RUNNER_R2_P1_REPAIR",
            "runner_contract_path": str(RUNNER_R2_CONTRACT),
            "runner_contract_sha256": sha256_path(RUNNER_R2_CONTRACT),
            "config_path": str(config_path),
            "config_sha256": sha256_path(config_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "checkpoint_sha256": sha256_path(checkpoint_path),
            "checkpoint_load_audit": checkpoint_load_audit,
            "device": str(device),
            "chunk_elements": args.chunk_elements,
            "input_layout_policy": (
                "REQUIRE_DEFAULT_C_CONTIGUOUS_NO_WHOLE_CALL_COPY"),
            "cuda_synchronization": synchronization_counts,
            "source_sha256": {
                "runner_r2": sha256_path(Path(__file__).resolve()),
                "writer_r2": sha256_path(
                    entrypoint_dir / "h67_binary_input_trace_r2.py"),
                "writer_r1_base": sha256_path(
                    entrypoint_dir / "h67_binary_input_trace.py"),
                "base_profiler": sha256_path(Path(profile.__file__).resolve()),
            },
            "claim_boundary": (
                "raw exact-binary inputs only; no output or performance claim"),
        })
        named = dict(model.named_modules())

        def make_hook(name, operator):
            def hook(_module, inputs, output):
                require(isinstance(inputs, tuple) and len(inputs) == 1 and
                        torch.is_tensor(inputs[0]),
                        "target hook must observe exactly one input tensor")
                require(torch.is_tensor(output),
                        "target hook output must be one tensor for shape identity")

                def payload_writer(handle, digest):
                    return stream_torch_binary_r2(
                        inputs[0], args.chunk_elements, handle, digest)

                writer.capture(
                    name=name, operator=operator,
                    input_shape=[int(item) for item in inputs[0].shape],
                    output_shape=[int(item) for item in output.shape],
                    payload_writer=payload_writer)
            return hook

        for target in plan["modules"]:
            handles.append(named[target["name"]].register_forward_hook(
                make_hook(target["name"], target["operator"])))

        processed = 0
        with torch.no_grad():
            for chunk, mask, label in loader:
                if processed >= args.samples:
                    break
                profile.functional.reset_net(model)
                sample_key, sequence_key = sample_identity(dataset, processed)
                writer.begin_sample(processed, sample_key, sequence_key)
                x, _transformed_label, _transformed_mask = profile.preprocess_chunk(
                    config, chunk, label, mask, transform_valid, device)
                model(x)
                # Mandatory completion fence: deferred errors cannot cross the
                # sample boundary or escape into a later PASS manifest.
                torch.cuda.synchronize(device)
                synchronization_counts["per_sample_post_forward"] += 1
                writer.end_sample()
                processed += 1
                print("[M51-r2] captured sample {}/10".format(processed),
                      flush=True)
        require(processed == 10, "M51-r2 dataset ended before ten samples")
        # A distinct final fence is required even after the tenth sample fence.
        torch.cuda.synchronize(device)
        synchronization_counts["final_pre_manifest"] += 1
        memory_after = cuda_memory_snapshot(device, "AFTER_FINAL_SYNCHRONIZE")
        writer.record_capture_memory(memory_before, memory_after)
        manifest = writer.close()
        completed = True
        print("PASS M51-r2 manifest {} {}".format(
            manifest, sha256_path(manifest)), flush=True)
    except BaseException as error:
        if not writer.closed and not writer.aborted:
            writer.abort(
                "{}: {}".format(type(error).__name__, error),
                failure_memory=failure_memory_snapshot(device))
        raise
    finally:
        for handle in handles:
            handle.remove()
    return 0 if completed else 1


if __name__ == "__main__":
    os.environ.setdefault("SDFORMER_USE_MLFLOW", "0")
    raise SystemExit(main())
