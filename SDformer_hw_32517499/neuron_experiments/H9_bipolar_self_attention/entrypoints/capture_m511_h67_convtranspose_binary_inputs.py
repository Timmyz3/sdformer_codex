"""Capture the four exact-binary H67 ConvTranspose2d input planes.

This GPU producer emits input bitpacks and immutable module identities only.
It does not emit outputs, cycles, speedup, accuracy, energy, PPA, or RTL claims.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid

import numpy as np
import torch
from spikingjelly.activation_based import functional

import profile_nts11_hardware_p0 as profile


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
REQUIRED_INPUT_NAMES = {
    "capture_script", "profile", "config", "checkpoint", "snn_models",
    "spiking_stswinnet", "spiking_modules", "bsa_attention", "atlif_init",
    "atlif_impl", "utils", "m51_manifest", "m510_result",
    "m510_result_manifest", "m510_result_seal",
    "m510_result_review_manifest", "m510_result_review_seal",
    "m512_review", "m512_review_manifest", "m512_review_seal", "docs359",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(token):
        raise RuntimeError("non-standard JSON token: " + token)

    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


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


def sample_identity(dataset, sample_id):
    row = dataset.files[sample_id]
    names = list(row) if isinstance(row, (list, tuple)) else [str(row)]
    sample_key = "|".join(str(item) for item in names)
    sequences = ["_".join(Path(str(item)).stem.split("_")[:-1])
                 for item in names]
    return sample_key, "|".join(sequences)


def sample_source_identity(dataset, sample_id):
    require(int(dataset.num_chunks) == 1,
            "M511 source identity requires one input chunk")
    row = dataset.files[sample_id]
    names = list(row) if isinstance(row, (list, tuple)) else [str(row)]
    require(len(names) == 1, "M511 source row population drift")
    name = str(names[0])
    sequence = "_".join(Path(name).stem.split("_")[:-1])
    paths = {
        "event": Path(dataset.events_path) / sequence / name,
        "mask": Path(dataset.mask_path) / name,
        "flow": Path(dataset.flow_path) / name,
    }
    identity = {"sample_id": sample_id, "sample_key": name, "files": {}}
    for kind, path in paths.items():
        resolved = path.resolve()
        require(resolved.is_file(),
                "M511 missing raw sample source {} {}".format(kind, resolved))
        identity["files"][kind] = {
            "path": str(resolved),
            "bytes": resolved.stat().st_size,
            "sha256": sha256(resolved),
        }
    return identity


def rehash_sample_sources(sequence_file, sequence_identity, sample_sources):
    require(sequence_file.is_file() and
            sequence_file.stat().st_size == sequence_identity["bytes"] and
            sha256(sequence_file) == sequence_identity["sha256"],
            "M511 validation sequence-list identity drift")
    for sample in sample_sources:
        for kind, entry in sample["files"].items():
            path = Path(entry["path"])
            require(path.is_file() and path.stat().st_size == entry["bytes"] and
                    sha256(path) == entry["sha256"],
                    "M511 raw sample source drift: {} {}".format(
                        sample["sample_id"], kind))


def cuda_snapshot(device, phase):
    return {
        "phase": phase,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_type": torch.device(device).type,
        "memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(
            torch.cuda.max_memory_reserved(device)),
    }


def stream_binary_input(tensor, chunk_elements, path):
    require(torch.is_tensor(tensor) and tensor.is_contiguous(),
            "M511 rejects non-tensor/non-contiguous hook input")
    require(chunk_elements > 0 and chunk_elements % 8 == 0,
            "M511 chunk size must be a positive multiple of eight")
    flat = tensor.detach().view(-1)
    elements = int(flat.numel())
    digest = hashlib.sha256()
    active = packed_bytes = 0
    partial = path.with_name(path.name + ".partial")
    require(not path.exists() and not partial.exists(),
            "M511 refuses an existing call payload")
    try:
        with partial.open("xb") as handle:
            for start in range(0, elements, chunk_elements):
                chunk = flat[start:min(elements, start + chunk_elements)]
                exact = torch.logical_or(chunk == 0, chunk == 1)
                require(bool(torch.all(exact).item()),
                        "M511 raw ConvTranspose2d input is not exact binary")
                values = chunk.to(device="cpu", dtype=torch.uint8).numpy()
                active += int(values.sum(dtype=np.uint64))
                payload = np.packbits(values, bitorder="little").tobytes(
                    order="C")
                require(len(payload) == (int(chunk.numel()) + 7) // 8,
                        "M511 packed chunk byte mismatch")
                handle.write(payload)
                digest.update(payload)
                packed_bytes += len(payload)
        require(packed_bytes == (elements + 7) // 8 and
                partial.stat().st_size == packed_bytes,
                "M511 final packed byte mismatch")
        os.link(str(partial), str(path))
        partial.unlink()
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    return {
        "elements": elements,
        "active": active,
        "packed_bytes": packed_bytes,
        "tail_used_bits": elements % 8 or 8,
        "file_sha256": digest.hexdigest(),
    }


def tuple2(value):
    if isinstance(value, tuple):
        return [int(item) for item in value]
    return [int(value), int(value)]


def module_identities(model, modules):
    named = dict(model.named_modules())
    result = {}
    for expected in modules:
        name = expected["name"]
        require(name in named, "M511 target module missing: " + name)
        module = named[name]
        require(isinstance(module, torch.nn.ConvTranspose2d) and
                module.__class__.__name__ == "ConvTranspose2d",
                "M511 target is not ConvTranspose2d: " + name)
        observed = {
            "operator": module.__class__.__name__,
            "in_channels": int(module.in_channels),
            "out_channels": int(module.out_channels),
            "kernel_size": tuple2(module.kernel_size),
            "stride": tuple2(module.stride),
            "padding": tuple2(module.padding),
            "output_padding": tuple2(module.output_padding),
            "dilation": tuple2(module.dilation),
            "groups": int(module.groups),
            "weight": parameter_payload(module.weight),
            "bias": parameter_payload(module.bias),
        }
        for key in ("operator", "in_channels", "out_channels", "kernel_size",
                    "stride", "padding", "output_padding", "dilation",
                    "groups"):
            require(observed[key] == expected[key],
                    "M511 module property mismatch {} {}".format(name, key))
        require(observed["weight"]["shape"] == expected["weight_shape"] and
                observed["bias"] is None,
                "M511 module parameter shape/bias mismatch: " + name)
        result[name] = observed
    return result


def verify_inputs(contract, script_sha):
    require(set(contract["inputs"]) == REQUIRED_INPUT_NAMES,
            "M511 contract input population drift")
    require(contract["inputs"]["capture_script"]["sha256"] == script_sha,
            "M511 contract does not pin the running producer")
    observed = {}
    for name, entry in contract["inputs"].items():
        path = (ROOT / entry["path"]).resolve()
        require(path.is_file() and sha256(path) == entry["sha256"],
                "M511 frozen input drift: " + name)
        observed[name] = {"path": str(path), "sha256": entry["sha256"]}
    require((ROOT / contract["inputs"]["capture_script"]["path"]).resolve() ==
            Path(__file__).resolve(), "M511 producer path drift")
    return observed


def write_seal(directory):
    members = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name not in (
                "SHA256SUMS", "SHA256SUMS.seal.sha256"):
            members.append(path.relative_to(directory))
    seal = directory / "SHA256SUMS"
    seal.write_text("".join(
        "{}  {}\n".format(sha256(directory / member), member.as_posix())
        for member in members), encoding="utf-8")
    outer = directory / "SHA256SUMS.seal.sha256"
    outer.write_text("{}  SHA256SUMS\n".format(sha256(seal)),
                     encoding="utf-8")


def verify_seal(directory):
    sealed_names = set()
    for line in (directory / "SHA256SUMS").read_text(
            encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        require(name not in sealed_names and name not in (
            "SHA256SUMS", "SHA256SUMS.seal.sha256"),
            "M511 unsafe/duplicate sealed member: " + name)
        sealed_names.add(name)
        require(sha256(directory / name) == expected,
                "M511 sealed member mismatch: " + name)
    expected, name = (directory / "SHA256SUMS.seal.sha256").read_text(
        encoding="utf-8").strip().split("  ", 1)
    require(name == "SHA256SUMS" and sha256(directory / name) == expected,
            "M511 outer seal mismatch")
    actual_names = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name not in (
            "SHA256SUMS", "SHA256SUMS.seal.sha256")
    }
    require(actual_names == sealed_names,
            "M511 sealed/actual member population mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chunk-elements", type=int,
                        default=8 * 1024 * 1024)
    args = parser.parse_args()
    require(sys.byteorder == "little", "M511 requires little endian")
    require(args.samples == 10 and args.num_workers == 0,
            "M511 requires S10 and num-workers=0")
    require(args.chunk_elements > 0 and args.chunk_elements % 8 == 0,
            "M511 invalid chunk-elements")

    script_start = sha256(Path(__file__).resolve())
    contract_path = args.contract.resolve()
    contract_start = sha256(contract_path)
    contract = strict_json(contract_path)
    require(contract.get("schema") ==
            "m511_h67_ep35_convtranspose_binary_input_capture_contract_v1" and
            contract.get("status") ==
            "LOCKED_STATIC_PREFLIGHT_REQUIRED_BEFORE_REMOTE_S10_CAPTURE",
            "M511 contract identity drift")
    output = args.output_dir.resolve()
    require((HW / contract["output"]["canonical_directory"]).resolve() == output,
            "M511 output directory is not canonical")
    require(output.parent.is_dir() and not output.exists(),
            "M511 output must be a new child of an existing directory")
    frozen_inputs = verify_inputs(contract, script_start)
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    require(config_path ==
            (ROOT / contract["inputs"]["config"]["path"]).resolve() and
            checkpoint_path ==
            (ROOT / contract["inputs"]["checkpoint"]["path"]).resolve(),
            "M511 runtime config/checkpoint path drift")
    require(checkpoint_path.stat().st_size ==
            contract["checkpoint_identity"]["size_bytes"],
            "M511 checkpoint size drift")

    staging = Path(tempfile.mkdtemp(
        prefix=output.name + ".staging.", dir=str(output.parent)))
    (staging / "calls").mkdir()
    handles = []
    records = []
    current = {"sample_id": None, "order": 0}
    published = False
    quarantine = None
    try:
        config, device = profile.load_config(config_path)
        require(torch.cuda.is_available() and torch.device(device).type == "cuda",
                "M511 requires an available CUDA device")
        expected_protocol = contract["eval_protocol"]
        require(config["model"]["name"] == "MS_SpikingformerFlowNet_en4" and
                config["model"]["use_upsample_conv"] is False and
                int(config["model"]["kernel_size"]) == 3,
                "M511 H67 decoder config drift")
        dataset = profile.DSECDatasetLite(
            config, file_list="valid", stereo=False,
            scale_factor=config.get("test", {}).get("scale_factor", 1))
        sequence_file = Path(dataset.sequence_file).resolve()
        require(sequence_file.is_file(),
                "M511 validation sequence list is missing")
        sequence_identity = {
            "path": str(sequence_file),
            "bytes": sequence_file.stat().st_size,
            "sha256": sha256(sequence_file),
        }
        sample_sources = [sample_source_identity(dataset, sample_id)
                          for sample_id in range(10)]
        require([row["sample_key"] for row in sample_sources] ==
                [row["sample_key"] for row in contract["samples"]],
                "M511 raw source/sample cohort drift")
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, drop_last=False,
            pin_memory=False, num_workers=args.num_workers)
        transform_valid = None
        if config["loader"].get("crop") is not None:
            transform_valid = profile.Compose([profile.CenterCrop((
                config["loader"]["crop"][0], config["loader"]["crop"][1]))])
        model = profile.build_model(config, checkpoint_path, device)
        load_audit = profile.validate_h9_load_audit(model, config)
        require(load_audit is not None and
                int(load_audit.get("missing_count", 0)) == 0 and
                int(load_audit.get("unexpected_count", 0)) == 0,
                "M511 checkpoint load is not exact")
        module_counts = profile.h9_module_counts(model)
        bn_policy = config.get("test", {}).get("bn_policy", "running")
        bn_changed = profile.configure_batch_norm_evaluation(model, bn_policy)
        observed_protocol = {
            "resolution": list(config["loader"]["resolution"]),
            "crop": config["loader"].get("crop"),
            "window_size": list(config["swin_transformer"]["window_size"]),
            "pretrained_window_size": config["swin_transformer"].get(
                "pretrained_window_size"),
            "tokens_per_window": int(np.prod(
                config["swin_transformer"]["window_size"])),
            "remap": config["loader"].get("remap"),
            "bn_policy": bn_policy,
            "bn_modules_changed": bn_changed,
            "eval_batch_size": 1,
            "num_workers": args.num_workers,
            "module_counts": module_counts,
        }
        require(observed_protocol == expected_protocol,
                "M511 frozen evaluation protocol mismatch")
        identities = module_identities(model, contract["modules"])
        named = dict(model.named_modules())
        runtime_convtranspose_names = [
            name for name, module in model.named_modules()
            if isinstance(module, torch.nn.ConvTranspose2d)
        ]
        require(runtime_convtranspose_names == [
            item["name"] for item in contract["modules"]
        ], "M511 complete runtime ConvTranspose2d module set drift")

        def make_hook(expected):
            def hook(module, inputs, output_tensor):
                require(current["sample_id"] is not None and
                        current["order"] == expected["module_index"],
                        "M511 target call order drift")
                require(isinstance(inputs, tuple) and len(inputs) == 1 and
                        torch.is_tensor(inputs[0]) and
                        torch.is_tensor(output_tensor),
                        "M511 hook tensor arity drift")
                require([int(item) for item in inputs[0].shape] ==
                        expected["input_shape"] and
                        [int(item) for item in output_tensor.shape] ==
                        expected["output_shape"],
                        "M511 hook shape drift: " + expected["name"])
                relative = "calls/s{:02d}_d{}.activation.le.bitpack".format(
                    current["sample_id"], expected["module_index"])
                stats = stream_binary_input(
                    inputs[0], args.chunk_elements, staging / relative)
                expected_elements = int(np.prod(expected["input_shape"]))
                require(stats["elements"] == expected_elements,
                        "M511 input element count drift")
                records.append({
                    "sample_id": current["sample_id"],
                    "sample_key": contract["samples"][current["sample_id"]][
                        "sample_key"],
                    "sequence_key": contract["samples"][current["sample_id"]][
                        "sequence_key"],
                    "module_index": expected["module_index"],
                    "name": expected["name"],
                    "operator": "ConvTranspose2d",
                    "input_shape": expected["input_shape"],
                    "output_shape": expected["output_shape"],
                    "relative_path": relative,
                    **stats,
                })
                current["order"] += 1
            return hook

        for expected in contract["modules"]:
            handles.append(named[expected["name"]].register_forward_hook(
                make_hook(expected)))

        sync_counts = {
            "before_capture": 0,
            "per_sample_post_forward": 0,
            "final_pre_manifest": 0,
        }
        torch.cuda.synchronize(device)
        sync_counts["before_capture"] += 1
        torch.cuda.reset_peak_memory_stats(device)
        memory_before = cuda_snapshot(device, "BEFORE_CAPTURE")
        processed = 0
        with torch.no_grad():
            for chunk, mask, label in loader:
                if processed >= args.samples:
                    break
                functional.reset_net(model)
                sample_key, sequence_key = sample_identity(dataset, processed)
                require(contract["samples"][processed] == {
                    "sample_id": processed,
                    "sample_key": sample_key,
                    "sequence_key": sequence_key,
                }, "M511 sample identity drift")
                current.update({"sample_id": processed, "order": 0})
                x, _label, _mask = profile.preprocess_chunk(
                    config, chunk, label, mask, transform_valid, device)
                model(x)
                torch.cuda.synchronize(device)
                sync_counts["per_sample_post_forward"] += 1
                require(current["order"] == len(contract["modules"]),
                        "M511 missing decoder call")
                current.update({"sample_id": None, "order": 0})
                processed += 1
                print("[M511] captured sample {}/10".format(processed),
                      flush=True)
        require(processed == 10 and len(records) == 40,
                "M511 capture population drift")
        torch.cuda.synchronize(device)
        sync_counts["final_pre_manifest"] += 1
        memory_after = cuda_snapshot(device, "AFTER_FINAL_SYNCHRONIZE")
        require(sync_counts == {
            "before_capture": 1,
            "per_sample_post_forward": 10,
            "final_pre_manifest": 1,
        }, "M511 synchronization count drift")
        # Hooks are part of the capture transaction.  Remove every one before
        # constructing or publishing a PASS artifact so a cleanup failure can
        # never leave a canonical successful directory behind.
        while handles:
            handles.pop().remove()
        require(sha256(Path(__file__).resolve()) == script_start,
                "M511 producer mutated during capture")
        verify_inputs(contract, script_start)
        require(sha256(contract_path) == contract_start,
                "M511 contract mutated during capture")
        rehash_sample_sources(
            sequence_file, sequence_identity, sample_sources)
        population = {
            "samples": 10,
            "modules": 4,
            "records": 40,
            "input_elements": sum(row["elements"] for row in records),
            "active_elements": sum(row["active"] for row in records),
            "packed_bytes": sum(row["packed_bytes"] for row in records),
        }
        expected_population = contract["expected_population"]
        require(all(population[key] == expected_population[key]
                    for key in ("samples", "modules", "records",
                                "input_elements", "packed_bytes")),
                "M511 final payload population mismatch")
        manifest = {
            "schema": "m511_h67_ep35_convtranspose_binary_input_trace_v1",
            "status": "PASS_EXACT_S10_FOUR_CONVTRANSPOSE_BINARY_INPUT_BITPACKS",
            "identity": {
                "contract_path": str(contract_path),
                "contract_sha256": sha256(contract_path),
                "inputs": frozen_inputs,
                "checkpoint_load_audit": load_audit,
            },
            "eval_protocol": observed_protocol,
            "raw_validation_sources": {
                "sequence_list": sequence_identity,
                "samples": sample_sources,
            },
            "cuda_synchronization": sync_counts,
            "cuda_memory": {"before": memory_before, "after": memory_after},
            "packing": {
                "order": "C_ORDER_FLAT",
                "bit_order": "little",
                "binary_values": [0, 1],
                "whole_call_contiguous_copy_allowed": False,
            },
            "module_identities": identities,
            "population": population,
            "records": records,
            "claim_boundary": {
                "exact_binary_input_bitpacks": True,
                "convtranspose_outputs": False,
                "cycles": False,
                "speedup": False,
                "rtl": False,
                "energy": False,
                "ppa": False,
                "system_speedup": False,
                "date_headline": False,
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        (staging / "RUN_COMPLETE.txt").write_text(
            "PASS_M511_EXACT_S10_CONVTRANSPOSE_INPUT_CAPTURE\n",
            encoding="utf-8")
        write_seal(staging)
        verify_seal(staging)
        require(not output.exists(), "M511 output appeared during capture")
        quarantine = output.with_name(
            output.name + ".quarantine.failed.{}.{}".format(
                os.getpid(), uuid.uuid4().hex))
        require(not quarantine.exists(),
                "M511 post-publication quarantine target already exists")
        os.replace(staging, output)
        published = True
        verify_seal(output)
        print("PASS M511 {} {}".format(
            output / "manifest.json", sha256(output / "manifest.json")),
            flush=True)
    except BaseException as error:
        if published:
            # The unique target was constructed and checked before publish, so
            # the first fallible post-publication recovery operation removes
            # the canonical directory.  A filesystem rename failure itself
            # remains externally non-admissible because this process exits.
            os.replace(output, quarantine)
            require(not output.exists() and quarantine.is_dir(),
                    "M511 failed to remove canonical output by quarantine")
            failure_root = quarantine
            failure_name = "FAILED_POSTPUBLICATION.json"
        else:
            failure_root = staging
            failure_name = "FAILED.json"
        failure = failure_root / failure_name
        if not failure.exists():
            failure.write_text(json.dumps({
                "schema": "m511_h67_ep35_convtranspose_capture_failure_v1",
                "status": "FAIL_CLOSED_NO_PASS_MANIFEST",
                "reason": "{}: {}".format(type(error).__name__, error),
                "completed_records": len(records),
                "staging_directory": str(staging),
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    finally:
        for handle in handles:
            try:
                handle.remove()
            except BaseException:
                # Any path reaching this fallback is already non-admissible:
                # successful publication requires handles to be empty above.
                pass
    return 0


if __name__ == "__main__":
    os.environ.setdefault("SDFORMER_USE_MLFLOW", "0")
    raise SystemExit(main())
