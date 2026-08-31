#!/usr/bin/env python3
"""Diagnose sample-0/d0 execution-mode drift between M649 and M660-r4.

This is a diagnostic only.  It publishes no activation payload and makes no
cycle, accuracy, energy, PPA, or paper claim.  Each invocation builds the
frozen H67 ep35 model in a fresh process and reports the first decoder input's
bit count/hash under one explicitly selected PyTorch execution mode.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch
from spikingjelly.activation_based import functional


ROOT = Path(__file__).resolve().parents[3]
M511 = ROOT / "neuron_experiments/H9_bipolar_self_attention/entrypoints/capture_m511_h67_convtranspose_binary_inputs.py"
M511_SHA256 = "e16a454d532acd15d96527cfddf43ebf9f95338a34ce9aeedbb10032cb26230a"
CONTRACT = ROOT / "hw_autoresearch_nts07/contracts/m511_h67_ep35_convtranspose_binary_input_capture_contract_r1_20260827.json"
CONFIG = ROOT / "neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml"
CHECKPOINT = ROOT / "hw_autoresearch_nts07/system_handoff/received/h67_ep35_system_trace_handoff_20260821/h67_ep35_system_trace_handoff_20260821/checkpoint/checkpoint_epoch35.pth"
REFERENCE = ROOT / "hw_autoresearch_nts07/system_handoff/outgoing/m511_h67_ep35_convtranspose_binary_inputs_s10_r1_20260827.staging.rn2m8c3a/calls/s00_d0.activation.le.bitpack"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_m511():
    require(sha256(M511) == M511_SHA256, "M681 frozen M511 producer drift")
    spec = importlib.util.spec_from_file_location("m681_frozen_m511", M511)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def configure(mode: str) -> dict:
    # Fresh-process defaults reproduce the unrecorded M511/M649 policy.  The
    # other modes change exactly one policy family at a time.
    if mode in ("deterministic_tf32", "m660"):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if mode == "m660":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return {
        "mode": mode,
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(
            torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("legacy", "deterministic_tf32", "m660"),
                        required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "M681 refuses existing output")
    require(args.output.parent.is_dir(), "M681 output parent missing")
    require(torch.cuda.is_available(), "M681 requires CUDA")
    mode = configure(args.mode)
    m511 = load_m511()
    contract = m511.strict_json(CONTRACT)
    m511.verify_inputs(contract, M511_SHA256)

    config, device = m511.profile.load_config(CONFIG)
    dataset = m511.profile.DSECDatasetLite(
        config, file_list="valid", stereo=False,
        scale_factor=config.get("test", {}).get("scale_factor", 1))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False,
        pin_memory=False, num_workers=0)
    transform = None
    if config["loader"].get("crop") is not None:
        transform = m511.profile.Compose([m511.profile.CenterCrop((
            config["loader"]["crop"][0], config["loader"]["crop"][1]))])
    model = m511.profile.build_model(config, CHECKPOINT, device)
    audit = m511.profile.validate_h9_load_audit(model, config)
    require(audit is not None and audit.get("missing_count") == 0 and
            audit.get("unexpected_count") == 0, "M681 checkpoint load drift")
    bn_policy = config.get("test", {}).get("bn_policy", "running")
    bn_changed = m511.profile.configure_batch_norm_evaluation(model, bn_policy)
    target = dict(model.named_modules())[contract["modules"][0]["name"]]
    observed = {}

    def hook(_module, inputs, _output):
        value = inputs[0].detach().contiguous().view(-1)
        exact = torch.logical_or(value == 0, value == 1)
        require(bool(torch.all(exact).item()), "M681 d0 is not exact binary")
        raw = value.to(device="cpu", dtype=torch.uint8).numpy()
        packed = np.packbits(raw, bitorder="little").tobytes(order="C")
        observed.update({
            "elements": int(raw.size),
            "one_count": int(raw.sum(dtype=np.uint64)),
            "zero_count": int(raw.size - raw.sum(dtype=np.uint64)),
            "packed_bytes": len(packed),
            "packed_sha256": hashlib.sha256(packed).hexdigest(),
        })

    handle = target.register_forward_hook(hook)
    chunk, mask, label = next(iter(loader))
    functional.reset_net(model)
    sample_key, sequence_key = m511.sample_identity(dataset, 0)
    require(contract["samples"][0]["sample_key"] == sample_key and
            contract["samples"][0]["sequence_key"] == sequence_key,
            "M681 sample-0 identity drift")
    x, _label, _mask = m511.profile.preprocess_chunk(
        config, chunk, label, mask, transform, device)
    with torch.no_grad():
        model(x)
    torch.cuda.synchronize(device)
    handle.remove()
    require(bool(observed), "M681 d0 hook not called")
    result = {
        "schema": "m681_m649_m660_execution_mode_drift_v1",
        "status": "DIAGNOSTIC_ONLY_NO_PERFORMANCE_CLAIM",
        "mode": mode,
        "sample": {"sample_id": 0, "sample_key": sample_key,
                   "sequence_key": sequence_key},
        "bn_policy": bn_policy,
        "bn_modules_changed": bn_changed,
        "checkpoint_load": audit,
        "observed_d0": observed,
        "m511_reference": {
            "path": str(REFERENCE), "sha256": sha256(REFERENCE),
            "one_count": 839586, "zero_count": 3768414,
        },
        "matches_m511_reference": (
            observed["packed_sha256"] == sha256(REFERENCE)),
        "identities": {
            "producer": {"path": str(Path(__file__).resolve()),
                         "sha256": sha256(Path(__file__).resolve())},
            "m511": {"path": str(M511), "sha256": sha256(M511)},
            "contract": {"path": str(CONTRACT), "sha256": sha256(CONTRACT)},
            "config": {"path": str(CONFIG), "sha256": sha256(CONFIG)},
            "checkpoint": {"path": str(CHECKPOINT),
                           "bytes": CHECKPOINT.stat().st_size,
                           "sha256": sha256(CHECKPOINT)},
        },
        "runtime": {
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": int(torch.backends.cudnn.version()),
            "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        },
        "claim_boundary": [
            "diagnostic only", "sample 0 and decoder d0 only",
            "not a cycle/speedup/accuracy/energy/PPA result",
            "does not admit M660 or decoder acceleration",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({"mode": args.mode, **observed,
                      "matches_m511": result["matches_m511_reference"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("SDFORMER_USE_MLFLOW", "0")
    raise SystemExit(main())
