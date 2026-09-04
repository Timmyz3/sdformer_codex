#!/usr/bin/env python3
"""Extract the ten M32 producer thresholds from the frozen H67 checkpoint.

This entrypoint is intentionally read-only with respect to the checkpoint and
refuses to overwrite its JSON output.  It must run in the SDFormer PyTorch
environment because the checkpoint contains a pickled model object.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


EXPECTED_CHECKPOINT_SHA256 = (
    "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158"
)
TARGET_PRODUCERS = [
    "sttmultires_unet.encoders.swin3d.layers.2.downsample.sn.spiking_neuron",
    "sttmultires_unet.encoders.swin3d.layers.3.swin_blocks.0.mlp.sn1.spiking_neuron",
    "sttmultires_unet.encoders.swin3d.layers.3.swin_blocks.1.mlp.sn1.spiking_neuron",
    "sttmultires_unet.preds.0.sn.spiking_neuron",
    "sttmultires_unet.preds.1.sn.spiking_neuron",
    "sttmultires_unet.preds.2.sn.spiking_neuron",
    "sttmultires_unet.resblocks.0.sn1.spiking_neuron",
    "sttmultires_unet.resblocks.0.sn2.spiking_neuron",
    "sttmultires_unet.resblocks.1.sn1.spiking_neuron",
    "sttmultires_unet.resblocks.1.sn2.spiking_neuron",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def install_pickle_import_paths(repo_root: Path) -> None:
    baseline_root = repo_root / "third_party" / "SDformerFlow"
    overlay_root = (
        repo_root / "neuron_experiments" / "H9_bipolar_self_attention" / "overlay"
    )
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(baseline_root))
    sys.path.insert(0, str(overlay_root))

    import models
    import models.STSwinNet_SNN as stsnn

    overlay_models = str(overlay_root / "models")
    overlay_stsnn = str(overlay_root / "models" / "STSwinNet_SNN")
    if overlay_models not in list(models.__path__):
        models.__path__.append(overlay_models)
    if overlay_stsnn not in list(stsnn.__path__):
        stsnn.__path__.append(overlay_stsnn)
    import models.STSwinNet_SNN.atlif_ternary_psn  # noqa: F401
    from models.STSwinNet_SNN.bsa_attention import register_shiftmax_pickle_compat

    register_shiftmax_pickle_compat()


def build_manifest(repo_root: Path, checkpoint: Path) -> dict:
    import torch

    checkpoint_hash = file_sha256(checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"H67 checkpoint hash drift: {checkpoint_hash} != "
            f"{EXPECTED_CHECKPOINT_SHA256}"
        )
    install_pickle_import_paths(repo_root)
    loaded = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if hasattr(loaded, "state_dict"):
        state_dict = loaded.state_dict()
    elif isinstance(loaded, dict):
        state_dict = loaded.get("state_dict", loaded.get("model", loaded))
    else:
        raise ValueError(f"unsupported checkpoint payload: {type(loaded).__name__}")
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()

    scalar_threshold_keys = sorted(
        key
        for key, value in state_dict.items()
        if key.endswith(".thresh") and hasattr(value, "numel") and value.numel() == 1
    )
    if len(scalar_threshold_keys) != 105:
        raise ValueError(
            f"expected 105 scalar ATLIF thresholds, got {len(scalar_threshold_keys)}"
        )
    rows = []
    for producer in TARGET_PRODUCERS:
        key = producer + ".thresh"
        if key not in state_dict:
            raise ValueError(f"missing M32 checkpoint threshold: {key}")
        value = state_dict[key]
        if value.numel() != 1 or tuple(value.shape) != ():
            raise ValueError(f"M32 threshold is not scalar: {key} {tuple(value.shape)}")
        rows.append(
            {
                "producer": producer,
                "state_dict_key": key,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "value_float32": float(value.item()),
                "value_raw_le_hex": value.detach().cpu().contiguous().numpy().tobytes().hex(),
            }
        )

    profile_source = (
        repo_root / "neuron_experiments" / "H9_bipolar_self_attention"
        / "entrypoints" / "profile_nts11_hardware_p0.py"
    )
    profile_text = profile_source.read_text(encoding="utf-8")
    if "model.eval()" not in profile_text or "threshold_update(" in profile_text:
        raise ValueError("profile inference threshold-static source contract drift")
    return {
        "schema": "m32_h67_checkpoint_threshold_manifest_v1",
        "status": "PASS_FROZEN_SCALAR_THRESHOLDS_INFERENCE_PROFILE_STATIC",
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": checkpoint_hash,
            "payload_type": type(loaded).__name__,
            "scalar_threshold_population": len(scalar_threshold_keys),
        },
        "inference_profile": {
            "path": str(profile_source.resolve()),
            "sha256": file_sha256(profile_source),
            "model_eval_token_present": True,
            "threshold_update_call_absent": True,
        },
        "extractor": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
            "torch_version": torch.__version__,
        },
        "producers": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite M32 threshold manifest: {args.output}")
    report = build_manifest(args.repo_root.resolve(), args.checkpoint.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
