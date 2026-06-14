"""Smoke-verify NTS-11 train/eval checkpoint chain on full SDFormer model."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import torchvision  # noqa: F401  # preload before overlay/models shadows torchvision.models

import torch
import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
OVERLAY = EXP_ROOT / "overlay"
BASELINE = REPO_ROOT / "experiments/baseline_stride_upstream/checkpoint_epoch59.pth"
DEFAULT_CONFIG = EXP_ROOT / "configs/generated/nts11b_hw_h60_s23_two_neuron_freeze1224_s1224.yml"


def _overlay_markers() -> tuple[str, ...]:
    return (".linear_v.", ".bn_v.", ".sn_v.", ".spiking_neuron.thresh", ".spiking_neuron.center")


def _is_overlay_key(key: str) -> bool:
    return any(marker in key for marker in _overlay_markers())


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _normalize_config(config: dict) -> dict:
    cfg = dict(config)
    if "spiking_neuron" in cfg:
        model_cfg = dict(cfg.get("model", {}))
        model_cfg["spiking_neuron"] = cfg["spiking_neuron"]
        cfg["model"] = model_cfg
    crop = cfg.get("loader", {}).get("crop")
    swin = dict(cfg.get("swin_transformer", {}))
    if crop is not None:
        swin["input_size"] = list(crop)
    else:
        swin["input_size"] = list(cfg.get("loader", {}).get("resolution", [288, 384]))
    cfg["swin_transformer"] = swin
    return cfg


def _build_model(config: dict, device: torch.device):
    sys.path.insert(0, str(REPO_ROOT / "third_party/SDformerFlow"))
    sys.path.insert(0, str(OVERLAY))
    from models.STSwinNet_SNN.Spiking_STSwinNet import MS_SpikingformerFlowNet_en4

    cfg = _normalize_config(config)
    model = MS_SpikingformerFlowNet_en4(cfg["model"].copy(), cfg["swin_transformer"].copy())
    model.init_weights()
    return model.to(device)


def _install_modules(model, config: dict) -> tuple[list[str], list[str]]:
    from models.STSwinNet_SNN.atlif_ternary_psn import atlif_ternary_summary, install_atlif_ternary_psn
    from models.STSwinNet_SNN.bsa_attention import install_shiftmax_attention, register_shiftmax_pickle_compat

    register_shiftmax_pickle_compat()
    neurons = install_atlif_ternary_psn(model, config.get("atlif_ternary_psn"))
    attn = install_shiftmax_attention(model, config.get("bsa_attention"))
    summary = atlif_ternary_summary(model)
    return neurons, attn, summary


def _load_baseline(model, device: torch.device) -> dict:
    from utils.utils import _extract_pretrained_state_dict

    pretrained = torch.load(BASELINE, map_location=device, weights_only=False)
    state = _extract_pretrained_state_dict(pretrained, test=False)
    overlay_ckpt = [k for k in state if _is_overlay_key(k)]
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    overlay_missing = [k for k in missing if _is_overlay_key(k)]
    overlay_unexpected = [k for k in unexpected if _is_overlay_key(k)]
    del pretrained
    return {
        "checkpoint_overlay_keys": len(overlay_ckpt),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "overlay_missing": overlay_missing,
        "overlay_unexpected": overlay_unexpected,
        "missing_sample": missing[:8],
    }


def _count_neuron_modes(model) -> dict[str, int]:
    from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN

    counts = {"ternary": 0, "binary": 0, "other": 0}
    for _, module in model.named_modules():
        if isinstance(module, ATLIFTernaryPSN):
            mode = str(getattr(module, "output_mode", "other"))
            counts[mode] = counts.get(mode, 0) + 1
    return counts


def _reload_saved(model, ckpt_path: Path, device: torch.device) -> dict:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    overlay_ckpt = [k for k in state if _is_overlay_key(k)]
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    overlay_missing = [k for k in missing if _is_overlay_key(k)]
    overlay_unexpected = [k for k in unexpected if _is_overlay_key(k)]
    return {
        "checkpoint_overlay_keys": len(overlay_ckpt),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "overlay_missing": overlay_missing,
        "overlay_unexpected": overlay_unexpected,
    }


def main() -> int:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    config = _load_yaml(config_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[verify] config={config_path}")
    print(f"[verify] baseline={BASELINE} exists={BASELINE.is_file()}")

    model = _build_model(config, device)
    vanilla_sn = sum(1 for _, m in model.named_modules() if m.__class__.__name__ == "Spiking_neuron")
    print(f"[verify] vanilla Spiking_neuron before install: {vanilla_sn}")

    neurons_pre, attn_pre, _ = _install_modules(model, config)
    print(f"[verify] preload install: neurons={len(neurons_pre)} attention={len(attn_pre)}")

    baseline_audit = _load_baseline(model, device)
    print(f"[verify] baseline load audit: {json.dumps(baseline_audit, indent=2)}")

    neurons_post, attn_post, summary = _install_modules(model, config)
    mode_counts = _count_neuron_modes(model)
    print(f"[verify] post-load install: neurons={len(neurons_post)} attention={len(attn_post)}")
    print(f"[verify] neuron modes: {mode_counts}")
    print(f"[verify] atlif summary: {summary}")

    if baseline_audit["overlay_unexpected"]:
        print("[verify] FAIL: overlay keys unexpected on baseline load")
        return 1
    if baseline_audit["checkpoint_overlay_keys"] and baseline_audit["overlay_missing"]:
        print("[verify] FAIL: baseline has overlay keys but model missing them")
        return 1
    if baseline_audit["checkpoint_overlay_keys"] == 0:
        print(
            f"[verify] expected: NB0 baseline has no overlay keys; "
            f"{len(baseline_audit['overlay_missing'])} ATLIF thresh/center init from install"
        )

    overlay_params = sum(1 for name, _ in model.named_parameters() if _is_overlay_key(name))
    print(f"[verify] overlay trainable params in model: {overlay_params}")

    with tempfile.TemporaryDirectory(prefix="nts11_chain_") as tmp:
        ckpt = Path(tmp) / "checkpoint_epoch0.pth"
        torch.save(model.state_dict(), ckpt)
        reload_audit = _reload_saved(model, ckpt, device)
        print(f"[verify] saved ckpt reload audit: {json.dumps(reload_audit, indent=2)}")
        if reload_audit["overlay_missing"] or reload_audit["overlay_unexpected"]:
            print("[verify] FAIL: saved checkpoint reload mismatch")
            return 1

    print("[verify] PASS: train preload -> baseline load -> forward -> save/reload chain ok")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("SDFORMER_USE_MLFLOW", "0")
    raise SystemExit(main())