"""Audit full-resolution geometry and checkpoint loading before DSEC training."""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import torch
import yaml


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
BASELINE = REPO / "third_party/SDformerFlow"
OVERLAY = EXP / "overlay"
DEFAULT_MANIFEST = EXP / "configs/generated/dsec_fullres_window9_manifest.json"
DEFAULT_OUTPUT = (
    REPO / "neuron_autoresearch/experiments/dsec_fullres_window9/load_chain_audit.json"
)
DERIVED_BUFFER_MARKERS = (
    "relative_position_index",
    "relative_coords_table",
    "attn_mask",
)


def setup_imports() -> None:
    try:
        __import__("mlflow")
    except ModuleNotFoundError:
        sys.modules["mlflow"] = types.ModuleType("mlflow")
    sys.path.insert(0, str(BASELINE))
    sys.path.insert(0, str(OVERLAY))


def build(config_path: Path, expected_window: int):
    from configs.parser import YAMLParser
    from models.STSwinNet_SNN.Spiking_STSwinNet import MS_SpikingformerFlowNet_en4
    from models.STSwinNet_SNN.atlif_ternary_psn import install_atlif_ternary_psn
    from models.STSwinNet_SNN.bsa_attention import install_shiftmax_attention

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = YAMLParser.combine_entries(raw)
    assert config["loader"]["crop"] is None
    assert config["loader"]["resolution"] == [480, 640]
    assert config["swin_transformer"]["window_size"] == [2, expected_window, expected_window]
    config["swin_transformer"]["input_size"] = [480, 640]
    model = MS_SpikingformerFlowNet_en4(
        config["model"].copy(), config["swin_transformer"].copy()
    )
    model.init_weights()
    neurons = install_atlif_ternary_psn(model, config.get("atlif_ternary_psn"))
    attentions = install_shiftmax_attention(model, config.get("bsa_attention"))
    return model, config, neurons, attentions


def source_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    from models.STSwinNet_SNN.bsa_attention import register_shiftmax_pickle_compat
    from utils.utils import _extract_pretrained_state_dict

    register_shiftmax_pickle_compat()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return _extract_pretrained_state_dict(payload, test=True)


def audit(row: dict, expected_window: int) -> dict:
    from models.STSwinNet_SNN.h9_load_audit import (
        is_h9_overlay_key,
        load_checkpoint_with_h9_audit,
    )

    config_path = Path(row["config"])
    checkpoint = Path(row["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model, config, neurons, attentions = build(config_path, expected_window)
    before = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if torch.is_floating_point(value)
    }
    state = source_state(checkpoint)
    model = load_checkpoint_with_h9_audit(
        checkpoint,
        model,
        torch.device("cpu"),
        config=config,
        remap="v1",
        test=True,
    )
    loaded = model.state_dict()
    remapped_state = {
        key: value.detach().clone()
        for key, value in state.items()
    }
    from utils.utils import load_pretrained_interpolate

    load_pretrained_interpolate(model, remapped_state)
    remapped_keys = [
        key
        for key, source_value in state.items()
        if "positional_encoding" in key
        and key in remapped_state
        and key in loaded
        and source_value.shape != loaded[key].shape
        and remapped_state[key].shape == loaded[key].shape
    ]
    remap_unequal = [
        key
        for key in remapped_keys
        if not torch.equal(
            remapped_state[key].detach().cpu(), loaded[key].detach().cpu()
        )
    ]

    comparable = [
        key
        for key, value in state.items()
        if key in loaded
        and value.shape == loaded[key].shape
        and not any(marker in key for marker in DERIVED_BUFFER_MARKERS)
    ]
    unequal = [
        key
        for key in comparable
        if not torch.equal(state[key].detach().cpu(), loaded[key].detach().cpu())
    ]
    changed = [
        key
        for key in comparable
        if key in before and not torch.equal(before[key], loaded[key].detach().cpu())
    ]
    overlay_keys = [key for key in state if is_h9_overlay_key(key)]
    result = {
        "id": row["id"],
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "geometry": {
            "resolution": config["loader"]["resolution"],
            "crop": config["loader"]["crop"],
            "window_size": config["swin_transformer"]["window_size"],
            "input_size": config["swin_transformer"]["input_size"],
        },
        "atlif_modules": len(neurons),
        "attention_modules": len(attentions),
        "checkpoint_overlay_keys": len(overlay_keys),
        "comparable_tensors": len(comparable),
        "unequal_after_load": unequal[:20],
        "changed_from_initialization": len(changed),
        "remapped_positional_tensors": len(remapped_keys),
        "remap_unequal_after_load": remap_unequal[:20],
        "remap": "v1",
    }
    assert len(neurons) == int(row["expected_atlif"]), result
    assert len(attentions) == int(row["expected_attention"]), result
    assert len(overlay_keys) == int(row["expected_overlay"]), result
    assert comparable and not unequal, result
    assert changed, result
    assert remapped_keys and not remap_unequal, result
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expected-window",
        type=int,
        default=9,
        help="expected spatial Swin window; default preserves the historical window9 audit",
    )
    args = parser.parse_args()
    setup_imports()
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    results = [audit(row, args.expected_window) for row in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(results, indent=2) + "\n"
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
