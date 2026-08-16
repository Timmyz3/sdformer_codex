"""CPU audit of Match-Code installation and TTX checkpoint warm-start semantics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
BASELINE = REPO / "third_party/SDformerFlow"
OVERLAY = EXP / "overlay"
DEFAULT_TTX = EXP / "results/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/checkpoint_epoch2.pth"
MARKERS = (".linear_v.", ".bn_v.", ".sn_v.", "._h9_match_code_weight", ".spiking_neuron.thresh", ".spiking_neuron.center")


def audit(config_path: Path, checkpoint: Path) -> dict:
    sys.path.insert(0, str(BASELINE))
    sys.path.insert(0, str(OVERLAY))
    from configs.parser import YAMLParser
    from models.STSwinNet_SNN.Spiking_STSwinNet import MS_SpikingformerFlowNet_en4
    from models.STSwinNet_SNN.atlif_ternary_psn import install_atlif_ternary_psn
    from models.STSwinNet_SNN.bsa_attention import (
        install_shiftmax_attention,
        register_shiftmax_pickle_compat,
        shiftmax_attention_summary,
    )
    from utils.utils import _extract_pretrained_state_dict

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = YAMLParser.combine_entries(config)
    crop = config["loader"].get("crop")
    config["swin_transformer"]["input_size"] = list(crop or config["loader"]["resolution"])
    model = MS_SpikingformerFlowNet_en4(config["model"].copy(), config["swin_transformer"].copy())
    model.init_weights()
    neurons = install_atlif_ternary_psn(model, config.get("atlif_ternary_psn"))
    attentions = install_shiftmax_attention(model, config.get("bsa_attention"))
    summary = shiftmax_attention_summary(model)

    register_shiftmax_pickle_compat()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = _extract_pretrained_state_dict(payload, test=False)
    overlay_keys = [key for key in state if any(marker in key for marker in MARKERS)]
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    match_missing = [key for key in missing if "._h9_match_code_weight" in key]
    non_match_missing = [key for key in missing if "._h9_match_code_weight" not in key]

    result = {
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "atlif_modules": len(neurons),
        "attention_modules": len(attentions),
        "match_code_modules": int(summary.get("match_code_modules", 0)),
        "match_code_parameters": int(summary.get("match_code_parameters", 0)),
        "checkpoint_overlay_keys": len(overlay_keys),
        "missing": len(missing),
        "match_code_missing": len(match_missing),
        "non_match_missing": non_match_missing,
        "unexpected": unexpected,
    }
    assert result["atlif_modules"] == 105, result
    assert result["attention_modules"] == 12, result
    assert result["match_code_modules"] == 12, result
    assert result["checkpoint_overlay_keys"] == 210, result
    assert result["missing"] == 12 and result["match_code_missing"] == 12, result
    assert not non_match_missing and not unexpected, result
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_TTX)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [audit(config, args.checkpoint) for config in args.config]
    text = json.dumps(rows, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
