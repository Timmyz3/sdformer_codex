"""Audit H76-H78 warm-start registration and strict trained-checkpoint loading."""

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
OUTPUT = REPO / "neuron_autoresearch/experiments/h76_h78_round3_match/load_chain_audit.json"
MARKERS = (
    ".linear_v.", ".bn_v.", ".sn_v.",
    "._h9_match_code_weight", "._h9_lc4_coefficients",
    ".spiking_neuron.thresh", ".spiking_neuron.center",
)
CANDIDATE_MARKERS = ("._h9_match_code_weight", "._h9_lc4_coefficients")
EXPECTED_NEW_KEYS = {
    "binary_pc9_patch_match_code": 12,
    "binary_lc4_match_code": 24,
    "binary_g4_match_code": 12,
}


def _imports() -> None:
    sys.path.insert(0, str(BASELINE))
    sys.path.insert(0, str(OVERLAY))


def _build(config_path: Path):
    from configs.parser import YAMLParser
    from models.STSwinNet_SNN.Spiking_STSwinNet import MS_SpikingformerFlowNet_en4
    from models.STSwinNet_SNN.atlif_ternary_psn import install_atlif_ternary_psn
    from models.STSwinNet_SNN.bsa_attention import install_shiftmax_attention, shiftmax_attention_summary

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = YAMLParser.combine_entries(raw)
    crop = config["loader"].get("crop")
    config["swin_transformer"]["input_size"] = list(crop or config["loader"]["resolution"])
    model = MS_SpikingformerFlowNet_en4(config["model"].copy(), config["swin_transformer"].copy())
    model.init_weights()
    neurons = install_atlif_ternary_psn(model, config.get("atlif_ternary_psn"))
    attentions = install_shiftmax_attention(model, config.get("bsa_attention"))
    return model, config, neurons, attentions, shiftmax_attention_summary(model)


def _state(checkpoint: Path, *, strip_module: bool = False) -> dict[str, torch.Tensor]:
    from models.STSwinNet_SNN.bsa_attention import register_shiftmax_pickle_compat
    from utils.utils import _extract_pretrained_state_dict

    register_shiftmax_pickle_compat()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return _extract_pretrained_state_dict(payload, test=strip_module)


def _base_result(config_path: Path, checkpoint: Path, model, config, neurons, attentions, summary) -> dict:
    return {
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "mode": str(config["bsa_attention"]["mode"]),
        "atlif_modules": len(neurons),
        "attention_modules": len(attentions),
        "candidate_modules": int(summary.get("match_code_modules", 0)),
        "candidate_parameters": int(summary.get("match_code_parameters", 0)),
    }


def audit_warmstart(config_path: Path, checkpoint: Path) -> dict:
    model, config, neurons, attentions, summary = _build(config_path)
    state = _state(checkpoint)
    overlay_keys = [key for key in state if any(marker in key for marker in MARKERS)]
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    candidate_missing = [
        key for key in missing if any(marker in key for marker in CANDIDATE_MARKERS)
    ]
    non_candidate_missing = [key for key in missing if key not in candidate_missing]
    mode = str(config["bsa_attention"]["mode"])
    result = _base_result(config_path, checkpoint, model, config, neurons, attentions, summary)
    result.update({
        "audit": "frozen_ttx_ep2_warmstart",
        "checkpoint_overlay_keys": len(overlay_keys),
        "missing": len(missing),
        "candidate_missing": candidate_missing,
        "non_candidate_missing": non_candidate_missing,
        "unexpected": unexpected,
    })
    assert result["atlif_modules"] == 105, result
    assert result["attention_modules"] == 12, result
    assert result["candidate_modules"] == 12, result
    assert result["checkpoint_overlay_keys"] == 210, result
    assert len(candidate_missing) == EXPECTED_NEW_KEYS[mode], result
    assert len(missing) == EXPECTED_NEW_KEYS[mode], result
    assert not non_candidate_missing and not unexpected, result

    fresh, _, _, _, _ = _build(config_path)
    registration = fresh.load_state_dict(model.state_dict(), strict=True)
    result["registered_state_strict_reload"] = {
        "missing": list(registration.missing_keys),
        "unexpected": list(registration.unexpected_keys),
    }
    assert not registration.missing_keys and not registration.unexpected_keys, result
    return result


def audit_trained(config_path: Path, checkpoint: Path) -> dict:
    model, config, neurons, attentions, summary = _build(config_path)
    state = _state(checkpoint, strip_module=True)
    incompatible = model.load_state_dict(state, strict=True)
    result = _base_result(config_path, checkpoint, model, config, neurons, attentions, summary)
    result.update({
        "audit": "trained_checkpoint_strict_load",
        "checkpoint_overlay_keys": sum(
            any(marker in key for marker in MARKERS) for key in state
        ),
        "missing": list(incompatible.missing_keys),
        "unexpected": list(incompatible.unexpected_keys),
    })
    assert result["atlif_modules"] == 105, result
    assert result["attention_modules"] == 12, result
    assert result["candidate_modules"] == 12, result
    assert not incompatible.missing_keys and not incompatible.unexpected_keys, result
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", default=[])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_TTX)
    parser.add_argument(
        "--trained", type=Path, nargs=2, action="append", metavar=("CONFIG", "CHECKPOINT"), default=[]
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not args.config and not args.trained:
        parser.error("provide at least one --config or --trained CONFIG CHECKPOINT")
    _imports()
    rows = [audit_warmstart(config, args.checkpoint) for config in args.config]
    rows.extend(audit_trained(config, checkpoint) for config, checkpoint in args.trained)
    text = json.dumps(rows, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
