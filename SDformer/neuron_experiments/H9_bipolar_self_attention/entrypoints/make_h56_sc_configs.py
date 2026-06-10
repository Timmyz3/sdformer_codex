"""Generate H56 SC-native agree/disagree improvement sweep configs.

All variants are based on H41 SC S012C (best SC result: AEE=1.622, AAE=9.455,
SOPs=3.128G). Only the attention mode and SC-specific params change.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = EXP_ROOT / "configs" / "generated"
BASE = GENERATED_DIR / "h41_scs012c_slowbb_full30_20260523_133312.yml"


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def main() -> int:
    base = read_yaml(BASE)

    variants: list[dict[str, Any]] = [
        # ── H56a: agree/disagree λ sweep ──
        {
            "name": "h56a_sc_agree_disagree_l03",
            "mode": "sc_agree_disagree_shiftmax",
            "bipolar_lambda": 0.3,
            "deadzone_epsilon": 0.0,
            "confidence_enabled": False,
            "k_consistency_mod": False,
            "consensus_score_norm": "head_dim",
        },
        {
            "name": "h56a_sc_agree_disagree_l05",
            "mode": "sc_agree_disagree_shiftmax",
            "bipolar_lambda": 0.5,
            "deadzone_epsilon": 0.0,
            "confidence_enabled": False,
            "k_consistency_mod": False,
            "consensus_score_norm": "head_dim",
        },
        {
            "name": "h56a_sc_agree_disagree_l08",
            "mode": "sc_agree_disagree_shiftmax",
            "bipolar_lambda": 0.8,
            "deadzone_epsilon": 0.0,
            "confidence_enabled": False,
            "k_consistency_mod": False,
            "consensus_score_norm": "head_dim",
        },
        {
            "name": "h56a_sc_agree_disagree_l10",
            "mode": "sc_agree_disagree_shiftmax",
            "bipolar_lambda": 1.0,
            "deadzone_epsilon": 0.0,
            "confidence_enabled": False,
            "k_consistency_mod": False,
            "consensus_score_norm": "head_dim",
        },
        # ── H56b: agree/disagree + deadzone ──
        {
            "name": "h56b_sc_ad_deadzone_e003",
            "mode": "sc_ad_deadzone_shiftmax",
            "bipolar_lambda": 0.5,
            "deadzone_epsilon": 1.0 / 32.0,  # ≈0.031: 1-vote margin = noise
            "confidence_enabled": False,
            "k_consistency_mod": False,
            "consensus_score_norm": "head_dim",
        },
        {
            "name": "h56b_sc_ad_deadzone_e006",
            "mode": "sc_ad_deadzone_shiftmax",
            "bipolar_lambda": 0.5,
            "deadzone_epsilon": 2.0 / 32.0,  # ≈0.063: 2-vote margin = noise
            "confidence_enabled": False,
            "k_consistency_mod": False,
            "consensus_score_norm": "head_dim",
        },
        # ── H56c: agree/disagree + deadzone + confidence ──
        {
            "name": "h56c_sc_ad_confidence",
            "mode": "sc_ad_confidence_shiftmax",
            "bipolar_lambda": 0.5,
            "deadzone_epsilon": 1.0 / 32.0,
            "confidence_enabled": True,
            "k_consistency_mod": False,
            "consensus_score_norm": "head_dim",
        },
        # ── H56d: agree/disagree + deadzone + confidence + K modulation ──
        {
            "name": "h56d_sc_ad_conf_kmod",
            "mode": "sc_ad_confidence_kmod_shiftmax",
            "bipolar_lambda": 0.5,
            "deadzone_epsilon": 1.0 / 32.0,
            "confidence_enabled": True,
            "k_consistency_mod": True,
            "consensus_score_norm": "head_dim",
        },
        # ── H56e: agree/disagree + active-norm denominator ──
        {
            "name": "h56e_sc_ad_activenorm",
            "mode": "sc_ad_activenorm_shiftmax",
            "bipolar_lambda": 0.5,
            "deadzone_epsilon": 0.0,
            "confidence_enabled": False,
            "k_consistency_mod": False,
            "consensus_score_norm": "active",
        },
    ]

    generated: list[str] = []
    for variant in variants:
        cfg = deepcopy(base)
        name = str(variant["name"])
        cfg["experiment"] = name
        cfg.pop("note", None)

        attn = cfg.setdefault("bsa_attention", {})
        attn["mode"] = str(variant["mode"])
        attn["bipolar_lambda"] = float(variant["bipolar_lambda"])
        attn["deadzone_epsilon"] = float(variant["deadzone_epsilon"])
        attn["confidence_enabled"] = bool(variant["confidence_enabled"])
        attn["k_consistency_mod"] = bool(variant["k_consistency_mod"])
        attn["consensus_score_norm"] = str(variant["consensus_score_norm"])

        cfg["note"] = (
            "H56 SC-native agree/disagree improvement. "
            f"variant={name}; mode={variant['mode']}; λ={variant['bipolar_lambda']}; "
            f"deadzone_ε={variant['deadzone_epsilon']:.4f}; "
            f"confidence={variant['confidence_enabled']}; kmod={variant['k_consistency_mod']}; "
            f"norm={variant['consensus_score_norm']}. "
            "Based on H41 SC S012C slowbb (best SC: AEE=1.622, AAE=9.455, SOPs=3.128G)."
        )

        out = GENERATED_DIR / f"{name}.yml"
        write_yaml(out, cfg)
        generated.append(f"generated/{out.name}")

    print("\n".join(generated))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
