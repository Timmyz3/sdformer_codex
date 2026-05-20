"""Generate H14 strict-BSA follow-up configs.

H14 differs from H13 by replacing the token-wise signed consensus gate with a
true ternary QK^T Shiftmax attention matrix. SDFormerFlow's attention block has
no separate V projection, so K is reused as the value carrier.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def load_config(name: str) -> dict:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_config(name: str, config: dict) -> None:
    with (CONFIG_DIR / name).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def make_variant(base: dict, experiment: str, *, full: bool, value_mode: str, norm: str, score_scale: float) -> dict:
    cfg = deepcopy(base)
    cfg["experiment"] = experiment
    cfg["runtime"]["max_train_steps"] = 0 if full else 120
    cfg["loader"]["n_epochs"] = 30 if full else 1
    cfg["bsa_attention"]["mode"] = "strict_bsa_shiftmax"
    cfg["bsa_attention"]["value_mode"] = value_mode
    cfg["bsa_attention"]["preserve_mean"] = False
    cfg["bsa_attention"]["center_scores"] = True
    cfg["bsa_attention"]["consensus_score_norm"] = norm
    cfg["bsa_attention"]["score_scale"] = score_scale
    cfg["note"] = (
        f"{experiment}. Strict BSA matrix attention: sign(Q) @ sign(K)^T then "
        f"Shiftmax, with K reused as V using value_mode={value_mode}, "
        f"norm={norm}, score_scale={score_scale}. FFN/downsample scope matches H13n."
    )
    return cfg


def main() -> None:
    guard_base = load_config("h13n_biascenter_shiftmax_target05_halfffn_down02_guard120.yml")
    full_base = load_config("h13n_biascenter_shiftmax_target05_halfffn_down02_full.yml")
    variants = [
        ("h14a_strict_bsa_thetav_sqrt", "threshold", "sqrt_head_dim", 1.0),
        ("h14b_strict_bsa_signv_sqrt", "sign", "sqrt_head_dim", 1.0),
        ("h14c_strict_bsa_thetav_mild", "threshold", "head_dim", 2.0),
    ]
    for stem, value_mode, norm, score_scale in variants:
        write_config(
            f"{stem}_guard120.yml",
            make_variant(
                guard_base,
                f"{stem}_guard120",
                full=False,
                value_mode=value_mode,
                norm=norm,
                score_scale=score_scale,
            ),
        )
        write_config(
            f"{stem}_full.yml",
            make_variant(
                full_base,
                f"{stem}_full",
                full=True,
                value_mode=value_mode,
                norm=norm,
                score_scale=score_scale,
            ),
        )


if __name__ == "__main__":
    main()
