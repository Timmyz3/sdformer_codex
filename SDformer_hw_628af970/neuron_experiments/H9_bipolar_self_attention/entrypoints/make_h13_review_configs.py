"""Generate H13 review-derived follow-up configs.

These configs encode the immediate recommendations from
`neuron_autoresearch/H13_SERIES_REVIEW.md` and
`experiments/h13_signed_consensus_attention/H13_DEEP_ANALYSIS.md`.
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


def prepare(base: dict, experiment: str, full: bool) -> dict:
    cfg = deepcopy(base)
    cfg["experiment"] = experiment
    cfg["runtime"]["max_train_steps"] = 0 if full else 120
    cfg["loader"]["n_epochs"] = 30 if full else 1
    return cfg


def make_ang(base: dict, stem: str, full: bool) -> dict:
    cfg = prepare(base, f"{stem}_{'full' if full else 'guard120'}", full)
    cfg["loss"]["lambda_ang"] = 0.2
    cfg["loss"]["use_angular_loss"] = True
    cfg["note"] = (
        f"{cfg['experiment']}. H13n plus angular loss lambda_ang=0.2 to test "
        "whether AAE drift in long full-parameter fine-tuning can be suppressed."
    )
    return cfg


def make_shiftnorm(base: dict, stem: str, full: bool) -> dict:
    cfg = prepare(base, f"{stem}_{'full' if full else 'guard120'}", full)
    cfg["bsa_attention"]["mode"] = "signed_consensus_shiftnorm"
    cfg["bsa_attention"]["center_scores"] = False
    cfg["bsa_attention"]["preserve_mean"] = True
    cfg["bsa_attention"]["consensus_bias"] = 1.0
    cfg["note"] = (
        f"{cfg['experiment']}. H13n scope with ShiftNorm: signed popcount evidence "
        "and next-power-of-two normalization, no exponent/LUT."
    )
    return cfg


def make_popcount_l1(base: dict, stem: str, full: bool) -> dict:
    cfg = prepare(base, f"{stem}_{'full' if full else 'guard120'}", full)
    cfg["bsa_attention"]["mode"] = "signed_consensus_popcount_l1"
    cfg["bsa_attention"]["center_scores"] = False
    cfg["bsa_attention"]["preserve_mean"] = True
    cfg["bsa_attention"]["consensus_bias"] = 1.0
    cfg["note"] = (
        f"{cfg['experiment']}. Pure popcount L1 attention ablation: no Shiftmax, "
        "used to isolate whether signed consensus alone can retain AAE/AEE."
    )
    return cfg


def make_negative_target(base: dict, stem: str, full: bool) -> dict:
    cfg = prepare(base, f"{stem}_{'full' if full else 'guard120'}", full)
    cfg["atlif_ternary_psn"]["negative_target_rate"] = 0.025
    cfg["atlif_ternary_psn"]["negative_target_eta"] = 0.02
    cfg["atlif_ternary_psn"]["threshold_mode"] = "asymmetric_scale"
    cfg["atlif_ternary_psn"]["negative_scale_min"] = 0.7
    cfg["atlif_ternary_psn"]["negative_scale_max"] = 1.3
    cfg["note"] = (
        f"{cfg['experiment']}. Independent negative firing feedback around "
        "negative_target_rate=0.025, testing whether negative events can be "
        "kept influential without symmetric target-rate drift."
    )
    return cfg


def main() -> None:
    guard_base = load_config("h13n_biascenter_shiftmax_target05_halfffn_down02_guard120.yml")
    full_base = load_config("h13n_biascenter_shiftmax_target05_halfffn_down02_full.yml")
    builders = [
        ("h13r_ang02_h13n", make_ang),
        ("h13s_shiftnorm_h13n", make_shiftnorm),
        ("h13t_popcount_l1_h13n", make_popcount_l1),
        ("h13u_negtarget_h13n", make_negative_target),
    ]
    for stem, builder in builders:
        write_config(f"{stem}_guard120.yml", builder(guard_base, stem, full=False))
        write_config(f"{stem}_full.yml", builder(full_base, stem, full=True))


if __name__ == "__main__":
    main()
