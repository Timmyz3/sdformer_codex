"""Generate H13 follow-up configs from existing H13 templates.

The generated configs stay inside the H9 experiment folder and only change the
overlay modules/config selected by the baseline entrypoint.
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


def partial_halfffn_down02_groups() -> list[dict]:
    return [
        {
            "name": "stage0_all_ffn_binary",
            "output_mode": "binary",
            "center_mode": "zero",
            "threshold_init": 0.1,
            "target_rate": None,
            "threshold_eta": 8.0e-05,
            "threshold_lr_scale": 8000.0,
            "max_threshold": 0.105,
            "activity_eta": 0.02,
            "paths": [
                "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.sn1",
                "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.sn2",
                "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.1.mlp.sn1",
                "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.1.mlp.sn2",
            ],
        },
        {
            "name": "stage1_half_even_ffn_binary",
            "output_mode": "binary",
            "center_mode": "zero",
            "threshold_init": 0.1,
            "target_rate": None,
            "threshold_eta": 8.0e-05,
            "threshold_lr_scale": 8000.0,
            "max_threshold": 0.105,
            "activity_eta": 0.02,
            "paths": [
                "sttmultires_unet.encoders.swin3d.layers.1.swin_blocks.0.mlp.sn1",
                "sttmultires_unet.encoders.swin3d.layers.1.swin_blocks.0.mlp.sn2",
            ],
        },
        {
            "name": "stage2_half_even_ffn_binary",
            "output_mode": "binary",
            "center_mode": "zero",
            "threshold_init": 0.1,
            "target_rate": None,
            "threshold_eta": 5.0e-05,
            "threshold_lr_scale": 6000.0,
            "max_threshold": 0.105,
            "activity_eta": 0.006,
            "paths": [
                "sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.0.mlp.sn1",
                "sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.0.mlp.sn2",
                "sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.2.mlp.sn1",
                "sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.2.mlp.sn2",
                "sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.4.mlp.sn1",
                "sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.4.mlp.sn2",
            ],
        },
        {
            "name": "stage3_block0_ffn_binary",
            "output_mode": "binary",
            "center_mode": "zero",
            "threshold_init": 0.1,
            "target_rate": None,
            "threshold_eta": 5.0e-05,
            "threshold_lr_scale": 6000.0,
            "max_threshold": 0.105,
            "activity_eta": 0.006,
            "paths": [
                "sttmultires_unet.encoders.swin3d.layers.3.swin_blocks.0.mlp.sn1",
                "sttmultires_unet.encoders.swin3d.layers.3.swin_blocks.0.mlp.sn2",
            ],
        },
        {
            "name": "downsample_stage0_stage2_binary",
            "output_mode": "binary",
            "center_mode": "zero",
            "threshold_init": 0.1,
            "target_rate": None,
            "threshold_eta": 8.0e-05,
            "threshold_lr_scale": 6000.0,
            "max_threshold": 0.105,
            "activity_eta": 0.02,
            "paths": [
                "sttmultires_unet.encoders.swin3d.layers.0.downsample.sn",
                "sttmultires_unet.encoders.swin3d.layers.2.downsample.sn",
            ],
        },
    ]


def make_common(base: dict, experiment: str, full: bool) -> dict:
    cfg = deepcopy(base)
    cfg["experiment"] = experiment
    cfg["runtime"]["max_train_steps"] = 0 if full else 120
    cfg["runtime"]["skip_state_save"] = True
    cfg["loader"]["n_epochs"] = 30 if full else 1
    cfg["atlif_ternary_psn"]["target_groups"] = partial_halfffn_down02_groups()
    cfg["note"] = (
        f"{experiment}. Bias-centered symmetric Q/K ternary firing with balanced "
        "positive/negative events; partial half-even FFN binary replacement plus "
        "downsample stage0/stage2 binary replacement."
    )
    return cfg


def main() -> None:
    shiftmax_base = load_config("h13j_biascenter_shiftmax_target05_guard120.yml")
    shiftnorm_base = deepcopy(shiftmax_base)
    shiftnorm_base["bsa_attention"]["mode"] = "signed_consensus_shiftnorm"
    shiftnorm_base["bsa_attention"]["center_scores"] = False
    shiftnorm_base["bsa_attention"]["consensus_bias"] = 1.0
    shiftnorm_base["note"] = "H13o template. Target05 bias-centered signed-consensus ShiftNorm."
    sparse_shiftmax_base = deepcopy(shiftmax_base)
    sparse_shiftmax_base["atlif_ternary_psn"]["target_rate"] = 0.02
    sparse_shiftmax_base["atlif_ternary_psn"]["target_rate_eta"] = 0.05
    sparse_shiftmax_base["atlif_ternary_psn"]["activity_eta"] = 1.2
    sparse_shiftmax_base["atlif_ternary_psn"]["max_threshold"] = 2.5
    sparse_shiftmax_base["note"] = (
        "H13p template. H13n attention/scope with stronger Q/K sparsity pressure "
        "to approach H9a-level SOPs while preserving signed ternary events."
    )
    mid_sparse_shiftmax_base = deepcopy(shiftmax_base)
    mid_sparse_shiftmax_base["atlif_ternary_psn"]["target_rate"] = 0.035
    mid_sparse_shiftmax_base["atlif_ternary_psn"]["target_rate_eta"] = 0.04
    mid_sparse_shiftmax_base["atlif_ternary_psn"]["activity_eta"] = 0.8
    mid_sparse_shiftmax_base["atlif_ternary_psn"]["max_threshold"] = 2.1
    mid_sparse_shiftmax_base["note"] = (
        "H13q template. Middle point between H13n target05 and H13p target02, "
        "intended to keep H13n-level AEE/AAE while reducing Q/K SOPs."
    )

    variants = [
        ("h13n_biascenter_shiftmax_target05_halfffn_down02", shiftmax_base),
        ("h13o_biascenter_shiftnorm_target05_halfffn_down02", shiftnorm_base),
        ("h13p_biascenter_shiftmax_target02_halfffn_down02", sparse_shiftmax_base),
        ("h13q_biascenter_shiftmax_target035_halfffn_down02", mid_sparse_shiftmax_base),
    ]
    for stem, base in variants:
        write_config(f"{stem}_guard120.yml", make_common(base, f"{stem}_guard120", full=False))
        write_config(f"{stem}_full.yml", make_common(base, f"{stem}_full", full=True))


if __name__ == "__main__":
    main()
