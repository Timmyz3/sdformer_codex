from __future__ import annotations

import sys
from pathlib import Path


ENTRYPOINTS = Path(__file__).resolve().parents[1] / "entrypoints"
sys.path.insert(0, str(ENTRYPOINTS))

from make_dsec_fullres_paper_w15_rescue_configs import build_config
from make_dsec_fullres_w15_h66d_local5_bb1e4_config import (
    build_config as build_local5_bb1e4_config,
)


def test_rescue_config_preserves_paper_geometry_and_uses_stronger_lr() -> None:
    config = build_config(
        "H67",
        "bb1e4",
        name="unit_rescue",
        batch_size=2,
        epochs=1,
        screen=True,
    )
    assert config["loader"]["resolution"] == [480, 640]
    assert config["loader"]["crop"] is None
    assert config["loader"]["remap"] == "v1"
    assert config["swin_transformer"]["window_size"] == [2, 15, 15]
    assert config["optimizer"]["param_groups"]["backbone_lr"] == 1.0e-4
    assert config["optimizer"]["param_groups"]["norm_lr"] == 1.0e-4
    assert config["optimizer"]["lr_warmup"]["enabled"] is False
    assert config["runtime"]["force_save_epochs"] == [0]
    assert config["runtime"]["skip_state_save"] is True


def test_formal_rescue_keeps_candidate_structure_and_checkpoint_schedule() -> None:
    config = build_config(
        "H66d",
        "bb2e5",
        name="unit_rescue_formal",
        batch_size=2,
        epochs=30,
        screen=False,
    )
    assert config["bsa_attention"]["mode"] == "binary_axnor_local5_shiftmax"
    assert config["atlif_ternary_psn"]["enabled"] is True
    assert config["atlif_ternary_psn"]["output_mode"] == "binary"
    assert config["loader"]["n_epochs"] == 30
    assert config["runtime"]["force_save_epochs"] == [0, 4, 9, 14, 19, 24, 29]
    assert config["runtime"]["state_save_epochs"] == [9, 19, 29]


def test_nb0_conversion_declares_matching_pretrained_window() -> None:
    config = build_config(
        "H67",
        "bb2e5",
        name="unit_nb0_conversion",
        batch_size=2,
        epochs=1,
        screen=True,
        init="nb0_fullres_conversion",
    )
    assert config["swin_transformer"]["pretrained_window_size"] == [2, 15, 15]


def test_local5_bb1e4_matches_h67_effective_lr_and_is_resumable() -> None:
    config = build_local5_bb1e4_config()
    assert config["bsa_attention"]["mode"] == "binary_axnor_local5_shiftmax"
    assert config["atlif_ternary_psn"]["output_mode"] == "binary"
    assert config["loader"]["resolution"] == [480, 640]
    assert config["loader"]["crop"] is None
    assert config["swin_transformer"]["window_size"] == [2, 15, 15]
    assert config["swin_transformer"]["pretrained_window_size"] == [2, 9, 9]
    assert config["optimizer"]["milestones"] == [13, 20]
    assert config["optimizer"]["param_groups"]["backbone_lr"] == 1.0e-4
    assert config["runtime"]["force_save_epochs"] == [9, 14, 19, 24, 29]
    assert config["runtime"]["state_save_epochs"] == [9, 19, 29]
