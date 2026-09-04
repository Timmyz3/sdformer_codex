"""Generate DATE11 all-binary FAPS configs.

FAPS keeps the TX-style hardware-friendly popcount selector, but makes the
score flow-aligned by splitting each attention head into x/y directional
groups and adding a sparse quantized K-magnitude lane on confident tokens.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GEN = EXP_ROOT / "configs/generated"
MANIFEST = GEN / "date11_allbinary_faps_manifest.json"

FULL30_BASE = GEN / "date11full_all_binary_atlif_tx_w720_fastlr_full30.yml"
FT5_BASE = GEN / "date11full_all_binary_atlif_tx_stdlr_ft_ep19_ft5.yml"

NB0_RESUME = Path("experiments/baseline_stride_upstream/checkpoint_epoch59.pth")
TX_EP19_RESUME = Path(
    "neuron_experiments/H9_bipolar_self_attention/results/"
    "date11full_all_binary_atlif_tx_w720_fastlr_full30_bs8_20260617_024526_setsid/"
    "checkpoint_epoch19.pth"
)
S2_BLOCKS = ["2:0", "2:1", "2:2", "2:3", "2:4", "2:5"]


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def apply_faps(
    cfg: dict[str, Any],
    *,
    target_blocks: list[str] | None = None,
    k_magnitude_alpha: float = 0.03125,
    faps_same_zero_weight: float = 1.0,
    faps_same_nonzero_weight: float = 4.0,
    faps_opposite_weight: float = 1.0,
    faps_single_active_weight: float = 4.0,
) -> None:
    attn = cfg.setdefault("bsa_attention", {})
    attn.update(
        {
            "enabled": True,
            "mode": "faps",
            "center_scores": True,
            "preserve_mean": True,
            "alpha0": 0.02,
            "mismatch_penalty": 0.25,
            "single_active_penalty": 0.05,
            "single_active_penalty_grad": "ste",
            "score_scale": 1.0,
            "consensus_bias": 0.02,
            "consensus_score_norm": "head_dim",
            "eps": 1.0e-6,
            "relu_k_floor": 0.0,
            "value_mode": "threshold",
            "target_rate": None,
            "sc_mu_schedule_enabled": False,
            "bipolar_mu": 0.0,
            "k_magnitude_alpha": float(k_magnitude_alpha),
            "directional_channels_enabled": True,
            "directional_merge_mode": "mean",
            "flow_disagreement_gamma": 0.0,
            "faps_same_nonzero_weight": float(faps_same_nonzero_weight),
            "faps_same_zero_weight": float(faps_same_zero_weight),
            "faps_opposite_weight": float(faps_opposite_weight),
            "faps_single_active_weight": float(faps_single_active_weight),
            "confidence_min_active": 8 if k_magnitude_alpha else 0,
            "kmag_quantize_bits": 2,
        }
    )
    if target_blocks is not None:
        attn["target_blocks"] = target_blocks


def make_full30() -> tuple[Path, dict[str, Any]]:
    cfg = deepcopy(read_yaml(FULL30_BASE))
    cfg["experiment"] = "date11full_all_binary_atlif_faps_all12_w720_fastlr_full30"
    cfg["note"] = (
        "DATE11 all-binary FAPS full30: 105 binary ATLIF modules; "
        "12 FAPS attention modules; flow-aligned x/y dyadic popcount, "
        "sparse 2-bit K magnitude on active>=8. Resume from NB0 ep59."
    )
    apply_faps(cfg)
    cfg.setdefault("loader", {})["n_epochs"] = 30
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = False
    runtime["force_save_epochs"] = [0, 4, 9, 14, 19, 24, 28, 29]
    runtime["use_mlflow_model_logging"] = False

    out = GEN / f"{cfg['experiment']}.yml"
    write_yaml(out, cfg)
    return out, {
        "name": cfg["experiment"],
        "kind": "full30",
        "config": str(out),
        "resume": str(NB0_RESUME),
        "attention_mode": "faps",
        "target_blocks": "all12",
        "k_magnitude_alpha": 0.03125,
        "directional_channels_enabled": True,
        "confidence_min_active": 8,
        "expected_atlif": 105,
        "expected_shiftmax": 12,
        "status": "config_generated_not_run",
    }


def make_ft5() -> tuple[Path, dict[str, Any]]:
    cfg = deepcopy(read_yaml(FT5_BASE))
    cfg["experiment"] = "date11full_all_binary_atlif_faps_all12_stdlr_ft_txep19_ft5"
    cfg["note"] = (
        "DATE11 all-binary FAPS FT5: switch existing all-binary+TX ep19 "
        "checkpoint to FAPS attention; 105 binary ATLIF modules; 12 FAPS "
        "attention modules; flow-aligned x/y dyadic popcount, sparse 2-bit "
        "K magnitude on active>=8."
    )
    apply_faps(cfg)
    cfg.setdefault("loader", {})["n_epochs"] = 5
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = False
    runtime["force_save_epochs"] = [0, 1, 2, 3, 4]
    runtime["use_mlflow_model_logging"] = False

    out = GEN / f"{cfg['experiment']}.yml"
    write_yaml(out, cfg)
    return out, {
        "name": cfg["experiment"],
        "kind": "ft5_from_tx_ep19",
        "config": str(out),
        "resume": str(TX_EP19_RESUME),
        "attention_mode": "faps",
        "target_blocks": "all12",
        "k_magnitude_alpha": 0.03125,
        "directional_channels_enabled": True,
        "confidence_min_active": 8,
        "expected_atlif": 105,
        "expected_shiftmax": 12,
        "status": "config_generated_not_run",
    }


def make_ft5_all12_nokmag() -> tuple[Path, dict[str, Any]]:
    cfg = deepcopy(read_yaml(FT5_BASE))
    cfg["experiment"] = "date11full_all_binary_atlif_faps_all12_nokmag_stdlr_ft_txep19_ft5"
    cfg["note"] = (
        "DATE11 all-binary FAPS FT5 strict mainline candidate: switch "
        "all-binary+TX ep19 checkpoint to FAPS attention on all 12 attention "
        "blocks; 105 binary ATLIF/PSN modules; no K magnitude side lane."
    )
    apply_faps(cfg, target_blocks=None, k_magnitude_alpha=0.0)
    cfg.setdefault("loader", {})["n_epochs"] = 5
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = False
    runtime["force_save_epochs"] = [0, 1, 2, 3, 4]
    runtime["use_mlflow_model_logging"] = False

    out = GEN / f"{cfg['experiment']}.yml"
    write_yaml(out, cfg)
    return out, {
        "name": cfg["experiment"],
        "kind": "ft5_from_tx_ep19_strict_all12_nokmag",
        "config": str(out),
        "resume": str(TX_EP19_RESUME),
        "attention_mode": "faps",
        "target_blocks": "all12",
        "k_magnitude_alpha": 0.0,
        "directional_channels_enabled": True,
        "confidence_min_active": 0,
        "expected_atlif": 105,
        "expected_shiftmax": 12,
        "status": "config_generated_not_run",
    }


def make_ft10_all12_nokmag_slowlr() -> tuple[Path, dict[str, Any]]:
    cfg = deepcopy(read_yaml(FT5_BASE))
    cfg["experiment"] = "date11full_all_binary_atlif_faps_all12_nokmag_slowlr_ft_txep19_ft10"
    cfg["note"] = (
        "DATE11 all-binary FAPS FT10 slowlr: same strict definition as the "
        "all12 noKmag FT5 run, but with lower backbone/neuron/threshold LR "
        "and a later LR milestone to test whether FT5/stdlr was too short or "
        "too aggressive."
    )
    apply_faps(cfg, target_blocks=None, k_magnitude_alpha=0.0)
    opt = cfg.setdefault("optimizer", {})
    opt["lr"] = 1.0e-05
    opt["milestones"] = [8]
    groups = opt.setdefault("param_groups", {})
    groups["backbone_lr"] = 5.0e-07
    groups["norm_lr"] = 5.0e-07
    groups["neuron_lr"] = 1.5e-05
    groups["threshold_lr"] = 2.5e-06
    warmup = opt.setdefault("lr_warmup", {})
    warmup["enabled"] = True
    warmup["steps"] = 720
    warmup["start_factor"] = 0.05
    cfg.setdefault("loader", {})["n_epochs"] = 10
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = False
    runtime["force_save_epochs"] = list(range(10))
    runtime["use_mlflow_model_logging"] = False

    out = GEN / f"{cfg['experiment']}.yml"
    write_yaml(out, cfg)
    return out, {
        "name": cfg["experiment"],
        "kind": "ft10_from_tx_ep19_strict_all12_nokmag_slowlr",
        "config": str(out),
        "resume": str(TX_EP19_RESUME),
        "attention_mode": "faps",
        "target_blocks": "all12",
        "k_magnitude_alpha": 0.0,
        "directional_channels_enabled": True,
        "confidence_min_active": 0,
        "lr_recipe": "slowlr_backbone5e-7_neuron1.5e-5_threshold2.5e-6_milestone8",
        "expected_atlif": 105,
        "expected_shiftmax": 12,
        "status": "config_generated_not_run",
    }


def make_ft5_s2_nokmag() -> tuple[Path, dict[str, Any]]:
    cfg = deepcopy(read_yaml(FT5_BASE))
    cfg["experiment"] = "date11full_all_binary_atlif_faps_s2only_nokmag_stdlr_ft_txep19_ft5"
    cfg["note"] = (
        "DATE11 all-binary FAPS FT5 selected from short sweep: switch "
        "all-binary+TX ep19 checkpoint to FAPS attention on stage2 only; "
        "105 binary ATLIF modules; 6 FAPS attention modules; no K magnitude "
        "side lane for the clean popcount-selector hardware story."
    )
    apply_faps(cfg, target_blocks=S2_BLOCKS, k_magnitude_alpha=0.0)
    cfg.setdefault("loader", {})["n_epochs"] = 5
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = False
    runtime["force_save_epochs"] = [0, 1, 2, 3, 4]
    runtime["use_mlflow_model_logging"] = False

    out = GEN / f"{cfg['experiment']}.yml"
    write_yaml(out, cfg)
    return out, {
        "name": cfg["experiment"],
        "kind": "ft5_from_tx_ep19_short_selected",
        "config": str(out),
        "resume": str(TX_EP19_RESUME),
        "attention_mode": "faps",
        "target_blocks": "s2only",
        "k_magnitude_alpha": 0.0,
        "directional_channels_enabled": True,
        "confidence_min_active": 0,
        "expected_atlif": 105,
        "expected_shiftmax": 6,
        "status": "config_generated_not_run",
    }


def make_ft5_txratio_integer_best() -> tuple[Path, dict[str, Any]]:
    cfg = deepcopy(read_yaml(FT5_BASE))
    cfg["experiment"] = "date11full_all_binary_atlif_faps_all12_nokmag_s64_z1_p6_sc0p015625_nosplit_stdlr_ft_txep19_ft5"
    cfg["note"] = (
        "DATE11 all-binary FAPS FT5 selected from integer TX-ratio short tune: "
        "all 12 attention blocks, no K magnitude side lane, no x/y split. "
        "Effective all-binary score is (64*same_active + 1*same_zero - "
        "6*single_active) >> 6, approximating TX ratio 1:0.02:0.10."
    )
    apply_faps(
        cfg,
        target_blocks=None,
        k_magnitude_alpha=0.0,
        faps_same_nonzero_weight=64.0,
        faps_same_zero_weight=1.0,
        faps_single_active_weight=6.0,
    )
    attn = cfg.setdefault("bsa_attention", {})
    attn["score_scale"] = 1.0 / 64.0
    attn["directional_channels_enabled"] = False
    cfg.setdefault("loader", {})["n_epochs"] = 5
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = False
    runtime["force_save_epochs"] = [0, 1, 2, 3, 4]
    runtime["use_mlflow_model_logging"] = False

    out = GEN / f"{cfg['experiment']}.yml"
    write_yaml(out, cfg)
    return out, {
        "name": cfg["experiment"],
        "kind": "ft5_from_tx_ep19_txratio_integer_best",
        "config": str(out),
        "resume": str(TX_EP19_RESUME),
        "attention_mode": "faps",
        "target_blocks": "all12",
        "k_magnitude_alpha": 0.0,
        "faps_same_nonzero_weight": 64,
        "faps_same_zero_weight": 1,
        "faps_single_active_weight": 6,
        "score_scale": 1.0 / 64.0,
        "directional_channels_enabled": False,
        "confidence_min_active": 0,
        "expected_atlif": 105,
        "expected_shiftmax": 12,
        "status": "config_generated_not_run",
    }


def make_ft5_txratio_integer_s32() -> tuple[Path, dict[str, Any]]:
    cfg = deepcopy(read_yaml(FT5_BASE))
    cfg["experiment"] = "date11full_all_binary_atlif_faps_all12_nokmag_s32_z1_p3_sc0p03125_nosplit_stdlr_ft_txep19_ft5"
    cfg["note"] = (
        "DATE11 all-binary FAPS FT5 integer TX-ratio follow-up: all 12 "
        "attention blocks, no K magnitude side lane, no x/y split. Effective "
        "all-binary score is (32*same_active + 1*same_zero - 3*single_active) "
        ">> 5, approximating TX ratio 1:0.02:0.10 with a slightly stronger "
        "same-zero term than the 64:1:6 run."
    )
    apply_faps(
        cfg,
        target_blocks=None,
        k_magnitude_alpha=0.0,
        faps_same_nonzero_weight=32.0,
        faps_same_zero_weight=1.0,
        faps_single_active_weight=3.0,
    )
    attn = cfg.setdefault("bsa_attention", {})
    attn["score_scale"] = 1.0 / 32.0
    attn["directional_channels_enabled"] = False
    cfg.setdefault("loader", {})["n_epochs"] = 5
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = False
    runtime["force_save_epochs"] = [0, 1, 2, 3, 4]
    runtime["use_mlflow_model_logging"] = False

    out = GEN / f"{cfg['experiment']}.yml"
    write_yaml(out, cfg)
    return out, {
        "name": cfg["experiment"],
        "kind": "ft5_from_tx_ep19_txratio_integer_s32",
        "config": str(out),
        "resume": str(TX_EP19_RESUME),
        "attention_mode": "faps",
        "target_blocks": "all12",
        "k_magnitude_alpha": 0.0,
        "faps_same_nonzero_weight": 32,
        "faps_same_zero_weight": 1,
        "faps_single_active_weight": 3,
        "score_scale": 1.0 / 32.0,
        "directional_channels_enabled": False,
        "confidence_min_active": 0,
        "expected_atlif": 105,
        "expected_shiftmax": 12,
        "status": "config_generated_not_run",
    }


def make_zero_weight_tune(
    *,
    same_zero_weight: float,
    resume: Path,
    resume_tag: str,
    base_path: Path,
) -> tuple[Path, dict[str, Any]]:
    cfg = deepcopy(read_yaml(base_path))
    weight_tag = str(same_zero_weight).replace(".", "p")
    cfg["experiment"] = f"date11full_all_binary_atlif_faps_all12_nokmag_z{weight_tag}_{resume_tag}_s360"
    cfg["note"] = (
        "DATE11 all-binary FAPS short precision tune: all 12 attention blocks, "
        "no K magnitude side lane, configurable FAPS silence/same-zero weight "
        f"{same_zero_weight}; resume track {resume_tag}."
    )
    apply_faps(cfg, target_blocks=None, k_magnitude_alpha=0.0, faps_same_zero_weight=same_zero_weight)
    cfg.setdefault("loader", {})["n_epochs"] = 1
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 360
    runtime["skip_state_save"] = False
    runtime["force_save_epochs"] = [0]
    runtime["use_mlflow_model_logging"] = False

    out = GEN / f"{cfg['experiment']}.yml"
    write_yaml(out, cfg)
    return out, {
        "name": cfg["experiment"],
        "kind": "short_precision_tune",
        "config": str(out),
        "resume": str(resume),
        "resume_tag": resume_tag,
        "attention_mode": "faps",
        "target_blocks": "all12",
        "k_magnitude_alpha": 0.0,
        "faps_same_zero_weight": same_zero_weight,
        "directional_channels_enabled": True,
        "confidence_min_active": 0,
        "expected_atlif": 105,
        "expected_shiftmax": 12,
        "status": "config_generated_not_run",
    }


def make_integer_value_tune(
    *,
    same_nonzero_weight: int,
    same_zero_weight: int,
    single_active_weight: int,
    score_scale: float,
    directional_channels_enabled: bool,
    resume: Path,
    resume_tag: str,
    base_path: Path,
) -> tuple[Path, dict[str, Any]]:
    cfg = deepcopy(read_yaml(base_path))
    scale_tag = str(score_scale).replace(".", "p")
    split_tag = "splitxy" if directional_channels_enabled else "nosplit"
    tag = f"s{same_nonzero_weight}_z{same_zero_weight}_p{single_active_weight}_sc{scale_tag}_{split_tag}"
    cfg["experiment"] = f"date11full_all_binary_atlif_faps_all12_nokmag_{tag}_{resume_tag}_s360"
    cfg["note"] = (
        "DATE11 all-binary FAPS integer-value short tune: all 12 attention blocks, "
        "no K magnitude side lane. In all-binary ATLIFPSN the effective score is "
        f"{same_nonzero_weight}*same_active + {same_zero_weight}*same_zero - "
        f"{single_active_weight}*single_active, then scaled by {score_scale}; "
        f"directional split enabled={directional_channels_enabled}; resume track {resume_tag}."
    )
    apply_faps(
        cfg,
        target_blocks=None,
        k_magnitude_alpha=0.0,
        faps_same_nonzero_weight=float(same_nonzero_weight),
        faps_same_zero_weight=float(same_zero_weight),
        faps_single_active_weight=float(single_active_weight),
    )
    attn = cfg.setdefault("bsa_attention", {})
    attn["score_scale"] = float(score_scale)
    attn["directional_channels_enabled"] = bool(directional_channels_enabled)
    cfg.setdefault("loader", {})["n_epochs"] = 1
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 360
    runtime["skip_state_save"] = False
    runtime["force_save_epochs"] = [0]
    runtime["use_mlflow_model_logging"] = False

    out = GEN / f"{cfg['experiment']}.yml"
    write_yaml(out, cfg)
    return out, {
        "name": cfg["experiment"],
        "kind": "short_integer_value_tune",
        "config": str(out),
        "resume": str(resume),
        "resume_tag": resume_tag,
        "attention_mode": "faps",
        "target_blocks": "all12",
        "k_magnitude_alpha": 0.0,
        "faps_same_nonzero_weight": same_nonzero_weight,
        "faps_same_zero_weight": same_zero_weight,
        "faps_single_active_weight": single_active_weight,
        "score_scale": score_scale,
        "directional_channels_enabled": directional_channels_enabled,
        "confidence_min_active": 0,
        "expected_atlif": 105,
        "expected_shiftmax": 12,
        "status": "config_generated_not_run",
    }


def main() -> int:
    if not FULL30_BASE.is_file():
        raise FileNotFoundError(f"missing base config: {FULL30_BASE}")
    if not FT5_BASE.is_file():
        raise FileNotFoundError(f"missing base config: {FT5_BASE}")

    rows: list[dict[str, Any]] = []
    for make in (
        make_full30,
        make_ft5,
        make_ft5_all12_nokmag,
        make_ft10_all12_nokmag_slowlr,
        make_ft5_s2_nokmag,
        make_ft5_txratio_integer_best,
        make_ft5_txratio_integer_s32,
    ):
        out, row = make()
        rows.append(row)
        print(out)
    for same_nonzero_weight, same_zero_weight, single_active_weight, score_scale, split in (
        (64, 1, 6, 1.0 / 64.0, False),
        (32, 1, 3, 1.0 / 32.0, False),
        (16, 1, 2, 1.0 / 16.0, False),
    ):
        out, row = make_integer_value_tune(
            same_nonzero_weight=same_nonzero_weight,
            same_zero_weight=same_zero_weight,
            single_active_weight=single_active_weight,
            score_scale=score_scale,
            directional_channels_enabled=split,
            resume=TX_EP19_RESUME,
            resume_tag="txep19",
            base_path=FT5_BASE,
        )
        rows.append(row)
        print(out)
    out, row = make_integer_value_tune(
        same_nonzero_weight=4,
        same_zero_weight=1,
        single_active_weight=4,
        score_scale=1.0,
        directional_channels_enabled=True,
        resume=NB0_RESUME,
        resume_tag="nb0",
        base_path=FULL30_BASE,
    )
    rows.append(row)
    print(out)
    for same_zero_weight in (0.02, 0.10, 0.25):
        out, row = make_zero_weight_tune(
            same_zero_weight=same_zero_weight,
            resume=TX_EP19_RESUME,
            resume_tag="txep19",
            base_path=FT5_BASE,
        )
        rows.append(row)
        print(out)
    MANIFEST.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
