"""生成 H34：按官方 ATLIF 范式重做历史短测配置。

H34 不再把所有变体都混称为 ATLIF。配置分两类：
- pure_official：Q/K 也改成 binary official ATLIF，关闭 BSA/Shiftmax；
- hybrid_h9：保留 H9 的 Q/K 三值 + Shiftmax，只把 FFN/downsample/attention aux
  这些二值稀疏层改成 official ATLIF。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = EXP_ROOT / "configs"
BASE_H28B_FULL = CONFIG_DIR / "h28b_diff_lr_newfast_steps360_auto_full_20260520_201852.yml"

DEPTHS = (2, 2, 6, 2)
ATTN_AUX_NAMES = ("sn2_q", "attn_sn", "proj_sn")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def block_prefix(stage: int, block: int) -> str:
    return f"sttmultires_unet.encoders.swin3d.layers.{stage}.swin_blocks.{block}"


def attention_aux_paths() -> list[str]:
    paths: list[str] = []
    for stage, depth in enumerate(DEPTHS):
        for block in range(depth):
            prefix = block_prefix(stage, block)
            for name in ATTN_AUX_NAMES:
                paths.append(f"{prefix}.attn.{name}")
    return paths


def set_runtime(cfg: dict[str, Any], *, name: str, note: str) -> None:
    cfg["experiment"] = name
    cfg["note"] = note
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = True
    runtime["force_save_epochs"] = [0]
    runtime["use_mlflow_model_logging"] = False
    loader = cfg.setdefault("loader", {})
    loader["n_epochs"] = 1
    loader["batch_size"] = 8
    loader["n_workers"] = 8
    loader["persistent_workers"] = True
    loader["prefetch_factor"] = 4
    loader["pin_memory"] = True
    cfg.setdefault("test", {})["sample"] = 10


def set_diff_lr(cfg: dict[str, Any]) -> None:
    opt = cfg.setdefault("optimizer", {})
    opt["use_amp"] = True
    opt["lr"] = 1.0e-6
    opt["param_groups"] = {
        "enabled": True,
        "backbone_lr": 1.0e-6,
        "neuron_lr": 3.0e-5,
        "threshold_lr": 1.0e-5,
        "norm_lr": 1.0e-6,
        "norm_wd": 0.0,
        "threshold_wd": 0.0,
    }


def official_group(name: str, paths: list[str], *, scale: float, activity_eta: float) -> dict[str, Any]:
    return {
        "name": name,
        "output_mode": "binary",
        "center_mode": "zero",
        "threshold_mode": "official_atlif",
        "threshold_init": 0.1,
        "threshold_eta": 0.001,
        "threshold_lr_scale": scale,
        "min_threshold": None,
        "max_threshold": None,
        "target_rate": None,
        "target_rate_eta": 0.0,
        "activity_eta": activity_eta,
        "paths": paths,
    }


def h28_group_paths(base: dict[str, Any]) -> dict[str, list[str]]:
    paths: dict[str, list[str]] = {}
    for group in base["atlif_ternary_psn"].get("target_groups", []) or []:
        paths[str(group["name"])] = [str(path) for path in group.get("paths", [])]
    return paths


def selected_highsop_paths(base: dict[str, Any]) -> list[str]:
    return [path for paths in h28_group_paths(base).values() for path in paths]


def ffn_sn1_paths(base: dict[str, Any]) -> list[str]:
    return [path for path in selected_highsop_paths(base) if ".mlp.sn1" in path]


def stage23_ffn_paths(base: dict[str, Any]) -> list[str]:
    return [
        path
        for path in selected_highsop_paths(base)
        if ".mlp." in path and (".layers.2." in path or ".layers.3." in path)
    ]


def stage02_highsop_paths(base: dict[str, Any]) -> list[str]:
    return [path for path in selected_highsop_paths(base) if ".layers.0." in path or ".layers.2." in path]


def configure_pure_official(
    cfg: dict[str, Any],
    *,
    scope_paths: list[str],
    scope_name: str,
    scale: float,
    activity_eta: float,
) -> None:
    atlif = cfg["atlif_ternary_psn"]
    atlif.update(
        {
            "enabled": True,
            "target": "qk",
            "stage_selection": "all",
            "output_mode": "binary",
            "threshold_mode": "official_atlif",
            "center_mode": "zero",
            "threshold_init": 0.1,
            "threshold_eta": 0.001,
            "threshold_lr_scale": scale,
            "threshold_base_lr": 1.0e-5,
            "min_threshold": None,
            "max_threshold": None,
            "target_rate": None,
            "target_rate_eta": 0.0,
            "activity_eta": activity_eta,
            "trainable": "all",
            "log_interval_steps": 20,
        }
    )
    atlif["target_groups"] = (
        [official_group(f"{scope_name}_official", scope_paths, scale=scale, activity_eta=activity_eta)]
        if scope_paths
        else []
    )
    cfg.setdefault("bsa_attention", {})["enabled"] = False


def configure_hybrid_h9(
    cfg: dict[str, Any],
    *,
    scope_paths: list[str],
    scope_name: str,
    scale: float,
    activity_eta: float,
) -> None:
    atlif = cfg["atlif_ternary_psn"]
    atlif["target_groups"] = [
        official_group(f"{scope_name}_official", scope_paths, scale=scale, activity_eta=activity_eta)
    ]
    atlif["threshold_base_lr"] = 1.0e-5
    atlif["log_interval_steps"] = 20
    cfg.setdefault("bsa_attention", {})["enabled"] = True


def make_name(prefix: str, scope: str, scale: float, activity_eta: float) -> str:
    scale_k = int(scale / 1000)
    act_text = str(activity_eta).replace(".", "p")
    return f"h34_{prefix}_{scope}_s{scale_k}k_act{act_text}"


def main() -> int:
    base = load_yaml(BASE_H28B_FULL)
    set_diff_lr(base)
    scopes = {
        "qkonly": [],
        "highsop": selected_highsop_paths(base),
        "ffn_sn1": ffn_sn1_paths(base),
        "stage23_ffn": stage23_ffn_paths(base),
        "stage02_highsop": stage02_highsop_paths(base),
        "attn_aux": attention_aux_paths(),
        "attn_aux_highsop": attention_aux_paths() + selected_highsop_paths(base),
    }
    hparams = [
        (150000.0, 1.0),
        (150000.0, 2.0),
        (300000.0, 2.0),
        (300000.0, 4.0),
    ]
    specs: list[tuple[str, dict[str, Any]]] = []
    for scope_name in ("qkonly", "highsop", "stage02_highsop"):
        for scale, activity_eta in hparams:
            cfg = deepcopy(base)
            name = make_name("pure_official", scope_name, scale, activity_eta)
            configure_pure_official(
                cfg,
                scope_paths=scopes[scope_name],
                scope_name=scope_name,
                scale=scale,
                activity_eta=activity_eta,
            )
            set_runtime(
                cfg,
                name=name,
                note=f"H34 纯 official ATLIF 重测：scope={scope_name}, scale={scale}, activity_eta={activity_eta}。",
            )
            specs.append((name, cfg))
    for scope_name in ("highsop", "ffn_sn1", "stage23_ffn", "stage02_highsop", "attn_aux", "attn_aux_highsop"):
        for scale, activity_eta in hparams:
            cfg = deepcopy(base)
            name = make_name("hybrid_h9", scope_name, scale, activity_eta)
            configure_hybrid_h9(
                cfg,
                scope_paths=scopes[scope_name],
                scope_name=scope_name,
                scale=scale,
                activity_eta=activity_eta,
            )
            set_runtime(
                cfg,
                name=name,
                note=f"H34 H9 三值注意力 + official ATLIF 重测：scope={scope_name}, scale={scale}, activity_eta={activity_eta}。",
            )
            specs.append((name, cfg))
    for name, cfg in specs:
        path = CONFIG_DIR / f"{name}.yml"
        dump_yaml(path, cfg)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
