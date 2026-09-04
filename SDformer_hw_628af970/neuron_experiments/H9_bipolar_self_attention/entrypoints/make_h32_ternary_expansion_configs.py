"""生成 H32 扩大三值替换配置。

H32 从当前 H28b 全量配置出发，只改变 ATLIF-PSN 的替换范围：
- Q/K 继续保持三值；
- 原 H28b 的 FFN/downsample 仍可保持 binary，或按方案扩大为 ternary；
- attention 内部的 sn2_q/attn_sn/proj_sn 可单独扩大为 ternary。

脚本同时生成 rapid 配置和 full 配置：
- rapid: 1 epoch + max_train_steps=360，用于短测筛选；
- full: 30 epoch，用于选中后全量续训。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

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


def ternary_group(name: str, paths: list[str], *, target_rate: float = 0.035) -> dict[str, Any]:
    return {
        "name": name,
        "output_mode": "ternary",
        "center_mode": "bias",
        "threshold_init": 1.0,
        "threshold_eta": 0.001,
        "threshold_lr_scale": 50000.0,
        "min_threshold": 0.001,
        "max_threshold": 1.8,
        "negative_threshold_scale": 1.0,
        "threshold_mode": "symmetric_target_rate",
        "target_rate": target_rate,
        "target_rate_eta": 0.08,
        "activity_eta": 1.0,
        "paths": paths,
    }


def split_existing_groups(
    cfg: dict[str, Any],
    should_ternary: Callable[[str], bool],
    *,
    suffix: str,
    target_rate: float = 0.035,
) -> None:
    atlif = cfg.setdefault("atlif_ternary_psn", {})
    new_groups: list[dict[str, Any]] = []
    for group in atlif.get("target_groups", []) or []:
        paths = [str(path) for path in group.get("paths", [])]
        ternary_paths = [path for path in paths if should_ternary(path)]
        binary_paths = [path for path in paths if path not in set(ternary_paths)]
        if binary_paths:
            binary_group = deepcopy(group)
            binary_group["paths"] = binary_paths
            new_groups.append(binary_group)
        if ternary_paths:
            new_groups.append(
                ternary_group(
                    f"{group.get('name', 'group')}_{suffix}",
                    ternary_paths,
                    target_rate=target_rate,
                )
            )
    atlif["target_groups"] = new_groups


def is_mlp_sn1(path: str) -> bool:
    return ".mlp.sn1" in path


def is_stage23_mlp(path: str) -> bool:
    return ".mlp." in path and (".layers.2." in path or ".layers.3." in path)


def is_selected_ffn_or_down(path: str) -> bool:
    return ".mlp." in path or ".downsample.sn" in path


def configure_runtime(cfg: dict[str, Any], *, name: str, rapid: bool, note: str) -> None:
    cfg["experiment"] = name
    cfg["note"] = note
    runtime = cfg.setdefault("runtime", {})
    runtime["skip_state_save"] = True
    runtime["use_mlflow_model_logging"] = False
    if rapid:
        runtime["max_train_steps"] = 360
        runtime["force_save_epochs"] = []
        cfg.setdefault("loader", {})["n_epochs"] = 1
        cfg.setdefault("test", {})["sample"] = 10
    else:
        runtime["max_train_steps"] = 0
        runtime["force_save_epochs"] = [9, 19, 29]
        cfg.setdefault("loader", {})["n_epochs"] = 30
    loader = cfg.setdefault("loader", {})
    loader["batch_size"] = 8
    loader["n_workers"] = 8
    loader["persistent_workers"] = True
    loader["prefetch_factor"] = 4
    loader["pin_memory"] = True


def build_specs() -> list[tuple[str, str, Callable[[dict[str, Any]], None]]]:
    def h32a(cfg: dict[str, Any]) -> None:
        cfg["atlif_ternary_psn"].setdefault("target_groups", []).append(
            ternary_group("attn_aux_all_blocks_ternary", attention_aux_paths(), target_rate=0.035)
        )

    def h32b(cfg: dict[str, Any]) -> None:
        split_existing_groups(cfg, is_mlp_sn1, suffix="sn1_ternary", target_rate=0.035)

    def h32c(cfg: dict[str, Any]) -> None:
        split_existing_groups(cfg, is_stage23_mlp, suffix="stage23_ternary", target_rate=0.035)

    def h32d(cfg: dict[str, Any]) -> None:
        h32a(cfg)
        split_existing_groups(cfg, is_mlp_sn1, suffix="sn1_ternary", target_rate=0.035)

    def h32e(cfg: dict[str, Any]) -> None:
        split_existing_groups(cfg, is_selected_ffn_or_down, suffix="all_selected_ternary", target_rate=0.035)

    return [
        (
            "h32a_expand_attn_aux_ternary",
            "H32a：H28b 基础上，Q/K 已三值，再把 attention 内 sn2_q/attn_sn/proj_sn 扩大为三值；FFN/downsample 仍为 binary ATLIF。",
            h32a,
        ),
        (
            "h32b_expand_ffn_sn1_selected_ternary",
            "H32b：H28b 基础上，仅把已选 FFN 的升维侧 sn1 扩大为三值，sn2 和 downsample 保持 binary，验证表达增强和 SOPs 的折中。",
            h32b,
        ),
        (
            "h32c_expand_stage23_ffn_selected_ternary",
            "H32c：H28b 基础上，把 stage2/stage3 已选 FFN 的 sn1/sn2 扩大为三值，观察后部高语义层替换是否更稳。",
            h32c,
        ),
        (
            "h32d_expand_attn_aux_ffn_sn1_ternary",
            "H32d：H32a + H32b，同时扩大 attention 内部脉冲和 FFN 升维侧三值，是更激进但仍分层的方案。",
            h32d,
        ),
        (
            "h32e_expand_all_selected_ffn_down_ternary",
            "H32e：把 H28b 已选择的 FFN/downsample 全部从 binary ATLIF 改为 ternary ATLIF，作为扩大三值替换上限测试。",
            h32e,
        ),
    ]


def main() -> int:
    base = load_yaml(BASE_H28B_FULL)
    for base_name, note, mutate in build_specs():
        for mode, rapid in (("rapid", True), ("full", False)):
            cfg = deepcopy(base)
            name = f"{base_name}_{mode}"
            mutate(cfg)
            configure_runtime(cfg, name=name, rapid=rapid, note=f"{note}\n\n生成来源：{BASE_H28B_FULL.name}。")
            path = CONFIG_DIR / f"{name}.yml"
            dump_yaml(path, cfg)
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
