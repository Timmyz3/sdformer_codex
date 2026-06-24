"""Generate final DATE11 all-binary mainline follow-up configs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GEN = EXP_ROOT / "configs/generated"


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def make_original_ft5() -> Path:
    base = read_yaml(GEN / "date11full_all_binary_atlif_original_w720_fastlr_full30.yml")
    template = read_yaml(GEN / "date11full_all_binary_atlif_nts_stdlr_ft_ep29_ft5.yml")

    cfg = deepcopy(base)
    cfg["experiment"] = "date11full_all_binary_atlif_original_stdlr_ft_ep29_ft5"
    cfg["optimizer"] = deepcopy(template["optimizer"])
    cfg["loader"]["n_epochs"] = 5
    cfg["runtime"]["force_save_epochs"] = [0, 1, 2, 3, 4]
    cfg["runtime"]["use_mlflow_model_logging"] = False
    cfg["note"] = (
        "DATE11 all-binary original-attention FT5 control: starts from "
        "all_binary+original ep29; keeps 105 binary ATLIF and no H60 attention. "
        "Purpose: test whether short FT alone recovers accuracy without NTS/H60."
    )
    out = GEN / f"{cfg['experiment']}.yml"
    write_yaml(out, cfg)
    return out


def _apply_ft5_recipe(cfg: dict[str, Any], experiment: str, note: str) -> dict[str, Any]:
    template = read_yaml(GEN / "date11full_all_binary_atlif_nts_stdlr_ft_ep29_ft5.yml")
    cfg = deepcopy(cfg)
    cfg["experiment"] = experiment
    cfg["optimizer"] = deepcopy(template["optimizer"])
    cfg["loader"]["n_epochs"] = 5
    cfg["runtime"]["force_save_epochs"] = [0, 1, 2, 3, 4]
    cfg["runtime"]["use_mlflow_model_logging"] = False
    cfg["note"] = note
    return cfg


def make_tx_ft5() -> Path:
    base = read_yaml(GEN / "date11full_all_binary_atlif_tx_w720_fastlr_full30.yml")
    cfg = _apply_ft5_recipe(
        base,
        "date11full_all_binary_atlif_tx_stdlr_ft_ep19_ft5",
        (
            "DATE11 all-binary TX FT5 control: starts from all_binary+TX ep19; "
            "keeps 105 binary ATLIF and 12 TX attention modules. Purpose: test "
            "whether simpler TX attention can match the all-binary NTS/H60 mainline "
            "after the same short fine-tune."
        ),
    )
    out = GEN / f"{cfg['experiment']}.yml"
    write_yaml(out, cfg)
    return out


def make_nts_ep19_ft5() -> Path:
    base = read_yaml(GEN / "date11full_all_binary_atlif_nts_w720_fastlr_full30.yml")
    cfg = _apply_ft5_recipe(
        base,
        "date11full_all_binary_atlif_nts_stdlr_ft_ep19_ft5",
        (
            "DATE11 all-binary NTS/H60 FT5 seed check: starts from all_binary+NTS/H60 "
            "ep19 instead of ep29; keeps 105 binary ATLIF and 12 H60 attention modules. "
            "Purpose: check whether the ep29 FT result is checkpoint cherry-picking."
        ),
    )
    out = GEN / f"{cfg['experiment']}.yml"
    write_yaml(out, cfg)
    return out


def make_binary_nts_deploy_quant() -> list[Path]:
    base = read_yaml(GEN / "date11full_all_binary_atlif_nts_stdlr_ft_ep29_ft5.yml")
    specs: dict[str, dict[str, Any]] = {
        "date11_binary_nts_ft_ep29_deploy_float_ref": {
            "hardware_quant_enabled": False,
        },
        "date11_binary_nts_ft_ep29_deploy_score_int8": {
            "hardware_quant_enabled": True,
            "hardware_score_step": 1.0 / 128.0,
            "hardware_score_min": -2.0,
            "hardware_score_max": 2.0,
            "hardware_gate_step": 0.0,
        },
        "date11_binary_nts_ft_ep29_deploy_score_int8_mu_pow2": {
            "hardware_quant_enabled": True,
            "hardware_mu_pow2_shift": 4,
            "hardware_score_step": 1.0 / 128.0,
            "hardware_score_min": -2.0,
            "hardware_score_max": 2.0,
            "hardware_gate_step": 0.0,
        },
        "date11_binary_nts_ft_ep29_deploy_score_int8_mu_pow2_gate_int8": {
            "hardware_quant_enabled": True,
            "hardware_mu_pow2_shift": 4,
            "hardware_score_step": 1.0 / 128.0,
            "hardware_score_min": -2.0,
            "hardware_score_max": 2.0,
            "hardware_gate_step": 1.0 / 128.0,
            "hardware_gate_min": 0.0,
            "hardware_gate_max": 2.0,
        },
    }

    paths: list[Path] = []
    for name, overrides in specs.items():
        cfg = deepcopy(base)
        cfg["experiment"] = name
        cfg["note"] = f"DATE11 all-binary NTS/H60 FT ep29 deployment quant eval: {name}"
        cfg.setdefault("loader", {})["batch_size"] = 1
        cfg.setdefault("loader", {})["n_workers"] = 0
        cfg.setdefault("loader", {})["persistent_workers"] = False
        cfg.setdefault("runtime", {})["use_mlflow_model_logging"] = False
        cfg.setdefault("bsa_attention", {}).update(overrides)
        out = GEN / f"{name}.yml"
        write_yaml(out, cfg)
        paths.append(out)
    return paths


def main() -> int:
    paths = [
        make_original_ft5(),
        make_tx_ft5(),
        make_nts_ep19_ft5(),
        *make_binary_nts_deploy_quant(),
    ]
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
