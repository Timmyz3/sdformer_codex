"""Generate NTS11 deployment-quantization eval configs.

The generated configs are inference-only ablations. All hardware quantization
switches default off in model code, so old experiments are unaffected.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = EXP_ROOT / "configs/generated/nts11bj_u12_ds_w720_stdlr_ftbd19_ft5.yml"
OUT_DIR = EXP_ROOT / "configs/generated"


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def make_config(name: str, overrides: dict[str, Any]) -> Path:
    cfg = deepcopy(read_yaml(BASE_CONFIG))
    cfg["experiment"] = name
    cfg["note"] = f"NTS11bj ep2 deployment quant eval: {name}"
    attn = cfg.setdefault("bsa_attention", {})
    attn.update(overrides)
    cfg.setdefault("loader", {})["batch_size"] = 1
    cfg.setdefault("loader", {})["n_workers"] = 0
    cfg.setdefault("loader", {})["persistent_workers"] = False
    cfg.setdefault("runtime", {})["use_mlflow_model_logging"] = False
    out = OUT_DIR / f"{name}.yml"
    write_yaml(out, cfg)
    return out


def main() -> int:
    if not BASE_CONFIG.is_file():
        raise FileNotFoundError(BASE_CONFIG)

    specs: dict[str, dict[str, Any]] = {
        "nts11bj_deploy_float_ref": {
            "hardware_quant_enabled": False,
        },
        "nts11bj_deploy_score_int8": {
            "hardware_quant_enabled": True,
            "hardware_score_step": 1.0 / 128.0,
            "hardware_score_min": -2.0,
            "hardware_score_max": 2.0,
            "hardware_gate_step": 0.0,
        },
        "nts11bj_deploy_score_int8_mu_pow2": {
            "hardware_quant_enabled": True,
            "hardware_mu_pow2_shift": 4,
            "hardware_score_step": 1.0 / 128.0,
            "hardware_score_min": -2.0,
            "hardware_score_max": 2.0,
            "hardware_gate_step": 0.0,
        },
        "nts11bj_deploy_score_int8_mu_pow2_gate_int8": {
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
    for name, overrides in specs.items():
        print(make_config(name, overrides))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
