"""Generate NTS07 no-Kmag H60 configs with softer FFN ATLIF pressure.

NTS06a kept qk ternary activity around 7%, but full training drove the
official binary ATLIF FFN groups below 1% activity by epoch1. NTS07 keeps the
same deployable H60 attention path and only changes the FFN ATLIF groups.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED = EXP_ROOT / "configs" / "generated"
BASE = GENERATED / "nts06a_hw_mu005_mis000_sap000_w720_s360.yml"


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def set_runtime(cfg: dict[str, Any], name: str, note: str) -> None:
    cfg["experiment"] = name
    cfg["note"] = note
    loader = cfg.setdefault("loader", {})
    loader["n_epochs"] = 1
    loader["batch_size"] = 8
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 1224
    runtime["skip_state_save"] = True
    runtime["force_save_epochs"] = [0]
    runtime["use_mlflow_model_logging"] = False


def set_ffn_group_pressure(cfg: dict[str, Any], threshold_eta: float, activity_eta: float) -> None:
    for group in cfg.setdefault("atlif_ternary_psn", {}).get("target_groups", []):
        group["threshold_eta"] = float(threshold_eta)
        group["activity_eta"] = float(activity_eta)


def make_config(base: dict[str, Any], spec: dict[str, Any]) -> Path:
    cfg = deepcopy(base)
    set_runtime(cfg, spec["name"], spec["note"])
    if spec.get("drop_ffn_groups", False):
        cfg.setdefault("atlif_ternary_psn", {})["target_groups"] = []
    else:
        set_ffn_group_pressure(cfg, spec["ffn_threshold_eta"], spec["ffn_activity_eta"])
    out = GENERATED / f"{spec['name']}.yml"
    write_yaml(out, cfg)
    return out


def main() -> int:
    base = read_yaml(BASE)
    specs: list[dict[str, Any]] = [
        {
            "name": "nts07a_hw_h60_ffn_soft_eta2e5_act05_s1224",
            "ffn_threshold_eta": 2.0e-05,
            "ffn_activity_eta": 0.5,
            "note": "NTS-07a: H60 no-Kmag; soften official FFN ATLIF threshold/activity pressure.",
        },
        {
            "name": "nts07b_hw_h60_ffn_update0_act0_s1224",
            "ffn_threshold_eta": 0.0,
            "ffn_activity_eta": 0.0,
            "note": "NTS-07b: H60 no-Kmag; keep FFN ATLIF binary modules but disable manual sparse update/loss.",
        },
        {
            "name": "nts07c_hw_h60_qk_only_noffn_s1224",
            "drop_ffn_groups": True,
            "note": "NTS-07c: H60 no-Kmag; only qk ternary replacement, remove FFN official ATLIF groups.",
        },
        {
            "name": "nts07d_hw_h60_ffn_update8e5_act0_s1224",
            "ffn_threshold_eta": 8.0e-05,
            "ffn_activity_eta": 0.0,
            "note": "NTS-07d: H60 no-Kmag; keep official FFN threshold update but remove activity loss.",
        },
    ]
    for spec in specs:
        print(make_config(base, spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
