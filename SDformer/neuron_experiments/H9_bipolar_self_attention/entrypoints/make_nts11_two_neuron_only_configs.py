"""Generate NTS11 configs with a two-neuron-only deployment story.

Narrative:
  - Ternary ATLIF-PSN on attention Q/K only (fine-grained signed events, higher expressiveness)
  - Binary official ATLIF-PSN on every other Spiking_neuron (no vanilla PSN left at inference)

Base: NTS10d/09e h60 + freeze1224 training policy. Only the neuron replacement scope changes.
Short-test variants sweep LR / threshold schedule knobs on the nts11b (S23) mainline.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED = EXP_ROOT / "configs" / "generated"
BASE = GENERATED / "nts10d_hw_h60_s23_freeze1224_s1224.yml"

STAGE_BLOCKS = {
    "s2": ("2:0", "2:1", "2:2", "2:3", "2:4", "2:5"),
    "s23": None,  # use blocks from nts10d base yaml
}


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


def apply_two_neuron_only_policy(cfg: dict[str, Any]) -> None:
    """Q/K ternary + all remaining Spiking_neuron -> binary official ATLIF."""
    atlif = cfg.setdefault("atlif_ternary_psn", {})
    atlif["enabled"] = True
    atlif["target"] = "qk"
    atlif["stage_selection"] = "all"
    atlif["output_mode"] = "ternary"
    atlif["threshold_mode"] = "symmetric_bsa_tsn"
    atlif["center_mode"] = "bias"
    atlif["threshold_eta"] = 6.5e-4
    atlif["threshold_lr_scale"] = 50000.0
    atlif["threshold_freeze_after_step"] = 1224
    atlif["max_threshold"] = 2.0
    atlif["target_rate"] = None
    atlif["target_rate_eta"] = 0.0
    atlif["activity_eta"] = 0.0

    atlif["target_groups"] = [
        {
            "name": "all_non_qk_binary_atlif",
            "path_selection": "all_non_qk",
            "output_mode": "binary",
            "threshold_mode": "official_atlif",
            "center_mode": "zero",
            "threshold_eta": 0.0,
            "activity_eta": 0.0,
            "target_rate": None,
            "target_rate_eta": 0.0,
        }
    ]


def apply_hparam_overrides(cfg: dict[str, Any], spec: dict[str, Any]) -> None:
    atlif = cfg.setdefault("atlif_ternary_psn", {})
    if "threshold_eta" in spec:
        atlif["threshold_eta"] = float(spec["threshold_eta"])
    if "threshold_lr_scale" in spec:
        atlif["threshold_lr_scale"] = float(spec["threshold_lr_scale"])
    if "threshold_freeze_after_step" in spec:
        atlif["threshold_freeze_after_step"] = int(spec["threshold_freeze_after_step"])

    groups = cfg.setdefault("optimizer", {}).setdefault("param_groups", {})
    if groups.get("enabled"):
        if "neuron_lr" in spec:
            groups["neuron_lr"] = float(spec["neuron_lr"])
        if "backbone_lr" in spec:
            groups["backbone_lr"] = float(spec["backbone_lr"])
        if "threshold_lr" in spec:
            groups["threshold_lr"] = float(spec["threshold_lr"])

    bsa = cfg.setdefault("bsa_attention", {})
    if "bipolar_mu" in spec:
        bsa["bipolar_mu"] = float(spec["bipolar_mu"])


def blocks_for(scope: str, base_cfg: dict[str, Any]) -> list[str]:
    if scope == "s2":
        return list(STAGE_BLOCKS["s2"])
    if scope == "s23":
        return list(base_cfg.get("bsa_attention", {}).get("target_blocks", []))
    raise ValueError(f"unknown scope: {scope}")


def make_config(base: dict[str, Any], spec: dict[str, Any]) -> Path:
    cfg = deepcopy(base)
    set_runtime(cfg, spec["name"], spec["note"])
    apply_two_neuron_only_policy(cfg)
    apply_hparam_overrides(cfg, spec)

    attn = cfg.setdefault("bsa_attention", {})
    attn["target_blocks"] = blocks_for(str(spec["scope"]), base)

    out = GENERATED / f"{spec['name']}.yml"
    write_yaml(out, cfg)
    return out


def main() -> int:
    base = read_yaml(BASE)
    specs: list[dict[str, Any]] = [
        {
            "name": "nts11a_hw_h60_s2_two_neuron_freeze1224_s1224",
            "scope": "s2",
            "note": (
                "NTS-11a: two-neuron-only deploy story. Q/K ternary ATLIF-PSN; "
                "all other Spiking_neuron -> binary official ATLIF-PSN (no vanilla PSN). "
                "Shiftmax h60 on S2 (6 blocks); freeze1224."
            ),
        },
        {
            "name": "nts11b_hw_h60_s23_two_neuron_freeze1224_s1224",
            "scope": "s23",
            "note": (
                "NTS-11b: same two-neuron policy as 11a but Shiftmax on S2+S3 (8 blocks). "
                "Closest full-coverage successor to current NTS-10d best line."
            ),
        },
        {
            "name": "nts11c_hw_h60_s23_two_neuron_fastlr_s1224",
            "scope": "s23",
            "neuron_lr": 5.0e-5,
            "backbone_lr": 2.0e-6,
            "note": (
                "NTS-11c: nts11b + faster neuron/backbone LR (mirror nts00d). "
                "Two-neuron policy unchanged."
            ),
        },
        {
            "name": "nts11d_hw_h60_s23_two_neuron_slowlr_s1224",
            "scope": "s23",
            "neuron_lr": 2.0e-5,
            "backbone_lr": 5.0e-7,
            "note": (
                "NTS-11d: nts11b + slower neuron/backbone LR (mirror nts00f). "
                "Two-neuron policy unchanged."
            ),
        },
        {
            "name": "nts11e_hw_h60_s23_two_neuron_qkscale25k_s1224",
            "scope": "s23",
            "threshold_lr_scale": 25000.0,
            "note": (
                "NTS-11e: nts11b + halved Q/K threshold_lr_scale (mirror nts08b). "
                "Reduces threshold drift during short adaptation."
            ),
        },
        {
            "name": "nts11f_hw_h60_s23_two_neuron_freeze816_s1224",
            "scope": "s23",
            "threshold_freeze_after_step": 816,
            "note": (
                "NTS-11f: nts11b + earlier threshold freeze at step 816 (mirror nts09a). "
                "Two-neuron policy unchanged."
            ),
        },
        {
            "name": "nts11g_hw_h60_s23_two_neuron_eta0325_s1224",
            "scope": "s23",
            "threshold_eta": 3.25e-4,
            "note": (
                "NTS-11g: nts11b + halved Q/K threshold_eta (mirror nts09c). "
                "Two-neuron policy unchanged."
            ),
        },
    ]
    for spec in specs:
        print(make_config(base, spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())