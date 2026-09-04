"""NTS-11 phase-3: 11j vanilla-decoder line + fast LR + long warmup variants."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_nts11_phase2_configs import apply_target_group_policy
from make_nts11_two_neuron_only_configs import (
    BASE,
    apply_hparam_overrides,
    apply_two_neuron_only_policy,
    blocks_for,
    read_yaml,
    set_runtime,
    write_yaml,
)


def apply_warmup_overrides(cfg: dict[str, Any], spec: dict[str, Any]) -> None:
    if "warmup_steps" not in spec and "warmup_start_factor" not in spec:
        return
    warmup = cfg.setdefault("optimizer", {}).setdefault("lr_warmup", {})
    warmup["enabled"] = True
    if "warmup_steps" in spec:
        warmup["steps"] = int(spec["warmup_steps"])
    if "warmup_start_factor" in spec:
        warmup["start_factor"] = float(spec["warmup_start_factor"])


def make_config(base: dict[str, Any], spec: dict[str, Any]) -> Path:
    cfg = deepcopy(base)
    set_runtime(cfg, spec["name"], spec["note"])
    apply_two_neuron_only_policy(cfg)
    apply_target_group_policy(cfg, spec)
    apply_hparam_overrides(cfg, spec)
    apply_warmup_overrides(cfg, spec)

    attn = cfg.setdefault("bsa_attention", {})
    attn["target_blocks"] = blocks_for(str(spec.get("scope", "s23")), base)

    out = Path(__file__).resolve().parents[1] / "configs" / "generated" / f"{spec['name']}.yml"
    write_yaml(out, cfg)
    return out


def main() -> int:
    base = read_yaml(BASE)
    vanilla_decoder = {"target_group_policy": "vanilla_decoder_head"}
    specs: list[dict[str, Any]] = [
        {
            "name": "nts11n_hw_h60_s23_vdec_fastlr_s1224",
            "note": (
                "NTS-11n: 11j vanilla decoder + fast neuron/backbone LR "
                "(neuron 5e-5, backbone 2e-6). Default warmup 200/0.1."
            ),
            **vanilla_decoder,
            "neuron_lr": 5.0e-5,
            "backbone_lr": 2.0e-6,
        },
        {
            "name": "nts11o_hw_h60_s23_vdec_fastlr_warm720_s1224",
            "note": (
                "NTS-11o: 11n + long LR warmup aligned with sc_mu (720 steps, start 0.05). "
                "Direction-first training for encoder ATLIF cold start."
            ),
            **vanilla_decoder,
            "neuron_lr": 5.0e-5,
            "backbone_lr": 2.0e-6,
            "warmup_steps": 720,
            "warmup_start_factor": 0.05,
        },
        {
            "name": "nts11p_hw_h60_s23_vdec_fastlr_warm720_freeze816_s1224",
            "note": (
                "NTS-11p: 11o + threshold freeze816. Long warmup then early threshold lock."
            ),
            **vanilla_decoder,
            "neuron_lr": 5.0e-5,
            "backbone_lr": 2.0e-6,
            "warmup_steps": 720,
            "warmup_start_factor": 0.05,
            "threshold_freeze_after_step": 816,
        },
    ]
    for spec in specs:
        print(make_config(base, spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())