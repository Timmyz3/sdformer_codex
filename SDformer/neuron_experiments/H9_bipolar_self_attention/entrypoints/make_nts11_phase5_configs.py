"""NTS-11 phase-5: post-11aah recipe+scope+attention sweep (two-neuron line).

After 11aah valid825 (ft ep0 AEE 1.516, −0.027 vs 11aa ep19), sweep:
  - neuron scope (downsample, +ffn_s0/s2, ffn_s2-only)
  - LR recipe (warm720/freeze1224 × std/fast/slow)
  - attention knobs (bipolar_mu, alpha0)
  - resume track (NB0 full30 vs finetune from 11aa ep19 / 11aah ft ep0)

Writes configs + phase5_manifest.json for autopilot.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_nts11_phase4_scope_configs import (
    apply_scope_policy,
    binary_group,
    build_path_sets,
    read_yaml,
    ternary_group,
    write_yaml,
)
from make_nts11_two_neuron_only_configs import apply_hparam_overrides, blocks_for

EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED = EXP_ROOT / "configs" / "generated"
MANIFEST = GENERATED / "nts11_phase5_manifest.json"

AA_FULL = EXP_ROOT / "configs/nts11aa_hw_h60_s23_scope_downsample_ternary_scope_full30_20260612_065413.yml"
NB0 = Path(__file__).resolve().parents[3] / "experiments/baseline_stride_upstream/checkpoint_epoch59.pth"
AA19 = (
    EXP_ROOT
    / "results/nts11aa_hw_h60_s23_scope_downsample_ternary_scope_full30_bs8_20260612_065413_setsid"
    / "checkpoint_epoch19.pth"
)
AAH0 = (
    EXP_ROOT
    / "results/nts11aah_hw_h60_s23_scope_downsample_ternary_warm720_freeze1224_stdlr_ft15_bs8_20260612_194020_setsid"
    / "checkpoint_epoch0.pth"
)

RECIPES: dict[str, dict[str, Any]] = {
    "w720_stdlr": {
        "neuron_lr": 3.0e-5,
        "backbone_lr": 1.0e-6,
        "warmup_steps": 720,
        "warmup_start": 0.05,
        "threshold_freeze_after_step": 1224,
    },
    "w720_fastlr": {
        "neuron_lr": 5.0e-5,
        "backbone_lr": 2.0e-6,
        "warmup_steps": 720,
        "warmup_start": 0.05,
        "threshold_freeze_after_step": 1224,
    },
    "w720_slowlr": {
        "neuron_lr": 2.0e-5,
        "backbone_lr": 5.0e-7,
        "warmup_steps": 720,
        "warmup_start": 0.05,
        "threshold_freeze_after_step": 1224,
    },
}


def apply_composite_scope(cfg: dict[str, Any], policy: str, paths: dict[str, list[str]]) -> None:
    if policy in {
        "downsample_ternary",
        "ffn_s0_ternary",
        "ffn_s2_ternary",
        "ffn_all_ternary",
        "attnaux_ternary",
        "patch_embed_ternary",
    }:
        apply_scope_policy(cfg, policy, paths)
        return

    atlif = cfg.setdefault("atlif_ternary_psn", {})
    atlif.pop("target_paths", None)
    groups: list[dict[str, Any]] = []

    if policy == "downsample_ffn_s2_ternary":
        groups.extend(
            [
                ternary_group("downsample_ternary", paths["downsample"]),
                ternary_group("ffn_s2_ternary", paths["ffn_s2"]),
                binary_group("sn2q_binary", paths=paths["sn2q"]),
            ]
        )
    elif policy == "downsample_ffn_s0_ternary":
        groups.extend(
            [
                ternary_group("downsample_ternary", paths["downsample"]),
                ternary_group("ffn_s0_ternary", paths["ffn_s0"]),
                binary_group("sn2q_binary", paths=paths["sn2q"]),
            ]
        )
    else:
        raise ValueError(f"unknown composite scope: {policy}")

    groups.append(binary_group("all_non_qk_binary_atlif", path_selection="all_non_qk"))
    atlif["target_groups"] = groups


def apply_recipe(cfg: dict[str, Any], recipe: dict[str, Any]) -> None:
    optimizer = cfg.setdefault("optimizer", {})
    groups = optimizer.setdefault("param_groups", {})
    groups["enabled"] = True
    groups["neuron_lr"] = float(recipe["neuron_lr"])
    groups["backbone_lr"] = float(recipe["backbone_lr"])
    warmup = optimizer.setdefault("lr_warmup", {})
    warmup["enabled"] = True
    warmup["steps"] = int(recipe["warmup_steps"])
    warmup["start_factor"] = float(recipe["warmup_start"])
    atlif = cfg.setdefault("atlif_ternary_psn", {})
    atlif["threshold_freeze_after_step"] = int(recipe["threshold_freeze_after_step"])


def make_short_config(spec: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    base = read_yaml(AA_FULL)
    paths = build_path_sets()
    cfg = deepcopy(base)
    cfg["experiment"] = spec["name"]
    cfg["note"] = spec["note"]

    apply_composite_scope(cfg, str(spec["scope_policy"]), paths)
    apply_recipe(cfg, RECIPES[str(spec["recipe"])])
    if spec.get("attn"):
        apply_hparam_overrides(cfg, spec["attn"])

    attn = cfg.setdefault("bsa_attention", {})
    attn["target_blocks"] = blocks_for("s23", base)

    loader = cfg.setdefault("loader", {})
    loader["n_epochs"] = 1
    loader["batch_size"] = 8
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 1224
    runtime["skip_state_save"] = True
    runtime["force_save_epochs"] = [0]
    runtime["use_mlflow_model_logging"] = False
    cfg.setdefault("metrics", {})["name"] = ["AEE", "AAE"]
    cfg.setdefault("test", {})["sample"] = 10

    out = GENERATED / f"{spec['name']}_s1224.yml"
    write_yaml(out, cfg)
    manifest = {
        "name": spec["name"],
        "config": str(out),
        "resume": str(spec["resume"]),
        "track": spec["track"],
        "full_epochs": int(spec["full_epochs"]),
        "scope_policy": spec["scope_policy"],
        "recipe": spec["recipe"],
    }
    return out, manifest


def main() -> int:
    if not AA19.is_file():
        raise FileNotFoundError(f"missing 11aa ep19: {AA19}")
    if not AAH0.is_file():
        raise FileNotFoundError(f"missing 11aah ft ep0: {AAH0}")

    specs: list[dict[str, Any]] = [
        # --- full30 from NB0 (11aah-style recipe, scope sweep) ---
        {
            "name": "nts11aj_hw_h60_s23_ds_w720_stdlr",
            "scope_policy": "downsample_ternary",
            "recipe": "w720_stdlr",
            "resume": str(NB0),
            "track": "full30",
            "full_epochs": 30,
            "note": "Phase-5: 11aa scope + warm720/stdlr/freeze1224 full30 from NB0.",
        },
        {
            "name": "nts11ak_hw_h60_s23_ds_w720_fastlr",
            "scope_policy": "downsample_ternary",
            "recipe": "w720_fastlr",
            "resume": str(NB0),
            "track": "full30",
            "full_epochs": 30,
            "note": "Phase-5: 11aa scope + warm720/fastlr full30 from NB0.",
        },
        {
            "name": "nts11al_hw_h60_s23_ds_w720_slowlr",
            "scope_policy": "downsample_ternary",
            "recipe": "w720_slowlr",
            "resume": str(NB0),
            "track": "full30",
            "full_epochs": 30,
            "note": "Phase-5: 11aa scope + warm720/slowlr full30 from NB0.",
        },
        {
            "name": "nts11am_hw_h60_s23_ds_ffn2_w720_stdlr",
            "scope_policy": "downsample_ffn_s2_ternary",
            "recipe": "w720_stdlr",
            "resume": str(NB0),
            "track": "full30",
            "full_epochs": 30,
            "note": "Phase-5: downsample+S2 FFN ternary + warm720/stdlr full30.",
        },
        {
            "name": "nts11an_hw_h60_s23_ds_ffn0_w720_stdlr",
            "scope_policy": "downsample_ffn_s0_ternary",
            "recipe": "w720_stdlr",
            "resume": str(NB0),
            "track": "full30",
            "full_epochs": 30,
            "note": "Phase-5: downsample+S0 FFN ternary + warm720/stdlr full30.",
        },
        {
            "name": "nts11ao_hw_h60_s23_ffn2_w720_stdlr",
            "scope_policy": "ffn_s2_ternary",
            "recipe": "w720_stdlr",
            "resume": str(NB0),
            "track": "full30",
            "full_epochs": 30,
            "note": "Phase-5: S2 FFN-only ternary + warm720/stdlr full30.",
        },
        # --- attention ablations (downsample + stdlr, NB0) ---
        {
            "name": "nts11at_hw_h60_s23_ds_w720_stdlr_mu03",
            "scope_policy": "downsample_ternary",
            "recipe": "w720_stdlr",
            "resume": str(NB0),
            "track": "full30",
            "full_epochs": 30,
            "attn": {"bipolar_mu": 0.03},
            "note": "Phase-5: downsample + stdlr + bipolar_mu=0.03.",
        },
        {
            "name": "nts11au_hw_h60_s23_ds_w720_stdlr_mu08",
            "scope_policy": "downsample_ternary",
            "recipe": "w720_stdlr",
            "resume": str(NB0),
            "track": "full30",
            "full_epochs": 30,
            "attn": {"bipolar_mu": 0.08},
            "note": "Phase-5: downsample + stdlr + bipolar_mu=0.08.",
        },
        {
            "name": "nts11av_hw_h60_s23_ds_w720_stdlr_a001",
            "scope_policy": "downsample_ternary",
            "recipe": "w720_stdlr",
            "resume": str(NB0),
            "track": "full30",
            "full_epochs": 30,
            "attn": {"alpha0": 0.01},
            "note": "Phase-5: downsample + stdlr + alpha0=0.01.",
        },
        # --- finetune tracks (early-stop friendly: 3ep full promotion) ---
        {
            "name": "nts11ap_hw_h60_s23_ds_w720_stdlr_ftaa19",
            "scope_policy": "downsample_ternary",
            "recipe": "w720_stdlr",
            "resume": str(AA19),
            "track": "finetune",
            "full_epochs": 3,
            "note": "Phase-5: finetune 3ep from 11aa ep19, stdlr (11aah line, early stop).",
        },
        {
            "name": "nts11aq_hw_h60_s23_ds_w720_fastlr_ftaa19",
            "scope_policy": "downsample_ternary",
            "recipe": "w720_fastlr",
            "resume": str(AA19),
            "track": "finetune",
            "full_epochs": 3,
            "note": "Phase-5: finetune 3ep from 11aa ep19, fastlr ablation.",
        },
        {
            "name": "nts11ar_hw_h60_s23_ds_ffn2_w720_stdlr_ftaa19",
            "scope_policy": "downsample_ffn_s2_ternary",
            "recipe": "w720_stdlr",
            "resume": str(AA19),
            "track": "finetune",
            "full_epochs": 3,
            "note": "Phase-5: finetune 3ep from 11aa ep19, expanded scope ds+ffn_s2.",
        },
        {
            "name": "nts11as_hw_h60_s23_ds_w720_stdlr_ftaah0",
            "scope_policy": "downsample_ternary",
            "recipe": "w720_stdlr",
            "resume": str(AAH0),
            "track": "finetune",
            "full_epochs": 2,
            "note": "Phase-5: polish 2ep from 11aah ft ep0 (valid825 best).",
        },
    ]

    manifests: list[dict[str, Any]] = []
    for spec in specs:
        _, manifest = make_short_config(spec)
        manifests.append(manifest)
        print(manifest["config"])

    MANIFEST.write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print(f"# manifest: {MANIFEST} ({len(manifests)} specs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())