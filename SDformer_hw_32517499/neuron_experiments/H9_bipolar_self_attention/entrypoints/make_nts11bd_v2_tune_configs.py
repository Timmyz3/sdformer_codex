"""Generate 11bd-v2 (u12_ds) finetune configs from valid825-best checkpoint.

Base model: unified h60 on all 12 blocks, downsample-only ternary scope (same as 11aa).
Resume: nts11bd_u12_ds_w720_fastlr full30 ep19.

Sweep axes (grounded in 11aah / 11aq / 11aqa evidence):
  - LR recipe: stdlr | fastlr | slowlr
  - finetune length: 3ep (11aq early-stop) | 5ep (11aqa polish)
  - warmup: warm720 | nowarm (11aqa-style)
  - optional: short warmup360 for 5ep fastlr
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_nts11bd_unified_attn_sweep_configs import RECIPES, apply_recipe, read_yaml, write_yaml

EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED = EXP_ROOT / "configs" / "generated"
MANIFEST = GENERATED / "nts11bd_v2_tune_manifest.json"

V2_FULL = EXP_ROOT / "configs/nts11bd_u12_ds_w720_fastlr_full30_20260613_223042.yml"
V2_RUN = (
    EXP_ROOT
    / "results"
    / "nts11bd_u12_ds_w720_fastlr_full30_20260613_223042_bs8_20260613_223042_setsid"
)
V2_BEST_CKPT = V2_RUN / "checkpoint_epoch19.pth"

WARM720 = {"enabled": True, "steps": 720, "start_factor": 0.05}
NOWARM = {"enabled": False, "steps": 720, "start_factor": 0.05}
WARM360 = {"enabled": True, "steps": 360, "start_factor": 0.1}


def make_ft_config(
    name: str,
    note: str,
    *,
    recipe_key: str,
    n_epochs: int,
    warmup: dict[str, Any],
    freeze_step: int = 1224,
    extra_recipe: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    if not V2_FULL.is_file():
        raise FileNotFoundError(V2_FULL)
    cfg = deepcopy(read_yaml(V2_FULL))
    cfg["experiment"] = name
    cfg["note"] = note

    recipe = dict(RECIPES[recipe_key])
    if extra_recipe:
        recipe.update(extra_recipe)
    recipe["threshold_freeze_after_step"] = int(freeze_step)
    apply_recipe(cfg, recipe)

    warmup_cfg = cfg.setdefault("optimizer", {}).setdefault("lr_warmup", {})
    warmup_cfg["enabled"] = bool(warmup["enabled"])
    warmup_cfg["steps"] = int(warmup["steps"])
    warmup_cfg["start_factor"] = float(warmup["start_factor"])

    loader = cfg.setdefault("loader", {})
    loader["n_epochs"] = int(n_epochs)

    optimizer = cfg.setdefault("optimizer", {})
    optimizer["milestones"] = [max(1, n_epochs - 1)] if n_epochs <= 5 else [10, 13]

    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = False
    if n_epochs <= 3:
        runtime["force_save_epochs"] = [0, 1, 2]
    else:
        runtime["force_save_epochs"] = [0, 2, 4] if n_epochs == 5 else [0, 4, 9, 14]

    cfg.setdefault("metrics", {})["name"] = ["AEE", "AAE"]
    cfg.setdefault("test", {})["sample"] = 10

    out = GENERATED / f"{name}.yml"
    write_yaml(out, cfg)
    meta = {
        "name": name,
        "config": str(out),
        "resume": str(V2_BEST_CKPT),
        "recipe": recipe_key,
        "n_epochs": n_epochs,
        "warmup": warmup,
        "freeze_step": freeze_step,
        "track": "finetune",
    }
    return out, meta


def main() -> int:
    if not V2_BEST_CKPT.is_file():
        raise FileNotFoundError(f"missing 11bd-v2 best checkpoint: {V2_BEST_CKPT}")

    specs: list[tuple[str, str, str, int, dict[str, Any], int, dict[str, Any] | None]] = [
        # --- 3ep: mimic 11aq early-stop lane ---
        (
            "nts11be_u12_ds_w720_stdlr_ftbd19_ft3",
            "11bd-v2 finetune 3ep: stdlr + warm720/freeze1224 (11aah-style, expect ep0–1 best).",
            "w720_stdlr",
            3,
            WARM720,
            1224,
            None,
        ),
        (
            "nts11bf_u12_ds_w720_fastlr_ftbd19_ft3",
            "11bd-v2 finetune 3ep: fastlr + warm720/freeze1224 (11aq winner recipe).",
            "w720_fastlr",
            3,
            WARM720,
            1224,
            None,
        ),
        (
            "nts11bg_u12_ds_w720_slowlr_ftbd19_ft3",
            "11bd-v2 finetune 3ep: slowlr + warm720/freeze1224 (conservative polish).",
            "w720_slowlr",
            3,
            WARM720,
            1224,
            None,
        ),
        (
            "nts11bh_u12_ds_w720_fastlr_nowarm_ftbd19_ft3",
            "11bd-v2 finetune 3ep: fastlr + no warmup + freeze1224 (11aqa late-stage style).",
            "w720_fastlr",
            3,
            NOWARM,
            1224,
            None,
        ),
        # --- 5ep: mimic 11aqa polish lane ---
        (
            "nts11bi_u12_ds_w720_fastlr_ftbd19_ft5",
            "11bd-v2 finetune 5ep: fastlr + warm720/freeze1224 (11aqa path, watch ep5 peak).",
            "w720_fastlr",
            5,
            WARM720,
            1224,
            None,
        ),
        (
            "nts11bj_u12_ds_w720_stdlr_ftbd19_ft5",
            "11bd-v2 finetune 5ep: stdlr + warm720/freeze1224 (stdlr longer — often ep0 best).",
            "w720_stdlr",
            5,
            WARM720,
            1224,
            None,
        ),
        (
            "nts11bk_u12_ds_w720_slowlr_ftbd19_ft5",
            "11bd-v2 finetune 5ep: slowlr + warm720/freeze1224.",
            "w720_slowlr",
            5,
            WARM720,
            1224,
            None,
        ),
        (
            "nts11bl_u12_ds_w720_fastlr_w360_ftbd19_ft5",
            "11bd-v2 finetune 5ep: fastlr + short warmup360/0.1 + freeze1224.",
            "w720_fastlr",
            5,
            WARM360,
            1224,
            None,
        ),
    ]

    entries: list[dict[str, Any]] = []
    print(f"# resume: {V2_BEST_CKPT}")
    for name, note, recipe_key, n_epochs, warmup, freeze_step, extra in specs:
        path, meta = make_ft_config(
            name,
            note,
            recipe_key=recipe_key,
            n_epochs=n_epochs,
            warmup=warmup,
            freeze_step=freeze_step,
            extra_recipe=extra,
        )
        entries.append(meta)
        print(path)

    MANIFEST.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"# manifest: {MANIFEST} ({len(entries)} configs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
