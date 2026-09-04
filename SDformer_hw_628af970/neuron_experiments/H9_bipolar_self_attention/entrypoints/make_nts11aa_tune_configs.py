"""Generate NTS-11aa accuracy-recovery finetune configs.

Keeps the Phase-4 winning neuron scope (Q/K + downsample ternary, sn2_q + all_non_qk
binary). Adjusts training recipe toward the 10d/09e line that reached AEE ~1.48:

  - longer LR warmup (720 steps, start_factor 0.05) aligned with sc_mu warmup
  - threshold freeze at 1224 (not 816)
  - optional standard vs fast param-group LRs

Finetune starts from 11aa valid825-best checkpoint (epoch19 by default).
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_nts11_phase4_scope_configs import (
    BEST_HPARAMS,
    apply_scope_policy,
    build_path_sets,
    read_yaml,
    write_yaml,
)
from make_nts11_two_neuron_only_configs import blocks_for

EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED = EXP_ROOT / "configs" / "generated"
AA_FULL = EXP_ROOT / "configs" / "nts11aa_hw_h60_s23_scope_downsample_ternary_scope_full30_20260612_065413.yml"
AA_RUN = (
    EXP_ROOT
    / "results"
    / "nts11aa_hw_h60_s23_scope_downsample_ternary_scope_full30_bs8_20260612_065413_setsid"
)
AA_BEST_CKPT = AA_RUN / "checkpoint_epoch19.pth"

STDLR = {"neuron_lr": 3.0e-5, "backbone_lr": 1.0e-6}
FASTLR = {"neuron_lr": 5.0e-5, "backbone_lr": 2.0e-6}


def make_tune_config(
    name: str,
    note: str,
    *,
    lr: dict[str, float],
    n_epochs: int = 15,
) -> Path:
    base = read_yaml(AA_FULL)
    paths = build_path_sets()
    cfg = deepcopy(base)
    cfg["experiment"] = name
    cfg["note"] = note

    loader = cfg.setdefault("loader", {})
    loader["n_epochs"] = int(n_epochs)

    optimizer = cfg.setdefault("optimizer", {})
    optimizer["milestones"] = [10, 13]
    groups = optimizer.setdefault("param_groups", {})
    groups["enabled"] = True
    groups["neuron_lr"] = float(lr["neuron_lr"])
    groups["backbone_lr"] = float(lr["backbone_lr"])
    warmup = optimizer.setdefault("lr_warmup", {})
    warmup["enabled"] = True
    warmup["steps"] = 720
    warmup["start_factor"] = 0.05

    atlif = cfg.setdefault("atlif_ternary_psn", {})
    atlif["threshold_freeze_after_step"] = 1224

    apply_scope_policy(cfg, "downsample_ternary", paths)
    attn = cfg.setdefault("bsa_attention", {})
    attn["target_blocks"] = blocks_for("s23", base)

    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = False
    runtime["force_save_epochs"] = [0, 4, 9, 14, 19, 24, 28, 29][: max(3, min(n_epochs, 8))]
    if n_epochs >= 15:
        runtime["force_save_epochs"] = sorted({0, 4, 9, 14, min(19, n_epochs - 1), min(24, n_epochs - 1), n_epochs - 1})

    cfg.setdefault("metrics", {})["name"] = ["AEE", "AAE"]
    cfg.setdefault("test", {})["sample"] = 10

    out = GENERATED / f"{name}.yml"
    write_yaml(out, cfg)
    return out


def main() -> int:
    if not AA_BEST_CKPT.is_file():
        raise FileNotFoundError(f"missing 11aa best checkpoint: {AA_BEST_CKPT}")

    specs = [
        (
            "nts11aah_hw_h60_s23_scope_downsample_ternary_warm720_freeze1224_stdlr_ft15",
            (
                "NTS-11aah: 11aa scope + finetune 15ep from ep19. "
                "warm720/0.05 + freeze1224 + 10d-style LR (neuron 3e-5, backbone 1e-6)."
            ),
            STDLR,
        ),
        (
            "nts11aai_hw_h60_s23_scope_downsample_ternary_warm720_freeze1224_fastlr_ft15",
            (
                "NTS-11aai: 11aa scope + finetune 15ep from ep19. "
                "warm720/0.05 + freeze1224 + phase-4 fast LR (5e-5 / 2e-6) for ablation."
            ),
            FASTLR,
        ),
    ]
    print(f"# resume checkpoint: {AA_BEST_CKPT}")
    for name, note, lr in specs:
        print(make_tune_config(name, note, lr=lr))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())