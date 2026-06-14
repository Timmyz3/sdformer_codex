"""Generate NTS-11ac for secondary-server full training (non-overlapping with phase-4 scope sweep)."""

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
from make_nts11_two_neuron_only_configs import BASE, apply_two_neuron_only_policy, blocks_for


EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED = EXP_ROOT / "configs" / "generated"


def main() -> int:
    base = read_yaml(BASE)
    paths = build_path_sets()
    cfg = deepcopy(base)
    cfg["experiment"] = "nts11ac_hw_h60_s23_sn2qbin_fastlr_freeze816_warm720_full30"
    cfg["note"] = (
        "NTS-11ac (secondary-server candidate): strict 105-module two-neuron deploy "
        "(Q/K ternary + sn2_q explicit binary + all_non_qk binary). "
        "Training: 11l fast LR + freeze816 + long warmup 720/0.05 aligned with sc_mu. "
        "Does NOT overlap phase-4 scope sweep (no warm720 there)."
    )

    loader = cfg.setdefault("loader", {})
    loader["n_epochs"] = 30
    loader["batch_size"] = 8
    loader["n_workers"] = 8
    loader["persistent_workers"] = True
    loader["prefetch_factor"] = 4
    loader["pin_memory"] = False
    loader["non_blocking"] = True

    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = False
    runtime["force_save_epochs"] = [0, 4, 9, 14, 19, 24, 28, 29]
    runtime["use_mlflow_model_logging"] = False

    optimizer = cfg.setdefault("optimizer", {})
    optimizer["use_amp"] = True
    optimizer["milestones"] = [20, 25]
    groups = optimizer.setdefault("param_groups", {})
    groups["enabled"] = True
    groups["neuron_lr"] = BEST_HPARAMS["neuron_lr"]
    groups["backbone_lr"] = BEST_HPARAMS["backbone_lr"]
    warmup = optimizer.setdefault("lr_warmup", {})
    warmup["enabled"] = True
    warmup["steps"] = 720
    warmup["start_factor"] = 0.05

    apply_two_neuron_only_policy(cfg)
    atlif = cfg.setdefault("atlif_ternary_psn", {})
    atlif["threshold_freeze_after_step"] = BEST_HPARAMS["threshold_freeze_after_step"]
    apply_scope_policy(cfg, "sn2q_binary", paths)

    attn = cfg.setdefault("bsa_attention", {})
    attn["target_blocks"] = blocks_for("s23", base)

    cfg.setdefault("metrics", {})["name"] = ["AEE", "AAE"]
    cfg.setdefault("test", {})["sample"] = 10

    out = GENERATED / "nts11ac_hw_h60_s23_sn2qbin_fastlr_freeze816_warm720_full30.yml"
    write_yaml(out, cfg)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())