"""Generate standard full30 configs for H67 and H68."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GEN = EXP_ROOT / "configs/generated"
BASE = GEN / "date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8.yml"
H67 = GEN / "h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30.yml"
H68 = GEN / "h68_allbinary_all12_castling_ttx_aux050_toep20_w720_fastlr_full30.yml"
H68_DEPLOY = GEN / "h68_allbinary_all12_castling_ttx_deploy_full30.yml"
MANIFEST = GEN / "h67_h68_full30_manifest.json"
TTX_CHECKPOINT = (
    EXP_ROOT
    / "results/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid"
    / "checkpoint_epoch2.pth"
)
SAVE_EPOCHS = [0, 4, 9, 14, 19, 24, 28, 29]
STEPS_PER_EPOCH = 918


def full30(base: dict, experiment: str) -> dict:
    cfg = deepcopy(base)
    cfg["experiment"] = experiment
    cfg["optimizer"].update({
        "lr": 2.0e-5,
        "milestones": [20, 25],
        "use_amp": True,
    })
    cfg["optimizer"]["param_groups"].update({
        "backbone_lr": 2.0e-6,
        "neuron_lr": 5.0e-5,
        "norm_lr": 1.0e-6,
        "threshold_lr": 5.0e-6,
    })
    cfg["optimizer"]["lr_warmup"].update({"enabled": True, "steps": 720, "start_factor": 0.05})
    cfg["loader"].update({
        "n_epochs": 30,
        "batch_size": 8,
        "n_workers": 8,
        "persistent_workers": True,
        "pin_memory": False,
        "prefetch_factor": 4,
        "non_blocking": True,
    })
    cfg.setdefault("runtime", {}).update({
        "max_train_steps": 0,
        "skip_state_save": False,
        "save_only_force_epochs": True,
        "state_save_epochs": [19, 24, 29],
        "force_save_epochs": list(SAVE_EPOCHS),
        "use_mlflow_model_logging": False,
    })
    cfg["metrics"]["name"] = ["AEE", "AAE"]
    cfg["test"].update({"sample": 10, "n_valid": 1})
    return cfg


def dump(path: Path, cfg: dict) -> None:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> int:
    if not BASE.is_file():
        raise FileNotFoundError(BASE)
    if not TTX_CHECKPOINT.is_file():
        raise FileNotFoundError(TTX_CHECKPOINT)
    base = yaml.safe_load(BASE.read_text(encoding="utf-8")) or {}

    h67 = full30(base, H67.stem)
    h67["bsa_attention"].update({
        "binary_motion_xor_alpha": 0.25,
        "castling_matrix_aux_weight": 0.0,
        "castling_matrix_aux_end_step": 0,
    })
    h67["note"] = (
        "H67 standard full30 from TTX epoch2. All12 H60 score adds one fixed "
        "same-position temporal K XOR-popcount term with dyadic weight 1/4."
    )
    dump(H67, h67)

    h68 = full30(base, H68.stem)
    h68["bsa_attention"].update({
        "binary_motion_xor_alpha": 0.0,
        "castling_matrix_aux_weight": 0.5,
        "castling_matrix_aux_end_step": 20 * STEPS_PER_EPOCH,
    })
    h68["note"] = (
        "H68 standard full30 from TTX epoch2. Training-only binary alpha-XNOR "
        "matrix output blend anneals from 0.5 to zero by epoch20; epochs20-29 "
        "train the deployed H60 path alone. No auxiliary parameters."
    )
    dump(H68, h68)

    deploy = deepcopy(h68)
    deploy["experiment"] = H68_DEPLOY.stem
    deploy["bsa_attention"].update({
        "castling_matrix_aux_weight": 0.0,
        "castling_matrix_aux_end_step": 0,
    })
    deploy["note"] = "H68 full30 deployment/evaluation config with matrix auxiliary explicitly disabled."
    dump(H68_DEPLOY, deploy)

    rows = [
        {
            "order": 1,
            "name": H67.stem,
            "train_config": str(H67),
            "eval_config": str(H67),
            "resume": str(TTX_CHECKPOINT),
            "epochs": 30,
            "status": "generated_not_run",
        },
        {
            "order": 2,
            "name": H68.stem,
            "train_config": str(H68),
            "eval_config": str(H68_DEPLOY),
            "resume": str(TTX_CHECKPOINT),
            "epochs": 30,
            "status": "generated_not_run",
        },
    ]
    MANIFEST.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    for path in (H67, H68, H68_DEPLOY, MANIFEST):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
