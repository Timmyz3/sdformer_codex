"""Generate H68 training and deployment configs for Castling-TTX."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GEN = EXP_ROOT / "configs/generated"
BASE = GEN / "date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8.yml"
TRAIN = GEN / "h68_allbinary_all12_castling_ttx_aux050_s360.yml"
DEPLOY = GEN / "h68_allbinary_all12_castling_ttx_deploy.yml"
MANIFEST = GEN / "h68_castling_ttx_manifest.json"
TTX_CHECKPOINT = (
    EXP_ROOT
    / "results/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid"
    / "checkpoint_epoch2.pth"
)


def dump(path: Path, cfg: dict) -> None:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> int:
    if not BASE.is_file():
        raise FileNotFoundError(BASE)
    if not TTX_CHECKPOINT.is_file():
        raise FileNotFoundError(TTX_CHECKPOINT)
    base = yaml.safe_load(BASE.read_text(encoding="utf-8")) or {}

    train = deepcopy(base)
    train["experiment"] = "h68_allbinary_all12_castling_ttx_aux050_s360"
    train["bsa_attention"].update({
        "binary_motion_xor_alpha": 0.0,
        "castling_matrix_aux_weight": 0.5,
        "castling_matrix_aux_end_step": 360,
    })
    train.setdefault("runtime", {}).update({
        "max_train_steps": 360,
        "skip_state_save": True,
        "force_save_epochs": [0],
        "use_mlflow_model_logging": False,
    })
    train["loader"].update({"n_epochs": 1, "batch_size": 8, "n_workers": 8})
    train["metrics"]["name"] = ["AEE", "AAE"]
    train["test"].update({"sample": 10, "n_valid": 1})
    train["note"] = (
        "H68 Castling-TTX: the deployed all12 H60 path is trained with a parameter-free "
        "binary alpha-XNOR matrix auxiliary whose output blend anneals linearly from "
        "0.5 to 0 by step 360. Evaluation always disables the auxiliary."
    )
    dump(TRAIN, train)

    deploy = deepcopy(train)
    deploy["experiment"] = "h68_allbinary_all12_castling_ttx_deploy"
    deploy["bsa_attention"].update({
        "castling_matrix_aux_weight": 0.0,
        "castling_matrix_aux_end_step": 0,
    })
    deploy["note"] = (
        "H68 deployment config: identical H60 dataflow with the training-only full-matrix "
        "auxiliary explicitly disabled. It loads an H68 checkpoint without extra keys."
    )
    dump(DEPLOY, deploy)

    rows = [{
        "name": train["experiment"],
        "train_config": str(TRAIN),
        "deploy_config": str(DEPLOY),
        "resume": str(TTX_CHECKPOINT),
        "deployed_attention": "12 x H60 dyadic TTX",
        "training_only_auxiliary": "binary alpha-XNOR full matrix, weight 0.5 -> 0",
        "new_parameters": 0,
        "status": "generated_not_run",
    }]
    MANIFEST.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(TRAIN)
    print(DEPLOY)
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
