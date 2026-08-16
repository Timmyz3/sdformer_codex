"""Generate the single-point H67 motion-XOR TTX accuracy screen."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GEN = EXP_ROOT / "configs/generated"
BASE = GEN / "date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8.yml"
OUTPUT = GEN / "h67_allbinary_all12_motionxor_ttx_w025_s120.yml"
MANIFEST = GEN / "h67_motion_xor_ttx_manifest.json"
TTX_CHECKPOINT = (
    EXP_ROOT
    / "results/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid"
    / "checkpoint_epoch2.pth"
)


def main() -> int:
    if not BASE.is_file():
        raise FileNotFoundError(BASE)
    if not TTX_CHECKPOINT.is_file():
        raise FileNotFoundError(TTX_CHECKPOINT)
    cfg = deepcopy(yaml.safe_load(BASE.read_text(encoding="utf-8")) or {})
    cfg["experiment"] = "h67_allbinary_all12_motionxor_ttx_w025_s120"
    cfg["bsa_attention"]["binary_motion_xor_alpha"] = 0.25
    cfg.setdefault("runtime", {}).update({
        "max_train_steps": 120,
        "skip_state_save": True,
        "force_save_epochs": [0],
        "use_mlflow_model_logging": False,
    })
    cfg["loader"].update({"n_epochs": 1, "batch_size": 8, "n_workers": 8})
    cfg["metrics"]["name"] = ["AEE", "AAE"]
    cfg["test"].update({"sample": 10, "n_valid": 1})
    cfg["note"] = (
        "H67 single-point accuracy screen. H60 dyadic TTX plus a 1/4-weight "
        "same-position temporal K XOR-popcount bias in every one of the 12 encoder "
        "attention blocks. Full 105 one-sided binary ATLIF wrappers; no native "
        "QKFormer carrier, SC mixing, Kmag, target-rate, or partial replacement."
    )
    OUTPUT.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    row = {
        "name": cfg["experiment"],
        "config": str(OUTPUT),
        "resume": str(TTX_CHECKPOINT),
        "neuron": "105 x one-sided binary official ATLIF",
        "attention": "12 x H60 dyadic TTX + temporal K XOR-popcount/4",
        "native_qkformer_carrier": False,
        "status": "generated_not_run",
    }
    MANIFEST.write_text(json.dumps([row], indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
