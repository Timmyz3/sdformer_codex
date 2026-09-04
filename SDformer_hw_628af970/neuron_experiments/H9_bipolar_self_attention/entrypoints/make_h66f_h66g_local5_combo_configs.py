"""Generate H66f (Local5+TP) and H66g (Local5+Motion) full30 configs from H66d."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
GEN = EXP / "configs/generated"
BASE = GEN / "h66d_allbinary_all12_lr_ttx_w720_fastlr_full30.yml"
MANIFEST = GEN / "h66f_h66g_local5_combo_full30_manifest.json"
SAVE_EPOCHS = [0, 4, 9, 14, 19, 24, 28, 29]

CANDIDATES = [
    {
        "name": "h66f_allbinary_all12_local5_tp_w720_fastlr_full30",
        "mode": "binary_axnor_local5_tp_shiftmax",
        "binary_motion_xor_alpha": 0.0,
        "note": (
            "H66f Scheme A: Local-5 spatial stencil plus temporal-peer candidate "
            "(6-way alpha-XNOR Shiftmax). Independent full30 from frozen TTX ep2. "
            "No new parameters. Compare against H66d Local-5 (1.4432) and H67 (1.4671)."
        ),
    },
    {
        "name": "h66g_allbinary_all12_local5_motion_w720_fastlr_full30",
        "mode": "binary_axnor_local5_motion_shiftmax",
        "binary_motion_xor_alpha": 0.25,
        "note": (
            "H66g: Local-5 spatial stencil with H67-style motion XOR-popcount bias "
            "on the self lane only (alpha=1/4). Independent full30 from frozen TTX ep2. "
            "Broadcasting motion to all lanes would leave Shiftmax invariant."
        ),
    },
]


def main() -> int:
    if not BASE.exists():
        raise FileNotFoundError(f"missing base config: {BASE}")
    base = yaml.safe_load(BASE.read_text(encoding="utf-8")) or {}
    rows = []
    for order, cand in enumerate(CANDIDATES, start=1):
        cfg = deepcopy(base)
        name = cand["name"]
        cfg["experiment"] = name
        cfg["bsa_attention"]["mode"] = cand["mode"]
        cfg["bsa_attention"]["binary_motion_xor_alpha"] = float(cand["binary_motion_xor_alpha"])
        cfg.setdefault("runtime", {}).update({
            "max_train_steps": 0,
            "skip_state_save": False,
            "save_only_force_epochs": True,
            "state_save_epochs": [19, 24, 29],
            "force_save_epochs": list(SAVE_EPOCHS),
            "use_mlflow_model_logging": False,
        })
        cfg["loader"]["n_epochs"] = 30
        cfg["metrics"]["name"] = ["AEE", "AAE"]
        cfg["note"] = cand["note"]
        path = GEN / f"{name}.yml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        rows.append({
            "order": order,
            "name": name,
            "config": str(path),
            "mode": cand["mode"],
            "binary_motion_xor_alpha": cand["binary_motion_xor_alpha"],
            "epochs": 30,
            "start_checkpoint": (
                "neuron_experiments/H9_bipolar_self_attention/results/"
                "date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/"
                "checkpoint_epoch2.pth"
            ),
        })
        print(path)
    MANIFEST.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
