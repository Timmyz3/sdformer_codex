"""Generate five-epoch continuations from the completed H67 rescue screens."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
GEN = EXP / "configs/generated"
SCREEN_ROOT = EXP / "results/dsec_fullres_w15_rescue_screen_20260801"

SPECS = (
    ("H67_crop_bb1e4", "bb1e4"),
    ("H67_crop_bb2e5", "bb2e5"),
)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def build_config(candidate: str, profile: str) -> tuple[dict[str, Any], Path]:
    source_config = GEN / f"dsec_fullres_w15_rescue_{candidate}_screen1.yml"
    source_checkpoint = SCREEN_ROOT / candidate / "checkpoint_epoch0.pth"
    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)

    config = deepcopy(load_yaml(source_config))
    config["experiment"] = f"dsec_fullres_w15_rescue_{candidate}_continue5"
    config["swin_transformer"]["pretrained_window_size"] = [2, 15, 15]
    config["loader"]["n_epochs"] = 5
    config["test"]["n_valid"] = 1
    runtime = config.setdefault("runtime", {})
    runtime.update(
        {
            "epoch_offset": 1,
            "skip_save": False,
            "skip_state_save": False,
            "save_only_force_epochs": True,
            "force_save_epochs": [4],
            "state_save_epochs": [4],
            "rescue_continuation": "screen_epoch0_model_only",
            "rescue_source_checkpoint": str(source_checkpoint),
        }
    )
    config["note"] = (
        f"Five-epoch convergence check for {candidate}; continue from its completed "
        "screen epoch0 model. The screen did not save optimizer/scaler state, so AdamW "
        "is rebuilt symmetrically for both LR branches. Saved epoch labels use offset=1."
    )
    return config, source_checkpoint


def main() -> int:
    rows = []
    for order, (candidate, profile) in enumerate(SPECS, start=1):
        config, checkpoint = build_config(candidate, profile)
        output = GEN / f"dsec_fullres_w15_rescue_{candidate}_continue5.yml"
        output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        rows.append(
            {
                "order": order,
                "id": candidate,
                "profile": profile,
                "config": str(output),
                "checkpoint": str(checkpoint),
                "epochs": 5,
                "epoch_offset": 1,
                "final_epoch": 5,
                "optimizer_resume": False,
            }
        )
        print(output)

    manifest = GEN / "dsec_fullres_w15_rescue_short5_manifest.json"
    manifest.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
