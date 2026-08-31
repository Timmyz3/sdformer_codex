#!/usr/bin/env python3
"""Generate the promoted C12 alpha=1/8 true-resume ep30-34 config."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
SOURCE = EXP / "configs/generated/dsec_fullres_w15_two_contrib_c12_binary_motion_ttx_nb0ep29_ft30_20260826.yml"
OUTPUT = EXP / "configs/generated/dsec_c12_alpha0125_ep29_resume5_20260830.yml"
MANIFEST = EXP / "configs/generated/dsec_c12_alpha0125_ep29_resume5_20260830.json"
PARENT = EXP / "results/date_two_contribution_full30_20260826/c12_binary_motion_ttx/checkpoint_epoch29.pth"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new_or_identical(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"refusing to overwrite changed artifact: {path}")
        return
    path.write_text(text, encoding="utf-8")


def main() -> int:
    for path in (SOURCE, PARENT, PARENT.with_name(PARENT.stem + "_state_dict.pth")):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = deepcopy(yaml.safe_load(SOURCE.read_text(encoding="utf-8")))
    config["experiment"] = "dsec_c12_alpha0125_ep29_resume5_20260830"
    config["bsa_attention"]["binary_motion_xor_alpha"] = 0.125
    config["loader"]["n_epochs"] = 35
    runtime = config.setdefault("runtime", {})
    runtime["force_save_epochs"] = [30, 32, 34]
    runtime["save_only_force_epochs"] = True
    runtime["state_save_epochs"] = [34]
    runtime["resume_protocol"] = "true_model_optimizer_scheduler_scaler_resume_ep29_to_ep34"
    runtime["resume_source_epoch"] = 29
    runtime["promoted_alpha"] = 0.125
    config["note"] = (
        "DATE optimized C12 recovery: true resume from same-parent C12 ep29 through global "
        "epochs 30-34, changing only the dyadic Motion alpha from 1/4 to 1/8. The strict "
        "same-parent full30 ablation remains unchanged."
    )
    write_new_or_identical(OUTPUT, yaml.safe_dump(config, sort_keys=False, width=120))
    payload = {
        "schema": "dsec_c12_alpha0125_resume5_manifest_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_config": str(SOURCE.resolve()),
        "source_config_sha256": sha256(SOURCE),
        "config": str(OUTPUT.resolve()),
        "config_sha256": sha256(OUTPUT),
        "parent_checkpoint": str(PARENT.resolve()),
        "parent_checkpoint_sha256": sha256(PARENT),
        "parent_state": str(PARENT.with_name(PARENT.stem + "_state_dict.pth").resolve()),
        "parent_state_sha256": sha256(PARENT.with_name(PARENT.stem + "_state_dict.pth")),
        "evaluation_epochs": [30, 32, 34],
        "claim_boundary": "optimized deployment mainline; not a replacement for the strict full30 causal row",
    }
    write_new_or_identical(MANIFEST, json.dumps(payload, indent=2) + "\n")
    print(MANIFEST.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
