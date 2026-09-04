#!/usr/bin/env python3
"""Audit the H67 Motion vs H81 no-motion training-control contract."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import yaml


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
GEN = EXP / "configs/generated"
RESULTS = EXP / "results"
H67_CROP_CONFIG = GEN / "h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30.yml"
H81_CROP_CONFIG = GEN / "h81_allbinary_all12_h60_nomotion_equalbudget_w720_fastlr_full30.yml"
H67_FULL_CONFIG = GEN / "dsec_fullres_w15_H67_crop_bb1e4_resume_ep30.yml"
H81_FULL_CONFIG = GEN / "dsec_fullres_w15_H81_nomotion_bb1e4_ft40.yml"
H67_LAUNCHER = RESULTS / "h67_h68_full30_queue_launcher_20260711_210210.log"
H81_LAUNCHER = RESULTS / "h81_equal_budget_after_h66_status.log"
H67_LINEAGE = REPO / "neuron_autoresearch/H67_FULLRES_LINEAGE_RECEIPT_20260805.json"
H67_CROP_RANKING = RESULTS / (
    "h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_bs8_"
    "full30_20260711_setsid/profile_ranking_valid825.md"
)
H81_CROP_RANKING = RESULTS / (
    "h81_allbinary_all12_h60_nomotion_equalbudget_w720_fastlr_full30_"
    "bs8_full30_20260717_setsid/profile_ranking_valid825.md"
)
OUTPUT = REPO / "neuron_autoresearch/H67_H81_TRAINING_FAIRNESS_20260812.json"
OUTPUT_MD = OUTPUT.with_suffix(".md")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def canonical_crop(config: dict) -> dict:
    value = deepcopy(config)
    value.pop("experiment", None)
    value.pop("note", None)
    value["bsa_attention"].pop("binary_motion_xor_alpha", None)
    return value


def selected_fullres_contract(config: dict) -> dict:
    loader = deepcopy(config["loader"])
    loader.pop("n_epochs", None)
    attention = deepcopy(config["bsa_attention"])
    attention.pop("binary_motion_xor_alpha", None)
    runtime_keys = (
        "allow_tf32",
        "cudnn_benchmark",
        "snn_backend",
        "max_train_steps",
        "skip_state_save",
        "use_mlflow_model_logging",
        "seed",
        "skip_save",
        "save_only_force_epochs",
        "physical_batch",
        "gradient_accumulation",
        "rescue_profile",
    )
    return {
        "data": config["data"],
        "model": config["model"],
        "swin_transformer": config["swin_transformer"],
        "spiking_neuron": config["spiking_neuron"],
        "atlif_ternary_psn": config["atlif_ternary_psn"],
        "bsa_attention_without_motion": attention,
        "experimental_neuron": config["experimental_neuron"],
        "loss": config["loss"],
        "optimizer": config["optimizer"],
        "loader_without_budget": loader,
        "metrics": config["metrics"],
        "test": config["test"],
        "runtime_common": {
            key: config.get("runtime", {}).get(key) for key in runtime_keys
        },
    }


def launcher_parent(log_path: Path, experiment: str) -> str:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        rf"START {re.escape(experiment)}.*?--prev_runid\s+(\S+)",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        raise RuntimeError(f"cannot locate launcher parent for {experiment}")
    return str(Path(match.group(1)).resolve())


def ranking_epochs(path: Path) -> list[int]:
    rows: list[tuple[int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|", line)
        if match:
            rows.append((int(match.group(1)), int(match.group(2))))
    if not rows or [rank for rank, _ in rows] != list(range(1, len(rows) + 1)):
        raise RuntimeError(f"invalid crop ranking: {path}")
    return [epoch for _, epoch in rows]


def main() -> int:
    required = (
        H67_CROP_CONFIG,
        H81_CROP_CONFIG,
        H67_FULL_CONFIG,
        H81_FULL_CONFIG,
        H67_LAUNCHER,
        H81_LAUNCHER,
        H67_LINEAGE,
        H67_CROP_RANKING,
        H81_CROP_RANKING,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    h67_crop = load_yaml(H67_CROP_CONFIG)
    h81_crop = load_yaml(H81_CROP_CONFIG)
    h67_full = load_yaml(H67_FULL_CONFIG)
    h81_full = load_yaml(H81_FULL_CONFIG)
    h67_parent = launcher_parent(H67_LAUNCHER, h67_crop["experiment"])
    h81_parent = launcher_parent(H81_LAUNCHER, "H81 equal-budget no-motion full30")
    parent = Path(h67_parent)
    lineage = json.loads(H67_LINEAGE.read_text(encoding="utf-8"))
    h67_initial = Path(lineage["initial_checkpoint"]["path"]).resolve()
    h81_source = Path(h81_full["runtime"]["source_crop_checkpoint"]).resolve()
    h67_crop_epochs = ranking_epochs(H67_CROP_RANKING)
    h81_crop_epochs = ranking_epochs(H81_CROP_RANKING)

    checks = {
        "crop configs differ only by experiment/note/motion alpha": (
            canonical_crop(h67_crop) == canonical_crop(h81_crop)
        ),
        "crop motion alpha H67=0.25": (
            float(h67_crop["bsa_attention"]["binary_motion_xor_alpha"]) == 0.25
        ),
        "crop motion alpha H81=0": (
            float(h81_crop["bsa_attention"]["binary_motion_xor_alpha"]) == 0.0
        ),
        "same crop parent checkpoint": h67_parent == h81_parent,
        "crop parent exists": parent.is_file(),
        "fullres common recipe equal": (
            selected_fullres_contract(h67_full) == selected_fullres_contract(h81_full)
        ),
        "fullres geometry 480x640": h81_full["loader"]["resolution"] == [480, 640],
        "fullres crop null": h81_full["loader"].get("crop") is None,
        "fullres window T2x15x15": h81_full["swin_transformer"]["window_size"] == [2, 15, 15],
        "fullres H81 budget40": int(h81_full["loader"]["n_epochs"]) == 40,
        "fullres H67 base budget30": int(h67_full["loader"]["n_epochs"]) == 30,
        "H67 lineage pass": lineage.get("status") == "PASS",
        "H67 lineage starts own crop ep19": h67_initial.name == "checkpoint_epoch19.pth",
        "H81 starts own crop ep19": h81_source.name == "checkpoint_epoch19.pth",
        "same crop evaluation epoch set": set(h67_crop_epochs) == set(h81_crop_epochs),
        "H67 crop ep19 is AEE rank1": h67_crop_epochs[0] == 19,
        "H81 crop ep19 is AEE rank1": h81_crop_epochs[0] == 19,
        "fullres motion alpha H67=0.25": (
            float(h67_full["bsa_attention"]["binary_motion_xor_alpha"]) == 0.25
        ),
        "fullres motion alpha H81=0": (
            float(h81_full["bsa_attention"]["binary_motion_xor_alpha"]) == 0.0
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"H67/H81 fairness audit failed: {failed}")

    output = {
        "schema": "h67_h81_training_fairness_v1",
        "status": "PASS_RECIPE_LEVEL_CONTROL_NOT_STEP_PAIRED",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "crop_parent_checkpoint": h67_parent,
        "crop_parent_checkpoint_sha256": sha256(parent),
        "h67_crop_checkpoint": str(h67_initial),
        "h67_crop_checkpoint_sha256": sha256(h67_initial),
        "h81_crop_checkpoint": str(h81_source),
        "h81_crop_checkpoint_sha256": sha256(h81_source),
        "crop_rankings": {
            "H67": {**binding(H67_CROP_RANKING), "epochs_by_AEE_rank": h67_crop_epochs},
            "H81": {**binding(H81_CROP_RANKING), "epochs_by_AEE_rank": h81_crop_epochs},
        },
        "config_bindings": {
            str(path.resolve()): sha256(path)
            for path in (H67_CROP_CONFIG, H81_CROP_CONFIG, H67_FULL_CONFIG, H81_FULL_CONFIG)
        },
        "interpretation": {
            "recipe_level_no_motion_control": True,
            "same_parent_and_seed_matched_crop_training": True,
            "fullres_common_recipe_matched": True,
            "step_paired_rng_or_optimizer_trajectory": False,
            "reason": (
                "H67 full-resolution adaptation is an audited five-stage historical "
                "rescue/continuation; H81 is an uninterrupted 40-epoch run."
            ),
            "allowed_claim": (
                "H81 is a seed-matched, same-parent, same-recipe no-motion control."
            ),
            "disallowed_claim": (
                "H67 and H81 are bit-exact paired training trajectories differing only "
                "by Motion-XOR at every optimizer step."
            ),
        },
        "h67_lineage": str(H67_LINEAGE.resolve()),
        "h67_lineage_sha256": sha256(H67_LINEAGE),
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text(
        "\n".join(
            (
                "# H67 Motion vs H81 no-motion training fairness",
                "",
                "Status: `PASS_RECIPE_LEVEL_CONTROL_NOT_STEP_PAIRED`.",
                "",
                "- Crop training uses the same parent checkpoint and identical config "
                "after removing experiment/note and `binary_motion_xor_alpha`; H67 uses "
                "`0.25`, H81 uses `0.0`.",
                "- Full-resolution geometry, model, neuron, optimizer, augmentation and "
                "evaluation contracts match. H81 runs the registered 40-epoch no-motion control.",
                "- H67's historical full-resolution path is a five-stage audited rescue/continuation, "
                "whereas H81 is uninterrupted. This supports a recipe-level control, not a "
                "bit-exact step-paired causal claim.",
                "- Final metrics remain pending until H81 valid825 finishes.",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
