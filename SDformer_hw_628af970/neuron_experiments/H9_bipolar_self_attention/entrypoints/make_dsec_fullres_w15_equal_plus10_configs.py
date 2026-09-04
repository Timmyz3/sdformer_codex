"""Generate equal-budget +10 full-resolution convergence-audit configs."""

from __future__ import annotations

from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
GEN = EXP / "configs/generated"
H67_SOURCE = GEN / "dsec_fullres_w15_H67_crop_bb1e4_resume_ep30.yml"
NB0_SOURCE = GEN / "dsec_fullres_paper_w15_nb0_ep59_ft30.yml"
LOCAL5_SOURCE = GEN / "dsec_fullres_w15_H66d_local5_bb1e4_ft30.yml"
H67_OUTPUT = GEN / "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40.yml"
NB0_OUTPUT = GEN / "dsec_fullres_w15_NB0_equal_plus10_ep40.yml"
LOCAL5_OUTPUT = GEN / "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus10_ep40.yml"


def build(
    source: Path, *, experiment: str, epoch_offset: int, source_checkpoint_label: int
) -> dict:
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["experiment"] = experiment
    config["loader"]["n_epochs"] = 40
    # The staged resume state also clears old milestones. Keeping the generated
    # config explicit makes fresh construction and resumed scheduler semantics agree.
    config["optimizer"]["milestones"] = []
    runtime = config.setdefault("runtime", {})
    runtime.update(
        {
            "force_save_epochs": [34, 39],
            "state_save_epochs": [34, 39],
            "save_only_force_epochs": True,
            "convergence_extension": "equal_plus10_from_completed_fullres30",
            "convergence_extension_lr": "fixed_source_optimizer_lr_2p5e-5",
            "resume_rng_scope": "seed_reset_not_bit_exact_rng_continuation",
            "resume_protocol": (
                "audited_model_optimizer_scheduler_scaler_equal_plus10_from_fullres30"
            ),
            "resume_source_budget": 30,
            "resume_source_checkpoint_label": source_checkpoint_label,
        }
    )
    # Remove inherited metadata from the earlier ep15->ep30 rescue continuation.
    runtime.pop("resume_source_epoch", None)
    if epoch_offset:
        runtime["epoch_offset"] = epoch_offset
    else:
        runtime.pop("epoch_offset", None)
    config["note"] = (
        "Equal-budget +10 full-resolution convergence audit. Resume model, AdamW, AMP "
        "scaler and current 2.5e-5 optimizer LR from the completed 30-epoch state; "
        "clear future scheduler milestones for both candidates. RNG state was not "
        "serialized by the historical trainer, so this is not a bit-exact continuation."
    )
    return config


def validate(config: dict, *, epoch_offset: int, source_checkpoint_label: int) -> None:
    runtime = config["runtime"]
    checks = {
        "resolution": config["loader"].get("resolution") == [480, 640],
        "crop": config["loader"].get("crop") is None,
        "window": config["swin_transformer"].get("window_size") == [2, 15, 15],
        "epochs": config["loader"].get("n_epochs") == 40,
        "batch": config["loader"].get("batch_size") == 2,
        "BN": config["test"].get("bn_policy") == "no_running",
        "milestones": config["optimizer"].get("milestones") == [],
        "model saves": runtime.get("force_save_epochs") == [34, 39],
        "state saves": runtime.get("state_save_epochs") == [34, 39],
        "offset": int(runtime.get("epoch_offset", 0) or 0) == epoch_offset,
        "resume protocol": runtime.get("resume_protocol")
        == "audited_model_optimizer_scheduler_scaler_equal_plus10_from_fullres30",
        "resume source budget": runtime.get("resume_source_budget") == 30,
        "resume checkpoint label": runtime.get("resume_source_checkpoint_label")
        == source_checkpoint_label,
        "no stale source epoch": "resume_source_epoch" not in runtime,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("invalid +10 convergence config: " + ", ".join(failed))


def main() -> None:
    h67 = build(
        H67_SOURCE,
        experiment="dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40",
        epoch_offset=1,
        source_checkpoint_label=30,
    )
    nb0 = build(
        NB0_SOURCE,
        experiment="dsec_fullres_w15_NB0_equal_plus10_ep40",
        epoch_offset=0,
        source_checkpoint_label=29,
    )
    local5 = build(
        LOCAL5_SOURCE,
        experiment="dsec_fullres_w15_H66d_local5_bb1e4_equal_plus10_ep40",
        epoch_offset=0,
        source_checkpoint_label=29,
    )
    validate(h67, epoch_offset=1, source_checkpoint_label=30)
    validate(nb0, epoch_offset=0, source_checkpoint_label=29)
    validate(local5, epoch_offset=0, source_checkpoint_label=29)
    H67_OUTPUT.write_text(yaml.safe_dump(h67, sort_keys=False), encoding="utf-8")
    NB0_OUTPUT.write_text(yaml.safe_dump(nb0, sort_keys=False), encoding="utf-8")
    LOCAL5_OUTPUT.write_text(yaml.safe_dump(local5, sort_keys=False), encoding="utf-8")
    print(H67_OUTPUT)
    print(NB0_OUTPUT)
    print(LOCAL5_OUTPUT)


if __name__ == "__main__":
    main()
