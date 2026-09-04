#!/usr/bin/env python3
"""Generate a fail-closed H67 crop-to-fullres checkpoint lineage receipt."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
RESULTS = EXP / "results"
OUTPUT = REPO / "neuron_autoresearch/H67_FULLRES_LINEAGE_RECEIPT_20260805.json"
OUTPUT_MD = REPO / "neuron_autoresearch/H67_FULLRES_LINEAGE_RECEIPT_20260805.md"
DELETION_AUDIT = RESULTS / "checkpoint_prune_audit_20260805.json"
SCHEDULER_AUDIT = RESULTS / (
    "dsec_fullres_w15_H67_crop_bb1e4_resume30_20260804/"
    "resume_source_ep15_scheduler_aligned/scheduler_alignment_audit.json"
)

CROP_EP19 = RESULTS / (
    "h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_bs8_"
    "full30_20260711_setsid/checkpoint_epoch19.pth"
)
SCREEN_EP0 = RESULTS / (
    "dsec_fullres_w15_rescue_screen_20260801/H67_crop_bb1e4/checkpoint_epoch0.pth"
)
EP15 = RESULTS / "dsec_fullres_w15_H67_crop_bb1e4_resume15_20260803/checkpoint_epoch15.pth"
EP30 = RESULTS / "dsec_fullres_w15_H67_crop_bb1e4_resume30_20260804/checkpoint_epoch30.pth"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def binding(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def main() -> int:
    deletion = json.loads(DELETION_AUDIT.read_text(encoding="utf-8"))
    deleted = {item["path"]: item for item in deletion.get("deleted", [])}
    stages = (
        {
            "name": "crop_ep19_to_fullres_ep0",
            "log": RESULTS / "dsec_fullres_w15_rescue_screen_20260801/H67_crop_bb1e4/train.log",
            "config": EXP / "configs/generated/dsec_fullres_w15_rescue_H67_crop_bb1e4_screen1.yml",
            "source": CROP_EP19,
            "output": SCREEN_EP0,
            "resume_state": None,
        },
        {
            "name": "fullres_ep0_to_ep5",
            "log": RESULTS / "dsec_fullres_w15_rescue_short5_20260801/H67_crop_bb1e4/train.log",
            "config": EXP / "configs/generated/dsec_fullres_w15_rescue_H67_crop_bb1e4_continue5.yml",
            "source": SCREEN_EP0,
            "output": RESULTS / "dsec_fullres_w15_rescue_short5_20260801/H67_crop_bb1e4/checkpoint_epoch5.pth",
            "resume_state": None,
        },
        {
            "name": "fullres_ep5_to_ep10",
            "log": RESULTS / "dsec_fullres_w15_H67_crop_bb1e4_resume10_20260802/train.log",
            "config": EXP / "configs/generated/dsec_fullres_w15_H67_crop_bb1e4_resume_ep10.yml",
            "source": RESULTS / "dsec_fullres_w15_rescue_short5_20260801/H67_crop_bb1e4/checkpoint_epoch5.pth",
            "output": RESULTS / "dsec_fullres_w15_H67_crop_bb1e4_resume10_20260802/checkpoint_epoch10.pth",
            "resume_state": RESULTS / "dsec_fullres_w15_rescue_short5_20260801/H67_crop_bb1e4/checkpoint_epoch5_state_dict.pth",
        },
        {
            "name": "fullres_ep10_to_ep15",
            "log": RESULTS / "dsec_fullres_w15_H67_crop_bb1e4_resume15_20260803/train.log",
            "config": EXP / "configs/generated/dsec_fullres_w15_H67_crop_bb1e4_resume_ep15.yml",
            "source": RESULTS / "dsec_fullres_w15_H67_crop_bb1e4_resume10_20260802/checkpoint_epoch10.pth",
            "output": EP15,
            "resume_state": RESULTS / "dsec_fullres_w15_H67_crop_bb1e4_resume10_20260802/checkpoint_epoch10_state_dict.pth",
        },
        {
            "name": "fullres_ep15_to_ep30",
            "log": RESULTS / "dsec_fullres_w15_H67_crop_bb1e4_resume30_20260804/train.log",
            "config": EXP / "configs/generated/dsec_fullres_w15_H67_crop_bb1e4_resume_ep30.yml",
            "source": EP15,
            "output": EP30,
            "resume_state": RESULTS / "dsec_fullres_w15_H67_crop_bb1e4_resume15_20260803/checkpoint_epoch15_state_dict.pth",
        },
    )

    records = []
    for stage in stages:
        log = stage["log"]
        config = stage["config"]
        source = stage["source"]
        output = stage["output"]
        resume_state = stage["resume_state"]
        text = log.read_text(encoding="utf-8", errors="replace")
        source_relative = relative(source)
        output_relative = relative(output)
        output_deleted = output_relative in deleted
        source_deleted = source_relative in deleted
        state_relative = relative(resume_state) if resume_state is not None else None
        state_deleted = state_relative in deleted if state_relative else False
        checks = {
            "log_exists": log.is_file(),
            "config_exists": config.is_file(),
            "source_named_in_command": str(source.resolve()) in text,
            "output_save_path_named": str(output.parent.resolve()) in text,
            "strict_overlay_load": (
                "checkpoint_overlay_keys=210, missing=0, unexpected=0" in text
            ),
            "source_preserved_or_audited_deleted": source.is_file() or source_deleted,
            "output_preserved_or_audited_deleted": output.is_file() or output_deleted,
            "resume_state_preserved_or_audited_deleted": (
                resume_state is None or resume_state.is_file() or state_deleted
            ),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(f"H67 lineage stage {stage['name']} failed: {failed}")
        records.append(
            {
                "name": stage["name"],
                "status": "PASS",
                "source": (
                    binding(source)
                    if source.is_file()
                    else {"path": str(source.resolve()), "deletion_audit": deleted[source_relative]}
                ),
                "output": (
                    binding(output)
                    if output.is_file()
                    else {"path": str(output.resolve()), "deletion_audit": deleted[output_relative]}
                ),
                "resume_state": (
                    None
                    if resume_state is None
                    else binding(resume_state)
                    if resume_state.is_file()
                    else {"path": str(resume_state.resolve()), "deletion_audit": deleted[state_relative]}
                ),
                "config": binding(config),
                "log": binding(log),
                "checks": checks,
            }
        )

    scheduler = json.loads(SCHEDULER_AUDIT.read_text(encoding="utf-8"))
    scheduler_checks = {
        "source_ep15_sha": scheduler.get("source_model_sha256") == sha256(EP15),
        "optimizer_scaler_unchanged": scheduler.get("optimizer_and_scaler_unchanged") is True,
        "scheduler_epoch15": int((scheduler.get("scheduler_after") or {}).get("last_epoch", -1)) == 15,
    }
    if not all(scheduler_checks.values()):
        raise RuntimeError(f"H67 scheduler alignment failed: {scheduler_checks}")

    source_paths = [str(stage["source"]["path"]).lower() for stage in records]
    lineage_checks = {
        "five_stages": len(records) == 5,
        "initial_source_is_h67_crop_ep19": Path(records[0]["source"]["path"]).resolve()
        == CROP_EP19.resolve(),
        "no_nb0_source": all("nb0" not in path for path in source_paths),
        "no_local5_h66d_source": all("local5" not in path and "h66d" not in path for path in source_paths),
        "all_stages_pass": all(stage["status"] == "PASS" for stage in records),
        "final_checkpoint_ep30": Path(records[-1]["output"]["path"]).resolve() == EP30.resolve(),
    }
    if not all(lineage_checks.values()):
        raise RuntimeError(f"H67 lineage identity failed: {lineage_checks}")

    receipt = {
        "schema": "h67_fullres_lineage_receipt_v1",
        "status": "PASS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": binding(Path(__file__)),
        "claim": "H67 own-crop ep19 to H67 fullres ep30; no NB0/fullres or Local5 initialization",
        "initial_checkpoint": binding(CROP_EP19),
        "final_checkpoint": binding(EP30),
        "stages": records,
        "scheduler_alignment": {
            "artifact": binding(SCHEDULER_AUDIT),
            "checks": scheduler_checks,
        },
        "deletion_audit": binding(DELETION_AUDIT),
        "checks": lineage_checks,
    }
    OUTPUT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text(
        "\n".join(
            [
                "# H67 Full-Resolution Lineage Receipt",
                "",
                "Status: **PASS**",
                "",
                "- Initial checkpoint: H67 Motion-XOR crop ep19.",
                "- Full-resolution stages: ep0 -> ep5 -> ep10 -> ep15 -> ep30.",
                "- Every stage loaded overlay210 with missing=0 and unexpected=0.",
                "- No NB0/fullres or Local5/H66d checkpoint is in the source chain.",
                "- Deleted ep5/ep10 intermediates are bound through the executed cleanup audit.",
                f"- Final ep30 SHA256: `{receipt['final_checkpoint']['sha256']}`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
