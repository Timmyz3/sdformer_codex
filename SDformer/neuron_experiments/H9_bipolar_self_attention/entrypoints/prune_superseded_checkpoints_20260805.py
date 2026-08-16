"""Prune superseded model checkpoints while retaining paper-relevant winners."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "neuron_experiments/H9_bipolar_self_attention/results"
AUDIT = RESULTS / "checkpoint_prune_audit_20260805.json"


KEEP_EPOCHS = {
    "h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_bs8_full30_20260711_setsid": {19, 29},
    "h68_allbinary_all12_castling_ttx_aux050_toep20_w720_fastlr_full30_bs8_full30_20260711_setsid": {19, 29},
    "h69_allbinary_all12_dyadic_temperature_ttx_x8_w720_fastlr_full30_bs8_full30_20260711_setsid": {19, 29},
    "h70_allbinary_all12_event_selective_ttx_maxshift3_w720_fastlr_full30_bs8_full30_20260711_setsid": {19, 29},
    "h71_allbinary_all12_window_context_ttx_w720_fastlr_full30_bs8_full30_20260711_setsid": {19, 29},
    "h66c_allbinary_all12_tp_ttx_w720_fastlr_full30_bs8_full30_20260712_setsid": {19, 29},
    "h66f_allbinary_all12_local5_tp_w720_fastlr_full30_bs8_full30_20260723_setsid": {19, 29},
    "h66g_allbinary_all12_local5_motion_w720_fastlr_full30_bs8_full30_20260723_setsid": {19, 29},
}


SUPERSEDED_SHORT_DIRS = (
    "h67_motionxor_w025_s360_20260711_200708",
    "h67_motionxor_w025_20260711_200046",
    "h68_castling_ttx_aux050_s360_retry_20260711_202914",
    "h69_dyadic_temperature_screen_20260713_132619",
    "h70_event_selective_health_20260714_041716",
    "h71_window_context_health_20260714_182732",
    "h66c_tp_accuracy_first_20260711_153624",
    "h66c_tp_accuracy360_20260711_154031",
    "h66d_lr_accuracy_first_20260711_153205",
    "h66e_tp_selfbias1_20260711_195504",
    "h63_stc_signedtx_symtern_recovery20_20260711_144352",
    "h63_stc_symtern_recovery20_20260711_034750",
)


EXPLICIT_SUPERSEDED = (
    "dsec_fullres_w15_rescue_short5_20260801/H67_crop_bb1e4/checkpoint_epoch5.pth",
    "dsec_fullres_w15_rescue_short5_20260801/H67_crop_bb1e4/checkpoint_epoch5_state_dict.pth",
    "dsec_fullres_w15_rescue_short5_20260801/H67_crop_bb2e5/checkpoint_epoch5.pth",
    "dsec_fullres_w15_rescue_short5_20260801/H67_crop_bb2e5/checkpoint_epoch5_state_dict.pth",
    "dsec_fullres_w15_H67_crop_bb1e4_resume10_20260802/checkpoint_epoch10.pth",
    "dsec_fullres_w15_H67_crop_bb1e4_resume10_20260802/checkpoint_epoch10_state_dict.pth",
    "dsec_fullres_w15_H67_crop_bb1e4_resume15_20260803/checkpoint_epoch12.pth",
    "dsec_fullres_w15_H67_crop_bb1e4_resume30_20260804/checkpoint_epoch20.pth",
    "dsec_fullres_w15_H67_crop_bb1e4_resume30_20260804/checkpoint_epoch25.pth",
    "dsec_fullres_w15_H67_crop_bb1e4_resume30_20260804/resume_source_ep15_scheduler_aligned/checkpoint_epoch15.pth",
    "dsec_fullres_w15_H67_crop_bb1e4_resume30_20260804/resume_source_ep15_scheduler_aligned/checkpoint_epoch15_state_dict.pth",
)


CHECKPOINT_RE = re.compile(r"^checkpoint_epoch(\d+)(?:_state_dict)?\.pth$")


def add_file(path: Path, reason: str, selected: dict[Path, str]) -> None:
    if path.is_file() or path.is_symlink():
        selected[path] = reason


def main() -> None:
    selected: dict[Path, str] = {}
    retained: dict[str, list[int]] = {}

    for directory_name, keep in KEEP_EPOCHS.items():
        directory = RESULTS / directory_name
        retained[directory_name] = sorted(keep)
        if not directory.is_dir():
            continue
        for path in directory.glob("checkpoint_epoch*.pth"):
            match = CHECKPOINT_RE.match(path.name)
            if match and int(match.group(1)) not in keep:
                add_file(path, f"intermediate; retain epochs {sorted(keep)}", selected)
        partial = directory / "checkpoint_epoch0_partial_before_resume_20260714_234909.pth"
        add_file(partial, "superseded partial checkpoint", selected)

    for directory_name in SUPERSEDED_SHORT_DIRS:
        directory = RESULTS / directory_name
        if not directory.is_dir():
            continue
        for path in directory.rglob("checkpoint_epoch*.pth"):
            add_file(path, "short screen superseded by completed full30", selected)

    for relative in EXPLICIT_SUPERSEDED:
        add_file(RESULTS / relative, "superseded H67 fullres intermediate", selected)

    deleted = []
    apparent_bytes = 0
    for path in sorted(selected):
        stat = path.stat()
        entry = {
            "path": str(path.relative_to(REPO)),
            "size": stat.st_size,
            "blocks_bytes": stat.st_blocks * 512,
            "inode": stat.st_ino,
            "reason": selected[path],
        }
        apparent_bytes += stat.st_size
        path.unlink()
        deleted.append(entry)

    audit = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "policy": (
            "retain best/final checkpoints for paper-relevant full30 runs; retain all configs, "
            "rankings, profiles, logs and hardware artifacts; remove superseded intermediate/short models"
        ),
        "retained_epoch_policy": retained,
        "deleted_count": len(deleted),
        "deleted_apparent_bytes": apparent_bytes,
        "deleted": deleted,
    }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"deleted_count": len(deleted), "apparent_bytes": apparent_bytes, "audit": str(AUDIT)}))


if __name__ == "__main__":
    main()
