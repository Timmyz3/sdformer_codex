"""Generate memory-safe H73-H80 configs without changing registered originals."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
GEN = EXP / "configs/generated"
MANIFEST = GEN / "h73_h80_bs4acc2_full30_manifest.json"

CANDIDATES = (
    ("H73", "h73_allbinary_all12_de9_match_code_w720_fastlr_full30", 12),
    ("H74", "h74_allbinary_all12_mc49_match_code_w720_fastlr_full30", 12),
    ("H75", "h75_allbinary_all12_ax17_match_code_w720_fastlr_full30", 12),
    ("H76", "h76_allbinary_all12_pc9_patch_match_code_w720_fastlr_full30", 12),
    ("H77", "h77_allbinary_all12_lc4_match_code_w720_fastlr_full30", 24),
    ("H78", "h78_allbinary_all12_g4_match_code_w720_fastlr_full30", 12),
    ("H79", "h79_allbinary_all12_cf10_match_code_w720_fastlr_full30", 24),
    ("H80", "h80_allbinary_all12_dn9_match_code_w720_fastlr_full30", 12),
)


def main() -> int:
    rows = []
    for candidate_id, source_name, expected_missing in CANDIDATES:
        source = GEN / f"{source_name}.yml"
        config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        safe_name = f"{source_name}_bs4acc2"
        config["experiment"] = safe_name
        config["loader"]["batch_size"] = 4
        config["optimizer"]["num_acc"] = 2
        config["optimizer"]["lr_warmup"]["steps"] = 1440
        config.setdefault("runtime", {})["memory_safe_effective_batch"] = 8
        config["note"] = (
            f"{config.get('note', '')} Memory-safe execution uses batch4 x gradient-"
            "accumulation2 (effective batch8); warmup1440 micro-steps preserves 720 "
            "optimizer updates and the same 5760-sample warmup as batch8."
        ).strip()

        target = GEN / f"{safe_name}.yml"
        target.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        rows.append({
            "id": candidate_id,
            "name": safe_name,
            "source_name": source_name,
            "config": str(target),
            "expected_missing": expected_missing,
            "batch_size": 4,
            "num_acc": 2,
            "effective_batch": 8,
            "warmup_micro_steps": 1440,
            "warmup_optimizer_updates": 720,
            "epochs": 30,
        })
        print(target)

    MANIFEST.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
