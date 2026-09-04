#!/usr/bin/env python3
"""在正式 profile 前冻结 Local5 v3 候选合同并写入 Git blob 锚点。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import evaluate_local5_joint_candidates_v3 as evaluator
import local5_joint_candidate_reference_v3 as reference


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def build_prereg(root: Path, frozen_at: str) -> dict[str, object]:
    calibration = json.loads(
        (
            root
            / "results/local5_ordered_frontend_rtl_calibration_20260810/report.json"
        ).read_text(encoding="utf-8")
    )
    if (
        calibration.get("schema") != "local5_ordered_frontend_rtl_calibration_v1"
        or calibration.get("calibration", {}).get("fixed_cycles_median") != 459
        or calibration.get("calibration", {}).get("residual", {}).get("max") != 475.0
        or calibration.get("decision", {}).get("v2_prereg")
        != "INVALIDATE_BEFORE_FORMAL_PROFILE"
    ):
        raise ValueError("RTL校准报告不能支持v3固定项")
    sources = evaluator.source_paths(root)
    return {
        "schema": "local5_joint_candidate_prereg_v3",
        "status": "FROZEN_BEFORE_PROFILE",
        "frozen_at_utc": frozen_at,
        "profile_status_at_freeze": "GPU_WAITING_NO_FORMAL_PAYLOAD",
        "supersedes": {
            "schema": "local5_joint_candidate_prereg_v2",
            "sha256": "324a67aec7b28a701945b958d5cc62e00c3914c4a2a6883aaa5c7b240932ec94",
            "status": "INVALIDATED_BEFORE_PROFILE",
            "reason": "max(relation,backend)与当前集成RTL相序不符，并在held-out上翻转Direct/GASR方向",
        },
        "baseline": "c0_direct_recompute",
        "candidates": reference.CANDIDATES,
        "stage_heads": list(evaluator.STAGE_HEADS),
        "stage_output_tiles": list(evaluator.STAGE_HEADS),
        "stage_windows": list(evaluator.STAGE_WINDOWS),
        "fixed_cycle_scenarios": reference.FIXED_SCENARIOS,
        "model_scope": {
            "unit": "same_sample_block_window_all_input_heads_all_output_tiles",
            "recompute": "fixed + active_descriptor_capture + ordered_term_service + backend_stall",
            "replay": "controller + memo_read + builder_capture + ordered_term_service + backend_stall + commit",
            "replay_overlap": "excluded_from_promotion",
            "common_accumulator_boundary": "B2v_cross_head_preserve",
            "common_final_vector_reads": 450,
            "common_scalar_serializer_cycles": 14400,
            "gasr2cp": "two source-resident contexts per bank; cross-head preserve remains model-only",
            "erm7": "7 KiB, 512x112-bit, head-order critical-only admission; known nonresident heads recompute",
        },
        "rtl_calibration": {
            "calibration_rows": 200,
            "median_fixed_cycles": 459,
            "conservative_fixed_cycles": 475,
            "heldout_direct_sequential_mae": 2.68,
            "heldout_gasr_sequential_mae": 2.43,
            "heldout_direct_v2_mae": 157.74,
            "heldout_gasr_v2_mae": 167.35,
            "evidence": "[rtl校准]+[模型校准]",
        },
        "bootstrap_trials": evaluator.BOOTSTRAP_TRIALS,
        "bootstrap_seed": evaluator.BOOTSTRAP_SEED,
        "bootstrap_units": ["sample", "sequence"],
        "candidate_comparisons": evaluator.CANDIDATE_COMPARISONS,
        "familywise_alpha": evaluator.FAMILYWISE_ALPHA,
        "bonferroni_alpha_per_candidate": evaluator.BONFERRONI_ALPHA,
        "promotion_speedup_lower_bound": evaluator.PROMOTION_SPEEDUP,
        "promotion_rule": {
            "scenario_intersection": "median459 and max475 must both pass",
            "lower_bound": "min(sample and sequence Bonferroni one-sided family-wise 95% lower bounds) >= 1.20",
            "tail": "overall and every-stage inverse-probability-weighted window p95 must not regress in either scenario",
            "meaning": "pass only permits minimal RTL; no architecture or PPA claim",
        },
        "source_bindings": {
            name: {"path": relative(root, path), "sha256": sha256(path)}
            for name, path in sources.items()
        },
        "excluded_from_promotion": [
            "v2 max-overlap cycles",
            "ideal FCSR recompute/replay overlap without integrated timing RTL",
            "GASR2C-P described as implemented before preserve RTL exists",
            "common readout or serializer claimed as timing-locked",
            "post-profile fixed-cycle or admission parameter changes",
            "OpenROAD/Yosys proxies described as ASIC PPA",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--prereg",
        type=Path,
        default=root / "contracts/local5_joint_candidate_prereg_v3_20260810.json",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=root
        / "contracts/local5_joint_candidate_prereg_v3_receipt_20260810.json",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.force and (args.prereg.exists() or args.receipt.exists()):
        raise FileExistsError("v3预注册或收据已存在；拒绝静默覆盖")
    frozen_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    args.prereg.parent.mkdir(parents=True, exist_ok=True)
    prereg = build_prereg(root, frozen_at)
    atomic_write(
        args.prereg,
        json.dumps(prereg, ensure_ascii=False, indent=2) + "\n",
    )
    oid = subprocess.run(
        ["git", "hash-object", "-w", str(args.prereg)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    blob = subprocess.run(
        ["git", "cat-file", "blob", oid],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    if blob != args.prereg.read_bytes():
        raise AssertionError("Git blob锚点与预注册字节不一致")
    receipt = {
        "schema": "local5_joint_candidate_prereg_receipt_v3",
        "status": "GIT_BLOB_ANCHORED_BEFORE_PROFILE",
        "created_at_utc": frozen_at,
        "profile_status_at_anchor": "GPU_WAITING_NO_FORMAL_PAYLOAD",
        "prereg": relative(root, args.prereg),
        "prereg_sha256": sha256(args.prereg),
        "git_blob_oid": oid,
        "verification": f"git cat-file blob {oid} must equal prereg bytes",
    }
    atomic_write(
        args.receipt,
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
