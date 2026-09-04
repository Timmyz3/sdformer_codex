#!/usr/bin/env python3
"""Run checkpoint-bound ep44 Local5 component RTL without changing profile identity."""

from __future__ import annotations

import json
import os
from pathlib import Path

import run_local5_ep44_hardware_rebind as profile_flow


HW_ROOT = Path(__file__).resolve().parents[1]
POSTSCORE_BUILD = HW_ROOT / "build_new_arch/local5_ep44_hardware_rebind_20260815_postscore"
SAFE_POSTSCORE_ENTRY = (
    HW_ROOT / "sim_new_arch/run_local5_qgasr2c_fivebank_checks_ranked.sh"
)
SAFE_INTEGRATED_ENTRY = (
    HW_ROOT / "sim_new_arch/run_local5_score_projection_checks_ranked.sh"
)


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def verify_frozen_profile_runner() -> None:
    identity = load_json(profile_flow.RUN_IDENTITY)
    binding = (identity.get("source_bindings") or {}).get(
        "ranked_rebind_runner"
    ) or {}
    if (
        Path(str(binding.get("path", ""))).resolve()
        != Path(profile_flow.__file__).resolve()
        or binding.get("sha256")
        != profile_flow.file_sha256(Path(profile_flow.__file__))
    ):
        raise RuntimeError("ranked profile runner drifted after post-G0 capture")


def main() -> int:
    verify_frozen_profile_runner()
    profile_flow.validate_acceptance()
    for path in (POSTSCORE_BUILD, profile_flow.SCORE_PROJECTION_BUILD):
        if path.exists():
            raise RuntimeError(f"ranked component RTL build already exists: {path}")

    profile_flow.run(
        [
            profile_flow.PYTHON,
            "scripts/generate_local5_active_projection_postg0_vectors.py",
            "--input-dir",
            str(profile_flow.OUTPUT),
            "--output-dir",
            str(profile_flow.POSTSCORE_VECTORS),
            "--per-stage",
            "25",
            "--out-dim",
            "2",
            "--weight-mode",
            "checkpoint_theta_folded_dyadic_int8_head_slice",
        ],
        "ep44 real-weight post-score vectors",
    )
    env = os.environ.copy()
    env.update(
        {
            "BUILD_ROOT": str(POSTSCORE_BUILD),
            "RESULT_DIR": str(profile_flow.POSTSCORE_RESULTS),
            "VECTOR_DIR": str(profile_flow.POSTSCORE_VECTORS),
            "CHECKPOINT_WEIGHTS": "1",
        }
    )
    profile_flow.run(
        ["bash", str(SAFE_POSTSCORE_ENTRY.relative_to(HW_ROOT))],
        "ep44 post-score projection RTL/SVA",
        env=env,
    )
    profile_flow.run(
        [
            profile_flow.PYTHON,
            "scripts/generate_local5_score_projection_vectors.py",
            "--postscore-vector-dir",
            str(profile_flow.POSTSCORE_VECTORS),
            "--output-dir",
            str(profile_flow.SCORE_PROJECTION_VECTORS),
        ],
        "ep44 raw-QK score-to-projection vectors",
    )
    env = os.environ.copy()
    env.update(
        {
            "BUILD_DIR": str(profile_flow.SCORE_PROJECTION_BUILD),
            "RESULT_DIR": str(profile_flow.SCORE_PROJECTION_RESULTS),
            "VECTOR_DIR": str(profile_flow.SCORE_PROJECTION_VECTORS),
            "POSTSCORE_REPORT": str(
                profile_flow.POSTSCORE_RESULTS / "report.json"
            ),
        }
    )
    profile_flow.run(
        ["bash", str(SAFE_INTEGRATED_ENTRY.relative_to(HW_ROOT))],
        "ep44 score/Shiftmax5-to-source-owned-TCFM5-Acc32 RTL/SVA",
        env=env,
    )
    report = load_json(profile_flow.SCORE_PROJECTION_RESULTS / "report.json")
    if report.get("status") != "PASS" or int(report.get("groups", 0)) != 100:
        raise RuntimeError("ep44 integrated score/projection RTL report failed")
    profile_flow.record("ALL COMPLETE Local5 ep44 checkpoint-bound component RTL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
