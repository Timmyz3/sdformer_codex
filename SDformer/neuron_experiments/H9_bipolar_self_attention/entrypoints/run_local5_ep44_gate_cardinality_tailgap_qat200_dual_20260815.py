#!/usr/bin/env python3
"""Run control/tail-gap-r010 200-step Local5 Q7 QAT."""

from __future__ import annotations

from pathlib import Path

import run_local5_ep44_gate_cardinality_qat20_sweep_20260815 as sweep


sweep.STEPS = 200
sweep.CALIBRATION_ROOT = sweep.RESULTS / (
    "dsec_fullres_w15_local5_ep44_gatecard_tailgap_q7_calibration1_20260815"
)
sweep.CALIBRATION = sweep.CALIBRATION_ROOT / "calibration_receipt.json"
sweep.EXPERIMENT_PREFIX = "dsec_fullres_w15_local5_ep44_gatecard_tailgap"
sweep.ROOT = sweep.RESULTS / "local5_ep44_gatecard_tailgap_qat200_dual_20260815"
sweep.STATUS = sweep.ROOT / "status.log"
sweep.SUMMARY = sweep.ROOT / "summary.json"
sweep.LOCK = Path("/tmp/sdformer_local5_ep44_gatecard_tailgap_qat200_dual.lock")


def dual_specs(calibration: dict) -> list[tuple[str, float, float]]:
    candidates = calibration["candidate_lambdas"]
    return [
        ("control", 0.0, 0.0),
        ("r010", 0.010, float(candidates["flow_loss_ratio_0.01"])),
    ]


sweep.branch_specs = dual_specs


if __name__ == "__main__":
    raise SystemExit(sweep.main())
