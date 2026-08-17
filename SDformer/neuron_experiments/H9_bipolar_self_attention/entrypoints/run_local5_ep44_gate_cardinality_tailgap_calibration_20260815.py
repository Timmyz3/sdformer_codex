#!/usr/bin/env python3
"""Calibrate the Local5 C<=2 tail-gap proxy on one real Q7 batch."""

from __future__ import annotations

from pathlib import Path

import run_local5_ep44_gate_cardinality_calibration_20260815 as calibration


calibration.PROXY_MODE = "tail_gap_c2"
calibration.EXPERIMENT = "dsec_fullres_w15_local5_ep44_gatecard_tailgap_q7_calibration1"
calibration.CONFIG = calibration.GENERATED / f"{calibration.EXPERIMENT}.yml"
calibration.ROOT = calibration.RESULTS / f"{calibration.EXPERIMENT}_20260815"
calibration.STATUS = calibration.ROOT / "status.log"
calibration.LOG = calibration.ROOT / "train.log"
calibration.RECEIPT = calibration.ROOT / "calibration_receipt.json"
calibration.LOCK = Path("/tmp/sdformer_local5_ep44_gatecard_tailgap_calibration.lock")


if __name__ == "__main__":
    raise SystemExit(calibration.main())
