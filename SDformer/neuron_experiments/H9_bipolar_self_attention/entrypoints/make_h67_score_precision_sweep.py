#!/usr/bin/env python3
"""Generate H67 ep35 score-precision sensitivity configs without changing RTL."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
GEN = EXP / "configs/generated"
RESULTS = EXP / "results"
SOURCE = GEN / "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_dyadic_q7q17_deploy.yml"
CHECKPOINT = RESULTS / (
    "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/checkpoint_epoch35.pth"
)
MANIFEST = GEN / "h67_ep35_score_precision_qf5_qf8_manifest.json"
FRACTIONAL_BITS = (5, 6, 7, 8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    for path in (SOURCE, CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(path)
    source = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    rows = []
    for bits in FRACTIONAL_BITS:
        config = deepcopy(source)
        experiment = f"dsec_fullres_w15_H67_ep35_score_qf{bits}_gate_q17_sensitivity"
        config["experiment"] = experiment
        attention = config["bsa_attention"]
        attention["hardware_quant_enabled"] = True
        attention["hardware_score_step"] = 1.0 / float(1 << bits)
        attention["hardware_score_min"] = -2.0
        attention["hardware_score_max"] = 2.0
        attention["hardware_gate_step"] = 1.0 / 128.0
        attention["hardware_gate_min"] = 0.0
        attention["hardware_gate_max"] = 2.0
        attention["hardware_rtl_shiftmax_enabled"] = False
        config.setdefault("runtime", {})["deployment_contract"] = {
            "scope": "algorithm_score_fractional_precision_sensitivity",
            "score_quantization": f"QF{bits}_step_2^-{bits}_range_-2_to_2",
            "shiftmax": "generic_float_exp2_ceil_pow2_after_score_quantization",
            "gate_quantization": "Q1.7_RNE_via_STE",
            "systemverilog_replay": False,
            "rtl_exists_for_this_precision": bits == 7,
            "warning": "QF denotes fractional bits, not total signed code width",
        }
        config["note"] = (
            f"H67 ep35 algorithm-only score precision sensitivity: QF{bits} score "
            "grid with fixed Q1.7 gate. Generic Shiftmax is used so QF5/QF6/QF8 "
            "must not be described as RTL-exact. Geometry and checkpoint are frozen."
        )
        path = GEN / f"{experiment}.yml"
        rendered = yaml.safe_dump(config, sort_keys=False, width=100)
        if path.exists() and path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"generated config drift: {path}")
        path.write_text(rendered, encoding="utf-8")
        rows.append(
            {
                "fractional_bits": bits,
                "score_step": 1.0 / float(1 << bits),
                "config": str(path.resolve()),
                "config_sha256": sha256(path),
                "result_dir": str((RESULTS / f"{experiment}_20260813").resolve()),
            }
        )
    payload = {
        "schema": "h67_ep35_score_precision_sweep_manifest_v1",
        "scope": "algorithm sensitivity only; no new RTL claim",
        "source_config": str(SOURCE.resolve()),
        "source_config_sha256": sha256(SOURCE),
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "rows": rows,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
