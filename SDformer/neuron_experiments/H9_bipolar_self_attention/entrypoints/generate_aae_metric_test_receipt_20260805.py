#!/usr/bin/env python3
"""Run AAE metric tests and bind the result to the evaluated source files."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
BASELINE = REPO / "third_party/SDformerFlow"
METRIC_SOURCE = BASELINE / "loss/flow_supervised.py"
EVALUATOR_SOURCE = BASELINE / "eval_DSEC_flow_SNN.py"
TEST_SOURCE = BASELINE / "tests/test_aae_metrics.py"
AGGREGATION_SOURCE = BASELINE / "utils/metric_aggregation.py"
AGGREGATION_TEST_SOURCE = BASELINE / "tests/test_metric_aggregation.py"
OUTPUT = REPO / "neuron_autoresearch/AAE_METRIC_TEST_RECEIPT_20260805.json"
OUTPUT_MD = REPO / "neuron_autoresearch/AAE_METRIC_TEST_RECEIPT_20260805.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASELINE)
    outputs = []
    for test_source in (TEST_SOURCE, AGGREGATION_TEST_SOURCE):
        result = subprocess.run(
            [sys.executable, str(test_source)],
            cwd=BASELINE,
            env=env,
            capture_output=True,
            text=True,
        )
        combined_test = result.stdout + result.stderr
        if result.returncode or "Ran 4 tests" not in combined_test or "OK" not in combined_test:
            raise RuntimeError(f"AAE metric tests failed:\n{combined_test}")
        outputs.append(combined_test.strip())
    combined = "\n\n".join(outputs)
    evaluator = EVALUATOR_SOURCE.read_text(encoding="utf-8")
    required_evaluator_tokens = (
        'if eval_batch_size != 1:',
        'val_results[metric]["it"] += 1',
        'val_results[metric]["metric"] / val_results[metric]["it"]',
        '"masked_mean_per_frame_then_equal_mean_over_validation_frames"',
        '"metric_aggregation_audit": metric_aggregation_summary',
        'results["DSEC_Fl"]',
    )
    missing = [token for token in required_evaluator_tokens if token not in evaluator]
    if missing:
        raise RuntimeError(f"AAE evaluator aggregation contract drift: {missing}")
    receipt = {
        "schema": "aae_metric_test_receipt_v2",
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "test_count": 8,
        "test_output": combined.strip(),
        "contracts": {
            "legacy_aae": "2d_direction_angle_degrees_between_uv",
            "benchmark_ae": "middlebury_barron_3d_angle_degrees_between_normalized_uv1",
            "aggregation": "masked_mean_per_frame_then_equal_mean_over_validation_frames",
            "dsec_fl": "percentage_epe_gt_3px_and_gt_5pct_of_ground_truth_flow_magnitude",
            "eval_batch_size": 1,
            "audited_aggregations": [
                "frame_equal_mean", "pixel_global_mean", "sequence_balanced_mean"
            ],
        },
        "sources": {
            "metric": {"path": str(METRIC_SOURCE), "sha256": sha256(METRIC_SOURCE)},
            "evaluator": {
                "path": str(EVALUATOR_SOURCE),
                "sha256": sha256(EVALUATOR_SOURCE),
            },
            "tests": {"path": str(TEST_SOURCE), "sha256": sha256(TEST_SOURCE)},
            "aggregation": {
                "path": str(AGGREGATION_SOURCE),
                "sha256": sha256(AGGREGATION_SOURCE),
            },
            "aggregation_tests": {
                "path": str(AGGREGATION_TEST_SOURCE),
                "sha256": sha256(AGGREGATION_TEST_SOURCE),
            },
        },
    }
    OUTPUT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text(
        "\n".join(
            [
                "# AAE Metric Test Receipt",
                "",
                "Status: **PASS**",
                "",
                "- Tests: 8/8",
                "- Legacy AAE: 2-D direction angle.",
                "- Benchmark AE: Barron/Middlebury `(u,v,1)` angle.",
                "- DSEC Fl-all: GT-magnitude 3px/5% criterion, reported in percent.",
                "- Aggregation: masked frame mean, then equal mean over frames.",
                "- Audit aggregations: frame-equal, pixel-global, sequence-balanced.",
                "- Evaluation batch size: 1.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
