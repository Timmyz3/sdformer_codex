#!/usr/bin/env python3
"""Negative identity tests for the frozen production M129 analyzer."""

import hashlib
import importlib.util
import json
import pathlib
import sys


REVIEW = pathlib.Path(__file__).resolve().parent
HW = REVIEW.parent.parent
ANALYZER = HW / (
    "system_simulator/scripts/"
    "analyze_m129_row_admission_bubble_and_descriptor_cost.py")
M122_RESULT = HW / (
    "results/m122_w384_row_synchronous_source_fold_dse_r1_20260824/"
    "m122_w384_row_synchronous_source_fold_dse.json")
M109_SCRIPT = HW / (
    "system_simulator/scripts/"
    "analyze_m109_r2_window_storage_dual_timeline_frontier.py")
PRODUCTION_RESULT = HW / (
    "results/m129_row_admission_bubble_and_descriptor_cost_r1_20260824/"
    "m129_row_admission_bubble_and_descriptor_cost.json")
OVERLAY = HW / (
    "contracts/m128_r1_independent_review_correction_overlay_r1_20260824.json")
EXPECTED_ANALYZER = "b755cc5492f6fabde359363454566265cdcd26d146c8984acc4a8e45764f66e1"
EXPECTED_RESULT = "2443a651675763c9e867a2186e83440c323cf20e381e7a49724d6cb0d9ab411e"
EXPECTED_OVERLAY = "e646cc71cc62ce0d50c128c1a57db9a59221909948413ad3493bfc23cf3d44ec"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise SystemExit("FAIL " + message)


def load_analyzer():
    spec = importlib.util.spec_from_file_location("m129_negative", str(ANALYZER))
    require(spec is not None and spec.loader is not None,
            "cannot load M129 analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


require(sha256(ANALYZER) == EXPECTED_ANALYZER, "analyzer SHA drift")
require(sha256(PRODUCTION_RESULT) == EXPECTED_RESULT, "result SHA drift")
require(sha256(OVERLAY) == EXPECTED_OVERLAY, "overlay SHA drift")

# Direct frozen input drift must be rejected before output creation.
direct_drift = REVIEW / "negative_m122_result_identity_drift.json"
direct_drift.write_bytes(M122_RESULT.read_bytes() + b"\n")
direct_output = REVIEW / "negative_direct_drift_output"
require(not direct_output.exists(), "stale direct negative output")
module = load_analyzer()
original_m122_result = module.M122_RESULT
module.M122_RESULT = direct_drift
sys.argv = [str(ANALYZER), "--output", str(direct_output)]
direct_message = None
try:
    module.main()
except ValueError as error:
    direct_message = str(error)
finally:
    module.M122_RESULT = original_m122_result
require(direct_message == "frozen input identity drift: m122_result",
        "direct M122 result drift was not rejected exactly")
require(not direct_output.exists(), "direct drift created output")

# A semantically identical but SHA-drifted transitive M109 script is injected
# after the frozen M122 script itself has passed its direct hash check.  The
# production M129 analyzer currently executes it without checking its SHA.
transitive_drift = REVIEW / "negative_m109_transitive_identity_drift.py"
transitive_drift.write_bytes(
    M109_SCRIPT.read_bytes()
    + b"\n# REVIEW NEGATIVE IDENTITY DRIFT; semantic no-op.\n")
transitive_output = REVIEW / "negative_transitive_drift_output"
require(not transitive_output.exists(), "stale transitive negative output")
module = load_analyzer()
original_loader = module.load_module


def injecting_loader(label, path):
    loaded = original_loader(label, path)
    if label == "m129_frozen_m122":
        loaded.M109_SCRIPT = transitive_drift
    return loaded


module.load_module = injecting_loader
sys.argv = [str(ANALYZER), "--output", str(transitive_output)]
module.main()
transitive_result = (
    transitive_output / "m129_row_admission_bubble_and_descriptor_cost.json")
require(transitive_result.is_file(), "transitive drift did not create output")
require(sha256(transitive_result) == EXPECTED_RESULT,
        "transitive drift changed numeric result")
require(sha256(transitive_drift) != sha256(M109_SCRIPT),
        "transitive drift SHA did not change")

analyzer_text = ANALYZER.read_text(encoding="utf-8")
overlay_referenced = (EXPECTED_OVERLAY in analyzer_text
                      or OVERLAY.name in analyzer_text)
require(not overlay_referenced,
        "expected overlay-unpinned negative condition disappeared")

receipt = {
    "schema": "m129_identity_negative_tests_v1",
    "status": "MIXED_DIRECT_FAIL_CLOSED_TRANSITIVE_FAIL_OPEN",
    "direct_m122_result_drift": {
        "rejected": True,
        "message": direct_message,
        "output_created": False,
    },
    "transitive_m109_script_drift": {
        "rejected": False,
        "analyzer_passed": True,
        "numeric_result_exact_match": True,
        "production_m109_sha256": sha256(M109_SCRIPT),
        "drift_m109_sha256": sha256(transitive_drift),
        "interpretation": (
            "M129 checks the frozen M122 script/result but does not execute "
            "M122 main's transitive M109 SHA checks before replaying live helpers."
        ),
    },
    "m128_correction_overlay": {
        "sha256": EXPECTED_OVERLAY,
        "referenced_by_analyzer": False,
        "pinned_by_analyzer": False,
    },
}
(REVIEW / "m129_identity_negative_tests.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("PASS M129 direct drift rejected; transitive drift accepted fail_open=true "
      "overlay_pinned=false")
