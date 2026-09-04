#!/usr/bin/env python3
"""Direct and transitive identity negative tests for M132."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


REVIEW = Path(__file__).resolve().parent
HW = REVIEW.parent.parent
ANALYZER = HW / ("system_simulator/scripts/"
                 "analyze_m132_dualrow512_pwp_compact_k4_schedule.py")
M129_RESULT = HW / ("results/"
                    "m129_row_admission_bubble_and_descriptor_cost_r1_20260824/"
                    "m129_row_admission_bubble_and_descriptor_cost.json")
M109_RESULT = HW / ("results/"
                    "m109_r2_window_storage_dual_timeline_frontier_r1_20260824/"
                    "m109_r2_window_storage_dual_timeline_frontier.json")
SEALED_RESULT = HW / ("results/"
                      "m132_dualrow512_pwp_compact_k4_schedule_r1_20260824/"
                      "m132_dualrow512_pwp_compact_k4_schedule.json")
EXPECTED_ANALYZER = "f140b6b72559f04cdac374eaf696c3f6650b20d3b00bd580419b88494d89c952"
EXPECTED_RESULT = "f74444576ec487b9b1034aced7add0da868a9dea5d4185e0a62c1e33fe1ad755"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise SystemExit("FAIL " + message)


def load_analyzer():
    spec = importlib.util.spec_from_file_location("m132_negative", str(ANALYZER))
    require(spec is not None and spec.loader is not None,
            "cannot load analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


require(sha256(ANALYZER) == EXPECTED_ANALYZER, "analyzer SHA")
require(sha256(SEALED_RESULT) == EXPECTED_RESULT, "sealed result SHA")

# Direct M129 input drift is pinned and must reject before output creation.
direct_drift = REVIEW / "negative_m129_result_identity_drift.json"
direct_drift.write_bytes(M129_RESULT.read_bytes() + b"\n")
direct_output = REVIEW / "negative_direct_drift_output"
require(not direct_output.exists(), "stale direct output")
module = load_analyzer()
original_m129 = module.M129_RESULT
module.M129_RESULT = direct_drift
sys.argv = [str(ANALYZER), "--output", str(direct_output)]
direct_message = None
try:
    module.main()
except ValueError as error:
    direct_message = str(error)
finally:
    module.M129_RESULT = original_m129
require(direct_message == "frozen input identity drift: m129_result",
        "direct drift did not fail closed")
require(not direct_output.exists(), "direct drift output exists")

# M132 reads m122.M109_RESULT for the fixed8 baseline but does not include that
# file in frozen_paths/EXPECTED_SHA256.  Inject a JSON-valid semantic no-op SHA
# drift after the M122 script's own direct SHA check.
transitive_drift = REVIEW / "negative_m109_result_transitive_identity_drift.json"
transitive_drift.write_bytes(M109_RESULT.read_bytes() + b"\n")
transitive_output = REVIEW / "negative_transitive_drift_output"
require(not transitive_output.exists(), "stale transitive output")
module = load_analyzer()
original_loader = module.load_module


def injecting_loader(label, path):
    loaded = original_loader(label, path)
    if label == "m132_frozen_m122":
        loaded.M109_RESULT = transitive_drift
    return loaded


module.load_module = injecting_loader
sys.argv = [str(ANALYZER), "--output", str(transitive_output)]
module.main()
transitive_result = transitive_output / SEALED_RESULT.name
require(transitive_result.is_file(), "transitive output missing")
require(sha256(transitive_result) == EXPECTED_RESULT,
        "transitive no-op drift changed output")
require(sha256(transitive_drift) != sha256(M109_RESULT),
        "transitive SHA did not drift")

receipt = {
    "schema": "m132_identity_negative_tests_v1",
    "status": "MIXED_DIRECT_FAIL_CLOSED_M109_RESULT_IDENTITY_FAIL_OPEN",
    "direct_m129_result_drift": {
        "rejected": True,
        "message": direct_message,
        "output_created": False,
    },
    "transitive_m109_result_drift": {
        "rejected": False,
        "analyzer_passed": True,
        "numeric_result_exact_match": True,
        "production_sha256": sha256(M109_RESULT),
        "drift_sha256": sha256(transitive_drift),
        "scope": (
            "M109 result supplies fixed-baseline service tokens. Semantic "
            "changes affecting compact256 are additionally constrained by "
            "the exact M129 recurrence equality, but exact-SHA provenance is open."
        ),
    },
}
(REVIEW / "m132_identity_negative_tests.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("PASS M132 direct drift rejected; unpinned M109 result drift accepted "
      "fail_open=true")
