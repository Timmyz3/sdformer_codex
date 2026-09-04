from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "profile_h67_active_payload_compaction",
    ROOT / "scripts/profile_h67_active_payload_compaction.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_percentile_is_fail_closed_and_nearest_rank() -> None:
    assert MODULE.percentile([0, 1, 2, 3], 0.50) == 1
    assert MODULE.percentile([0, 1, 2, 3], 0.95) == 3
    try:
        MODULE.percentile([], 0.95)
    except ValueError:
        pass
    else:
        raise AssertionError("empty percentile input must fail")


def test_real_report_keeps_all_four_payload_macros() -> None:
    source = json.loads(MODULE.DEFAULT_INPUT.read_text(encoding="utf-8"))
    report = MODULE.build_report(source)
    assert report["sample0_rows"]["active_pairs_p95"] == 225
    assert report["sample0_rows"]["active_pairs_max"] == 225
    assert report["physical_capacity"]["baseline_macro_count"] == 4
    assert report["physical_capacity"]["candidate_macro_count"] == 4
    assert report["write_activity"]["packed_vs_sparse_addressed_incremental_payload_bits"] == 0
    assert report["verdict"] == "NO_GO_AS_DATE_CONTRIBUTION"
