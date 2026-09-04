from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prove_dual_line_optimality_bounds",
    ROOT / "scripts/prove_dual_line_optimality_bounds.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_motion_pair_local_lower_bound() -> None:
    equal = MODULE.motion_bound(10, 7, 13)
    assert equal["descriptor_lower_bound"] == 13
    assert equal["attains_descriptor_lower_bound"]
    assert equal["temporal_membership_bit_lower_bound"] == 2


def test_local5_cross_bounds() -> None:
    coloring = {
        "banks": 5,
        "conflict_free_all_neighborhoods": True,
        "interior_k5_witnesses": 1,
        "injective_bank_address": True,
    }
    result = MODULE.local5_bounds(
        [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], 3, 5, coloring
    )
    assert result["row_span_lower_bound"] == 3
    assert result["attains_row_span_lower_bound"]
    assert result["bank_count_lower_bound"] == 5
    assert result["attains_bank_count_lower_bound"]


def test_locked_artifacts_attain_all_bounds() -> None:
    report = MODULE.build_report(
        MODULE.DEFAULT_FAIR_LOG.read_text(encoding="utf-8"),
        __import__("json").loads(MODULE.DEFAULT_COMPILER.read_text(encoding="utf-8")),
        __import__("json").loads(MODULE.DEFAULT_COLORING.read_text(encoding="utf-8")),
    )
    assert report["status"] == "PASS"
    assert report["motion"]["actual_rqtb_descriptors"] == 34099
    assert report["local5"]["actual_row_span"] == 3
