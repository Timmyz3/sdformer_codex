from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "profile_local5_neighbor_score_object_reuse",
    ROOT / "scripts/profile_local5_neighbor_score_object_reuse.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_role_values_and_input_key() -> None:
    word = 3 | (5 << 32) | (7 << 64) | (11 << 96) | (13 << 128)
    assert MODULE.role_values(word, 32) == [3, 5, 7, 11, 13]
    assert MODULE.input_key(0x1F, 9, word) == (0x1F, 9, word)


def test_ident_k_respects_valid_mask() -> None:
    word = 7 | (7 << 32) | (9 << 64)
    assert MODULE.ident_k(word, 0b00011)
    assert not MODULE.ident_k(word, 0b00111)


def test_real_profile_is_exact_and_preserves_relation_slots() -> None:
    report = MODULE.build_profile(MODULE.DEFAULT_VECTOR_DIR)
    assert report["exactness"]["equal_input_score_gate_mismatches"] == 0
    assert report["rates"]["relation_slot_reduction"] == 0.0
    assert report["verdict"] == "NO_GO_AS_DATE_CONTRIBUTION"
