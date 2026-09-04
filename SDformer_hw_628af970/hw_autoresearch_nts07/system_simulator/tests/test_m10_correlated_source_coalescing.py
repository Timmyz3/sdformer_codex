import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "analyze_m10_correlated_source_coalescing.py"
)
SPEC = importlib.util.spec_from_file_location("m10_coalescing", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_same_source_broadcasts_to_four_contexts():
    active = np.zeros((4, 32), dtype=bool)
    active[:, 3] = True
    result = MODULE.coalesced_issue(active, issue_width=16, reduce_slots=4)
    assert result == {
        "cycles": 1,
        "bank_reads": 1,
        "context_updates": 4,
        "multi_context_reads": 1,
    }


def test_disjoint_sources_in_same_bank_remain_serial():
    active = np.zeros((4, 64), dtype=bool)
    for context, source in enumerate((1, 17, 33, 49)):
        active[context, source] = True
    result = MODULE.coalesced_issue(active, issue_width=16, reduce_slots=4)
    assert result["cycles"] == 4
    assert result["bank_reads"] == 4
    assert result["context_updates"] == 4
    assert result["multi_context_reads"] == 0


def test_context_reducer_limit_is_preserved():
    active = np.zeros((1, 16), dtype=bool)
    active[0, :5] = True
    result = MODULE.coalesced_issue(active, issue_width=16, reduce_slots=4)
    assert result["cycles"] == 2
    assert result["bank_reads"] == 5
    assert result["context_updates"] == 5


def test_broadcast_never_displaces_baseline_bank_grants():
    active = np.zeros((4, 64), dtype=bool)
    # Context 0 already consumes all four reducer slots from separate banks.
    active[0, (0, 1, 2, 3)] = True
    # The shared source in bank 0 may update context 1, but must not displace
    # any of context 0's four frozen baseline grants.
    active[1, 0] = True
    result = MODULE.coalesced_issue(active, issue_width=16, reduce_slots=4)
    assert result["cycles"] == 1
    assert result["bank_reads"] == 4
    assert result["context_updates"] == 5
    assert result["multi_context_reads"] == 1


def test_invalid_geometry_fails_closed():
    active = np.zeros((4, 16), dtype=bool)
    try:
        MODULE.coalesced_issue(active, issue_width=0, reduce_slots=4)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid issue width was accepted")
