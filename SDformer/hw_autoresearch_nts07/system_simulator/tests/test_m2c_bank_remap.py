from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_m2c_bank_remap.py"
SPEC = importlib.util.spec_from_file_location("analyze_m2c_bank_remap", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_modulo_issue_beats_matches_manual_bank_max() -> None:
    bits = np.zeros((2, 256), dtype=bool)
    bits[0, [0, 4, 8, 1]] = True
    bits[1, [2, 3, 4, 5]] = True
    assignment = MODULE.bank_assignment(256, 4, None)
    assert MODULE.issue_beats(bits, assignment, 4).tolist() == [3, 1]


@pytest.mark.parametrize("issue_width", [2, 4, 8])
def test_xor_remap_is_reversible_with_bank_local_address(issue_width: int) -> None:
    bank_bits = int(np.log2(issue_width))
    source = np.arange(256)
    for shift in range(bank_bits, 8):
        bank = MODULE.bank_assignment(256, issue_width, shift)
        local_address = source >> bank_bits
        recovered_low = bank ^ ((source >> shift) & (issue_width - 1))
        recovered = (local_address << bank_bits) | recovered_low
        assert np.array_equal(recovered, source)


def test_rejects_non_power_of_two_or_low_xor_shift() -> None:
    with pytest.raises(ValueError):
        MODULE.bank_assignment(256, 3, None)
    with pytest.raises(ValueError):
        MODULE.bank_assignment(256, 8, 2)
