#!/usr/bin/env python3
"""Synthetic mapping, debt-domain and policy tests for M1554."""

from __future__ import print_function

import importlib.util
from pathlib import Path


SOURCE = Path(__file__).resolve().parent.parent / "scripts/analyze_m1554_ep34_s2_ccbs_destination_debt_successor.py"
SPEC = importlib.util.spec_from_file_location("m1554", str(SOURCE))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def rejects(function):
    try:
        function()
    except Exception:
        return
    raise AssertionError("attack accepted")


def main():
    attacks = []
    assert M.destination_sources(0, 0, 3, 3) == ((0, 0, 1, 1),)
    interior = M.destination_sources(1, 1, 3, 3)
    assert len(interior) == 4
    assert set(interior) == set(((1, 1, 0, 0), (1, 0, 0, 2),
                                 (0, 1, 2, 0), (0, 0, 2, 2)))
    attacks.append("exact_k3s2_mapping")
    rejects(lambda: M.destination_sources(-10, -10, 3, 3))
    attacks.append("empty_destination")

    # M1547 accepted each source-local 0.09 against epsilon 0.1.  M1554 first
    # accumulates the four contributors, so 0.36 is kept against a unit budget.
    local = [M.B.fixed_order_drop([0.09], 0.1, 1.0)[0][0]
             for _unused in range(4)]
    assert local == [True, True, True, True]
    mask, debt = M.B.fixed_order_drop([4.0 * 0.09], 0.1, 1.0)
    assert mask == [False] and debt == 0.0
    attacks.append("destination_debt_counterexample_closed")

    mask, debt = M.B.fixed_order_drop([0.0, 0.2], 0.0, 1.0)
    assert mask == [True, False] and debt == 0.0
    attacks.append("zero_epsilon")
    contract = M.B.strict_json(M.CONTRACT)
    assert contract["required_accounting"]["owner"] == "destination_x_output_tile"
    assert contract["required_accounting"]["maximum_interior_spatial_contributors"] == 4
    assert all(contract["claim_boundary"][name] is False for name in
               ["aee", "capture_executed", "cycles", "traffic", "speedup",
                "energy", "rtl", "eda", "paper_headline"])
    attacks.append("contract_boundary")
    assert M.sha256(M.BASE_SOURCE) == M.BASE_SOURCE_SHA256
    attacks.append("base_identity")
    assert len(attacks) == 6
    print("PASS_M1554_SYNTHETIC_TEST attacks=6 destination_owned=true cpython_compatible=true")


if __name__ == "__main__":
    main()
