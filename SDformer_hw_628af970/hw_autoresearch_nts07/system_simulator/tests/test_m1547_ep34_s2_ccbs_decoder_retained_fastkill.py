#!/usr/bin/env python3
"""Synthetic and policy attacks for the M1547 CCBS retained-data screen."""

from __future__ import print_function

import importlib.util
import json
import shutil
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "scripts/analyze_m1547_ep34_s2_ccbs_decoder_retained_fastkill.py"
spec = importlib.util.spec_from_file_location("m1547_source", str(SOURCE))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def expect_failure(function, label):
    try:
        function()
    except Exception:
        return label
    raise AssertionError("attack unexpectedly passed: " + label)


def main():
    attacks = []
    module.validate_block_configs(module.BLOCK_CONFIGS)
    attacks.append(expect_failure(lambda: module.validate_block_configs(
        ((16, 16), (8, 16), (32, 16))), "block_order"))
    attacks.append(expect_failure(lambda: module.validate_block_configs(
        ((8, 16), (16, 16))), "block_missing"))

    account = module.metadata_account(770, 192, 9, 16, 16)
    assert account["metadata_to_int8_weight_bytes"] < 0.02
    assert account["reduction_vs_old_g11"] >= 8.0
    weak = module.metadata_account(7, 16, 1, 8, 16)
    assert weak["reduction_vs_old_g11"] < 8.0
    attacks.append("metadata_gate")

    mask, debt = module.fixed_order_drop([0.0, 1.0, 2.0], 0.0, 10.0)
    assert mask == [True, False, False] and debt == 0.0
    attacks.append("zero_epsilon")
    mask, debt = module.fixed_order_drop([0.4, 0.7, 0.2], 0.1, 10.0)
    assert mask == [True, False, True] and abs(debt - 0.6) < 1.0e-12
    attacks.append("fixed_order_debt")
    attacks.append(expect_failure(lambda: module.fixed_order_drop(
        [float("nan")], 0.1, 1.0), "nonfinite_bound"))

    assert module.dynamic_witness_count({"a": 1, "b": 2, "c": 3}) == 1
    assert module.dynamic_witness_count({"a": 1, "b": 2}) == 0
    attacks.append("dynamic_witness")

    with tempfile.TemporaryDirectory(prefix="m1547_attacks_") as temp:
        root = Path(temp)
        payload = root / "payload.bin"
        payload.write_bytes(bytes([1, 2, 3]))
        expected = module.sha256(payload)
        payload.write_bytes(bytes([1, 2, 4]))
        attacks.append(expect_failure(lambda: module.require(
            module.sha256(payload) == expected, "payload SHA drift"), "sha_mutation"))
        missing = root / "missing.bin"
        attacks.append(expect_failure(lambda: module.require(
            missing.is_file(), "missing input"), "missing_input"))

    contract = json.loads(module.CONTRACT.read_text())
    assert contract["block_configs"] == [list(row) for row in module.BLOCK_CONFIGS]
    assert contract["epsilon_grid"] == list(module.EPSILON_GRID)
    assert contract["claim_boundary"] == module.CLAIM_BOUNDARY
    assert all(contract["claim_boundary"][name] is False for name in
               ["aee", "cycles", "speedup", "traffic", "energy", "rtl", "eda"])
    attacks.append("contract_boundary")

    assert len(attacks) == 10
    print("PASS_M1547_SYNTHETIC_TEST attacks={} cpython_compatible=true".format(len(attacks)))


if __name__ == "__main__":
    main()
