#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("analyze_atlif_lifetime_contract.py")
SPEC = importlib.util.spec_from_file_location("analyze_atlif_lifetime_contract", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_classifier() -> None:
    assert MODULE.classify("x.attn.attn_sn.spiking_neuron")[0] == "dead_debug"
    assert MODULE.classify("x.attn.proj_sn.spiking_neuron")[0] == "dual_consumer_fanout"
    assert MODULE.classify("x.attn.sn_q.spiking_neuron")[0] == "temporal_pair_assembly"
    assert MODULE.classify("x.mlp.sn1.spiking_neuron")[0] == "single_immediate_consumer"


def test_real_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    exp = root / "neuron_experiments/H9_bipolar_self_attention/results"
    result = MODULE.analyze(
        exp / "h67_ep19_true_ttb_profile100_20260712/atlif_activity.csv",
        "H67",
    )
    categories = {row["category"]: row for row in result["categories"]}
    assert result["called_modules"] == 93
    assert result["live_modules"] == 81
    assert result["dead_modules"] == 12
    assert result["live_output_elements_per_frame"] == 526_046_400
    assert categories["single_immediate_consumer"]["modules"] == 45
    assert categories["dual_consumer_fanout"]["modules"] == 12
    assert categories["temporal_pair_assembly"]["modules"] == 24
    assert categories["single_immediate_consumer"]["elements_per_frame"] == 421_536_960


if __name__ == "__main__":
    test_classifier()
    test_real_contract()
    print("2项ATLIF生命周期合同测试通过")
