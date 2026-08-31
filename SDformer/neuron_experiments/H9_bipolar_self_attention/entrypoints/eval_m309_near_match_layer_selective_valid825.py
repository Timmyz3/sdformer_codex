#!/usr/bin/env python3
"""Fail-closed launcher for the one frozen M309 valid825 candidate."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[3]
M284_WRAPPER = (REPO / "neuron_experiments/H9_bipolar_self_attention/"
                "entrypoints/eval_m284_near_match_residual_elision.py")
SELECTIVE_MODULE = (
    REPO / "neuron_experiments/H9_bipolar_self_attention/overlay/models/"
    "STSwinNet_SNN/near_match_residual_elision_selective.py")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def option(name):
    require(name in sys.argv, "M309 missing " + name)
    index = sys.argv.index(name)
    require(index + 1 < len(sys.argv), "M309 missing value for " + name)
    return index, sys.argv[index + 1]


def main():
    launcher_start = sha256(Path(__file__).resolve())
    _, contract_raw = option("--contract")
    contract_path = Path(contract_raw).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    require(contract.get("schema") ==
            "m284_near_match_modified_forward_contract_v1" and
            contract.get("milestone") == "M309_VALID825",
            "M309 contract schema/milestone drift")
    runtime = contract["runtime_identity"]
    require(launcher_start == runtime["launcher_sha256"],
            "M309 launcher SHA drift")
    require(sha256(M284_WRAPPER) == runtime["wrapper_sha256"],
            "M309 pinned M284 wrapper SHA drift")
    require(sha256(SELECTIVE_MODULE) == runtime["module_sha256"],
            "M309 selective module SHA drift")

    freeze = contract["inputs"]["selection_freeze"]
    freeze_path = REPO / freeze["path"]
    require(freeze_path.is_file() and sha256(freeze_path) == freeze["sha256"],
            "M309 selection-freeze SHA drift")
    freeze_value = json.loads(freeze_path.read_text(encoding="utf-8"))
    correction = freeze_value["selection_rule_correction"]
    require(correction["unique_enabled_operator_indices"] == [0, 2, 3] and
            correction["dropped_operator_index"] == 1 and
            correction["further_s10_combination_search_allowed"] is False,
            "M309 selection freeze semantic drift")

    enabled_index, enabled_raw = option("--enabled-operator-indices")
    enabled = [int(value) for value in enabled_raw.split(",") if value != ""]
    require(enabled == [0, 2, 3] and
            enabled == contract["policy"]["enabled_operator_indices"],
            "M309 permits only frozen enable set [0,2,3]")
    _, threshold_raw = option("--distance-threshold")
    _, samples_raw = option("--max-samples")
    _, bn_raw = option("--bn-policy")
    require(int(threshold_raw) == 1 and int(samples_raw) == 0 and
            bn_raw == "running",
            "M309 requires tau1, full valid825, and running BN")
    del sys.argv[enabled_index:enabled_index + 2]
    os.environ["M306_ENABLED_OPERATOR_INDICES"] = "0,2,3"

    spec = importlib.util.spec_from_file_location("m309_pinned_m284_wrapper",
                                                  str(M284_WRAPPER))
    require(spec is not None and spec.loader is not None,
            "M309 cannot import pinned M284 wrapper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MODULE = SELECTIVE_MODULE
    module.main()
    require(sha256(Path(__file__).resolve()) == launcher_start,
            "M309 launcher changed during execution")


if __name__ == "__main__":
    main()
