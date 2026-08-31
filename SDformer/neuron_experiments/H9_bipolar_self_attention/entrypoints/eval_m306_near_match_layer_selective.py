#!/usr/bin/env python3
"""Fail-closed launcher for layer-selective M284 modified-forward screens."""

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
    require(name in sys.argv, "M306 missing " + name)
    index = sys.argv.index(name)
    require(index + 1 < len(sys.argv), "M306 missing value for " + name)
    return index, sys.argv[index + 1]


def main():
    launcher_start = sha256(Path(__file__).resolve())
    contract_index, contract_raw = option("--contract")
    del contract_index
    contract_path = Path(contract_raw).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    require(contract.get("schema") ==
            "m284_near_match_modified_forward_contract_v1" and
            contract.get("milestone") == "M306",
            "M306 contract schema/milestone drift")
    runtime = contract["runtime_identity"]
    require(launcher_start == runtime["launcher_sha256"],
            "M306 launcher SHA drift")
    require(sha256(M284_WRAPPER) == runtime["wrapper_sha256"],
            "M306 pinned M284 wrapper SHA drift")
    require(sha256(SELECTIVE_MODULE) == runtime["module_sha256"],
            "M306 selective module SHA drift")

    enabled_index, enabled_raw = option("--enabled-operator-indices")
    enabled = [int(value) for value in enabled_raw.split(",") if value != ""]
    allowed = contract["policy"]["allowed_enabled_operator_indices"]
    require(enabled in allowed, "M306 operator enable set outside frozen DSE")
    del sys.argv[enabled_index:enabled_index + 2]
    os.environ["M306_ENABLED_OPERATOR_INDICES"] = ",".join(
        str(value) for value in enabled)

    spec = importlib.util.spec_from_file_location("m306_pinned_m284_wrapper",
                                                  str(M284_WRAPPER))
    require(spec is not None and spec.loader is not None,
            "M306 cannot import pinned M284 wrapper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MODULE = SELECTIVE_MODULE
    module.main()
    require(sha256(Path(__file__).resolve()) == launcher_start,
            "M306 launcher changed during execution")


if __name__ == "__main__":
    main()
