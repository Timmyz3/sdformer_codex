#!/usr/bin/env python3
"""Independent source-only checker for M1007; no full replay and no EDA."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HERE / "m1007_c1_matched_common_charge_address_replay_source.py"
SOURCE_SHA = "150f22eaa11d219bfa20561b91a38049f14abbc541a6b40db04bd73533ec3442"
TEST = HW / "system_simulator/tests/test_m1007_c1_matched_common_charge_address_replay_source.py"
CONTRACT = HW / "contracts/m1007_m1000_c1_matched_common_charge_address_replay_source_contract_r1_20260829.json"
RESULT_GLOB = "m1007*m1000*c1*matched*replay*"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + value)))


def load_source():
    require(sha(SOURCE) == SOURCE_SHA, "M1007 source drift")
    spec = importlib.util.spec_from_file_location("m1007_source_checked", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_static(contract=CONTRACT):
    source_text = SOURCE.read_text()
    tree = ast.parse(source_text)
    imported = {alias.name.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names}
    require(not imported.intersection({"subprocess", "socket", "requests", "urllib"}),
            "execution/network module in source")
    calls = {node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    require(not calls.intersection({"system", "exec", "eval", "compile"}),
            "unsafe execution primitive in source")
    module = load_source()
    require(inspect.isgeneratorfunction(module.stream_parent_memh),
            "full parent replay is not streaming")
    require(module.ROWS_SHA ==
            "6e03352b89eff1955825334b4dedd991db8c975a9ef6662fe0317e73ccfa8334",
            "M410 identity constant drift")
    require(module.M528_RESULT_SHA ==
            "778c8e1bed6a19852c14bc61e00761f798008d67042b7a74efbaaffdde4b3de1",
            "M528 identity constant drift")
    require(module.M505_RESULT_SHA ==
            "b8a29f2fafc0e7d051d66ed206cd5c25efb866d4a1ab02082aa71bad4b14eb61",
            "M505 identity constant drift")
    value = strict_json(contract)
    require(value.get("status") == "PASS_M1007_SOURCE_ONLY__NO_FULL_REPLAY_NO_EDA" and
            value.get("launch_now") is False, "M1007 contract state drift")
    identity = value["source_identity"]
    require(identity["source"]["sha256"] == sha(SOURCE) and
            identity["tests"]["sha256"] == sha(TEST), "M1007 source/test identity drift")
    boundary = value["claim_boundary"]
    require(all(boundary[key] is False for key in
                ("full_51840000_replay_executed", "vcs_executed", "dc_executed",
                 "pt_executed", "ptpx_executed", "gpu_remote_used",
                 "matched_total_cycles", "rtl_speedup", "paper_ppa_ready")),
            "M1007 false execution/claim boundary")
    require(value["admission_gates"]["capacity_only_214912B_requires_complete_coverage"] is True,
            "214912B coverage gate weakened")
    require(value["admission_gates"]["m528_1p7467534301_remains_cpu_only"] is True,
            "M528 CPU-only boundary weakened")
    require(not list((HW / "results").glob(RESULT_GLOB)),
            "M1007 result exists despite source-only contract")
    oracle = module.small_oracle()
    require(oracle["status"] == "PASS_M1007_SMALL_ORACLE__NO_FULL_REPLAY" and
            oracle["asymmetric_charge_rejected"] is True and
            oracle["packing_negative_oracle"]["capacity_only_214912B_admitted"] is False,
            "M1007 small oracle drift")
    return {
        "schema": "m1007_m1000_c1_matched_common_charge_source_check_v1",
        "status": "PASS_M1007_SOURCE_CHECK__NO_FULL_REPLAY_NO_EDA",
        "source_sha256": sha(SOURCE),
        "test_sha256": sha(TEST),
        "contract_sha256": sha(contract),
        "parent_oracle_cases": len(oracle["parent_cases"]),
        "asymmetric_common_charge_rejected": True,
        "synthetic_packing_conflict_rejected": True,
        "full_51840000_replayed": False,
        "eda_gpu_remote_used": False,
        "matched_total_cycles": False,
        "capacity_only_214912B_admitted": False,
        "m528_1p7467534301_promoted": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    args = parser.parse_args()
    print(json.dumps(validate_static(args.contract), sort_keys=True))


if __name__ == "__main__":
    main()
