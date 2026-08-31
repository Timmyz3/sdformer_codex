#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Exact-pinned launch-time tests for the M1433 runtime-present state."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
CHECKER = HERE / "check_m1433_c1_r16_vcs_runtime_split_source.py"
SPEC = importlib.util.spec_from_file_location("m1433_runtime_checker", CHECKER)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def changed(value):
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is str:
        return "M1433_RUNTIME_MUTATED"
    if type(value) is dict:
        result = dict(value); result["m1433_extra"] = True; return result
    raise TypeError(type(value))


def predecessor_regression_cases():
    base = M.expected_contract()
    cases = [
        ("contract_extra_top_level", lambda d: d.__setitem__("m1433_extra", True)),
        ("contract_date_changed", lambda d: d.__setitem__("date", "2099-01-01")),
        ("contract_future_execution_removed", lambda d: d.pop("future_execution")),
        ("contract_future_execution_extra", lambda d:
         d["future_execution"].__setitem__("m1433_extra", True)),
    ]
    cases.extend(("future_execution_" + key, lambda d, key=key:
                  d["future_execution"].__setitem__(key, changed(d["future_execution"][key])))
                 for key in base["future_execution"])
    cases.extend([
        ("author_execution_extra", lambda d:
         d["author_execution"].__setitem__("m1433_extra", False)),
        ("claim_boundary_extra", lambda d:
         d["claim_boundary"].__setitem__("m1433_extra", False)),
    ])
    if len(cases) != 16:
        raise AssertionError(len(cases))
    return cases


def validate_contract_regressions() -> dict[str, int]:
    rejected = 0
    for _, mutate in predecessor_regression_cases():
        candidate = copy.deepcopy(M.expected_contract()); mutate(candidate)
        try:
            M.check_contract_dict(candidate)
        except AssertionError:
            rejected += 1
    if rejected != 16:
        raise AssertionError("predecessor regression false negative")
    return {"attacks": 16, "rejected": rejected, "false_negatives": 16 - rejected}


def run_runtime_checks() -> dict[str, object]:
    common = M.validate_common(skip_author=False)
    future = M.validate_future("runtime_present")
    regressions = validate_contract_regressions()
    if M.strict_json(M.CONTRACT) != M.expected_contract():
        raise AssertionError("canonical contract drift")
    if common["runner"]["runtime_suite_only"] is not True:
        raise AssertionError("runtime suite split drift")
    return {"common": common, "future": future, "regressions": regressions}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("runtime_present",), required=True)
    parser.parse_args()
    checks = run_runtime_checks()
    print(json.dumps({
        "schema": "m1433_c1_r16_vcs_runtime_present_test_r1_v1",
        "status": "PASS_M1433_RUNTIME_PRESENT_LAUNCH_TESTS",
        "checks": checks,
        "license_queries": 0, "vcs_runs": 0, "simv_runs": 0, "eda_runs": 0,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
