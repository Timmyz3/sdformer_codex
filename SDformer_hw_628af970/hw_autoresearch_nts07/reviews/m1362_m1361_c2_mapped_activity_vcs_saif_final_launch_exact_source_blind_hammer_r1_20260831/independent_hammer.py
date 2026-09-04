#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Fresh no-EDA blind hammer for the exact M1361 contract successor."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
CHECKER = HW / "verif_m1361_c2_activity_final_launch_exact/static_check_m1361_c2_activity_final_launch_exact_source.py"
CHECKER_SHA = "13a98be09ec5e00d5f6ec7f07e53f27bc2d66c5d72d11b778c19e5a511422745"
TEST = HW / "verif_m1361_c2_activity_final_launch_exact/test_m1361_c2_activity_final_launch_exact_source.py"
TEST_SHA = "2938595d4192528e05b1aea22201f4086f35a5789756348e2d9034f35afdc8dd"
CONTRACT = HW / "contracts/m1361_c2_mapped_activity_vcs_saif_final_launch_exact_source_contract_r1_20260831.json"
CONTRACT_SHA = "fb2e5f83a4befef0252a030402c2e18f8babc336e326d30f7d91d90969c00c9a"
CONTRACT_DIGEST = CONTRACT.with_suffix(CONTRACT.suffix + ".sha256")
CONTRACT_DIGEST_SHA = "db556d08e4b69274589594e4d8c03bd0010309fc9cdded56cc6fd11a4799fead"
CONTRACT_OUTER = Path(str(CONTRACT_DIGEST) + ".seal.sha256")
CONTRACT_OUTER_SHA = "448eca4c4e99daf81b064d7a2efbb7f0f8475f45a65098cbe1a8a8eb1e3f1cb0"
AUTHOR = HW / "reviews/m1361_c2_mapped_activity_vcs_saif_final_launch_exact_source_author_r1_20260831"
AUTHOR_REVIEW_SHA = "d4369a78849b7f3f7411cc1c21365e17450275b01ed906468c368781b140126c"
AUTHOR_MANIFEST_SHA = "e00f9cfc6222c92ecd7f6b7e0ca7d0f1c46204634f208cdac3545e707e4edaaa"
AUTHOR_OUTER_SHA = "634258227ac5143d820fa696ed8cb572f8c622d7b4ad8e3c0db404a0b2adbdaf"
M1357_OUTPUT = HW / "reviews/m1357_m1356_c2_mapped_activity_vcs_saif_final_launch_authority_blind_hammer_r1_20260831/hammer_output.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict(path: Path) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            assert key not in result
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           AssertionError("nonfinite " + token)))
    assert type(value) is dict
    return value


def load():
    assert sha(CHECKER) == CHECKER_SHA and sha(TEST) == TEST_SHA
    spec = importlib.util.spec_from_file_location("m1362_blind_m1361", CHECKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = load()
CHECKS: list[str] = []
ATTACKS = 0
FALSE_NEGATIVES = 0


def passed(name: str) -> None:
    CHECKS.append(name)


def changed(value: Any) -> Any:
    if type(value) is bool: return not value
    if type(value) is int: return value + 1
    if type(value) is str: return "M1362_MUTATED"
    if type(value) is list: return list(value) + ["M1362_MUTATED"]
    if type(value) is dict:
        result = dict(value); result["m1362_extra"] = True; return result
    raise TypeError(type(value))


def rejected(name: str, candidate: dict[str, Any]) -> None:
    global ATTACKS, FALSE_NEGATIVES
    ATTACKS += 1
    try:
        with mock.patch.object(M, "strict_json", return_value=candidate):
            M.validate_contract(skip_author=True)
    except AssertionError:
        passed(name)
        return
    FALSE_NEGATIVES += 1
    raise AssertionError("accepted contract attack: " + name)


def mutate_at(candidate: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = candidate
    for key in path[:-1]: cursor = cursor[key]
    cursor[path[-1]] = value


def delete_at(candidate: dict[str, Any], path: tuple[str, ...]) -> None:
    cursor = candidate
    for key in path[:-1]: cursor = cursor[key]
    del cursor[path[-1]]


def historical_cases() -> list[tuple[str, Callable[[dict[str, Any]], None]]]:
    base = M.expected_contract()
    cases: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        ("contract_extra_top_level", lambda d: d.__setitem__("m1361_extra", True)),
        ("contract_date_changed", lambda d: d.__setitem__("date", "2099-01-01")),
        ("contract_purpose_changed", lambda d: d.__setitem__("purpose", "M1361_MUTATED")),
        ("one_shot_removed", lambda d: d.pop("one_shot")),
    ]
    cases.extend(("one_shot_" + key, lambda d, key=key:
                  d["one_shot"].__setitem__(key, changed(d["one_shot"][key])))
                 for key in base["one_shot"])
    cases.append(("resource_fail_close_removed", lambda d: d.pop("resource_fail_close")))
    cases.extend(("resource_fail_close_" + key, lambda d, key=key:
                  d["resource_fail_close"].__setitem__(
                      key, changed(d["resource_fail_close"][key])))
                 for key in base["resource_fail_close"])
    cases.append(("receipt_contract_removed", lambda d: d.pop("receipt_contract")))
    cases.extend(("receipt_contract_" + key, lambda d, key=key:
                  d["receipt_contract"].__setitem__(
                      key, changed(d["receipt_contract"][key])))
                 for key in base["receipt_contract"])
    cases.extend([
        ("authorization_automatic_retry_true", lambda d:
         d["authorization"].__setitem__("automatic_retry", True)),
        ("authorization_source_only_tests_false", lambda d:
         d["authorization"].__setitem__("source_only_tests", False)),
        ("future_blind_zero_false_negatives_false", lambda d:
         d["future_blind"].__setitem__("zero_false_negatives_required", False)),
        ("future_blind_fresh_different_author_false", lambda d:
         d["future_blind"].__setitem__("fresh_different_author", False)),
        ("protected_files_removed", lambda d: d.pop("protected_files")),
    ])
    assert len(cases) == 30
    return cases


def leaves(value: Any, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    if type(value) is dict:
        output = []
        for key, child in value.items(): output.extend(leaves(child, prefix + (key,)))
        return output
    return [prefix]


def main() -> int:
    assert sha(CONTRACT) == CONTRACT_SHA
    assert sha(CONTRACT_DIGEST) == CONTRACT_DIGEST_SHA
    assert sha(CONTRACT_OUTER) == CONTRACT_OUTER_SHA
    assert CONTRACT_DIGEST.read_text(encoding="ascii") == f"{CONTRACT_SHA}  {CONTRACT.name}\n"
    assert CONTRACT_OUTER.read_text(encoding="ascii") == \
        f"{CONTRACT_DIGEST_SHA}  {CONTRACT_DIGEST.name}\n"
    passed("contract_file_recursive_two_level_seal")

    author = M.verify_dir(AUTHOR, AUTHOR_REVIEW_SHA, AUTHOR_MANIFEST_SHA, AUTHOR_OUTER_SHA)
    assert author["status"] == "PASS_M1361_EXACT_SOURCE_AUTHOR__FRESH_M1362_BLIND_REQUIRED"
    assert author["authorization"] == {
        "different_author_blind_hammer": True, "launch_authorized": False,
        "license_query": False, "vcs": False, "simv": False, "saif": False,
        "ptpx": False, "eda": False, "automatic_retry": False}
    passed("author_recursive_seal_and_authorization")

    common = M.validate_common(skip_author=False)
    assert common["m1357_false_negatives_repaired"] == 30
    assert common["launch_authorized"] is False
    passed("full_static_chain_positive")
    expected = M.expected_contract(); actual = strict(CONTRACT)
    assert actual == expected and set(actual) == set(expected)
    passed("contract_exact_positive")

    old = strict(M1357_OUTPUT)
    old_names = set(old["false_negative_names"])
    generated = {name for name, _ in historical_cases()}
    assert old["false_negatives"] == 30 and generated == old_names
    passed("m1357_exact_30_inventory_bound")
    for name, mutate in historical_cases():
        candidate = copy.deepcopy(expected); mutate(candidate)
        rejected("m1357_closed_" + name, candidate)

    # Exhaustively exercise every leaf plus deletion/addition at each requested object.
    objects = ("one_shot", "resource_fail_close", "receipt_contract", "future_blind",
               "authorization", "claim_boundary", "protected_files")
    for object_name in objects:
        candidate = copy.deepcopy(expected); candidate[object_name]["m1362_extra"] = True
        rejected(object_name + "_extra_key", candidate)
        for path in leaves(expected[object_name], (object_name,)):
            candidate = copy.deepcopy(expected)
            cursor: Any = expected
            for key in path: cursor = cursor[key]
            mutate_at(candidate, path, changed(cursor))
            rejected("leaf_value_" + ".".join(path), candidate)
            candidate = copy.deepcopy(expected); delete_at(candidate, path)
            rejected("leaf_delete_" + ".".join(path), candidate)

    for key in tuple(expected):
        candidate = copy.deepcopy(expected); candidate.pop(key)
        rejected("top_level_delete_" + key, candidate)
    candidate = copy.deepcopy(expected); candidate["m1362_extra"] = True
    rejected("top_level_extra", candidate)

    assert M.EXACT_CLAIMS == expected["claim_boundary"]
    assert expected["authorization"]["launch_authorized"] is False
    assert expected["authorization"]["automatic_retry"] is False
    assert expected["future_blind"] == {
        "path": "reviews/m1362_m1361_c2_mapped_activity_vcs_saif_final_launch_exact_source_blind_hammer_r1_20260831",
        "must_be_absent_during_authoring": True,
        "fresh_different_author": True,
        "zero_false_negatives_required": True}
    passed("final_authoring_only_boundary")

    # Static script audit: this hammer and checker have no process launcher.
    combined = Path(__file__).read_text(encoding="utf-8") + CHECKER.read_text(encoding="utf-8")
    assert "sub" + "process" not in combined
    assert "os" + ".system" not in combined
    assert "Po" + "pen(" not in combined
    passed("no_external_tool_execution_primitive")
    assert FALSE_NEGATIVES == 0
    output = {
        "schema": "m1362_m1361_c2_final_launch_exact_source_blind_hammer_output_r1_v1",
        "status": "PASS_ZERO_FALSE_NEGATIVE_GATE",
        "checks_passed": len(CHECKS),
        "attacks": ATTACKS,
        "false_negatives": FALSE_NEGATIVES,
        "m1357_historical_attacks": 30,
        "m1357_historical_false_negatives_now": 0,
        "authorization": {
            "final_launch_authority_authoring": True,
            "launch": False, "license_query": False, "vcs": False,
            "simv": False, "saif": False, "ptpx": False, "eda": False,
            "automatic_retry": False},
        "claim_boundary": dict(M.EXACT_CLAIMS),
        "protected_files": expected["protected_files"],
    }
    print(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
