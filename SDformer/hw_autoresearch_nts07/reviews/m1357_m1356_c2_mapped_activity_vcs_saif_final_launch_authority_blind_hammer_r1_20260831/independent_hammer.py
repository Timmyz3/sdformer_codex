#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author, source-only blind hammer for M1356 (never launches EDA)."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest import mock


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CHECKER = HW / "verif_m1356_c2_activity_final_launch/static_check_m1356_c2_activity_final_launch_source.py"
TEST = HW / "verif_m1356_c2_activity_final_launch/test_m1356_c2_activity_final_launch_source.py"
CONTRACT = HW / "contracts/m1356_c2_mapped_activity_vcs_saif_final_launch_authority_source_contract_r1_20260831.json"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1344_c2_headline_mapped_production_activity_one_shot_exact_sha.sh"
AUTHOR = HW / "reviews/m1356_c2_mapped_activity_vcs_saif_final_launch_authority_source_author_r1_20260831"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = load("m1357_blind_target", CHECKER)


def changed(value):
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is str:
        return "M1357_MUTATED"
    if type(value) is list:
        return value + ["M1357_MUTATED"]
    if type(value) is dict:
        result = dict(value)
        result["m1357_extra"] = True
        return result
    raise TypeError(type(value))


def main() -> int:
    # This is safe after the blind directory exists: validate_common does not
    # assert that the future blind namespace is absent.  The author-stage
    # source_absent check and 20/20 suite were run before creating HERE.
    baseline = M.validate_common(skip_author=False)
    base = M.strict_json(CONTRACT)
    attacks: dict[str, bool] = {}

    def attack(name, mutate):
        candidate = copy.deepcopy(base)
        mutate(candidate)
        try:
            with mock.patch.object(M, "strict_json", return_value=candidate):
                M.validate_contract(skip_author=True)
            attacks[name] = False
        except Exception:
            attacks[name] = True

    attack("contract_extra_top_level", lambda d: d.__setitem__("m1357_extra", True))
    attack("contract_date_changed", lambda d: d.__setitem__("date", "2099-01-01"))
    attack("contract_purpose_changed", lambda d: d.__setitem__("purpose", "M1357_MUTATED"))
    attack("one_shot_removed", lambda d: d.pop("one_shot"))
    for key in base["one_shot"]:
        attack("one_shot_" + key, lambda d, key=key:
               d["one_shot"].__setitem__(key, changed(d["one_shot"][key])))
    attack("resource_fail_close_removed", lambda d: d.pop("resource_fail_close"))
    for key in base["resource_fail_close"]:
        attack("resource_fail_close_" + key, lambda d, key=key:
               d["resource_fail_close"].__setitem__(
                   key, changed(d["resource_fail_close"][key])))
    attack("receipt_contract_removed", lambda d: d.pop("receipt_contract"))
    for key in base["receipt_contract"]:
        attack("receipt_contract_" + key, lambda d, key=key:
               d["receipt_contract"].__setitem__(
                   key, changed(d["receipt_contract"][key])))
    attack("authorization_automatic_retry_true", lambda d:
           d["authorization"].__setitem__("automatic_retry", True))
    attack("authorization_source_only_tests_false", lambda d:
           d["authorization"].__setitem__("source_only_tests", False))
    attack("future_blind_zero_false_negatives_false", lambda d:
           d["future_blind"].__setitem__("zero_false_negatives_required", False))
    attack("future_blind_fresh_different_author_false", lambda d:
           d["future_blind"].__setitem__("fresh_different_author", False))
    attack("protected_files_removed", lambda d: d.pop("protected_files"))

    # Positive controls: exact identity and exact all-false claim mutations
    # must be rejected and are indeed rejected.
    attack("positive_control_runner_sha_changed", lambda d:
           d["identity"].__setitem__("runner_sha256", "0" * 64))
    attack("positive_control_m1353_review_sha_changed", lambda d:
           d["identity"].__setitem__("m1353_review_sha256", "0" * 64))
    attack("positive_control_claim_headline_lift", lambda d:
           d["claim_boundary"].__setitem__("headline", True))
    attack("positive_control_claim_extra", lambda d:
           d["claim_boundary"].__setitem__("launch_authorized", False))

    false_negatives = sorted(name for name, rejected in attacks.items()
                             if not rejected and not name.startswith("positive_control_"))
    positive_controls = {name: rejected for name, rejected in attacks.items()
                         if name.startswith("positive_control_")}
    output = {
        "schema": "m1357_m1356_c2_final_launch_source_blind_hammer_output_r1_v1",
        "status": "FAIL_ZERO_FALSE_NEGATIVE_GATE",
        "baseline": {
            "precreation_unit_tests": "20/20 PASS",
            "precreation_source_absent_self_check": "PASS",
            "validate_common_after_creation": True,
            "m1353_false_negatives": baseline["m1353_false_negatives"],
            "runner_sha256": sha(RUNNER),
            "checker_sha256": sha(CHECKER),
            "test_sha256": sha(TEST),
            "contract_sha256": sha(CONTRACT),
            "author_review_sha256": sha(AUTHOR / "review.json"),
            "author_manifest_sha256": sha(AUTHOR / "SHA256SUMS"),
            "author_outer_file_sha256": sha(AUTHOR / "SHA256SUMS.seal.sha256"),
        },
        "attacks": len(attacks),
        "rejected": sum(attacks.values()),
        "false_negatives": len(false_negatives),
        "false_negative_names": false_negatives,
        "positive_controls": positive_controls,
        "runner_facts": {
            "namespace_count": baseline["resources"]["namespace_count"],
            "attempt_fresh_before_blind_creation": baseline["resources"]["attempt_fresh"],
            "collision_before_attempt": baseline["resources"]["collision_before_attempt"],
            "success_claim_count": baseline["receipts"]["success_claims"],
            "launch_authorized": baseline["launch_authorized"],
        },
        "authorization": {
            "additive_source_successor": True,
            "launch": False,
            "license_query": False,
            "vcs": False,
            "simv": False,
            "saif": False,
            "ptpx": False,
            "eda": False,
            "automatic_retry": False,
        },
        "claim_boundary": M.EXACT_CLAIMS,
        "docs359_sha256": sha(DOCS359),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 1 if false_negatives else 0


if __name__ == "__main__":
    raise SystemExit(main())
