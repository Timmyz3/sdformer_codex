#!/usr/bin/env python3
"""Independent M65-r2 release linkage and coordinated-tamper validator."""

from __future__ import print_function

import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
REVIEW = HERE / "m65_exact_sha_release_r2_independent_hammer_review.json"
RELEASE = HW / "contracts/m65_exact_sha_release_contract_r2_20260823.json"
RELEASE_VALIDATOR = HW / "system_simulator/scripts/validate_m65_exact_sha_release_r2.py"
PRODUCER_RECEIPT = HW / "results/m65_m53_m63_nonoverlap_joint_r1_20260823/m65_exact_sha_release_validation_receipt_r2.json"

EXPECTED = {
    "review": "e813cc36123361211509b980ac3f04b0f10948df527ef4b718fbcb13ff8129dd",
    "release": "cba46273de617bbb4f28f13baf0adc3999ca2c465a9f2308c863f95ca5213185",
    "release_validator": "27281de4968ec141bfc4da0c568f20789b3cd50ed1afc2e5d8a203e38e9c3180",
    "producer_receipt": "6437f262749083045daf422c98c5b60a5766a6841c2b7dadba4f169069301ba1",
    "exact_rational_validator": "6b5abbd5a22a953167daf965924a02b3b374bf85c41f4d10b33e37b87349795c",
}

RESULT_REL = "results/m65_m53_m63_nonoverlap_joint_r1_20260823/m65_m53_m63_nonoverlap_joint_result_r1.json"
CONTRACT_REL = "contracts/m65_m53_m63_nonoverlap_joint_contract_r1_20260823.json"
ANALYZER_REL = "system_simulator/scripts/analyze_m65_m53_m63_nonoverlap_joint.py"
EXACT_VALIDATOR_REL = "reviews/m65_independent_hammer_r1_20260823/validate_m65_independent_hammer_review.py"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: {}".format(key))
        result[key] = value
    return result


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=no_duplicates,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError("non-standard constant: " + value)))


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")


def entry_map(release):
    result = {}
    for row in release["entries"]:
        require(row["path"] not in result,
                "duplicate release entry: {}".format(row["path"]))
        result[row["path"]] = row["sha256"]
    return result


def validate_review(review):
    require(sha256_path(REVIEW) == EXPECTED["review"], "review SHA drift")
    require(review["schema"] ==
            "m65_exact_sha_release_r2_independent_hammer_review_v1" and
            review["status"] ==
            "PASS_PREVIOUS_PRODUCER_SEAL_P1_CLOSED_WITH_PACKAGING_P2",
            "review schema/status drift")
    require(review["producer_evidence_modified"] is False,
            "review claims production mutation")
    require(review["issues"]["P0"] == [] and
            review["issues"]["P1"] == [] and
            len(review["issues"]["P2"]) == 3,
            "review severity inventory drift")
    require(review["admission"] == {
        "previous_producer_seal_p1_closed": True,
        "content_chain_fail_closed": True,
        "producer_package_physically_immutable": False,
        "system_or_headline_claim": False,
        "verdict": "ADMIT_EXACT_SHA_DUAL_NO_GO_RELEASE_ONLY",
    }, "review admission drift")
    exact = review["exact_identity"]
    require(exact["release_contract_sha256"] == EXPECTED["release"] and
            exact["release_validator_sha256"] ==
            EXPECTED["release_validator"] and
            exact["release_receipt_sha256"] ==
            EXPECTED["producer_receipt"] and
            exact["release_entry_count"] == 12,
            "review identity drift")
    rational = exact["independent_exact_rational_validator"]
    require(rational["path"] == EXACT_VALIDATOR_REL and
            rational["sha256"] == EXPECTED["exact_rational_validator"] and
            rational["present_in_release_entries"] is True and
            rational["sha_checked_before_dynamic_import"] is True and
            rational["used_for_exact_fraction_recomputation"] is True and
            rational["used_for_independent_semantic_guard"] is True,
            "review rational-validator pin claim drift")
    require(len(review["required_attacks"]) == 8,
            "review attack inventory drift")


def validate_release(release):
    require(sha256_path(RELEASE) == EXPECTED["release"],
            "release SHA drift")
    require(release["schema"] == "m65_exact_sha_release_contract_v2" and
            release["status"] ==
            "FROZEN_M65_ARITHMETIC_AND_DUAL_NO_GO_RELEASE",
            "release schema/status drift")
    entries = entry_map(release)
    require(len(entries) == 12, "release entry count drift")
    for name, expected in entries.items():
        path = Path(name)
        if not path.is_absolute():
            path = HW / path
        require(path.is_file() and not path.is_symlink(),
                "release entry missing/symlink: {}".format(name))
        require(sha256_path(path) == expected,
                "release entry SHA drift: {}".format(name))
    require(entries[EXACT_VALIDATOR_REL] ==
            EXPECTED["exact_rational_validator"],
            "exact-rational validator is not pinned")
    validator_text = RELEASE_VALIDATOR.read_text(encoding="utf-8")
    require(EXPECTED["release"] in validator_text,
            "release validator does not hardcode release SHA")
    require("for entry in release[\"entries\"]" in validator_text and
            "independent = load_module(review_validator_path)" in validator_text and
            "summary = independent.validate_semantics(" in validator_text and
            "not independent.independent_guard(contract, result)" in validator_text,
            "release validator exact-rational linkage drift")
    return entries


def validate_producer_receipt(receipt, release):
    require(sha256_path(PRODUCER_RECEIPT) == EXPECTED["producer_receipt"],
            "producer receipt SHA drift")
    require(receipt["schema"] ==
            "m65_exact_sha_release_validation_receipt_v2" and
            receipt["status"] ==
            "PASS_EXACT_SHA_INDEPENDENT_RECOMPUTE_DUAL_NO_GO",
            "producer receipt schema/status drift")
    require(receipt["release_contract_sha256"] == EXPECTED["release"] and
            receipt["release_validator_sha256"] ==
            EXPECTED["release_validator"],
            "producer receipt release binding drift")
    require(receipt["entries_sha256"] == entry_map(release),
            "producer receipt entry ledger drift")
    require(receipt["exact_speed"] == [5158877, 860504] and
            receipt["joint_cycle_interval"] == [204002475, 204002476] and
            receipt["joint_ratio_interval_not_system_speedup"] == [
                3.0434348404673286, 3.0434348553859456] and
            receipt["replacement_regression_interval"] == [2742965, 2742966],
            "producer receipt rational conclusion drift")
    require(receipt["spatial_k4_decision"] ==
            "NO_GO_AS_ADDITIVE_M53_ACCELERATOR" and
            receipt["temporal_k4_decision"] ==
            "KILLED_BY_11_OF_24_LOCAL_CAPACITY_FAILURES",
            "producer receipt no-go drift")
    require(receipt["claim_boundary"] == release["claim_boundary"],
            "producer receipt claim boundary drift")


def clone_release_tree(directory):
    root = Path(directory)
    release = load_json(RELEASE)
    for row in release["entries"]:
        relative = Path(row["path"])
        if relative.is_absolute():
            continue
        source = HW / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(target))
    release_target = root / "contracts/m65_exact_sha_release_contract_r2_20260823.json"
    release_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(RELEASE), str(release_target))
    validator_target = root / "system_simulator/scripts/validate_m65_exact_sha_release_r2.py"
    validator_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(RELEASE_VALIDATOR), str(validator_target))
    return root, release_target, validator_target


def update_entry_sha(release, relative, root):
    matches = [row for row in release["entries"] if row["path"] == relative]
    require(len(matches) == 1, "entry update target missing: {}".format(relative))
    matches[0]["sha256"] = sha256_path(root / relative)


def patch_validator_root(validator_path, release_sha):
    text = validator_path.read_text(encoding="utf-8")
    require(EXPECTED["release"] in text, "cloned validator root missing")
    validator_path.write_text(text.replace(EXPECTED["release"], release_sha, 1),
                              encoding="utf-8")


def run_validator(validator, release):
    receipt = release.parent.parent / "attack_receipt.json"
    completed = subprocess.run([
        sys.executable, str(validator), "--release", str(release),
        "--receipt", str(receipt)], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, universal_newlines=True)
    return {
        "returncode": completed.returncode,
        "accepted": completed.returncode == 0,
        "diagnostic": (completed.stdout + completed.stderr).strip()[-500:],
    }


def require_rejected(name, outcome, expected_classification,
                     expected_diagnostic):
    require(not outcome["accepted"], "tamper accepted: {}".format(name))
    require(expected_diagnostic in outcome["diagnostic"],
            "tamper rejected for unexpected reason: {}: {}".format(
                name, outcome["diagnostic"]))
    return {
        "name": name,
        "result": "REJECTED",
        "classification": expected_classification,
        "returncode": outcome["returncode"],
        "diagnostic": outcome["diagnostic"],
    }


def attack_release_sha():
    with tempfile.TemporaryDirectory(prefix="m65_r2_release_sha_") as raw:
        root, release_path, _ = clone_release_tree(raw)
        payload = load_json(release_path)
        payload["frozen_conclusion"]["exact_m4_speed_numerator"] += 1
        write_json(release_path, payload)
        outcome = run_validator(RELEASE_VALIDATOR, release_path)
        return require_rejected("release_sha_mutation", outcome,
                                "REJECTED_BY_OUTER_RELEASE_SHA",
                                "release contract SHA drift")


def attack_entry_content(relative, attack_name, classification):
    with tempfile.TemporaryDirectory(prefix="m65_r2_entry_") as raw:
        root, release_path, validator_path = clone_release_tree(raw)
        target = root / relative
        target.write_bytes(target.read_bytes() + b"\n ")
        outcome = run_validator(validator_path, release_path)
        return require_rejected(
            attack_name, outcome, classification,
            "release entry SHA drift: {}".format(relative))


def coordinated_attack(name, mutate, classification, expected_diagnostic):
    with tempfile.TemporaryDirectory(prefix="m65_r2_linked_") as raw:
        root, release_path, validator_path = clone_release_tree(raw)
        release = load_json(release_path)
        mutate(root, release)
        write_json(release_path, release)
        release_sha = sha256_path(release_path)
        patch_validator_root(validator_path, release_sha)
        outcome = run_validator(validator_path, release_path)
        return require_rejected(name, outcome, classification,
                                expected_diagnostic)


def mutate_ratio(root, release):
    path = root / RESULT_REL
    result = load_json(path)
    result["spatial_k4_nonoverlap_joint"][
        "conditional_ratio_interval_not_system_speedup"] = {
            "minimum": 99.0, "maximum": 100.0}
    write_json(path, result)
    update_entry_sha(release, RESULT_REL, root)


def mutate_input_ledger(root, release):
    path = root / RESULT_REL
    result = load_json(path)
    first = sorted(result["identity"]["inputs_sha256"])[0]
    result["identity"]["inputs_sha256"][first] = "0" * 64
    write_json(path, result)
    update_entry_sha(release, RESULT_REL, root)


def mutate_analyzer(root, release):
    analyzer = root / ANALYZER_REL
    analyzer.write_bytes(analyzer.read_bytes() + b"\n# linked analyzer attack\n")
    result_path = root / RESULT_REL
    result = load_json(result_path)
    result["identity"]["analyzer_sha256"] = sha256_path(analyzer)
    write_json(result_path, result)
    update_entry_sha(release, ANALYZER_REL, root)
    update_entry_sha(release, RESULT_REL, root)


def mutate_forbidden(root, release):
    contract_path = root / CONTRACT_REL
    contract = load_json(contract_path)
    contract["claim_policy"]["forbidden"] = []
    write_json(contract_path, contract)
    result_path = root / RESULT_REL
    result = load_json(result_path)
    result["claim_boundary"] = copy.deepcopy(contract["claim_policy"])
    result["identity"]["contract_sha256"] = sha256_path(contract_path)
    write_json(result_path, result)
    update_entry_sha(release, CONTRACT_REL, root)
    update_entry_sha(release, RESULT_REL, root)


def receipt_mutation_attack(receipt, release):
    bad = copy.deepcopy(receipt)
    bad["release_contract_sha256"] = "0" * 64
    rejected = False
    diagnostic = ""
    try:
        validate_producer_receipt(bad, release)
    except Exception as error:
        rejected = True
        diagnostic = str(error)
    require(rejected, "receipt release binding mutation accepted")
    return {
        "name": "receipt_release_binding_mutation",
        "result": "REJECTED",
        "classification": "REJECTED_BY_INDEPENDENT_RECEIPT_VALIDATION",
        "returncode": 1,
        "diagnostic": diagnostic,
    }


def baseline_rerun():
    with tempfile.TemporaryDirectory(prefix="m65_r2_baseline_") as raw:
        receipt = Path(raw) / "receipt.json"
        completed = subprocess.run([
            sys.executable, str(RELEASE_VALIDATOR), "--release", str(RELEASE),
            "--receipt", str(receipt)], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, universal_newlines=True)
        require(completed.returncode == 0 and receipt.is_file(),
                "baseline release-validator rerun failed")
        payload = load_json(receipt)
        require(payload["status"] ==
                "PASS_EXACT_SHA_INDEPENDENT_RECOMPUTE_DUAL_NO_GO",
                "baseline rerun receipt status drift")
        return {
            "returncode": completed.returncode,
            "status": payload["status"],
            "exact_speed": payload["exact_speed"],
            "joint_cycle_interval": payload["joint_cycle_interval"],
            "replacement_regression_interval":
                payload["replacement_regression_interval"],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing independent receipt overwrite")
    review = load_json(REVIEW)
    release = load_json(RELEASE)
    receipt = load_json(PRODUCER_RECEIPT)
    validate_review(review)
    entries = validate_release(release)
    require(sha256_path(RELEASE_VALIDATOR) == EXPECTED["release_validator"],
            "release validator SHA drift")
    validate_producer_receipt(receipt, release)
    rerun = baseline_rerun()

    attacks = [
        attack_release_sha(),
        attack_entry_content(
            RESULT_REL, "entry_content_without_entry_sha_update",
            "REJECTED_BY_ENTRY_SHA"),
        coordinated_attack(
            "ratio_linked_resign", mutate_ratio,
            "REJECTED_BY_PINNED_EXACT_RATIONAL_VALIDATOR",
            "ValueError: joint intervals"),
        coordinated_attack(
            "input_ledger_linked_resign", mutate_input_ledger,
            "REJECTED_BY_PINNED_EXACT_RATIONAL_VALIDATOR",
            "ValueError: result input SHA ledger"),
        coordinated_attack(
            "analyzer_linked_resign", mutate_analyzer,
            "REJECTED_BY_PINNED_INDEPENDENT_GUARD",
            "ValueError: independent semantic guard rejected frozen result"),
        coordinated_attack(
            "forbidden_linked_resign", mutate_forbidden,
            "REJECTED_BY_PINNED_EXACT_RATIONAL_VALIDATOR",
            "ValueError: claim forbidden text system speedup"),
        attack_entry_content(
            EXACT_VALIDATOR_REL, "independent_validator_entry_mutation",
            "REJECTED_BY_ENTRY_SHA_BEFORE_IMPORT"),
        receipt_mutation_attack(receipt, release),
    ]
    require(len(attacks) == 8 and
            all(row["result"] == "REJECTED" for row in attacks),
            "attack ledger incomplete")

    payload = {
        "schema": "m65_exact_sha_release_r2_independent_hammer_validation_receipt_v1",
        "status": "PASS_M65_R2_RELEASE_P1_CLOSED_EIGHT_TAMPERS_REJECTED",
        "generated_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "review_sha256": EXPECTED["review"],
        "validator_sha256": sha256_path(Path(__file__).resolve()),
        "production_release_sha256": EXPECTED["release"],
        "production_release_validator_sha256": EXPECTED["release_validator"],
        "production_receipt_sha256": EXPECTED["producer_receipt"],
        "release_entry_count": len(entries),
        "exact_rational_validator_pinned": True,
        "exact_rational_validator_sha256":
            EXPECTED["exact_rational_validator"],
        "baseline_release_validator_rerun": rerun,
        "coordinated_tamper_attacks": attacks,
        "severity_counts": {"P0": 0, "P1": 0, "P2": 3},
        "previous_producer_seal_p1_closed": True,
        "producer_package_physically_immutable": False,
        "claim_boundary": review["claim_boundary"],
        "admission": review["admission"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("PASS M65-r2 independent release review P0/P1/P2=0/0/3 previous_seal_P1=CLOSED tamper=8/8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("FAIL M65-r2 independent release review: {}".format(error))
        raise SystemExit(1)
