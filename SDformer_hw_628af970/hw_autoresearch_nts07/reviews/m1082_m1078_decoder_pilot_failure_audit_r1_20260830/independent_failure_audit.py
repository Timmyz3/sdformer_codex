#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Receipt-blind, non-rerunning M1082 audit of the ended M1078 failure.

The diagnostic probe executes only the first D0 SOURCE_INIT_CENSUS window via
the frozen lower-level exact pair function.  It never calls M1078 run_pilot,
the M1078 runner, replay_layer, GPU, EDA, or remote services.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
DRIVER = HW / "system_simulator/scripts/execute_m1076_decoder_exact_bool_repair.py"
RUNNER = HW / "system_simulator/scripts/run_m1078_m1076_decoder_exact_bool_pilot_one_shot.sh"
CONTRACT = HW / "contracts/m1076_decoder_exact_bool_repair_contract_r1_20260830.json"
M1077 = HW / "reviews/m1077_m1076_decoder_exact_bool_repair_hammer_r1_20260830"
ATTEMPT = HW / "results/.m1078_m1076_decoder_exact_bool_pilot_attempt_consumed"
RESULT = HW / "results/m1078_m1076_decoder_exact_bool_pilot_r1_20260830"
QUARANTINE = HW / "results/m1078_m1076_decoder_exact_bool_pilot_r1_20260830.failed_or_incomplete.2631940.23079.18872"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "driver": "d3b98ec71c3123c856d6a7ce8c8cee431e4d8d0da75aebf92eee8e144123ec15",
    "runner": "15b53d4d8be73d12ee0b5847cfcbd856bc2b9e06c8c4bfa4a3df509ca330a5c6",
    "contract": "ba702d74e6ddfd4cd152bacd35e26a19f293c9e038e237120ca376fd9f969413",
    "m1077_review": "3228372f7f35ec68d5eee97795a4ec4174a634adb7dddde45b99b253b0cb9b00",
    "m1077_manifest": "4999b94bca9a173701a387537cae2d4b258cc78dae473a730dec63cc6b7aa962",
    "m1077_outer": "a293c6c6593892a1c83289847e4984fd54a1e63880249518b3b4ab30e06e1e02",
    "attempt_json": "7ea0a3c95b5674461c585097df461f37ffe384f71cc9bba614cb8ee853c63131",
    "canonical_context": "3298912963fc1068569fa36fa2a2e3eafc26085346ebb345e8de46b81ccf7581",
    "payload_validation": "d6e023a3a2b536c8f77099e4377f5b3966bebc47975dba456cee2d869128fce2",
    "failure": "5f859ed4db4ac9cbc585a2149094e60da050489eafed8ec14416711d1a05ec96",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def pairs(rows):
        value = {}
        for key, item in rows:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + token)))


def verify_flat(directory: Path) -> dict[str, str]:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(directory.is_dir() and not directory.is_symlink() and
            manifest.is_file() and outer.is_file(), "sealed directory absent")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        target = directory / name.lstrip("*")
        require(target.is_file() and sha(target) == digest,
                "manifest member drift: " + name)
    tokens = outer.read_text(encoding="utf-8").split()
    require(tokens == [sha(manifest), "SHA256SUMS"], "outer content drift")
    return {"manifest_sha256": sha(manifest),
            "outer_seal_file_sha256": sha(outer)}


def load_driver():
    require(sha(DRIVER) == EXPECTED["driver"], "driver drift")
    spec = importlib.util.spec_from_file_location("m1082_frozen_m1076", DRIVER)
    require(spec is not None and spec.loader is not None, "cannot load driver")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def recursive_differences(left: Any, right: Any, path=()) -> list[dict[str, Any]]:
    label = ".".join(map(str, path)) or "$"
    if type(left) is not type(right):
        return [{"path": label, "candidate": left, "baseline": right,
                 "class": "type"}]
    if type(left) is dict:
        output = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                output.append({"path": ".".join(map(str, path + (key,))),
                               "candidate": left.get(key), "baseline": right.get(key),
                               "class": "key"})
            else:
                output.extend(recursive_differences(left[key], right[key],
                                                    path + (key,)))
        return output
    if type(left) is list:
        output = []
        if len(left) != len(right):
            output.append({"path": label + ".length", "candidate": len(left),
                           "baseline": len(right), "class": "length"})
        for index, (a, b) in enumerate(zip(left, right)):
            output.extend(recursive_differences(a, b, path + (index,)))
        return output
    return [] if left == right else [{"path": label, "candidate": left,
                                      "baseline": right, "class": "value"}]


def main() -> dict[str, Any]:
    identity = {
        "driver_sha256": sha(DRIVER), "runner_sha256": sha(RUNNER),
        "contract_sha256": sha(CONTRACT),
        "m1077_review_sha256": sha(M1077 / "review.json"),
        "m1077_manifest_sha256": sha(M1077 / "SHA256SUMS"),
        "m1077_outer_seal_file_sha256": sha(M1077 / "SHA256SUMS.seal.sha256"),
        "attempt_json_sha256": sha(ATTEMPT / "attempt.json"),
        "canonical_context_sha256": sha(QUARANTINE / "canonical_context.json"),
        "payload_validation_sha256": sha(QUARANTINE / "payload_validation.json"),
        "failure_sha256": sha(QUARANTINE / "FAILURE.json"),
        "docs359_sha256": sha(DOCS359),
    }
    require(identity == {
        "driver_sha256": EXPECTED["driver"], "runner_sha256": EXPECTED["runner"],
        "contract_sha256": EXPECTED["contract"],
        "m1077_review_sha256": EXPECTED["m1077_review"],
        "m1077_manifest_sha256": EXPECTED["m1077_manifest"],
        "m1077_outer_seal_file_sha256": EXPECTED["m1077_outer"],
        "attempt_json_sha256": EXPECTED["attempt_json"],
        "canonical_context_sha256": EXPECTED["canonical_context"],
        "payload_validation_sha256": EXPECTED["payload_validation"],
        "failure_sha256": EXPECTED["failure"],
        "docs359_sha256": EXPECTED["docs359"],
    }, "M1078 failure identity drift")
    m1077_seal = verify_flat(M1077)
    require(m1077_seal == {"manifest_sha256": EXPECTED["m1077_manifest"],
                           "outer_seal_file_sha256": EXPECTED["m1077_outer"]},
            "M1077 seal recomputation drift")

    all_m1078 = sorted(path.name for path in (HW / "results").iterdir()
                       if "m1078" in path.name.lower())
    expected_names = sorted([ATTEMPT.name, QUARANTINE.name])
    require(all_m1078 == expected_names and not RESULT.exists(),
            "M1078 namespace population drift")
    attempt_members = sorted(path.name for path in ATTEMPT.iterdir())
    quarantine_members = sorted(path.name for path in QUARANTINE.iterdir())
    require(attempt_members == ["attempt.json"] and quarantine_members ==
            ["FAILURE.json", "canonical_context.json", "payload_validation.json"],
            "M1078 failure member population drift")

    attempt = strict_json(ATTEMPT / "attempt.json")
    failure = strict_json(QUARANTINE / "FAILURE.json")
    context = strict_json(QUARANTINE / "canonical_context.json")
    payload = strict_json(QUARANTINE / "payload_validation.json")
    require(attempt["status"] ==
            "M1078_ATTEMPT_CONSUMED_BEFORE_PAYLOAD_MEMBER_ACCESS" and
            attempt["runner_sha256"] == EXPECTED["runner"] and
            attempt["contract_sha256"] == EXPECTED["contract"] and
            attempt["m1077_authority"] == {
                "review_sha256": EXPECTED["m1077_review"],
                "manifest_sha256": EXPECTED["m1077_manifest"],
                "outer_seal_file_sha256": EXPECTED["m1077_outer"],
            } and failure == {"schema": "m1078_failure_v1",
                              "status": "FAILED_OR_INCOMPLETE_DO_NOT_CITE",
                              "return_code": 1, "paper_citable": False},
            "attempt/failure receipt drift")

    module = load_driver()
    module.validate_canonical_context(context, context)
    module.validate_payload_receipt(payload, context)
    require(module.canonical_sha(context) ==
            payload["canonical_context_sha256"] ==
            "93f863325b1dd4792db8f24f10d031100fb9beb7f26863d6fec0c0e0bc2acc22",
            "canonical context receipt binding drift")
    verified_members = {}
    for selected in context["payload"]["selected_records"]:
        target = HW / context["payload"]["directory"] / selected["relative_path"]
        actual = sha(target)
        require(actual == selected["packed_sha256"] ==
                selected["payload_member_sha256"], "payload member drift")
        verified_members[selected["layer"]] = actual

    # Minimal diagnostic: first actual D0 source block only.  This does not
    # call replay_layer/run_pilot and cannot create a result or attempt.
    payload_root, records, mapper, oracles = module.M1048._context()
    record = module.M1048.select_record(records, "D0")
    stream = module.M1048.M785.iter_record_transactions(
        mapper, record, payload_root, module.M1048.POPULATION_ID,
        module.M1048.CONFIG, module.M1048.TIMESTEP, oracles)
    metadata, body = next(module.M1048.iter_semantic_blocks(stream, "D0"))
    require(metadata["block_id"] == "M1048:D0:SOURCE:000000000" and
            module.M1048.M1041.classify_stratum(metadata) == "SOURCE_INIT_CENSUS",
            "first D0 window identity drift")
    spec = module.M1048.M1041.WindowSpec(
        metadata["block_id"], "D0", "SOURCE_INIT_CENSUS", 1, 0, 0)
    pair = module.M1048.M1041.paired_replay(body, body, spec)
    candidate, baseline = pair["candidate"], pair["baseline"]
    exact_differences = recursive_differences(candidate, baseline)
    reset_differences = recursive_differences(pair["candidate_reset"],
                                              pair["baseline_reset"])
    require(pair["candidate_cycles"] == pair["baseline_cycles"] ==
            candidate["total_cycles"] == baseline["total_cycles"] and
            candidate != baseline and exact_differences,
            "first-window failure mechanism not reproduced")
    transform_source = module.M1060.BASE.transform_layer.__code__
    transform_text = module.M1060.BASE_PATH.read_text(encoding="utf-8")
    transform_section = transform_text[
        transform_text.index("def transform_layer("):
        transform_text.index("def validate_metadata(")
    ]
    require(transform_source.co_name == "transform_layer" and
            '"candidate_exact", "baseline_exact"' in transform_section and
            "{key: copy.deepcopy(row[key])" in transform_section,
            "transform no-longer preserves exact pair fields verbatim")
    source = module.M1060.BASE_PATH.read_text(encoding="utf-8")
    require('window["candidate_exact"] == window["baseline_exact"]' in source and
            '"exact replay binding drift"' in source,
            "frozen validator failure predicate drift")
    classification = {
        "algorithm_or_candidate_numeric_mismatch": False,
        "schema_transform_bug": False,
        "exact_bool_repair_side_effect": False,
        "validator_identity_bug": True,
        "reason": (
            "The self-matched pair has equal cycles and each side independently "
            "passes the exact scheduler miter, but side-tagged reset/dependency "
            "identities make the full candidate_exact and baseline_exact dicts "
            "unequal. M1052 validate_layer incorrectly requires whole-dict equality."
        ),
    }
    require(not RESULT.exists() and sha(ATTEMPT / "attempt.json") ==
            EXPECTED["attempt_json"] and sha(DOCS359) == EXPECTED["docs359"],
            "audit modified frozen evidence")

    return {
        "schema": "m1082_m1078_decoder_pilot_failure_audit_mechanical_v1",
        "status": "PASS_M1082_M1078_FAILURE_AUDIT__ADDITIVE_VALIDATOR_REPAIR_ALLOWED__M1078_DO_NOT_RETRY",
        "identity": identity,
        "namespace": {
            "attempt": ATTEMPT.name, "result_absent": True,
            "work_absent_after_quarantine": True, "quarantine": QUARANTINE.name,
            "attempt_members": attempt_members,
            "quarantine_members": quarantine_members,
            "attempt_sealed": False, "quarantine_sealed": False,
            "unsealed_is_by_frozen_quarantine_implementation": True,
        },
        "m1077_seal_recomputed": m1077_seal,
        "canonical_context_digest_sha256": module.canonical_sha(context),
        "payload_members_rehashed": verified_members,
        "first_concrete_failure": {
            "layer": "D0", "window_index": 0,
            "block_id": metadata["block_id"],
            "stratum": "SOURCE_INIT_CENSUS",
            "candidate_cycles": pair["candidate_cycles"],
            "baseline_cycles": pair["baseline_cycles"],
            "candidate_exact_total_cycles": candidate["total_cycles"],
            "baseline_exact_total_cycles": baseline["total_cycles"],
            "first_exact_field_difference": exact_differences[0],
            "all_exact_field_differences": exact_differences,
            "reset_field_differences": reset_differences,
            "failing_predicate": "candidate_exact == baseline_exact",
            "passing_predicates": [
                "candidate_cycles == baseline_cycles",
                "candidate_exact.total_cycles == candidate_cycles",
                "baseline_exact.total_cycles == baseline_cycles",
                "candidate and baseline independently pass M768/M861/M890/M896 exact miter"
            ],
        },
        "classification": classification,
        "repair_recommendation": {
            "additive_source_repair_allowed": True,
            "m1076_m1078_may_be_modified": False,
            "m1078_retry_allowed": False,
            "new_attempt_namespace_required": True,
            "replace_full_dict_equality_with": (
                "validate each exact result independently, bind both total_cycles "
                "to candidate/baseline cycles, and compare an explicitly defined "
                "side-normalized invariant projection; retain side-specific hashes "
                "as separate provenance rather than requiring them equal"
            ),
            "new_different_author_source_hammer_required": True,
            "one_new_attempt_only_after_hammer": True,
        },
        "scope": {
            "m1078_runner_called": False, "m1078_run_pilot_called": False,
            "m1048_replay_layer_called": False,
            "minimal_first_window_exact_pair_only": True,
            "gpu_eda_remote_used": False, "sources_modified": False,
            "docs359_modified": False,
        },
    }


if __name__ == "__main__":
    result = main()
    temporary = HERE / ".mechanical_checks.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(HERE / "mechanical_checks.json")
    print(result["status"])
