#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Receipt-blind M1084 hammer for the source-only M1083 validator repair.

No real M699 payload member is opened or statted, and no pilot, GPU, EDA or
remote service is invoked.  All dynamic attacks use M1083's synthetic 623/623
window or isolated non-existing runtime paths.
"""
from __future__ import annotations

import builtins
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
DRIVER = HW / "system_simulator/scripts/execute_m1083_decoder_side_normalized_validator_repair.py"
RUNNER = HW / "system_simulator/scripts/run_m1085_m1083_decoder_side_normalized_pilot_one_shot.sh"
CONTRACT = HW / "contracts/m1083_decoder_side_normalized_validator_repair_contract_r1_20260830.json"
RELEASE = HW / "contracts/m1083_decoder_side_normalized_validator_repair_release_r1_20260830.json"
AUTHOR = HW / "reviews/m1083_decoder_side_normalized_validator_source_receipt_r1_20260830"
M1082 = HW / "reviews/m1082_m1078_decoder_pilot_failure_audit_r1_20260830"
M1077 = HW / "reviews/m1077_m1076_decoder_exact_bool_repair_hammer_r1_20260830"
M1085_C1 = HW / "reviews/m1085_m1074_c1_full_replay_failure_audit_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
PAYLOAD_MARKER = "system_handoff/outgoing/m699_h67_ep35_multisequence_decoder_payload_s3x10_r1_20260828"

EXPECTED = {
    "driver": "44aa7367ea8a679ff2b067c80d79320989030bde24a6e75edf96c07340e9bbec",
    "runner": "5d78df62700bd924237915ccf4cb454487760dee045d2d977ebce69d5ff42580",
    "contract": "e7d17a777b4e051d1648d9aed2ef6fa9a6caa8f3b3052c2232b41b781a15ac16",
    "contract_side": "c259c1885ef98a10f3d968826f435f667d74318a30fcf177d01a060bac42fddb",
    "contract_outer": "cb5942ffe983b4deadb3e5e37be679ac3a7a838499ebaea63ffe736c99072cad",
    "release": "19df5679a8f856237b38716e6d31ce555e0e99c3c7cc70598ade00bbf67817ac",
    "release_side": "99390576c6b27ea5d04ea286eadb974225a5f84151a4d7206d501d2d30c8fd6e",
    "release_outer": "4f205d40a54a667469f7409217f1dc0ef50a4e0befbaa0c975e79f9f49d49a22",
    "author_review": "4333f9e8a8e9626700b3918c6d9cfb6d748882fcb8a9c4cba943a6c1d50a4518",
    "author_manifest": "f90d05572816d0eb24162c7cd631b6520eb2537e0c8760acb2accf46752aa7a1",
    "author_outer": "2e37fdd9cfda54904d6992dffa1488df95aaad582edd0eacf254c57cc1859de6",
    "m1082_review": "73e0740258ca00c5fde02bbde46b41e3b7891d01ed2a8033bf2ac56f63f37243",
    "m1082_manifest": "33460b491c524a707818f4faba412ffd1d756215bf07e36aba13b138a892d012",
    "m1082_outer": "c40d02006a2f5e564cc37c0e2d9c1437f02e18fa9816a1c2c56cc8cc76c72956",
    "m1077_review": "3228372f7f35ec68d5eee97795a4ec4174a634adb7dddde45b99b253b0cb9b00",
    "m1077_manifest": "4999b94bca9a173701a387537cae2d4b258cc78dae473a730dec63cc6b7aa962",
    "m1077_outer": "a293c6c6593892a1c83289847e4984fd54a1e63880249518b3b4ab30e06e1e02",
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
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + token)))


def verify_flat(directory: Path, expected: tuple[str, str, str]) -> dict[str, str]:
    review, manifest, outer = (directory / "review.json", directory / "SHA256SUMS",
                               directory / "SHA256SUMS.seal.sha256")
    require(directory.is_dir() and not directory.is_symlink() and
            (sha(review), sha(manifest), sha(outer)) == expected,
            "flat identity drift: " + directory.name)
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        require(name not in listed and (directory / name).is_file() and
                not (directory / name).is_symlink() and
                sha(directory / name) == digest, "flat member drift: " + name)
        listed.add(name)
    require(outer.read_text(encoding="utf-8").split() ==
            [sha(manifest), "SHA256SUMS"], "flat outer content drift")
    return {"review_sha256": sha(review), "manifest_sha256": sha(manifest),
            "outer_seal_file_sha256": sha(outer)}


def verify_double(path: Path, expected: tuple[str, str, str]) -> dict[str, str]:
    side, outer = Path(str(path) + ".sha256"), Path(str(path) + ".sha256.seal.sha256")
    require((sha(path), sha(side), sha(outer)) == expected and
            side.read_text(encoding="utf-8").split() == [expected[0], path.name] and
            outer.read_text(encoding="utf-8").split() == [expected[1], side.name],
            "double seal drift: " + path.name)
    return {"primary_sha256": expected[0], "sidecar_sha256": expected[1],
            "outer_seal_file_sha256": expected[2]}


def load_driver_with_payload_guard():
    attempts: list[str] = []
    original_builtin_open = builtins.open
    original_path_open = Path.open
    original_os_open = os.open

    def forbidden(value: Any) -> bool:
        try:
            return PAYLOAD_MARKER in os.fspath(value)
        except TypeError:
            return False

    def guarded_builtin(file, *args, **kwargs):
        if forbidden(file):
            attempts.append("builtins.open:" + os.fspath(file))
            raise RuntimeError("M1084 forbids real payload open")
        return original_builtin_open(file, *args, **kwargs)

    def guarded_path(path, *args, **kwargs):
        if forbidden(path):
            attempts.append("Path.open:" + os.fspath(path))
            raise RuntimeError("M1084 forbids real payload open")
        return original_path_open(path, *args, **kwargs)

    def guarded_os(path, *args, **kwargs):
        if forbidden(path):
            attempts.append("os.open:" + os.fspath(path))
            raise RuntimeError("M1084 forbids real payload open")
        return original_os_open(path, *args, **kwargs)

    builtins.open, Path.open, os.open = guarded_builtin, guarded_path, guarded_os
    try:
        spec = importlib.util.spec_from_file_location("m1084_frozen_m1083", DRIVER)
        require(spec is not None and spec.loader is not None, "cannot load M1083")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        builtins.open, Path.open, os.open = (original_builtin_open,
                                             original_path_open, original_os_open)
    require(not attempts, "real payload access attempted")
    return module, attempts


def rejected(module, value: dict[str, Any], expected_sha: str) -> bool:
    try:
        module.validate_window_pair(value, expected_sha,
                                    value["side_specific_provenance"])
    except (RuntimeError, TypeError, ValueError, KeyError, IndexError):
        return True
    return False


def semantic_attacks(module, base: dict[str, Any]) -> dict[str, bool]:
    output = {}
    classes = sorted(module.M1052.CYCLE_CLASSES)

    def run(name: str, mutate: Callable[[dict[str, Any]], None]) -> None:
        value = copy.deepcopy(base)
        mutate(value)
        value["side_specific_provenance"] = module.window_bindings(value)
        output[name] = rejected(module, value,
                                base["candidate_exact"]["window_identity_sha256"])

    run("status", lambda w: w["baseline_exact"].__setitem__("status", "PASS_FORGED"))
    run("window_identity_sha256", lambda w: w["baseline_exact"].__setitem__(
        "window_identity_sha256", "d" * 64))

    def total(w):
        w["baseline_exact"]["total_cycles"] = 624
        w["baseline_exact"]["cycle_classes"][classes[0]] = 624
        w["baseline_cycles"] = 624
    run("total_cycles", total)
    run("expanded_request_count", lambda w: w["baseline_exact"].__setitem__(
        "expanded_request_count", 18))
    run("compressed_transaction_count", lambda w: w["baseline_exact"].__setitem__(
        "compressed_transaction_count", 8))
    run("commit_request_count", lambda w: w["baseline_exact"].__setitem__(
        "commit_request_count", 4))

    def cycle_classes(w):
        w["baseline_exact"]["cycle_classes"][classes[0]] = 622
        w["baseline_exact"]["cycle_classes"][classes[1]] = 1
    run("cycle_classes", cycle_classes)
    run("commit_sequence_sha256", lambda w: w["baseline_exact"].__setitem__(
        "commit_sequence_sha256", "d" * 64))
    run("port_calendars_sha256", lambda w: w["baseline_exact"].__setitem__(
        "port_calendars_sha256", "d" * 64))
    run("live_token_final_zero", lambda w: w["baseline_exact"].__setitem__(
        "live_token_final_zero", False))
    run("outstanding_return_final_zero", lambda w: w["baseline_exact"].__setitem__(
        "outstanding_return_final_zero", False))
    run("cycle_class_sum_equals_total", lambda w: w["baseline_exact"].__setitem__(
        "cycle_class_sum_equals_total", False))
    run("exact_fields", lambda w: w["baseline_exact"]["exact_fields"].__setitem__(
        0, "forged"))
    return output


def total_binding_attacks(module, base: dict[str, Any]) -> dict[str, bool]:
    output = {}
    first = sorted(module.M1052.CYCLE_CLASSES)[0]
    for side in ("candidate", "baseline"):
        value = copy.deepcopy(base)
        value[side + "_exact"]["total_cycles"] = 624
        value[side + "_exact"]["cycle_classes"][first] = 624
        value["side_specific_provenance"] = module.window_bindings(value)
        output[side + "_total_not_bound_to_reported_cycles"] = rejected(
            module, value, base["candidate_exact"]["window_identity_sha256"])
        value = copy.deepcopy(base)
        value[side + "_cycles"] = True
        value["side_specific_provenance"] = module.window_bindings(value)
        output[side + "_cycles_bool_alias"] = rejected(
            module, value, base["candidate_exact"]["window_identity_sha256"])
    return output


def provenance_attacks(module, base: dict[str, Any]) -> dict[str, str]:
    output = {}

    def run(name: str, mutate: Callable[[dict[str, Any]], None], resign: bool) -> None:
        value = copy.deepcopy(base)
        mutate(value)
        if resign:
            value["side_specific_provenance"] = module.window_bindings(value)
        output[name] = ("REJECTED" if rejected(
            module, value, base["candidate_exact"]["window_identity_sha256"])
                        else "ACCEPTED")

    run("stale_homogenize_transaction", lambda w: w["baseline_exact"].__setitem__(
        "transaction_address_sha256",
        w["candidate_exact"]["transaction_address_sha256"]), False)
    run("stale_arbitrary_terminal", lambda w: w["candidate_exact"].__setitem__(
        "terminal_readiness_sha256", "e" * 64), False)
    run("stale_binding_mutation", lambda w: w["side_specific_provenance"][
        "candidate"].__setitem__("binding_sha256", "f" * 64), False)
    run("resigned_homogenize_transaction_both_sides", lambda w: [
        w[side].__setitem__("transaction_address_sha256", "d" * 64)
        for side in ("candidate_exact", "baseline_exact")], True)
    run("resigned_arbitrary_candidate_terminal", lambda w: w[
        "candidate_exact"].__setitem__("terminal_readiness_sha256", "e" * 64), True)
    run("resigned_homogenize_all_reset_side_hashes", lambda w: [
        w[side].__setitem__(field, "f" * 64)
        for side in ("candidate_reset", "baseline_reset")
        for field in module.RESET_SIDE_FIELDS], True)
    run("resigned_arbitrary_paired_reset_semantics_sha", lambda w: w.__setitem__(
        "paired_reset_semantics_sha256", "0" * 64), True)
    return output


def namespace_and_runner_audit(module) -> dict[str, Any]:
    result = HW / "results/m1085_m1083_decoder_side_normalized_pilot_r1_20260830"
    attempt = HW / "results/.m1085_m1083_decoder_side_normalized_pilot_attempt_consumed"
    work = HW / "results/.m1085_m1083_decoder_side_normalized_pilot_r1_20260830.work.m1084_probe"
    quarantine = HW / "results/m1085_m1083_decoder_side_normalized_pilot_r1_20260830.failed_or_incomplete.m1084_probe"
    for role, path in (("result", result), ("attempt", attempt), ("work", work),
                       ("quarantine", quarantine)):
        require(module.safe_path(path, role) == path, role + " namespace rejects exact")
    rejected_paths = 0
    for role, path in (("attempt", HW / "results/.m1078_m1076_decoder_exact_bool_pilot_attempt_consumed"),
                       ("result", HW / "results/m1078_m1076_decoder_exact_bool_pilot_r1_20260830"),
                       ("attempt", M1085_C1)):
        try:
            module.safe_path(path, role)
        except RuntimeError:
            rejected_paths += 1
    require(rejected_paths == 3 and M1085_C1.is_dir() and
            M1085_C1.parent.name == "reviews" and not attempt.exists() and
            not result.exists(), "namespace isolation drift")

    text = RUNNER.read_text(encoding="utf-8")
    positions = {token: text.index(token) for token in (
        "--validate-source-only", "--validate-authority", "${m1085_flock}\" -n 9",
        "--consume-attempt", "/usr/bin/mkdir -m 700", "--validate-payload-after-attempt",
        "--run-pilot", "--assemble", "--publish")}
    order = [positions[key] for key in (
        "--validate-source-only", "--validate-authority", "${m1085_flock}\" -n 9",
        "--consume-attempt", "/usr/bin/mkdir -m 700", "--validate-payload-after-attempt",
        "--run-pilot", "--assemble", "--publish")]
    require(order == sorted(order) and text.count("--consume-attempt") == 1 and
            "m1085_started=1" in text and
            "${m1085_started}\" -eq 1 && \"${m1085_published}\" -eq 0" in text and
            "--quarantine --work" in text and
            text.index("--consume-attempt") < text.index("m1085_started=1") <
            text.index("--validate-payload-after-attempt"),
            "attempt/payload/quarantine runner ordering drift")
    envs = set(re.findall(r"M1085_EXPECTED_[A-Z0-9_]+", text))
    require(envs == {"M1085_EXPECTED_CONTRACT_SHA",
                     "M1085_EXPECTED_M1084_REVIEW_SHA",
                     "M1085_EXPECTED_M1084_MANIFEST_SHA",
                     "M1085_EXPECTED_M1084_OUTER_SHA"}, "runner pin surface drift")
    driver_text = DRIVER.read_text(encoding="utf-8")
    require("os.replace(work, quarantine)" in
            module.M1076.quarantine.__code__.co_consts or
            "os.replace(work, quarantine)" in
            (HW / "system_simulator/scripts/execute_m1076_decoder_exact_bool_repair.py").read_text(encoding="utf-8"),
            "frozen quarantine move absent")
    require("configure_frozen_runtime" in driver_text and
            "M1076.ATTEMPT_NAME = ATTEMPT_NAME" in driver_text and
            "M1076.RESULT_NAME = RESULT_NAME" in driver_text,
            "runtime namespace retarget drift")
    return {
        "exact_new_namespaces_accept": True,
        "old_or_review_namespaces_rejected": rejected_paths,
        "m1085_c1_review_conflict": False,
        "decoder_attempt_absent": True, "decoder_result_absent": True,
        "attempt_before_payload_static_order": True,
        "one_consume_attempt_call": True,
        "failure_quarantine_static_path": True,
        "runner_pin_environment": sorted(envs),
    }


def main() -> dict[str, Any]:
    identity = {
        "driver_sha256": sha(DRIVER), "runner_sha256": sha(RUNNER),
        "contract_sha256": sha(CONTRACT),
        "release_sha256": sha(RELEASE), "docs359_sha256": sha(DOCS359),
    }
    require(identity == {"driver_sha256": EXPECTED["driver"],
                          "runner_sha256": EXPECTED["runner"],
                          "contract_sha256": EXPECTED["contract"],
                          "release_sha256": EXPECTED["release"],
                          "docs359_sha256": EXPECTED["docs359"]},
            "M1083 top-level identity drift")
    seals = {
        "contract": verify_double(CONTRACT, (EXPECTED["contract"],
            EXPECTED["contract_side"], EXPECTED["contract_outer"])),
        "release": verify_double(RELEASE, (EXPECTED["release"],
            EXPECTED["release_side"], EXPECTED["release_outer"])),
        "author": verify_flat(AUTHOR, (EXPECTED["author_review"],
            EXPECTED["author_manifest"], EXPECTED["author_outer"])),
        "m1082": verify_flat(M1082, (EXPECTED["m1082_review"],
            EXPECTED["m1082_manifest"], EXPECTED["m1082_outer"])),
        "m1077": verify_flat(M1077, (EXPECTED["m1077_review"],
            EXPECTED["m1077_manifest"], EXPECTED["m1077_outer"])),
    }
    contract, release = strict_json(CONTRACT), strict_json(RELEASE)
    require(contract["status"] == "M1083_SOURCE_ONLY__M1084_HAMMER_REQUIRED" and
            contract["launch_now"] is False and
            release["status"] == "M1083_RELEASE_FROZEN__M1084_REQUIRED__NO_LAUNCH" and
            release["launch_now"] is False and
            release["authorization"]["m1085_attempt"] is False,
            "pre-hammer no-launch boundary drift")

    module, payload_attempts = load_driver_with_payload_guard()
    require(tuple(module.SEMANTIC_FIELDS) == tuple(contract["semantic_projection_fields"]) and
            tuple(module.SIDE_SPECIFIC_FIELDS) ==
            tuple(contract["side_specific_provenance_fields"]) and
            tuple(module.RESET_SEMANTIC_FIELDS) ==
            tuple(contract["paired_reset"]["semantic_fields"]) and
            tuple(module.RESET_SIDE_FIELDS) ==
            tuple(contract["paired_reset"]["side_specific_fields"]),
            "contract/source projection field drift")
    base = module.synthetic_window()
    module.validate_window_pair(base, base["candidate_exact"]["window_identity_sha256"],
                                base["side_specific_provenance"])
    require(base["candidate_cycles"] == base["baseline_cycles"] == 623 and
            base["candidate_exact"]["transaction_address_sha256"] !=
            base["baseline_exact"]["transaction_address_sha256"] and
            base["candidate_exact"]["terminal_readiness_sha256"] !=
            base["baseline_exact"]["terminal_readiness_sha256"],
            "M1082 D0 synthetic pattern drift")
    semantic = semantic_attacks(module, base)
    totals = total_binding_attacks(module, base)
    provenance = provenance_attacks(module, base)
    require(set(semantic) == set(module.SEMANTIC_FIELDS) and all(semantic.values()),
            "semantic projection field attack survived")
    require(all(totals.values()), "per-side total/cycle binding attack survived")
    require(all(value == "REJECTED" for key, value in provenance.items()
                if key.startswith("stale_")), "stale provenance attack survived")
    resigned = {key: value for key, value in provenance.items()
                if key.startswith("resigned_")}
    require(resigned and all(value == "ACCEPTED" for value in resigned.values()),
            "expected re-signed provenance vulnerability not reproduced")
    namespace = namespace_and_runner_audit(module)
    require(not payload_attempts and sha(DOCS359) == EXPECTED["docs359"],
            "payload/docs guard drift")
    return {
        "schema": "m1084_m1083_decoder_side_normalized_validator_hammer_mechanical_v1",
        "status": "STOP_M1084_M1083_RESIGNED_SIDE_PROVENANCE_FORGERY__NO_M1085_ATTEMPT",
        "identity": identity, "seals": seals,
        "positive_checks": {
            "d0_623_623_unequal_side_hash_pattern_accepted": True,
            "all_semantic_projection_fields_mutated_and_rejected": semantic,
            "per_side_total_and_bool_alias_attacks_rejected": totals,
            "stale_binding_attacks_rejected": {key: value for key, value in
                                                provenance.items() if key.startswith("stale_")},
            "namespace_and_runner": namespace,
        },
        "p0_counterexamples": resigned,
        "payload_guard": {
            "real_payload_open_attempts": payload_attempts,
            "real_payload_members_opened_or_statted": False,
            "real_pilot_executed": False,
            "gpu_eda_remote_used": False,
        },
        "authorization": {"one_m1085_attempt": False,
                          "real_payload_after_attempt_only": False,
                          "eda_gpu_remote": False},
        "claim_boundary": {"source_ready": False, "paper_citable": False,
                           "decoder_complete": False, "table_a_row": False,
                           "system_speedup": False},
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
