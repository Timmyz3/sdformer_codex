#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1083 additive side-normalized decoder exact-result validator repair.

This source imports the frozen M1076 implementation by exact SHA.  Both sides
still pass the complete frozen exact-result validator.  Only two intentionally
side-tagged provenance hashes are excluded from cross-side semantic equality;
they remain independently validated and bound in a side-provenance receipt.
No production payload is touched by the author self-test.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
RESULTS = HW / "results"
M1076_PATH = HERE / "execute_m1076_decoder_exact_bool_repair.py"
M1076_SHA256 = "d3b98ec71c3123c856d6a7ce8c8cee431e4d8d0da75aebf92eee8e144123ec15"
M1082_DIR = HW / "reviews/m1082_m1078_decoder_pilot_failure_audit_r1_20260830"
M1082_OUTER_SHA256 = "c40d02006a2f5e564cc37c0e2d9c1437f02e18fa9816a1c2c56cc8cc76c72956"
M1084_DIR = HW / "reviews/m1084_m1083_decoder_side_normalized_validator_hammer_r1_20260830"
CONTRACT = HW / "contracts/m1083_decoder_side_normalized_validator_repair_contract_r1_20260830.json"
RELEASE = HW / "contracts/m1083_decoder_side_normalized_validator_repair_release_r1_20260830.json"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOC359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
ATTEMPT_NAME = ".m1085_m1083_decoder_side_normalized_pilot_attempt_consumed"
RESULT_NAME = "m1085_m1083_decoder_side_normalized_pilot_r1_20260830"

SIDE_SPECIFIC_FIELDS = (
    "transaction_address_sha256",
    "terminal_readiness_sha256",
)
SEMANTIC_FIELDS = (
    "status",
    "window_identity_sha256",
    "total_cycles",
    "expanded_request_count",
    "compressed_transaction_count",
    "commit_request_count",
    "cycle_classes",
    "commit_sequence_sha256",
    "port_calendars_sha256",
    "live_token_final_zero",
    "outstanding_return_final_zero",
    "cycle_class_sum_equals_total",
    "exact_fields",
)
RESET_SEMANTIC_FIELDS = (
    "window_identity_sha256",
    "body_expanded_request_count",
    "reset_expanded_request_count",
    "external_dependency_remap_count",
)
RESET_SIDE_FIELDS = (
    "original_transaction_id_census_sha256",
    "reset_transaction_id_census_sha256",
    "boundary_ready_token_sha256",
)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def strict_json(path):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            RuntimeError("non-finite JSON token: " + token)))


def verify_file_double_seal(path):
    path = Path(path)
    side = Path(str(path) + ".sha256")
    outer = Path(str(side) + ".seal.sha256")
    require(path.is_file() and side.is_file() and outer.is_file() and
            not path.is_symlink() and not side.is_symlink() and not outer.is_symlink(),
            "double-seal member absent/unsafe")
    exact_tree(side.read_text(encoding="utf-8"),
        sha256(path) + "  " + path.name + "\n", ("sidecar", path.name))
    exact_tree(outer.read_text(encoding="utf-8"),
        sha256(side) + "  " + side.name + "\n", ("outer", path.name))
    return {"primary_sha256": sha256(path), "sidecar_sha256": sha256(side),
            "outer_seal_file_sha256": sha256(outer)}


def load_m1076():
    require(M1076_PATH.is_file() and not M1076_PATH.is_symlink() and
            sha256(M1076_PATH) == M1076_SHA256, "frozen M1076 identity drift")
    spec = importlib.util.spec_from_file_location("m1083_frozen_m1076", M1076_PATH)
    require(spec is not None and spec.loader is not None, "cannot load M1076")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M1076 = load_m1076()
M1052 = M1076.M1060.BASE
FROZEN_M1076_VALIDATE_LAYER = M1076.validate_layer
FROZEN_M1052_TRANSFORM_LAYER = M1052.transform_layer


def exact_tree(actual, expected, path=()):
    return M1076.exact_tree(actual, expected, path)


def hash_value(value):
    require(type(value) is str and len(value) == 64 and
            all(char in "0123456789abcdef" for char in value), "invalid SHA256")
    return value


def semantic_projection(exact):
    """Documented equality surface; no implicit or cycles-only comparison."""
    require(type(exact) is dict, "exact result must be a dictionary")
    require(set(SEMANTIC_FIELDS).issubset(exact), "semantic field missing")
    return {key: copy.deepcopy(exact[key]) for key in SEMANTIC_FIELDS}


def reset_semantic_projection(reset):
    require(type(reset) is dict, "reset must be a dictionary")
    require(set(RESET_SEMANTIC_FIELDS).issubset(reset), "reset semantic field missing")
    return {key: copy.deepcopy(reset[key]) for key in RESET_SEMANTIC_FIELDS}


def side_provenance(side, exact, reset):
    require(side in ("candidate", "baseline"), "unknown replay side")
    value = {
        "side": side,
        "window_identity_sha256": exact["window_identity_sha256"],
        "exact": {key: exact[key] for key in SIDE_SPECIFIC_FIELDS},
        "reset": {key: reset[key] for key in RESET_SIDE_FIELDS},
    }
    for group in (value["exact"], value["reset"]):
        for item in group.values():
            hash_value(item)
    value["binding_sha256"] = canonical_sha(value)
    return value


def paired_reset_binding(candidate_reset, baseline_reset, paired_sha):
    hash_value(paired_sha)
    csemantic = reset_semantic_projection(candidate_reset)
    bsemantic = reset_semantic_projection(baseline_reset)
    exact_tree(csemantic, bsemantic, ("paired_reset", "semantic_projection"))
    return {"paired_reset_semantics_sha256": paired_sha,
        "semantic_projection": csemantic,
        "binding_sha256": canonical_sha({
            "paired_reset_semantics_sha256": paired_sha,
            "semantic_projection": csemantic})}


def window_bindings(window):
    return {
        "candidate": side_provenance("candidate", window["candidate_exact"],
                                     window["candidate_reset"]),
        "baseline": side_provenance("baseline", window["baseline_exact"],
                                    window["baseline_reset"]),
        "paired_reset": paired_reset_binding(window["candidate_reset"],
            window["baseline_reset"], window["paired_reset_semantics_sha256"]),
    }


def validate_window_pair(window, expected_window_sha, bindings):
    """Strictly validate each side, then compare only documented semantics."""
    require(type(window) is dict and type(bindings) is dict and
            set(bindings) == {"candidate", "baseline", "paired_reset"},
            "side binding schema drift")
    candidate = window["candidate_exact"]
    baseline = window["baseline_exact"]
    M1052.validate_exact_result(candidate, expected_window_sha)
    M1052.validate_exact_result(baseline, expected_window_sha)
    M1052.validate_reset(window["candidate_reset"], expected_window_sha)
    M1052.validate_reset(window["baseline_reset"], expected_window_sha)
    require(type(window["candidate_cycles"]) is int and
            type(window["baseline_cycles"]) is int,
            "side cycle must be exact non-bool int")
    exact_tree(candidate["total_cycles"], window["candidate_cycles"],
               ("candidate", "total_cycles"))
    exact_tree(baseline["total_cycles"], window["baseline_cycles"],
               ("baseline", "total_cycles"))
    exact_tree(semantic_projection(candidate), semantic_projection(baseline),
               ("side_normalized_semantic_projection",))
    expected = window_bindings(window)
    exact_tree(bindings, expected, ("side_specific_provenance",))
    return True


def transform_layer(old):
    row = FROZEN_M1052_TRANSFORM_LAYER(old)
    for window in row["windows"]:
        window["side_specific_provenance"] = window_bindings(window)
    return row


def validate_layer(value, layer, selected=None):
    """Use the frozen full layer validator through a provenance-only shadow."""
    require(type(value) is dict and type(value.get("windows")) is list,
            "layer/windows schema drift")
    shadow = copy.deepcopy(value)
    for original, projected in zip(value["windows"], shadow["windows"]):
        require("side_specific_provenance" in original,
                "side-specific provenance absent")
        bindings = projected.pop("side_specific_provenance")
        validate_window_pair(original, original["window_identity_sha256"], bindings)
        # The frozen validator still compares whole dictionaries.  Normalize only
        # the two documented side-tagged fields in this private validation shadow.
        for key in SIDE_SPECIFIC_FIELDS:
            projected["baseline_exact"][key] = projected["candidate_exact"][key]
    FROZEN_M1076_VALIDATE_LAYER(shadow, layer, selected) if selected is not None else \
        M1052.validate_layer(shadow, layer)
    return True


def make_synthetic_exact(window_sha, total=623):
    value = {
        "status": "PASS_M768_M861_M890_M896_BLOCK_RESET_EXACT_MITER",
        "window_identity_sha256": window_sha,
        "total_cycles": total,
        "expanded_request_count": 17,
        "compressed_transaction_count": 7,
        "commit_request_count": 3,
        "cycle_classes": {name: 0 for name in M1052.CYCLE_CLASSES},
        "transaction_address_sha256": "1" * 64,
        "commit_sequence_sha256": "2" * 64,
        "terminal_readiness_sha256": "3" * 64,
        "port_calendars_sha256": "4" * 64,
        "live_token_final_zero": True,
        "outstanding_return_final_zero": True,
        "cycle_class_sum_equals_total": True,
        "exact_fields": list(M1076.M1048.M1041.M946.EXACT_FIELDS),
    }
    value["cycle_classes"][sorted(M1052.CYCLE_CLASSES)[0]] = total
    return value


def make_synthetic_reset(window_sha, side):
    salt = "5" if side == "candidate" else "6"
    return {"window_identity_sha256": window_sha,
        "original_transaction_id_census_sha256": salt * 64,
        "reset_transaction_id_census_sha256": ("7" if side == "candidate" else "8") * 64,
        "body_expanded_request_count": 17,
        "reset_expanded_request_count": 3,
        "external_dependency_remap_count": 1,
        "boundary_ready_token_sha256": ("9" if side == "candidate" else "a") * 64}


def synthetic_window():
    window_sha = "b" * 64
    candidate = make_synthetic_exact(window_sha)
    baseline = copy.deepcopy(candidate)
    # Exact shape of M1082's first D0 failure: 623/623 with unequal side hashes.
    candidate["transaction_address_sha256"] = (
        "be3f74b424a5a52d1cc14316ba0adbb0fb71cb51e2286d2d61d51b4a69960a0f")
    baseline["transaction_address_sha256"] = (
        "ecfe8c85b1e24f2e228809279c43a838f13d90bde4f12cf8bb0e6bf88617113d")
    candidate["terminal_readiness_sha256"] = (
        "cd7b8dd49dfa1ac3b20b408d37f87bef87891c11a4d7d69399665cca87819c2b")
    baseline["terminal_readiness_sha256"] = (
        "97e9f0ace9a6e362ed961cb2b8de7ea4778c8717f89a3d551f1a968277bfff10")
    row = {"candidate_cycles": 623, "baseline_cycles": 623,
        "candidate_exact": candidate, "baseline_exact": baseline,
        "candidate_reset": make_synthetic_reset(window_sha, "candidate"),
        "baseline_reset": make_synthetic_reset(window_sha, "baseline"),
        "paired_reset_semantics_sha256": "c" * 64}
    row["side_specific_provenance"] = window_bindings(row)
    return row


def self_test():
    row = synthetic_window()
    validate_window_pair(row, row["candidate_exact"]["window_identity_sha256"],
                         row["side_specific_provenance"])
    rejected = []
    attacks = []
    for field in SEMANTIC_FIELDS:
        if field in ("status", "window_identity_sha256", "total_cycles",
                     "cycle_classes", "exact_fields"):
            continue
        attacks.append(("semantic_" + field, lambda value, key=field:
            value["baseline_exact"].__setitem__(key,
                False if type(value["baseline_exact"][key]) is bool else
                "d" * 64 if type(value["baseline_exact"][key]) is str else
                value["baseline_exact"][key] + 1)))
    attacks.extend([
        ("semantic_status", lambda value: value["baseline_exact"].__setitem__("status", "PASS_BUT_WRONG")),
        ("semantic_window", lambda value: value["baseline_exact"].__setitem__("window_identity_sha256", "d" * 64)),
        ("baseline_total", lambda value: value["baseline_exact"].__setitem__("total_cycles", 624)),
        ("candidate_total_bool", lambda value: value["candidate_exact"].__setitem__("total_cycles", True)),
        ("cycle_class", lambda value: value["baseline_exact"]["cycle_classes"].__setitem__(sorted(M1052.CYCLE_CLASSES)[0], 622)),
        ("exact_fields", lambda value: value["baseline_exact"]["exact_fields"].__setitem__(0, "forged")),
        ("homogenize_side_hash", lambda value: value["baseline_exact"].__setitem__("transaction_address_sha256", value["candidate_exact"]["transaction_address_sha256"])),
        ("arbitrary_side_hash", lambda value: value["candidate_exact"].__setitem__("terminal_readiness_sha256", "e" * 64)),
        ("mutate_binding", lambda value: value["side_specific_provenance"]["candidate"].__setitem__("binding_sha256", "f" * 64)),
        ("reset_semantic", lambda value: value["baseline_reset"].__setitem__("external_dependency_remap_count", 2)),
        ("reset_bool", lambda value: value["candidate_reset"].__setitem__("reset_expanded_request_count", True)),
        ("paired_hash", lambda value: value.__setitem__("paired_reset_semantics_sha256", "0" * 64)),
    ])
    for name, mutate in attacks:
        attacked = copy.deepcopy(row)
        mutate(attacked)
        try:
            validate_window_pair(attacked,
                row["candidate_exact"]["window_identity_sha256"],
                attacked["side_specific_provenance"])
        except (RuntimeError, TypeError, IndexError):
            rejected.append(name)
    require(len(rejected) == len(attacks), "validator attack survived")
    require(sha256(M1076_PATH) == M1076_SHA256 and
            sha256(DOC359) == DOC359_SHA256, "frozen source/docs drift")
    return {"status": "PASS_M1083_SIDE_NORMALIZED_VALIDATOR_SELFTEST__M1084_REQUIRED",
        "m1082_623_623_side_specific_difference_accepted": True,
        "semantic_projection_fields": list(SEMANTIC_FIELDS),
        "side_specific_fields": list(SIDE_SPECIFIC_FIELDS),
        "attacks_rejected": rejected,
        "real_payload_members_opened": False,
        "real_pilot_executed": False,
        "cpu_full_gpu_eda_remote_used": False,
        "launch_now": False}


def validate_source_only(contract_path, release_path, runner):
    contract_seal = verify_file_double_seal(contract_path)
    release_seal = verify_file_double_seal(release_path)
    contract = strict_json(contract_path)
    release = strict_json(release_path)
    require(contract["schema"] == "m1083_decoder_side_normalized_validator_repair_contract_v1" and
            contract["status"] == "M1083_SOURCE_ONLY__M1084_HAMMER_REQUIRED" and
            contract["launch_now"] is False, "M1083 contract drift")
    require(release["schema"] == "m1083_decoder_side_normalized_validator_repair_release_v1" and
            release["status"] == "M1083_RELEASE_FROZEN__M1084_REQUIRED__NO_LAUNCH" and
            release["launch_now"] is False and release["authorization"]["m1085_attempt"] is False,
            "M1083 release drift")
    identities = contract["identity"]
    for name, path in (("driver", Path(__file__).resolve()), ("runner", Path(runner)),
                       ("m1076", M1076_PATH), ("docs359", DOC359)):
        require(path.is_file() and not path.is_symlink() and
                sha256(path) == identities[name]["sha256"], name + " identity drift")
    require(sha256(M1082_DIR / "SHA256SUMS.seal.sha256") == M1082_OUTER_SHA256,
            "M1082 outer identity drift")
    exact_tree(release["contract_sha256"], sha256(contract_path), ("release", "contract_sha256"))
    return {"status": "PASS_M1083_SOURCE_ONLY__M1084_REQUIRED_NO_LAUNCH",
        "contract_sha256": sha256(contract_path),
        "release_sha256": sha256(release_path),
        "contract_outer_seal_file_sha256": contract_seal["outer_seal_file_sha256"],
        "release_outer_seal_file_sha256": release_seal["outer_seal_file_sha256"],
        "driver_sha256": sha256(Path(__file__).resolve()),
        "runner_sha256": sha256(runner),
        "real_payload_members_opened": False, "launch_now": False}


def safe_path(path, role):
    path = Path(path)
    require(path.is_absolute() and path.parent.resolve() == RESULTS.resolve() and
            not path.is_symlink(), role + " runtime path drift")
    if role == "attempt": require(path.name == ATTEMPT_NAME, "attempt namespace drift")
    elif role == "result": require(path.name == RESULT_NAME, "result namespace drift")
    elif role == "work": require(path.name.startswith("." + RESULT_NAME + ".work."), "work namespace drift")
    elif role == "quarantine": require(path.name.startswith(RESULT_NAME + ".failed_or_incomplete."), "quarantine namespace drift")
    else: raise RuntimeError("unknown runtime role")
    return path


def validate_m1084(review_sha, manifest_sha, outer_sha):
    require(M1084_DIR.is_dir(), "M1084 absent")
    sealed = M1076.BASE.verify_flat_seal(M1084_DIR)
    require(sha256(M1084_DIR / "review.json") == review_sha and
            sealed["manifest_sha256"] == manifest_sha and
            sealed["outer_seal_file_sha256"] == outer_sha,
            "M1084 caller-pinned authority drift")
    review = strict_json(M1084_DIR / "review.json")
    require(review["status"] == "PASS_M1084_M1083_SIDE_NORMALIZED_VALIDATOR_HAMMER__GO_ONE_M1085_ATTEMPT" and
            review["authorization"]["one_m1085_attempt"] is True and
            review["authorization"]["real_payload_after_attempt_only"] is True and
            review["authorization"]["eda_gpu_remote"] is False,
            "M1084 authorization drift")
    return {"review_sha256": review_sha, "manifest_sha256": manifest_sha,
            "outer_seal_file_sha256": outer_sha}


def runtime_contract_value(path=CONTRACT):
    value = strict_json(path)
    require(type(value) is dict and value["schema"] ==
            "m1083_decoder_side_normalized_validator_repair_contract_v1" and
            value["status"] == "M1083_SOURCE_ONLY__M1084_HAMMER_REQUIRED" and
            value["launch_now"] is False, "runtime contract drift")
    for key in ("source_identity", "frozen_payload", "authority"):
        require(type(value.get(key)) is dict, "runtime contract field absent: " + key)
    return value


def configure_frozen_runtime():
    """Retarget frozen M1076 machinery without editing any frozen source file."""
    M1076.CONTRACT = CONTRACT
    M1076.M1077_DIR = M1084_DIR
    M1076.ATTEMPT_NAME = ATTEMPT_NAME
    M1076.RESULT_NAME = RESULT_NAME
    M1076.ATTEMPT_SCHEMA = "m1085_decoder_side_normalized_attempt_v1"
    M1076.CONTEXT_SCHEMA = "m1085_decoder_side_normalized_context_v1"
    M1076.PAYLOAD_SCHEMA = "m1085_decoder_side_normalized_payload_receipt_v1"
    M1076.RAW_SCHEMA = "m1085_decoder_side_normalized_raw_cycles_v1"
    M1076.RESULT_SCHEMA = "m1085_decoder_side_normalized_result_v1"
    M1076.contract_value = runtime_contract_value
    M1076.validate_m1077 = validate_m1084
    M1076.validate_layer = validate_layer
    M1076.M1060.BASE.transform_layer = transform_layer
    return M1076


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--validate-source-only", action="store_true")
    modes.add_argument("--validate-authority", action="store_true")
    modes.add_argument("--validate-namespace", action="store_true")
    for name in ("consume-attempt", "validate-payload-after-attempt", "run-pilot",
                 "assemble", "publish", "quarantine"):
        modes.add_argument("--" + name, action="store_true")
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--release", type=Path, default=RELEASE)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--expected-review-sha")
    parser.add_argument("--expected-manifest-sha")
    parser.add_argument("--expected-outer-sha")
    parser.add_argument("--path", type=Path)
    parser.add_argument("--role", choices=("attempt", "result", "work", "quarantine"))
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--quarantine-path", type=Path)
    parser.add_argument("--return-code", type=int, default=1)
    parser.add_argument("--expected-contract-sha")
    args = parser.parse_args()
    if args.self_test:
        output = self_test()
    elif args.validate_source_only:
        require(args.runner, "runner absent")
        output = validate_source_only(args.contract, args.release, args.runner)
    elif args.validate_authority:
        require(args.expected_review_sha and args.expected_manifest_sha and
                args.expected_outer_sha, "M1084 pins absent")
        output = validate_m1084(args.expected_review_sha,
            args.expected_manifest_sha, args.expected_outer_sha)
    elif args.validate_namespace:
        require(args.path and args.role, "namespace path/role absent")
        output = {"status": "PASS_M1083_NEW_NAMESPACE", "path": str(safe_path(args.path, args.role))}
    else:
        require(args.runner and args.expected_contract_sha == sha256(args.contract) and
                args.expected_review_sha and args.expected_manifest_sha and
                args.expected_outer_sha, "runtime caller pins absent")
        authority = validate_m1084(args.expected_review_sha,
            args.expected_manifest_sha, args.expected_outer_sha)
        runtime = configure_frozen_runtime()
        if args.consume_attempt:
            require(args.attempt, "attempt absent")
            output = runtime.consume_attempt(args.attempt, args.runner,
                args.expected_contract_sha, authority)
        elif args.validate_payload_after_attempt:
            require(args.attempt and args.work, "payload paths absent")
            output = runtime.validate_payload_after_attempt(args.attempt, args.work,
                args.runner, args.expected_contract_sha, authority)
        elif args.run_pilot:
            require(args.attempt and args.work, "run paths absent")
            output = runtime.run_pilot(args.attempt, args.work, args.runner,
                args.expected_contract_sha, authority)
        elif args.assemble:
            require(args.attempt and args.work, "assemble paths absent")
            output = runtime.assemble(args.work, args.attempt, args.runner,
                args.expected_contract_sha, authority)
        elif args.publish:
            require(args.attempt and args.work and args.result, "publish paths absent")
            output = runtime.publish(args.work, args.result, args.attempt, args.runner,
                args.expected_contract_sha, authority)
        else:
            require(args.work and args.quarantine_path, "quarantine paths absent")
            output = runtime.quarantine(args.work, args.quarantine_path,
                                        args.return_code)
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
