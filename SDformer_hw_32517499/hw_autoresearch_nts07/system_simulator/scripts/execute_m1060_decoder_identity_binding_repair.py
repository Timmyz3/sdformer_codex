#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1060 additive identity-binding repair for the stopped decoder pilot.

Pre-attempt validation reads only source/review seals and the M699 root seal
metadata.  Full manifest and payload-member access is allowed only after the
canonical M1062 attempt exists.  A canonical context derived from the frozen
contract, canonical attempt, fully verified M699 manifest and actual selected
member files is then re-derived by run and assemble; user-supplied 64-hex
strings are never authority.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
REPO = HW.parent
RESULTS = HW / "results"
BASE_PATH = HERE / "execute_m1052_decoder_stratified_block_reset_pilot_repair.py"
BASE_SHA256 = "756bf90d52505a68f089dd42296244b94b9c9a50cf013efc0dbc02cd6bb25cec"
CONTRACT = HW / "contracts/m1060_decoder_identity_binding_repair_contract_r1_20260830.json"
M1053_DIR = HW / "reviews/m1053_m1052_decoder_stratified_block_reset_pilot_repair_hammer_r1_20260829"
M1061_DIR = HW / "reviews/m1061_m1060_decoder_identity_binding_repair_hammer_r1_20260830"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOC359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
SCHEMA = "m1060_decoder_identity_binding_repair_v1"
CONTEXT_SCHEMA = "m1062_decoder_canonical_payload_context_v1"
PAYLOAD_SCHEMA = "m1062_postattempt_payload_identity_receipt_v1"
RAW_SCHEMA = "m1062_decoder_stratified_block_reset_raw_cycles_v1"
RESULT_SCHEMA = "m1062_decoder_stratified_block_reset_result_v1"
ATTEMPT_SCHEMA = "m1062_decoder_pilot_attempt_v1"
ATTEMPT_NAME = ".m1062_m1060_decoder_identity_binding_pilot_attempt_consumed"
RESULT_NAME = "m1062_m1060_decoder_identity_binding_pilot_r1_20260830"
LAYERS = ("D0", "D2", "D3")
FROZEN_SELECTED = (
    {"layer": "D0", "population_id": "M699_INTERLAKEN_01_A_S10",
     "sequence": "interlaken_01_a", "sample_id": 0, "module_index": 0,
     "route": "EXACT_BINARY_BITPACK",
     "relative_path": "calls/s00_d0.binary.le.bitpack",
     "packed_sha256": "2af601cc112e1c39c1e850f7c776f71a28957d52df2164f0e79a988a9dbdf1be"},
    {"layer": "D2", "population_id": "M699_INTERLAKEN_01_A_S10",
     "sequence": "interlaken_01_a", "sample_id": 0, "module_index": 2,
     "route": "EXACT_BINARY_BITPACK",
     "relative_path": "calls/s00_d2.binary.le.bitpack",
     "packed_sha256": "948d72523e23384e603a83739408aee4decbb1afb2c21cd8d2a77f3bff9a3e64"},
    {"layer": "D3", "population_id": "M699_INTERLAKEN_01_A_S10",
     "sequence": "interlaken_01_a", "sample_id": 0, "module_index": 3,
     "route": "EXACT_BINARY_BITPACK",
     "relative_path": "calls/s00_d3.binary.le.bitpack",
     "packed_sha256": "0a8567d62df9aaf31ab19d7f1ad78366171be850a63562837d86f12570be86e3"},
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
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")).hexdigest()


def strict_json(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("non-finite JSON token: " + token)))


def atomic_json(path, value):
    temporary = Path(path).with_name(Path(path).name + ".tmp." + str(os.getpid()))
    payload = json.dumps(value, indent=2, sort_keys=True,
                         ensure_ascii=False, allow_nan=False) + "\n"
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_pinned(path, expected, name):
    require(Path(path).is_file() and not Path(path).is_symlink() and
            sha256(path) == expected, name + " identity drift")
    spec = importlib.util.spec_from_file_location("m1060_" + name, path)
    require(spec is not None and spec.loader is not None, "cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_pinned(BASE_PATH, BASE_SHA256, "frozen_m1052")
M1048, M785 = BASE.M1048, BASE.M785


def exact_dict(value, keys, name):
    require(type(value) is dict and set(value) == set(keys), name + " schema drift")
    return value


def exact_list(value, length, name):
    require(type(value) is list and len(value) == length, name + " shape drift")
    return value


def exact_bool(value, expected):
    require(type(value) is bool and value is expected, "boolean boundary drift")
    return value


def exact_int(value, minimum=0):
    require(type(value) is int and value >= minimum, "integer/range drift")
    return value


def selected_record(value, expected_layer):
    exact_dict(value, {"layer", "population_id", "sequence", "sample_id",
               "module_index", "route", "relative_path", "packed_sha256"},
               "frozen selected record")
    require(value["layer"] == expected_layer and
            value["population_id"] == M1048.POPULATION_ID and
            value["sequence"] == M1048.SEQUENCE and
            exact_int(value["sample_id"]) == 0 and
            exact_int(value["module_index"]) == M1048.MODULE_BY_LAYER[expected_layer] and
            value["route"] == "EXACT_BINARY_BITPACK", "selected identity drift")
    BASE.hash_value(value["packed_sha256"])
    relative = PurePosixPath(value["relative_path"])
    require(not relative.is_absolute() and relative.parts and
            relative.parts[0] == "calls" and ".." not in relative.parts and
            "." not in relative.parts, "selected path unsafe")
    return value


def validate_contract(value):
    BASE.reject_forbidden_semantic_keys(value)
    exact_dict(value, {"schema", "status", "launch_now", "objective", "workload",
               "d1", "pair", "sampling", "pre_attempt", "post_attempt", "output",
               "authority", "source_identity", "frozen_payload", "claim_boundary",
               "next_gate"}, "contract")
    require(value["schema"] == SCHEMA and value["status"] ==
            "IDENTITY_BINDING_REPAIR_SOURCE_ONLY__M1061_HAMMER_REQUIRED",
            "contract header drift")
    exact_bool(value["launch_now"], False)
    require(value["workload"] == {"population_id": M1048.POPULATION_ID,
            "sequence": M1048.SEQUENCE, "sample_id": 0, "timestep": 0,
            "config": M1048.CONFIG, "layers": list(LAYERS)}, "workload drift")
    require(value["d1"] == {"status": "DIAGNOSTIC_ONLY",
            "generator_allowed": False, "scheduler_allowed": False,
            "numeric_equivalence_admitted": False}, "D1 drift")
    require(value["pair"] == {"role": "SELF_MATCHED_PROTOCOL_CALIBRATION",
            "candidate_body": "A1_OSG", "baseline_body": "A1_OSG",
            "performance_comparison": False}, "pair drift")
    require(value["sampling"] == {"strata": list(BASE.STRATA),
            "source_census": 1, "pilot_per_noncensus_stratum": BASE.PILOT,
            "selection_seed": M1048.SELECTION_SEED,
            "window_expanded_request_cap": M1048.CAP,
            "selection_before_replay": True}, "sampling drift")
    require(value["pre_attempt"] == {"payload_member_access": "FORBIDDEN",
            "allowed_checks": ["contract", "code", "review_seals",
                               "payload_root_seal_metadata"],
            "canonical_attempt_before_payload_validation": True},
            "pre-attempt boundary drift")
    require(value["post_attempt"] == {"full_payload_member_verification": True,
            "canonical_context_written": True,
            "run_rederives_canonical_context": True,
            "assemble_rederives_canonical_context": True,
            "raw_record_selected_member_cross_binding": True,
            "failure_quarantine": True}, "post-attempt boundary drift")
    require(value["output"] == {"raw_cycle_samples": True,
            "cycle_ci_bounds": True, "derived_performance_values": False,
            "recursive_exact_schema": True}, "output boundary drift")
    exact_dict(value["authority"], {"m1053_negative", "m1061_required_status"},
               "authority")
    neg = exact_dict(value["authority"]["m1053_negative"], {"directory",
        "review_sha256", "manifest_sha256", "outer_seal_file_sha256"},
        "M1053 negative")
    require(neg == {"directory":
            "reviews/m1053_m1052_decoder_stratified_block_reset_pilot_repair_hammer_r1_20260829",
            "review_sha256": "a0c544a0fd081e0589da6a91d9d7c9a694d4d5bb8c7a8e6fca48fbbb327e3e05",
            "manifest_sha256": "18fa9a077ba835afd6bf518fd04fb32aa756375019745e4c051197ce42352cf3",
            "outer_seal_file_sha256": "3c13b12faaf8956e947191d832b2f75439a8bdd01e421327c288cfc5876f02ea"},
            "M1053 frozen identity drift")
    require(value["authority"]["m1061_required_status"] ==
            "PASS_M1061_M1060_IDENTITY_BINDING_HAMMER__GO_ONE_M1062_ATTEMPT",
            "M1061 status drift")
    exact_dict(value["source_identity"], {"driver", "runner", "m1052",
               "m785_contract", "docs359"}, "source identity")
    for name in value["source_identity"]:
        exact_dict(value["source_identity"][name], {"path", "sha256"}, name)
        BASE.hash_value(value["source_identity"][name]["sha256"])
    frozen = exact_dict(value["frozen_payload"], {"directory",
        "m699_manifest_sha256", "m699_root_manifest_sha256",
        "m699_outer_seal_file_sha256", "selected_records"}, "frozen payload")
    require(frozen["directory"] ==
            "system_handoff/outgoing/m699_h67_ep35_multisequence_decoder_payload_s3x10_r1_20260828" and
            frozen["m699_manifest_sha256"] ==
            "e2d7c92a038c213b590603ff534a33f3579bf1224cc3f56c11629e1d4c813dc0" and
            frozen["m699_root_manifest_sha256"] ==
            "27b35748b81d32907410ada0fbecfaa869a6ce1c3039e94ab3da2e52a8f46053" and
            frozen["m699_outer_seal_file_sha256"] ==
            "eaf975a9a1a4829b2c0a2251e7ef297abd53b83b30e23630e5ce51db5c5de18c",
            "frozen M699 root identity drift")
    exact_list(frozen["selected_records"], len(LAYERS), "frozen selected records")
    require(frozen["selected_records"] == list(FROZEN_SELECTED),
            "frozen selected record path/SHA drift")
    for layer, row in zip(LAYERS, frozen["selected_records"]):
        selected_record(row, layer)
    exact_dict(value["claim_boundary"], {"paper_citable", "decoder_complete",
               "table_a_row", "system_performance_claim", "local_performance_claim",
               "continuous_row_cycles", "d1_scheduled", "eda_gpu_remote_used"},
               "claim boundary")
    require(all(exact_bool(item, False) is False
                for item in value["claim_boundary"].values()), "claim expansion")
    return True


def contract_value(path=CONTRACT):
    value = strict_json(path)
    validate_contract(value)
    return value


def validate_pre_attempt_source(contract_path, runner):
    value = contract_value(contract_path)
    identities = value["source_identity"]
    for name, path in (("driver", Path(__file__).resolve()),
                       ("runner", Path(runner)), ("m1052", BASE_PATH),
                       ("m785_contract", HW / identities["m785_contract"]["path"]),
                       ("docs359", DOC359)):
        require(Path(path).is_file() and not Path(path).is_symlink() and
                sha256(path) == identities[name]["sha256"], name + " drift")
    neg = value["authority"]["m1053_negative"]
    sealed = BASE.verify_flat_seal(HW / neg["directory"])
    require(sha256(HW / neg["directory"] / "review.json") == neg["review_sha256"] and
            sealed["manifest_sha256"] == neg["manifest_sha256"] and
            sealed["outer_seal_file_sha256"] == neg["outer_seal_file_sha256"],
            "M1053 seal drift")
    negative = strict_json(HW / neg["directory"] / "review.json")
    require(negative["status"] ==
            "FAIL_M1053_M1052_POSTRUN_PAYLOAD_IDENTITY_REBINDING_ESCAPE__STOP_M1054",
            "M1053 negative status drift")
    frozen = value["frozen_payload"]
    meta = BASE.verify_payload_root_seal_metadata_only(
        HW / frozen["directory"], frozen["m699_root_manifest_sha256"],
        frozen["m699_outer_seal_file_sha256"], frozen["m699_manifest_sha256"])
    return {"status":
            "PASS_M1060_PREATTEMPT_SOURCE_WITH_ZERO_PAYLOAD_MEMBER_ACCESS",
            "contract_sha256": sha256(contract_path),
            "driver_sha256": sha256(Path(__file__).resolve()),
            "runner_sha256": sha256(runner), "payload_root_seal": meta,
            "payload_members_opened": False, "payload_members_statted": False,
            "payload_members_hashed": False, "launch_now": False}


def validate_m1061(review_sha, manifest_sha, outer_sha):
    sealed = BASE.verify_flat_seal(M1061_DIR)
    require(sha256(M1061_DIR / "review.json") == review_sha and
            sealed["manifest_sha256"] == manifest_sha and
            sealed["outer_seal_file_sha256"] == outer_sha,
            "M1061 caller-pinned authority drift")
    value = strict_json(M1061_DIR / "review.json")
    require(value["status"] ==
            "PASS_M1061_M1060_IDENTITY_BINDING_HAMMER__GO_ONE_M1062_ATTEMPT" and
            value["authorization"]["one_m1062_attempt"] is True and
            value["authorization"]["real_payload_after_attempt_only"] is True and
            value["authorization"]["eda_gpu_remote"] is False,
            "M1061 authorization drift")
    return {"review_sha256": review_sha, "manifest_sha256": manifest_sha,
            "outer_seal_file_sha256": outer_sha}


def safe_path(path, role):
    path = Path(path)
    require(path.is_absolute() and path.parent.resolve() == RESULTS.resolve() and
            not path.is_symlink(), role + " runtime path drift")
    if role == "attempt":
        require(path.name == ATTEMPT_NAME, "attempt namespace drift")
    elif role == "result":
        require(path.name == RESULT_NAME, "result namespace drift")
    elif role == "work":
        require(path.name.startswith("." + RESULT_NAME + ".work."),
                "work namespace drift")
    elif role == "quarantine":
        require(path.name.startswith(RESULT_NAME + ".failed_or_incomplete."),
                "quarantine namespace drift")
    else:
        raise RuntimeError("unknown runtime role")
    return path


def consume_attempt(path, runner, contract_sha, authority):
    path = safe_path(path, "attempt")
    contract = contract_value()
    require(contract_sha == sha256(CONTRACT) and Path(runner).is_file() and
            not Path(runner).is_symlink() and
            sha256(runner) == contract["source_identity"]["runner"]["sha256"],
            "attempt source/contract identity drift")
    require(not path.exists(), "M1062 attempt already consumed")
    os.mkdir(path, 0o700)
    receipt = {"schema": ATTEMPT_SCHEMA,
        "status": "M1062_ATTEMPT_CONSUMED_BEFORE_PAYLOAD_MEMBER_ACCESS",
        "consumed_unix_ns": time.time_ns(), "runner_sha256": sha256(runner),
        "contract_sha256": contract_sha, "m1061_authority": dict(authority),
        "payload_members_opened": False, "payload_members_statted": False,
        "payload_members_hashed": False, "paper_citable": False}
    atomic_json(path / "attempt.json", receipt)
    return receipt


def validate_attempt(path, authority):
    path = safe_path(path, "attempt")
    require(path.is_dir() and not path.is_symlink() and
            {item.name for item in path.iterdir()} == {"attempt.json"},
            "attempt receipt absent")
    value = strict_json(path / "attempt.json")
    exact_dict(value, {"schema", "status", "consumed_unix_ns", "runner_sha256",
               "contract_sha256", "m1061_authority", "payload_members_opened",
               "payload_members_statted", "payload_members_hashed",
               "paper_citable"}, "attempt")
    contract = contract_value()
    require(value["schema"] == ATTEMPT_SCHEMA and value["status"] ==
            "M1062_ATTEMPT_CONSUMED_BEFORE_PAYLOAD_MEMBER_ACCESS" and
            exact_int(value["consumed_unix_ns"], 1) > 0 and
            value["runner_sha256"] == contract["source_identity"]["runner"]["sha256"] and
            value["contract_sha256"] == sha256(CONTRACT) and
            value["m1061_authority"] == dict(authority) and
            all(value[key] is False for key in ("payload_members_opened",
                "payload_members_statted", "payload_members_hashed", "paper_citable")),
            "attempt exact identity drift")
    return value


def normalized_selected(row, layer):
    return {"layer": layer, "population_id": row["population_id"],
        "sequence": row["sequence"], "sample_id": int(row["sample_id"]),
        "module_index": int(row["module_index"]), "route": row["route"],
        "relative_path": row["relative_path"], "packed_sha256": row["packed_sha256"]}


def verified_member(payload_dir, frozen_record):
    relative = PurePosixPath(frozen_record["relative_path"])
    require(not relative.is_absolute() and ".." not in relative.parts and
            relative.parts and relative.parts[0] == "calls", "payload path unsafe")
    target = payload_dir.joinpath(*relative.parts)
    require(target.is_file() and not target.is_symlink() and
            target.resolve().is_relative_to(payload_dir.resolve()),
            "selected payload member nonexistent/unsafe")
    actual = sha256(target)
    require(actual == frozen_record["packed_sha256"],
            "selected payload member SHA drift")
    return {**copy.deepcopy(frozen_record), "payload_member_sha256": actual}


def build_canonical_context(attempt, runner, contract_sha, authority):
    attempt_value = validate_attempt(attempt, authority)
    contract = contract_value()
    require(Path(runner).is_file() and not Path(runner).is_symlink() and
            sha256(runner) == attempt_value["runner_sha256"] ==
            contract["source_identity"]["runner"]["sha256"] and
            contract_sha == attempt_value["contract_sha256"] == sha256(CONTRACT),
            "canonical attempt/runner/contract cross-binding drift")
    m785_path = HW / contract["source_identity"]["m785_contract"]["path"]
    M785.validate_source_contract(REPO, m785_path)
    frozen = contract["frozen_payload"]
    payload_dir = HW / frozen["directory"]
    sealed = M785.verify_sealed_directory(payload_dir)
    require(sha256(payload_dir / "manifest.json") == frozen["m699_manifest_sha256"] and
            sealed["manifest_sha256"] == frozen["m699_root_manifest_sha256"] and
            sealed["outer_seal_file_sha256"] == frozen["m699_outer_seal_file_sha256"],
            "frozen M699 root/manifest/outer drift")
    manifest = M785.strict_json(payload_dir / "manifest.json")
    records = M785.normalized_population_records(manifest, M1048.POPULATION_ID)
    selected = []
    for layer, frozen_record in zip(LAYERS, frozen["selected_records"]):
        actual_record = normalized_selected(M1048.select_record(records, layer), layer)
        require(actual_record == frozen_record,
                "selected manifest record differs from frozen contract")
        selected.append(verified_member(payload_dir, frozen_record))
    return {"schema": CONTEXT_SCHEMA,
        "attempt": {"attempt_json_sha256": sha256(Path(attempt) / "attempt.json"),
            "runner_sha256": attempt_value["runner_sha256"],
            "contract_sha256": attempt_value["contract_sha256"],
            "m1061_authority": copy.deepcopy(attempt_value["m1061_authority"])},
        "payload": {"directory": frozen["directory"],
            "m699_manifest_sha256": frozen["m699_manifest_sha256"],
            "m699_root_manifest_sha256": frozen["m699_root_manifest_sha256"],
            "m699_outer_seal_file_sha256": frozen["m699_outer_seal_file_sha256"],
            "selected_records": selected},
        "identity_binding": "FROZEN_CONTRACT_PLUS_CANONICAL_ATTEMPT_PLUS_VERIFIED_MEMBERS",
        "d1_scheduled": False, "paper_citable": False}


def validate_canonical_context(value, expected):
    BASE.reject_forbidden_semantic_keys(value)
    require(value == expected, "canonical context exact identity drift")
    return True


def make_payload_receipt(context):
    return {"schema": PAYLOAD_SCHEMA,
        "status": "PASS_M1062_POSTATTEMPT_CANONICAL_PAYLOAD_IDENTITY",
        "canonical_context_sha256": canonical_sha(context),
        "attempt": copy.deepcopy(context["attempt"]),
        "payload": copy.deepcopy(context["payload"]),
        "payload_members_verified": True, "post_attempt": True,
        "d1_scheduled": False, "paper_citable": False}


def validate_payload_receipt(value, context):
    BASE.reject_forbidden_semantic_keys(value)
    require(value == make_payload_receipt(context),
            "payload receipt not exactly bound to canonical context")
    return True


def validate_payload_after_attempt(attempt, work, runner, contract_sha, authority):
    work = safe_path(work, "work")
    require(work.is_dir() and not any(work.iterdir()), "postattempt work not fresh")
    context = build_canonical_context(attempt, runner, contract_sha, authority)
    receipt = make_payload_receipt(context)
    atomic_json(work / "canonical_context.json", context)
    atomic_json(work / "payload_validation.json", receipt)
    return receipt


def expected_raw_record(selected):
    return {"population_id": selected["population_id"],
        "sequence": selected["sequence"], "sample_id": selected["sample_id"],
        "module_index": selected["module_index"], "timestep": 0,
        "config": M1048.CONFIG, "route": selected["route"],
        "relative_path": selected["relative_path"],
        "packed_sha256": selected["packed_sha256"]}


def bind_raw_records(raw, context):
    exact_list(raw["layers"], len(LAYERS), "raw layers")
    selected_rows = context["payload"]["selected_records"]
    for layer, row, selected in zip(LAYERS, raw["layers"], selected_rows):
        require(row["layer"] == layer and
                row["record_identity"] == expected_raw_record(selected) and
                row["verified_payload_member_sha256"] ==
                selected["payload_member_sha256"] == selected["packed_sha256"],
                "raw layer/selected/verified-member cross-binding drift")
    return True


def validate_layer(value, expected_layer, selected):
    require(type(value) is dict and "verified_payload_member_sha256" in value,
            "verified member identity absent")
    base_value = copy.deepcopy(value)
    verified_sha = base_value.pop("verified_payload_member_sha256")
    BASE.validate_layer(base_value, expected_layer)
    require(base_value["record_identity"] == expected_raw_record(selected) and
            verified_sha == selected["payload_member_sha256"] ==
            selected["packed_sha256"], "layer payload identity drift")
    return True


def validate_raw(value, context):
    BASE.reject_forbidden_semantic_keys(value)
    exact_dict(value, {"schema", "status", "workload", "pair_role",
               "canonical_context_sha256", "layers", "exact_mismatch_count",
               "d1", "claim_boundary"}, "raw")
    require(value["schema"] == RAW_SCHEMA and value["status"] ==
            "PASS_M1062_RAW_CYCLES__RESULT_HAMMER_REQUIRED" and
            value["pair_role"] == "SELF_MATCHED_PROTOCOL_CALIBRATION" and
            value["canonical_context_sha256"] == canonical_sha(context),
            "raw header/context drift")
    require(value["workload"] == {"population_id": M1048.POPULATION_ID,
            "sequence": M1048.SEQUENCE, "sample_id": 0, "timestep": 0,
            "config": M1048.CONFIG, "layers": list(LAYERS)}, "raw workload drift")
    bind_raw_records(value, context)
    for layer, row, selected in zip(LAYERS, value["layers"],
                                    context["payload"]["selected_records"]):
        validate_layer(row, layer, selected)
    require(exact_int(value["exact_mismatch_count"]) == 0 and
            value["d1"] == {"status": "DIAGNOSTIC_ONLY", "scheduled": False,
                             "numeric_equivalence_admitted": False},
            "raw exact/D1 drift")
    require(value["claim_boundary"] == {"paper_citable": False,
            "decoder_complete": False, "table_a_row": False,
            "system_performance_claim": False, "local_performance_claim": False,
            "continuous_row_cycles": False}, "raw claim drift")
    return True


def make_result(raw, raw_sha, payload_sha, context_sha):
    return {"schema": RESULT_SCHEMA,
        "status": "PASS_M1062_DIAGNOSTIC_RAW_CYCLE_PILOT__RESULT_HAMMER_REQUIRED",
        "result_role": "PROTOCOL_CALIBRATION_ONLY",
        "canonical_context_sha256": context_sha,
        "raw_windows_sha256": raw_sha, "payload_validation_sha256": payload_sha,
        "layers": [{"layer": row["layer"],
            "record_identity": copy.deepcopy(row["record_identity"]),
            "verified_payload_member_sha256": row["verified_payload_member_sha256"],
            "selection_identity_sha256": row["selection_identity_sha256"],
            "block_population_index_sha256": row["block_population_index_sha256"],
            "transaction_assignment_census_sha256":
                row["transaction_assignment_census_sha256"],
            "generated_compressed_transactions": row["generated_compressed_transactions"],
            "assigned_compressed_transactions": row["assigned_compressed_transactions"],
            "coverage": copy.deepcopy(row["coverage"]),
            "source_census_cycles": copy.deepcopy(row["source_census_cycles"]),
            "cycle_ci_envelope": copy.deepcopy(row["cycle_ci_envelope"]),
            "window_count": len(row["windows"]),
            "exact_mismatch_count": row["exact_mismatch_count"]}
            for row in raw["layers"]],
        "total_window_count": sum(len(row["windows"]) for row in raw["layers"]),
        "exact_mismatch_count": 0, "d1_scheduled": False,
        "claim_boundary": {"paper_citable": False, "decoder_complete": False,
            "table_a_row": False, "system_performance_claim": False,
            "local_performance_claim": False, "continuous_row_cycles": False,
            "eda_gpu_remote_used": False},
        "next_gate": "Independent receipt-blind M1063 result hammer"}


def validate_result(value, raw, raw_sha, payload_sha, context_sha):
    BASE.reject_forbidden_semantic_keys(value)
    require(value == make_result(raw, raw_sha, payload_sha, context_sha),
            "result exact derivation drift")
    return True


def load_work_context(work, expected_context):
    stored = strict_json(work / "canonical_context.json")
    validate_canonical_context(stored, expected_context)
    payload = strict_json(work / "payload_validation.json")
    validate_payload_receipt(payload, expected_context)
    return stored, payload


def run_pilot(attempt, work, runner, contract_sha, authority):
    work = safe_path(work, "work")
    require(work.is_dir() and not work.is_symlink() and
            {item.name for item in work.iterdir()} ==
            {"canonical_context.json", "payload_validation.json"},
            "canonical context receipts required before run")
    context = build_canonical_context(attempt, runner, contract_sha, authority)
    load_work_context(work, context)
    payload_root, records, mapper, oracles = M1048._context()
    old_layers = []
    for layer, selected in zip(LAYERS, context["payload"]["selected_records"]):
        record = M1048.select_record(records, layer)
        require(normalized_selected(record, layer) ==
                {key: selected[key] for key in ("layer", "population_id", "sequence",
                    "sample_id", "module_index", "route", "relative_path",
                    "packed_sha256")}, "run selected record rebinding drift")
        old_layers.append(M1048.replay_layer(layer, record, payload_root,
                                             mapper, oracles))
    layers = []
    for old, selected in zip(old_layers, context["payload"]["selected_records"]):
        row = BASE.transform_layer(old)
        row["verified_payload_member_sha256"] = selected["payload_member_sha256"]
        layers.append(row)
    raw = {"schema": RAW_SCHEMA,
        "status": "PASS_M1062_RAW_CYCLES__RESULT_HAMMER_REQUIRED",
        "workload": {"population_id": M1048.POPULATION_ID,
            "sequence": M1048.SEQUENCE, "sample_id": 0, "timestep": 0,
            "config": M1048.CONFIG, "layers": list(LAYERS)},
        "pair_role": "SELF_MATCHED_PROTOCOL_CALIBRATION",
        "canonical_context_sha256": canonical_sha(context), "layers": layers,
        "exact_mismatch_count": 0,
        "d1": {"status": "DIAGNOSTIC_ONLY", "scheduled": False,
               "numeric_equivalence_admitted": False},
        "claim_boundary": {"paper_citable": False, "decoder_complete": False,
            "table_a_row": False, "system_performance_claim": False,
            "local_performance_claim": False, "continuous_row_cycles": False}}
    validate_raw(raw, context)
    atomic_json(work / "raw_windows.json", raw)
    result = make_result(raw, sha256(work / "raw_windows.json"),
        sha256(work / "payload_validation.json"), canonical_sha(context))
    validate_result(result, raw, result["raw_windows_sha256"],
                    result["payload_validation_sha256"], canonical_sha(context))
    atomic_json(work / "result.json", result)
    (work / "RUN_COMPLETE.txt").write_text(result["status"] + "\n", encoding="utf-8")
    return result


def assemble(work, attempt, runner, contract_sha, authority):
    work = safe_path(work, "work")
    expected_files = {"canonical_context.json", "payload_validation.json",
                      "raw_windows.json", "result.json", "RUN_COMPLETE.txt"}
    require(work.is_dir() and not work.is_symlink() and
            {item.name for item in work.iterdir()} == expected_files and
            all(item.is_file() and not item.is_symlink() for item in work.iterdir()),
            "work exact-set drift")
    context = build_canonical_context(attempt, runner, contract_sha, authority)
    load_work_context(work, context)
    raw = strict_json(work / "raw_windows.json")
    result = strict_json(work / "result.json")
    validate_raw(raw, context)
    validate_result(result, raw, sha256(work / "raw_windows.json"),
                    sha256(work / "payload_validation.json"), canonical_sha(context))
    require((work / "RUN_COMPLETE.txt").read_text(encoding="utf-8") ==
            result["status"] + "\n", "completion token drift")
    lines = [sha256(work / name) + "  " + name for name in sorted(expected_files)]
    (work / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_sha = sha256(work / "SHA256SUMS")
    (work / "SHA256SUMS.seal.sha256").write_text(
        manifest_sha + "  SHA256SUMS\n", encoding="utf-8")
    return {"status": "PASS_M1062_IDENTITY_BOUND_OUTPUT_SEALED",
            "manifest_sha256": manifest_sha,
            "outer_seal_file_sha256": sha256(work / "SHA256SUMS.seal.sha256")}


def publish(work, result, attempt, runner, contract_sha, authority):
    work, result = safe_path(work, "work"), safe_path(result, "result")
    require(work.is_dir() and not result.exists(), "publish namespace drift")
    BASE.verify_flat_seal(work)
    require({item.name for item in work.iterdir()} == {
        "canonical_context.json", "payload_validation.json", "raw_windows.json",
        "result.json", "RUN_COMPLETE.txt", "SHA256SUMS",
        "SHA256SUMS.seal.sha256"}, "sealed publish exact-set drift")
    context = build_canonical_context(attempt, runner, contract_sha, authority)
    load_work_context(work, context)
    raw = strict_json(work / "raw_windows.json")
    result_value = strict_json(work / "result.json")
    validate_raw(raw, context)
    validate_result(result_value, raw, sha256(work / "raw_windows.json"),
                    sha256(work / "payload_validation.json"), canonical_sha(context))
    require((work / "RUN_COMPLETE.txt").read_text(encoding="utf-8") ==
            result_value["status"] + "\n", "publish completion token drift")
    os.replace(work, result)
    return {"status": "PASS_M1062_ATOMIC_RESULT_PUBLISHED"}


def quarantine(work, quarantine, return_code):
    work, quarantine = safe_path(work, "work"), safe_path(quarantine, "quarantine")
    require(work.is_dir() and not quarantine.exists(), "quarantine namespace drift")
    os.replace(work, quarantine)
    atomic_json(quarantine / "FAILURE.json", {"schema": "m1062_failure_v1",
        "status": "FAILED_OR_INCOMPLETE_DO_NOT_CITE", "return_code": int(return_code),
        "paper_citable": False})
    return {"status": "PASS_M1062_FAILURE_QUARANTINED"}


def synthetic_context():
    selected = []
    for layer, path, digest in (("D0", "calls/d0.bitpack", "1" * 64),
                                ("D2", "calls/d2.bitpack", "2" * 64),
                                ("D3", "calls/d3.bitpack", "3" * 64)):
        selected.append({"layer": layer, "population_id": M1048.POPULATION_ID,
            "sequence": M1048.SEQUENCE, "sample_id": 0,
            "module_index": M1048.MODULE_BY_LAYER[layer],
            "route": "EXACT_BINARY_BITPACK", "relative_path": path,
            "packed_sha256": digest, "payload_member_sha256": digest})
    return {"schema": CONTEXT_SCHEMA,
        "attempt": {"attempt_json_sha256": "4" * 64, "runner_sha256": "5" * 64,
            "contract_sha256": "6" * 64,
            "m1061_authority": {"review_sha256": "7" * 64,
                "manifest_sha256": "8" * 64,
                "outer_seal_file_sha256": "9" * 64}},
        "payload": {"directory": "synthetic", "m699_manifest_sha256": "a" * 64,
            "m699_root_manifest_sha256": "b" * 64,
            "m699_outer_seal_file_sha256": "c" * 64,
            "selected_records": selected},
        "identity_binding": "FROZEN_CONTRACT_PLUS_CANONICAL_ATTEMPT_PLUS_VERIFIED_MEMBERS",
        "d1_scheduled": False, "paper_citable": False}


def self_test():
    old = BASE.self_test()
    require(old["m1048_transactions"] == 85, "base synthetic drift")
    context = synthetic_context()
    receipt = make_payload_receipt(context)
    validate_payload_receipt(receipt, context)
    rejected = []
    for name, mutate in (
        ("all_fake_sha", lambda x: x["attempt"].update(
            {"attempt_json_sha256": "f" * 64, "runner_sha256": "e" * 64,
             "contract_sha256": "d" * 64})),
        ("nonexistent_path", lambda x: x["payload"]["selected_records"][0].update(
            {"relative_path": "calls/FORGED_DOES_NOT_EXIST.bitpack"})),
        ("relabel_and_rehash", lambda x: x["payload"]["selected_records"][1].update(
            {"relative_path": "calls/relabeled.bitpack", "packed_sha256": "f" * 64,
             "payload_member_sha256": "f" * 64}))):
        attack = copy.deepcopy(receipt)
        mutate(attack)
        try:
            validate_payload_receipt(attack, context)
        except RuntimeError:
            rejected.append(name)
    require(rejected == ["all_fake_sha", "nonexistent_path", "relabel_and_rehash"],
            "identity attack survived")
    raw_stub = {"layers": [{"layer": row["layer"],
        "record_identity": expected_raw_record(row),
        "verified_payload_member_sha256": row["payload_member_sha256"]}
        for row in context["payload"]["selected_records"]]}
    bind_raw_records(raw_stub, context)
    raw_stub["layers"][2]["record_identity"]["relative_path"] = "calls/relabel.bitpack"
    try:
        bind_raw_records(raw_stub, context)
    except RuntimeError:
        rejected.append("raw_relabel")
    require(rejected[-1] == "raw_relabel", "raw identity relabel survived")
    return {"status": "PASS_M1060_SMALL_SYNTHETIC_IDENTITY_BINDING_SELFTEST",
        "m1052_transactions": 85, "identity_attacks_rejected": rejected,
        "real_payload_members_opened": False, "real_pilot_executed": False,
        "eda_gpu_remote_used": False}


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("validate-pre-attempt-source", "validate-authority",
                 "consume-attempt", "validate-payload-after-attempt", "run-pilot",
                 "assemble", "publish", "quarantine", "self-test"):
        modes.add_argument("--" + name, action="store_true")
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--expected-contract-sha")
    parser.add_argument("--expected-review-sha")
    parser.add_argument("--expected-manifest-sha")
    parser.add_argument("--expected-outer-sha")
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--quarantine-path", type=Path)
    parser.add_argument("--return-code", type=int, default=1)
    args = parser.parse_args()
    pins = (args.expected_review_sha, args.expected_manifest_sha,
            args.expected_outer_sha)
    if args.validate_pre_attempt_source:
        require(args.runner and args.expected_contract_sha and
                sha256(args.contract) == args.expected_contract_sha,
                "pre-attempt caller identity absent")
        output = validate_pre_attempt_source(args.contract, args.runner)
    elif args.validate_authority:
        require(all(pins), "M1061 pins absent")
        output = validate_m1061(*pins)
    elif args.consume_attempt:
        require(args.attempt and args.runner and args.expected_contract_sha and
                all(pins), "attempt authority absent")
        output = consume_attempt(args.attempt, args.runner,
            args.expected_contract_sha, validate_m1061(*pins))
    elif args.validate_payload_after_attempt:
        require(args.attempt and args.work and args.runner and
                args.expected_contract_sha and all(pins), "payload authority absent")
        output = validate_payload_after_attempt(args.attempt, args.work, args.runner,
            args.expected_contract_sha, validate_m1061(*pins))
    elif args.run_pilot:
        require(args.attempt and args.work and args.runner and
                args.expected_contract_sha and all(pins), "run authority absent")
        output = run_pilot(args.attempt, args.work, args.runner,
            args.expected_contract_sha, validate_m1061(*pins))
    elif args.assemble:
        require(args.attempt and args.work and args.runner and
                args.expected_contract_sha and all(pins), "assemble authority absent")
        output = assemble(args.work, args.attempt, args.runner,
            args.expected_contract_sha, validate_m1061(*pins))
    elif args.publish:
        require(args.work and args.result and args.attempt and args.runner and
                args.expected_contract_sha and all(pins), "publish authority absent")
        output = publish(args.work, args.result, args.attempt, args.runner,
            args.expected_contract_sha, validate_m1061(*pins))
    elif args.quarantine:
        require(args.work and args.quarantine_path, "quarantine paths absent")
        output = quarantine(args.work, args.quarantine_path, args.return_code)
    else:
        output = self_test()
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
