#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1076 additive exact-bool/int repair over frozen M1060 identity binding.

Every schema comparison is type-strict recursively: bool and int never alias,
including at arbitrary nesting depth.  M1060 identity binding and the zero
pre-attempt payload-member-access boundary remain unchanged.
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
M1060_PATH = HERE / "execute_m1060_decoder_identity_binding_repair.py"
M1060_SHA256 = "440d6a12e19ac5561627ae9181d9b6f8ae1be23b1e988c139816a5261c760eb1"
CONTRACT = HW / "contracts/m1076_decoder_exact_bool_repair_contract_r1_20260830.json"
M1061_DIR = HW / "reviews/m1061_m1060_decoder_identity_binding_repair_hammer_r1_20260830"
M1077_DIR = HW / "reviews/m1077_m1076_decoder_exact_bool_repair_hammer_r1_20260830"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOC359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
SCHEMA = "m1076_decoder_exact_bool_repair_v1"
CONTEXT_SCHEMA = "m1078_decoder_canonical_payload_context_v1"
PAYLOAD_SCHEMA = "m1078_postattempt_payload_identity_receipt_v1"
RAW_SCHEMA = "m1078_decoder_stratified_block_reset_raw_cycles_v1"
RESULT_SCHEMA = "m1078_decoder_stratified_block_reset_result_v1"
ATTEMPT_SCHEMA = "m1078_decoder_pilot_attempt_v1"
ATTEMPT_NAME = ".m1078_m1076_decoder_exact_bool_pilot_attempt_consumed"
RESULT_NAME = "m1078_m1076_decoder_exact_bool_pilot_r1_20260830"
LAYERS = ("D0", "D2", "D3")


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
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_pinned(path, expected, name):
    require(Path(path).is_file() and not Path(path).is_symlink() and
            sha256(path) == expected, name + " identity drift")
    spec = importlib.util.spec_from_file_location("m1076_" + name, path)
    require(spec is not None and spec.loader is not None, "cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M1060 = load_pinned(M1060_PATH, M1060_SHA256, "frozen_m1060")
BASE, M1048, M785 = M1060.BASE, M1060.M1048, M1060.M785


def exact_tree(actual, expected, path=()):
    """Require equal JSON topology, value and exact Python type recursively."""
    label = ".".join(map(str, path)) or "$"
    require(type(actual) is type(expected), "exact leaf type drift at " + label)
    if type(expected) is dict:
        require(set(actual) == set(expected), "exact dict keys drift at " + label)
        for key in expected:
            exact_tree(actual[key], expected[key], path + (key,))
    elif type(expected) is list:
        require(len(actual) == len(expected), "exact list length drift at " + label)
        for index, (left, right) in enumerate(zip(actual, expected)):
            exact_tree(left, right, path + (index,))
    else:
        require(actual == expected, "exact leaf value drift at " + label)
    return True


def exact_int(value, minimum=0):
    require(type(value) is int and value >= minimum, "exact non-bool int required")
    return value


def contract_value(path=CONTRACT):
    value = strict_json(path)
    validate_contract(value)
    return value


def validate_contract(value):
    BASE.reject_forbidden_semantic_keys(value)
    require(type(value) is dict and set(value) == {"schema", "status", "launch_now",
        "objective", "workload", "d1", "pair", "sampling", "pre_attempt",
        "post_attempt", "output", "authority", "source_identity",
        "frozen_payload", "claim_boundary", "next_gate"}, "contract schema drift")
    exact_tree(value["schema"], SCHEMA, ("schema",))
    exact_tree(value["status"],
        "EXACT_BOOL_REPAIR_SOURCE_ONLY__M1077_HAMMER_REQUIRED", ("status",))
    exact_tree(value["launch_now"], False, ("launch_now",))
    exact_tree(value["workload"], {"population_id": M1048.POPULATION_ID,
        "sequence": M1048.SEQUENCE, "sample_id": 0, "timestep": 0,
        "config": M1048.CONFIG, "layers": list(LAYERS)}, ("workload",))
    exact_tree(value["d1"], {"status": "DIAGNOSTIC_ONLY",
        "generator_allowed": False, "scheduler_allowed": False,
        "numeric_equivalence_admitted": False}, ("d1",))
    exact_tree(value["pair"], {"role": "SELF_MATCHED_PROTOCOL_CALIBRATION",
        "candidate_body": "A1_OSG", "baseline_body": "A1_OSG",
        "performance_comparison": False}, ("pair",))
    exact_tree(value["sampling"], {"strata": list(BASE.STRATA),
        "source_census": 1, "pilot_per_noncensus_stratum": BASE.PILOT,
        "selection_seed": M1048.SELECTION_SEED,
        "window_expanded_request_cap": M1048.CAP,
        "selection_before_replay": True}, ("sampling",))
    exact_tree(value["pre_attempt"], {"payload_member_access": "FORBIDDEN",
        "allowed_checks": ["contract", "code", "review_seals",
                           "payload_root_seal_metadata"],
        "canonical_attempt_before_payload_validation": True}, ("pre_attempt",))
    exact_tree(value["post_attempt"], {"full_payload_member_verification": True,
        "canonical_context_written": True, "run_rederives_canonical_context": True,
        "assemble_rederives_canonical_context": True,
        "publish_rederives_canonical_context": True,
        "raw_record_selected_member_cross_binding": True,
        "recursive_exact_bool_int_validation": True,
        "failure_quarantine": True}, ("post_attempt",))
    exact_tree(value["output"], {"raw_cycle_samples": True,
        "cycle_ci_bounds": True, "derived_performance_values": False,
        "recursive_exact_schema": True, "bool_int_alias_allowed": False},
        ("output",))
    require(type(value["authority"]) is dict and
            set(value["authority"]) == {"m1061_negative", "m1077_required_status"},
            "authority schema drift")
    exact_tree(value["authority"]["m1061_negative"], {
        "directory": "reviews/m1061_m1060_decoder_identity_binding_repair_hammer_r1_20260830",
        "review_sha256": "40a4b530f9937d9044139b42b6dedc60ba9272a0179bef843adb1eedcc32650a",
        "manifest_sha256": "5014560a85f32dad8ce9de6385032fd09c761a7c6062ca3a57a29f7632f92a20",
        "outer_seal_file_sha256": "cdb8d9686f26a335a34b76c7055e6ae4a6ba960ba200968eb8f8852933a0551d"},
        ("authority", "m1061_negative"))
    exact_tree(value["authority"]["m1077_required_status"],
        "PASS_M1077_M1076_EXACT_BOOL_HAMMER__GO_ONE_M1078_ATTEMPT",
        ("authority", "m1077_required_status"))
    require(type(value["source_identity"]) is dict and
            set(value["source_identity"]) == {"driver", "runner", "m1060",
                "m785_contract", "docs359"}, "source identity schema drift")
    expected_paths = {"driver":
        "system_simulator/scripts/execute_m1076_decoder_exact_bool_repair.py",
        "runner": "system_simulator/scripts/run_m1078_m1076_decoder_exact_bool_pilot_one_shot.sh",
        "m1060": "system_simulator/scripts/execute_m1060_decoder_identity_binding_repair.py",
        "m785_contract": "contracts/m785_h67_decoder_physical_residency_repair_contract_r1_20260828.json",
        "docs359": "docs/359_DATE终局冻结_20260813.md"}
    for name, expected_path in expected_paths.items():
        entry = value["source_identity"][name]
        require(type(entry) is dict and set(entry) == {"path", "sha256"},
                name + " identity schema drift")
        exact_tree(entry["path"], expected_path, ("source_identity", name, "path"))
        BASE.hash_value(entry["sha256"])
    frozen = value["frozen_payload"]
    require(type(frozen) is dict and set(frozen) == {"directory",
        "m699_manifest_sha256", "m699_root_manifest_sha256",
        "m699_outer_seal_file_sha256", "selected_records"},
        "frozen payload schema drift")
    exact_tree(frozen["directory"],
        "system_handoff/outgoing/m699_h67_ep35_multisequence_decoder_payload_s3x10_r1_20260828",
        ("frozen_payload", "directory"))
    exact_tree(frozen["m699_manifest_sha256"],
        "e2d7c92a038c213b590603ff534a33f3579bf1224cc3f56c11629e1d4c813dc0",
        ("frozen_payload", "m699_manifest_sha256"))
    exact_tree(frozen["m699_root_manifest_sha256"],
        "27b35748b81d32907410ada0fbecfaa869a6ce1c3039e94ab3da2e52a8f46053",
        ("frozen_payload", "m699_root_manifest_sha256"))
    exact_tree(frozen["m699_outer_seal_file_sha256"],
        "eaf975a9a1a4829b2c0a2251e7ef297abd53b83b30e23630e5ce51db5c5de18c",
        ("frozen_payload", "m699_outer_seal_file_sha256"))
    exact_tree(frozen["selected_records"], list(M1060.FROZEN_SELECTED),
               ("frozen_payload", "selected_records"))
    exact_tree(value["claim_boundary"], {"paper_citable": False,
        "decoder_complete": False, "table_a_row": False,
        "system_performance_claim": False, "local_performance_claim": False,
        "continuous_row_cycles": False, "d1_scheduled": False,
        "eda_gpu_remote_used": False}, ("claim_boundary",))
    require(type(value["objective"]) is str and type(value["next_gate"]) is str,
            "contract narrative type drift")
    return True


def validate_pre_attempt_source(contract_path, runner):
    value = contract_value(contract_path)
    identities = value["source_identity"]
    for name, path in (("driver", Path(__file__).resolve()),
                       ("runner", Path(runner)), ("m1060", M1060_PATH),
                       ("m785_contract", HW / identities["m785_contract"]["path"]),
                       ("docs359", DOC359)):
        require(Path(path).is_file() and not Path(path).is_symlink() and
                sha256(path) == identities[name]["sha256"], name + " drift")
    negative = value["authority"]["m1061_negative"]
    sealed = BASE.verify_flat_seal(HW / negative["directory"])
    require(sha256(HW / negative["directory"] / "review.json") ==
            negative["review_sha256"] and
            sealed["manifest_sha256"] == negative["manifest_sha256"] and
            sealed["outer_seal_file_sha256"] == negative["outer_seal_file_sha256"],
            "M1061 negative seal drift")
    report = strict_json(HW / negative["directory"] / "review.json")
    require(report["status"] ==
            "FAIL_M1061_M1060_BOOL_INTEGER_EXACT_SCHEMA_ESCAPE__STOP_M1062",
            "M1061 negative status drift")
    frozen = value["frozen_payload"]
    meta = BASE.verify_payload_root_seal_metadata_only(
        HW / frozen["directory"], frozen["m699_root_manifest_sha256"],
        frozen["m699_outer_seal_file_sha256"], frozen["m699_manifest_sha256"])
    return {"status":
        "PASS_M1076_PREATTEMPT_SOURCE_WITH_ZERO_PAYLOAD_MEMBER_ACCESS",
        "contract_sha256": sha256(contract_path),
        "driver_sha256": sha256(Path(__file__).resolve()),
        "runner_sha256": sha256(runner), "payload_root_seal": meta,
        "payload_members_opened": False, "payload_members_statted": False,
        "payload_members_hashed": False, "launch_now": False}


def validate_m1077(review_sha, manifest_sha, outer_sha):
    sealed = BASE.verify_flat_seal(M1077_DIR)
    require(sha256(M1077_DIR / "review.json") == review_sha and
            sealed["manifest_sha256"] == manifest_sha and
            sealed["outer_seal_file_sha256"] == outer_sha,
            "M1077 caller-pinned authority drift")
    value = strict_json(M1077_DIR / "review.json")
    require(value["status"] ==
            "PASS_M1077_M1076_EXACT_BOOL_HAMMER__GO_ONE_M1078_ATTEMPT" and
            value["authorization"]["one_m1078_attempt"] is True and
            value["authorization"]["real_payload_after_attempt_only"] is True and
            value["authorization"]["eda_gpu_remote"] is False,
            "M1077 authorization drift")
    return {"review_sha256": review_sha, "manifest_sha256": manifest_sha,
            "outer_seal_file_sha256": outer_sha}


def safe_path(path, role):
    path = Path(path)
    require(path.is_absolute() and path.parent.resolve() == RESULTS.resolve() and
            not path.is_symlink(), role + " runtime path drift")
    expected = {"attempt": ATTEMPT_NAME, "result": RESULT_NAME}
    if role in expected:
        require(path.name == expected[role], role + " namespace drift")
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
    require(not path.exists(), "M1078 attempt already consumed")
    os.mkdir(path, 0o700)
    receipt = {"schema": ATTEMPT_SCHEMA,
        "status": "M1078_ATTEMPT_CONSUMED_BEFORE_PAYLOAD_MEMBER_ACCESS",
        "consumed_unix_ns": time.time_ns(), "runner_sha256": sha256(runner),
        "contract_sha256": contract_sha, "m1077_authority": dict(authority),
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
    contract = contract_value()
    expected = {"schema": ATTEMPT_SCHEMA,
        "status": "M1078_ATTEMPT_CONSUMED_BEFORE_PAYLOAD_MEMBER_ACCESS",
        "consumed_unix_ns": value.get("consumed_unix_ns"),
        "runner_sha256": contract["source_identity"]["runner"]["sha256"],
        "contract_sha256": sha256(CONTRACT), "m1077_authority": dict(authority),
        "payload_members_opened": False, "payload_members_statted": False,
        "payload_members_hashed": False, "paper_citable": False}
    exact_int(value.get("consumed_unix_ns"), 1)
    exact_tree(value, expected, ("attempt",))
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
            "canonical attempt/runner/contract drift")
    m785_path = HW / contract["source_identity"]["m785_contract"]["path"]
    M785.validate_source_contract(REPO, m785_path)
    frozen = contract["frozen_payload"]
    payload_dir = HW / frozen["directory"]
    sealed = M785.verify_sealed_directory(payload_dir)
    require(sha256(payload_dir / "manifest.json") == frozen["m699_manifest_sha256"] and
            sealed["manifest_sha256"] == frozen["m699_root_manifest_sha256"] and
            sealed["outer_seal_file_sha256"] == frozen["m699_outer_seal_file_sha256"],
            "frozen M699 root identity drift")
    manifest = M785.strict_json(payload_dir / "manifest.json")
    records = M785.normalized_population_records(manifest, M1048.POPULATION_ID)
    selected = []
    for layer, frozen_record in zip(LAYERS, frozen["selected_records"]):
        actual = normalized_selected(M1048.select_record(records, layer), layer)
        exact_tree(actual, frozen_record, ("selected", layer))
        selected.append(verified_member(payload_dir, frozen_record))
    context = {"schema": CONTEXT_SCHEMA,
        "attempt": {"attempt_json_sha256": sha256(Path(attempt) / "attempt.json"),
            "runner_sha256": attempt_value["runner_sha256"],
            "contract_sha256": attempt_value["contract_sha256"],
            "m1077_authority": copy.deepcopy(attempt_value["m1077_authority"])},
        "payload": {"directory": frozen["directory"],
            "m699_manifest_sha256": frozen["m699_manifest_sha256"],
            "m699_root_manifest_sha256": frozen["m699_root_manifest_sha256"],
            "m699_outer_seal_file_sha256": frozen["m699_outer_seal_file_sha256"],
            "selected_records": selected},
        "identity_binding":
            "FROZEN_CONTRACT_PLUS_CANONICAL_ATTEMPT_PLUS_VERIFIED_MEMBERS",
        "d1_scheduled": False, "paper_citable": False}
    validate_canonical_context(context, context)
    return context


def validate_canonical_context(value, expected):
    BASE.reject_forbidden_semantic_keys(value)
    exact_tree(value, expected, ("canonical_context",))
    return True


def make_payload_receipt(context):
    return {"schema": PAYLOAD_SCHEMA,
        "status": "PASS_M1078_POSTATTEMPT_CANONICAL_PAYLOAD_IDENTITY",
        "canonical_context_sha256": canonical_sha(context),
        "attempt": copy.deepcopy(context["attempt"]),
        "payload": copy.deepcopy(context["payload"]),
        "payload_members_verified": True, "post_attempt": True,
        "d1_scheduled": False, "paper_citable": False}


def validate_payload_receipt(value, context):
    BASE.reject_forbidden_semantic_keys(value)
    exact_tree(value, make_payload_receipt(context), ("payload_receipt",))
    return True


def validate_payload_after_attempt(attempt, work, runner, contract_sha, authority):
    work = safe_path(work, "work")
    require(work.is_dir() and not any(work.iterdir()), "postattempt work not fresh")
    context = build_canonical_context(attempt, runner, contract_sha, authority)
    atomic_json(work / "canonical_context.json", context)
    atomic_json(work / "payload_validation.json", make_payload_receipt(context))
    return make_payload_receipt(context)


def expected_raw_record(selected):
    return {"population_id": selected["population_id"],
        "sequence": selected["sequence"], "sample_id": selected["sample_id"],
        "module_index": selected["module_index"], "timestep": 0,
        "config": M1048.CONFIG, "route": selected["route"],
        "relative_path": selected["relative_path"],
        "packed_sha256": selected["packed_sha256"]}


def bind_raw_records(raw, context):
    require(type(raw) is dict and type(raw.get("layers")) is list and
            len(raw["layers"]) == len(LAYERS), "raw layer shape drift")
    for layer, row, selected in zip(LAYERS, raw["layers"],
                                    context["payload"]["selected_records"]):
        require(type(row) is dict, "raw row type drift")
        exact_tree(row.get("layer"), layer, ("raw", layer, "layer"))
        exact_tree(row.get("record_identity"), expected_raw_record(selected),
                   ("raw", layer, "record_identity"))
        exact_tree(row.get("verified_payload_member_sha256"),
                   selected["payload_member_sha256"],
                   ("raw", layer, "verified_payload_member_sha256"))
        exact_tree(selected["payload_member_sha256"], selected["packed_sha256"],
                   ("selected", layer, "member_sha"))
    return True


def validate_layer(value, layer, selected):
    require(type(value) is dict and "verified_payload_member_sha256" in value,
            "verified member absent")
    projected = copy.deepcopy(value)
    member_sha = projected.pop("verified_payload_member_sha256")
    BASE.validate_layer(projected, layer)
    exact_tree(projected["record_identity"], expected_raw_record(selected),
               ("layer", layer, "record_identity"))
    exact_tree(member_sha, selected["payload_member_sha256"],
               ("layer", layer, "member_sha"))
    return True


def validate_raw(value, context):
    BASE.reject_forbidden_semantic_keys(value)
    require(type(value) is dict and set(value) == {"schema", "status", "workload",
        "pair_role", "canonical_context_sha256", "layers",
        "exact_mismatch_count", "d1", "claim_boundary"}, "raw schema drift")
    exact_tree(value["schema"], RAW_SCHEMA, ("raw", "schema"))
    exact_tree(value["status"], "PASS_M1078_RAW_CYCLES__RESULT_HAMMER_REQUIRED",
               ("raw", "status"))
    exact_tree(value["workload"], {"population_id": M1048.POPULATION_ID,
        "sequence": M1048.SEQUENCE, "sample_id": 0, "timestep": 0,
        "config": M1048.CONFIG, "layers": list(LAYERS)}, ("raw", "workload"))
    exact_tree(value["pair_role"], "SELF_MATCHED_PROTOCOL_CALIBRATION",
               ("raw", "pair_role"))
    exact_tree(value["canonical_context_sha256"], canonical_sha(context),
               ("raw", "canonical_context_sha256"))
    bind_raw_records(value, context)
    for layer, row, selected in zip(LAYERS, value["layers"],
                                    context["payload"]["selected_records"]):
        validate_layer(row, layer, selected)
    exact_tree(value["exact_mismatch_count"], 0, ("raw", "exact_mismatch_count"))
    exact_tree(value["d1"], {"status": "DIAGNOSTIC_ONLY", "scheduled": False,
        "numeric_equivalence_admitted": False}, ("raw", "d1"))
    exact_tree(value["claim_boundary"], {"paper_citable": False,
        "decoder_complete": False, "table_a_row": False,
        "system_performance_claim": False, "local_performance_claim": False,
        "continuous_row_cycles": False}, ("raw", "claim_boundary"))
    return True


def make_result(raw, raw_sha, payload_sha, context_sha):
    return {"schema": RESULT_SCHEMA,
        "status": "PASS_M1078_DIAGNOSTIC_RAW_CYCLE_PILOT__RESULT_HAMMER_REQUIRED",
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
        "next_gate": "Independent receipt-blind M1079 result hammer"}


def validate_result(value, raw, raw_sha, payload_sha, context_sha):
    BASE.reject_forbidden_semantic_keys(value)
    exact_tree(value, make_result(raw, raw_sha, payload_sha, context_sha),
               ("result",))
    return True


def load_work_context(work, expected):
    stored = strict_json(work / "canonical_context.json")
    validate_canonical_context(stored, expected)
    payload = strict_json(work / "payload_validation.json")
    validate_payload_receipt(payload, expected)
    return stored, payload


def run_pilot(attempt, work, runner, contract_sha, authority):
    work = safe_path(work, "work")
    require(work.is_dir() and not work.is_symlink() and
            {item.name for item in work.iterdir()} ==
            {"canonical_context.json", "payload_validation.json"},
            "canonical receipts required")
    context = build_canonical_context(attempt, runner, contract_sha, authority)
    load_work_context(work, context)
    payload_root, records, mapper, oracles = M1048._context()
    old_layers = []
    for layer, selected in zip(LAYERS, context["payload"]["selected_records"]):
        record = M1048.select_record(records, layer)
        exact_tree(normalized_selected(record, layer),
            {key: selected[key] for key in ("layer", "population_id", "sequence",
                "sample_id", "module_index", "route", "relative_path", "packed_sha256")},
            ("run_selected", layer))
        old_layers.append(M1048.replay_layer(layer, record, payload_root,
                                             mapper, oracles))
    layers = []
    for old, selected in zip(old_layers, context["payload"]["selected_records"]):
        row = M1060.BASE.transform_layer(old)
        row["verified_payload_member_sha256"] = selected["payload_member_sha256"]
        layers.append(row)
    raw = {"schema": RAW_SCHEMA,
        "status": "PASS_M1078_RAW_CYCLES__RESULT_HAMMER_REQUIRED",
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


def validate_completed_work(work, context, sealed):
    core = {"canonical_context.json", "payload_validation.json",
            "raw_windows.json", "result.json", "RUN_COMPLETE.txt"}
    expected = core | ({"SHA256SUMS", "SHA256SUMS.seal.sha256"} if sealed else set())
    require(work.is_dir() and not work.is_symlink() and
            {item.name for item in work.iterdir()} == expected and
            all(item.is_file() and not item.is_symlink() for item in work.iterdir()),
            "completed work exact-set drift")
    load_work_context(work, context)
    raw = strict_json(work / "raw_windows.json")
    result = strict_json(work / "result.json")
    validate_raw(raw, context)
    validate_result(result, raw, sha256(work / "raw_windows.json"),
                    sha256(work / "payload_validation.json"), canonical_sha(context))
    exact_tree((work / "RUN_COMPLETE.txt").read_text(encoding="utf-8"),
               result["status"] + "\n", ("completion",))
    return core


def assemble(work, attempt, runner, contract_sha, authority):
    work = safe_path(work, "work")
    context = build_canonical_context(attempt, runner, contract_sha, authority)
    core = validate_completed_work(work, context, False)
    lines = [sha256(work / name) + "  " + name for name in sorted(core)]
    (work / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_sha = sha256(work / "SHA256SUMS")
    (work / "SHA256SUMS.seal.sha256").write_text(
        manifest_sha + "  SHA256SUMS\n", encoding="utf-8")
    return {"status": "PASS_M1078_EXACT_BOOL_IDENTITY_BOUND_OUTPUT_SEALED",
            "manifest_sha256": manifest_sha,
            "outer_seal_file_sha256": sha256(work / "SHA256SUMS.seal.sha256")}


def publish(work, result, attempt, runner, contract_sha, authority):
    work, result = safe_path(work, "work"), safe_path(result, "result")
    require(work.is_dir() and not result.exists(), "publish namespace drift")
    BASE.verify_flat_seal(work)
    context = build_canonical_context(attempt, runner, contract_sha, authority)
    validate_completed_work(work, context, True)
    os.replace(work, result)
    return {"status": "PASS_M1078_ATOMIC_RESULT_PUBLISHED"}


def quarantine(work, quarantine, return_code):
    work, quarantine = safe_path(work, "work"), safe_path(quarantine, "quarantine")
    require(work.is_dir() and not quarantine.exists(), "quarantine namespace drift")
    os.replace(work, quarantine)
    atomic_json(quarantine / "FAILURE.json", {"schema": "m1078_failure_v1",
        "status": "FAILED_OR_INCOMPLETE_DO_NOT_CITE",
        "return_code": exact_int(return_code, 0), "paper_citable": False})
    return {"status": "PASS_M1078_FAILURE_QUARANTINED"}


def synthetic_context():
    old = M1060.synthetic_context()
    old["schema"] = CONTEXT_SCHEMA
    old["attempt"]["m1077_authority"] = old["attempt"].pop("m1061_authority")
    return old


def self_test():
    base = M1060.self_test()
    require(base["m1052_transactions"] == 85, "M1060 synthetic drift")
    expected = {"outer": [{"flag": False, "count": 0},
                           {"flag": True, "count": 1}]}
    attacks = []
    for name, path in (("false_to_zero", ("outer", 0, "flag")),
                       ("zero_to_false", ("outer", 0, "count")),
                       ("true_to_one", ("outer", 1, "flag")),
                       ("one_to_true", ("outer", 1, "count"))):
        value = copy.deepcopy(expected)
        parent = value[path[0]][path[1]]
        parent[path[2]] = {"false_to_zero": 0, "zero_to_false": False,
                           "true_to_one": 1, "one_to_true": True}[name]
        try:
            exact_tree(value, expected)
        except RuntimeError:
            attacks.append(name)
    context = synthetic_context(); receipt = make_payload_receipt(context)
    for key, replacement in (("payload_members_verified", 1),
                             ("post_attempt", 1), ("paper_citable", 0)):
        attacked = copy.deepcopy(receipt); attacked[key] = replacement
        try:
            validate_payload_receipt(attacked, context)
        except RuntimeError:
            attacks.append("payload_" + key)
    raw = {"layers": [{"layer": row["layer"],
        "record_identity": expected_raw_record(row),
        "verified_payload_member_sha256": row["payload_member_sha256"]}
        for row in context["payload"]["selected_records"]]}
    for field in ("sample_id", "timestep"):
        attacked = copy.deepcopy(raw)
        attacked["layers"][0]["record_identity"][field] = False
        try:
            bind_raw_records(attacked, context)
        except RuntimeError:
            attacks.append("raw_" + field)
    require(len(attacks) == 9, "bool/int attack survived")
    return {"status": "PASS_M1076_SMALL_SYNTHETIC_EXACT_BOOL_INT_SELFTEST",
        "bool_int_attacks_rejected": attacks, "m1060_identity_repair_preserved": True,
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
        require(all(pins), "M1077 pins absent")
        output = validate_m1077(*pins)
    elif args.consume_attempt:
        require(args.attempt and args.runner and args.expected_contract_sha and
                all(pins), "attempt authority absent")
        output = consume_attempt(args.attempt, args.runner,
            args.expected_contract_sha, validate_m1077(*pins))
    elif args.validate_payload_after_attempt:
        require(args.attempt and args.work and args.runner and
                args.expected_contract_sha and all(pins), "payload authority absent")
        output = validate_payload_after_attempt(args.attempt, args.work, args.runner,
            args.expected_contract_sha, validate_m1077(*pins))
    elif args.run_pilot:
        require(args.attempt and args.work and args.runner and
                args.expected_contract_sha and all(pins), "run authority absent")
        output = run_pilot(args.attempt, args.work, args.runner,
            args.expected_contract_sha, validate_m1077(*pins))
    elif args.assemble:
        require(args.attempt and args.work and args.runner and
                args.expected_contract_sha and all(pins), "assemble authority absent")
        output = assemble(args.work, args.attempt, args.runner,
            args.expected_contract_sha, validate_m1077(*pins))
    elif args.publish:
        require(args.work and args.result and args.attempt and args.runner and
                args.expected_contract_sha and all(pins), "publish authority absent")
        output = publish(args.work, args.result, args.attempt, args.runner,
            args.expected_contract_sha, validate_m1077(*pins))
    elif args.quarantine:
        require(args.work and args.quarantine_path, "quarantine paths absent")
        output = quarantine(args.work, args.quarantine_path, args.return_code)
    else:
        output = self_test()
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
