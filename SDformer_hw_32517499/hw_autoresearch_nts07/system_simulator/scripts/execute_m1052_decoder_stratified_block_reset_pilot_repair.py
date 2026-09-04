#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Additive M1052 repair of the stopped M1048/M1050 pilot release.

No M699 payload member is opened or hashed by pre-attempt validation.  The
canonical M1054 attempt must exist before full payload validation.  Every JSON
that can be sealed is recursively strong-typed and contains raw cycle samples
and cycle CI bounds only; point, mean, normalized and speedup fields are
forbidden at every depth.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Dict, Mapping, Sequence


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
REPO = HW.parent
RESULTS = HW / "results"
CONTRACT = HW / "contracts/m1052_decoder_stratified_block_reset_pilot_repair_contract_r1_20260829.json"
M1048_PATH = HERE / "execute_m1048_decoder_stratified_block_reset_pilot_release.py"
M1048_SHA256 = "3e2fa596e7cb0406feecc4124280643eaa093df80e9dcc7915fa9dcc7074267a"
M1049_DIR = HW / "reviews/m1049_m1048_decoder_stratified_block_reset_pilot_release_hammer_r1_20260829"
M1042_DIR = HW / "reviews/m1042_m1041_decoder_stratified_block_reset_windows_source_r4_hammer_r1_20260829"
M1053_DIR = HW / "reviews/m1053_m1052_decoder_stratified_block_reset_pilot_repair_hammer_r1_20260829"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOC359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
SCHEMA = "m1052_decoder_stratified_block_reset_pilot_repair_v1"
RAW_SCHEMA = "m1054_decoder_stratified_block_reset_raw_cycles_v1"
RESULT_SCHEMA = "m1054_decoder_stratified_block_reset_result_v1"
PAYLOAD_SCHEMA = "m1054_postattempt_payload_validation_v1"
ENVELOPE_SCHEMA = "m1054_cycles_only_ci_envelope_v1"
RESULT_NAME = "m1054_m1052_decoder_stratified_block_reset_pilot_r1_20260829"
ATTEMPT_NAME = ".m1054_m1052_decoder_stratified_block_reset_pilot_attempt_consumed"
LAYERS = ("D0", "D2", "D3")
STRATA = ("SOURCE_INIT_CENSUS", "COMPUTE_REGULAR",
          "DEPENDENCY_STRESS", "COMMIT_TAIL")
NONCENSUS = STRATA[1:]
PILOT = 8
HASH = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_SEMANTIC_TOKENS = {
    "mean", "average", "point", "speedup", "normalized", "normalization",
    "estimate", "throughput", "fps",
}


def require(value, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")).hexdigest()


def strict_json(path: Path):
    def pairs(rows):
        output = {}
        for key, value in rows:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("non-finite JSON token: " + token)))


def atomic_json(path: Path, value: object) -> None:
    temporary = Path(path).with_name(Path(path).name + ".tmp." + str(os.getpid()))
    payload = json.dumps(value, indent=2, sort_keys=True,
                         ensure_ascii=False, allow_nan=False) + "\n"
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_pinned(path: Path, expected: str, name: str):
    require(path.is_file() and not path.is_symlink() and
            sha256(path) == expected, name + " identity drift")
    spec = importlib.util.spec_from_file_location("m1052_" + name, path)
    require(spec is not None and spec.loader is not None,
            "cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M1048 = load_pinned(M1048_PATH, M1048_SHA256, "frozen_m1048")
M1041, M785, M890 = M1048.M1041, M1048.M785, M1048.M890


def exact_dict(value, keys, name: str):
    require(type(value) is dict and set(value) == set(keys), name + " schema drift")
    return value


def exact_list(value, length: int, name: str):
    require(type(value) is list and len(value) == length, name + " shape drift")
    return value


def exact_bool(value, expected=None):
    require(type(value) is bool, "exact bool required")
    if expected is not None:
        require(value is expected, "boolean boundary drift")
    return value


def exact_int(value, minimum=0):
    require(type(value) is int and value >= minimum, "exact integer/range drift")
    return value


def finite(value, minimum=None):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(value), "finite scalar required")
    if minimum is not None:
        require(value >= minimum, "finite scalar range drift")
    return value


def hash_value(value):
    require(type(value) is str and HASH.fullmatch(value), "SHA256 value drift")
    return value


def semantic_tokens(key: str):
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    tokens = set()
    for raw in re.findall(r"[A-Za-z]+", expanded):
        token = raw.lower()
        tokens.add(token)
        if token.endswith("ies") and len(token) > 3:
            tokens.add(token[:-3] + "y")
        if token.endswith("es") and len(token) > 2:
            tokens.add(token[:-2])
        if token.endswith("s") and len(token) > 1:
            tokens.add(token[:-1])
    return tokens


def reject_forbidden_semantic_keys(value, path=()):
    if type(value) is dict:
        for key, item in value.items():
            require(type(key) is str and key, "public JSON key type drift")
            child = path + (key,)
            require(not (semantic_tokens(key) & FORBIDDEN_SEMANTIC_TOKENS),
                    "forbidden derived semantic key at depth: " + ".".join(child))
            reject_forbidden_semantic_keys(item, child)
    elif type(value) is list:
        for index, item in enumerate(value):
            reject_forbidden_semantic_keys(item, path + (str(index),))
    elif isinstance(value, bool) or value is None or isinstance(value, str):
        return
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        require(math.isfinite(value), "non-finite public value at " + ".".join(path))
    else:
        raise RuntimeError("non-JSON public value at " + ".".join(path))


def verify_flat_seal(directory: Path) -> Dict[str, str]:
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(), "sealed directory absent")
    manifest, outer = directory / "SHA256SUMS", directory / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and not manifest.is_symlink() and
            outer.is_file() and not outer.is_symlink(), "seal absent")
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and len(fields[0]) == 64, "manifest malformed")
        expected, name = fields
        target = directory / name
        require(name not in listed and target.is_file() and not target.is_symlink() and
                sha256(target) == expected, "sealed member drift: " + name)
        listed.add(name)
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and not path.is_symlink() and
              path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")}
    require(listed == actual, "sealed exact-set drift")
    manifest_sha = sha256(manifest)
    require(outer.read_text(encoding="utf-8") ==
            manifest_sha + "  SHA256SUMS\n", "outer seal drift")
    return {"manifest_sha256": manifest_sha,
            "outer_seal_file_sha256": sha256(outer)}


def verify_payload_root_seal_metadata_only(directory: Path,
                                           expected_manifest_file_sha: str,
                                           expected_outer_file_sha: str,
                                           expected_payload_manifest_sha: str):
    """Read only root seal metadata; never stat/open any listed member."""
    root_manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(sha256(root_manifest) == expected_manifest_file_sha and
            sha256(outer) == expected_outer_file_sha, "root seal identity drift")
    require(outer.read_text(encoding="utf-8") ==
            expected_manifest_file_sha + "  SHA256SUMS\n", "root outer drift")
    listed_manifest = None
    calls = 0
    for line in root_manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and len(fields[0]) == 64, "root seal malformed")
        if fields[1] == "manifest.json":
            listed_manifest = fields[0]
        if fields[1].startswith("calls/"):
            calls += 1
    require(listed_manifest == expected_payload_manifest_sha and calls == 120,
            "root seal payload population drift")
    return {"root_manifest_sha256": expected_manifest_file_sha,
            "outer_seal_file_sha256": expected_outer_file_sha,
            "listed_payload_manifest_sha256": listed_manifest,
            "listed_call_members": calls,
            "payload_members_opened": False,
            "payload_members_statted": False,
            "payload_members_hashed": False}


def validate_contract(value):
    reject_forbidden_semantic_keys(value)
    exact_dict(value, {"schema", "status", "launch_now", "objective",
                       "workload", "d1", "pair", "sampling", "pre_attempt",
                       "post_attempt", "output", "authority", "source_identity",
                       "claim_boundary", "next_gate"}, "contract")
    require(value["schema"] == SCHEMA and
            value["status"] == "REPAIR_SOURCE_ONLY__M1053_HAMMER_REQUIRED" and
            exact_bool(value["launch_now"], False) is False, "contract header drift")
    exact_dict(value["workload"], {"population_id", "sequence", "sample_id",
               "timestep", "config", "layers"}, "workload")
    require(value["workload"] == {
        "population_id": M1048.POPULATION_ID, "sequence": M1048.SEQUENCE,
        "sample_id": 0, "timestep": 0, "config": M1048.CONFIG,
        "layers": list(LAYERS)}, "workload identity drift")
    exact_dict(value["d1"], {"status", "generator_allowed", "scheduler_allowed",
               "numeric_equivalence_admitted"}, "d1")
    require(value["d1"] == {"status": "DIAGNOSTIC_ONLY",
            "generator_allowed": False, "scheduler_allowed": False,
            "numeric_equivalence_admitted": False}, "D1 boundary drift")
    exact_dict(value["pair"], {"role", "candidate_body", "baseline_body",
               "performance_comparison"}, "pair")
    require(value["pair"] == {"role": "SELF_MATCHED_PROTOCOL_CALIBRATION",
            "candidate_body": "A1_OSG", "baseline_body": "A1_OSG",
            "performance_comparison": False}, "pair boundary drift")
    exact_dict(value["sampling"], {"strata", "source_census",
               "pilot_per_noncensus_stratum", "selection_seed",
               "window_expanded_request_cap", "selection_before_replay"}, "sampling")
    require(value["sampling"] == {"strata": list(STRATA), "source_census": 1,
            "pilot_per_noncensus_stratum": PILOT,
            "selection_seed": M1048.SELECTION_SEED,
            "window_expanded_request_cap": M1048.CAP,
            "selection_before_replay": True}, "sampling drift")
    exact_dict(value["pre_attempt"], {"payload_member_access", "allowed_checks",
               "canonical_attempt_before_payload_validation"}, "pre_attempt")
    require(value["pre_attempt"]["payload_member_access"] == "FORBIDDEN" and
            value["pre_attempt"]["allowed_checks"] ==
            ["contract", "code", "review_seals", "payload_root_seal_metadata"] and
            value["pre_attempt"]["canonical_attempt_before_payload_validation"] is True,
            "pre-attempt boundary drift")
    exact_dict(value["post_attempt"], {"full_payload_member_verification",
               "failure_quarantine", "run_requires_payload_validation_receipt"},
               "post_attempt")
    require(all(value["post_attempt"].values()), "post-attempt gate drift")
    exact_dict(value["output"], {"raw_cycle_samples", "cycle_ci_bounds",
               "exact_miter_fields", "coverage", "forbidden_derived_fields",
               "recursive_exact_schema"}, "output")
    require(value["output"]["forbidden_derived_fields"] ==
            ["candidate_mean", "baseline_mean", "point", "speedup",
             "normalized", "estimate", "throughput", "fps"] and
            all(value["output"][key] is True for key in
                ("raw_cycle_samples", "cycle_ci_bounds", "exact_miter_fields",
                 "coverage", "recursive_exact_schema")), "output boundary drift")
    exact_dict(value["authority"], {"m1049_negative", "m1042_positive",
               "m1053_required_status"}, "authority")
    for name in ("m1049_negative", "m1042_positive"):
        exact_dict(value["authority"][name], {"directory", "review_sha256",
                   "manifest_sha256", "outer_seal_file_sha256"}, name)
        for key in ("review_sha256", "manifest_sha256", "outer_seal_file_sha256"):
            hash_value(value["authority"][name][key])
    require(value["authority"]["m1053_required_status"] ==
            "PASS_M1053_M1052_REPAIR_HAMMER__GO_ONE_M1054_ATTEMPT",
            "M1053 required status drift")
    exact_dict(value["source_identity"], {"driver", "runner", "m1048",
               "m785_contract", "m699_root_seal", "m705_review", "docs359"},
               "source_identity")
    for name in ("driver", "runner", "m1048", "m785_contract",
                 "docs359"):
        exact_dict(value["source_identity"][name], {"path", "sha256"}, name)
        hash_value(value["source_identity"][name]["sha256"])
    exact_dict(value["source_identity"]["m705_review"], {"path",
               "review_sha256", "manifest_sha256", "outer_seal_file_sha256"},
               "m705_review")
    for key in ("review_sha256", "manifest_sha256", "outer_seal_file_sha256"):
        hash_value(value["source_identity"]["m705_review"][key])
    exact_dict(value["source_identity"]["m699_root_seal"], {"directory",
               "root_manifest_sha256", "outer_seal_file_sha256",
               "payload_manifest_sha256"}, "m699 root seal")
    for key in ("root_manifest_sha256", "outer_seal_file_sha256",
                "payload_manifest_sha256"):
        hash_value(value["source_identity"]["m699_root_seal"][key])
    exact_dict(value["claim_boundary"], {"paper_citable", "decoder_complete",
               "table_a_row", "system_performance_claim",
               "local_performance_claim", "continuous_row_cycles",
               "d1_scheduled", "eda_gpu_remote_used"}, "claim boundary")
    require(all(exact_bool(item, False) is False
                for item in value["claim_boundary"].values()),
            "claim boundary expanded")
    return True


def contract_value(path=CONTRACT):
    value = strict_json(path)
    validate_contract(value)
    return value


def validate_pre_attempt_source(contract_path: Path, runner: Path):
    value = contract_value(contract_path)
    identities = value["source_identity"]
    for name, path in (("driver", Path(__file__).resolve()),
                       ("runner", Path(runner)), ("m1048", M1048_PATH),
                       ("m785_contract", HW / identities["m785_contract"]["path"]),
                       ("docs359", DOC359)):
        require(path.is_file() and not path.is_symlink() and
                sha256(path) == identities[name]["sha256"], name + " drift")
    for name in ("m1049_negative", "m1042_positive"):
        entry = value["authority"][name]
        directory = HW / entry["directory"]
        sealed = verify_flat_seal(directory)
        require(sha256(directory / "review.json") == entry["review_sha256"] and
                sealed["manifest_sha256"] == entry["manifest_sha256"] and
                sealed["outer_seal_file_sha256"] ==
                entry["outer_seal_file_sha256"], name + " authority drift")
    m1049 = strict_json(HW / value["authority"]["m1049_negative"]["directory"] /
                        "review.json")
    require(m1049["status"] ==
            "FAIL_M1049_M1048_PREATTEMPT_PAYLOAD_OPEN_AND_ASSEMBLE_SEMANTIC_INJECTION__STOP_M1050" and
            m1049["authorization"]["author_additive_m1052_repair_source"] is True and
            m1049["authorization"]["execute_real_windows"] is False,
            "M1049 negative authority drift")
    m705_entry = identities["m705_review"]
    m705_dir = HW / m705_entry["path"]
    m705_seal = verify_flat_seal(m705_dir)
    require(sha256(m705_dir / "review.json") == m705_entry["review_sha256"] and
            m705_seal["manifest_sha256"] == m705_entry["manifest_sha256"] and
            m705_seal["outer_seal_file_sha256"] ==
            m705_entry["outer_seal_file_sha256"],
            "M705 review drift")
    root = identities["m699_root_seal"]
    root_meta = verify_payload_root_seal_metadata_only(
        HW / root["directory"], root["root_manifest_sha256"],
        root["outer_seal_file_sha256"], root["payload_manifest_sha256"])
    return {"status": "PASS_M1052_PREATTEMPT_SOURCE_WITH_ZERO_PAYLOAD_MEMBER_ACCESS",
            "contract_sha256": sha256(contract_path),
            "driver_sha256": sha256(Path(__file__).resolve()),
            "runner_sha256": sha256(Path(runner)), "payload_root_seal": root_meta,
            "payload_members_opened": False, "payload_members_statted": False,
            "payload_members_hashed": False, "launch_now": False}


def validate_m1053(review_sha, manifest_sha, outer_sha):
    sealed = verify_flat_seal(M1053_DIR)
    require(sha256(M1053_DIR / "review.json") == review_sha and
            sealed["manifest_sha256"] == manifest_sha and
            sealed["outer_seal_file_sha256"] == outer_sha,
            "M1053 caller-pinned authority drift")
    value = strict_json(M1053_DIR / "review.json")
    require(value["status"] ==
            "PASS_M1053_M1052_REPAIR_HAMMER__GO_ONE_M1054_ATTEMPT" and
            value["authorization"]["one_m1054_attempt"] is True and
            value["authorization"]["real_payload_after_attempt_only"] is True and
            value["authorization"]["eda_gpu_remote"] is False,
            "M1053 authorization drift")
    return {"review_sha256": review_sha, "manifest_sha256": manifest_sha,
            "outer_seal_file_sha256": outer_sha}


def safe_path(path: Path, role: str):
    path = Path(path)
    require(path.is_absolute() and path.parent.resolve() == RESULTS.resolve() and
            not path.is_symlink(), role + " runtime path drift")
    if role == "attempt": require(path.name == ATTEMPT_NAME, "attempt namespace drift")
    elif role == "result": require(path.name == RESULT_NAME, "result namespace drift")
    elif role == "work": require(path.name.startswith("." + RESULT_NAME + ".work."), "work namespace drift")
    elif role == "quarantine": require(path.name.startswith(RESULT_NAME + ".failed_or_incomplete."), "quarantine namespace drift")
    else: raise RuntimeError("unknown runtime role")
    return path


def consume_attempt(path, runner, contract_sha, authority):
    path = safe_path(path, "attempt")
    contract = contract_value()
    require(contract_sha == sha256(CONTRACT) and
            Path(runner).is_file() and not Path(runner).is_symlink() and
            sha256(Path(runner)) == contract["source_identity"]["runner"]["sha256"],
            "attempt source/contract identity drift")
    require(not path.exists(), "M1054 attempt already consumed")
    os.mkdir(path, 0o700)
    receipt = {"schema": "m1054_attempt_v1",
        "status": "M1054_ATTEMPT_CONSUMED_BEFORE_PAYLOAD_MEMBER_ACCESS",
        "consumed_unix_ns": time.time_ns(), "runner_sha256": sha256(Path(runner)),
        "contract_sha256": contract_sha, "m1053_authority": dict(authority),
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
               "contract_sha256", "m1053_authority", "payload_members_opened",
               "payload_members_statted", "payload_members_hashed",
               "paper_citable"}, "attempt")
    require(value["schema"] == "m1054_attempt_v1" and value["status"] ==
            "M1054_ATTEMPT_CONSUMED_BEFORE_PAYLOAD_MEMBER_ACCESS" and
            value["contract_sha256"] == sha256(CONTRACT) and
            value["m1053_authority"] == dict(authority) and
            all(value[key] is False for key in ("payload_members_opened",
                 "payload_members_statted", "payload_members_hashed",
                 "paper_citable")), "attempt boundary drift")
    return value


def validate_payload_after_attempt(attempt, work, authority):
    validate_attempt(attempt, authority)
    work = safe_path(work, "work")
    require(work.is_dir() and not any(work.iterdir()), "postattempt work not fresh")
    value = contract_value()
    m785_path = HW / value["source_identity"]["m785_contract"]["path"]
    M785.validate_source_contract(REPO, m785_path)
    root = value["source_identity"]["m699_root_seal"]
    payload_dir = HW / root["directory"]
    sealed = M785.verify_sealed_directory(payload_dir)
    require(sha256(payload_dir / "manifest.json") == root["payload_manifest_sha256"] and
            sealed["manifest_sha256"] == root["root_manifest_sha256"] and
            sealed["outer_seal_file_sha256"] == root["outer_seal_file_sha256"],
            "postattempt M699 full identity drift")
    manifest = M785.strict_json(payload_dir / "manifest.json")
    records = M785.normalized_population_records(manifest, M1048.POPULATION_ID)
    selected = []
    for layer in LAYERS:
        row = M1048.select_record(records, layer)
        selected.append({"layer": layer, "sequence": row["sequence"],
            "sample_id": int(row["sample_id"]), "module_index": int(row["module_index"]),
            "route": row["route"], "relative_path": row["relative_path"],
            "packed_sha256": row["packed_sha256"]})
    receipt = {"schema": PAYLOAD_SCHEMA,
        "status": "PASS_M1054_POSTATTEMPT_FULL_PAYLOAD_IDENTITY",
        "attempt_receipt_sha256": sha256(Path(attempt) / "attempt.json"),
        "m699_manifest_sha256": sha256(payload_dir / "manifest.json"),
        "m699_root_manifest_sha256": sealed["manifest_sha256"],
        "m699_outer_seal_file_sha256": sealed["outer_seal_file_sha256"],
        "selected_records": selected, "payload_members_verified": True,
        "post_attempt": True, "d1_scheduled": False,
        "paper_citable": False}
    validate_payload_receipt(receipt)
    atomic_json(work / "payload_validation.json", receipt)
    return receipt


CYCLE_CLASSES = {"active_service", "compute", "dependency_completion",
                 "memory", "psum_bank", "weight_bank"}
EXACT_RESULT_KEYS = {"status", "window_identity_sha256", "total_cycles",
    "expanded_request_count", "compressed_transaction_count",
    "commit_request_count", "cycle_classes", "transaction_address_sha256",
    "commit_sequence_sha256", "terminal_readiness_sha256",
    "port_calendars_sha256", "live_token_final_zero",
    "outstanding_return_final_zero", "cycle_class_sum_equals_total",
    "exact_fields"}
RESET_KEYS = {"window_identity_sha256", "original_transaction_id_census_sha256",
    "reset_transaction_id_census_sha256", "body_expanded_request_count",
    "reset_expanded_request_count", "external_dependency_remap_count",
    "boundary_ready_token_sha256"}
METADATA_KEYS = set(M1041.BASE.BASE.METADATA_SCHEMA)


def validate_payload_receipt(value):
    reject_forbidden_semantic_keys(value)
    exact_dict(value, {"schema", "status", "attempt_receipt_sha256",
               "m699_manifest_sha256", "m699_root_manifest_sha256",
               "m699_outer_seal_file_sha256", "selected_records",
               "payload_members_verified", "post_attempt", "d1_scheduled",
               "paper_citable"}, "payload receipt")
    require(value["schema"] == PAYLOAD_SCHEMA and value["status"] ==
            "PASS_M1054_POSTATTEMPT_FULL_PAYLOAD_IDENTITY", "payload status drift")
    for key in ("attempt_receipt_sha256", "m699_manifest_sha256",
                "m699_root_manifest_sha256", "m699_outer_seal_file_sha256"):
        hash_value(value[key])
    exact_list(value["selected_records"], 3, "selected records")
    for layer, row in zip(LAYERS, value["selected_records"]):
        exact_dict(row, {"layer", "sequence", "sample_id", "module_index",
                   "route", "relative_path", "packed_sha256"}, "selected record")
        require(row["layer"] == layer and row["sequence"] == M1048.SEQUENCE and
                exact_int(row["sample_id"]) == 0 and
                exact_int(row["module_index"]) == M1048.MODULE_BY_LAYER[layer] and
                row["route"] == "EXACT_BINARY_BITPACK", "selected record drift")
        hash_value(row["packed_sha256"])
    exact_bool(value["payload_members_verified"], True)
    exact_bool(value["post_attempt"], True)
    exact_bool(value["d1_scheduled"], False)
    exact_bool(value["paper_citable"], False)
    return True


def validate_exact_result(value, window_sha):
    exact_dict(value, EXACT_RESULT_KEYS, "exact replay")
    require(value["status"] == "PASS_M768_M861_M890_M896_BLOCK_RESET_EXACT_MITER" and
            value["window_identity_sha256"] == window_sha, "exact status/identity drift")
    for key in ("window_identity_sha256", "transaction_address_sha256",
                "commit_sequence_sha256", "terminal_readiness_sha256",
                "port_calendars_sha256"):
        hash_value(value[key])
    total = exact_int(value["total_cycles"], 1)
    for key in ("expanded_request_count", "compressed_transaction_count",
                "commit_request_count"):
        exact_int(value[key], 0)
    exact_dict(value["cycle_classes"], CYCLE_CLASSES, "cycle classes")
    require(all(type(item) is int and item >= 0 for item in
                value["cycle_classes"].values()) and
            sum(value["cycle_classes"].values()) == total, "cycle class drift")
    for key in ("live_token_final_zero", "outstanding_return_final_zero",
                "cycle_class_sum_equals_total"):
        exact_bool(value[key], True)
    require(value["exact_fields"] == list(M1048.M1041.M946.EXACT_FIELDS),
            "exact field identity drift")
    return True


def validate_reset(value, window_sha):
    exact_dict(value, RESET_KEYS, "reset")
    require(value["window_identity_sha256"] == window_sha, "reset identity drift")
    for key in ("window_identity_sha256", "original_transaction_id_census_sha256",
                "reset_transaction_id_census_sha256", "boundary_ready_token_sha256"):
        hash_value(value[key])
    exact_int(value["body_expanded_request_count"], 1)
    require(exact_int(value["reset_expanded_request_count"], 0) == 3,
            "reset request charge drift")
    exact_int(value["external_dependency_remap_count"], 0)


def cycles_only_envelope(old):
    widths = old["uncertainty"]
    output = {"schema": ENVELOPE_SCHEMA,
        "status": ({"HARD_STOP_ABOVE_10_PERCENT": "CYCLE_CI_HARD_STOP_ABOVE_10_PERCENT",
                    "DIAGNOSTIC_5_TO_10_PERCENT": "CYCLE_CI_DIAGNOSTIC_5_TO_10_PERCENT",
                    "CANDIDATE_AT_MOST_5_PERCENT": "CYCLE_CI_AT_MOST_5_PERCENT_NO_DERIVED_VALUES"}[old["state"]]),
        "state": old["state"],
        "bounds": {"candidate_total_cycles_ci95": old["bounds"]["candidate_total_cycles_ci95"],
                   "baseline_total_cycles_ci95": old["bounds"]["baseline_total_cycles_ci95"]},
        "uncertainty": {"candidate_cycles_relative_halfwidth": widths["candidate_cycles_relative_halfwidth"],
                        "baseline_cycles_relative_halfwidth": widths["baseline_cycles_relative_halfwidth"],
                        "maximum_relative_halfwidth": max(widths["candidate_cycles_relative_halfwidth"], widths["baseline_cycles_relative_halfwidth"]),
                        "t_critical": widths["t_critical"]},
        "coverage": copy.deepcopy(old["coverage"]),
        "identity": {"metric": "serial block-reset executable schedule raw cycles"},
        "admission": {"derived_values_emitted": False, "paper_citable": False}}
    validate_envelope(output)
    return output


def validate_envelope(value):
    reject_forbidden_semantic_keys(value)
    exact_dict(value, {"schema", "status", "state", "bounds", "uncertainty",
               "coverage", "identity", "admission"}, "cycle CI envelope")
    require(value["schema"] == ENVELOPE_SCHEMA and value["state"] in
            ("HARD_STOP_ABOVE_10_PERCENT", "DIAGNOSTIC_5_TO_10_PERCENT",
             "CANDIDATE_AT_MOST_5_PERCENT"), "envelope state drift")
    expected_status = {"HARD_STOP_ABOVE_10_PERCENT": "CYCLE_CI_HARD_STOP_ABOVE_10_PERCENT",
       "DIAGNOSTIC_5_TO_10_PERCENT": "CYCLE_CI_DIAGNOSTIC_5_TO_10_PERCENT",
       "CANDIDATE_AT_MOST_5_PERCENT": "CYCLE_CI_AT_MOST_5_PERCENT_NO_DERIVED_VALUES"}[value["state"]]
    require(value["status"] == expected_status, "envelope status drift")
    exact_dict(value["bounds"], {"candidate_total_cycles_ci95",
               "baseline_total_cycles_ci95"}, "bounds")
    for interval in value["bounds"].values():
        exact_list(interval, 2, "CI interval")
        require(all(finite(item, 0) is not None for item in interval) and
                interval[0] <= interval[1], "CI interval drift")
    exact_dict(value["uncertainty"], {"candidate_cycles_relative_halfwidth",
               "baseline_cycles_relative_halfwidth", "maximum_relative_halfwidth",
               "t_critical"}, "uncertainty")
    require(all(finite(value["uncertainty"][key], 0) is not None for key in
                ("candidate_cycles_relative_halfwidth",
                 "baseline_cycles_relative_halfwidth",
                 "maximum_relative_halfwidth")) and
            finite(value["uncertainty"]["t_critical"], 0) > 0 and
            math.isclose(value["uncertainty"]["maximum_relative_halfwidth"],
                max(value["uncertainty"]["candidate_cycles_relative_halfwidth"],
                    value["uncertainty"]["baseline_cycles_relative_halfwidth"]),
                rel_tol=1e-12, abs_tol=1e-12), "uncertainty identity drift")
    worst = value["uncertainty"]["maximum_relative_halfwidth"]
    expected_state = ("HARD_STOP_ABOVE_10_PERCENT" if worst > 0.10 else
                      "DIAGNOSTIC_5_TO_10_PERCENT" if worst > 0.05 else
                      "CANDIDATE_AT_MOST_5_PERCENT")
    require(value["state"] == expected_state, "uncertainty/state binding drift")
    exact_dict(value["coverage"], {"strata"}, "envelope coverage")
    exact_list(value["coverage"]["strata"], 3, "envelope coverage rows")
    for stratum, row in zip(NONCENSUS, value["coverage"]["strata"]):
        exact_dict(row, {"stratum", "population_blocks", "sample_blocks",
                   "finite_population_fraction"}, "envelope coverage row")
        pop = exact_int(row["population_blocks"], 1)
        sample = exact_int(row["sample_blocks"], 1)
        require(row["stratum"] == stratum and sample == PILOT and
                sample <= pop and math.isclose(
                    finite(row["finite_population_fraction"], 0), sample / pop,
                    rel_tol=1e-12, abs_tol=1e-12),
                "envelope coverage identity drift")
    exact_dict(value["identity"], {"metric"}, "envelope identity")
    require(value["identity"]["metric"] ==
            "serial block-reset executable schedule raw cycles", "metric drift")
    exact_dict(value["admission"], {"derived_values_emitted", "paper_citable"},
               "envelope admission")
    exact_bool(value["admission"]["derived_values_emitted"], False)
    exact_bool(value["admission"]["paper_citable"], False)
    return True


def transform_layer(old):
    windows = []
    for row in old["windows"]:
        windows.append({key: copy.deepcopy(row[key]) for key in (
            "block_id", "window_identity_sha256", "stratum", "metadata",
            "body_transaction_ids_sha256", "body_compressed_transaction_count",
            "body_expanded_request_count", "candidate_cycles", "baseline_cycles",
            "exact_mismatch_count", "candidate_exact", "baseline_exact",
            "candidate_reset", "baseline_reset", "paired_reset_semantics_sha256",
            "pair_role")})
    return {"layer": old["layer"], "record_identity": copy.deepcopy(old["record_identity"]),
        "selection_identity_sha256": old["selection_identity_sha256"],
        "selection_frozen_before_cycle_replay": True,
        "block_population_index_sha256": old["block_population_index_sha256"],
        "transaction_assignment_census_sha256": old["transaction_assignment_census_sha256"],
        "generated_compressed_transactions": old["generated_compressed_transactions"],
        "assigned_compressed_transactions": old["assigned_compressed_transactions"],
        "coverage": copy.deepcopy(old["coverage"]),
        "source_census_cycles": copy.deepcopy(old["source_census_cycles"]),
        "ci_raw_inputs": copy.deepcopy(old["ci_raw_inputs"]),
        "cycle_ci_envelope": cycles_only_envelope(old["ci_publication_envelope"]),
        "windows": windows, "exact_mismatch_count": old["exact_mismatch_count"],
        "continuous_row_cycles": False}


def validate_metadata(value):
    exact_dict(value, METADATA_KEYS, "window metadata")
    M1041.validate_metadata_row(value)


def validate_layer(value, expected_layer):
    reject_forbidden_semantic_keys(value)
    exact_dict(value, {"layer", "record_identity", "selection_identity_sha256",
               "selection_frozen_before_cycle_replay", "block_population_index_sha256",
               "transaction_assignment_census_sha256", "generated_compressed_transactions",
               "assigned_compressed_transactions", "coverage", "source_census_cycles",
               "ci_raw_inputs", "cycle_ci_envelope", "windows",
               "exact_mismatch_count", "continuous_row_cycles"}, "layer")
    require(value["layer"] == expected_layer, "layer identity drift")
    for key in ("selection_identity_sha256", "block_population_index_sha256",
                "transaction_assignment_census_sha256"):
        hash_value(value[key])
    exact_bool(value["selection_frozen_before_cycle_replay"], True)
    generated = exact_int(value["generated_compressed_transactions"], 1)
    require(exact_int(value["assigned_compressed_transactions"], 1) == generated,
            "transaction conservation drift")
    exact_int(value["exact_mismatch_count"], 0)
    require(value["exact_mismatch_count"] == 0, "exact mismatch nonzero")
    exact_bool(value["continuous_row_cycles"], False)
    record = exact_dict(value["record_identity"], {"population_id", "sequence",
        "sample_id", "module_index", "timestep", "config", "route",
        "relative_path", "packed_sha256"}, "record identity")
    require(record["population_id"] == M1048.POPULATION_ID and
            record["sequence"] == M1048.SEQUENCE and record["sample_id"] == 0 and
            record["module_index"] == M1048.MODULE_BY_LAYER[expected_layer] and
            record["timestep"] == 0 and record["config"] == M1048.CONFIG and
            record["route"] == "EXACT_BINARY_BITPACK", "record identity drift")
    hash_value(record["packed_sha256"])
    exact_list(value["coverage"], 4, "coverage")
    population = {}
    for stratum, row in zip(STRATA, value["coverage"]):
        exact_dict(row, {"stratum", "population_blocks", "sample_blocks",
                   "finite_population_fraction"}, "coverage row")
        require(row["stratum"] == stratum, "coverage stratum drift")
        pop = exact_int(row["population_blocks"], 1)
        sample = exact_int(row["sample_blocks"], 1)
        require(sample == (1 if stratum == STRATA[0] else PILOT) and sample <= pop and
                math.isclose(finite(row["finite_population_fraction"], 0), sample/pop,
                             rel_tol=1e-12, abs_tol=1e-12), "coverage identity drift")
        population[stratum] = pop
    source = exact_dict(value["source_census_cycles"], {"candidate", "baseline"},
                        "source cycles")
    require(exact_int(source["candidate"], 1) == exact_int(source["baseline"], 1),
            "self-matched source cycles drift")
    exact_list(value["ci_raw_inputs"], 3, "CI raw rows")
    raw_by_stratum = {}
    for stratum, row in zip(NONCENSUS, value["ci_raw_inputs"]):
        exact_dict(row, {"stratum", "population_blocks", "candidate_cycles",
                   "baseline_cycles"}, "CI raw row")
        require(row["stratum"] == stratum and
                exact_int(row["population_blocks"], 1) == population[stratum],
                "CI raw identity drift")
        exact_list(row["candidate_cycles"], PILOT, "candidate cycle samples")
        exact_list(row["baseline_cycles"], PILOT, "baseline cycle samples")
        require(all(type(item) is int and item > 0 for item in
                    row["candidate_cycles"] + row["baseline_cycles"]) and
                row["candidate_cycles"] == row["baseline_cycles"],
                "raw self-matched cycle drift")
        raw_by_stratum[stratum] = row
    exact_list(value["windows"], 25, "windows")
    cycle_by_stratum = {name: [] for name in STRATA}
    metadata_for_selection = {name: [] for name in STRATA}
    for window in value["windows"]:
        exact_dict(window, {"block_id", "window_identity_sha256", "stratum",
            "metadata", "body_transaction_ids_sha256",
            "body_compressed_transaction_count", "body_expanded_request_count",
            "candidate_cycles", "baseline_cycles", "exact_mismatch_count",
            "candidate_exact", "baseline_exact", "candidate_reset",
            "baseline_reset", "paired_reset_semantics_sha256", "pair_role"}, "window")
        require(window["stratum"] in STRATA and
                window["pair_role"] == "SELF_MATCHED_A1_OSG_PROTOCOL_CALIBRATION",
                "window role drift")
        validate_metadata(window["metadata"])
        require(window["metadata"]["block_id"] == window["block_id"] and
                M1041.classify_stratum(window["metadata"]) == window["stratum"],
                "window metadata identity drift")
        spec = M1041.WindowSpec(window["block_id"], expected_layer,
            window["stratum"], population[window["stratum"]], 0, 0)
        require(window["window_identity_sha256"] == spec.identity_sha256,
                "window identity recompute drift")
        for key in ("window_identity_sha256", "body_transaction_ids_sha256",
                    "paired_reset_semantics_sha256"):
            hash_value(window[key])
        exact_int(window["body_compressed_transaction_count"], 1)
        body_expanded = exact_int(window["body_expanded_request_count"], 1)
        require(body_expanded + 3 <= M1048.CAP and
                exact_int(window["candidate_cycles"], 1) ==
                exact_int(window["baseline_cycles"], 1) and
                exact_int(window["exact_mismatch_count"], 0) == 0,
                "window raw cycle/count drift")
        validate_exact_result(window["candidate_exact"], spec.identity_sha256)
        validate_exact_result(window["baseline_exact"], spec.identity_sha256)
        require(window["candidate_exact"] == window["baseline_exact"] and
                window["candidate_exact"]["total_cycles"] ==
                window["candidate_cycles"], "exact replay binding drift")
        validate_reset(window["candidate_reset"], spec.identity_sha256)
        validate_reset(window["baseline_reset"], spec.identity_sha256)
        cycle_by_stratum[window["stratum"]].append(window["candidate_cycles"])
        metadata_for_selection[window["stratum"]].append(window["metadata"])
    require(len(cycle_by_stratum[STRATA[0]]) == 1 and
            all(len(cycle_by_stratum[name]) == PILOT for name in NONCENSUS) and
            cycle_by_stratum[STRATA[0]][0] == source["candidate"],
            "window stratum cardinality/source binding drift")
    for stratum in NONCENSUS:
        require(cycle_by_stratum[stratum] == raw_by_stratum[stratum]["candidate_cycles"],
                "window/CI raw cycle order drift")
    selection_sha = canonical_sha({stratum: [[
        M1048._selection_key(row["block_id"]), row["block_id"], row]
        for row in metadata_for_selection[stratum]] for stratum in STRATA})
    require(selection_sha == value["selection_identity_sha256"],
            "selection identity recompute drift")
    raw_old = M1041.estimate_paired_totals(value["ci_raw_inputs"],
        fixed_candidate=source["candidate"], fixed_baseline=source["baseline"])
    expected_envelope = cycles_only_envelope(raw_old)
    require(value["cycle_ci_envelope"] == expected_envelope,
            "CI envelope recompute drift")
    validate_envelope(value["cycle_ci_envelope"])
    return True


def validate_raw(value):
    reject_forbidden_semantic_keys(value)
    exact_dict(value, {"schema", "status", "workload", "pair_role", "layers",
               "exact_mismatch_count", "d1", "claim_boundary"}, "raw")
    require(value["schema"] == RAW_SCHEMA and value["status"] ==
            "PASS_M1054_RAW_CYCLES__RESULT_HAMMER_REQUIRED" and
            value["pair_role"] == "SELF_MATCHED_PROTOCOL_CALIBRATION",
            "raw header drift")
    require(value["workload"] == {"population_id": M1048.POPULATION_ID,
            "sequence": M1048.SEQUENCE, "sample_id": 0, "timestep": 0,
            "config": M1048.CONFIG, "layers": list(LAYERS)}, "raw workload drift")
    exact_list(value["layers"], 3, "raw layers")
    for layer, row in zip(LAYERS, value["layers"]): validate_layer(row, layer)
    require(exact_int(value["exact_mismatch_count"], 0) == 0, "raw mismatch drift")
    require(value["d1"] == {"status": "DIAGNOSTIC_ONLY", "scheduled": False,
            "numeric_equivalence_admitted": False}, "raw D1 drift")
    exact_dict(value["claim_boundary"], {"paper_citable", "decoder_complete",
               "table_a_row", "system_performance_claim", "local_performance_claim",
               "continuous_row_cycles"}, "raw claims")
    require(all(exact_bool(item, False) is False
                for item in value["claim_boundary"].values()), "raw claim drift")
    return True


def make_result(raw, raw_sha, payload_sha):
    return {"schema": RESULT_SCHEMA,
        "status": "PASS_M1054_DIAGNOSTIC_RAW_CYCLE_PILOT__RESULT_HAMMER_REQUIRED",
        "result_role": "PROTOCOL_CALIBRATION_ONLY",
        "raw_windows_sha256": raw_sha, "payload_validation_sha256": payload_sha,
        "layers": [{"layer": row["layer"],
            "selection_identity_sha256": row["selection_identity_sha256"],
            "block_population_index_sha256": row["block_population_index_sha256"],
            "transaction_assignment_census_sha256": row["transaction_assignment_census_sha256"],
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
        "next_gate": "Independent receipt-blind M1055 result hammer"}


def validate_result(value, raw, raw_sha, payload_sha):
    reject_forbidden_semantic_keys(value)
    expected = make_result(raw, raw_sha, payload_sha)
    require(value == expected, "result exact derivation/schema drift")
    return True


def run_pilot(attempt, work, authority):
    validate_attempt(attempt, authority)
    work = safe_path(work, "work")
    require({item.name for item in work.iterdir()} == {"payload_validation.json"},
            "payload validation receipt required before run")
    payload = strict_json(work / "payload_validation.json")
    validate_payload_receipt(payload)
    require(payload["attempt_receipt_sha256"] ==
            sha256(Path(attempt) / "attempt.json"), "payload/attempt binding drift")
    payload_root, records, mapper, oracles = M1048._context()
    old_layers = [M1048.replay_layer(layer, M1048.select_record(records, layer),
                                    payload_root, mapper, oracles)
                  for layer in LAYERS]
    layers = [transform_layer(row) for row in old_layers]
    raw = {"schema": RAW_SCHEMA,
        "status": "PASS_M1054_RAW_CYCLES__RESULT_HAMMER_REQUIRED",
        "workload": {"population_id": M1048.POPULATION_ID,
            "sequence": M1048.SEQUENCE, "sample_id": 0, "timestep": 0,
            "config": M1048.CONFIG, "layers": list(LAYERS)},
        "pair_role": "SELF_MATCHED_PROTOCOL_CALIBRATION", "layers": layers,
        "exact_mismatch_count": 0,
        "d1": {"status": "DIAGNOSTIC_ONLY", "scheduled": False,
               "numeric_equivalence_admitted": False},
        "claim_boundary": {"paper_citable": False, "decoder_complete": False,
            "table_a_row": False, "system_performance_claim": False,
            "local_performance_claim": False, "continuous_row_cycles": False}}
    validate_raw(raw)
    atomic_json(work / "raw_windows.json", raw)
    result = make_result(raw, sha256(work / "raw_windows.json"),
                         sha256(work / "payload_validation.json"))
    validate_result(result, raw, result["raw_windows_sha256"],
                    result["payload_validation_sha256"])
    atomic_json(work / "result.json", result)
    (work / "RUN_COMPLETE.txt").write_text(result["status"] + "\n", encoding="utf-8")
    return result


def assemble(work):
    work = safe_path(work, "work")
    expected_files = {"payload_validation.json", "raw_windows.json",
                      "result.json", "RUN_COMPLETE.txt"}
    require(work.is_dir() and not work.is_symlink() and
            {item.name for item in work.iterdir()} == expected_files and
            all(item.is_file() and not item.is_symlink() for item in work.iterdir()),
            "work exact-set drift")
    payload = strict_json(work / "payload_validation.json")
    raw = strict_json(work / "raw_windows.json")
    result = strict_json(work / "result.json")
    validate_payload_receipt(payload)
    validate_raw(raw)
    validate_result(result, raw, sha256(work / "raw_windows.json"),
                    sha256(work / "payload_validation.json"))
    require((work / "RUN_COMPLETE.txt").read_text(encoding="utf-8") ==
            result["status"] + "\n", "completion token drift")
    lines = [sha256(work / name) + "  " + name for name in sorted(expected_files)]
    (work / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_sha = sha256(work / "SHA256SUMS")
    (work / "SHA256SUMS.seal.sha256").write_text(
        manifest_sha + "  SHA256SUMS\n", encoding="utf-8")
    return {"status": "PASS_M1054_STRICT_OUTPUT_SEALED",
            "manifest_sha256": manifest_sha,
            "outer_seal_file_sha256": sha256(work / "SHA256SUMS.seal.sha256")}


def publish(work, result):
    work, result = safe_path(work, "work"), safe_path(result, "result")
    require(work.is_dir() and not result.exists(), "publish namespace drift")
    verify_flat_seal(work)
    os.replace(work, result)
    return {"status": "PASS_M1054_ATOMIC_RESULT_PUBLISHED"}


def quarantine(work, quarantine, return_code):
    work, quarantine = safe_path(work, "work"), safe_path(quarantine, "quarantine")
    require(work.is_dir() and not quarantine.exists(), "quarantine namespace drift")
    os.replace(work, quarantine)
    atomic_json(quarantine / "FAILURE.json", {"schema": "m1054_failure_v1",
        "status": "FAILED_OR_INCOMPLETE_DO_NOT_CITE", "return_code": int(return_code),
        "paper_citable": False})
    return {"status": "PASS_M1054_FAILURE_QUARANTINED"}


def self_test():
    # Reuse the frozen 85-transaction partition/selector test, then hammer the
    # repaired recursive validators without opening any payload member.
    old = M1048.self_test()
    require(old["compressed_transactions"] == 85 and
            old["streaming_selection_matches_m1041"] is True, "M1048 synthetic drift")
    base = {"schema": ENVELOPE_SCHEMA,
        "status": "CYCLE_CI_AT_MOST_5_PERCENT_NO_DERIVED_VALUES",
        "state": "CANDIDATE_AT_MOST_5_PERCENT",
        "bounds": {"candidate_total_cycles_ci95": [1.0, 2.0],
                   "baseline_total_cycles_ci95": [1.0, 2.0]},
        "uncertainty": {"candidate_cycles_relative_halfwidth": 0.04,
            "baseline_cycles_relative_halfwidth": 0.04,
            "maximum_relative_halfwidth": 0.04, "t_critical": 2.365},
        "coverage": {"strata": [{"stratum": name, "population_blocks": 8,
            "sample_blocks": 8, "finite_population_fraction": 1.0}
            for name in NONCENSUS]},
        "identity": {"metric": "serial block-reset executable schedule raw cycles"},
        "admission": {"derived_values_emitted": False, "paper_citable": False}}
    validate_envelope(base)
    attacks = []
    for key in ("candidate_mean_cycles", "baselineMean", "point_speedup",
                "speedups", "normalizedCycles", "runtimeEstimate", "throughput",
                "FPS"):
        attack = copy.deepcopy(base)
        attack["coverage"]["strata"][0]["nested"] = {"deeper": {key: 1.0}}
        try: validate_envelope(attack)
        except RuntimeError: attacks.append(key)
    require(len(attacks) == 8, "semantic attack survived")
    return {"status": "PASS_M1052_SMALL_SYNTHETIC_REPAIR_SELFTEST",
        "m1048_transactions": 85, "semantic_attacks_rejected": attacks,
        "real_payload_members_opened": False, "real_window_execution": False,
        "eda_gpu_remote_used": False}


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("validate-pre-attempt-source", "validate-authority",
                 "consume-attempt", "validate-payload-after-attempt",
                 "run-pilot", "assemble", "publish", "quarantine", "self-test"):
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
    authority = None
    if args.validate_pre_attempt_source:
        require(args.runner and args.expected_contract_sha and
                sha256(args.contract) == args.expected_contract_sha,
                "pre-attempt caller identity absent")
        output = validate_pre_attempt_source(args.contract, args.runner)
    elif args.validate_authority:
        require(all(pins), "M1053 pins absent")
        output = validate_m1053(*pins)
    elif args.consume_attempt:
        require(args.attempt and args.runner and args.expected_contract_sha and
                all(pins), "attempt authority absent")
        authority = validate_m1053(*pins)
        output = consume_attempt(args.attempt, args.runner,
                                 args.expected_contract_sha, authority)
    elif args.validate_payload_after_attempt:
        require(args.attempt and args.work and all(pins), "payload authority absent")
        authority = validate_m1053(*pins)
        output = validate_payload_after_attempt(args.attempt, args.work, authority)
    elif args.run_pilot:
        require(args.attempt and args.work and all(pins), "run authority absent")
        authority = validate_m1053(*pins)
        output = run_pilot(args.attempt, args.work, authority)
    elif args.assemble:
        require(args.work and all(pins), "assemble authority absent")
        validate_m1053(*pins); output = assemble(args.work)
    elif args.publish:
        require(args.work and args.result and all(pins), "publish authority absent")
        validate_m1053(*pins); output = publish(args.work, args.result)
    elif args.quarantine:
        require(args.work and args.quarantine_path, "quarantine paths absent")
        output = quarantine(args.work, args.quarantine_path, args.return_code)
    else:
        output = self_test()
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
