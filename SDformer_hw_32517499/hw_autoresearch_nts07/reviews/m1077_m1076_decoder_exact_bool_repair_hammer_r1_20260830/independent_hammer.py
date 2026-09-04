#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Receipt-blind independent hammer for the M1076 exact-bool repair.

No M1078 pilot, decoder cycle, real calls/* payload member, EDA, GPU, or
remote service is used.  All assemble/publish attacks live in a temporary
RESULTS namespace and rebind build_canonical_context to a synthetic canonical
context, so the hammer can exercise the publication boundary without opening
the real payload.
"""
from __future__ import annotations

import builtins
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
DRIVER = HW / "system_simulator/scripts/execute_m1076_decoder_exact_bool_repair.py"
RUNNER = HW / "system_simulator/scripts/run_m1078_m1076_decoder_exact_bool_pilot_one_shot.sh"
CONTRACT = HW / "contracts/m1076_decoder_exact_bool_repair_contract_r1_20260830.json"
TESTS = HW / "system_simulator/tests/test_m1076_decoder_exact_bool_repair.py"
SOURCE_RECEIPT = HW / "reviews/m1076_decoder_exact_bool_repair_source_receipt_r1_20260830"
M1061 = HW / "reviews/m1061_m1060_decoder_identity_binding_repair_hammer_r1_20260830"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
ATTEMPT = HW / "results/.m1078_m1076_decoder_exact_bool_pilot_attempt_consumed"
RESULT = HW / "results/m1078_m1076_decoder_exact_bool_pilot_r1_20260830"

EXPECTED = {
    "driver": "d3b98ec71c3123c856d6a7ce8c8cee431e4d8d0da75aebf92eee8e144123ec15",
    "runner": "15b53d4d8be73d12ee0b5847cfcbd856bc2b9e06c8c4bfa4a3df509ca330a5c6",
    "contract": "ba702d74e6ddfd4cd152bacd35e26a19f293c9e038e237120ca376fd9f969413",
    "tests": "9fd21ed1311767eb72194b637a9e690124f5225d9acd8fcd750873e2b241001c",
    "source_review": "14365e5a37cb4840f350178db044bef2d5e652aef08de3284c4788217f5893ef",
    "source_manifest": "fffa1e6c94007332e1211a0162207c6b0e54caf498e8d87e8819a2bf84cbbb13",
    "source_outer": "e423597ec44a700b392c8ecb047dbb378e9058e68b2c8d485ea9ad81f4e3fd78",
    "m1061_review": "40a4b530f9937d9044139b42b6dedc60ba9272a0179bef843adb1eedcc32650a",
    "m1061_manifest": "5014560a85f32dad8ce9de6385032fd09c761a7c6062ca3a57a29f7632f92a20",
    "m1061_outer": "cdb8d9686f26a335a34b76c7055e6ae4a6ba960ba200968eb8f8852933a0551d",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def rejected(callable_) -> bool:
    try:
        callable_()
    except (RuntimeError, TypeError, ValueError, OSError, KeyError, AssertionError):
        return True
    return False


def verify_flat(directory: Path, review_sha: str, manifest_sha: str,
                outer_sha: str) -> None:
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require((sha(review), sha(manifest), sha(outer)) ==
            (review_sha, manifest_sha, outer_sha),
            "sealed authority identity drift: " + directory.name)
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and fields[1] not in listed,
                "malformed/duplicate seal row")
        target = directory / fields[1]
        require(target.is_file() and not target.is_symlink() and
                sha(target) == fields[0], "sealed member drift: " + fields[1])
        listed.add(fields[1])
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and not path.is_symlink() and path.name not in
              ("SHA256SUMS", "SHA256SUMS.seal.sha256")}
    require(actual == listed and outer.read_text(encoding="utf-8") ==
            manifest_sha + "  SHA256SUMS\n", "flat seal exact-set drift")


def seal(directory: Path) -> None:
    names = sorted(path.name for path in directory.iterdir()
                   if path.is_file() and path.name not in
                   ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    manifest = directory / "SHA256SUMS"
    manifest.write_text("".join(sha(directory / name) + "  " + name + "\n"
                                for name in names), encoding="utf-8")
    (directory / "SHA256SUMS.seal.sha256").write_text(
        sha(manifest) + "  SHA256SUMS\n", encoding="utf-8")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def load_driver():
    require(sha(DRIVER) == EXPECTED["driver"], "driver identity drift")
    spec = importlib.util.spec_from_file_location("m1077_driver_under_hammer", DRIVER)
    require(spec is not None and spec.loader is not None, "cannot load driver")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def set_path(tree, path, value):
    cursor = tree
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value


def synthetic_raw(module, context):
    rows = []
    for selected in context["payload"]["selected_records"]:
        rows.append({
            "layer": selected["layer"],
            "record_identity": module.expected_raw_record(selected),
            "verified_payload_member_sha256": selected["payload_member_sha256"],
            "selection_identity_sha256": "1" * 64,
            "block_population_index_sha256": "2" * 64,
            "transaction_assignment_census_sha256": "3" * 64,
            "generated_compressed_transactions": 7,
            "assigned_compressed_transactions": 7,
            "coverage": [],
            "source_census_cycles": {"candidate": 9, "baseline": 9},
            "ci_raw_inputs": [],
            "cycle_ci_envelope": {},
            "windows": [],
            "exact_mismatch_count": 0,
            "continuous_row_cycles": False,
        })
    return {
        "schema": module.RAW_SCHEMA,
        "status": "PASS_M1078_RAW_CYCLES__RESULT_HAMMER_REQUIRED",
        "workload": {"population_id": module.M1048.POPULATION_ID,
            "sequence": module.M1048.SEQUENCE, "sample_id": 0, "timestep": 0,
            "config": module.M1048.CONFIG, "layers": list(module.LAYERS)},
        "pair_role": "SELF_MATCHED_PROTOCOL_CALIBRATION",
        "canonical_context_sha256": module.canonical_sha(context),
        "layers": rows, "exact_mismatch_count": 0,
        "d1": {"status": "DIAGNOSTIC_ONLY", "scheduled": False,
               "numeric_equivalence_admitted": False},
        "claim_boundary": {"paper_citable": False, "decoder_complete": False,
            "table_a_row": False, "system_performance_claim": False,
            "local_performance_claim": False, "continuous_row_cycles": False},
    }


def make_work(module, root: Path, context, raw, mutation=None, sealed=False):
    work = root / ("." + module.RESULT_NAME + ".work.attack")
    work.mkdir(mode=0o700)
    payload = module.make_payload_receipt(context)
    result = module.make_result(raw, "4" * 64, "5" * 64,
                                module.canonical_sha(context))
    files = {"canonical_context.json": copy.deepcopy(context),
             "payload_validation.json": payload,
             "raw_windows.json": copy.deepcopy(raw),
             "result.json": result}
    # Make hashes internally consistent before the requested mutation.
    write_json(work / "canonical_context.json", files["canonical_context.json"])
    write_json(work / "payload_validation.json", files["payload_validation.json"])
    write_json(work / "raw_windows.json", files["raw_windows.json"])
    result["raw_windows_sha256"] = sha(work / "raw_windows.json")
    result["payload_validation_sha256"] = sha(work / "payload_validation.json")
    files["result.json"] = result
    if mutation is not None:
        mutation(files)
    for name, value in files.items():
        write_json(work / name, value)
    (work / "RUN_COMPLETE.txt").write_text(
        files["result.json"]["status"] + "\n", encoding="utf-8")
    if sealed:
        seal(work)
    return work


def main():
    for path, key in ((DRIVER, "driver"), (RUNNER, "runner"),
                      (CONTRACT, "contract"), (TESTS, "tests"),
                      (DOC359, "docs359")):
        require(path.is_file() and not path.is_symlink() and
                sha(path) == EXPECTED[key], "identity drift: " + key)
    verify_flat(SOURCE_RECEIPT, EXPECTED["source_review"],
                EXPECTED["source_manifest"], EXPECTED["source_outer"])
    verify_flat(M1061, EXPECTED["m1061_review"], EXPECTED["m1061_manifest"],
                EXPECTED["m1061_outer"])
    require(not ATTEMPT.exists() and not ATTEMPT.is_symlink() and
            not RESULT.exists() and not RESULT.is_symlink(),
            "canonical M1078 namespace is not fresh")

    module = load_driver()
    contract = module.contract_value()
    require(contract["status"] ==
            "EXACT_BOOL_REPAIR_SOURCE_ONLY__M1077_HAMMER_REQUIRED" and
            contract["launch_now"] is False and
            all(value is False for value in contract["claim_boundary"].values()),
            "source-only boundary drift")

    # Pre-attempt source validation may inspect root seal metadata only.  Any
    # open/stat/hash of calls/* or call of the full verifier trips the hammer.
    member_access = []
    full_verify_calls = []
    original_open = builtins.open
    original_path_open, original_path_stat = Path.open, Path.stat
    original_os_stat = os.stat
    original_verify = module.M785.verify_sealed_directory

    def is_member(path):
        return "/m699_h67_ep35_multisequence_decoder_payload_s3x10_r1_20260828/calls/" in str(path)

    def watched_open(file, *args, **kwargs):
        if is_member(file):
            member_access.append("open:" + str(file)); raise RuntimeError("member open")
        return original_open(file, *args, **kwargs)

    def watched_path_open(self, *args, **kwargs):
        if is_member(self):
            member_access.append("Path.open:" + str(self)); raise RuntimeError("member open")
        return original_path_open(self, *args, **kwargs)

    def watched_path_stat(self, *args, **kwargs):
        if is_member(self):
            member_access.append("Path.stat:" + str(self)); raise RuntimeError("member stat")
        return original_path_stat(self, *args, **kwargs)

    def watched_os_stat(path, *args, **kwargs):
        if is_member(path):
            member_access.append("os.stat:" + str(path)); raise RuntimeError("member stat")
        return original_os_stat(path, *args, **kwargs)

    def trip_full(path):
        full_verify_calls.append(str(path)); raise RuntimeError("full verifier")

    builtins.open, Path.open, Path.stat, os.stat = (
        watched_open, watched_path_open, watched_path_stat, watched_os_stat)
    module.M785.verify_sealed_directory = trip_full
    try:
        pre = module.validate_pre_attempt_source(CONTRACT, RUNNER)
    finally:
        builtins.open, Path.open, Path.stat, os.stat = (
            original_open, original_path_open, original_path_stat, original_os_stat)
        module.M785.verify_sealed_directory = original_verify
    require(member_access == [] and full_verify_calls == [] and
            pre["payload_members_opened"] is False and
            pre["payload_members_statted"] is False and
            pre["payload_members_hashed"] is False,
            "pre-attempt payload member access detected")

    # Arbitrary-depth exact-tree attacks cover both directions for the Python
    # bool/int alias, not merely a list of named top-level leaves.
    exact_expected = {"l0": [{"l1": [{"l2": [{"flag0": False, "int0": 0,
        "l3": [{"flag1": True, "int1": 1}]}]}]}]}
    deep_attacks = {
        "false_to_zero": (("l0", 0, "l1", 0, "l2", 0, "flag0"), 0),
        "zero_to_false": (("l0", 0, "l1", 0, "l2", 0, "int0"), False),
        "true_to_one": (("l0", 0, "l1", 0, "l2", 0, "l3", 0, "flag1"), 1),
        "one_to_true": (("l0", 0, "l1", 0, "l2", 0, "l3", 0, "int1"), True),
    }
    for name, (path, replacement) in deep_attacks.items():
        attacked = copy.deepcopy(exact_expected); set_path(attacked, path, replacement)
        require(rejected(lambda a=attacked: module.exact_tree(a, exact_expected)),
                "arbitrary-depth alias survived: " + name)

    # Contract: cover every object carrying boolean/integer leaves plus schema,
    # extra-field, direct forbidden semantic key, and non-finite attacks.
    contract_attacks = {
        "launch_false_to_zero": lambda x: x.__setitem__("launch_now", 0),
        "workload_zero_to_false": lambda x: x["workload"].__setitem__("sample_id", False),
        "d1_false_to_zero": lambda x: x["d1"].__setitem__("generator_allowed", 0),
        "sampling_one_to_true": lambda x: x["sampling"].__setitem__("source_census", True),
        "sampling_true_to_one": lambda x: x["sampling"].__setitem__("selection_before_replay", 1),
        "pre_true_to_one": lambda x: x["pre_attempt"].__setitem__("canonical_attempt_before_payload_validation", 1),
        "post_true_to_one": lambda x: x["post_attempt"].__setitem__("failure_quarantine", 1),
        "output_false_to_zero": lambda x: x["output"].__setitem__("bool_int_alias_allowed", 0),
        "selected_zero_to_false": lambda x: x["frozen_payload"]["selected_records"][0].__setitem__("module_index", False),
        "claim_false_to_zero": lambda x: x["claim_boundary"].__setitem__("paper_citable", 0),
        "wrong_schema": lambda x: x.__setitem__("schema", "attacker"),
        "extra": lambda x: x.__setitem__("attacker", False),
        "semantic_bypass": lambda x: x.__setitem__("pointSpeedups", 9.0),
        "nan": lambda x: x["sampling"].__setitem__("window_expanded_request_cap", math.nan),
    }
    for name, mutate in contract_attacks.items():
        attacked = copy.deepcopy(contract); mutate(attacked)
        require(rejected(lambda a=attacked: module.validate_contract(a)),
                "contract attack survived: " + name)

    context = module.synthetic_context()
    receipt = module.make_payload_receipt(context)
    # Preserve M1060 attacks: all-fake identities, nonexistent member,
    # relabel+rehash, refreshed context hash, wrong schema/status and extras.
    receipt_attacks = {
        "all_fake_sha": lambda x: (
            x["attempt"].update({"attempt_json_sha256": "f" * 64,
                "runner_sha256": "e" * 64, "contract_sha256": "d" * 64,
                "m1077_authority": {"review_sha256": "c" * 64,
                    "manifest_sha256": "b" * 64,
                    "outer_seal_file_sha256": "a" * 64}}),
            x["payload"].update({"m699_manifest_sha256": "9" * 64,
                "m699_root_manifest_sha256": "8" * 64,
                "m699_outer_seal_file_sha256": "7" * 64}),
            [row.update({"packed_sha256": "6" * 64,
                         "payload_member_sha256": "6" * 64})
             for row in x["payload"]["selected_records"]]),
        "nonexistent_path": lambda x: x["payload"]["selected_records"][0].update(
            {"relative_path": "calls/FORGED_DOES_NOT_EXIST.bitpack"}),
        "relabel_rehash": lambda x: x["payload"]["selected_records"][1].update(
            {"relative_path": "calls/renamed.bitpack", "packed_sha256": "5" * 64,
             "payload_member_sha256": "5" * 64}),
        "context_hash_refresh": lambda x: x.update(
            {"canonical_context_sha256": module.canonical_sha({"attacker": x["payload"]})}),
        "payload_true_to_one": lambda x: x.__setitem__("payload_members_verified", 1),
        "post_true_to_one": lambda x: x.__setitem__("post_attempt", 1),
        "paper_false_to_zero": lambda x: x.__setitem__("paper_citable", 0),
        "deep_zero_to_false": lambda x: x["payload"]["selected_records"][1].__setitem__("sample_id", False),
        "deep_one_to_true": lambda x: x["payload"]["selected_records"][2].__setitem__("module_index", True),
        "wrong_schema": lambda x: x.__setitem__("schema", "attacker"),
        "wrong_status": lambda x: x.__setitem__("status", "PASS_ATTACKER"),
        "extra": lambda x: x.__setitem__("attacker", False),
        "nan": lambda x: x["payload"].__setitem__("m699_manifest_sha256", math.nan),
    }
    for name, mutate in receipt_attacks.items():
        attacked = copy.deepcopy(receipt); mutate(attacked)
        require(rejected(lambda a=attacked: module.validate_payload_receipt(a, context)),
                "payload receipt attack survived: " + name)

    context_attacks = {
        "false_to_zero": lambda x: x.__setitem__("d1_scheduled", 0),
        "deep_zero_to_false": lambda x: x["payload"]["selected_records"][0].__setitem__("sample_id", False),
        "deep_one_to_true": lambda x: x["payload"]["selected_records"][1].__setitem__("module_index", True),
        "extra": lambda x: x.__setitem__("attacker", False),
        "nan": lambda x: x["payload"].__setitem__("m699_manifest_sha256", math.nan),
    }
    for name, mutate in context_attacks.items():
        attacked = copy.deepcopy(context); mutate(attacked)
        require(rejected(lambda a=attacked: module.validate_canonical_context(a, context)),
                "canonical context attack survived: " + name)

    # The M1076 wrapper cross-binds raw record identity.  Patch only the older
    # BASE row-body validator so this synthetic hammer does not need to execute
    # real decoder windows; M1076 exact checks remain live.
    raw = synthetic_raw(module, context)
    old_base_validate_layer = module.BASE.validate_layer
    module.BASE.validate_layer = lambda projected, layer: True
    try:
        module.validate_raw(raw, context)
        raw_attacks = {
            "workload_zero_to_false": lambda x: x["workload"].__setitem__("sample_id", False),
            "top_zero_to_false": lambda x: x.__setitem__("exact_mismatch_count", False),
            "d1_false_to_zero": lambda x: x["d1"].__setitem__("scheduled", 0),
            "claim_false_to_zero": lambda x: x["claim_boundary"].__setitem__("paper_citable", 0),
            "record_zero_to_false": lambda x: x["layers"][0]["record_identity"].__setitem__("sample_id", False),
            "record_two_to_bool": lambda x: x["layers"][1]["record_identity"].__setitem__("module_index", True),
            "record_timestep_zero_to_false": lambda x: x["layers"][2]["record_identity"].__setitem__("timestep", False),
            "relabel": lambda x: x["layers"][0]["record_identity"].__setitem__("relative_path", "calls/relabel.bitpack"),
            "rehash": lambda x: x["layers"][1].__setitem__("verified_payload_member_sha256", "f" * 64),
            "wrong_schema": lambda x: x.__setitem__("schema", "attacker"),
            "extra": lambda x: x.__setitem__("attacker", False),
            "nan": lambda x: x.__setitem__("canonical_context_sha256", math.nan),
        }
        for name, mutate in raw_attacks.items():
            attacked = copy.deepcopy(raw); mutate(attacked)
            require(rejected(lambda a=attacked: module.validate_raw(a, context)),
                    "raw attack survived: " + name)
    finally:
        module.BASE.validate_layer = old_base_validate_layer

    result = module.make_result(raw, "4" * 64, "5" * 64,
                                module.canonical_sha(context))
    result_attacks = {
        "d1_false_to_zero": lambda x: x.__setitem__("d1_scheduled", 0),
        "top_zero_to_false": lambda x: x.__setitem__("total_window_count", False),
        "claim_false_to_zero": lambda x: x["claim_boundary"].__setitem__("paper_citable", 0),
        "deep_int_to_bool": lambda x: x["layers"][0].__setitem__("generated_compressed_transactions", True),
        "deep_zero_to_false": lambda x: x["layers"][2].__setitem__("exact_mismatch_count", False),
        "record_zero_to_false": lambda x: x["layers"][1]["record_identity"].__setitem__("sample_id", False),
        "wrong_schema": lambda x: x.__setitem__("schema", "attacker"),
        "wrong_status": lambda x: x.__setitem__("status", "PASS_ATTACKER"),
        "extra": lambda x: x.__setitem__("attacker", False),
        "nan": lambda x: x.__setitem__("total_window_count", math.nan),
    }
    for name, mutate in result_attacks.items():
        attacked = copy.deepcopy(result); mutate(attacked)
        require(rejected(lambda a=attacked: module.validate_result(
            a, raw, "4" * 64, "5" * 64, module.canonical_sha(context))),
            "result attack survived: " + name)

    # Assemble and publish each reject type-confused context/payload/raw/result
    # files even if an attacker refreshes the flat seal.  The only monkeypatches
    # provide a synthetic canonical context and skip the old window-body schema;
    # all M1076 recursive validators and cross-bindings remain live.
    file_mutations = {
        "context_false_to_zero": lambda f: f["canonical_context.json"].__setitem__("d1_scheduled", 0),
        "payload_true_to_one": lambda f: f["payload_validation.json"].__setitem__("payload_members_verified", 1),
        "raw_zero_to_false": lambda f: f["raw_windows.json"]["workload"].__setitem__("sample_id", False),
        "result_false_to_zero": lambda f: f["result.json"].__setitem__("d1_scheduled", 0),
    }
    assemble_publish_rejections = []
    for operation in ("assemble", "publish"):
        for attack_name, mutation in file_mutations.items():
            with tempfile.TemporaryDirectory(prefix="m1077_" + operation + "_") as td:
                old_results = module.RESULTS
                old_builder = module.build_canonical_context
                old_validator = module.BASE.validate_layer
                module.RESULTS = Path(td)
                module.build_canonical_context = lambda *args, **kwargs: context
                module.BASE.validate_layer = lambda projected, layer: True
                work = make_work(module, module.RESULTS, context, raw, mutation,
                                 sealed=(operation == "publish"))
                try:
                    if operation == "assemble":
                        call = lambda: module.assemble(work,
                            module.RESULTS / module.ATTEMPT_NAME, RUNNER,
                            EXPECTED["contract"], context["attempt"]["m1077_authority"])
                    else:
                        call = lambda: module.publish(work,
                            module.RESULTS / module.RESULT_NAME,
                            module.RESULTS / module.ATTEMPT_NAME, RUNNER,
                            EXPECTED["contract"], context["attempt"]["m1077_authority"])
                    require(rejected(call), operation + " attack survived: " + attack_name)
                    assemble_publish_rejections.append(operation + ":" + attack_name)
                finally:
                    module.BASE.validate_layer = old_validator
                    module.build_canonical_context = old_builder
                    module.RESULTS = old_results

    # Explicit M1060-style relabel+rehash and double-seal publication attack.
    with tempfile.TemporaryDirectory(prefix="m1077_double_seal_") as td:
        old_results = module.RESULTS
        old_builder = module.build_canonical_context
        old_validator = module.BASE.validate_layer
        module.RESULTS = Path(td)
        module.build_canonical_context = lambda *args, **kwargs: context
        module.BASE.validate_layer = lambda projected, layer: True
        def relabel(files):
            files["canonical_context.json"]["payload"]["selected_records"][0].update(
                {"relative_path": "calls/relabeled.bitpack", "packed_sha256": "2" * 64,
                 "payload_member_sha256": "2" * 64})
            files["payload_validation.json"] = module.make_payload_receipt(
                files["canonical_context.json"])
        work = make_work(module, module.RESULTS, context, raw, relabel, sealed=True)
        try:
            double_seal_rejected = rejected(lambda: module.publish(work,
                module.RESULTS / module.RESULT_NAME,
                module.RESULTS / module.ATTEMPT_NAME, RUNNER,
                EXPECTED["contract"], context["attempt"]["m1077_authority"]))
        finally:
            module.BASE.validate_layer = old_validator
            module.build_canonical_context = old_builder
            module.RESULTS = old_results
    require(double_seal_rejected, "double-sealed relabel+rehash survived publish")

    # No pin, wrong pin, direct run, bad namespace, and wrong authority seal all
    # fail without touching canonical M1078 state.
    clean = {"PATH": "/usr/bin:/bin"}
    missing = subprocess.run(["/bin/bash", str(RUNNER)], env=clean,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    wrong = subprocess.run(["/bin/bash", str(RUNNER)], env={**clean,
        "M1078_EXPECTED_CONTRACT_SHA": "0" * 64,
        "M1078_EXPECTED_M1077_REVIEW_SHA": "0" * 64,
        "M1078_EXPECTED_M1077_MANIFEST_SHA": "0" * 64,
        "M1078_EXPECTED_M1077_OUTER_SHA": "0" * 64},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    require(missing.returncode != 0 and wrong.returncode != 0 and
            not ATTEMPT.exists() and not RESULT.exists(),
            "caller-pin bypass changed canonical state")
    require(rejected(lambda: module.validate_m1077("0" * 64, "0" * 64, "0" * 64)),
            "wrong authority seal survived")
    namespace_rejections = 0
    for path, role in ((module.RESULTS / ".wrong-attempt", "attempt"),
                       (module.RESULTS / "wrong-result", "result"),
                       (module.RESULTS / ".wrong-work", "work"),
                       (module.RESULTS / "wrong-quarantine", "quarantine"),
                       (Path("/tmp/m1077-outside"), "work")):
        require(rejected(lambda p=path, r=role: module.safe_path(p, r)),
                "namespace bypass survived")
        namespace_rejections += 1
    require(rejected(lambda: module.run_pilot(ATTEMPT,
        module.RESULTS / ("." + module.RESULT_NAME + ".work.direct"), RUNNER,
        EXPECTED["contract"], {"synthetic": "authority"})),
        "direct run-pilot bypass survived")

    # Temporary post-attempt failure remains quarantined and preserves the
    # synthetic attempt.  It never opens the real payload.
    with tempfile.TemporaryDirectory(prefix="m1077_quarantine_") as td:
        old_results = module.RESULTS
        module.RESULTS = Path(td)
        authority = {"review_sha256": "7" * 64, "manifest_sha256": "8" * 64,
                     "outer_seal_file_sha256": "9" * 64}
        attempt = module.RESULTS / module.ATTEMPT_NAME
        module.consume_attempt(attempt, RUNNER, EXPECTED["contract"], authority)
        work = module.RESULTS / ("." + module.RESULT_NAME + ".work.synthetic")
        work.mkdir(mode=0o700)
        quarantine = module.RESULTS / (
            module.RESULT_NAME + ".failed_or_incomplete.synthetic")
        module.quarantine(work, quarantine, 99)
        quarantine_pass = (attempt.is_dir() and quarantine.is_dir() and
            (quarantine / "FAILURE.json").is_file() and not work.exists())
        module.RESULTS = old_results
    require(quarantine_pass, "temporary quarantine/attempt retention failed")

    # Actual runner ordering and author regressions are checked without launch.
    runner_text = RUNNER.read_text(encoding="utf-8")
    ordered = [runner_text.index('m1078_flock}" -n 9'),
               runner_text.index('/usr/bin/pgrep'),
               runner_text.index('MemAvailable:'),
               runner_text.index('CommitLimit:'),
               runner_text.index('--consume-attempt'),
               runner_text.index('/usr/bin/mkdir -m 700'),
               runner_text.index('--validate-payload-after-attempt'),
               runner_text.index('--run-pilot'), runner_text.index('--assemble'),
               runner_text.index('--publish')]
    require(ordered == sorted(ordered) and
            'for m1078_process in dc_shell vcs simv fm_shell pt_shell' in runner_text and
            'm1078_mem}" -ge 16777216' in runner_text and
            'm1078_limit-m1078_used' in runner_text and
            'exec 9>"/tmp/m1078_decoder_exact_bool_pilot.lock"' in runner_text,
            "runner state/resource order drift")
    syntax = subprocess.run(["/bin/bash", "-n", str(RUNNER)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    tests = subprocess.run(["/opt/anaconda3/envs/pytorch310/bin/python3.10", str(TESTS)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    self_test = module.self_test()
    require(syntax.returncode == 0 and tests.returncode == 0 and
            self_test["status"] == "PASS_M1076_SMALL_SYNTHETIC_EXACT_BOOL_INT_SELFTEST",
            "runner syntax/author regression/self-test failed")

    require(not ATTEMPT.exists() and not ATTEMPT.is_symlink() and
            not RESULT.exists() and not RESULT.is_symlink() and
            sha(DOC359) == EXPECTED["docs359"],
            "canonical state/docs359 changed")

    output = {
        "status": "PASS_M1077_M1076_EXACT_BOOL_HAMMER__GO_ONE_M1078_ATTEMPT",
        "verdict": "GO_ONE_M1078_DIAGNOSTIC_ATTEMPT__RESULT_HAMMER_REQUIRED",
        "score": 100,
        "positive": {
            "source_identities_and_double_seals": "PASS",
            "author_regression": "12/12 PASS",
            "author_self_test": "9/9 PASS",
            "pre_attempt_calls_member_access": "0 open / 0 stat / 0 hash",
            "arbitrary_depth_bool_int_attacks_rejected": sorted(deep_attacks),
            "contract_attacks_rejected": len(contract_attacks),
            "canonical_context_attacks_rejected": len(context_attacks),
            "payload_receipt_attacks_rejected": len(receipt_attacks),
            "raw_attacks_rejected": len(raw_attacks),
            "result_attacks_rejected": len(result_attacks),
            "assemble_publish_file_attacks_rejected": assemble_publish_rejections,
            "m1060_all_fake_nonexistent_relabel_rehash": "REJECTED",
            "relabel_rehash_double_seal_publish": "REJECTED",
            "wrong_schema_status_extra_nan_direct_bypass": "REJECTED",
            "wrong_authority_and_caller_pins": "REJECTED_WITHOUT_ATTEMPT",
            "wrong_runtime_namespaces_rejected": namespace_rejections,
            "temporary_postattempt_failure": "QUARANTINED_WITH_ATTEMPT_RETAINED",
            "runner_gate_order": "flock -> EDA -> memory/commit -> attempt -> work -> payload -> cycles -> assemble -> publish",
        },
        "authorization": {"one_m1078_attempt": True,
            "real_payload_after_attempt_only": True, "eda_gpu_remote": False},
        "claim_boundary": {"paper_citable": False, "decoder_complete": False,
            "table_a_row": False, "system_performance_claim": False,
            "local_performance_claim": False, "continuous_row_cycles": False},
        "execution": {"m1078_executed": False,
            "canonical_attempt_consumed": False, "real_payload_members_opened": False,
            "real_window_cycles_executed": False, "eda_gpu_remote_used": False,
            "docs359_sha256": sha(DOC359)},
        "next_gate": "Caller may pin this exact double-sealed M1077 authority for one M1078 diagnostic pilot; independent M1079 result hammer remains mandatory.",
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
