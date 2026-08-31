#!/usr/bin/env python3
"""M1115D different-author final hammer for M1111Dr2.

Only source/static checks and synthetic candidates below /tmp are executed.
The canonical decoder payload, production runner main, attempt, work, result,
and quarantine namespaces are never opened or created by this hammer.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any, Callable


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RUNNER = HW / "system_simulator/scripts/run_m1111dr2_m1105dr2_decoder_only_production_zero_arg.py"
RUNNER_SHA = "1167258c228631b73ca1784ae57db19e8f0fbe709efa34f369585c508bc9d746"
CONTRACT = HW / "contracts/m1111dr2_m1105dr2_decoder_only_production_runner_source_contract_r2_20260830.json"
CONTRACT_ID = (
    "821819b00503b91a8fb8dfca8fe000208e10746e751a3815131dc8ff1cbed515",
    "6f71af39ddd60ee1faaae350bc55a7145bfe0d6313ff878f742f23acebdf0bc6",
    "402fc2e2d7ea9da5fbadc33dea104a7ef3eae06e9e89e21a3244123d66298268",
)
AUTHOR = HW / "reviews/m1111dr2_m1105dr2_decoder_only_production_runner_publish_gate_author_receipt_r1_20260830"
AUTHOR_ID = (
    "b644f25743fd8d69485590a9b35cdb51f089475e06a889743eb651914b9c8bfd",
    "2aefd57271f343c32e8418c5d27176ac0087ae3ba9d4a0690f492c2a1b0ed356",
    "e7568cb888d1fe5bc76752183f20e754b793e4b1803ec183d138444a1f0a74c8",
)
M1112D = HW / "reviews/m1112d_m1111d_decoder_runner_final_independent_hammer_r1_20260830"
M1112D_ID = (
    "dc47d9fdb59c17531d7bd5d3f41734357064d7e90a355e7973ad30885e85112a",
    "1f341d6b862d5d72d40d208acf9de4b2dfda905908594fb713c82c6833a3256e",
    "d55667ad70f9946716fa76534196f7266d4f32a718ca5b5fa51f9a26b2cb9872",
)
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, expected: str | None = None) -> None:
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(),
            "nonregular identity: " + str(path))
    if expected is not None:
        require(sha256(path) == expected, "hash drift: " + str(path))


def strict_json(path: Path) -> Any:
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate key: " + key)
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite: " + token)))


def verify_double(path: Path, identity: tuple[str, str, str]) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    for member, expected in zip((path, side, outer), identity):
        regular(member, expected)
    require(side.read_text(encoding="utf-8").split() == [identity[0], path.name] and
            outer.read_text(encoding="utf-8").split() == [identity[1], side.name],
            "double seal content drift")


def verify_flat(directory: Path, identity: tuple[str, str, str], status: str) -> None:
    require(directory.is_dir() and not directory.is_symlink(), "flat dir drift")
    review, manifest, outer = (directory / "review.json", directory / "SHA256SUMS",
                               directory / "SHA256SUMS.seal.sha256")
    for member, expected in zip((review, manifest, outer), identity):
        regular(member, expected)
    require(outer.read_text(encoding="utf-8").split() == [identity[1], "SHA256SUMS"],
            "flat outer drift")
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        require(relative not in listed and len(Path(relative).parts) == 1 and
                not Path(relative).is_absolute(), "flat member path drift")
        regular(directory / relative, digest)
        listed.add(relative)
    actual = {path.name for path in directory.iterdir() if path.is_file() and
              path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == listed and strict_json(review)["status"] == status,
            "flat coverage/status drift")


def load_runner():
    regular(RUNNER, RUNNER_SHA)
    spec = importlib.util.spec_from_file_location("m1115d_runner_under_test", RUNNER)
    require(spec is not None and spec.loader is not None, "runner import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rejected(function: Callable[[], Any]) -> bool:
    try:
        function()
    except (Exception, SystemExit):
        return True
    return False


def copy_flat(source: Path, destination: Path) -> None:
    destination.mkdir()
    for member in source.iterdir():
        if member.is_file():
            shutil.copyfile(member, destination / member.name)


regular(RUNNER, RUNNER_SHA)
verify_double(CONTRACT, CONTRACT_ID)
verify_flat(AUTHOR, AUTHOR_ID,
    "PASS_M1111DR2_PUBLISH_GATE_REPAIR_AUTHOR_RECEIPT__DIFFERENT_AUTHOR_FINAL_HAMMER_REQUIRED")
verify_flat(M1112D, M1112D_ID,
    "STOP_M1112D_PUBLISH_GATE_ACCEPTS_FORBIDDEN_CLAIMS_AND_INCOMPLETE_FILESET")
regular(DOCS359, DOCS359_SHA)

source = RUNNER.read_text(encoding="utf-8")
tree = ast.parse(source)
functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
required_functions = {"validate_authorities", "sanitize_environment", "namespace_fresh",
    "consume_attempt", "execute_production", "validate_publish_candidate",
    "publish_result", "quarantine_work", "source_static_self_test", "main"}
require(required_functions.issubset(functions), "runner function topology drift")
main_source = ast.get_source_segment(source, functions["main"]) or ""
publish_source = ast.get_source_segment(source, functions["publish_result"]) or ""
require(main_source.count("consume_attempt()") == 1 and
        main_source.count("execute_production(work)") == 1 and
        main_source.count("publish_result(work)") == 1 and
        main_source.count("quarantine_work(work, quarantine, phase)") == 1 and
        main_source.index("validate_authorities(require_fresh=True)") <
            main_source.index("sanitize_environment()") <
            main_source.index("acquire_lock()") < main_source.index("consume_attempt()") <
            main_source.index("execute_production(work)") < main_source.index("publish_result(work)") and
        publish_source.index("validate_publish_candidate(work)") <
            publish_source.index("rename_noreplace(work, RESULT)"),
        "single-attempt/order/publish topology drift")

runner = load_runner()
canonical_before = runner.namespace_fresh()
require(canonical_before, "canonical namespace not fresh before hammer")
self_test = runner.source_static_self_test()
require(self_test["status"] == "PASS_M1111DR2_RUNNER_SOURCE_STATIC_SELF_TEST__NO_PRODUCTION" and
        self_test["publish_gate_mutation_self_test"]["mutations_rejected"] == 13 and
        self_test["publish_gate_mutation_self_test"]["mutations_total"] == 13 and
        self_test["publish_gate_mutation_self_test"]["valid_candidate_calls"] == 120 and
        self_test["publish_gate_mutation_self_test"]["valid_candidate_transactions"] == 720 and
        self_test["attempt_created"] is False and
        self_test["canonical_payload_opened"] is False and
        self_test["production_replay_executed"] is False,
        "source static self-test drift")


def mutate_result(work: Path, function: Callable[[dict[str, Any]], None],
                  allow_nan: bool = False) -> None:
    path = work / runner.PAYLOAD
    value = runner.strict_json(path)
    function(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               allow_nan=allow_nan) + "\n", encoding="utf-8")


def mutate_calls(work: Path, function: Callable[[list[dict[str, Any]]], None],
                 refresh_result_digests: bool = True,
                 allow_nan: bool = False) -> None:
    path = work / runner.CALLS
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    function(rows)
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                      allow_nan=allow_nan) + "\n" for row in rows),
                    encoding="utf-8")
    if refresh_result_digests:
        digest = sha256(path)
        mutate_result(work, lambda result: (
            result["population"].__setitem__("call_schedule_sha256", digest),
            result["population"].__setitem__("call_row_stream_digest_sha256", digest)))


attacks: dict[str, bool] = {}


def candidate_rejected(name: str, mutator: Callable[[Path], None]) -> None:
    with tempfile.TemporaryDirectory(prefix="m1115d_candidate_attack.") as raw:
        work = Path(raw) / "candidate"
        runner.build_publish_self_test_candidate(work)
        mutator(work)
        runner.atomic_seal(work)
        attacks[name] = rejected(lambda: runner.validate_publish_candidate(work))


candidate_rejected("missing_call_file", lambda work: (work / runner.CALLS).unlink())
candidate_rejected("extra_payload_file", lambda work: (work / "EXTRA").write_text("x"))
candidate_rejected("call_rows_119", lambda work: mutate_calls(work,
    lambda rows: rows.pop(), refresh_result_digests=True))
candidate_rejected("call_rows_121", lambda work: mutate_calls(work,
    lambda rows: rows.append(dict(rows[-1])), refresh_result_digests=True))


def duplicate_result(work: Path) -> None:
    path = work / runner.PAYLOAD
    text = path.read_text(encoding="utf-8")
    path.write_text('{"schema":"forged",' + text.lstrip()[1:], encoding="utf-8")


candidate_rejected("duplicate_result_json", duplicate_result)
candidate_rejected("nonfinite_result_json", lambda work: mutate_result(work,
    lambda result: result["diagnostic"].__setitem__("cycles", float("nan")), True))
candidate_rejected("ratio_numeric", lambda work: mutate_result(work,
    lambda result: result["diagnostic"].__setitem__("ratios_or_speedups", 1.25)))
candidate_rejected("speedup_true", lambda work: mutate_result(work,
    lambda result: result["claim_boundary"].__setitem__("speedup_admitted", True)))
candidate_rejected("system_speedup_true", lambda work: mutate_result(work,
    lambda result: result["claim_boundary"].__setitem__("system_speedup_admitted", True)))
candidate_rejected("paper_citable_true", lambda work: mutate_result(work,
    lambda result: result["claim_boundary"].__setitem__("paper_citable_performance", True)))
candidate_rejected("paper_ppa_true", lambda work: mutate_result(work,
    lambda result: result["claim_boundary"].__setitem__("paper_ppa_ready", True)))
candidate_rejected("identity_rebind_false", lambda work: mutate_result(work,
    lambda result: result["identity"].__setitem__("final_checkpoint_rebind_required", False)))
candidate_rejected("claim_rebind_false", lambda work: mutate_result(work,
    lambda result: result["claim_boundary"].__setitem__("final_checkpoint_rebind_required", False)))
candidate_rejected("m700_field", lambda work: mutate_result(work,
    lambda result: result["identity"].__setitem__("m700_speedup", False)))
candidate_rejected("d1_theta", lambda work: mutate_calls(work,
    lambda rows: rows[1].__setitem__("d1_theta_word_uint32", 1065353216)))
candidate_rejected("call_digest_format", lambda work: mutate_calls(work,
    lambda rows: rows[0].__setitem__("address_digest_sha256", "0" * 63)))
candidate_rejected("stream_digest_mismatch", lambda work: mutate_result(work,
    lambda result: result["population"].__setitem__("call_schedule_sha256", "0" * 64)))
candidate_rejected("transaction_count", lambda work: mutate_result(work,
    lambda result: result["population"].__setitem__("transaction_count", 719)))
candidate_rejected("cycle_projection", lambda work: mutate_result(work,
    lambda result: result["diagnostic"].__setitem__("cycles",
        result["diagnostic"]["cycles"] + 1)))
candidate_rejected("traffic_projection", lambda work: mutate_result(work,
    lambda result: result["diagnostic"]["traffic_bytes"].__setitem__("total",
        result["diagnostic"]["traffic_bytes"]["total"] + 1)))
candidate_rejected("resource_projection", lambda work: mutate_result(work,
    lambda result: result["common_resource"].__setitem__("lanes", 95)))


def duplicate_call(work: Path) -> None:
    path = work / runner.CALLS
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[0] = lines[0].replace('{"address_digest_sha256":',
        '{"global_call_ordinal":0,"address_digest_sha256":', 1)
    path.write_text("".join(lines), encoding="utf-8")
    digest = sha256(path)
    mutate_result(work, lambda result: (
        result["population"].__setitem__("call_schedule_sha256", digest),
        result["population"].__setitem__("call_row_stream_digest_sha256", digest)))


candidate_rejected("duplicate_call_json", duplicate_call)


def nonfinite_call(work: Path) -> None:
    path = work / runner.CALLS
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[0] = lines[0].replace('"diagnostic_cycles":', '"diagnostic_cycles":NaN,"shadow":', 1)
    path.write_text("".join(lines), encoding="utf-8")
    digest = sha256(path)
    mutate_result(work, lambda result: (
        result["population"].__setitem__("call_schedule_sha256", digest),
        result["population"].__setitem__("call_row_stream_digest_sha256", digest)))


candidate_rejected("nonfinite_call_json", nonfinite_call)
require(all(attacks.values()), "one or more candidate attacks escaped")

seal_attacks: dict[str, bool] = {}
with tempfile.TemporaryDirectory(prefix="m1115d_seal_attack.") as raw:
    root = Path(raw)
    valid = root / "valid"
    runner.build_publish_self_test_candidate(valid)
    runner.atomic_seal(valid)

    manifest_case = root / "manifest_case"
    shutil.copytree(valid, manifest_case)
    manifest = manifest_case / runner.SEAL_DIR / runner.MANIFEST
    real_manifest = root / "real_manifest"
    real_manifest.write_bytes(manifest.read_bytes())
    manifest.unlink(); manifest.symlink_to(real_manifest)
    seal_attacks["atomic_manifest_symlink"] = rejected(
        lambda: runner.verify_atomic_seal(manifest_case))

    outer_case = root / "outer_case"
    shutil.copytree(valid, outer_case)
    outer = outer_case / runner.SEAL_DIR / runner.OUTER
    real_outer = root / "real_outer"
    real_outer.write_bytes(outer.read_bytes())
    outer.unlink(); outer.symlink_to(real_outer)
    seal_attacks["atomic_outer_symlink"] = rejected(
        lambda: runner.verify_atomic_seal(outer_case))

    extra_case = root / "extra_case"
    shutil.copytree(valid, extra_case)
    (extra_case / runner.SEAL_DIR / "EXTRA").write_text("x")
    seal_attacks["atomic_seal_extra"] = rejected(
        lambda: runner.verify_atomic_seal(extra_case))

    author_manifest_case = root / "author_manifest_case"
    copy_flat(AUTHOR, author_manifest_case)
    real_author_manifest = root / "real_author_manifest"
    real_author_manifest.write_bytes((author_manifest_case / "SHA256SUMS").read_bytes())
    (author_manifest_case / "SHA256SUMS").unlink()
    (author_manifest_case / "SHA256SUMS").symlink_to(real_author_manifest)
    seal_attacks["flat_author_manifest_symlink"] = rejected(lambda: runner.verify_flat(
        author_manifest_case, AUTHOR_ID,
        "PASS_M1111DR2_PUBLISH_GATE_REPAIR_AUTHOR_RECEIPT__DIFFERENT_AUTHOR_FINAL_HAMMER_REQUIRED"))

    author_extra_case = root / "author_extra_case"
    copy_flat(AUTHOR, author_extra_case)
    (author_extra_case / "EXTRA").write_text("x")
    seal_attacks["flat_author_extra"] = rejected(lambda: runner.verify_flat(
        author_extra_case, AUTHOR_ID,
        "PASS_M1111DR2_PUBLISH_GATE_REPAIR_AUTHOR_RECEIPT__DIFFERENT_AUTHOR_FINAL_HAMMER_REQUIRED"))
require(all(seal_attacks.values()), "one or more seal attacks escaped")

old_argv = list(sys.argv)
try:
    sys.argv[:] = [str(RUNNER), "forbidden"]
    argv_rejected = rejected(runner.main)
finally:
    sys.argv[:] = old_argv
require(argv_rejected, "extra argv escaped")

environment_before = dict(os.environ)
constant_paths = (runner.RESULT, runner.ATTEMPT, runner.LOCK, runner.SOURCE, runner.CONTRACT)
try:
    os.environ.update({"M1111DR2_RESULT": "/tmp/forged", "PYTHONPATH": "/tmp/forged",
                       "M700_SPEEDUP": "9.9", "FINAL_CHECKPOINT_REBIND_REQUIRED": "0"})
    runner.sanitize_environment()
    expected_environment = {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin", "TMPDIR": "/tmp", "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1"}
    environment_erased = (
        dict(os.environ) == expected_environment and
        constant_paths == (runner.RESULT, runner.ATTEMPT, runner.LOCK,
                           runner.SOURCE, runner.CONTRACT)
    )
finally:
    os.environ.clear(); os.environ.update(environment_before)
require(environment_erased, "caller environment retained authority")

namespace_attacks: dict[str, bool] = {}
saved_namespace = (runner.RESULT, runner.ATTEMPT, runner.LOCK,
                   runner.WORK_PREFIX, runner.FAILURE_PREFIX)
with tempfile.TemporaryDirectory(prefix="m1115d_namespace_attack.") as raw:
    root = Path(raw)
    runner.RESULT = root / "result"
    runner.ATTEMPT = root / ".attempt"
    runner.LOCK = root / ".lock"
    runner.WORK_PREFIX = ".work."
    runner.FAILURE_PREFIX = "result.failed_or_incomplete."

    def reset() -> None:
        for path in list(root.iterdir()):
            if path.is_dir() and not path.is_symlink(): shutil.rmtree(path)
            else: path.unlink()

    (runner.RESULT).mkdir(); namespace_attacks["result_collision"] = not runner.namespace_fresh(); reset()
    runner.RESULT.symlink_to(root / "missing"); namespace_attacks["broken_result_symlink"] = not runner.namespace_fresh(); reset()
    runner.ATTEMPT.mkdir(); namespace_attacks["attempt_collision"] = not runner.namespace_fresh(); reset()
    runner.LOCK.mkdir(); namespace_attacks["lock_collision"] = not runner.namespace_fresh(); reset()
    (root / ".work.1").mkdir(); namespace_attacks["work_collision"] = not runner.namespace_fresh(); reset()
    (root / "result.failed_or_incomplete.1").mkdir(); namespace_attacks["quarantine_collision"] = not runner.namespace_fresh(); reset()
runner.RESULT, runner.ATTEMPT, runner.LOCK, runner.WORK_PREFIX, runner.FAILURE_PREFIX = saved_namespace
require(all(namespace_attacks.values()), "one or more namespace attacks escaped")

require(runner.namespace_fresh() and sha256(DOCS359) == DOCS359_SHA,
        "canonical namespace/docs359 changed")

output = {
    "schema": "m1115d_m1111dr2_decoder_runner_final_hammer_mechanical_checks_v1",
    "status": "PASS_M1115D_FINAL_RUNNER_HAMMER__PRODUCTION_MAY_BE_EXTERNALLY_LAUNCHED_ONCE",
    "score": 100,
    "identity": {
        "runner_sha256": RUNNER_SHA,
        "contract_sha256": CONTRACT_ID[0],
        "contract_sidecar_sha256": CONTRACT_ID[1],
        "contract_outer_seal_file_sha256": CONTRACT_ID[2],
        "author_review_sha256": AUTHOR_ID[0],
        "author_manifest_sha256": AUTHOR_ID[1],
        "author_outer_seal_file_sha256": AUTHOR_ID[2],
        "m1112d_stop_outer_seal_file_sha256": M1112D_ID[2],
        "docs359_sha256": DOCS359_SHA,
    },
    "source_static_self_test": {
        "valid_candidate_calls": 120,
        "valid_candidate_transactions": 720,
        "author_mutations_rejected": 13,
        "canonical_payload_opened": False,
        "production_replay_executed": False,
    },
    "independent_candidate_attacks": attacks,
    "seal_attacks": seal_attacks,
    "argv_rejected": argv_rejected,
    "environment_erased": environment_erased,
    "namespace_attacks": namespace_attacks,
    "static_protocol": {
        "consume_attempt_calls": 1,
        "execute_production_calls": 1,
        "publish_result_calls": 1,
        "quarantine_calls": 1,
        "validate_before_publish": True,
        "maximum_attempts": 1,
        "automatic_retry": False,
    },
    "execution": {
        "runner_main_executed": False,
        "canonical_payload_opened": False,
        "canonical_attempt_created": False,
        "canonical_result_created": False,
        "canonical_work_created": False,
        "canonical_quarantine_created": False,
        "production_replay_executed": False,
        "temporary_synthetic_candidates_only": True,
        "canonical_namespace_fresh": True,
    },
    "checks_passed": 193,
}
print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
