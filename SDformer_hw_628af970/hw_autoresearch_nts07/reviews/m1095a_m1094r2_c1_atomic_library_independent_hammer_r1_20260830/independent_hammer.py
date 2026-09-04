#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Receipt-blind M1095a hammer of the non-launch M1094r2 atomic library."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
ENGINE = HW / "system_simulator/scripts/execute_m1094_m1086_c1_zero_work_exact_1rw_full_replay_one_shot.py"
STUB = HW / "system_simulator/scripts/run_m1094_m1086_c1_zero_work_exact_1rw_full_replay_one_shot.sh"
TESTS = HW / "system_simulator/tests/test_m1094_m1086_c1_zero_work_exact_1rw_full_replay_one_shot_source.py"
CONTRACT = HW / "contracts/m1094r2_m1087r3_m1086r2_c1_zero_work_full_replay_atomic_library_source_contract_r1_20260830.json"
RECEIPT = HW / "reviews/m1094r2_m1087r3_m1086r2_c1_atomic_library_source_receipt_r1_20260830"
M1087R3 = HW / "reviews/m1087r3_m1086r2_c1_zero_work_population_source_hammer_r1_20260830"
DOCS = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "python": "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    "engine": "c8808c0d4cf37a8f279afa128e089c08af3718606061658db8f2047c198c824a",
    "stub": "745b6e112d0e33457a64a6411b1563afa58418a3f9c175039ebc9ecadb902e10",
    "tests": "3622d77898675567580dd585947e732a04e98b1305c3f45937df0247c84c6073",
    "contract": "5278c5fa03a74cf9e3364325865b1bd52a5f75f372de15d5172b0b38bda64be4",
    "contract_side": "963315ed0cd04080eeeb7271dab2da0fa808891919d6aa119f4ed89d4b44fffa",
    "contract_outer": "c35cdf984fb51c584c9ca99f5ff7a638884eb7db3aabab994a62ddc0221b4c5f",
    "receipt_outer": "3bbeb9624b064021298c7f9d4e4cb2b91777dc9b274326570ec54671f7b7336b",
    "m1087r3_outer": "c8901ff70a8a22fa171f0fc47ae6ea40ee91c3af793c9dc5ca09670113369ae5",
    "docs": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def req(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_flat(directory: Path, expected_outer: str) -> dict[str, str]:
    req(directory.is_dir() and not directory.is_symlink(), "sealed dir absent/symlink")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    seen = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        member = directory / relative
        req(relative not in seen and member.is_file() and not member.is_symlink() and
            sha(member) == expected, "sealed member drift")
        seen.add(relative)
    req(outer.read_text(encoding="utf-8").split() == [sha(manifest), "SHA256SUMS"] and
        sha(outer) == expected_outer, "outer seal drift")
    return {"manifest_sha256": sha(manifest), "outer_seal_file_sha256": sha(outer)}


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    req(spec is not None and spec.loader is not None, "cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def static_audit(module) -> dict:
    for path, key in ((PYTHON, "python"), (ENGINE, "engine"), (STUB, "stub"),
                      (TESTS, "tests"), (CONTRACT, "contract"), (DOCS, "docs")):
        req(path.is_file() and not path.is_symlink() and sha(path) == EXPECTED[key],
            key + " identity drift")
    side = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    req(sha(side) == EXPECTED["contract_side"] and
        sha(outer) == EXPECTED["contract_outer"] and
        side.read_text(encoding="utf-8").split() == [EXPECTED["contract"], CONTRACT.name] and
        outer.read_text(encoding="utf-8").split() == [EXPECTED["contract_side"], side.name],
        "contract double seal drift")
    receipt_id = verify_flat(RECEIPT, EXPECTED["receipt_outer"])
    hammer_id = verify_flat(M1087R3, EXPECTED["m1087r3_outer"])
    contract = module.strict_json(CONTRACT)
    req(contract["status"] ==
        "PASS_M1094R2_ATOMIC_LIBRARY_SOURCE_CONTRACT__NO_EXECUTABLE_LAUNCH" and
        contract["launch_now"] is False and contract["max_attempts_now"] == 0 and
        contract["claim_boundary"]["executable_launch_present"] is False,
        "contract launch boundary drift")
    population = contract["canonical_population"]
    req(population["tasks"] == module.TASKS == 812160 and
        population["designs"] == list(module.DESIGNS) ==
        ["candidate", "strongest_zero", "same_coordinate_bit"] and
        population["design_count"] == len(module.DESIGNS) == 3 and
        population["task_design_work_values"] == module.VALUES == 2436480 and
        population["required_preflight_values_checked"] == 2436480,
        "population drift")
    main_source = textwrap.dedent(inspect.getsource(module.main))
    for forbidden in ("consume_attempt(", "execute_full(", "publish_result(",
                      "quarantine_work(", "validate-authority", "expected-m1095",
                      "EXPECTED_M1095"):
        req(forbidden not in main_source, "mutating/caller-authority CLI surface")
    req("os.environ" not in ENGINE.read_text(encoding="utf-8") and
        "getenv(" not in ENGINE.read_text(encoding="utf-8"),
        "environment authority surface")
    execute = textwrap.dedent(inspect.getsource(module.execute_full))
    tree = ast.parse(execute)
    calls = [(node.func.attr, node.lineno, node) for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
             node.func.attr in {"canonical_work_domain_preflight",
                                "iter_canonical_full_replay_results"}]
    calls.sort(key=lambda item: item[1])
    req([item[0] for item in calls] ==
        ["canonical_work_domain_preflight", "iter_canonical_full_replay_results"] and
        all(not item[2].args and not item[2].keywords for item in calls),
        "production call count/order/arguments drift")
    return {"receipt": receipt_id, "m1087r3": hammer_id,
            "tasks": 812160, "design_count": 3, "values": 2436480,
            "main_mutating_calls": 0, "caller_authority_cli_or_env": False,
            "preflight_calls": 1, "iterator_calls": 1,
            "preflight_before_iterator": True}


def run_tests() -> dict:
    tests = load(TESTS, "m1095a_pinned_m1094r2_tests")
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (HERE / "pinned_tests.stdout.txt").write_text(stream.getvalue(), encoding="utf-8")
    req(result.wasSuccessful() and result.testsRun == 11,
        "M1094r2 tests failed: " + stream.getvalue())
    return {"tests": 11, "failures": 0, "errors": 0}


def cli_attacks(module) -> dict:
    before = (module.ATTEMPT.exists(), module.RESULT.exists(),
              tuple(module.RESULT.parent.glob(module.WORK_PREFIX + "*")),
              tuple(module.RESULT.parent.glob(module.FAILURE_PREFIX + "*")))
    base_env = {"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1", "PYTHONPATH": ""}
    attacks = []
    for argv in ([], ["--consume-attempt"], ["--execute-full"],
                 ["--validate-authority", "x"], ["--expected-m1095", "0" * 64],
                 ["--self-test", "--authority", "x"]):
        result = subprocess.run([str(PYTHON), "-I", str(ENGINE), *argv],
                                text=True, capture_output=True, env=base_env,
                                check=False)
        req(result.returncode != 0, "forbidden CLI accepted: " + repr(argv))
        attacks.append({"argv": argv, "returncode": result.returncode})
    # --runner exists only to bind the read-only validate-source mode.  In the
    # self-test mode it is ignored and cannot select authority or mutate state.
    ignored_runner = subprocess.run(
        [str(PYTHON), "-I", str(ENGINE), "--self-test", "--runner", "/tmp/forged"],
        text=True, capture_output=True, env=base_env, check=False)
    req(ignored_runner.returncode == 0 and
        "PASS_M1094R2_SOURCE_SELF_TEST__NO_ATTEMPT_NO_PAYLOAD" in ignored_runner.stdout,
        "ignored read-only runner coordinate changed self-test")
    env = dict(base_env)
    env.update({"M1095_REVIEW_SHA256": "1" * 64,
                "M1095_OUTER_SHA256": "2" * 64,
                "M1094_EXPECTED_CYCLES": "1"})
    result = subprocess.run([str(PYTHON), "-I", str(ENGINE), "--self-test"],
                            text=True, capture_output=True, env=env, check=False)
    req(result.returncode == 0 and
        "PASS_M1094R2_SOURCE_SELF_TEST__NO_ATTEMPT_NO_PAYLOAD" in result.stdout,
        "environment changed read-only self-test")
    stub = subprocess.run([str(STUB)], text=True, capture_output=True,
                          env=base_env, check=False)
    stub_arg = subprocess.run([str(STUB), "x"], text=True, capture_output=True,
                              env=base_env, check=False)
    req(stub.returncode == 86 and stub_arg.returncode == 2 and
        "DIFFERENT_AUTHOR_M1095_HARDCODED_WRAPPER_REQUIRED" in stub.stderr,
        "non-launch stub mutated/launchable")
    after = (module.ATTEMPT.exists(), module.RESULT.exists(),
             tuple(module.RESULT.parent.glob(module.WORK_PREFIX + "*")),
             tuple(module.RESULT.parent.glob(module.FAILURE_PREFIX + "*")))
    req(after == before, "CLI/stub polluted runtime namespace")
    # Re-derive atomic semantics independently in temporary namespaces.
    authority = {
        "status": "PASS_DIFFERENT_AUTHOR_HARDCODED_LAUNCH_AUTHORITY",
        "m1095_review_sha256": "1" * 64,
        "m1095_manifest_sha256": "2" * 64,
        "m1095_outer_seal_file_sha256": "3" * 64,
        "m1095_launch_wrapper_sha256": "4" * 64,
        "m1094_engine_sha256": EXPECTED["engine"],
        "m1094_contract_sha256": EXPECTED["contract"],
        "m1086_source_sha256": module.M1086_SHA,
        "m1087r3_outer_seal_file_sha256": EXPECTED["m1087r3_outer"],
    }
    with tempfile.TemporaryDirectory(prefix="m1095a_atomic_") as raw:
        root = Path(raw)
        first = module.consume_attempt(authority, root)
        req(first["receipt"]["canonical_payload_opened_or_hashed_before_attempt"] is False and
            first["receipt"]["maximum_attempts"] == 1,
            "attempt semantic drift")
        try:
            module.consume_attempt(authority, root)
        except RuntimeError:
            pass
        else:
            raise RuntimeError("duplicate attempt accepted")
        work = root / (module.WORK_PREFIX + "bounded"); work.mkdir()
        (work / "partial").write_text("partial", encoding="utf-8")
        quarantine = root / (module.FAILURE_PREFIX + "bounded")
        value = module.quarantine_work(work, quarantine, 17, "BOUNDED", root)
        req(value["status"] == "PASS_M1094_SEALED_FAILURE_QUARANTINE" and
            module.verify_atomic_seal(quarantine)["members"] >= 2,
            "quarantine semantic drift")
    return {"forbidden_cli_attacks": attacks,
            "authority_environment_ignored": True,
            "stub_no_arg_exit": 86, "stub_arg_exit": 2,
            "self_test_runner_coordinate_ignored_read_only": True,
            "attempt_before_payload": True, "duplicate_attempt_rejected": True,
            "recursive_quarantine": True, "production_namespace_unchanged": True}


def seal(directory: Path) -> tuple[str, str]:
    members = sorted(path for path in directory.rglob("*")
                     if path.is_file() and path.name not in
                     {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    manifest = directory / "SHA256SUMS"
    manifest.write_text("".join(
        f"{sha(path)}  {path.relative_to(directory).as_posix()}\n" for path in members),
        encoding="utf-8")
    outer = directory / "SHA256SUMS.seal.sha256"
    outer.write_text(f"{sha(manifest)}  SHA256SUMS\n", encoding="utf-8")
    return sha(manifest), sha(outer)


def publish(static: dict, tests: dict, attacks: dict) -> None:
    review = {
        "schema": "m1095a_m1094r2_c1_atomic_library_independent_hammer_review_v1",
        "status": "PASS_M1095A_M1094R2_C1_ATOMIC_LIBRARY_INDEPENDENT_HAMMER",
        "verdict": "GO_M1095_ZERO_ARGUMENT_HARDCODED_LAUNCH_WRAPPER_SOURCE_AUTHORING_ONLY",
        "score": 100, "p0_count": 0, "p1_count": 0,
        "receipt_blind": True,
        "identity": {"m1094r2_engine_sha256": EXPECTED["engine"],
                     "m1094r2_non_launch_stub_sha256": EXPECTED["stub"],
                     "m1094r2_tests_sha256": EXPECTED["tests"],
                     "m1094r2_contract_sha256": EXPECTED["contract"],
                     "m1094r2_contract_outer_seal_file_sha256": EXPECTED["contract_outer"],
                     "m1094r2_receipt_outer_seal_file_sha256": EXPECTED["receipt_outer"],
                     "m1087r3_outer_seal_file_sha256": EXPECTED["m1087r3_outer"],
                     "python_sha256": EXPECTED["python"],
                     "docs359_sha256": EXPECTED["docs"]},
        "static_audit": static, "bounded_tests": tests, "cli_and_atomic_attacks": attacks,
        "authority_boundary": {
            "caller_selectable_cli_or_environment_authority": False,
            "library_functions_are_not_an_executable_trust_boundary": True,
            "future_wrapper_must_construct_authority_from_hardcoded_frozen_identities": True,
            "future_m1096_independent_launch_hammer_required": True,
        },
        "claim_boundary": {"m1095_source_authoring_authorized": True,
                           "m1095_execution_authorized": False,
                           "attempt_consumed": False, "preflight_executed": False,
                           "full_replay_executed": False,
                           "matched_cycles_admitted": False,
                           "speedup_admitted": False, "rtl_cycles": False,
                           "paper_ppa_ready": False},
    }
    (HERE / "review.json").write_text(json.dumps(review, sort_keys=True, indent=2) + "\n",
                                      encoding="utf-8")
    (HERE / "mechanical_checks.json").write_text(json.dumps(
        {"status": "PASS", "static": static, "tests": tests, "attacks": attacks},
        sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (HERE / "review.md").write_text(
        "# M1095a independent M1094r2 atomic-library hammer\n\n"
        "**GO 仅限编写 M1095 零参数、硬编码 authority launcher 源码；不授权执行。**\n\n"
        "11/11 tests 与独立 CLI/env/stub/attempt/seal/quarantine 攻击通过。M1094r2 只是原子库，"
        "CLI 无 mutating/authority 入口，non-launch stub 稳定 exit 86。未访问 canonical payload，"
        "未消费生产 attempt。\n", encoding="utf-8")
    (HERE / "RUN_COMPLETE.txt").write_text(
        "PASS_M1095A_M1094R2_C1_ATOMIC_LIBRARY_INDEPENDENT_HAMMER\n",
        encoding="utf-8")
    manifest, outer = seal(HERE)
    print("M1095A_REVIEW_SHA=" + sha(HERE / "review.json"))
    print("M1095A_MANIFEST_SHA=" + manifest)
    print("M1095A_OUTER_SHA=" + outer)


def main() -> None:
    module = load(ENGINE, "m1095a_target_m1094r2")
    static = static_audit(module)
    tests = run_tests()
    attacks = cli_attacks(module)
    req(sha(DOCS) == EXPECTED["docs"], "docs359 changed")
    publish(static, tests, attacks)


if __name__ == "__main__":
    main()
