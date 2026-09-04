#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Receipt-blind bounded hammer for M1086.

This hammer deliberately does not invoke either production entry point.  In
particular, it never runs the 812160-value work-domain preflight or the full
cycle replay.  It may open only the two canonical rows needed by the bounded
task-207/task-208 regression and a single row for a provenance-forgery check.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import replace
import hashlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HW / "system_simulator/scripts/run_m1086_c1_zero_work_exact_1rw_source.py"
TESTS = HW / "system_simulator/tests/test_m1086_c1_zero_work_exact_1rw_source.py"
CONTRACT = HW / "contracts/m1086_c1_zero_work_exact_1rw_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1086_c1_zero_work_exact_1rw_source_receipt_r1_20260830"
M1085 = HW / "reviews/m1085_m1074_c1_full_replay_failure_audit_r1_20260830"
M1072 = HW / "system_simulator/scripts/run_m1072_c1_row_provenance_exact_1rw_source.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "source": "3925c97de922393786b4aa8ae6ca6b4942489e3cf10485f5d1b6cd423e797a51",
    "tests": "c7528af1a799f549d82193fbb2b297507da8c19daabd35487fd244158d20ae08",
    "contract": "cd4a315d0f153925acee893fd24d9d2b227d45ef9e40e2534f76e35c8abfebe8",
    "contract_outer": "dcc3ab8f6271657fab93e27604465f544db963b9621c0e366f33cec9d687db5c",
    "author_outer": "c7a79b704160e3323c6c7ec70a0c019bf79270f6f26574a9f6231d9a01b33372",
    "m1085_outer": "ea6a4f8853ccc534be36db355b7c2e57612b2dae8af4681b500134961d2ec2a9",
    "m1072": "879712a59785acc79776990236884582431adea81103a222d5415905199a1e4c",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
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


def strict_json(path: Path):
    def pairs(items):
        result = {}
        for key, value in items:
            req(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            RuntimeError("nonfinite JSON: " + token)
        ),
    )


def verify_flat(directory: Path, expected_outer: str) -> dict[str, str]:
    req(directory.is_dir() and not directory.is_symlink(), "sealed dir absent/symlink")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    seen = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        req(name not in seen, "duplicate manifest member")
        seen.add(name)
        member = directory / name
        req(member.is_file() and not member.is_symlink() and sha(member) == expected,
            "sealed member drift: " + name)
    req(outer.read_text(encoding="utf-8").split() == [sha(manifest), "SHA256SUMS"],
        "inner seal drift")
    req(sha(outer) == expected_outer, "outer seal drift")
    return {"manifest_sha256": sha(manifest), "outer_seal_file_sha256": sha(outer)}


def load_source():
    spec = importlib.util.spec_from_file_location("m1087_target_m1086", SOURCE)
    req(spec is not None and spec.loader is not None, "cannot load M1086")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def static_audit(module) -> dict:
    for path, key in ((SOURCE, "source"), (TESTS, "tests"), (CONTRACT, "contract"),
                      (M1072, "m1072"), (DOCS359, "docs359")):
        req(path.is_file() and not path.is_symlink() and sha(path) == EXPECTED[key],
            key + " identity drift")
    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    req(sidecar.read_text(encoding="utf-8").split() ==
        [EXPECTED["contract"], CONTRACT.name], "contract sidecar drift")
    req(outer.read_text(encoding="utf-8").split() == [sha(sidecar), sidecar.name] and
        sha(outer) == EXPECTED["contract_outer"], "contract outer drift")
    author_id = verify_flat(AUTHOR, EXPECTED["author_outer"])
    m1085_id = verify_flat(M1085, EXPECTED["m1085_outer"])
    contract = strict_json(CONTRACT)
    req(contract["status"] == "SOURCE_ONLY__INDEPENDENT_M1087_HAMMER_REQUIRED" and
        contract["launch_now"] is False and
        contract["authorization"]["full_replay_attempt"] is False and
        contract["authorization"]["exhaustive_work_preflight"] is False,
        "M1086 source contract boundary drift")
    # Receipt is identity evidence only.  Its status is intentionally not used
    # to determine the verdict.
    author_review = strict_json(AUTHOR / "review.json")
    req(author_review["identity"]["driver_sha256"] == EXPECTED["source"] and
        author_review["identity"]["tests_sha256"] == EXPECTED["tests"],
        "author identity payload drift")
    req(len(inspect.signature(module.canonical_work_domain_preflight).parameters) == 0,
        "preflight acquired caller arguments")
    req(inspect.isgeneratorfunction(module.iter_canonical_full_replay_results) and
        len(inspect.signature(module.iter_canonical_full_replay_results).parameters) == 0,
        "full iterator acquired caller arguments")
    preflight_source = inspect.getsource(module.canonical_work_domain_preflight)
    req("schedule_task(" not in preflight_source and "DesignStream(" not in preflight_source and
        "cycles_after_commit" not in preflight_source,
        "preflight derives cycle/arbitration payload")
    full_source = inspect.getsource(module.iter_canonical_full_replay_results)
    for token in ("CanonicalRowReader()", "ProvenanceCoverage()", "DesignStream()",
                  "validate_production_work", "proof['full_coverage_pass']"):
        req(token in full_source, "full iterator missing frozen token: " + token)
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    req("canonical_work_domain_preflight" in functions and
        "iter_canonical_full_replay_results" in functions,
        "production functions absent")
    req(module.M1072.TASKS == 812160 and module.M1072.SAMPLES == 10 and
        list(module.DESIGNS) == ["candidate", "strongest_zero", "same_coordinate_bit"],
        "population/design boundary drift")
    req(module.M1064.derive_physical_capacity()["derived_total_bytes"] == 214912,
        "capacity drift")
    return {
        "exact_identities": 8,
        "contract_double_seal": "PASS",
        "author_receipt_identity_only_not_status": True,
        "author_receipt": author_id,
        "m1085_authority": m1085_id,
        "production_preflight_arguments": 0,
        "production_iterator_arguments": 0,
        "tasks": module.M1072.TASKS,
        "values_reserved_for_production_preflight": module.M1072.TASKS * 3,
        "samples": module.M1072.SAMPLES,
        "designs": list(module.DESIGNS),
        "capacity_bytes": 214912,
    }


def run_pinned_tests() -> dict:
    spec = importlib.util.spec_from_file_location("m1087_pinned_m1086_tests", TESTS)
    req(spec is not None and spec.loader is not None, "cannot load M1086 tests")
    test_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = test_module
    spec.loader.exec_module(test_module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_module)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (HERE / "pinned_tests.stdout.txt").write_text(stream.getvalue(), encoding="utf-8")
    req(result.wasSuccessful() and result.testsRun == 12,
        "pinned tests failed: " + stream.getvalue())
    return {"tests": result.testsRun, "failures": 0, "errors": 0}


def expect_runtime_error(callable_, label: str) -> None:
    try:
        callable_()
    except (RuntimeError, TypeError):
        return
    raise RuntimeError(label + " attack accepted")


def dynamic_audit(module) -> dict:
    out = {}
    # Zero work must be a literal no-op on the memory protocol and state.
    zero_cases = 0
    for task_id, preprocess, row, start in ((0, 0, 0, 0), (207, 146, 15, 500),
                                             (811999, 4096, 63, 99999)):
        state = {(0, row): 17, (7, 127): 91}
        before = dict(state)
        result = module.schedule_task(
            module.M1056.TaskPlan(task_id, preprocess, 0, row), start, state)
        req(state == before and result.events == [] and result.grants == {} and
            result.queue_peak == result.nominal_excess_accesses ==
            result.delayed_accesses == result.maximum_read_write_lifetime == 0 and
            result.raw_dependencies_pass is True and
            result.nominal_work_end == result.effective_work_end == start,
            "zero-work semantic invariant drift")
        zero_cases += 1
    out["zero_work_cases"] = zero_cases

    # Every positive helper call is exactly the frozen M1056 behavior, including
    # unsupported short values; production rejects those before reaching it.
    positive_cases = 0
    for work in (8, 15, 16, 31, 224, 280, 4096):
        for row in (0, 16, 63):
            left = {(0, row): 8}; right = dict(left)
            plan = module.M1056.TaskPlan(208, 158, work, row)
            got = module.schedule_task(plan, 700, left)
            expected = module.M1056.schedule_task(plan, 700, right)
            req(got == expected and left == right, "positive delegate drift")
            positive_cases += 1
    # The helper still delegates work=1 exactly: both paths raise the frozen
    # dependency error and leave identical state.  Production rejects it before
    # reaching this helper, which is checked below.
    left = {(0, 0): 8}; right = dict(left)
    plan = module.M1056.TaskPlan(208, 158, 1, 0)
    left_error = right_error = None
    try:
        module.schedule_task(plan, 700, left)
    except RuntimeError as error:
        left_error = str(error)
    try:
        module.M1056.schedule_task(plan, 700, right)
    except RuntimeError as error:
        right_error = str(error)
    req(left_error == right_error == "invalid event dependency" and left == right,
        "positive short-work exception delegate drift")
    positive_cases += 1
    out["positive_delegate_cases"] = positive_cases

    rejected_work = []
    for value in (True, False, -1, *range(1, 15), 1.0, "15", None):
        expect_runtime_error(lambda value=value: module.validate_production_work(value),
                             "work " + repr(value))
        plan = module.M1056.TaskPlan(9, 7, value, 3)
        expect_runtime_error(lambda plan=plan: module.DesignStream().consume_internal(plan),
                             "production stream work " + repr(value))
        rejected_work.append(repr(value))
    out["production_work_attacks_rejected"] = rejected_work

    valid = module.M1056.PortEvent("r", 0, 0, 0, 0, 0, "READ", 0)
    module.validate_dependencies([valid])
    dependency_attacks = []
    for dep in (module.M1056.Dependency("", 0), module.M1056.Dependency("r", True),
                module.M1056.Dependency("r", False), module.M1056.Dependency("r", -1)):
        event = module.M1056.PortEvent("w", 0, 1, 0, 0, 0, "WRITE", 0, (dep,))
        expect_runtime_error(lambda event=event: module.validate_dependencies([event]),
                             "dependency")
        dependency_attacks.append([dep.event_id, dep.delay_cycles])
    out["dependency_attacks_rejected"] = dependency_attacks

    # Bounded canonical check: exactly task 207 and 208, not a population scan.
    regression = module.real_task207_next_regression()
    req(regression["status"] == "PASS_M1086_REAL_TASK207_NEXT_RAW_REGRESSION" and
        regression["task207_coordinate"] == [0, 0, 0, 207] and
        regression["task208_coordinate"] == [0, 0, 0, 208] and
        regression["production_iterator_called"] is False,
        "task207/task208 regression drift")
    out["task207_task208_raw"] = regression

    # Caller-supplied provenance cannot drive production and is rejected even
    # after recomputing its self-digest.
    with module.M1072.CanonicalRowReader() as reader:
        real = reader.derive(0)
    forged = replace(real, shared_preprocess_cycles=0,
                     works={name: 0 for name in module.DESIGNS},
                     parents={name: {"reads": 0, "writes": 0, "forwards": 0,
                                            "work_cycles": 0}
                              for name in module.DESIGNS})
    forged = replace(
        forged,
        provenance_sha256=hashlib.sha256(
            module.M1072._canonical_provenance_payload(
                module.M1072.record_payload(forged))).hexdigest())
    expect_runtime_error(
        lambda: module.M1072.validate_external_records_against_frozen([forged]),
        "row provenance forgery")
    out["row_provenance_forgery"] = "REJECTED_BY_CANONICAL_REDERIVATION"

    for function, label in ((module.canonical_work_domain_preflight, "preflight"),
                            (module.iter_canonical_full_replay_results, "iterator")):
        for args in ((1,), ([],), ({"work": 0},)):
            expect_runtime_error(lambda function=function, args=args: function(*args),
                                 label + " caller injection")
    out["production_argument_injection"] = "REJECTED"

    # A mutated or symlinked authority cannot pass the source's authority gate.
    with tempfile.TemporaryDirectory(prefix="m1087_authority_attack_") as raw:
        copied = Path(raw) / "m1085"
        shutil.copytree(M1085, copied)
        review = copied / "review.json"
        review.write_text(review.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        expect_runtime_error(lambda: module.verify_dir(copied, EXPECTED["m1085_outer"]),
                             "mutated authority")
        link = Path(raw) / "m1085_link"
        link.symlink_to(M1085, target_is_directory=True)
        expect_runtime_error(lambda: module.verify_dir(link, EXPECTED["m1085_outer"]),
                             "symlink authority")
    out["authority_mutation_and_symlink"] = "REJECTED"

    oracle = module.source_small_oracle()
    req(oracle["zero_events"] == 0 and oracle["zero_grants"] == 0 and
        oracle["frozen_cascade_20_to_22"] is True and
        oracle["full_replay_executed"] is False,
        "source oracle drift")
    out["frozen_cascade"] = "PASS_20_TO_22"
    out["production_preflight_executed"] = False
    out["full_replay_executed"] = False
    out["attempt_consumed"] = False
    return out


def seal_directory(directory: Path) -> tuple[str, str]:
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


def publish(static: dict, tests: dict, dynamic: dict) -> None:
    review = {
        "schema": "m1087_m1086_c1_zero_work_exact_1rw_source_hammer_review_v1",
        "status": "PASS_M1087_M1086_C1_ZERO_WORK_EXACT_1RW_SOURCE_HAMMER",
        "verdict": "GO_AUTHOR_ONE_NEW_NAMESPACE_CPU_ONE_SHOT_WRAPPER_ONLY__NO_EXECUTION",
        "score": 100,
        "p0_count": 0,
        "p1_count": 0,
        "receipt_blind": True,
        "identity": {
            "m1086_source_sha256": EXPECTED["source"],
            "m1086_tests_sha256": EXPECTED["tests"],
            "m1086_contract_sha256": EXPECTED["contract"],
            "m1086_contract_outer_seal_file_sha256": EXPECTED["contract_outer"],
            "m1086_author_receipt_outer_seal_file_sha256": EXPECTED["author_outer"],
            "m1085_outer_seal_file_sha256": EXPECTED["m1085_outer"],
            "m1072_source_sha256": EXPECTED["m1072"],
            "docs359_sha256": EXPECTED["docs359"],
        },
        "static_audit": static,
        "pinned_tests": tests,
        "dynamic_attacks": dynamic,
        "deferred_runner_gates": {
            "reason": "M1086 intentionally contains no runner/attempt/quarantine implementation",
            "dynamically_verified_by_m1087": False,
            "required_before_execution": [
                "fresh source/result/attempt/lock namespace distinct from M1074/M1086/M1087",
                "zero-argument runner and no caller work/cycle/row/coverage injection",
                "atomically consume exactly one attempt before canonical payload access",
                "run zero-argument 812160-task x 3-design work preflight as first attempted payload phase",
                "quarantine every preflight/replay/seal/publish failure and forbid automatic retry",
                "publish no partial raw/result and use no-replace atomic final rename",
                "independent runner source hammer before execution",
                "independent result hammer after any successful execution",
            ],
        },
        "claim_boundary": {
            "source_semantics_admitted": True,
            "one_shot_wrapper_source_authoring_authorized": True,
            "exhaustive_preflight_authorized_now": False,
            "full_replay_execution_authorized_now": False,
            "attempt_consumed": False,
            "matched_cycles_admitted": False,
            "speedup_admitted": False,
            "rtl_cycles": False,
            "eda_gpu_remote": False,
            "paper_ppa_ready": False,
        },
    }
    (HERE / "review.json").write_text(json.dumps(review, sort_keys=True, indent=2) + "\n",
                                      encoding="utf-8")
    (HERE / "mechanical_checks.json").write_text(json.dumps(
        {"status": "PASS", "static": static, "tests": tests, "dynamic": dynamic},
        sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (HERE / "review.md").write_text(
        "# M1087 independent bounded source hammer\n\n"
        "**GO 仅限下一位作者建立新 namespace 的 CPU one-shot wrapper 源码；尚不授权执行。**\n\n"
        "M1086 zero-work 语义、正工作冻结 delegate、task207→208 RAW、工作域、依赖、行 provenance、"
        "authority 和 214912B capacity 均通过 receipt-blind bounded hammer。本 hammer 未运行 812160×3 preflight、"
        "full replay、EDA/GPU/remote，未消费 attempt。\n\n"
        "M1086 故意尚无 runner，因此 attempt-before-payload、fresh namespace、quarantine 并未被动态验证。"
        "一次性 release 只允许编写 wrapper；wrapper 必须另经独立 hammer 才可执行。\n",
        encoding="utf-8")
    (HERE / "RUN_COMPLETE.txt").write_text(
        "PASS_M1087_M1086_C1_ZERO_WORK_EXACT_1RW_SOURCE_HAMMER\n",
        encoding="utf-8")
    manifest_sha, outer_sha = seal_directory(HERE)
    print("M1087_REVIEW_SHA=" + sha(HERE / "review.json"))
    print("M1087_MANIFEST_SHA=" + manifest_sha)
    print("M1087_OUTER_SHA=" + outer_sha)


def main() -> None:
    req(not (HW / "results/.m1086_c1_zero_work_exact_1rw_attempt_consumed").exists(),
        "unexpected M1086 attempt namespace exists")
    module = load_source()
    static = static_audit(module)
    tests = run_pinned_tests()
    dynamic = dynamic_audit(module)
    req(sha(DOCS359) == EXPECTED["docs359"], "docs359 changed during hammer")
    publish(static, tests, dynamic)


if __name__ == "__main__":
    main()
