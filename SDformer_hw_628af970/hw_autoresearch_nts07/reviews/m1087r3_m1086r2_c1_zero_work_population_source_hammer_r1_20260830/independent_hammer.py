#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Fresh receipt-blind M1087r3 hammer for the M1086r2 population repair.

Bounded only: this file must not call canonical_work_domain_preflight() or
iter_canonical_full_replay_results().
"""
from __future__ import annotations

from dataclasses import replace
import ast
import hashlib
import importlib.util
import inspect
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SRC = HW / "system_simulator/scripts/run_m1086_c1_zero_work_exact_1rw_source.py"
TESTS = HW / "system_simulator/tests/test_m1086_c1_zero_work_exact_1rw_source.py"
CONTRACT = HW / "contracts/m1086r2_c1_zero_work_exact_1rw_population_repair_contract_r1_20260830.json"
RECEIPT = HW / "reviews/m1086r2_c1_zero_work_exact_1rw_population_repair_source_receipt_r1_20260830"
STOP = HW / "reviews/m1087r2_m1087_m1086_population_supersession_hammer_r1_20260830"
OLD_GO = HW / "reviews/m1087_m1086_c1_zero_work_exact_1rw_source_hammer_r1_20260830"
M1085 = HW / "reviews/m1085_m1074_c1_full_replay_failure_audit_r1_20260830"
M1072 = HW / "system_simulator/scripts/run_m1072_c1_row_provenance_exact_1rw_source.py"
DOCS = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "src": "3925c97de922393786b4aa8ae6ca6b4942489e3cf10485f5d1b6cd423e797a51",
    "tests": "c7528af1a799f549d82193fbb2b297507da8c19daabd35487fd244158d20ae08",
    "contract": "351bbec8d7c4b538f035077f18f670ec6deccae4d4a995ec4ce250a6e960ed6f",
    "contract_outer": "a45bf483f0bf77a48ddce23e7d1d5e0194bc7c5ef2c3893afe29211577dc4243",
    "receipt_outer": "8447490431b8474d67e08539bd7cd52aefd4457e7c49262a92bbd7e1d6a5e837",
    "stop_outer": "2301881ad38e431bbc1b49f08ef05a2a1f8be977f94b90a47e86b4ec7160df36",
    "old_go_outer": "dfe945edfce00b3a8d2995279daa432bcefae3feda001ee265c5ff63f3219a55",
    "m1085_outer": "ea6a4f8853ccc534be36db355b7c2e57612b2dae8af4681b500134961d2ec2a9",
    "m1072": "879712a59785acc79776990236884582431adea81103a222d5415905199a1e4c",
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


def strict_json(path: Path):
    def pairs(items):
        out = {}
        for key, value in items:
            req(key not in out, "duplicate key: " + key)
            out[key] = value
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + token)))


def verify_flat(directory: Path, outer_sha: str) -> dict[str, str]:
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
    req(outer.read_text(encoding="utf-8").split() == [sha(manifest), "SHA256SUMS"] and
        sha(outer) == outer_sha, "outer seal drift")
    return {"manifest_sha256": sha(manifest), "outer_seal_file_sha256": sha(outer)}


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    req(spec is not None and spec.loader is not None, "cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def crosscheck(module) -> dict:
    for path, key in ((SRC, "src"), (TESTS, "tests"), (CONTRACT, "contract"),
                      (M1072, "m1072"), (DOCS, "docs")):
        req(path.is_file() and not path.is_symlink() and sha(path) == EXPECTED[key],
            key + " identity drift")
    side = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    req(side.read_text(encoding="utf-8").split() == [EXPECTED["contract"], CONTRACT.name],
        "contract sidecar drift")
    req(outer.read_text(encoding="utf-8").split() == [sha(side), side.name] and
        sha(outer) == EXPECTED["contract_outer"], "contract outer drift")
    identities = {
        "receipt": verify_flat(RECEIPT, EXPECTED["receipt_outer"]),
        "stop": verify_flat(STOP, EXPECTED["stop_outer"]),
        "old_go": verify_flat(OLD_GO, EXPECTED["old_go_outer"]),
        "m1085": verify_flat(M1085, EXPECTED["m1085_outer"]),
    }
    contract = strict_json(CONTRACT)
    receipt = strict_json(RECEIPT / "review.json")
    stop = strict_json(STOP / "review.json")
    old_go = strict_json(OLD_GO / "review.json")
    req(stop["status"] == "STOP_M1087R2_M1086_CONTRACT_POPULATION_MISMATCH" and
        stop["supersession"]["m1087_go_usable"] is False and
        stop["supersession"]["m1087_release_created"] is False,
        "M1087r2 supersession drift")
    req(old_go["verdict"].endswith("NO_EXECUTION"), "old GO identity drift")
    req(contract["status"] ==
        "SOURCE_CONTRACT_POPULATION_REPAIRED__NEW_INDEPENDENT_HAMMER_REQUIRED" and
        contract["launch_now"] is False and
        contract["supersession"]["old_contract_usable"] is False and
        contract["supersession"]["old_m1087_go_usable"] is False and
        contract["authorization"]["m1092_authoring"] is False and
        contract["authorization"]["full_replay"] is False,
        "repaired contract boundary drift")
    # The receipt is used as sealed identity/context, never as the source of the
    # verdict.  Its population fields must nevertheless agree mechanically.
    populations = [contract["canonical_population"], receipt["canonical_population"]]
    tasks = module.M1072.TASKS
    designs = list(module.DESIGNS)
    values = tasks * len(designs)
    req((tasks, designs, values) ==
        (812160, ["candidate", "strongest_zero", "same_coordinate_bit"], 2436480),
        "frozen source population drift")
    for population in populations:
        req(population["tasks"] == tasks and population["designs"] == designs and
            population["design_count"] == len(designs) and
            population["task_design_work_values"] == values,
            "contract/receipt population mismatch")
    req(contract["canonical_population"]["preflight_expected_values_checked"] == values and
        receipt["canonical_population"]["expected_values_checked"] == values and
        receipt["canonical_population"]["equation_pass"] is True,
        "expected values_checked mismatch")
    req(len(inspect.signature(module.canonical_work_domain_preflight).parameters) == 0 and
        inspect.isgeneratorfunction(module.iter_canonical_full_replay_results) and
        len(inspect.signature(module.iter_canonical_full_replay_results).parameters) == 0,
        "production caller surface drift")
    preflight = inspect.getsource(module.canonical_work_domain_preflight)
    req("for task_id in range(M1072.TASKS)" in preflight and
        "for name in DESIGNS" in preflight and
        "values_checked':M1072.TASKS*3" in preflight and
        "schedule_task(" not in preflight and "DesignStream(" not in preflight,
        "preflight source population/scope drift")
    tree = ast.parse(preflight)
    req(not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
                node.func.id == "iter_canonical_full_replay_results" for node in ast.walk(tree)),
        "preflight calls full replay")
    req(module.M1064.derive_physical_capacity()["derived_total_bytes"] == 214912,
        "capacity drift")
    return {
        "tasks": tasks,
        "designs": designs,
        "design_count": len(designs),
        "task_design_work_values": values,
        "equation": f"{tasks}*{len(designs)}={values}",
        "source_preflight_return_expression": "M1072.TASKS*3",
        "production_preflight_arguments": 0,
        "production_iterator_arguments": 0,
        "capacity_bytes": 214912,
        "old_go_superseded": True,
        "sealed_authorities": identities,
        "author_receipt_status_trusted_for_verdict": False,
    }


def run_tests() -> dict:
    tests = load(TESTS, "m1087r3_pinned_tests")
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (HERE / "pinned_tests.stdout.txt").write_text(stream.getvalue(), encoding="utf-8")
    req(result.wasSuccessful() and result.testsRun == 12,
        "bounded pinned tests failed: " + stream.getvalue())
    return {"tests": 12, "failures": 0, "errors": 0}


def rejected(callable_, label: str) -> None:
    try:
        callable_()
    except (RuntimeError, TypeError):
        return
    raise RuntimeError(label + " accepted")


def attacks(module) -> dict:
    zero = 0
    for task, preprocess, row, start in ((0, 0, 0, 0), (207, 146, 15, 500),
                                         (811999, 4096, 63, 99999)):
        state = {(0, row): 17, (7, 127): 91}; before = dict(state)
        result = module.schedule_task(
            module.M1056.TaskPlan(task, preprocess, 0, row), start, state)
        req(state == before and result.events == [] and result.grants == {} and
            result.queue_peak == result.nominal_excess_accesses ==
            result.delayed_accesses == result.maximum_read_write_lifetime == 0 and
            result.raw_dependencies_pass is True and
            result.nominal_work_end == result.effective_work_end == start,
            "zero-work invariant drift")
        zero += 1
    positive = 0
    for work in (8, 15, 16, 31, 224, 280, 4096):
        for row in (0, 16, 63):
            left = {(0, row): 8}; right = dict(left)
            plan = module.M1056.TaskPlan(208, 158, work, row)
            req(module.schedule_task(plan, 700, left) ==
                module.M1056.schedule_task(plan, 700, right) and left == right,
                "positive delegate drift")
            positive += 1
    work_rejections = []
    for value in (True, False, -1, *range(1, 15), 1.0, "15", None):
        rejected(lambda value=value: module.validate_production_work(value), "work")
        plan = module.M1056.TaskPlan(9, 7, value, 3)
        rejected(lambda plan=plan: module.DesignStream().consume_internal(plan),
                 "stream work")
        work_rejections.append(repr(value))
    dependency_rejections = []
    for dep in (module.M1056.Dependency("", 0), module.M1056.Dependency("r", True),
                module.M1056.Dependency("r", False), module.M1056.Dependency("r", -1)):
        event = module.M1056.PortEvent("w", 0, 1, 0, 0, 0, "WRITE", 0, (dep,))
        rejected(lambda event=event: module.validate_dependencies([event]), "dependency")
        dependency_rejections.append([dep.event_id, dep.delay_cycles])
    regression = module.real_task207_next_regression()
    req(regression["status"] == "PASS_M1086_REAL_TASK207_NEXT_RAW_REGRESSION" and
        regression["production_iterator_called"] is False,
        "task207/task208 regression drift")
    with module.M1072.CanonicalRowReader() as reader:
        real = reader.derive(0)
    forged = replace(real, shared_preprocess_cycles=0,
                     works={name: 0 for name in module.DESIGNS},
                     parents={name: {"reads": 0, "writes": 0, "forwards": 0,
                                            "work_cycles": 0}
                              for name in module.DESIGNS})
    forged = replace(forged, provenance_sha256=hashlib.sha256(
        module.M1072._canonical_provenance_payload(
            module.M1072.record_payload(forged))).hexdigest())
    rejected(lambda: module.M1072.validate_external_records_against_frozen([forged]),
             "row provenance forgery")
    for function in (module.canonical_work_domain_preflight,
                     module.iter_canonical_full_replay_results):
        for args in ((1,), ([],), ({"work": 0},)):
            rejected(lambda function=function, args=args: function(*args),
                     "production argument")
    with tempfile.TemporaryDirectory(prefix="m1087r3_authority_") as raw:
        copied = Path(raw) / "m1085"
        shutil.copytree(M1085, copied)
        review = copied / "review.json"
        review.write_text(review.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        rejected(lambda: module.verify_dir(copied, EXPECTED["m1085_outer"]),
                 "authority mutation")
        link = Path(raw) / "link"; link.symlink_to(M1085, target_is_directory=True)
        rejected(lambda: module.verify_dir(link, EXPECTED["m1085_outer"]),
                 "authority symlink")
    oracle = module.source_small_oracle()
    req(oracle["zero_events"] == 0 and oracle["zero_grants"] == 0 and
        oracle["frozen_cascade_20_to_22"] is True and
        oracle["full_replay_executed"] is False, "small oracle drift")
    return {
        "zero_work_cases": zero,
        "positive_delegate_cases": positive,
        "production_work_attacks_rejected": work_rejections,
        "dependency_attacks_rejected": dependency_rejections,
        "task207_task208_raw": regression,
        "row_provenance_forgery": "REJECTED_BY_CANONICAL_REDERIVATION",
        "production_argument_injection": "REJECTED",
        "authority_mutation_and_symlink": "REJECTED",
        "frozen_cascade": "PASS_20_TO_22",
        "exhaustive_preflight_executed": False,
        "full_replay_executed": False,
        "attempt_consumed": False,
    }


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


def publish(population: dict, tests: dict, dynamic: dict) -> None:
    review = {
        "schema": "m1087r3_m1086r2_c1_zero_work_population_source_hammer_review_v1",
        "status": "PASS_M1087R3_M1086R2_C1_ZERO_WORK_POPULATION_SOURCE_HAMMER",
        "verdict": "GO_M1092_ONE_SHOT_RUNNER_SOURCE_AUTHORING_ONLY__NO_EXECUTION",
        "score": 100,
        "p0_count": 0,
        "p1_count": 0,
        "receipt_blind": True,
        "identity": {
            "m1086_source_sha256": EXPECTED["src"],
            "m1086_tests_sha256": EXPECTED["tests"],
            "m1086r2_contract_sha256": EXPECTED["contract"],
            "m1086r2_contract_outer_seal_file_sha256": EXPECTED["contract_outer"],
            "m1086r2_receipt_outer_seal_file_sha256": EXPECTED["receipt_outer"],
            "m1087r2_stop_outer_seal_file_sha256": EXPECTED["stop_outer"],
            "superseded_old_m1087_go_outer_seal_file_sha256": EXPECTED["old_go_outer"],
            "m1085_outer_seal_file_sha256": EXPECTED["m1085_outer"],
            "m1072_source_sha256": EXPECTED["m1072"],
            "docs359_sha256": EXPECTED["docs"],
        },
        "population_crosscheck": population,
        "bounded_tests": tests,
        "dynamic_attacks": dynamic,
        "required_m1092_runner_hammer_gates": [
            "fresh M1092 source/result/attempt/lock/quarantine namespaces",
            "zero-argument runner with no caller metric/work/row/coverage injection",
            "atomically consume exactly one attempt before canonical payload access",
            "first attempted payload phase invokes zero-argument preflight and requires values_checked=2436480",
            "full replay begins only after preflight passes",
            "every failure is sealed into quarantine; no partial result and no automatic retry",
            "atomic no-replace publication",
            "independent source hammer before execution and result hammer after success",
        ],
        "claim_boundary": {
            "m1092_runner_source_authoring_authorized": True,
            "m1092_runner_execution_authorized": False,
            "exhaustive_preflight_authorized_now": False,
            "full_replay_authorized_now": False,
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
        {"status": "PASS", "population": population, "tests": tests,
         "dynamic": dynamic}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (HERE / "review.md").write_text(
        "# M1087r3 independent population/source hammer\n\n"
        "**GO 仅限 M1092 one-shot runner 源码编写；不授权执行。**\n\n"
        "独立交叉核对已确认 812,160 tasks × 3 designs = 2,436,480 work values，"
        "contract、receipt 和 source `M1072.TASKS*3` 一致。旧 M1087 GO 继续由 M1087r2 STOP 撤销。"
        "12/12 bounded tests 与 zero-work、positive delegate、dependency、task207→208 RAW、provenance、"
        "authority/caller attacks 通过。未执行 exhaustive preflight/full replay，未消费 attempt。\n",
        encoding="utf-8")
    (HERE / "RUN_COMPLETE.txt").write_text(
        "PASS_M1087R3_M1086R2_C1_ZERO_WORK_POPULATION_SOURCE_HAMMER\n",
        encoding="utf-8")
    manifest, outer = seal(HERE)
    print("M1087R3_REVIEW_SHA=" + sha(HERE / "review.json"))
    print("M1087R3_MANIFEST_SHA=" + manifest)
    print("M1087R3_OUTER_SHA=" + outer)


def main() -> None:
    req(not any((HW / "results").glob("*m1092*")), "M1092 namespace already exists")
    module = load(SRC, "m1087r3_target_m1086")
    population = crosscheck(module)
    tests = run_tests()
    dynamic = attacks(module)
    req(sha(DOCS) == EXPECTED["docs"], "docs359 changed during hammer")
    publish(population, tests, dynamic)


if __name__ == "__main__":
    main()
