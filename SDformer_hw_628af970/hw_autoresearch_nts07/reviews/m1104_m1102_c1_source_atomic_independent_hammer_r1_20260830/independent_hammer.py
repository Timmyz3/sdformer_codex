#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1104 independent read-only/mutation hammer for M1102 C1 sources.

The only filesystem mutations occur below a TemporaryDirectory and in this
review directory after a PASS/STOP verdict is authored externally.  This
program never creates the production launcher, attempt, work, result, lock or
quarantine and never calls the full-cycle iterator.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any


HW = Path(__file__).resolve().parents[2]
SOURCE = HW / "system_simulator/scripts/run_m1102_c1_work8_exact_1rw_source.py"
SOURCE_SHA = "95bd50aebcc473ab69cdea6ccf27d54743c89926c5e0f31199dc469ced9bf7cc"
ATOMIC = HW / "system_simulator/scripts/execute_m1102_c1_work8_exact_1rw_full_replay_atomic.py"
ATOMIC_SHA = "0325a4c901e945656ad6d74b12cae6b066f5b75bb426326143f8b0a8f24d1157"
CONTRACT = HW / "contracts/m1102_c1_legal_work8_exact_1rw_additive_source_contract_r1_20260830.json"
CONTRACT_SHA = "fad9c381fc1e55fc78d6cf4b95ad0959b5a7089989a7acce1ccfafa73714db6e"
CONTRACT_SIDE_SHA = "e6754574c804a7ed2cfd39e5a99c991db38402389901fef570359decf43e3607"
CONTRACT_OUTER_SHA = "b17774b1b3fad06f104081b2ab2b0de4b3b539c72fd9e6adcb2171a46d55770c"
RECEIPT = HW / "reviews/m1102_c1_legal_work8_exact_1rw_additive_source_receipt_r1_20260830"
RECEIPT_OUTER_SHA = "326cc8ba37dd839a8447d89cdbb7156b623207bf6405ae57a0954c71a8db6377"
M1100 = HW / "reviews/m1100_m1095_c1_failed_preflight_audit_r1_20260830"
M1100_OUTER_SHA = "867102e3529a8c4bc10b4ad3fe2336e4ddfcc6350cdcc3d38fdb783c7dc71376"
M1101 = HW / "reviews/m1101_c1_short_work_semantics_first_principles_review_r1_20260830"
M1101_OUTER_SHA = "d9f95f7c9b3fb15bef9f369c365603dd7060529b08b4bab5f0626f06d5bb7539"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
ATTEMPT = HW / "results/.m1102_c1_work8_exact_1rw_full_replay_attempt_consumed"
RESULT = HW / "results/m1102_c1_work8_exact_1rw_full_replay_r1_20260830"
LOCK = HW / "results/.m1102_c1_work8_exact_1rw_full_replay.lock"
WORK_GLOB = ".m1102_c1_work8_exact_1rw_full_replay_work.*"
FAILURE_GLOB = RESULT.name + ".failed_or_incomplete.*"


checks: list[str] = []


def require(value: bool, label: str) -> None:
    if not value:
        raise RuntimeError(label)
    checks.append(label)


def reject(callable_value, label: str) -> None:
    try:
        callable_value()
    except (RuntimeError, TypeError, ValueError, SystemExit):
        checks.append(label)
    else:
        raise RuntimeError(label + " was accepted")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, expected: str, label: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(), label + " regular")
    require(sha(path) == expected, label + " sha")


def verify_flat(directory: Path, expected_outer: str, label: str) -> dict[str, Any]:
    require(directory.is_dir() and not directory.is_symlink(), label + " directory")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    regular(outer, expected_outer, label + " outer")
    require(outer.read_text(encoding="utf-8").split() ==
            [sha(manifest), "SHA256SUMS"], label + " outer content")
    seen: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        require(relative not in seen and not Path(relative).is_absolute() and
                ".." not in Path(relative).parts, label + " member path")
        regular(directory / relative, expected, label + " member " + relative)
        seen.add(relative)
    return json.loads((directory / "review.json").read_text(encoding="utf-8"))


def verify_double() -> None:
    side = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    regular(CONTRACT, CONTRACT_SHA, "contract")
    regular(side, CONTRACT_SIDE_SHA, "contract side")
    regular(outer, CONTRACT_OUTER_SHA, "contract outer")
    require(side.read_text(encoding="utf-8").split() ==
            [CONTRACT_SHA, CONTRACT.name], "contract side content")
    require(outer.read_text(encoding="utf-8").split() ==
            [CONTRACT_SIDE_SHA, side.name], "contract outer content")
    with tempfile.TemporaryDirectory(prefix="m1104_contract_mutation.") as raw:
        root = Path(raw)
        mutated = root / CONTRACT.name
        mutated.write_bytes(CONTRACT.read_bytes() + b"\n")
        require(sha(mutated) != CONTRACT_SHA, "contract byte mutation rejected")
        stale_side = root / side.name
        stale_side.write_bytes(side.read_bytes())
        require(stale_side.read_text().split()[0] != sha(mutated),
                "old contract side seal rejected")


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, name + " import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def production_namespaces_absent(label: str) -> None:
    require(not ATTEMPT.exists() and not ATTEMPT.is_symlink(), label + " attempt absent")
    require(not RESULT.exists() and not RESULT.is_symlink(), label + " result absent")
    require(not LOCK.exists() and not LOCK.is_symlink(), label + " lock absent")
    require(not any(RESULT.parent.glob(WORK_GLOB)), label + " work absent")
    require(not any(RESULT.parent.glob(FAILURE_GLOB)), label + " quarantine absent")


regular(SOURCE, SOURCE_SHA, "semantic source")
regular(ATOMIC, ATOMIC_SHA, "atomic library")
regular(DOCS359, DOCS359_SHA, "docs359")
verify_double()
receipt = verify_flat(RECEIPT, RECEIPT_OUTER_SHA, "M1102 receipt")
m1100 = verify_flat(M1100, M1100_OUTER_SHA, "M1100")
m1101 = verify_flat(M1101, M1101_OUTER_SHA, "M1101")
require(receipt["status"] ==
        "PASS_M1102_ADDITIVE_SOURCE_AUTHOR_RECEIPT__DIFFERENT_AUTHOR_HAMMER_REQUIRED",
        "M1102 receipt status")
require(m1100["status"] ==
        "PASS_M1100_M1095_FAILURE_AUDIT__LEGAL_WORK8_DOMAIN_REPAIR_ALLOWED__M1095_DO_NOT_RETRY",
        "M1100 status")
require(m1101["status"] ==
        "GO_UNIQUE_ADDITIVE_DOMAIN_REPAIR__STOP_15_BANK_REINTERPRETATION",
        "M1101 status")
production_namespaces_absent("before")

source = load(SOURCE, "m1104_m1102_semantics")
atomic = load(ATOMIC, "m1104_m1102_atomic")
require(sha(Path(source.M1056.__file__)) == source.M1056_SHA, "frozen M1056 identity")

# Exact short-work semantics and direct positive delegation count.
original_schedule = source.M1056.schedule_task
calls: list[int] = []


def counted_schedule(plan, work_start, state, config=source.M1056.ArbiterConfig()):
    calls.append(plan.work_cycles)
    return original_schedule(plan, work_start, state, config)


source.M1056.schedule_task = counted_schedule
try:
    sentinel = {(0, 7): 91, (3, 11): 44}
    before = dict(sentinel)
    zero_plan = source.M1056.TaskPlan(700, 146, 0, 15)
    zero = source.schedule_task(zero_plan, 500, sentinel)
    require(sentinel == before and zero.events == [] and zero.grants == {} and
            zero.nominal_work_end == zero.effective_work_end == 500 and calls == [],
            "work0 exact no-event no-state no-M1056-call")
    delegated: dict[int, int] = {}
    for work in (8, 9, 15, 16, 24):
        calls.clear()
        repaired_state: dict[tuple[int, int], int] = {}
        frozen_state: dict[tuple[int, int], int] = {}
        plan = source.M1056.TaskPlan(701 + work, 158, work, 16)
        repaired = source.schedule_task(plan, 700, repaired_state)
        repaired_calls = list(calls)
        calls.clear()
        frozen = original_schedule(plan, 700, frozen_state)
        require(repaired == frozen and repaired_state == frozen_state and
                repaired_calls == [work], "positive exact single frozen M1056 call w=" + str(work))
        delegated[work] = len(repaired.events)
finally:
    source.M1056.schedule_task = original_schedule

for value in (True, False, -1, *range(1, 8)):
    reject(lambda value=value: source.validate_work(value),
           "generic illegal rejected " + repr(value))
require(source.validate_work(8) == 8 and source.validate_work(16) == 16,
        "generic work8/work16 admitted")
for value in (9, 10, 11, 12, 13, 14, 15, 17, 23):
    require(source.validate_work(value) == value, "generic positive nonmod admitted " + str(value))
    reject(lambda value=value: source.validate_canonical_work(value),
           "canonical nonmod rejected " + str(value))
require(source.validate_canonical_work(0) == 0 and
        source.validate_canonical_work(8) == 8 and
        source.validate_canonical_work(16) == 16,
        "canonical 0/8/16 admitted")

# Independently rerun the complete read-only 812160x3/12522 gate.  This is not
# the production iterator and cannot create an attempt or cycle result.
preflight = source.canonical_work_domain_and_work8_preflight()
require(preflight["status"] ==
        "PASS_M1102_EXHAUSTIVE_812160X3_AND_12522_WORK8_REGRESSION" and
        preflight["tasks"] == 812160 and preflight["values_checked"] == 2436480 and
        preflight["work8_occurrences_total"] == 12522 and
        preflight["full_coverage_pass"] is True and
        preflight["production_full_cycle_iterator_called"] is False and
        preflight["attempt_created"] is False,
        "independent full-domain/work8 preflight")
atomic.validate_preflight(preflight)
checks.append("atomic accepts exact independent preflight")

mutations: list[tuple[str, Any]] = [
    ("values_checked", 2436479),
    ("tasks", 812159),
    ("work8_occurrences_total", 12521),
    ("task_design_work_digest_sha256", "0" * 64),
    ("row_work_execution_provenance_digest_sha256", "1" * 64),
    ("full_coverage_pass", False),
    ("production_full_cycle_iterator_called", True),
    ("attempt_created", True),
    ("cycles_or_speedup_admitted", True),
]
for key, value in mutations:
    changed = copy.deepcopy(preflight)
    changed[key] = value
    reject(lambda changed=changed: atomic.validate_preflight(changed),
           "preflight mutation rejected " + key)
changed = copy.deepcopy(preflight)
changed["counts"]["candidate"]["work8"] -= 1
reject(lambda: atomic.validate_preflight(changed), "count work8 mutation rejected")
changed = copy.deepcopy(preflight)
changed["work8_geometry"]["candidate"]["minimum_dependency_delay"] = -1
reject(lambda: atomic.validate_preflight(changed), "geometry delay mutation rejected")
changed = copy.deepcopy(preflight)
changed["work8_geometry"].pop("candidate")
reject(lambda: atomic.validate_preflight(changed), "geometry coverage mutation rejected")

# Caller API/environment attacks.  The library has no production CLI; future
# launcher and launch hammer must hard-code the two not-yet-existing hashes.
require(len(inspect.signature(source.canonical_work_domain_preflight).parameters) == 0 and
        inspect.isgeneratorfunction(source.iter_canonical_full_replay_results) and
        len(inspect.signature(source.iter_canonical_full_replay_results).parameters) == 0,
        "zero-argument production semantic interfaces")
reject(lambda: source.canonical_work_domain_preflight(8), "preflight caller arg rejected")
reject(lambda: source.iter_canonical_full_replay_results(8), "iterator caller arg rejected")
reject(lambda: atomic.main([]), "atomic no-mode CLI rejected")
reject(lambda: atomic.main(["--self-test", "--validate-source"]),
       "atomic multi-mode CLI rejected")

old_environment = dict(os.environ)
try:
    os.environ["M1102_EXPECTED_SOURCE_SHA256"] = "0" * 64
    os.environ["M1102_EXPECTED_CONTRACT_SHA256"] = "1" * 64
    os.environ["M1102_RESULT"] = "/tmp/forged-result"
    identity = atomic.validate_source_contract(require_fresh=True)
    require(identity["source_sha256"] == SOURCE_SHA and
            identity["contract_sha256"] == CONTRACT_SHA,
            "caller environment has no identity/path authority")
finally:
    os.environ.clear()
    os.environ.update(old_environment)

valid_authority = {
    "status": "PASS_DIFFERENT_AUTHOR_M1102_HARDCODED_LAUNCH_AUTHORITY",
    "launch_wrapper_sha256": "a" * 64,
    "launch_hammer_outer_seal_file_sha256": "b" * 64,
    "m1102_atomic_library_sha256": ATOMIC_SHA,
    "m1102_semantic_source_sha256": SOURCE_SHA,
    "m1102_contract_sha256": CONTRACT_SHA,
    "m1100_outer_seal_file_sha256": M1100_OUTER_SHA,
    "m1101_outer_seal_file_sha256": M1101_OUTER_SHA,
}
atomic._validate_launch_authority(valid_authority)
checks.append("future hardcoded authority shape accepted")
for label, mutation in (
    ("missing key", lambda value: value.pop("launch_wrapper_sha256")),
    ("extra key", lambda value: value.update({"caller_path": "/tmp/x"})),
    ("source hash", lambda value: value.update({"m1102_semantic_source_sha256": "0" * 64})),
    ("contract hash", lambda value: value.update({"m1102_contract_sha256": "0" * 64})),
    ("M1100 hash", lambda value: value.update({"m1100_outer_seal_file_sha256": "0" * 64})),
    ("uppercase", lambda value: value.update({"launch_wrapper_sha256": "A" * 64})),
):
    changed = dict(valid_authority)
    mutation(changed)
    reject(lambda changed=changed: atomic._validate_launch_authority(changed),
           "authority mutation rejected " + label)

# Atomic partial-output and seal attacks, isolated to temporary directories.
with tempfile.TemporaryDirectory(prefix="m1104_atomic_mutation.") as raw:
    root = Path(raw)
    partial = root / "partial_output"
    partial.mkdir()
    (partial / "partial.json").write_text('{"status":"PARTIAL"}\n', encoding="utf-8")
    seal = atomic.atomic_seal(partial)
    require(seal["members"] == 1 and atomic.verify_atomic_seal(partial) == seal,
            "partial output seal exact")
    (partial / "late.bin").write_bytes(b"late")
    reject(lambda: atomic.verify_atomic_seal(partial),
           "unmanifested partial member rejected")
    (partial / "late.bin").unlink()
    (partial / "partial.json").write_text('{"status":"MUTATED"}\n', encoding="utf-8")
    reject(lambda: atomic.verify_atomic_seal(partial), "sealed payload mutation rejected")
    symlink_dir = root / "symlink_output"
    symlink_dir.mkdir()
    target = root / "target.bin"
    target.write_bytes(b"target")
    (symlink_dir / "link.bin").symlink_to(target)
    reject(lambda: atomic.atomic_seal(symlink_dir), "partial symlink rejected")
    reject(lambda: atomic.normalize_raw({"status":
           "PASS_M1102_RAW_CPU_MODEL_FULL_REPLAY_PENDING_RESULT_HAMMER"}),
           "partial raw result rejected")

atomic_source = ATOMIC.read_text(encoding="utf-8")
atomic_tree = ast.parse(atomic_source)
require("execute_full(" not in ast.get_source_segment(
            atomic_source,
            next(node for node in atomic_tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "main")),
        "atomic CLI does not expose execute_full")
require("consume_attempt(" not in ast.get_source_segment(
            atomic_source,
            next(node for node in atomic_tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "main")),
        "atomic CLI does not expose consume_attempt")
production_namespaces_absent("after")
require(sha(DOCS359) == DOCS359_SHA, "docs359 unchanged after hammer")

print(json.dumps({
    "schema": "m1104_m1102_c1_source_atomic_independent_hammer_r1_v1",
    "status": "PASS_M1104_M1102_SOURCE_ATOMIC_HAMMER__AUTHOR_ZERO_ARG_LAUNCHER_ONLY",
    "checks_passed": len(checks),
    "identity": {
        "source_sha256": SOURCE_SHA,
        "atomic_library_sha256": ATOMIC_SHA,
        "contract_sha256": CONTRACT_SHA,
        "contract_outer_seal_file_sha256": CONTRACT_OUTER_SHA,
        "source_receipt_outer_seal_file_sha256": RECEIPT_OUTER_SHA,
        "m1100_outer_seal_file_sha256": M1100_OUTER_SHA,
        "m1101_outer_seal_file_sha256": M1101_OUTER_SHA,
        "docs359_sha256": DOCS359_SHA,
    },
    "independent_exhaustive": {
        "tasks": preflight["tasks"],
        "values_checked": preflight["values_checked"],
        "work8_occurrences_total": preflight["work8_occurrences_total"],
        "task_design_work_digest_sha256": preflight[
            "task_design_work_digest_sha256"],
        "row_work_execution_provenance_digest_sha256": preflight[
            "row_work_execution_provenance_digest_sha256"],
        "full_cycle_iterator_called": False,
    },
    "short_semantics": {
        "work0_state_unchanged": True,
        "work0_m1056_calls": 0,
        "positive_single_frozen_m1056_calls": sorted(delegated),
        "work_1_to_7_rejected": True,
        "generic_nonmod_positive_delegated": True,
        "canonical_nonmod_positive_rejected": True,
    },
    "atomic_attacks": {
        "preflight_mutations_rejected": len(mutations) + 3,
        "caller_authority_mutations_rejected": 6,
        "caller_environment_authority": False,
        "partial_unmanifested_member_rejected": True,
        "partial_payload_mutation_rejected": True,
        "partial_symlink_rejected": True,
    },
    "execution": {
        "launcher_created": False,
        "production_attempt_created": False,
        "full_cycle_replay_executed": False,
        "production_result_created": False,
        "eda_commands": 0,
    },
}, indent=2, sort_keys=True))
