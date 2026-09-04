#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1102 additive legal-work8 repair over frozen M1086/M1056.

Generic scheduling admits exactly ``work == 0 or work >= 8``.  Zero work uses
the frozen M1086 event-free repair; every positive work delegates directly to
frozen M1056.  Canonical provenance is stricter: every work value must lie on
the frozen eight-output-block lattice (``work % 8 == 0``).

The only executable production-facing interfaces are zero-argument functions.
The CLI is source validation only.  ``--exhaustive-readonly`` performs one
full work/provenance scan plus bounded work=8 geometry checks, but never calls
the full cycle iterator and never creates an attempt or result namespace.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterator

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
M1086_PATH = HERE / "run_m1086_c1_zero_work_exact_1rw_source.py"
M1086_SHA = "3925c97de922393786b4aa8ae6ca6b4942489e3cf10485f5d1b6cd423e797a51"
M1056_SHA = "95e276a7afe7a049faa2b967bed1431762c72a5e0b521c3e9857121ece5c816f"
M1072_SHA = "879712a59785acc79776990236884582431adea81103a222d5415905199a1e4c"
M1100 = HW / "reviews/m1100_m1095_c1_failed_preflight_audit_r1_20260830"
M1100_ID = (
    "84094e424b92814e111bc732e39df5de852f81d0ff2823bc824e055fc9b122b1",
    "10c2ee1b782d27d5f1b9ba6a8fe446481594f564d45a6bab536808c8a96a0cda",
    "867102e3529a8c4bc10b4ad3fe2336e4ddfcc6350cdcc3d38fdb783c7dc71376",
)
M1101 = HW / "reviews/m1101_c1_short_work_semantics_first_principles_review_r1_20260830"
M1101_ID = (
    "4f927917f09faa43a2412298ec71f8b2d650b62d41bb1fdab3d544d2db324626",
    "ba47ef39e5175b0a8802706e6e9ef3d049ee1bece39ca9e970b6ce6272818ad4",
    "d9f95f7c9b3fb15bef9f369c365603dd7060529b08b4bab5f0626f06d5bb7539",
)
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
DESIGNS = ("candidate", "strongest_zero", "same_coordinate_bit")
TASKS = 812160
VALUES = TASKS * len(DESIGNS)
EXPECTED_WORK8_PER_DESIGN = 4174
EXPECTED_WORK8_TOTAL = 12522


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + token)))


def verify_flat(directory: Path, identity: tuple[str, str, str]) -> None:
    review, manifest, outer = (directory / "review.json",
                               directory / "SHA256SUMS",
                               directory / "SHA256SUMS.seal.sha256")
    require(directory.is_dir() and not directory.is_symlink() and
            (sha256(review), sha256(manifest), sha256(outer)) == identity,
            "sealed authority identity drift: " + directory.name)
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        member = directory / relative
        require(relative not in listed and member.is_file() and
                not member.is_symlink() and sha256(member) == expected,
                "sealed authority member drift: " + relative)
        listed.add(relative)
    expected, relative = outer.read_text(encoding="utf-8").split()
    require(relative == "SHA256SUMS" and expected == sha256(manifest),
            "sealed authority outer drift")


def load_m1086():
    require(M1086_PATH.is_file() and not M1086_PATH.is_symlink() and
            sha256(M1086_PATH) == M1086_SHA, "M1086 identity drift")
    spec = importlib.util.spec_from_file_location("m1102_frozen_m1086", M1086_PATH)
    require(spec is not None and spec.loader is not None, "cannot load M1086")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M1086 = load_m1086()
M1072 = M1086.M1072
M1056 = M1086.M1056
M1064 = M1086.M1064


def validate_frozen_authorities() -> dict[str, Any]:
    verify_flat(M1100, M1100_ID)
    verify_flat(M1101, M1101_ID)
    m1100 = strict_json(M1100 / "review.json")
    m1101 = strict_json(M1101 / "review.json")
    require(m1100.get("status") ==
            "PASS_M1100_M1095_FAILURE_AUDIT__LEGAL_WORK8_DOMAIN_REPAIR_ALLOWED__M1095_DO_NOT_RETRY" and
            m1100.get("minimum_additive_repair", {}).get("domain") ==
            "work == 0 or work >= 8" and
            m1101.get("status") ==
            "GO_UNIQUE_ADDITIVE_DOMAIN_REPAIR__STOP_15_BANK_REINTERPRETATION" and
            m1101.get("first_principles_findings", {}).get("logical_psum_banks") == 8 and
            sha256(M1072.__file__ and Path(M1072.__file__)) == M1072_SHA and
            sha256(M1056.__file__ and Path(M1056.__file__)) == M1056_SHA and
            sha256(DOCS359) == DOCS359_SHA,
            "M1100/M1101 frozen authority content drift")
    return {"status": "PASS_M1102_FROZEN_AUTHORITIES",
            "m1100_outer_seal_file_sha256": M1100_ID[2],
            "m1101_outer_seal_file_sha256": M1101_ID[2],
            "docs359_sha256": DOCS359_SHA}


def validate_work(value: Any) -> int:
    require(type(value) is int and value >= 0,
            "work must be exact nonnegative int")
    require(value == 0 or value >= 8,
            "unsupported positive work interval 1..7")
    return value


def validate_canonical_work(value: Any) -> int:
    value = validate_work(value)
    require(value % 8 == 0, "canonical work violates eight-block lattice")
    return value


def validate_dependencies(events: Any) -> None:
    require(type(events) is list and all(
        bool(dependency.event_id) and type(dependency.delay_cycles) is int and
        dependency.delay_cycles >= 0
        for event in events for dependency in event.dependencies),
        "dependency type/value drift")


def schedule_task(plan: Any, work_start: int,
                  last_write_cycle: dict[tuple[int, int], int],
                  config: Any = M1056.ArbiterConfig()) -> Any:
    require(type(plan) is M1056.TaskPlan and type(plan.work_cycles) is int,
            "exact M1056 plan/work type required")
    plan.validate()
    validate_work(plan.work_cycles)
    require(type(work_start) is int and work_start >= 0,
            "exact nonnegative work start required")
    if plan.work_cycles == 0:
        result = M1086.schedule_task(plan, work_start, last_write_cycle, config)
        require(result.events == [] and result.grants == {} and
                result.effective_work_end == work_start,
                "M1086 zero-work delegation drift")
        return result
    result = M1056.schedule_task(plan, work_start, last_write_cycle, config)
    validate_dependencies(result.events)
    return result


@dataclass
class DesignStream:
    last_write: dict[tuple[int, int], int] = field(default_factory=dict)
    previous_start: int | None = None
    previous_effective_end: int | None = None
    delayed_accesses: int = 0
    nominal_excess_accesses: int = 0

    def consume_internal(self, plan: Any) -> Any:
        validate_canonical_work(plan.work_cycles)
        if self.previous_start is None:
            start = plan.preprocess_cycles
        else:
            require(self.previous_effective_end is not None, "stream state drift")
            start = max(self.previous_effective_end,
                        self.previous_start + plan.preprocess_cycles) + 2
        result = schedule_task(plan, start, self.last_write)
        self.previous_start = start
        self.previous_effective_end = result.effective_work_end
        self.delayed_accesses += result.delayed_accesses
        self.nominal_excess_accesses += result.nominal_excess_accesses
        return result

    def finish_sample(self) -> dict[str, int]:
        require(self.previous_effective_end is not None, "empty design sample")
        return {
            "cycles_after_commit": self.previous_effective_end + 2 +
                M1064.COMMIT_CYCLES_PER_SAMPLE,
            "delayed_accesses": self.delayed_accesses,
            "nominal_excess_accesses": self.nominal_excess_accesses,
        }


def _delayed_raw_last_write(record: Any, start: int) -> dict[tuple[int, int], int]:
    state = {}
    for bank in range(M1056.BLOCKS):
        group = bank // 2
        address = M1056.packed_address(bank, record.row)
        state[(group, address)] = start + 31
    return state


def canonical_work_domain_and_work8_preflight() -> dict[str, Any]:
    """Exhaustive work/provenance gate plus bounded work=8 schedule regression."""
    started = time.time()
    authorities = validate_frozen_authorities()
    coverage = M1072.ProvenanceCoverage()
    counts = {name: Counter() for name in DESIGNS}
    digest = hashlib.sha256()
    work8_geometry = {name: Counter() for name in DESIGNS}
    with M1072.CanonicalRowReader() as reader:
        for task_id in range(TASKS):
            record = reader.derive(task_id)
            coverage.consume_internal(record)
            for name in DESIGNS:
                work = validate_canonical_work(record.works[name])
                bucket = "zero" if work == 0 else ("work8" if work == 8 else "positive_ge16")
                counts[name][bucket] += 1
                digest.update(f"{task_id}:{name}:{work}\n".encode())
                if work != 8:
                    continue
                plan = M1056.TaskPlan(record.task_id,
                                      record.shared_preprocess_cycles,
                                      work, record.row)
                start = record.shared_preprocess_cycles
                fresh_state: dict[tuple[int, int], int] = {}
                fresh = schedule_task(plan, start, fresh_state)
                delayed_state = _delayed_raw_last_write(record, start)
                delayed = schedule_task(plan, start, delayed_state)
                dependencies = [dependency.delay_cycles
                    for event in M1056.nominal_task_events(plan, start, {})
                    for dependency in event.dependencies]
                require(fresh.raw_dependencies_pass is True and
                        delayed.raw_dependencies_pass is True and dependencies and
                        min(dependencies) >= 0,
                        "work8 frozen M1056 geometry regression")
                row = work8_geometry[name]
                row["occurrences"] += 1
                row["fresh_pass"] += 1
                row["delayed_raw_pass"] += 1
                row["raw_dependencies_pass"] += 1
                if "minimum_dependency_delay" not in row:
                    row["minimum_dependency_delay"] = min(dependencies)
                else:
                    row["minimum_dependency_delay"] = min(
                        row["minimum_dependency_delay"], min(dependencies))
    proof = coverage.proof()
    require(proof.get("full_coverage_pass") is True and
            all(sum(row.values()) == TASKS for row in counts.values()) and
            all(row["work8"] == EXPECTED_WORK8_PER_DESIGN for row in counts.values()) and
            sum(row["work8"] for row in counts.values()) == EXPECTED_WORK8_TOTAL and
            all(row["occurrences"] == EXPECTED_WORK8_PER_DESIGN and
                row["fresh_pass"] == EXPECTED_WORK8_PER_DESIGN and
                row["delayed_raw_pass"] == EXPECTED_WORK8_PER_DESIGN and
                row["raw_dependencies_pass"] == EXPECTED_WORK8_PER_DESIGN and
                row["minimum_dependency_delay"] == 0
                for row in work8_geometry.values()),
            "M1102 exhaustive population/geometry drift")
    return {
        "schema": "m1102_c1_work8_domain_and_geometry_preflight_v1",
        "status": "PASS_M1102_EXHAUSTIVE_812160X3_AND_12522_WORK8_REGRESSION",
        "authorities": authorities,
        "tasks": TASKS,
        "designs": list(DESIGNS),
        "values_checked": VALUES,
        "domain": "exact_int && work%8==0 && (work==0 || work>=8)",
        "counts": {name: dict(row) for name, row in counts.items()},
        "work8_geometry": {name: dict(row) for name, row in work8_geometry.items()},
        "work8_occurrences_total": EXPECTED_WORK8_TOTAL,
        "task_design_work_digest_sha256": digest.hexdigest(),
        "row_work_execution_provenance_digest_sha256":
            proof["execution_provenance_digest_sha256"],
        "full_coverage_pass": True,
        "production_full_cycle_iterator_called": False,
        "attempt_created": False,
        "cycles_or_speedup_admitted": False,
        "runtime_seconds": time.time() - started,
    }


def canonical_work_domain_preflight() -> dict[str, Any]:
    """Zero-argument production preflight; includes the required work8 regression."""
    return canonical_work_domain_and_work8_preflight()


def iter_canonical_full_replay_results() -> Iterator[dict[str, Any]]:
    """Zero-argument production iterator.  Not called by this source receipt."""
    validate_frozen_authorities()
    capacity = M1064.derive_physical_capacity()
    coverage = M1072.ProvenanceCoverage()
    sample_rows = []
    with M1072.CanonicalRowReader() as reader:
        streams = {name: DesignStream() for name in DESIGNS}
        for task_id in range(TASKS):
            record = reader.derive(task_id)
            coverage.consume_internal(record)
            for name in DESIGNS:
                work = validate_canonical_work(record.works[name])
                streams[name].consume_internal(M1056.TaskPlan(
                    record.task_id, record.shared_preprocess_cycles,
                    work, record.row))
            if (task_id + 1) % M1072.TASKS_PER_SAMPLE == 0:
                sample = task_id // M1072.TASKS_PER_SAMPLE
                sample_rows.append({
                    "sample": sample,
                    "first_task_id": sample * M1072.TASKS_PER_SAMPLE,
                    "last_task_id": task_id,
                    "designs": {name: streams[name].finish_sample()
                                for name in DESIGNS},
                })
                streams = {name: DesignStream() for name in DESIGNS}
    proof = coverage.proof()
    require(proof.get("full_coverage_pass") is True and
            len(sample_rows) == M1072.SAMPLES,
            "M1102 full provenance coverage failed")
    yield {
        "schema": "m1102_canonical_full_work8_exact_1rw_replay_result_v1",
        "status": "PASS_M1102_RAW_CPU_MODEL_FULL_REPLAY_PENDING_RESULT_HAMMER",
        "samples": sample_rows,
        "coverage": proof,
        "capacity": capacity,
        "claim_boundary": {
            "matched_cycles_admitted": False,
            "speedup_admitted": False,
            "rtl_cycles": False,
            "paper_ppa_ready": False,
            "independent_result_hammer_required": True,
        },
    }


def source_small_oracle() -> dict[str, Any]:
    validate_frozen_authorities()
    sentinel = {(0, 7): 91}
    before = dict(sentinel)
    zero_plan = M1056.TaskPlan(207, 146, 0, 15)
    zero = schedule_task(zero_plan, 500, sentinel)
    require(sentinel == before and zero.events == [] and zero.grants == {},
            "zero-work state drift")
    delegated = []
    for work in range(8, 15):
        repaired_state: dict[tuple[int, int], int] = {}
        frozen_state: dict[tuple[int, int], int] = {}
        plan = M1056.TaskPlan(208, 158, work, 16)
        repaired = schedule_task(plan, 700, repaired_state)
        frozen = M1056.schedule_task(plan, 700, frozen_state)
        require(repaired == frozen and repaired_state == frozen_state,
                "positive delegation drift")
        delegated.append(work)
    rejected = []
    for value in (True, False, -1, *range(1, 8)):
        try:
            validate_work(value)
        except RuntimeError:
            rejected.append(value)
        else:
            raise RuntimeError("illegal generic work admitted")
    canonical_rejected = []
    for value in range(9, 15):
        try:
            validate_canonical_work(value)
        except RuntimeError:
            canonical_rejected.append(value)
        else:
            raise RuntimeError("non-lattice canonical work admitted")
    require(len(inspect.signature(canonical_work_domain_preflight).parameters) == 0 and
            inspect.isgeneratorfunction(iter_canonical_full_replay_results) and
            len(inspect.signature(iter_canonical_full_replay_results).parameters) == 0,
            "production interface shape drift")
    return {
        "status": "PASS_M1102_WORK8_SOURCE_SMALL_ORACLE",
        "zero_semantics_exact": True,
        "positive_behavior_equivalent_work_values": delegated,
        "illegal_generic_values_rejected": len(rejected),
        "non_lattice_canonical_values_rejected": canonical_rejected,
        "production_preflight_called": False,
        "production_full_cycle_iterator_called": False,
        "attempt_created": False,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--exhaustive-readonly", action="store_true")
    args = parser.parse_args()
    require(args.self_test ^ args.exhaustive_readonly, "select exactly one read-only mode")
    result = (source_small_oracle() if args.self_test else
              canonical_work_domain_and_work8_preflight())
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
