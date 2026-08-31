#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Receipt-blind M1095 failure audit; full work scan, bounded short-work geometry only."""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
HW = Path(__file__).resolve().parents[2]
RESULTS = HW / "results"
ATTEMPT = RESULTS / ".m1094_m1086_c1_zero_work_exact_1rw_full_replay_attempt_consumed"
QUARANTINE = RESULTS / "m1094_m1086_c1_zero_work_exact_1rw_full_replay_r1_20260830.failed_or_incomplete.3351976.1788028823132798404.quarantine"
M1072_PATH = HW / "system_simulator/scripts/run_m1072_c1_row_provenance_exact_1rw_source.py"
M1056_PATH = HW / "system_simulator/scripts/run_m1056_c1_exact_1rw_arbitration_replay_source.py"
M1086_PATH = HW / "system_simulator/scripts/run_m1086_c1_zero_work_exact_1rw_source.py"
M1094_PATH = HW / "system_simulator/scripts/execute_m1094_m1086_c1_zero_work_exact_1rw_full_replay_one_shot.py"
M1095_PATH = HW / "system_simulator/scripts/run_m1095_m1094r2_c1_zero_work_full_replay_zero_arg.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
OUT = Path(__file__).with_name("mechanical_checks.json")

EXPECTED = {
    M1072_PATH: "879712a59785acc79776990236884582431adea81103a222d5415905199a1e4c",
    M1056_PATH: "95e276a7afe7a049faa2b967bed1431762c72a5e0b521c3e9857121ece5c816f",
    M1086_PATH: "3925c97de922393786b4aa8ae6ca6b4942489e3cf10485f5d1b6cd423e797a51",
    M1094_PATH: "c8808c0d4cf37a8f279afa128e089c08af3718606061658db8f2047c198c824a",
    M1095_PATH: "74576584bcf3140a17d935f7f2bce2fb7fe6a373e8e4b2b0666f5e797e0a5f3b",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
DESIGN_ORDER = ("candidate", "strongest_zero", "same_coordinate_bit")
SHORT_VALUES = tuple(range(1, 15))
TASKS = 812160


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(path: Path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise RuntimeError("duplicate key " + key)
            out[key] = value
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite " + token)))


def verify_atomic(directory: Path) -> dict:
    bundle = directory / ".m1094_atomic_seal"
    manifest = bundle / "SHA256SUMS"
    outer = bundle / "SHA256SUMS.seal.sha256"
    if (not directory.is_dir() or directory.is_symlink() or
            not bundle.is_dir() or bundle.is_symlink() or
            not manifest.is_file() or manifest.is_symlink() or
            not outer.is_file() or outer.is_symlink()):
        raise RuntimeError("atomic seal shape")
    if outer.read_text() != sha(manifest) + "  SHA256SUMS\n":
        raise RuntimeError("outer content")
    listed = {}
    for line in manifest.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        member = directory / relative
        if (relative in listed or not member.is_file() or member.is_symlink() or sha(member) != digest):
            raise RuntimeError("manifest drift " + relative)
        listed[relative] = digest
    actual = set()
    for item in directory.rglob("*"):
        relative = item.relative_to(directory)
        if relative.parts and relative.parts[0] == ".m1094_atomic_seal":
            continue
        if item.is_symlink():
            raise RuntimeError("symlink in evidence")
        if item.is_file():
            actual.add(relative.as_posix())
    if set(listed) != actual:
        raise RuntimeError("manifest coverage")
    return {
        "manifest_sha256": sha(manifest),
        "outer_seal_file_sha256": sha(outer),
        "members": len(actual),
    }


def load_m1072(name: str):
    spec = importlib.util.spec_from_file_location(name, M1072_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def occurrence(record, design: str, value: int) -> dict:
    return {
        "task_id": record.task_id,
        "sample": record.sample,
        "operator": record.operator,
        "chunk": record.chunk,
        "partition": record.partition,
        "row": record.row,
        "row_count": record.row_count,
        "file_offset": record.file_offset,
        "design": design,
        "value": value,
        "shared_preprocess_cycles": record.shared_preprocess_cycles,
        "raw_row_bytes_sha256": record.raw_row_bytes_sha256,
        "provenance_sha256": record.provenance_sha256,
    }


def schedule_geometry(module, record, design: str, value: int) -> dict:
    m1056 = module.M1056
    plan = m1056.TaskPlan(record.task_id, record.shared_preprocess_cycles,
                         value, record.row)
    start = record.shared_preprocess_cycles
    nominal = m1056.nominal_task_events(plan, start, {})
    delays = [dep.delay_cycles for event in nominal for dep in event.dependencies]
    fresh_error = None
    fresh_result = None
    try:
        fresh_result = m1056.schedule_task(plan, start, {})
    except Exception as exc:
        fresh_error = type(exc).__name__ + ": " + str(exc)
    raw_last_write = {}
    for bank in range(m1056.BLOCKS):
        group = bank // 2
        address = m1056.packed_address(bank, record.row)
        raw_last_write[(group, address)] = start + 31
    raw_error = None
    raw_result = None
    try:
        raw_result = m1056.schedule_task(plan, start, raw_last_write)
    except Exception as exc:
        raw_error = type(exc).__name__ + ": " + str(exc)
    return {
        "min_dependency_delay": min(delays),
        "negative_dependency_count": sum(delay < 0 for delay in delays),
        "fresh_pass": fresh_error is None,
        "fresh_error": fresh_error,
        "fresh_raw_dependencies_pass": None if fresh_result is None else fresh_result.raw_dependencies_pass,
        "raw_predecessor_pass": raw_error is None,
        "raw_predecessor_error": raw_error,
        "raw_dependencies_pass": None if raw_result is None else raw_result.raw_dependencies_pass,
    }


def worker(args):
    worker_id, begin, end = args
    module = load_m1072("m1100_worker_%d" % worker_id)
    counts = {design: Counter() for design in DESIGN_ORDER}
    minimum = {design: None for design in DESIGN_ORDER}
    maximum = {design: None for design in DESIGN_ORDER}
    minimum_positive = {design: None for design in DESIGN_ORDER}
    maximum_positive = {design: None for design in DESIGN_ORDER}
    first = None
    first_by_value_design = {}
    geometry = {design: {str(value): Counter() for value in SHORT_VALUES}
                for design in DESIGN_ORDER}
    geometry_examples = {}
    digest = hashlib.sha256()
    started = time.time()
    with module.CanonicalRowReader() as reader:
        for task_id in range(begin, end):
            record = reader.derive(task_id)
            for design in DESIGN_ORDER:
                value = record.works[design]
                minimum[design] = value if minimum[design] is None else min(minimum[design], value)
                maximum[design] = value if maximum[design] is None else max(maximum[design], value)
                if value > 0:
                    minimum_positive[design] = value if minimum_positive[design] is None else min(minimum_positive[design], value)
                    maximum_positive[design] = value if maximum_positive[design] is None else max(maximum_positive[design], value)
                digest.update(f"{task_id}:{design}:{value}\n".encode())
                if 1 <= value <= 14:
                    counts[design][value] += 1
                    row = occurrence(record, design, value)
                    key = f"{design}:{value}"
                    if first is None:
                        first = row
                    if key not in first_by_value_design:
                        first_by_value_design[key] = row
                    check = schedule_geometry(module, record, design, value)
                    bucket = geometry[design][str(value)]
                    bucket["occurrences"] += 1
                    bucket["negative_dependency_occurrences"] += int(check["negative_dependency_count"] > 0)
                    bucket["fresh_pass"] += int(check["fresh_pass"])
                    bucket["raw_predecessor_pass"] += int(check["raw_predecessor_pass"])
                    bucket["raw_dependencies_pass"] += int(check["raw_dependencies_pass"] is True)
                    bucket["minimum_dependency_delay"] = min(
                        bucket.get("minimum_dependency_delay", check["min_dependency_delay"]),
                        check["min_dependency_delay"],
                    )
                    if key not in geometry_examples:
                        geometry_examples[key] = {"occurrence": row, "check": check}
    return {
        "worker": worker_id,
        "begin": begin,
        "end": end,
        "tasks": end - begin,
        "counts": {design: {str(value): counts[design][value] for value in SHORT_VALUES}
                   for design in DESIGN_ORDER},
        "minimum": minimum,
        "maximum": maximum,
        "minimum_positive": minimum_positive,
        "maximum_positive": maximum_positive,
        "first": first,
        "first_by_value_design": first_by_value_design,
        "geometry": {design: {value: dict(counter) for value, counter in rows.items()}
                     for design, rows in geometry.items()},
        "geometry_examples": geometry_examples,
        "work_digest_sha256": digest.hexdigest(),
        "seconds": time.time() - started,
    }


def merge(parts):
    counts = {design: Counter() for design in DESIGN_ORDER}
    minimum = {design: None for design in DESIGN_ORDER}
    maximum = {design: None for design in DESIGN_ORDER}
    minimum_positive = {design: None for design in DESIGN_ORDER}
    maximum_positive = {design: None for design in DESIGN_ORDER}
    first_rows = [part["first"] for part in parts if part["first"] is not None]
    first = min(first_rows, key=lambda row: (row["task_id"], DESIGN_ORDER.index(row["design"])))
    first_by = {}
    geometry = {design: {str(value): Counter() for value in SHORT_VALUES}
                for design in DESIGN_ORDER}
    examples = {}
    for part in parts:
        for design in DESIGN_ORDER:
            for value in SHORT_VALUES:
                counts[design][value] += part["counts"][design][str(value)]
                source_geometry = part["geometry"][design][str(value)]
                target_geometry = geometry[design][str(value)]
                for field, amount in source_geometry.items():
                    if field == "minimum_dependency_delay":
                        if field not in target_geometry:
                            target_geometry[field] = amount
                        else:
                            target_geometry[field] = min(target_geometry[field], amount)
                    else:
                        target_geometry[field] += amount
            rows = [part["minimum"][design], minimum[design]]
            minimum[design] = min(v for v in rows if v is not None)
            rows = [part["maximum"][design], maximum[design]]
            maximum[design] = max(v for v in rows if v is not None)
            rows = [part["minimum_positive"][design], minimum_positive[design]]
            minimum_positive[design] = min(v for v in rows if v is not None)
            rows = [part["maximum_positive"][design], maximum_positive[design]]
            maximum_positive[design] = max(v for v in rows if v is not None)
        for key, row in part["first_by_value_design"].items():
            if key not in first_by or row["task_id"] < first_by[key]["task_id"]:
                first_by[key] = row
        for key, value in part["geometry_examples"].items():
            if key not in examples or value["occurrence"]["task_id"] < examples[key]["occurrence"]["task_id"]:
                examples[key] = value
    return {
        "tasks": sum(part["tasks"] for part in parts),
        "values_checked": sum(part["tasks"] for part in parts) * 3,
        "counts_1_to_14_by_design": {
            design: {str(value): counts[design][value] for value in SHORT_VALUES}
            for design in DESIGN_ORDER
        },
        "minimum_all_work_by_design": minimum,
        "maximum_all_work_by_design": maximum,
        "minimum_positive_work_by_design": minimum_positive,
        "maximum_positive_work_by_design": maximum_positive,
        "first_short_positive": first,
        "first_by_value_design": first_by,
        "geometry": {design: {value: dict(counter) for value, counter in rows.items()}
                     for design, rows in geometry.items()},
        "geometry_examples": examples,
        "parts": [{key: part[key] for key in ("worker", "begin", "end", "tasks", "work_digest_sha256", "seconds")}
                  for part in parts],
    }


def main():
    started = time.time()
    for path, expected in EXPECTED.items():
        if not path.is_file() or path.is_symlink() or sha(path) != expected:
            raise RuntimeError("frozen identity drift " + str(path))
    attempt_seal = verify_atomic(ATTEMPT)
    quarantine_seal = verify_atomic(QUARANTINE)
    attempt = strict_json(ATTEMPT / "attempt.json")
    failure = strict_json(QUARANTINE / "failure.json")
    traceback_text = (QUARANTINE / "partial_result/traceback.log").read_text(encoding="utf-8")
    workers = 4
    bounds = []
    for index in range(workers):
        begin = TASKS * index // workers
        end = TASKS * (index + 1) // workers
        bounds.append((index, begin, end))
    context = mp.get_context("fork")
    with context.Pool(workers) as pool:
        parts = pool.map(worker, bounds)
    scan = merge(parts)
    present_values = sorted({int(value) for design in DESIGN_ORDER
                             for value, count in scan["counts_1_to_14_by_design"][design].items()
                             if count})
    unsafe_present = [value for value in present_values if value <= 6]
    geometry_failures = []
    for design in DESIGN_ORDER:
        for value in map(str, SHORT_VALUES):
            row = scan["geometry"][design][value]
            if row.get("occurrences", 0) and (
                    row.get("fresh_pass", 0) != row["occurrences"] or
                    row.get("raw_predecessor_pass", 0) != row["occurrences"] or
                    row.get("raw_dependencies_pass", 0) != row["occurrences"] or
                    row.get("negative_dependency_occurrences", 0) != 0):
                geometry_failures.append({"design": design, "value": int(value), "row": row})
    legal_short_work = bool(present_values) and not unsafe_present and not geometry_failures
    output = {
        "schema": "m1100_m1095_c1_failed_preflight_audit_checks_v1",
        "receipt_blind": True,
        "production_preflight_called": False,
        "production_iterator_called": False,
        "cycle_scheduler_full_replay_called": False,
        "bounded_m1056_schedule_task_only_for_short_occurrences": True,
        "attempt": {
            "seal": attempt_seal,
            "status": attempt["status"],
            "maximum_attempts": attempt["maximum_attempts"],
            "automatic_retry": attempt["automatic_retry"],
            "canonical_payload_opened_before_attempt": attempt["canonical_payload_opened_or_hashed_before_attempt"],
        },
        "quarantine": {
            "seal": quarantine_seal,
            "status": failure["status"],
            "phase": failure["phase"],
            "attempt_consumed": failure["attempt_consumed"],
            "automatic_retry": failure["automatic_retry"],
            "traceback_exact_terminal": traceback_text.strip().splitlines()[-1],
        },
        "scan": scan,
        "present_short_values": present_values,
        "unsafe_geometry_values_present": unsafe_present,
        "geometry_failures": geometry_failures,
        "classification": {
            "legal_short_positive_work": legal_short_work,
            "trace_error": False if legal_short_work else None,
            "reason": "All short positives are rederived from frozen raw rows and preserve provenance. Every present short value passes frozen M1056 fresh and delayed-RAW schedule geometry with nonnegative dependencies." if legal_short_work else "Unsafe or failing short-work geometry requires further source audit.",
        },
        "minimum_additive_repair_gate": {
            "m1095_do_not_retry": True,
            "freeze_m1086_m1094_m1095_and_evidence": True,
            "new_source_contract_release_attempt_namespace_required": True,
            "domain_change": "admit work==0 or work>=8; keep 1..7 fail-closed",
            "zero_work_semantics_unchanged": True,
            "positive_work_delegates_exact_frozen_m1056": True,
            "required_exhaustive_preflight": "812160 tasks x 3 designs; exact 2436480 values; bind counts/digest/provenance",
            "required_short_geometry": "every present work 8..14 occurrence fresh + delayed-RAW M1056 schedule passes; dependencies >=0",
            "required_regressions": [
                "work 0 emits no event/grant and does not mutate last_write",
                "work 1..7 rejects before attempt",
                "work 8..14 delegates bit-identically to frozen M1056",
                "task207 zero-work and task208 positive RAW regression remains exact",
                "attempt precedes preflight and iterator; no retry/no-replace/quarantine remain exact"
            ],
            "different_author_source_hammer_required": True,
            "one_new_cpu_attempt_only_after_hammer": True
        },
        "runtime_seconds": time.time() - started,
        "docs359_sha256": sha(DOCS359),
        "verdict": "PASS_M1100_LEGAL_SHORT_POSITIVE_WORK__ADDITIVE_DOMAIN_REPAIR_ALLOWED" if legal_short_work else "STOP_M1100_SHORT_WORK_GEOMETRY_OR_TRACE_UNRESOLVED"
    }
    OUT.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(output["verdict"])


if __name__ == "__main__":
    main()
