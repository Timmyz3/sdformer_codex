#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent bounded hammer for the M1169 II=2 interval source.

This checker deliberately implements its own beat-accurate service oracle.  It
does not open the production schedule, canonical source rows, or an M1161
production result, and it does not invoke EDA, GPU, or remote work.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import stat
import sys
from typing import Any, Iterable, Sequence

HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HERE / "build_m1169_c1_ii2_service_aware_interval_replay_source.py"
SOURCE_SHA = "bd243ca34760757cadbf9c1104049480197f1fb77bf6ad6ec1071870250ebc4f"
TESTS = HW / "system_simulator/tests/test_m1169_c1_ii2_service_aware_interval_replay_source.py"
TESTS_SHA = "558d0deccf29543af26c8b8d855c187c9a1d0722c1a8964b61b8593b123e46a8"
CONTRACT = HW / "contracts/m1169_c1_ii2_service_aware_interval_replay_source_contract_r1_20260830.json"
CONTRACT_SHA = "275214c40e1a53b922c1db448dcedff8792f5232124fc1ea5d474360ded861dc"
AUTHOR = HW / "reviews/m1169_c1_ii2_service_aware_interval_replay_source_author_receipt_r1_20260830"
AUTHOR_REVIEW_SHA = "50dd3649b5ca5eb216ce6926fb77ffdba56570bf76e419a8df47037833f46229"
AUTHOR_MANIFEST_SHA = "edc2bbe2d817cce31983e2e11be5c4137caaef87f8f1bbaab94d7afa462202c9"
AUTHOR_OUTER_SHA = "19472c48ddfe53b2c5aa2ef9ad647a5b0d378c6d0c7143d789505895412b270b"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
PRODUCTION_SCHEDULE = HW / "results/m1141ca_c1_production_schedule_release_r1_20260830/m1141ca_per_task_schedule_records.jsonl"


class HammerFailure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise HammerFailure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_regular(path: Path, expected: str) -> None:
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and not path.is_symlink() and
            sha256(path) == expected, f"identity drift: {path}")


def strict_json(path: Path) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          HammerFailure("nonfinite JSON: " + token)))


def verify_flat(directory: Path, review_sha: str, manifest_sha: str,
                outer_sha: str) -> dict[str, Any]:
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    verify_regular(review, review_sha)
    verify_regular(manifest, manifest_sha)
    verify_regular(outer, outer_sha)
    require(outer.read_text(encoding="utf-8").split() ==
            [manifest_sha, "SHA256SUMS"], "author outer seal content drift")
    names: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]),
                "author manifest row drift")
        name = fields[1].lstrip("*")
        relative = Path(name)
        require(name not in names and name == relative.as_posix() and
                not relative.is_absolute() and ".." not in relative.parts,
                "author manifest member drift")
        names.add(name)
        verify_regular(directory / name, fields[0])
    require("review.json" in names, "author review absent from manifest")
    return strict_json(review)


def load_source():
    spec = importlib.util.spec_from_file_location("m1170_frozen_m1169", SOURCE)
    require(spec is not None and spec.loader is not None, "source import spec drift")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def independent_service(intervals: Iterable[tuple[int, int]],
                        accept_stalls: Sequence[int] | None = None,
                        response_stalls: Sequence[int] | None = None,
                        initial_completed: int | None = None) -> list[int]:
    """Independent depth-one ready/valid model, returning completions."""
    eligible: list[int] = []
    for requested_first, beats in intervals:
        require(type(requested_first) is int and requested_first >= 0 and
                type(beats) is int and beats > 0, "independent interval drift")
        eligible.extend(requested_first + beat for beat in range(beats))
    if accept_stalls is None:
        accept_stalls = [0] * len(eligible)
    if response_stalls is None:
        response_stalls = [0] * len(eligible)
    require(len(accept_stalls) == len(eligible) and
            len(response_stalls) == len(eligible), "stall length drift")
    completions: list[int] = []
    require(initial_completed is None or
            type(initial_completed) is int and initial_completed >= 0,
            "initial completion drift")
    occupied_until = -1 if initial_completed is None else initial_completed
    for request_cycle, accept_extra, response_extra in zip(
            eligible, accept_stalls, response_stalls):
        require(type(accept_extra) is int and accept_extra >= 0 and
                type(response_extra) is int and response_extra >= 0,
                "stall value drift")
        accept_cycle = max(request_cycle, occupied_until + 1) + accept_extra
        complete_cycle = accept_cycle + 1 + response_extra
        completions.append(complete_cycle)
        occupied_until = complete_cycle
    return completions


def expect_failure(callable_, label: str) -> None:
    try:
        callable_()
    except Exception:
        return
    raise HammerFailure("attack escaped: " + label)


def run() -> dict[str, Any]:
    verify_regular(SOURCE, SOURCE_SHA)
    verify_regular(TESTS, TESTS_SHA)
    verify_regular(CONTRACT, CONTRACT_SHA)
    verify_regular(DOCS359, DOCS359_SHA)
    author = verify_flat(AUTHOR, AUTHOR_REVIEW_SHA, AUTHOR_MANIFEST_SHA,
                         AUTHOR_OUTER_SHA)
    require(author["status"].startswith("PASS_M1169_EXACT_II2_INTERVAL") and
            author["scope"]["production_schedule_opened"] is False and
            author["scope"]["m1161_result_opened"] is False,
            "author boundary/status drift")

    module = load_source()
    require(module.EVENTS_PER_AXIS == 70_853_184 and module.TASKS == 812_160 and
            module.RECORDS == 2_436_480 and tuple(module.AXES) ==
            ("candidate", "strongest_zero", "same_coordinate_bit"),
            "production geometry drift")

    # Exhaustive first-completion and within-interval last-completion audit.
    exhaustive = 0
    for previous in [None, *range(0, 33)]:
        for requested in range(0, 33):
            for beats in range(1, 18):
                expected = independent_service(
                    [(requested, beats)], initial_completed=previous)
                offset = 0
                actual = module.advance_zero_stall_ii2(previous, requested, beats)
                require(actual.first_completed_cycle == expected[offset] and
                        actual.last_completed_cycle == expected[-1] and
                        actual.beats == beats,
                        "first/last completion off-by-one")
                nominal = [requested + k + 1 for k in range(beats)]
                queue_delay = sum(c - n for c, n in zip(expected[offset:], nominal))
                require(actual.aggregate_queue_delay_cycles == queue_delay,
                        "aggregate queue delay mismatch")
                exhaustive += 1

    # Random multi-task gaps/overlaps, independent explicit comparison.
    rng = random.Random(0x1170_1169)
    random_trials = 0
    for _ in range(10_000):
        count = rng.randrange(1, 24)
        requested = rng.randrange(0, 64)
        intervals = []
        for _task in range(count):
            requested += rng.randrange(0, 40)
            intervals.append((requested, rng.randrange(1, 24)))
        expected = independent_service(intervals)
        previous = None
        offset = 0
        for first, beats in intervals:
            actual = module.advance_zero_stall_ii2(previous, first, beats)
            require(actual.first_completed_cycle == expected[offset] and
                    actual.last_completed_cycle == expected[offset + beats - 1],
                    "random interval mismatch")
            previous = actual.last_completed_cycle
            offset += beats
        random_trials += 1

    # Stalls may be absorbed by a later task gap, but can never improve any
    # completion.  On a contiguous stream every injected stall must persist.
    stall_attacks = 0
    contiguous = [(0, 37)]
    zero = independent_service(contiguous)
    for index in range(37):
        for kind in ("accept", "response"):
            accept = [0] * 37
            response = [0] * 37
            (accept if kind == "accept" else response)[index] = 1 + index % 5
            attacked = independent_service(contiguous, accept, response)
            require(all(a >= b for a, b in zip(attacked, zero)) and
                    attacked != zero and attacked[-1] > zero[-1],
                    "stall falsely absorbed in contiguous stream")
            stall_attacks += 1
    for _ in range(1_000):
        intervals = [(0, rng.randrange(1, 10)),
                     (rng.randrange(0, 60), rng.randrange(1, 10))]
        count = sum(beats for _, beats in intervals)
        accept = [rng.randrange(0, 4) for _ in range(count)]
        response = [rng.randrange(0, 4) for _ in range(count)]
        base = independent_service(intervals)
        attacked = independent_service(intervals, accept, response)
        require(all(a >= b for a, b in zip(attacked, base)),
                "stalled service improved a completion")
        stall_attacks += 1

    # Exact production floor quota proof, including interval continuity.
    counts: dict[int, int] = {}
    previous_end = 0
    quota_sum = 0
    for task in range(module.TASKS):
        begin, end = module.floor_quota(task, module.TASKS,
                                        module.EVENTS_PER_AXIS)
        require(begin == previous_end and begin < end, "quota hole/overlap")
        count = end - begin
        counts[count] = counts.get(count, 0) + 1
        quota_sum += count
        previous_end = end
    require(counts == {87: 616_896, 88: 195_264} and
            quota_sum == previous_end == 70_853_184,
            "87/88 quota distribution drift")

    # Replay conservation, ordering, drop/duplicate/reorder/wrong-II attacks.
    replay = module.IntervalReplay(13, 1_139, module.AXES)
    explicit_by_axis: dict[str, list[tuple[int, int]]] = {
        axis: [] for axis in module.AXES}
    last_requested = {axis: 0 for axis in module.AXES}
    for task in range(13):
        for axis_index, axis in enumerate(module.AXES):
            last_requested[axis] += (task + axis_index) % 7
            begin, end = module.floor_quota(task, 13, 1_139)
            explicit_by_axis[axis].append((last_requested[axis], end - begin))
            replay.consume_interval(axis, task, last_requested[axis])
    terminal = replay.finalize()
    for axis in module.AXES:
        expected = independent_service(explicit_by_axis[axis])
        row = terminal["axes"][axis]
        require(row["records"] == 13 and row["beats"] == 1_139 and
                row["first_completed_issue_cycle"] == expected[0] and
                row["last_completed_issue_cycle"] == expected[-1] and
                row["ii2_service_makespan_coordinate"] == expected[-1] + 1,
                "replay terminal mismatch")
    require(terminal["expanded_beats"] == 0 and
            terminal["state_complexity"] == "O(axes)" and
            terminal["claim_boundary"] == {
                "component_weight_service_schedule_only": True,
                "rtl_cycles": False,
                "system_speedup": False,
                "traffic_energy_or_ppa": False,
                "paper_citable_without_future_result_hammer": False,
            }, "claim boundary drift")

    attacks = 0
    attack_calls = [
        (lambda: module.IntervalReplay(2, 4, module.AXES).consume_interval(
            "strongest_zero", 0, 0), "axis reorder"),
        (lambda: module.IntervalReplay(2, 4, module.AXES).consume_interval(
            "candidate", 1, 0), "task drop"),
        (lambda: module.IntervalReplay(True, 4, module.AXES), "bool tasks"),
        (lambda: module.IntervalReplay(2, True, module.AXES), "bool beats"),
        (lambda: module.floor_quota(True, 2, 4), "bool task"),
        (lambda: module.advance_zero_stall_ii2(None, True, 1), "bool request"),
        (lambda: module.advance_zero_stall_ii2(None, 0, True), "bool interval"),
        (lambda: module.advance_zero_stall_ii2(True, 0, 1), "bool previous"),
        (lambda: module.ProductionReplay(), "production release"),
    ]
    for callable_, label in attack_calls:
        expect_failure(callable_, label)
        attacks += 1
    incomplete = module.IntervalReplay(2, 4, module.AXES)
    incomplete.consume_interval("candidate", 0, 0)
    expect_failure(incomplete.finalize, "dropped terminal records")
    attacks += 1

    source_task = hashlib.sha256(b"m1170-independent-provenance").hexdigest()
    good = {
        "axis": "candidate", "chunk": 0, "operator": 0, "partition": 0,
        "requested_cycle_first": 0, "sample": 0,
        "source_task_provenance_sha256": source_task,
        "task_sequence_ordinal": 0,
    }
    good["schedule_record_provenance_sha256"] = module.record_provenance(
        "candidate", 0, 0, 0, 0, 0, 0, source_task)
    module.ScheduleRecord.from_mapping(good)
    for field, value in (
            ("axis", "strongest_zero"),
            ("task_sequence_ordinal", 1),
            ("requested_cycle_first", True),
            ("source_task_provenance_sha256", "0" * 64),
            ("schedule_record_provenance_sha256", "f" * 64)):
        attacked = dict(good); attacked[field] = value
        expect_failure(lambda a=attacked: module.ScheduleRecord.from_mapping(a),
                       "provenance/coordinate " + field)
        attacks += 1
    extra = dict(good); extra["extra"] = 0
    expect_failure(lambda: module.ScheduleRecord.from_mapping(extra),
                   "schema extension")
    attacks += 1

    # Verify source preflight while proving the 836 MB production JSONL was
    # only lstat'ed and never opened/read.
    original_open = Path.open
    opened_production = []
    def guarded_open(path: Path, *args, **kwargs):
        if path.resolve() == PRODUCTION_SCHEDULE.resolve():
            opened_production.append(str(path))
            raise HammerFailure("production schedule open attempted")
        return original_open(path, *args, **kwargs)
    Path.open = guarded_open
    try:
        preflight = module.source_preflight()
    finally:
        Path.open = original_open
    require(not opened_production and preflight["m1141_records_opened"] is False and
            preflight["m1161_result_opened"] is False and
            preflight["production_execution"] is False,
            "source-only boundary escaped")

    return {
        "schema": "m1170_m1169_c1_ii2_interval_source_hammer_result_r1_v1",
        "status": "PASS_M1170_M1169_II2_INTERVAL_SOURCE_HAMMER__AUTHORIZE_ONLY_FUTURE_GATED_SUCCESSOR_SOURCE",
        "identity": {
            "m1169_source_sha256": SOURCE_SHA,
            "m1169_tests_sha256": TESTS_SHA,
            "m1169_contract_sha256": CONTRACT_SHA,
            "m1169_author_outer_seal_file_sha256": AUTHOR_OUTER_SHA,
            "docs359_sha256": DOCS359_SHA,
        },
        "independent_derivation": {
            "accept_cycle": "max(request_cycle, previous_completion+1)",
            "completion_cycle": "accept_cycle+1",
            "closed_form_first": "max(requested_first+1, previous_completion+2)",
            "closed_form_last": "first+2*(beats-1)",
            "exhaustive_first_last_delay_cases": exhaustive,
            "random_multitask_gap_overlap_trials": random_trials,
            "stall_attacks": stall_attacks,
        },
        "production_geometry_proof": {
            "tasks_per_axis": module.TASKS,
            "records_all_axes": module.RECORDS,
            "quota_counts": {str(key): value for key, value in sorted(counts.items())},
            "beats_per_axis": quota_sum,
            "beats_all_axes": quota_sum * len(module.AXES),
            "state_complexity": "O(axes)",
            "event_materialization": False,
        },
        "attacks": {
            "drop_duplicate_reorder_bool_provenance_schema": attacks,
            "wrong_ii_detected_by_independent_explicit_oracle": True,
            "off_by_one_first_completion_detected": True,
            "task_gaps_and_overlaps_checked": True,
            "nonzero_stalls_never_improve": True,
            "contiguous_nonzero_stalls_not_absorbed": True,
        },
        "scope": {
            "production_schedule_opened": False,
            "production_records_consumed": 0,
            "production_beats_expanded": 0,
            "m1161_result_consumed": False,
            "canonical_rows_opened": False,
            "vcs_dc_ptpx_gpu_remote": False,
            "docs359_modified": False,
        },
        "authorization": {
            "m1169_source_hammer_passed": True,
            "production_execution_now": False,
            "future_successor_source_only_after_sealed_m1161_result": True,
            "fresh_different_author_m1161_result_hammer_required": True,
            "future_successor_must_pin_exact_outer_seals": True,
        },
        "claim_boundary": {
            "component_weight_service_schedule_only": True,
            "rtl_cycles": False,
            "system_speedup": False,
            "paper_citable_performance_now": False,
            "ppa_energy_traffic": False,
        },
    }


def main() -> None:
    require(len(sys.argv) == 1, "M1170 hammer accepts zero arguments")
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
