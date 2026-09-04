#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1169 source-only interval replay for the M1162 depth-one service.

The frozen M1141 schedule gives one requested-cycle origin per task and axis.
M1137 assigns that task a contiguous interval of weight beats by the exact
floor quota ``[floor(t*E/T), floor((t+1)*E/T))``.  Under the M1162 contract
(zero request stalls, one-cycle responses, one outstanding transaction), the
completed issue cycles for an interval are therefore an arithmetic sequence
with step two.  This module evaluates that sequence in O(tasks), without
materialising the 212,559,552 beats.

This is deliberately not a production launcher.  It never opens the 836 MB
M1141 JSONL, the canonical source rows, or an M1161 result.  A successor must
pin a sealed M1161 production result and a fresh independent result hammer
before it may stream the sealed schedule through ``ProductionReplay``.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import stat
import struct
from typing import Any, Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent

DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

M1162_RTL = HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
M1162_RTL_SHA = "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595"
M1162_CONTRACT = HW / "contracts/m1162_m1160_m1116c_c1_common_charge_protocol_repair_source_contract_r1_20260830.json"
M1162_CONTRACT_SHA = "5787f3302aa3308485e357c41385e69da93e6b41bfdea92410690af5a95ecbdc"

M1166 = HW / "reviews/m1166_m1162_c1_common_charge_protocol_repair_independent_hammer_r1_20260830"
M1166_REVIEW_SHA = "7f2cdf4cb1f979c0680b491c27c1088bc35624a2fd801b97c304c5b403076b4c"
M1166_MANIFEST_SHA = "da8daaef6b6832dd2d3278fcbdf61613170f07da5bb65e311915a3c421e76363"
M1166_OUTER_FILE_SHA = "afc25e37fa8b3b5c5bd8e8c1b3582fecc5d2d75450df86b7c48f71e992ea02ef"

M1141 = HW / "results/m1141ca_c1_production_schedule_release_r1_20260830"
M1141_RELEASE = M1141 / "m1141ca_schedule_release.json"
M1141_RECORDS = M1141 / "m1141ca_per_task_schedule_records.jsonl"
M1141_RELEASE_SHA = "4c4d264a9ac1e084c8c0acf0a6d150140f95ee96ee967b038ea4c1eefcc2b58c"
M1141_RECORDS_SHA = "4d4e0e6396ac1061aca7ada142bc2761bf12a785e5373640a28503e3d73a0a81"
M1141_RECORDS_BYTES = 836_268_740
M1141_MANIFEST_SHA = "852b48c0d8098ef69a619925f82a8e1a308e87f2faf9ea76becabf51d52caace"
M1141_OUTER_FILE_SHA = "0b6549ce38a62bcb22e8a97d0c038860f5698fabc0d9bff162dc6af95d4f043a"

M1161_SOURCE = HERE / "run_m1161ca_c1_production_real_replay_driver_one_shot_source.py"
M1161_SOURCE_SHA = "d7ffb8dbab289e83fd8a32f4ed5244cd005a4b6d0785b586df932fd6a97ee20d"
M1161_CONTRACT = HW / "contracts/m1161ca_c1_production_real_replay_driver_source_contract_r1_20260830.json"
M1161_CONTRACT_SHA = "93471a51d5f9d9270ece1629688b10b0cf88047abed9a5e7b6e71048cd63ef63"
M1164 = HW / "reviews/m1164_m1161ca_c1_production_real_replay_driver_hammer_r1_20260830"
M1164_REVIEW_SHA = "a2d53aa814da8c29ee335b1782594253c7cf9f69d1312984c207e540c2899f0b"
M1164_MANIFEST_SHA = "d1153fa67a8af96d679a7a04475f34b7c7aeacd69cb0f59c3078b4d18e97c84f"
M1164_OUTER_FILE_SHA = "7e6d8e23326b65e59db4347887da1f31822a0cc7ae8fcca59ca191309cac57c4"

CONTRACT = HW / "contracts/m1169_c1_ii2_service_aware_interval_replay_source_contract_r1_20260830.json"

M1016_SOURCE_SHA = "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa"
M1102_SOURCE_SHA = "95bd50aebcc473ab69cdea6ccf27d54743c89926c5e0f31199dc469ced9bf7cc"
M1137_SOURCE_SHA = "9ec640ae8c9fa75f9cbf706e15d2d26a4233def77e5be4d67e94c084347b20a6"

AXES = ("candidate", "strongest_zero", "same_coordinate_bit")
SAMPLES = 10
OPERATORS = 4
CHUNKS = 47
PARTITIONS = 432
TASKS = 812_160
EVENTS_PER_AXIS = 70_853_184
RECORDS = TASKS * len(AXES)
SCHEDULE_FIELDS = (
    "axis", "chunk", "operator", "partition", "requested_cycle_first",
    "sample", "schedule_record_provenance_sha256",
    "source_task_provenance_sha256", "task_sequence_ordinal",
)


class Failure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise Failure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_regular(path: Path, expected: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and
            sha256(path) == expected, "identity drift: " + str(path))


def strict_json(path: Path) -> Any:
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("nonfinite JSON: " + token)))


def verify_flat(directory: Path, review_sha: str, manifest_sha: str,
                outer_file_sha: str | None = None) -> dict[str, Any]:
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    verify_regular(review, review_sha)
    verify_regular(manifest, manifest_sha)
    if outer_file_sha is not None:
        verify_regular(outer, outer_file_sha)
    require(outer.read_text(encoding="utf-8").split() ==
            [manifest_sha, "SHA256SUMS"], "outer seal content drift")
    listed: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]),
                "manifest row drift")
        name = fields[1].lstrip("*")
        relative = Path(name)
        require(name not in listed and name == relative.as_posix() and
                not relative.is_absolute() and ".." not in relative.parts,
                "manifest member drift")
        listed[name] = fields[0]
    require(listed.get("review.json") == review_sha, "review manifest drift")
    for name, digest in listed.items():
        verify_regular(directory / name, digest)
    return strict_json(review)


def verify_m1141_metadata_without_records_open() -> dict[str, Any]:
    """Verify the small sealed metadata and lstat the JSONL; never open it."""
    verify_regular(M1141_RELEASE, M1141_RELEASE_SHA)
    verify_regular(M1141 / "SHA256SUMS", M1141_MANIFEST_SHA)
    verify_regular(M1141 / "SHA256SUMS.seal.sha256", M1141_OUTER_FILE_SHA)
    require((M1141 / "SHA256SUMS.seal.sha256").read_text(encoding="utf-8").split() ==
            [M1141_MANIFEST_SHA, "SHA256SUMS"], "M1141 outer content drift")
    rows = {}
    for line in (M1141 / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2, "M1141 manifest row drift")
        rows[fields[1].lstrip("*")] = fields[0]
    require(rows.get(M1141_RECORDS.name) == M1141_RECORDS_SHA and
            rows.get(M1141_RELEASE.name) == M1141_RELEASE_SHA,
            "M1141 identity set drift")
    value = M1141_RECORDS.lstat()
    require(stat.S_ISREG(value.st_mode) and not M1141_RECORDS.is_symlink() and
            value.st_size == M1141_RECORDS_BYTES,
            "M1141 records metadata drift")
    release = strict_json(M1141_RELEASE)
    require(release["schema"] == "m1141ca_c1_production_schedule_release_r1_v1" and
            release["records"]["count"] == RECORDS and
            release["records"]["sha256"] == M1141_RECORDS_SHA and
            release["geometry"]["tasks"] == TASKS and
            tuple(release["geometry"]["axes"]) == AXES,
            "M1141 release geometry drift")
    return release


def source_preflight() -> dict[str, Any]:
    verify_regular(DOCS359, DOCS359_SHA)
    verify_regular(M1162_RTL, M1162_RTL_SHA)
    verify_regular(M1162_CONTRACT, M1162_CONTRACT_SHA)
    m1166 = verify_flat(M1166, M1166_REVIEW_SHA, M1166_MANIFEST_SHA,
                        M1166_OUTER_FILE_SHA)
    verify_regular(M1161_SOURCE, M1161_SOURCE_SHA)
    verify_regular(M1161_CONTRACT, M1161_CONTRACT_SHA)
    m1164 = verify_flat(M1164, M1164_REVIEW_SHA, M1164_MANIFEST_SHA,
                        M1164_OUTER_FILE_SHA)
    release = verify_m1141_metadata_without_records_open()
    contract = strict_json(CONTRACT)
    require(m1166["status"].startswith("PASS_M1166_M1162_PROTOCOL_REPAIR") and
            m1166["protocol_findings"][
                "minimum_completed_issue_ii_zero_stall_one_cycle_response"] == 2 and
            m1164["status"].startswith("PASS_M1164_M1161CA") and
            contract["schema"] ==
                "m1169_c1_ii2_service_aware_interval_replay_source_contract_r1_v1",
            "frozen authorization/status drift")
    return {
        "status": "PASS_M1169_SOURCE_PREFLIGHT__NO_SCHEDULE_OPEN_NO_PRODUCTION",
        "m1141_records_opened": False,
        "m1161_result_opened": False,
        "canonical_rows_opened": False,
        "production_execution": False,
        "m1141_records": release["records"]["count"],
        "m1162_completed_issue_ii": 2,
    }


def _u64(value: int) -> bytes:
    require(type(value) is int and 0 <= value < (1 << 64), "u64 drift")
    return struct.pack(">Q", value)


def task_index(sample: int, operator: int, chunk: int, partition: int) -> int:
    require(all(type(v) is int for v in (sample, operator, chunk, partition)) and
            0 <= sample < SAMPLES and 0 <= operator < OPERATORS and
            0 <= chunk < CHUNKS and 0 <= partition < PARTITIONS,
            "task coordinate drift")
    return (((sample * OPERATORS + operator) * CHUNKS + chunk) *
            PARTITIONS + partition)


def record_provenance(axis: str, task: int, sample: int, operator: int,
                      chunk: int, partition: int, requested: int,
                      source_task_provenance: str) -> str:
    require(axis in AXES and type(source_task_provenance) is str and
            re.fullmatch(r"[0-9a-f]{64}", source_task_provenance) is not None,
            "record provenance input drift")
    payload = b"".join((
        b"M1139CA_SCHEDULE_RECORD\x00\x01", bytes.fromhex(M1016_SOURCE_SHA),
        bytes.fromhex(M1102_SOURCE_SHA), bytes.fromhex(M1137_SOURCE_SHA),
        struct.pack(">B", AXES.index(axis)), _u64(task), _u64(sample),
        _u64(operator), _u64(chunk), _u64(partition), _u64(requested),
        bytes.fromhex(source_task_provenance),
    ))
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ScheduleRecord:
    axis: str
    chunk: int
    operator: int
    partition: int
    requested_cycle_first: int
    sample: int
    schedule_record_provenance_sha256: str
    source_task_provenance_sha256: str
    task_sequence_ordinal: int

    @classmethod
    def from_mapping(cls, value: Any) -> "ScheduleRecord":
        require(type(value) is dict and set(value) == set(SCHEDULE_FIELDS),
                "schedule exact-field-set drift")
        record = cls(**value)
        record.validate()
        return record

    def validate(self) -> None:
        require(self.axis in AXES and all(type(v) is int and v >= 0 for v in (
                    self.chunk, self.operator, self.partition,
                    self.requested_cycle_first, self.sample,
                    self.task_sequence_ordinal)) and
                self.task_sequence_ordinal == task_index(
                    self.sample, self.operator, self.chunk, self.partition) and
                self.schedule_record_provenance_sha256 == record_provenance(
                    self.axis, self.task_sequence_ordinal, self.sample,
                    self.operator, self.chunk, self.partition,
                    self.requested_cycle_first,
                    self.source_task_provenance_sha256),
                "schedule coordinate/provenance drift")


def floor_quota(task: int, total_tasks: int, total_beats: int) -> tuple[int, int]:
    require(type(task) is int and type(total_tasks) is int and
            type(total_beats) is int and total_tasks > 0 and
            total_beats >= total_tasks and 0 <= task < total_tasks,
            "quota geometry drift")
    begin = (task * total_beats) // total_tasks
    end = ((task + 1) * total_beats) // total_tasks
    require(begin < end, "empty interval unsupported")
    return begin, end


@dataclass(frozen=True)
class IntervalResult:
    first_completed_cycle: int
    last_completed_cycle: int
    beats: int
    aggregate_queue_delay_cycles: int


def advance_zero_stall_ii2(previous_completed: int | None,
                           requested_cycle_first: int,
                           beats: int) -> IntervalResult:
    """Exact closed form for one contiguous requested-beat interval.

    Requested beat ``k`` is eligible at ``requested_cycle_first + k``.
    A request completes one cycle after acceptance.  Because depth is one, a
    new request accepts no earlier than one cycle after the prior completion.
    Thus completed issues have II=2.
    """
    require((previous_completed is None or
             type(previous_completed) is int and previous_completed >= 0) and
            type(requested_cycle_first) is int and requested_cycle_first >= 0 and
            type(beats) is int and beats > 0, "interval input drift")
    first = requested_cycle_first + 1
    if previous_completed is not None:
        first = max(first, previous_completed + 2)
    last = first + 2 * (beats - 1)
    delay0 = first - (requested_cycle_first + 1)
    aggregate_delay = beats * delay0 + beats * (beats - 1) // 2
    return IntervalResult(first, last, beats, aggregate_delay)


@dataclass
class AxisState:
    records: int = 0
    beats: int = 0
    first_requested: int | None = None
    last_requested_first: int | None = None
    max_requested: int | None = None
    first_completed: int | None = None
    last_completed: int | None = None
    aggregate_queue_delay_cycles: int = 0


class IntervalReplay:
    """O(axes) matched replay over task-level schedule records."""
    def __init__(self, total_tasks: int, total_beats_per_axis: int,
                 axes: Sequence[str] = AXES):
        require(type(total_tasks) is int and type(total_beats_per_axis) is int and
                total_tasks > 0 and total_beats_per_axis >= total_tasks and
                tuple(axes) and len(set(axes)) == len(tuple(axes)),
                "replay geometry drift")
        self.total_tasks = total_tasks
        self.total_beats_per_axis = total_beats_per_axis
        self.axes = tuple(axes)
        self.state = {axis: AxisState() for axis in self.axes}
        self.records = 0

    def consume_interval(self, axis: str, task: int,
                         requested_cycle_first: int) -> IntervalResult:
        expected_task = self.records // len(self.axes)
        expected_axis = self.axes[self.records % len(self.axes)]
        require(axis == expected_axis and task == expected_task and
                task < self.total_tasks, "task-major/axis-minor order drift")
        state = self.state[axis]
        require(state.records == task and type(requested_cycle_first) is int and
                requested_cycle_first >= 0 and
                (state.last_requested_first is None or
                 requested_cycle_first >= state.last_requested_first),
                "axis requested-cycle/order drift")
        begin, end = floor_quota(task, self.total_tasks,
                                 self.total_beats_per_axis)
        beats = end - begin
        result = advance_zero_stall_ii2(
            state.last_completed, requested_cycle_first, beats)
        if state.first_requested is None:
            state.first_requested = requested_cycle_first
            state.first_completed = result.first_completed_cycle
        state.last_requested_first = requested_cycle_first
        state.max_requested = max(
            requested_cycle_first + beats - 1,
            state.max_requested if state.max_requested is not None else 0)
        state.last_completed = result.last_completed_cycle
        state.records += 1
        state.beats += beats
        state.aggregate_queue_delay_cycles += result.aggregate_queue_delay_cycles
        self.records += 1
        return result

    def finalize(self) -> dict[str, Any]:
        require(self.records == self.total_tasks * len(self.axes),
                "terminal record conservation drift")
        rows: dict[str, Any] = {}
        for axis in self.axes:
            state = self.state[axis]
            require(state.records == self.total_tasks and
                    state.beats == self.total_beats_per_axis and
                    state.first_requested is not None and
                    state.max_requested is not None and
                    state.first_completed is not None and
                    state.last_completed is not None,
                    "terminal axis conservation drift")
            rows[axis] = {
                "records": state.records,
                "beats": state.beats,
                "first_requested_cycle": state.first_requested,
                "requested_schedule_makespan_coordinate": state.max_requested + 1,
                "first_completed_issue_cycle": state.first_completed,
                "last_completed_issue_cycle": state.last_completed,
                "ii2_service_makespan_coordinate": state.last_completed + 1,
                "aggregate_queue_delay_cycles":
                    state.aggregate_queue_delay_cycles,
            }
        ratios: dict[str, Any] = {}
        if tuple(self.axes) == AXES:
            cand = rows["candidate"]["ii2_service_makespan_coordinate"]
            zero = rows["strongest_zero"]["ii2_service_makespan_coordinate"]
            bit = rows["same_coordinate_bit"]["ii2_service_makespan_coordinate"]
            ratios = {
                "strongest_zero_over_candidate": {
                    "numerator": zero, "denominator": cand,
                    "ratio_decimal": float(Fraction(zero, cand))},
                "same_coordinate_bit_over_candidate": {
                    "numerator": bit, "denominator": cand,
                    "ratio_decimal": float(Fraction(bit, cand))},
            }
        return {
            "schema": "m1169_c1_ii2_interval_replay_terminal_v1",
            "status": "PASS_EXACT_INTERVAL_RECURRENCE__COMPONENT_SCHEDULE_ONLY",
            "service_model": {
                "outstanding_depth": 1,
                "request_stalls": 0,
                "response_latency_cycles": 1,
                "completed_issue_ii": 2,
            },
            "axes": rows,
            "component_schedule_ratios": ratios,
            "state_complexity": "O(axes)",
            "expanded_beats": 0,
            "claim_boundary": {
                "component_weight_service_schedule_only": True,
                "rtl_cycles": False,
                "system_speedup": False,
                "traffic_energy_or_ppa": False,
                "paper_citable_without_future_result_hammer": False,
            },
        }


class ProductionReplay(IntervalReplay):
    def __init__(self):
        raise Failure(
            "STOP: M1169 is source-only; a successor must pin a sealed M1161 "
            "production result plus a fresh independent result hammer")

    def consume_mapping(self, value: Any) -> IntervalResult:
        record = ScheduleRecord.from_mapping(value)
        return self.consume_interval(record.axis, record.task_sequence_ordinal,
                                     record.requested_cycle_first)


def explicit_beat_simulation(
        intervals: Iterable[tuple[int, int]],
        request_accept_stalls: Sequence[int] = (),
        response_extra_stalls: Sequence[int] = ()) -> list[int]:
    """Bounded explicit oracle.  Nonzero stalls are diagnostic, not M1169."""
    requested = []
    for first, beats in intervals:
        require(type(first) is int and first >= 0 and type(beats) is int and beats > 0,
                "explicit interval drift")
        requested.extend(first + ordinal for ordinal in range(beats))
    if not request_accept_stalls:
        request_accept_stalls = (0,) * len(requested)
    if not response_extra_stalls:
        response_extra_stalls = (0,) * len(requested)
    require(len(request_accept_stalls) == len(requested) and
            len(response_extra_stalls) == len(requested) and
            all(type(v) is int and v >= 0 for v in
                tuple(request_accept_stalls) + tuple(response_extra_stalls)),
            "explicit stall vector drift")
    completed: list[int] = []
    previous = None
    for eligible, accept_stall, response_stall in zip(
            requested, request_accept_stalls, response_extra_stalls):
        accept = eligible if previous is None else max(eligible, previous + 1)
        accept += accept_stall
        complete = accept + 1 + response_stall
        completed.append(complete)
        previous = complete
    return completed


def bounded_source_oracle() -> dict[str, Any]:
    """Exercise exact closed form, gaps, overlap and stall dominance."""
    checks = 0
    for first0 in range(4):
        for beats0 in range(1, 7):
            for gap in range(7):
                for beats1 in range(1, 7):
                    intervals = [(first0, beats0), (first0 + gap, beats1)]
                    explicit = explicit_beat_simulation(intervals)
                    first = advance_zero_stall_ii2(None, first0, beats0)
                    second = advance_zero_stall_ii2(
                        first.last_completed_cycle, first0 + gap, beats1)
                    require((first.first_completed_cycle,
                             first.last_completed_cycle,
                             second.first_completed_cycle,
                             second.last_completed_cycle) ==
                            (explicit[0], explicit[beats0 - 1],
                             explicit[beats0], explicit[-1]),
                            "closed-form/explicit mismatch")
                    checks += 1

    fixture = [(0, 3), (11, 2), (11, 4)]
    zero = explicit_beat_simulation(fixture)
    accept_stalls = [0] * len(zero); accept_stalls[2] = 5
    response_stalls = [0] * len(zero); response_stalls[-2] = 7
    stalled_accept = explicit_beat_simulation(fixture, accept_stalls, ())
    stalled_response = explicit_beat_simulation(fixture, (), response_stalls)
    require(all(a >= b for a, b in zip(stalled_accept, zero)) and
            all(a >= b for a, b in zip(stalled_response, zero)) and
            stalled_accept != zero and stalled_response != zero,
            "adversarial stalls escaped or improved lower bound")

    replay = IntervalReplay(3, 12, AXES)
    starts = {
        "candidate": (0, 5, 10),
        "strongest_zero": (0, 9, 20),
        "same_coordinate_bit": (0, 7, 17),
    }
    for task in range(3):
        for axis in AXES:
            replay.consume_interval(axis, task, starts[axis][task])
    terminal = replay.finalize()
    require(terminal["axes"]["candidate"]["beats"] == 12 and
            terminal["component_schedule_ratios"][
                "strongest_zero_over_candidate"]["ratio_decimal"] > 1.0 and
            terminal["expanded_beats"] == 0,
            "bounded multi-axis terminal drift")

    quota_counts = {}
    quota_sum = 0
    for task in range(TASKS):
        begin, end = floor_quota(task, TASKS, EVENTS_PER_AXIS)
        count = end - begin
        quota_counts[count] = quota_counts.get(count, 0) + 1
        quota_sum += count
    require(quota_counts == {87: 616_896, 88: 195_264} and
            quota_sum == EVENTS_PER_AXIS, "production quota proof drift")

    return {
        "schema": "m1169_c1_ii2_interval_replay_bounded_oracle_v1",
        "status": "PASS_M1169_BOUNDED_EXPLICIT_AND_INTERVAL_ORACLE__FRESH_HAMMER_REQUIRED",
        "closed_form_explicit_cases": checks,
        "adversarial_gap_cases_included": True,
        "adversarial_accept_stall_detected": True,
        "adversarial_response_stall_detected": True,
        "production_floor_quota_counts": quota_counts,
        "production_beats_per_axis": quota_sum,
        "bounded_terminal": terminal,
        "production_schedule_opened": False,
        "production_records_consumed": 0,
        "production_beats_expanded": 0,
        "m1161_result_consumed": False,
        "claim_boundary": terminal["claim_boundary"],
    }


def main() -> None:
    preflight = source_preflight()
    oracle = bounded_source_oracle()
    print(json.dumps({"preflight": preflight, "oracle": oracle},
                     indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
