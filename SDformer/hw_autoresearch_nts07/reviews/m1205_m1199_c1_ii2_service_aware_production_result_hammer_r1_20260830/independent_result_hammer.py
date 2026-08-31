#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Fresh-author, read-only result hammer for the M1199 C1 II=2 replay.

The hammer independently streams and hashes all 2,436,480 sealed schedule
records, re-derives the depth-one II=2 recurrence without importing M1169,
checks the complete admission chain and immutable one-shot namespace, and
writes evidence only inside this additive M1205 review directory.
"""
from __future__ import annotations

import copy
from decimal import Decimal, getcontext
import hashlib
import json
from pathlib import Path
import re
import stat
import struct
import sys
from typing import Any

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
RESULT = HW / "results/m1199_c1_ii2_service_aware_production_replay_r1_20260830"
ATTEMPT = HW / "results/.m1199_c1_ii2_service_aware_production_replay_attempt_consumed"
SOURCE = HW / "system_simulator/scripts/run_m1199_c1_ii2_service_aware_production_consumer_one_shot_source.py"
TESTS = HW / "system_simulator/tests/test_m1199_c1_ii2_service_aware_production_consumer_source.py"
CONTRACT = HW / "contracts/m1199_c1_ii2_service_aware_production_consumer_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1199_c1_ii2_service_aware_production_consumer_source_author_receipt_r1_20260830"
M1202 = HW / "reviews/m1202_m1199_c1_ii2_service_aware_production_consumer_source_hammer_r1_20260830"
M1141 = HW / "results/m1141ca_c1_production_schedule_release_r1_20260830"
SCHEDULE = M1141 / "m1141ca_per_task_schedule_records.jsonl"
M1161 = HW / "results/m1161ca_c1_production_real_replay_r1_20260830"
M1196 = HW / "reviews/m1196_m1161ca_c1_production_real_replay_result_hammer_r1_20260830"
M1169_SOURCE = HW / "system_simulator/scripts/build_m1169_c1_ii2_service_aware_interval_replay_source.py"
M1169_CONTRACT = HW / "contracts/m1169_c1_ii2_service_aware_interval_replay_source_contract_r1_20260830.json"
M1170 = HW / "reviews/m1170_m1169_c1_ii2_service_aware_interval_replay_source_hammer_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
SOURCE_SHA = "b77bde3c15e74e6320e39ea2b0f4066ff3d8cbc7af945d88d77324f148a24768"
TESTS_SHA = "781e45dded3a17df8d098e584ef1ecc06778d4dd4f00c3a3ab0bc241e858ab64"
CONTRACT_ID = (
    "0f277f7f8f9437ce0692d5e4ce8c167d288894be6486cadef968455a2eae3ecb",
    "69eb75bf0a471316945cbdc67927d9f99467c8e245fe053284983189f7cd46d4",
    "bba87f25ace5739f780cccf0de98a101928a051d678fbb98da5f5a2b8d539015",
)
AUTHOR_ID = (
    "c7ef9bfd26a1f5f381bd4a5e9245bc83d8c81bfda3f416599a2a6de6a4b86274",
    "3a9bbe91eec177064aa64ea79389023b18e3a425369f0e143f7ee2cfa1da9935",
    "46cfdfdeab1d687da591a6c000049868e5ceeffdbf3489a5e35ac522fd56c0e4",
)
M1202_ID = (
    "1326a96dd33df37e83970b8f6a72059a6fd2a36d11160d8d8885513f93292fad",
    "6cbe730045fe2abbf43fe0a34ff6b13401e1921fa31614e045aa6bbd3fb98d82",
    "c9a6b4096bd14a2f0966b2aac73d0cb7bcb369caec3bbccd4b561aa4f49ee738",
)
RESULT_ID = (
    "cdc36eca7c2f89c3bb1681dddfbe74e940dca8c5112d584b50f40745f4dc0de8",
    "9e83c81a908e7e30910c386d2eb7a46350901f74f5796885d8ca633406d85721",
)
RESULT_MEMBERS = {
    "RUN_COMPLETE.txt": "646bb90ba3651c55b3d60616f842523dd9a15a15ec4e2363cf2cf32386d3ed14",
    "ii2_service_aware_terminal.json": "a40166b4607648cf806a807672a2d7b1b570fb1e6f391d7153285593d5bd5713",
    "receipt.json": "7649d41c083051412588925729f952fe6f48aaf1bef1d9060520204456860e82",
    "runtime_resources.json": "bd7c96535059290b175bb42a3e279a947537ef9efb7c38ab89bbffe550da07c4",
}
ATTEMPT_ID = (
    "525b8799f92604668990bf06e48205c7d7e0b550c54df2da8181dfe496e172ad",
    "f9ed86d4d8feb9c7864f323524c28be89bd2342d5cbf77cd16ae268f4a5b33a0",
    "58cc3735cc3067aa343865cd3426661e0fc5e185bd0ec4ae240b019ba85088f3",
)
M1141_ID = (
    "852b48c0d8098ef69a619925f82a8e1a308e87f2faf9ea76becabf51d52caace",
    "0b6549ce38a62bcb22e8a97d0c038860f5698fabc0d9bff162dc6af95d4f043a",
)
SCHEDULE_SHA = "4d4e0e6396ac1061aca7ada142bc2761bf12a785e5373640a28503e3d73a0a81"
SCHEDULE_BYTES = 836_268_740
M1161_ID = (
    "b6c2be64d8cb32fcf0c31ae44070b5efdcb10d0db2661dddb0ec2c4cc3733198",
    "7bb4ff9dc40a9764d9312c1639a022756305c0170c483854a84c02d2a6cf5b5c",
)
M1196_ID = (
    "7b1a8b4fa8f1e2a6c361817c65ba198f76e332f5ed09a5199b96c699e241a65e",
    "174dee393c022db03dc315266e0d90f4ba45892147d4d69b01b970ffb1f16092",
    "8b919a0ad6e6ba6638ba6c21a5fbe993dfde0097fddc327001b5c4c5543a8dd0",
)
M1169_SOURCE_SHA = "bd243ca34760757cadbf9c1104049480197f1fb77bf6ad6ec1071870250ebc4f"
M1169_CONTRACT_SHA = "275214c40e1a53b922c1db448dcedff8792f5232124fc1ea5d474360ded861dc"
M1170_ID = (
    "c52c7bb2086e2ad638b7b91656c9c21c1fe517d81fa032a158973a2867f57f16",
    "5a3d7a821190c39d4b1213517e81f240ec2cd8e1a1e557832d6c404c74291af0",
    "0e1cf625aee653b734b2e949a459fe9d8ac3c9b95d830c772a9682b5e7c3bebd",
)

AXES = ("candidate", "strongest_zero", "same_coordinate_bit")
TASKS = 812_160
RECORDS = 2_436_480
BEATS = 70_853_184
FIELDS = {
    "axis", "chunk", "operator", "partition", "requested_cycle_first",
    "sample", "schedule_record_provenance_sha256",
    "source_task_provenance_sha256", "task_sequence_ordinal",
}
EXPECTED = {
    "candidate": (434_146_781, 434_146_913, 61_003_102_963),
    "strongest_zero": (752_971_318, 752_971_491, 213_973_617_194),
    "same_coordinate_bit": (752_971_318, 752_971_491, 213_973_617_194),
}
PROV_SOURCES = (
    "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa",
    "95bd50aebcc473ab69cdea6ccf27d54743c89926c5e0f31199dc469ced9bf7cc",
    "9ec640ae8c9fa75f9cbf706e15d2d26a4233def77e5be4d67e94c084347b20a6",
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


def regular(path: Path, expected: str | None = None) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(), "not regular: " + str(path))
    if expected is not None:
        require(sha256(path) == expected, "SHA drift: " + str(path))


def strict_loads(payload: bytes) -> Any:
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key")
            output[key] = value
        return output
    return json.loads(payload.decode("utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("nonfinite JSON: " + token)))


def strict_json(path: Path) -> Any:
    return strict_loads(path.read_bytes())


def manifest_tree(directory: Path, manifest_sha: str, outer_sha: str) -> dict[str, str]:
    require(directory.is_dir() and not directory.is_symlink(), "tree drift: " + str(directory))
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    regular(manifest, manifest_sha); regular(outer, outer_sha)
    require(outer.read_text(encoding="ascii").split() == [manifest_sha, "SHA256SUMS"],
            "outer seal drift: " + str(directory))
    rows: dict[str, str] = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]) is not None,
                "manifest syntax")
        name = fields[1].lstrip("*")
        rel = Path(name)
        require(name not in rows and name == rel.as_posix() and not rel.is_absolute()
                and ".." not in rel.parts, "manifest path")
        rows[name] = fields[0]
    actual: set[str] = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "sealed symlink")
        if stat.S_ISREG(mode):
            actual.add(name)
        else:
            require(stat.S_ISDIR(mode), "sealed special member")
    require(actual == set(rows), "sealed exact member set drift: " + str(directory))
    for name, digest in rows.items():
        regular(directory / name, digest)
    return rows


def double_file(path: Path, identity: tuple[str, str, str]) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    regular(path, identity[0]); regular(side, identity[1]); regular(outer, identity[2])
    require(side.read_text(encoding="ascii").split() == [identity[0], path.name],
            "side seal drift")
    require(outer.read_text(encoding="ascii").split() == [identity[1], side.name],
            "double outer drift")


def review_tree(path: Path, identity: tuple[str, str, str], review_name: str) -> dict[str, Any]:
    rows = manifest_tree(path, identity[1], identity[2])
    require(rows.get(review_name) == identity[0], "review identity drift")
    return strict_json(path / review_name)


def u64(value: int) -> bytes:
    return struct.pack(">Q", value)


def provenance(row: dict[str, Any]) -> str:
    payload = b"".join((
        b"M1139CA_SCHEDULE_RECORD\x00\x01",
        *(bytes.fromhex(value) for value in PROV_SOURCES),
        struct.pack(">B", AXES.index(row["axis"])),
        u64(row["task_sequence_ordinal"]), u64(row["sample"]),
        u64(row["operator"]), u64(row["chunk"]), u64(row["partition"]),
        u64(row["requested_cycle_first"]),
        bytes.fromhex(row["source_task_provenance_sha256"]),
    ))
    return hashlib.sha256(payload).hexdigest()


def task_index(sample: int, operator: int, chunk: int, partition: int) -> int:
    return (((sample * 4 + operator) * 47 + chunk) * 432 + partition)


def scan_schedule() -> dict[str, Any]:
    digest = hashlib.sha256()
    states = {axis: {
        "records": 0, "beats": 0, "first_requested": None,
        "requested_makespan": 0, "first_completed": None,
        "last_completed": None, "queue": 0,
    } for axis in AXES}
    size = 0
    count = 0
    triplet_source: str | None = None
    with SCHEDULE.open("rb") as stream:
        for line in stream:
            digest.update(line); size += len(line)
            require(line.endswith(b"\n") and not line.endswith(b"\r\n"), "line framing drift")
            row = strict_loads(line)
            require(type(row) is dict and set(row) == FIELDS, "schedule field-set drift")
            axis = AXES[count % 3]
            task = count // 3
            require(row["axis"] == axis and type(row["task_sequence_ordinal"]) is int
                    and row["task_sequence_ordinal"] == task, "axis/order drift")
            ints = (row["sample"], row["operator"], row["chunk"],
                    row["partition"], row["requested_cycle_first"])
            require(all(type(value) is int and value >= 0 for value in ints),
                    "non-canonical integer")
            require(0 <= ints[0] < 10 and 0 <= ints[1] < 4 and
                    0 <= ints[2] < 47 and 0 <= ints[3] < 432 and
                    task_index(*ints[:4]) == task, "coordinate drift")
            source = row["source_task_provenance_sha256"]
            require(type(source) is str and re.fullmatch(r"[0-9a-f]{64}", source)
                    and row["schedule_record_provenance_sha256"] == provenance(row),
                    "provenance drift")
            if count % 3 == 0:
                triplet_source = source
            else:
                require(source == triplet_source, "triplet source drift")

            begin = task * BEATS // TASKS
            end = (task + 1) * BEATS // TASKS
            beats = end - begin
            state = states[axis]
            requested = row["requested_cycle_first"]
            first = requested + 1
            if state["last_completed"] is not None:
                first = max(first, state["last_completed"] + 2)
            last = first + 2 * (beats - 1)
            delay0 = first - (requested + 1)
            state["records"] += 1
            state["beats"] += beats
            state["first_requested"] = requested if state["first_requested"] is None else state["first_requested"]
            state["requested_makespan"] = max(state["requested_makespan"], requested + beats)
            state["first_completed"] = first if state["first_completed"] is None else state["first_completed"]
            state["last_completed"] = last
            state["queue"] += beats * delay0 + beats * (beats - 1) // 2
            count += 1
    require(count == RECORDS and size == SCHEDULE_BYTES and digest.hexdigest() == SCHEDULE_SHA,
            "schedule count/bytes/SHA drift")
    for axis, state in states.items():
        expected_requested, expected_service, expected_queue = EXPECTED[axis]
        require(state == {
            "records": TASKS,
            "beats": BEATS,
            "first_requested": 0,
            "requested_makespan": expected_requested,
            "first_completed": 1,
            "last_completed": expected_service - 1,
            "queue": expected_queue,
        }, "independent recurrence drift: " + axis)
    return states


def validate_terminal(value: Any, states: dict[str, Any]) -> None:
    require(type(value) is dict and value["status"] ==
            "PASS_M1199_EXACT_II2_COMPONENT_SCHEDULE__RESULT_HAMMER_REQUIRED",
            "terminal status")
    require(value["sealed_schedule"] == {
        "bytes": SCHEDULE_BYTES, "records": RECORDS, "sha256": SCHEDULE_SHA,
    }, "terminal schedule identity")
    require(value["m1169_source_sha256"] == M1169_SOURCE_SHA and
            value["m1161_result_outer_seal_file_sha256"] == M1161_ID[1] and
            value["m1170_outer_seal_file_sha256"] == M1170_ID[2] and
            value["m1196_outer_seal_file_sha256"] == M1196_ID[2],
            "terminal admission chain")
    outer = value["claim_boundary"]
    require(outer == {
        "component_weight_service_schedule_only": True,
        "different_author_result_hammer_required": True,
        "rtl_cycles_or_system_speedup": False,
        "traffic_energy_or_paper_ppa": False,
    } and value["per_event_output_written"] is False and
            value["retained_schedule_record_or_event_history"] is False,
            "terminal outer claim boundary")
    service = value["service_terminal"]
    require(service["status"] == "PASS_EXACT_INTERVAL_RECURRENCE__COMPONENT_SCHEDULE_ONLY"
            and service["expanded_beats"] == 0 and
            service["state_complexity"] == "O(axes)" and
            service["service_model"] == {
                "completed_issue_ii": 2, "outstanding_depth": 1,
                "request_stalls": 0, "response_latency_cycles": 1,
            } and service["claim_boundary"] == {
                "component_weight_service_schedule_only": True,
                "paper_citable_without_future_result_hammer": False,
                "rtl_cycles": False, "system_speedup": False,
                "traffic_energy_or_ppa": False,
            }, "service claim/model drift")
    for axis, state in states.items():
        actual = service["axes"][axis]
        require(actual == {
            "aggregate_queue_delay_cycles": state["queue"],
            "beats": state["beats"],
            "first_completed_issue_cycle": state["first_completed"],
            "first_requested_cycle": state["first_requested"],
            "ii2_service_makespan_coordinate": state["last_completed"] + 1,
            "last_completed_issue_cycle": state["last_completed"],
            "records": state["records"],
            "requested_schedule_makespan_coordinate": state["requested_makespan"],
        }, "axis terminal drift: " + axis)
    getcontext().prec = 50
    expected_ratio = Decimal(752_971_491) / Decimal(434_146_913)
    for key in ("same_coordinate_bit_over_candidate", "strongest_zero_over_candidate"):
        ratio = service["component_schedule_ratios"][key]
        require(ratio["numerator"] == 752_971_491 and ratio["denominator"] == 434_146_913
                and abs(Decimal(str(ratio["ratio_decimal"])) - expected_ratio) < Decimal("5e-16"),
                "ratio arithmetic drift")


def synthetic_attacks(terminal: dict[str, Any], states: dict[str, Any]) -> int:
    attacks = []
    for mutate in (
        lambda x: x.__setitem__("status", "PASS"),
        lambda x: x["sealed_schedule"].__setitem__("records", RECORDS - 1),
        lambda x: x["service_terminal"]["axes"]["candidate"].__setitem__(
            "ii2_service_makespan_coordinate", 434_146_912),
        lambda x: x["service_terminal"]["component_schedule_ratios"][
            "strongest_zero_over_candidate"].__setitem__("ratio_decimal", 9.0),
        lambda x: x["claim_boundary"].__setitem__("rtl_cycles_or_system_speedup", True),
        lambda x: x["service_terminal"]["service_model"].__setitem__("completed_issue_ii", 1),
    ):
        value = copy.deepcopy(terminal); mutate(value)
        try:
            validate_terminal(value, states)
        except Failure:
            attacks.append(True)
        else:
            raise Failure("synthetic terminal mutation survived")
    return len(attacks)


def main() -> None:
    regular(DOCS359, DOCS359_SHA)
    regular(SOURCE, SOURCE_SHA); regular(TESTS, TESTS_SHA)
    double_file(CONTRACT, CONTRACT_ID)
    author = review_tree(AUTHOR, AUTHOR_ID, "review.json")
    source_hammer = review_tree(M1202, M1202_ID, "review.json")
    require(author["status"].startswith("PASS_M1199_ONE_SHOT_II2") and
            source_hammer["status"].startswith("PASS_M1202_M1199_SOURCE_HAMMER") and
            source_hammer["authorization"]["exactly_one_zero_argument_production_launch"] is True,
            "source authorization drift")

    result_rows = manifest_tree(RESULT, RESULT_ID[0], RESULT_ID[1])
    require(result_rows == RESULT_MEMBERS, "result exact member identities")
    attempt_rows = manifest_tree(ATTEMPT, ATTEMPT_ID[1], ATTEMPT_ID[2])
    require(attempt_rows == {"attempt.json": ATTEMPT_ID[0]}, "attempt exact member identity")
    attempt = strict_json(ATTEMPT / "attempt.json")
    require(attempt["status"] == "M1199_SINGLE_ATTEMPT_CONSUMED__NO_AUTOMATIC_RETRY"
            and attempt["automatic_retry"] is False
            and attempt["schedule_opened_before_attempt"] is False,
            "attempt semantics drift")
    residue = list((HW / "results").glob(".m1199_c1_ii2_service_aware_production_replay_work.*"))
    residue += list((HW / "results").glob("m1199_c1_ii2_service_aware_production_replay_r1_20260830.failed_or_incomplete.*"))
    require(not residue, "work/failure residue exists")

    m1141_rows = manifest_tree(M1141, M1141_ID[0], M1141_ID[1])
    require(m1141_rows.get(SCHEDULE.name) == SCHEDULE_SHA, "M1141 schedule member drift")
    m1161_rows = manifest_tree(M1161, M1161_ID[0], M1161_ID[1])
    require(m1161_rows.get("producer_replay_terminal.json") ==
            "e681c65f25a42b7960b2a68f0709fff2b4c2bfe7d4ac7e69cccf689b9723add8",
            "M1161 terminal drift")
    m1196 = review_tree(M1196, M1196_ID, "review.json")
    regular(M1169_SOURCE, M1169_SOURCE_SHA); regular(M1169_CONTRACT, M1169_CONTRACT_SHA)
    m1170 = review_tree(M1170, M1170_ID, "hammer_result.json")
    require(m1196["status"].startswith("PASS_M1196_M1161CA") and
            m1170["status"].startswith("PASS_M1170_M1169"), "upstream hammer drift")

    states = scan_schedule()
    terminal = strict_json(RESULT / "ii2_service_aware_terminal.json")
    validate_terminal(terminal, states)
    attacks = synthetic_attacks(terminal, states)
    receipt = strict_json(RESULT / "receipt.json")
    resources = strict_json(RESULT / "runtime_resources.json")
    require(receipt["status"] ==
            "PASS_M1199_II2_PRODUCTION_CONSUMER__DIFFERENT_AUTHOR_RESULT_HAMMER_REQUIRED"
            and receipt["attempt_consumed"] is True
            and receipt["automatic_retry"] is False
            and receipt["production_schedule_opened_after_attempt"] is True
            and receipt["component_schedule_only"] is True
            and receipt["rtl_or_system_speedup"] is False,
            "receipt semantics drift")
    require(resources["input_bytes_streamed"] == SCHEDULE_BYTES and
            resources["input_records_streamed"] == RECORDS and
            resources["events_expanded"] == 0 and
            resources["retained_schedule_record_or_event_history"] is False,
            "resource accounting drift")
    require((RESULT / "RUN_COMPLETE.txt").read_text(encoding="ascii").strip() ==
            "PASS_M1199_II2_PRODUCTION_CONSUMER__RESULT_HAMMER_REQUIRED",
            "completion marker drift")

    review = {
        "schema": "m1205_m1199_c1_ii2_production_result_hammer_r1_v1",
        "status": "PASS_M1205_M1199_EXACT_II2_COMPONENT_SCHEDULE_RESULT_HAMMER",
        "date": "2026-08-30",
        "score": 100,
        "p0": [], "p1": [],
        "identity": {
            "result_manifest_sha256": RESULT_ID[0],
            "result_outer_seal_file_sha256": RESULT_ID[1],
            "terminal_sha256": RESULT_MEMBERS["ii2_service_aware_terminal.json"],
            "receipt_sha256": RESULT_MEMBERS["receipt.json"],
            "attempt_manifest_sha256": ATTEMPT_ID[1],
            "attempt_outer_seal_file_sha256": ATTEMPT_ID[2],
            "source_sha256": SOURCE_SHA,
            "contract_sha256": CONTRACT_ID[0],
            "source_hammer_outer_seal_file_sha256": M1202_ID[2],
            "docs359_sha256": DOCS359_SHA,
        },
        "full_schedule_scan": {
            "records": RECORDS, "bytes": SCHEDULE_BYTES, "sha256": SCHEDULE_SHA,
            "records_per_axis": TASKS, "beats_per_axis": BEATS,
            "schedule_record_provenance_rederived": True,
            "exact_axis_and_task_order_verified": True,
        },
        "independent_ii2_recurrence": {
            "candidate_cycles": 434_146_913,
            "same_coordinate_bit_cycles": 752_971_491,
            "strongest_zero_cycles": 752_971_491,
            "same_coordinate_bit_over_candidate": 1.7343702522192068,
            "strongest_zero_over_candidate": 1.7343702522192068,
            "candidate_queue_delay_cycles": 61_003_102_963,
            "bit_and_zero_queue_delay_cycles": 213_973_617_194,
            "expanded_beats": 0,
        },
        "one_shot_and_completeness": {
            "attempt_consumed": True, "automatic_retry": False,
            "schedule_opened_after_attempt": True,
            "failure_or_work_residue": 0,
            "result_exact_member_set": sorted(RESULT_MEMBERS),
            "completion_marker_verified": True,
        },
        "mechanical_evidence": {
            "synthetic_terminal_attacks_rejected": attacks,
            "recursive_seals_verified": 7,
            "strict_duplicate_key_json": True,
        },
        "claim_boundary": {
            "component_weight_service_schedule_only": True,
            "rtl_cycles_or_system_speedup": False,
            "traffic_energy_or_paper_ppa": False,
            "checkpoint_bound_final_system_metric": False,
        },
        "verdict": (
            "Admit 434,146,913 candidate component service cycles and the "
            "1.7343702522192068x bit/zero-over-candidate ratio only as an exact "
            "C1 weight-service schedule result. It is not RTL timing, an end-to-end "
            "cycle speedup, system speedup, traffic, energy, or paper-ready PPA."
        ),
    }
    mechanical = {
        "schema": "m1205_m1199_mechanical_checks_r1_v1",
        "status": "PASS",
        "checks": {
            "result_recursive_seal_and_exact_members": True,
            "attempt_recursive_seal_and_single_attempt": True,
            "source_contract_author_and_source_hammer": True,
            "m1141_m1161_m1196_m1169_m1170_chain": True,
            "full_schedule_hash_bytes_count_order_provenance": True,
            "independent_exact_ii2_recurrence_and_arithmetic": True,
            "claim_boundary_fail_closed": True,
            "docs359_unchanged": True,
        },
        "synthetic_attacks_rejected": attacks,
    }
    review_md = """# M1205 independent result hammer — M1199 C1 II=2 service replay

**Verdict: PASS (100/100).** The complete 836,268,740-byte schedule was
independently streamed: 2,436,480 records, SHA-256
`4d4e0e6396ac1061aca7ada142bc2761bf12a785e5373640a28503e3d73a0a81`.
Axis/task order and every schedule-record provenance were re-derived.

The independent depth-one, one-cycle-response, zero-stall II=2 recurrence gives
434,146,913 candidate cycles versus 752,971,491 for both same-coordinate bit and
strongest-zero, or 1.7343702522192068x. The one-shot attempt, complete result
tree, recursive seals, authorities, arithmetic, and six mutation rejections all
pass.

This admission is deliberately narrow: it is a component weight-service
schedule result. It is not RTL timing, end-to-end/system speedup, traffic,
energy, checkpoint-bound final-system evidence, or paper-ready PPA.
"""
    (HERE / "review.json").write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (HERE / "mechanical_checks.json").write_text(json.dumps(mechanical, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (HERE / "review.md").write_text(review_md, encoding="utf-8")
    (HERE / "RUN_COMPLETE.txt").write_text(
        "PASS_M1205_M1199_EXACT_II2_COMPONENT_SCHEDULE_RESULT_HAMMER\n", encoding="ascii")


if __name__ == "__main__":
    main()
