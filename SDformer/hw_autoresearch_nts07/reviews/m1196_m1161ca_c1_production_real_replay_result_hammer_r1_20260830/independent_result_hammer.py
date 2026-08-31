#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1196 fresh-author read-only result hammer for M1161CA.

This checker streams the sealed 836 MB schedule once, but never replays the
212,559,552 producer rows.  It independently checks schedule SHA/bytes/count,
task/axis order, ordinals and record provenance, then cross-checks every
terminal producer/validator count and digest against the independently sealed
M1148CA authority.  It writes only a new review directory.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import tempfile
from typing import Any

HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
RESULT = HW / "results/m1161ca_c1_production_real_replay_r1_20260830"
ATTEMPT = HW / "results/.m1161ca_c1_production_real_replay_attempt_consumed"
SOURCE = HW / "system_simulator/scripts/run_m1161ca_c1_production_real_replay_driver_one_shot_source.py"
CONTRACT = HW / "contracts/m1161ca_c1_production_real_replay_driver_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1161ca_c1_production_real_replay_driver_author_receipt_r1_20260830"
M1164 = HW / "reviews/m1164_m1161ca_c1_production_real_replay_driver_hammer_r1_20260830"
M1141 = HW / "results/m1141ca_c1_production_schedule_release_r1_20260830"
SCHEDULE = M1141 / "m1141ca_per_task_schedule_records.jsonl"
RELEASE = M1141 / "m1141ca_schedule_release.json"
M1148 = HW / "results/m1148ca_c1_production_expected_digest_compiler_r1_20260830"
AUTHORITY = M1148 / "expected_digest_authority.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

SOURCE_SHA = "d7ffb8dbab289e83fd8a32f4ed5244cd005a4b6d0785b586df932fd6a97ee20d"
CONTRACT_ID = (
    "93471a51d5f9d9270ece1629688b10b0cf88047abed9a5e7b6e71048cd63ef63",
    "89345e94816a72f3672920d4eb9c984afa085789fc47213ef8c981b824f437ea",
    "5c7fdc73e9a69211fea340fa6c9862d19531df551176aa0351f6c914a2f12272",
)
AUTHOR_ID = (
    "7d2dbd0f7019f7bf9f462bf9e5fb0575313a896b29d9bda7a673d6699a4b763c",
    "b6361e95b5e4f16414e923a0c1b56928028c81102a3be051670e0da4988f97bf",
    "9d3e6dbad63761090eb60e06fb4dfa220690a3651643e26c96f9948ec10f71f4",
)
M1164_ID = (
    "a2d53aa814da8c29ee335b1782594253c7cf9f69d1312984c207e540c2899f0b",
    "d1153fa67a8af96d679a7a04475f34b7c7aeacd69cb0f59c3078b4d18e97c84f",
    "7e6d8e23326b65e59db4347887da1f31822a0cc7ae8fcca59ca191309cac57c4",
)
RESULT_MANIFEST_SHA = "b6c2be64d8cb32fcf0c31ae44070b5efdcb10d0db2661dddb0ec2c4cc3733198"
RESULT_OUTER_SHA = "7bb4ff9dc40a9764d9312c1639a022756305c0170c483854a84c02d2a6cf5b5c"
ATTEMPT_ID = (
    "54d0e884ee5be2bcca7c30c500464ab7208762f9cb3b7f7d2b3098c2bdaf9681",
    "5f0e79bfbc4fd165b1d65ab1dc0d499677ef482ca138e141594bcbf1cda10510",
    "492d89e87f22e972e4df95764b3ad0d95a65e17fb844c946e02fd96d1430bf1f",
)
SCHEDULE_SHA = "4d4e0e6396ac1061aca7ada142bc2761bf12a785e5373640a28503e3d73a0a81"
SCHEDULE_BYTES = 836_268_740
RELEASE_SHA = "4c4d264a9ac1e084c8c0acf0a6d150140f95ee96ee967b038ea4c1eefcc2b58c"
M1141_ID = (
    "852b48c0d8098ef69a619925f82a8e1a308e87f2faf9ea76becabf51d52caace",
    "0b6549ce38a62bcb22e8a97d0c038860f5698fabc0d9bff162dc6af95d4f043a",
)
AUTHORITY_SHA = "c45fd835db7fddca268a8891051a5d24bf9492806c6e3610b8e52b8730e705b2"
M1148_ID = (
    "6fc0048c84409cc7afc114f540ad17c83a2a00d0d1db19b0684881d8f2dadf5f",
    "98d69e2799af300b2babe72ac3cceb97f3ecc9a435ac7d12c6c7b8fdd13979d1",
)
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
M1016_SHA = "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa"
M1102_SHA = "95bd50aebcc473ab69cdea6ccf27d54743c89926c5e0f31199dc469ced9bf7cc"
M1137_SHA = "9ec640ae8c9fa75f9cbf706e15d2d26a4233def77e5be4d67e94c084347b20a6"
AUTHORITY_ID = "a53f0141ff9f01b32ed8920c0c3fc10a2d70848773e9b99e02b8905ea05a6fbf"

AXES = ("candidate", "strongest_zero", "same_coordinate_bit")
TASKS = 812_160
RECORDS = 2_436_480
EVENTS_AXIS = 70_853_184
EVENTS = 212_559_552
BYTES_AXIS = 9_069_207_552
NATIVE_AXIS = 566_825_472
FIELDS = {
    "axis", "chunk", "operator", "partition", "requested_cycle_first",
    "sample", "schedule_record_provenance_sha256",
    "source_task_provenance_sha256", "task_sequence_ordinal",
}
EXPECTED_MAKESPAN = {
    "candidate": 434_146_781,
    "strongest_zero": 752_971_318,
    "same_coordinate_bit": 752_971_318,
}
EXPECTED_ROW_DIGEST = {
    "candidate": "a5d9e99da453045ea4e2158d1bb032660c020caa7f67551843d6f4ca72b3232a",
    "strongest_zero": "b066959e789d093b2da4b09613cbfd3da7348317d90fbccf5c0e8c7118d548f6",
    "same_coordinate_bit": "1a85689aa21b0625d62b79bc1e3eb1edceeb9f82de8f028239c96926696f3247",
}


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


def regular(path: Path, expected: str | None = None) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(), "not regular: " + str(path))
    if expected is not None:
        require(sha256(path) == expected, "SHA drift: " + str(path))


def manifest_tree(directory: Path, manifest_sha: str, outer_sha: str,
                  skip_hash: set[str] | None = None) -> dict[str, str]:
    require(directory.is_dir() and not directory.is_symlink(), "tree drift")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    regular(manifest, manifest_sha); regular(outer, outer_sha)
    require(outer.read_text(encoding="ascii").split() == [manifest_sha, "SHA256SUMS"],
            "outer seal content drift")
    rows: dict[str, str] = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]) is not None,
                "manifest syntax")
        name = fields[1].lstrip("*")
        rel = Path(name)
        require(name not in rows and name == rel.as_posix() and not rel.is_absolute()
                and ".." not in rel.parts, "manifest member")
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
    require(actual == set(rows), "exact member set drift")
    for name, expected in rows.items():
        if name not in (skip_hash or set()):
            regular(directory / name, expected)
    return rows


def review_tree(path: Path, identity: tuple[str, str, str]) -> dict[str, Any]:
    rows = manifest_tree(path, identity[1], identity[2])
    require(rows.get("review.json") == identity[0], "review identity")
    return strict_json(path / "review.json")


def double_file(path: Path, identity: tuple[str, str, str]) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    regular(path, identity[0]); regular(side, identity[1]); regular(outer, identity[2])
    require(side.read_text(encoding="ascii").split() == [identity[0], path.name],
            "side seal content")
    require(outer.read_text(encoding="ascii").split() == [identity[1], side.name],
            "double outer content")


def task_index(sample: int, operator: int, chunk: int, partition: int) -> int:
    require(0 <= sample < 10 and 0 <= operator < 4 and 0 <= chunk < 47 and
            0 <= partition < 432, "coordinate range")
    return (((sample * 4 + operator) * 47 + chunk) * 432 + partition)


def u64(value: int) -> bytes:
    return struct.pack(">Q", value)


def provenance(row: dict[str, Any]) -> str:
    payload = b"".join((
        b"M1139CA_SCHEDULE_RECORD\x00\x01", bytes.fromhex(M1016_SHA),
        bytes.fromhex(M1102_SHA), bytes.fromhex(M1137_SHA),
        struct.pack(">B", AXES.index(row["axis"])),
        u64(row["task_sequence_ordinal"]), u64(row["sample"]),
        u64(row["operator"]), u64(row["chunk"]), u64(row["partition"]),
        u64(row["requested_cycle_first"]),
        bytes.fromhex(row["source_task_provenance_sha256"]),
    ))
    return hashlib.sha256(payload).hexdigest()


def validate_schedule_row(row: Any, record_index: int,
                          triplet_source: str | None) -> str:
    require(type(row) is dict and set(row) == FIELDS, "schedule exact fields")
    axis = AXES[record_index % 3]
    ordinal = record_index // 3
    require(row["axis"] == axis and type(row["task_sequence_ordinal"]) is int and
            row["task_sequence_ordinal"] == ordinal, "axis/ordinal order")
    ints = (row["sample"], row["operator"], row["chunk"], row["partition"],
            row["requested_cycle_first"])
    require(all(type(value) is int and value >= 0 for value in ints), "integer field")
    require(task_index(*ints[:4]) == ordinal, "ordinal geometry")
    source = row["source_task_provenance_sha256"]
    require(type(source) is str and re.fullmatch(r"[0-9a-f]{64}", source) is not None,
            "source provenance")
    if triplet_source is not None:
        require(source == triplet_source, "triplet source provenance")
    require(row["schedule_record_provenance_sha256"] == provenance(row),
            "record provenance")
    return source


def scan_schedule() -> dict[str, Any]:
    regular(SCHEDULE)
    require(SCHEDULE.stat().st_size == SCHEDULE_BYTES, "schedule bytes")
    digest = hashlib.sha256()
    counts = {axis: 0 for axis in AXES}
    first = {axis: None for axis in AXES}
    last = {axis: None for axis in AXES}
    records = 0
    triplet_source: str | None = None
    with SCHEDULE.open("rb") as stream:
        for record_index, line in enumerate(stream):
            digest.update(line)
            row = strict_loads(line)
            if record_index % 3 == 0:
                triplet_source = None
            triplet_source = validate_schedule_row(row, record_index, triplet_source)
            axis = row["axis"]
            counts[axis] += 1
            value = row["requested_cycle_first"]
            if first[axis] is None:
                first[axis] = value
            last[axis] = value
            records += 1
    require(records == RECORDS and digest.hexdigest() == SCHEDULE_SHA,
            "schedule terminal count/SHA")
    require(counts == {axis: TASKS for axis in AXES}, "schedule axis counts")
    require(first == {axis: 0 for axis in AXES}, "schedule first coordinates")
    require(last == {"candidate": 434_146_693,
                     "strongest_zero": 752_971_230,
                     "same_coordinate_bit": 752_971_230}, "schedule last coordinates")
    return {"bytes": SCHEDULE_BYTES, "records": records, "sha256": digest.hexdigest(),
            "records_per_axis": counts, "first_requested_cycle": first,
            "last_requested_cycle_first": last,
            "ordinals": [0, TASKS - 1], "axis_order": list(AXES)}


def validate_result(result: dict[str, Any], receipt: dict[str, Any],
                    resources: dict[str, Any], authority: dict[str, Any],
                    schedule_scan: dict[str, Any]) -> dict[str, Any]:
    require(result["schema"] == "m1161ca_c1_production_real_replay_terminal_r1_v1" and
            result["status"] == "PASS_REAL_M1137_PRODUCER_TO_M1135_VALIDATOR_REPLAY__RESULT_HAMMER_REQUIRED",
            "result status")
    require(result["claim_boundary"] == {
                "different_author_result_hammer_required": True,
                "real_producer_replay": True,
                "rtl_cycle_or_system_speedup": False,
                "traffic_energy_or_paper_ppa": False,
                "weight_service_schedule_cycles": True,
            }, "result claim boundary")
    require(result["sealed_schedule"] == {"bytes": SCHEDULE_BYTES,
            "records": RECORDS, "sha256": SCHEDULE_SHA}, "sealed schedule receipt")
    require(result["sealed_schedule"] == {key: schedule_scan[key]
            for key in ("bytes", "records", "sha256")}, "schedule independent scan")
    require(result["events_emitted"] == EVENTS and result["per_event_output_written"] is False
            and result["retained_schedule_event_row_or_key_history"] is False,
            "result O(axes) scope")
    driver = result["driver_terminal"]
    require(driver["scope"] == "production" and driver["records"] == RECORDS and
            driver["tasks_per_axis"] == TASKS and
            driver["records_per_axis"] == {axis: TASKS for axis in AXES} and
            driver["retained_schedule_event_row_or_key_history"] is False and
            driver["state_complexity"] == "O(axes + axes*24)", "driver terminal")
    producer = driver["m1137c_terminal"]
    validator = producer["m1135c_terminal"]
    require(producer["authority_scope"] == "production" and
            producer["tasks_per_axis"] == TASKS and
            producer["events_per_axis"] == {axis: EVENTS_AXIS for axis in AXES} and
            producer["retained_rows_events_or_key_history"] is False and
            producer["state_complexity"] == "O(axes + axes*24)", "producer terminal")
    require(authority["authority_id_sha256"] == AUTHORITY_ID and
            validator["authority_id_sha256"] == AUTHORITY_ID and
            validator["authority_scope"] == "production" and
            validator["state_complexity"] == "O(axes + axes*24)", "authority binding")
    rows = result["row_terminal"]
    require(rows["cycle_claim"] == "weight-service schedule coordinates only; not RTL or system cycles"
            and rows["retained_schedule_event_row_or_key_history"] is False and
            rows["state_complexity"] == "O(axes)", "row scope")
    for axis in AXES:
        expected = authority["axes"][axis]
        vaxis = validator["axes"][axis]
        require(expected == {"bytes": BYTES_AXIS, "events": EVENTS_AXIS,
                "native_activations": NATIVE_AXIS, "records": TASKS,
                "stall_cycles": 0, "stalled_transactions": 0}, "authority axis")
        require(vaxis["events"] == expected["events"] and
                vaxis["bytes"] == expected["bytes"] and
                vaxis["native_activations"] == expected["native_activations"] and
                vaxis["stall_cycles"] == 0 and vaxis["stalled_transactions"] == 0 and
                vaxis["digest"] == authority["expected_digest_by_axis"][axis] and
                vaxis["first_beat"] == 0 and vaxis["first_transaction"] == 0 and
                vaxis["last_beat"] == EVENTS_AXIS - 1 and
                vaxis["last_transaction"] == EVENTS_AXIS - 1, "validator axis")
        raxis = rows["axes"][axis]
        require(raxis["rows"] == EVENTS_AXIS and raxis["bytes"] == BYTES_AXIS and
                raxis["native_activations"] == NATIVE_AXIS and
                raxis["first_requested_cycle"] == 0 and
                raxis["first_scheduled_cycle"] == 0 and
                raxis["last_requested_cycle"] == EXPECTED_MAKESPAN[axis] - 1 and
                raxis["last_scheduled_cycle"] == EXPECTED_MAKESPAN[axis] - 1 and
                raxis["max_scheduled_cycle"] == EXPECTED_MAKESPAN[axis] - 1 and
                raxis["weight_service_makespan_coordinate"] == EXPECTED_MAKESPAN[axis] and
                raxis["stall_cycles"] == 0 and raxis["stalled_transactions"] == 0 and
                raxis["row_digest_sha256"] == EXPECTED_ROW_DIGEST[axis], "row axis")
    require(receipt["attempt_consumed"] is True and receipt["automatic_retry"] is False and
            receipt["event_output_written"] is False and
            receipt["source_sha256"] == SOURCE_SHA and
            receipt["source_preflight"]["authority_id_sha256"] == AUTHORITY_ID and
            receipt["source_preflight"]["m1141_records_sha256_expected"] == SCHEDULE_SHA and
            receipt["claim_boundary"] == {"paper_citable_performance": False,
                "producer_replay_and_1rw_schedule_receipt": True,
                "rtl_cycle_or_system_speedup": False}, "receipt scope")
    require(resources == {"cpu_seconds": resources["cpu_seconds"],
            "events_replayed": EVENTS, "input_bytes_streamed": SCHEDULE_BYTES,
            "input_records_streamed": RECORDS, "max_rss_kib": resources["max_rss_kib"],
            "retained_schedule_event_row_or_key_history": False,
            "schema": "m1161ca_c1_production_real_replay_resources_r1_v1",
            "state_complexity": "O(axes + axes*24) plus one JSON line and one row",
            "wall_seconds": resources["wall_seconds"]} and
            all(type(resources[key]) in (int, float) and math.isfinite(resources[key]) and
                resources[key] > 0 for key in ("cpu_seconds", "max_rss_kib", "wall_seconds")),
            "runtime resources")
    speedup_bit = EXPECTED_MAKESPAN["same_coordinate_bit"] / EXPECTED_MAKESPAN["candidate"]
    speedup_zero = EXPECTED_MAKESPAN["strongest_zero"] / EXPECTED_MAKESPAN["candidate"]
    require(math.isclose(speedup_bit, 1.734370381062436, rel_tol=0, abs_tol=1e-15) and
            speedup_bit == speedup_zero, "independent speedup arithmetic")
    return {"candidate": EXPECTED_MAKESPAN["candidate"],
            "same_coordinate_bit": EXPECTED_MAKESPAN["same_coordinate_bit"],
            "strongest_zero": EXPECTED_MAKESPAN["strongest_zero"],
            "candidate_vs_same_coordinate_bit": speedup_bit,
            "candidate_vs_strongest_zero": speedup_zero}


def expect_result_reject(result: dict[str, Any], receipt: dict[str, Any],
                         resources: dict[str, Any], authority: dict[str, Any],
                         schedule_scan: dict[str, Any], mutate) -> str:
    attacked = copy.deepcopy(result)
    mutate(attacked)
    try:
        validate_result(attacked, receipt, resources, authority, schedule_scan)
    except BaseException as error:
        return type(error).__name__ + ": " + str(error)
    raise Failure("result mutation escaped")


def write_review(review: dict[str, Any], mechanical: dict[str, Any], attacks: dict[str, str]) -> None:
    require(not (HERE / "SHA256SUMS").exists(), "review already sealed")
    payloads = {
        "RUN_COMPLETE.txt": b"PASS_M1196_M1161CA_DIFFERENT_AUTHOR_RESULT_HAMMER__WEIGHT_SERVICE_SCHEDULE_COORDINATES_ADMITTED\n",
        "review.json": (json.dumps(review, indent=2, sort_keys=True) + "\n").encode(),
        "mechanical_checks.json": (json.dumps(mechanical, indent=2, sort_keys=True) + "\n").encode(),
        "mutation_checks.json": (json.dumps(attacks, indent=2, sort_keys=True) + "\n").encode(),
        "review.md": (
            "# M1196 / M1161CA production replay result hammer\n\n"
            "PASS (99/100). The exact sealed 2,436,480-record, 836,268,740-byte schedule was independently streamed and its SHA, task-major/axis-minor ordering, ordinals, coordinates, source triplets and record provenance all closed. The real M1137C producer to M1135C validator terminals close all three axes at 812,160 tasks, 70,853,184 rows and 566,825,472 native activations per axis, with no retained event output. Candidate/bit/zero 1RW weight-service makespans are 434,146,781 / 752,971,318 / 752,971,318, giving independently recomputed 1.734370381x. This admits a production real-producer replay and weight-service schedule-coordinate comparison only; it is not RTL or system speedup, traffic/energy, PPA, or paper headline evidence.\n"
        ).encode(),
    }
    for name, data in payloads.items():
        (HERE / name).write_bytes(data)
    members = ["RUN_COMPLETE.txt", "independent_result_hammer.py", "mechanical_checks.json",
               "mutation_checks.json", "review.json", "review.md"]
    manifest = "".join(f"{sha256(HERE / name)}  {name}\n" for name in members)
    (HERE / "SHA256SUMS").write_text(manifest, encoding="ascii")
    (HERE / "SHA256SUMS.seal.sha256").write_text(
        f"{sha256(HERE / 'SHA256SUMS')}  SHA256SUMS\n", encoding="ascii")


def main() -> None:
    regular(DOCS359, DOCS359_SHA); regular(SOURCE, SOURCE_SHA)
    double_file(CONTRACT, CONTRACT_ID)
    author = review_tree(AUTHOR, AUTHOR_ID)
    m1164 = review_tree(M1164, M1164_ID)
    require(author["subject"]["sha256"] == SOURCE_SHA and
            author["status"] == "PASS_M1161CA_SOURCE_AND_BOUNDED_LIVE_HOOK__DIFFERENT_AUTHOR_HAMMER_REQUIRED",
            "author chain")
    require(m1164["subject"]["sha256"] == SOURCE_SHA and
            m1164["status"] == "PASS_M1164_M1161CA_DIFFERENT_AUTHOR_SOURCE_HAMMER__EXACTLY_ONE_PRODUCTION_LAUNCH_AUTHORIZED_AFTER_FRESH_PREFLIGHT" and
            m1164["authorization"]["automatic_retry"] is False,
            "M1164 chain")
    attempt_rows = manifest_tree(ATTEMPT, ATTEMPT_ID[1], ATTEMPT_ID[2])
    require(attempt_rows == {"attempt.json": ATTEMPT_ID[0]}, "attempt exact members")
    attempt = strict_json(ATTEMPT / "attempt.json")
    require(attempt == {"authority_id_sha256": AUTHORITY_ID, "automatic_retry": False,
            "m1137_source_sha256": M1137_SHA,
            "m1141_records_sha256_expected": SCHEDULE_SHA,
            "schedule_opened_before_attempt": False,
            "schema": "m1161ca_c1_production_real_replay_attempt_r1_v1",
            "source_sha256": SOURCE_SHA,
            "status": "M1161CA_SINGLE_ATTEMPT_CONSUMED__NO_AUTOMATIC_RETRY"},
            "attempt identity")
    result_rows = manifest_tree(RESULT, RESULT_MANIFEST_SHA, RESULT_OUTER_SHA)
    require(result_rows == {"RUN_COMPLETE.txt": "21b1500f40c7c9baa55b3c4dd99ddf805215c8fe101ad7c4a35a131d0933b6d1",
            "producer_replay_terminal.json": "e681c65f25a42b7960b2a68f0709fff2b4c2bfe7d4ac7e69cccf689b9723add8",
            "receipt.json": "2e6d5ae223f4057e66916ee46c483b523ec233d4a621a070e1438e50b559c751",
            "runtime_resources.json": "9cc5cf0022812b39c437ee433a83cd45d3b07e43a468aff922b2fb291ac2b000"},
            "result exact members")
    require((RESULT / "RUN_COMPLETE.txt").read_text(encoding="ascii") ==
            "PASS_M1161CA_REAL_PRODUCER_REPLAY__RESULT_HAMMER_REQUIRED\n", "completion token")
    rows1141 = manifest_tree(M1141, M1141_ID[0], M1141_ID[1], {SCHEDULE.name})
    require(rows1141[SCHEDULE.name] == SCHEDULE_SHA and rows1141[RELEASE.name] == RELEASE_SHA,
            "M1141 identity")
    rows1148 = manifest_tree(M1148, M1148_ID[0], M1148_ID[1])
    require(rows1148[AUTHORITY.name] == AUTHORITY_SHA, "M1148 identity")
    authority = strict_json(AUTHORITY)
    release = strict_json(RELEASE)
    require(release["records"]["count"] == RECORDS and
            release["records"]["sha256"] == SCHEDULE_SHA and
            release["geometry"]["tasks"] == TASKS, "release metadata")
    schedule_scan = scan_schedule()
    result = strict_json(RESULT / "producer_replay_terminal.json")
    receipt = strict_json(RESULT / "receipt.json")
    resources = strict_json(RESULT / "runtime_resources.json")
    ratios = validate_result(result, receipt, resources, authority, schedule_scan)

    mutations = {
        "result_status": lambda x: x.__setitem__("status", "PASS"),
        "schedule_records": lambda x: x["sealed_schedule"].__setitem__("records", RECORDS - 1),
        "event_total": lambda x: x.__setitem__("events_emitted", EVENTS - 1),
        "driver_scope": lambda x: x["driver_terminal"].__setitem__("scope", "synthetic"),
        "task_count": lambda x: x["driver_terminal"].__setitem__("tasks_per_axis", TASKS - 1),
        "producer_event_count": lambda x: x["driver_terminal"]["m1137c_terminal"]["events_per_axis"].__setitem__("candidate", EVENTS_AXIS - 1),
        "validator_digest": lambda x: x["driver_terminal"]["m1137c_terminal"]["m1135c_terminal"]["axes"]["candidate"].__setitem__("digest", "0" * 64),
        "validator_last_ordinal": lambda x: x["driver_terminal"]["m1137c_terminal"]["m1135c_terminal"]["axes"]["candidate"].__setitem__("last_beat", EVENTS_AXIS),
        "native_activations": lambda x: x["row_terminal"]["axes"]["candidate"].__setitem__("native_activations", NATIVE_AXIS - 1),
        "makespan": lambda x: x["row_terminal"]["axes"]["candidate"].__setitem__("weight_service_makespan_coordinate", EXPECTED_MAKESPAN["candidate"] - 1),
        "row_digest": lambda x: x["row_terminal"]["axes"]["candidate"].__setitem__("row_digest_sha256", "f" * 64),
        "claim_boundary": lambda x: x["claim_boundary"].__setitem__("rtl_cycle_or_system_speedup", True),
    }
    attacks = {name: expect_result_reject(result, receipt, resources, authority,
                                           schedule_scan, mutate)
               for name, mutate in mutations.items()}
    try:
        strict_loads(b'{"x":1,"x":2}')
    except BaseException as error:
        attacks["duplicate_json_key"] = type(error).__name__ + ": " + str(error)
    else:
        raise Failure("duplicate JSON escaped")
    try:
        strict_loads(b'{"x":NaN}')
    except BaseException as error:
        attacks["nonfinite_json"] = type(error).__name__ + ": " + str(error)
    else:
        raise Failure("nonfinite JSON escaped")
    sample = {"axis": "candidate", "chunk": 0, "operator": 0, "partition": 0,
              "requested_cycle_first": 0, "sample": 0,
              "schedule_record_provenance_sha256": "0" * 64,
              "source_task_provenance_sha256": "0" * 64,
              "task_sequence_ordinal": 0}
    try:
        validate_schedule_row(sample, 0, None)
    except BaseException as error:
        attacks["schedule_provenance"] = type(error).__name__ + ": " + str(error)
    else:
        raise Failure("schedule provenance attack escaped")

    mechanical = {
        "docs359_sha256": sha256(DOCS359), "result_manifest_sha256": sha256(RESULT / "SHA256SUMS"),
        "result_outer_seal_file_sha256": sha256(RESULT / "SHA256SUMS.seal.sha256"),
        "attempt_outer_seal_file_sha256": sha256(ATTEMPT / "SHA256SUMS.seal.sha256"),
        "source_sha256": sha256(SOURCE), "author_outer_seal_file_sha256": AUTHOR_ID[2],
        "m1164_outer_seal_file_sha256": M1164_ID[2], "schedule_scan": schedule_scan,
        "events_total": EVENTS, "events_per_axis": EVENTS_AXIS,
        "native_activations_per_axis": NATIVE_AXIS, "event_output_written": False,
        "retained_schedule_event_row_or_key_history": False,
        "state_complexity": ["O(axes)", "O(axes + axes*24)"],
        "makespan_and_ratios": ratios, "attacks_rejected": len(attacks),
    }
    review = {
        "schema": "m1196_m1161ca_c1_production_real_replay_result_hammer_r1_v1",
        "date": "2026-08-30", "score": 99, "p0": [],
        "p1": ["Row-receipt SHA values are identity-checked from the sealed real replay; the hammer does not spend another 2.5 h regenerating 212,559,552 rows."],
        "status": "PASS_M1196_M1161CA_DIFFERENT_AUTHOR_RESULT_HAMMER__WEIGHT_SERVICE_SCHEDULE_COORDINATES_ADMITTED",
        "verdict": "ADMIT_PRODUCTION_REAL_PRODUCER_TO_VALIDATOR_REPLAY_AND_1RW_WEIGHT_SERVICE_COORDINATES_ONLY",
        "claim_boundary": {"production_real_producer_replay": True,
            "weight_service_schedule_coordinates": True, "rtl_cycles": False,
            "system_speedup": False, "traffic_or_energy": False,
            "ppa_or_timing": False, "paper_headline": False,
            "schedule_streamed_by_hammer": True, "event_rows_replayed_by_hammer": False},
        "sealed_chain": {"source_sha256": SOURCE_SHA,
            "author_outer_seal_file_sha256": AUTHOR_ID[2],
            "m1164_outer_seal_file_sha256": M1164_ID[2],
            "attempt_outer_seal_file_sha256": ATTEMPT_ID[2],
            "result_outer_seal_file_sha256": RESULT_OUTER_SHA,
            "schedule_sha256": SCHEDULE_SHA, "authority_id_sha256": AUTHORITY_ID},
        "production_evidence": {"schedule": schedule_scan,
            "tasks_per_axis": TASKS, "rows_per_axis": EVENTS_AXIS,
            "rows_total": EVENTS, "native_activations_per_axis": NATIVE_AXIS,
            "bytes_per_axis": BYTES_AXIS, "makespan_and_ratios": ratios,
            "expected_event_digests": authority["expected_digest_by_axis"],
            "row_receipt_digests": EXPECTED_ROW_DIGEST,
            "event_output_written": False,
            "retained_schedule_event_row_or_key_history": False},
        "mutation_attacks_rejected": len(attacks),
        "docs359_sha256": sha256(DOCS359),
    }
    write_review(review, mechanical, attacks)


if __name__ == "__main__":
    main()
