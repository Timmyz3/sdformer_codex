#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Receipt-blind, bounded M1085 audit of the ended M1074 failure.

The probe opens only canonical task rows 0..207, stopping at the first invalid
event dependency.  It never calls the M1072 production iterator, M1074
runner/engine, a full replay, GPU, EDA, or remote services.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
M1074_ENGINE = HW / "system_simulator/scripts/execute_m1074_m1072_c1_full_exact_1rw_one_shot.py"
M1074_RUNNER = HW / "system_simulator/scripts/run_m1074_m1072_c1_full_exact_1rw_one_shot.sh"
M1072_SOURCE = HW / "system_simulator/scripts/run_m1072_c1_row_provenance_exact_1rw_source.py"
M1064_SOURCE = HW / "system_simulator/scripts/run_m1064_c1_frozen_exact_1rw_replay_source.py"
M1056_SOURCE = HW / "system_simulator/scripts/run_m1056_c1_exact_1rw_arbitration_replay_source.py"
CONTRACT = HW / "contracts/m1074_m1073_m1072_c1_full_exact_1rw_one_shot_source_contract_r1_20260830.json"
CONTRACT_SIDECAR = Path(str(CONTRACT) + ".sha256")
CONTRACT_OUTER = Path(str(CONTRACT) + ".sha256.seal.sha256")
M1075 = HW / "reviews/m1075_m1074_c1_full_exact_1rw_one_shot_source_hammer_r1_20260830"
ATTEMPT = HW / "results/.m1074_m1072_c1_full_exact_1rw_replay_attempt_consumed"
RESULT = HW / "results/m1074_m1072_c1_full_exact_1rw_replay_r1_20260830"
QUARANTINE = HW / "results/m1074_m1072_c1_full_exact_1rw_replay_r1_20260830.failed_or_incomplete.2844327.quarantine"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "engine": "90ead8cb4a0196114dbb6c51f4fe9e042fee1bf2816855687327221c8c3274e5",
    "runner": "cec9da5f0faaef281c705f46b41020fe6572be0f98317f6f8ab29f5e1a090812",
    "m1072": "879712a59785acc79776990236884582431adea81103a222d5415905199a1e4c",
    "m1064": "ecf2625ae60a9f7848fc32b852b67f8efd3439c5fb24b9904ef397d39aafed09",
    "m1056": "95e276a7afe7a049faa2b967bed1431762c72a5e0b521c3e9857121ece5c816f",
    "contract": "5d385afe4c0b5875568b19f903d1ed56a224d79790c206a62a28fdeefb967a67",
    "contract_sidecar": "259532aa54f20c02bfbb04c2e3722b9fb821ba82b4b9d025c45bc8b5fd3c348d",
    "contract_outer": "b2892273abf602787f8d857d97ef9d9a5c9282fa380ba8787fbd9e55c15214aa",
    "m1075_review": "3394efc09302e5a95fc24c3f4dff0e23299448aabb8c5d3de91f6ab17ee45421",
    "m1075_manifest": "97c40d6974a6ca73d1cd35830e4baf70e6ec678b7ee3ee7f102fa8ed27ec3dbf",
    "m1075_outer": "8eb2c07d3ed7e0616226684e04180198a5aad7eb85ddfd3dcea0a758df2a12da",
    "attempt_json": "5f3addebb82a5b493e118ca0709986ae2113a22fc5011f700b30681c273824b4",
    "attempt_manifest": "a320ff65e8fa4b963b2f7e81beede7a257e1c92a91302e206ae06f119fd8c2f6",
    "attempt_outer": "2e7b51e2d19449c53da65bc01ff73b93bdae46c0627b2b1ff3ff09f0d20625c0",
    "run_started": "19652ecc8b99e66bdf0af01abe3b52ef14812b1d4f1bee8ec995965f09aa366a",
    "traceback": "30fd4b25eac7c684c15b7dfb4573fb43235be207e6fca4fe87f7205a7dd05e25",
    "failure": "6255b45d48e73a046a68eb72bb2cdd6baf5335fe58cc98b8c3ef488c61ada4cf",
    "quarantine_manifest": "d0290d1ab3711b075988d951e1812b5fb0a1b85bfa1df11dc062ba67a2f87a9c",
    "quarantine_outer": "fbb2b631d5b1aee56b1e80523a224cecc9b9f79bcaf4cb2b3e07f33b35d263a2",
    "target_raw": "e8636aaf63033f5c8520c127205c519a0da3f3b4e599888dcb8fe5569446f9e9",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def pairs(rows):
        value = {}
        for key, item in rows:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + token)))


def verify_flat(directory: Path) -> dict[str, str]:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(directory.is_dir() and not directory.is_symlink(),
            "flat authority absent/symlink")
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        require(name not in listed, "duplicate flat manifest member")
        listed.add(name)
        target = directory / name
        require(target.is_file() and not target.is_symlink() and sha(target) == digest,
                "flat authority member drift: " + name)
    require(outer.read_text(encoding="utf-8").split() ==
            [sha(manifest), "SHA256SUMS"], "flat outer content drift")
    return {"manifest_sha256": sha(manifest),
            "outer_seal_file_sha256": sha(outer)}


def verify_atomic(directory: Path, expected_members: set[str]) -> dict[str, str]:
    seal = directory / ".m1074_atomic_seal"
    manifest = seal / "SHA256SUMS"
    outer = seal / "SHA256SUMS.seal.sha256"
    require(directory.is_dir() and not directory.is_symlink() and
            seal.is_dir() and not seal.is_symlink(), "atomic directory absent/symlink")
    payload_members = {path.relative_to(directory).as_posix()
                       for path in directory.rglob("*") if path.is_file() and
                       ".m1074_atomic_seal" not in path.parts}
    require(payload_members == expected_members, "atomic payload population drift")
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        require(name not in listed and name in expected_members,
                "atomic manifest population drift")
        listed.add(name)
        target = directory / name
        require(target.is_file() and not target.is_symlink() and sha(target) == digest,
                "atomic member drift: " + name)
    require(listed == expected_members and
            outer.read_text(encoding="utf-8").split() ==
            [sha(manifest), "SHA256SUMS"], "atomic outer/member closure drift")
    return {"manifest_sha256": sha(manifest),
            "outer_seal_file_sha256": sha(outer)}


def load_m1072():
    require(sha(M1072_SOURCE) == EXPECTED["m1072"], "M1072 source drift")
    spec = importlib.util.spec_from_file_location("m1085_frozen_m1072", M1072_SOURCE)
    require(spec is not None and spec.loader is not None, "cannot load M1072")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bounded_first_invalid_dependency(module) -> dict[str, Any]:
    """Read the minimum prefix and stop exactly at the first invalid dependency."""
    info = os.lstat(module.ROWS)
    require(stat.S_ISREG(info.st_mode) and not module.ROWS.is_symlink() and
            info.st_size == module.ROWS_BYTES, "canonical row path/type/size drift")
    fd = os.open(module.ROWS, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    opened = 0
    try:
        for task_id in range(module.TASKS):
            sample, operator, chunk, partition = module.M1064.decode_task_id(task_id)
            count = module.M1064.row_count_for_chunk(chunk)
            phase = ((sample * module.M1064.OPERATORS + operator) *
                     module.M1064.PARTITIONS + partition)
            offset = ((phase * module.M1064.ROWS_PER_PHASE +
                       chunk * module.M1064.ROW_TILE) * module.M1064.BYTES_PER_LINE)
            raw = os.pread(fd, count * module.M1064.BYTES_PER_LINE, offset)
            opened += 1
            require(len(raw) == count * module.M1064.BYTES_PER_LINE,
                    "short bounded task read")
            record = module.derive_record_from_exact_raw(task_id, raw, offset)
            module.validate_record_shape(record)
            for design in module.DESIGNS:
                plan = module.M1056.TaskPlan(
                    record.task_id, record.shared_preprocess_cycles,
                    record.works[design], record.row)
                events = module.M1056.nominal_task_events(plan, 0, {})
                for event in events:
                    for dependency in event.dependencies:
                        if not bool(dependency.event_id) or dependency.delay_cycles < 0:
                            return {
                                "task_id": task_id,
                                "coordinate": [sample, operator, chunk, partition],
                                "design": design,
                                "shared_preprocess_cycles": record.shared_preprocess_cycles,
                                "work_cycles": record.works[design],
                                "row": record.row,
                                "row_count": count,
                                "file_offset": offset,
                                "raw_row_bytes_sha256": hashlib.sha256(raw).hexdigest(),
                                "raw_unique_lines": sorted(set(line.decode()
                                                               for line in raw.splitlines())),
                                "task_rows_opened": opened,
                                "full_canonical_file_hashed_by_audit": False,
                                "group": event.group,
                                "logical_bank": event.logical_bank,
                                "event_id": event.event_id,
                                "event_op": event.op,
                                "dependency_event_id": dependency.event_id,
                                "dependency_event_id_bool": bool(dependency.event_id),
                                "dependency_delay_cycles": dependency.delay_cycles,
                                "dependency_delay_exact_int":
                                    type(dependency.delay_cycles) is int,
                                "failing_dependency_field": "delay_cycles",
                                "prior_tasks_all_dependency_valid": task_id == opened - 1,
                            }
    finally:
        os.close(fd)
    raise RuntimeError("no invalid dependency in bounded scan")


def main() -> dict[str, Any]:
    identity = {
        "m1074_engine_sha256": sha(M1074_ENGINE),
        "m1074_runner_sha256": sha(M1074_RUNNER),
        "m1072_source_sha256": sha(M1072_SOURCE),
        "m1064_source_sha256": sha(M1064_SOURCE),
        "m1056_source_sha256": sha(M1056_SOURCE),
        "contract_sha256": sha(CONTRACT),
        "contract_sidecar_sha256": sha(CONTRACT_SIDECAR),
        "contract_outer_seal_file_sha256": sha(CONTRACT_OUTER),
        "m1075_review_sha256": sha(M1075 / "review.json"),
        "m1075_manifest_sha256": sha(M1075 / "SHA256SUMS"),
        "m1075_outer_seal_file_sha256": sha(M1075 / "SHA256SUMS.seal.sha256"),
        "attempt_json_sha256": sha(ATTEMPT / "attempt.json"),
        "run_started_sha256": sha(QUARANTINE / "partial_result/RUN_STARTED.json"),
        "traceback_sha256": sha(QUARANTINE / "partial_result/traceback.log"),
        "failure_sha256": sha(QUARANTINE / "failure.json"),
        "docs359_sha256": sha(DOCS359),
    }
    require(identity == {
        "m1074_engine_sha256": EXPECTED["engine"],
        "m1074_runner_sha256": EXPECTED["runner"],
        "m1072_source_sha256": EXPECTED["m1072"],
        "m1064_source_sha256": EXPECTED["m1064"],
        "m1056_source_sha256": EXPECTED["m1056"],
        "contract_sha256": EXPECTED["contract"],
        "contract_sidecar_sha256": EXPECTED["contract_sidecar"],
        "contract_outer_seal_file_sha256": EXPECTED["contract_outer"],
        "m1075_review_sha256": EXPECTED["m1075_review"],
        "m1075_manifest_sha256": EXPECTED["m1075_manifest"],
        "m1075_outer_seal_file_sha256": EXPECTED["m1075_outer"],
        "attempt_json_sha256": EXPECTED["attempt_json"],
        "run_started_sha256": EXPECTED["run_started"],
        "traceback_sha256": EXPECTED["traceback"],
        "failure_sha256": EXPECTED["failure"],
        "docs359_sha256": EXPECTED["docs359"],
    }, "M1074 failure identity drift")

    require(CONTRACT_SIDECAR.read_text(encoding="utf-8").split() ==
            [EXPECTED["contract"], CONTRACT.name] and
            CONTRACT_OUTER.read_text(encoding="utf-8").split() ==
            [EXPECTED["contract_sidecar"], CONTRACT_SIDECAR.name],
            "contract double seal content drift")
    m1075_seal = verify_flat(M1075)
    require(m1075_seal == {"manifest_sha256": EXPECTED["m1075_manifest"],
                            "outer_seal_file_sha256": EXPECTED["m1075_outer"]},
            "M1075 seal drift")
    attempt_seal = verify_atomic(ATTEMPT, {"attempt.json"})
    quarantine_seal = verify_atomic(QUARANTINE, {
        "failure.json", "partial_result/RUN_STARTED.json",
        "partial_result/traceback.log"})
    require(attempt_seal == {"manifest_sha256": EXPECTED["attempt_manifest"],
                              "outer_seal_file_sha256": EXPECTED["attempt_outer"]} and
            quarantine_seal == {
                "manifest_sha256": EXPECTED["quarantine_manifest"],
                "outer_seal_file_sha256": EXPECTED["quarantine_outer"]},
            "M1074 runtime atomic seal drift")

    namespace = sorted(path.name for path in (HW / "results").iterdir()
                       if "m1074" in path.name.lower())
    require(namespace == sorted([ATTEMPT.name, QUARANTINE.name]) and
            not RESULT.exists(), "M1074 namespace/retry drift")
    attempt = strict_json(ATTEMPT / "attempt.json")
    started = strict_json(QUARANTINE / "partial_result/RUN_STARTED.json")
    failure = strict_json(QUARANTINE / "failure.json")
    require(attempt == {
        "automatic_retry": False,
        "canonical_rows_opened_or_hashed_before_attempt": False,
        "m1072_source_sha256": EXPECTED["m1072"],
        "m1075_outer_seal_file_sha256": EXPECTED["m1075_outer"],
        "maximum_attempts": 1,
        "schema": "m1074_full_exact_1rw_attempt_v1",
        "status": "CONSUMED_BEFORE_CANONICAL_ROWS_OPEN",
    } and started == {
        "m1072_source_sha256": EXPECTED["m1072"],
        "m1075_outer_seal_file_sha256": EXPECTED["m1075_outer"],
        "status": "ATTEMPT_ALREADY_CONSUMED__OPEN_ROWS_NEXT",
    } and failure == {
        "attempt_consumed": True, "automatic_retry": False,
        "phase": "FULL_812160_TASK_51840000_ROW_EXACT_1RW_REPLAY",
        "return_code": 1, "schema": "m1074_failure_quarantine_v1",
        "status": "FAILED_OR_INTERRUPTED__NO_RETRY",
    }, "M1074 attempt/failure content drift")

    traceback_text = (QUARANTINE / "partial_result/traceback.log").read_text(
        encoding="utf-8")
    require("RuntimeError: invalid event dependency" in traceback_text and
            "dependency.validate()" in traceback_text and
            "M1056.schedule_task(plan, start, self.last_write)" in traceback_text,
            "traceback call-chain drift")
    m1056_text = M1056_SOURCE.read_text(encoding="utf-8")
    require("bool(self.event_id) and self.delay_cycles >= 0" in m1056_text and
            "write_cycle = min(work_end, read_cycle + span - 1)" in m1056_text and
            "dependencies=(Dependency(read_id, write_cycle - read_cycle),)" in
            m1056_text, "M1056 failure predicate/generator drift")

    module = load_m1072()
    first = bounded_first_invalid_dependency(module)
    require(first == {
        "task_id": 207, "coordinate": [0, 0, 0, 207],
        "design": "candidate", "shared_preprocess_cycles": 146,
        "work_cycles": 0, "row": 15, "row_count": 64,
        "file_offset": 5589000,
        "raw_row_bytes_sha256": EXPECTED["target_raw"],
        "raw_unique_lines": ["00420000"], "task_rows_opened": 208,
        "full_canonical_file_hashed_by_audit": False,
        "group": 0, "logical_bank": 1, "event_id": "t207:b1:W",
        "event_op": "WRITE", "dependency_event_id": "t207:b1:R",
        "dependency_event_id_bool": True, "dependency_delay_cycles": -1,
        "dependency_delay_exact_int": True,
        "failing_dependency_field": "delay_cycles",
        "prior_tasks_all_dependency_valid": True,
    }, "first invalid dependency rederivation drift")

    classification = {
        "canonical_trace_illegal": False,
        "validator_bool_or_empty_id_issue": False,
        "m1072_record_or_provenance_bug": False,
        "m1056_event_geometry_bug_exposed_by_m1072": True,
        "explanation": (
            "The exact canonical task is schema-valid and rederives candidate "
            "work_cycles=0. Its dependency id is nonempty and delay has exact int "
            "type, but M1056 emits all sixteen bank read/write pairs even for a "
            "zero-work task. For bank 1, read_cycle=start+1 while write_cycle is "
            "clamped to work_end=start, producing delay_cycles=-1."
        ),
    }
    require(not RESULT.exists() and sha(ATTEMPT / "attempt.json") ==
            EXPECTED["attempt_json"] and sha(DOCS359) == EXPECTED["docs359"],
            "audit changed frozen runtime evidence")

    return {
        "schema": "m1085_m1074_c1_full_replay_failure_audit_mechanical_v1",
        "status": "PASS_M1085_M1074_FAILURE_AUDIT__ADDITIVE_ZERO_WORK_REPAIR_ALLOWED__M1074_DO_NOT_RETRY",
        "identity": identity,
        "sealed_authority_recomputation": {
            "m1075": m1075_seal, "attempt": attempt_seal,
            "quarantine": quarantine_seal,
        },
        "runtime_state": {
            "namespace": namespace, "attempt_status": attempt["status"],
            "attempt_consumed": True, "maximum_attempts": 1,
            "automatic_retry": False, "result_absent": True,
            "work_absent_after_quarantine": True,
            "quarantine_status": failure["status"],
            "m1074_retry_allowed": False,
        },
        "bounded_probe": {
            "production_iterator_called": False, "m1074_runner_called": False,
            "m1074_engine_execute_full_called": False,
            "full_replay_executed": False, "gpu_eda_remote_used": False,
            "canonical_task_rows_opened": 208,
            "stopped_at_first_invalid_dependency": True,
        },
        "first_invalid_dependency": first,
        "root_cause_classification": classification,
        "claim_boundary": {
            "paper_citable_result": False, "matched_cycles_admitted": False,
            "speedup_admitted": False, "full_trace_port_feasibility": False,
            "rtl_cycles": False, "paper_ppa_ready": False,
        },
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
