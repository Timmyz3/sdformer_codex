#!/usr/bin/env python3
"""M1072 source-only row-provenance-bound exact-1RW full replay library.

The sole production cycle entry is ``iter_canonical_full_replay_results()``.
It accepts no arguments, reads only the exact canonical M410 file through
``pread``, rederives every cycle-driving field from those bytes, and exposes no
result until the whole file, coverage, provenance and final file identity have
closed.  Caller records can only enter a read-only validation function that
reopens and rederives the canonical bytes; they can never drive cycles.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field, replace
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
M1064_PATH = HERE / "run_m1064_c1_frozen_exact_1rw_replay_source.py"
M1065 = HW / "reviews/m1065_m1064_c1_frozen_exact_1rw_source_hammer_r1_20260830"
CONTRACT = HW / "contracts/m1072_m1065_c1_row_provenance_exact_1rw_source_contract_r1_20260830.json"
CONTRACT_SIDECAR = Path(str(CONTRACT) + ".sha256")
CONTRACT_OUTER = Path(str(CONTRACT) + ".sha256.seal.sha256")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

M1064_SHA = "ecf2625ae60a9f7848fc32b852b67f8efd3439c5fb24b9904ef397d39aafed09"
M1065_ID = (
    "3dedad8967263ccc65d8709da98c66883a54db89a643763a0bf6280724c64268",
    "1f100fee161fc87cc455c6f16373cddda12bd33c18189a001d05ef39170dcb7d",
    "c20db372093934d548a7d50c7772b78833ddf1ebb5f3d06d2000da57cae33277",
)
CONTRACT_SHA = "017d5254346e54a24c3082cb9cd17f61e19d4f895ef6366e55345784e6b4ec03"
CONTRACT_SIDECAR_SHA = "84568e3920d73ed1c053fef57a007b5aede91319a78d7f625b07b50f8e9e0951"
CONTRACT_OUTER_SHA = "4c55522fbf34cd430d93a4b84d6d3a17eb2f6068fa7027b898d187dd9b6fac1d"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

CONTRACT_ROOT_KEYS = {
    "schema", "status", "launch_now", "max_attempts_now", "scope",
    "frozen_authorities", "unique_production_boundary", "canonical_row_reader",
    "record_provenance", "coverage", "preserved_resources",
    "m1073_required_attacks", "future_execution", "source_identity",
    "claim_boundary",
}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def exact_int(value: Any) -> bool:
    return type(value) is int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            RuntimeError("nonfinite JSON: " + token)
        ),
    )


def verify_flat(directory: Path, identity: tuple[str, str, str]) -> None:
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(directory.is_dir() and not directory.is_symlink() and
            (sha256(review), sha256(manifest), sha256(outer)) == identity,
            "sealed authority identity drift")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        require(sha256(directory / name.lstrip("*")) == expected,
                "sealed authority member drift")
    expected, name = outer.read_text(encoding="utf-8").split()
    require(name == "SHA256SUMS" and expected == sha256(manifest),
            "sealed authority outer drift")


def load_frozen(path: Path, expected_sha: str, module_name: str):
    require(path.is_file() and not path.is_symlink() and sha256(path) == expected_sha,
            module_name + " identity drift")
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None, "cannot load " + module_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M1064 = load_frozen(M1064_PATH, M1064_SHA, "m1072_frozen_m1064")
M1016 = M1064.M1016
M1056 = M1064.M1056

ROWS = M1064.ROWS
ROWS_SHA = M1064.ROWS_SHA
ROWS_BYTES = M1064.RAW_ROWS * M1064.BYTES_PER_LINE
DESIGNS = M1064.DESIGNS
TASKS = M1064.TASKS
TASKS_PER_SAMPLE = M1064.TASKS_PER_SAMPLE
SAMPLES = M1064.SAMPLES
EXPECTED_SERVICE_DIGEST = M1064.EXPECTED_SERVICE_DIGEST
EXPECTED_SERVICES = M1064.EXPECTED_SERVICES
EXPECTED_CANDIDATE_PARENT = {
    "reads": 131_926_088,
    "writes": 79_581_608,
    "forwards": 13_717_024,
    "work_cycles": 409_734_336,
}


def validate_sealed_contract(contract_path: Path = CONTRACT) -> dict[str, Any]:
    supplied = Path(contract_path)
    require(supplied.absolute() == CONTRACT.absolute() and
            supplied.is_file() and not supplied.is_symlink(),
            "only canonical nonsymlink contract accepted")
    require(sha256(CONTRACT) == CONTRACT_SHA and
            sha256(CONTRACT_SIDECAR) == CONTRACT_SIDECAR_SHA and
            sha256(CONTRACT_OUTER) == CONTRACT_OUTER_SHA,
            "contract double-seal identity drift")
    expected, name = CONTRACT_SIDECAR.read_text(encoding="utf-8").split()
    require(expected == CONTRACT_SHA and name == CONTRACT.name,
            "contract sidecar content drift")
    expected, name = CONTRACT_OUTER.read_text(encoding="utf-8").split()
    require(expected == CONTRACT_SIDECAR_SHA and name == CONTRACT_SIDECAR.name,
            "contract outer content drift")
    value = strict_json(CONTRACT)
    require(type(value) is dict and set(value) == CONTRACT_ROOT_KEYS and
            value["schema"] ==
            "m1072_m1065_c1_row_provenance_exact_1rw_source_contract_v1" and
            value["status"] ==
            "PASS_M1072_SEALED_SOURCE_CONTRACT__M1073_REQUIRED_NO_LAUNCH" and
            value["launch_now"] is False and
            exact_int(value["max_attempts_now"]) and
            value["max_attempts_now"] == 0,
            "contract exact schema/content drift")
    return {
        "status": "PASS_M1072_EXACT_DOUBLE_SEALED_CONTRACT",
        "contract_sha256": CONTRACT_SHA,
        "sidecar_sha256": CONTRACT_SIDECAR_SHA,
        "outer_seal_file_sha256": CONTRACT_OUTER_SHA,
    }


def validate_frozen_authorities() -> dict[str, Any]:
    validate_sealed_contract()
    verify_flat(M1065, M1065_ID)
    review = strict_json(M1065 / "review.json")
    require(review.get("status") ==
            "STOP_M1065_M1064_C1_FROZEN_EXACT_1RW_SOURCE_HAMMER" and
            review.get("claim_boundary", {}).get(
                "m1066_full_execution_release_authorized") is False and
            sha256(M1064_PATH) == M1064_SHA and sha256(DOCS359) == DOCS359_SHA,
            "M1065/M1064/docs authority drift")
    return {"status": "PASS_M1072_FROZEN_AUTHORITIES"}


def _fd_signature(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _sha256_fd(fd: int, size: int) -> str:
    require(exact_int(fd) and exact_int(size) and fd >= 0 and size >= 0,
            "invalid fd hash coordinate")
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        count = min(1 << 20, size - offset)
        block = os.pread(fd, count, offset)
        require(len(block) == count, "short pread while hashing canonical rows")
        digest.update(block)
        offset += count
    return digest.hexdigest()


class CanonicalRowReader:
    """Zero-argument, no-follow canonical file reader with before/after identity."""

    def __init__(self) -> None:
        require(ROWS.absolute() ==
                (HW / "results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/m410r2_h67_q32_runtime_rows_32.memh").absolute(),
                "canonical row path drift")
        info = os.lstat(ROWS)
        require(stat.S_ISREG(info.st_mode) and not ROWS.is_symlink() and
                info.st_size == ROWS_BYTES, "canonical row path/type/size drift")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        self._fd = os.open(ROWS, flags)
        opened = os.fstat(self._fd)
        require(stat.S_ISREG(opened.st_mode) and opened.st_size == ROWS_BYTES and
                (opened.st_dev, opened.st_ino) == (info.st_dev, info.st_ino),
                "opened canonical row identity drift")
        self._signature = _fd_signature(opened)
        require(_sha256_fd(self._fd, ROWS_BYTES) == ROWS_SHA,
                "canonical row initial SHA drift")
        self._closed = False
        self._reads = 0

    def _verify_unchanged(self, final_hash: bool) -> None:
        require(not self._closed and _fd_signature(os.fstat(self._fd)) == self._signature,
                "canonical row file drift during replay")
        if final_hash:
            require(_sha256_fd(self._fd, ROWS_BYTES) == ROWS_SHA,
                    "canonical row final SHA drift")

    def raw_for_task(self, task_id: int) -> tuple[bytes, int]:
        self._verify_unchanged(final_hash=False)
        sample, operator, chunk, partition = M1064.decode_task_id(task_id)
        count = M1064.row_count_for_chunk(chunk)
        phase = (sample * M1064.OPERATORS + operator) * M1064.PARTITIONS + partition
        offset = (phase * M1064.ROWS_PER_PHASE + chunk * M1064.ROW_TILE) * M1064.BYTES_PER_LINE
        raw = os.pread(self._fd, count * M1064.BYTES_PER_LINE, offset)
        require(len(raw) == count * M1064.BYTES_PER_LINE,
                "short pread for canonical task row bytes")
        lines = raw.splitlines()
        require(len(lines) == count and all(len(line) == 8 for line in lines),
                "canonical task row text/reorder geometry drift")
        self._reads += 1
        return raw, offset

    def derive(self, task_id: int) -> "ProvenanceRecord":
        raw, offset = self.raw_for_task(task_id)
        return derive_record_from_exact_raw(task_id, raw, offset)

    def close(self) -> None:
        if not self._closed:
            try:
                self._verify_unchanged(final_hash=True)
            finally:
                os.close(self._fd)
                self._closed = True

    def __enter__(self) -> "CanonicalRowReader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


@dataclass(frozen=True)
class ProvenanceRecord:
    task_id: int
    sample: int
    operator: int
    chunk: int
    partition: int
    file_offset: int
    row: int
    row_count: int
    raw_row_bytes_sha256: str
    masks_le16_sha256: str
    shared_preprocess_cycles: int
    works: Mapping[str, int]
    parents: Mapping[str, Mapping[str, int]]
    common_receipt: Mapping[str, Any]
    provenance_sha256: str


def _canonical_provenance_payload(record_values: Mapping[str, Any]) -> bytes:
    return json.dumps(record_values, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def derive_record_from_exact_raw(task_id: int, raw: bytes,
                                 file_offset: int) -> ProvenanceRecord:
    """Derive every cycle field from exact bytes; not a production cycle entry."""
    sample, operator, chunk, partition = M1064.decode_task_id(task_id)
    count = M1064.row_count_for_chunk(chunk)
    require(type(raw) is bytes and len(raw) == count * M1064.BYTES_PER_LINE and
            exact_int(file_offset) and file_offset >= 0,
            "raw provenance geometry drift")
    lines = raw.splitlines()
    require(len(lines) == count and all(len(line) == 8 for line in lines),
            "raw provenance row text drift")
    masks = [int(line, 16) & 0xFFFF for line in lines]
    masks_le16 = b"".join(int(mask).to_bytes(2, "little") for mask in masks)
    receipt = M1016.common_receipt(task_id, count)
    M1064.validate_receipt_exact(receipt, task_id, count)
    works: dict[str, int] = {}
    parents: dict[str, dict[str, int]] = {}
    preprocess_values = []
    masks_array = np.asarray(masks, dtype=np.uint16)
    for design in DESIGNS:
        work, parent = M1016.parent_for_design(design, masks_array)
        work = int(work)
        works[design] = work
        parents[design] = {
            "reads": int(parent.get("reads", 0)),
            "writes": int(parent.get("writes", 0)),
            "forwards": int(parent.get("forwards", 0)),
            "work_cycles": work,
        }
        preprocess_values.append(int(
            M1016.preprocess_for_design(design, masks_array, receipt)
        ))
    shared_preprocess = max(preprocess_values)
    row = task_id % 64
    values = {
        "task_id": task_id,
        "coordinate": [sample, operator, chunk, partition],
        "file_offset": file_offset,
        "row": row,
        "row_count": count,
        "raw_row_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "masks_le16_sha256": hashlib.sha256(masks_le16).hexdigest(),
        "shared_preprocess_cycles": shared_preprocess,
        "works": works,
        "parents": parents,
        "common_receipt": receipt,
    }
    provenance = hashlib.sha256(_canonical_provenance_payload(values)).hexdigest()
    return ProvenanceRecord(
        task_id, sample, operator, chunk, partition, file_offset, row, count,
        values["raw_row_bytes_sha256"], values["masks_le16_sha256"],
        shared_preprocess, works, parents, receipt, provenance,
    )


def record_payload(record: ProvenanceRecord) -> dict[str, Any]:
    require(type(record) is ProvenanceRecord, "record exact type required")
    return {
        "task_id": record.task_id,
        "coordinate": [record.sample, record.operator, record.chunk, record.partition],
        "file_offset": record.file_offset,
        "row": record.row,
        "row_count": record.row_count,
        "raw_row_bytes_sha256": record.raw_row_bytes_sha256,
        "masks_le16_sha256": record.masks_le16_sha256,
        "shared_preprocess_cycles": record.shared_preprocess_cycles,
        "works": dict(record.works),
        "parents": {name: dict(record.parents[name]) for name in DESIGNS},
        "common_receipt": dict(record.common_receipt),
    }


def validate_record_shape(record: ProvenanceRecord) -> None:
    require(type(record) is ProvenanceRecord and all(exact_int(value) for value in (
        record.task_id, record.sample, record.operator, record.chunk, record.partition,
        record.file_offset, record.row, record.row_count,
        record.shared_preprocess_cycles,
    )), "record exact scalar/type drift")
    sample, operator, chunk, partition = M1064.decode_task_id(record.task_id)
    require((record.sample, record.operator, record.chunk, record.partition) ==
            (sample, operator, chunk, partition) and
            record.row == record.task_id % 64 and
            record.row_count == M1064.row_count_for_chunk(chunk) and
            record.shared_preprocess_cycles >= 0 and
            type(record.works) is dict and set(record.works) == set(DESIGNS) and
            all(exact_int(value) and value >= 0 for value in record.works.values()) and
            type(record.parents) is dict and set(record.parents) == set(DESIGNS),
            "record coordinate/work population drift")
    for design in DESIGNS:
        parent = record.parents[design]
        require(type(parent) is dict and set(parent) == {
            "reads", "writes", "forwards", "work_cycles"
        } and all(exact_int(value) and value >= 0 for value in parent.values()) and
                parent["work_cycles"] == record.works[design],
                "record parent schema/work drift")
    M1064.validate_receipt_exact(record.common_receipt, record.task_id,
                                 record.row_count)
    require(all(type(value) is str and len(value) == 64 and
                all(char in "0123456789abcdef" for char in value)
                for value in (record.raw_row_bytes_sha256,
                              record.masks_le16_sha256,
                              record.provenance_sha256)),
            "record digest schema drift")
    require(hashlib.sha256(_canonical_provenance_payload(record_payload(record))).hexdigest()
            == record.provenance_sha256, "record provenance self-digest drift")


def validate_external_records_against_frozen(
    records: Sequence[ProvenanceRecord],
) -> dict[str, Any]:
    """Read-only validator. External records never enter the cycle scheduler."""
    require(type(records) in (list, tuple) and len(records) > 0,
            "external validation requires nonempty record sequence")
    validate_frozen_authorities()
    with CanonicalRowReader() as reader:
        for record in records:
            validate_record_shape(record)
            expected = reader.derive(record.task_id)
            require(record == expected,
                    "external record differs from canonical row rederivation")
    return {
        "status": "PASS_M1072_EXTERNAL_RECORDS_MATCH_CANONICAL_ROWS",
        "records": len(records),
        "cycle_entry": False,
    }


@dataclass(init=False)
class ProvenanceCoverage:
    next_task_id: int
    next_sample_commit: int
    services: dict[str, Counter]
    service_digests: dict[str, Any]
    execution_digest: Any
    parents: dict[str, Counter]
    raw_rows: int

    def __init__(self) -> None:
        self.next_task_id = 0
        self.next_sample_commit = 0
        self.services = {name: Counter() for name in DESIGNS}
        self.service_digests = {name: hashlib.sha256() for name in DESIGNS}
        self.execution_digest = hashlib.sha256()
        self.parents = {name: Counter() for name in DESIGNS}
        self.raw_rows = 0

    def consume_internal(self, record: ProvenanceRecord) -> None:
        validate_record_shape(record)
        require(record.task_id == self.next_task_id,
                "missing, duplicate or out-of-order provenance task")
        canonical_receipt = json.dumps(record.common_receipt, sort_keys=True,
                                       separators=(",", ":")).encode()
        for design in DESIGNS:
            self.services[design].update(record.common_receipt["counts"])
            self.service_digests[design].update(canonical_receipt)
            self.parents[design].update(record.parents[design])
        self.execution_digest.update(record.provenance_sha256.encode())
        self.raw_rows += record.row_count
        self.next_task_id += 1
        if self.next_task_id % TASKS_PER_SAMPLE == 0:
            sample = self.next_sample_commit
            commit = {
                "task": TASKS + sample,
                "counts": {resource: M1064.COMMIT_CYCLES_PER_SAMPLE
                           if resource == "commit" else 0
                           for resource in M1064.RESOURCES},
                "sample_commit": sample,
            }
            encoded = json.dumps(commit, sort_keys=True, separators=(",", ":")).encode()
            for design in DESIGNS:
                self.services[design].update(commit["counts"])
                self.service_digests[design].update(encoded)
            self.execution_digest.update(hashlib.sha256(encoded).hexdigest().encode())
            self.next_sample_commit += 1

    def proof(self) -> dict[str, Any]:
        service_rows = {name: self.service_digests[name].hexdigest()
                        for name in DESIGNS}
        candidate_parent = dict(self.parents["candidate"])
        baseline_parent_zero = all(
            self.parents[name][key] == 0
            for name in ("strongest_zero", "same_coordinate_bit")
            for key in ("reads", "writes", "forwards")
        )
        checks = {
            "exact_tasks": self.next_task_id == TASKS,
            "exact_sample_commits": self.next_sample_commit == SAMPLES,
            "exact_raw_rows": self.raw_rows == M1064.RAW_ROWS,
            "exact_services": all(dict(self.services[name]) == EXPECTED_SERVICES
                                  for name in DESIGNS),
            "exact_service_digest": all(value == EXPECTED_SERVICE_DIGEST
                                        for value in service_rows.values()),
            "candidate_parent_conservation": all(
                candidate_parent.get(key, 0) == value
                for key, value in EXPECTED_CANDIDATE_PARENT.items()
            ),
            "baseline_parent_accesses_zero": baseline_parent_zero,
            "baseline_work_equal": self.parents["strongest_zero"]["work_cycles"] ==
                                   self.parents["same_coordinate_bit"]["work_cycles"],
        }
        return {
            "schema": "m1072_row_provenance_coverage_v1",
            "checks": checks,
            "full_coverage_pass": all(checks.values()),
            "service_digests": service_rows,
            "execution_provenance_digest_sha256": self.execution_digest.hexdigest(),
            "parent": {name: dict(self.parents[name]) for name in DESIGNS},
            "caller_supplied_coverage_or_digest": False,
        }


@dataclass
class _DesignStream:
    last_write: dict[tuple[int, int], int] = field(default_factory=dict)
    previous_start: int | None = None
    previous_effective_end: int | None = None
    delayed_accesses: int = 0
    nominal_excess_accesses: int = 0

    def consume_internal(self, plan: Any) -> None:
        require(type(plan) is M1056.TaskPlan, "internal plan type drift")
        if self.previous_start is None:
            start = plan.preprocess_cycles
        else:
            require(self.previous_effective_end is not None,
                    "stream pipeline state drift")
            start = max(self.previous_effective_end,
                        self.previous_start + plan.preprocess_cycles) + 2
        result = M1056.schedule_task(plan, start, self.last_write)
        self.previous_start = start
        self.previous_effective_end = result.effective_work_end
        self.delayed_accesses += result.delayed_accesses
        self.nominal_excess_accesses += result.nominal_excess_accesses

    def finish_sample(self) -> dict[str, int]:
        require(self.previous_effective_end is not None, "empty design sample")
        return {
            "cycles_after_commit": self.previous_effective_end + 2 +
                                   M1064.COMMIT_CYCLES_PER_SAMPLE,
            "delayed_accesses": self.delayed_accesses,
            "nominal_excess_accesses": self.nominal_excess_accesses,
        }


def iter_canonical_full_replay_results() -> Iterator[dict[str, Any]]:
    """Only production cycle entry. It takes no caller-controlled arguments."""
    validate_frozen_authorities()
    capacity = M1064.derive_physical_capacity()
    coverage = ProvenanceCoverage()
    sample_rows = []
    with CanonicalRowReader() as reader:
        streams = {name: _DesignStream() for name in DESIGNS}
        for task_id in range(TASKS):
            record = reader.derive(task_id)
            coverage.consume_internal(record)
            for design in DESIGNS:
                plan = M1056.TaskPlan(
                    record.task_id, record.shared_preprocess_cycles,
                    record.works[design], record.row,
                )
                streams[design].consume_internal(plan)
            if (task_id + 1) % TASKS_PER_SAMPLE == 0:
                sample = task_id // TASKS_PER_SAMPLE
                sample_rows.append({
                    "sample": sample,
                    "first_task_id": sample * TASKS_PER_SAMPLE,
                    "last_task_id": task_id,
                    "designs": {name: streams[name].finish_sample()
                                for name in DESIGNS},
                })
                streams = {name: _DesignStream() for name in DESIGNS}
    proof = coverage.proof()
    require(proof["full_coverage_pass"] and len(sample_rows) == SAMPLES,
            "full row-provenance coverage failed")
    full = {
        "schema": "m1072_canonical_full_exact_1rw_replay_result_v1",
        "status": "PASS_M1072_CANONICAL_FULL_REPLAY_PENDING_RESULT_HAMMER",
        "samples": sample_rows,
        "coverage": proof,
        "capacity": capacity,
        "claim_boundary": {
            "capacity_only_214912B_admitted": False,
            "matched_cycles_admitted": False,
            "speedup_admitted": False,
            "rtl_cycles": False,
            "paper_ppa_ready": False,
            "independent_result_hammer_required": True,
        },
    }
    # No result is visible until the with-block performs final stat+SHA checks.
    yield full


def small_oracle() -> dict[str, Any]:
    validate_frozen_authorities()
    require(len(inspect.signature(iter_canonical_full_replay_results).parameters) == 0,
            "production iterator acquired caller arguments")
    with CanonicalRowReader() as reader:
        real = reader.derive(0)
        raw1, offset1 = reader.raw_for_task(1)
    require(real.shared_preprocess_cycles == 210 and
            real.works == {
                "candidate": 1664,
                "strongest_zero": 4392,
                "same_coordinate_bit": 4392,
            }, "frozen task-0 anchor drift")

    forged_works = replace(
        real,
        shared_preprocess_cycles=0,
        works={"candidate": 0, "strongest_zero": 999_999,
               "same_coordinate_bit": 999_999},
        parents={
            "candidate": {"reads": 0, "writes": 0, "forwards": 0,
                          "work_cycles": 0},
            "strongest_zero": {"reads": 0, "writes": 0, "forwards": 0,
                               "work_cycles": 999_999},
            "same_coordinate_bit": {"reads": 0, "writes": 0, "forwards": 0,
                                    "work_cycles": 999_999},
        },
    )
    forged_works = replace(
        forged_works,
        provenance_sha256=hashlib.sha256(
            _canonical_provenance_payload(record_payload(forged_works))
        ).hexdigest(),
    )
    zero_raw = b"00000000\n" * 64
    zero = derive_record_from_exact_raw(0, zero_raw, real.file_offset)
    reordered = derive_record_from_exact_raw(0, raw1, real.file_offset)
    require(zero.shared_preprocess_cycles == 146 and
            all(value == 0 for value in zero.works.values()) and
            reordered.raw_row_bytes_sha256 != real.raw_row_bytes_sha256,
            "M1065 zero/reorder attack construction drift")
    rejected = {}
    for name, record in (("work_0_999999", forged_works),
                         ("all_zero_masks", zero),
                         ("row_reorder", reordered)):
        try:
            validate_external_records_against_frozen([record])
        except RuntimeError:
            rejected[name] = True
        else:
            rejected[name] = False
    require(all(rejected.values()), "M1065 provenance forgery accepted")

    empty = ProvenanceCoverage().proof()
    require(not empty["full_coverage_pass"], "empty provenance coverage admitted")
    capacity = M1064.derive_physical_capacity()
    require(capacity["derived_total_bytes"] == 214_912 and
            capacity["capacity_only_214912B_admitted"] is False,
            "capacity boundary drift")
    return {
        "schema": "m1072_row_provenance_source_small_oracle_v1",
        "status": "PASS_M1072_SMALL_ORACLE__M1073_REQUIRED_NO_FULL_REPLAY",
        "task0": {
            "raw_row_bytes_sha256": real.raw_row_bytes_sha256,
            "masks_le16_sha256": real.masks_le16_sha256,
            "provenance_sha256": real.provenance_sha256,
            "shared_preprocess_cycles": real.shared_preprocess_cycles,
            "works": dict(real.works),
            "parents": {name: dict(real.parents[name]) for name in DESIGNS},
        },
        "m1065_attacks_rejected": rejected,
        "file_identity_before_and_after_first_reads": True,
        "production_iterator_arguments": 0,
        "full_iterator_called": False,
        "capacity": capacity,
        "claim_boundary": {
            "source_only": True,
            "m1073_passed": False,
            "launch_now": False,
            "full_51840000_replay": False,
            "full_trace_port_feasibility": False,
            "capacity_only_214912B_admitted": False,
            "matched_cycles_admitted": False,
            "speedup_admitted": False,
            "rtl_cycles": False,
            "paper_ppa_ready": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate-contract", action="store_true")
    args = parser.parse_args(argv)
    require(args.self_test ^ args.validate_contract,
            "select exactly one source-only action")
    value = small_oracle() if args.self_test else validate_sealed_contract()
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
