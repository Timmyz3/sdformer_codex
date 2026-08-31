#!/usr/bin/env python3
"""M1064 fail-closed source for a future frozen C1 exact-1RW replay.

M1064 wraps the sound M1056 arbitration kernel with a frozen production
boundary.  Task geometry, rows, shared preprocess, common-service receipts and
the 214,912-byte physical organization are all internally derived.  No caller
can set coverage, service, row, preprocess, capacity, port-feasibility or
admission values.

This is source-only.  The full iterator exists for a future M1066 runner but is
never invoked by this module's CLI or tests before an independent M1065 hammer.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
M1016_PATH = HERE / "run_m1016_c1_full_matched_address_replay.py"
M1056_PATH = HERE / "run_m1056_c1_exact_1rw_arbitration_replay_source.py"
ROWS = HW / "results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/m410r2_h67_q32_runtime_rows_32.memh"
M1057 = HW / "reviews/m1057_m1056_c1_exact_1rw_arbitration_source_hammer_r1_20260830"
CONTRACT = HW / "contracts/m1064_m1057_c1_frozen_exact_1rw_replay_source_contract_r1_20260830.json"
CONTRACT_SIDECAR = Path(str(CONTRACT) + ".sha256")
CONTRACT_OUTER = Path(str(CONTRACT) + ".sha256.seal.sha256")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

M1016_SHA = "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa"
M1056_SHA = "95e276a7afe7a049faa2b967bed1431762c72a5e0b521c3e9857121ece5c816f"
ROWS_SHA = "6e03352b89eff1955825334b4dedd991db8c975a9ef6662fe0317e73ccfa8334"
M1057_ID = (
    "350439ac9e08adb00e7dcd84fb21a5f6e08ec39c5e6834a452d0e14a8f1a360d",
    "0c49e1a24ca1620756df407f66c9d84963d1ca373885a79e96eacd8b5179e130",
    "11e4e041726c3789cdd7feec91545182357bd491b6b57601ea6564e887d8968c",
)
CONTRACT_SHA = "203392094fed8dc29bcd65abd400a21a1a7a7607686fae77c1eb19e1eefeaa24"
CONTRACT_SIDECAR_SHA = "6056723078ad7008a7f949b4b5eb70cca7a64eab554e3eb6966ed96e26d6784e"
CONTRACT_OUTER_SHA = "eb2b74ed09c575b0f48ab05eb89bf4e2778976eb479bff50d0d420886cb9f318"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
EXPECTED_SERVICE_DIGEST = "a38589ba99715b0962fb88744c03dd6019a68c72bae35d3787ca9f48eb3680ea"

SAMPLES, OPERATORS, CHUNKS, PARTITIONS = 10, 4, 47, 432
ROWS_PER_PHASE, ROW_TILE, BLOCKS = 3000, 64, 8
PHASES = SAMPLES * OPERATORS * PARTITIONS
TASKS = SAMPLES * OPERATORS * CHUNKS * PARTITIONS
TASKS_PER_SAMPLE = OPERATORS * CHUNKS * PARTITIONS
RAW_ROWS = PHASES * ROWS_PER_PHASE
BLOCK_TASKS = TASKS * BLOCKS
BYTES_PER_LINE = 9
COMMIT_CYCLES_PER_SAMPLE = 96_000
DESIGNS = ("candidate", "strongest_zero", "same_coordinate_bit")
RESOURCES = ("psum", "weight", "source", "dma", "commit")
EXPECTED_SERVICES = {
    "psum": 12_994_560,
    "weight": 70_853_184,
    "source": 51_840_000,
    "dma": 1_476_108,
    "commit": 960_000,
}

CONTRACT_ROOT_KEYS = {
    "schema", "status", "launch_now", "max_attempts_now", "scope",
    "frozen_geometry", "frozen_m1016_authority", "frozen_task_identity",
    "physical_capacity_ledger", "arbiter_coordinate", "strict_schema",
    "m1057_stop_authority", "m1065_required_attacks", "future_execution",
    "source_identity", "claim_boundary", "docs359_sha256",
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
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
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
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        require(name not in listed and sha256(directory / name) == expected,
                "sealed authority member drift")
        listed.add(name)
    expected, name = outer.read_text(encoding="utf-8").split()
    require(name == "SHA256SUMS" and expected == sha256(manifest),
            "sealed authority outer drift")


def load_frozen(path: Path, expected_sha: str, module_name: str):
    require(path.is_file() and not path.is_symlink() and sha256(path) == expected_sha,
            module_name + " source identity drift")
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None,
            "cannot load " + module_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M1016 = load_frozen(M1016_PATH, M1016_SHA, "m1064_frozen_m1016")
M1056 = load_frozen(M1056_PATH, M1056_SHA, "m1064_frozen_m1056")


def validate_sealed_contract(contract_path: Path = CONTRACT) -> dict[str, Any]:
    """Accept only the canonical, exact, double-sealed M1064 contract."""
    supplied = Path(contract_path)
    require(supplied.absolute() == CONTRACT.absolute() and
            supplied.is_file() and not supplied.is_symlink(),
            "only canonical nonsymlink contract path is accepted")
    require(CONTRACT_SIDECAR.is_file() and not CONTRACT_SIDECAR.is_symlink() and
            CONTRACT_OUTER.is_file() and not CONTRACT_OUTER.is_symlink(),
            "contract double seal absent")
    require(sha256(CONTRACT) == CONTRACT_SHA and
            sha256(CONTRACT_SIDECAR) == CONTRACT_SIDECAR_SHA and
            sha256(CONTRACT_OUTER) == CONTRACT_OUTER_SHA,
            "contract or double-seal exact identity drift")
    expected, name = CONTRACT_SIDECAR.read_text(encoding="utf-8").split()
    require(expected == CONTRACT_SHA and name == CONTRACT.name,
            "contract sidecar content drift")
    expected, name = CONTRACT_OUTER.read_text(encoding="utf-8").split()
    require(expected == CONTRACT_SIDECAR_SHA and name == CONTRACT_SIDECAR.name,
            "contract outer content drift")
    value = strict_json(CONTRACT)
    require(type(value) is dict and set(value) == CONTRACT_ROOT_KEYS,
            "contract exact root schema drift")
    require(value["schema"] ==
            "m1064_m1057_c1_frozen_exact_1rw_replay_source_contract_v1" and
            value["status"] ==
            "PASS_M1064_SEALED_CONTRACT_SOURCE_ONLY__M1065_REQUIRED_NO_LAUNCH" and
            value["launch_now"] is False and
            exact_int(value["max_attempts_now"]) and
            value["max_attempts_now"] == 0 and
            value["docs359_sha256"] == DOCS359_SHA,
            "contract content drift")
    return {
        "status": "PASS_M1064_EXACT_DOUBLE_SEALED_CONTRACT",
        "contract_sha256": CONTRACT_SHA,
        "sidecar_sha256": CONTRACT_SIDECAR_SHA,
        "outer_seal_file_sha256": CONTRACT_OUTER_SHA,
    }


def validate_frozen_authorities(hash_rows: bool = True) -> dict[str, Any]:
    validate_sealed_contract()
    verify_flat(M1057, M1057_ID)
    m1057 = strict_json(M1057 / "review.json")
    require(m1057.get("status") ==
            "STOP_M1057_M1056_C1_EXACT_1RW_SOURCE_HAMMER" and
            m1057.get("claim_boundary", {}).get(
                "future_full_replay_release_authorized") is False,
            "M1057 STOP authority drift")
    require(sha256(M1016_PATH) == M1016_SHA and sha256(M1056_PATH) == M1056_SHA and
            ROWS.is_file() and not ROWS.is_symlink() and
            ROWS.stat().st_size == RAW_ROWS * BYTES_PER_LINE and
            sha256(DOCS359) == DOCS359_SHA,
            "frozen source/row-size/docs authority drift")
    if hash_rows:
        require(sha256(ROWS) == ROWS_SHA, "frozen row identity drift")
    return {
        "status": "PASS_M1064_FROZEN_AUTHORITIES",
        "rows_sha_checked": hash_rows,
        "m1016_service_digest": EXPECTED_SERVICE_DIGEST,
    }


def derive_physical_capacity() -> dict[str, Any]:
    """No arguments: capacity has no caller-controlled coordinate."""
    macro_bytes = (128 * 128) // 8
    psum_slices = math.ceil(1824 / 128)
    psum_groups = BLOCKS // 2
    psum_macros = psum_slices * psum_groups
    psum_bytes = psum_macros * macro_bytes
    weight_macros = 24
    weight_bytes = weight_macros * macro_bytes
    other = {
        "parent_scratch_bytes": 18_432,
        "active_bitmap_bytes": 1_152,
        "descriptor_pingpong_bytes": 2_304,
        "fifo_control_reserve_bytes": 16_384,
        "parent_liveness_class_bytes": 1_152,
        "psum_valid_sidecar_bytes": 1_152,
        "source_mask_pingpong_bytes": 2_304,
    }
    parent_plus_other = sum(other.values())
    total = psum_bytes + weight_bytes + parent_plus_other
    require(macro_bytes == 2048 and psum_slices == 15 and psum_groups == 4 and
            psum_macros == 60 and psum_bytes == 122_880 and
            weight_bytes == 49_152 and parent_plus_other == 42_880 and
            total == 214_912,
            "physical capacity derivation drift")
    return {
        "schema": "m1064_frozen_physical_capacity_v1",
        "macro_bytes": macro_bytes,
        "psum": {
            "wide_slices_per_group": psum_slices,
            "groups": psum_groups,
            "macro_count": psum_macros,
            "bytes": psum_bytes,
        },
        "weight": {"macro_count": weight_macros, "bytes": weight_bytes},
        "parent_plus_other": {**other, "bytes": parent_plus_other},
        "derived_total_bytes": total,
        "budget_bytes": 245_760,
        "derived_margin_bytes": 245_760 - total,
        "capacity_bytes_pass": total <= 245_760,
        "caller_supplied_capacity": False,
        "capacity_only_214912B_admitted": False,
    }


def decode_task_id(task_id: int) -> tuple[int, int, int, int]:
    require(exact_int(task_id) and 0 <= task_id < TASKS, "task ID outside frozen geometry")
    partition = task_id % PARTITIONS
    quotient = task_id // PARTITIONS
    chunk = quotient % CHUNKS
    quotient //= CHUNKS
    operator = quotient % OPERATORS
    sample = quotient // OPERATORS
    require(0 <= sample < SAMPLES, "decoded sample drift")
    return sample, operator, chunk, partition


def row_count_for_chunk(chunk: int) -> int:
    require(exact_int(chunk) and 0 <= chunk < CHUNKS, "chunk outside geometry")
    return min(ROW_TILE, ROWS_PER_PHASE - chunk * ROW_TILE)


def validate_receipt_exact(receipt: Mapping[str, Any], task_id: int,
                           row_count: int) -> dict[str, Any]:
    require(type(receipt) is dict and set(receipt) == {
        "task", "counts", "source_address_first", "source_address_count",
        "weight_beat_first", "dma_first", "psum_addresses", "commit_first",
    }, "common receipt exact root schema drift")
    require(exact_int(receipt["task"]) and receipt["task"] == task_id and
            exact_int(receipt["source_address_first"]) and
            exact_int(receipt["source_address_count"]) and
            exact_int(receipt["weight_beat_first"]) and
            exact_int(receipt["dma_first"]) and
            receipt["commit_first"] is None,
            "common receipt scalar type/value drift")
    counts = receipt["counts"]
    require(type(counts) is dict and set(counts) == set(RESOURCES) and
            all(exact_int(value) and value >= 0 for value in counts.values()),
            "common receipt counts reject bool/extra/noninteger")
    addresses = receipt["psum_addresses"]
    require(type(addresses) is list and len(addresses) == BLOCKS and all(
        type(pair) is list and len(pair) == 2 and
        all(exact_int(value) for value in pair) for pair in addresses
    ), "common receipt psum address schema drift")
    expected = M1016.common_receipt(task_id, row_count)
    require(receipt == expected, "common receipt does not equal frozen M1016")
    return dict(receipt)


@dataclass(frozen=True)
class DesignTaskReceipt:
    task_id: int
    row: int
    row_count: int
    preprocess_cycles: int
    common_receipt: Mapping[str, Any]
    plan: Any


@dataclass(frozen=True)
class FrozenTaskRecord:
    task_id: int
    sample: int
    operator: int
    chunk: int
    partition: int
    row: int
    row_count: int
    preprocess_cycles: int
    design_receipts: Mapping[str, DesignTaskReceipt]


def build_frozen_record(task_id: int, masks: Sequence[int]) -> FrozenTaskRecord:
    """Internal derivation primitive; the future production iterator supplies masks."""
    sample, operator, chunk, partition = decode_task_id(task_id)
    row_count = row_count_for_chunk(chunk)
    require(type(masks) in (list, tuple, np.ndarray) and len(masks) == row_count and
            all(not isinstance(value, (bool, np.bool_)) and
                isinstance(value, (int, np.integer)) and
                0 <= int(value) <= 0xFFFF for value in masks),
            "mask tile geometry/value drift")
    masks_array = np.asarray(masks, dtype=np.uint16)
    receipt = M1016.common_receipt(task_id, row_count)
    validate_receipt_exact(receipt, task_id, row_count)
    work = {}
    design_preprocess = {}
    for design in DESIGNS:
        work[design] = int(M1016.parent_for_design(design, masks_array)[0])
        design_preprocess[design] = int(
            M1016.preprocess_for_design(design, masks_array, receipt)
        )
    shared_preprocess = max(design_preprocess.values())
    row = task_id % 64
    design_receipts = {
        design: DesignTaskReceipt(
            task_id=task_id,
            row=row,
            row_count=row_count,
            preprocess_cycles=shared_preprocess,
            common_receipt=dict(receipt),
            plan=M1056.TaskPlan(task_id, shared_preprocess, work[design], row),
        )
        for design in DESIGNS
    }
    record = FrozenTaskRecord(task_id, sample, operator, chunk, partition, row,
                              row_count, shared_preprocess, design_receipts)
    validate_frozen_record(record)
    return record


def validate_frozen_record(record: FrozenTaskRecord) -> None:
    require(type(record) is FrozenTaskRecord, "record must be exact FrozenTaskRecord")
    sample, operator, chunk, partition = decode_task_id(record.task_id)
    expected_row_count = row_count_for_chunk(chunk)
    require(all(exact_int(value) for value in (
        record.task_id, record.sample, record.operator, record.chunk,
        record.partition, record.row, record.row_count, record.preprocess_cycles
    )) and (record.sample, record.operator, record.chunk, record.partition) ==
            (sample, operator, chunk, partition) and
            record.row == record.task_id % 64 and
            record.row_count == expected_row_count and record.preprocess_cycles >= 0,
            "frozen task coordinate/row/preprocess drift")
    require(type(record.design_receipts) is dict and
            set(record.design_receipts) == set(DESIGNS),
            "three-design receipt population drift")
    identities = []
    common_receipts = []
    for design in DESIGNS:
        design_receipt = record.design_receipts[design]
        require(type(design_receipt) is DesignTaskReceipt and
                all(exact_int(value) for value in (
                    design_receipt.task_id, design_receipt.row,
                    design_receipt.row_count, design_receipt.preprocess_cycles
                )) and design_receipt.task_id == record.task_id and
                design_receipt.row == record.row and
                design_receipt.row_count == record.row_count and
                design_receipt.preprocess_cycles == record.preprocess_cycles,
                "three-design receipt ID/row/preprocess drift")
        validate_receipt_exact(design_receipt.common_receipt,
                               record.task_id, record.row_count)
        plan = design_receipt.plan
        require(type(plan) is M1056.TaskPlan and exact_int(plan.task_id) and
                exact_int(plan.preprocess_cycles) and exact_int(plan.work_cycles) and
                exact_int(plan.psum_row) and plan.task_id == record.task_id and
                plan.preprocess_cycles == record.preprocess_cycles and
                plan.psum_row == record.row and plan.work_cycles >= 0,
                "three-design per-task ID/row/preprocess drift")
        identities.append((plan.task_id, plan.preprocess_cycles, plan.psum_row))
        common_receipts.append(design_receipt.common_receipt)
    require(len(set(identities)) == 1,
            "three-design per-task identity mismatch")
    require(common_receipts[0] == common_receipts[1] == common_receipts[2],
            "three-design common receipt mismatch")


@dataclass(init=False)
class FrozenCoverage:
    next_task_id: int = 0
    next_sample_commit: int = 0
    services: dict[str, Counter] = field(
        default_factory=lambda: {name: Counter() for name in DESIGNS}
    )
    digests: dict[str, Any] = field(
        default_factory=lambda: {name: hashlib.sha256() for name in DESIGNS}
    )
    nonempty: bool = False

    def __init__(self) -> None:
        self.next_task_id = 0
        self.next_sample_commit = 0
        self.services = {name: Counter() for name in DESIGNS}
        self.digests = {name: hashlib.sha256() for name in DESIGNS}
        self.nonempty = False

    def consume(self, record: FrozenTaskRecord) -> None:
        validate_frozen_record(record)
        require(record.task_id == self.next_task_id,
                "missing, duplicate or out-of-order frozen task ID")
        for design in DESIGNS:
            design_receipt = record.design_receipts[design]
            canonical = json.dumps(design_receipt.common_receipt, sort_keys=True,
                                   separators=(",", ":")).encode()
            self.services[design].update(design_receipt.common_receipt["counts"])
            self.digests[design].update(canonical)
        self.next_task_id += 1
        self.nonempty = True
        if self.next_task_id % TASKS_PER_SAMPLE == 0:
            self._consume_sample_commit(self.next_sample_commit)
            self.next_sample_commit += 1

    def _consume_sample_commit(self, sample: int) -> None:
        require(exact_int(sample) and sample == self.next_sample_commit and
                0 <= sample < SAMPLES, "sample commit boundary drift")
        receipt = {
            "task": TASKS + sample,
            "counts": {
                resource: COMMIT_CYCLES_PER_SAMPLE if resource == "commit" else 0
                for resource in RESOURCES
            },
            "sample_commit": sample,
        }
        canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        for design in DESIGNS:
            self.services[design].update(receipt["counts"])
            self.digests[design].update(canonical)

    def proof(self) -> dict[str, Any]:
        digest_rows = {name: self.digests[name].hexdigest() for name in DESIGNS}
        checks = {
            "nonempty": self.nonempty,
            "exact_task_count_and_ids": self.next_task_id == TASKS,
            "exact_sample_commits": self.next_sample_commit == SAMPLES,
            "exact_services": all(dict(self.services[name]) == EXPECTED_SERVICES
                                  for name in DESIGNS),
            "three_design_digests_equal": len(set(digest_rows.values())) == 1,
            "frozen_m1016_digest": all(value == EXPECTED_SERVICE_DIGEST
                                       for value in digest_rows.values()),
        }
        return {
            "schema": "m1064_internal_frozen_coverage_v1",
            "checks": checks,
            "full_coverage_pass": all(checks.values()),
            "task_count": self.next_task_id,
            "sample_commits": self.next_sample_commit,
            "service_counts": {name: dict(self.services[name]) for name in DESIGNS},
            "service_digests": digest_rows,
            "caller_supplied_coverage": False,
        }


def iter_frozen_task_records() -> Iterator[FrozenTaskRecord]:
    """Future M1066-only full iterator; source CLI never calls this function."""
    validate_frozen_authorities(hash_rows=True)
    fd = os.open(ROWS, os.O_RDONLY)
    try:
        for task_id in range(TASKS):
            sample, operator, chunk, partition = decode_task_id(task_id)
            count = row_count_for_chunk(chunk)
            phase = (sample * OPERATORS + operator) * PARTITIONS + partition
            offset = (phase * ROWS_PER_PHASE + chunk * ROW_TILE) * BYTES_PER_LINE
            raw = os.pread(fd, count * BYTES_PER_LINE, offset)
            require(len(raw) == count * BYTES_PER_LINE, "short frozen row tile read")
            lines = raw.splitlines()
            require(len(lines) == count and all(len(line) == 8 for line in lines),
                    "frozen row tile text drift")
            masks = [int(line, 16) & 0xFFFF for line in lines]
            yield build_frozen_record(task_id, masks)
    finally:
        os.close(fd)


def replay_frozen_sample(records: Sequence[FrozenTaskRecord]) -> dict[str, Any]:
    """Sanctioned future M1066 sample entry; capacity has no caller argument."""
    require(type(records) in (list, tuple) and len(records) == TASKS_PER_SAMPLE,
            "sample replay requires exact nonempty frozen task population")
    for record in records:
        validate_frozen_record(record)
    sample = records[0].sample
    expected_first = sample * TASKS_PER_SAMPLE
    require(0 <= sample < SAMPLES and
            [record.task_id for record in records] ==
            list(range(expected_first, expected_first + TASKS_PER_SAMPLE)) and
            all(record.sample == sample for record in records),
            "sample task IDs/order/boundary drift")
    plans = {
        design: [record.design_receipts[design].plan for record in records]
        for design in DESIGNS
    }
    receipts = {
        design: [record.design_receipts[design].common_receipt for record in records]
        for design in DESIGNS
    }
    capacity = derive_physical_capacity()
    replay = M1056.replay_three_design_sequences(
        plans, receipts, commit_cycles=COMMIT_CYCLES_PER_SAMPLE,
        capacity_bytes=capacity["derived_total_bytes"],
    )
    require(replay["capacity_bytes"] == capacity["derived_total_bytes"] and
            replay["capacity_bytes_pass"] == capacity["capacity_bytes_pass"] and
            replay["port_feasibility_pass"] is True,
            "sample replay capacity/port coordinate drift")
    return {
        "schema": "m1064_frozen_sample_exact_1rw_replay_v1",
        "status": "PASS_M1064_FROZEN_SAMPLE_EXACT_1RW_REPLAY",
        "sample": sample,
        "first_task_id": expected_first,
        "last_task_id": expected_first + TASKS_PER_SAMPLE - 1,
        "task_count": TASKS_PER_SAMPLE,
        "cycles_after_commit": replay["cycles_after_commit"],
        "capacity": capacity,
        "port_feasibility_pass": True,
        "caller_supplied_capacity": False,
    }


def parse_receipt_json_for_attack(payload: str, task_id: int,
                                  row_count: int) -> dict[str, Any]:
    """Strict external-schema oracle; production never ingests receipt JSON."""
    value = json.loads(
        payload,
        object_pairs_hook=lambda pairs: _strict_pairs(pairs),
        parse_constant=lambda token: (_ for _ in ()).throw(
            RuntimeError("nonfinite JSON: " + token)
        ),
    )
    return validate_receipt_exact(value, task_id, row_count)


def _strict_pairs(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "duplicate JSON key: " + key)
        value[key] = item
    return value


def small_oracle() -> dict[str, Any]:
    validate_frozen_authorities(hash_rows=False)
    capacity = derive_physical_capacity()
    require(len(inspect.signature(derive_physical_capacity).parameters) == 0 and
            len(inspect.signature(FrozenCoverage).parameters) == 0 and
            capacity["derived_total_bytes"] == 214_912 and
            capacity["capacity_bytes_pass"] is True and
            capacity["capacity_only_214912B_admitted"] is False,
            "capacity caller-authority repair drift")

    masks0 = [1] * 64
    record0 = build_frozen_record(0, masks0)
    require(all(record0.design_receipts[name].task_id == 0 and
                record0.design_receipts[name].row == 0 and
                record0.design_receipts[name].preprocess_cycles ==
                record0.preprocess_cycles
                for name in DESIGNS), "three-design task identity repair drift")

    # Preserve M1056 one-port/cascade kernel with internally derived plans.
    tiny_plans = {
        name: [M1056.TaskPlan(0, 0, 8 if name == "candidate" else 16, 3),
               M1056.TaskPlan(1, 0, 8 if name == "candidate" else 16, 3)]
        for name in DESIGNS
    }
    receipt = {"task": 0, "counts": {
        "psum": 16, "weight": 1, "source": 64, "dma": 0, "commit": 0,
    }}
    replay = M1056.replay_three_design_sequences(
        tiny_plans, {name: [receipt] for name in DESIGNS}
    )
    require(replay["results"]["candidate"].sample_cycles_after_commit == 22 and
            replay["results"]["candidate"].total_nominal_excess_accesses == 16,
            "M1056 cascade kernel drift")

    # M1057 attack 1: empty coverage cannot complete.
    empty = FrozenCoverage().proof()
    require(not empty["full_coverage_pass"] and not empty["checks"]["nonempty"],
            "empty coverage admitted")

    # M1057 attack 2: boolean count and extra key are rejected.
    expected_receipt = M1016.common_receipt(0, 64)
    boolean_receipt = json.loads(json.dumps(expected_receipt))
    boolean_receipt["counts"]["dma"] = True
    try:
        validate_receipt_exact(boolean_receipt, 0, 64)
    except RuntimeError:
        boolean_rejected = True
    else:
        boolean_rejected = False
    extra_receipt = json.loads(json.dumps(expected_receipt))
    extra_receipt["coverage_pass"] = True
    try:
        validate_receipt_exact(extra_receipt, 0, 64)
    except RuntimeError:
        extra_rejected = True
    else:
        extra_rejected = False
    require(boolean_rejected and extra_rejected, "bool/extra receipt accepted")

    # M1057 attack 3: mismatched ID/row/preprocess is rejected before coverage.
    bad_receipts = dict(record0.design_receipts)
    original_bad = bad_receipts["strongest_zero"]
    bad_receipts["strongest_zero"] = DesignTaskReceipt(
        99, 7, original_bad.row_count, record0.preprocess_cycles + 1,
        original_bad.common_receipt,
        M1056.TaskPlan(
            99, record0.preprocess_cycles + 1,
            original_bad.plan.work_cycles, 7
        ),
    )
    bad_record = FrozenTaskRecord(
        record0.task_id, record0.sample, record0.operator, record0.chunk,
        record0.partition, record0.row, record0.row_count,
        record0.preprocess_cycles, bad_receipts,
    )
    try:
        validate_frozen_record(bad_record)
    except RuntimeError:
        geometry_rejected = True
    else:
        geometry_rejected = False
    require(geometry_rejected, "three-design geometry mismatch accepted")

    # Missing/duplicate/out-of-order task IDs fail immediately.
    coverage = FrozenCoverage()
    coverage.consume(record0)
    try:
        coverage.consume(record0)
    except RuntimeError:
        duplicate_rejected = True
    else:
        duplicate_rejected = False
    require(duplicate_rejected and not coverage.proof()["full_coverage_pass"],
            "duplicate/incomplete coverage admitted")

    # M1057 attack 4: no capacity argument or caller pass exists.
    try:
        derive_physical_capacity(0)  # type: ignore[call-arg]
    except TypeError:
        capacity_argument_rejected = True
    else:
        capacity_argument_rejected = False
    require(capacity_argument_rejected, "caller capacity accepted")

    # M1057 attack 5: temporary unsealed contract is rejected by path first.
    try:
        validate_sealed_contract(Path("/tmp/m1064_fake_contract.json"))
    except RuntimeError:
        fake_contract_rejected = True
    else:
        fake_contract_rejected = False
    require(fake_contract_rejected, "temporary/unsealed contract accepted")

    return {
        "schema": "m1064_c1_frozen_exact_1rw_source_small_oracle_v1",
        "status": "PASS_M1064_SMALL_ORACLE__M1065_REQUIRED_NO_FULL_REPLAY",
        "m1057_attacks_rejected": {
            "equal_empty_receipts": True,
            "boolean_service_count": boolean_rejected,
            "extra_receipt_key_or_coverage_boolean": extra_rejected,
            "three_design_task_geometry_mismatch": geometry_rejected,
            "duplicate_or_incomplete_task_population": duplicate_rejected,
            "caller_capacity": capacity_argument_rejected,
            "temporary_unsealed_contract": fake_contract_rejected,
        },
        "capacity": capacity,
        "frozen_geometry": {
            "tasks": TASKS,
            "samples": SAMPLES,
            "operators": OPERATORS,
            "chunks": CHUNKS,
            "partitions": PARTITIONS,
            "raw_rows": RAW_ROWS,
        },
        "frozen_service_digest": EXPECTED_SERVICE_DIGEST,
        "m1056_cascade_kernel_preserved": True,
        "full_iterator_called": False,
        "claim_boundary": {
            "source_only": True,
            "m1065_passed": False,
            "launch_now": False,
            "full_51840000_replay": False,
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
