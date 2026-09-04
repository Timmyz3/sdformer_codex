#!/usr/bin/env python3
"""M1016 future full C1 matched-address replay engine.

Production mode is intentionally expensive and has not been executed while
authoring this source package.  Coverage is derived only from the frozen M410
identity, exact geometry, unique tile/block observations and completed
three-design common-service merges.  No CLI/JSON coverage override exists.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
M1007_PATH = HERE / "m1007_c1_matched_common_charge_address_replay_source.py"
M1007_SHA = "150f22eaa11d219bfa20561b91a38049f14abbc541a6b40db04bd73533ec3442"
M1010 = HW / "reviews/m1010_m1007_c1_matched_common_charge_address_replay_source_hammer_r1_20260829"
M1010_ID = (
    "c74812b03ca17b698ec5f80d086427937aea312668fd8d34df35544a930d669e",
    "5bc8ea19bfb658cf737e227d632461a21096d5035efad8e88a20fc5cdb704e27",
    "4885bee6283a09551fa5f95088a01683ce2b561e9305a33365ad807bfeb618f7",
)
CONTRACT = HW / "contracts/m1016_m1010_c1_full_matched_address_replay_source_contract_r1_20260829.json"
ROWS = HW / "results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/m410r2_h67_q32_runtime_rows_32.memh"
ROWS_SHA = "6e03352b89eff1955825334b4dedd991db8c975a9ef6662fe0317e73ccfa8334"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

SAMPLES, OPERATORS, PARTITIONS = 10, 4, 432
ROWS_PER_PHASE, ROW_TILE, BLOCKS = 3000, 64, 8
CHUNKS = math.ceil(ROWS_PER_PHASE / ROW_TILE)
PHASES = SAMPLES * OPERATORS * PARTITIONS
TASKS = PHASES * CHUNKS
RAW_ROWS = SAMPLES * OPERATORS * PARTITIONS * ROWS_PER_PHASE
BLOCK_TASKS = TASKS * BLOCKS
BYTES_PER_LINE = 9
DESIGNS = ("candidate", "strongest_zero", "same_coordinate_bit")
RESOURCES = ("psum", "weight", "source", "dma", "commit")
EXPECTED_SERVICE_COUNTS = {
    "psum": BLOCK_TASKS * 2,
    "weight": 9_069_207_552 // 128,
    "source": RAW_ROWS,
    "dma": 1_476_108,
    "commit": 960_000,
}
EXPECTED_PARENT = {
    "candidate": {"reads": 131_926_088, "writes": 79_581_608,
                  "forwards": 13_717_024, "work_cycles": 51_216_792 * BLOCKS},
    "strongest_zero": {"reads": 0, "writes": 0, "forwards": 0},
    "same_coordinate_bit": {"reads": 0, "writes": 0, "forwards": 0},
}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + value)))


def verify_flat(directory: Path, identity: tuple[str, str, str]) -> None:
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require((sha256(review), sha256(manifest), sha256(outer)) == identity,
            "M1010 identity drift")
    listed = set()
    for line in manifest.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        require(name not in listed and sha256(directory / name) == expected,
                "M1010 member drift")
        listed.add(name)
    require(outer.read_text().split() == [identity[1], "SHA256SUMS"],
            "M1010 outer drift")


def load_m1007():
    require(sha256(M1007_PATH) == M1007_SHA, "frozen M1007 source drift")
    spec = importlib.util.spec_from_file_location("m1016_frozen_m1007", M1007_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M1007 = load_m1007()


def quota(total: int, index: int, population: int = TASKS) -> int:
    require(0 <= index < population and total >= 0, "quota coordinate drift")
    return ((index + 1) * total) // population - (index * total) // population


def task_index(sample: int, operator: int, chunk: int, partition: int) -> int:
    require(0 <= sample < SAMPLES and 0 <= operator < OPERATORS and
            0 <= chunk < CHUNKS and 0 <= partition < PARTITIONS,
            "task coordinate outside frozen geometry")
    return (((sample * OPERATORS + operator) * CHUNKS + chunk) * PARTITIONS + partition)


def phase_index(sample: int, operator: int, partition: int) -> int:
    return (sample * OPERATORS + operator) * PARTITIONS + partition


def source_row_base(index: int) -> int:
    partition = index % PARTITIONS
    quotient = index // PARTITIONS
    chunk = quotient % CHUNKS
    quotient //= CHUNKS
    operator = quotient % OPERATORS
    sample = quotient // OPERATORS
    return phase_index(sample, operator, partition) * ROWS_PER_PHASE + chunk * ROW_TILE


def common_receipt(index: int, row_count: int, include_commit: int = 0) -> dict[str, Any]:
    """Canonical common logical accesses; independent of design timing."""
    require(0 < row_count <= ROW_TILE, "row-count drift")
    counts = {
        "psum": BLOCKS * 2,
        "weight": quota(EXPECTED_SERVICE_COUNTS["weight"], index),
        "source": row_count,
        "dma": quota(EXPECTED_SERVICE_COUNTS["dma"], index),
        "commit": include_commit,
    }
    return {
        "task": index,
        "counts": counts,
        "source_address_first": source_row_base(index),
        "source_address_count": row_count,
        "weight_beat_first": (index * EXPECTED_SERVICE_COUNTS["weight"]) // TASKS,
        "dma_first": (index * EXPECTED_SERVICE_COUNTS["dma"]) // TASKS,
        "psum_addresses": [[bank, index % 64] for bank in range(BLOCKS)],
        "commit_first": index // (TASKS // SAMPLES) * 96_000 if include_commit else None,
    }


@dataclass
class DesignCoverage:
    block_tasks: int = 0
    parent: Counter = field(default_factory=Counter)
    services: Counter = field(default_factory=Counter)
    service_digest: Any = field(default_factory=hashlib.sha256)
    service_merge_finished: bool = False

    def consume(self, receipt: Mapping[str, Any]) -> None:
        for resource, count in receipt["counts"].items():
            self.services[resource] += int(count)
        canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
        self.service_digest.update(canonical.encode())


@dataclass
class DerivedCoverage:
    """There is deliberately no public coverage setter or constructor flag."""
    seen_tiles: bytearray = field(default_factory=lambda: bytearray(TASKS))
    phase_rows: np.ndarray = field(default_factory=lambda: np.zeros(PHASES, dtype=np.int32))
    raw_rows: int = 0
    unique_tiles: int = 0
    ledger_sha256: str | None = None
    designs: dict[str, DesignCoverage] = field(
        default_factory=lambda: {name: DesignCoverage() for name in DESIGNS})

    def observe_tile(self, sample: int, operator: int, chunk: int,
                     partition: int, row_count: int) -> int:
        index = task_index(sample, operator, chunk, partition)
        require(self.seen_tiles[index] == 0, "duplicate frozen tile")
        self.seen_tiles[index] = 1
        self.unique_tiles += 1
        self.raw_rows += row_count
        self.phase_rows[phase_index(sample, operator, partition)] += row_count
        return index

    def observe_design(self, design: str, receipt: Mapping[str, Any],
                       parent_summary: Mapping[str, int]) -> None:
        require(design in self.designs, "unknown design")
        state = self.designs[design]
        state.block_tasks += BLOCKS
        state.consume(receipt)
        state.parent.update(parent_summary)

    def finish_design_merges(self) -> None:
        for state in self.designs.values():
            state.service_merge_finished = True

    def set_frozen_ledger_identity(self, digest: str) -> None:
        require(self.ledger_sha256 is None, "ledger identity set twice")
        self.ledger_sha256 = digest

    def proof(self) -> dict[str, Any]:
        service_digests = {name: state.service_digest.hexdigest()
                           for name, state in self.designs.items()}
        checks = {
            "frozen_ledger_sha": self.ledger_sha256 == ROWS_SHA,
            "raw_rows": self.raw_rows == RAW_ROWS,
            "unique_tiles": self.unique_tiles == TASKS and all(self.seen_tiles),
            "all_17280_phases_have_3000_rows": bool(np.all(self.phase_rows == ROWS_PER_PHASE)),
            "all_designs_have_6497280_blocks": all(
                state.block_tasks == BLOCK_TASKS for state in self.designs.values()),
            "all_three_service_merges_finished": all(
                state.service_merge_finished for state in self.designs.values()),
            "service_counts_exact": all(
                dict(state.services) == EXPECTED_SERVICE_COUNTS
                for state in self.designs.values()),
            "service_digests_equal": len(set(service_digests.values())) == 1,
            "parent_conservation": all(
                all(int(self.designs[name].parent[key]) == int(value)
                    for key, value in expected.items())
                for name, expected in EXPECTED_PARENT.items()),
        }
        return {
            "schema": "m1016_internal_derived_coverage_v1",
            "checks": checks,
            "raw_full_replay_complete": all(checks.values()),
            "service_digests": service_digests,
            "caller_supplied_coverage": False,
        }


@dataclass
class Pipeline:
    work_start: int | None = None
    previous_work: int = 0
    total: int = 0

    def push(self, preprocess: int, work: int) -> tuple[int, int]:
        require(preprocess >= 0 and work >= 0, "negative pipeline interval")
        if self.work_start is None:
            start = preprocess
        else:
            start = self.work_start + max(self.previous_work, preprocess) + 2
        self.work_start = start
        self.previous_work = work
        self.total = start + work + 2
        return start, start + work

    def commit(self, cycles: int) -> None:
        self.total += cycles
        if self.work_start is not None:
            self.work_start += cycles


@dataclass
class PackingAudit:
    psum_last_cycle: dict[int, int] = field(default_factory=dict)
    psum_conflicts: int = 0
    weight_runs: list[tuple[int, int, int]] = field(default_factory=list)
    weight_conflicts: int = 0
    weight_half_slot_overlap: int = 0
    psum_max_lifetime: int = 0

    def psum_task(self, work_start: int, work_end: int, row: int) -> None:
        span = max(1, (work_end - work_start) // BLOCKS)
        events = []
        for bank in range(BLOCKS):
            read_cycle = work_start + bank * span
            write_cycle = min(work_end, read_cycle + span - 1)
            group = bank // 2
            events.extend(((read_cycle, group), (write_cycle, group)))
            self.psum_max_lifetime = max(self.psum_max_lifetime, write_cycle - read_cycle)
        for cycle, group in sorted(events):
            if self.psum_last_cycle.get(group) == cycle:
                self.psum_conflicts += 1
            self.psum_last_cycle[group] = cycle

    def weight_task(self, start: int, beats: int, half_slot: int) -> None:
        if beats == 0:
            return
        end = start + beats
        for old_start, old_end, old_half in self.weight_runs[-2:]:
            overlap = max(0, min(end, old_end) - max(start, old_start))
            if overlap:
                self.weight_conflicts += overlap
                if old_half != half_slot:
                    self.weight_half_slot_overlap += overlap
        self.weight_runs.append((start, end, half_slot))
        if len(self.weight_runs) > 2:
            del self.weight_runs[:-2]

    def summary(self, proof: Mapping[str, Any]) -> dict[str, Any]:
        complete = bool(proof["raw_full_replay_complete"])
        conflict_free = (self.psum_conflicts == 0 and self.weight_conflicts == 0 and
                         self.weight_half_slot_overlap == 0)
        return {
            "schema": "m1016_full_packing_audit_v1",
            "coverage_internally_derived": complete,
            "paired_psum_1rw_conflicts": self.psum_conflicts,
            "weight_1rw_conflicts": self.weight_conflicts,
            "weight_half_slot_overlap_cycles": self.weight_half_slot_overlap,
            "psum_maximum_lifetime_cycles": self.psum_max_lifetime,
            "capacity_only_214912B_raw_gate_pass": complete and conflict_free,
            "capacity_only_214912B_admitted": False,
            "pending_independent_result_hammer": True,
        }


def parent_for_design(design: str, masks: Sequence[int]) -> tuple[int, dict[str, int]]:
    if design == "candidate":
        trace = list(M1007.parent_cycle_trace(masks))
        summary = M1007.parent_summary(trace)
        return summary["cycles"] * BLOCKS, {
            "reads": summary["macro_reads"] * BLOCKS,
            "writes": summary["macro_writes"] * BLOCKS,
            "forwards": summary["forwarded_reads"] * BLOCKS,
            "work_cycles": summary["cycles"] * BLOCKS,
        }
    input_nnz = sum(int(mask).bit_count() for mask in masks)
    return input_nnz * BLOCKS, {"reads": 0, "writes": 0, "forwards": 0}


def preprocess_for_design(design: str, masks: Sequence[int], receipt: Mapping[str, Any]) -> int:
    rows = len(masks)
    nonempty = any(masks)
    input_pc = np.asarray([int(mask).bit_count() for mask in masks], dtype=np.int32)
    source_cycles = math.ceil(receipt["counts"]["source"] / 64)
    common_cycles = max(source_cycles, receipt["counts"]["weight"],
                        receipt["counts"]["dma"], BLOCKS * 2)
    if design == "strongest_zero":
        frontend = rows + 5
    elif design == "same_coordinate_bit":
        frontend = math.ceil(rows / 8) + 2
    else:
        capture = math.ceil(rows / 8)
        search_rows = int(np.count_nonzero(input_pc > 1))
        frontend = capture + search_rows * math.ceil(rows / 64) + 17 * capture + 2
    return max(frontend, common_cycles) if nonempty else max(frontend, common_cycles)


def iter_parent_address_events(masks: Sequence[int], block: int,
                               work_start: int) -> Iterator[dict[str, Any]]:
    """Public memory-bounded expansion path for exact candidate parent events."""
    trace = list(M1007.parent_cycle_trace(masks))
    offset = work_start + block * len(trace)
    for event in trace:
        yield {"cycle": offset + int(event["cycle"]), "block": block,
               "op": event["op"], "address": event["address"],
               "forward_address": event["forward_address"],
               "free_address": event["free_address"]}


def validate_source_only(contract: Path = CONTRACT) -> dict[str, Any]:
    verify_flat(M1010, M1010_ID)
    value = strict_json(contract)
    require(value["status"] == "PASS_M1016_SOURCE_ONLY__NO_FULL_REPLAY_NO_EDA" and
            value["launch_now"] is False, "M1016 contract drift")
    require(sha256(HW / "docs/359_DATE终局冻结_20260813.md") == DOCS359_SHA,
            "docs359 drift")
    return {"status": "PASS_M1016_SOURCE_PREFLIGHT__NO_FULL_REPLAY",
            "coverage_cli_override": False, "full_replay_executed": False}


def run_full(contract: Path, out: Path) -> dict[str, Any]:
    """Future production entry. Not invoked by this source milestone."""
    validate_source_only(contract)
    require(not out.exists(), "refuse to overwrite M1016 output")
    require(ROWS.stat().st_size == RAW_ROWS * BYTES_PER_LINE, "M410 size drift")
    coverage = DerivedCoverage()
    coverage.set_frozen_ledger_identity(sha256(ROWS))
    pipelines = {name: Pipeline() for name in DESIGNS}
    packing = {name: PackingAudit() for name in DESIGNS}
    sample_cycles = {name: [] for name in DESIGNS}
    global_offsets = {name: 0 for name in DESIGNS}
    fd = os.open(ROWS, os.O_RDONLY)
    try:
        for sample in range(SAMPLES):
            for operator in range(OPERATORS):
                for chunk in range(CHUNKS):
                    count = min(ROW_TILE, ROWS_PER_PHASE - chunk * ROW_TILE)
                    for partition in range(PARTITIONS):
                        phase = phase_index(sample, operator, partition)
                        offset = (phase * ROWS_PER_PHASE + chunk * ROW_TILE) * BYTES_PER_LINE
                        raw = os.pread(fd, count * BYTES_PER_LINE, offset)
                        require(len(raw) == count * BYTES_PER_LINE, "short frozen tile read")
                        masks = [int(line, 16) & 0xffff for line in raw.splitlines()]
                        require(len(masks) == count, "tile row parse drift")
                        index = coverage.observe_tile(sample, operator, chunk, partition, count)
                        receipt = common_receipt(index, count)
                        for design in DESIGNS:
                            work, parent = parent_for_design(design, masks)
                            preprocess = preprocess_for_design(design, masks, receipt)
                            start, end = pipelines[design].push(preprocess, work)
                            coverage.observe_design(design, receipt, parent)
                            absolute_start = global_offsets[design] + start
                            absolute_end = global_offsets[design] + end
                            packing[design].psum_task(absolute_start, absolute_end, index % 64)
                            packing[design].weight_task(global_offsets[design] + start - preprocess,
                                                       receipt["counts"]["weight"], index & 1)
            for design in DESIGNS:
                # Commit receipts are generated from frozen sample completion,
                # never from a caller-provided completion flag.
                commit_receipt = {"task": TASKS + sample,
                                  "counts": {r: 96_000 if r == "commit" else 0
                                             for r in RESOURCES},
                                  "sample_commit": sample}
                coverage.designs[design].consume(commit_receipt)
                pipelines[design].commit(96_000)
                sample_cycles[design].append(pipelines[design].total)
                global_offsets[design] += pipelines[design].total
                pipelines[design] = Pipeline()
    finally:
        os.close(fd)
    coverage.finish_design_merges()
    proof = coverage.proof()
    packing_rows = {name: packing[name].summary(proof) for name in DESIGNS}
    result = {
        "schema": "m1016_c1_full_matched_address_replay_result_v1",
        "status": "PASS_RAW_FULL_REPLAY_PENDING_INDEPENDENT_RESULT_HAMMER"
                  if proof["raw_full_replay_complete"] else "FAIL_CLOSED_INCOMPLETE_REPLAY",
        "coverage": proof,
        "cycles_raw_unadmitted": {name: sum(sample_cycles[name]) for name in DESIGNS},
        "sample_cycle_boundaries": sample_cycles,
        "packing": packing_rows,
        "claim_boundary": {
            "matched_total_cycles_admitted": False,
            "capacity_only_214912B_admitted": False,
            "speedup_admitted": False,
            "m528_1p7467534301_promoted": False,
            "independent_result_hammer_required": True,
            "rtl_cycles": False,
            "paper_ppa_ready": False,
        },
    }
    out.mkdir()
    payload = out / "m1016_c1_full_matched_address_replay_result_r1.json"
    payload.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    manifest = out / "SHA256SUMS"
    manifest.write_text(f"{sha256(payload)}  {payload.name}\n")
    (out / "SHA256SUMS.seal.sha256").write_text(
        f"{sha256(manifest)}  SHA256SUMS\n")
    return result


def small_oracle() -> dict[str, Any]:
    coverage = DerivedCoverage()
    # A caller cannot make an empty or tiny replay complete: no boolean enters
    # the proof path and every frozen production count remains unsatisfied.
    empty = coverage.proof()
    require(not empty["raw_full_replay_complete"], "empty coverage admitted")
    index = coverage.observe_tile(0, 0, 0, 0, 64)
    receipt = common_receipt(index, 64)
    pipelines = {name: Pipeline() for name in DESIGNS}
    audits = {name: PackingAudit() for name in DESIGNS}
    for design in DESIGNS:
        work, parent = parent_for_design(design, [1, 3, 5, 7])
        pre = preprocess_for_design(design, [1, 3, 5, 7], receipt)
        start, end = pipelines[design].push(pre, work)
        coverage.observe_design(design, receipt, parent)
        audits[design].psum_task(start, end, 0)
        audits[design].weight_task(0, receipt["counts"]["weight"], 0)
    tiny = coverage.proof()
    require(not tiny["raw_full_replay_complete"], "tiny coverage admitted")
    negative = audits["candidate"].summary(tiny)
    require(not negative["capacity_only_214912B_raw_gate_pass"] and
            not negative["capacity_only_214912B_admitted"], "packing fail-open")
    shifted = list(iter_parent_address_events([1, 3, 5], 2, 100))
    require(shifted and shifted[0]["cycle"] >= 100, "parent stream offset drift")
    return {
        "status": "PASS_M1016_SMALL_ORACLE__NO_FULL_REPLAY",
        "empty_coverage_rejected": True,
        "tiny_coverage_rejected": True,
        "coverage_cli_override": False,
        "parent_address_events": len(shifted),
        "capacity_admitted": False,
        "speedup_admitted": False,
        "full_51840000_replayed": False,
        "eda_gpu_remote_used": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    require(args.self_test ^ (args.out is not None),
            "select exactly one of --self-test or production --out")
    value = small_oracle() if args.self_test else run_full(args.contract, args.out.resolve())
    print(json.dumps({"status": value["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
