#!/usr/bin/env python3
"""M1051 receipt-blind hammer for the M1040/M1016 full C1 replay.

The hammer never imports M1016 or its M1007 wrapper and never trusts the
published coverage booleans, counters, digests, cycles, or packing summary.
It replays the frozen M410 rows through the earlier frozen M505 recurrence,
independently rebuilds the common-service receipts and schedule, and then
compares the derived values to the sealed M1040 payload.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
RESULT_DIR = HW / "results/m1040_m1016_c1_full_matched_address_replay_r1_20260829"
RESULT_JSON = RESULT_DIR / "m1016_c1_full_matched_address_replay_result_r1.json"
ATTEMPT_DIR = HW / "results/.m1040_m1016_c1_full_matched_address_replay_attempt_consumed"
ROWS = HW / "results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/m410r2_h67_q32_runtime_rows_32.memh"
M505_PATH = HW / "system_simulator/scripts/analyze_m505_h67_liveness_aware_single_port_parent_scratch.py"
M504_PATH = HW / "system_simulator/scripts/analyze_m504_h67_single_port_parent_scratch.py"
M1016_PATH = HW / "system_simulator/scripts/run_m1016_c1_full_matched_address_replay.py"
M1040_RUNNER = HW / "system_simulator/scripts/run_m1040_m1016_c1_full_matched_address_replay_one_shot.sh"
RELEASE = HW / "contracts/m1038_m1037_m1016_c1_full_matched_address_replay_launch_release_r1_20260829.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

ROWS_SHA = "6e03352b89eff1955825334b4dedd991db8c975a9ef6662fe0317e73ccfa8334"
M505_SHA = "9d55d960d237a1940fb8e9efaa4e227a4ec1025489f80804d1c677e12bc9aced"
M504_SHA = "9a7586b096e5ffa47867a8c20f32f49a607a5724f5df835827b7a28f9d230a5e"
M1016_SHA = "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa"
M1040_RUNNER_SHA = "47d73bcff61cc0721d79223c3e2f398e406ad87aba5359d2c1418674990d2c34"
RELEASE_SHA = "ce96a98abcf8fbbb75e98c0ef1c407c2b804aa6d231e36c12a4c13f9d03fd8d5"
RESULT_SHA = "5b49e5d8a2b995af5734463805f9beefaece1792d437f9a5639fbe65707ab278"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

SAMPLES, OPERATORS, PARTITIONS = 10, 4, 432
ROWS_PER_PHASE, ROW_TILE, BLOCKS = 3000, 64, 8
CHUNKS = math.ceil(ROWS_PER_PHASE / ROW_TILE)
PHASES = SAMPLES * OPERATORS * PARTITIONS
TASKS = PHASES * CHUNKS
RAW_ROWS = PHASES * ROWS_PER_PHASE
BLOCK_TASKS = TASKS * BLOCKS
BYTES_PER_LINE = 9
DESIGNS = ("candidate", "strongest_zero", "same_coordinate_bit")
RESOURCES = ("psum", "weight", "source", "dma", "commit")
EXPECTED_SERVICES = {
    "psum": BLOCK_TASKS * 2,
    "weight": 9_069_207_552 // 128,
    "source": RAW_ROWS,
    "dma": 1_476_108,
    "commit": 960_000,
}
EXPECTED_PARENT = {
    "reads": 131_926_088,
    "writes": 79_581_608,
    "forwards": 13_717_024,
    "work_cycles": 409_734_336,
}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            RuntimeError("nonfinite JSON: " + token)
        ),
    )


def payload_set(directory: Path) -> set[str]:
    return {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }


def verify_flat_seal(directory: Path, expected_payloads: set[str]) -> dict[str, Any]:
    require(directory.is_dir() and not directory.is_symlink(), "sealed directory absent")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and outer.is_file(), "seal files absent")
    listed: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        if name.startswith("./"):
            name = name[2:]
        require(name not in listed and "/" not in name, "manifest member drift")
        member = directory / name
        require(member.is_file() and not member.is_symlink(), "sealed member absent")
        require(sha256(member) == expected, "sealed member hash drift")
        listed.add(name)
    require(listed == expected_payloads, "sealed payload list drift")
    expected_outer, outer_name = outer.read_text(encoding="utf-8").split()
    require(outer_name == "SHA256SUMS" and sha256(manifest) == expected_outer,
            "outer seal drift")
    require(payload_set(directory) == expected_payloads | {
        "SHA256SUMS", "SHA256SUMS.seal.sha256"
    }, "directory exact-set drift")
    return {
        "manifest_sha256": sha256(manifest),
        "outer_seal_file_sha256": sha256(outer),
        "payloads": sorted(expected_payloads),
        "pass": True,
    }


def load_m505():
    require(sha256(M505_PATH) == M505_SHA and sha256(M504_PATH) == M504_SHA,
            "frozen M504/M505 identity drift")
    spec = importlib.util.spec_from_file_location("m1051_frozen_m505", M505_PATH)
    require(spec is not None and spec.loader is not None, "cannot load M505")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M505 = load_m505()


def quota(total: int, index: int) -> int:
    return ((index + 1) * total) // TASKS - (index * total) // TASKS


def task_index(sample: int, operator: int, chunk: int, partition: int) -> int:
    return (((sample * OPERATORS + operator) * CHUNKS + chunk) * PARTITIONS + partition)


def phase_index(sample: int, operator: int, partition: int) -> int:
    return (sample * OPERATORS + operator) * PARTITIONS + partition


def common_receipt(index: int, row_count: int) -> dict[str, Any]:
    partition = index % PARTITIONS
    quotient = index // PARTITIONS
    chunk = quotient % CHUNKS
    quotient //= CHUNKS
    operator = quotient % OPERATORS
    sample = quotient // OPERATORS
    source_first = phase_index(sample, operator, partition) * ROWS_PER_PHASE + chunk * ROW_TILE
    counts = {
        "psum": BLOCKS * 2,
        "weight": quota(EXPECTED_SERVICES["weight"], index),
        "source": row_count,
        "dma": quota(EXPECTED_SERVICES["dma"], index),
        "commit": 0,
    }
    return {
        "task": index,
        "counts": counts,
        "source_address_first": source_first,
        "source_address_count": row_count,
        "weight_beat_first": (index * EXPECTED_SERVICES["weight"]) // TASKS,
        "dma_first": (index * EXPECTED_SERVICES["dma"]) // TASKS,
        "psum_addresses": [[bank, index % 64] for bank in range(BLOCKS)],
        "commit_first": None,
    }


@dataclass
class Pipeline:
    work_start: int | None = None
    previous_work: int = 0
    total: int = 0

    def push(self, preprocess: int, work: int) -> tuple[int, int]:
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
class Packing:
    last_cycle: dict[int, int] = field(default_factory=dict)
    psum_conflicts: int = 0
    psum_max_lifetime: int = 0
    weight_runs: list[tuple[int, int, int]] = field(default_factory=list)
    weight_conflicts: int = 0
    weight_half_slot_overlap: int = 0
    conflict_tasks: int = 0
    event_cycles: np.ndarray = field(
        default_factory=lambda: np.empty((BLOCKS // 2, TASKS * 4), dtype=np.int64),
        repr=False,
    )
    event_cursor: np.ndarray = field(
        default_factory=lambda: np.zeros(BLOCKS // 2, dtype=np.int64), repr=False
    )

    def psum_task(self, start: int, end: int) -> None:
        before = self.psum_conflicts
        span = max(1, (end - start) // BLOCKS)
        events: list[tuple[int, int]] = []
        for bank in range(BLOCKS):
            read_cycle = start + bank * span
            write_cycle = min(end, read_cycle + span - 1)
            group = bank // 2
            events.extend(((read_cycle, group), (write_cycle, group)))
            self.psum_max_lifetime = max(
                self.psum_max_lifetime, write_cycle - read_cycle
            )
            cursor = int(self.event_cursor[group])
            self.event_cycles[group, cursor:cursor + 2] = (read_cycle, write_cycle)
            self.event_cursor[group] += 2
        for cycle, group in sorted(events):
            if self.last_cycle.get(group) == cycle:
                self.psum_conflicts += 1
            self.last_cycle[group] = cycle
        self.conflict_tasks += int(self.psum_conflicts != before)

    def global_port_sweep(self) -> dict[str, int]:
        """Exact group/cycle occupancy sweep, unlike append-order last_cycle."""
        require(bool(np.all(self.event_cursor == TASKS * 4)),
                "global psum event population incomplete")
        conflict_cycles = 0
        excess_accesses = 0
        accesses_in_conflicting_slots = 0
        maximum_multiplicity = 0
        for group in range(BLOCKS // 2):
            row = self.event_cycles[group]
            row.sort()
            starts = np.r_[0, np.flatnonzero(row[1:] != row[:-1]) + 1]
            counts = np.diff(np.r_[starts, row.size])
            conflicts = counts[counts > 1]
            conflict_cycles += int(conflicts.size)
            excess_accesses += int(np.sum(conflicts - 1, dtype=np.int64))
            accesses_in_conflicting_slots += int(np.sum(conflicts, dtype=np.int64))
            maximum_multiplicity = max(
                maximum_multiplicity, int(conflicts.max(initial=0))
            )
        return {
            "total_accesses": int(self.event_cycles.size),
            "conflict_cycles": conflict_cycles,
            "excess_accesses_over_1rw": excess_accesses,
            "accesses_in_conflicting_slots": accesses_in_conflicting_slots,
            "maximum_same_group_cycle_multiplicity": maximum_multiplicity,
        }

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


def frontend(design: str, masks: np.ndarray, receipt: Mapping[str, Any]) -> int:
    rows = int(masks.size)
    source_cycles = math.ceil(int(receipt["counts"]["source"]) / 64)
    common = max(source_cycles, int(receipt["counts"]["weight"]),
                 int(receipt["counts"]["dma"]), BLOCKS * 2)
    if design == "strongest_zero":
        front = rows + 5
    elif design == "same_coordinate_bit":
        front = math.ceil(rows / 8) + 2
    else:
        capture = math.ceil(rows / 8)
        search_rows = int(np.count_nonzero(M505.M504.POPCOUNT[masks] > 1))
        front = capture + search_rows * math.ceil(rows / 64) + 17 * capture + 2
    return max(front, common)


def full_rederive() -> dict[str, Any]:
    require(ROWS.stat().st_size == RAW_ROWS * BYTES_PER_LINE,
            "frozen row byte count drift")
    services = {name: Counter() for name in DESIGNS}
    digests = {name: hashlib.sha256() for name in DESIGNS}
    parent = Counter()
    pipelines = {name: Pipeline() for name in DESIGNS}
    packings = {name: Packing() for name in DESIGNS}
    global_offsets = {name: 0 for name in DESIGNS}
    sample_cycles = {name: [] for name in DESIGNS}
    phase_rows = np.zeros(PHASES, dtype=np.int32)
    tasks_seen = 0
    rows_seen = 0
    first_source = None
    last_source_end = None
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
                        require(len(raw) == count * BYTES_PER_LINE, "short row read")
                        lines = raw.splitlines()
                        require(len(lines) == count and all(len(line) == 8 for line in lines),
                                "row text geometry drift")
                        masks = np.asarray(
                            [int(line, 16) & 0xFFFF for line in lines], dtype=np.uint16
                        )
                        index = task_index(sample, operator, chunk, partition)
                        require(index == tasks_seen, "task traversal/order drift")
                        receipt = common_receipt(index, count)
                        source_first = int(receipt["source_address_first"])
                        if first_source is None:
                            first_source = source_first
                        last_source_end = source_first + count
                        phase_rows[phase] += count
                        tasks_seen += 1
                        rows_seen += count

                        m505 = M505.simulate_liveness_task(masks, False)
                        candidate_work = int(m505["liveness_cycles"]) * BLOCKS
                        baseline_work = int(M505.M504.POPCOUNT[masks].sum()) * BLOCKS
                        parent.update({
                            "reads": int(m505["macro_reads"]) * BLOCKS,
                            "writes": int(m505["macro_writes"]) * BLOCKS,
                            "forwards": int(m505["forwarded_reads"]) * BLOCKS,
                            "work_cycles": candidate_work,
                        })
                        works = {
                            "candidate": candidate_work,
                            "strongest_zero": baseline_work,
                            "same_coordinate_bit": baseline_work,
                        }
                        canonical = json.dumps(
                            receipt, sort_keys=True, separators=(",", ":")
                        ).encode()
                        for design in DESIGNS:
                            services[design].update(receipt["counts"])
                            digests[design].update(canonical)
                            pre = frontend(design, masks, receipt)
                            start, end = pipelines[design].push(pre, works[design])
                            absolute_start = global_offsets[design] + start
                            absolute_end = global_offsets[design] + end
                            packings[design].psum_task(absolute_start, absolute_end)
                            packings[design].weight_task(
                                global_offsets[design] + start - pre,
                                int(receipt["counts"]["weight"]), index & 1
                            )
            for design in DESIGNS:
                commit = {
                    "task": TASKS + sample,
                    "counts": {resource: 96_000 if resource == "commit" else 0
                               for resource in RESOURCES},
                    "sample_commit": sample,
                }
                services[design].update(commit["counts"])
                digests[design].update(
                    json.dumps(commit, sort_keys=True, separators=(",", ":")).encode()
                )
                pipelines[design].commit(96_000)
                sample_cycles[design].append(pipelines[design].total)
                global_offsets[design] += pipelines[design].total
                pipelines[design] = Pipeline()
    finally:
        os.close(fd)

    return {
        "tasks": tasks_seen,
        "rows": rows_seen,
        "phase_count": int(phase_rows.size),
        "phases_all_3000": bool(np.all(phase_rows == ROWS_PER_PHASE)),
        "first_source_address": first_source,
        "last_source_address_exclusive": last_source_end,
        "block_tasks_per_design": tasks_seen * BLOCKS,
        "services": {name: dict(services[name]) for name in DESIGNS},
        "service_digests": {name: digests[name].hexdigest() for name in DESIGNS},
        "parent": dict(parent),
        "sample_cycles": sample_cycles,
        "total_cycles": {name: sum(sample_cycles[name]) for name in DESIGNS},
        "packing": {
            name: {
                "paired_psum_1rw_conflicts": packings[name].psum_conflicts,
                "conflict_tasks": packings[name].conflict_tasks,
                "psum_maximum_lifetime_cycles": packings[name].psum_max_lifetime,
                "weight_1rw_conflicts": packings[name].weight_conflicts,
                "weight_half_slot_overlap_cycles":
                    packings[name].weight_half_slot_overlap,
                "global_port_sweep": packings[name].global_port_sweep(),
            }
            for name in DESIGNS
        },
    }


def main() -> dict[str, Any]:
    require(sha256(ROWS) == ROWS_SHA, "M410 row identity drift")
    require(sha256(M1016_PATH) == M1016_SHA, "M1016 identity drift")
    require(sha256(M1040_RUNNER) == M1040_RUNNER_SHA, "M1040 runner drift")
    require(sha256(RELEASE) == RELEASE_SHA, "M1038 release drift")
    require(sha256(RESULT_JSON) == RESULT_SHA, "M1040 result identity drift")
    require(sha256(DOCS359) == DOCS359_SHA, "docs359 drift")
    attempt_seal = verify_flat_seal(ATTEMPT_DIR, {"attempt.json"})
    result_seal = verify_flat_seal(
        RESULT_DIR, {"m1016_c1_full_matched_address_replay_result_r1.json"}
    )
    attempt = strict_json(ATTEMPT_DIR / "attempt.json")
    require(attempt == {
        "status": "M1040_ATTEMPT_CONSUMED",
        "runner_sha256": M1040_RUNNER_SHA,
        "contract_sha256":
            "b980f51017778b1958845547601de5d343ba5a1f3db1b046963afa7549644c90",
        "release_sha256": RELEASE_SHA,
        "lockfile": str(HW / "results/.c1_full_matched_address_replay_global.lock"),
        "min_commit_headroom_kb": 16_777_216,
        "min_mem_available_kb": 16_777_216,
    }, "attempt content drift")
    published = strict_json(RESULT_JSON)
    derived = full_rederive()

    require(derived["tasks"] == TASKS and derived["rows"] == RAW_ROWS and
            derived["phase_count"] == PHASES and derived["phases_all_3000"] and
            derived["block_tasks_per_design"] == BLOCK_TASKS,
            "coverage rederivation mismatch")
    require(derived["first_source_address"] == 0 and
            derived["last_source_address_exclusive"] == RAW_ROWS,
            "source address boundary mismatch")
    require(all(derived["services"][name] == EXPECTED_SERVICES for name in DESIGNS),
            "service count mismatch")
    require(len(set(derived["service_digests"].values())) == 1,
            "service digest mismatch across designs")
    require(derived["parent"] == EXPECTED_PARENT, "parent conservation mismatch")
    require(derived["service_digests"] == published["coverage"]["service_digests"],
            "published service digest mismatch")
    require(derived["sample_cycles"] == published["sample_cycle_boundaries"],
            "published sample cycles mismatch")
    require(derived["total_cycles"] == published["cycles_raw_unadmitted"],
            "published total cycles mismatch")
    for name in DESIGNS:
        observed = derived["packing"][name]
        reported = published["packing"][name]
        for key in ("paired_psum_1rw_conflicts", "psum_maximum_lifetime_cycles",
                    "weight_1rw_conflicts", "weight_half_slot_overlap_cycles"):
            require(observed[key] == reported[key], name + " packing mismatch: " + key)
        require(reported["capacity_only_214912B_raw_gate_pass"] is False and
                reported["capacity_only_214912B_admitted"] is False,
                name + " capacity gate expanded")

    boundary = published["claim_boundary"]
    require(all(boundary[key] is False for key in (
        "capacity_only_214912B_admitted", "m528_1p7467534301_promoted",
        "matched_total_cycles_admitted", "paper_ppa_ready", "rtl_cycles",
        "speedup_admitted"
    )), "published claim boundary expanded")
    require(published["status"] ==
            "PASS_RAW_FULL_REPLAY_PENDING_INDEPENDENT_RESULT_HAMMER" and
            published["coverage"]["raw_full_replay_complete"] is True,
            "published status drift")

    candidate = derived["total_cycles"]["candidate"]
    zero = derived["total_cycles"]["strongest_zero"]
    bit = derived["total_cycles"]["same_coordinate_bit"]
    raw_vs_zero = zero / candidate
    raw_vs_bit = bit / candidate
    all_conflicts = {
        name: derived["packing"][name]["paired_psum_1rw_conflicts"]
        for name in DESIGNS
    }
    require(set(all_conflicts.values()) == {403_922}, "403922 conflict anchor drift")
    require(derived["packing"]["candidate"]["conflict_tasks"] > 0,
            "conflict task population unexpectedly empty")

    return {
        "schema": "m1051_m1040_m1016_c1_full_replay_result_hammer_mechanical_v1",
        "status": "PASS_M1051_M1040_M1016_C1_FULL_REPLAY_RAW_RESULT_HAMMER",
        "verdict": "ADMIT_RAW_CPU_CYCLE_OPPORTUNITY_ONLY__BLOCK_214912B_AND_SPEEDUP",
        "identity": {
            "m1040_result_json_sha256": sha256(RESULT_JSON),
            "m1040_runner_sha256": sha256(M1040_RUNNER),
            "m1038_release_sha256": sha256(RELEASE),
            "m1016_engine_sha256": sha256(M1016_PATH),
            "m505_analyzer_sha256": sha256(M505_PATH),
            "m504_analyzer_sha256": sha256(M504_PATH),
            "m410_rows_sha256": sha256(ROWS),
            "docs359_sha256": sha256(DOCS359),
        },
        "seals": {"attempt": attempt_seal, "result": result_seal},
        "coverage_rederived": {
            "raw_rows": derived["rows"],
            "tasks": derived["tasks"],
            "phases": derived["phase_count"],
            "rows_per_phase": ROWS_PER_PHASE,
            "blocks_per_design": derived["block_tasks_per_design"],
            "samples": SAMPLES,
            "operators": OPERATORS,
            "partitions": PARTITIONS,
            "chunks_per_phase": CHUNKS,
            "first_source_address": derived["first_source_address"],
            "last_source_address_exclusive": derived["last_source_address_exclusive"],
            "pass": True,
        },
        "services_rederived": {
            "counts": derived["services"],
            "digests": derived["service_digests"],
            "all_three_equal": True,
            "pass": True,
        },
        "parent_rederived": {**derived["parent"], "pass": True},
        "cycles_rederived_raw_unadmitted": {
            "totals": derived["total_cycles"],
            "sample_boundaries": derived["sample_cycles"],
            "candidate_vs_strongest_zero_raw_opportunity": raw_vs_zero,
            "candidate_vs_same_coordinate_bit_raw_opportunity": raw_vs_bit,
            "old_m528_1p7467534301_reproduced": False,
            "pass": True,
        },
        "packing_rederived": {
            "designs": derived["packing"],
            "all_designs_conflicts": all_conflicts,
            "conflicts_feed_back_into_pipeline_cycles": False,
            "capacity_only_214912B_raw_gate_pass": False,
            "capacity_only_214912B_admitted": False,
            "pass": True,
        },
        "claim_boundary": {
            "legal": (
                "On the frozen 51.84M-row CPU replay with equal logical common "
                "service counts, the conflict-unrepaired candidate has 434242823 "
                "raw model cycles versus 753067320 for either baseline, a "
                "1.734207867x raw opportunity."
            ),
            "illegal": [
                "214912-byte capacity-feasible result",
                "matched executable cycle speedup",
                "RTL-measured cycle speedup",
                "paper headline speedup",
                "promotion of the old M528 1.7467534301x ratio",
                "paper-PPA-ready result",
            ],
            "speedup_admitted": False,
            "capacity_only_214912B_admitted": False,
            "rtl_cycles": False,
            "paper_ppa_ready": False,
        },
        "rerun": {
            "m1040_result_modified": False,
            "m1016_engine_executed": False,
            "independent_frozen_row_rederivation": True,
            "eda_gpu_remote_used": False,
        },
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
