#!/usr/bin/env python3
"""Audit Q1/Q2 streaming VCS evidence and summarize the queue tradeoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ACC_BITS = 96 * 32
IDENTITY_BITS = 2 + 5 + 16 + 32 + 4 + 4 + 1 + 1 + 1 + 32
ENTRY_BITS = ACC_BITS + IDENTITY_BITS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_evidence(run_dir: Path) -> None:
    ledger = run_dir / "evidence.sha256"
    if not ledger.is_file():
        raise ValueError(f"missing evidence ledger: {ledger}")
    for line in ledger.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError(f"malformed evidence line: {line}")
        path = Path(match.group(2))
        if not path.is_absolute():
            path = run_dir / path
        if not path.is_file() or sha256(path) != match.group(1):
            raise ValueError(f"evidence SHA mismatch: {path}")


def cover(report: str, name: str) -> tuple[int, int]:
    match = re.search(
        rf"{re.escape(name)},\s+(\d+) attempts,\s+(\d+) match", report
    )
    if match is None:
        raise ValueError(f"missing VCS cover: {name}")
    return int(match.group(1)), int(match.group(2))


def load_point(depth: int, run_dir: Path) -> dict[str, Any]:
    verify_evidence(run_dir)
    simulation = (run_dir / "simulation.log").read_text(
        encoding="utf-8", errors="replace"
    )
    report = (run_dir / "assertion_report.txt").read_text(
        encoding="utf-8", errors="replace"
    )
    required_pass = (
        "PASS_M4_STATEFUL_REAL sequences=80 batches=800 descriptors=12880 "
        "outputs=9360 local_outputs=7677 motion_outputs=1683"
    )
    required_stream = (
        "PASS_M4_STATEFUL_STREAMING sequences=80 batches=800 outputs=9360 "
        "fifo_writes=9360 fifo_reads=9360"
    )
    if required_pass not in simulation or required_stream not in simulation:
        raise ValueError(f"depth {depth} lacks streaming VCS admission markers")
    if re.search(r"Fatal:|Assertion failed|failed at", simulation + report):
        raise ValueError(f"depth {depth} contains a VCS failure")
    perf_match = re.search(
        r"PASS_M4_STATEFUL_STREAMING_PERF sequences=80 batches=800 "
        r"outputs=9360 cycles=(\d+)",
        simulation,
    )
    finish_match = re.search(r"\$finish at simulation time\s+(\d+)", simulation)
    stall_match = re.search(
        r"PASS_M4_STATEFUL_REAL .*?request_stalls=(\d+) "
        r"output_stalls=(\d+)",
        simulation,
    )
    if perf_match is None or finish_match is None or stall_match is None:
        raise ValueError(f"depth {depth} lacks always-ready streaming timing evidence")
    if int(stall_match.group(1)) != 0 or int(stall_match.group(2)) != 0:
        raise ValueError(f"depth {depth} streaming performance run was stalled")

    attempts, local_matches = cover(report, "cp_local_commit")
    other_covers = {
        name: cover(report, name)[1]
        for name in (
            "cp_motion_commit",
            "cp_queue_decouples_rmw",
            "cp_queue_push_pop_overlap",
            "cp_full_queue_push_pop_overlap",
            "cp_tail_pointer_wrap",
            "cp_head_pointer_wrap",
            "cp_next_batch_accepts_with_state_pending",
        )
    }
    if local_matches != 7677 or other_covers["cp_motion_commit"] != 1683:
        raise ValueError(f"depth {depth} cover population changed")
    for name in (
        "cp_queue_push_pop_overlap",
        "cp_full_queue_push_pop_overlap",
        "cp_tail_pointer_wrap",
        "cp_head_pointer_wrap",
        "cp_next_batch_accepts_with_state_pending",
    ):
        if other_covers[name] <= 0:
            raise ValueError(f"depth {depth} missed required cover: {name}")
    return {
        "queue_depth": depth,
        "payload_bits": depth * ENTRY_BITS,
        "streaming_cycles": int(perf_match.group(1)),
        "finish_time_ps": int(finish_match.group(1)),
        "assertion_sampled_cycles": attempts,
        "covers": other_covers,
        "run_dir": str(run_dir.resolve()),
        "evidence_ledger_sha256": sha256(run_dir / "evidence.sha256"),
    }


def summarize(points: list[dict[str, Any]]) -> dict[str, Any]:
    if not points:
        raise ValueError("streaming queue DSE needs at least one point")
    points = sorted(points, key=lambda point: point["queue_depth"])
    if len({point["queue_depth"] for point in points}) != len(points):
        raise ValueError("streaming queue depths must be unique")
    best_cycles = min(point["streaming_cycles"] for point in points)
    q1_cycles = next(
        (point["streaming_cycles"] for point in points
         if point["queue_depth"] == 1),
        None,
    )
    for point in points:
        point["speedup_vs_slowest"] = max(
            row["streaming_cycles"] for row in points
        ) / point["streaming_cycles"]
        point["cycle_penalty_vs_best"] = (
            point["streaming_cycles"] - best_cycles
        )
        if q1_cycles is not None:
            point["speedup_vs_q1"] = q1_cycles / point["streaming_cycles"]
            point["cycle_reduction_vs_q1_fraction"] = (
                (q1_cycles - point["streaming_cycles"]) / q1_cycles
            )
    return {
        "entry_bits": ENTRY_BITS,
        "candidate": "Q1_PREMACRO_PENDING_LOGIC_AREA",
        "throughput_ablation": min(
            point["queue_depth"] for point in points
            if point["streaming_cycles"] == best_cycles
        ),
        "points": points,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--point", action="append", nargs=2, metavar=("DEPTH", "RUN_DIR"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    points = [load_point(int(depth), Path(path)) for depth, path in args.point]
    result = {
        "schema": "m4_state_queue_streaming_vcs_dse_v1",
        "status": "PASS_M4_STATE_QUEUE_STREAMING_VCS_DSE_PREMACRO",
        "claim_boundary": (
            "Synopsys VCS assertion-sampled wall-cycle comparison on the same "
            "bounded checkpoint-bitmap, Motion-enriched 80-sequence workload "
            "with always-ready weight-request and output interfaces. This is "
            "not the full network population, real checkpoint-weight Acc32, "
            "energy, or paper PPA."
        ),
        **summarize(points),
        "summarizer_sha256": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"PASS streaming queue DSE -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
