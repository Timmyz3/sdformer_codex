#!/usr/bin/env python3
"""Compose the M4 source kernel with its synchronous Local/Motion state cost.

The executable M4 wall-cycle model already charges one output-retirement cycle
per 96-lane accumulator tile.  The integrated state engine accepts an absolute
Local value and writes all six banks during that retirement cycle.  A Motion
value instead performs a synchronous read followed by a writeback.  This
script conservatively charges one non-overlapped bubble for every Motion
output.  Real M4 execution can hide some of those bubbles behind source work
or the state-transaction queue, so this result is an explicit upper bound.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_state_outputs(
    wall: Any,
    records: list[dict[str, str]],
    *,
    line: str,
    output_lanes: int,
) -> dict[str, Any]:
    """Count direct writes and RMWs on the same row bundles used by M4."""
    if line not in ("local", "hybrid"):
        raise ValueError(f"unsupported line: {line}")
    if output_lanes <= 0:
        raise ValueError("output_lanes must be positive")
    local_outputs = 0
    motion_outputs = 0
    per_sample: dict[str, dict[str, int]] = {}
    for bundle in wall.ordered_row_bundles(records):
        row = records[bundle[0]]
        sample = str(int(row["sample_id"]))
        lane_tiles = math.ceil(int(row["output_channel_fanout"]) / output_lanes)
        use_motion = line == "hybrid" and row["row_use_motion"].lower() == "true"
        item = per_sample.setdefault(
            sample, {"local_outputs": 0, "motion_outputs": 0}
        )
        field = "motion_outputs" if use_motion else "local_outputs"
        item[field] += lane_tiles
        if use_motion:
            motion_outputs += lane_tiles
        else:
            local_outputs += lane_tiles
    return {
        "local_outputs": local_outputs,
        "motion_outputs": motion_outputs,
        "outputs": local_outputs + motion_outputs,
        "per_sample": per_sample,
    }


def compose_state_cost(kernel: dict[str, Any], counts: dict[str, int]) -> dict[str, Any]:
    """Add a conservative non-overlapped synchronous-state cost bound."""
    outputs = int(counts["outputs"])
    local_outputs = int(counts["local_outputs"])
    motion_outputs = int(counts["motion_outputs"])
    if outputs != int(kernel["output_cycles"]):
        raise ValueError(
            "state-output population does not match M4 output cycles: "
            f"state={outputs} m4={kernel['output_cycles']}"
        )
    m4_wall = int(kernel["m4_wall_cycles"])
    dense_wall = int(kernel["same_width_dense_wall_cycles"])
    p1_wall = int(kernel["p1_sparse_wall_cycles"])
    stateful_upper_bound = m4_wall + motion_outputs
    result = {
        **kernel,
        "state_transactions": outputs,
        "state_direct_write_transactions": local_outputs,
        "state_rmw_transactions": motion_outputs,
        "state_bank_reads": motion_outputs * 6,
        "state_bank_writes": outputs * 6,
        "state_rmw_extra_cycles": motion_outputs,
        "stateful_nonoverlap_cycles_upper_bound": stateful_upper_bound,
        "speedup_vs_p1_sparse_lower_bound": p1_wall / stateful_upper_bound,
        "speedup_vs_same_width_dense_lower_bound": dense_wall / stateful_upper_bound,
    }
    per_sample_counts = counts.get("per_sample", {})
    for sample, sample_kernel in result.get("per_sample", {}).items():
        sample_counts = per_sample_counts[sample]
        sample_outputs = (
            sample_counts["local_outputs"] + sample_counts["motion_outputs"]
        )
        if sample_outputs != int(sample_kernel["output_cycles"]):
            raise ValueError(f"sample {sample} output population mismatch")
        sample_stateful_upper_bound = (
            int(sample_kernel["m4_wall_cycles"])
            + sample_counts["motion_outputs"]
        )
        sample_kernel.update({
            "state_transactions": sample_outputs,
            "state_direct_write_transactions": sample_counts["local_outputs"],
            "state_rmw_transactions": sample_counts["motion_outputs"],
            "state_rmw_extra_cycles": sample_counts["motion_outputs"],
            "stateful_nonoverlap_cycles_upper_bound": sample_stateful_upper_bound,
            "speedup_vs_p1_sparse_lower_bound": (
                sample_kernel["p1_sparse_wall_cycles"] /
                sample_stateful_upper_bound
            ),
            "speedup_vs_same_width_dense_lower_bound": (
                sample_kernel["same_width_dense_wall_cycles"] /
                sample_stateful_upper_bound
            ),
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--identity", action="append", nargs=2,
        metavar=("LABEL", "TILE_DIR"), required=True,
    )
    parser.add_argument("--issue-width", type=int, default=16)
    parser.add_argument("--contexts", type=int, default=4)
    parser.add_argument("--reduce-slots", type=int, default=4)
    parser.add_argument("--output-lanes", type=int, default=96)
    parser.add_argument(
        "--availability-mode",
        choices=("temporal_fenced", "layer_materialized_greedy"),
        default="temporal_fenced",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    wall_path = script_dir / "analyze_m4_descriptor_resident_wall_cycles.py"
    validator_path = script_dir / "build_dual_line_tile_memory_trace.py"
    wall = load_module(wall_path, "m4_stateful_wall_kernel")
    validator = load_module(validator_path, "m4_stateful_tile_validator")

    loaded = []
    identities: dict[str, Any] = {}
    for label, raw_directory in args.identity:
        directory = Path(raw_directory).resolve()
        manifest, records, current, previous = validator.validate(directory)
        loaded.append((label, directory, records, current, previous))
        identities[label] = {
            "directory": str(directory),
            "records": len(records),
            "sample_ids": sorted({int(row["sample_id"]) for row in records}),
            "checkpoint_sha256": manifest["run_context"]["artifact_identity"][
                "checkpoint_sha256"
            ],
            "manifest_sha256": sha256(directory / "manifest.json"),
            "records_sha256": sha256(directory / "tile_records.csv"),
            "packed_tiles_sha256": sha256(directory / "packed_tiles.npz"),
        }

    variants: dict[str, Any] = {}
    for line in ("local", "hybrid"):
        per_identity = {}
        for label, _directory, records, current, previous in loaded:
            kernel = wall.analyze_identity(
                records, current, previous,
                line=line,
                issue_width=args.issue_width,
                contexts=args.contexts,
                reduce_slots=args.reduce_slots,
                output_lanes=args.output_lanes,
                availability_mode=args.availability_mode,
            )
            counts = count_state_outputs(
                wall, records, line=line, output_lanes=args.output_lanes
            )
            per_identity[label] = compose_state_cost(kernel, counts)
        stateful_upper_bound = sum(
            item["stateful_nonoverlap_cycles_upper_bound"]
            for item in per_identity.values()
        )
        m4 = sum(item["m4_wall_cycles"] for item in per_identity.values())
        dense = sum(
            item["same_width_dense_wall_cycles"] for item in per_identity.values()
        )
        p1 = sum(item["p1_sparse_wall_cycles"] for item in per_identity.values())
        variants[line] = {
            "m4_wall_cycles": m4,
            "state_rmw_extra_cycles": sum(
                item["state_rmw_extra_cycles"] for item in per_identity.values()
            ),
            "stateful_nonoverlap_cycles_upper_bound": stateful_upper_bound,
            "p1_sparse_wall_cycles": p1,
            "same_width_dense_wall_cycles": dense,
            "speedup_vs_p1_sparse_lower_bound": p1 / stateful_upper_bound,
            "speedup_vs_same_width_dense_lower_bound": dense / stateful_upper_bound,
            "per_identity": per_identity,
        }

    variants["hybrid"]["speedup_vs_local_lower_bound"] = (
        variants["local"]["stateful_nonoverlap_cycles_upper_bound"]
        / variants["hybrid"]["stateful_nonoverlap_cycles_upper_bound"]
    )
    output = {
        "schema": "m4_stateful_wall_cycles_v1",
        "status": "PASS_M4_STATEFUL_NONOVERLAP_CYCLE_UPPER_BOUND",
        "claim_boundary": (
            "M4 temporal-fenced source-kernel wall cycles plus a conservative "
            "non-overlapped synchronous 1RW RMW cycle per Motion output; Local "
            "absolute state writes overlap the already charged output-retirement "
            "cycle. This is an upper bound because RTL source work and the state "
            "transaction queue may hide RMW service. Excludes external output "
            "backpressure, narrow-client/abort "
            "contention, target SRAM macro timing/energy, DRAM, unrelated network "
            "operators, and full-system overlap. This is not paper PPA."
        ),
        "state_contract": {
            "banks": 6,
            "lanes_per_bank": 16,
            "accumulator_width_bits": 32,
            "local_service": "six-bank atomic absolute write in output cycle",
            "motion_service": "six-bank synchronous read plus next-cycle writeback",
            "motion_extra_cycles_per_output": 1,
            "external_output_ready": "always",
        },
        "architecture": {
            "issue_width": args.issue_width,
            "contexts": args.contexts,
            "reduce_slots": args.reduce_slots,
            "output_lanes": args.output_lanes,
            "availability_mode": args.availability_mode,
        },
        "identities": identities,
        "variants": variants,
        "source_sha256": {
            "analyzer": sha256(Path(__file__)),
            "m4_kernel_model": sha256(wall_path),
            "tile_validator": sha256(validator_path),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        "PASS M4 stateful wall cycles: "
        "upper-bound local="
        f"{variants['local']['stateful_nonoverlap_cycles_upper_bound']} "
        "hybrid="
        f"{variants['hybrid']['stateful_nonoverlap_cycles_upper_bound']} "
        "hybrid/local lower-bound="
        f"{variants['hybrid']['speedup_vs_local_lower_bound']:.6f}x "
        "hybrid/dense lower-bound="
        f"{variants['hybrid']['speedup_vs_same_width_dense_lower_bound']:.6f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
