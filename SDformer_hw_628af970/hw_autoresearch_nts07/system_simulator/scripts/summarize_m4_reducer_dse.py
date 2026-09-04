#!/usr/bin/env python3
"""Select the M4 compact-reducer knee from admitted wall-cycle runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def select_knee(variants: dict[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    if sorted(variants) != [2, 4, 8]:
        raise ValueError("M4 reducer knee requires R2/R4/R8")
    ratios: dict[str, Any] = {}
    for line in ("local", "hybrid"):
        speeds = {
            slots: item["variants"][line]["speedup_vs_same_width_dense_wall"]
            for slots, item in variants.items()
        }
        ratios[line] = {
            "r4_speedup_over_r2": speeds[4] / speeds[2],
            "r8_speedup_over_r4": speeds[8] / speeds[4],
            "r4_dense_wall_speedup": speeds[4],
            "r8_dense_wall_speedup": speeds[8],
        }
    selected = 4 if (
        min(item["r4_speedup_over_r2"] for item in ratios.values()) >= 1.5
        and max(item["r8_speedup_over_r4"] for item in ratios.values()) <= 1.25
    ) else 2
    return selected, ratios


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", action="append", nargs=2, metavar=("SLOTS", "JSON"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    variants: dict[int, dict[str, Any]] = {}
    identity_contract = None
    architecture_contract = None
    for raw_slots, raw_path in args.variant:
        slots = int(raw_slots)
        payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        if payload.get("status") != "PASS_M4_EXECUTABLE_SINGLE_BUFFER_WALL_CYCLE_MODEL":
            raise ValueError(f"unadmitted M4 DSE: {raw_path}")
        if payload["architecture"]["reduce_slots_per_context"] != slots:
            raise ValueError("reducer slot identity mismatch")
        current_identity = payload["identities"]
        current_arch = {
            key: value
            for key, value in payload["architecture"].items()
            if key not in {"reduce_slots_per_context", "shared_reducer_signed_adders"}
        }
        if identity_contract is None:
            identity_contract = current_identity
            architecture_contract = current_arch
        elif current_identity != identity_contract or current_arch != architecture_contract:
            raise ValueError("R2/R4/R8 did not use one trace/architecture contract")
        variants[slots] = payload
    selected, ratios = select_knee(variants)
    table = {}
    for slots in sorted(variants):
        item = variants[slots]
        table[str(slots)] = {
            "signed_adders": item["architecture"]["shared_reducer_signed_adders"],
            "local": {
                key: item["variants"]["local"][key]
                for key in (
                    "speedup_vs_p1_sparse_wall",
                    "speedup_vs_same_width_dense_wall",
                    "p1_sparse_sample_speedup_min",
                    "same_width_dense_sample_speedup_min",
                )
            },
            "hybrid": {
                key: item["variants"]["hybrid"][key]
                for key in (
                    "speedup_vs_p1_sparse_wall",
                    "speedup_vs_same_width_dense_wall",
                    "p1_sparse_sample_speedup_min",
                    "same_width_dense_sample_speedup_min",
                )
            },
        }
    result = {
        "schema": "m4_compact_reducer_knee_v1",
        "status": "PASS_M4_R4_REDUCER_KNEE_SELECTED" if selected == 4 else "REVIEW",
        "selected_reduce_slots": selected,
        "selection_rule": (
            "select R4 when both lines gain at least 1.5x over R2 and R8 adds at "
            "most 1.25x over R4 while doubling shared SIMD adders at each step"
        ),
        "marginal_speed": ratios,
        "variants": table,
        "claim_boundary": (
            "wall-cycle versus shared-adder-count knee; mapped area/power and SRAM macros "
            "remain required before an EDAP selection"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = [
        "# M4 compact-reducer DSE\n\n",
        "| slots/context | signed adders | Local vs dense P16 | Hybrid vs dense P16 | Local sample min | Hybrid sample min |\n",
        "|---:|---:|---:|---:|---:|---:|\n",
    ]
    for slots, item in table.items():
        report.append(
            f"| {slots} | {item['signed_adders']} | "
            f"{item['local']['speedup_vs_same_width_dense_wall']:.6f}x | "
            f"{item['hybrid']['speedup_vs_same_width_dense_wall']:.6f}x | "
            f"{item['local']['same_width_dense_sample_speedup_min']:.6f}x | "
            f"{item['hybrid']['same_width_dense_sample_speedup_min']:.6f}x |\n"
        )
    report.append(
        f"\nSelected functional knee: **R{selected}**. This is not yet a mapped-EDAP decision.\n"
    )
    args.output.with_suffix(".md").write_text("".join(report), encoding="utf-8")
    print(f"PASS: selected R{selected} and wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
