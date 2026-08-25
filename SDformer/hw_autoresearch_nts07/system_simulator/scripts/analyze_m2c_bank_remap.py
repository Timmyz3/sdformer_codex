#!/usr/bin/env python3
"""Audit reversible XOR bank remaps on admitted Local/Motion real tiles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tile_validator() -> Any:
    path = Path(__file__).with_name("build_dual_line_tile_memory_trace.py")
    spec = importlib.util.spec_from_file_location("dual_line_tile_memory_trace", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import tile validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bank_assignment(tile_bits: int, issue_width: int, xor_shift: int | None) -> np.ndarray:
    if issue_width <= 0 or issue_width & (issue_width - 1):
        raise ValueError("issue_width must be a power of two")
    bank_bits = int(math.log2(issue_width))
    if xor_shift is not None and xor_shift < bank_bits:
        raise ValueError("xor_shift must use only source bits above the bank field")
    source = np.arange(tile_bits, dtype=np.int64)
    mapped = source if xor_shift is None else np.bitwise_xor(source, source >> xor_shift)
    return mapped & (issue_width - 1)


def issue_beats(bitmaps: np.ndarray, assignment: np.ndarray, issue_width: int) -> np.ndarray:
    if bitmaps.ndim != 2 or bitmaps.shape[1] != len(assignment):
        raise ValueError("bitmap/assignment shape mismatch")
    counts = np.stack(
        [bitmaps[:, assignment == bank].sum(axis=1) for bank in range(issue_width)], axis=1
    )
    return counts.max(axis=1).astype(np.int64)


def mapping_name(xor_shift: int | None) -> str:
    return "modulo" if xor_shift is None else f"xor_shift_{xor_shift}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", action="append", nargs=2, metavar=("LABEL", "TILE_DIR"), required=True)
    parser.add_argument("--issue-width", action="append", type=int, choices=(2, 4, 8), required=True)
    parser.add_argument("--survival-gain", type=float, default=0.03)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validator = load_tile_validator()
    args.output.mkdir(parents=True, exist_ok=True)

    identities: dict[str, Any] = {}
    loaded: list[tuple[str, list[dict[str, str]], np.ndarray, np.ndarray]] = []
    for label, raw_path in args.identity:
        directory = Path(raw_path).resolve()
        manifest, records, current_packed, previous_packed = validator.validate(directory)
        current = np.unpackbits(current_packed, axis=1, bitorder="little").astype(bool)
        previous = np.unpackbits(previous_packed, axis=1, bitorder="little").astype(bool)
        loaded.append((label, records, current, previous))
        identities[label] = {
            "directory": str(directory),
            "records": len(records),
            "tile_manifest_sha256": sha256(directory / "manifest.json"),
            "packed_tiles_sha256": sha256(directory / "packed_tiles.npz"),
            "tile_manifest": manifest,
        }

    widths: dict[str, Any] = {}
    for issue_width in sorted(set(args.issue_width)):
        bank_bits = int(math.log2(issue_width))
        shifts = [None, *range(bank_bits, 8)]
        assignments = {
            mapping_name(shift): bank_assignment(256, issue_width, shift) for shift in shifts
        }
        line_results: dict[str, Any] = {}
        for line in ("local", "hybrid"):
            global_totals = Counter({name: 0 for name in assignments})
            operator_totals: dict[str, Counter[str]] = defaultdict(Counter)
            identity_totals: dict[str, Counter[str]] = {}
            selected_sources = 0
            for label, records, current, previous in loaded:
                use_motion = np.asarray(
                    [row["row_use_motion"].lower() == "true" for row in records], dtype=bool
                )[:, None]
                selected = current if line == "local" else np.where(use_motion, current ^ previous, current)
                selected_sources += int(selected.sum())
                names = np.asarray([f"{label}:{row['name']}" for row in records])
                per_identity = Counter()
                for name, assignment in assignments.items():
                    beats = issue_beats(selected, assignment, issue_width)
                    total = int(beats.sum())
                    global_totals[name] += total
                    per_identity[name] = total
                    for operator in np.unique(names):
                        operator_totals[operator][name] += int(beats[names == operator].sum())
                identity_totals[label] = per_identity

            modulo = global_totals["modulo"]
            best_global = min(global_totals, key=global_totals.get)
            per_operator_beats = sum(min(values.values()) for values in operator_totals.values())
            operator_choices = Counter(min(values, key=values.get) for values in operator_totals.values())
            operator_rows = []
            for operator, values in operator_totals.items():
                best = min(values, key=values.get)
                base = values["modulo"]
                operator_rows.append({
                    "operator": operator,
                    "modulo_beats": base,
                    "best_mapping": best,
                    "best_beats": values[best],
                    "speedup": base / values[best] if values[best] else 1.0,
                })
            operator_rows.sort(key=lambda row: (row["speedup"], row["modulo_beats"]), reverse=True)
            per_operator_gain = modulo / per_operator_beats - 1.0
            line_results[line] = {
                "selected_sources": selected_sources,
                "global_mapping_beats": dict(global_totals),
                "identity_mapping_beats": {
                    label: dict(values) for label, values in identity_totals.items()
                },
                "best_global_mapping": best_global,
                "best_global_speedup_vs_modulo": modulo / global_totals[best_global],
                "per_operator_config_beats": per_operator_beats,
                "per_operator_config_speedup_vs_modulo": modulo / per_operator_beats,
                "per_operator_gain": per_operator_gain,
                "survives_gate": per_operator_gain >= args.survival_gain,
                "operator_choice_histogram": dict(operator_choices),
                "top_operator_opportunities": operator_rows[:20],
            }
        widths[f"p{issue_width}"] = line_results

    survived = any(
        line["survives_gate"] for width in widths.values() for line in width.values()
    )
    payload = {
        "schema": "m2c_reversible_xor_bank_remap_dse_v1",
        "status": "PASS_REMAP_SURVIVES" if survived else "PASS_REMAP_BELOW_SURVIVAL_GATE",
        "claim_boundary": (
            "exact bank-conflict issue beats on admitted real tile descriptors; reversible XOR uses "
            "bank=(source xor (source>>shift)) mod P and bank-local address=source/P; no RTL/PPA claim"
        ),
        "survival_gain": args.survival_gain,
        "identities": identities,
        "issue_widths": widths,
        "script_sha256": sha256(Path(__file__)),
        "validator_sha256": sha256(Path(__file__).with_name("build_dual_line_tile_memory_trace.py")),
    }
    json_path = args.output / "bank_remap_dse.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = [
        "# M2C reversible XOR bank-remap DSE\n\n",
        "| P | line | best global | global speedup | per-operator speedup | survives gate |\n",
        "|---:|---|---|---:|---:|---|\n",
    ]
    for width, lines in widths.items():
        for line_name, result in lines.items():
            report.append(
                f"| {width[1:]} | {line_name} | {result['best_global_mapping']} | "
                f"{result['best_global_speedup_vs_modulo']:.6f}x | "
                f"{result['per_operator_config_speedup_vs_modulo']:.6f}x | "
                f"{'YES' if result['survives_gate'] else 'NO'} |\n"
            )
    report.append(
        "\nThis is a bank-conflict DSE, not accelerator speedup. A per-operator result assumes offline "
        "weight prepacking and a mapping selector that are not present in M2B RTL.\n"
    )
    (args.output / "REPORT.md").write_text("".join(report), encoding="utf-8")
    print(f"PASS: {payload['status']} -> {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
