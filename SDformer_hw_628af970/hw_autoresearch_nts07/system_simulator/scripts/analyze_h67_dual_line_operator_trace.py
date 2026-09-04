#!/usr/bin/env python3
"""Audit and compare Local/Motion per-operator temporal work traces."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PASS = "PASS_EXACT_SOURCE_WORK"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def integer(row: dict[str, str], key: str) -> int:
    return int(row[key])


def truth(value: str) -> bool:
    return value.lower() == "true"


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize_pass_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    local = sum(integer(row, "local_work") for row in rows)
    motion = sum(integer(row, "motion_work") for row in rows)
    selected = sum(integer(row, "selected_work") for row in rows)
    saved = sum(integer(row, "selector_saved_work") for row in rows)
    selector_rows = sum(integer(row, "selector_rows") for row in rows)
    motion_rows = sum(integer(row, "motion_selected_rows") for row in rows)
    state_rows = [row for row in rows if truth(row["state_valid"])]
    state_local = sum(integer(row, "local_work") for row in state_rows)
    state_selected = sum(integer(row, "selected_work") for row in state_rows)
    return {
        "trace_rows": len(rows),
        "operator_calls": len({(row["name"], row["operator_call_index"]) for row in rows}),
        "local_work": local,
        "motion_only_work": motion,
        "selected_work": selected,
        "selector_saved_work": saved,
        "selector_rows": selector_rows,
        "motion_selected_rows": motion_rows,
        "motion_selected_row_fraction": ratio(motion_rows, selector_rows),
        "local_to_selected_work_ratio": ratio(local, selected),
        "local_to_motion_only_work_ratio": ratio(local, motion),
        "selected_work_reduction": ratio(saved, local),
        "state_valid_local_work": state_local,
        "state_valid_selected_work": state_selected,
        "state_valid_local_to_selected_work_ratio": ratio(state_local, state_selected),
    }


def audit_rows(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    for index, row in enumerate(rows):
        if row.get("status") != PASS:
            continue
        local = integer(row, "local_work")
        motion = integer(row, "motion_work")
        selected = integer(row, "selected_work")
        saved = integer(row, "selector_saved_work")
        selector_rows = integer(row, "selector_rows")
        local_rows = integer(row, "local_selected_rows")
        motion_rows = integer(row, "motion_selected_rows")
        fanout = integer(row, "output_channel_fanout")
        current_sources = integer(row, "current_source_count")
        positive_sources = integer(row, "positive_transition_source_count")
        negative_sources = integer(row, "negative_transition_source_count")
        valid_source_work = integer(row, "valid_source_work")
        if fanout <= 0:
            errors.append(f"row {index}: nonpositive fanout")
            continue
        if local != current_sources * fanout:
            errors.append(f"row {index}: Local source/fanout conservation mismatch")
        if motion != (positive_sources + negative_sources) * fanout:
            errors.append(f"row {index}: Motion source/fanout conservation mismatch")
        if any(value % fanout for value in (local, motion, selected, valid_source_work)):
            errors.append(f"row {index}: work is not divisible by fanout")
        if valid_source_work < local:
            errors.append(f"row {index}: Local exceeds valid dense source work")
        if selected > local:
            errors.append(f"row {index}: selected exceeds Local")
        if selected > motion and truth(row["state_valid"]):
            errors.append(f"row {index}: selected exceeds Motion with valid state")
        if saved != local - selected:
            errors.append(f"row {index}: saved work mismatch")
        if local_rows + motion_rows != selector_rows:
            errors.append(f"row {index}: selector row conservation mismatch")
        if not truth(row["state_valid"]) and motion_rows:
            errors.append(f"row {index}: invalid prior state selected Motion")
    return errors


def summarize(label: str, path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    errors = audit_rows(rows)
    status = Counter(row.get("status", "") for row in rows)
    pass_rows = [row for row in rows if row.get("status") == PASS]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    bypass: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("status") == PASS:
            grouped[row["name"]].append(row)
        else:
            bypass[row.get("status", "")].add(row.get("name", ""))
    operators = []
    for name, operator_rows in grouped.items():
        item = summarize_pass_rows(operator_rows)
        item.update({
            "name": name,
            "operator": operator_rows[0]["operator"],
            "scope": operator_rows[0]["scope"],
        })
        operators.append(item)
    operators.sort(key=lambda row: (-row["selector_saved_work"], row["name"]))
    return {
        "label": label,
        "path": str(path.resolve()),
        "status": "PASS_TRACE_AUDIT" if not errors else "FAIL_TRACE_AUDIT",
        "errors": errors,
        "row_status": dict(status),
        "totals": summarize_pass_rows(pass_rows),
        "bypass_operators": {
            key: sorted(value) for key, value in sorted(bypass.items())
        },
        "operators": operators,
    }


def write_csv(path: Path, traces: list[dict[str, Any]]) -> None:
    rows = []
    for trace in traces:
        for operator in trace["operators"]:
            rows.append({"trace": trace["label"], **operator})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, traces: list[dict[str, Any]]) -> None:
    lines = [
        "# Local/Motion full-network source-work trace audit\n",
        "This is an exact ordered operation-count audit across the supplied traces, not RTL cycles, energy, or a publication mean.\n",
        "\n| trace | exact calls | Local work | Motion-only work | selected work | Local/selected | saved | Motion rows |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for trace in traces:
        total = trace["totals"]
        lines.append(
            f"| {trace['label']} | {total['operator_calls']} | {total['local_work']} | "
            f"{total['motion_only_work']} | {total['selected_work']} | "
            f"{total['local_to_selected_work_ratio']:.6f}x | "
            f"{total['selected_work_reduction']:.4%} | "
            f"{total['motion_selected_row_fraction']:.4%} |\n"
        )
    for trace in traces:
        lines.extend([
            f"\n## {trace['label']} top absolute savings\n",
            "| operator | Local work | selected work | ratio | saved | Motion rows |\n",
            "|---|---:|---:|---:|---:|---:|\n",
        ])
        for row in trace["operators"][:12]:
            lines.append(
                f"| `{row['name']}` | {row['local_work']} | {row['selected_work']} | "
                f"{row['local_to_selected_work_ratio']:.6f}x | "
                f"{row['selector_saved_work']} | {row['motion_selected_row_fraction']:.4%} |\n"
            )
        lines.append("\nBypass/unqualified operator counts:\n")
        for status, names in trace["bypass_operators"].items():
            lines.append(f"- `{status}`: {len(names)}\n")
    path.write_text("".join(lines), encoding="utf-8")


def parse_trace(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("trace must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", type=parse_trace, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    traces = [summarize(label, path) for label, path in args.trace]
    if any(trace["status"].startswith("FAIL") for trace in traces):
        raise RuntimeError("trace audit failed")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(
        json.dumps({"schema": "h67_dual_line_trace_analysis_v0", "traces": traces}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    write_csv(args.output / "operator_comparison.csv", traces)
    write_report(args.output / "REPORT.md", traces)
    print(json.dumps({trace["label"]: trace["totals"] for trace in traces}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
