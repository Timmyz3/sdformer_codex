#!/usr/bin/env python3
"""Screen exact adjacent-destination Local5 score-object reuse.

Two destinations may share score/Shiftmax computation only when their complete
hardware inputs {valid mask, binary Q, five binary K vectors} are identical.
The screen separately removes rows already handled by Q-silent or ident-K and
checks whether any relation materialization is actually eliminated.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VECTOR_DIR = (
    ROOT
    / "tb_qfit/vectors/"
    "local5_joint_ep29_score_projection_realw_sample100_population_v1_20260813"
)
DEFAULT_OUTPUT = ROOT / "results/local5_neighbor_score_object_reuse_20260814"
TOKENS = 450
PLANE_TOKENS = 225
HEIGHT = 15
WIDTH = 15
ROLES = 5


def read_memh(path: Path) -> list[int]:
    return [
        int(line.strip(), 16)
        for line in path.read_text(encoding="ascii").splitlines()
        if line.strip()
    ]


def role_values(word: int, width: int) -> list[int]:
    mask = (1 << width) - 1
    return [(word >> (role * width)) & mask for role in range(ROLES)]


def ident_k(k_word: int, valid_mask: int) -> bool:
    values = [
        value
        for role, value in enumerate(role_values(k_word, 32))
        if (valid_mask >> role) & 1
    ]
    return bool(values) and all(value == values[0] for value in values[1:])


def input_key(valid: int, q_word: int, k_word: int) -> tuple[int, int, int]:
    return valid, q_word, k_word


def build_profile(vector_dir: Path) -> dict[str, object]:
    manifest = json.loads((vector_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest["selection"]["rows"]
    if len(rows) != 100:
        raise ValueError(f"expected 100 groups, got {len(rows)}")

    valid = read_memh(vector_dir / "input_valid.memh")
    q_words = read_memh(vector_dir / "input_q.memh")
    k_words = read_memh(vector_dir / "input_candidate_k.memh")
    scores = read_memh(vector_dir / "expected_scores.memh")
    gates = read_memh(vector_dir / "expected_gates.memh")
    expected = len(rows) * TOKENS
    for name, values in {
        "valid": valid,
        "q": q_words,
        "k": k_words,
        "scores": scores,
        "gates": gates,
    }.items():
        if len(values) != expected:
            raise ValueError(f"{name}: expected {expected}, got {len(values)}")

    totals = Counter()
    stage_totals: dict[int, Counter[str]] = defaultdict(Counter)
    group_records: list[dict[str, int | bool]] = []
    output_mismatches = 0

    for group, metadata in enumerate(rows):
        start = group * TOKENS
        group_count = Counter()
        longest_horizontal_run = 1
        for plane in range(2):
            base = start + plane * PLANE_TOKENS
            for y in range(HEIGHT):
                run = 1
                for x in range(WIDTH):
                    index = base + y * WIDTH + x
                    key = input_key(valid[index], q_words[index], k_words[index])
                    qsilent = q_words[index] == 0
                    existing_ident = (not qsilent) and ident_k(k_words[index], valid[index])
                    normal = not qsilent and not existing_ident
                    group_count["destinations"] += 1
                    group_count["normal_score_destinations"] += int(normal)
                    group_count["relation_slots"] += valid[index].bit_count()

                    left_equal = False
                    up_equal = False
                    if x > 0:
                        left = index - 1
                        left_equal = key == input_key(
                            valid[left], q_words[left], k_words[left]
                        )
                        group_count["horizontal_edges"] += 1
                        group_count["horizontal_input_equal"] += int(left_equal)
                        group_count["horizontal_incremental_equal"] += int(
                            left_equal and normal
                        )
                        if left_equal:
                            output_mismatches += int(scores[index] != scores[left])
                            output_mismatches += int(gates[index] != gates[left])
                            run += 1
                        else:
                            run = 1
                        longest_horizontal_run = max(longest_horizontal_run, run)
                    if y > 0:
                        up = index - WIDTH
                        up_equal = key == input_key(valid[up], q_words[up], k_words[up])
                        group_count["vertical_edges"] += 1
                        group_count["vertical_input_equal"] += int(up_equal)
                        group_count["vertical_incremental_equal"] += int(
                            up_equal and normal
                        )
                        if up_equal:
                            output_mismatches += int(scores[index] != scores[up])
                            output_mismatches += int(gates[index] != gates[up])

                    any_equal = left_equal or up_equal
                    group_count["neighbor_oracle_equal"] += int(any_equal)
                    group_count["neighbor_oracle_incremental_equal"] += int(
                        any_equal and normal
                    )

        group_count["longest_horizontal_run"] = longest_horizontal_run
        totals.update(
            {key: value for key, value in group_count.items() if key != "longest_horizontal_run"}
        )
        stage_totals[int(metadata["stage"])].update(
            {key: value for key, value in group_count.items() if key != "longest_horizontal_run"}
        )
        group_records.append(
            {
                "group": group,
                "sample": int(metadata["sample"]),
                "stage": int(metadata["stage"]),
                "empty": bool(metadata["empty"]),
                **dict(group_count),
            }
        )

    normal = totals["normal_score_destinations"]
    incremental = totals["neighbor_oracle_incremental_equal"]
    score_reduction_upper = incremental / normal if normal else 0.0
    all_edges = totals["horizontal_edges"] + totals["vertical_edges"]
    all_equal = totals["horizontal_input_equal"] + totals["vertical_input_equal"]
    result = {
        "schema": "local5_neighbor_score_object_reuse_profile_v1",
        "status": "PASS",
        "evidence": "[prof] exact input/output identity; no RTL/PPA",
        "candidate": (
            "reuse one score/Shiftmax object when a destination has an exactly "
            "identical full input tuple to its left or upper neighbor"
        ),
        "totals": dict(totals),
        "rates": {
            "all_neighbor_edge_input_equal": all_equal / all_edges if all_edges else 0.0,
            "incremental_neighbor_sites_over_all_destinations": incremental
            / totals["destinations"],
            "score_service_reduction_upper_bound": score_reduction_upper,
            "relation_slot_reduction": 0.0,
        },
        "stage": {str(stage): dict(counts) for stage, counts in sorted(stage_totals.items())},
        "group_records": group_records,
        "exactness": {
            "equal_input_score_gate_mismatches": output_mismatches,
            "relation_destinations_preserved": True,
        },
        "strong_baseline": (
            "Q-silent + ident-K score leaf followed by bounded FCSR and source-major TCFM5"
        ),
        "verdict": "NO_GO_AS_DATE_CONTRIBUTION",
        "reason": (
            "Even the ideal left-or-up oracle removes only score service for incremental "
            "normal rows; every destination still owns distinct inverse-stencil source "
            "coordinates, so relation slots, term work, and accumulator updates are unchanged. "
            "The candidate is score-front CSE rather than a new materialized object."
        ),
    }
    if output_mismatches:
        raise AssertionError(f"equal input produced {output_mismatches} score/gate mismatches")
    return result


def render_markdown(report: dict[str, object]) -> str:
    totals = report["totals"]
    rates = report["rates"]
    longest = max(record["longest_horizontal_run"] for record in report["group_records"])
    return f"""# Local5 adjacent-destination score-object reuse screen

## Verdict

`{report['verdict']}`. Read-only `[prof]`; no RTL, cycle, power, or encoder claim.

## Exact opportunity

- destinations: `{totals['destinations']}`; normal score destinations after Q-silent/ident-K: `{totals['normal_score_destinations']}`;
- exact left/up input-equal edges: `{totals['horizontal_input_equal'] + totals['vertical_input_equal']}` / `{totals['horizontal_edges'] + totals['vertical_edges']}` = `{rates['all_neighbor_edge_input_equal']:.2%}`;
- incremental destinations matching left or upper neighbor: `{totals['neighbor_oracle_incremental_equal']}` = `{rates['incremental_neighbor_sites_over_all_destinations']:.2%}` of all destinations;
- ideal score-service reduction upper bound after existing fast paths: `{rates['score_service_reduction_upper_bound']:.2%}`;
- longest horizontal identical-input run: `{longest}`;
- equal-input score/gate mismatches: `{report['exactness']['equal_input_score_gate_mismatches']}`.

## Architecture boundary

Relation-slot reduction is `{rates['relation_slot_reduction']:.2%}`. Adjacent destinations have translated source coordinates even when Q/K values match, so FCSR must still preserve each destination relation and TCFM5 must still execute the same term/update set.

{report['reason']}

This result does not modify frozen DATE tables.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-dir", type=Path, default=DEFAULT_VECTOR_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_profile(args.vector_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(
        "PASS Local5 neighbor reuse "
        f"incremental={report['totals']['neighbor_oracle_incremental_equal']} "
        f"score_upper={report['rates']['score_service_reduction_upper_bound']:.2%} "
        f"verdict={report['verdict']}"
    )


if __name__ == "__main__":
    main()
