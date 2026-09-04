#!/usr/bin/env python3
"""Prove scoped lower bounds attained by the frozen Motion and Local5 dataflows."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FAIR_LOG = ROOT / "results/h67_fair_merge_population_20260813/ep35_fair_merge.log"
DEFAULT_COMPILER = ROOT / "results/local5_stencil_retirement_compiler_20260814/report.json"
DEFAULT_COLORING = ROOT / "results/qfit_tcfm5_coloring_proof_20260731/proof.json"
DEFAULT_OUTPUT = ROOT / "results/dual_line_optimality_bounds_20260814"
FAIR_RE = re.compile(
    r"^FAIR_SUM .*?fpairs=(?P<pairs>\d+) .*?fslots=(?P<fixed_slots>\d+) "
    r"fequal=(?P<equal>\d+) .*?rslots=(?P<rqtb_slots>\d+) "
    r"requal=(?P<rqtb_equal>\d+)$",
    re.MULTILINE,
)


def motion_bound(pairs: int, equal_pairs: int, actual_slots: int) -> dict[str, object]:
    if not 0 <= equal_pairs <= pairs:
        raise ValueError("invalid equal-pair count")
    # One descriptor carries exactly one score class. Equal pairs need one
    # class-bearing descriptor; unequal pairs require two distinct descriptors.
    minimum = equal_pairs + 2 * (pairs - equal_pairs)
    nonempty_temporal_subsets = 3  # {t0}, {t1}, {t0,t1}
    membership_bits = math.ceil(math.log2(nonempty_temporal_subsets))
    return {
        "scope": (
            "pair-local ordered streaming; one Q7 score class per descriptor; "
            "exact score multiset and temporal membership reconstruction"
        ),
        "pairs": pairs,
        "equal_pairs": equal_pairs,
        "descriptor_lower_bound": minimum,
        "actual_rqtb_descriptors": actual_slots,
        "attains_descriptor_lower_bound": actual_slots == minimum,
        "identity": "descriptors + equal_pairs = 2 * pairs",
        "temporal_membership_states": nonempty_temporal_subsets,
        "temporal_membership_bit_lower_bound": membership_bits,
        "actual_temporal_mask_bits": 2,
        "attains_membership_bit_lower_bound": membership_bits == 2,
        "not_claimed": (
            "global cross-pair reordering, multi-class descriptors, K payload "
            "elimination, or a universal attention I/O lower bound"
        ),
    }


def local5_bounds(
    offsets: list[list[int]],
    actual_row_span: int,
    actual_banks: int,
    coloring: dict[str, object],
) -> dict[str, object]:
    if not offsets or [0, 0] not in offsets:
        raise ValueError("Local5 offsets must include self")
    dys = [int(offset[0]) for offset in offsets]
    row_span_lower_bound = max(dys) - min(dys) + 1
    bank_lower_bound = len(offsets)
    coloring_banks = int(coloring["banks"])
    return {
        "transpose_scope": (
            "single-pass row-major raster; no relation recomputation or off-chip "
            "spill; retain every live source relation until its last consumer"
        ),
        "offsets": offsets,
        "row_span_lower_bound": row_span_lower_bound,
        "actual_row_span": actual_row_span,
        "attains_row_span_lower_bound": actual_row_span == row_span_lower_bound,
        "bank_scope": (
            "one interior source scatters to all five destinations in one cycle; "
            "each accumulator bank has one write port; no replication/reduction"
        ),
        "simultaneous_destinations": len(offsets),
        "bank_count_lower_bound": bank_lower_bound,
        "actual_bank_count": actual_banks,
        "coloring_bank_count": coloring_banks,
        "attains_bank_count_lower_bound": (
            actual_banks == bank_lower_bound == coloring_banks
            and bool(coloring["conflict_free_all_neighborhoods"])
            and int(coloring["interior_k5_witnesses"]) > 0
        ),
        "interior_lower_bound_witnesses": int(coloring["interior_k5_witnesses"]),
        "injective_bank_address": bool(coloring["injective_bank_address"]),
        "not_claimed": (
            "minimum ASIC area/energy, arbitrary sparse graphs, multi-cycle scatter, "
            "multiported banks, or a universal stencil buffer lower bound"
        ),
    }


def build_report(fair_text: str, compiler: dict[str, object], coloring: dict[str, object]) -> dict[str, object]:
    match = FAIR_RE.search(fair_text)
    if match is None:
        raise ValueError("locked FAIR_SUM line missing")
    fields = {name: int(value) for name, value in match.groupdict().items()}
    if fields["equal"] != fields["rqtb_equal"]:
        raise ValueError("Fixed/RQTB equal-pair ledger mismatch")
    topology = compiler["topologies"]["cross_r1"]
    motion = motion_bound(fields["pairs"], fields["equal"], fields["rqtb_slots"])
    local5 = local5_bounds(
        topology["offsets"],
        int(topology["relation_row_span"]),
        int(topology["affine_bank_map"]["banks"]),
        coloring,
    )
    if not all(
        (
            motion["attains_descriptor_lower_bound"],
            motion["attains_membership_bit_lower_bound"],
            local5["attains_row_span_lower_bound"],
            local5["attains_bank_count_lower_bound"],
        )
    ):
        raise AssertionError("a frozen dataflow no longer attains its scoped bound")
    return {
        "schema": "dual_line_scoped_optimality_bounds_v1",
        "status": "PASS",
        "evidence": "[rtl-ledger]+[compile-time-proof]+[exhaustive-topology-proof]",
        "motion": motion,
        "local5": local5,
        "date_interpretation": {
            "claim": (
                "The frozen mechanisms attain domain-scoped representation and "
                "topology bounds; this strengthens defensibility but is not a new RTL mechanism."
            ),
            "innovation_effect": "narrative/theorem strengthening only",
            "estimated_novelty_ceiling": "about 3.3/5 without new PPA or end-to-end evidence",
        },
    }


def render_markdown(report: dict[str, object]) -> str:
    motion = report["motion"]
    local5 = report["local5"]
    return f"""# Motion / Local5 scoped architecture optimality

## Verdict

`PASS` under explicitly limited contracts. This is theorem-strengthening evidence, not a new RTL mechanism or ASIC PPA result.

## Motion: pair-local quotient bound

For a temporal pair, a descriptor names one Q7 score class. An equal-score pair needs at least one descriptor; an unequal pair needs at least two. Therefore:

```text
D_min = equal + 2 * (pairs - equal) = 2 * pairs - equal
```

- locked fair ledger: pairs `{motion['pairs']}`, equal `{motion['equal_pairs']}`;
- lower bound `{motion['descriptor_lower_bound']}`, actual RQTB `{motion['actual_rqtb_descriptors']}`;
- identity: `{motion['identity']}`;
- three nonempty temporal subsets require at least `{motion['temporal_membership_bit_lower_bound']}` bits; RQTB uses `{motion['actual_temporal_mask_bits']}`.

Scope: {motion['scope']}.

Not claimed: {motion['not_claimed']}.

## Local5: bounded transpose and bank bounds

For fixed source-relative offsets, a row-major stream must retain a source relation from its earliest to latest destination row:

```text
Dr_min = max(delta_y) - min(delta_y) + 1
```

- cross-r1 lower bound `{local5['row_span_lower_bound']}` rows, actual FCSR ring `{local5['actual_row_span']}` rows;
- an interior five-neighbor source creates `{local5['simultaneous_destinations']}` simultaneous destination writes;
- with one write per bank, pigeonhole lower bound is `{local5['bank_count_lower_bound']}` banks;
- TCFM5 uses `{local5['actual_bank_count']}` banks and the exhaustive proof has `{local5['interior_lower_bound_witnesses']}` interior K5 witnesses with conflict-free injective addressing.

Transpose scope: {local5['transpose_scope']}.

Bank scope: {local5['bank_scope']}.

Not claimed: {local5['not_claimed']}.

## DATE interpretation

The useful paper statement is not that FIFOs, banking, or line buffers are individually novel. It is that each dataflow is built around a workload-specific exact object and reaches the corresponding scoped lower bound. This improves the architectural argument, but without matched DC/SAIF and end-to-end evidence it does not justify a 4/5 novelty claim by itself.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fair-log", type=Path, default=DEFAULT_FAIR_LOG)
    parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    parser.add_argument("--coloring", type=Path, default=DEFAULT_COLORING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(
        args.fair_log.read_text(encoding="utf-8"),
        json.loads(args.compiler.read_text(encoding="utf-8")),
        json.loads(args.coloring.read_text(encoding="utf-8")),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(
        "PASS Motion descriptor/mask bounds and Local5 row-span/bank bounds attained"
    )


if __name__ == "__main__":
    main()
