#!/usr/bin/env python3
"""Compare Local5 gate-cardinality candidates on one identical ordered trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from screen_local5_execution_object_codesign import analyze_arrays


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_profile(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = path / "ordered_term_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload_path = path / manifest["payload_file"]
    if sha256(payload_path) != manifest.get("payload_sha256"):
        raise ValueError(f"payload SHA mismatch: {payload_path}")
    qualification = manifest.get("qualification") or {}
    checks = qualification.get("checks") or {}
    required = (
        "captured_modules_12",
        "exact_target_block_set",
        "module_sample_pair_coverage",
        "rotating_flat_group_coverage",
        "exact_rotating_indices",
        "shape_t450_l32",
        "sampling_contract",
    )
    if not all(checks.get(key) is True for key in required):
        raise ValueError(f"profile coverage contract failed: {path}")
    groups = manifest.get("groups") or []
    if len(groups) != int(qualification.get("captured_groups", -1)):
        raise ValueError(f"group count mismatch: {path}")
    with np.load(payload_path, allow_pickle=False) as payload:
        analysis = analyze_arrays(payload, [int(group["stage"]) for group in groups])
    return manifest, analysis


def group_identity(group: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        group.get(key)
        for key in (
            "sample",
            "stage",
            "block",
            "window",
            "head",
            "flat_group",
            "module",
            "selection",
        )
    )


def metric_delta(candidate: float, baseline: float) -> float | None:
    return candidate / baseline - 1.0 if baseline else None


def build_report(baseline_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    baseline_manifest, baseline = load_profile(baseline_dir)
    candidate_manifest, candidate = load_profile(candidate_dir)
    baseline_groups = [group_identity(group) for group in baseline_manifest["groups"]]
    candidate_groups = [group_identity(group) for group in candidate_manifest["groups"]]
    if baseline_manifest.get("cohort_sha256") != candidate_manifest.get("cohort_sha256"):
        raise ValueError("cohort SHA differs")
    if baseline_groups != candidate_groups:
        raise ValueError("ordered group identities differ")
    if baseline["totals"]["groups"] != candidate["totals"]["groups"]:
        raise ValueError("analyzed group count differs")

    bt = baseline["totals"]
    ct = candidate["totals"]
    baseline_weighted_c = bt["source_owned_terms"] / bt["ideal_one_gate_issue_cycles"]
    candidate_weighted_c = ct["source_owned_terms"] / ct["ideal_one_gate_issue_cycles"]
    source_term_delta = metric_delta(ct["source_owned_terms"], bt["source_owned_terms"])
    weighted_c_delta = metric_delta(candidate_weighted_c, baseline_weighted_c)
    tail_delta = ct["active_source_gate_cardinality_gt2"] - bt[
        "active_source_gate_cardinality_gt2"
    ]
    if (
        source_term_delta is not None
        and source_term_delta <= -0.05
        and weighted_c_delta is not None
        and weighted_c_delta <= -0.03
        and tail_delta <= 0
    ):
        status = "GO_TO_BOUNDED_QAT_EXTENSION"
    elif (
        source_term_delta is not None
        and source_term_delta < 0.0
        and weighted_c_delta is not None
        and weighted_c_delta < 0.0
    ):
        status = "HOLD_SMALL_POSITIVE_NOT_RTL_READY"
    else:
        status = "NO_GO_GATE_CARDINALITY_NOT_IMPROVED"
    return {
        "schema": "local5_gatecard_fixed_trace_compare_v1",
        "status": status,
        "evidence": "[prof] one fixed full-resolution sample, 12 blocks, 48 ordered groups",
        "claim_boundary": (
            "Not valid825, converged QAT, RTL, cycle, energy, encoder, or PPA evidence."
        ),
        "baseline": {
            "directory": str(baseline_dir.resolve()),
            "manifest_sha256": sha256(baseline_dir / "ordered_term_manifest.json"),
            "checkpoint_sha256": baseline_manifest.get("checkpoint_sha256"),
            "totals": bt,
            "weighted_gate_cardinality": baseline_weighted_c,
        },
        "candidate": {
            "directory": str(candidate_dir.resolve()),
            "manifest_sha256": sha256(candidate_dir / "ordered_term_manifest.json"),
            "checkpoint_sha256": candidate_manifest.get("checkpoint_sha256"),
            "totals": ct,
            "weighted_gate_cardinality": candidate_weighted_c,
        },
        "comparison": {
            "cohort_sha256": baseline_manifest.get("cohort_sha256"),
            "group_identities_equal": True,
            "groups": len(baseline_groups),
            "source_owned_term_delta": source_term_delta,
            "active_source_descriptor_delta": metric_delta(
                ct["active_source_descriptors"], bt["active_source_descriptors"]
            ),
            "destination_update_delta": metric_delta(
                ct["destination_updates"], bt["destination_updates"]
            ),
            "weighted_gate_cardinality_delta": weighted_c_delta,
            "gate_cardinality_gt2_absolute_delta": tail_delta,
            "c_le2_ratio_delta": (
                ct["active_source_gate_cardinality_le2_ratio"]
                - bt["active_source_gate_cardinality_le2_ratio"]
            ),
        },
        "promotion_gate": {
            "source_owned_term_reduction": ">=5% on this pilot",
            "weighted_gate_cardinality_reduction": ">=3% on this pilot",
            "c_gt2_tail": "must not increase",
            "before_rtl": (
                "repeat on >=10 samples, valid825 AEE <=0.5% relative degradation, "
                "then compare generic two-wide issue under legal 1RW"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.baseline_dir, args.candidate_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    comparison = report["comparison"]
    lines = [
        "# Local5 fixed-trace gate-cardinality comparison",
        "",
        f"Status: `{report['status']}`.",
        "",
        "| Metric | Delta |",
        "|---|---:|",
        f"| source-owned terms | {100*comparison['source_owned_term_delta']:.3f}% |",
        f"| active source descriptors | {100*comparison['active_source_descriptor_delta']:.3f}% |",
        f"| destination updates | {100*comparison['destination_update_delta']:.3f}% |",
        f"| weighted gate cardinality | {100*comparison['weighted_gate_cardinality_delta']:.3f}% |",
        f"| C>2 absolute count | {comparison['gate_cardinality_gt2_absolute_delta']:+d} |",
        f"| C<=2 ratio | {100*comparison['c_le2_ratio_delta']:.3f} pp |",
        "",
        report["claim_boundary"],
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
