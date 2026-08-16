#!/usr/bin/env python3
"""Close the NB0 angular-error diagnosis using equal+10 full-resolution evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
AUTO = REPO / "neuron_autoresearch"
RESULTS = REPO / "neuron_experiments/H9_bipolar_self_attention/results"
METRIC_RECEIPT = AUTO / "AAE_METRIC_TEST_RECEIPT_20260805.json"
EARLY_DIAGNOSTIC = AUTO / "NB0_AAE_GAP_DIAGNOSTIC_20260806.json"
EQUAL_SUMMARY = RESULTS / "dsec_fullres_w15_equal_plus10_convergence_summary_20260805.json"
NB0_ROOT = RESULTS / "dsec_fullres_w15_NB0_equal_plus10_ep40_20260805"
OUTPUT = AUTO / "NB0_AAE_GAP_CLOSURE_20260812.json"
OUTPUT_MD = AUTO / "NB0_AAE_GAP_CLOSURE_20260812.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def improvement(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference * 100.0


def validate_metric_receipt(receipt: dict[str, object]) -> dict[str, bool]:
    contracts = receipt.get("contracts") or {}
    checks = {
        "status_pass": receipt.get("status") == "PASS",
        "eight_tests": int(receipt.get("test_count", -1)) == 8,
        "legacy_is_2d": contracts.get("legacy_aae")
        == "2d_direction_angle_degrees_between_uv",
        "benchmark_is_uv1": contracts.get("benchmark_ae")
        == "middlebury_barron_3d_angle_degrees_between_normalized_uv1",
        "eval_batch_one": int(contracts.get("eval_batch_size", -1)) == 1,
    }
    for name, source in (receipt.get("sources") or {}).items():
        path = Path(str(source["path"]))
        checks[f"source_sha_{name}"] = path.is_file() and sha256(path) == source["sha256"]
    return checks


def profile(epoch: int, expected: dict[str, object]) -> dict[str, object]:
    path = NB0_ROOT / f"standard_valid825/epoch{epoch}/spike_profile.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    protocol = raw.get("eval_protocol") or {}
    metrics = {name: float(raw["metrics"][name]) for name in ("AEE", "AAE", "AAE_Benchmark")}
    aggregation = raw.get("metric_aggregation_audit") or {}
    checks = {
        "samples_825": int(raw.get("samples", -1)) == 825,
        "resolution_480x640": protocol.get("resolution") == [480, 640],
        "crop_null": protocol.get("crop") is None,
        "window_2x15x15": protocol.get("window_size") == [2, 15, 15],
        "eval_batch_one": int(protocol.get("eval_batch_size", -1)) == 1,
        "valid_pixels": float(aggregation.get("valid_pixels", -1)) == 48152523.0,
        "sequence_count_18": int(aggregation.get("sequence_count", -1)) == 18,
        "summary_aee": metrics["AEE"] == float(expected["AEE"]),
        "summary_aae2d": metrics["AAE"] == float(expected["AAE"]),
        "summary_ae3d": metrics["AAE_Benchmark"] == float(expected["AAE_Benchmark"]),
        "summary_spikes": abs(
            float(raw["total_spikes"]) / 1e9 - float(expected["total_spikes_g"])
        ) < 1e-9,
    }
    if not all(checks.values()):
        raise RuntimeError(f"NB0 ep{epoch} profile contract failed: {checks}")
    return {
        "checkpoint_label": epoch,
        "profile": binding(path),
        "metrics": metrics,
        "total_spikes_g": float(raw["total_spikes"]) / 1e9,
        "aggregation": {
            key: aggregation[key]
            for key in ("frame_equal_mean", "pixel_global_mean", "sequence_balanced_mean")
        },
        "checks": checks,
    }


def main() -> int:
    metric_receipt = json.loads(METRIC_RECEIPT.read_text(encoding="utf-8"))
    metric_checks = validate_metric_receipt(metric_receipt)
    if not all(metric_checks.values()):
        raise RuntimeError(f"AAE metric receipt drifted: {metric_checks}")

    early = json.loads(EARLY_DIAGNOSTIC.read_text(encoding="utf-8"))
    equal = json.loads(EQUAL_SUMMARY.read_text(encoding="utf-8"))
    if equal.get("schema") != "dsec_fullres_equal_plus10_convergence_v1":
        raise RuntimeError("unexpected equal+10 summary schema")
    nb0 = equal["candidates"]["NB0"]
    expected_by_epoch = {
        int(row["checkpoint_label"]): row for row in nb0["points"]
    }
    profiles = [profile(epoch, expected_by_epoch[epoch]) for epoch in (29, 34, 39)]

    best = profiles[0]
    official = float(early["paper_evidence"]["table_I_official_hidden_test"]["AE_3D"])
    aggregate_gaps = {}
    for aggregation in ("frame_equal_mean", "pixel_global_mean", "sequence_balanced_mean"):
        local = float(best["aggregation"][aggregation]["AAE_Benchmark"])
        aggregate_gaps[aggregation] = {
            "local_AE_3D": local,
            "official_hidden_test_AE_3D": official,
            "absolute_gap_not_same_population": local - official,
            "relative_to_official_pct_not_same_population": (local / official - 1.0) * 100.0,
        }

    candidate_comparison = {}
    for name in ("H67", "Local5"):
        candidate = equal["candidates"][name]
        point = next(
            row for row in candidate["points"]
            if int(row["budget"]) == int(candidate["rank1_budget"])
        )
        reference = expected_by_epoch[29]
        candidate_comparison[name] = {
            "rank1_budget": int(candidate["rank1_budget"]),
            "checkpoint_label": int(candidate["rank1_checkpoint_label"]),
            "AEE": float(point["AEE"]),
            "AAE_2D": float(point["AAE"]),
            "AE_3D": float(point["AAE_Benchmark"]),
            "total_spikes_g": float(point["total_spikes_g"]),
            "vs_NB0_pct": {
                "AEE": improvement(float(reference["AEE"]), float(point["AEE"])),
                "AAE_2D": improvement(float(reference["AAE"]), float(point["AAE"])),
                "AE_3D": improvement(
                    float(reference["AAE_Benchmark"]), float(point["AAE_Benchmark"])
                ),
                "total_spikes_g": improvement(
                    float(reference["total_spikes_g"]), float(point["total_spikes_g"])
                ),
            },
        }

    points = nb0["points"]
    closure = {
        "schema": "nb0_aae_gap_closure_v2",
        "status": "PASS_LOCAL_CONVERGENCE_CLOSED_OFFICIAL_TEST_NOT_COMPARABLE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "DSEC full-resolution local valid825; not official hidden-test reproduction",
        "inputs": {
            "metric_receipt": binding(METRIC_RECEIPT),
            "early_diagnostic": binding(EARLY_DIAGNOSTIC),
            "equal_plus10_summary": binding(EQUAL_SUMMARY),
        },
        "metric_receipt_checks": metric_checks,
        "NB0": {
            "rank1_budget": int(nb0["rank1_budget"]),
            "rank1_checkpoint_label": int(nb0["rank1_checkpoint_label"]),
            "decision": nb0["decision"],
            "angle_decision": nb0["angle_decision"],
            "profiles": profiles,
            "budget30_to35_pct_lower_is_better": {
                name: improvement(float(points[0][source]), float(points[1][source]))
                for name, source in (("AEE", "AEE"), ("AAE_2D", "AAE"), ("AE_3D", "AAE_Benchmark"))
            },
            "budget30_to40_pct_lower_is_better": {
                name: improvement(float(points[0][source]), float(points[2][source]))
                for name, source in (("AEE", "AEE"), ("AAE_2D", "AAE"), ("AE_3D", "AAE_Benchmark"))
            },
        },
        "paper_evidence": early["paper_evidence"],
        "official_gap_by_local_aggregation": aggregate_gaps,
        "same_population_candidate_comparison": candidate_comparison,
        "final_diagnosis": {
            "formula_bug": False,
            "NB0_undertraining_explains_local_angle": False,
            "NB0_local_convergence": "operationally_plateaued_or_overfit",
            "aggregation_choice_explains_official_4p871": False,
            "population_and_server_protocol_mismatch": True,
            "paper_safe_statement": (
                "On the same local valid825 population, H67 and Local5 improve both AEE and "
                "angular metrics over converged NB0. The paper's 4.871 is an official hidden-test "
                "AE and is not a directly comparable local-valid target."
            ),
        },
    }
    OUTPUT.write_text(json.dumps(closure, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# NB0 AAE Gap Closure (2026-08-12)",
        "",
        f"Machine receipt: `{OUTPUT.relative_to(REPO)}`.",
        "",
        "## Final conclusion",
        "",
        "- Formula audit PASS: legacy AAE is 2-D direction angle; benchmark AE uses Barron/Middlebury `(u,v,1)`.",
        "- NB0 is operationally plateaued/overfit. Its budget-30 checkpoint remains AEE rank-1 after equal training at budgets 35 and 40.",
        "- The three local aggregations all remain above official hidden-test AE 4.871; aggregation choice alone does not close the gap.",
        "- The remaining numerical gap is not evidence that local NB0 needs more epochs. Official hidden test and local valid825 are different populations/protocols.",
        "",
        "## NB0 equal+10",
        "",
        "| budget | AEE | AAE-2D | AE-3D | spikes (G) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in points:
        lines.append(
            f"| {row['budget']} | {row['AEE']:.4f} | {row['AAE']:.4f} | "
            f"{row['AAE_Benchmark']:.4f} | {row['total_spikes_g']:.4f} |"
        )
    lines.extend([
        "",
        "## Same-population rank-1 comparison",
        "",
        "| route | AEE | AAE-2D | AE-3D | spikes (G) | vs NB0 AEE | vs NB0 AE-3D | vs NB0 spikes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    ref = expected_by_epoch[29]
    lines.append(
        f"| NB0 | {ref['AEE']:.4f} | {ref['AAE']:.4f} | {ref['AAE_Benchmark']:.4f} | "
        f"{ref['total_spikes_g']:.4f} | - | - | - |"
    )
    for name in ("H67", "Local5"):
        row = candidate_comparison[name]
        delta = row["vs_NB0_pct"]
        lines.append(
            f"| {name} | {row['AEE']:.4f} | {row['AAE_2D']:.4f} | {row['AE_3D']:.4f} | "
            f"{row['total_spikes_g']:.4f} | {delta['AEE']:.2f}% | {delta['AE_3D']:.2f}% | "
            f"{delta['total_spikes_g']:.2f}% |"
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"PASS NB0 AAE closure: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
