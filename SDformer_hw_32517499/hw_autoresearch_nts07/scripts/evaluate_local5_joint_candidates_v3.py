#!/usr/bin/env python3
"""用 RTL 校准串行边界评估 Local5 同窗全 head 候选。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_local5_joint_candidates_v2 as common
import local5_joint_candidate_reference_v3 as reference


SAMPLING_ID = common.SAMPLING_ID
STAGE_DEPTHS = common.STAGE_DEPTHS
STAGE_HEADS = common.STAGE_HEADS
STAGE_WINDOWS = common.STAGE_WINDOWS
BOOTSTRAP_TRIALS = common.BOOTSTRAP_TRIALS
BOOTSTRAP_SEED = common.BOOTSTRAP_SEED
PROMOTION_SPEEDUP = common.PROMOTION_SPEEDUP
FAMILYWISE_ALPHA = common.FAMILYWISE_ALPHA
CANDIDATE_COMPARISONS = common.CANDIDATE_COMPARISONS
BONFERRONI_ALPHA = common.BONFERRONI_ALPHA
REQUIRED_PAYLOAD_ARRAYS = common.REQUIRED_PAYLOAD_ARRAYS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


weighted_percentile = common.weighted_percentile
bootstrap_ratio = common.bootstrap_ratio
load_plan = common.load_plan
validate_profile = common.validate_profile


def source_paths(root: Path) -> dict[str, Path]:
    return {
        "reference_model": Path(reference.__file__).resolve(),
        "primitive_v2_model": root / "scripts/local5_joint_candidate_reference_v2.py",
        "evaluator": Path(__file__).resolve(),
        "prereg_generator": root / "scripts/freeze_local5_joint_candidate_prereg_v3.py",
        "reference_test": root / "tests/test_local5_joint_candidate_reference_v3.py",
        "primitive_v2_test": root / "tests/test_local5_joint_candidate_reference_v2.py",
        "evaluator_test": root / "tests/test_evaluate_local5_joint_candidates_v3.py",
        "calibration_script": root / "scripts/calibrate_local5_ordered_frontend_rtl.py",
        "calibration_report": root / "results/local5_ordered_frontend_rtl_calibration_20260810/report.json",
        "calibration_direct_20260804": root / "results/local5_qgasr2c_fivebank_postg0_rtl_20260804/direct_profile100.log",
        "calibration_gasr_20260804": root / "results/local5_qgasr2c_fivebank_postg0_rtl_20260804/qgasr_profile100.log",
        "heldout_direct_20260805": root / "results/local5_bb1e4_qgasr2c_fivebank_postg0_rtl_20260805/direct_profile100.log",
        "heldout_gasr_20260805": root / "results/local5_bb1e4_qgasr2c_fivebank_postg0_rtl_20260805/qgasr_profile100.log",
        "rtl_timing_tb": root / "tb_qfit/tb_qfit_direct_1rw_reference_timing.sv",
        "regression_runner": root / "sim_qfit/run_local5_joint_candidate_reference_v3_checks.sh",
        "source_builder": root / "rtl_qfit/qfit_source_multicast_term_builder.sv",
        "source_builder_fifo2": root / "rtl_qfit/qfit_source_multicast_term_builder_fifo2.sv",
        "direct_acc_bank": root / "rtl_qfit/qfit_direct_1rw_acc_bank.sv",
        "tcfm5_top": root / "rtl_qfit/qfit_tcfm5_projection_top.sv",
        "gasr2c_acc_bank": root / "rtl_qfit/qfit_gasr2c_acc_bank.sv",
        "relation_vault": root / "rtl_qfit/qfit_exposure_relation_vault.sv",
        "relation_controller": root / "rtl_qfit/qfit_relation_memo_tile_controller.sv",
        "fcsr_relation_top": root / "rtl_qfit/qfit_fcsr_relation_memo_projection_top.sv",
        "fcsr_relation_tb": root / "tb_qfit/tb_qfit_fcsr_relation_memo_projection_top.sv",
    }


def load_prereg(path: Path, receipt_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {
        "schema": "local5_joint_candidate_prereg_v3",
        "status": "FROZEN_BEFORE_PROFILE",
        "bootstrap_trials": BOOTSTRAP_TRIALS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "promotion_speedup_lower_bound": PROMOTION_SPEEDUP,
        "stage_heads": list(STAGE_HEADS),
        "stage_output_tiles": list(STAGE_HEADS),
        "stage_windows": list(STAGE_WINDOWS),
        "candidates": reference.CANDIDATES,
        "fixed_cycle_scenarios": reference.FIXED_SCENARIOS,
        "candidate_comparisons": CANDIDATE_COMPARISONS,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "bonferroni_alpha_per_candidate": BONFERRONI_ALPHA,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"预注册合同字段失效: {key}")
    root = Path(__file__).resolve().parents[1]
    bindings = value.get("source_bindings") or {}
    current = source_paths(root)
    if set(bindings) != set(current):
        raise ValueError("预注册源码绑定集合失效")
    for name, source in current.items():
        binding = bindings.get(name) or {}
        bound_path = Path(str(binding.get("path", "")))
        if not bound_path.is_absolute():
            bound_path = root / bound_path
        if bound_path.resolve() != source.resolve() or binding.get("sha256") != sha256(source):
            raise ValueError(f"预注册源码SHA失效: {name}")
    if (
        receipt.get("schema") != "local5_joint_candidate_prereg_receipt_v3"
        or receipt.get("status") != "GIT_BLOB_ANCHORED_BEFORE_PROFILE"
        or receipt.get("prereg_sha256") != sha256(path)
    ):
        raise ValueError("预注册外部收据字段失效")
    oid = str(receipt.get("git_blob_oid", ""))
    git_blob = subprocess.run(
        ["git", "cat-file", "blob", oid],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    if git_blob != path.read_bytes():
        raise ValueError("Git blob与预注册字节不一致")
    return value, receipt


def build_comparison(
    baseline_frame: np.ndarray,
    candidate_frame: np.ndarray,
    baseline_windows: np.ndarray,
    candidate_windows: np.ndarray,
    weights: np.ndarray,
    stages: np.ndarray,
    sequence_keys: list[str],
) -> dict[str, Any]:
    sample_bootstrap = bootstrap_ratio(
        baseline_frame,
        candidate_frame,
        trials=BOOTSTRAP_TRIALS,
        seed=BOOTSTRAP_SEED,
        clusters=None,
    )
    sequence_bootstrap = bootstrap_ratio(
        baseline_frame,
        candidate_frame,
        trials=BOOTSTRAP_TRIALS,
        seed=BOOTSTRAP_SEED,
        clusters=sequence_keys,
    )
    overall_base = weighted_percentile(baseline_windows, weights, 95)
    overall_candidate = weighted_percentile(candidate_windows, weights, 95)
    p95_ok = overall_candidate <= overall_base
    per_stage = []
    for stage in range(4):
        mask = stages == stage
        base_p95 = weighted_percentile(baseline_windows[mask], weights[mask], 95)
        candidate_p95 = weighted_percentile(
            candidate_windows[mask], weights[mask], 95
        )
        stage_ok = candidate_p95 <= base_p95
        p95_ok &= stage_ok
        per_stage.append(
            {
                "stage": stage,
                "baseline_p95": base_p95,
                "candidate_p95": candidate_p95,
                "p95_non_regression": stage_ok,
            }
        )
    lower_bound = min(
        float(sample_bootstrap["one_sided_familywise_lower"]),
        float(sequence_bootstrap["one_sided_familywise_lower"]),
    )
    return {
        "sample_bootstrap": sample_bootstrap,
        "sequence_cluster_bootstrap": sequence_bootstrap,
        "decision_lower_bound": lower_bound,
        "overall_window_p95": {
            "baseline": overall_base,
            "candidate": overall_candidate,
            "non_regression": overall_candidate <= overall_base,
        },
        "per_stage_window_p95": per_stage,
        "scenario_gate": (
            "PASS_SCENARIO"
            if lower_bound >= PROMOTION_SPEEDUP and p95_ok
            else "FAIL_SCENARIO"
        ),
    }


def aggregate_candidate_gate(scenarios: dict[str, dict[str, Any]]) -> str:
    if set(scenarios) != set(reference.FIXED_SCENARIOS):
        raise ValueError("候选缺少预提交 fixed scenario")
    return (
        "PROMOTE_TO_MINIMAL_RTL"
        if all(row["scenario_gate"] == "PASS_SCENARIO" for row in scenarios.values())
        else "REJECT_MODEL_PROMOTION"
    )


def evaluate(
    manifest_path: Path, plan_path: Path, prereg_path: Path, receipt_path: Path
) -> dict[str, Any]:
    prereg, receipt = load_prereg(prereg_path, receipt_path)
    manifest, plan, plan_value, payload, paths, sequence_keys = validate_profile(
        manifest_path, plan_path
    )
    if (
        plan_value.get("candidate_prereg_receipt_sha256") != sha256(receipt_path)
        or plan_value.get("candidate_prereg_git_blob_oid") != receipt.get("git_blob_oid")
    ):
        raise ValueError("selection plan未绑定v3预注册收据")

    groups = manifest.get("groups") or []
    offsets = np.asarray(payload["descriptor_group_offsets"], dtype=np.int64)
    descriptor_count = len(groups) * reference.TOKENS
    if (
        len(groups) != 13800
        or offsets.shape != (len(groups) + 1,)
        or offsets[0] != 0
        or offsets[-1] != descriptor_count
        or np.any(np.diff(offsets) != reference.TOKENS)
    ):
        raise ValueError("group/descriptor offset数量失效")

    rows_by_window: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(groups):
        key3 = (int(row["sample"]), int(row["stage"]), int(row["block"]))
        planned = plan.get(key3)
        if planned is None or int(row["window"]) != int(planned["window"]):
            raise ValueError(f"group未绑定selection plan: {index}")
        rows_by_window[(*key3, int(row["window"]))].append(index)
    if len(rows_by_window) != 1200:
        raise ValueError("joint window数量不是1200")

    scenario_names = tuple(reference.FIXED_SCENARIOS)
    candidate_names = tuple(reference.CANDIDATES)
    frame_cycles = {
        scenario: {
            name: np.zeros(100, dtype=np.float64) for name in candidate_names
        }
        for scenario in scenario_names
    }
    window_cycles = {
        scenario: {name: [] for name in candidate_names}
        for scenario in scenario_names
    }
    window_weights: list[float] = []
    window_stages: list[int] = []
    arrays = {
        "source_ids": payload["descriptor_source_id"],
        "planes": payload["descriptor_source_plane"],
        "ys": payload["descriptor_source_y"],
        "xs": payload["descriptor_source_x"],
        "k_bitmaps": payload["descriptor_k_bitmap"],
        "gates": payload["descriptor_incoming_gates"],
        "valid_masks": payload["descriptor_valid_mask"],
    }
    source_term_count = np.asarray(payload["source_term_count"], dtype=np.int64)
    expected_ids = np.tile(np.arange(reference.TOKENS, dtype=np.int64), len(groups))
    expected_spatial = expected_ids % (reference.HEIGHT * reference.WIDTH)
    if (
        np.asarray(arrays["source_ids"]).shape != (descriptor_count,)
        or np.asarray(arrays["planes"]).shape != (descriptor_count,)
        or np.asarray(arrays["ys"]).shape != (descriptor_count,)
        or np.asarray(arrays["xs"]).shape != (descriptor_count,)
        or np.asarray(arrays["k_bitmaps"]).shape != (descriptor_count,)
        or np.asarray(arrays["gates"]).shape != (descriptor_count, reference.ROLES)
        or np.asarray(arrays["valid_masks"]).shape != (descriptor_count,)
        or source_term_count.shape != (descriptor_count,)
        or not np.array_equal(np.asarray(arrays["source_ids"]), expected_ids)
        or not np.array_equal(
            np.asarray(arrays["planes"]),
            expected_ids // (reference.HEIGHT * reference.WIDTH),
        )
        or not np.array_equal(
            np.asarray(arrays["ys"]), expected_spatial // reference.WIDTH
        )
        or not np.array_equal(
            np.asarray(arrays["xs"]), expected_spatial % reference.WIDTH
        )
    ):
        raise ValueError("payload source-id/坐标/完整shape合同失效")

    for key in sorted(rows_by_window):
        sample, stage, block, _ = key
        indices = rows_by_window[key]
        expected_heads = STAGE_HEADS[stage]
        if (
            len(indices) != expected_heads
            or [int(groups[index]["head"]) for index in indices]
            != list(range(expected_heads))
        ):
            raise ValueError(f"同窗head顺序/覆盖失效: {key}")
        traces: list[reference.HeadTrace] = []
        for index in indices:
            begin = int(offsets[index])
            end = int(offsets[index + 1])
            trace = reference.build_head_trace(
                *(np.asarray(values[begin:end]) for values in arrays.values())
            )
            if len(trace.terms) != int(source_term_count[begin:end].sum()):
                raise ValueError(f"term重建与producer计数不一致: group={index}")
            traces.append(trace)
        cycle_scenarios = reference.candidate_window_cycle_scenarios(
            tuple(traces), output_tiles=expected_heads
        )
        weight = float(plan[(sample, stage, block)]["analysis_weight"])
        window_weights.append(weight)
        window_stages.append(stage)
        for scenario in scenario_names:
            for name in candidate_names:
                value = int(cycle_scenarios[scenario][name])
                window_cycles[scenario][name].append(value)
                frame_cycles[scenario][name][sample] += value * weight

    weights = np.asarray(window_weights, dtype=np.float64)
    stages = np.asarray(window_stages, dtype=np.int64)
    scenario_comparisons: dict[str, dict[str, Any]] = {}
    for scenario in scenario_names:
        baseline = frame_cycles[scenario]["c0_direct_recompute"]
        base_windows = np.asarray(
            window_cycles[scenario]["c0_direct_recompute"], dtype=np.float64
        )
        scenario_comparisons[scenario] = {}
        for name in candidate_names[1:]:
            scenario_comparisons[scenario][name] = build_comparison(
                baseline,
                frame_cycles[scenario][name],
                base_windows,
                np.asarray(window_cycles[scenario][name], dtype=np.float64),
                weights,
                stages,
                sequence_keys,
            )

    aggregate = {}
    for name in candidate_names[1:]:
        per_scenario = {
            scenario: scenario_comparisons[scenario][name]
            for scenario in scenario_names
        }
        aggregate[name] = {
            "worst_decision_lower_bound": min(
                float(row["decision_lower_bound"]) for row in per_scenario.values()
            ),
            "all_scenarios_p95_non_regression": all(
                row["overall_window_p95"]["non_regression"]
                and all(
                    stage["p95_non_regression"]
                    for stage in row["per_stage_window_p95"]
                )
                for row in per_scenario.values()
            ),
            "model_gate": aggregate_candidate_gate(per_scenario),
        }

    return {
        "schema": "local5_joint_candidate_evaluation_v3",
        "status": "MODEL_DECISION_COMPLETE_NOT_RTL",
        "evidence": "[prof]+[rtl校准模型]",
        "input": {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256(manifest_path),
            "payload": str(paths["payload_path"].resolve()),
            "payload_sha256": sha256(paths["payload_path"]),
            "selection_plan": str(plan_path.resolve()),
            "selection_plan_sha256": sha256(plan_path),
            "prereg": str(prereg_path.resolve()),
            "prereg_sha256": sha256(prereg_path),
            "prereg_receipt": str(receipt_path.resolve()),
            "prereg_receipt_sha256": sha256(receipt_path),
            "prereg_git_blob_oid": receipt.get("git_blob_oid"),
            "identity_sha256": sha256(paths["identity_path"]),
            "cohort_file_sha256": sha256(paths["cohort_path"]),
            "gpu_exclusivity_audit_sha256": sha256(paths["gpu_path"]),
            "checkpoint_sha256": manifest.get("checkpoint_sha256"),
            "config_sha256": manifest.get("config_sha256"),
            "samples": 100,
            "joint_windows": 1200,
            "head_groups": 13800,
        },
        "decision_contract": prereg,
        "frame_cycle_statistics": {
            scenario: {
                name: {
                    "mean": float(values.mean()),
                    "sample_observed_p50": float(np.percentile(values, 50)),
                    "sample_observed_p95": float(np.percentile(values, 95)),
                    "sample_observed_p99": float(np.percentile(values, 99)),
                    "sample_observed_max": float(values.max()),
                }
                for name, values in candidates.items()
            }
            for scenario, candidates in frame_cycles.items()
        },
        "scenario_comparisons_vs_c0": scenario_comparisons,
        "aggregate_decision_vs_c0": aggregate,
        "source_bindings": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in source_paths(Path(__file__).resolve().parents[1]).items()
        },
        "limits": [
            "recompute主裁决是旧五bank RTL held-out校准过的串行相序，不是新joint-head顶层RTL实测。",
            "fixed=459与475同时评估；两场景求交不会把理想重叠作为晋级证据。",
            "replay保守串行计memo read、builder capture、term和stall；流式重叠不进入晋级。",
            "GASR2C-P是跨head preserve候选模型，尚无对应RTL，不得写成已实现能力。",
            "共同readout与scalar serializer尚未用联合timing miter校准。",
            "过模型门槛只允许实现最小RTL；DATE贡献仍需Acc32 bit-exact、随机反压和物理证据。",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Local5 RTL 校准候选决策",
        "",
        "## 证据边界",
        "",
        "本报告为 **[prof]+[rtl校准模型]**，不是新 joint-head RTL 周期或 ASIC PPA。",
        "晋级必须同时通过 `fixed=459` 与 `fixed=475` 两个预提交场景。",
        "",
        "## 场景结果",
        "",
        "| 固定项场景 | 候选 | ratio-of-means | sample LB | sequence LB | 总体p95 | 各stage p95 | 场景裁决 |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for scenario, candidates in report["scenario_comparisons_vs_c0"].items():
        for name, row in candidates.items():
            sample = row["sample_bootstrap"]
            sequence = row["sequence_cluster_bootstrap"]
            overall = row["overall_window_p95"]["non_regression"]
            stages = all(item["p95_non_regression"] for item in row["per_stage_window_p95"])
            lines.append(
                f"| {scenario} | {name} | {sample['ratio_of_means']:.4f}x | "
                f"{sample['one_sided_familywise_lower']:.4f}x | "
                f"{sequence['one_sided_familywise_lower']:.4f}x | "
                f"{'PASS' if overall else 'FAIL'} | {'PASS' if stages else 'FAIL'} | "
                f"{row['scenario_gate']} |"
            )
    lines += [
        "",
        "## 最终裁决",
        "",
        "| 候选 | 最差下界 | 双场景p95 | 裁决 |",
        "|---|---:|---|---|",
    ]
    for name, row in report["aggregate_decision_vs_c0"].items():
        lines.append(
            f"| {name} | {row['worst_decision_lower_bound']:.4f}x | "
            f"{'PASS' if row['all_scenarios_p95_non_regression'] else 'FAIL'} | "
            f"{row['model_gate']} |"
        )
    lines += ["", "## 限制", ""]
    lines.extend(f"- {item}" for item in report["limits"])
    return "\n".join(lines) + "\n"


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection-plan", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--prereg-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(
        args.manifest, args.selection_plan, args.prereg, args.prereg_receipt
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(
        args.output_dir / "report.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write(args.output_dir / "report.md", render_markdown(report))
    commit = {
        "schema": "local5_joint_candidate_evaluation_commit_v3",
        "status": "COMMITTED",
        "report_json_sha256": sha256(args.output_dir / "report.json"),
        "report_md_sha256": sha256(args.output_dir / "report.md"),
        "prereg_sha256": sha256(args.prereg),
        "prereg_receipt_sha256": sha256(args.prereg_receipt),
        "prereg_git_blob_oid": report["input"]["prereg_git_blob_oid"],
    }
    atomic_write(
        args.output_dir / "commit.json",
        json.dumps(commit, ensure_ascii=False, indent=2) + "\n",
    )
    print("PASS Local5 joint candidate v3 decision")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
