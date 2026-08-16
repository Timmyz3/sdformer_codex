#!/usr/bin/env python3
"""用预注册公平 ordered-frontend/1RW v2 模型评估 Local5 候选。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import local5_joint_candidate_reference_v2 as reference


SAMPLING_ID = "uniform_plan_window_all_heads_v1"
STAGE_DEPTHS = (2, 2, 6, 2)
STAGE_HEADS = (3, 6, 12, 24)
STAGE_WINDOWS = (440, 120, 30, 10)
BOOTSTRAP_TRIALS = 20_000
BOOTSTRAP_SEED = 20260810
PROMOTION_SPEEDUP = 1.20
FAMILYWISE_ALPHA = 0.05
CANDIDATE_COMPARISONS = 3
BONFERRONI_ALPHA = FAMILYWISE_ALPHA / CANDIDATE_COMPARISONS
REQUIRED_PAYLOAD_ARRAYS = {
    "descriptor_group_offsets",
    "descriptor_source_id",
    "descriptor_source_plane",
    "descriptor_source_y",
    "descriptor_source_x",
    "descriptor_k_bitmap",
    "descriptor_incoming_gates",
    "descriptor_valid_mask",
    "source_term_count",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.shape != weights.shape or values.size == 0:
        raise ValueError("加权分位数输入非法")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    if cumulative[-1] <= 0:
        raise ValueError("加权分位数权重非法")
    target = min(max(q, 0.0), 100.0) / 100.0 * cumulative[-1]
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(values[order[min(index, len(order) - 1)]])


def bootstrap_ratio(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    trials: int,
    seed: int,
    clusters: list[str] | None,
    one_sided_alpha: float = BONFERRONI_ALPHA,
) -> dict[str, float | int | str]:
    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if baseline.shape != candidate.shape or baseline.shape != (100,):
        raise ValueError("bootstrap必须输入100个配对sample")
    rng = np.random.default_rng(seed)
    ratios = np.zeros(trials, dtype=np.float64)
    if clusters is None:
        for trial in range(trials):
            selected = rng.integers(0, len(baseline), size=len(baseline))
            ratios[trial] = baseline[selected].sum() / candidate[selected].sum()
        unit = "sample"
        cluster_count = len(baseline)
    else:
        if len(clusters) != len(baseline):
            raise ValueError("sequence cluster数与sample不一致")
        unique = sorted(set(clusters))
        members = {
            key: np.asarray([index for index, value in enumerate(clusters) if value == key])
            for key in unique
        }
        for trial in range(trials):
            selected_clusters = rng.integers(0, len(unique), size=len(unique))
            base_sum = 0.0
            candidate_sum = 0.0
            for selected in selected_clusters:
                indices = members[unique[int(selected)]]
                base_sum += float(baseline[indices].sum())
                candidate_sum += float(candidate[indices].sum())
            ratios[trial] = base_sum / candidate_sum
        unit = "sequence"
        cluster_count = len(unique)
    if not 0.0 < one_sided_alpha < 0.5:
        raise ValueError("单侧alpha非法")
    return {
        "unit": unit,
        "clusters": cluster_count,
        "trials": trials,
        "seed": seed,
        "ratio_of_means": float(baseline.mean() / candidate.mean()),
        "one_sided_alpha": one_sided_alpha,
        "one_sided_familywise_lower": float(
            np.percentile(ratios, 100.0 * one_sided_alpha)
        ),
        "two_sided_95_lower": float(np.percentile(ratios, 2.5)),
        "two_sided_95_upper": float(np.percentile(ratios, 97.5)),
    }


def load_prereg(path: Path, receipt_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {
        "schema": "local5_joint_candidate_prereg_v2",
        "status": "FROZEN_BEFORE_PROFILE",
        "bootstrap_trials": BOOTSTRAP_TRIALS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "promotion_speedup_lower_bound": PROMOTION_SPEEDUP,
        "stage_heads": list(STAGE_HEADS),
        "stage_output_tiles": list(STAGE_HEADS),
        "stage_windows": list(STAGE_WINDOWS),
        "candidates": reference.CANDIDATES,
        "candidate_comparisons": CANDIDATE_COMPARISONS,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "bonferroni_alpha_per_candidate": BONFERRONI_ALPHA,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"预注册合同字段失效: {key}")
    bindings = value.get("source_bindings") or {}
    root = Path(__file__).resolve().parents[1]
    current = {
        "reference_model": Path(reference.__file__).resolve(),
        "evaluator": Path(__file__).resolve(),
        "reference_test": root / "tests/test_local5_joint_candidate_reference_v2.py",
        "evaluator_test": root / "tests/test_evaluate_local5_joint_candidates_v2.py",
        "rtl_timing_tb": root / "tb_qfit/tb_qfit_direct_1rw_reference_timing.sv",
        "regression_runner": root / "sim_qfit/run_local5_joint_candidate_reference_v2_checks.sh",
        "source_builder": root / "rtl_qfit/qfit_source_multicast_term_builder.sv",
        "source_builder_fifo2": root / "rtl_qfit/qfit_source_multicast_term_builder_fifo2.sv",
        "direct_acc_bank": root / "rtl_qfit/qfit_direct_1rw_acc_bank.sv",
        "tcfm5_top": root / "rtl_qfit/qfit_tcfm5_projection_top.sv",
        "gasr2c_acc_bank": root / "rtl_qfit/qfit_gasr2c_acc_bank.sv",
        "relation_vault": root / "rtl_qfit/qfit_exposure_relation_vault.sv",
        "relation_controller": root / "rtl_qfit/qfit_relation_memo_tile_controller.sv",
    }
    if set(bindings) != set(current):
        raise ValueError("预注册源码绑定集合失效")
    for name, source in current.items():
        binding = bindings.get(name) or {}
        if (
            Path(str(binding.get("path", ""))).resolve() != source
            or binding.get("sha256") != sha256(source)
        ):
            raise ValueError(f"预注册源码SHA失效: {name}")
    if (
        receipt.get("schema") != "local5_joint_candidate_prereg_receipt_v2"
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


def load_plan(path: Path) -> tuple[dict[tuple[int, int, int], dict], dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != "local5_uniform_joint_window_plan_v1"
        or value.get("sampling_id") != SAMPLING_ID
        or value.get("seed") != 20260809
    ):
        raise ValueError("selection plan格式/sampling/seed失效")
    rows: dict[tuple[int, int, int], dict] = {}
    for row in value.get("records") or []:
        key = (int(row["sample"]), int(row["stage"]), int(row["block"]))
        sample, stage, block = key
        if (
            key in rows
            or not 0 <= sample < 100
            or not 0 <= stage < 4
            or not 0 <= block < STAGE_DEPTHS[stage]
            or int(row["heads"]) != STAGE_HEADS[stage]
            or int(row["batch_windows"]) != STAGE_WINDOWS[stage]
            or not 0 <= int(row["window"]) < STAGE_WINDOWS[stage]
            or float(row["inclusion_probability"]) != 1.0 / STAGE_WINDOWS[stage]
            or float(row["analysis_weight"]) != float(STAGE_WINDOWS[stage])
        ):
            raise ValueError(f"selection plan记录失效: {key}")
        rows[key] = row
    expected = {
        (sample, stage, block)
        for sample in range(100)
        for stage, depth in enumerate(STAGE_DEPTHS)
        for block in range(depth)
    }
    if set(rows) != expected:
        raise ValueError("selection plan未覆盖100x12完整key set")
    return rows, value


def validate_profile(
    manifest_path: Path, plan_path: Path
) -> tuple[dict, dict, dict, np.lib.npyio.NpzFile, dict, list[str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qualification = manifest.get("qualification") or {}
    if (
        manifest.get("schema") != "et3_ordered_term_trace_v2"
        or manifest.get("sampling", {}).get("method") != SAMPLING_ID
        or qualification.get("qualified") is not True
        or qualification.get("processed_samples") != 100
        or qualification.get("attached_blocks") != 12
        or qualification.get("captured_groups") != 13800
        or manifest.get("sampling", {}).get("selection_plan_sha256") != sha256(plan_path)
    ):
        raise ValueError("正式joint-head manifest合同失效")
    plan, plan_value = load_plan(plan_path)
    if manifest.get("cohort_sha256") != plan_value.get("cohort_sha256"):
        raise ValueError("manifest/plan cohort不一致")
    payload_path = manifest_path.parent / str(manifest.get("payload_file", ""))
    if not payload_path.is_file() or manifest.get("payload_sha256") != sha256(payload_path):
        raise ValueError("payload缺失或SHA失效")
    payload = np.load(payload_path, allow_pickle=False, mmap_mode="r")
    if not REQUIRED_PAYLOAD_ARRAYS.issubset(payload.files):
        raise ValueError("payload缺少候选评估必需数组")

    identity_path = Path(str(manifest.get("run_identity_file", ""))).resolve()
    cohort_path = manifest_path.parent / str(manifest.get("cohort_file", ""))
    gpu_path = manifest_path.parent / "gpu_exclusivity_audit.json"
    if (
        not identity_path.is_file()
        or manifest.get("run_identity_file_sha256") != sha256(identity_path)
        or not cohort_path.is_file()
        or manifest.get("cohort_file_sha256") != sha256(cohort_path)
        or not gpu_path.is_file()
    ):
        raise ValueError("身份/cohort/GPU审计缺失或SHA失效")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    gpu = json.loads(gpu_path.read_text(encoding="utf-8"))
    if (
        identity.get("selection_plan_sha256") != sha256(plan_path)
        or identity.get("checkpoint_sha256") != manifest.get("checkpoint_sha256")
        or identity.get("config_sha256") != manifest.get("config_sha256")
        or identity.get("cohort_sha256") != manifest.get("cohort_sha256")
        or gpu.get("schema") != "local5_joint_gpu_exclusivity_audit_v1"
        or gpu.get("status") != "PASS"
        or gpu.get("manifest_sha256") != sha256(manifest_path)
        or gpu.get("payload_sha256") != sha256(payload_path)
        or gpu.get("identity_sha256") != sha256(identity_path)
        or gpu.get("foreign_compute_pids") != []
    ):
        raise ValueError("身份或GPU独占审计失效")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    sample_keys = [str(value) for value in cohort.get("sample_keys") or []]
    sequence_keys = [str(value) for value in cohort.get("sequence_keys") or []]
    derived_sequences = [re.sub(r"_\d+\.npy$", "", key) for key in sample_keys]
    sequence_counts = dict(Counter(sequence_keys))
    dataset_indices = cohort.get("dataset_indices") or []
    if (
        cohort.get("schema") != "ordered_trace_cohort_v2"
        or cohort.get("count") != 100
        or len(sample_keys) != 100
        or len(set(sample_keys)) != 100
        or len(sequence_keys) != 100
        or sequence_keys != derived_sequences
        or len(set(sequence_keys)) != 18
        or cohort.get("sequence_counts") != sequence_counts
        or len(dataset_indices) != 100
        or len(set(dataset_indices)) != 100
        or list(dataset_indices) != sorted(dataset_indices)
        or cohort.get("sample_key_sha256") != manifest.get("cohort_sha256")
    ):
        raise ValueError("cohort sequence cluster合同失效")
    return manifest, plan, plan_value, payload, {
        "payload_path": payload_path,
        "identity_path": identity_path,
        "cohort_path": cohort_path,
        "gpu_path": gpu_path,
    }, sequence_keys


def evaluate(
    manifest_path: Path, plan_path: Path, prereg_path: Path, receipt_path: Path
) -> dict[str, Any]:
    prereg, receipt = load_prereg(prereg_path, receipt_path)
    manifest, plan, plan_value, payload, paths, sequence_keys = validate_profile(
        manifest_path, plan_path
    )
    if (
        plan_value.get("candidate_prereg_receipt_sha256") != sha256(receipt_path)
        or plan_value.get("candidate_prereg_git_blob_oid")
        != receipt.get("git_blob_oid")
    ):
        raise ValueError("selection plan未绑定v2预注册收据")
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

    candidate_names = tuple(reference.CANDIDATES)
    frame_cycles = {
        name: np.zeros(100, dtype=np.float64) for name in candidate_names
    }
    window_cycles = {name: [] for name in candidate_names}
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
    expected_source_ids = np.tile(
        np.arange(reference.TOKENS, dtype=np.int64), len(groups)
    )
    expected_spatial = expected_source_ids % (reference.HEIGHT * reference.WIDTH)
    if (
        np.asarray(arrays["source_ids"]).shape != (descriptor_count,)
        or np.asarray(arrays["planes"]).shape != (descriptor_count,)
        or np.asarray(arrays["ys"]).shape != (descriptor_count,)
        or np.asarray(arrays["xs"]).shape != (descriptor_count,)
        or np.asarray(arrays["k_bitmaps"]).shape != (descriptor_count,)
        or np.asarray(arrays["gates"]).shape != (descriptor_count, reference.ROLES)
        or np.asarray(arrays["valid_masks"]).shape != (descriptor_count,)
        or source_term_count.shape != (descriptor_count,)
        or not np.array_equal(np.asarray(arrays["source_ids"]), expected_source_ids)
        or not np.array_equal(
            np.asarray(arrays["planes"]),
            expected_source_ids // (reference.HEIGHT * reference.WIDTH),
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
                *(
                    np.asarray(values[begin:end])
                    for values in arrays.values()
                )
            )
            if len(trace.terms) != int(source_term_count[begin:end].sum()):
                raise ValueError(f"term重建与producer计数不一致: group={index}")
            traces.append(trace)
        cycles = reference.candidate_window_cycles(
            tuple(traces), output_tiles=expected_heads
        )
        weight = float(plan[(sample, stage, block)]["analysis_weight"])
        window_weights.append(weight)
        window_stages.append(stage)
        for name in candidate_names:
            value = int(cycles[name])
            window_cycles[name].append(value)
            frame_cycles[name][sample] += value * weight

    weights = np.asarray(window_weights, dtype=np.float64)
    stages = np.asarray(window_stages, dtype=np.int64)
    baseline = frame_cycles["c0_direct_recompute"]
    comparisons: dict[str, Any] = {}
    for name in candidate_names[1:]:
        sample_bootstrap = bootstrap_ratio(
            baseline,
            frame_cycles[name],
            trials=BOOTSTRAP_TRIALS,
            seed=BOOTSTRAP_SEED,
            clusters=None,
        )
        sequence_bootstrap = bootstrap_ratio(
            baseline,
            frame_cycles[name],
            trials=BOOTSTRAP_TRIALS,
            seed=BOOTSTRAP_SEED,
            clusters=sequence_keys,
        )
        base_window = np.asarray(window_cycles["c0_direct_recompute"], dtype=np.float64)
        candidate_window = np.asarray(window_cycles[name], dtype=np.float64)
        overall_p95_base = weighted_percentile(base_window, weights, 95)
        overall_p95_candidate = weighted_percentile(candidate_window, weights, 95)
        per_stage = []
        p95_ok = overall_p95_candidate <= overall_p95_base
        for stage in range(4):
            mask = stages == stage
            base_p95 = weighted_percentile(base_window[mask], weights[mask], 95)
            candidate_p95 = weighted_percentile(
                candidate_window[mask], weights[mask], 95
            )
            p95_ok &= candidate_p95 <= base_p95
            per_stage.append(
                {
                    "stage": stage,
                    "baseline_p95": base_p95,
                    "candidate_p95": candidate_p95,
                    "p95_non_regression": candidate_p95 <= base_p95,
                }
            )
        lower_bound = min(
            float(sample_bootstrap["one_sided_familywise_lower"]),
            float(sequence_bootstrap["one_sided_familywise_lower"]),
        )
        comparisons[name] = {
            "sample_bootstrap": sample_bootstrap,
            "sequence_cluster_bootstrap": sequence_bootstrap,
            "decision_lower_bound": lower_bound,
            "overall_window_p95": {
                "baseline": overall_p95_base,
                "candidate": overall_p95_candidate,
                "non_regression": overall_p95_candidate <= overall_p95_base,
            },
            "per_stage_window_p95": per_stage,
            "model_gate": (
                "PROMOTE_TO_MINIMAL_RTL"
                if lower_bound >= PROMOTION_SPEEDUP and p95_ok
                else "REJECT_MODEL_PROMOTION"
            ),
        }

    return {
        "schema": "local5_joint_candidate_evaluation_v2",
        "status": "MODEL_DECISION_COMPLETE_NOT_RTL",
        "evidence": "[prof]+[模型]",
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
            name: {
                "mean": float(values.mean()),
                "sample_observed_p50": float(np.percentile(values, 50)),
                "sample_observed_p95": float(np.percentile(values, 95)),
                "sample_observed_p99": float(np.percentile(values, 99)),
                "sample_observed_max": float(values.max()),
            }
            for name, values in frame_cycles.items()
        },
        "comparisons_vs_c0": comparisons,
        "source_bindings": {
            "reference_model": {
                "path": str(Path(reference.__file__).resolve()),
                "sha256": sha256(Path(reference.__file__).resolve()),
            },
            "evaluator": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
        },
        "limits": [
            "周期是预注册的ordered行为模型，不是RTL顶层实测。",
            "四候选共用B2v跨head状态边界和最终scalar serializer，B2v本身不是候选创新。",
            "GASR2C-P是跨head preserve候选模型，尚无对应RTL，不得写成已实现能力。",
            "Direct/GASR2C-P共计active-scan、active descriptor capture与ordered builder开销。",
            "ERM7参数固定为7KiB、112-bit word和critical-only顺序admission。",
            "三个候选使用Bonferroni校正的family-wise 5%单侧下界。",
            "过模型门槛只允许进入最小RTL；正文贡献仍需Acc32 bit-exact、反压和物理证据。",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Local5 同窗全 Head 候选预注册决策",
        "",
        "## 证据边界",
        "",
        "本报告为 **[prof]+[模型]**，不是 RTL 顶层周期或 ASIC PPA。",
        "候选、参数、强基线、bootstrap seed 和模型 SHA 由 selection-plan 绑定的 Git blob 收据冻结。",
        "",
        "## 配对结果",
        "",
        "| 候选 | ratio-of-means | sample 95% LB | sequence 95% LB | 总体p95 | 各stage p95 | 裁决 |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for name, row in report["comparisons_vs_c0"].items():
        sample = row["sample_bootstrap"]
        sequence = row["sequence_cluster_bootstrap"]
        overall = row["overall_window_p95"]["non_regression"]
        stages = all(item["p95_non_regression"] for item in row["per_stage_window_p95"])
        lines.append(
            f"| {name} | {sample['ratio_of_means']:.4f}x | "
            f"{sample['one_sided_familywise_lower']:.4f}x | "
            f"{sequence['one_sided_familywise_lower']:.4f}x | "
            f"{'PASS' if overall else 'FAIL'} | {'PASS' if stages else 'FAIL'} | "
            f"{row['model_gate']} |"
        )
    lines += [
        "",
        "## 门槛",
        "",
        "`PROMOTE_TO_MINIMAL_RTL` 当且仅当 sample 与 sequence-cluster 两种配对",
        "bootstrap 中更保守的 Bonferroni family-wise 95% 加速下界仍 `>=1.20x`，且总体及",
        "每个 stage 的加权 window p95 均不退化。否则停止 Local5 微机制扩展。",
        "",
        "## 候选含义",
        "",
        "- `C0`：合法 1RW TCFM5 + 每 output tile 重算 relation。",
        "- `C1`：GASR2C-P 两槽 source-resident Acc context 候选模型，relation 仍重算。",
        "- `C2`：直接 1RW 后端 + 7 KiB critical-only exact Relation Memo。",
        "- `C3`：C1+C2 双生命期精确状态层次候选模型。",
        "",
        "## 边界",
        "",
    ]
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
        "schema": "local5_joint_candidate_evaluation_commit_v2",
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
    print("PASS Local5 joint candidate preregistered decision")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
