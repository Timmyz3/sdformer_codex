#!/usr/bin/env python3
"""用预注册有序1RW模型评估Local5同窗全head候选。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import local5_joint_candidate_reference as reference


SAMPLING_ID = "uniform_plan_window_all_heads_v1"
STAGE_DEPTHS = (2, 2, 6, 2)
STAGE_HEADS = (3, 6, 12, 24)
STAGE_WINDOWS = (440, 120, 30, 10)
BOOTSTRAP_TRIALS = 20_000
BOOTSTRAP_SEED = 20260810
PROMOTION_SPEEDUP = 1.20


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
    return {
        "unit": unit,
        "clusters": cluster_count,
        "trials": trials,
        "seed": seed,
        "ratio_of_means": float(baseline.mean() / candidate.mean()),
        "one_sided_95_lower": float(np.percentile(ratios, 5.0)),
        "two_sided_95_lower": float(np.percentile(ratios, 2.5)),
        "two_sided_95_upper": float(np.percentile(ratios, 97.5)),
    }


def load_prereg(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": "local5_joint_candidate_prereg_v1",
        "status": "FROZEN_BEFORE_PROFILE",
        "bootstrap_trials": BOOTSTRAP_TRIALS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "promotion_speedup_lower_bound": PROMOTION_SPEEDUP,
        "stage_heads": list(STAGE_HEADS),
        "stage_output_tiles": list(STAGE_HEADS),
        "stage_windows": list(STAGE_WINDOWS),
        "candidates": reference.CANDIDATES,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"预注册合同字段失效: {key}")
    bindings = value.get("source_bindings") or {}
    current = {
        "reference_model": Path(reference.__file__).resolve(),
        "evaluator": Path(__file__).resolve(),
    }
    for name, source in current.items():
        binding = bindings.get(name) or {}
        if (
            Path(str(binding.get("path", ""))).resolve() != source
            or binding.get("sha256") != sha256(source)
        ):
            raise ValueError(f"预注册源码SHA失效: {name}")
    return value


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
) -> tuple[dict, dict, np.lib.npyio.NpzFile, dict, list[str]]:
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
    payload = np.load(payload_path, allow_pickle=False)

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
    sequence_keys = [str(value) for value in cohort.get("sequence_keys") or []]
    if len(sequence_keys) != 100 or len(set(sequence_keys)) != 18:
        raise ValueError("cohort sequence cluster合同失效")
    return manifest, plan, payload, {
        "payload_path": payload_path,
        "identity_path": identity_path,
        "cohort_path": cohort_path,
        "gpu_path": gpu_path,
    }, sequence_keys


def evaluate(
    manifest_path: Path, plan_path: Path, prereg_path: Path
) -> dict[str, Any]:
    prereg = load_prereg(prereg_path)
    manifest, plan, payload, paths, sequence_keys = validate_profile(
        manifest_path, plan_path
    )
    groups = manifest.get("groups") or []
    offsets = np.asarray(payload["descriptor_group_offsets"], dtype=np.int64)
    if len(groups) != 13800 or len(offsets) != len(groups) + 1:
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
            float(sample_bootstrap["one_sided_95_lower"]),
            float(sequence_bootstrap["one_sided_95_lower"]),
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
        "schema": "local5_joint_candidate_evaluation_v1",
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
            "SRAC2参数固定为每bank两context和descriptor latency=3，不在profile后扫参。",
            "ERM7参数固定为7KiB、112-bit word和critical-only顺序admission。",
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
        "候选、参数、强基线、bootstrap seed 和模型 SHA 均在 profile 揭晓前冻结。",
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
            f"{sample['one_sided_95_lower']:.4f}x | "
            f"{sequence['one_sided_95_lower']:.4f}x | "
            f"{'PASS' if overall else 'FAIL'} | {'PASS' if stages else 'FAIL'} | "
            f"{row['model_gate']} |"
        )
    lines += [
        "",
        "## 门槛",
        "",
        "`PROMOTE_TO_MINIMAL_RTL` 当且仅当 sample 与 sequence-cluster 两种配对",
        "bootstrap 中更保守的单侧 95% 加速下界仍 `>=1.20x`，且总体及",
        "每个 stage 的加权 window p95 均不退化。否则停止 Local5 微机制扩展。",
        "",
        "## 候选含义",
        "",
        "- `C0`：合法 1RW TCFM5 + 每 output tile 重算 relation。",
        "- `C1`：两槽 source-resident Acc context，relation 仍重算。",
        "- `C2`：直接 1RW 后端 + 7 KiB critical-only exact Relation Memo。",
        "- `C3`：C1+C2 双生命期精确状态层次。",
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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.manifest, args.selection_plan, args.prereg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(
        args.output_dir / "report.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write(args.output_dir / "report.md", render_markdown(report))
    print("PASS Local5 joint candidate preregistered decision")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
