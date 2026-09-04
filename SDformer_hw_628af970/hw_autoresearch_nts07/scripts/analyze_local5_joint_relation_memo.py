#!/usr/bin/env python3
"""用同窗全 head 正式 trace 评估 Local5 exact Relation Memo。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import model_local5_relation_vault as vault


SAMPLING_ID = "uniform_plan_window_all_heads_v1"
SAMPLING_SEED = 20260809
STAGE_HEADS = (3, 6, 12, 24)
STAGE_WINDOWS = (440, 120, 30, 10)
STAGE_DEPTHS = (2, 2, 6, 2)
BLOCK_PAIRS = tuple(
    (stage, block)
    for stage, depth in enumerate(STAGE_DEPTHS)
    for block in range(depth)
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def string_list_sha256(values: list[str]) -> str:
    return hashlib.sha256(
        ("\n".join(values) + "\n").encode("utf-8")
    ).hexdigest()


def uniform_window(sample: int, stage: int, block: int) -> int:
    material = f"{SAMPLING_SEED}:{sample}:{stage}:{block}".encode("ascii")
    local_seed = int.from_bytes(hashlib.sha256(material).digest()[:16], "big")
    return random.Random(local_seed).randrange(STAGE_WINDOWS[stage])


@dataclass(frozen=True)
class JointWindow:
    sample: int
    stage: int
    block: int
    window: int
    analysis_weight: float
    service_cycles: np.ndarray
    packet_storage_bits: np.ndarray


def load_selection_plan(path: Path) -> tuple[dict[tuple[int, int, int], dict], dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != "local5_uniform_joint_window_plan_v1"
        or value.get("sampling_id") != SAMPLING_ID
        or value.get("seed") != SAMPLING_SEED
    ):
        raise ValueError("selection plan schema/sampling错误")
    source_manifest = Path(str(value.get("source_cohort_manifest", ""))).resolve()
    if (
        not source_manifest.is_file()
        or value.get("source_cohort_manifest_sha256") != sha256(source_manifest)
    ):
        raise ValueError("selection plan source cohort manifest绑定失效")
    source_value = json.loads(source_manifest.read_text(encoding="utf-8"))
    if source_value.get("cohort_sha256") != value.get("cohort_sha256"):
        raise ValueError("selection plan source cohort SHA不一致")
    records: dict[tuple[int, int, int], dict] = {}
    for row in value.get("records") or []:
        key = (int(row["sample"]), int(row["stage"]), int(row["block"]))
        stage = key[1]
        if (
            key in records
            or stage not in range(4)
            or int(row["heads"]) != STAGE_HEADS[stage]
            or int(row["batch_windows"]) != STAGE_WINDOWS[stage]
            or not 0 <= int(row["window"]) < STAGE_WINDOWS[stage]
            or int(row["window"]) != uniform_window(*key)
            or float(row["inclusion_probability"])
            != 1.0 / STAGE_WINDOWS[stage]
            or float(row["analysis_weight"]) != float(STAGE_WINDOWS[stage])
        ):
            raise ValueError(f"selection plan记录错误: {key}")
        records[key] = row
    expected = {
        (sample, stage, block)
        for sample in range(100)
        for stage, block in BLOCK_PAIRS
    }
    if set(records) != expected:
        raise ValueError("selection plan未覆盖100 samples x 12 blocks")
    return records, value


def load_joint_windows(
    manifest_path: Path, selection_path: Path
) -> tuple[list[JointWindow], dict, dict, Path, dict, Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("sampling", {}).get("method") != SAMPLING_ID
        or manifest.get("qualification", {}).get("qualified") is not True
        or manifest.get("qualification", {}).get("captured_groups") != 13800
    ):
        raise ValueError("joint manifest未通过正式qualification")
    if manifest.get("sampling", {}).get("selection_plan_sha256") != sha256(
        selection_path
    ):
        raise ValueError("manifest与selection plan哈希不一致")
    plan, plan_value = load_selection_plan(selection_path)
    if manifest.get("cohort_sha256") != plan_value.get("cohort_sha256"):
        raise ValueError("manifest与selection plan cohort不一致")
    payload_path = manifest_path.parent / str(manifest["payload_file"])
    if not payload_path.is_file() or manifest.get("payload_sha256") != sha256(
        payload_path
    ):
        raise ValueError("joint payload缺失或哈希失效")
    payload = np.load(payload_path, mmap_mode="r")
    offsets = np.asarray(payload["descriptor_group_offsets"], dtype=np.int64)
    term_count = np.asarray(payload["source_term_count"], dtype=np.int64)
    groups = manifest.get("groups") or []
    if len(groups) != 13800 or len(offsets) != len(groups) + 1:
        raise ValueError("joint group/payload offset数量错误")
    cohort_path = manifest_path.parent / str(manifest.get("cohort_file", ""))
    if (
        not cohort_path.is_file()
        or manifest.get("cohort_file_sha256") != sha256(cohort_path)
    ):
        raise ValueError("joint cohort文件缺失或哈希失效")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    sequence_keys = cohort.get("sequence_keys") or []
    observed_sequence_counts = {
        key: sequence_keys.count(key) for key in sorted(set(sequence_keys))
    }
    if (
        len(sequence_keys) != 100
        or any(not str(key) for key in sequence_keys)
        or len(set(sequence_keys)) != 18
        or cohort.get("sequence_key_sha256")
        != string_list_sha256([str(key) for key in sequence_keys])
        or cohort.get("sequence_counts") != observed_sequence_counts
        or cohort.get("sample_key_sha256") != manifest.get("cohort_sha256")
    ):
        raise ValueError("joint cohort sequence/sample合同失效")
    identity_path = Path(str(manifest.get("run_identity_file", ""))).resolve()
    if (
        not identity_path.is_file()
        or manifest.get("run_identity_file_sha256") != sha256(identity_path)
    ):
        raise ValueError("joint run identity绑定失效")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if (
        identity.get("selection_plan_sha256") != sha256(selection_path)
        or identity.get("cohort_sha256") != manifest.get("cohort_sha256")
        or identity.get("checkpoint_sha256") != manifest.get("checkpoint_sha256")
    ):
        raise ValueError("joint identity与manifest/plan不一致")
    gpu_audit_path = manifest_path.parent / "gpu_exclusivity_audit.json"
    if not gpu_audit_path.is_file():
        raise ValueError("joint GPU独占审计缺失")
    gpu_audit = json.loads(gpu_audit_path.read_text(encoding="utf-8"))
    if (
        gpu_audit.get("schema") != "local5_joint_gpu_exclusivity_audit_v1"
        or gpu_audit.get("status") != "PASS"
        or gpu_audit.get("identity_sha256") != sha256(identity_path)
        or gpu_audit.get("manifest_sha256") != sha256(manifest_path)
        or gpu_audit.get("payload_sha256") != sha256(payload_path)
        or gpu_audit.get("foreign_compute_pids") != []
    ):
        raise ValueError("joint GPU独占审计未通过或哈希失效")

    rows_by_key: dict[tuple[int, int, int, int], list[tuple[int, dict]]] = defaultdict(list)
    for index, row in enumerate(groups):
        sample = int(row["sample"])
        stage = int(row["stage"])
        block = int(row["block"])
        window = int(row["window"])
        plan_row = plan.get((sample, stage, block))
        if (
            plan_row is None
            or row.get("selection") != SAMPLING_ID
            or window != int(plan_row["window"])
        ):
            raise ValueError(f"group未绑定selection plan: index={index}")
        rows_by_key[(sample, stage, block, window)].append((index, row))

    windows: list[JointWindow] = []
    expected_joint_keys = {
        (sample, stage, block, int(row["window"]))
        for (sample, stage, block), row in plan.items()
    }
    if set(rows_by_key) != expected_joint_keys:
        raise ValueError("joint window key集合不完整")
    for key in sorted(rows_by_key):
        sample, stage, block, window = key
        rows = rows_by_key[key]
        heads = STAGE_HEADS[stage]
        if (
            len(rows) != heads
            or [int(row["head"]) for _, row in rows] != list(range(heads))
        ):
            raise ValueError(f"joint head顺序/覆盖错误: {key}")
        service = np.zeros(heads, dtype=np.int64)
        packets = np.zeros(heads, dtype=np.int64)
        for head, (index, _) in enumerate(rows):
            begin = int(offsets[index])
            end = int(offsets[index + 1])
            terms = term_count[begin:end]
            active_sources = int(np.count_nonzero(terms))
            service[head] = 15 + int(terms.sum(dtype=np.int64))
            packets[head] = active_sources * vault.RELATION_MACRO_WORD_BITS
        windows.append(
            JointWindow(
                sample=sample,
                stage=stage,
                block=block,
                window=window,
                analysis_weight=float(plan[(sample, stage, block)]["analysis_weight"]),
                service_cycles=service,
                packet_storage_bits=packets,
            )
        )
    return windows, manifest, plan_value, payload_path, cohort, gpu_audit_path


def sequence_cluster_ratio_ci(
    baseline_by_sample: np.ndarray,
    candidate_by_sample: np.ndarray,
    sequence_keys: list[str],
    *,
    trials: int,
    seed: int,
) -> dict[str, float]:
    if baseline_by_sample.shape != candidate_by_sample.shape:
        raise ValueError("cluster数组shape不一致")
    samples = int(baseline_by_sample.size)
    if len(sequence_keys) != samples:
        raise ValueError("sequence key数量与sample不一致")
    clusters: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(sequence_keys):
        clusters[str(key)].append(index)
    cluster_names = sorted(clusters)
    baseline_cluster = np.asarray(
        [baseline_by_sample[clusters[name]].sum() for name in cluster_names],
        dtype=np.float64,
    )
    candidate_cluster = np.asarray(
        [candidate_by_sample[clusters[name]].sum() for name in cluster_names],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    ratios = np.empty(trials, dtype=np.float64)
    for trial in range(trials):
        selected = rng.integers(
            0, len(cluster_names), size=len(cluster_names)
        )
        ratios[trial] = float(
            baseline_cluster[selected].sum()
            / candidate_cluster[selected].sum()
        )
    return {
        "ratio_of_means": float(
            baseline_by_sample.sum() / candidate_by_sample.sum()
        ),
        "bootstrap_mean": float(ratios.mean()),
        "ci95_lower": float(np.percentile(ratios, 2.5)),
        "ci95_upper": float(np.percentile(ratios, 97.5)),
        "sequence_clusters": len(cluster_names),
        "max_samples_per_sequence": max(len(indices) for indices in clusters.values()),
    }


def evaluate(
    windows: list[JointWindow],
    *,
    capacity_kib: int,
    policy: str,
    trials: int,
    seed: int,
    sequence_keys: list[str],
) -> dict:
    capacity_bits = int(capacity_kib) * 8192
    baseline_by_sample = np.zeros(100, dtype=np.float64)
    vault_by_sample = np.zeros(100, dtype=np.float64)
    builds_baseline = 0.0
    builds_vault = 0.0
    resident_heads = 0.0
    total_heads = 0.0
    stage_windows: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for row in windows:
        baseline, candidate, resident, build_count, _ = vault.cycles_for_window(
            row.service_cycles,
            row.packet_storage_bits,
            capacity_bits,
            policy,
        )
        weight = row.analysis_weight
        baseline_by_sample[row.sample] += baseline * weight
        vault_by_sample[row.sample] += candidate * weight
        builds_baseline += len(row.service_cycles) ** 2 * weight
        builds_vault += build_count * weight
        resident_heads += resident * weight
        total_heads += len(row.service_cycles) * weight
        stage_windows[row.stage].append((baseline * weight, candidate * weight, weight))
    ci = sequence_cluster_ratio_ci(
        baseline_by_sample,
        vault_by_sample,
        sequence_keys,
        trials=trials,
        seed=seed,
    )
    per_stage = []
    for stage in range(4):
        values = stage_windows[stage]
        baseline = sum(value[0] for value in values)
        candidate = sum(value[1] for value in values)
        per_stage.append(
            {
                "stage": stage,
                "joint_windows": len(values),
                "speedup_ratio_of_means": baseline / candidate,
            }
        )
    return {
        "capacity_kib": capacity_kib,
        "policy": policy,
        "cluster_bootstrap": {
            "unit": "sequence",
            "samples": 100,
            "trials": trials,
            "seed": seed,
            **ci,
        },
        "weighted_frame_cycles": {
            "recompute_mean": float(baseline_by_sample.mean()),
            "vault_mean": float(vault_by_sample.mean()),
        },
        "relation_build_reduction": 1.0 - builds_vault / builds_baseline,
        "resident_head_fraction": resident_heads / total_heads,
        "per_stage": per_stage,
        "passes_model_gate": ci["ci95_lower"] >= 1.15,
    }


def render_markdown(report: dict) -> str:
    main = report["evaluations"]["critical_only_7kib"]
    first_fit = report["evaluations"]["first_fit_all_7kib"]
    ci = main["cluster_bootstrap"]
    lines = [
        "# Local5 同窗全 Head Relation Memo 评估",
        "",
        "## 结论",
        "",
        (
            "本报告使用同一 sample/block/window 的完整 head 集合，不再从 "
            "per-head 池独立 bootstrap。window 采用预提交均匀概率计划，按 "
            "sequence 聚类 bootstrap。"
        ),
        "",
        f"7 KiB critical-only 的全帧 ratio-of-means 为 `{ci['ratio_of_means']:.4f}x`，"
        f"95% CI 为 `[{ci['ci95_lower']:.4f}, {ci['ci95_upper']:.4f}]`。",
        "",
        (
            "模型门槛结论：`PASS`，可进入 checkpoint-bound 单顶层 RTL。"
            if main["passes_model_gate"]
            else "模型门槛结论：`FAIL`，不得扩大 Relation Memo RTL。"
        ),
        "",
        "## 公平口径",
        "",
        "- 强基线仍为每个 head/output-tile 的 `max(450, projection service)`。",
        "- 7 KiB memo 容量和 112-bit 原生 relation record 与旧模型一致。",
        "- 容量 miss 与未 admission 的 head 走 exact recompute fallback。",
        "- 报告是 `[prof]+[模型]`，不是 RTL 周期、ASIC PPA、功耗或 EDP。",
        "",
        "## Admission 消融",
        "",
        "| 策略 | ratio-of-means | 95% CI lower | build 减少 | 驻留 head |",
        "|---|---:|---:|---:|---:|",
        f"| critical-only | {ci['ratio_of_means']:.4f}x | {ci['ci95_lower']:.4f}x | "
        f"{100*main['relation_build_reduction']:.2f}% | "
        f"{100*main['resident_head_fraction']:.2f}% |",
        f"| first-fit all | {first_fit['cluster_bootstrap']['ratio_of_means']:.4f}x | "
        f"{first_fit['cluster_bootstrap']['ci95_lower']:.4f}x | "
        f"{100*first_fit['relation_build_reduction']:.2f}% | "
        f"{100*first_fit['resident_head_fraction']:.2f}% |",
        "",
        "## 分 Stage",
        "",
        "| Stage | joint windows | 周期比 |",
        "|---:|---:|---:|",
    ]
    for row in main["per_stage"]:
        lines.append(
            f"| S{row['stage']} | {row['joint_windows']} | "
            f"{row['speedup_ratio_of_means']:.4f}x |"
        )
    lines += [
        "",
        "## 下一门槛",
        "",
        "即使模型门槛 PASS，仍需同一 trace 下 recompute 与 pack/replay/fallback "
        "单顶层 Acc32 零失配，并使用相同端口、反压和周期边界。模型 PASS 不等于 "
        "Relation Memo 已成为 DATE 贡献。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()
    windows, manifest, plan, payload_path, cohort, gpu_audit_path = load_joint_windows(
        args.manifest, args.selection_plan
    )
    sequence_keys = [str(value) for value in cohort["sequence_keys"]]
    evaluations = {
        "critical_only_7kib": evaluate(
            windows,
            capacity_kib=7,
            policy="critical_only",
            trials=args.trials,
            seed=args.seed,
            sequence_keys=sequence_keys,
        ),
        "first_fit_all_7kib": evaluate(
            windows,
            capacity_kib=7,
            policy="first_fit_all",
            trials=args.trials,
            seed=args.seed,
            sequence_keys=sequence_keys,
        ),
    }
    if any(
        item["cluster_bootstrap"]["sequence_clusters"] != 18
        for item in evaluations.values()
    ):
        raise RuntimeError("正式分析未形成18个sequence cluster")
    report = {
        "schema": "local5_joint_relation_memo_analysis_v1",
        "status": "PROFILE_MODEL_COMPLETE_NOT_RTL",
        "evidence": "[prof]+[模型]+[待验证]",
        "input": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "payload": str(payload_path.resolve()),
            "payload_sha256": sha256(payload_path),
            "selection_plan": str(args.selection_plan.resolve()),
            "selection_plan_sha256": sha256(args.selection_plan),
            "cohort_file": str(
                (args.manifest.parent / str(manifest["cohort_file"])).resolve()
            ),
            "cohort_file_sha256": manifest["cohort_file_sha256"],
            "gpu_exclusivity_audit": str(gpu_audit_path.resolve()),
            "gpu_exclusivity_audit_sha256": sha256(gpu_audit_path),
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "cohort_sha256": manifest["cohort_sha256"],
            "sampling_id": plan["sampling_id"],
            "joint_windows": len(windows),
            "head_groups": len(manifest["groups"]),
        },
        "method": {
            "window_estimator": "Horvitz-Thompson, weight=batch_windows",
            "confidence_interval": "sequence-cluster percentile bootstrap",
            "trials": args.trials,
            "seed": args.seed,
            "model_gate": "95% CI lower bound >= 1.15x",
        },
        "evaluations": evaluations,
        "source_bindings": {
            "analyzer": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "vault_model": {
                "path": str(Path(vault.__file__).resolve()),
                "sha256": sha256(Path(vault.__file__).resolve()),
            },
        },
        "decision": (
            "CONDITIONAL_RTL_GATE_PASS"
            if evaluations["critical_only_7kib"]["passes_model_gate"]
            else "REJECT_RELATION_MEMO_EXPANSION"
        ),
        "limitations": [
            "每个sample/block只抽一个window；使用已知纳入概率和sequence聚类CI",
            "周期为模型，不是RTL、ASIC PPA、功耗或EDP",
            "尚未完成同trace单顶层Acc32 miter",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(report["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
