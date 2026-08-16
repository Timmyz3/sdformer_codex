"""Analyze exact TTX/H67 temporal-pair traces for architecture DSE."""

from __future__ import annotations

import argparse
import base64
import json
import math
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


LANES = (4, 8, 16, 32)


def decode_trace(encoded: dict[str, Any]) -> np.ndarray:
    if encoded.get("codec") != "zlib_base64" or encoded.get("dtype") != "int16_le":
        raise ValueError(f"unsupported ordered trace encoding: {encoded}")
    raw = zlib.decompress(base64.b64decode(encoded["data"]))
    shape = tuple(int(item) for item in encoded["shape"])
    array = np.frombuffer(raw, dtype="<i2")
    if array.size != math.prod(shape):
        raise ValueError(f"ordered trace size mismatch: {array.size} != {shape}")
    return array.reshape(shape).astype(np.int64, copy=False)


def rne_div16(numerator: np.ndarray) -> np.ndarray:
    quotient = numerator // 16
    remainder = numerator % 16
    return quotient + ((remainder > 8) | ((remainder == 8) & ((quotient & 1) != 0)))


def record_arrays(record: dict[str, Any]) -> dict[str, np.ndarray]:
    q = decode_trace(record["pair_q_count_ordered_trace"])
    k = decode_trace(record["pair_k_count_ordered_trace"])
    overlap = decode_trace(record["pair_overlap_ordered_trace"])
    union = decode_trace(record["pair_four_vector_union_ordered_trace"])
    if q.shape != k.shape or q.shape != overlap.shape or q.shape[0] != 2:
        raise ValueError("Q/K/overlap trace shape mismatch")
    if union.shape != q.shape[1:]:
        raise ValueError("four-vector union trace shape mismatch")
    events = (q + k).sum(axis=0)
    motion = decode_trace(record["pair_motion_ordered_trace"])
    update = decode_trace(record["pair_update_ordered_trace"])
    if motion.shape != union.shape or update.shape != union.shape:
        raise ValueError("motion/update trace shape mismatch")
    same_zero = int(record["head_dim"]) - q - k + overlap
    ttx_score = rne_div16(64 * overlap + same_zero)
    h67_score = rne_div16(64 * overlap + same_zero + 16 * motion[None, ...])
    return {
        "q": q,
        "k": k,
        "overlap": overlap,
        "union": union,
        "events": events,
        "motion": motion,
        "update": update,
        "ttx_score": ttx_score,
        "h67_score": h67_score,
    }


def traffic_bits(events: np.ndarray, union: np.ndarray, head_dim: int) -> dict[str, np.ndarray]:
    index_bits = max(1, math.ceil(math.log2(head_dim)))
    count_bits = math.ceil(math.log2(head_dim + 1))
    dense = np.full(events.shape, 4 * head_dim, dtype=np.int64)
    separate = 4 * count_bits + index_bits * events
    fused_union = count_bits + (index_bits + 4) * union
    return {"dense_bitmap": dense, "separate_index": separate, "fused_union": fused_union}


def aggregate_record(record: dict[str, Any]) -> dict[str, Any]:
    arrays = record_arrays(record)
    events = arrays["events"].reshape(-1)
    union = arrays["union"].reshape(-1)
    head_dim = int(record["head_dim"])
    traffic = traffic_bits(events, union, head_dim)
    names = tuple(traffic)
    traffic_stack = np.stack([traffic[name] for name in names], axis=0)
    best_traffic_route = np.argmin(traffic_stack, axis=0)
    kzero = arrays["k"] == 0
    both_kzero = kzero[0] & kzero[1]
    one_kzero = kzero[0] ^ kzero[1]
    both_active = ~kzero[0] & ~kzero[1]
    same_class_ttx = arrays["ttx_score"][0] == arrays["ttx_score"][1]
    same_class_h67 = arrays["h67_score"][0] == arrays["h67_score"][1]

    def row_fold_classes(scores: np.ndarray) -> np.ndarray:
        batch, heads, spatial = scores.shape[1:]
        flat_scores = scores.transpose(1, 2, 0, 3).reshape(batch, heads, 2 * spatial)
        flat_kzero = kzero.transpose(1, 2, 0, 3).reshape(batch, heads, 2 * spatial)
        result = np.zeros((batch, heads), dtype=np.int64)
        for batch_idx in range(batch):
            for head_idx in range(heads):
                result[batch_idx, head_idx] = np.unique(
                    flat_scores[batch_idx, head_idx][flat_kzero[batch_idx, head_idx]]
                ).size
        return result

    active_per_row = (~kzero).sum(axis=(0, 3)).astype(np.int64, copy=False)
    fold_ttx_per_row = row_fold_classes(arrays["ttx_score"])
    fold_h67_per_row = row_fold_classes(arrays["h67_score"])
    pairs_per_row = int(arrays["union"].shape[-1])
    dual_active_per_row = both_active.sum(axis=-1).astype(np.int64, copy=False)
    dual_class_ttx_per_row = (both_kzero & ~same_class_ttx).sum(axis=-1).astype(np.int64, copy=False)
    dual_class_h67_per_row = (both_kzero & ~same_class_h67).sum(axis=-1).astype(np.int64, copy=False)
    front_oneport_ttx = pairs_per_row + dual_active_per_row + dual_class_ttx_per_row
    front_oneport_h67 = pairs_per_row + dual_active_per_row + dual_class_h67_per_row
    front_dualactive_ttx = pairs_per_row + dual_class_ttx_per_row
    front_dualactive_h67 = pairs_per_row + dual_class_h67_per_row
    backend_ttx = np.maximum(active_per_row, 1) + fold_ttx_per_row + active_per_row + 3
    backend_h67 = np.maximum(active_per_row, 1) + 2 * fold_h67_per_row + active_per_row + 3
    result: dict[str, Any] = {
        "stage": int(record["stage"]),
        "block": int(record["block"]),
        "pairs": int(events.size),
        "pair_empty": int(np.count_nonzero(events == 0)),
        "events": int(events.sum()),
        "union_lanes": int(union.sum()),
        "traffic_bits": {name: int(values.sum()) for name, values in traffic.items()},
        "adaptive_traffic_bits": int(traffic_stack.min(axis=0).sum()),
        "adaptive_traffic_routes": {
            name: int(np.count_nonzero(best_traffic_route == index))
            for index, name in enumerate(names)
        },
        "score_equal_h67": int(np.count_nonzero(arrays["h67_score"][0] == arrays["h67_score"][1])),
        "score_equal_ttx": int(np.count_nonzero(arrays["ttx_score"][0] == arrays["ttx_score"][1])),
        "update_zero": int(np.count_nonzero(arrays["update"] == 0)),
        "motion_zero": int(np.count_nonzero(arrays["motion"] == 0)),
        "both_kzero": int(np.count_nonzero(both_kzero)),
        "one_kzero": int(np.count_nonzero(one_kzero)),
        "both_active": int(np.count_nonzero(both_active)),
        "both_kzero_same_class_ttx": int(np.count_nonzero(both_kzero & same_class_ttx)),
        "both_kzero_same_class_h67": int(np.count_nonzero(both_kzero & same_class_h67)),
        "both_kzero_dual_class_ttx": int(np.count_nonzero(both_kzero & ~same_class_ttx)),
        "both_kzero_dual_class_h67": int(np.count_nonzero(both_kzero & ~same_class_h67)),
        "row_active_entries": active_per_row.reshape(-1).tolist(),
        "row_fold_classes_ttx": fold_ttx_per_row.reshape(-1).tolist(),
        "row_fold_classes_h67": fold_h67_per_row.reshape(-1).tolist(),
        "row_front_oneport_ttx": front_oneport_ttx.reshape(-1).tolist(),
        "row_front_oneport_h67": front_oneport_h67.reshape(-1).tolist(),
        "row_front_dualactive_ttx": front_dualactive_ttx.reshape(-1).tolist(),
        "row_front_dualactive_h67": front_dualactive_h67.reshape(-1).tolist(),
        "row_backend_ttx": backend_ttx.reshape(-1).tolist(),
        "row_backend_h67": backend_h67.reshape(-1).tolist(),
        "row_current_front": np.full(active_per_row.size, 2 * pairs_per_row, dtype=np.int64).tolist(),
    }
    for lanes in LANES:
        dense_cycles = np.full(events.shape, math.ceil(head_dim / lanes), dtype=np.int64)
        separate_cycles = 1 + np.ceil(events / lanes).astype(np.int64)
        union_cycles = 1 + np.ceil(union / lanes).astype(np.int64)
        separate_cycles[events == 0] = 1
        union_cycles[union == 0] = 1
        cycle_stack = np.stack([dense_cycles, separate_cycles, union_cycles], axis=0)
        best_cycle_route = np.argmin(cycle_stack, axis=0)
        result[f"cycles_l{lanes}"] = {
            "dense_bitmap": int(dense_cycles.sum()),
            "separate_index": int(separate_cycles.sum()),
            "fused_union": int(union_cycles.sum()),
            "oracle_adaptive": int(cycle_stack.min(axis=0).sum()),
            "metadata_issue_lower_bound": int(events.size),
            "oracle_routes": {
                name: int(np.count_nonzero(best_cycle_route == index))
                for index, name in enumerate(names)
            },
        }
    return result


def sum_records(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    totals: dict[str, Any] = {
        "records": len(rows),
        "pairs": sum(row["pairs"] for row in rows),
        "pair_empty": sum(row["pair_empty"] for row in rows),
        "events": sum(row["events"] for row in rows),
        "union_lanes": sum(row["union_lanes"] for row in rows),
        "score_equal_h67": sum(row["score_equal_h67"] for row in rows),
        "score_equal_ttx": sum(row["score_equal_ttx"] for row in rows),
        "update_zero": sum(row["update_zero"] for row in rows),
        "motion_zero": sum(row["motion_zero"] for row in rows),
        "both_kzero": sum(row["both_kzero"] for row in rows),
        "one_kzero": sum(row["one_kzero"] for row in rows),
        "both_active": sum(row["both_active"] for row in rows),
        "both_kzero_same_class_ttx": sum(row["both_kzero_same_class_ttx"] for row in rows),
        "both_kzero_same_class_h67": sum(row["both_kzero_same_class_h67"] for row in rows),
        "both_kzero_dual_class_ttx": sum(row["both_kzero_dual_class_ttx"] for row in rows),
        "both_kzero_dual_class_h67": sum(row["both_kzero_dual_class_h67"] for row in rows),
        "traffic_bits": {
            name: sum(row["traffic_bits"][name] for row in rows)
            for name in ("dense_bitmap", "separate_index", "fused_union")
        },
        "adaptive_traffic_bits": sum(row["adaptive_traffic_bits"] for row in rows),
        "adaptive_traffic_routes": {
            name: sum(row["adaptive_traffic_routes"][name] for row in rows)
            for name in ("dense_bitmap", "separate_index", "fused_union")
        },
    }
    for lanes in LANES:
        key = f"cycles_l{lanes}"
        totals[key] = {
            field: sum(row[key][field] for row in rows)
            for field in (
                "dense_bitmap", "separate_index", "fused_union",
                "oracle_adaptive", "metadata_issue_lower_bound",
            )
        }
        totals[key]["oracle_routes"] = {
            name: sum(row[key]["oracle_routes"][name] for row in rows)
            for name in ("dense_bitmap", "separate_index", "fused_union")
        }
    pairs = totals["pairs"]
    dense_bits = totals["traffic_bits"]["dense_bitmap"]
    totals.update({
        "pair_empty_ratio": totals["pair_empty"] / pairs if pairs else 0.0,
        "mean_events_per_pair": totals["events"] / pairs if pairs else 0.0,
        "mean_union_lanes_per_pair": totals["union_lanes"] / pairs if pairs else 0.0,
        "score_equal_h67_ratio": totals["score_equal_h67"] / pairs if pairs else 0.0,
        "score_equal_ttx_ratio": totals["score_equal_ttx"] / pairs if pairs else 0.0,
        "update_zero_ratio": totals["update_zero"] / pairs if pairs else 0.0,
        "motion_zero_ratio": totals["motion_zero"] / pairs if pairs else 0.0,
        "adaptive_traffic_reduction_vs_dense": (
            1.0 - totals["adaptive_traffic_bits"] / dense_bits if dense_bits else 0.0
        ),
        "both_kzero_ratio": totals["both_kzero"] / pairs if pairs else 0.0,
        "one_kzero_ratio": totals["one_kzero"] / pairs if pairs else 0.0,
        "both_active_ratio": totals["both_active"] / pairs if pairs else 0.0,
        "both_kzero_same_class_ttx_ratio": totals["both_kzero_same_class_ttx"] / pairs if pairs else 0.0,
        "both_kzero_same_class_h67_ratio": totals["both_kzero_same_class_h67"] / pairs if pairs else 0.0,
    })
    for mode in ("ttx", "h67"):
        active = [value for row in rows for value in row["row_active_entries"]]
        folds = [value for row in rows for value in row[f"row_fold_classes_{mode}"]]
        totals[f"row_active_entries_mean_{mode}"] = float(np.mean(active)) if active else 0.0
        totals[f"row_fold_classes_mean_{mode}"] = float(np.mean(folds)) if folds else 0.0
        for front_kind in ("oneport", "dualactive"):
            front = np.asarray(
                [value for row in rows for value in row[f"row_front_{front_kind}_{mode}"]],
                dtype=np.int64,
            )
            backend = np.asarray(
                [value for row in rows for value in row[f"row_backend_{mode}"]],
                dtype=np.int64,
            )
            current_front = np.asarray(
                [value for row in rows for value in row["row_current_front"]],
                dtype=np.int64,
            )
            totals[f"row_pipeline_{mode}_{front_kind}"] = pipeline_sweep(
                front, backend, current_front=current_front
            )
    return totals


def flowshop_cycles(front: np.ndarray, backend: np.ndarray, contexts: int) -> int:
    if front.size != backend.size:
        raise ValueError("front/backend row count mismatch")
    if contexts <= 0:
        raise ValueError("contexts must be positive")
    front_available = 0
    backend_available = 0
    releases: list[int] = []
    for index, (front_cycles, backend_cycles) in enumerate(zip(front, backend, strict=True)):
        context_available = releases[index - contexts] if index >= contexts else 0
        front_start = max(front_available, context_available)
        front_done = front_start + int(front_cycles)
        backend_start = max(backend_available, front_done)
        backend_done = backend_start + int(backend_cycles)
        front_available = front_done
        backend_available = backend_done
        releases.append(backend_done)
    return backend_available


def pipeline_sweep(
    front: np.ndarray,
    backend: np.ndarray,
    *,
    current_front: np.ndarray,
) -> dict[str, Any]:
    current = flowshop_cycles(current_front, backend, contexts=1)
    result: dict[str, Any] = {
        "rows": int(front.size),
        "current_token_serial_single_context": current,
        "front_work": int(front.sum()),
        "backend_work": int(backend.sum()),
        "infinite_context_lower_bound": int(max(front.sum(), backend.sum())),
    }
    for contexts in (1, 2, 4, 8):
        cycles = flowshop_cycles(front, backend, contexts=contexts)
        result[f"contexts_{contexts}"] = cycles
        result[f"reduction_vs_current_{contexts}"] = 1.0 - cycles / current if current else 0.0
    return result


def analyze(profile: dict[str, Any]) -> dict[str, Any]:
    if not profile.get("ordered_trace"):
        raise ValueError("profile must be collected with --ordered-trace")
    records = profile["summary"]["h60_records"]
    required = (
        "pair_q_count_ordered_trace", "pair_k_count_ordered_trace",
        "pair_overlap_ordered_trace", "pair_motion_ordered_trace",
        "pair_update_ordered_trace", "pair_four_vector_union_ordered_trace",
    )
    missing = [record.get("name", "unknown") for record in records if any(key not in record for key in required)]
    if missing:
        raise ValueError(f"profile lacks binary temporal-pair traces in {len(missing)} records")
    rows = [aggregate_record(record) for record in records]
    stage_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        stage_rows[int(row["stage"])].append(row)
    return {
        "schema_version": 1,
        "experiment": profile.get("experiment"),
        "checkpoint": profile.get("checkpoint"),
        "samples": profile.get("samples"),
        "model_summary": sum_records(rows),
        "stage_summary": {str(stage): sum_records(items) for stage, items in sorted(stage_rows.items())},
        "model_pair_histograms": profile["summary"].get("binary_temporal_pairs", {}),
        "scope": {
            "traffic": "Q0/Q1/K0/K1 pair payload only; excludes SRAM framing, score/class storage, projection and ATLIF",
            "cycles": "pair-front-end work cycles; one metadata issue per cycle; excludes SCS backend and memory stalls",
            "adaptive": "per-pair oracle lower bound, not an implemented speedup claim",
        },
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def render(result: dict[str, Any]) -> str:
    model = result["model_summary"]
    lines = [
        "# 二值时间对 Workload 与架构 DSE 报告",
        "",
        f"- 实验：`{result.get('experiment')}`",
        f"- checkpoint：`{result.get('checkpoint')}`",
        f"- samples：{result.get('samples')}",
        "- 本报告只评估 attention 时间对前端；周期是工作量下界，不是整网吞吐或芯片 speedup。",
        "",
        "## 全局特征",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 时间对 | {model['pairs']} |",
        f"| 四向量全空 | {pct(model['pair_empty_ratio'])} |",
        f"| 每对平均事件条目 | {model['mean_events_per_pair']:.4f} / 128 |",
        f"| 每对平均四向量并集 lane | {model['mean_union_lanes_per_pair']:.4f} / 32 |",
        f"| Q/K temporal update=0 | {pct(model['update_zero_ratio'])} |",
        f"| K motion=0 | {pct(model['motion_zero_ratio'])} |",
        f"| TTX 成对分数相等 | {pct(model['score_equal_ttx_ratio'])} |",
        f"| H67 成对分数相等 | {pct(model['score_equal_h67_ratio'])} |",
        f"| 双 K-zero pair | {pct(model['both_kzero_ratio'])} |",
        f"| 单侧 K-zero pair | {pct(model['one_kzero_ratio'])} |",
        f"| 双 K-active pair | {pct(model['both_active_ratio'])} |",
        f"| 双 K-zero 且同 TTX 类 | {pct(model['both_kzero_same_class_ttx_ratio'])} |",
        f"| 双 K-zero 且同 H67 类 | {pct(model['both_kzero_same_class_h67_ratio'])} |",
        "",
        "## 精确编码流量",
        "",
        "| 编码 | 总 bit | 相对 4x32 bitmap |",
        "|---|---:|---:|",
    ]
    dense = model["traffic_bits"]["dense_bitmap"]
    for name, label in (
        ("dense_bitmap", "4x32 bitmap"),
        ("separate_index", "四流 count+index"),
        ("fused_union", "union-index+4bit membership"),
    ):
        bits = model["traffic_bits"][name]
        lines.append(f"| {label} | {bits} | {pct(1.0 - bits / dense if dense else 0.0)} |")
    lines.append(
        f"| 每对自适应 oracle | {model['adaptive_traffic_bits']} | "
        f"{pct(model['adaptive_traffic_reduction_vs_dense'])} |"
    )
    lines += [
        "",
        "## 前端工作周期下界",
        "",
        "| 并行 lane | dense bitmap | 四流 index | fused union | 每对 oracle | metadata issue LB |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for lanes in LANES:
        row = model[f"cycles_l{lanes}"]
        lines.append(
            f"| {lanes} | {row['dense_bitmap']} | {row['separate_index']} | "
            f"{row['fused_union']} | {row['oracle_adaptive']} | {row['metadata_issue_lower_bound']} |"
        )
    lines += [
        "",
        "## Pair-fused 多行上下文流水模型",
        "",
        "`oneport` 假定一个 active-entry 写口和一个 class-hist 写口；同拍两个 active 或两个不同 class 需要追加一拍。",
        "`dualactive` 额外提供两个 active-entry 写口；class-hist 仍通过同类合并或串行解决冲突。Backend 使用当前 RTL 的",
        "active exp、K-zero class scan 和 gated-K emit 周期，不包含 SRAM 宏读延迟与外部 backpressure。",
        "",
        "| score mode | commit | current single-context | pair ctx1 | ctx2 | ctx4 | ctx8 | ctx4 reduction |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("ttx", "h67"):
        for commit in ("oneport", "dualactive"):
            row = model[f"row_pipeline_{mode}_{commit}"]
            lines.append(
                f"| {mode.upper()} | {commit} | {row['current_token_serial_single_context']} | "
                f"{row['contexts_1']} | {row['contexts_2']} | {row['contexts_4']} | {row['contexts_8']} | "
                f"{pct(row['reduction_vs_current_4'])} |"
            )
    lines += [
        "",
        "## 分 stage 特征",
        "",
        "| stage | pairs | empty | events/pair | union/pair | H67 pair-score equal | adaptive traffic reduction |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, row in result["stage_summary"].items():
        lines.append(
            f"| {stage} | {row['pairs']} | {pct(row['pair_empty_ratio'])} | "
            f"{row['mean_events_per_pair']:.4f} | {row['mean_union_lanes_per_pair']:.4f} | "
            f"{pct(row['score_equal_h67_ratio'])} | {pct(row['adaptive_traffic_reduction_vs_dense'])} |"
        )
    lines += [
        "",
        "## 架构含义",
        "",
        "1. `union-index+membership` 是无损时间对包：一个 lane 索引携带 Q0/Q1/K0/K1 四个 membership，",
        "   可在同一归约通路同时累加 qcount/kcount/overlap/motion，避免四条稀疏流重复索引。",
        "2. 若自适应编码显著优于任一固定编码，优先设计单核 representation-adaptive front-end；",
        "   只有当不同编码的服务时间造成持续队头阻塞时，才有证据升级为异构双核。",
        "3. H67 成对分数相等只允许复用分数/指数类，不能复用 gated-K 输出；K0/K1 仍须分别发射。",
        "4. `metadata issue LB` 与 SCS/backend、SRAM bank、FIFO 联合后才能形成吞吐结论。",
        "5. 多行上下文模型只证明解耦流水的调度潜力；提交端口数量、同类 collision、同步 SRAM 和 out-ready",
        "   必须进入 RTL 离散事件回放后才能形成论文数字。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = json.loads(args.profile_json.read_text(encoding="utf-8"))
    result = analyze(profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(render(result), encoding="utf-8")
    print(args.output.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
