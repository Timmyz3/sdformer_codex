#!/usr/bin/env python3
"""FCIP有限资源逐row周期边界与强基线模型。"""

from __future__ import annotations

import argparse
import base64
import json
import math
import zlib
from pathlib import Path

import numpy as np

try:
    from scripts.analyze_hit_flow_ordered_profiles import decode_count_trace
    from scripts.model_architecture_innovation_round12 import (
        storage_ledger,
        summarize,
    )
except ModuleNotFoundError:
    from analyze_hit_flow_ordered_profiles import decode_count_trace
    from model_architecture_innovation_round12 import storage_ledger, summarize


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = (
    ROOT.parent
    / "neuron_experiments/H9_bipolar_self_attention/results"
    / "h67_ep19_ttb_delta_cycle_v2_profile100_20260713"
    / "nts11_hardware_p0_profile.json"
)
DEFAULT_OUT = ROOT / "results/fcip_finite_resource_model_20260730"


def ceil_div(value: int, width: int) -> int:
    return (value + width - 1) // width


def decode_numpy_trace(encoded: dict) -> np.ndarray:
    dtypes = {
        "int16_le": np.dtype("<i2"),
        "int32_le": np.dtype("<i4"),
    }
    if encoded.get("dtype") not in dtypes:
        raise ValueError("不支持的ordered trace dtype")
    raw = zlib.decompress(base64.b64decode(encoded["data"]))
    values = np.frombuffer(raw, dtype=dtypes[encoded["dtype"]])
    expected = math.prod(int(value) for value in encoded["shape"])
    if values.size != expected:
        raise ValueError("ordered trace字节数与shape不一致")
    return values.reshape(encoded["shape"])


def load_rows(profile: dict) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    fields = {
        "active_classes": "projection_active_classes_h67_ordered_trace",
        "score_class_lane_terms": (
            "projection_class_channel_terms_h67_ordered_trace"
        ),
        "final_gate_lane_terms": (
            "projection_gate_class_channel_terms_deploy_ordered_trace"
        ),
        "active_lane_events": (
            "projection_baseline_active_lanes_ordered_trace"
        ),
    }
    for record in profile["summary"]["h60_records"]:
        decoded = {
            key: [
                int(value)
                for value in decode_count_trace(record[trace_name])
            ]
            for key, trace_name in fields.items()
        }
        encoded_k_count = record["pair_k_count_ordered_trace"]
        k_count = decode_numpy_trace(encoded_k_count)
        if k_count.ndim != 4 or k_count.shape[0] != 2:
            raise ValueError("pair_k_count trace必须是[2,B,H,N]")
        active_tokens = k_count.astype(bool).sum(axis=(0, 3)).reshape(-1)
        lengths = {len(values) for values in decoded.values()}
        if len(lengths) != 1 or len(active_tokens) != next(iter(lengths)):
            raise ValueError("FCIP ordered trace长度不一致")
        for index in range(next(iter(lengths))):
            rows.append(
                {
                    **{
                        key: values[index]
                        for key, values in decoded.items()
                    },
                    "active_tokens": int(active_tokens[index]),
                }
            )
    return rows


def fragment_bounds(row: dict[str, int], segments: int) -> tuple[int, int]:
    lower = row["score_class_lane_terms"]
    upper = min(
        row["score_class_lane_terms"] * segments,
        row["active_lane_events"],
    )
    if lower > upper:
        raise ValueError("fragment上下界不闭合")
    return lower, upper


def fcip_cycles(
    row: dict[str, int],
    *,
    fragments: int,
    and_width: int,
    emit_width: int,
    product_width: int,
) -> int:
    alias_cycles = row["active_classes"]
    intersection_cycles = ceil_div(fragments, and_width)
    emit_cycles = ceil_div(fragments, emit_width)
    product_cycles = ceil_div(row["final_gate_lane_terms"], product_width)
    return alias_cycles + max(
        intersection_cycles,
        emit_cycles,
        product_cycles,
    )


def current_postnorm_cycles(
    *,
    tokens: int,
    g1_slots: int = 4,
    lanes: int = 32,
) -> int:
    # member replay/build与G1定长{slot,lane}扫描；共同normalization不计。
    return tokens + g1_slots * lanes


def materialized_relation_storage(
    tokens: int,
    *,
    active_class_slots: int = 16,
    lanes: int = 32,
) -> int:
    base = storage_ledger(
        tokens,
        active_class_slots=active_class_slots,
        lanes=lanes,
    )
    # 用联合S×L×T位图替换FCIP的S×T和L×T因子平面；其余控制公平保留。
    return int(
        base["factorized_total_bits"]
        - base["factorized_active_class_bitmap_bits"]
        - base["factorized_k_lane_bitmap_bits"]
        + active_class_slots * lanes * tokens
    )


def model(profile: dict) -> dict:
    rows = load_rows(profile)
    tokens = 162
    segments = math.ceil(tokens / 64)
    baseline = current_postnorm_cycles(tokens=tokens)
    strong_baseline_cycles = [
        row["active_tokens"] + row["final_gate_lane_terms"]
        for row in rows
    ]
    lower_fragments = []
    upper_fragments = []
    for row in rows:
        lower, upper = fragment_bounds(row, segments)
        lower_fragments.append(lower)
        upper_fragments.append(upper)

    configurations = []
    for and_width, emit_width, product_width in (
        (1, 1, 1),
        (4, 1, 1),
        (4, 4, 1),
        (4, 4, 4),
        (8, 4, 4),
        (8, 8, 4),
        (8, 8, 8),
        (32, 8, 8),
    ):
        lower_cycles = [
            fcip_cycles(
                row,
                fragments=fragments,
                and_width=and_width,
                emit_width=emit_width,
                product_width=product_width,
            )
            for row, fragments in zip(rows, lower_fragments)
        ]
        upper_cycles = [
            fcip_cycles(
                row,
                fragments=fragments,
                and_width=and_width,
                emit_width=emit_width,
                product_width=product_width,
            )
            for row, fragments in zip(rows, upper_fragments)
        ]
        lower_summary = summarize(lower_cycles)
        upper_summary = summarize(upper_cycles)
        configurations.append(
            {
                "and_width": and_width,
                "emit_width": emit_width,
                "product_width": product_width,
                "lower_cycles": lower_summary,
                "upper_cycles": upper_summary,
                "mean_speedup_lower": baseline / lower_summary["mean"],
                "mean_speedup_upper": baseline / upper_summary["mean"],
                "p99_speedup_lower": baseline / lower_summary["p99"],
                "p99_speedup_upper": baseline / upper_summary["p99"],
                "aggregate_speedup_vs_strong_lower": (
                    sum(strong_baseline_cycles) / max(1, sum(lower_cycles))
                ),
                "aggregate_speedup_vs_strong_upper": (
                    sum(strong_baseline_cycles) / max(1, sum(upper_cycles))
                ),
                "p99_speedup_vs_strong_lower": (
                    summarize(strong_baseline_cycles)["p99"]
                    / max(1, lower_summary["p99"])
                ),
                "p99_speedup_vs_strong_upper": (
                    summarize(strong_baseline_cycles)["p99"]
                    / max(1, upper_summary["p99"])
                ),
            }
        )

    overflow_rows = sum(row["active_classes"] > 16 for row in rows)
    return {
        "schema": "fcip_finite_resource_bound_v1",
        "evidence": "[ordered prof]+[finite-resource bound model]，不是逐拍RTL/PPA",
        "rows": len(rows),
        "tokens": tokens,
        "segments": segments,
        "current_scs_g1_postnorm_cycles_per_row": baseline,
        "strong_sparse_replay_cycles": summarize(strong_baseline_cycles),
        "fragment_lower": summarize(lower_fragments),
        "fragment_upper": summarize(upper_fragments),
        "s16_overflow_rows": overflow_rows,
        "s16_overflow_row_ratio": overflow_rows / len(rows),
        "configurations": configurations,
        "storage_bits": {
            "current_scs_g1_t162": storage_ledger(tokens)[
                "current_total_bits"
            ],
            "materialized_s16_class_lane_relation_t162": (
                materialized_relation_storage(tokens)
            ),
            "factorized_s16_fcip_t162": storage_ledger(tokens)[
                "factorized_total_bits"
            ],
        },
        "model_contract": {
            "common_normalization_excluded": True,
            "fcip_alias_cycles": "active score classes/row",
            "fcip_service": (
                "alias + max(class-lane-segment/AND width, fragment/emit width, "
                "final-gate-lane term/product width)"
            ),
            "fallback": (
                "统计overflow比例但未把rare whole-row replay摊入均值；"
                "真实逐拍模型必须加入"
            ),
            "materialized_baseline": (
                "仓库gatestack_transposed_bitmap_bank式S×L×T联合关系；"
                "本轮只比较storage，不虚构其周期"
            ),
        },
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# FCIP 有限资源周期边界",
        "",
        f"- ordered rows：{report['rows']}",
        (
            "- 当前SCS+G1共同normalization之后的基线："
            f"{report['current_scs_g1_postnorm_cycles_per_row']} cycle/row"
        ),
        (
            "- S16 overflow row："
            f"{report['s16_overflow_row_ratio']:.6%}"
        ),
        "",
        "| AND | fragment emit | product | mean speedup区间 | p99 speedup区间 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in report["configurations"]:
        lines.append(
            f"| {row['and_width']} | {row['emit_width']} | "
            f"{row['product_width']} | "
            f"{row['mean_speedup_upper']:.3f}x–"
            f"{row['mean_speedup_lower']:.3f}x | "
            f"{row['p99_speedup_upper']:.3f}x–"
            f"{row['p99_speedup_lower']:.3f}x |"
        )
    lines += [
        "",
        "### 相对B1强基线",
        "",
        "B1按逐row `active token + occupied final-gate/lane term` 计数，",
        "代表稀疏member replay与occupied term扫描，不包含定长162-token建表。",
        "",
        "| AND | fragment emit | product | aggregate speedup区间 | p99 speedup区间 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in report["configurations"]:
        lines.append(
            f"| {row['and_width']} | {row['emit_width']} | "
            f"{row['product_width']} | "
            f"{row['aggregate_speedup_vs_strong_upper']:.3f}x–"
            f"{row['aggregate_speedup_vs_strong_lower']:.3f}x | "
            f"{row['p99_speedup_vs_strong_upper']:.3f}x–"
            f"{row['p99_speedup_vs_strong_lower']:.3f}x |"
        )
    storage = report["storage_bits"]
    lines += [
        "",
        "## 逻辑状态强基线",
        "",
        "| 实现 | bit |",
        "|---|---:|",
        f"| 当前SCS+G1 | {storage['current_scs_g1_t162']} |",
        (
            "| S16联合class-lane关系平面 | "
            f"{storage['materialized_s16_class_lane_relation_t162']} |"
        ),
        (
            "| S16 FCIP因子平面 | "
            f"{storage['factorized_s16_fcip_t162']} |"
        ),
        "",
        "## 解释",
        "",
        "- `1/1/1`表示单AND、单fragment出口、单product出口；增加AND但不增加",
        "  fragment出口不会消除串行瓶颈。",
        "- fragment上下界来自旧ordered trace；新增真实segment hook尚未运行。",
        "- B0 current只计G1定长建表和扫描；B1用active-token sparse replay与",
        "  occupied final-gate term，是FCIP必须面对的更强软件/硬件下界。",
        "- 共同normalization与共同projection backend不计。",
        "- rare fallback、SRAM latency、RMW、多口代价和backpressure尚未摊入。",
        "- 因此表中是证伪边界，不是论文加速比。",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    report = model(profile)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.out / "report.md").write_text(
        render_markdown(report) + "\n",
        encoding="utf-8",
    )
    print(args.out / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
