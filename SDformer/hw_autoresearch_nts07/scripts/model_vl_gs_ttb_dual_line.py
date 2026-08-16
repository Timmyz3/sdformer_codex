#!/usr/bin/env python3
"""Motion/Local5 双线 Vocabulary-Lifecycle Gate-Slot TTB 模型。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

try:
    from scripts.analyze_hit_flow_ordered_profiles import decode_count_trace
    from scripts.evaluate_qfit_product_cache_policies import (
        bundle_bits_dynamic,
        descriptor_gate_dictionary,
        simulate_no_replace,
    )
except ModuleNotFoundError:
    from analyze_hit_flow_ordered_profiles import decode_count_trace
    from evaluate_qfit_product_cache_policies import (
        bundle_bits_dynamic,
        descriptor_gate_dictionary,
        simulate_no_replace,
    )


ROOT = Path(__file__).resolve().parents[1]
MOTION_PROFILE = (
    ROOT.parent
    / "neuron_experiments/H9_bipolar_self_attention/results"
    / "h67_ep19_ttb_delta_cycle_v2_profile100_20260713"
    / "nts11_hardware_p0_profile.json"
)
LOCAL_TRACE = (
    ROOT
    / "results/qfit_local5_projection_tile_yosys_20260731"
    / "ordered_term_trace.csv"
)
DEFAULT_OUT = ROOT / "results/vl_gs_ttb_dual_line_model_20260731"


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(probability * len(ordered)) - 1),
    )
    return ordered[index]


def distribution(values: Iterable[float]) -> dict[str, float]:
    body = [float(value) for value in values]
    return {
        "count": float(len(body)),
        "mean": statistics.fmean(body) if body else 0.0,
        "p50": percentile(body, 0.50),
        "p95": percentile(body, 0.95),
        "p99": percentile(body, 0.99),
        "max": max(body, default=0.0),
    }


def motion_eager_header(
    active_classes: list[int],
    terms: list[int],
    slots: int,
) -> dict[str, int | float]:
    if len(active_classes) != len(terms):
        raise ValueError("Motion active-class与term trace长度不一致")
    if slots <= 0:
        raise ValueError("slot数量必须为正")
    slot_bits = max(1, math.ceil(math.log2(slots)))
    count_bits = max(1, math.ceil(math.log2(slots + 1)))
    baseline_bits = 0
    encoded_bits = 0
    active_contexts = 0
    fallback_contexts = 0
    fallback_terms = 0
    active_class_sum = 0
    term_sum = 0
    for classes, context_terms in zip(active_classes, terms):
        if classes < 0 or context_terms < 0 or classes > context_terms:
            raise ValueError("Motion gate-class/term计数不守恒")
        if context_terms == 0:
            if classes != 0:
                raise ValueError("零term context含active gate class")
            continue
        active_contexts += 1
        active_class_sum += classes
        term_sum += context_terms
        baseline_bits += context_terms * 9
        encoded_bits += 1  # fast/raw mode
        if classes <= slots:
            # slot编号由header顺序隐式确定，不需要逐项携带slot id。
            encoded_bits += count_bits + classes * 9
            encoded_bits += context_terms * slot_bits
        else:
            fallback_contexts += 1
            fallback_terms += context_terms
            encoded_bits += context_terms * 9
    return {
        "slots": slots,
        "slot_bits": slot_bits,
        "active_contexts": active_contexts,
        "active_classes": active_class_sum,
        "terms": term_sum,
        "fallback_contexts": fallback_contexts,
        "fallback_context_ratio": (
            fallback_contexts / active_contexts if active_contexts else 0.0
        ),
        "fallback_terms": fallback_terms,
        "fallback_term_ratio": fallback_terms / term_sum if term_sum else 0.0,
        "baseline_gate_key_bits": baseline_bits,
        "vl_gs_ttb_gate_key_bits": encoded_bits,
        "gate_key_reduction": (
            1.0 - encoded_bits / baseline_bits if baseline_bits else 0.0
        ),
    }


def load_motion() -> tuple[list[dict], dict[int, tuple[list[int], list[int]]]]:
    profile = json.loads(MOTION_PROFILE.read_text(encoding="utf-8"))
    records = profile["summary"]["h60_records"]
    by_sample: dict[int, tuple[list[int], list[int]]] = {}
    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        grouped[int(record["sample_id"])].append(record)
    all_classes: list[int] = []
    all_terms: list[int] = []
    for sample, sample_records in sorted(grouped.items()):
        classes: list[int] = []
        terms: list[int] = []
        for record in sample_records:
            classes.extend(
                int(value)
                for value in decode_count_trace(
                    record[
                        "projection_active_gate_classes_deploy_ordered_trace"
                    ]
                )
            )
            terms.extend(
                int(value)
                for value in decode_count_trace(
                    record[
                        "projection_gate_class_channel_terms_deploy_ordered_trace"
                    ]
                )
            )
        by_sample[sample] = (classes, terms)
        all_classes.extend(classes)
        all_terms.extend(terms)
    return records, {-1: (all_classes, all_terms), **by_sample}


def evaluate_motion() -> dict[str, object]:
    records, traces = load_motion()
    all_classes, all_terms = traces[-1]
    policies = []
    for slots in (2, 4, 6, 8):
        aggregate = motion_eager_header(all_classes, all_terms, slots)
        sample_reductions = [
            float(motion_eager_header(classes, terms, slots)["gate_key_reduction"])
            for sample, (classes, terms) in traces.items()
            if sample >= 0
        ]
        aggregate["sample_gate_key_reduction"] = distribution(sample_reductions)
        policies.append(aggregate)
    return {
        "evidence": "[prof-ordered]+[exact bit model]，不是总线SAIF或PPA",
        "profile": str(MOTION_PROFILE),
        "records": len(records),
        "contexts": len(all_classes),
        "active_class_distribution": distribution(all_classes),
        "term_distribution": distribution(all_terms),
        "policies": policies,
        "product_start_claim": (
            "不主张减少；当前NMF的class-channel term已是一项一次product"
        ),
        "protocol": (
            "SCS完成整行class枚举后发送slot->gate eager header；超出S的整"
            "context走raw-gate exact fallback，禁止fast/raw部分提交混合"
        ),
    }


def load_local_rows() -> list[dict[str, int]]:
    with LOCAL_TRACE.open(newline="") as handle:
        rows = [
            {name: int(value) for name, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    if not rows:
        raise ValueError("Local5 ordered term trace为空")
    return rows


def evaluate_local5() -> dict[str, object]:
    rows = load_local_rows()
    policies = []
    for slots in (4, 6, 8):
        policy = simulate_no_replace(rows, slots)
        bundle = bundle_bits_dynamic(
            len(rows),
            int(policy["fills"]),
            int(policy["bypasses"]),
            slots,
        )
        policies.append({**policy, **bundle})
    return {
        "evidence": "[rtl-directed W6 trace]+[exact bit model]，不是fullres多样本",
        "trace": str(LOCAL_TRACE),
        "terms": len(rows),
        "policies": policies,
        "descriptor_gate_ttb": descriptor_gate_dictionary(rows),
        "protocol": (
            "FCSR流式到达时first-bind空slot；hit走primary slot，fill/bypass"
            "才向exception流发送9-bit gate；满表不替换并精确旁路"
        ),
    }


def build_report() -> dict[str, object]:
    return {
        "schema": "vl_gs_ttb_dual_line_model_v1",
        "architecture": {
            "name": "VL-GS-TTB",
            "full_name": "Vocabulary-Lifecycle Gate-Slot Token-Term Bundle",
            "shared_principle": (
                "gate vocabulary在attention与projection之间成为显式、有生命周期"
                "的slot对象，而非每term重复携带的立即数"
            ),
            "motion_mode": "eager context header + whole-context exact fallback",
            "local5_mode": "incremental primary/exception + no-replace exact bypass",
        },
        "motion": evaluate_motion(),
        "local5": evaluate_local5(),
        "claim_boundary": [
            "Motion只主张gate-key/control traffic机会，不主张product减少",
            "Local5 product-start数字仅来自单W6定向trace",
            "所有bit结果均未包含FIFO、wire capacitance、SRAM macro和SAIF",
            "Local5正式结论必须等待post-G0分层100-sample trace",
        ],
    }


def render_markdown(report: dict[str, object]) -> str:
    motion = report["motion"]
    local = report["local5"]
    lines = [
        "# VL-GS-TTB 双线架构机会模型",
        "",
        "## 1. 架构定义",
        "",
        "VL-GS-TTB 将 gate vocabulary 从每条 term 的9-bit立即数提升为具有",
        "明确建立、使用和失效时刻的硬件对象。Motion在SCS完成后一次性发布",
        "eager header；Local5在FCSR流中用primary/exception双流增量填槽。两者",
        "都保留raw-gate exact fallback，不做剪枝或近似。",
        "",
        "## 2. Motion eager-header结果",
        "",
        f"- ordered contexts：{motion['contexts']}；",
        f"- 证据：{motion['evidence']}；",
        "",
        "| slots | active context | fallback context | fallback term | gate-key减少 | sample mean/p95 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in motion["policies"]:
        sample = row["sample_gate_key_reduction"]
        lines.append(
            f"| {row['slots']} | {row['active_contexts']} | "
            f"{row['fallback_context_ratio']:.4%} | "
            f"{row['fallback_term_ratio']:.4%} | "
            f"{row['gate_key_reduction']:.4%} | "
            f"{sample['mean']:.4%}/{sample['p95']:.4%} |"
        )
    lines += [
        "",
        "Motion的NMF已经按class-channel生成一次product，因此本表不宣称减少",
        "乘法。收益对象仅为gate-key/control流；需RTL FIFO和SAIF才能晋级。",
        "",
        "## 3. Local5 incremental结果",
        "",
        f"- W6 term：{local['terms']}；",
        f"- 证据：{local['evidence']}；",
        "",
        "| slots | product start | product减少 | bypass | gate-key减少 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in local["policies"]:
        lines.append(
            f"| {row['policy'].split('_')[-1]} | {row['product_starts']} | "
            f"{row['reuse_ratio']:.4%} | {row['bypasses']} | "
            f"{row['exception_split_reduction']:.4%} |"
        )
    dictionary = local["descriptor_gate_ttb"]
    lines += [
        "",
        f"DG-TTB安全变长格式为{dictionary['variable_safe_total_bits']} bit，",
        f"原展开term为{dictionary['baseline_total_bits']} bit。该结果仍是单W6",
        "定向trace，fullres post-G0前不进入论文主表。",
        "",
        "## 4. 与已有工作的差分",
        "",
        "- Bishop：借header/body TTB纪律；本工作不用ECP，改为gate vocabulary",
        "  生命周期和整context/逐term exact fallback。",
        "- Prosperity：借精确值复用思想；本工作不用通用关联cache预测，slot由",
        "  attention上游确定并随bundle传递。",
        "- PHI：借primary/residual分层；本工作异常流携带原gate并精确执行，",
        "  不是pattern近似或残差预测。",
        "- Sanger：借stationary数据流表述；本工作驻留的是gate vocabulary和",
        "  Local5 product，Motion只驻留gate映射。",
        "",
        "## 5. 当前判定",
        "",
        "VL-GS-TTB已形成可区分的跨阶段架构合同，但当前等级仍是Motion",
        "`[prof]+[模型]`、Local5 `[rtl-directed]+[模型]`。下一步必须先实现",
        "共同slot resolver、Motion eager-header encoder/decoder和Local5",
        "primary/exception join，再与raw-gate、LRU和固定宽包做同producer回放。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    report = build_report()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
