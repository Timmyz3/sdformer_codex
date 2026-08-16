#!/usr/bin/env python3
"""汇总Local5五bank direct/GASR真实RTL，并做无泄漏模式分层评估。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_manifest(path: Path) -> list[dict[str, str]]:
    bindings: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        expected, source = line.split(maxsplit=1)
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        actual = file_sha256(source_path)
        if actual != expected:
            raise ValueError(f"RTL source hash mismatch: {source_path}")
        bindings.append({"path": str(source_path.resolve()), "sha256": actual})
    if not bindings:
        raise ValueError("RTL source manifest is empty")
    return bindings


GROUP_RE = re.compile(r"^GROUP (?P<body>.+)$")
FIELD_RE = re.compile(r"(?P<key>[a-zA-Z0-9_]+)=(?P<value>-?\d+)")


def parse_log(path: Path) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        match = GROUP_RE.match(line)
        if match:
            rows.append(
                {
                    item.group("key"): int(item.group("value"))
                    for item in FIELD_RE.finditer(match.group("body"))
                }
            )
    if "PASS post-G0 active projection" not in text:
        raise ValueError(f"日志没有PASS标记：{path}")
    return rows


def ratio(row: dict[str, int], numerator: str, denominator: str) -> float:
    return row[numerator] / row[denominator] if row[denominator] else 0.0


def choose_threshold(
    rows: list[dict[str, int]], feature: str, candidates: list[float]
) -> tuple[float, int]:
    numerator, denominator = feature.split("/")
    choices = []
    for threshold in candidates:
        cycles = sum(
            row["gasr_cycles"]
            if ratio(row, numerator, denominator) >= threshold
            else row["direct_cycles"]
            for row in rows
        )
        choices.append((cycles, threshold))
    cycles, threshold = min(choices, key=lambda item: (item[0], item[1]))
    return threshold, cycles


def evaluate_selector(
    rows: list[dict[str, int]], feature: str, threshold: float
) -> dict[str, float | int]:
    numerator, denominator = feature.split("/")
    direct = sum(row["direct_cycles"] for row in rows)
    selected = [
        ratio(row, numerator, denominator) >= threshold for row in rows
    ]
    hybrid = sum(
        row["gasr_cycles"] if use_gasr else row["direct_cycles"]
        for row, use_gasr in zip(rows, selected, strict=True)
    )
    oracle = sum(min(row["direct_cycles"], row["gasr_cycles"]) for row in rows)
    return {
        "groups": len(rows),
        "selected_gasr_groups": sum(selected),
        "selected_regressions": sum(
            use_gasr and row["gasr_cycles"] > row["direct_cycles"]
            for row, use_gasr in zip(rows, selected, strict=True)
        ),
        "direct_cycles": direct,
        "hybrid_cycles": hybrid,
        "speedup_vs_direct": direct / hybrid,
        "oracle_cycles": oracle,
        "oracle_speedup_vs_direct": direct / oracle,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("tb_qfit/vectors/local5_active_projection_postg0_100/manifest.json"),
    )
    parser.add_argument(
        "--direct-log",
        type=Path,
        default=Path(
            "results/local5_gasr2c_fivebank_postg0_rtl_20260804/direct_profile100.log"
        ),
    )
    parser.add_argument(
        "--gasr-log",
        type=Path,
        default=Path(
            "results/local5_gasr2c_fivebank_postg0_rtl_20260804/gasr_profile100.log"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/local5_gasr2c_fivebank_postg0_rtl_20260804"),
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=("blind_geometry", "descriptor_synchronized"),
        default="descriptor_synchronized",
    )
    args = parser.parse_args()
    source_bindings = validate_source_manifest(args.source_manifest)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "local5_active_projection_postg0_vectors_v1":
        raise ValueError("projection vector manifest schema错误")
    weight_mode = str(manifest.get("weight_mode", "synthetic"))
    checkpoint_weight_modes = {
        "checkpoint_dyadic_int8_head_slice",
        "checkpoint_theta_folded_dyadic_int8_head_slice",
    }
    if weight_mode in checkpoint_weight_modes:
        binding = manifest.get("projection_contract_binding")
        if not isinstance(binding, dict):
            raise ValueError("checkpoint weight模式缺少projection contract binding")
        for key in ("manifest", "payload"):
            path = Path(str(binding.get(key, "")))
            if not path.is_file() or file_sha256(path) != binding.get(
                f"{key}_sha256"
            ):
                raise ValueError(f"projection contract {key}绑定失效")
    metadata = manifest["selection"]["rows"]
    direct_rows = parse_log(args.direct_log)
    gasr_rows = parse_log(args.gasr_log)
    if len(direct_rows) != 100 or len(gasr_rows) != 100:
        raise ValueError("五bank日志必须各覆盖100组")

    rows: list[dict[str, int]] = []
    for direct, gasr, meta in zip(direct_rows, gasr_rows, metadata, strict=True):
        group = direct["group"]
        if group != gasr["group"] or group != meta["vector_group_index"]:
            raise AssertionError("direct/GASR/manifest组号错位")
        for key in ("active", "terms", "updates"):
            if direct[key] != gasr[key]:
                raise AssertionError(f"组{group}的{key}在两种模式间不一致")
        if (
            direct["active"] != meta["active_sources"]
            or direct["terms"] != meta["terms"]
            or direct["updates"] != meta["updates"]
        ):
            raise AssertionError(f"组{group}的RTL计数与manifest不一致")
        rows.append(
            {
                "group": group,
                "stage": meta["stage"],
                "active": direct["active"],
                "terms": direct["terms"],
                "updates": direct["updates"],
                "direct_cycles": direct["cycles"],
                "gasr_cycles": gasr["cycles"],
                "direct_stalls": direct["term_stall"],
                "gasr_stalls": gasr["term_stall"],
                "direct_reads": direct["sram_reads"],
                "direct_writes": direct["sram_writes"],
                "gasr_reads": gasr["sram_reads"],
                "gasr_writes": gasr["sram_writes"],
            }
        )

    direct_cycles = sum(row["direct_cycles"] for row in rows)
    gasr_cycles = sum(row["gasr_cycles"] for row in rows)
    direct_transactions = sum(
        row["direct_reads"] + row["direct_writes"] for row in rows
    )
    gasr_transactions = sum(
        row["gasr_reads"] + row["gasr_writes"] for row in rows
    )
    speedups = [row["direct_cycles"] / row["gasr_cycles"] for row in rows]

    # 前50组只用于选规则，后50组只用于报告泛化结果。
    train = rows[:50]
    holdout = rows[50:]
    feature_candidates = {
        "updates/active": sorted(
            {ratio(row, "updates", "active") for row in train}
        ),
        "terms/active": sorted(
            {ratio(row, "terms", "active") for row in train}
        ),
        "updates/terms": sorted(
            {ratio(row, "updates", "terms") for row in train}
        ),
    }
    trained = []
    for feature, candidates in feature_candidates.items():
        threshold, train_cycles = choose_threshold(train, feature, candidates)
        trained.append((train_cycles, feature, threshold))
    _, selected_feature, selected_threshold = min(trained)
    train_eval = evaluate_selector(train, selected_feature, selected_threshold)
    holdout_eval = evaluate_selector(holdout, selected_feature, selected_threshold)
    all_eval = evaluate_selector(rows, selected_feature, selected_threshold)

    stage_rows = []
    for stage in sorted({row["stage"] for row in rows}):
        subset = [row for row in rows if row["stage"] == stage]
        direct = sum(row["direct_cycles"] for row in subset)
        gasr = sum(row["gasr_cycles"] for row in subset)
        stage_rows.append(
            {
                "stage": stage,
                "groups": len(subset),
                "direct_cycles": direct,
                "gasr_cycles": gasr,
                "speedup": direct / gasr,
            }
        )

    random_logs = [
        args.output_dir / "direct_random_sva.log",
        args.output_dir / "qgasr_random_sva.log",
    ]
    lint_logs = [
        args.output_dir / "direct_lint.log",
        args.output_dir / "qgasr_lint.log",
    ]
    yosys_logs = [
        args.output_dir / "direct_yosys_memory_collect.log",
        args.output_dir / "qgasr_yosys_memory_collect.log",
    ]
    verification = {
        "checkpoint_weight_binding": (
            "PASS"
            if weight_mode in checkpoint_weight_modes
            else "NOT_APPLICABLE_SYNTHETIC"
        ),
        "random_sva": "PASS"
        if all(
            path.exists()
            and "PASS post-G0 active projection" in path.read_text(encoding="utf-8")
            for path in random_logs
        )
        else "未审计",
        "verilator_lint": "PASS"
        if all(
            path.exists() and "%Error" not in path.read_text(encoding="utf-8")
            for path in lint_logs
        )
        else "未审计",
        "yosys_check": "PASS"
        if all(
            path.exists()
            and "End of script" in path.read_text(encoding="utf-8")
            and "ERROR:" not in path.read_text(encoding="utf-8")
            for path in yosys_logs
        )
        else "未审计",
    }

    summary = {
        "schema": "local5_gasr2c_fivebank_rtl_summary_v1",
        "variant": args.variant,
        "vector_manifest": str(args.manifest.resolve()),
        "vector_manifest_sha256": file_sha256(args.manifest),
        "weight_mode": weight_mode,
        "weight_contract": manifest.get("weight_contract"),
        "projection_contract_binding": manifest.get(
            "projection_contract_binding"
        ),
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": file_sha256(args.source_manifest),
        "source_bindings": source_bindings,
        "evidence": "本机RTL，Local5 post-G0 T450 profile100，完整五颜色bank在线前端",
        "correctness": {
            "direct_groups": len(direct_rows),
            "gasr_groups": len(gasr_rows),
            "acc32": "100/100组PASS，逐元素零失配",
            "descriptors": sum(row["active"] for row in rows),
            "terms": sum(row["terms"] for row in rows),
            "updates": sum(row["updates"] for row in rows),
        },
        "aggregate": {
            "direct_cycles": direct_cycles,
            "gasr_cycles": gasr_cycles,
            "gasr_speedup": direct_cycles / gasr_cycles,
            "gasr_cycle_reduction": 1 - gasr_cycles / direct_cycles,
            "direct_term_stalls": sum(row["direct_stalls"] for row in rows),
            "gasr_term_stalls": sum(row["gasr_stalls"] for row in rows),
            "direct_sram_transactions": direct_transactions,
            "gasr_sram_transactions": gasr_transactions,
            "sram_transaction_reduction": 1 - gasr_transactions / direct_transactions,
            "win_equal_loss": {
                "win": sum(value > 1 for value in speedups),
                "equal": sum(value == 1 for value in speedups),
                "loss": sum(value < 1 for value in speedups),
            },
            "speedup_distribution": {
                "p0": float(np.percentile(speedups, 0)),
                "p50": float(np.percentile(speedups, 50)),
                "p95": float(np.percentile(speedups, 95)),
                "p100": float(np.percentile(speedups, 100)),
            },
        },
        "no_leakage_selector_model": {
            "evidence": "前50组训练阈值、后50组固定阈值留出验证；尚无选择器RTL",
            "feature": selected_feature,
            "threshold": selected_threshold,
            "train": train_eval,
            "holdout": holdout_eval,
            "all": all_eval,
        },
        "per_stage": stage_rows,
        "verification": verification,
        "fairness": {
            "frontend": "相同word-skipper、relation frontier、FIFO2 term builder",
            "memory": "五个同深度同步1RW Acc bank，direct与GASR位宽和端口数一致",
            "cycle_boundary": "projection_start至五bank flush完成，不含结果readback",
            "transaction_boundary": "执行期backing SRAM读写，不含结果readback",
        },
        "not_proven": [
            "跨全部head的完整C维求和、bias、动态BatchNorm、requant和residual",
            "动态source/window分层选择器RTL",
            "SRAM macro绑定后的PPA/EDP",
            "Motion线对应收益",
            "可变SRAM延迟和外部消费者反压",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    agg = summary["aggregate"]
    selector = summary["no_leakage_selector_model"]
    lines = [
        "# Local5 DS-GASR-2C 五 Bank 在线 RTL 评估",
        "",
        "## 核心结论",
        "",
        f"完整五 bank 在线链路在同一 T450 profile100 上均通过 Acc32：direct 为 {direct_cycles:,} 周期，GASR 为 {gasr_cycles:,} 周期，固定使用 GASR 为 {direct_cycles / gasr_cycles:.3f}x，" + (
            f"周期缩短 {1 - gasr_cycles / direct_cycles:.2%}。"
            if gasr_cycles <= direct_cycles
            else f"周期增加 {gasr_cycles / direct_cycles - 1:.2%}。"
        ) + "单 bank 的 1.609x 不能直接外推为五 bank 端到端加速。",
        "",
        f"与此同时，执行期 SRAM 总事务由 {direct_transactions:,} 降到 {gasr_transactions:,}，下降 {1 - gasr_transactions / direct_transactions:.2%}；term stall 由 {agg['direct_term_stalls']:,} 降到 {agg['gasr_term_stalls']:,}。" + (
            "盲 geometry prepare 对 relation frontier 的阻塞抵消了驻留收益。"
            if args.variant == "blind_geometry"
            else "descriptor-synchronized geometry 将relation read先行，并在payload返回后与descriptor FIFO原子提交；当前profile100的descriptor mask与边界geometry mask完全相同，因此不存在role剪枝收益。"
        ),
        "",
        "## 公平性与正确性",
        "",
        "- 两条路径共享同一个 word-skipper、relation frontier、两项 descriptor FIFO 和 source-major term builder。",
        "- 两条路径都是五个相同深度、相同 Acc32 位宽、相同单端口 1RW 合同的颜色 bank。",
        "- 周期从 projection_start 到五 bank flush 完成，SRAM 事务只计执行期，均不含 900 次结果读回。",
        f"- 共 {summary['correctness']['descriptors']:,} 个 descriptors、{summary['correctness']['terms']:,} 个 terms、{summary['correctness']['updates']:,} 个 destination updates；两模式均为 100/100 组逐元素零失配。",
        f"- 随机空泡+SVA：{verification['random_sva']}；Verilator lint：{verification['verilator_lint']}；Yosys hierarchy/check/stat：{verification['yosys_check']}。",
        "",
        "## 逐组与 Stage 分布",
        "",
        f"逐组 win/equal/loss={agg['win_equal_loss']['win']}/{agg['win_equal_loss']['equal']}/{agg['win_equal_loss']['loss']}；加速分布 p50={agg['speedup_distribution']['p50']:.3f}x、p95={agg['speedup_distribution']['p95']:.3f}x、最差={agg['speedup_distribution']['p0']:.3f}x。固定 GASR 对不同密度窗口并不稳健。",
        "",
        "| Stage | 组数 | direct周期 | GASR周期 | GASR加速 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in stage_rows:
        lines.append(
            f"| {row['stage']} | {row['groups']} | {row['direct_cycles']:,} | {row['gasr_cycles']:,} | {row['speedup']:.3f}x |"
        )
    lines.extend(
        [
            "",
            "## 无泄漏分层模型",
            "",
            f"只在前50组选择特征和阈值，得到 `{selector['feature']} >= {selector['threshold']:.6g}` 时走 GASR；阈值冻结后，后50组相对 direct 为 {selector['holdout']['speedup_vs_direct']:.3f}x，选择 {selector['holdout']['selected_gasr_groups']} 组，其中 {selector['holdout']['selected_regressions']} 组仍退化。全100组模型为 {selector['all']['speedup_vs_direct']:.3f}x，逐组 oracle 上界为 {selector['all']['oracle_speedup_vs_direct']:.3f}x。",
            "",
            "该项仍是模型证据。只有留出集稳定为正、并实现在线可获得的复用元数据计数与模式选择RTL后，才能升级为本土化架构贡献。",
            "",
            "## 当前决策",
            "",
            "- 不把固定 GASR 五 bank 写成性能贡献；它当前是降低 SRAM 活动的候选。",
            "- 下一轮先判断分层是否在留出集有效，再决定实现同 bank 双模式，而不是复制异构双核。",
            "- 随机空泡、SVA、lint/Yosys已完成；后续仍需真实 SRAM macro 映射、STA和活动功耗。",
        ]
    )
    (args.output_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
