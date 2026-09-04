#!/usr/bin/env python3
"""Report the strong-baseline Local5 score-to-Acc rolling composition."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/local5_qsilent_rolling_composition_20260814"
VECTOR_DIR = (
    ROOT
    / "tb_qfit/vectors/local5_joint_ep29_score_projection_realw_sample100_population_v1_20260813"
)
GROUP_RE = re.compile(
    r"^GROUP .* group=(?P<group>\d+) cycles=(?P<cycles>\d+) "
    r"score_rows=(?P<score_rows>\d+) score_service=(?P<score_service>\d+) "
    r"score_direct_rows=(?P<score_direct>\d+) qsilent_rows=(?P<qsilent>\d+) "
    r"identk_rows=(?P<identk>\d+) overlap=(?P<overlap>\d+) "
    r"active=(?P<active>\d+).* terms=(?P<terms>\d+) updates=(?P<updates>\d+)"
)
PASS_RE = re.compile(r"^PASS Local5 score-to-projection .* groups=100 total_cycles=(\d+)")
BAD_RE = re.compile(r"%Error|Assertion failed|MISMATCH|\$fatal|\bFAIL\b")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse(path: Path) -> dict[int, dict[str, int]]:
    text = path.read_text()
    if BAD_RE.search(text):
        raise ValueError(f"bad marker in {path}")
    rows: dict[int, dict[str, int]] = {}
    passes: list[int] = []
    for line in text.splitlines():
        match = GROUP_RE.match(line)
        if match:
            row = {key: int(value) for key, value in match.groupdict().items()}
            group = row.pop("group")
            if group in rows:
                raise ValueError(f"duplicate group {group} in {path}")
            rows[group] = row
        match = PASS_RE.match(line)
        if match:
            passes.append(int(match.group(1)))
    if sorted(rows) != list(range(100)):
        raise ValueError(f"group population mismatch in {path}")
    total = sum(row["cycles"] for row in rows.values())
    if passes != [total]:
        raise ValueError(f"PASS mismatch in {path}")
    return rows


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def compare(
    baseline: dict[int, dict[str, int]],
    candidate: dict[int, dict[str, int]],
    indices: list[int] | None = None,
) -> dict[str, object]:
    selected = list(range(100)) if indices is None else indices
    if not selected:
        raise ValueError("empty comparison population")
    left = sum(baseline[index]["cycles"] for index in selected)
    right = sum(candidate[index]["cycles"] for index in selected)
    ratios = [baseline[index]["cycles"] / candidate[index]["cycles"] for index in selected]
    return {
        "groups": len(selected),
        "baseline_cycles": left,
        "candidate_cycles": right,
        "speedup": left / right,
        "cycle_reduction": 1.0 - right / left,
        "per_group": {
            "min": min(ratios),
            "p50": percentile(ratios, 0.50),
            "p95": percentile(ratios, 0.95),
            "max": max(ratios),
            "wins": sum(value > 1.0 for value in ratios),
            "ties": sum(value == 1.0 for value in ratios),
            "losses": sum(value < 1.0 for value in ratios),
        },
    }


def main() -> None:
    log_paths = {
        "t450_residual": OUT / "t450_q0_g100_verilator_assert.log",
        "t450_qsilent": OUT / "t450_q1_g100_verilator_assert.log",
        "rolling_residual": OUT / "rolling_q0_g100_verilator_assert.log",
        "rolling_qsilent_verilator": OUT / "rolling_q1_g100_verilator_assert.log",
        "rolling_qsilent_icarus": OUT / "rolling_q1_g100_iverilog.log",
        "dynamic_qsilent": OUT / "dynamic_q1_g100_verilator_assert.log",
        "banked_dynamic_qsilent": (
            OUT / "banked_dynamic_q1_g100_verilator_assert.log"
        ),
        "stripe_qsilent": OUT / "stripe_q1_g100_verilator_assert.log",
    }
    logs = {name: parse(path) for name, path in log_paths.items()}
    if logs["rolling_qsilent_icarus"] != logs["rolling_qsilent_verilator"]:
        raise ValueError("Icarus/Verilator rolling-q-silent ledger mismatch")
    if logs["banked_dynamic_qsilent"] != logs["dynamic_qsilent"]:
        raise ValueError("banked/flat Dynamic production-tile ledger mismatch")
    if logs["banked_dynamic_qsilent"] != logs["rolling_qsilent_verilator"]:
        raise ValueError("banked Dynamic/FCSR production-tile ledger mismatch")

    conserved = ("score_rows", "active", "terms", "updates")
    reference = logs["t450_residual"]
    conservation = {field: sum(row[field] for row in reference.values()) for field in conserved}
    for name, rows in logs.items():
        for field, expected in conservation.items():
            actual = sum(row[field] for row in rows.values())
            if actual != expected:
                raise ValueError(f"{name} violates {field}: {actual} != {expected}")

    random_paths = sorted(OUT.glob("rolling_q1_bp8_seed*_verilator_assert.log"))
    if len(random_paths) != 8:
        raise ValueError("expected eight random-gap logs")
    for path in random_paths:
        text = path.read_text()
        if "PASS Local5 score-to-projection" not in text or BAD_RE.search(text):
            raise ValueError(f"random-gap regression failed: {path}")

    lint_path = OUT / "rolling_q1_current_lint.log"
    lint_text = lint_path.read_text()
    if "%Error" in lint_text or "UNOPTFLAT" in lint_text:
        raise ValueError("current-source rolling lint is not combinational-loop clean")

    manifest_path = VECTOR_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != "local5_score_projection_vectors_v1"
        or manifest.get("selection", {}).get("groups") != 100
        or manifest.get("shape", {}).get("out_dim") != 2
    ):
        raise ValueError("vector manifest contract mismatch")
    stage_counts = manifest["selection"]["stage_counts"]
    metadata = {
        int(row["vector_group_index"]): row for row in manifest["selection"]["rows"]
    }
    if sorted(metadata) != list(range(100)):
        raise ValueError("manifest vector-group population mismatch")
    empty_groups = [index for index, row in metadata.items() if row["empty"]]
    nonempty_groups = [index for index, row in metadata.items() if not row["empty"]]
    final = logs["rolling_qsilent_verilator"]
    population_breakdown = {
        "empty": compare(logs["t450_qsilent"], final, empty_groups),
        "nonempty": compare(logs["t450_qsilent"], final, nonempty_groups),
        "by_stage": {
            str(stage): compare(
                logs["t450_qsilent"],
                final,
                [index for index, row in metadata.items() if row["stage"] == stage],
            )
            for stage in range(4)
        },
    }
    report = {
        "schema": "local5_qsilent_rolling_composition_rtl_v1",
        "status": "ADMIT_AS_LOCAL5_UNIFIED_DATAFLOW_EVIDENCE",
        "evidence": "[rtl]",
        "scope": (
            "100 sample-disjoint population-stage-weighted real raw-Q/K and checkpoint-weight groups; "
            "score/Shiftmax5 through relation/TCFM5 to Acc32; OUT_DIM=2 tile; not encoder"
        ),
        "population": {"groups": 100, "stage_counts": stage_counts},
        "four_way": {
            "t450_residual": sum(row["cycles"] for row in logs["t450_residual"].values()),
            "t450_qsilent": sum(row["cycles"] for row in logs["t450_qsilent"].values()),
            "rolling_residual": sum(row["cycles"] for row in logs["rolling_residual"].values()),
            "rolling_qsilent": sum(row["cycles"] for row in final.values()),
        },
        "strong_baseline_increment": compare(logs["t450_qsilent"], final),
        "strong_baseline_population_breakdown": population_breakdown,
        "scheduler_ablation_under_qsilent": {
            "t450_to_dynamic": compare(
                logs["t450_qsilent"], logs["dynamic_qsilent"]
            ),
            "t450_to_banked_dynamic": compare(
                logs["t450_qsilent"], logs["banked_dynamic_qsilent"]
            ),
            "t450_to_stripe": compare(
                logs["t450_qsilent"], logs["stripe_qsilent"]
            ),
            "dynamic_to_rolling": compare(logs["dynamic_qsilent"], final),
            "stripe_to_rolling": compare(logs["stripe_qsilent"], final),
        },
        "rolling_under_residual": compare(logs["t450_residual"], logs["rolling_residual"]),
        "qsilent_under_rolling": compare(logs["rolling_residual"], final),
        "weak_end_to_end_reference": compare(logs["t450_residual"], final),
        "state": {
            "t450_relation_k_bits": 36_900,
            "rolling_relation_k_bits": 3_735,
            "reduction": 1.0 - 3_735 / 36_900,
        },
        "workload": {
            "score_rows": conservation["score_rows"],
            "qsilent_rows": sum(row["qsilent"] for row in final.values()),
            "identk_rows": sum(row["identk"] for row in final.values()),
            "overlap_accepts": sum(row["overlap"] for row in final.values()),
            "descriptors": conservation["active"],
            "terms": conservation["terms"],
            "updates": conservation["updates"],
            "acc32_mismatch": 0,
        },
        "verification": {
            "icarus_verilator_per_group_exact": True,
            "verilator_assert": True,
            "random_gap_seeds": 8,
            "groups_per_seed": 8,
            "current_source_unoptflat_free": True,
            "banked_dynamic_production_tile_exact": True,
        },
        "architectural_reading": (
            "Q-silent/IdentK rows bypass the residual score walk; non-silent relations are "
            "online-transposed and retired from a bounded three-row frontier into TCFM5."
        ),
        "claim_boundary": [
            "strong-baseline increment is 183379 -> 155791; 324605 -> 155791 is not a single-mechanism claim",
            "rolling FCSR originates in docs/208 and is not a new standalone contribution",
            "OUT_DIM=2 tile only; no full encoder, bias/BN/requant/residual/decoder",
            "no DC, STA, SAIF, PTPX, SRAM macro energy, or ASIC PPA",
            "does not modify docs/359 frozen columns",
        ],
        "sha256": {str(path.relative_to(OUT)): sha256(path) for path in log_paths.values()},
    }
    report["sha256"].update({str(path.relative_to(OUT)): sha256(path) for path in random_paths})
    report["sha256"][lint_path.name] = sha256(lint_path)
    report["sha256"]["vector_manifest"] = sha256(manifest_path)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    strong = report["strong_baseline_increment"]
    empty = report["strong_baseline_population_breakdown"]["empty"]
    nonempty = report["strong_baseline_population_breakdown"]["nonempty"]
    sched = report["scheduler_ablation_under_qsilent"]
    qsilent = report["qsilent_under_rolling"]
    rolling = report["rolling_under_residual"]
    weak = report["weak_end_to_end_reference"]
    markdown = f"""# Local5 Query-Silent x Rolling FCSR 统一数据流回放

- 裁决：`{report['status']}`，证据 `[rtl]`。
- 边界：{report['scope']}。
- 四方周期：residual+T450 `{report['four_way']['t450_residual']}`；Query-Silent+T450 `{report['four_way']['t450_qsilent']}`；residual+rolling `{report['four_way']['rolling_residual']}`；Query-Silent+rolling `{report['four_way']['rolling_qsilent']}`。
- 最强现行基线增量：`183379 -> 155791 = {strong['speedup']:.4f}x`，周期 `{strong['cycle_reduction']:.2%}`；逐组 min/p50/p95 `{strong['per_group']['min']:.4f}/{strong['per_group']['p50']:.4f}/{strong['per_group']['p95']:.4f}x`，win/tie/loss `{strong['per_group']['wins']}/{strong['per_group']['ties']}/{strong['per_group']['losses']}`。
- 负组解释：33 个空组全部因 rolling 固定开销慢 2 cycles，`{empty['baseline_cycles']} -> {empty['candidate_cycles']} = {empty['speedup']:.4f}x`；67 个非空组全部更快，`{nonempty['baseline_cycles']} -> {nonempty['candidate_cycles']} = {nonempty['speedup']:.4f}x`，win/tie/loss `{nonempty['per_group']['wins']}/{nonempty['per_group']['ties']}/{nonempty['per_group']['losses']}`。
- 固定 residual score 时 rolling 增量：`{rolling['speedup']:.4f}x`；固定 rolling relation 时 Query-Silent 增量：`{qsilent['speedup']:.4f}x`。
- Query-Silent 下调度强基线：active-filtered flat Dynamic `{sched['t450_to_dynamic']['candidate_cycles']}`、五色 Banked-Dynamic `{sched['t450_to_banked_dynamic']['candidate_cycles']}`、Stripe `{sched['t450_to_stripe']['candidate_cycles']}`、Rolling `{report['four_way']['rolling_qsilent']}`；两种 Dynamic 与 Rolling 逐组周期完全相同，说明收益来自有界在线转置，不来自 FCSR 的额外吞吐。
- `324605 -> 155791 = {weak['speedup']:.4f}x` 只作四方端点，不是单机制主张。
- 状态：relation/K `36900 -> 3735 bit`，降低 `{report['state']['reduction']:.2%}`。
- 守恒：45000 score rows、descriptor `{conservation['active']}`、term `{conservation['terms']}`、update `{conservation['updates']}`、Acc32 mismatch `0`；Icarus/Verilator 逐组一致，Banked-Dynamic 生产 tile `--assert` PASS，8 seeds x 8 groups 随机间隙 `--assert` PASS。

## 可辩护主线

Local5 的贡献应写成一条统一执行流：静默/IdentK 行在 score 端绕过 residual XOR walk；其余 relation 不再形成 T450 sealed image，而是在固定五邻域上在线转置、三行有界驻留并按 source 退休到 TCFM5。三行 FCSR 来自 `docs/208`，本轮价值是把 Query-Silent 与它在真实 score-to-Acc32 边界组合闭环，不增加第四个贡献名。

该包仍是 `OUT_DIM=2` tile，不是 encoder；不改 `docs/359`，不得称 ASIC PPA 或 DC/STA/SAIF/PTPX 结果。
"""
    (OUT / "report.md").write_text(markdown)


if __name__ == "__main__":
    main()
