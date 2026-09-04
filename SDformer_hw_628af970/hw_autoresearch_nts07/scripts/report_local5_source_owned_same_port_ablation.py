#!/usr/bin/env python3
"""Report Local5 source-owned execution under 1R1W and legal 1RW Acc32."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


GROUP_RE = re.compile(
    r"^GROUP .* group=(?P<group>\d+) cycles=(?P<cycles>\d+) "
    r"score_rows=(?P<score_rows>\d+) score_service=(?P<score_service>\d+) "
    r"score_direct_rows=(?P<score_direct>\d+) qsilent_rows=(?P<qsilent>\d+) "
    r"identk_rows=(?P<identk>\d+) overlap=(?P<overlap>\d+) "
    r"active=(?P<active>\d+) memory_wait=(?P<memory_wait>\d+) "
    r"terms=(?P<terms>\d+) updates=(?P<updates>\d+)"
)
PASS_RE = re.compile(
    r"^PASS Local5 score-to-projection .* groups=100 total_cycles=(?P<cycles>\d+)"
)
BAD_RE = re.compile(r"%Error|Assertion failed|MISMATCH|\$fatal|\bFAIL\b")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_log(path: Path) -> dict[int, dict[str, int]]:
    text = path.read_text(encoding="utf-8")
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
                raise ValueError(f"duplicate group {group}")
            rows[group] = row
        match = PASS_RE.match(line)
        if match:
            passes.append(int(match.group("cycles")))
    if sorted(rows) != list(range(100)):
        raise ValueError("log does not contain the sealed 100 groups")
    total = sum(row["cycles"] for row in rows.values())
    if passes != [total]:
        raise ValueError("PASS cycle total mismatch")
    return rows


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def cycle_subset(
    one_r_one_w: dict[int, dict[str, int]],
    one_rw: dict[int, dict[str, int]],
    groups: list[int],
) -> dict[str, Any]:
    left = sum(one_r_one_w[group]["cycles"] for group in groups)
    right = sum(one_rw[group]["cycles"] for group in groups)
    ratios = [
        one_rw[group]["cycles"] / one_r_one_w[group]["cycles"]
        for group in groups
    ]
    return {
        "groups": len(groups),
        "one_r_one_w_cycles": left,
        "one_rw_cycles": right,
        "one_rw_over_one_r_one_w": right / left,
        "one_rw_slower": sum(value > 1.0 for value in ratios),
        "tie": sum(value == 1.0 for value in ratios),
        "one_rw_faster": sum(value < 1.0 for value in ratios),
    }


def activity(path: Path, *, design: str, busy: int) -> dict[str, Any]:
    row = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "status_pass": row.get("status") == "PASS",
        "design": row.get("design_name") == design,
        "paper_power_eligible": row.get("paper_power_eligible") is True,
        "scope": row.get("measurement_scope") == "full_load_compute_readback",
        "purpose": row.get("activity_purpose") == "paper_power_with_io",
        "busy": row.get("busy_cycles") == busy,
        "single_interval": row.get("vcd_active_intervals") == 1,
    }
    if not all(checks.values()):
        raise ValueError(f"activity contract failed: {path}: {checks}")
    return row


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--one-r-one-w-log",
        type=Path,
        default=(
            root
            / "results/local5_source_owned_gate_quotient_current_rtl_20260814/"
            "verilator_assert.log"
        ),
    )
    parser.add_argument(
        "--one-rw-log",
        type=Path,
        default=(
            root
            / "results/local5_source_owned_1rw_population_20260814/"
            "verilator_assert.log"
        ),
    )
    parser.add_argument(
        "--one-r-one-w-activity",
        type=Path,
        default=(
            root
            / "dc_handoff/runs/local5_dc_activity_full_population100/"
            "activity_contract.json"
        ),
    )
    parser.add_argument(
        "--one-rw-activity",
        type=Path,
        default=(
            root
            / "dc_handoff/runs/local5_1rw_activity_population100_full/"
            "activity_contract.json"
        ),
    )
    parser.add_argument(
        "--rejected-busy-activity",
        type=Path,
        default=(
            root
            / "dc_handoff/runs/local5_1rw_activity_population100_busy/"
            "activity_contract.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results/local5_source_owned_same_port_ablation_v3_20260814",
    )
    args = parser.parse_args()

    one_r_one_w = parse_log(args.one_r_one_w_log)
    one_rw = parse_log(args.one_rw_log)
    conserved_fields = (
        "score_rows",
        "score_service",
        "score_direct",
        "qsilent",
        "identk",
        "overlap",
        "active",
        "terms",
        "updates",
    )
    mismatch: dict[str, int] = {}
    for field in conserved_fields:
        mismatch[field] = sum(
            one_r_one_w[group][field] != one_rw[group][field]
            for group in range(100)
        )
    if any(mismatch.values()):
        raise AssertionError(f"1R1W/1RW workload drift: {mismatch}")

    cycles_1r1w = sum(row["cycles"] for row in one_r_one_w.values())
    cycles_1rw = sum(row["cycles"] for row in one_rw.values())
    if cycles_1r1w != 155_791 or cycles_1rw != 170_269:
        raise AssertionError("sealed same-port cycle ledger drift")
    ratios = [
        one_rw[group]["cycles"] / one_r_one_w[group]["cycles"]
        for group in range(100)
    ]
    empty_groups = [group for group in range(100) if one_rw[group]["terms"] == 0]
    nonempty_groups = [group for group in range(100) if one_rw[group]["terms"] != 0]

    activity_1r1w = activity(
        args.one_r_one_w_activity,
        design="local5_unified_out2_dc_top",
        busy=cycles_1r1w,
    )
    activity_1rw = activity(
        args.one_rw_activity,
        design="local5_unified_out2_1rw_dc_top",
        busy=cycles_1rw,
    )
    if activity_1r1w["trace_sha256"] != activity_1rw["trace_sha256"]:
        raise AssertionError("activity contracts do not bind the same vector population")

    rejected = json.loads(args.rejected_busy_activity.read_text(encoding="utf-8"))
    rejected_checks = {
        "status_fail": rejected.get("status") == "FAIL",
        "not_paper_power": rejected.get("paper_power_eligible") is False,
        "busy_matches": rejected.get("busy_cycles") == cycles_1rw,
        "hundred_intervals": rejected.get("vcd_active_intervals") == 100,
    }
    if not all(rejected_checks.values()):
        raise AssertionError(f"rejected busy artifact drift: {rejected_checks}")

    terms = sum(row["terms"] for row in one_rw.values())
    updates = sum(row["updates"] for row in one_rw.values())
    report = {
        "schema": "local5_source_owned_same_port_ablation_v1",
        "status": "RTL_AND_ACTIVITY_READY_POWER_PENDING",
        "evidence": ["[rtl]", "[activity]", "[待验证:SAIF/DC/PTPX]"],
        "scope": (
            "same 100 sample-disjoint population-stage-weighted groups; OUT_DIM=2 "
            "score-to-Acc32 tile; not encoder"
        ),
        "workload_conservation": {
            "per_group_mismatch": mismatch,
            "terms": terms,
            "updates": updates,
            "acc32_mismatch": 0,
            "builder_backend_sva": True,
        },
        "cycle_ablation": {
            "one_r_one_w": cycles_1r1w,
            "one_rw": cycles_1rw,
            "one_rw_over_one_r_one_w": cycles_1rw / cycles_1r1w,
            "one_rw_cycle_overhead": cycles_1rw / cycles_1r1w - 1.0,
            "per_group_ratio": {
                "min": min(ratios),
                "p50": percentile(ratios, 0.50),
                "p95": percentile(ratios, 0.95),
                "max": max(ratios),
                "one_rw_slower": sum(value > 1.0 for value in ratios),
                "tie": sum(value == 1.0 for value in ratios),
                "one_rw_faster": sum(value < 1.0 for value in ratios),
            },
            "empty": cycle_subset(one_r_one_w, one_rw, empty_groups),
            "nonempty": cycle_subset(one_r_one_w, one_rw, nonempty_groups),
            "memory_wait_cycles": {
                "one_r_one_w": sum(row["memory_wait"] for row in one_r_one_w.values()),
                "one_rw": sum(row["memory_wait"] for row in one_rw.values()),
            },
        },
        "matched_full_window_activity": {
            "trace_sha256": activity_1r1w["trace_sha256"],
            "one_r_one_w": {
                "measured_cycles": activity_1r1w["measured_cycles"],
                "busy_cycles": activity_1r1w["busy_cycles"],
                "overhead_cycles": activity_1r1w["measurement_overhead_cycles"],
                "vcd_sha256": activity_1r1w["source_vcd_sha256"],
                "paper_power_eligible": True,
            },
            "one_rw": {
                "measured_cycles": activity_1rw["measured_cycles"],
                "busy_cycles": activity_1rw["busy_cycles"],
                "overhead_cycles": activity_1rw["measurement_overhead_cycles"],
                "vcd_sha256": activity_1rw["source_vcd_sha256"],
                "paper_power_eligible": True,
            },
            "power_or_energy_available": False,
        },
        "rejected_busy_activity": {
            "status": "REJECTED_AS_MULTI_INTERVAL_SAIF_INPUT",
            "vcd_intervals": 100,
            "paper_power_eligible": False,
            "checks": rejected_checks,
        },
        "claim_boundary": [
            "Cycle overhead is measured RTL; it is not energy or PPA.",
            "VCD contracts are matched inputs for later SAIF/PTPX and contain no power number.",
            "Full-window activity includes weight load, compute, Acc readback, and inter-group reset.",
            "The 100-interval busy VCD is explicitly rejected rather than time-compressed.",
            "OUT_DIM=2 tile only; not full encoder and no modification to docs/359.",
        ],
        "sha256": {
            "one_r_one_w_log": sha256(args.one_r_one_w_log),
            "one_rw_log": sha256(args.one_rw_log),
            "one_r_one_w_activity": sha256(args.one_r_one_w_activity),
            "one_rw_activity": sha256(args.one_rw_activity),
            "rejected_busy_activity": sha256(args.rejected_busy_activity),
            "one_rw_wrapper": sha256(
                root / "dc_handoff/rtl/date_local5_1rw_dc_top.sv"
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    cycle = report["cycle_ablation"]
    markdown = f"""# Local5 source-owned 同端口消融

- 裁决：`{report['status']}`。
- 边界：{report['scope']}。
- 逐组守恒：score/QS/IdentK/active/term/update 100/100 一致；term `{terms}`、update `{updates}`、Acc32 mismatch `0`，builder/backend bind SVA PASS。
- 周期 `[rtl]`：1R1W `{cycles_1r1w}`；合法单端口 1RW `{cycles_1rw}`，1RW 开销 `{cycle['one_rw_cycle_overhead']:.2%}`；逐组 1RW slower/tie/faster `{cycle['per_group_ratio']['one_rw_slower']}/{cycle['per_group_ratio']['tie']}/{cycle['per_group_ratio']['one_rw_faster']}`。
- 分布：空组 `{cycle['empty']['groups']}` 组，`{cycle['empty']['one_r_one_w_cycles']} -> {cycle['empty']['one_rw_cycles']}`，基本持平；非空组 `{cycle['nonempty']['groups']}` 组，`{cycle['nonempty']['one_r_one_w_cycles']} -> {cycle['nonempty']['one_rw_cycles']}`。memory-wait `{cycle['memory_wait_cycles']['one_r_one_w']} -> {cycle['memory_wait_cycles']['one_rw']}`，说明非空组的单端口 RMW stall 主导总开销。
- matched full-window VCD `[activity]`：1R1W measured `{activity_1r1w['measured_cycles']}`，1RW measured `{activity_1rw['measured_cycles']}`；同一 trace SHA、单活动区间、两边 `paper_power_eligible=true`。
- 被拒绝的 compute-only 尝试：100 次 dumpon/dumpoff 形成 100 个 interval，不能作为单一 population SAIF，保留 `paper_power_eligible=false`。

目前没有 SAIF/DC/PTPX 数字，不得把 VCD、周期或 measured-window 差异写成能量/PPA。该包只证明 source-owned Local5 在合法 1RW Acc32 下 bit-exact，并量化单端口 RMW 周期代价；仍是 `OUT_DIM=2` tile，不是 encoder。
"""
    (args.output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"status": report["status"], "cycle_ablation": cycle}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
