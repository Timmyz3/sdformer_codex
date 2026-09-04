#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


PASS_RE = re.compile(
    r"^PASS tb_h67_denominator_certificate rows=(\d+) gates=(\d+) errors=0$",
    re.MULTILINE,
)
BAD_RE = re.compile(r"\b(?:ERROR|FATAL):|\bFAIL\b", re.IGNORECASE)


def require_pass(path: Path) -> tuple[int, int]:
    text = path.read_text(errors="replace")
    if BAD_RE.search(text):
        raise ValueError(f"failure marker in {path}")
    match = PASS_RE.search(text)
    if match is None:
        raise ValueError(f"missing exact PASS marker in {path}")
    return int(match.group(1)), int(match.group(2))


def parse_top_area(path: Path, top: str) -> float:
    text = path.read_text(errors="replace")
    if BAD_RE.search(text):
        raise ValueError(f"failure marker in {path}")
    top_pattern = re.compile(
        rf"Chip area for top module '\\{re.escape(top)}':\s+([0-9.]+)"
    )
    module_pattern = re.compile(
        rf"Chip area for module '\\{re.escape(top)}':\s+([0-9.]+)"
    )
    top_matches = top_pattern.findall(text)
    module_matches = module_pattern.findall(text)
    if len(top_matches) == 1:
        return float(top_matches[0])
    if not top_matches and len(module_matches) == 1:
        return float(module_matches[0])
    raise ValueError(
        f"ambiguous top area for {top} in {path}: "
        f"top={top_matches}, module={module_matches}"
    )


def parse_sta(path: Path) -> dict[str, float | str]:
    text = path.read_text(errors="replace")
    if BAD_RE.search(text):
        raise ValueError(f"failure marker in {path}")
    arrivals = re.findall(r"^\s*([0-9.]+)\s+data arrival time\s*$", text, re.MULTILINE)
    slacks = re.findall(
        r"^\s*(-?[0-9.]+)\s+slack \((MET|VIOLATED)\)\s*$", text, re.MULTILINE
    )
    if len(arrivals) != 1 or len(slacks) != 1:
        raise ValueError(f"ambiguous STA summary in {path}")
    return {
        "data_arrival_ns": float(arrivals[0]),
        "slack_ns": float(slacks[0][0]),
        "status": slacks[0][1],
    }


def parse_macro_area(path: Path) -> float:
    text = path.read_text(errors="replace")
    match = re.search(
        r"cell\(fakeram45_256x16\)\s*\{.*?\barea\s*:\s*([0-9.]+)\s*;",
        text,
        re.DOTALL,
    )
    if match is None:
        raise ValueError(f"missing fakeram45_256x16 area in {path}")
    return float(match.group(1))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(output_dir: Path, profile_path: Path, macro_lib: Path) -> dict:
    logs = output_dir / "logs"
    sim_results = {
        simulator: require_pass(logs / f"{simulator}.log")
        for simulator in ("iverilog", "verilator")
    }
    if len(set(sim_results.values())) != 1:
        raise ValueError(f"simulator result mismatch: {sim_results}")
    rows, gates = sim_results["iverilog"]

    tops = (
        "h67_row_qmax_denominator_certificate_core",
        "h67_row_qmax_denominator_certificate",
        "h67_row_qkm_denominator_certificate_core",
        "h67_row_qkm_denominator_certificate",
        "h67_row_qkm_denominator_certificate_reuse_qcounts",
        "ttx_gate_quant_q17",
        "h67_certified_gate_quant_q17",
    )
    mapped = {
        top: {
            "area": parse_top_area(logs / f"nangate45_{top}.log", top),
            "sta": parse_sta(logs / f"sta_{top}.log"),
        }
        for top in tops
    }

    profile = json.loads(profile_path.read_text())
    if profile.get("status") != "FROZEN_LEAF_ONLY_NO_DIRECTORY":
        raise ValueError(f"unexpected profile status in {profile_path}")
    fair = profile["fair_ep35_sample0_window0"]
    population = profile["profile100"]
    if fair["baseline_gate_mismatch"] or fair["forced17_gate_mismatch"]:
        raise ValueError("profile gate mismatch is nonzero")

    macro_area = parse_macro_area(macro_lib)
    current_two_hist_area = 2.0 * macro_area
    standalone_candidate_area = (
        macro_area + mapped["h67_row_qmax_denominator_certificate"]["area"]
    )
    shared_count_candidate_area = (
        macro_area + mapped["h67_row_qmax_denominator_certificate_core"]["area"]
    )
    qkm_standalone_candidate_area = (
        macro_area + mapped["h67_row_qkm_denominator_certificate"]["area"]
    )
    qkm_shared_count_candidate_area = (
        macro_area + mapped["h67_row_qkm_denominator_certificate_core"]["area"]
    )
    qkm_reuse_qcount_candidate_area = (
        macro_area
        + mapped["h67_row_qkm_denominator_certificate_reuse_qcounts"]["area"]
    )

    report = {
        "schema": "h67_denominator_certificate_rtl_screen_v1",
        "status": "FROZEN_LEAF_ONLY_NO_DIRECTORY",
        "evidence": "[rtl]+[prof]+[开放逻辑映射代理]+[开放STA代理]",
        "scope": {
            "rtl": "standalone load-time certificate and gate-select miter",
            "profile": "frozen fair sample0/window0 plus profile100",
            "not_claimed": [
                "shared fallback histogram RTL",
                "fail-fail backpressure closure",
                "SAIF energy",
                "target-process ASIC PPA",
                "full encoder speedup",
            ],
        },
        "rtl_miter": {
            "rows": rows,
            "gates": gates,
            "simulators": list(sim_results),
            "gate_mismatch": 0,
            "adversarial_cases": [
                "qmax=15 score=93 certificate pass",
                "qmax=16 score>=97 certificate fail",
                "QKM upper bound never falls below actual row max",
                "short row fail-closed",
                "out-of-order pair fail-closed",
                "deterministic load gaps",
            ],
        },
        "profile_anchor": {
            "fair_static_pass_rows": fair["static_pass_rows"],
            "fair_rows": fair["rows"],
            "profile100_static_pass_rows": population["static_pass_rows"],
            "fair_qkm_pass_rows": fair["qkm_pass_rows"],
            "profile100_qkm_pass_rows": population["qkm_pass_rows"],
            "profile100_rows": population["rows"],
            "hist_update_reduction": population["hist_update_reduction"],
            "class_scan_reduction": population["class_scan_reduction"],
            "qkm_hist_update_reduction": population["qkm_hist_update_reduction"],
            "qkm_class_scan_reduction": population["qkm_class_scan_reduction"],
            "claim": "transaction counts only; not energy",
        },
        "open_mapping_proxy": {
            "library": str(macro_lib),
            "macro_area_fakeram45_256x16": macro_area,
            "mapped": mapped,
            "histogram_subsystem_area_model": {
                "current_two_macros": current_two_hist_area,
                "one_macro_plus_standalone_certificate": standalone_candidate_area,
                "one_macro_plus_shared_qcount_core": shared_count_candidate_area,
                "one_macro_plus_qkm_standalone_certificate": (
                    qkm_standalone_candidate_area
                ),
                "one_macro_plus_qkm_count_core": qkm_shared_count_candidate_area,
                "one_macro_plus_qkm_reuse_qcounts": qkm_reuse_qcount_candidate_area,
                "reduction_current_over_standalone": (
                    current_two_hist_area / standalone_candidate_area
                ),
                "reduction_current_over_shared_qcount": (
                    current_two_hist_area / shared_count_candidate_area
                ),
                "reduction_current_over_qkm_standalone": (
                    current_two_hist_area / qkm_standalone_candidate_area
                ),
                "reduction_current_over_qkm_count_core": (
                    current_two_hist_area / qkm_shared_count_candidate_area
                ),
                "reduction_current_over_qkm_reuse_qcounts": (
                    current_two_hist_area / qkm_reuse_qcount_candidate_area
                ),
                "boundary": (
                    "open Nangate45 macro+logic area model; directory control, "
                    "routing, target SRAM, and activity are not included"
                ),
            },
            "gate_area_ratio_candidate_over_baseline": (
                mapped["h67_certified_gate_quant_q17"]["area"]
                / mapped["ttx_gate_quant_q17"]["area"]
            ),
        },
        "decision": {
            "architecture": "FROZEN_SUPPORT_ONLY",
            "novelty": "supporting exact fast path, not an independent DATE contribution",
            "next_gate": (
                "none; do not integrate a shared histogram or modify the frozen "
                "LAWS top; retain as a default-off RQTB support leaf"
            ),
        },
        "identity": {
            "profile_path": str(profile_path),
            "profile_sha256": sha256(profile_path),
            "macro_lib_sha256": sha256(macro_lib),
        },
    }
    return report


def render_markdown(report: dict) -> str:
    miter = report["rtl_miter"]
    anchor = report["profile_anchor"]
    proxy = report["open_mapping_proxy"]
    area = proxy["histogram_subsystem_area_model"]
    mapped = proxy["mapped"]
    return f"""# Motion Load-Time Denominator Certificate RTL Go/No-Go

## 裁决

`{report['status']}`

证书 leaf 与 gate-select 已有 `[rtl]` 证据，但共享 fallback histogram 不再接入 LAWS。该方向冻结为 RQTB 的 default-off 支撑，不是新的独立 DATE 贡献，也不能进入冻结性能主表。

## Exact RTL

- Icarus/Verilator `--assert`：`{miter['rows']}` 行、`{miter['gates']}` 个 gate，mismatch `0`。
- 对抗边界：`qmax=15` 可构造 score `93` 并命中；`qmax=16` 可构造 score `>=97`，证书必须失败并回到原 `ceil_log2(row_sum)`。
- 短行、乱序 pair 均 fail-closed；load gap 不影响 pair ledger。

## Profile Anchor

- 公平包静态证书：`{anchor['fair_static_pass_rows']}/{anchor['fair_rows']}` 行。
- 公平包 Q/K/motion 上界证书：`{anchor['fair_qkm_pass_rows']}/{anchor['fair_rows']}` 行。
- profile100：`{anchor['profile100_static_pass_rows']}/{anchor['profile100_rows']}` 行。
- profile100 Q/K/motion 上界证书：`{anchor['profile100_qkm_pass_rows']}/{anchor['profile100_rows']}` 行。
- histogram update / class scan 事务减少分别为 `{anchor['hist_update_reduction']:.2%}` / `{anchor['class_scan_reduction']:.2%}`；它们不是能量数字。
- Q/K/motion 证书在该 profile 的对应事务覆盖为 `{anchor['qkm_hist_update_reduction']:.2%}` / `{anchor['qkm_class_scan_reduction']:.2%}`；仍不是能量。

## Open Proxy

- certificate core（复用上游 Q-popcount）：面积 `{mapped['h67_row_qmax_denominator_certificate_core']['area']:.3f}`，arrival `{mapped['h67_row_qmax_denominator_certificate_core']['sta']['data_arrival_ns']:.6f} ns`。
- standalone certificate（含两棵 popcount32）：面积 `{mapped['h67_row_qmax_denominator_certificate']['area']:.3f}`，arrival `{mapped['h67_row_qmax_denominator_certificate']['sta']['data_arrival_ns']:.6f} ns`。
- Q/K/motion core（复用 Q/K/motion counts）：面积 `{mapped['h67_row_qkm_denominator_certificate_core']['area']:.3f}`；standalone 五棵 popcount32 版本 `{mapped['h67_row_qkm_denominator_certificate']['area']:.3f}`。
- 现实接点（复用已有两路 Q-count、新增 K0/K1/motion 三棵树）：面积 `{mapped['h67_row_qkm_denominator_certificate_reuse_qcounts']['area']:.3f}`。
- 一份开放 `fakeram45_256x16` 面积代理 `{proxy['macro_area_fakeram45_256x16']:.3f}`。双 histogram `{area['current_two_macros']:.3f}`；一 macro + standalone certificate `{area['one_macro_plus_standalone_certificate']:.3f}`，比值 `{area['reduction_current_over_standalone']:.3f}x`；若复用 qcount 则 `{area['reduction_current_over_shared_qcount']:.3f}x`。
- Q/K/motion standalone 对应局部比值 `{area['reduction_current_over_qkm_standalone']:.3f}x`；若五个 count 均由上游复用则 `{area['reduction_current_over_qkm_count_core']:.3f}x`。
- 复用已有 Q-count 的现实局部比值为 `{area['reduction_current_over_qkm_reuse_qcounts']:.3f}x`。
- certified gate 相对原 gate 的开放面积比 `{proxy['gate_area_ratio_candidate_over_baseline']:.4f}x`。

以上均为 Nangate45 开放逻辑/宏面积与 OpenSTA 代理，不是 DC、目标 SRAM 或 ASIC PPA；没有 SAIF，不能声称能量收益。

## 冻结边界

不拆共享 fallback directory，不修改 `h67_laws_shared_backend_2s_top`。原因是 class-scan-only 性能上界过小，且 fail-fail 会破坏双 workspace 解耦；开放宏面积不足以抵消该架构风险。后续只允许在目标 SAIF/DC 已具备时重新审计能量，不自动恢复 RTL 集成。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--macro-lib", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(args.output_dir, args.profile, args.macro_lib)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output_dir / "report.md").write_text(render_markdown(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
