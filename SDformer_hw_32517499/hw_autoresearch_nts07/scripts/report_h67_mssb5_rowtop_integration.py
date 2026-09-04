#!/usr/bin/env python3
"""Build the fail-closed MSSB5 row-top integration evidence report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FAIR_RE = re.compile(
    r"FAIR_SUM rows=(?P<rows>\d+) skip=(?P<skip>\d+) "
    r"fixed=(?P<fixed>\d+) rqtb=(?P<rqtb>\d+) shared=(?P<shared>\d+) "
    r"fpairs=(?P<fpairs>\d+) fslots=(?P<fslots>\d+) "
    r"fequal=(?P<fequal>\d+) rpairs=(?P<rpairs>\d+) "
    r"rslots=(?P<rslots>\d+) requal=(?P<requal>\d+)"
)


def read(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text()


def parse_fair(path: Path) -> dict[str, int]:
    text = read(path)
    match = FAIR_RE.search(text)
    if not match or "PASS tb_h67_laws_fair_lfsr_threeway_2s" not in text:
        raise ValueError(f"incomplete fair log: {path}")
    return {key: int(value) for key, value in match.groupdict().items()}


def parse_area(path: Path) -> dict[str, float | int]:
    text = read(path)
    cells = re.findall(r"Number of cells:\s+(\d+)", text)
    areas = re.findall(r"Chip area for module.*?:\s+([0-9.]+)", text)
    if not cells or not areas or "Found and reported 0 problems" not in text:
        raise ValueError(f"incomplete mapping log: {path}")
    return {"cells": int(cells[-1]), "area_proxy": float(areas[-1])}


def parse_sta(path: Path) -> dict[str, float | str]:
    text = read(path)
    arrivals = re.findall(r"^\s*([0-9.]+)\s+data arrival time$", text, re.M)
    slacks = re.findall(r"^\s*(-?[0-9.]+)\s+slack \((MET|VIOLATED)\)$", text, re.M)
    if not arrivals or not slacks:
        raise ValueError(f"incomplete STA log: {path}")
    return {
        "arrival_ns": float(arrivals[-1]),
        "slack_ns": float(slacks[-1][0]),
        "status": slacks[-1][1],
    }


def ratio_reduction(candidate: float, baseline: float) -> float:
    return 1.0 - candidate / baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slot-dir", type=Path,
        default=Path("results/h67_mssb5_slot_ep35_rtl_20260814")
    )
    parser.add_argument(
        "--mssb-fair-dir", type=Path,
        default=Path("results/h67_mssb5_fair_ep35_rtl_20260814")
    )
    parser.add_argument(
        "--direct-fair-dir", type=Path,
        default=Path("results/h67_direct_fair_ep35_rtl_20260814")
    )
    parser.add_argument(
        "--proxy-dir", type=Path,
        default=Path("results/h67_mssb5_slot_integration_openproxy_20260814")
    )
    parser.add_argument(
        "--leaf-report", type=Path,
        default=Path("results/h67_mssb5_score_pair_screen_20260810/report.json")
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/h67_mssb5_rowtop_integration_20260814")
    )
    args = parser.parse_args()

    for engine in ("iverilog", "verilator"):
        text = read(args.slot_dir / "logs" / f"{engine}.log")
        required = (
            "MSSB5_SLOT_EP35 rows=138 pairs=31050 packet_mismatch=0",
            "PASS tb_h67_mssb5_temporal_slot_encoder_ep35",
        )
        if any(item not in text for item in required):
            raise ValueError(f"incomplete {engine} slot miter")

    mssb_fair = parse_fair(
        args.mssb_fair_dir / "fair_lfsr_threeway_iverilog.log"
    )
    direct_fair = parse_fair(
        args.direct_fair_dir / "fair_lfsr_threeway_iverilog.log"
    )
    frozen = {
        "rows": 138, "skip": 33, "fixed": 112589, "rqtb": 94891,
        "shared": 87034, "fpairs": 31050, "fslots": 62100,
        "fequal": 28001, "rpairs": 31050, "rslots": 34099,
        "requal": 28001,
    }
    if mssb_fair != frozen or direct_fair != frozen:
        raise ValueError(
            f"frozen fair mismatch: direct={direct_fair}, mssb={mssb_fair}"
        )

    proxy: dict[str, dict[str, dict[str, float | int | str]]] = {}
    for target in ("t3", "t2"):
        proxy[target] = {}
        for name in ("cse7", "mssb5"):
            module = f"h67_{name}_temporal_slot_encoder"
            area = parse_area(
                args.proxy_dir / "logs" / f"nangate45_dff_{target}_{module}.log"
            )
            timing = parse_sta(
                args.proxy_dir / "logs" / f"sta_dff_{target}_{module}.log"
            )
            proxy[target][name] = {**area, **timing}

    cse3 = proxy["t3"]["cse7"]
    mssb3 = proxy["t3"]["mssb5"]
    area_reduction = ratio_reduction(
        float(mssb3["area_proxy"]), float(cse3["area_proxy"])
    )
    delay_reduction = ratio_reduction(
        float(mssb3["arrival_ns"]), float(cse3["arrival_ns"])
    )
    adp_reduction = ratio_reduction(
        float(mssb3["area_proxy"]) * float(mssb3["arrival_ns"]),
        float(cse3["area_proxy"]) * float(cse3["arrival_ns"]),
    )
    if area_reduction <= 0 or delay_reduction <= 0:
        raise ValueError("MSSB5 does not beat CSE7 under the matched 3ns proxy")
    if proxy["t2"]["mssb5"]["status"] != "MET":
        raise ValueError("MSSB5 must meet the 2ns proxy constraint")

    leaf = json.loads(read(args.leaf_report))
    if leaf.get("schema") != "h67_mssb5_score_pair_screen_v1":
        raise ValueError(f"unexpected leaf report schema: {args.leaf_report}")
    leaf_candidates = leaf.get("candidates", {})
    try:
        ssr5_leaf = leaf_candidates["h67_ssr5_score_pair"]
        mssb5_leaf = leaf_candidates["h67_mssb5_score_pair"]
    except KeyError as exc:
        raise ValueError(f"missing SSR5/MSSB5 leaf baseline: {args.leaf_report}") from exc
    leaf_area_reduction = ratio_reduction(
        float(mssb5_leaf["area"]), float(ssr5_leaf["area"])
    )
    leaf_delay_reduction = ratio_reduction(
        float(mssb5_leaf["delay_ns"]), float(ssr5_leaf["delay_ns"])
    )
    if not (0.0 < leaf_area_reduction < 0.05):
        raise ValueError("unexpected MSSB5 vs SSR5 leaf area delta")
    if not (0.0 < leaf_delay_reduction < 0.02):
        raise ValueError("unexpected MSSB5 vs SSR5 leaf delay delta")

    report = {
        "schema": "h67_mssb5_rowtop_integration_v2",
        "decision": "ADMIT_AS_MOTION_CSE_SUPPORT_ONLY",
        "evidence": ["[rtl]", "[开放逻辑映射代理]", "[开放网表STA代理]"],
        "slot_packet_miter": {
            "engines": ["iverilog", "verilator"],
            "rows": 138,
            "pairs": 31050,
            "mismatches": 0,
            "implementations": ["direct", "cse7", "mssb5"],
        },
        "fair_rtl": mssb_fair,
        "matched_proxy": proxy,
        "t3_reduction_vs_cse7": {
            "area": area_reduction,
            "delay": delay_reduction,
            "area_delay_product": adp_reduction,
        },
        "leaf_strong_baseline": {
            "baseline": "h67_ssr5_score_pair",
            "candidate": "h67_mssb5_score_pair",
            "ssr5": ssr5_leaf,
            "mssb5": mssb5_leaf,
            "mssb5_area_reduction": leaf_area_reduction,
            "mssb5_delay_reduction": leaf_delay_reduction,
            "packed_butterfly_is_independent_contribution": False,
        },
        "claim_boundary": {
            "standalone_date_contribution": False,
            "dc_sta_saif_ptpx": False,
            "post_layout": False,
            "full_encoder": False,
            "frozen_main_table_changed": False,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    md = f"""# Motion MSSB5 行顶层集成与强基线筛选

## 裁决

`ADMIT_AS_MOTION_CSE_SUPPORT_ONLY`。MSSB5 只作为 Motion 精确
score-to-quotient 数据流的 score-front CSE 支撑，不单列为新的 DATE 主贡献。

## RTL 等价

- `[rtl]`：Direct/CSE7/MSSB5 三方 packet miter，Icarus 与 Verilator 均覆盖
  138 行、31050 个真实 ep35 temporal pair，packet mismatch 为 0；
- `[rtl]`：MSSB5 行顶层公平回放仍严格得到
  `112589/94891/34099/28001`，与 default-direct 回放逐项相同；
- 本轮没有产生或改写任何主表加速数字。

## 行顶层 CSE7 对照代理

| 3 ns 时序驱动映射 | cells | 面积代理 | 关键路径(ns) | slack(ns) |
|---|---:|---:|---:|---:|
| CSE7 packet encoder | {cse3['cells']} | {cse3['area_proxy']:.3f} | {cse3['arrival_ns']:.6f} | {cse3['slack_ns']:.6f} |
| MSSB5 packet encoder | {mssb3['cells']} | {mssb3['area_proxy']:.3f} | {mssb3['arrival_ns']:.6f} | {mssb3['slack_ns']:.6f} |

MSSB5 相对 CSE7：面积代理下降 `{area_reduction:.2%}`，关键路径下降
`{delay_reduction:.2%}`，面积延迟积代理下降 `{adp_reduction:.2%}`。
2 ns 代理下 MSSB5 为 `{proxy['t2']['mssb5']['status']}`，CSE7 为
`{proxy['t2']['cse7']['status']}`。

该 15.20% 级结果不能用于声称 MSSB5 本身具有同量级的新颖收益，因为 CSE7
不是最强算术基线。已有同边界叶级 SSR5 已直接形成相同五个充分统计量：

| 叶级强基线 | 面积代理 | 关键路径(ns) |
|---|---:|---:|
| SSR5 | {float(ssr5_leaf['area']):.3f} | {float(ssr5_leaf['delay_ns']):.6f} |
| MSSB5 | {float(mssb5_leaf['area']):.3f} | {float(mssb5_leaf['delay_ns']):.6f} |

MSSB5 相对 SSR5 仅减少 `{leaf_area_reduction:.2%}` 面积代理和
`{leaf_delay_reduction:.2%}` 关键路径。故主要收益来自“五充分统计量替代七计数”
这一 domain CSE；packed reduction tree 只是小幅实现增量。

## 可辩护架构边界

MSSB5 的价值不是“蝶形”本身，而是把双时间 Motion-XOR score 所需的
七个常规计数重写为五个精确充分统计量
`{{overlap0,same-zero0,overlap1,same-zero1,motion}}`，随后直接生成 RQTB slot。
它改变的是 score front 的归约对象；RQTB 改变的是归一化前的存储/调度对象。
二者组成一条精确流水，但仍只计为一条 Motion 主贡献的两个实现阶段。
SSR5 强基线表明，这一阶段只能作为 RQTB 主贡献的前端实现支撑，不能独立抬高
架构创新评分。

## 证据边界

- 面积和时序均为 Yosys/ABC + Nangate45 + OpenSTA 开放代理，不是 DC/STA 签核；
- 无 SRAM 宏、布局布线寄生、SAIF、功耗或 PTPX；
- 仅覆盖 attention row/tile，不是 full encoder；
- 面积优先 `abc -fast` 映射曾出现面积下降但路径变长，说明结论依赖映射目标；
  论文只允许引用同约束时序驱动结果，并需由 DC 复核。
"""
    (args.output_dir / "report.md").write_text(md)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
