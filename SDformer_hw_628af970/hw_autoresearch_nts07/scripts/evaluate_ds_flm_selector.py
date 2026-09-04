#!/usr/bin/env python3
"""评估既有late materializer两种展开顺序的状态不变量与成本敏感性。"""

from __future__ import annotations

import csv
import itertools
import json
from collections import OrderedDict, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACE = (
    ROOT
    / "results/qfit_local5_projection_tile_yosys_20260731"
    / "ordered_term_trace.csv"
)
OUT = ROOT / "results/ds_flm_selector_model_20260731"


def load_rows(path: Path = TRACE) -> list[dict[str, int]]:
    with path.open(newline="") as handle:
        rows = [
            {key: int(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    if not rows:
        raise RuntimeError("ordered term trace为空")
    return rows


def hamming(value: int) -> int:
    return value.bit_count()


def descriptors(
    rows: list[dict[str, int]],
) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for key, group in itertools.groupby(
        rows, key=lambda row: (row["plane"], row["y"], row["x"])
    ):
        body = list(group)
        lanes: OrderedDict[int, None] = OrderedDict()
        gate_masks: OrderedDict[int, int] = OrderedDict()
        for row in body:
            lanes.setdefault(row["lane"], None)
            previous = gate_masks.setdefault(row["gate"], row["mask"])
            if previous != row["mask"]:
                raise RuntimeError("同一descriptor的gate对应多个mask")
        lane_values = list(lanes)
        gate_values = list(gate_masks.items())
        expected = {
            (lane, gate, mask)
            for lane in lane_values
            for gate, mask in gate_values
        }
        observed = {
            (row["lane"], row["gate"], row["mask"]) for row in body
        }
        if observed != expected or len(body) != len(expected):
            raise RuntimeError("descriptor不是严格lane×gate笛卡尔积")
        lane_major = [
            (lane, gate, mask)
            for lane in lane_values
            for gate, mask in gate_values
        ]
        gate_major = [
            (lane, gate, mask)
            for gate, mask in gate_values
            for lane in lane_values
        ]
        if lane_major[0] != gate_major[0] or lane_major[-1] != gate_major[-1]:
            raise RuntimeError("两模式首尾项不一致，局部选择将产生跨descriptor状态")
        values.append(
            {
                "key": key,
                "lanes": lane_values,
                "gates": gate_values,
                "lane_major": lane_major,
                "gate_major": gate_major,
            }
        )
    return values


def output_toggles(
    sequence: list[tuple[int, int, int]],
    previous: tuple[int, int, int],
) -> tuple[dict[str, int], tuple[int, int, int]]:
    totals = {"lane": 0, "gate": 0, "mask": 0}
    before = previous
    for after in sequence:
        totals["lane"] += hamming(before[0] ^ after[0])
        totals["gate"] += hamming(before[1] ^ after[1])
        totals["mask"] += hamming(before[2] ^ after[2])
        before = after
    return totals, before


def scan_state_toggles(
    lane_count: int, gate_count: int, mode: str
) -> dict[str, int]:
    if mode == "lane":
        # capture时0->K，随后每个active lane恰好清除一次。
        bitmap = 2 * lane_count
        gate_indices = [
            gate_index
            for _ in range(lane_count)
            for gate_index in range(gate_count)
        ]
    elif mode == "gate":
        # 每个gate扫描完K；除最后一轮外都需要K bitmap重载。
        bitmap = 2 * lane_count * gate_count
        gate_indices = [
            gate_index
            for gate_index in range(gate_count)
            for _ in range(lane_count)
        ]
    else:
        raise ValueError(mode)
    gate_index_hamming = 0
    before = 0
    for after in gate_indices:
        gate_index_hamming += hamming(before ^ after)
        before = after
    return {
        "bitmap": bitmap,
        "gate_index": gate_index_hamming,
    }


def lru_miss_map(
    values: list[dict[str, object]], ways: int
) -> dict[tuple[tuple[int, int, int], int, int], bool]:
    caches: dict[int, OrderedDict[int, None]] = defaultdict(OrderedDict)
    misses: dict[tuple[tuple[int, int, int], int, int], bool] = {}
    for descriptor in values:
        key = descriptor["key"]
        for lane, gate, _ in descriptor["lane_major"]:
            cache = caches[lane]
            miss = gate not in cache
            misses[(key, lane, gate)] = miss
            if miss:
                if len(cache) == ways:
                    cache.popitem(last=False)
                cache[gate] = None
            else:
                cache.move_to_end(gate)
    return misses


def weight_loads(
    descriptor: dict[str, object],
    mode: str,
    miss_map: dict[tuple[tuple[int, int, int], int, int], bool],
) -> int:
    sequence = descriptor[f"{mode}_major"]
    key = descriptor["key"]
    loads = 0
    index = 0
    while index < len(sequence):
        lane = sequence[index][0]
        end = index + 1
        while end < len(sequence) and sequence[end][0] == lane:
            end += 1
        if any(
            miss_map[(key, item_lane, gate)]
            for item_lane, gate, _ in sequence[index:end]
        ):
            loads += 1
        index = end
    return loads


def metric(
    output: dict[str, int],
    scan: dict[str, int],
    loads: int,
    weight_cost: int,
) -> int:
    return (
        output["lane"]
        + output["gate"]
        + output["mask"]
        + scan["bitmap"]
        + scan["gate_index"]
        + loads * weight_cost
    )


def evaluate(rows: list[dict[str, int]]) -> dict[str, object]:
    values = descriptors(rows)
    per_way: dict[str, object] = {}
    for ways in (4, 6, 8):
        miss_map = lru_miss_map(values, ways)
        descriptors_cost = []
        previous = (0, 0, 0)
        for descriptor in values:
            lane_output, lane_last = output_toggles(
                descriptor["lane_major"], previous
            )
            gate_output, gate_last = output_toggles(
                descriptor["gate_major"], previous
            )
            if lane_last != gate_last:
                raise RuntimeError("两模式descriptor末状态不一致")
            lane_count = len(descriptor["lanes"])
            gate_count = len(descriptor["gates"])
            descriptors_cost.append(
                {
                    "source": list(descriptor["key"]),
                    "lanes": lane_count,
                    "gates": gate_count,
                    "lane_output": lane_output,
                    "gate_output": gate_output,
                    "lane_scan": scan_state_toggles(
                        lane_count, gate_count, "lane"
                    ),
                    "gate_scan": scan_state_toggles(
                        lane_count, gate_count, "gate"
                    ),
                    "lane_weight_loads": weight_loads(
                        descriptor, "lane", miss_map
                    ),
                    "gate_weight_loads": weight_loads(
                        descriptor, "gate", miss_map
                    ),
                }
            )
            previous = lane_last

        sweep = []
        for weight_cost in (0, 1, 2, 4, 8, 16, 32, 64, 128):
            lane_total = 0
            gate_total = 0
            selector_total = 0
            gate_choices = 0
            for item in descriptors_cost:
                lane_cost = metric(
                    item["lane_output"],
                    item["lane_scan"],
                    item["lane_weight_loads"],
                    weight_cost,
                )
                gate_cost = metric(
                    item["gate_output"],
                    item["gate_scan"],
                    item["gate_weight_loads"],
                    weight_cost,
                )
                lane_total += lane_cost
                gate_total += gate_cost
                selector_total += min(lane_cost, gate_cost)
                gate_choices += gate_cost < lane_cost
            best_static = min(lane_total, gate_total)
            sweep.append(
                {
                    "weight_cost": weight_cost,
                    "lane_total": lane_total,
                    "gate_total": gate_total,
                    "best_static": best_static,
                    "selector_total": selector_total,
                    "selector_gain_vs_best_static": (
                        1.0 - selector_total / best_static
                        if best_static
                        else 0.0
                    ),
                    "gate_choices": gate_choices,
                }
            )
        per_way[str(ways)] = {
            "product_misses": sum(miss_map.values()),
            "lane_weight_loads": sum(
                item["lane_weight_loads"] for item in descriptors_cost
            ),
            "gate_weight_loads": sum(
                item["gate_weight_loads"] for item in descriptors_cost
            ),
            "lane_bitmap_hamming": sum(
                item["lane_scan"]["bitmap"] for item in descriptors_cost
            ),
            "gate_bitmap_hamming": sum(
                item["gate_scan"]["bitmap"] for item in descriptors_cost
            ),
            "sweep": sweep,
            "descriptors": descriptors_cost,
        }
    return {
        "evidence": (
            "QFIT 3x6x2定向RTL slice TB导出的ordered term；"
            "不是部署或post-G0 trace。Hamming不是功耗，"
            "weight_cost不是物理能量，weight-vector buffer尚未实现。"
        ),
        "terms": len(rows),
        "descriptor_count": len(values),
        "first_last_state_invariant": True,
        "per_lane_gate_sequence_invariant": True,
        "per_way": per_way,
    }


def write_report(value: dict[str, object]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    )
    lines = [
        "# DS-FLM状态不变量与模式选择成本敏感性",
        "",
        "## 1. 证据边界",
        "",
        f"- 输入：QFIT 3x6x2定向RTL slice TB导出的"
        f"{value['terms']}条ordered term、{value['descriptor_count']}个descriptor；",
        "- 证据等级为`[rtl-directed]`，不是部署、post-G0或fullres workload；",
        "- 所有数字均为结构计数或Hamming代理，不是功耗/PPA；",
        "- `weight_cost`表示一次weight-vector读取相对一次bit toggle的假设成本；",
        "- 容量1 weight-vector驻留尚未进入现有W6 LRU RTL，读取次数是候选模型；",
        "- 两模式首项、末项及每lane gate子序列完全一致，因此局部选择不改变"
        "跨descriptor输出边界和LRU未来状态。",
        "",
        "## 2. 扫描状态与weight驻留",
        "",
        "| ways | product miss | lane-major weight load | gate-major weight load | "
        "lane bitmap flip | gate bitmap flip |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for ways, item in value["per_way"].items():
        lines.append(
            f"| {ways} | {item['product_misses']} | "
            f"{item['lane_weight_loads']} | {item['gate_weight_loads']} | "
            f"{item['lane_bitmap_hamming']} | "
            f"{item['gate_bitmap_hamming']} |"
        )
    lines.extend(
        [
            "",
            "## 3. 成本系数敏感性",
            "",
            "下表中的selector是每descriptor选择模型成本更低的模式；"
            "它是oracle敏感性上界，不是已实现的在线selector。",
            "",
            "| ways | weight_cost | lane total | gate total | selector | "
            "相对最佳静态收益 | 选择gate descriptor |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for ways, item in value["per_way"].items():
        for row in item["sweep"]:
            lines.append(
                f"| {ways} | {row['weight_cost']} | {row['lane_total']} | "
                f"{row['gate_total']} | {row['selector_total']} | "
                f"{row['selector_gain_vs_best_static']:.2%} | "
                f"{row['gate_choices']} |"
            )
    lines.extend(
        [
            "",
            "## 4. 架构判定",
            "",
            "1. Factorized late materialization由既有lane-major builder已经实现，"
            "不是本轮DS-FLM新增；",
            "2. gate-major会显著增加工作bitmap重载和weight-vector读取，"
            "不能用外部总线Hamming下降直接推导能耗下降；",
            "3. 若所有合理`weight_cost`下最佳静态模式不变，自动selector应淘汰；",
            "4. 只有在物理标定系数下出现稳定的descriptor级互补性，"
            "selector才有资格进入DATE贡献；",
            "5. 下一步先需真实descriptor trace，再接W6 LRU/TCFM；最后才用"
            "集成RTL VCD/SAIF和真实SRAM接口标定各项系数。",
            "",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines))


def main() -> None:
    value = evaluate(load_rows())
    write_report(value)
    print(json.dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    main()
