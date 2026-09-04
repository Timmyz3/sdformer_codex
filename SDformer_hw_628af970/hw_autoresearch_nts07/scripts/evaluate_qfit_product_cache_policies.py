#!/usr/bin/env python3
"""评估Local5 gate-product复用策略与GS-TTB命令编码。"""

from __future__ import annotations

import csv
import json
import math
import statistics
import itertools
from collections import Counter, OrderedDict, defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACE = (
    ROOT
    / "results/qfit_local5_projection_tile_yosys_20260731"
    / "ordered_term_trace.csv"
)
OUT = ROOT / "results/qfit_product_policy_matrix_20260731"


def load_rows(path: Path = TRACE) -> list[dict[str, int]]:
    with path.open(newline="") as handle:
        rows = [
            {key: int(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    if not rows:
        raise RuntimeError("ordered term trace为空")
    return rows


def result(name: str, terms: int, misses: int, **extra: object) -> dict[str, object]:
    return {
        "policy": name,
        "terms": terms,
        "hits": terms - misses,
        "product_starts": misses,
        "reuse_ratio": 1.0 - misses / terms,
        **extra,
    }


def simulate_lru(rows: list[dict[str, int]], ways: int) -> dict[str, object]:
    caches: dict[int, OrderedDict[int, None]] = defaultdict(OrderedDict)
    misses = 0
    for row in rows:
        cache = caches[row["lane"]]
        gate = row["gate"]
        if gate in cache:
            cache.move_to_end(gate)
        else:
            misses += 1
            if len(cache) == ways:
                cache.popitem(last=False)
            cache[gate] = None
    return result(f"lru_{ways}", len(rows), misses)


def lru_miss_flags(
    rows: list[dict[str, int]], ways: int
) -> list[bool]:
    caches: dict[int, OrderedDict[int, None]] = defaultdict(OrderedDict)
    flags = []
    for row in rows:
        cache = caches[row["lane"]]
        gate = row["gate"]
        miss = gate not in cache
        flags.append(miss)
        if miss:
            if len(cache) == ways:
                cache.popitem(last=False)
            cache[gate] = None
        else:
            cache.move_to_end(gate)
    return flags


def weight_vector_loads(
    rows: list[dict[str, int]], miss_flags: list[bool]
) -> int:
    loads = 0
    index = 0
    while index < len(rows):
        lane = rows[index]["lane"]
        end = index + 1
        while end < len(rows) and rows[end]["lane"] == lane:
            end += 1
        if any(miss_flags[index:end]):
            loads += 1
        index = end
    return loads


def simulate_fifo(rows: list[dict[str, int]], ways: int) -> dict[str, object]:
    entries: dict[int, set[int]] = defaultdict(set)
    queues: dict[int, deque[int]] = defaultdict(deque)
    misses = 0
    for row in rows:
        lane = row["lane"]
        gate = row["gate"]
        if gate in entries[lane]:
            continue
        misses += 1
        if len(entries[lane]) == ways:
            entries[lane].remove(queues[lane].popleft())
        entries[lane].add(gate)
        queues[lane].append(gate)
    return result(f"fifo_{ways}", len(rows), misses)


def simulate_srrip(
    rows: list[dict[str, int]], ways: int, rrpv_bits: int = 2
) -> dict[str, object]:
    max_rrpv = (1 << rrpv_bits) - 1
    insert_rrpv = max_rrpv - 1
    tags: dict[int, list[int | None]] = defaultdict(lambda: [None] * ways)
    rrpv: dict[int, list[int]] = defaultdict(lambda: [max_rrpv] * ways)
    misses = 0
    for row in rows:
        lane = row["lane"]
        gate = row["gate"]
        if gate in tags[lane]:
            way = tags[lane].index(gate)
            rrpv[lane][way] = 0
            continue
        misses += 1
        if None in tags[lane]:
            victim = tags[lane].index(None)
        else:
            while max_rrpv not in rrpv[lane]:
                rrpv[lane] = [min(max_rrpv, value + 1) for value in rrpv[lane]]
            victim = rrpv[lane].index(max_rrpv)
        tags[lane][victim] = gate
        rrpv[lane][victim] = insert_rrpv
    return result(f"srrip_{ways}", len(rows), misses)


def simulate_no_replace(
    rows: list[dict[str, int]], ways: int
) -> dict[str, object]:
    slots: dict[int, list[int]] = defaultdict(list)
    fills = 0
    bypasses = 0
    hits = 0
    for row in rows:
        table = slots[row["lane"]]
        gate = row["gate"]
        if gate in table:
            hits += 1
        elif len(table) < ways:
            table.append(gate)
            fills += 1
        else:
            bypasses += 1
    return result(
        f"first_bind_{ways}",
        len(rows),
        fills + bypasses,
        fills=fills,
        bypasses=bypasses,
        hits=hits,
    )


def simulate_belady(rows: list[dict[str, int]], ways: int) -> dict[str, object]:
    future: dict[tuple[int, int], deque[int]] = defaultdict(deque)
    for index, row in enumerate(rows):
        future[(row["lane"], row["gate"])].append(index)
    cache: dict[int, set[int]] = defaultdict(set)
    misses = 0
    bypasses = 0
    for index, row in enumerate(rows):
        lane = row["lane"]
        gate = row["gate"]
        future[(lane, gate)].popleft()
        if gate in cache[lane]:
            continue
        misses += 1
        if len(cache[lane]) < ways:
            cache[lane].add(gate)
            continue
        requested_next = (
            future[(lane, gate)][0]
            if future[(lane, gate)]
            else len(rows) + 1
        )
        if len(cache[lane]) == ways:
            victim = max(
                cache[lane],
                key=lambda item: (
                    future[(lane, item)][0]
                    if future[(lane, item)]
                    else len(rows) + 1,
                    item,
                ),
            )
            victim_next = (
                future[(lane, victim)][0]
                if future[(lane, victim)]
                else len(rows) + 1
            )
            if requested_next < victim_next:
                cache[lane].remove(victim)
                cache[lane].add(gate)
            else:
                bypasses += 1
    return result(
        f"belady_admission_oracle_{ways}",
        len(rows),
        misses,
        bypasses=bypasses,
    )


def global_codebook(
    rows: list[dict[str, int]], ways: int
) -> list[int]:
    frequency = Counter(row["gate"] for row in rows)
    return [
        gate
        for gate, _ in sorted(
            frequency.items(), key=lambda item: (-item[1], item[0])
        )[:ways]
    ]


def lane_codebooks(
    rows: list[dict[str, int]], ways: int
) -> dict[int, list[int]]:
    frequency: dict[int, Counter[int]] = defaultdict(Counter)
    for row in rows:
        frequency[row["lane"]][row["gate"]] += 1
    return {
        lane: [
            gate
            for gate, _ in sorted(
                counter.items(), key=lambda item: (-item[1], item[0])
            )[:ways]
        ]
        for lane, counter in frequency.items()
    }


def evaluate_static(
    rows: list[dict[str, int]],
    ways: int,
    codebook: list[int] | dict[int, list[int]],
    name: str,
) -> dict[str, object]:
    valid_products: set[tuple[int, int]] = set()
    misses = 0
    bypasses = 0
    code_hits = 0
    for row in rows:
        lane = row["lane"]
        gate = row["gate"]
        codes = codebook.get(lane, []) if isinstance(codebook, dict) else codebook
        if gate not in codes:
            misses += 1
            bypasses += 1
        elif (lane, gate) not in valid_products:
            misses += 1
            valid_products.add((lane, gate))
        else:
            code_hits += 1
    return result(
        name,
        len(rows),
        misses,
        bypasses=bypasses,
        code_hits=code_hits,
        codebook=codebook,
    )


def bundle_bits_dynamic(
    terms: int, fills: int, bypasses: int, ways: int, gate_bits: int = 9
) -> dict[str, int | float]:
    slot_bits = max(1, math.ceil(math.log2(ways)))
    baseline = terms * gate_bits
    fixed_packet = terms * (2 + slot_bits + gate_bits)
    split_stream = terms * (2 + slot_bits) + (fills + bypasses) * gate_bits
    return {
        "baseline_gate_bits": baseline,
        "fixed_packet_bits": fixed_packet,
        "exception_split_bits": split_stream,
        "exception_split_reduction": 1.0 - split_stream / baseline,
    }


def bundle_bits_frozen(
    terms: int, bypasses: int, ways: int, gate_bits: int = 9
) -> dict[str, int | float]:
    slot_bits = max(1, math.ceil(math.log2(ways)))
    baseline = terms * gate_bits
    fixed_packet = terms * (1 + slot_bits + gate_bits)
    split_stream = terms * (1 + slot_bits) + bypasses * gate_bits
    return {
        "baseline_gate_bits": baseline,
        "fixed_packet_bits": fixed_packet,
        "exception_split_bits": split_stream,
        "exception_split_reduction": 1.0 - split_stream / baseline,
    }


def row_group_holdout(rows: list[dict[str, int]], ways: int) -> dict[str, object]:
    groups = sorted({(row["plane"], row["y"]) for row in rows})
    global_ratios = []
    lane_ratios = []
    folds = []
    for held_out in groups:
        train = [
            row
            for row in rows
            if (row["plane"], row["y"]) != held_out
        ]
        test = [
            row
            for row in rows
            if (row["plane"], row["y"]) == held_out
        ]
        global_result = evaluate_static(
            test,
            ways,
            global_codebook(train, ways),
            "row_holdout_global",
        )
        lane_result = evaluate_static(
            test,
            ways,
            lane_codebooks(train, ways),
            "row_holdout_per_lane",
        )
        global_ratios.append(float(global_result["reuse_ratio"]))
        lane_ratios.append(float(lane_result["reuse_ratio"]))
        folds.append(
            {
                "held_out": list(held_out),
                "terms": len(test),
                "global_reuse": global_result["reuse_ratio"],
                "per_lane_reuse": lane_result["reuse_ratio"],
            }
        )
    return {
        "warning": "仅为同一W6 trace的row-group诊断，不是sample-held-out证据",
        "folds": folds,
        "global_mean": statistics.mean(global_ratios),
        "global_min": min(global_ratios),
        "per_lane_mean": statistics.mean(lane_ratios),
        "per_lane_min": min(lane_ratios),
    }


def descriptor_gate_dictionary(rows: list[dict[str, int]]) -> dict[str, object]:
    segments = []
    for key, group in itertools.groupby(
        rows, key=lambda row: (row["plane"], row["y"], row["x"])
    ):
        body = list(group)
        segments.append(
            {
                "source": list(key),
                "terms": len(body),
                "gates": sorted({row["gate"] for row in body}),
            }
        )
    unique_sources = {tuple(item["source"]) for item in segments}
    if len(unique_sources) != len(segments):
        raise RuntimeError("同一source descriptor在ordered trace中不连续")

    terms = len(rows)
    descriptor_count = len(segments)
    dictionary_entries = sum(len(item["gates"]) for item in segments)
    source_bits = 9 + 4 + 4
    baseline_term_bits = source_bits + 5 + 9 + 5 + 1
    body_bits = 5 + 3 + 5 + 1
    ideal_header_bits = (
        descriptor_count * (source_bits + 3)
        + dictionary_entries * 9
    )
    variable_safe_header_bits = ideal_header_bits + descriptor_count * 8
    fixed_header_bits = descriptor_count * (
        source_bits + 3 + 8 + 5 * 9
    )
    baseline_total = terms * baseline_term_bits
    ideal_dictionary_total = ideal_header_bits + terms * body_bits
    variable_safe_total = variable_safe_header_bits + terms * body_bits
    fixed_header_total = fixed_header_bits + terms * body_bits
    return {
        "descriptors": descriptor_count,
        "terms": terms,
        "dictionary_entries": dictionary_entries,
        "max_dictionary_entries": max(len(item["gates"]) for item in segments),
        "baseline_term_bits": baseline_term_bits,
        "dictionary_body_bits": body_bits,
        "ideal_header_bits": ideal_header_bits,
        "variable_safe_header_bits": variable_safe_header_bits,
        "fixed_header_bits": fixed_header_bits,
        "baseline_total_bits": baseline_total,
        "ideal_dictionary_total_bits": ideal_dictionary_total,
        "variable_safe_total_bits": variable_safe_total,
        "fixed_header_total_bits": fixed_header_total,
        "fixed_header_reduction": 1.0 - fixed_header_total / baseline_total,
        "gate_only_baseline_bits": terms * 9,
        "gate_only_dictionary_bits": descriptor_count * 3
        + dictionary_entries * 9
        + terms * 3,
        "segments": segments,
    }


def hamming(value: int) -> int:
    return value.bit_count()


def gate_stationary_reorder(
    rows: list[dict[str, int]]
) -> tuple[list[dict[str, int]], dict[str, object]]:
    reordered = []
    descriptor_count = 0
    for _, group in itertools.groupby(
        rows, key=lambda row: (row["plane"], row["y"], row["x"])
    ):
        descriptor_count += 1
        body = list(group)
        gate_masks: OrderedDict[int, int] = OrderedDict()
        lanes: OrderedDict[int, None] = OrderedDict()
        for row in body:
            if row["gate"] in gate_masks:
                if gate_masks[row["gate"]] != row["mask"]:
                    raise RuntimeError("同一descriptor的gate对应多个mask")
            else:
                gate_masks[row["gate"]] = row["mask"]
            lanes.setdefault(row["lane"], None)
        expected = {
            (row["lane"], row["gate"], row["mask"]) for row in body
        }
        product = {
            (lane, gate, mask)
            for gate, mask in gate_masks.items()
            for lane in lanes
        }
        if expected != product or len(body) != len(product):
            raise RuntimeError("descriptor term不是lane×gate笛卡尔积")
        template = body[0]
        for gate, mask in gate_masks.items():
            for lane in lanes:
                reordered.append(
                    {
                        **template,
                        "seq": len(reordered),
                        "lane": lane,
                        "gate": gate,
                        "mask": mask,
                    }
                )

    def toggles(sequence: list[dict[str, int]]) -> dict[str, int]:
        totals = {
            "gate_hamming": 0,
            "mask_hamming": 0,
            "lane_hamming": 0,
            "source_hamming": 0,
            "gate_transitions": 0,
            "mask_transitions": 0,
            "lane_transitions": 0,
            "control_hamming": 0,
        }
        for before, after in zip(sequence, sequence[1:]):
            source_before = (
                (before["plane"] << 8)
                | (before["y"] << 4)
                | before["x"]
            )
            source_after = (
                (after["plane"] << 8)
                | (after["y"] << 4)
                | after["x"]
            )
            totals["gate_hamming"] += hamming(before["gate"] ^ after["gate"])
            totals["mask_hamming"] += hamming(before["mask"] ^ after["mask"])
            totals["lane_hamming"] += hamming(before["lane"] ^ after["lane"])
            totals["source_hamming"] += hamming(source_before ^ source_after)
            totals["gate_transitions"] += before["gate"] != after["gate"]
            totals["mask_transitions"] += before["mask"] != after["mask"]
            totals["lane_transitions"] += before["lane"] != after["lane"]
        totals["control_hamming"] = (
            totals["gate_hamming"]
            + totals["mask_hamming"]
            + totals["lane_hamming"]
        )
        return totals

    original_toggles = toggles(rows)
    reordered_toggles = toggles(reordered)
    return reordered, {
        "descriptors": descriptor_count,
        "terms": len(rows),
        "original": original_toggles,
        "gate_stationary": reordered_toggles,
        "reductions": {
            key: (
                1.0 - reordered_toggles[key] / original_toggles[key]
                if original_toggles[key]
                else 0.0
            )
            for key in original_toggles
        },
    }


def evaluate(rows: list[dict[str, int]]) -> dict[str, object]:
    policy_rows = []
    bundle_rows = []
    holdout = {}
    for ways in (4, 6, 8):
        lru = simulate_lru(rows, ways)
        fifo = simulate_fifo(rows, ways)
        srrip = simulate_srrip(rows, ways)
        first_bind = simulate_no_replace(rows, ways)
        belady = simulate_belady(rows, ways)
        global_static = evaluate_static(
            rows,
            ways,
            global_codebook(rows, ways),
            f"same_trace_global_top{ways}",
        )
        lane_static = evaluate_static(
            rows,
            ways,
            lane_codebooks(rows, ways),
            f"same_trace_per_lane_top{ways}",
        )
        policy_rows.extend(
            [lru, fifo, srrip, first_bind, global_static, lane_static, belady]
        )
        bundle_rows.append(
            {
                "name": f"dynamic_gs_ttb_{ways}",
                **bundle_bits_dynamic(
                    len(rows),
                    int(first_bind["fills"]),
                    int(first_bind["bypasses"]),
                    ways,
                ),
            }
        )
        bundle_rows.append(
            {
                "name": f"pf_gs_ttb_{ways}",
                **bundle_bits_frozen(
                    len(rows),
                    int(global_static["bypasses"]),
                    ways,
                ),
            }
        )
        holdout[f"ways_{ways}"] = row_group_holdout(rows, ways)
    reordered, stationary = gate_stationary_reorder(rows)
    stationary["cache_invariance"] = {}
    for ways in (4, 6, 8):
        original_lru = simulate_lru(rows, ways)
        reordered_lru = simulate_lru(reordered, ways)
        stationary["cache_invariance"][f"lru_{ways}"] = {
            "original_product_starts": original_lru["product_starts"],
            "reordered_product_starts": reordered_lru["product_starts"],
            "equal": original_lru["product_starts"]
            == reordered_lru["product_starts"],
            "lane_major_weight_vector_loads": weight_vector_loads(
                rows, lru_miss_flags(rows, ways)
            ),
            "gate_major_weight_vector_loads": weight_vector_loads(
                reordered, lru_miss_flags(reordered, ways)
            ),
        }
    return {
        "evidence": "单W6 ordered trace；same-trace静态策略存在数据泄漏",
        "terms": len(rows),
        "policies": policy_rows,
        "bundles": bundle_rows,
        "row_group_holdout": holdout,
        "descriptor_gate_dictionary": descriptor_gate_dictionary(rows),
        "gate_stationary_late_materialization": stationary,
    }


def write_report(payload: dict[str, object]) -> None:
    lines = [
        "# Product Cache策略矩阵与GS-TTB编码审计",
        "",
        "## 1. 证据边界",
        "",
        "- 输入为1494条W6 ordered term；",
        "- LRU/FIFO/SRRIP/first-bind是在线精确策略；",
        "- Belady是离线oracle上界，不能实现为当前硬件；",
        "- global/per-lane top-K在同一trace选择并评估，存在数据泄漏；",
        "- row-group holdout仍来自同一样本，只用于诊断，不能替代sample-held-out。",
        "",
        "## 2. 策略结果",
        "",
        "| 策略 | hit | product start | 减少 |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["policies"]:
        lines.append(
            f"| {row['policy']} | {row['hits']} | {row['product_starts']} | "
            f"{row['reuse_ratio']:.2%} |"
        )
    lines += [
        "",
        "## 3. Bundle位宽",
        "",
        "| 编码 | baseline gate bit | 固定宽包bit | 异常拆分bit | 拆分减少 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["bundles"]:
        lines.append(
            f"| {row['name']} | {row['baseline_gate_bits']} | "
            f"{row['fixed_packet_bits']} | {row['exception_split_bits']} | "
            f"{row['exception_split_reduction']:.2%} |"
        )
    lines += [
        "",
        "固定宽GS-TTB必须同时携带op、slot和gate，反而比原9-bit gate流更宽。"
        "只有把每term的窄slot命令与仅fill/bypass出现的gate sideband拆成两条"
        "弹性流，才产生净带宽收益。该结构暂命名为",
        "**ES-GS-TTB（Exception-Split Gate-Slot TTB）**。",
        "",
        "## 4. Row-group诊断",
        "",
        "| ways | global mean/min | per-lane mean/min |",
        "|---:|---:|---:|",
    ]
    for key, value in payload["row_group_holdout"].items():
        lines.append(
            f"| {key.split('_')[1]} | {value['global_mean']:.2%}/"
            f"{value['global_min']:.2%} | {value['per_lane_mean']:.2%}/"
            f"{value['per_lane_min']:.2%} |"
        )
    lines += [
        "",
        "## 5. 架构判定",
        "",
        "1. 后续GS候选必须同时对比LRU、FIFO、SRRIP、投影端first-bind、"
        "static top-K和Belady上界；",
        "2. PF的product-start收益不能归因于producer-side bundle；必须单独"
        "消融static codebook与跨阶段slot传递；",
        "3. 固定宽GS-TTB为NO-GO；仅ES-GS-TTB进入接口设计；",
        "4. ES双流必须证明sideband FIFO、join、反压和异常顺序成本后仍有净收益；",
        "5. 没有独立sample-held-out前，不把PF或row-group数字写入论文主结论。",
        "",
        "## 6. Descriptor-Gate TTB",
        "",
    ]
    dictionary = payload["descriptor_gate_dictionary"]
    lines += [
        f"- descriptor数：{dictionary['descriptors']}；",
        f"- term数：{dictionary['terms']}；",
        f"- gate字典项总数：{dictionary['dictionary_entries']}；",
        f"- 单descriptor最大gate项：{dictionary['max_dictionary_entries']}；",
        f"- 原term格式：{dictionary['baseline_term_bits']} bit/term，"
        f"总计{dictionary['baseline_total_bits']} bit；",
        f"- 理想变长下界：{dictionary['ideal_dictionary_total_bits']} bit；",
        f"- 带8-bit body_count的安全变长格式："
        f"{dictionary['variable_safe_total_bits']} bit；",
        f"- 单拍固定5-entry header格式：header共"
        f"{dictionary['fixed_header_bits']} bit，body为"
        f"{dictionary['dictionary_body_bits']} bit/term，总计"
        f"{dictionary['fixed_header_total_bits']} bit；",
        f"- 固定header全term流位数减少："
        f"{dictionary['fixed_header_reduction']:.2%}。",
        "",
        "该结果利用现有source-multicast builder已经完成的五角色gate去重："
        "每个source descriptor只发送一次`source metadata + gate dictionary`，"
        "header同时携带8-bit body_count，后续lane-term仅发送"
        "`lane + 3-bit gate_ref + destination mask + last`。"
        "它借鉴Bishop TTB的header/body打包纪律，但不使用ECP，也不改变任何term。"
        "后续命名为**DG-TTB（Descriptor-Gate Token-Term Bundle）**。",
        "",
        "## 7. Gate-Stationary延迟展开",
        "",
    ]
    stationary = payload["gate_stationary_late_materialization"]
    original = stationary["original"]
    gate_stationary = stationary["gate_stationary"]
    reductions = stationary["reductions"]
    lines += [
        "当前builder按`lane-major: lane -> unique gate`展开。因term集合严格等于"
        "`active lane × unique gate`，可无损交换循环为"
        "`gate-major: gate -> active lane`，并在projection端本地延迟展开。",
        "",
        "| 活动 | lane-major | gate-major | 减少 |",
        "|---|---:|---:|---:|",
        f"| gate切换次数 | {original['gate_transitions']} | "
        f"{gate_stationary['gate_transitions']} | "
        f"{reductions['gate_transitions']:.2%} |",
        f"| gate总Hamming翻转 | {original['gate_hamming']} | "
        f"{gate_stationary['gate_hamming']} | "
        f"{reductions['gate_hamming']:.2%} |",
        f"| mask切换次数 | {original['mask_transitions']} | "
        f"{gate_stationary['mask_transitions']} | "
        f"{reductions['mask_transitions']:.2%} |",
        f"| mask总Hamming翻转 | {original['mask_hamming']} | "
        f"{gate_stationary['mask_hamming']} | "
        f"{reductions['mask_hamming']:.2%} |",
        f"| lane总Hamming翻转 | {original['lane_hamming']} | "
        f"{gate_stationary['lane_hamming']} | "
        f"{reductions['lane_hamming']:.2%} |",
        f"| lane切换次数 | {original['lane_transitions']} | "
        f"{gate_stationary['lane_transitions']} | "
        f"{reductions['lane_transitions']:.2%} |",
        f"| gate+mask+lane Hamming | {original['control_hamming']} | "
        f"{gate_stationary['control_hamming']} | "
        f"{reductions['control_hamming']:.2%} |",
        "",
        "| cache | product start | lane-major weight-vector load | "
        "gate-major weight-vector load |",
        "|---|---:|---:|---:|",
    ]
    for name, value in stationary["cache_invariance"].items():
        lines.append(
            f"| {name} | {value['original_product_starts']} | "
            f"{value['lane_major_weight_vector_loads']} | "
            f"{value['gate_major_weight_vector_loads']} |"
        )
    lines += [
        "",
        "W4/W6/W8的每lane gate访问子序列不变，因此LRU product-start逐项相同。"
        "lane-major可在连续同lane的多个gate miss之间复用一次weight-vector读取；"
        "gate-major减少gate/mask翻转，但增加lane/tag/weight地址活动。"
        "该候选不再传输展开后的DG body，而是让原始descriptor驻留在projection端，"
        "由同一控制器选择lane-major或gate-major。暂命名为"
        "**DS-FLM（Dual-Stationary Factorized Late Materialization）**。"
        "当前只报告分账，不在没有SRAM/比较器能耗系数时宣布哪一模式获胜。",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = evaluate(load_rows())
    (OUT / "report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_report(payload)
    print(OUT / "report.md")


if __name__ == "__main__":
    main()
