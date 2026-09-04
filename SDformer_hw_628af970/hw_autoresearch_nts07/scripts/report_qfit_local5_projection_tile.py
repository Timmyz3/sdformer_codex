#!/usr/bin/env python3
"""生成 Local5 C1-C3 端到端架构切片中文报告。"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/qfit_local5_projection_tile_yosys_20260731"


def main() -> None:
    score_shiftmax_marker = "PASS tb_local5_score_shiftmax_vectors vectors=1024"
    fullchain_generation_marker = (
        "PASS Local5 independent fullchain oracle "
        "inputs=36 acc32=72 terms=1498 updates=2332"
    )
    fullchain_rtl_marker = (
        "python_fullchain_miter=1 python_acc32_mismatch=0"
    )
    for simulator in ("iverilog", "verilator"):
        log = (OUT / f"score_shiftmax_pyref_{simulator}.log").read_text()
        if score_shiftmax_marker not in log:
            raise RuntimeError(
                f"{simulator} 独立 Python score/Shiftmax 向量回归未通过"
            )
    if fullchain_generation_marker not in (
        OUT / "fullchain_oracle_generation.log"
    ).read_text():
        raise RuntimeError("独立 Python Q/K->Acc32 全链 oracle 生成失败")
    directed_generation_counts = (1498, 2332)
    random_generation_match = re.search(
        r"inputs=36 acc32=72 terms=(\d+) updates=(\d+) seed=1592594996",
        (OUT / "fullchain_random_oracle_generation.log").read_text(),
    )
    if not random_generation_match:
        raise RuntimeError("固定种子随机 Python 全链 oracle 生成失败")
    random_generation_counts = tuple(map(int, random_generation_match.groups()))
    stat = json.loads((OUT / "stat.json").read_text())
    module = stat["modules"]["\\qfit_local5_projection_tile"]
    types = module["num_cells_by_type"]
    logs = {
        name: (OUT / f"{name}_iverilog.log").read_text()
        for name in ("tcfm5", "affine4", "linear5", "role_sharded")
    }
    pattern = re.compile(
        r"backend=(\d+) tile_cycles=(\d+) descriptors=(\d+) "
        r"terms=(\d+) updates=(\d+) stalls=(\d+) "
        r"issue_seed=(\d+) issue_stall_cycles=(\d+) "
        r"issue_block_hits=(\d+) "
        r"lane_gate_products=(\d+) lane1_hits=(\d+) lane1_misses=(\d+) "
        r"lane2_hits=(\d+) lane2_misses=(\d+) "
        r"dm16_hits=(\d+) dm16_misses=(\d+) linear5_cycles=(\d+) "
        r"affine4_cycles=(\d+) single_cycles=(\d+) acc32_mismatch=(\d+)",
    )
    parsed = {}
    for name, log in logs.items():
        if fullchain_rtl_marker not in log:
            raise RuntimeError(f"{name} 未使用独立 Python 全链 Acc32 oracle")
        match = pattern.search(log)
        if not match:
            raise RuntimeError(f"无法从 {name} Icarus 日志提取端到端计数")
        parsed[name] = tuple(map(int, match.groups()))
    random_counts = None
    for simulator in ("iverilog", "verilator_sva"):
        random_log = (OUT / f"fullchain_random_{simulator}.log").read_text()
        if fullchain_rtl_marker not in random_log:
            raise RuntimeError(f"{simulator} 随机全链 miter 未启用")
        random_match = pattern.search(random_log)
        if not random_match:
            raise RuntimeError(f"{simulator} 随机全链计数无法解析")
        simulator_counts = tuple(map(int, random_match.groups()))
        if simulator_counts[2] != 36 or simulator_counts[19] != 0:
            raise RuntimeError(f"{simulator} 随机全链合同不成立")
        if random_counts is None:
            random_counts = simulator_counts
        elif random_counts != simulator_counts:
            raise RuntimeError("随机全链 Icarus 与 Verilator/SVA 计数不一致")
    if random_counts[3:5] != random_generation_counts:
        raise RuntimeError("随机全链 Python 与 RTL 工作量账本不一致")
    (
        _,
        tcfm_tile_cycles,
        descriptors,
        terms,
        updates,
        stalls,
        issue_seed,
        issue_stall_cycles,
        issue_block_hits,
        lane_gate_products,
        lane1_hits,
        lane1_misses,
        lane2_hits,
        lane2_misses,
        dm16_hits,
        dm16_misses,
        linear5,
        affine4,
        single,
        acc32_mismatch,
    ) = parsed["tcfm5"]
    if (terms, updates) != directed_generation_counts:
        raise RuntimeError("定向 Python 与 Icarus term/update 账本不一致")
    verilator_baseline_log = (OUT / "verilator.log").read_text()
    if fullchain_rtl_marker not in verilator_baseline_log:
        raise RuntimeError("Verilator/SVA 基线未使用独立 Python 全链 oracle")
    verilator_baseline_match = pattern.search(verilator_baseline_log)
    if not verilator_baseline_match:
        raise RuntimeError("Verilator/SVA 定向全链计数无法解析")
    if tuple(map(int, verilator_baseline_match.groups())) != parsed["tcfm5"]:
        raise RuntimeError("定向全链 Icarus 与 Verilator/SVA 计数不一致")
    if acc32_mismatch != 0:
        raise RuntimeError("TCFM-5 基线 Acc32 出现失配")
    _, affine_tile_cycles, *affine_counts = parsed["affine4"]
    _, linear_tile_cycles, *linear_counts = parsed["linear5"]
    _, role_tile_cycles, *role_counts = parsed["role_sharded"]
    affine_stalls = affine_counts[3]
    linear_stalls = linear_counts[3]
    if issue_seed != 0 or issue_stall_cycles != 0 or issue_block_hits != 0:
        raise RuntimeError("无暂停基线意外启用了 term issue backpressure")
    if affine_counts[:3] != [descriptors, terms, updates]:
        raise RuntimeError("Affine4 与 TCFM5 的工作量计数不一致")
    if linear_counts[:3] != [descriptors, terms, updates]:
        raise RuntimeError("Linear5 与 TCFM5 的工作量计数不一致")
    if role_counts[:3] != [descriptors, terms, updates]:
        raise RuntimeError("Role-Sharded 与 TCFM5 的工作量计数不一致")
    role_stalls = role_counts[3]
    for name, counts in (
        ("Affine4", affine_counts),
        ("Linear5", linear_counts),
        ("Role-Sharded", role_counts),
    ):
        if counts[17] != 0:
            raise RuntimeError(f"{name} Acc32 出现失配")
        if counts[4:7] != [0, 0, 0]:
            raise RuntimeError(f"{name} 无暂停基线出现 issue stall")
        if counts[7:14] != [
            lane_gate_products,
            lane1_hits,
            lane1_misses,
            lane2_hits,
            lane2_misses,
            dm16_hits,
            dm16_misses,
        ]:
            raise RuntimeError(f"{name} 与 TCFM5 的值键复用计数不一致")
    backpressure = []
    for seed in (1, 44257, 48879):
        log = (OUT / f"tcfm5_backpressure_seed_{seed}_iverilog.log").read_text()
        match = pattern.search(log)
        if not match:
            raise RuntimeError(f"无法从 backpressure seed={seed} 日志提取计数")
        counts = tuple(map(int, match.groups()))
        if (
            counts[0] != 0
            or counts[6] != seed
            or counts[7] <= 0
            or counts[8] <= 0
        ):
            raise RuntimeError(f"backpressure seed={seed} 合同不成立")
        if counts[2:5] != (descriptors, terms, updates):
            raise RuntimeError(f"backpressure seed={seed} 改变了有效工作量")
        if counts[19] != 0:
            raise RuntimeError(f"backpressure seed={seed} Acc32 出现失配")
        verilator_log = (
            OUT / f"tcfm5_backpressure_seed_{seed}_verilator_sva.log"
        ).read_text()
        if fullchain_rtl_marker not in verilator_log:
            raise RuntimeError(
                f"Verilator/SVA backpressure seed={seed} 未启用全链 miter"
            )
        verilator_match = pattern.search(verilator_log)
        if not verilator_match:
            raise RuntimeError(
                f"无法从 Verilator/SVA backpressure seed={seed} 日志提取计数"
            )
        verilator_counts = tuple(map(int, verilator_match.groups()))
        if verilator_counts != counts:
            raise RuntimeError(
                f"Icarus 与 Verilator/SVA backpressure seed={seed} 计数不一致"
            )
        backpressure.append(
            {
                "seed": seed,
                "tile_cycles": counts[1],
                "relation_stalls": counts[5],
                "issue_stall_cycles": counts[7],
                "term_boundary_block_hits": counts[8],
                "acc32_mismatch": 0,
            }
        )
    protocol_pattern = re.compile(
        r"PASS Local5 adversarial protocol negative=(\d+) "
        r"consecutive_nonzero_runs=(\d+) descriptors_per_run=(\d+) "
        r"descriptor_last_pauses=(\d+) run_last_pauses=(\d+) "
        r"terms_per_run=(\d+) updates_per_run=(\d+) "
        r"descriptor_fault_terms=(\d+) descriptor_fault_updates=(\d+) "
        r"descriptor_fault_bank_writes=(\d+) acc32_mismatch=(\d+)"
    )
    protocol_counts = None
    for simulator in ("iverilog", "verilator_sva"):
        protocol_log = (OUT / f"protocol_{simulator}.log").read_text()
        protocol_match = protocol_pattern.search(protocol_log)
        if not protocol_match:
            raise RuntimeError(f"{simulator} 协议回归未通过完整检查")
        simulator_counts = tuple(map(int, protocol_match.groups()))
        if protocol_counts is None:
            protocol_counts = simulator_counts
        elif protocol_counts != simulator_counts:
            raise RuntimeError("Icarus 与 Verilator/SVA 协议计数不一致")
    expected_protocol = (9, 2, 8, 16, 2, 256, 768, 0, 0, 0, 0)
    if protocol_counts != expected_protocol:
        raise RuntimeError(f"adversarial 协议覆盖不完整: {protocol_counts}")
    linear_reduction = (linear5 - terms) / linear5 * 100
    affine4_overhead = (affine4 - terms) / terms * 100
    single_reduction = (single - terms) / single * 100
    cycle_evidence = {
        "evidence": "同一端到端TB、同一producer分别连接四个真实后端RTL",
        "workload": "W6定向C1-C3输入；非post-G0",
        "descriptors": descriptors,
        "product_terms": terms,
        "unique_lane_gate_products": lane_gate_products,
        "cross_source_product_reuse_upper_bound": (
            1.0 - lane_gate_products / terms
        ),
        "ordered_product_memo": {
            "lane_local_1_entry": {
                "hits": lane1_hits,
                "misses": lane1_misses,
                "hit_rate": lane1_hits / terms,
            },
            "lane_local_2_entry": {
                "hits": lane2_hits,
                "misses": lane2_misses,
                "hit_rate": lane2_hits / terms,
            },
            "global_direct_mapped_16": {
                "hits": dm16_hits,
                "misses": dm16_misses,
                "hit_rate": dm16_hits / terms,
            },
        },
        "destination_updates": updates,
        "relation_stalls": stalls,
        "relation_stalls_by_backend": {
            "TCFM-5": stalls,
            "Affine-4": affine_stalls,
            "Linear-5": linear_stalls,
            "Role-Sharded": role_stalls,
        },
        "fixed_seed_backpressure": backpressure,
        "independent_python_fullchain_oracle": {
            "input_rows": 36,
            "acc32_outputs": 72,
            "product_terms": terms,
            "destination_updates": updates,
            "icarus_backends": 4,
            "icarus_fixed_seed_backpressure": 3,
            "verilator_sva_baseline": "PASS",
            "verilator_sva_fixed_seed_backpressure": 3,
            "acc32_mismatch": 0,
            "random_seed": "0x5eed1234",
            "random_product_terms": random_counts[3],
            "random_destination_updates": random_counts[4],
            "random_icarus": "PASS",
            "random_verilator_sva": "PASS",
            "random_acc32_mismatch": 0,
        },
        "run_scoped_protocol": {
            "negative_cases": protocol_counts[0],
            "consecutive_nonzero_runs": protocol_counts[1],
            "descriptors_per_run": protocol_counts[2],
            "descriptor_last_pauses": protocol_counts[3],
            "run_last_pauses": protocol_counts[4],
            "terms_per_run": protocol_counts[5],
            "updates_per_run": protocol_counts[6],
            "descriptor_fault_terms": protocol_counts[7],
            "descriptor_fault_updates": protocol_counts[8],
            "descriptor_fault_bank_writes": protocol_counts[9],
            "acc32_mismatch": protocol_counts[10],
            "icarus": "PASS",
            "verilator_sva": "PASS",
        },
        "cycles": {
            "TCFM-5": tcfm_tile_cycles,
            "Linear-5": linear_tile_cycles,
            "Role-Sharded": role_tile_cycles,
            "Affine-4": affine_tile_cycles,
            "Single-bank": single,
        },
        "analytical_product_delivery_cycles": {
            "TCFM-5": terms,
            "Linear-5": linear5,
            "Affine-4": affine4,
            "Single-bank": single,
        },
    }
    (OUT / "cycle_evidence.json").write_text(
        json.dumps(cycle_evidence, ensure_ascii=False, indent=2) + "\n"
    )

    lines = [
        "# Local5 C1-C3 端到端架构切片结果",
        "",
        "## 功能闭环",
        "",
        "- 数据流：`XBF-DBDR -> FCSR-RX -> source gate-equivalence "
        "term -> TCFM-5 -> 五 bank Acc`；",
        f"- 两个平面共 {descriptors} 个 descriptor，生成 {terms} 个 "
        f"product term，完成 {updates} 次 destination update；",
        f"- 其中全窗口唯一 `(lane,gate)` product key 为 "
        f"{lane_gate_products} 个；这只是跨 source 精确复用上界，尚未计"
        "缓存容量、生命周期和访问冲突；",
        f"- 按原始有序 term 流观测：每lane 1项memo命中 "
        f"{lane1_hits}/{terms}（{lane1_hits / terms:.2%}），每lane "
        f"2项命中 {lane2_hits}/{terms}（{lane2_hits / terms:.2%}），"
        f"全局16项direct-mapped命中 {dm16_hits}/{terms}"
        f"（{dm16_hits / terms:.2%}）；这些是TB观测模型，不是已实现RTL；",
        "- 从 Q/K、真实 invalid-candidate mask 到最终每个 "
        "`plane/y/x/out` 的 Acc，均与整数金参考逐项一致；",
        "- 独立 Python 整数参考生成 1024 组 masked score/Shiftmax 向量，"
        "Icarus 与 Verilator 均逐 score/gate 零失配；",
        "- 独立 Python 全链参考显式执行 relation transpose、source-major "
        "term、INT8 projection 与 Acc32，生成 36 个 Q/K/mask 输入和 72 个 "
        "Acc32 期望；四个 Icarus 后端及 TCFM-5 Verilator/SVA 基线/三组反压"
        "均逐 Acc32 零失配；",
        f"- 固定随机种子 `0x5eed1234` 的第二组全链向量包含边界、零/全1与"
        f"随机 Q/K，产生 {random_counts[3]} 个 term、{random_counts[4]} 次 update；"
        "Icarus 与 Verilator/SVA 工作量一致且 72 个 Acc32 零失配；",
        "- Icarus、Verilator/SVA、Yosys `check -assert` 通过。",
        "- 同一 producer 已分别连接 TCFM-5、Affine-4、Linear-5、"
        "Role-Sharded 真实 RTL，四个后端最终 Acc 均与同一整数金参考一致。",
        "- TCFM-5 另以 3 个固定 16-bit LFSR 种子暂停 term issue；每次均"
        "保持 descriptor/term/update 计数不变且 Acc32 零失配。",
        "- 运行级协议以 Icarus 和 Verilator/SVA 交叉验证 9 类负例、"
        "连续 2 次非零运行、16 次 descriptor-last 暂停和 2 次 run-last 暂停，"
        "非零非法 descriptor 产生 0 term/0 update/0 bank write，逐次读回 Acc32 "
        "均为零失配。",
        "",
        "## 等端口周期基线",
        "",
        "| 映射 | 写端口口径 | product-delivery 周期 | 相对 TCFM-5 |",
        "|---|---|---:|---:|",
        f"| TCFM-5 | 5 个同步1R1W bank，拓扑着色 | {terms} | 1.000x |",
        f"| Linear-5 | 同样 5 个同步1R1W bank，raster-id `%5` | "
        f"{linear5} | {linear5 / terms:.3f}x |",
        f"| Affine-4 | 4 个同步1R1W bank，`(x+2y)%4`，精确replay | "
        f"{affine4} | {affine4 / terms:.3f}x |",
        f"| Single-bank | 1 个单写 bank | {single} | "
        f"{single / terms:.3f}x |",
        "",
        f"在该定向向量上，TCFM-5 相对等容量、等端口的 Linear-5 "
        f"减少 {linear_reduction:.2f}% product-delivery 周期；相对 "
        f"single-bank 减少 {single_reduction:.2f}%。这些数字只用于"
        "验证比较方法，不代表部署 trace。",
        "",
        f"Affine-4 在同一 gate-mask 定向流上相对 TCFM-5 多 "
        f"{affine4_overhead:.2f}% delivery 周期；它使用 4 个 bank，"
        "但部署窗口因容量不均衡需 480 个槽，反而高于 TCFM-5 的 450 个槽。"
        "是否节省宏外围功耗必须由同宏 PPA 判断。",
        "",
        "五色映射的架构主张不是经验性 hash：对每个 source，"
        "`{self,north,south,west,east}` 五个 destination 同时构成"
        "写冲突图中的五节点团，因此单周期无冲突至少需要 5 个"
        "一拍一更新的同步1R1W bank；"
        "`bank(x,y)=(x+2y) mod 5` 恰好使用 5 色，达到该端口模型下的"
        "最小 bank 数。",
        "",
        "## 开放结构统计",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| flatten 后 cells | {module['num_cells']} |",
        f"| wire bits | {module['num_wire_bits']} |",
        f"| `$mem_v2` | {types.get('$mem_v2', 0)} |",
        f"| `$mux` | {types.get('$mux', 0)} |",
        f"| `$dffe`/`$sdffe` | "
        f"{types.get('$dffe', 0) + types.get('$sdffe', 0)} |",
        "",
        "以上是 Yosys 开放结构统计，不是目标工艺面积、频率或功耗。",
        "",
        "## 证据边界",
        "",
        "- 当前输入是确定性定向向量，不是 post-G0 Local5 workload；",
        "- 随机反压已覆盖 term 边界，但尚未覆盖真实 T450 多窗口和外部 SRAM "
        "可变延迟；",
        "- 周期只覆盖 projection product-delivery，不含完整 encoder；",
        f"- 全 tile 实际 `projection_busy` 周期：TCFM-5={tcfm_tile_cycles}、"
        f"Affine-4={affine_tile_cycles}、Linear-5={linear_tile_cycles}、"
        f"Role-Sharded={role_tile_cycles}；该计数包含 Acc clear、producer "
        "反压和 close/drain，不含最终readback；",
        f"- 后端反压传播后的 relation stall：TCFM-5={stalls}、"
        f"Affine-4={affine_stalls}、Linear-5={linear_stalls}；",
        f"- Role-Sharded 的 relation stall={role_stalls}；其五份 partial-Acc "
        "还需在 readback 阶段做五路归约，该成本未计入上述 busy 周期；",
        f"- 本向量关系转置生产者 stall 为 {stalls}；K-bitmap "
        "next-event iterator 已跳过零 K lane，但单 descriptor capture "
        "和一 term/拍供给仍需进入端到端相序账本；",
        "- 目标 SRAM macro、DC/STA/SAIF 和 full-resolution 多样本 "
        "mean/p95/p99 仍未完成。",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
