#!/usr/bin/env python3
"""用真实 Local5 profile 评估首遍填充、后续重放的精确 relation vault。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TOKENS = 450
HEAD_DIM = 32
GATE_COUNT = 5
GATE_W = 9
SOURCE_ID_W = 9
LOGICAL_RECORD_BITS = SOURCE_ID_W + HEAD_DIM + GATE_COUNT * GATE_W + 5
RAW_RELATION_BITS = TOKENS * (HEAD_DIM + GATE_COUNT * GATE_W + 5)
RELATION_BUILD_CYCLES = TOKENS
CLEAR_CYCLES = 90
FINAL_DRAIN_CYCLES = TOKENS
RELATION_MACRO_CAPACITY_BITS = 57344
RELATION_MACRO_WORD_BITS = 112
RELATION_MACRO_DEPTH = 512
RELATION_MACRO_AREA_UM2 = 52084.928
DIRECT_TILE_TOTAL_AREA_UM2 = 1060747.884

STAGE_DEPTHS = (2, 2, 6, 2)
STAGE_WINDOWS = (440, 120, 30, 10)
STAGE_HEADS = (3, 6, 12, 24)
STAGE_CHANNELS = (96, 192, 384, 768)


@dataclass(frozen=True)
class Group:
    sample: int
    stage: int
    active_sources: int
    product_cycles: int

    @property
    def service_cycles(self) -> int:
        return 15 + self.product_cycles

    @property
    def packet_bits(self) -> int:
        return LOGICAL_RECORD_BITS * self.active_sources

    @property
    def packet_words(self) -> int:
        return self.active_sources

    @property
    def packet_storage_bits(self) -> int:
        return self.packet_words * RELATION_MACRO_WORD_BITS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values.astype(np.float64), q))


def load_groups(manifest_path: Path) -> tuple[list[Group], Path, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload_path = manifest_path.parent / manifest["payload_file"]
    payload = np.load(payload_path, mmap_mode="r")
    offsets = np.asarray(payload["descriptor_group_offsets"])
    source_term_count = np.asarray(payload["source_term_count"])
    groups: list[Group] = []
    for index, row in enumerate(manifest["groups"]):
        begin = int(offsets[index])
        end = int(offsets[index + 1])
        terms = source_term_count[begin:end]
        groups.append(
            Group(
                sample=int(row["sample"]),
                stage=int(row["stage"]),
                active_sources=int(np.count_nonzero(terms)),
                product_cycles=int(np.sum(terms, dtype=np.int64)),
            )
        )
    return groups, payload_path, manifest


def admit_packets(
    service_cycles: np.ndarray,
    packet_bits: np.ndarray,
    capacity_bits: int,
    policy: str,
) -> np.ndarray:
    """按 head 顺序首遍填充；不合适的 head 精确回退为重算。"""
    if policy not in {"first_fit_all", "critical_only"}:
        raise ValueError(f"未知 admission policy: {policy}")
    admitted = np.zeros(packet_bits.shape, dtype=np.bool_)
    used = 0
    for index, bits in enumerate(packet_bits):
        if policy == "critical_only" and int(service_cycles[index]) >= RELATION_BUILD_CYCLES:
            continue
        value = int(bits)
        if used + value <= capacity_bits:
            admitted[index] = True
            used += value
    return admitted


def online_admit_packets(
    service_cycles: np.ndarray,
    packet_storage_bits: np.ndarray,
    capacity_bits: int,
    policy: str,
) -> tuple[np.ndarray, int, int, int]:
    """模拟顺序 head 的 speculative write、commit/rollback 与容量 miss。"""
    if policy not in {"first_fit_all", "critical_only"}:
        raise ValueError(f"未知 admission policy: {policy}")
    admitted = np.zeros(packet_storage_bits.shape, dtype=np.bool_)
    committed_bits = 0
    speculative_words = 0
    discarded_words = 0
    capacity_misses = 0
    for index, storage_bits_raw in enumerate(packet_storage_bits):
        storage_bits = int(storage_bits_raw)
        remaining_bits = capacity_bits - committed_bits
        writable_bits = max(0, min(storage_bits, remaining_bits))
        written_words = writable_bits // RELATION_MACRO_WORD_BITS
        speculative_words += written_words
        fits = storage_bits <= remaining_bits
        critical = int(service_cycles[index]) < RELATION_BUILD_CYCLES
        should_commit = policy == "first_fit_all" or critical
        if fits and should_commit:
            admitted[index] = True
            committed_bits += storage_bits
        else:
            discarded_words += written_words
            if not fits:
                capacity_misses += 1
    return admitted, speculative_words, discarded_words, capacity_misses


def cycles_for_window(
    service_cycles: np.ndarray,
    packet_bits: np.ndarray,
    capacity_bits: int,
    policy: str = "critical_only",
) -> tuple[int, int, int, int, int]:
    """返回强重算基线、vault、驻留 head、关系生成次数和占用 bit。"""
    heads = int(service_cycles.size)
    output_tiles = heads
    admitted, _, _, _ = online_admit_packets(
        service_cycles, packet_bits, capacity_bits, policy
    )
    packet_words = packet_bits // RELATION_MACRO_WORD_BITS

    baseline = int(np.maximum(RELATION_BUILD_CYCLES, service_cycles).sum())
    baseline *= output_tiles

    # Vault 宏与 FCSR/TCFM5 分离，但每拍至多做一次原生 112-bit 1RW 访问。
    # +1 覆盖 head-directory commit；若访问超过 service，显式暴露到周期。
    first_tile_service = np.maximum(service_cycles, packet_words + 1)
    first_tile = int(
        np.maximum(RELATION_BUILD_CYCLES, first_tile_service).sum()
    )
    replay_service = np.maximum(service_cycles[admitted], packet_words[admitted] + 1)
    replay = int(replay_service.sum()) * (output_tiles - 1)
    fallback = int(
        np.maximum(RELATION_BUILD_CYCLES, service_cycles[~admitted]).sum()
    ) * (output_tiles - 1)
    vault = first_tile + replay + fallback

    fixed = output_tiles * (CLEAR_CYCLES + FINAL_DRAIN_CYCLES)
    baseline += fixed
    vault += fixed
    build_count = heads + int((~admitted).sum()) * (output_tiles - 1)
    return (
        baseline,
        vault,
        int(admitted.sum()),
        build_count,
        int(packet_bits[admitted].sum()),
    )


def simulate_stage(
    groups: list[Group],
    stage: int,
    capacity_bits: int,
    trials: int,
    seed: int,
    policy: str,
) -> dict:
    stage_groups = [group for group in groups if group.stage == stage]
    if not stage_groups:
        raise ValueError(f"stage {stage} 没有 profile group")
    heads = STAGE_HEADS[stage]
    # 所有容量和策略必须复用同一批 synthetic window，保证单变量比较。
    rng = np.random.default_rng(seed + stage * 1009)
    service_source = np.asarray(
        [group.service_cycles for group in stage_groups], dtype=np.int64
    )
    packet_source = np.asarray(
        [group.packet_storage_bits for group in stage_groups], dtype=np.int64
    )

    baseline_cycles = np.zeros(trials, dtype=np.int64)
    vault_cycles = np.zeros(trials, dtype=np.int64)
    resident_heads = np.zeros(trials, dtype=np.int64)
    build_counts = np.zeros(trials, dtype=np.int64)
    resident_bits = np.zeros(trials, dtype=np.int64)
    all_head_bits = np.zeros(trials, dtype=np.int64)
    speculative_words = np.zeros(trials, dtype=np.int64)
    discarded_words = np.zeros(trials, dtype=np.int64)
    capacity_misses = np.zeros(trials, dtype=np.int64)

    for trial in range(trials):
        selection = rng.integers(0, len(stage_groups), size=heads)
        service = service_source[selection]
        packets = packet_source[selection]
        (
            baseline_cycles[trial],
            vault_cycles[trial],
            resident_heads[trial],
            build_counts[trial],
            resident_bits[trial],
        ) = cycles_for_window(service, packets, capacity_bits, policy)
        (
            online_admitted,
            speculative_words[trial],
            discarded_words[trial],
            capacity_misses[trial],
        ) = online_admit_packets(service, packets, capacity_bits, policy)
        if int(online_admitted.sum()) != resident_heads[trial]:
            raise AssertionError("online admission 与周期模型不一致")
        all_head_bits[trial] = int(packets.sum())

    recompute_builds = heads * heads
    return {
        "stage": stage,
        "profile_groups": len(stage_groups),
        "heads": heads,
        "output_tiles": heads,
        "capacity_bits": capacity_bits,
        "capacity_kib": capacity_bits / 8192.0,
        "admission_policy": policy,
        "raw_all_head_worst_case_bits": heads * RAW_RELATION_BITS,
        "packed_all_head_worst_case_bits": heads
        * RELATION_MACRO_WORD_BITS
        * TOKENS,
        "all_head_fit_rate": float(np.mean(all_head_bits <= capacity_bits)),
        "resident_head_fraction_mean": float(np.mean(resident_heads) / heads),
        "resident_heads_p05": percentile(resident_heads, 5),
        "resident_heads_p50": percentile(resident_heads, 50),
        "all_head_packet_kib": {
            "mean": float(np.mean(all_head_bits) / 8192.0),
            "p95": percentile(all_head_bits, 95) / 8192.0,
            "p99": percentile(all_head_bits, 99) / 8192.0,
            "max_bootstrap": int(all_head_bits.max()) / 8192.0,
        },
        "window_cycles": {
            "recompute_overlap_mean": float(np.mean(baseline_cycles)),
            "vault_mean": float(np.mean(vault_cycles)),
            "speedup_mean_ratio": float(
                np.mean(baseline_cycles) / np.mean(vault_cycles)
            ),
            "vault_p95": percentile(vault_cycles, 95),
            "vault_p99": percentile(vault_cycles, 99),
        },
        "relation_builds": {
            "recompute_per_window": recompute_builds,
            "vault_mean_per_window": float(np.mean(build_counts)),
            "reduction": 1.0 - float(np.mean(build_counts)) / recompute_builds,
        },
        "resident_bits_mean": float(np.mean(resident_bits)),
        "online_port_model": {
            "macro_word_bits": RELATION_MACRO_WORD_BITS,
            "macro_depth": RELATION_MACRO_DEPTH,
            "speculative_write_words_mean": float(np.mean(speculative_words)),
            "discarded_write_words_mean": float(np.mean(discarded_words)),
            "capacity_misses_mean": float(np.mean(capacity_misses)),
            "pack_or_replay_port_stall_mean": float(
                np.mean(
                    np.maximum(
                        0,
                        np.asarray(
                            [group.packet_words + 1 for group in stage_groups],
                            dtype=np.int64,
                        )
                        - service_source,
                    )
                )
            ),
            "max_packet_words": int(
                max(group.packet_words for group in stage_groups)
            ),
        },
    }


def model_capacity(
    groups: list[Group], capacity_kib: int, trials: int, seed: int, policy: str
) -> dict:
    stage_results = []
    total_baseline = 0.0
    total_vault = 0.0
    total_build_baseline = 0.0
    total_build_vault = 0.0
    for stage in range(4):
        capacity_bits = int(capacity_kib) * 8192
        result = simulate_stage(groups, stage, capacity_bits, trials, seed, policy)
        stage_results.append(result)
        windows = STAGE_DEPTHS[stage] * STAGE_WINDOWS[stage]
        total_baseline += result["window_cycles"]["recompute_overlap_mean"] * windows
        total_vault += result["window_cycles"]["vault_mean"] * windows
        total_build_baseline += (
            result["relation_builds"]["recompute_per_window"] * windows
        )
        total_build_vault += (
            result["relation_builds"]["vault_mean_per_window"] * windows
        )
    macro_slices = math.ceil((int(capacity_kib) * 8192) / RELATION_MACRO_CAPACITY_BITS)
    incremental_macro_area = max(0, macro_slices - 1) * RELATION_MACRO_AREA_UM2
    area_ratio = (
        DIRECT_TILE_TOTAL_AREA_UM2 + incremental_macro_area
    ) / DIRECT_TILE_TOTAL_AREA_UM2
    speedup = total_baseline / total_vault
    return {
        "capacity": capacity_kib,
        "admission_policy": policy,
        "macro_proxy": {
            "relation_macro_slices": macro_slices,
            "incremental_macro_area_um2": incremental_macro_area,
            "area_ratio_vs_direct_tile": area_ratio,
            "throughput_per_area_ratio": speedup / area_ratio,
            "scope": "Nangate45+FakeRAM45 open proxy; not ASIC PPA",
        },
        "stage_results": stage_results,
        "frame_model": {
            "recompute_overlap_cycles_mean": total_baseline,
            "vault_cycles_mean": total_vault,
            "speedup": speedup,
            "relation_build_reduction": 1.0
            - total_build_vault / total_build_baseline,
            "relation_builds_recompute": total_build_baseline,
            "relation_builds_vault": total_build_vault,
        },
    }


def render_markdown(report: dict) -> str:
    existing = next(item for item in report["capacity_results"] if item["capacity"] == 7)
    lines = [
        "# Local5 首遍精确 Relation Vault DSE",
        "",
        "## 结论",
        "",
        "本轮先否决了简单的多 relation 槽分组驻留：当强基线允许双槽重叠 relation build 与投影时，分组方案需要跨 head-group 搬运 Acc32 部分和，面积归一吞吐没有稳定优势。",
        "",
        "更有价值的候选是 `[prof]+[模型]` 的暴露感知精确 relation memoization：第一个输出 tile 仍执行完整 score、Shiftmax5、relation transpose 和 source-major term；同时把活跃 source 写成连续 112-bit 原生记录 `{source-id, K32, 5xgate9, valid5}`。只有 `projection service < relation build=450 cycle`、即前端未被后端隐藏的 head 才尝试驻留。后续输出 tile 重放完全相同的 relation；容量不足或后端已经隐藏前端的 head 走精确重算。",
        "",
        f"在当前 profile100 的独立 bootstrap 模型中，使用现有 Direct 物理基线已经计入的 7 KiB relation 宏容量，整帧周期代理为 `{existing['frame_model']['speedup']:.3f}x`，relation build 次数减少 `{100*existing['frame_model']['relation_build_reduction']:.2f}%`，开放宏面积代理不增加。该结果不是 RTL 周期，也不是 ASIC PPA。",
        "",
        "## 数据与合同",
        "",
        f"- 输入：`{report['input']['manifest']}`；payload SHA256 `{report['input']['payload_sha256']}`。",
        f"- 真实 profile group：{report['input']['groups']}，T={TOKENS}，full-resolution。",
        f"- 每个活跃 source 使用一个原生 {RELATION_MACRO_WORD_BITS}-bit 物理记录；有效字段为 source-id {SOURCE_ID_W}、K32、五个 gate9 和 valid5，共 {LOGICAL_RECORD_BITS} bit，其余为 padding。",
        f"- 单 head 最坏占 {TOKENS} 行、{report['contracts']['packed_head_worst_storage_bits']} bit，仍可放入现有 512 行 relation 宏；多个 head 通过 512 行全局 committed pointer 共享容量。",
        "- 强基线按双上下文理想重叠计费：每个 head/output-tile 的前端与投影周期取 `max(450, service)`，不是串行相加。",
        "- admission 只依赖首遍已经精确观测的 service count，不预测后续数值；容量溢出采用 exact recompute fallback。",
        "- 旧 FCSR 只有三行 Q/K buffer，不能把整窗 `H x 450 x 64 bit` 当成空闲 scratch；此前的 Q/K 生命周期覆盖假设已否决。",
        "",
        "## 容量扫描",
        "",
        "| 容量 | 整帧周期加速 | relation build 减少 | 开放宏面积比 | 面积归一吞吐代理 |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in report["capacity_results"]:
        label = f"{item['capacity']} KiB"
        lines.append(
            f"| {label} | {item['frame_model']['speedup']:.3f}x | "
            f"{100*item['frame_model']['relation_build_reduction']:.2f}% | "
            f"{item['macro_proxy']['area_ratio_vs_direct_tile']:.3f}x | "
            f"{item['macro_proxy']['throughput_per_area_ratio']:.3f}x |"
        )
    all_policy = report["policy_ablation_7kib"]["first_fit_all"]
    critical_policy = report["policy_ablation_7kib"]["critical_only"]
    lines.extend(
        [
            "",
            "7 KiB 单变量 admission 消融：",
            "",
            "| 策略 | 周期加速 | relation build 减少 | 解释 |",
            "|---|---:|---:|---|",
            f"| first-fit all | {all_policy['frame_model']['speedup']:.3f}x | "
            f"{100*all_policy['frame_model']['relation_build_reduction']:.2f}% | "
            "缓存更多总工作，但可能挤掉关键路径上的稀疏 head |",
            f"| critical-only | {critical_policy['frame_model']['speedup']:.3f}x | "
            f"{100*critical_policy['frame_model']['relation_build_reduction']:.2f}% | "
            "只缓存 relation build 未被 projection 隐藏的 head |",
        ]
    )
    lines.extend(
        [
            "",
            "## 现有 7 KiB relation 宏容量的分 stage 结果",
            "",
            "| Stage | heads/tiles | 容量 | 全 head 包大小 mean/p95/p99 | 驻留 head 比例 | 周期加速 | build 减少 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for stage in existing["stage_results"]:
        packet = stage["all_head_packet_kib"]
        lines.append(
            f"| S{stage['stage']} | {stage['heads']} | {stage['capacity_kib']:.1f} KiB | "
            f"{packet['mean']:.1f}/{packet['p95']:.1f}/{packet['p99']:.1f} KiB | "
            f"{100*stage['resident_head_fraction_mean']:.2f}% | "
            f"{stage['window_cycles']['speedup_mean_ratio']:.3f}x | "
            f"{100*stage['relation_builds']['reduction']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "112-bit 1RW 在线相序分账（每 synthetic window 均值）：",
            "",
            "| Stage | speculative write words | discarded words | capacity misses | pack/replay port stall |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for stage in existing["stage_results"]:
        port = stage["online_port_model"]
        lines.append(
            f"| S{stage['stage']} | {port['speculative_write_words_mean']:.2f} | "
            f"{port['discarded_write_words_mean']:.2f} | "
            f"{port['capacity_misses_mean']:.3f} | "
            f"{port['pack_or_replay_port_stall_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 架构解释",
            "",
            "1. `first tile observe`：正常执行 Local5 前端；同时累计该 head 的 exact product-service count，并把候选 source descriptor 顺序写入可回滚区域。",
            "2. `exposure-aware commit`：首遍必须由尚未正式集成的 FCSR 三行在线转置生产 descriptor，使已有 7 KiB relation 宏可专用于候选 packet。head 结束时，若 service 小于 450 且 packet fits，则原子提交；否则回滚写指针。",
            "   在线实现使用 `committed_ptr/speculative_ptr`。每个活跃 source 直接写一个 112-bit 原生宏字，source-id 和 valid-mask 填入现有 gate 宏的 padding。非 critical 或 overflow head 在边界恢复 speculative pointer，后续 head 严格在 rollback 后启动。",
            "3. `exact replay`：后续输出 tile 按 head directory 的 base/length 顺序读取 source/K/gate/valid 记录；权重和 output-tile base 改变，relation 不变。",
            "4. `fallback`：容量不足的 head 不驻留，后续 tile 走原 score/Shiftmax/relation 路径。重放与回退共享全局 head 序，保证可审计。",
            "",
            "这不是把 FIFO、banking 或双缓冲单列成贡献。可候选的架构主张是：利用 Local5 relation 对输出通道 tile 的不变性，并用首遍测得的前后端服务差决定是否驻留，把有限 relation SRAM 只分配给真正暴露在关键路径上的 exact topology operand。TTB/Phi 提供打包范式，Prosperity 提供 exact reuse 范式，Bishop 提供 density stratification 启发；本土化差异是五邻域 relation、output-tile 不变性和 latency-exposure admission 的联合合同。",
            "",
            "## 负结果与限制",
            "",
            "- bootstrap 从同 stage 的 profile group 独立抽取 head，不能替代同一真实 window 的全 head 联合分布；全拟合率为 `[模型]`。",
            "- 当前 Direct 路径的完整 relation plane 会占用这组 7 KiB 宏；候选必须先闭合 FCSR 三行在线转置，才能把宏改作 memoization。FCSR 与 memoization 尚未形成单顶层。",
            "- 模型已计 112-bit 原生记录、speculative/discarded writes 和 replay read 下界，但还不是 RTL；FCSR ring、head directory、双指针和控制标准单元面积未计。",
            "- 周期模型给了重算基线理想双上下文重叠；未包含 SRAM 真实延迟、随机反压、全 encoder 调度和片外流量。",
            "- build 次数减少不是能耗结果；没有 SAIF/DC/PTPX 前不得换算成功耗或 EDP。",
            "- 若新 rank-1 的真实同 window 全 head profile 使 critical-head 拟合率或面积归一收益不足，该机制应降级或淘汰。",
            "",
            "## 晋级门槛",
            "",
            "1. 新 rank-1 导出同一 window 的完整 head 集合，实测 packet occupancy、critical-head 拟合率和 exact fallback 比例。",
            "2. 先证明 FCSR 三行 source frontier 能替代完整 relation plane，再做 pack/replay；不得把两个未集成模块同时视为闭环。",
            "3. 叶级 pack/replay miter 覆盖 450 source、容量边界、回退、随机反压，Acc32 与重算路径逐项一致。",
            "4. 集成后同时报告 score/Shiftmax/relation 执行次数、vault SRAM 读写、总周期和 stall，不只报前端 work。",
            "5. 与理想双槽重算基线在相同 SRAM 宏、频率和反压下比较；至少需要非负面积归一吞吐，并在 SAIF/PTPX 后证明 EDP 收益。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "results/local5_fullres_postg0_qfsa_profile100_20260730/"
            "ordered_term_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/local5_relation_vault_dse_20260806"),
    )
    parser.add_argument(
        "--capacities-kib",
        type=int,
        nargs="+",
        default=[4, 6, 7, 8, 16, 24, 32, 48, 64],
    )
    parser.add_argument("--trials", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()

    groups, payload_path, manifest = load_groups(args.manifest)
    capacity_results = [
        model_capacity(groups, capacity, args.trials, args.seed, "critical_only")
        for capacity in args.capacities_kib
    ]
    policy_ablation = {
        policy: model_capacity(groups, 7, args.trials, args.seed, policy)
        for policy in ("first_fit_all", "critical_only")
    }

    report = {
        "schema": "local5_exposure_aware_relation_memoization_dse_v2",
        "status": "PROFILE_MODEL_COMPLETE_NOT_RTL",
        "evidence": "[prof]+[模型]+[待验证]",
        "input": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "payload": str(payload_path.resolve()),
            "payload_sha256": sha256(payload_path),
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "groups": len(groups),
            "samples": len({group.sample for group in groups}),
            "full_resolution": manifest["resolution"]["full_resolution"],
        },
        "contracts": {
            "tokens": TOKENS,
            "head_dim": HEAD_DIM,
            "gate_width": GATE_W,
            "logical_record_bits": LOGICAL_RECORD_BITS,
            "source_id_width": SOURCE_ID_W,
            "packed_head_worst_bits": LOGICAL_RECORD_BITS * TOKENS,
            "packed_head_worst_storage_bits": RELATION_MACRO_WORD_BITS * TOKENS,
            "raw_relation_bits": RAW_RELATION_BITS,
            "relation_build_cycles": RELATION_BUILD_CYCLES,
            "strong_baseline": "two-context ideal overlap: max(build, projection service)",
            "fallback": "capacity miss recomputes exact relation on later output tiles",
            "admission": "commit only when exact first-pass service < 450 cycles and packet fits",
            "relation_macro_capacity_bits": RELATION_MACRO_CAPACITY_BITS,
            "relation_macro_word_bits": RELATION_MACRO_WORD_BITS,
            "relation_macro_depth": RELATION_MACRO_DEPTH,
            "relation_macro_area_um2": RELATION_MACRO_AREA_UM2,
            "direct_tile_total_area_um2": DIRECT_TILE_TOTAL_AREA_UM2,
        },
        "capacity_results": capacity_results,
        "policy_ablation_7kib": policy_ablation,
        "decision": {
            "grouped_relation_slots": "REJECT_BEFORE_RTL",
            "qk_liveness_overlay": "REJECT_UNDER_EXISTING_THREE_LINE_FCSR",
            "exposure_aware_relation_memoization": "CONDITIONAL_CANDIDATE",
            "next_gate": "new-rank1 same-window all-head joint profile and leaf pack/replay miter",
        },
        "limitations": [
            "head joint distribution is bootstrap-modeled from real per-head groups",
            "whole-window Q/K liveness overlay was rejected under the existing three-line FCSR",
            "existing relation macro port reuse is not yet closed in RTL",
            "cycle estimates are not RTL, ASIC PPA, power, or EDP",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(json.dumps(report["decision"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
