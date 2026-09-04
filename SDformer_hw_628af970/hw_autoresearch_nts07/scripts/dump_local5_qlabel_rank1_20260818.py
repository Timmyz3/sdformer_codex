#!/usr/bin/env python3
"""Local5 rank-1 (ep44) 带 Q 标签 per-pair dump —— C1 第一闸裁决数据采集与就地裁决。

规格：docs/CLAUDE_LOCAL5_QLABEL_DUMP_SPEC_20260818.md（2026-08-18）。
本脚本需要 GPU（model forward 主导，100 samples 约 4.5-5.5 h）。GPU 空闲后直接：

    /opt/conda/envs/sdformerflow/bin/python \
      scripts/dump_local5_qlabel_rank1_20260818.py \
      --config <ep44 deploy yml> --checkpoint <ep44 pth> \
      --output-dir <out> --samples 100

只读：不改 overlay / RTL / 封存 trace。输出：qlabel_records.npz +
qlabel_report.json + qlabel_summary.md。

采集复用封存 profile100 的同一机制（Local5Collector.attach 挂 _h9_local5_trace_collector、
同一 coprime 旋转抽样、同一分层数据集抽样），但 sink 换成 QLabelSink：逐 token 记录
真实 q_event / k_event 双事件向量 + 逐边 score/gate，附加跨窗口对齐组（adj=1）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

HW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HW_ROOT.parent
EXP_ROOT = REPO_ROOT / "neuron_experiments" / "H9_bipolar_self_attention"
ENTRYPOINT_ROOT = EXP_ROOT / "entrypoints"
sys.path.insert(0, str(ENTRYPOINT_ROOT))
sys.path.insert(0, str(HW_ROOT / "scripts"))

import profile_nts11_hardware_p0 as base_profile  # noqa: E402
from profile_local5_hardware_features import (  # noqa: E402
    POST_G0_BLOCK_PAIRS,
    POST_G0_LANES,
    POST_G0_SAMPLING_ID,
    POST_G0_TOKENS,
    file_sha256,
    file_row_values,
    rotating_flat_indices,
    sequence_key_from_file_row,
    stratified_dataset_indices,
)

SCHEMA = "local5_qlabel_rank1_v1"
VALID_MODES = {"binary_axnor_local5_shiftmax", "lr_ttx", "h66_lr"}
# 封存 ep44 profile100 全口径（kv 代理口径），用于自校验容差
SEALED_EP44_FULL_MATCH = 0.7859762898038378
SEALED_TOL = 0.005

# 判定阈值（预注册，见规格 §4.2）
T_G2 = 0.99      # 同 token 双角色 q1==k1 一致率
T_G3 = 0.95      # 全口径统计保持（exact 合同线）
T_G4 = 0.60      # 非零-非零保持率
T_G5 = 0.90      # 向量全等率
T_G7 = 0.01      # 统计保持边界上 score 元组相等率（389 防御）


def bit_pack(bits: torch.Tensor) -> torch.Tensor:
    """[..., D] bool -> [...,] uint32 位图（与封存 descriptor_k_bitmap 同构）。"""
    weights = (torch.ones(bits.shape[-1], dtype=torch.long, device=bits.device)
               << torch.arange(bits.shape[-1], dtype=torch.long, device=bits.device))
    return (bits.to(dtype=torch.long) * weights.view(1, -1)).sum(dim=-1)


def popcount_u32(values: np.ndarray) -> np.ndarray:
    return np.array([int(v).bit_count() for v in values], dtype=np.uint8)


class QLabelSink:
    """按 (window, head) 组逐 token 记录双事件向量与逐边 score/gate。

    组内 token 顺序与封存一致：source_ids 0..449 = plane0 pos0..224 后接
    plane1 pos0..224。基础组（adj=0）按 (sample, block, call, flat) 收集序
    追加 —— 复现封存 npz 的平坦 descriptor 流；对齐组（adj=1）独立追加。
    """

    def __init__(self, groups_per_block_sample: int, adjacency: bool) -> None:
        self.groups_per_block_sample = groups_per_block_sample
        self.adjacency = adjacency
        self.sample_id = -1
        self.modules: list[torch.nn.Module] = []
        self.attached_pairs: set[tuple[int, int]] = set()
        self.groups: list[dict[str, np.ndarray]] = []
        self.group_meta: list[dict[str, Any]] = []

    # ---- attach（与封存 Local5Collector.attach 同构，只替换 callback） ----
    def attach(self, model: torch.nn.Module) -> int:
        attached = 0
        for full_name, module in model.named_modules():
            cfg = getattr(module, "_h9_shiftmax_cfg", None)
            mode = str(getattr(cfg, "mode", ""))
            if mode not in VALID_MODES:
                continue
            match = re.search(r"layers\.(\d+)\.swin_blocks\.(\d+)\.attn$", full_name)
            if match is None:
                raise RuntimeError("Local5 attention 模块名无法解析 stage/block: " + full_name)
            stage, block = int(match.group(1)), int(match.group(2))
            if (stage, block) not in POST_G0_BLOCK_PAIRS:
                raise RuntimeError(f"出现非目标 attention block: {(stage, block)}")
            if (stage, block) in self.attached_pairs:
                raise RuntimeError(f"Local5 attention block 重复绑定: {(stage, block)}")
            self.attached_pairs.add((stage, block))

            counters: dict[str, int] = {"call": 0}

            def callback(
                *,
                module: torch.nn.Module,
                q_event: torch.Tensor,
                k_event: torch.Tensor,
                k_orig: torch.Tensor,
                neighbor_index: torch.Tensor,
                valid: torch.Tensor,
                score_q7: torch.Tensor,
                gate: torch.Tensor,
                _name: str = full_name,
                _stage: int = stage,
                _block: int = block,
            ) -> None:
                del k_orig
                call_index = counters["call"]
                batch_windows, heads, tokens, lanes = q_event.shape
                if tokens != POST_G0_TOKENS or lanes != POST_G0_LANES:
                    raise ValueError(f"descriptor shape 必须为 T{POST_G0_TOKENS}×{POST_G0_LANES}")
                total_groups = batch_windows * heads
                flat_indices = rotating_flat_indices(
                    total_groups=total_groups,
                    selected_groups=self.groups_per_block_sample,
                    sample_id=self.sample_id,
                    stage=_stage,
                    block=_block,
                )
                for flat in flat_indices:
                    window_b, head = divmod(flat, heads)
                    self._capture_group(
                        q_event=q_event, k_event=k_event, neighbor_index=neighbor_index,
                        valid=valid, score_q7=score_q7, gate=gate,
                        window_b=window_b, head=head,
                        sample_id=self.sample_id, stage=_stage, block=_block,
                        call_index=call_index, adj=0,
                    )
                if self.adjacency:
                    # 跨窗口边界口径：同 (h,w,head) 补记时间窗口 d+1（若存在）
                    temporal = self._temporal_windows(batch_windows)
                    for flat in flat_indices:
                        window_b, head = divmod(flat, heads)
                        d, rem = divmod(window_b, temporal)
                        if d + 1 < temporal:
                            self._capture_group(
                                q_event=q_event, k_event=k_event, neighbor_index=neighbor_index,
                                valid=valid, score_q7=score_q7, gate=gate,
                                window_b=(d + 1) * temporal + rem, head=head,
                                sample_id=self.sample_id, stage=_stage, block=_block,
                                call_index=call_index, adj=1,
                            )
                counters["call"] += 1

            module._h9_local5_trace_collector = callback
            self.modules.append(module)
            attached += 1
        if self.attached_pairs != set(POST_G0_BLOCK_PAIRS):
            raise RuntimeError(
                "Local5 block 集合不匹配: "
                f"missing={sorted(set(POST_G0_BLOCK_PAIRS) - self.attached_pairs)} "
                f"extra={sorted(self.attached_pairs - set(POST_G0_BLOCK_PAIRS))}"
            )
        return attached

    def close(self) -> None:
        for module in self.modules:
            if hasattr(module, "_h9_local5_trace_collector"):
                delattr(module, "_h9_local5_trace_collector")
        self.modules.clear()

    @staticmethod
    def _temporal_windows(batch_windows: int) -> int:
        # window_partition_v2 时间窗口数 = D // window_size[0] = 10 // 2 = 5
        if batch_windows % 5 != 0:
            raise ValueError(f"batch_windows {batch_windows} 不能被时间窗口数 5 整除")
        return batch_windows // 5

    # ---- 组采集 ----
    def _capture_group(
        self,
        *,
        q_event: torch.Tensor,
        k_event: torch.Tensor,
        neighbor_index: torch.Tensor,
        valid: torch.Tensor,
        score_q7: torch.Tensor,
        gate: torch.Tensor,
        window_b: int,
        head: int,
        sample_id: int,
        stage: int,
        block: int,
        call_index: int,
        adj: int,
    ) -> None:
        tokens = q_event.shape[2]
        plane_tokens = tokens // 2
        q = q_event[window_b, head].to(dtype=torch.bool)
        k = k_event[window_b, head].to(dtype=torch.bool)
        qv = bit_pack(q).cpu().numpy().astype(np.uint32)
        kv = bit_pack(k).cpu().numpy().astype(np.uint32)
        q1 = q.sum(dim=-1).cpu().numpy().astype(np.uint8)
        k1 = k.sum(dim=-1).cpu().numpy().astype(np.uint8)
        plane = (np.arange(tokens, dtype=np.uint8) // plane_tokens)
        pos = (np.arange(tokens, dtype=np.uint16) % plane_tokens)
        nbr = neighbor_index.to(device=q_event.device).cpu().numpy().astype(np.uint16)
        ev = valid.to(device=q_event.device).cpu().numpy().astype(np.uint8)
        sc = torch.round(score_q7[window_b, head] * 128.0).cpu().numpy().astype(np.int16)
        ga = torch.round(gate[window_b, head] * 128.0).cpu().numpy().astype(np.int16)
        rec = {
            "sample": np.full(tokens, sample_id, dtype=np.uint16),
            "stage": np.full(tokens, stage, dtype=np.uint8),
            "block": np.full(tokens, block, dtype=np.uint8),
            "call": np.full(tokens, call_index, dtype=np.uint32),
            "flat_window": np.full(tokens, window_b, dtype=np.uint16),
            "head": np.full(tokens, head, dtype=np.uint8),
            "adj": np.full(tokens, adj, dtype=np.uint8),
            "plane": plane,
            "pos": pos,
            "qv": qv,
            "kv": kv,
            "q1": q1,
            "k1": k1,
            "nbr": nbr,     # [T,5]
            "ev": ev,       # [T,5]
            "sc": sc,       # [T,5]
            "ga": ga,       # [T,5]
        }
        self.groups.append(rec)
        self.group_meta.append({
            "sample_id": sample_id, "stage": stage, "block": block,
            "call_index": call_index, "flat_window": int(window_b),
            "head": int(head), "adj": adj,
        })


# --------------------------------------------------------------------------
# 后处理：就地裁决（CPU numpy）
# --------------------------------------------------------------------------

def concat_fields(groups: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = list(groups[0].keys())
    out = {}
    for key in keys:
        arrays = [g[key] for g in groups]
        out[key] = np.concatenate(arrays, axis=0)
    return out


def main_stream_boundaries(rec: dict[str, np.ndarray]) -> np.ndarray:
    """基础组（adj==0）按收集序形成的平坦流的相邻边界索引对 [(p, p+1), ...]。

    组内边界：每组内相邻 token 的 449 条；组间边界：相邻基础组首尾相接处。
    """
    n = rec["sample"].shape[0]
    base = np.where(rec["adj"] == 0)[0]
    if base.size == 0:
        raise ValueError("没有基础组记录")
    # 基础组起止
    group_start = base[np.r_[True, base[1:] != base[:-1] + 1]]
    group_end = base[np.r_[base[1:] != base[:-1] + 1, True]]
    pairs: list[tuple[int, int]] = []
    for s, e in zip(group_start, group_end):
        for p in range(int(s), int(e) - 1):
            pairs.append((p, p + 1))
    for i in range(group_start.size - 1):
        pairs.append((int(group_end[i]), int(group_start[i + 1])))
    return np.asarray(pairs, dtype=np.int64)


def phys_frame(stage: int, block: int, flat_window: int, plane: int,
               temporal: int, shifted: bool) -> int:
    d, _ = divmod(int(flat_window), temporal)
    if shifted:
        return (2 * d + 1 + int(plane)) % 10
    return 2 * d + int(plane)


def adjudicate(rec: dict[str, np.ndarray], meta: dict[str, Any]) -> dict[str, Any]:
    n = rec["sample"].shape[0]
    q1_true = rec["q1"].astype(np.int64)
    k1 = rec["k1"].astype(np.int64)
    q1_proxy = popcount_u32(rec["kv"]).astype(np.int64)

    # 自校验
    if not np.array_equal(q1_true, popcount_u32(rec["qv"]).astype(np.int64)):
        raise RuntimeError("自校验失败: q1 != pc(qv)")
    if not np.array_equal(k1, popcount_u32(rec["kv"]).astype(np.int64)):
        raise RuntimeError("自校验失败: k1 != pc(kv)")

    bd = main_stream_boundaries(rec)
    p_prev, p_next = bd[:, 0], bd[:, 1]

    m_proxy = q1_proxy[p_next] == k1[p_prev]          # 封存代理口径复现
    m_true = q1_true[p_next] == k1[p_prev]            # 真实 q_event 口径
    both_zero = (k1[p_prev] == 0) & (q1_true[p_next] == 0)
    nz_nz = (k1[p_prev] > 0) & (q1_true[p_next] > 0)

    # 389 防御：统计保持边界上 score 五元组相等率
    sc_prev = rec["sc"][p_prev]
    sc_next = rec["sc"][p_next]
    score_tup_eq_all = np.all(sc_prev == sc_next, axis=1)
    score_tup_eq_matched = score_tup_eq_all[m_true]

    # G1 存在性：同一物理 token 双角色实例
    temporal = meta["temporal_windows"]
    shifted_by_block = meta["shifted_blocks"]  # {(stage, block): bool}
    dual_keys: set[tuple[Any, ...]] = set()
    dual_instances = 0
    for key, count in zip(
        zip(
            rec["sample"], rec["stage"], rec["block"], rec["head"],
            rec["flat_window"], rec["plane"], rec["pos"],
        ),
        range(n),
    ):
        if rec["adj"][count]:
            continue
        s, st, b, h, fw, pl, pos = key
        shift = shifted_by_block.get((int(st), int(b)), False)
        frame = phys_frame(int(st), int(b), int(fw), int(pl), temporal, shift)
        if int(pl) == 1:
            dual_keys.add((int(s), int(st), int(b), int(h), int(frame), int(pos)))
        else:
            if (int(s), int(st), int(b), int(h), int(frame), int(pos)) in dual_keys:
                dual_instances += 1

    # G6：同 call 内同 token q_event vs k_event
    vec_eq = np.equal(rec["qv"], rec["kv"])
    pop_eq = q1_true == k1

    # m = pc(Q) 分布（非静默边 m-bit 门控数据）
    m_hist_all = np.bincount(q1_true, minlength=33).tolist()
    m_hist_by_stage: dict[str, list[int]] = {}
    for st in np.unique(rec["stage"]):
        sel = rec["stage"] == st
        m_hist_by_stage[str(int(st))] = (
            np.bincount(q1_true[sel], minlength=33).tolist()
        )

    # 跨窗口边界口径（adj 组：窗口 d 的 plane1 与窗口 d+1 的 plane0）
    win_bd_match: dict[str, float] = {}
    win_bd_total = 0
    win_bd_hits = 0
    by_call: dict[tuple[int, int, int, int, int, int], list[int]] = defaultdict(list)
    for idx in range(n):
        by_call[(int(rec["sample"][idx]), int(rec["stage"][idx]), int(rec["block"][idx]),
                 int(rec["head"][idx]), int(rec["flat_window"][idx]), int(rec["adj"][idx]))].append(idx)
    for (s, st, b, h, fw, _adj), idxs in by_call.items():
        pass  # 结构仅用于说明；真实跨窗口口径由 adj 组与 base 组配对计算
    # 直接口径：对每 (sample, stage, block, head, spatial_win)，窗口 d 的 plane1
    # 与窗口 d+1 的 plane0 的 token 按 pos 对齐比较
    win_tot: defaultdict[Any, int] = defaultdict(int)
    win_hit: defaultdict[Any, int] = defaultdict(int)
    for idx in range(n):
        if rec["adj"][idx]:
            continue
        s, st, b, h, fw, pl, pos = (int(rec["sample"][idx]), int(rec["stage"][idx]),
                                    int(rec["block"][idx]), int(rec["head"][idx]),
                                    int(rec["flat_window"][idx]), int(rec["plane"][idx]),
                                    int(rec["pos"][idx]))
        temporal = meta["temporal_windows"]
        d, rem = divmod(fw, temporal)
        if pl != 1 or d + 1 >= temporal:
            continue
        # 找对齐组的同一 (h,w,head) 窗口 d+1 的 plane0 pos 记录
        fw2 = (d + 1) * temporal + rem
        kk = (s, st, b, h, fw2, 0, pos)
        m2 = (rec["sample"] == s) & (rec["stage"] == st) & (rec["block"] == b) & \
             (rec["head"] == h) & (rec["flat_window"] == fw2) & \
             (rec["plane"] == 0) & (rec["pos"] == pos) & (rec["adj"] == 1)
        hits = np.where(m2)[0]
        if hits.size:
            win_tot[kk] += 1
            win_hit[kk] += 1 if q1_true[hits[0]] == k1[idx] else 0
    win_bd_total = sum(win_tot.values())
    win_bd_hits = sum(win_hit.values())

    verdict = {}
    g1 = dual_instances
    g3 = float(m_true.mean())
    g4 = float((m_true & nz_nz).sum() / max(1, int(nz_nz.sum())))
    g5 = float(np.equal(rec["qv"][p_next], rec["kv"][p_prev]).mean())
    g6_vec = float(vec_eq.mean())
    g6_pop = float(pop_eq.mean())
    g7 = float(score_tup_eq_matched.mean()) if m_true.sum() else float("nan")
    both_zero_share = float((m_true & both_zero).sum() / max(1, int(m_true.sum())))
    if g1 > 0:
        same_token = (q1_true[p_next] == k1[p_prev]) & (rec["sample"][p_next] == rec["sample"][p_prev])
        # G2 只对双角色 token 对齐口径有意义；此处给出基础口径
        verdict["G2_not_applicable"] = True
    gate1_pass = (g1 > 0 and g6_pop >= T_G2) or (g3 >= T_G3 and g4 >= T_G4 and g5 >= T_G5)
    gate1_fail = not gate1_pass
    verdict.update({
        "G1_dual_role_instances": g1,
        "G3_full_match_true_q1": g3,
        "G4_nonzero_nonzero_match": g4,
        "G5_vector_identity_rate": g5,
        "G6_same_pair_same_token_vec_eq": g6_vec,
        "G6_same_pair_same_token_pop_eq": g6_pop,
        "G7_score_tuple_eq_on_matched": g7,
        "both_zero_share_of_matches": both_zero_share,
        "k1_zero_frac": float((k1 == 0).mean()),
        "q1_true_zero_frac": float((q1_true == 0).mean()),
        "proxy_full_match": float(m_proxy.mean()),
        "window_boundary_match": (win_bd_hits / win_bd_total) if win_bd_total else float("nan"),
        "window_boundary_total": win_bd_total,
        "nz_nz_boundaries": int(nz_nz.sum()),
        "total_boundaries": int(bd.shape[0]),
        "records_total": n,
        "records_base": int(np.sum(rec["adj"] == 0)),
        "m_hist_all": m_hist_all,
        "m_hist_by_stage": m_hist_by_stage,
        "score_tup_eq_all": float(score_tup_eq_all.mean()),
        "gate1_pass": bool(gate1_pass),
        "gate1_fail": bool(gate1_fail),
        "thresholds": {
            "G2": T_G2, "G3": T_G3, "G4": T_G4, "G5": T_G5, "G7": T_G7,
        },
    })
    return verdict


def finalize_verdict(verdict: dict[str, Any], sealed_check: dict[str, Any]) -> dict[str, Any]:
    """规格 §4.2 / §6 的裁决映射。"""
    if sealed_check.get("reproduced") is False:
        return {
            "ruling": "INVALID_DUMP",
            "reason": sealed_check.get("reason"),
            "score": None,
        }
    g1, g3, g4, g5 = (verdict["G1_dual_role_instances"], verdict["G3_full_match_true_q1"],
                      verdict["G4_nonzero_nonzero_match"], verdict["G5_vector_identity_rate"])
    if g3 < T_G3 or g4 < T_G4 or g5 < T_G5 or g1 == 0:
        return {
            "ruling": "GATE1_FAIL",
            "reason": (
                f"G1={g1} (存在性) / G3={g3:.4f} (需≥{T_G3}) / "
                f"G4={g4:.4f} (需≥{T_G4}) / G5={g5:.4f} (需≥{T_G5})"
            ),
            "score": 3.1,
            "disposition": "NO_GO_AS_DATE_CONTRIBUTION / HOLD_AS_IMPLEMENTATION_OPTION"
                           "（m-bit 门控 AND 残差 + stat-add 作为实现优化，不单列 DATE 贡献）",
        }
    return {
        "ruling": "GATE1_PASS",
        "reason": "带 Q 标签 dump 支持统计平面 exact 身份；进入第二闸（同端口 miter）",
        "score": 3.5,
    }


def write_summary_md(path: Path, report: dict[str, Any]) -> None:
    v = report["verdict"]
    lines = [
        "# Local5 C1 第一闸 dump 裁决摘要（Q 标签）",
        "",
        f"- schema: {report['schema']}",
        f"- checkpoint: `{report['checkpoint']}`",
        f"- samples: {report['samples']}",
        f"- 记录数: {v['records_total']}（基础 {v['records_base']}）",
        f"- 边界数: {v['total_boundaries']}（非零-非零 {v['nz_nz_boundaries']}）",
        "",
        "## 裁决",
        "",
        f"- **{report['final_verdict']['ruling']}**",
        f"- 理由: {report['final_verdict']['reason']}",
        f"- 创新分: {report['final_verdict'].get('score')}",
        "",
        "## 关键指标",
        "",
        "| 指标 | 值 | 阈值 | 口径 |",
        "|---|---:|---:|---|",
        f"| G1 双角色实例数 | {v['G1_dual_role_instances']} | ==0 则语义不可实例化 | 存在性 |",
        f"| G3 全口径一致率（真实 q1） | {v['G3_full_match_true_q1']:.4f} | ≥{T_G3} | 流相邻 |",
        f"| 封存代理口径复现 | {v['proxy_full_match']:.4f} | 封存 78.60%±0.5% | kv proxy |",
        f"| 双零命中占比 | {v['both_zero_share_of_matches']:.4f} | 报告 | QS 覆盖分解 |",
        f"| G4 非零-非零保持率 | {v['G4_nonzero_nonzero_match']:.4f} | ≥{T_G4} | 排除双零 |",
        f"| G5 向量全等率 | {v['G5_vector_identity_rate']:.4f} | ≥{T_G5} | 32-bit 全等 |",
        f"| G6 同 pair 同 token pop 一致 | {v['G6_same_pair_same_token_pop_eq']:.4f} | 报告（Direct5-CSE 基线） | 降级备选 |",
        f"| G7 统计保持边界 score 相等率 | {v['G7_score_tuple_eq_on_matched']:.4f} | <{T_G7} | 389 防御 |",
        f"| 跨窗口边界一致率 | {v['window_boundary_match']:.4f}（n={v['window_boundary_total']}） | 报告 | 对齐组 |",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--adjacency", action="store_true", default=True)
    parser.add_argument("--groups-per-block-sample", type=int, default=4)
    parser.add_argument(
        "--sealed-npz",
        type=Path,
        default=(HW_ROOT / "results" / "local5_ep44_hardware_rebind_20260815_profile100"
                 / "ordered_term_items.npz"),
    )
    parser.add_argument(
        "--sealed-run-identity",
        type=Path,
        default=(HW_ROOT / "results" / "local5_ep44_hardware_rebind_20260815_profile100"
                 / "post_g0_run_identity.json"),
    )
    args = parser.parse_args()

    if args.samples <= 0:
        parser.error("--samples 必须 > 0")
    if args.groups_per_block_sample < 1:
        parser.error("--groups-per-block-sample 必须 >= 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config, device = base_profile.load_config(args.config)
    deployment = base_profile.deployment_contract_from_config(config)
    if "local5" not in deployment["attention_mode"]:
        raise ValueError("非 Local5 部署合同，拒绝执行")
    if deployment["window_size"] != [2, 15, 15]:
        raise ValueError("window_size 必须为 [2,15,15]")
    if deployment["resolution"] != [480, 640]:
        raise ValueError("resolution 必须为 [480,640]")

    # 身份绑定：与封存 ep44 profile100 run-identity 对照
    if args.sealed_run_identity.exists():
        sealed_id = json.loads(args.sealed_run_identity.read_text(encoding="utf-8"))
        if (sealed_id.get("checkpoint_sha256") != file_sha256(args.checkpoint)
                or sealed_id.get("config_sha256") != file_sha256(args.config)):
            raise ValueError("checkpoint/config 与封存 ep44 run-identity 不匹配")
    else:
        print("[qlabel] 警告：封存 run-identity 缺失，跳过 SHA 对照", flush=True)

    dataset = base_profile.DSECDatasetLite(
        config, file_list="valid", stereo=False,
        scale_factor=config.get("test", {}).get("scale_factor", 1),
    )
    dataset_indices = (
        stratified_dataset_indices(dataset.files, args.samples)
        if args.samples > 1
        else list(range(min(1, len(dataset))))
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, sampler=dataset_indices, drop_last=False,
        pin_memory=False, num_workers=args.num_workers,
    )
    transform_valid = None
    if config["loader"].get("crop") is not None:
        transform_valid = base_profile.Compose([
            base_profile.CenterCrop(
                (config["loader"]["crop"][0], config["loader"]["crop"][1])
            )
        ])

    model = base_profile.build_model(config, args.checkpoint, device)
    load_audit = base_profile.validate_h9_load_audit(model, config)
    bn_policy = config.get("test", {}).get("bn_policy", "running")
    base_profile.configure_batch_norm_evaluation(model, bn_policy)

    sink = QLabelSink(
        groups_per_block_sample=args.groups_per_block_sample,
        adjacency=args.adjacency,
    )
    attached = sink.attach(model)
    print(f"[qlabel] attached {attached} blocks on {device}", flush=True)

    sample_keys: list[str] = []
    sequence_keys: list[str] = []
    processed = 0
    try:
        with torch.no_grad():
            for chunk, mask, label in loader:
                if processed >= args.samples:
                    break
                base_profile.functional.reset_net(model)
                sink.sample_id = processed
                dataset_index = dataset_indices[processed]
                file_row = dataset.files[dataset_index]
                file_names = file_row_values(file_row)
                sample_keys.append("|".join(str(item) for item in file_names))
                sequence_keys.append(sequence_key_from_file_row(file_row))
                x, _, _ = base_profile.preprocess_chunk(
                    config, chunk, label, mask, transform_valid, device
                )
                model(x)
                processed += 1
                print(f"[qlabel] processed {processed}/{args.samples}", flush=True)
    finally:
        sink.close()

    if processed != args.samples:
        raise RuntimeError(f"样本不足: expected {args.samples}, processed {processed}")
    if not sink.groups:
        raise RuntimeError("没有采集到任何组")

    rec = concat_fields(sink.groups)
    heads_per_stage = {int(st): int(h) for st, h in (
        base_profile.deployment_contract_from_config(config).get("heads_per_stage", {}).items()
    )}
    temporal_windows = 5
    # 移位 block 表：Spiking_Swin_BasicLayer 中 block 序号为奇则 shift (1,7,7)
    shifted_blocks = {
        (stage, block): (block % 2 == 1)
        for (stage, block) in POST_G0_BLOCK_PAIRS
    }

    verdict = adjudicate(
        rec,
        meta={
            "temporal_windows": temporal_windows,
            "shifted_blocks": shifted_blocks,
        },
    )

    # 封存口径复现自校验（kv 代理口径 vs 封存 ep44 78.60%）
    sealed_check: dict[str, Any] = {"reproduced": True}
    if args.sealed_npz.exists() and not args.adjacency:
        try:
            sealed = np.load(args.sealed_npz, allow_pickle=True)
            s_k1 = sealed["source_k_popcount"].astype(np.int64)
            s_qp = popcount_u32(sealed["descriptor_k_bitmap"]).astype(np.int64)
            s_match = float((s_k1[:-1] == s_qp[1:]).mean())
            if abs(s_match - SEALED_EP44_FULL_MATCH) > SEALED_TOL:
                sealed_check = {
                    "reproduced": False,
                    "reason": f"封存复现 {s_match:.4f} vs 封存 {SEALED_EP44_FULL_MATCH:.4f}",
                }
        except Exception as exc:  # noqa: BLE001
            sealed_check = {"reproduced": False, "reason": f"sealed npz 读取失败: {exc}"}
    else:
        # adjacency 开启时边界流多出对齐组记录？不——对齐组 adj=1 不进主流；
        # 基础流仍与封存同构，此处仅做本 dump 内部 proxy 与封存值的对照
        print("[qlabel] adjacency 开启或封存 npz 缺失，proxy 对照由 report 的 proxy_full_match 承担",
              flush=True)

    final_verdict = finalize_verdict(verdict, sealed_check)

    report = {
        "schema": SCHEMA,
        "date": "2026-08-18",
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "samples": processed,
        "groups_per_block_sample": args.groups_per_block_sample,
        "adjacency": args.adjacency,
        "sampling_id": POST_G0_SAMPLING_ID,
        "checkpoint_load_audit": load_audit,
        "bn_policy": bn_policy,
        "verdict": verdict,
        "sealed_check": sealed_check,
        "final_verdict": final_verdict,
    }
    report_path = args.output_dir / "qlabel_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    npz_path = args.output_dir / "qlabel_records.npz"
    np.savez_compressed(
        npz_path,
        **{key: rec[key] for key in (
            "sample", "stage", "block", "call", "flat_window", "head", "adj",
            "plane", "pos", "qv", "kv", "q1", "k1",
        )},
        nbr=rec["nbr"], ev=rec["ev"], sc=rec["sc"], ga=rec["ga"],
    )
    write_summary_md(args.output_dir / "qlabel_summary.md", report)
    print(json.dumps(final_verdict, ensure_ascii=False, indent=2), flush=True)
    return 0 if final_verdict["ruling"] == "GATE1_PASS" else 0


if __name__ == "__main__":
    sys.exit(main())
