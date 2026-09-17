"""BitL (MICRO'25) hybrid bit-serial/parallel cycle proxy on Swin MLP FC1.

Identity: AT-LIF {0,theta} absorb, binary spikes. Not silicon PPA.
Weights-only tiles are static. One captured fc1 frame for spike AND.

Cycle model on a G x B bit subtile (G grouped input channels, B bits MSB->LSB):
  naive            = B
  serial           = max row popcount (densest weight; group waits while walking bits)
  parallel         = max col popcount (densest bit-plane)
  bitl_min         = min(serial, parallel) per tile  (assigned approx)
  bitl_greedy      = greedy cover: each cycle clear densest remaining row or col
  serial_nz_cols   = #bit-planes with a 1  (Pragmatic-style skip empty columns)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ALGO = HERE.parents[1] / "algorithm"
REPO = Path("/home/zhumd/work/sdformer_codex/SDformer")
DATA = REPO / "data/Datasets/DSEC/saved_flow_data"
INCOMING = REPO / "hw_autoresearch_nts07/system_handoff/incoming/m2041_ep34_quant_binding_inputs"
sys.path[:0] = [str(ALGO), str(ALGO / "nrv_cost_probe")]
os.environ["SDFORMER_USE_MLFLOW"] = "0"
os.environ["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
from run_bn_probe import build_model, input_frame
from spikingjelly.activation_based import functional

PREFIX = "sttmultires_unet.encoders.swin3d."
SHORT = [
    "layers.0.swin_blocks.0.mlp.fc1",
    "layers.1.swin_blocks.0.mlp.fc1",
]
NET_MAC = 596.5464288e9
LAYER_MAC = 7.08e9
PY = "/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/r0_execution_trials_20260913/data_and_quality/py312/bin/python"


def jnum(x):
    if isinstance(x, dict):
        return {k: jnum(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jnum(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, np.ndarray):
        return jnum(x.tolist())
    if torch.is_tensor(x):
        return jnum(x.detach().cpu())
    return x


def summarize(t: torch.Tensor) -> dict:
    t = t.float().reshape(-1)
    if t.numel() == 0:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    q = torch.quantile(t, torch.tensor([0.5, 0.9], device=t.device, dtype=t.dtype))
    return {
        "n": int(t.numel()),
        "mean": float(t.mean()),
        "p50": float(q[0]),
        "p90": float(q[1]),
        "max": float(t.max()),
    }


def pad_groups(bits: torch.Tensor, G: int) -> tuple[torch.Tensor, int]:
    """bits [O, I, B] -> tiles [O, Ng, G, B], padded with zero weights."""
    o, inn, b = bits.shape
    pad = (G - inn % G) % G
    if pad:
        bits = F.pad(bits, (0, 0, 0, pad))
        inn = inn + pad
    ng = inn // G
    return bits.reshape(o, ng, G, b), pad


def greedy_bitl_cycles(tiles: torch.Tensor) -> torch.Tensor:
    """tiles [N, R, C] in {0,1}. One cycle clears densest remaining row or col."""
    m = tiles.to(torch.uint8).clone()
    n, r, c = m.shape
    cyc = torch.zeros(n, device=m.device, dtype=torch.int32)
    for _ in range(r + c + 1):
        live = m.reshape(n, -1).any(dim=1)
        if not bool(live.any()):
            break
        row_n = m.sum(dim=2)
        col_n = m.sum(dim=1)
        rmax, ridx = row_n.max(dim=1)
        cmax, cidx = col_n.max(dim=1)
        pick_row = rmax >= cmax
        sel_r = torch.where(pick_row & live)[0]
        if sel_r.numel():
            m[sel_r, ridx[sel_r], :] = 0
        sel_c = torch.where((~pick_row) & live)[0]
        if sel_c.numel():
            m[sel_c, :, cidx[sel_c]] = 0
        cyc += live.to(torch.int32)
    return cyc


def tile_cycle_stats(bits: torch.Tensor, G: int, naive_bits: int | None = None) -> dict:
    tiles, pad = pad_groups(bits, G)
    o, ng, g, b = tiles.shape
    naive = int(naive_bits if naive_bits is not None else b)
    row_pop = tiles.sum(dim=3)
    col_pop = tiles.sum(dim=2)
    serial = row_pop.amax(dim=2)
    parallel = col_pop.amax(dim=2)
    bitl_min = torch.minimum(serial, parallel)
    nz_cols = (col_pop > 0).sum(dim=2)
    nz_rows = (row_pop > 0).sum(dim=2)
    greedy = greedy_bitl_cycles(tiles.reshape(-1, g, b)).reshape(o, ng)
    ones = tiles.reshape(o, ng, -1).sum(dim=2)
    best_uni = torch.minimum(serial, parallel)
    mean_ones = float(ones.float().mean())
    mean_serial = float(serial.float().mean())
    mean_min = float(bitl_min.float().mean())
    mean_greedy = float(greedy.float().mean())
    pe_bound = mean_ones / max(g, 1)
    return {
        "G": g,
        "B": b,
        "naive": naive,
        "n_tiles": int(o * ng),
        "out": o,
        "in_padded": int(ng * g),
        "in_pad": int(pad),
        "bit_density": float(tiles.float().mean()),
        "mean_ones": mean_ones,
        "pe_bound_ones_over_G": pe_bound,
        "serial_max_row_pop": summarize(serial),
        "parallel_max_col_pop": summarize(parallel),
        "serial_nz_cols": summarize(nz_cols),
        "parallel_nz_rows": summarize(nz_rows),
        "bitl_min": summarize(bitl_min),
        "bitl_greedy": summarize(greedy),
        "mean_vs_naive": {
            "serial_max_row_pop": mean_serial / naive,
            "parallel_max_col_pop": float(parallel.float().mean()) / naive,
            "serial_nz_cols": float(nz_cols.float().mean()) / naive,
            "bitl_min": mean_min / naive,
            "bitl_greedy": mean_greedy / naive,
        },
        "speedup_vs_naive": {
            "serial_max_row_pop": naive / max(mean_serial, 1e-9),
            "bitl_min": naive / max(mean_min, 1e-9),
            "bitl_greedy": naive / max(mean_greedy, 1e-9),
            "best_unidirectional": naive / max(float(best_uni.float().mean()), 1e-9),
        },
        "speedup_vs_serial": {
            "bitl_min": mean_serial / max(mean_min, 1e-9),
            "bitl_greedy": mean_serial / max(mean_greedy, 1e-9),
            "pe_bound": mean_serial / max(pe_bound, 1e-9),
        },
        "speedup_vs_best_uni": {
            "bitl_min": 1.0,
            "bitl_greedy": float(best_uni.float().mean()) / max(mean_greedy, 1e-9),
        },
        "frac_serial_full_naive": float((serial == naive).float().mean()),
        "frac_bitl_min_lt_serial": float((bitl_min < serial).float().mean()),
        "frac_greedy_lt_min": float((greedy < bitl_min).float().mean()),
    }


def quant_int8(w: torch.Tensor):
    """Per-tensor symmetric int8. Returns int16 codes in [-128,127] and scale."""
    wf = w.detach().float().contiguous()
    maxabs = float(wf.abs().max().clamp_min(1e-12))
    scale = maxabs / 127.0
    q = torch.quantize_per_tensor(wf.cpu(), scale, 0, torch.qint8)
    qi = q.int_repr().to(torch.int16)
    return qi.to(w.device), scale


def bits_sign_mag(qi: torch.Tensor, n_mag: int, include_sign: bool) -> torch.Tensor:
    mag = qi.abs().clamp(max=(1 << n_mag) - 1).to(torch.int32)
    planes = [((mag >> (n_mag - 1 - b)) & 1).to(torch.uint8) for b in range(n_mag)]
    if include_sign:
        sign = (qi < 0).to(torch.uint8)
        planes = [sign] + planes
    return torch.stack(planes, dim=-1)


def bits_twos(qi: torch.Tensor, nbits: int = 8) -> torch.Tensor:
    u = qi.to(torch.int32) & ((1 << nbits) - 1)
    planes = [((u >> (nbits - 1 - b)) & 1).to(torch.uint8) for b in range(nbits)]
    return torch.stack(planes, dim=-1)


def bits_mag_n(w: torch.Tensor, n_mag: int) -> tuple[torch.Tensor, float]:
    wf = w.detach().float()
    qmax = (1 << n_mag) - 1
    maxabs = float(wf.abs().max().clamp_min(1e-12))
    scale = maxabs / qmax
    mag = (wf.abs() / scale).round().clamp(0, qmax).to(torch.int32)
    planes = [((mag >> (n_mag - 1 - b)) & 1).to(torch.uint8) for b in range(n_mag)]
    return torch.stack(planes, dim=-1), scale


def plane_density(bits: torch.Tensor) -> list[float]:
    return [float(bits[..., b].float().mean()) for b in range(bits.shape[-1])]


def and_density(bits: torch.Tensor, spikes: torch.Tensor) -> dict:
    """AND of 1-bit spikes and weight bits. Exact mean without materializing N x O x I x B."""
    spike_rate = spikes.float().mean(dim=0)
    w_bit = bits.float().mean(dim=(0, 2))
    and_per_in = spike_rate * w_bit
    spike_rate_mean = float(spikes.float().mean())
    w_bit_mean = float(bits.float().mean())
    and_mean = float(and_per_in.mean())
    return {
        "spike_rate": spike_rate_mean,
        "weight_bit_density": w_bit_mean,
        "and_density": and_mean,
        "and_sparsity": 1.0 - and_mean,
        "and_over_weight_bits": and_mean / max(w_bit_mean, 1e-12),
        "expected_if_independent": spike_rate_mean * w_bit_mean,
    }


def and_tile_cycles(bits: torch.Tensor, spikes: torch.Tensor, G: int, chunk: int = 4096) -> dict:
    """min(row_crit, col_crit) after zeroing weight rows whose spike is 0.

    Averaged over AAC-surviving tokens (any spike in the full vector).
    Empty input-groups (all G spikes 0) are 0-cycle tiles.
    """
    tiles, pad = pad_groups(bits, G)
    o, ng, g, b = tiles.shape
    ntok = spikes.shape[0]
    pad_s = (G - spikes.shape[1] % G) % G
    sp = F.pad(spikes, (0, pad_s)) if pad_s else spikes
    sp = sp.view(ntok, ng, g).to(tiles.dtype)
    live_tok = spikes.reshape(ntok, -1).any(dim=1)
    sp_live = sp[live_tok.bool()]
    n_live = int(sp_live.shape[0])
    naive = b
    device = bits.device
    sum_serial = sum_par = sum_min = sum_nzg = 0.0
    n_tiles = 0
    n_work = 0
    ham = tiles.sum(dim=3).float()
    for n0 in range(0, n_live, chunk):
        s = sp_live[n0 : n0 + chunk].float()
        nb = s.shape[0]
        serial = (ham.unsqueeze(0) * s.unsqueeze(1)).amax(dim=3)
        col = torch.einsum("tng,ongb->tong", s, tiles.float())
        parallel = col.amax(dim=3)
        bitl = torch.minimum(serial, parallel)
        grp_nz = s.any(dim=2)
        work = grp_nz.unsqueeze(1).expand_as(bitl)
        sum_serial += float(serial.sum())
        sum_par += float(parallel.sum())
        sum_min += float(bitl.sum())
        n_tiles += int(bitl.numel())
        n_work += int(work.sum())
        if work.any():
            sum_nzg += float(bitl[work].sum())
        del serial, parallel, bitl, col, s, grp_nz, work
    mean = lambda acc: acc / max(n_tiles, 1)
    mean_w = lambda acc: acc / max(n_work, 1)
    rec = {
        "G": g,
        "B": b,
        "naive": naive,
        "n_aac_tokens": n_live,
        "n_tiles": n_tiles,
        "n_work_tiles": n_work,
        "work_tile_frac": n_work / max(n_tiles, 1),
        "all_tiles": {
            "serial": mean(sum_serial),
            "parallel": mean(sum_par),
            "bitl_min": mean(sum_min),
            "vs_naive_bitl_min": mean(sum_min) / naive,
        },
        "work_tiles_only": {
            "bitl_min": mean_w(sum_nzg),
            "vs_naive_bitl_min": mean_w(sum_nzg) / naive,
        },
        "in_pad": int(pad),
    }
    return rec


def greedy_and_sample(bits: torch.Tensor, spikes: torch.Tensor, G: int, n_sample: int, seed: int = 20260916) -> dict:
    tiles, _ = pad_groups(bits, G)
    o, ng, g, b = tiles.shape
    live = torch.where(spikes.reshape(spikes.shape[0], -1).any(dim=1))[0]
    if live.numel() == 0:
        return {"n_sample": 0}
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    take = min(n_sample, int(live.numel()))
    pick = live[torch.randperm(int(live.numel()), generator=rng)[:take]]
    pad_s = (G - spikes.shape[1] % G) % G
    sp = F.pad(spikes, (0, pad_s)) if pad_s else spikes
    sp = sp.view(spikes.shape[0], ng, g)
    s = sp[pick]
    # tiles [O, Ng, G, B] * spikes [T, Ng, G] -> [T, O, Ng, G, B]
    gated = tiles.to(torch.uint8).unsqueeze(0) * s.unsqueeze(1).unsqueeze(-1)
    n_t = take
    flat = gated.reshape(n_t * o * ng, g, b)
    serial = flat.sum(dim=2).amax(dim=1)
    parallel = flat.sum(dim=1).amax(dim=1)
    bitl_min = torch.minimum(serial, parallel)
    greedy = greedy_bitl_cycles(flat)
    work = flat.reshape(flat.shape[0], -1).any(dim=1)
    return {
        "n_sample_tokens": take,
        "n_tiles": int(flat.shape[0]),
        "bitl_min": summarize(bitl_min),
        "bitl_greedy": summarize(greedy),
        "serial": summarize(serial),
        "parallel": summarize(parallel),
        "work_frac": float(work.float().mean()),
        "greedy_vs_naive": float(greedy.float().mean()) / b,
        "greedy_vs_min": float(bitl_min.float().mean()) / max(float(greedy.float().mean()), 1e-9),
    }


def fmt(x, nd=3):
    return f"{x:.{nd}f}"


def write_md(summary: dict, path: Path) -> None:
    layers = summary["layers"]
    l0 = layers["layers.0.swin_blocks.0.mlp.fc1"]
    l1 = layers["layers.1.swin_blocks.0.mlp.fc1"]

    def row(layer, enc, gkey):
        st = layer["encodings"][enc][gkey]
        n8 = st["naive"]
        s = st["serial_max_row_pop"]["mean"]
        p = st["parallel_max_col_pop"]["mean"]
        m = st["bitl_min"]["mean"]
        g = st["bitl_greedy"]["mean"]
        zc = st["serial_nz_cols"]["mean"]
        vs_s = st["speedup_vs_serial"]["bitl_min"]
        return (
            f"{s:.3f}",
            f"{p:.3f}",
            f"{zc:.3f}",
            f"{m:.3f}",
            f"{g:.3f}",
            f"{st['speedup_vs_naive']['bitl_min']:.3f}",
            f"{vs_s:.3f}",
            f"{st['speedup_vs_naive']['bitl_greedy']:.3f}",
            n8,
            st,
        )

    def enc_block(title, enc, gkey):
        a = row(l0, enc, gkey)
        b = row(l1, enc, gkey)
        naive = a[-2]
        lines = [
            f"### {title}",
            "",
            f"naive {naive}-bit serial = **{naive}** cycles / tile. 不是硅 PPA。",
            "",
            "| 层 | serial 最密行 | parallel 最密列 | serial 非空列 | BitL min | 行/列 greedy | min vs naive | min vs serial | greedy vs naive |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| L0.b0.fc1 {l0['shape']} | {a[0]} | {a[1]} | {a[2]} | {a[3]} | {a[4]} | {a[5]}× | {a[6]}× | {a[7]}× |",
            f"| L1.b0.fc1 {l1['shape']} | {b[0]} | {b[1]} | {b[2]} | {b[3]} | {b[4]} | {b[5]}× | {b[6]}× | {b[7]}× |",
            "",
        ]
        return lines

    sm8_l0 = l0["encodings"]["sign_mag_8"]["G8"]
    sm8_l1 = l1["encodings"]["sign_mag_8"]["G8"]
    and0 = l0["spikes_1frame"]
    and1 = l1["spikes_1frame"]
    g0 = and0["G8_and_on_aac_rows"]
    g1 = and1["G8_and_on_aac_rows"]
    gv0 = and0.get("G8_and_greedy_sample", {})
    gv1 = and1.get("G8_and_greedy_sample", {})

    net0 = LAYER_MAC / NET_MAC
    save0_naive = (1.0 - sm8_l0["mean_vs_naive"]["bitl_min"]) * net0
    save1_naive = (1.0 - sm8_l1["mean_vs_naive"]["bitl_min"]) * net0
    save0_ser = (1.0 - 1.0 / sm8_l0["speedup_vs_serial"]["bitl_min"]) * net0
    save1_ser = (1.0 - 1.0 / sm8_l1["speedup_vs_serial"]["bitl_min"]) * net0
    p0 = [round(x, 3) for x in l0["encodings"]["sign_mag_8"]["bit_density_per_plane"]]
    p1 = [round(x, 3) for x in l1["encodings"]["sign_mag_8"]["bit_density_per_plane"]]

    md = []
    md += [
        "# BitL 横/纵关键路径（MLP FC1，AAC 之后仍最热）",
        "",
        "论文：Lee et al., MICRO’25, BitL。身份：**AT-LIF `{0,θ}` 吸入 W，脉冲路径二值 GeMM**。",
        "对象：AAC 跳过全零 token 行之后仍最热的 `L0.b0.fc1`（本帧空行 2.5%），对照 `L1.b0.fc1`（空行 46%）。",
        "周期是 **G×B 权重比特子块** 的拍数代理，**不是硅 PPA**，不改 nts07。",
        "",
        "## 抄什么",
        "",
        "BitL：一组 8-bit 权排成 8×8 比特阵。纯 bit-serial 被 **1 最多的那一行** 卡住；纯 bit-parallel 被最密的位平面卡住。",
        "本脚本（题目指定的近似，不是硅上的 A* PE）：",
        "",
        "- `serial` = 子块最大行 popcount（沿 MSB→LSB 走，一组等最密权）",
        "- `parallel` = 最大列 popcount",
        "- **BitL min** = 每块 `min(serial, parallel)`（静态选更好的单向）",
        "- **行/列 greedy** = 每拍清掉当前最密剩余行或列（整行/整列覆盖；论文 Fig.1 的 4×4 例子是 2 拍）",
        "- naive = 恒 8 拍（无跳零 bit-serial）",
        "",
        "量化：`torch.quantize_per_tensor` 对称 int8。主表 **符号幅值**（1 sign + 7 mag）。对照 two’s complement 和 4-bit 幅值 4×4。",
        "",
        "## 权重-only（静态 W 比特）",
        "",
    ]
    md += enc_block("8×8 符号幅值 int8（主结果）", "sign_mag_8", "G8")
    md += [
        f"L0 位密度 **{l0['encodings']['sign_mag_8']['bit_density']:.3f}**，平面（sign→LSB） `{p0}`；"
        f"每块均 {sm8_l0['mean_ones']:.2f} 个 1，PE 满载下界 ones/8 = **{sm8_l0['pe_bound_ones_over_G']:.2f}**。",
        f"L1 密度 **{l1['encodings']['sign_mag_8']['bit_density']:.3f}**，`{p1}`，下界 **{sm8_l1['pe_bound_ones_over_G']:.2f}**。",
        "",
        f"相对 naive 8：BitL min 是 **{sm8_l0['speedup_vs_naive']['bitl_min']:.2f}× / {sm8_l1['speedup_vs_naive']['bitl_min']:.2f}×**（L0/L1）。",
        f"这几乎全是「最密行 popcount≈4.5–4.7」相对恒 8 拍，**不是换向**：min 优于 serial 的块只占 "
        f"L0 **{100*sm8_l0['frac_bitl_min_lt_serial']:.1f}%**、L1 **{100*sm8_l1['frac_bitl_min_lt_serial']:.1f}%**，"
        f"min vs serial 只有 **{sm8_l0['speedup_vs_serial']['bitl_min']:.3f}× / {sm8_l1['speedup_vs_serial']['bitl_min']:.3f}×**。",
        f"整行/整列 greedy 在散开的 1 上 **更慢**（L0 {sm8_l0['bitl_greedy']['mean']:.2f} 拍 vs serial {sm8_l0['serial_max_row_pop']['mean']:.2f}），"
        "因为覆盖所有 1 需要接近 8 条线；Fig.1 那种「一行全 1 + 一列全 1」在本层几乎不出现。",
        "",
        f"一层名义 MAC 各占全网 {net0*100:.2f}%。把 naive 8 换成 BitL min，折合全网 L0 **{save0_naive*100:.2f}%**、L1 **{save1_naive*100:.2f}%**；"
        f"相对已经跳零的 serial 关键路径只剩 **{save0_ser*100:.3f}% / {save1_ser*100:.3f}%**。**成不了倍。**",
        "",
    ]
    md += enc_block("8×8 two’s complement int8", "twos_8", "G8")
    md += [
        "符号扩展把各平面密度都拉到 ~0.5，非空列 ≈8。BitL min 只剩 ~1.41× naive，换向仍然吃不到关键路径。",
        "",
    ]
    md += enc_block("4×4 四比特幅值", "mag_4", "G4")
    md += [
        "4-bit 幅值更稀（MSB 平面 ≈0.02），serial 已经 ~1.8–1.9 / 4。BitL min 几乎等于 serial。",
        "",
        "## 1-bit 发放 AND（一帧，AAC 非零行）",
        "",
        f"帧 `{summary['frame']}`。激活按非零当脉冲（θ 已吸入 W）。有效比特 = `spike AND weight_bit`。",
        "",
        "| 层 | token 发放率 | 空行(AAC) | W 位密度 | AND 密度 | AND 稀疏 | 8×8 BitL min（含空组） | 只计非空组 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| L0.b0.fc1 | {and0['and']['spike_rate']:.3f} | {and0['zero_row_frac']:.3f} | "
            f"{and0['and']['weight_bit_density']:.3f} | {and0['and']['and_density']:.4f} | "
            f"{and0['and']['and_sparsity']:.3f} | {g0['all_tiles']['bitl_min']:.3f}/{g0['naive']} "
            f"({g0['all_tiles']['vs_naive_bitl_min']:.3f}) | {g0['work_tiles_only']['bitl_min']:.3f} |"
        ),
        (
            f"| L1.b0.fc1 | {and1['and']['spike_rate']:.3f} | {and1['zero_row_frac']:.3f} | "
            f"{and1['and']['weight_bit_density']:.3f} | {and1['and']['and_density']:.4f} | "
            f"{and1['and']['and_sparsity']:.3f} | {g1['all_tiles']['bitl_min']:.3f}/{g1['naive']} "
            f"({g1['all_tiles']['vs_naive_bitl_min']:.3f}) | {g1['work_tiles_only']['bitl_min']:.3f} |"
        ),
        "",
        f"AND 密度 ≈ 发放率 × W 位密度（L0 {and0['and']['expected_if_independent']:.4f} 期望 vs 实测 {and0['and']['and_density']:.4f}），通道间几乎独立。",
        "8 输入一组里大约 1.4 个脉冲（L0）/ 0.4 个（L1），所以 AND 之后 serial≈parallel≈BitL min：",
        f"L0 全块 {g0['all_tiles']['serial']:.3f}，非空组 {g0['work_tiles_only']['bitl_min']:.3f}。",
        "从 8 拍掉到 2.6 拍，是 **发放把整行权位置零**，AAC/脉冲稀疏已经记账，不是横/纵换向。",
        "",
    ]
    if gv0 and gv1 and gv0.get("n_sample_tokens"):
        md += [
            f"Greedy 抽样 {gv0['n_sample_tokens']} 个 AAC token：L0 均 {gv0['bitl_greedy']['mean']:.3f}，"
            f"L1 {gv1['bitl_greedy']['mean']:.3f}。空组多时整行覆盖看起来很快，仍然是发放门控。",
            "",
        ]
    md += [
        "## 去留",
        "",
        "- **相对 naive 8 拍：叶子 ~1.7×。** 来源是符号幅值 MSB 稀、最密行只有 ~4.6 个 1，不是 BitL 换向。论文相对早期 bit-serial 的 1.92× 是这一档。",
        "- **相对已经跳零的 serial 关键路径：~1.02×。** 换向几乎不切最密行。论文相对近期跳零的 1.24× 在本层不成立。",
        "- **整行/整列 greedy 在 MLP 权上失败**（比 serial 慢）：1 是散的，不是 Fig.1 那种 L 形关键路径。",
        "- **two’s complement 不要用**：符号扩展把平面填满，跳零/换向都没东西。",
        "- **1-bit 发放 AND 稀疏 0.93–0.98**，但是 AAC/发放账，不要写成 BitL。",
        "- L0.b0.fc1 仍是 AAC 之后最热的 MLP 核；BitL 不能把「几乎每行都要算」变成整行取消。",
        "- 失败不扔：保留「最密行卡住跳零」；不要上硅、不要写进 nts07。",
        "",
        f"原始 json：`results/bitl.json`。GPU `{summary['gpu']}`，{summary['seconds']:.1f}s，1 帧权重+AND。",
        "",
    ]
    path.write_text("\n".join(md), encoding="utf-8")


def _selfcheck_fig1():
    """Paper Fig.1: w={15,8,8,8} as 4x4 bits. Serial/parallel crit=4, BitL greedy=2."""
    tile = torch.tensor(
        [[1, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.uint8
    )
    serial = int(tile.sum(1).max())
    parallel = int(tile.sum(0).max())
    greedy = int(greedy_bitl_cycles(tile.unsqueeze(0))[0])
    assert serial == 4 and parallel == 4 and greedy == 2, (serial, parallel, greedy)
    print("SELFCHECK_FIG1 serial=4 parallel=4 greedy=2 ok", flush=True)


def main():
    t_all = time.time()
    dest = HERE / "results"
    dest.mkdir(exist_ok=True)
    _selfcheck_fig1()
    names = [p.name for p in sorted((DATA / "gt_tensors").glob("*.npy"))]
    frame = names[0]
    args = SimpleNamespace(
        code_root=REPO,
        config=INCOMING / "dsec_c12_alpha0125_ep29_resume5_20260830.yml",
        checkpoint=INCOMING / "checkpoint_epoch34.pth",
        data=DATA,
    )
    model, *_ = build_model(args)
    mods = dict(model.named_modules())
    weights = {}
    captured = {}
    for short in SHORT:
        full = PREFIX + short
        lin = mods[full]
        weights[short] = lin.weight.detach().float().contiguous()
        cin = lin.in_features

        def pre(m, inp, short=short, cin=cin):
            x = inp[0].detach()
            flat = x.reshape(-1, cin) if x.shape[-1] == cin else x.reshape(-1, x.shape[-1])
            captured[short] = (flat != 0).to(torch.uint8)

        lin.register_forward_pre_hook(pre)
        print("W", short, tuple(weights[short].shape), flush=True)

    with torch.no_grad():
        functional.reset_net(model)
        x, _, _ = input_frame(DATA, frame)
        model(x)
        del x
    del model
    torch.cuda.empty_cache()

    layers = {}
    for short in SHORT:
        w = weights[short]
        qi, scale8 = quant_int8(w)
        sm8 = bits_sign_mag(qi, n_mag=7, include_sign=True)
        tw8 = bits_twos(qi, 8)
        mag4, scale4 = bits_mag_n(w, 4)
        spikes = captured[short]
        zero_row = float((spikes.sum(dim=1) == 0).float().mean())
        enc = {
            "sign_mag_8": {
                "scale": scale8,
                "bit_density": float(sm8.float().mean()),
                "bit_density_per_plane": plane_density(sm8),
                "G8": tile_cycle_stats(sm8, 8),
                "G4": tile_cycle_stats(sm8, 4),
            },
            "twos_8": {
                "scale": scale8,
                "bit_density": float(tw8.float().mean()),
                "bit_density_per_plane": plane_density(tw8),
                "G8": tile_cycle_stats(tw8, 8),
            },
            "mag_4": {
                "scale": scale4,
                "bit_density": float(mag4.float().mean()),
                "bit_density_per_plane": plane_density(mag4),
                "G4": tile_cycle_stats(mag4, 4),
            },
        }
        and_sm = and_density(sm8, spikes)
        and_cycles = and_tile_cycles(sm8, spikes, 8)
        and_greedy = greedy_and_sample(sm8, spikes, 8, n_sample=64)
        layers[short] = {
            "module": PREFIX + short,
            "shape": list(w.shape),
            "mac_G": LAYER_MAC / 1e9,
            "net_share": LAYER_MAC / NET_MAC,
            "encodings": enc,
            "spikes_1frame": {
                "file": frame,
                "n_tokens": int(spikes.shape[0]),
                "cin": int(spikes.shape[1]),
                "zero_row_frac": zero_row,
                "and": and_sm,
                "G8_and_on_aac_rows": and_cycles,
                "G8_and_greedy_sample": and_greedy,
            },
        }
        g8 = enc["sign_mag_8"]["G8"]
        print(
            short,
            "sm8 dens",
            round(enc["sign_mag_8"]["bit_density"], 3),
            "serial",
            round(g8["serial_max_row_pop"]["mean"], 3),
            "par",
            round(g8["parallel_max_col_pop"]["mean"], 3),
            "min",
            round(g8["bitl_min"]["mean"], 3),
            "greedy",
            round(g8["bitl_greedy"]["mean"], 3),
            "AND",
            round(and_sm["and_sparsity"], 3),
            flush=True,
        )

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    summary = {
        "identity": {
            "neuron": "AT-LIF {0,theta} absorb into W",
            "activations": "binary spikes",
            "not_silicon_ppa": True,
            "no_nts07_edits": True,
        },
        "assumption": (
            "Subtile = G consecutive input-channel weights x B bits (MSB->LSB). "
            "cycles_serial = max popcount along each weight (row) while walking bit columns. "
            "cycles_parallel = max popcount along a bit-plane (column). "
            "BitL min = min(row_crit, col_crit) per subtile. "
            "BitL greedy = each cycle clears densest remaining row or column. "
            "naive 8-bit serial always 8. Cycle proxy only, not ASIC PPA."
        ),
        "python": PY,
        "gpu": gpu,
        "frame": frame,
        "seconds": time.time() - t_all,
        "net_mac": NET_MAC,
        "layers": layers,
    }
    (dest / "bitl.json").write_text(json.dumps(jnum(summary), indent=2), encoding="utf-8")
    write_md(summary, HERE / "RESULTS.md")
    print("WROTE", dest / "bitl.json", "seconds", round(summary["seconds"], 2), flush=True)


if __name__ == "__main__":
    main()
