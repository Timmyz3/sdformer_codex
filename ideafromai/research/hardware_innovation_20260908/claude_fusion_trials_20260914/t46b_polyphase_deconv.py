#!/usr/bin/env python3
"""T46-2：stride-2 转置卷积的相位分解 —— 精确等价验证 + 抽头普查 + 收益的正确归因。

## 为什么做这个

T45b3 证明 decoder 四个 stride-2 转置卷积在"零插值"口径（A）下比"输入散射"口径（B）
多 **4 倍**。T46-1 进一步量出：全网 946.38G dense MAC 里 **240.25G（25.4%）是结构零**
（全部来自这 4 个转置卷积的零插值）——这是**纯映射决策**就能去掉的部分，不需要重训练。

但"怎么去掉"要说清楚，而且**收益要归因正确**。本脚本做三件事：

1. **精确等价**：在**真实权重 + 真实输入脉冲**上验证
   `ConvTranspose2d(k=3,s=2,p=1,op=1)` ≡ 4 个相位子卷积，逐元素 0 误差。
2. **抽头普查**：四个相位的抽头数是 **1 / 2 / 2 / 4**，合计 9 = k²；
   而零插值口径每个输入元素要枚举 **36 = s²·k²** 个抽头。比值恰好 4。
3. **收益归因（诚实版）**：4× 是**对稠密引擎**说的。对一个**已带零跳过**的
   spike-driven 引擎，零插值出来的零本来就会被跳过 ⇒ **非零乘法次数不变
   （都是 9·Cout/输入元素）**；真正的收益在**缓冲 / 掩码 / 地址生成**三处各 4×。
   这个区分必须写进论文，否则会把"映射省下 4×"错记成"活动稀疏省下 4×"。

## 相位公式（推导）

一维 `s=2,p=1,k=3,op=1`：`y[n] = Σ_{i,kh} x[i]·W[kh]·[n = 2i−1+kh]`。
令 `n = 2j+a`（a∈{0,1}）⇒ `kh = 2(j−i)+a+1`，要求 `0≤kh≤2`：

- `a=0`：唯一解 `kh=1, i=j`（1 个抽头）
- `a=1`：`kh=0, i=j+1` 与 `kh=2, i=j`（2 个抽头）

二维按轴可分，四个相位 `(a,b)∈{0,1}²` 的抽头数 = `(1或2)×(1或2)` = 1/2/2/4，合计 9。

用法：/opt/anaconda3/envs/pytorch310/bin/python t46b_polyphase_deconv.py
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CODE = Path('/home/zhumd/work/sdformer_codex/SDformer')
CAP = (CODE / 'hw_autoresearch_nts07/results'
       / 'm1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831')
CKPT = (CODE / 'hw_autoresearch_nts07/system_handoff/incoming'
        / 'm2041_ep34_quant_binding_inputs/checkpoint_epoch34.pth')
CASES = ['sttmultires_unet.decoders.0.deconv.0',
         'sttmultires_unet.decoders.1.deconv.0',
         'sttmultires_unet.decoders.2.deconv.0',
         'sttmultires_unet.decoders.3.deconv.0']
NCH_OUT = 16          # 采样多少输出通道做等价性核对（共 384/96）
S, P, K, OP = 2, 1, 3, 1


def phase_taps(a: int) -> list:
    """一维相位 a 的 (kh, dih) 抽头表。dih 表示 ih = j + dih。"""
    return [(K // 2, 0)] if a == 0 else [(0, +1), (K - 1, 0)]


def convtranspose_ref(X, W):
    """X: (Cin,H,W), W: (Cin,Cout,K,K) → (Cout,Ho,Wo)。逐 (kh,kw) 显式散射。"""
    Ci, H, Wd = X.shape
    _, Co, _, _ = W.shape
    Ho = (H - 1) * S - 2 * P + K + OP
    Wo = (Wd - 1) * S - 2 * P + K + OP
    Y = np.zeros((Co, Ho, Wo), dtype=np.float64)
    for kh in range(K):
        oh = np.arange(H) * S - P + kh
        mh = np.where((oh >= 0) & (oh < Ho))[0]
        for kw in range(K):
            ow = np.arange(Wd) * S - P + kw
            mw = np.where((ow >= 0) & (ow < Wo))[0]
            patch = X[:, mh, :][:, :, mw]                       # (Cin, |mh|, |mw|)
            Y[:, oh[mh][:, None], ow[mw][None, :]] += np.einsum(
                'chw,co->ohw', patch, W[:, :, kh, kw])
    return Y


def convtranspose_polyphase(X, W):
    """同样的输出，但按四个相位用子卷积算（每个相位只枚举自己的抽头）。"""
    Ci, H, Wd = X.shape
    _, Co, _, _ = W.shape
    Ho, Wo = (H - 1) * S - 2 * P + K + OP, (Wd - 1) * S - 2 * P + K + OP
    Y = np.zeros((Co, Ho, Wo), dtype=np.float64)
    for a in (0, 1):
        for b in (0, 1):
            Yp = np.zeros((Co, H, Wd), dtype=np.float64)
            for kh, dih in phase_taps(a):
                sh = np.s_[dih:H] if dih else np.s_[0:H]
                dh = np.s_[0:H - dih] if dih else np.s_[0:H]
                for kw, diw in phase_taps(b):
                    sw = np.s_[diw:Wd] if diw else np.s_[0:Wd]
                    dw = np.s_[0:Wd - diw] if diw else np.s_[0:Wd]
                    patch = X[:, sh, sw]
                    Yp[:, dh, dw] += np.einsum('chw,co->ohw', patch, W[:, :, kh, kw])
            Y[:, a::S, b::S] = Yp
    return Y


def main():
    import torch

    st = torch.load(CKPT, map_location='cpu', weights_only=False)['model_state_dict']
    recs = {}
    for line in (CAP / 'unified_ordered_records.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r.get('payload', {}).get('retained') and r['category'] == 'decoder_convtranspose':
            recs.setdefault(r['name'], r)

    out = {'phase_tap_counts': {'00': 1, '01': 2, '10': 2, '11': 4},
           'phase_taps_total': 9, 'zero_insert_taps_per_input': 36, 'ratio': 4,
           'cases': []}
    for name in CASES:
        Wt = st[name + '.weight'].numpy().astype(np.float64)      # (Cin,Cout,3,3)
        r = recs[name]
        x = np.frombuffer(zlib.decompress((CAP / r['payload']['compressed_fp32'])
                                         .read_bytes()), '<f4').astype(np.float64)
        shape = r['input']['shape']                                # [T,1,Cin,H,W]
        x = x.reshape(shape)
        T, B, Cin, H, Wd = x.shape
        assert Wt.shape[0] == Cin and Wt.shape[2:] == (K, K), (Wt.shape, Cin)
        Co = Wt.shape[1]

        co = np.random.default_rng(0).choice(Co, size=min(NCH_OUT, Co), replace=False)
        Ws = Wt[:, co]                                             # (Cin,nch,3,3)
        X0 = x[0, 0]                                               # (Cin,H,W) 一个时刻
        Yref = convtranspose_ref(X0, Ws)
        Ypoly = convtranspose_polyphase(X0, Ws)
        err = float(np.max(np.abs(Yref - Ypoly)))
        scale = float(np.max(np.abs(Yref)))

        # 第三方核对：torch 的 ConvTranspose2d（零插值口径）
        Xt = torch.from_numpy(X0[None]).float()                    # (1,Cin,H,W)
        Wtt = torch.from_numpy(Ws).float()                         # (Cin,Cout,3,3)
        Yt = torch.nn.functional.conv_transpose2d(
            Xt, Wtt, stride=S, padding=P, output_padding=OP)[0].numpy().astype(np.float64)
        err_t = float(np.max(np.abs(Yref - Yt)))

        active = int((X0 != 0).sum())
        active_all = int((x != 0).sum())   # 全 (T,Cin,H,W) 张量
        # 零插值口径下 T=0 的全部乘法数 = (Ho·Wo 输出位置) × Cin × K²
        mac_zer = (S * H) * (S * Wd) * Cin * K * K
        # 全张量、含全部 Cout 的两个口径
        mac_zer_allC = T * (S * H) * (S * Wd) * Cin * K * K * Co
        nz_allC = active_all * K * K * Co
        out['cases'].append({
            'conv': name, 'W_shape': list(Wt.shape), 'input_shape': shape,
            'Cin': Cin, 'Cout': Co, 'H': H, 'W': Wd, 'T': T,
            'input_density_T0': float((X0 != 0).mean()), 'active_in_T0': active,
            'input_density_all_T': float((x != 0).mean()), 'active_all_T': active_all,
            'n_out_ch_checked': int(co.size),
            'max_abs_err_ref_vs_polyphase': err,
            'max_abs_err_ref_vs_torch': err_t,
            'max_abs_ref': scale,
            'rel_err': err / scale if scale else 0.0,
            'mac_per_input_elem_polyphase': 9 * Co,
            'mac_per_input_elem_zero_insert': 36 * Co,
            'mac_zero_insert_T0_perCout': mac_zer,
            'mac_polyphase_dense_T0_perCout': mac_zer // 4,
            'mac_zero_insert_allT_allCout': mac_zer_allC,
            'mac_polyphase_dense_allT_allCout': mac_zer_allC // 4,
            'structural_zero_allT_allCout': mac_zer_allC - mac_zer_allC // 4,
            'mac_nonzero_T0_sampledCout': int(active * K * K * co.size),
            'mac_nonzero_allT_allCout': int(nz_allC),
            'mac_nonzero_spikeskip_zero_insert_allT_allCout': int(nz_allC),
            'buffer_elem_zero_insert': 4 * Cin * H * Wd,
            'buffer_elem_polyphase': Cin * H * Wd,
            'sram_ratio': 4.0,
        })
        c = out['cases'][-1]
        print('=== %s ===' % name)
        print('  W %s，输入 %s，density(T0) %.4f，活跃输入 %d（全 T：%d，density %.4f）'
              % (tuple(Wt.shape), shape, c['input_density_T0'], active,
                 active_all, c['input_density_all_T']))
        print('  相位抽头数 00:1 01:2 10:2 11:4 = 9 = k²；零插值口径每输入元素枚举 36 ⇒ 4×')
        print('  等价性（本脚本散射 vs 相位分解）：max|Y_ref − Y_poly| = %.3e（max|Y_ref| = %.3e）%s'
              % (err, scale, '✅ 逐元素为 0' if err == 0.0 else '⚠ 非零'))
        print('  等价性（本脚本散射 vs torch ConvTranspose2d）：max 差 = %.3e %s'
              % (err_t, '✅' if err_t < 1e-3 * scale else '⚠'))
        print('  每个输入元素的乘数：相位分解 9×Cout = %d；零插值 36×Cout = %d'
              % (9 * Co, 36 * Co))
        print('  缓冲元素数：零插值 %d → 相位分解 %d（4×）'
              % (c['buffer_elem_zero_insert'], c['buffer_elem_polyphase']))
        print('  全张量全 Cout 稠密乘法（对稠密引擎）：零插值 %.3fG → 相位分解 %.3fG（4×），结构零 %.3fG'
              % (mac_zer_allC / 1e9, mac_zer_allC / 4e9, (mac_zer_allC - mac_zer_allC // 4) / 1e9))
        print('  **对已带零跳过的引擎：非零乘法 = active×9×Cout = %.3fG，两种实现相同' % (nz_allC / 1e9))
        print()

    (ROOT / 'results' / 't46b_polyphase_deconv.json').write_text(
        json.dumps(out, indent=1) + '\n')
    print('wrote results/t46b_polyphase_deconv.json')


if __name__ == '__main__':
    main()
