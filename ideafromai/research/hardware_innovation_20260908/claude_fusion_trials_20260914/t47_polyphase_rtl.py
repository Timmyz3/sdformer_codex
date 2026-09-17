#!/usr/bin/env python3
"""T47：把 T46-2 的相位分解结论落成 RTL 对照，并给出 PPA 与槽位数。

## 为什么做

T46-2 已在真实权重 + 真实输入脉冲上证明：零插值口径与相位分解**逐元素 0 误差**，
且非零乘法次数完全相同（`active×9×Cout`）。既然算术量一模一样，两者的一切硬件差异
**全落在控制路径**：地址生成、抽头槽位数、有效性判定、输入缓冲。

所以本脚本生成三个激励用例（真实权重 + 真实脉冲 / 全 1 / 伪随机），以及 Python 金标，
交给 `t47_verify.sh` 做 Verilator 回放（两版 RTL 都必须与金标逐元素相同），
再用 Vivado 对 `t47_deconv_z` 与 `t47_deconv_p` 各自做 out-of-context 综合，取 LUT/FF/DSP。

规模：H=4, W=5, CIN=4 ⇒ Ho=8, Wo=10（80 个输出元素），一个输出通道。
选这么小是为了让 out-of-context 综合的面积差异**只来自控制路径**，不被 MAC 阵列淹没。

用法：/opt/anaconda3/envs/pytorch310/bin/python t47_polyphase_rtl.py
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

CONV = 'sttmultires_unet.decoders.3.deconv.0'
OC = 0                      # 取第 0 个输出通道
H, W, CIN = 4, 5, 4         # 小块（Ho=8, Wo=10）
S, P, K, OP = 2, 1, 3, 1


def convtranspose_ref(X, Wt):
    """X: (Cin,H,W) 二值脉冲, Wt: (Cin,Cout,K,K) int → (Cout,Ho,Wo) int。"""
    Ci, Hh, Wd = X.shape
    Co = Wt.shape[1]
    Ho = (Hh - 1) * S - 2 * P + K + OP
    Wo = (Wd - 1) * S - 2 * P + K + OP
    Y = np.zeros((Co, Ho, Wo), dtype=np.int64)
    for kh in range(K):
        oh = np.arange(Hh) * S - P + kh
        mh = np.where((oh >= 0) & (oh < Ho))[0]
        for kw in range(K):
            ow = np.arange(Wd) * S - P + kw
            mw = np.where((ow >= 0) & (ow < Wo))[0]
            patch = X[:, mh, :][:, :, mw]
            Y[:, oh[mh][:, None], ow[mw][None, :]] += np.einsum(
                'chw,co->ohw', patch, Wt[:, :, kh, kw])
    return Y


def main():
    import torch
    out_dir = ROOT / 't47_stim'
    out_dir.mkdir(exist_ok=True)

    st = torch.load(CKPT, map_location='cpu', weights_only=False)['model_state_dict']
    Wall = st[CONV + '.weight'].numpy()                       # (Cin,Cout,3,3)
    rec = None
    for line in (CAP / 'unified_ordered_records.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r['name'] == CONV and r.get('payload', {}).get('retained'):
            rec = r
            break
    x = np.frombuffer(zlib.decompress((CAP / rec['payload']['compressed_fp32'])
                                      .read_bytes()), '<f4')
    x = x.reshape(rec['input']['shape'])                      # [T,1,Cin,H,W]
    T, B, Cin, Hh, Wd = x.shape
    assert (Cin, Hh, Wd) == (194, 120, 160), x.shape

    # 取前 CIN 个最活跃的输入通道 + 前 H 行 W 列 + 第 0 时刻
    act = (x[0, 0] != 0).reshape(Cin, -1).sum(1)
    ch = np.argsort(-act)[:CIN]
    ch = np.sort(ch)
    Xspk = (x[0, 0][ch][:, :H, :W] != 0).astype(np.int64)      # (CIN,H,W)
    Wsub = Wall[ch][:, OC:OC + 1]                              # (CIN,1,3,3)

    # 权重量化到 int8（按 max|W| 归一）
    scale = float(np.abs(Wsub).max()) / 127.0
    Wq = np.round(Wsub / scale).astype(np.int64).clip(-127, 127)   # (CIN,1,3,3)

    cases = {}
    # 用例 1：真实权重 + 真实脉冲
    cases['real'] = (Xspk, Wq)
    # 用例 2：真实权重 + 全 1（打满所有抽头，含全部边界条件）
    cases['ones'] = (np.ones((CIN, H, W), dtype=np.int64), Wq)
    # 用例 3：真实权重 + 伪随机 30%
    rng = np.random.default_rng(0)
    cases['rand'] = ((rng.random((CIN, H, W)) < 0.30).astype(np.int64), Wq)

    meta = {'conv': CONV, 'oc': OC, 'channels': [int(c) for c in ch],
            'H': H, 'W': W, 'CIN': CIN, 'widget_scale': scale,
            'W_int8': Wq[:, 0].tolist(), 'cases': {}}

    for name, (Xc, Wc) in cases.items():
        Y = convtranspose_ref(Xc, Wc)[0]                       # (Ho,Wo)
        (out_dir / f'x_{name}.txt').write_text(
            ''.join(f'{int(v)}\n' for v in Xc.reshape(-1)))
        (out_dir / f'w_{name}.txt').write_text(
            ''.join(f'{int(v)}\n' for v in Wc.reshape(-1)))
        (out_dir / f'exp_{name}.txt').write_text(
            ''.join(f'{int(v)}\n' for v in Y.reshape(-1)))
        meta['cases'][name] = {'input_active': int(Xc.sum()),
                               'input_elements': int(Xc.size),
                               'y_min': int(Y.min()), 'y_max': int(Y.max())}

    # 槽位普查（金标算出来的期望值，供 RTL 核对）
    Ho, Wo = 2 * H, 2 * W
    meta['slots_expected'] = {
        'z_slots': int(Ho * Wo * K * K * CIN),
        'p_slots': int(Ho * Wo * (1 + 2 + 2 + 4) // 4 * CIN),
        'p_slots_exact': int(sum(
            (1 if a == 0 else 2) * (1 if b == 0 else 2)
            for a in (0, 1) for b in (0, 1)) * (H * W) * CIN),
        'ratio': 4.0,
    }
    (ROOT / 'results' / 't47_stim_meta.json').write_text(
        json.dumps(meta, indent=1) + '\n')

    print('用例：', list(cases))
    print('输入通道（实数里最活跃的 %d 个）：%s' % (CIN, ch.tolist()))
    print('权重 int8 尺度 %.6g；int8 权重范围 [%d, %d]'
          % (scale, Wq.min(), Wq.max()))
    for name in cases:
        c = meta['cases'][name]
        print('  %-5s 输入活跃 %2d/%d，金标 Y ∈ [%d, %d]'
              % (name, c['input_active'], c['input_elements'], c['y_min'], c['y_max']))
    s = meta['slots_expected']
    print('槽位（金标）：Z %d vs P %d（精确 %d）⇒ %.3f×'
          % (s['z_slots'], s['p_slots'], s['p_slots_exact'],
             s['z_slots'] / s['p_slots_exact']))
    print('wrote t47_stim/*.txt 与 results/t47_stim_meta.json')


if __name__ == '__main__':
    main()
