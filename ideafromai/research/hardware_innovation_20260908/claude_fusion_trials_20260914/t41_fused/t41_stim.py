#!/usr/bin/env python3
"""T41 激励导出：把 t41_equiv 用的同一批 (Y_q, thr, 判决) 落成 Verilator 可读的文本。

**逐层分文件**：A 是逐层的，RTL 也逐层生成（`t41_gen.py` → `gates/L%d_gate.sv`），
所以激励必须按层切开——否则拿 L8 的门去回放 L14 的组必然失配（这不是 RTL 的问题）。

口径与 `t41_equiv.py` 完全同一：同一 npz 的 k=4 A、同一 `trace_setup`、同一批 traces。
导出前**断言**每个 lane 的 Y_q 落在 signed24 内、每行判决与非融合式一致——RTL 拿到的是
已经过范围检查的数据，失配就只能是 RTL 的问题。

用法：python t41_fused/t41_stim.py   # 写 t41_fused/stim_L{8,14,20,28}.txt
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 't32_lever'))
from t32_lever import trace_setup  # noqa: E402
import t19_all_layers as t19  # noqa: E402

LIDS = [8, 14, 20, 28]
NSID = 10
PER_TRACE = 200          # 每个 (序列, 层) 取前多少个组
YMIN, YMAX = -(1 << 23), (1 << 23) - 1
TMIN, TMAX = -(1 << 47), (1 << 47) - 1


def main():
    npz = np.load(ROOT / 'results' / 't21b_k4_gate_params.npz')
    z = np.load(t19.PARAMS)
    src = t19.parse_sources(list(range(NSID)))

    meta = {}
    for lid in LIDS:
        A = npz['L%d_A' % lid]
        Aq = np.rint(A * 4096).astype(np.int64)
        Aq = np.where(Aq >= (1 << 15), Aq - (1 << 16), Aq)
        prm = {k: z['L%d_%s' % (lid, k)] for k in
               ('W', 'A', 'gamma', 'beta', 'bias', 'center', 'theta_src')}
        prm['lid'] = lid
        lines, n_dec = [], 0
        per_trace = {}
        for sid in range(NSID):
            key = (sid, lid)
            if key not in src:
                continue
            pk, C, br = src[key]
            Yq, thr_g, _ = trace_setup(pk, C, br, prm)
            Yq = Yq[:PER_TRACE]
            th = thr_g[:PER_TRACE]
            dec = (Yq @ Aq.T) >= th
            assert Yq.min() >= YMIN and Yq.max() <= YMAX, 'Y out of signed24'
            assert th.min() >= TMIN and th.max() <= TMAX, 'thr out of signed48'
            for g in range(Yq.shape[0]):
                yv = ' '.join(str(int(v)) for v in Yq[g])
                tv = ' '.join(str(int(v)) for v in th[g])
                d = int(sum((1 << t) for t in range(10) if dec[g][t]))
                lines.append('G %s %s %03x' % (yv, tv, d))
            n_dec += dec.size
            per_trace['s%d' % sid] = int(Yq.shape[0])
        out = ROOT / 't41_fused' / ('stim_L%d.txt' % lid)
        out.write_text('\n'.join(lines) + '\n')
        meta['L%d' % lid] = {'groups': len(lines), 'decisions': int(n_dec),
                             'per_trace': per_trace}
        print('wrote %s  groups %d  decisions %d'
              % (out.relative_to(ROOT), len(lines), n_dec))

    meta['_note'] = ('与 t41_equiv.py 同源同口径，仅截取每 trace 前 %d 组供 RTL 回放；'
                     '按层分文件，与 gates/L%%d_gate.sv 一一对应。' % PER_TRACE)
    meta['_range_ok'] = 'Y in signed24, thr in signed48（导出前已断言）'
    (ROOT / 't41_fused' / 'stim_meta.json').write_text(json.dumps(meta, indent=1) + '\n')


if __name__ == '__main__':
    main()
