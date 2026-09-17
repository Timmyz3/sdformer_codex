#!/usr/bin/env python3
"""T41 等价性断言：C1 证书递归的判决 **恒等于** 融合式的 `A_q·Y ≥ thr`。

这是 T41 融合基线成立的前提，也是"融合送判决"能作为主对照的依据：
如果证书终态判决 ≠ 直接算 A·Y 的判决，那么"送 10 判决位"就不是同一个功能。
（T27 已证 MSB-first + 精确区间证书是精确的，这里把它显式对着融合式再断言一次。）

用法：python t41_fused/t41_equiv.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 't32_lever'))
from t32_lever import cert_planes, trace_setup  # noqa: E402
import t19_all_layers as t19  # noqa: E402


def main():
    npz = np.load(ROOT / 'results' / 't21b_k4_gate_params.npz')
    z = np.load(t19.PARAMS)
    src = t19.parse_sources(list(range(10)))
    bad = tot = groups = 0
    per_layer = {}
    for key in sorted(src):
        sid, lid = key
        if ('L%d_A' % lid) not in npz:
            continue
        A = npz['L%d_A' % lid]
        Aq = np.rint(A * 4096).astype(np.int64)
        Aq = np.where(Aq >= (1 << 15), Aq - (1 << 16), Aq)
        prm = {k: z['L%d_%s' % (lid, k)] for k in
               ('W', 'A', 'gamma', 'beta', 'bias', 'center', 'theta_src')}
        prm['lid'] = lid
        pk, C, br = src[key]
        Yq, thr_g, _ = trace_setup(pk, C, br, prm)
        _, dec_cert, _ = cert_planes(Yq, thr_g, Aq)
        dec_fused = (Yq @ Aq.T) >= thr_g
        m = int((dec_cert != dec_fused).sum())
        bad += m
        tot += dec_cert.size
        groups += Yq.shape[0]
        per_layer['L%d_s%d' % (lid, sid)] = {
            'groups': int(Yq.shape[0]), 'mismatch': m,
            'dec_rate_cert': float(dec_cert.mean()),
        }
    out = {
        'groups': groups, 'decisions': tot,
        'cert_vs_fused_mismatch': bad,
        'verdict': 'PASS' if bad == 0 else 'FAIL',
        'note': '证书递归终态判决 vs 融合式 (A_q·Y ≥ thr)。相等才允许用 "送 10 判决位" 作主对照。',
        'per_trace': per_layer,
    }
    (ROOT / 'results' / 't41_equiv.json').write_text(json.dumps(out, indent=1) + '\n')
    print('groups %d  decisions %d  mismatch %d  -> %s'
          % (groups, tot, bad, out['verdict']))
    print('wrote results/t41_equiv.json')


if __name__ == '__main__':
    main()
