"""T5 回归（审计修正验证）：负 gamma 方向 + 48bit thr 合同。

Codex 独立审计（shared_execution_20260915/claude_review）发现两个模型侧 bug：
1. D=−1 时 thr 少一次取反（tau 已乘 direction 而 RTL Vtop 是 U 侧，比较再取反
   = 方向折两次）——四条真实 trace 全 gamma>0 未触发，但合同错误；
2. 模型允许 49bit thr 而 DUT 只存 48bit。
修正后本回归：构造含负 gamma 的合成 trace（P=8, C=6, H=5，gamma 符号混合），
(a) 审计反例数值单查；(b) 模型 dec_full 必须等于原生 BN 门公式判决；
(c) RTL 四模式必须与模型零差。自有代码。
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from t5_c1_cert_core_model import one_trace  # noqa: E402

RTL = ROOT / 't5_rtl'
OUT = ROOT / 'results' / 't5_rtl' / 'synthetic_neggamma'
SYNTH = ROOT / 'results' / 't5_synth_trace.npz'


def build_synth():
    rng = np.random.default_rng(20260915)
    T, P, C, H = 10, 8, 6, 5
    N = T * P
    S = (rng.random((N, C)) < 0.4).astype(np.uint8)
    W = rng.normal(0, 0.8, (H, C))
    A = rng.normal(0, 0.3, (T, T))
    gamma = np.array([1.0, -1.0, -0.7, 1.3, -1.0])
    beta = rng.normal(0, 0.2, H)
    bias = rng.normal(0, 0.1, (T, 1))
    center = rng.normal(0, 0.1, (T, 1))
    np.savez(SYNTH, source_packed=np.packbits(S, axis=1, bitorder='little'),
             W=W.astype(np.float32), A=A, gamma=gamma, beta=beta,
             bias=bias, center=center)
    return dict(T=T, P=P, C=C, H=H, S=S, W=W, A=A, gamma=gamma, beta=beta,
                bias=bias, center=center)


def audit_counterexample():
    """审计反例：gamma=-1, Y=0, mu=0, var=1, theta=1, A=I, beta/bias/center=0。
    原生门 = false；修正后公式也必须给 false（修正前 true）。"""
    gamma, mu, var, theta = -1.0, 0.0, 1.0, 1.0
    R, direction = 1.0, -1.0
    tau = (mu * R + np.sqrt(var + 1e-5) / gamma * theta) * direction
    tau_q = np.rint(tau * (1 << 14))
    thr = int(-(int(tau_q) << 12) + 1)             # 修正后的 U 侧 thr
    U = 0
    raw = U >= thr
    dec = (not raw) if direction < 0 else raw
    normalized = gamma * (0.0 - mu) / np.sqrt(var + 1e-5)
    native = normalized + 0.0 >= theta
    assert bool(dec) == bool(native) == False, (dec, native)
    print('audit counterexample: fixed dec=%s == native=%s' % (dec, native))


def native_gate(prm):
    """原生 BN 门公式（build_traces 口径）：ordered >= theta。"""
    T, P, C, H = prm['T'], prm['P'], prm['C'], prm['H']
    Y = (prm['S'].astype(np.float64) @ prm['W'].astype(np.float64).T)
    Y = Y.reshape(T, P, H)
    mu = Y.mean((0, 1))
    var = ((Y - mu) ** 2).mean((0, 1))
    normalized = prm['gamma'][None, None, :] * (Y - mu[None, None, :]) \
        / np.sqrt(var + 1e-5)[None, None, :] + prm['beta'][None, None, :]
    ordered = np.einsum('ts,sph->tph', prm['A'], normalized) \
        + prm['bias'][:, None, :] - prm['center'][:, None, :]
    return ordered >= 1.0                          # (T,P,H)


def run_rtl(mode):
    exe = str(RTL / 'obj_dir' / 'Vcert_gate_core')
    args = [exe, '+a=%s/a.hex' % OUT, '+pn=%s/pn.hex' % OUT]
    for t in range(10):
        args.append('+tau_t%d=%s/tau_t%d.hex' % (t, OUT, t))
    stim = 'stim_%s.txt' % mode.split('_')[0]
    args += ['+stim=%s/%s' % (OUT, stim), '+mode=%s' % mode,
             '+out=%s/rtl_%s.txt' % (OUT, mode)]
    subprocess.run(args, check=True, capture_output=True)
    dec, cyc = [], []
    for ln in (OUT / ('rtl_%s.txt' % mode)).read_text().splitlines():
        if ln.startswith('#'):
            continue
        _, d, p = ln.split()
        dec.append(int(d, 16))
        cyc.append(1 + int(p))
    return np.array(dec, np.uint16), np.array(cyc, np.int32)


def main():
    audit_counterexample()
    prm = build_synth()
    r = one_trace(SYNTH, OUT)
    print('model groups=%d H=%d P=%d' % (r['groups'], r['H'], r['P']))

    exp = np.load(OUT / 'expected.npz')
    dec_exp = exp['dec']                                  # (G,10)
    G = dec_exp.shape[0]

    # (b) 模型 dec_full == 原生 BN 门（逐组逐 t）
    nat = native_gate(prm)                                # (T,P,H)
    # one_trace 用同一个 SEED 生成器顺序采样：ps 先、hs 后，这里必须复刻
    rng = np.random.default_rng(20260914)
    ps = rng.integers(0, prm['P'], G)
    hs = rng.integers(0, prm['H'], G)
    nat_g = nat[:, ps, hs].T                              # (G,T)
    mism = int((nat_g.astype(np.uint8) != dec_exp).sum())
    assert mism == 0, 'model vs native BN gate mismatch: %d' % mism
    print('model vs native BN gate: %d groups x 10, 0 mismatch' % G)

    # (c) RTL 四模式 vs 模型
    for mode in ('fx_full', 'fx_cert', 'bf_full', 'bf_cert'):
        dec, cyc = run_rtl(mode)
        dec_rtl = ((dec[:, None] >> np.arange(10)[None, :]) & 1).astype(np.uint8)
        assert int((dec_rtl != dec_exp).sum()) == 0, mode
        if mode == 'fx_cert':
            assert int((cyc != exp['cyc_fx']).sum()) == 0, mode
        if mode == 'bf_cert':
            assert int((cyc != exp['cyc_bf']).sum()) == 0, mode
        print('RTL %-8s dec 0 mismatch, mean_cycles=%.3f' % (mode, cyc.mean()))
    print('NEG-GAMMA REGRESSION PASS')


if __name__ == '__main__':
    main()
