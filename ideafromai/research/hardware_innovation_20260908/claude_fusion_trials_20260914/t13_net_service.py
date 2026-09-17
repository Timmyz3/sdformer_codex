#!/usr/bin/env python3
"""T13b：C1 净服务实验（同端口全计费，部署口径）。

前置结论（T13a + 模型结构查证，2026-09-15）：
1. 部署 BN 是 track 模式（checkpoint 带 running stats，eval 冻结）→ thr(t,h) 为
   层级静态常数，可 SRAM 常驻（tau_q f14：stage0 10×384×16b≈7.5KB，
   stage3 10×3072×16b≈60KB；层切换时装载，逐组 0 传输）；
2. fc1 的 Y 在部署链上是门核独占消费者（fc1→bn1→sn2 严格串行，残差在脉冲侧；
   连续 I24 通路在 r0.conv2 边，与本边不同——Codex cert_transport README 同判）；
   因此生产者↔门核接口可改为位平面串行（producer 移位输出 sop+planes，
   指数由 producer 对 10 词 OR-tree 求得，无额外拍）。

本实验：逐组周期级模拟（含背压日历，与 Codex cert_transport 同日历以便对比），
四模式 × 两阈值来源 × 两传输形态：

模式（供数策略，与 cert_transport 一致）：
  FX full / FX cert / BF full / BF cert
阈值来源：
  thr_port = 每组 10 次 64bit 口读（Codex 原计费）
  thr_sram = 部署口径：SRAM 常驻，逐组 0 读（T13a）
Y 传输形态：
  raw_words = 每组 5×64bit 传输装 10 个 signed24 词（Codex 原接口）
  plane_ser = 部署接口：producer 串行 sop+planes，10bit/平面，6 平面/64bit 传输

每组服务恒等式（无停顿时）：
  1 cmd + [Y 传输] + [thr 传输] + 1 sign/sop + planes + 1 retire + 输出
  raw_words 模式 words 先到、指数在门核生成，内部平面拍照付；
  plane_ser 模式 sop+planes 流式，传输与内部消费同数。
平面数（与 T5/t10_rtl 合同一致）：
  FX full=23（bit22..0）；FX cert=max(23−j*,1)；
  BF full=max(e,1)（bit e−1..0）；BF cert=max(e−j*,0)→至少 1。

背压日历（照抄 cert_transport）：请求在 cycle%7∈{0,1} 拒绝；响应比最早下一拍
额外延迟 1/2/3 拍；输出在 cycle%5∈{3,4} 拒绝。日历时钟与全部消耗拍同步推进。
净服务 = 1 − cert 周期 / 同配置 full 周期。判决/平面数与 T5/T13a 同 SEED 同组
（复算断言 cert==full 判决）。自有代码；只读 traces 与 checkpoint。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HW = ROOT.parents[0]
GH = HW.parents[0] / 'mechanism_rebuild_gh_20260906'
CKPT = Path('/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07'
            '/system_handoff/incoming/motion_c12_ep34_live93_checkpoint_epoch34.pth')
sys.path.insert(0, str(GH / 'scripts'))
from checkpoint_numpy import read_checkpoint  # noqa: E402

TRACES = sorted((HW / 'bn_state').glob('trace_*.npz'))
G_PER_TRACE = 20000
SEED = 20260914
T = 10


def to_signed(v, bits):
    out = np.asarray(v, dtype=np.int64)
    assert np.all(out >= -(1 << (bits - 1))) and np.all(out <= (1 << (bits - 1)) - 1)
    return out


class Cal:
    """事件驱动逐拍模拟：一个 64bit 请求/响应口，一次一个在途请求。
    日历时钟 t 与全部消耗拍（命令/内部平面/retire/输出）同步推进。"""

    __slots__ = ('bp', 't', 'busy_until', 'ph')

    def __init__(self, bp):
        self.bp = bp
        self.t = 0
        self.busy_until = -1
        self.ph = 0

    def do_request(self):
        start = self.t
        while True:
            if self.t >= self.busy_until and (
                    not self.bp or self.t % 7 not in (0, 1)):
                delay = 1 + (self.ph % 3) if self.bp else 1
                self.ph += 1
                self.busy_until = self.t + delay + 1
                self.t = self.busy_until
                return self.t - start
            self.t += 1

    def do_output(self):
        while self.bp and self.t % 5 in (3, 4):
            self.t += 1
        self.t += 1


def service_cycles(mode, thr_src, y_ser, msb, j_star, cal):
    planes = (max(23 - j_star, 1) if mode == 'FX cert' else
              max(msb - j_star, 0) if mode == 'BF cert' else
              23 if mode == 'FX full' else max(msb, 1))
    cal.t += 1                                    # command
    if y_ser == 'raw_words':
        for _ in range(5):
            cal.do_request()
    else:
        ntrans = max(1, -(-(planes + 1) // 6))    # +1 sop 槽，6 平面/64b
        for _ in range(ntrans):
            cal.do_request()
    if thr_src == 'thr_port':
        for _ in range(T):
            cal.do_request()
    cal.t += 1                                    # sign/sop 拍
    cal.t += max(planes, 1)                       # 内部平面消费拍
    cal.t += 1                                    # retire
    cal.do_output()
    return cal.t


def one_trace(trace, sd):
    z = np.load(trace)
    stage = int(trace.stem.split('stage')[1])
    pre = f'sttmultires_unet.encoders.swin3d.layers.{stage}.swin_blocks.0.mlp.'
    C = int(z['W'].shape[1])
    S = np.unpackbits(z['source_packed'], axis=1, bitorder='little')[:, :C].astype(np.float64)
    N = S.shape[0]
    P = N // T
    H = z['W'].shape[0]
    W = z['W'].astype(np.float64)
    A, gamma = z['A'], z['gamma']
    R = A.sum(1).reshape(T, 1)
    direction = np.sign(gamma)

    rm = sd[pre + 'bn1.norm_layer.running_mean'].astype(np.float64)
    rv = sd[pre + 'bn1.norm_layer.running_var'].astype(np.float64)
    beta = sd[pre + 'bn1.norm_layer.bias'].astype(np.float64)
    core = (1.0 + z['center'] - z['bias'] - beta[None, :] * R)
    tau_ck = (rm[None, :] * R + np.sqrt(rv[None, :] + 1e-5) / gamma[None, :] * core) \
        * direction[None, :]
    tau_q = np.rint(tau_ck * (1 << 14)).astype(np.int64)
    thr = to_signed(np.where(direction[None, :] < 0, -(tau_q << 12) + 1, tau_q << 12), 48)

    rng = np.random.default_rng(SEED)
    ps = rng.integers(0, P, G_PER_TRACE)
    hs = rng.integers(0, H, G_PER_TRACE)
    Ssub = S.reshape(T, P, C)[:, ps, :]
    Ysub = np.einsum('tgc,gc->tg', Ssub, W[hs])
    Yq = np.clip(np.rint(Ysub * (1 << 14)), -(1 << 23), (1 << 23) - 1).astype(np.int64).T

    A_q = to_signed(np.rint(A.astype(np.float64) * 4096), 16)
    P_t = A_q.clip(min=0).sum(1)
    N_t = A_q.clip(max=0).sum(1)
    thr_g = thr[:, hs].T
    Vfull = np.einsum('gs,ts->gt', Yq, A_q)
    D_g = np.broadcast_to((direction > 0).astype(np.int64)[hs][:, None], (G_PER_TRACE, T))
    dec_full = np.where(D_g > 0, Vfull >= thr_g, ~(Vfull >= thr_g))

    j_first = np.full((G_PER_TRACE, T), -1, np.int8)
    frozen = np.zeros((G_PER_TRACE, T), bool)
    dec_raw = np.zeros((G_PER_TRACE, T), bool)
    for j in range(23, -1, -1):
        Vtop = np.einsum('gs,ts->gt', Yq >> j, A_q)
        Vmin = (Vtop << j) + N_t[None, :] * ((1 << j) - 1)
        Vmax = (Vtop << j) + P_t[None, :] * ((1 << j) - 1)
        lock = (Vmin >= thr_g) | (Vmax < thr_g)
        newly = lock & ~frozen
        j_first[newly] = j
        dec_raw[newly] = Vmin[newly] >= thr_g[newly]
        frozen |= lock
    assert frozen.all()
    assert np.array_equal(np.where(D_g > 0, dec_raw, ~dec_raw), dec_full)
    j_star = j_first.min(1)
    msb_g = np.array([int(np.abs(Yq[g]).max()).bit_length() for g in range(G_PER_TRACE)])

    out = {'trace': trace.name, 'groups': G_PER_TRACE,
           'bf_planes_mean': float(np.maximum(msb_g - j_star, 0).mean()),
           'msb_mean': float(msb_g.mean()), 'j_star_mean': float(j_star.mean())}
    for bp in (False, True):
        tag = 'bp' if bp else 'cold'
        res = {}
        for thr_src in ('thr_port', 'thr_sram'):
            for y_ser in ('raw_words', 'plane_ser'):
                for mode in ('FX full', 'FX cert', 'BF full', 'BF cert'):
                    tot = np.zeros(G_PER_TRACE, np.int64)
                    for g in range(G_PER_TRACE):
                        tot[g] = service_cycles(mode, thr_src, y_ser,
                                                int(msb_g[g]), int(j_star[g]), Cal(bp))
                    res[f'{thr_src}|{y_ser}|{mode}'] = float(tot.mean())
        out[tag] = res
    return out


def main():
    sd = read_checkpoint(CKPT)['model_state_dict']
    results = [one_trace(tr, sd) for tr in TRACES]
    for r in results:
        print('%-22s planes=%.2f msb=%.2f j*=%.2f  (BFcert cold thr_sram|plane_ser=%.1f 拍)'
              % (r['trace'], r['bf_planes_mean'], r['msb_mean'], r['j_star_mean'],
                 r['cold']['thr_sram|plane_ser|BF cert']))

    cfgs = []
    for thr_src in ('thr_port', 'thr_sram'):
        for y_ser in ('raw_words', 'plane_ser'):
            for mode in ('FX', 'BF'):
                for tag in ('cold', 'bp'):
                    full = np.mean([r[tag][f'{thr_src}|{y_ser}|{mode} full'] for r in results])
                    cert = np.mean([r[tag][f'{thr_src}|{y_ser}|{mode} cert'] for r in results])
                    cfgs.append({'thr': thr_src, 'y': y_ser, 'mode': mode, 'bp': tag,
                                 'full_cyc': full, 'cert_cyc': cert,
                                 'net_service': 1 - cert / full})
    print()
    for c in cfgs:
        print('%-9s %-10s %-2s %-4s full=%7.2f cert=%7.2f 净服务=%6.2f%%' %
              (c['thr'], c['y'], c['mode'], c['bp'], c['full_cyc'], c['cert_cyc'],
               100 * c['net_service']))
    summary = {
        'per_trace': results, 'configs': cfgs,
        'gate': '>=15% net service',
        'calendar': {'req_reject': 'cycle%7 in {0,1}',
                     'resp_delay': 'rotate 1/2/3 extra', 'out_reject': 'cycle%5 in {3,4}'},
        'deployment_notes': {
            'thr': 'T13a: track-mode BN; thr(t,h) layer-static; SRAM resident '
                   '(tau_q f14: s0 ~7.5KB / s3 ~60KB); 0 per-group reads',
            'y_consumer': 'fc1->bn1->sn2 strictly serial; I24 continuous path on '
                          'r0.conv2 edge (different graph edge); plane serialization '
                          'legal because gate is sole Y consumer',
        },
    }
    (ROOT / 'results' / 't13_net_service.json').write_text(json.dumps(summary, indent=1))
    print('saved results/t13_net_service.json')


if __name__ == '__main__':
    main()
