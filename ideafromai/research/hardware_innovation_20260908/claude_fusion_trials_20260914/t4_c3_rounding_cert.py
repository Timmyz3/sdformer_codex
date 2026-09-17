"""T-C3（C3 卡）：lifting40 半步舍入证书动态命中率——自有执行器。

证书（精确 RNE 不变式）：半步 new = RNE24((old<<12)+q·other) 与"跳过该步"
（state 保持 old）一致当且仅当 |q·other| < Q/2=2048（严格不等号避开 tie）。
本脚本在真实输入（capture_train4_ped 的 r1_sn1_input = source sn 输入探针，
(10,96,64,4)×4帧）上执行 fast_raw_diagonal 前向 8 个半步（40 次写回），
测每写的跳过命中率；并用自有 da4ml 图执行器（读 constant_compilation_graphs.json）
与直接公式交叉验证。静态杀门（系数 |q| 分级）一并输出。
样本内数值试验，非 RTL 周期。自有代码，只读他人产出的数据文件。
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HW = ROOT.parents[0]
LIFT = (HW / 'algorithm' / 'patch_probe' / 'residual_consumer_probe'
        / 'projection_chain' / 'fast_temporal_recovery_lifting40')
CAP = (HW / 'algorithm' / 'patch_probe' / 'residual_consumer_probe'
       / 'capture_train4_ped')
Q2 = 2048
XLO, XHI = -(1 << 23), (1 << 23) - 1


def rne24(n):
    q = np.floor_divide(n, 4096)
    r = n - q * 4096
    inc = (r > Q2) | ((r == Q2) & ((q & 1) == 1))
    return np.clip(q + inc, XLO, XHI)


def eval_graph(graph, states):
    """自有 da4ml 图执行器：states (V,10) int64 -> 分子 (V,5) int64。"""
    vals = []
    for n in graph['nodes']:
        if n['kind'] == 'input':
            vals.append(states[:, n['source']])
        elif n['kind'] == 'addsub':
            a = vals[n['lhs']] << n['lhs_shift']
            b = vals[n['rhs']] << n['rhs_shift']
            vals.append(a - b if n['subtract'] else a + b)
        else:
            raise ValueError(n['kind'])
    out = np.stack([vals[o['node']] * o['sign'] << o['shift']
                    for o in graph['outputs']], axis=1)
    return out


def main():
    src = np.load(LIFT / 'stage320' / 'fast_raw_diagonal.npz')
    q12 = np.rint(src['basis_lifting'].astype(np.float64) * 4096).astype(np.int64)
    matching = src['basis_matchings'].astype(np.int64)
    res = json.loads((LIFT / 'constant_compilation_result.json').read_text())
    assert np.array_equal(q12, np.asarray(res['models']['fast_raw_diagonal']['coefficients_q12']))

    graphs = json.loads((LIFT / 'constant_compilation_graphs.json').read_text())
    records = [graphs['whole_halfstage_graphs'][f'fast_raw_diagonal/forward/{i}']
               for i in range(8)]

    # 图执行器 vs 直接公式：随机合法域输入逐半步交叉验证
    rng = np.random.default_rng(7)
    probe = rng.integers(XLO, XHI + 1, size=(4096, 10))
    for rec in records:
        layer, half = rec['layer'], rec['coefficient_half']
        assert list(rec['q']) == list(q12[layer, :, half])
        assert list(rec['write_time_indices']) == list(matching[layer, :, half])
        tgt = matching[layer, :, half]
        oth = matching[layer, :, 1 - half]
        n_direct = probe[:, tgt] * 4096 + probe[:, oth] * q12[layer, :, half][None, :]
        assert np.array_equal(eval_graph(rec['whole_five_pairs_graph'], probe), n_direct)
    print('graph-vs-direct: 8 half-steps x 4096 random legal inputs, 0 mismatch')

    # 真实输入：r1_sn1_input (10,96,64,4) x4 帧 -> (V,10) signed24/f14
    states = []
    for f in sorted(CAP.glob('0*.npz')):
        x = np.load(f)['r1_sn1_input'].astype(np.float64)  # (T,C,H,W)
        states.append(np.clip(np.rint(x * 4096), XLO, XHI)
                      .reshape(10, -1).T.astype(np.int64))
    S0 = np.concatenate(states)
    print(f'inputs: {S0.shape[0]} T10 vectors '
          f'({len(list(CAP.glob("0*.npz")))} frames), '
          f'I24 range [{S0.min()}, {S0.max()}]')

    # 静态杀门：系数量级分级
    absq = np.abs(q12).ravel()
    static_gate = {
        'n_coefficients': int(absq.size),
        'lt_Q2': float((absq < Q2).mean()),
        'lt_Q4': float((absq < 1024).mean()),
        'lt_Q8': float((absq < 512).mean()),
        'absq_min': int(absq.min()), 'absq_max': int(absq.max()),
    }

    # 动态：8 半步 x 5 写，逐写判 |q*other| < 2048
    st = S0.copy()
    rows = []
    for rec in records:
        layer, half = rec['layer'], rec['coefficient_half']
        tgt = matching[layer, :, half]
        oth = matching[layer, :, 1 - half]
        for p in range(5):
            old = st[:, tgt[p]]
            other = st[:, oth[p]]
            prod = other * int(q12[layer, p, half])
            hit = np.abs(prod) < Q2
            new = rne24(old * 4096 + prod)
            # 证书精确性自检：命中处 new 必须 == old
            assert np.array_equal(new[hit], old[hit]), (layer, half, p)
            st[:, tgt[p]] = new
            rows.append(dict(layer=layer, half=half, pair=p,
                             q=int(q12[layer, p, half]),
                             n=int(hit.size), hits=int(hit.sum()),
                             other_zero=int((other == 0).sum()),
                             absq_other_p999=float(np.quantile(np.abs(prod), 0.999))))
    n_writes = sum(r['n'] for r in rows)
    n_hits = sum(r['hits'] for r in rows)
    hit_rate = n_hits / n_writes
    # 部署 raw 链最后半步 5 写是门出口（无 RNE 写回），另报 35 写口径
    n35 = sum(r['n'] for r in rows if not (r['layer'] == 3 and r['half'] == 1))
    h35 = sum(r['hits'] for r in rows if not (r['layer'] == 3 and r['half'] == 1))

    out = {
        'certificate': '|q*other| < 2048  <=>  RNE24((old<<12)+q*other) == old '
                       '(tie 严格避开，饱和不影响: |q*other|<Q/2 时 RNE 不可能离开 old)',
        'verification': '自有图执行器 vs 直接公式：8 半步 x 4096 随机合法域输入零差；'
                        '命中处 RNE 结果==old 逐写断言通过',
        'static_gate': static_gate,
        'n_vectors': int(S0.shape[0]),
        'n_writes': n_writes, 'n_hits': n_hits,
        'hit_rate_40writes': hit_rate,
        'hit_rate_35writes_numerical_only': h35 / n35,
        'per_write': [{k: v for k, v in r.items()} for r in rows],
        'note': ('输入=capture_train4_ped r1_sn1_input（source sn 输入探针，4帧/每帧96ch x 256位置，'
                 '非全帧）。跳过一个半步省：乘法+恒等合并+RNE/饱和+写回（1/40 写义务）。'
                 '门出口 5 写（layer3 half1）部署无写回，另计。'
                 '样本内数值试验，非 RTL 周期/PPA。'),
    }
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results' / 't4_c3_rounding_cert.json').write_text(
        json.dumps(out, indent=1) + '\n')

    lines = [
        '# T-C3：lifting40 半步舍入证书动态命中率（C3 卡）', '',
        f"- 输入：{S0.shape[0]:,} 个真实 T10 向量（r1_sn1_input 探针，4 帧 × 96ch × 256 位置）",
        f"- 验证：自有图执行器 vs 直接公式 8 半步零差；命中处 RNE==old 逐写断言通过", '',
        '| layer.half.pair | q | 命中/写 | 命中率 | other==0 占比 |',
        '|---|---:|---|---:|---:|',
    ]
    for r in rows:
        lines.append(f"| {r['layer']}.{r['half']}.{r['pair']} | {r['q']} | "
                     f"{r['hits']:,}/{r['n']:,} | {r['hits']/r['n']:.4f} | "
                     f"{r['other_zero']/r['n']:.4f} |")
    lines += [
        '',
        f"- **全 40 写命中率：{hit_rate:.4f}**（{n_hits:,}/{n_writes:,}）",
        f"- 35 数值写口径：{h35/n35:.4f}",
        f"- 静态杀门：|q|<2048 占比 {static_gate['lt_Q2']:.3f}，"
        f"<1024 占比 {static_gate['lt_Q4']:.3f}，<512 占比 {static_gate['lt_Q8']:.3f}"
        f"（|q| 范围 {static_gate['absq_min']}–{static_gate['absq_max']}）",
        '',
        '## 判读',
        '',
        '- 命中条件 |q·other|<2048 在 |q|≥1258 时要求 |other|≤1 量子（f14 连续残差'
        '状态几乎不可能），只有 |q|<2048 的系数（224/894 两枚）有真实命中窗口，'
        '且窗口内仍需 |other|<2048/|q|≈9.1/2.3 量子；',
        '- 杀门 2（≥15% 服务减少）：见上表命中率，若 <15% 则 **C3 动态形态杀**；',
        '- 这是 gptpro 条件 |q|·d < Q/2−|r₀| 的严格化（r₀≡0：old·4096 恒被 4096 整除），'
        '并把 tie（=2048 且 old 奇偶）排除在证书外；',
        '- 边界：输入为 256 位置/帧探针而非全帧；跳过省的是乘法+RNE+写回拍，'
        '非同端口周期结论。',
        '',
    ]
    verdict = '**C3 动态形态存活**' if hit_rate >= 0.15 else '**C3 动态形态杀**（命中率 < 15% 服务门）'
    lines.append(verdict + '：' + f'命中率 {hit_rate:.4f}。')
    (ROOT / 'results' / 'T4_C3_REPORT.md').write_text('\n'.join(lines) + '\n')
    print('hit_rate(40 writes) = %.4f  (%d/%d)' % (hit_rate, n_hits, n_writes))
    print('hit_rate(35 numerical writes) = %.4f' % (h35 / n35))
    print('static: |q|<Q2 %.3f <Q4 %.3f <Q8 %.3f' %
          (static_gate['lt_Q2'], static_gate['lt_Q4'], static_gate['lt_Q8']))


if __name__ == '__main__':
    main()
