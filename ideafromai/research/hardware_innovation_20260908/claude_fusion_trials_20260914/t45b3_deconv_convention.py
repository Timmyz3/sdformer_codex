#!/usr/bin/env python3
"""T45b-3：decoder 反卷积的两种实现口径差多少（stride-2 ConvTranspose）。

## 问题

`MS_SpikingTransposeDecoderLayer.forward`（Spiking_modules.py:467-474）是

    x = self.sn(x)        # 0/1 脉冲
    x = self.deconv(x)    # ConvTranspose2d, stride=2, output_padding=1
    x = self.norm_layer(x)

deconv 是 **stride-2 转置卷积**。它的 MAC 数取决于实现：

- **口径 A（零插值+普通卷积）**：框架/稠密阵列的标准做法
  （cuDNN 就这么干，ptflops 也这么数）。先把输入零插值到 2×分辨率再做 k×k 卷积。
  代价 = out_elems × in_ch × k² —— 但其中只有 1/stride² 的抽头落在非零输入上，
  其余 3/4 是**纯浪费**。
- **口径 B（直接散射 / 输入驻留）**：每个**输入**脉冲散射到 out_ch × k² 个输出位置。
  代价 = in_elems × in_ch × out_ch × k²（的脉冲驱动版本）
       = in_elems × r × out_ch × k²。
  数学上与口径 A 等价，但**不做零插值**，因此省掉 stride² 倍。

两者恰好差 stride² = 4 倍。T45/T45b 用的是口径 A（forward hook 按输出数），
所以 decoder 那 60.2% 里有一大块是**零插值浪费**，不是真工作量。

本脚本按同一份真实张量形状把两个口径都算出来，供 T47 决定映射策略。

用法：/opt/anaconda3/envs/pytorch310/bin/python t45b3_deconv_convention.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

from t45_mapping_cost import build_model, load_rates  # noqa: E402

ROOT = Path(__file__).resolve().parent
H, W, T = 480, 640, 10


def main():
    model = build_model()
    model.eval()
    rates, label = load_rates('ep34')
    names = {m: n for n, m in model.named_modules()}

    shapes = {}

    def pre(mod, inp):
        shapes[names[mod]] = list(inp[0].shape)

    targets = [(n, m) for n, m in model.named_modules()
               if 'decoders.' in n and isinstance(m, torch.nn.ConvTranspose2d)]
    hs = [m.register_forward_pre_hook(pre) for _, m in targets]
    with torch.no_grad():
        model(torch.randn(1, T, 2, H, W))
    for h in hs:
        h.remove()

    rows = []
    for n, m in targets:
        in_shape = shapes[n]
        s = int(m.stride[0])
        k = int(m.kernel_size[0])
        N_in = 1
        for d in in_shape:
            N_in *= d
        in_ch, out_ch = m.in_channels, m.out_channels
        r = rates.get('sttmultires_unet.decoders.%s.sn' % n.split('decoders.')[1].split('.')[0])
        in_pos = N_in // in_ch                      # 每通道的 (T,B,H,W) 位置数
        mac_a = in_pos * s * s * in_ch * out_ch * k * k   # 口径 A：零插值 + 卷积
        mac_b = in_pos * in_ch * out_ch * k * k           # 口径 B：输入散射
        rows.append({
            'module': n, 'in_shape': in_shape, 'in_ch': in_ch, 'out_ch': out_ch,
            'stride': s, 'kernel': k,
            'output_padding': int(m.output_padding[0]), 'padding': int(m.padding[0]),
            'input_rate': r,
            'mac_A_zero_insert': mac_a, 'mac_B_direct': mac_b,
            'ratio_A_over_B': mac_a / mac_b,
            'sop_A': mac_a * r, 'sop_B': mac_b * r,
        })

    tot = {kk: sum(x[kk] for x in rows) for kk in
           ('mac_A_zero_insert', 'mac_B_direct', 'sop_A', 'sop_B')}
    alljson = json.loads((ROOT / 'results' / 't45b_mapping_cost_ep34.json').read_text())
    tot_all = {'macs': alljson['total_macs'], 'sops': alljson['total_sops']}

    out = {'rate_src': label, 'rows': rows, 'decoder_totals': tot,
           'network_totals_hook': tot_all}
    (ROOT / 'results' / 't45b3_deconv_convention.json').write_text(
        json.dumps(out, indent=1) + '\n')

    print('=== decoder 各级 stride-2 转置卷积的两种口径 ===')
    print('%-18s %-24s %5s %5s %8s %10s %10s %7s'
          % ('stage', 'in_shape', 's', 'k', 'rate', 'MAC_A(G)', 'MAC_B(G)', 'A/B'))
    for x in rows:
        print('%-18s %-24s %5d %5d %8.4f %10.3f %10.3f %7.3f'
              % (x['module'].split('sttmultires_unet.')[-1], str(x['in_shape'])[:24],
                 x['stride'], x['kernel'], x['input_rate'],
                 x['mac_A_zero_insert'] / 1e9, x['mac_B_direct'] / 1e9,
                 x['ratio_A_over_B']))
    print('\n%-28s %10s %10s %10s %10s' % ('', 'MAC_A(G)', 'MAC_B(G)', 'SOP_A(G)', 'SOP_B(G)'))
    print('%-28s %10.3f %10.3f %10.3f %10.3f'
          % ('decoder 合计', tot['mac_A_zero_insert'] / 1e9, tot['mac_B_direct'] / 1e9,
             tot['sop_A'] / 1e9, tot['sop_B'] / 1e9))
    print('%-28s %10.3f %10s %10.3f %10s'
          % ('全网（hook 口径=A）', tot_all['macs'] / 1e9, '-', tot_all['sops'] / 1e9, '-'))
    net_b = tot_all['sops'] - tot['sop_A'] + tot['sop_B']
    print('\n若 decoder 用口径 B：全网活动加权 SOP = %.3fG（原 %.3fG，省 %.1f%%）'
          % (net_b / 1e9, tot_all['sops'] / 1e9,
             100 * (1 - net_b / tot_all['sops'])))
    print('decoder 占全网 SOP：口径 A %.1f%% → 口径 B %.1f%%'
          % (100 * tot['sop_A'] / tot_all['sops'], 100 * tot['sop_B'] / net_b))
    print('\nwrote results/t45b3_deconv_convention.json')


if __name__ == '__main__':
    main()
