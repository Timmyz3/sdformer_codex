#!/usr/bin/env python3
"""T45b-2：decoder/frontend 的**输入到底是不是脉冲**（决定 SOP 口径是否成立）。

## 为什么必须查

T45/T45b 用 "MAC × 输入发放率" 算活动加权 SOP，前提是**算子输入是 0/1 脉冲**。
但 `Spiking_modules.py:381-396` 的 `SpikingDecoderLayer.forward` 是：

    x_up = F.interpolate(x[i], scale_factor=2, mode=self.upsample_mode)   # 默认 bilinear
    x    = self.deconv(x_out)                                            # Conv2d
    out  = self.sn(x)

**bilinear 插值会把 0/1 脉冲变成 [0,1] 的连续值**。若部署模型真用 bilinear，
decoder 的卷积输入就是稠密的 ⇒ 它根本不是 SOP，而是 dense MAC；
T45b 里给 decoder 乘一个 <1 的发放率是**口径错误**，decoder 的真实代价
= 全 dense MAC（320.3G），而不是 61.2G。

反之若 `upsample_mode == 'nearest'`，输入仍是 0/1，SOP 口径成立。

## 另外查

1. deconv 究竟是 Conv2d（bilinear 上采样路线）还是 ConvTranspose2d；
2. 每个 decoder stage 的**输入**是谁的脉冲（本 stage 的 sn 是**输出**，
   所以 T45b 里 "decoders.N → decoders.N.sn" 的映射是**错位**的）；
3. frontend 的 `head.conv`（rate=1.0）与 `conv.conv` 输入性质。

用法：/opt/anaconda3/envs/pytorch310/bin/python t45b2_probe_inputs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

from t45_mapping_cost import PRE, build_model  # noqa: E402

ROOT = Path(__file__).resolve().parent
H, W, T = 480, 640, 10


def describe(t: torch.Tensor):
    if not torch.is_tensor(t):
        return {'type': str(type(t))}
    f = t.detach().float()
    n = f.numel()
    is01 = ((f == 0) | (f == 1)).sum().item()
    return {
        'shape': list(f.shape), 'numel': n,
        'min': float(f.min()), 'max': float(f.max()), 'mean': float(f.mean()),
        'frac_exactly_0_or_1': is01 / n,
        'frac_nonzero': float((f != 0).float().mean()),
    }


def main():
    model = build_model()
    model.eval()

    # decoder 层本体：类型 + upsample_mode
    root = getattr(model, 'sttmultires_unet', None) or model
    dec = getattr(root, 'decoders', None)
    info = {'decoder_modules': [], 'probes': {}}
    if dec is not None:
        for i, layer in enumerate(dec):
            info['decoder_modules'].append({
                'index': i, 'class': type(layer).__name__,
                'upsample_mode': getattr(layer, 'upsample_mode', None),
                'deconv_type': type(getattr(layer, 'deconv', None)[0]).__name__
                if hasattr(layer, 'deconv') else None,
                'scale': getattr(layer, 'scale', None),
            })

    names = {m: n for n, m in model.named_modules()}
    targets = []
    for n, m in model.named_modules():
        if 'decoders.' in n and isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            targets.append((n, m))
        if n.endswith('patch_embed.head.conv.0') or n.endswith('patch_embed.conv.conv.0'):
            targets.append((n, m))

    def pre(mod, inp):
        nm = names.get(mod, '?')
        info['probes'][nm] = describe(inp[0])

    hs = [m.register_forward_pre_hook(pre) for _, m in targets]
    with torch.no_grad():
        model(torch.randn(1, T, 2, H, W))
    for h in hs:
        h.remove()

    (ROOT / 'results' / 't45b2_input_probe.json').write_text(
        json.dumps(info, indent=1) + '\n')

    print('=== decoder 层构造 ===')
    print('%-4s %-32s %-12s %-18s %s'
          % ('idx', 'class', 'upsample_mode', 'deconv[0]', 'scale'))
    for d in info['decoder_modules']:
        print('%-4s %-32s %-12s %-18s %s'
              % (d['index'], d['class'], d['upsample_mode'], d['deconv_type'],
                 d['scale']))
    print('\n=== 算子输入张量性质（randn 输入，判 binarity 是结构性）===')
    print('%-64s %-22s %8s %8s %12s' % ('module', 'input shape', 'min', 'max',
                                        'frac∈{0,1}'))
    for nm in sorted(info['probes']):
        p = info['probes'][nm]
        if 'shape' not in p:
            continue
        print('%-64s %-22s %8.3f %8.3f %11.4f'
              % (nm.split('sttmultires_unet.')[-1], str(p['shape'])[:22], p['min'],
                 p['max'], p['frac_exactly_0_or_1']))
    print('\nwrote results/t45b2_input_probe.json')


if __name__ == '__main__':
    main()
