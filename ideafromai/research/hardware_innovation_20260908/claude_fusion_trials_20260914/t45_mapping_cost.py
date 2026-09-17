#!/usr/bin/env python3
"""T45：**整个硬件映射**的成本分布（逐模块 MAC / 参数 / 存储），不是只数 C1 那条链路。

## 为什么要做这一步（用户 2026-09-17 指示）

C1（组级 BFP + 位平面证书 + 逐 t 退休）只覆盖 **fc1→sn2 这一个门路链接**
（12 个 swin-MLP 层）。"抄 BitFair 抄全"同样只动这个门的阈值。
但整张网络的映射里，注意力分支、卷积前端、decoder/pred 头、以及膜的 T 步状态
全都在这条链路之外。**必须先量出钱花在哪**，否则我们是在优化一个次级项
（T40f 的教训：目标函数选错 ⇒ 收敛到退化解）。

## 两个口径（必须分开报，不能混）

1. **dense 等效 MAC**：forward hook 在真实张量上数出来的乘加数（所有神经元全发放）。
   这是"如果做成稠密阵列要花多少"。
2. **活动加权 SOP**：dense MAC × 该算子**输入**的发放率（spike-driven 阵列的实际运算数）。
   这是 SNN 加速器的标准口径。

⚠ 活动率来源：`neuron_experiments/_profiles/sops_20260511_120258/layer_firing_rates.csv`
（crop 288×384、40 样本）。**该 profile 用的是另一个 checkpoint（tokenmix_pool），
不是 ep34 部署权重**——架构相同（层名逐一对上），但活动率必须用我们的权重复测才算数。
本表的活动率列标为 `RATE_SRC=foreign_ckpt`，只作量级判断，不作论文数字。

用法：/opt/anaconda3/envs/pytorch310/bin/python t45_mapping_cost.py [--dump-names]
"""
from __future__ import annotations

import copy
import json
import random
import sys
import types
from pathlib import Path

import os

os.environ['SDFORMER_USE_MLFLOW'] = '0'
os.environ['SDFORMER_MLFLOW_MODEL_LOGGING'] = '0'
sys.modules.setdefault('mlflow', types.ModuleType('mlflow'))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parent
CODE = Path('/home/zhumd/work/sdformer_codex/SDformer')
CONFIG = (CODE / 'hw_autoresearch_nts07/system_handoff/incoming'
          / 'm2041_ep34_quant_binding_inputs/dsec_c12_alpha0125_ep29_resume5_20260830.yml')
H, W = 480, 640
T = 10


def build_model():
    sys.path[:0] = [str(CODE / 'neuron_experiments/H9_bipolar_self_attention/overlay'),
                    str(CODE / 'third_party/SDformerFlow')]
    from configs.parser import YAMLParser
    from models.STSwinNet_SNN.Spiking_STSwinNet import MS_SpikingformerFlowNet_en4
    from models.STSwinNet_SNN.atlif_ternary_psn import install_atlif_ternary_psn
    from models.STSwinNet_SNN.bsa_attention import (install_shiftmax_attention,
                                                    register_shiftmax_pickle_compat)

    cfg = YAMLParser.combine_entries(yaml.safe_load(CONFIG.read_text()))
    cfg['swin_transformer']['input_size'] = [H, W]
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    model = MS_SpikingformerFlowNet_en4(copy.deepcopy(cfg['model']),
                                       copy.deepcopy(cfg['swin_transformer']))
    model.init_weights()
    install_atlif_ternary_psn(model, cfg['atlif_ternary_psn'])
    install_shiftmax_attention(model, cfg['bsa_attention'])
    register_shiftmax_pickle_compat()
    from spikingjelly.activation_based import functional
    functional.set_step_mode(model, 'm')
    return model


def bucket(name: str) -> str:
    p = name.split('.')
    if name.startswith('sttmultires_unet.encoders.swin3d.patch_embed'):
        seg = name.split('patch_embed.')[-1].split('.')[0]
        return 'frontend.patch_embed.%s' % seg
    if 'encoders.swin3d.layers.' in name:
        i = p.index('layers') + 1
        st = p[i]
        rest = '.'.join(p[i + 1:])
        if rest.startswith('swin_blocks.'):
            j = p.index('swin_blocks') + 1
            blk, tail = p[j], '.'.join(p[j + 1:])
            which = 'attn' if tail.startswith('attn') else ('mlp' if tail.startswith('mlp') else 'other')
            if which in ('attn', 'mlp'):
                leaf = '.'.join(tail.split('.')[1:2])
                return 'enc.stage%s.%s.%s' % (st, which, leaf if leaf else 'body')
            return 'enc.stage%s.block%s.%s' % (st, blk, which)
        if rest.startswith('downsample'):
            return 'enc.stage%s.downsample' % st
        return 'enc.stage%s.%s' % (st, rest.split('.')[0])
    if 'sttmultires_unet.resblocks' in name:
        return 'bottleneck.resblocks'
    if 'sttmultires_unet.decoders' in name:
        return 'decoder.%s' % name.split('decoders.')[1].split('.')[0]
    if 'sttmultires_unet.preds' in name:
        return 'pred.%s' % name.split('preds.')[1].split('.')[0]
    return 'other.' + name.split('.', 1)[0]


PRE = 'sttmultires_unet.encoders.swin3d.'
RATES_CSV = (CODE / 'neuron_experiments/_profiles/sops_20260511_120258'
             / 'layer_firing_rates.csv')
EP34_PROFILE = (CODE / 'hw_autoresearch_nts07/system_handoff/incoming'
                / 'm2041_ep34_quant_binding_inputs/spike_profile.json')


def load_rates(src: str = 'ep34'):
    """neuron 名 → 发放率。返回 (rates, label)；不可用时 rates=None。

    src='ep34'   : 我们自己的 ep34 部署权重，spike_profile.json（480x640, 825 样本，
                   T=10, no_running BN）。这是唯一可作论文数字的来源。
    src='foreign': neuron_experiments 里旧 profile（另一个 ckpt，288x384 crop，
                   40 样本）。只作量级对照。
    """
    if src == 'ep34':
        if not EP34_PROFILE.exists():
            return None, 'none'
        d = json.loads(EP34_PROFILE.read_text())
        out = {k: float(v['firing_rate'])
               for k, v in d['layer_firing_rates'].items()}
        return out, 'ep34_m2041_spike_profile(480x640,825samples,T=10,no_running_bn)'
    if not RATES_CSV.exists():
        return None, 'none'
    import csv
    out = {}
    with RATES_CSV.open() as f:
        for row in csv.DictReader(f):
            out[row['layer']] = float(row['firing_rate'])
    return out, 'foreign_ckpt:sops_20260511_120258(288x384,40samples)'


def norm_name(n: str) -> str:
    """去掉 spikingjelly 的容器后缀：至多一个尾部索引 + 至多一个 .conv* 包装。"""
    parts = n.split('.')
    if parts and parts[-1].isdigit():
        parts.pop()
    if (len(parts) >= 2 and parts[-1] in ('conv', 'conv_res')
            and parts[-2] not in ('patch_embed',)):
        parts.pop()
    return '.'.join(parts)


def rate_key(name: str):
    """该算子的输入发放率对应**哪个 neuron 的键**；None 表示无对应神经元。

    tuple 表示输入是**两个脉冲张量的逐元素积**（双操作数），此时
    firing fraction = P(两者同时发放)，独立假设下取 r_a·r_b。
    """
    n = norm_name(name)
    if n.endswith('mlp.fc1'):
        return n[:-3] + 'sn1'
    if n.endswith('mlp.fc2'):
        return n[:-3] + 'sn2'
    if n.endswith('attn.linear_q') or n.endswith('attn.linear_k'):
        return n.rsplit('.', 1)[0] + '.proj_sn'
    if n.endswith('attn.proj'):
        # Spiking_swin_transformer3D.py:709-712：proj 吃的是
        # x = reshape(k.mul(att_token))，即 sn_k ⊙ sn2_q 的逐元素积。
        # `attn = self.attn_sn(x)` 是只进调试返回的死结果（生产树已确认
        # 12 个 attn_sn 全为 dead_debug），**不是** proj 的输入。
        # 真值应为 P(sn_k ∧ sn2_q)；但 sn2_q **未被 profile 收录**
        # （93 层里只有 sn_q，没有 sn2_q），故退而用 P ≤ min(r_k, r_sn2q) ≤ r_k
        # 作为**上界**，并在表里标 ok_upperbound。
        stem = n.rsplit('.', 1)[0]
        return ('@UB', stem + '.sn_k')
    if n.endswith('patch_embed.head'):
        return '@dense_input'             # 前端输入是连续 voxel，不是脉冲
    if n.endswith('patch_embed.conv'):
        return PRE + 'patch_embed.head.sn'
    if 'patch_embed.residual_encoding.resblocks.' in n:
        j = n.index('resblocks.')
        stem = n[:j] + 'resblocks.' + n[j + len('resblocks.'):].split('.')[0]
        return stem + ('.sn1' if n.endswith('conv1') else '.sn2')
    if n.endswith('patch_embed.proj'):
        return PRE + 'patch_embed.residual_encoding.resblocks.1.sn2'
    if '.downsample' in n:
        return n.rsplit('.', 1)[0] + '.sn'
    if 'sttmultires_unet.resblocks.' in n:
        j = n.index('resblocks.')
        stem = 'sttmultires_unet.resblocks.' + n[j + len('resblocks.'):].split('.')[0]
        return stem + ('.sn1' if n.endswith('conv1') else '.sn2')
    if 'sttmultires_unet.decoders.' in n:
        return 'sttmultires_unet.decoders.' + n.split('decoders.')[1].split('.')[0] + '.sn'
    if 'sttmultires_unet.preds.' in n:
        return 'sttmultires_unet.preds.' + n.split('preds.')[1].split('.')[0] + '.sn'
    return None


def input_rate(name: str, rates: dict):
    """该算子的输入发放率（由它前面那个神经元的名字查表）。"""
    if rates is None:
        return None, 'none'
    k = rate_key(name)
    if k is None:
        return None, 'no_neuron_mapped'
    if k == '@dense_input':
        return 1.0, 'ok'
    if isinstance(k, tuple):
        if k[0] == '@UB':
            v = rates.get(k[1])
            if v is None:
                return None, 'missing:%s' % k[1]
            return float(v), 'ok_upperbound:%s' % k[1]
        vs = [rates.get(x) for x in k]
        if any(v is None for v in vs):
            miss = [x for x, v in zip(k, vs) if v is None]
            return None, 'missing:%s' % ','.join(miss)
        return float(np.prod(vs)), 'ok_prod2(independent)'
    v = rates.get(k)
    return v, ('ok' if v is not None else 'missing:%s' % k)


def main():
    argv = sys.argv[1:]
    dump_names = '--dump-names' in argv
    src = 'ep34'
    if '--rate-src' in argv:
        src = argv[argv.index('--rate-src') + 1]
    out_name = ('t45b_mapping_cost_ep34.json' if src == 'ep34'
                else 't45_mapping_cost.json')
    model = build_model()
    model.eval()

    macs = {}
    params = {}

    def hook(mod, inp, out):
        n = out[0] if isinstance(out, (tuple, list)) else out
        if not torch.is_tensor(n) or n.dim() == 0:
            return
        if isinstance(mod, torch.nn.Linear):
            per = mod.in_features
            cnt = n.numel() * per
        else:
            k = mod.kernel_size
            per = mod.in_channels * (k[0] * k[1] if len(k) > 1 else k[0])
            cnt = n.numel() * per
        macs[mod] = cnt

    handles = []
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d,
                          torch.nn.ConvTranspose2d, torch.nn.Conv3d)):
            handles.append(m.register_forward_hook(hook))
            params[m] = sum(p.numel() for p in m.parameters())

    x = torch.randn(1, T, 2, H, W)
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()

    names = {m: n for n, m in model.named_modules()}
    rates, rate_label = load_rates(src)
    agg = {}
    missing = {}
    used_keys = set()
    detail = {}
    for m, c in macs.items():
        nm = names[m]
        b = bucket(nm)
        r, why = input_rate(nm, rates)
        d = agg.setdefault(b, {'macs': 0, 'params': 0, 'nmods': 0, 'sops': 0.0,
                               'rated_mods': 0})
        d['macs'] += int(c)
        d['params'] += int(params.get(m, 0))
        d['nmods'] += 1
        if r is not None:
            d['sops'] += float(c) * r
            d['rated_mods'] += 1
        else:
            missing[nm] = why
        detail[nm] = {'macs': int(c), 'rate': r, 'rate_src': why,
                      'sops': (float(c) * r) if r is not None else None,
                      'group': b}
    for nm in detail:
        r, _ = input_rate(nm, rates)
        if r is not None:
            k = rate_key(nm)
            used_keys.update(k if isinstance(k, tuple) else (k,))
    unused_keys = sorted(set(rates or {}) - used_keys)

    if dump_names:
        for nm in sorted(detail):
            print('%-88s %10.3fM  rate=%s' % (nm, detail[nm]['macs'] / 1e6,
                                              detail[nm]['rate']))

    total_mac = sum(d['macs'] for d in agg.values())
    total_par = sum(d['params'] for d in agg.values())
    # 未标定活动率的算子按 dense 计入（保守：高估其 SOP）
    total_sop = sum(d['sops'] if d['rated_mods'] else d['macs'] for d in agg.values())
    out = {
        'crop': [H, W], 'T': T,
        'rate_src': rate_label,
        'total_macs': total_mac, 'total_params': total_par, 'total_sops': total_sop,
        'unrated_modules': missing,
        'detail': detail,
        'groups': {k: {'macs': v['macs'], 'mac_pct': 100.0 * v['macs'] / total_mac,
                       'sops': v['sops'] if v['rated_mods'] else v['macs'],
                       'sop_pct': 100.0 * (v['sops'] if v['rated_mods'] else v['macs']) / total_sop,
                       'params': v['params'], 'nmods': v['nmods'],
                       'rated_mods': v['rated_mods']}
                   for k, v in sorted(agg.items(),
                                      key=lambda kv: -(kv[1]['sops'] if kv[1]['rated_mods']
                                                       else kv[1]['macs']))},
    }
    (ROOT / 'results' / out_name).write_text(json.dumps(out, indent=1) + '\n')

    print('=== 总 dense MAC %.4fG / 活动加权 SOP %.4fG / 参数 %.3fM'
          ' （%dx%d, T=%d） ===' % (total_mac / 1e9, total_sop / 1e9,
                                    total_par / 1e6, H, W, T))
    print('%-32s %9s %7s %10s %7s %9s' % ('module group', 'MAC(G)', 'MAC%',
                                          'SOP(G)', 'SOP%', 'params(M)'))
    for k, v in out['groups'].items():
        print('%-32s %9.3f %6.2f%% %10.3f %6.2f%% %9.3f'
              % (k, v['macs'] / 1e9, v['mac_pct'], v['sops'] / 1e9, v['sop_pct'],
                 v['params'] / 1e6))
    if missing:
        print('\n⚠ 未标定活动率的算子（按 dense 计入，高估）：%d 个' % len(missing))
        for nm in list(missing)[:8]:
            print('   ', nm)
    print('wrote results/%s' % out_name)


if __name__ == '__main__':
    main()
