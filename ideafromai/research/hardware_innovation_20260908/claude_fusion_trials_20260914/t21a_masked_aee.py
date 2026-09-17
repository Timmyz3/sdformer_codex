#!/usr/bin/env python3
"""T21a：静态 A 稀疏化（top-k per row）的 valid825 AEE 评测（sd5ai A800）。

对 12 个 encoder fc1 的 sn2.spiking_neuron.weight（A，10×10 时间混合）做逐行
|a| top-k 掩码，k∈{10(基线),5,2}；动态 BN（no_running，与参考 AEE 同口径）。
每帧同时测全头 AEE（model(x)['flow'][-1]）与粗头 AEE（preds.2 时间和 +
双线性上采样，与 evaluate_coarse_readout 同式）。基线粗头应复现 ≈1.081367。

自有代码（参照 run_bn_probe/evaluate_coarse_readout 的公开调用模式重写）。
用法：
  python t21a_masked_aee.py --code-root <SDformer> --config <yml> \
      --checkpoint /tmp/ck.pth --data <saved_flow_data> --output <dir> \
      --ks 10,5,2
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
import types
from pathlib import Path

import os
os.environ['SDFORMER_USE_MLFLOW'] = '0'
os.environ['SDFORMER_MLFLOW_MODEL_LOGGING'] = '0'
sys.modules.setdefault('mlflow', types.ModuleType('mlflow'))

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PREFIX = 'sttmultires_unet.encoders.swin3d.'
A_KEYS = [PREFIX + f'layers.{s}.swin_blocks.{b}.mlp.sn2.spiking_neuron.weight'
          for s, count in enumerate((2, 2, 6, 2)) for b in range(count)]
COARSE = 'sttmultires_unet.preds.2'


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def read_names(data, split):
    with (data / 'sequence_lists' / f'{split}_split_seq.csv').open() as f:
        return [row[0] for row in csv.reader(f) if row]


def input_frame(data, name):
    seq = name.rsplit('_', 1)[0]
    x = torch.from_numpy(np.load(data / 'event_tensors' / '10bins' / 'left' / seq / name)).cuda().float()[None]
    pos, neg = x.relu(), (-x).relu()
    x = torch.stack((pos, neg), dim=2)
    nz = x != 0
    if nz.any():
        lo, hi = x[nz].min(), x[nz].max()
        if hi != lo:
            x[nz] = (x[nz] - lo) / (hi - lo)
    label = torch.from_numpy(np.load(data / 'gt_tensors' / name)).cuda().float()[None]
    mask = torch.from_numpy(np.load(data / 'mask_tensors' / name)).cuda().bool().reshape(1, 480, 640)
    return x, label, mask


def set_bn_dynamic(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
            module.num_batches_tracked = None


def mask_state(sd, k):
    if k >= 10:
        return sd, 0
    out = dict(sd)
    zeroed = 0
    for key in A_KEYS:
        a = sd[key].clone()
        keep = torch.topk(a.abs(), k, dim=1).indices
        m = torch.zeros_like(a, dtype=torch.bool)
        m.scatter_(1, keep, True)
        zeroed += int((~m).sum())
        out[key] = a * m
    return out, zeroed


def build_model(args):
    sys.path[:0] = [str(args.code_root / 'neuron_experiments/H9_bipolar_self_attention/overlay'),
                    str(args.code_root / 'third_party/SDformerFlow')]
    from configs.parser import YAMLParser
    from models.STSwinNet_SNN.Spiking_STSwinNet import MS_SpikingformerFlowNet_en4
    from models.STSwinNet_SNN.atlif_ternary_psn import install_atlif_ternary_psn
    from models.STSwinNet_SNN.bsa_attention import install_shiftmax_attention, register_shiftmax_pickle_compat
    from spikingjelly.activation_based import functional

    cfg = YAMLParser.combine_entries(yaml.safe_load(args.config.read_text()))
    cfg['data']['path'] = str(args.data)
    cfg['swin_transformer']['input_size'] = [480, 640]
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg['runtime']['allow_tf32'])
    torch.backends.cudnn.allow_tf32 = bool(cfg['runtime']['allow_tf32'])
    torch.backends.cudnn.benchmark = bool(cfg['runtime']['cudnn_benchmark'])
    model = MS_SpikingformerFlowNet_en4(copy.deepcopy(cfg['model']), copy.deepcopy(cfg['swin_transformer'])).cuda()
    model.init_weights()
    install_atlif_ternary_psn(model, cfg['atlif_ternary_psn'])
    install_shiftmax_attention(model, cfg['bsa_attention'])
    register_shiftmax_pickle_compat()
    base_sd = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['model_state_dict']
    assert all(k in base_sd for k in A_KEYS), 'A keys missing in checkpoint'
    functional.set_step_mode(model, 'm')
    model.eval()
    set_bn_dynamic(model)
    return model, base_sd


def main():
    p = argparse.ArgumentParser()
    for name in ('code-root', 'config', 'checkpoint', 'data', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--ks', type=str, default='10,5,2')
    p.add_argument('--max-frames', type=int, default=0)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    model, base_sd = build_model(args)
    from spikingjelly.activation_based import functional

    names = read_names(args.data, 'valid')
    if args.max_frames:
        names = names[:args.max_frames]

    coarse_mod = dict(model.named_modules())[COARSE]
    captured = {}

    def coarse_hook(module, inputs, output):
        captured['flow'] = output.detach().sum(0)
    handle = coarse_mod.register_forward_hook(coarse_hook)

    results = {}
    for k in [int(x) for x in args.ks.split(',')]:
        sd, zeroed = mask_state(base_sd, k)
        inc = model.load_state_dict(sd, strict=False)
        assert not inc.missing_keys, inc.missing_keys[:5]
        ok_tail = ('running_mean', 'running_var', 'num_batches_tracked')
        bad = [key for key in inc.unexpected_keys if not key.endswith(ok_tail)]
        assert not bad, bad[:5]
        set_bn_dynamic(model)
        rows = []
        t0 = time.monotonic()
        with torch.no_grad():
            for i, name in enumerate(names):
                functional.reset_net(model)
                x, label, mask = input_frame(args.data, name)
                pred = model(x)['flow'][-1]
                err_full = (pred - label).square().sum(1).sqrt()
                low = captured.pop('flow')
                pred_c = F.interpolate(low, size=(480, 640), mode='bilinear', align_corners=False)
                err_c = (pred_c - label).square().sum(1).sqrt()
                count = int(mask.sum())
                row = {'file': name, 'valid_pixels': count,
                       'aee_full': float(err_full[mask].sum()) / count,
                       'aee_coarse': float(err_c[mask].sum()) / count}
                rows.append(row)
                del x, label, mask, pred, pred_c, err_full, err_c, low
                if (i + 1) % 25 == 0 or i + 1 == len(names):
                    save_json(args.output / f'k{k}_frames.json', rows)
                    print(f'K{k} FRAME {i+1}/{len(names)} '
                          f'full={np.mean([r["aee_full"] for r in rows]):.6f} '
                          f'coarse={np.mean([r["aee_coarse"] for r in rows]):.6f} '
                          f'({time.monotonic()-t0:.0f}s)', flush=True)
        vp = sum(r['valid_pixels'] for r in rows)
        results[f'k{k}'] = {
            'frames': len(rows), 'valid_pixels': vp, 'a_zeroed': zeroed,
            'AEE_full_frame_mean': float(np.mean([r['aee_full'] for r in rows])),
            'AEE_full_pixel_mean': sum(r['aee_full'] * r['valid_pixels'] for r in rows) / vp,
            'AEE_coarse_frame_mean': float(np.mean([r['aee_coarse'] for r in rows])),
            'AEE_coarse_pixel_mean': sum(r['aee_coarse'] * r['valid_pixels'] for r in rows) / vp,
            'elapsed_s': time.monotonic() - t0,
        }
        save_json(args.output / 'summary.json', results)
        print(f'K{k} DONE', json.dumps(results[f'k{k}']), flush=True)
    handle.remove()
    print('COMPLETE', json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
