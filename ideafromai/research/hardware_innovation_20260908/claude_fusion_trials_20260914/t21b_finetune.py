#!/usr/bin/env python3
"""T21b：masked-A 掩码微调恢复（sd5ai A800）。

T21a 静态掩码的 AEE 损失用 1 epoch 微调恢复：固定 top-k 掩码（每步重投影），
标准 EPE 损失，全参数微调。产出微调后 checkpoint + valid825 评测（调用
t21a_masked_aee.py 同口径）。

自有代码。用法：
  python t21b_finetune.py --code-root <SDformer> --config <yml> \
      --checkpoint /tmp/ck.pth --data <saved_flow_data> --output <dir> \
      --k 5 --epochs 1 --lr 1e-5
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
import yaml

PREFIX = 'sttmultires_unet.encoders.swin3d.'
MLPS = [PREFIX + f'layers.{s}.swin_blocks.{b}.mlp.'
        for s, count in enumerate((2, 2, 6, 2)) for b in range(count)]
A_PARAMS = [p + 'sn2.spiking_neuron.weight' for p in MLPS]


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
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    inc = model.load_state_dict(sd, strict=False)
    assert not inc.missing_keys, inc.missing_keys[:5]
    ok_tail = ('running_mean', 'running_var', 'num_batches_tracked')
    bad = [k for k in inc.unexpected_keys if not k.endswith(ok_tail)]
    assert not bad, bad[:5]
    functional.set_step_mode(model, 'm')
    return model, cfg


def build_mask(model, k):
    mods = dict(model.named_parameters())
    masks = {}
    for name in A_PARAMS:
        a = mods[name].detach()
        keep = torch.topk(a.abs(), k, dim=1).indices
        m = torch.zeros_like(a, dtype=torch.bool)
        m.scatter_(1, keep, True)
        masks[name] = m.cuda()
    return masks


def apply_mask(model, masks):
    with torch.no_grad():
        for name, m in masks.items():
            dict(model.named_parameters())[name].mul_(m)


def main():
    p = argparse.ArgumentParser()
    for name in ('code-root', 'config', 'checkpoint', 'data', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--log-every', type=int, default=50)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    model, cfg = build_model(args)
    from spikingjelly.activation_based import functional

    masks = build_mask(model, args.k)
    apply_mask(model, masks)

    model.train()
    # 全参数微调；A 的 top-k 结构由每步掩码重投影保证
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    names = read_names(args.data, 'train')
    rng = random.Random(0)
    t0 = time.monotonic()
    hist = []
    step = 0
    for ep in range(args.epochs):
        rng.shuffle(names)
        for i, name in enumerate(names if not args.max_frames else names[:args.max_frames]):
            functional.reset_net(model)
            x, label, mask = input_frame(args.data, name)
            pred = model(x)['flow'][-1]
            err = (pred - label).square().sum(1).sqrt()
            loss = err[mask].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            apply_mask(model, masks)
            hist.append({'step': step, 'file': name, 'epe': float(loss)})
            step += 1
            del x, label, mask, pred, err, loss
            if step % args.log_every == 0:
                save_json(args.output / 'hist.json', hist[-2000:])
                print(f'EP{ep} {i+1}/{len(names)} epe={np.mean([h["epe"] for h in hist[-args.log_every:]]):.4f} '
                      f'({time.monotonic()-t0:.0f}s)', flush=True)
    model.eval()
    sd_out = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({'model_state_dict': sd_out,
                'meta': {'base': str(args.checkpoint), 'k': args.k,
                         'epochs': args.epochs, 'lr': args.lr, 'steps': step}},
               args.output / 'finetuned.pth')
    print('COMPLETE steps=%d saved=%s' % (step, args.output / 'finetuned.pth'), flush=True)


if __name__ == '__main__':
    main()
