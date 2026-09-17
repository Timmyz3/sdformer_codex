#!/usr/bin/env python3
"""T21d：微调后网络的自有 fc1 输入 capture（sd5ai）。

对 12 个 encoder fc1 的输入（sn1 输出脉冲）逐样本捕获并打包为
{(sid,lid): packed uint8} npz，供 t21 供数测量。cohort 复刻 m1707
sample_order.json 的 40 个 valid 样本（global_sample_id 顺序 = sid）。

验证模式（--verify）：对基线 checkpoint 跑前 2 个样本，与 m1707
fc_frames.bin 对应 (sid,lid) 的 packed 位流逐位比对。

自有代码。用法：
  python t21d_capture.py --code-root <SDformer> --config <yml> \
      --checkpoint <pth> --data <saved_flow_data> --fcframes <m1707>/fc_frames.bin \
      --output /tmp/t21d_cap.npz [--verify]
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import struct
import sys
import types
import zlib
from pathlib import Path

import os
os.environ['SDFORMER_USE_MLFLOW'] = '0'
os.environ['SDFORMER_MLFLOW_MODEL_LOGGING'] = '0'
sys.modules.setdefault('mlflow', types.ModuleType('mlflow'))

import numpy as np
import torch
import yaml

PREFIX = 'sttmultires_unet.encoders.swin3d.'
FC1 = [PREFIX + f'layers.{s}.swin_blocks.{b}.mlp.fc1'
       for s, count in enumerate((2, 2, 6, 2)) for b in range(count)]
LIDS = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
HEADER = struct.Struct('<8sHH11I')


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
    return x


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
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            m.num_batches_tracked = None
    return model


def read_m1707(fcframes, sids):
    """读 m1707 fc_frames.bin 的 {(sid,lid): packed}（自有解析，验证用）。"""
    want = {(sid, lid) for sid in sids for lid in LIDS}
    chunks = {k: [] for k in want}
    meta = {}
    with open(fcframes, 'rb') as f:
        while raw := f.read(HEADER.size):
            magic, ver, hs, lid, sid, fi, start, n, C, br, nnz, rb, cb, _ = \
                HEADER.unpack(raw)
            if (sid, lid) not in chunks:
                f.seek(cb, 1)
                continue
            payload = zlib.decompress(f.read(cb))
            if (sid, lid) in meta and meta[sid, lid][0] != C:
                raise ValueError('C changed')
            meta[sid, lid] = (C, br)
            chunks[sid, lid].append(payload[:n * br])
    out = {}
    for k, lst in chunks.items():
        if lst:
            C, br = meta[k]
            out[k] = (np.concatenate([np.frombuffer(p, np.uint8) for p in lst]), C, br)
    return out


def main():
    p = argparse.ArgumentParser()
    for name in ('code-root', 'config', 'checkpoint', 'data', 'fcframes'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--sample-order', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--verify', action='store_true')
    args = p.parse_args()

    order = json.loads(args.sample_order.read_text())
    names = [s['sample_key'] for s in order['samples']]
    sids = list(range(len(names)))
    if args.verify:
        names, sids = names[:2], sids[:2]

    model = build_model(args)
    from spikingjelly.activation_based import functional

    mods = dict(model.named_modules())
    captured = {}

    def make_hook(lid):
        def hook(module, args_):
            x = args_[0].detach()
            bits = (x != 0).reshape(-1, x.shape[-1]).cpu().numpy().astype(np.uint8)
            captured[lid] = np.packbits(bits, axis=-1, bitorder='little')
        return hook
    handles = [mods[name].register_forward_pre_hook(make_hook(lid))
               for name, lid in zip(FC1, LIDS)]

    store = {}
    with torch.no_grad():
        for sid, name in zip(sids, names):
            functional.reset_net(model)
            x = input_frame(args.data, name)
            model(x)
            for lid in LIDS:
                packed = captured.pop(lid)
                store['s%02d_L%02d' % (sid, lid)] = packed
            print('CAPTURED', sid, name, flush=True)
            del x
    for h in handles:
        h.remove()

    if args.verify:
        ref = read_m1707(args.fcframes, sids)
        nbad = 0
        for (sid, lid), (ref_packed, C, br) in ref.items():
            mine = store['s%02d_L%02d' % (sid, lid)].reshape(-1)
            ref_flat = ref_packed.reshape(-1)
            same = mine.size == ref_flat.size and bool((mine == ref_flat).all())
            diff_bits = None if same or mine.size != ref_flat.size else \
                int((mine != ref_flat).sum())
            print('VERIFY sid=%d L%02d mine=%d ref=%d %s%s'
                  % (sid, lid, mine.size, ref_flat.size,
                     'EXACT' if same else 'DIFF',
                     '' if diff_bits is None else ' bits=%d' % diff_bits),
                  flush=True)
            nbad += 0 if same else 1
        print('VERIFY_RESULT bad=%d/%d' % (nbad, len(ref)), flush=True)
        np.savez_compressed(args.output, **store)
        print('VERIFY_SAVED %s' % args.output, flush=True)
        return

    arrays = {k: v for k, v in store.items()}
    np.savez_compressed(args.output, **arrays)
    print('COMPLETE saved=%s arrays=%d' % (args.output, len(arrays)), flush=True)


if __name__ == '__main__':
    main()
