"""Bitmap-G=32 skip on every Swin MLP Linear. 1 cycle/token dense, inspect 32 valids/cycle + 1 MAC/nz."""
from __future__ import annotations
import json, os, sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ALGO = HERE.parents[1] / 'algorithm'
REPO = Path('/home/zhumd/work/sdformer_codex/SDformer')
DATA = REPO / 'data/Datasets/DSEC/saved_flow_data'
INCOMING = REPO / 'hw_autoresearch_nts07/system_handoff/incoming/m2041_ep34_quant_binding_inputs'
sys.path[:0] = [str(ALGO), str(ALGO / 'nrv_cost_probe')]
os.environ['SDFORMER_USE_MLFLOW'] = '0'
os.environ['SDFORMER_MLFLOW_MODEL_LOGGING'] = '0'
from run_bn_probe import build_model, input_frame
from spikingjelly.activation_based import functional

G = 32
NET = 596.5464288e9
MLP = 156.5411328e9
MAC = {
    'layers.0.swin_blocks.0.mlp.fc1': 7.08e9, 'layers.0.swin_blocks.0.mlp.fc2': 7.08e9,
    'layers.0.swin_blocks.1.mlp.fc1': 7.08e9, 'layers.0.swin_blocks.1.mlp.fc2': 7.08e9,
    'layers.1.swin_blocks.0.mlp.fc1': 7.08e9, 'layers.1.swin_blocks.0.mlp.fc2': 7.08e9,
    'layers.1.swin_blocks.1.mlp.fc1': 7.08e9, 'layers.1.swin_blocks.1.mlp.fc2': 7.08e9,
    'layers.2.swin_blocks.0.mlp.fc2': 7.08e9, 'layers.2.swin_blocks.1.mlp.fc2': 7.08e9,
    'layers.2.swin_blocks.2.mlp.fc2': 7.08e9, 'layers.2.swin_blocks.3.mlp.fc2': 7.08e9,
    'layers.2.swin_blocks.4.mlp.fc2': 7.08e9, 'layers.2.swin_blocks.5.mlp.fc2': 7.08e9,
    'layers.3.swin_blocks.0.mlp.fc1': 7.08e9, 'layers.3.swin_blocks.0.mlp.fc2': 7.08e9,
    'layers.3.swin_blocks.1.mlp.fc1': 7.08e9, 'layers.3.swin_blocks.1.mlp.fc2': 7.08e9,
}


def main():
    names = [p.name for p in sorted((DATA / 'gt_tensors').glob('*.npy'))]
    args = SimpleNamespace(code_root=REPO, config=INCOMING / 'dsec_c12_alpha0125_ep29_resume5_20260830.yml',
                           checkpoint=INCOMING / 'checkpoint_epoch34.pth', data=DATA)
    model, *_ = build_model(args)
    mods = dict(model.named_modules())
    fc = [n for n in mods if n.endswith('.mlp.fc1') or n.endswith('.mlp.fc2')]
    bits = {}

    def make(full):
        cin = mods[full].in_features
        short = full.replace('sttmultires_unet.encoders.swin3d.', '')

        def pre(m, inp, short=short, cin=cin):
            x = inp[0].detach()
            flat = x.reshape(-1, cin) if x.shape[-1] == cin else x.reshape(-1, x.shape[-1])
            bits.setdefault(short, []).append((flat.abs().sum(1) != 0).cpu().numpy())
        return pre

    for n in fc:
        mods[n].register_forward_pre_hook(make(n))
    with torch.no_grad():
        for name in names:
            functional.reset_net(model)
            x, _, _ = input_frame(DATA, name)
            model(x)
            print('CAP', name, flush=True)
            del x

    per = {}
    dense_c = skip_c = 0
    w_dense = w_skip = 0.0
    for short, masks in bits.items():
        d = s = 0
        for valid in masks:
            n = valid.size
            d += n
            s += (n + G - 1) // G + int(valid.sum())
        vs = s / max(d, 1)
        mac = MAC.get(short, 0.0)
        per[short] = {'dense_cycles': d, 'bitmap_cycles': s, 'vs_dense': vs, 'mac_G': mac / 1e9}
        dense_c += d
        skip_c += s
        w_dense += mac
        w_skip += vs * mac

    summary = {
        'G': G,
        'unweighted_cycle_vs_dense': skip_c / max(dense_c, 1),
        'mac_weighted_vs_dense': w_skip / max(w_dense, 1),
        'mlp_cycle_save_frac': 1.0 - w_skip / max(w_dense, 1),
        'net_save_frac': (w_dense - w_skip) / NET,
        'per_module': per,
        'note': '1 cycle/token dense; bitmap G=32 inspect + 1 cycle/nz. MAC-weighted uses 7.08G for profiled FCs only. Layer2 fc1 bmm not in MAC table.',
    }
    dest = HERE / 'results'
    (dest / 'bitmap_all.json').write_text(json.dumps(summary, indent=2))
    print('SUMMARY', json.dumps({k: summary[k] for k in summary if k != 'per_module'}, indent=2), flush=True)
    for k, v in sorted(per.items(), key=lambda kv: kv[1]['vs_dense']):
        print(f"{k:48s} vs={v['vs_dense']:.3f} dense={v['dense_cycles']}", flush=True)


if __name__ == '__main__':
    main()
