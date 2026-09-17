"""FireFly-style bitmap row-skip cycle model for L1.b0.fc1 (not GPU gather).

Dense: 1 cycle per token (192-wide MAC in 1 cycle).
test_plus_mac: 1 inspect + 1 MAC if nz (usually worse than dense).
bitmap-G: inspect G validity bits per cycle, then 1 MAC cycle per nz token.
"""
from __future__ import annotations
import json, os, sys
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

TARGET = 'sttmultires_unet.encoders.swin3d.layers.1.swin_blocks.0.mlp.fc1'
NET_MAC = 596.5464288e9
LAYER_MAC = 7.08e9


def bitmap_cycles(valid, g):
    n = valid.size
    inspect = (n + g - 1) // g
    mac = int(valid.sum())
    return inspect + mac


def main():
    names = [p.name for p in sorted((DATA / 'gt_tensors').glob('*.npy'))]
    args = SimpleNamespace(
        code_root=REPO,
        config=INCOMING / 'dsec_c12_alpha0125_ep29_resume5_20260830.yml',
        checkpoint=INCOMING / 'checkpoint_epoch34.pth',
        data=DATA,
    )
    model, *_ = build_model(args)
    mods = dict(model.named_modules())
    cin = mods[TARGET].in_features
    captured = []

    def pre(m, inp):
        x = inp[0].detach()
        flat = x.reshape(-1, cin) if x.shape[-1] == cin else x.reshape(-1, x.shape[-1])
        captured.append((flat.abs().sum(dim=1) != 0).cpu().numpy().astype(np.uint8))

    mods[TARGET].register_forward_pre_hook(pre)
    frames = []
    with torch.no_grad():
        for name in names:
            captured.clear()
            functional.reset_net(model)
            x, _, _ = input_frame(DATA, name)
            model(x)
            valid = captured[-1]
            dense = int(valid.size)
            nz = int(valid.sum())
            rec = {
                'file': name, 'rows': dense, 'nz': nz,
                'zero_frac': 1.0 - nz / dense,
                'dense_cycles': dense,
                'test_plus_mac': dense + nz,
                'test_plus_mac_vs_dense': (dense + nz) / dense,
            }
            for g in (8, 16, 32, 64, 128):
                c = bitmap_cycles(valid, g)
                rec[f'g{g}'] = {'cycles': c, 'vs_dense': c / dense}
            frames.append(rec)
            print(name[-12:], 'zero', round(rec['zero_frac'], 3),
                  'g32', round(rec['g32']['vs_dense'], 3), flush=True)
            del x

    def mean_vs(attr):
        if attr == 'test_plus_mac_vs_dense':
            return float(np.mean([f[attr] for f in frames]))
        return float(np.mean([f[attr]['vs_dense'] for f in frames]))

    g32 = mean_vs('g32')
    summary = {
        'target': TARGET,
        'assumption': '1 cycle/token dense 192-wide; bitmap inspects G valids/cycle then 1 MAC/nz. No gather. Not ASIC PPA.',
        'mean_zero_frac': float(np.mean([f['zero_frac'] for f in frames])),
        'mean_vs_dense': {
            'test_plus_mac': mean_vs('test_plus_mac_vs_dense'),
            **{f'bitmap_g{g}': mean_vs(f'g{g}') for g in (8, 16, 32, 64, 128)},
        },
        'layer_mac_share_of_net': LAYER_MAC / NET_MAC,
        'net_save_if_bitmap_g32': (1.0 - g32) * LAYER_MAC / NET_MAC,
        'frames': frames,
    }
    dest = HERE / 'results'
    dest.mkdir(exist_ok=True)
    (dest / 'bitmap_cycles.json').write_text(json.dumps(summary, indent=2))
    print('SUMMARY', json.dumps({k: summary[k] for k in summary if k != 'frames'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
