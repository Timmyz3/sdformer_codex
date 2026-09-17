"""C-STEP locally-common spike: AND of spike bits across K=2 horizontal neighbors per CHANNEL.

Not full-vector equality (already measured ~1.1% on r0.conv2 and failed).
AT-LIF {0, theta}: absorb theta into W; spike bit = (abs(x) > 0).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from spikingjelly.activation_based import functional

HERE = Path(__file__).resolve().parent
ALGO = HERE.parents[1] / 'algorithm'
REPO = Path('/home/zhumd/work/sdformer_codex/SDformer')
DATA = REPO / 'data/Datasets/DSEC/saved_flow_data'
INCOMING = REPO / 'hw_autoresearch_nts07/system_handoff/incoming/m2041_ep34_quant_binding_inputs'
sys.path[:0] = [str(ALGO), str(ALGO / 'nrv_cost_probe')]
os.environ['SDFORMER_USE_MLFLOW'] = '0'
os.environ['SDFORMER_MLFLOW_MODEL_LOGGING'] = '0'
from run_bn_probe import build_model, input_frame

CONV2 = 'sttmultires_unet.encoders.swin3d.patch_embed.residual_encoding.resblocks.0.conv2.0'
FC1 = 'sttmultires_unet.encoders.swin3d.layers.1.swin_blocks.0.mlp.fc1'
PAPER_K2 = 0.827  # C-STEP DATE'26 Fig.2(a): up to 82.7% of spike positions common at K=2


def to_tchw(x, cin):
    x = x.detach()
    if x.ndim == 5:
        x = x[:, 0] if x.shape[1] == 1 or x.shape[0] != 1 else x[0]
    if x.ndim != 4:
        raise ValueError(f'expected 4D TCHW/THWC, got {tuple(x.shape)}')
    if x.shape[1] == cin:
        return x
    if x.shape[-1] == cin:
        return x.permute(0, 3, 1, 2).contiguous()
    raise ValueError(f'cannot find C={cin} in {tuple(x.shape)}')


def horiz_and(spk):
    """spk: bool [T,C,H,W]. K=2 horizontal neighbors, per-channel bits."""
    left = spk[:, :, :, :-1]
    right = spk[:, :, :, 1:]
    common = int((left & right).sum().item())
    union = int((left | right).sum().item())
    left_n = int(left.sum().item())
    right_n = int(right.sum().item())
    jaccard = common / union if union else 0.0
    precision = common / left_n if left_n else 0.0  # P(right | left spike in channel c)
    both = left_n + right_n
    return {
        'common': common,
        'union': union,
        'left_spikes': left_n,
        'right_spikes': right_n,
        'jaccard': jaccard,
        'precision_reuse': precision,
        'dice': (2.0 * common / both) if both else 0.0,
        'reuse_save_frac': (common / both) if both else 0.0,  # common computed once; not PPA
    }


def identity_check(act):
    nz = act[act != 0]
    if nz.numel() == 0:
        return {'nnz': 0, 'nz_abs_min': None, 'nz_abs_max': None}
    a = nz.abs()
    return {
        'nnz': int(nz.numel()),
        'nz_abs_min': float(a.min().item()),
        'nz_abs_max': float(a.max().item()),
        'n_unique_abs_cap': int(torch.unique(a.flatten()[:65536]).numel()),
    }


def layer_rec(act, cin):
    tchw = to_tchw(act, cin)
    spk = tchw.abs() > 0
    rec = horiz_and(spk)
    rec['shape_tchw'] = [int(v) for v in tchw.shape]
    rec['nnz'] = float(spk.float().mean().item())
    rec['identity'] = identity_check(tchw)
    return rec


def verdict(jaccard):
    if jaccard > 0.5:
        return 'KEEP as FireFly/C-STEP A for this layer'
    if jaccard < 0.3:
        return 'WEAK: jaccard << 0.3, local-common reuse is weak'
    return 'MID: 0.3-0.5, below C-STEP K=2 82.7%'


def pool(rows, key):
    c = sum(r[key]['common'] for r in rows)
    u = sum(r[key]['union'] for r in rows)
    l = sum(r[key]['left_spikes'] for r in rows)
    rgt = sum(r[key]['right_spikes'] for r in rows)
    both = l + rgt
    return {
        'jaccard_micro': c / u if u else 0.0,
        'jaccard_macro': float(np.mean([r[key]['jaccard'] for r in rows])),
        'precision_reuse_micro': c / l if l else 0.0,
        'precision_reuse_macro': float(np.mean([r[key]['precision_reuse'] for r in rows])),
        'dice_micro': (2.0 * c / both) if both else 0.0,
        'reuse_save_frac_micro': (c / both) if both else 0.0,
        'mean_nnz': float(np.mean([r[key]['nnz'] for r in rows])),
        'common': c,
        'union': u,
        'left_spikes': l,
        'right_spikes': rgt,
        'verdict': verdict(c / u if u else 0.0),
        'cstep_paper_k2': PAPER_K2,
    }


def main():
    out = HERE / 'results'
    out.mkdir(exist_ok=True)
    args = SimpleNamespace(
        code_root=REPO,
        config=INCOMING / 'dsec_c12_alpha0125_ep29_resume5_20260830.yml',
        checkpoint=INCOMING / 'checkpoint_epoch34.pth',
        data=DATA,
    )
    model, *_ = build_model(args)
    mods = dict(model.named_modules())
    cin_conv = int(mods[CONV2].in_channels)
    cin_fc1 = int(mods[FC1].in_features)
    bucket = {}
    mods[CONV2].register_forward_pre_hook(lambda m, a: bucket.__setitem__('conv2', a[0].detach()))
    mods[FC1].register_forward_pre_hook(lambda m, a: bucket.__setitem__('fc1', a[0].detach()))
    names = [p.name for p in sorted((DATA / 'gt_tensors').glob('*.npy'))]
    if len(names) != 10:
        raise SystemExit(f'expected 10 GT frames, got {len(names)}')
    rows = []
    with torch.no_grad():
        for name in names:
            functional.reset_net(model)
            x, _, _ = input_frame(DATA, name)
            model(x)
            rec = {
                'file': name,
                'r0_conv2': layer_rec(bucket.pop('conv2'), cin_conv),
                'l1_fc1': layer_rec(bucket.pop('fc1'), cin_fc1),
            }
            rows.append(rec)
            print(
                'AND', name,
                'conv2 j', round(rec['r0_conv2']['jaccard'], 4),
                'p', round(rec['r0_conv2']['precision_reuse'], 4),
                'fc1 j', round(rec['l1_fc1']['jaccard'], 4),
                'p', round(rec['l1_fc1']['precision_reuse'], 4),
                flush=True,
            )
            del x
    summary = {
        'metric': 'per-channel AND of K=2 horizontal neighbor spike bits; jaccard=|A&B|/|A|B|',
        'not': 'full 96/192-bit vector equality (copy_hotspot/results/common.json ~1.1%)',
        'identity': 'AT-LIF {0,theta} absorb; bit = abs(x)>0',
        'K': 2,
        'layers': {
            'r0_conv2': {'module': CONV2, 'in_channels': cin_conv},
            'l1_fc1': {'module': FC1, 'in_features': cin_fc1},
        },
        'r0_conv2': pool(rows, 'r0_conv2'),
        'l1_fc1': pool(rows, 'l1_fc1'),
        'frames': rows,
    }
    (out / 'channel_and.json').write_text(json.dumps(summary, indent=2))
    slim = {k: summary[k] for k in summary if k != 'frames'}
    print('SUMMARY', json.dumps(slim, indent=2), flush=True)


if __name__ == '__main__':
    main()
