"""C-STEP copy: does early-timestep silence predict later silence on r0.conv2 input spikes?"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ALGO = BASE / 'algorithm'
REPO = Path('/home/zhumd/work/sdformer_codex/SDformer')
DATA = REPO / 'data/Datasets/DSEC/saved_flow_data'
INCOMING = REPO / 'hw_autoresearch_nts07/system_handoff/incoming/m2041_ep34_quant_binding_inputs'
sys.path[:0] = [str(ALGO), str(ALGO / 'nrv_cost_probe')]
os.environ['SDFORMER_USE_MLFLOW'] = '0'
os.environ['SDFORMER_MLFLOW_MODEL_LOGGING'] = '0'
from run_bn_probe import build_model, input_frame
from spikingjelly.activation_based import functional

TARGET = 'sttmultires_unet.encoders.swin3d.patch_embed.residual_encoding.resblocks.0.conv2.0'


def main():
    out = HERE / 'results'
    out.mkdir(exist_ok=True)
    names = [p.name for p in sorted((DATA / 'gt_tensors').glob('*.npy'))]
    args = SimpleNamespace(code_root=REPO, config=INCOMING / 'dsec_c12_alpha0125_ep29_resume5_20260830.yml',
                           checkpoint=INCOMING / 'checkpoint_epoch34.pth', data=DATA)
    model, *_ = build_model(args)
    bucket = {}
    def hook(m, inp):
        bucket['x'] = inp[0].detach()
    h = dict(model.named_modules())[TARGET].register_forward_pre_hook(hook)
    rows = []
    with torch.no_grad():
        for name in names:
            functional.reset_net(model)
            x, _, _ = input_frame(DATA, name)
            model(x)
            act = bucket.pop('x')  # expect [T,B,C,H,W] or [T,C,H,W]
            print('SHAPE', name, tuple(act.shape), flush=True)
            if act.ndim == 5:
                act = act[:, 0]
            # binary-ish
            spk = (act.abs() > 0).float()
            T, C = spk.shape[0], spk.shape[1]
            early = spk[:2].sum(dim=(0, 2, 3))  # C
            late = spk[2:].sum(dim=(0, 2, 3))
            silent_early = early == 0
            late_sfr = late / (late.numel() and 1 or 1)
            # per-channel late activity if silent early
            late_rate = late / float((T - 2) * spk.shape[2] * spk.shape[3])
            all_rate = spk.mean(dim=(0, 2, 3))
            n_silent = int(silent_early.sum())
            rec = {
                'file': name, 'T': T, 'C': int(C),
                'silent_early_channels': n_silent,
                'silent_frac': n_silent / float(C),
                'mean_late_rate_if_silent': float(late_rate[silent_early].mean()) if n_silent else None,
                'mean_late_rate_if_active': float(late_rate[~silent_early].mean()),
                'mean_all_rate': float(all_rate.mean()),
                'nnz': float(spk.mean()),
            }
            rows.append(rec)
            print('FRAME', rec, flush=True)
            del x, act, spk
    h.remove()
    silent = np.array([r['silent_frac'] for r in rows])
    ratio = []
    for r in rows:
        if r['mean_late_rate_if_silent'] is None:
            continue
        a, b = r['mean_late_rate_if_silent'], r['mean_late_rate_if_active']
        ratio.append(b / max(a, 1e-12))
    summary = {
        'target': TARGET,
        'mean_silent_frac': float(silent.mean()),
        'mean_late_active_over_silent': float(np.mean(ratio)) if ratio else None,
        'frames': rows,
        'cstep_paper': 'silent channels later SFR 6.7-20x lower; ~10-15% silent',
        'copy_verdict': None,
    }
    if summary['mean_silent_frac'] < 0.02:
        summary['copy_verdict'] = 'FAIL: almost no early-silent channels to prune'
    elif summary['mean_late_active_over_silent'] and summary['mean_late_active_over_silent'] >= 5:
        summary['copy_verdict'] = 'KEEP: early silence predicts later quiet; try skip those channels in later T'
    else:
        summary['copy_verdict'] = 'WEAK: silent fraction or later contrast too small for C-STEP-scale skip'
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('SUMMARY', json.dumps({k: summary[k] for k in summary if k != 'frames'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
