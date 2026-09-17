"""Copy Prosperity unique-rows + C-STEP early-T token skip onto Swin MLP (26% dense MAC)."""
from __future__ import annotations
import json, os, sys, time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F

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

PRED2 = 'sttmultires_unet.preds.2'
T = 10


def aee(pred, label, mask):
    err = (pred - label).square().sum(1).sqrt()
    return float(err[mask].sum()) / int(mask.sum())


def as_tcn(x, cin=None):
    """Return (T, N, C) tokens or None if T does not divide."""
    if cin is not None and x.shape[-1] == cin:
        if x.ndim == 2:
            rows, c = x.shape
            if rows % T:
                return None
            return x.reshape(T, rows // T, c)
        if x.ndim == 3 and x.shape[0] == T:
            return x
        if x.ndim == 5:
            t, b, h, w, c = x.shape
            return x.reshape(t, b * h * w, c)
        if x.ndim == 4 and x.shape[0] == T:
            return x.reshape(T, -1, cin)
        return x.reshape(T, -1, cin) if x.shape[0] == T else None
    if x.ndim == 2:
        rows, c = x.shape
        if rows % T:
            return None
        return x.reshape(T, rows // T, c)
    if x.ndim == 3 and x.shape[0] == T:
        return x
    if x.ndim == 5:
        t, b, c, h, w = x.shape
        return x.permute(0, 1, 3, 4, 2).reshape(t, b * h * w, c)
    return None


def stats_from_x(name, x, cin=None):
    spk = (x.abs() > 0)
    tc = as_tcn(spk, cin)
    rec = {'module': name, 'shape': list(x.shape), 'nnz': float(spk.float().mean())}
    if tc is None:
        rec['layout'] = 'unparsed'
        return rec
    t, n, c = tc.shape
    rec['layout'] = f'TNC:{t}x{n}x{c}'
    tok_nz = tc.any(dim=2)  # T,N
    rec['token_any_frac'] = float(tok_nz.float().mean())
    rec['spatial_any_frac'] = float(tok_nz.any(dim=0).float().mean())
    early = tok_nz[:2].any(dim=0)
    late = tok_nz[2:].any(dim=0)
    silent_early = ~early
    rec['early_silent_token_frac'] = float(silent_early.float().mean())
    rec['late_active_given_early_silent'] = float(late[silent_early].float().mean()) if int(silent_early.sum()) else 0.0
    rec['later_t_skip_frac'] = float(silent_early.float().mean()) * 0.8
    # Prosperity: unique binary patterns among tokens at one T (use T=0)
    row = tc[0].cpu().numpy()  # N,C
    packed = np.packbits(row, axis=1)
    # only among nonzero tokens
    nz = row.any(axis=1)
    uniq = len({packed[i].tobytes() for i in np.where(nz)[0]}) if nz.any() else 0
    rec['t0_nz_tokens'] = int(nz.sum())
    rec['t0_unique_patterns'] = uniq
    rec['t0_unique_over_nz'] = uniq / max(int(nz.sum()), 1)
    return rec


def main():
    out = HERE / 'results'
    out.mkdir(exist_ok=True)
    names = [p.name for p in sorted((DATA / 'gt_tensors').glob('*.npy'))]
    args = SimpleNamespace(code_root=REPO, config=INCOMING / 'dsec_c12_alpha0125_ep29_resume5_20260830.yml',
                           checkpoint=INCOMING / 'checkpoint_epoch34.pth', data=DATA)
    t0 = time.time()
    model, *_ = build_model(args)
    mods = dict(model.named_modules())
    fc_names = [n for n, m in mods.items() if n.endswith('.mlp.fc1') or n.endswith('.mlp.fc2')]
    print('FC_HOOKS', len(fc_names), flush=True)

    bucket = {}
    mods[PRED2].register_forward_hook(lambda m, i, o: bucket.__setitem__('p2', o.detach()))

    captured = defaultdict(list)

    def make_pre(name, cin):
        def pre(m, inp):
            captured[name].append(stats_from_x(name, inp[0].detach(), cin))
        return pre

    for n in fc_names:
        mods[n].register_forward_pre_hook(make_pre(n, mods[n].in_features))

    # pass 1: measure + baseline AEE
    aees = []
    with torch.no_grad():
        for name in names:
            functional.reset_net(model)
            x, label, mask = input_frame(DATA, name)
            model(x)
            pred = F.interpolate(bucket.pop('p2').sum(0), size=(480, 640), mode='bilinear', align_corners=False)
            val = aee(pred, label, mask)
            aees.append(val)
            print('BASE', name, round(val, 4), flush=True)
            del x, label, mask, pred

    # aggregate per module (mean over frames)
    per = {}
    for n, recs in captured.items():
        keys = [k for k in recs[0] if isinstance(recs[0][k], (int, float))]
        agg = {k: float(np.mean([r[k] for r in recs if k in r])) for k in keys}
        agg['layout'] = recs[0].get('layout')
        agg['shape'] = recs[0].get('shape')
        per[n.replace('sttmultires_unet.encoders.swin3d.', '')] = agg

    # pass 2: C-STEP token early skip on ALL fc1 (spike inputs of MS-MLP)
    def gate_fc1(m, inp):
        x = inp[0]
        cin = m.in_features
        tc = as_tcn(x, cin)
        if tc is None:
            return
        early = tc[:2].abs().any(dim=2).any(dim=0)
        silent = ~early
        if not bool(silent.any()):
            return
        tc = tc.clone()
        tc[2:] = tc[2:] * early.to(tc.dtype)[None, :, None]
        return (tc.reshape_as(x),)

    for n in fc_names:
        if n.endswith('.fc1'):
            mods[n].register_forward_pre_hook(gate_fc1)

    skip_aees = []
    with torch.no_grad():
        for name in names:
            functional.reset_net(model)
            x, label, mask = input_frame(DATA, name)
            model(x)
            pred = F.interpolate(bucket.pop('p2').sum(0), size=(480, 640), mode='bilinear', align_corners=False)
            val = aee(pred, label, mask)
            skip_aees.append(val)
            print('SKIP', name, round(val, 4), flush=True)
            del x, label, mask, pred

    # weighted later-T skip using profile shares 1.19% each fc1 roughly
    fc1_skip = [v['later_t_skip_frac'] for k, v in per.items() if k.endswith('fc1') and 'later_t_skip_frac' in v]
    summary = {
        'seconds': time.time() - t0,
        'gpu': torch.cuda.get_device_name(),
        'AEE_full': float(np.mean(aees)),
        'AEE_cstep_early_token_skip_all_fc1': float(np.mean(skip_aees)),
        'mean_fc1_later_t_skip_frac': float(np.mean(fc1_skip)) if fc1_skip else None,
        'per_module': per,
        'NB0_ref': 1.454602861,
        'note': 'unique_over_nz is Prosperity copy metric (1=no sharing). early_silent is C-STEP on tokens.',
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('SUMMARY_AEE', summary['AEE_full'], summary['AEE_cstep_early_token_skip_all_fc1'], flush=True)
    # compact table
    for k, v in sorted(per.items()):
        if 'fc1' in k or 'fc2' in k:
            print(f"{k:48s} nnz={v.get('nnz',0):.4f} uniq={v.get('t0_unique_over_nz',0):.3f} early_sil={v.get('early_silent_token_frac',0):.3f} late|sil={v.get('late_active_given_early_silent',0):.3f}", flush=True)


if __name__ == '__main__':
    main()
