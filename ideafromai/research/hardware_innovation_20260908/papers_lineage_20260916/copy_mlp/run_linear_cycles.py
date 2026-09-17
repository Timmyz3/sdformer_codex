"""Same-GPU work: dense Linear vs skip-zero-rows vs unique-nonzero-rows.

Replaces every Swin MLP fc1/fc2 for the timed path. AEE must stay on preds.2.
Times only the Linear bodies (CUDA events), not the rest of the net.
"""
from __future__ import annotations
import json, os, sys, time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
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


def aee(pred, label, mask):
    err = (pred - label).square().sum(1).sqrt()
    return float(err[mask].sum()) / int(mask.sum())


def gemm(flat, weight):
    return F.linear(flat, weight, None)


def aac_gemm(flat, weight):
    nz = flat.abs().sum(dim=1) != 0
    out = flat.new_zeros(flat.shape[0], weight.shape[0])
    n = int(nz.sum())
    if n:
        out[nz] = gemm(flat[nz], weight)
    return out, n, int(flat.shape[0])


def unique_gemm(flat, weight):
    nz = flat.abs().sum(dim=1) != 0
    out = flat.new_zeros(flat.shape[0], weight.shape[0])
    n = int(nz.sum())
    if n == 0:
        return out, 0, 0, int(flat.shape[0])
    rows = flat[nz]
    uniq, inv = torch.unique(rows, dim=0, return_inverse=True)
    u = gemm(uniq, weight)
    out[nz] = u[inv]
    return out, int(uniq.shape[0]), n, int(flat.shape[0])


def wrap(mod, mode, meters):
    w = mod.weight
    cin = mod.in_features

    def fwd(x, mode=mode, w=w, cin=cin, meters=meters):
        shape = x.shape
        flat = x.reshape(-1, cin) if x.shape[-1] == cin else x.reshape(-1, x.shape[-1])
        if flat.shape[-1] != w.shape[1]:
            flat = x.reshape(-1, w.shape[1])
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if mode == 'dense':
            y = gemm(flat, w)
            n_u = n_nz = flat.shape[0]
        elif mode == 'aac':
            y, n_nz, n_all = aac_gemm(flat, w)
            n_u = n_nz
            meters['rows'] += n_all
            meters['nz'] += n_nz
        else:
            y, n_u, n_nz, n_all = unique_gemm(flat, w)
            meters['rows'] += n_all
            meters['nz'] += n_nz
            meters['uniq'] += n_u
        end.record()
        torch.cuda.synchronize()
        meters['ms'] += start.elapsed_time(end)
        meters['calls'] += 1
        return y.reshape(*shape[:-1], w.shape[0]) if x.shape[-1] == cin else y.reshape(shape[0], w.shape[0], *shape[2:])

    return fwd


def run_mode(model, mods, fc, mode, names):
    meters = defaultdict(float)
    orig = {}
    for n in fc:
        orig[n] = mods[n].forward
        mods[n].forward = wrap(mods[n], mode, meters)
    bucket = {}
    h = mods[PRED2].register_forward_hook(lambda m, i, o: bucket.__setitem__('p2', o.detach()))
    aees = []
    with torch.no_grad():
        for name in names:
            functional.reset_net(model)
            x, label, mask = input_frame(DATA, name)
            model(x)
            pred = F.interpolate(bucket.pop('p2').sum(0), size=(480, 640), mode='bilinear', align_corners=False)
            aees.append(aee(pred, label, mask))
            print(mode.upper(), name, round(aees[-1], 4), 'ms', round(meters['ms'], 2), flush=True)
            del x, label, mask, pred
    h.remove()
    for n in fc:
        mods[n].forward = orig[n]
    return {
        'AEE': float(sum(aees) / len(aees)),
        'linear_ms': float(meters['ms']),
        'calls': int(meters['calls']),
        'rows': int(meters['rows']),
        'nz': int(meters['nz']),
        'uniq': int(meters['uniq']),
    }


def main():
    names = [p.name for p in sorted((DATA / 'gt_tensors').glob('*.npy'))]
    args = SimpleNamespace(code_root=REPO, config=INCOMING / 'dsec_c12_alpha0125_ep29_resume5_20260830.yml',
                           checkpoint=INCOMING / 'checkpoint_epoch34.pth', data=DATA)
    t0 = time.time()
    model, *_ = build_model(args)
    mods = dict(model.named_modules())
    fc = [n for n in mods if n.endswith('.mlp.fc1') or n.endswith('.mlp.fc2')]
    print('FC', len(fc), flush=True)
    # warmup dense
    with torch.no_grad():
        functional.reset_net(model)
        x, _, _ = input_frame(DATA, names[0])
        model(x)
        del x
        torch.cuda.synchronize()
    out = {}
    for mode in ('dense', 'aac', 'unique'):
        out[mode] = run_mode(model, mods, fc, mode, names)
    dens = out['dense']['linear_ms']
    for m in out:
        out[m]['vs_dense'] = out[m]['linear_ms'] / dens if dens else None
    summary = {
        'seconds': time.time() - t0,
        'gpu': torch.cuda.get_device_name(),
        'modes': out,
        'note': 'linear_ms is CUDA-event time of MLP fc1/fc2 only, 10 frames, same GPU. unique=torch.unique then gemm. Not ASIC cycles.',
        'AEE_ref_unpatched': 0.7157699492039079,
    }
    dest = HERE / 'results'
    dest.mkdir(exist_ok=True)
    (dest / 'linear_cycles.json').write_text(json.dumps(summary, indent=2))
    print('SUMMARY', json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    main()
