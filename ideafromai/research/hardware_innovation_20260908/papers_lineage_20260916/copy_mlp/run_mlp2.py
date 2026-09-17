"""MLP GeMM-batch lossless skips + Phi/LUT codebook copy on hottest FC1.

Previous unique/nz used only T=0. The Linear actually sees all T*N rows in one GeMM.
"""
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
TOTAL = 596.5464288e9
MLP = 156.5411328e9
HOT_FC1 = 'sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.fc1'


def aee(pred, label, mask):
    err = (pred - label).square().sum(1).sqrt()
    return float(err[mask].sum()) / int(mask.sum())


def rows_2d(x, cin):
    if x.shape[-1] != cin:
        x = x.transpose(-1, -2) if x.shape[-2] == cin else x
    return x.reshape(-1, cin)


def pack_rows(spk_nc):
    return np.packbits(spk_nc, axis=1)


def main():
    out = HERE / 'results'
    out.mkdir(exist_ok=True)
    names = [p.name for p in sorted((DATA / 'gt_tensors').glob('*.npy'))]
    args = SimpleNamespace(code_root=REPO, config=INCOMING / 'dsec_c12_alpha0125_ep29_resume5_20260830.yml',
                           checkpoint=INCOMING / 'checkpoint_epoch34.pth', data=DATA)
    t0 = time.time()
    model, *_ = build_model(args)
    mods = dict(model.named_modules())
    fc_names = [n for n in mods if n.endswith('.mlp.fc1') or n.endswith('.mlp.fc2')]
    bucket = {}
    mods[PRED2].register_forward_hook(lambda m, i, o: bucket.__setitem__('p2', o.detach()))
    acc = defaultdict(list)

    def make_pre(full):
        cin = mods[full].in_features
        short = full.replace('sttmultires_unet.encoders.swin3d.', '')

        def pre(m, inp):
            x = inp[0].detach()
            r = rows_2d(x, cin)
            spk = (r.abs() > 0)
            nz = spk.any(dim=1)
            n_rows = int(spk.shape[0])
            n_nz = int(nz.sum())
            packed = pack_rows(spk.cpu().numpy())
            uniq_all = len({packed[i].tobytes() for i in range(n_rows)})
            uniq_nz = len({packed[i].tobytes() for i in np.where(nz.cpu().numpy())[0]}) if n_nz else 0
            zero_frac = 1.0 - n_nz / max(n_rows, 1)
            # lossless GeMM work: unique nonzero patterns (zero row is 1 pattern)
            work = uniq_nz / max(n_rows, 1)
            acc[short].append({
                'n_rows': n_rows, 'n_nz': n_nz, 'zero_frac': zero_frac,
                'uniq_all': uniq_all, 'uniq_nz': uniq_nz,
                'uniq_over_nz': uniq_nz / max(n_nz, 1),
                'lossless_work_frac': work,
                'aac_save': zero_frac,
                'unique_save_on_nz': 1.0 - uniq_nz / max(n_nz, 1),
            })
        return pre

    for n in fc_names:
        mods[n].register_forward_pre_hook(make_pre(n))

    aees = []
    with torch.no_grad():
        for name in names:
            functional.reset_net(model)
            x, label, mask = input_frame(DATA, name)
            model(x)
            pred = F.interpolate(bucket.pop('p2').sum(0), size=(480, 640), mode='bilinear', align_corners=False)
            aees.append(aee(pred, label, mask))
            print('BASE', name, round(aees[-1], 4), flush=True)
            del x, label, mask, pred

    per = {}
    saved_mlp = saved_all = 0.0
    for short, recs in acc.items():
        keys = recs[0].keys()
        agg = {k: float(np.mean([r[k] for r in recs])) for k in keys}
        mac = MAC.get(short, 0.0)
        # remaining work after unique-nz + zero skip
        work = agg['lossless_work_frac']
        save = (1.0 - work) * mac
        agg['mac'] = mac
        agg['lossless_save_G'] = save / 1e9
        per[short] = agg
        saved_mlp += save
        saved_all += save

    # ---- Phi/LUT copy: codebook of K most frequent patterns on hottest FC1 ----
    # Fit codebook on frame 0, apply to all frames (nearest Hamming).
    Ks = (64, 256)
    code_aees = {}
    cin = mods[HOT_FC1].in_features
    codebooks = {}

    def fit_and_apply(K):
        fitted = {'cb': None}

        def pre(m, inp):
            x = inp[0]
            r = rows_2d(x, cin)
            spk = (r.abs() > 0)
            packed = pack_rows(spk.detach().cpu().numpy())
            if fitted['cb'] is None:
                # frequency
                freq = {}
                for i, row in enumerate(packed):
                    k = row.tobytes()
                    freq[k] = freq.get(k, 0) + 1
                top = sorted(freq.items(), key=lambda z: -z[1])[:K]
                cb = np.stack([np.frombuffer(k, dtype=np.uint8) for k, _ in top])
                fitted['cb'] = cb
                fitted['packed'] = {k: j for j, (k, _) in enumerate(top)}
            cb = fitted['cb']
            # nearest hamming in packed space
            # packed: N x nbytes, cb: K x nbytes
            # XOR popcount
            xr = packed[:, None, :] ^ cb[None, :, :]
            pop = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
            dist = pop[xr].sum(axis=2)
            nn = dist.argmin(axis=1)
            new_packed = cb[nn]
            bits = np.unpackbits(new_packed, axis=1)[:, :cin]
            new = torch.from_numpy(bits.astype(np.float32)).to(x.device, x.dtype).reshape_as(x)
            # keep original amplitudes on kept ones? spikes are 0/theta — use 1/0 then * orig scale
            scale = x.abs().amax()
            if float(scale) == 0:
                return (new,)
            return (new * scale,)

        h = mods[HOT_FC1].register_forward_pre_hook(pre)
        vals = []
        with torch.no_grad():
            for name in names:
                functional.reset_net(model)
                x, label, mask = input_frame(DATA, name)
                model(x)
                pred = F.interpolate(bucket.pop('p2').sum(0), size=(480, 640), mode='bilinear', align_corners=False)
                vals.append(aee(pred, label, mask))
                print(f'PHI{K}', name, round(vals[-1], 4), flush=True)
                del x, label, mask, pred
        h.remove()
        return float(np.mean(vals)), fitted['cb'].shape[0] if fitted['cb'] is not None else 0

    for K in Ks:
        mean_aee, k_used = fit_and_apply(K)
        code_aees[str(K)] = {'AEE': mean_aee, 'K': k_used}

    summary = {
        'seconds': time.time() - t0,
        'gpu': torch.cuda.get_device_name(),
        'AEE_full': float(np.mean(aees)),
        'lossless_mlp_save_G': saved_mlp / 1e9,
        'lossless_mlp_save_frac_of_mlp': saved_mlp / MLP,
        'lossless_mlp_save_frac_of_net': saved_mlp / TOTAL,
        'phi_lut_on_L0B0_fc1': code_aees,
        'per_module': per,
        'NB0_ref': 1.454602861,
        'note': 'lossless_work_frac=unique_nz_patterns/n_rows (zero rows share one empty pattern). Phi replaces hottest FC1 rows by K frequent patterns (Hamming).',
    }
    (out / 'mlp2_summary.json').write_text(json.dumps(summary, indent=2))
    print('LOSSLESS_MLP_SAVE', summary['lossless_mlp_save_G'], 'G',
          'mlp%', round(100 * summary['lossless_mlp_save_frac_of_mlp'], 2),
          'net%', round(100 * summary['lossless_mlp_save_frac_of_net'], 2), flush=True)
    print('PHI', code_aees, flush=True)
    for k, v in sorted(per.items(), key=lambda kv: -kv[1]['lossless_save_G']):
        print(f"{k:48s} zero={v['zero_frac']:.3f} uniq_nz={v['uniq_over_nz']:.3f} work={v['lossless_work_frac']:.3f} saveG={v['lossless_save_G']:.2f}", flush=True)


if __name__ == '__main__':
    main()
