"""FireFly AAC + Jung empty-token skip on Swin attention (12.6% dense MAC)."""
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
TOTAL = 596.5464288e9
POP = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def aee(pred, label, mask):
    err = (pred - label).square().sum(1).sqrt()
    return float(err[mask].sum()) / int(mask.sum())


def load_mac():
    p = Path('/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/open_fusion_execution/major_operator_fusions_20260913/root_owned/profile.json')
    d = json.loads(p.read_text())
    mac = {}
    for r in d['rows']:
        n = r['module'].replace('sttmultires_unet.encoders.swin3d.', '')
        if any(s in n for s in ('linear_q', 'linear_k', 'linear_v', '.attn.proj')):
            mac[n] = mac.get(n, 0) + r['MACs']
    return mac


def rows_cin(x, cin):
    if x.shape[-1] == cin:
        return x.reshape(-1, cin)
    return x.reshape(-1, x.shape[-1])


def main():
    out = HERE / 'results'
    out.mkdir(exist_ok=True)
    mac_tbl = load_mac()
    names = [p.name for p in sorted((DATA / 'gt_tensors').glob('*.npy'))]
    args = SimpleNamespace(code_root=REPO, config=INCOMING / 'dsec_c12_alpha0125_ep29_resume5_20260830.yml',
                           checkpoint=INCOMING / 'checkpoint_epoch34.pth', data=DATA)
    t0 = time.time()
    model, *_ = build_model(args)
    mods = dict(model.named_modules())
    lin_names = [n for n in mods if n.endswith(('.attn.linear_q', '.attn.linear_k', '.attn.proj'))]
    print('HOOKS', len(lin_names), flush=True)
    bucket = {}
    mods[PRED2].register_forward_hook(lambda m, i, o: bucket.__setitem__('p2', o.detach()))
    acc = defaultdict(list)
    qempty = defaultdict(list)

    def make_pre(full):
        cin = mods[full].in_features
        short = full.replace('sttmultires_unet.encoders.swin3d.', '')

        def pre(m, inp):
            x = inp[0].detach()
            r = rows_cin(x, cin)
            spk = (r.abs() > 0)
            n_rows = spk.shape[0]
            nz = spk.any(dim=1)
            n_nz = int(nz.sum())
            packed = np.packbits(spk.cpu().numpy(), axis=1)
            uniq_nz = len({packed[i].tobytes() for i in np.where(nz.cpu().numpy())[0]}) if n_nz else 0
            acc[short].append({
                'n_rows': int(n_rows), 'n_nz': n_nz,
                'zero_frac': 1.0 - n_nz / max(int(n_rows), 1),
                'elem_nnz': float(spk.float().mean()),
                'uniq_over_nz': uniq_nz / max(n_nz, 1),
                'work_frac': uniq_nz / max(int(n_rows), 1),
            })
            if short.endswith('linear_q'):
                # Jung: empty Q tokens (all-zero rows)
                qempty[short].append(float((~nz).float().mean()))
        return pre

    for n in lin_names:
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
    save = 0.0
    aac = 0.0
    uni = 0.0
    for short, recs in acc.items():
        agg = {k: float(np.mean([r[k] for r in recs])) for k in recs[0]}
        m = mac_tbl.get(short, 0.0)
        agg['mac'] = m
        agg['save_G'] = (1.0 - agg['work_frac']) * m / 1e9
        per[short] = agg
        save += (1.0 - agg['work_frac']) * m
        aac += agg['zero_frac'] * m
        uni += agg['uniq_over_nz'] and (1 - agg['uniq_over_nz']) * (1 - agg['zero_frac']) * m

    # Jung copy: zero the attention output for windows/tokens with empty Q
    # Lossy if Q empty but K/V still used via residual... SSA output zeroed when Q row empty is
    # often exact for QK^T V. We zero linear_q *output* rows that were empty inputs? That's identity.
    # Instead zero sn_q outputs that are already empty — no-op.
    # Lossy experiment: drop entire attn.proj input (set 0) when mean |q| in that tensor is low.
    # More faithful Jung: after sn_q, if a token is 0, skip its query (already 0).
    # Do a lossy "skip whole SSA block if >p empty Q tokens" by zeroing attn output.
    def make_jung(p_empty, stats):
        def hook(m, inp, output, p_empty=p_empty, stats=stats):
            q = output
            row = q.reshape(-1, q.shape[-1])
            empty = (row.abs().sum(dim=1) == 0)
            frac = float(empty.float().mean())
            stats['fired'] += 1
            if frac >= p_empty:
                stats['skipped'] += 1
                return torch.zeros_like(output)
            return output
        return hook

    jung = {}
    snq = [n for n in mods if n.endswith('.attn.sn_q')]
    for p_empty in (0.90, 0.95, 0.99):
        hooks = []
        st = {'fired': 0, 'skipped': 0}
        for n in snq:
            hooks.append(mods[n].register_forward_hook(make_jung(p_empty, st)))
        vals = []
        with torch.no_grad():
            for name in names:
                functional.reset_net(model)
                x, label, mask = input_frame(DATA, name)
                model(x)
                pred = F.interpolate(bucket.pop('p2').sum(0), size=(480, 640), mode='bilinear', align_corners=False)
                vals.append(aee(pred, label, mask))
                print(f'JUNG{p_empty}', name, round(vals[-1], 4), flush=True)
                del x, label, mask, pred
        for h in hooks:
            h.remove()
        jung[str(p_empty)] = {
            'AEE': float(np.mean(vals)),
            'blocks_zeroed_frac': (st['skipped'] / max(st['fired'], 1)) if st else None,
        }

    summary = {
        'seconds': time.time() - t0,
        'AEE_full': float(np.mean(aees)),
        'attn_lin_G': sum(mac_tbl.values()) / 1e9,
        'lossless_save_G': save / 1e9,
        'lossless_frac_of_net': save / TOTAL,
        'aac_save_G': aac / 1e9,
        'unique_extra_G': uni / 1e9,
        'jung_zero_whole_Q_if_empty_frac': jung,
        'mean_Q_empty_row': {k: float(np.mean(v)) for k, v in qempty.items()},
        'per_module': per,
        'NB0_ref': 1.454602861,
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('ATTN_SAVE', summary['lossless_save_G'], 'net%', round(100 * summary['lossless_frac_of_net'], 2), flush=True)
    print('JUNG', jung, flush=True)
    for k, v in sorted(per.items(), key=lambda kv: -kv[1]['save_G'])[:12]:
        print(f"{k:52s} zero={v['zero_frac']:.3f} elem={v['elem_nnz']:.4f} uniq={v['uniq_over_nz']:.3f} saveG={v['save_G']:.2f}", flush=True)


if __name__ == '__main__':
    main()
