"""Stack Jung empty-Q SSA skip with Zhang-style P1 readout (skip decoder2 in the metric)."""
from __future__ import annotations
import json, os, sys, time
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

PRED1 = 'sttmultires_unet.preds.1'
PRED2 = 'sttmultires_unet.preds.2'


def aee(pred, label, mask):
    err = (pred - label).square().sum(1).sqrt()
    return float(err[mask].sum()) / int(mask.sum())


def up(x, hw):
    return F.interpolate(x, size=hw, mode='bilinear', align_corners=False)


def main():
    names = [p.name for p in sorted((DATA / 'gt_tensors').glob('*.npy'))]
    args = SimpleNamespace(code_root=REPO, config=INCOMING / 'dsec_c12_alpha0125_ep29_resume5_20260830.yml',
                           checkpoint=INCOMING / 'checkpoint_epoch34.pth', data=DATA)
    t0 = time.time()
    model, *_ = build_model(args)
    mods = dict(model.named_modules())
    bucket = {}
    mods[PRED1].register_forward_hook(lambda m, i, o: bucket.__setitem__('p1', o.detach()))
    mods[PRED2].register_forward_hook(lambda m, i, o: bucket.__setitem__('p2', o.detach()))

    def jung_hook(m, inp, output, st=None):
        row = output.reshape(-1, output.shape[-1])
        frac = float((row.abs().sum(1) == 0).float().mean())
        st['n'] += 1
        if frac >= 0.90:
            st['z'] += 1
            return torch.zeros_like(output)
        return output

    st = {'n': 0, 'z': 0}
    for n in mods:
        if n.endswith('.attn.sn_q'):
            mods[n].register_forward_hook(lambda m, i, o, st=st: jung_hook(m, i, o, st))

    rows = []
    with torch.no_grad():
        for name in names:
            functional.reset_net(model)
            x, label, mask = input_frame(DATA, name)
            model(x)
            p1 = up(bucket.pop('p1').sum(0), (480, 640))
            p2 = up(bucket.pop('p2').sum(0), (480, 640))
            rec = {
                'file': name,
                'AEE_P1_with_Jung': aee(p1, label, mask),
                'AEE_P2_with_Jung': aee(p2, label, mask),
            }
            rows.append(rec)
            print(name[-12:], rec, flush=True)
            del x, label, mask, p1, p2

    summary = {
        'seconds': time.time() - t0,
        'jung_blocks_zeroed': st['z'] / max(st['n'], 1),
        'AEE_P1_Jung': float(np.mean([r['AEE_P1_with_Jung'] for r in rows])),
        'AEE_P2_Jung': float(np.mean([r['AEE_P2_with_Jung'] for r in rows])),
        'AEE_P1_no_Jung_ref': 0.7094136208632279,
        'AEE_P2_no_Jung_ref': 0.7157699492039079,
        'NB0_ref': 1.454602861,
        'frames': rows,
        'note': 'Jung zeros sn_q when >=90% rows empty. P1 metric = skip decoder2. Decoder still ran in this forward (quality only).',
    }
    dest = HERE / 'results'
    dest.mkdir(exist_ok=True)
    (dest / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('SUMMARY', json.dumps({k: summary[k] for k in summary if k != 'frames'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
