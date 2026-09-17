"""C-STEP adapted to pixels on r0.conv2: skip later T where T0-T1 are silent."""
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
CONV2 = 'sttmultires_unet.encoders.swin3d.patch_embed.residual_encoding.resblocks.0.conv2.0'
PRED2 = 'sttmultires_unet.preds.2'

def aee(pred, label, mask):
    err = (pred - label).square().sum(1).sqrt()
    return float(err[mask].sum()) / int(mask.sum())

def main():
    args = SimpleNamespace(code_root=REPO, config=INCOMING/'dsec_c12_alpha0125_ep29_resume5_20260830.yml',
                           checkpoint=INCOMING/'checkpoint_epoch34.pth', data=DATA)
    model, *_ = build_model(args)
    mods = dict(model.named_modules())
    bucket, pix = {}, {'n': 0, 'silent': 0}
    mods[PRED2].register_forward_hook(lambda m,i,o: bucket.__setitem__('p2', o.detach()))
    def gate(m, args):
        x = args[0]
        if x.ndim == 4:
            x = x.unsqueeze(1)
        silent = x[:2].abs().sum(dim=(0, 1, 2)) == 0
        pix['n'] += int(silent.numel())
        pix['silent'] += int(silent.sum())
        x = x.clone()
        x[2:] *= (~silent).to(x.dtype)
        return (x,)
    mods[CONV2].register_forward_pre_hook(gate)
    names = [p.name for p in sorted((DATA/'gt_tensors').glob('*.npy'))]
    aees = []
    with torch.no_grad():
        for name in names:
            functional.reset_net(model)
            x, label, mask = input_frame(DATA, name)
            model(x)
            pred = F.interpolate(bucket.pop('p2').sum(0), size=(480, 640), mode='bilinear', align_corners=False)
            val = aee(pred, label, mask)
            aees.append(val)
            print('PIXEL', name, round(val, 4), 'silent', pix['silent']/max(pix['n'],1), flush=True)
            del x, label, mask, pred
    frac = pix['silent'] / max(pix['n'], 1)
    out = {'AEE': float(np.mean(aees)), 'silent_pixel_frac': frac,
           'conv2_mac_saved_proxy': frac * 0.8 * 0.1068, 'full_AEE_ref': 0.7157699492039079}
    (HERE/'results'/'pixel.json').write_text(json.dumps(out, indent=2))
    print('SUMMARY', json.dumps(out, indent=2), flush=True)
if __name__ == '__main__':
    main()
