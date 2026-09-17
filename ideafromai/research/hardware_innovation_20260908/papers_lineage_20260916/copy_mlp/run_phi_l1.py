"""Phi/LUT on the MLP layer that actually has row sharing: L1.b0.fc1 (zero 46%, uniq_nz 0.73)."""
from __future__ import annotations
import json, os, sys
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
TARGET = 'sttmultires_unet.encoders.swin3d.layers.1.swin_blocks.0.mlp.fc1'
PRED2 = 'sttmultires_unet.preds.2'
POP = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

def aee(pred, label, mask):
    err = (pred - label).square().sum(1).sqrt()
    return float(err[mask].sum()) / int(mask.sum())

def main():
    names = [p.name for p in sorted((DATA / 'gt_tensors').glob('*.npy'))]
    args = SimpleNamespace(code_root=REPO, config=INCOMING / 'dsec_c12_alpha0125_ep29_resume5_20260830.yml',
                           checkpoint=INCOMING / 'checkpoint_epoch34.pth', data=DATA)
    model, *_ = build_model(args)
    mods = dict(model.named_modules())
    cin = mods[TARGET].in_features
    bucket = {}
    mods[PRED2].register_forward_hook(lambda m, i, o: bucket.__setitem__('p2', o.detach()))
    out = {}
    for K in (64, 256):
        fitted = {}
        def pre(m, inp, K=K, fitted=fitted):
            x = inp[0]
            r = x.reshape(-1, cin) if x.shape[-1] == cin else x.reshape(-1, cin)
            spk = (r.abs() > 0).detach().cpu().numpy()
            packed = np.packbits(spk, axis=1)
            if 'cb' not in fitted:
                freq = {}
                for row in packed:
                    k = row.tobytes()
                    freq[k] = freq.get(k, 0) + 1
                top = sorted(freq.items(), key=lambda z: -z[1])[:K]
                fitted['cb'] = np.stack([np.frombuffer(k, dtype=np.uint8) for k, _ in top])
            cb = fitted['cb']
            dist = POP[packed[:, None, :] ^ cb[None, :, :]].sum(axis=2)
            bits = np.unpackbits(cb[dist.argmin(1)], axis=1)[:, :cin]
            scale = x.abs().amax()
            new = torch.from_numpy(bits.astype(np.float32)).to(x.device, x.dtype).reshape_as(x)
            return (new if float(scale) == 0 else new * scale,)
        h = mods[TARGET].register_forward_pre_hook(pre)
        vals = []
        with torch.no_grad():
            for name in names:
                functional.reset_net(model)
                x, label, mask = input_frame(DATA, name)
                model(x)
                pred = F.interpolate(bucket.pop('p2').sum(0), size=(480, 640), mode='bilinear', align_corners=False)
                vals.append(aee(pred, label, mask))
                print(f'L1PHI{K}', name, round(vals[-1], 4), flush=True)
                del x, label, mask, pred
        h.remove()
        out[str(K)] = float(np.mean(vals))
    (HERE / 'results' / 'phi_l1.json').write_text(json.dumps({'AEE_full_ref': 0.7157699492039079, 'phi_L1B0_fc1': out}, indent=2))
    print('SUMMARY', out, flush=True)
if __name__ == '__main__':
    main()
