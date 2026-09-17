"""C-STEP local-common-spike copy on r0.conv2: how often neighbors share the same 96-bit spike vector?"""
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
CONV2 = 'sttmultires_unet.encoders.swin3d.patch_embed.residual_encoding.resblocks.0.conv2.0'

def main():
    args = SimpleNamespace(code_root=REPO, config=INCOMING/'dsec_c12_alpha0125_ep29_resume5_20260830.yml',
                           checkpoint=INCOMING/'checkpoint_epoch34.pth', data=DATA)
    model, *_ = build_model(args)
    bucket = {}
    dict(model.named_modules())[CONV2].register_forward_pre_hook(lambda m, a: bucket.__setitem__('x', a[0].detach()))
    names = [p.name for p in sorted((DATA/'gt_tensors').glob('*.npy'))]
    rows = []
    with torch.no_grad():
        for name in names:
            functional.reset_net(model)
            x, _, _ = input_frame(DATA, name)
            model(x)
            act = bucket.pop('x')
            if act.ndim == 5:
                act = act[:, 0]
            spk = (act.abs() > 0).cpu().numpy()  # T,C,H,W
            T, C, H, W = spk.shape
            heq = hden = veq = vden = nnz = 0
            for t in range(T):
                packed = np.packbits(spk[t], axis=0)  # Ceil(C/8),H,W
                nz = spk[t].any(axis=0)
                nnz += int(nz.sum())
                same_h = (packed[:, :, :-1] == packed[:, :, 1:]).all(axis=0)
                same_v = (packed[:, :-1, :] == packed[:, 1:, :]).all(axis=0)
                pair_h = nz[:, :-1] & nz[:, 1:]
                pair_v = nz[:-1, :] & nz[1:, :]
                heq += int((same_h & pair_h).sum())
                hden += int(pair_h.sum())
                veq += int((same_v & pair_v).sum())
                vden += int(pair_v.sum())
            rec = {
                'file': name,
                'nnz_spatial_frac': nnz / (T * H * W),
                'horiz_equal_given_both_nz': heq / max(hden, 1),
                'vert_equal_given_both_nz': veq / max(vden, 1),
                'horiz_pairs_nz': hden,
            }
            rows.append(rec)
            print('COMMON', rec, flush=True)
            del x, act, spk
    hrate = float(np.mean([r['horiz_equal_given_both_nz'] for r in rows]))
    vrate = float(np.mean([r['vert_equal_given_both_nz'] for r in rows]))
    # C-STEP paper: K=2 common fraction up to 82.7% of spike *positions* (channels)
    out = {'mean_horiz_same_vector': hrate, 'mean_vert_same_vector': vrate, 'frames': rows,
           'verdict': 'if neighbor equality << 0.3, local-common reuse is weak on this layer'}
    (HERE/'results'/'common.json').write_text(json.dumps(out, indent=2))
    print('SUMMARY', json.dumps({k: out[k] for k in out if k != 'frames'}, indent=2), flush=True)
if __name__ == '__main__':
    main()
