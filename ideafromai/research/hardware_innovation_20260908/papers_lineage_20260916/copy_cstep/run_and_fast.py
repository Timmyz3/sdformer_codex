"""C-STEP per-channel AND (not full-vector equality) on r0.conv2 and L1.b0.fc1."""
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

HOOKS = {
    'r0.conv2': 'sttmultires_unet.encoders.swin3d.patch_embed.residual_encoding.resblocks.0.conv2.0',
    'L1.b0.fc1': 'sttmultires_unet.encoders.swin3d.layers.1.swin_blocks.0.mlp.fc1',
}


def metrics_nchw(spk):
    # spk: T,C,H,W bool
    left, right = spk[..., :, :-1], spk[..., :, 1:]
    inter = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    jac = float(inter) / max(float(union), 1.0)
    # reuse: P(right|left)
    rec = float(inter) / max(float(left.sum()), 1.0)
    return jac, rec, int(spk.sum())


def metrics_tokens(spk_tnc):
    # T,N,C with N = H*W row-major
    T, N, C = spk_tnc.shape
    # unknown H,W: treat as 1D neighbors along N
    left, right = spk_tnc[:, :-1], spk_tnc[:, 1:]
    inter = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    jac = float(inter) / max(float(union), 1.0)
    rec = float(inter) / max(float(left.sum()), 1.0)
    return jac, rec, int(spk_tnc.sum())


def main():
    names = [p.name for p in sorted((DATA / 'gt_tensors').glob('*.npy'))]
    args = SimpleNamespace(code_root=REPO, config=INCOMING / 'dsec_c12_alpha0125_ep29_resume5_20260830.yml',
                           checkpoint=INCOMING / 'checkpoint_epoch34.pth', data=DATA)
    model, *_ = build_model(args)
    mods = dict(model.named_modules())
    bucket = {}

    def make(key):
        def pre(m, inp, key=key):
            bucket[key] = inp[0].detach()
        return pre

    for k, n in HOOKS.items():
        mods[n].register_forward_pre_hook(make(k))
    rows = []
    with torch.no_grad():
        for name in names:
            functional.reset_net(model)
            x, _, _ = input_frame(DATA, name)
            model(x)
            rec = {'file': name}
            x2 = bucket['r0.conv2']
            if x2.ndim == 5:
                x2 = x2[:, 0]
            spk = (x2.abs() > 0).cpu().numpy()
            jac, reca, nnz = metrics_nchw(spk)
            rec['r0_jaccard'] = jac
            rec['r0_P_right_given_left'] = reca
            rec['r0_nnz'] = nnz
            fc = bucket['L1.b0.fc1']
            cin = mods[HOOKS['L1.b0.fc1']].in_features
            r = fc.reshape(-1, cin) if fc.shape[-1] == cin else fc.reshape(-1, fc.shape[-1])
            spk2 = (r.abs() > 0).cpu().numpy()
            # reshape T=10
            if spk2.shape[0] % 10 == 0:
                tnc = spk2.reshape(10, -1, cin)
                jac2, rec2, nnz2 = metrics_tokens(tnc)
            else:
                jac2 = rec2 = float('nan'); nnz2 = int(spk2.sum())
            rec['fc1_jaccard'] = jac2
            rec['fc1_P_right_given_left'] = rec2
            rec['fc1_nnz'] = nnz2
            rows.append(rec)
            print(name[-12:], 'r0', round(jac, 3), 'p|', round(reca, 3), 'fc1', round(jac2, 3), flush=True)
            del x
    summary = {
        'r0_mean_jaccard': float(np.mean([r['r0_jaccard'] for r in rows])),
        'r0_mean_P_right_given_left': float(np.mean([r['r0_P_right_given_left'] for r in rows])),
        'fc1_mean_jaccard': float(np.mean([r['fc1_jaccard'] for r in rows])),
        'fc1_mean_P_right_given_left': float(np.mean([r['fc1_P_right_given_left'] for r in rows])),
        'cstep_paper_ref': 0.827,
        'frames': rows,
    }
    dest = HERE / 'results'
    dest.mkdir(exist_ok=True)
    (dest / 'channel_and.json').write_text(json.dumps(summary, indent=2))
    print('SUMMARY', json.dumps({k: summary[k] for k in summary if k != 'frames'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
