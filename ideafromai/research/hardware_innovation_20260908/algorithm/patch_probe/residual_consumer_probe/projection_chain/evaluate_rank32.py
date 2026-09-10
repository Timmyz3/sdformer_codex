"""One ordinary rank32 PED control on the common branch-deletion parent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--split', choices=('diverse', 'valid'), default='diverse')
    parser.add_argument('--count', type=int, default=10)
    args = parser.parse_args()
    alg = args.root/'algorithm'
    area = alg/'patch_probe'
    residual = area/'residual_consumer_probe'
    latent = area/'factor_completion_20260909/latent_stage_train16'
    args.output = args.output or residual/f'projection_chain/rank32_{args.split}{args.count}'
    args.output.mkdir(parents=True, exist_ok=True)
    for path in (alg, alg/'nrv_cost_probe', latent, residual):
        sys.path.insert(0, str(path))
    import run_probe as probe
    from run_bn_probe import read_names, save_json
    from adapter import install_latent_factor
    from capture import BLOCK, PROJECT
    from evaluate_branch_control import evaluate_axis, mask_nonanchors

    system = probe.load_system(args)
    model, modules, _, current, sources, _, _, _ = system
    probe.install_sources(system, sources)
    current['count_codes'] = False
    fixed = torch.load(area/'patch_train_calibration.pt', map_location='cpu', weights_only=False)
    for name, values in fixed.items():
        bn = modules[name]
        bn.track_running_stats = True
        bn.running_mean = values['mean'].to(bn.weight)
        bn.running_var = values['var'].to(bn.weight)
    model.eval()
    parent = latent/'flow_recovery64/preview_only/shared48_u8_vq5.npz'
    conv1, sn2 = modules[BLOCK+'.conv1.0'], modules[BLOCK+'.sn2.spiking_neuron']
    _, originals = install_latent_factor(conv1, sn2, parent, conditional=False)
    conv = modules[PROJECT+'.conv_res']
    original_forward = conv.forward
    c = conv.weight.detach()[:, :, 0, 0].double().cpu()
    left, singular, right = torch.linalg.svd(c, full_matrices=False)
    first = right[:32].float().to(conv.weight)[:, :, None, None]
    second = (left[:, :32]*singular[:32]).float().to(conv.weight)[:, :, None, None]
    np.savez_compressed(args.output/'parameters.npz', C=c.numpy(),
        U=first[:, :, 0, 0].cpu().numpy(), V=second[:, :, 0, 0].cpu().numpy(),
        singular_values=singular.numpy(), rank=np.array(32),
        bias=np.zeros(96, np.float32) if conv.bias is None else conv.bias.detach().cpu().numpy(),
        has_bias=np.array(conv.bias is not None))
    def projected(x):
        return F.conv2d(F.conv2d(x[:, :, ::2, ::2], first), second, conv.bias)
    masks = {}
    def delete_nonanchor(module, inputs, output):
        key = (output.shape[-2:], output.device)
        if key not in masks:
            mask = torch.zeros(output.shape[-2:], device=output.device, dtype=torch.bool)
            mask[::2, ::2] = True
            masks[key] = mask
        return mask_nonanchors(output, masks[key])
    hook = modules[BLOCK+'.norm2'].register_forward_hook(delete_nonanchor)
    names = (read_names(args.data, 'valid') if args.split == 'valid' else
             json.loads((alg/'samples.json').read_text())['valid'])[:args.count]
    run = dict(complete=False, parent=str(parent), files=names, split=args.split, rank=32,
        common_parent='R32 U8/VQ5 preview-only plus nonanchor whole normalized branch deletion',
        method='Unweighted FP64 SVD of original C; U=right vectors, V=left*singular. Two actual FP32 conv2d calls; stride2 anchor selection first, original bias once.',
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32,
        TF32_cudnn=torch.backends.cudnn.allow_tf32,
        scope='Ordinary changed projection student; no training, integer equivalence or speed claim.')
    save_json(args.output/'run.json', run)
    try:
        conv.forward = projected
        run['result'] = evaluate_axis(args, model, current, names,
                                     'ordinary_projection_rank32', progress_tag='RANK32_CONTROL')
        run['complete'] = True
        save_json(args.output/'run.json', run)
    finally:
        hook.remove()
        conv.forward = original_forward
        conv1.forward, sn2.forward = originals['conv1_forward'], originals['neuron_forward']
    print('DONE', json.dumps(run['result']), flush=True)


if __name__ == '__main__':
    main()
