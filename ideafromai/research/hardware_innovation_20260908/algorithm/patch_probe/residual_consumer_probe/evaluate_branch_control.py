"""Three fixed ordinary nonanchor branch controls on the same R32 parent.

Root launches CUDA. All operators are still computed before software masks;
only network AEE is measured, never GPU/hardware acceleration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

from capture import BLOCK, PROJECT, convolution_anchor_mask

AXES = ('full', 'nonanchor_conv2_raw_zero', 'nonanchor_norm2_branch_zero')


def mask_nonanchors(output, anchor):
    if output.ndim != 5 or output.shape[:3] != (10, 1, 96):
        raise ValueError('Expected the actual r1 Conv2/norm2 T10,B1,C96 output.')
    return torch.where(anchor[None, None, None], output, torch.zeros((), device=output.device, dtype=output.dtype))


@torch.no_grad()
def evaluate_axis(args, model, current, names, axis, *, progress_tag='BRANCH_CONTROL_AEE'):
    """Shared original coarse-head/AEE loop for the ordinary controls."""
    from run_bn_probe import input_frame, save_json
    from evaluate_stage2_deployment import CoarseReady, summarize
    from spikingjelly.activation_based import functional
    rows, started = [], time.monotonic()
    for i, name in enumerate(names):
        functional.reset_net(model)
        current.pop('flow', None)
        x, label, valid = input_frame(args.data, name)
        try:
            model(x)
        except CoarseReady:
            pred = F.interpolate(current.pop('flow'), (480, 640), mode='bilinear', align_corners=False)
        else:
            raise RuntimeError('The existing real preds.2 coarse exit was not reached.')
        # Exact same metric calculation as evaluate_factors_network.
        error = torch.linalg.vector_norm(pred.permute(0, 2, 3, 1)[valid]
            -label.permute(0, 2, 3, 1)[valid], dim=1)
        total, count = float(error.double().sum()), error.numel()
        rows.append(dict(file=name, valid_pixels=count, aee_sum=total, AEE=total/count))
        summary = dict(**summarize(rows, i+1 == len(names)), axis=axis,
            wall_seconds=time.monotonic()-started,
            claim='Dense function/accuracy check; elapsed time is not a speed comparison')
        save_json(args.output/(axis+'_frames.json'), rows)
        save_json(args.output/(axis+'_summary.json'), summary)
        if args.split == 'diverse' or (i+1) % 50 == 0 or i+1 == len(names):
            print(progress_tag, axis, i+1, summary['AEE_frame_mean'], flush=True)
        del x, label, valid, pred, error
    return summary


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--split', choices=('diverse', 'valid'), default='diverse')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--axes', nargs='+', choices=AXES, default=list(AXES))
    args = parser.parse_args()
    algorithm = args.root/'algorithm'
    area = algorithm/'patch_probe'
    latent = area/'factor_completion_20260909/latent_stage_train16'
    args.parent = args.parent or latent/'flow_recovery64/preview_only/shared48_u8_vq5.npz'
    args.output = args.output or area/f'residual_consumer_probe/branch_control_{args.split}{args.count}'
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(algorithm/'nrv_cost_probe'))
    sys.path.insert(0, str(algorithm))
    sys.path.insert(0, str(latent))
    import run_probe as probe
    from run_bn_probe import read_names, save_json
    from adapter import install_latent_factor
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
    conv_res = modules.get(PROJECT+'.conv_res')
    if conv_res is None:
        raise ValueError('The actual continuous PED projection is missing; do not assume nonanchor positions.')
    if (tuple(conv_res.kernel_size), tuple(conv_res.stride), tuple(conv_res.padding), tuple(conv_res.dilation)) != ((1, 1), (2, 2), (0, 0), (1, 1)):
        raise ValueError('This fixed control was specified for the observed1x1/stride2/no-padding PED projection.')
    conv1, neuron = modules[BLOCK+'.conv1.0'], modules[BLOCK+'.sn2.spiking_neuron']
    pair, originals = install_latent_factor(conv1, neuron, args.parent, conditional=False)
    names = (read_names(args.data, 'valid') if args.split == 'valid' else
             json.loads((algorithm/'samples.json').read_text())['valid'])[:args.count]
    geometry = dict(kernel=tuple(conv_res.kernel_size), stride=tuple(conv_res.stride),
        padding=tuple(conv_res.padding), dilation=tuple(conv_res.dilation))
    metadata = dict(complete=False, files=names, split=args.split,
        requested_count=args.count, evaluated_count=len(names), selected_axes=args.axes, parent=str(args.parent),
        model='same preview-only R32 dequantized U8/VQ5 LatentPair full; fixed4patchBN; original real Conv2, PED, S2 and coarse successors',
        axes=dict(full='No change after installing the common parent.',
            nonanchor_conv2_raw_zero='Conv2.0 forward output is zeroed only at nonanchors; real fixed BN2 subsequently runs, retaining BN2(0).',
            nonanchor_norm2_branch_zero='The complete norm2 forward output is zeroed only at nonanchors; identity is retained.'),
        anchor_definition='Input positions used by the actual PED conv_res; geometry-derived even/even for observed1x1stride2. Every source value at these anchors remains unchanged.',
        continuous_projection_geometry=geometry,
        metric='Unchanged evaluate_factors_network formula and evaluate_stage2_deployment.summarize: real preds.2 sum over time, bilinear480x640 with align_corners=False, no flow-value scaling, EPE at existing valid GT pixels.',
        numeric='FP32/dequantized parent; no new integer source or residual arithmetic',
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32, TF32_cudnn=torch.backends.cudnn.allow_tf32,
        claim='Accuracy ablations only. Dense Conv/BN is executed before masking; no software or hardware speed is measured.',
        results={})
    save_json(args.output/'run.json', metadata)
    mask_cache = {}
    hook = None
    def anchor_for(output):
        height, width = output.shape[-2:]
        key = (height, width, output.device)
        if key not in mask_cache:
            positions = torch.arange(height*width, device=output.device).reshape(-1, 4)
            anchor = convolution_anchor_mask(positions, (height, width), **geometry).reshape(height, width)
            mask_cache[key] = anchor
            metadata['anchor_spatial_counts'] = dict(height=height, width=width,
                anchors=int(anchor.sum()), nonanchors=int((~anchor).sum()),
                anchor_fraction=float(anchor.float().mean()))
        return mask_cache[key]
    def zero_hook(module, inputs, output):
        return mask_nonanchors(output, anchor_for(output))
    try:
        for axis in args.axes:
            if hook is not None:
                hook.remove(); hook = None
            if axis == 'nonanchor_conv2_raw_zero':
                hook = modules[BLOCK+'.conv2.0'].register_forward_hook(zero_hook)
            elif axis == 'nonanchor_norm2_branch_zero':
                hook = modules[BLOCK+'.norm2'].register_forward_hook(zero_hook)
            summary = evaluate_axis(args, model, current, names, axis)
            metadata['results'][axis] = summary
            save_json(args.output/'run.json', metadata)
        metadata['complete'] = True
        save_json(args.output/'run.json', metadata)
    finally:
        if hook is not None:
            hook.remove()
        conv1.forward, neuron.forward = originals['conv1_forward'], originals['neuron_forward']
    print('DONE', json.dumps(metadata['results']), flush=True)


if __name__ == '__main__':
    main()
