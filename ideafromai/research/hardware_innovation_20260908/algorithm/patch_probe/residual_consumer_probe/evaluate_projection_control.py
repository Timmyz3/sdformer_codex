"""Fixed X12/W8 ordinary PED projection control; root launches CUDA.

The shared parent includes nonanchor_norm2_branch_zero. Only conv_res changes.
Calibration uses the existing four training captures, never validation inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from capture import BLOCK, PROJECT, convolution_anchor_mask
from evaluate_branch_control import evaluate_axis, mask_nonanchors

AXES = ('float_projection', 'x12_w8_projection')


def dyadic_scale(maximum, limit):
    maximum = np.asarray(maximum, dtype=np.float64)
    return np.exp2(np.ceil(np.log2(np.where(maximum > 0, maximum/limit, 1.))))


def prepare_parameters(capture, output):
    with np.load(capture/'projection_parameters.npz') as data:
        c = data['conv_res_weight'].copy()
        bias = data['conv_res_bias'].copy()
        geometry = {key: data[key].copy() for key in (
            'conv_res_kernel_size', 'conv_res_stride', 'conv_res_padding',
            'conv_res_dilation', 'conv_res_groups')}
    if c.shape != (96, 96, 1, 1) or bias.shape != (96,):
        raise ValueError('Expected the captured96x96 pointwise projection and96 bias values.')
    c = c[:, :, 0, 0]
    vectors, files, source_files, offsets, positions = [], [], [], [0], []
    for file in sorted(capture.glob('[0-9][0-9]_*.npz')):
        with np.load(file) as data:
            if str(data['split']) != 'train':
                raise ValueError('Projection calibration accepts only captured training frames.')
            anchor = data['anchor_mask'].astype(bool)
            x = data['r1out'][:, :, anchor].transpose(0, 2, 1).reshape(-1, 96).copy()
            vectors.append(x)
            files.append(str(data['frame_name']))
            source_files.append(str(file))
            positions.append(data['positions'][anchor].copy())
            offsets.append(offsets[-1]+len(x))
    x = np.concatenate(vectors).astype(np.float32)
    sx = float(dyadic_scale(np.abs(x.astype(np.float64)).max(), 2047))
    sw = dyadic_scale(np.abs(c.astype(np.float64)).max(axis=1), 127)
    xr = np.rint(x.astype(np.float64)/sx)
    wr = np.rint(c.astype(np.float64)/sw[:, None])
    xq = np.clip(xr, -2048, 2047).astype(np.int16)
    wq = np.clip(wr, -127, 127).astype(np.int8)
    dot = xq.astype(np.int64) @ wq.astype(np.int64).T
    # Match the declared deployment conversion order, not a second FP64 dequant.
    result = dot.astype(np.float32)*(sx*sw).astype(np.float32)[None, :]+bias[None, :]
    reference = x.astype(np.float64) @ c.astype(np.float64).T+bias.astype(np.float64)
    error = result.astype(np.float64)-reference
    abs_bound = 2048*np.abs(wq.astype(np.int64)).sum(axis=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, C=c, Wq=wq, input_scale=np.array(sx),
        row_scale=sw, original_bias=bias, calibration_Xq=xq,
        calibration_files=np.asarray(files), calibration_source_files=np.asarray(source_files),
        calibration_vector_offsets=np.asarray(offsets),
        calibration_anchor_positions=np.stack(positions),
        input_integer_range=np.array([-2048, 2047]), weight_integer_range=np.array([-127, 127]),
        dot_absolute_bound=abs_bound, **geometry,
        definition=np.array('Select stride2 anchor inputs first. Xq=RNE(X/input_scale) clamp[-2048,2047]. Wq=RNE(C/row_scale[h]) clamp[-127,127]. Exact integer dot via FP64; cast dot toFP32, multiply by input_scale*row_scale[h], add originalFP32 bias once. No theta/BN folding.'))
    summary = dict(parameters=str(output), capture=str(capture), calibration_files=files,
        calibration_source_files=source_files, calibration_vectors=len(x),
        calibration_input_values=int(x.size), input_scale=sx, row_scale=sw.tolist(),
        C_shape=list(c.shape), Wq_shape=list(wq.shape), original_bias_shape=list(bias.shape),
        original_bias_max_abs=float(np.abs(bias).max()),
        bias_presence='Capture saves a zero-filled vector when absent; evaluator uses the actual live module bias object, once.',
        input_integer_range=[-2048, 2047], weight_integer_range=[-127, 127],
        calibration_input_clipped_values=int(((xr < -2048) | (xr > 2047)).sum()),
        calibration_input_clipped_fraction=float(((xr < -2048) | (xr > 2047)).mean()),
        weight_clipped_values=int(((wr < -127) | (wr > 127)).sum()),
        local_projection_output_values=int(result.size),
        local_error_reference='Captured anchorFP32 values with original C in NumPyFP64 dot; not CUDAFP32 bit-equivalence.',
        local_max_abs_error=float(np.abs(error).max()),
        local_RMSE=float(np.sqrt(np.square(error).mean())),
        local_mean_abs_error=float(np.abs(error).mean()),
        local_relative_RMSE=float(np.sqrt(np.square(error).sum()/np.square(reference).sum())),
        legal_input_dot_max_abs=int(abs_bound.max()),
        legal_input_dot_within_FP64_exact_integer=bool(abs_bound.max() < 2**53),
        calibration_Xq_layout='[vector,C96], vectors ordered frame thenT then captured anchor order; offsets identify each frame.',
        shared_parent='R32 preview-only U8/VQ5 + nonanchor_norm2_branch_zero; calibration anchors equal the complete captured parent because this deletion leaves all anchors unchanged.',
        scope='Four training frames and64 sampled anchors per frame; no validation calibration and no whole-network fixed-point claim.')
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
    return summary


class IntegerProjection:
    def __init__(self, conv, arrays):
        if (tuple(conv.kernel_size), tuple(conv.stride), tuple(conv.padding),
                tuple(conv.dilation), conv.groups) != ((1, 1), (2, 2), (0, 0), (1, 1), 1):
            raise ValueError('This control is for the actual1x1/stride2 ungrouped PED conv_res.')
        device = conv.weight.device
        if not torch.equal(conv.weight.detach()[:, :, 0, 0], torch.as_tensor(arrays['C'], device=device)):
            raise ValueError('Live projection weight differs from the captured calibration weight.')
        self.original_forward = conv.forward
        self.w = torch.as_tensor(arrays['Wq'], device=device, dtype=torch.float64)[:, :, None, None]
        self.input_scale = float(arrays['input_scale'])
        self.output_scale = torch.as_tensor(arrays['row_scale']*self.input_scale,
                                            device=device, dtype=torch.float32)[None, :, None, None]
        self.bias = conv.bias
        if self.bias is not None and not torch.equal(self.bias.detach(), torch.as_tensor(arrays['original_bias'], device=device)):
            raise ValueError('Live projection bias differs from its captured value.')
        self.counts = {}
        self.reset_counts()

    def reset_counts(self):
        self.counts = dict(frames=0, input_values=0, clipped_values=0,
                          runtime_has_bias=self.bias is not None)

    def forward(self, x):
        anchor = x[:, :, ::2, ::2]
        rounded = torch.round(anchor/self.input_scale)
        self.counts['frames'] += 1
        self.counts['input_values'] += rounded.numel()
        self.counts['clipped_values'] += int(((rounded < -2048) | (rounded > 2047)).sum())
        integer = rounded.clamp(-2048, 2047).double()
        dot = F.conv2d(integer, self.w)
        output = dot.float()*self.output_scale
        if self.bias is not None:
            output = output+self.bias[None, :, None, None]
        return output


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--parameters', type=Path)
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--split', choices=('diverse', 'valid'), default='diverse')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--axes', nargs='+', choices=AXES, default=list(AXES))
    parser.add_argument('--integer-label', default='x12_w8_projection',
                        help='Result label for a supplied alternative integer coefficient package.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    algorithm = args.root/'algorithm'
    area = algorithm/'patch_probe'
    residual = area/'residual_consumer_probe'
    latent = area/'factor_completion_20260909/latent_stage_train16'
    args.parent = args.parent or latent/'flow_recovery64/preview_only/shared48_u8_vq5.npz'
    args.parameters = args.parameters or residual/'projection_control_parameters.npz'
    args.capture = args.capture or residual/'capture_train4_ped'
    args.output = args.output or residual/f'projection_control_{args.split}{args.count}'
    if not args.parameters.exists():
        prepare_parameters(args.capture, args.parameters)
    if args.prepare_only:
        print('PARAMETERS_READY', str(args.parameters), flush=True)
        return
    with np.load(args.parameters) as data:
        arrays = {key: data[key].copy() for key in data.files}
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
    conv1, neuron = modules[BLOCK+'.conv1.0'], modules[BLOCK+'.sn2.spiking_neuron']
    pair, originals = install_latent_factor(conv1, neuron, args.parent, conditional=False)
    conv = modules[PROJECT+'.conv_res']
    quantized = IntegerProjection(conv, arrays)
    geometry = dict(kernel=tuple(conv.kernel_size), stride=tuple(conv.stride),
                    padding=tuple(conv.padding), dilation=tuple(conv.dilation))
    masks = {}
    def delete_nonanchor_branch(module, inputs, output):
        height, width = output.shape[-2:]
        key = (height, width, output.device)
        if key not in masks:
            positions = torch.arange(height*width, device=output.device).reshape(-1, 4)
            masks[key] = convolution_anchor_mask(positions, (height, width), **geometry).reshape(height, width)
        return mask_nonanchors(output, masks[key])
    hook = modules[BLOCK+'.norm2'].register_forward_hook(delete_nonanchor_branch)
    names = (read_names(args.data, 'valid') if args.split == 'valid' else
             json.loads((algorithm/'samples.json').read_text())['valid'])[:args.count]
    metadata = dict(complete=False, files=names, split=args.split,
        requested_count=args.count, selected_axes=args.axes, parent=str(args.parent),
        parameters=str(args.parameters), shared_control='nonanchor_norm2_branch_zero',
        numeric=str(arrays['definition']),
        axes={'float_projection': 'Original conv_res forward on the common branch-deletion parent.',
            args.integer_label: str(arrays['definition'])},
        integer_result_label=args.integer_label,
        original_bias_is_present=conv.bias is not None,
        original_bias_shape=None if conv.bias is None else list(conv.bias.shape),
        metric='Unchanged evaluate_branch_control.evaluate_axis coarse-head AEE.',
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32,
        TF32_cudnn=torch.backends.cudnn.allow_tf32,
        claim='New projection-quantized student only; native proj.sn and all other real successors preserved. No full-network fixed-point or software/hardware speed claim.',
        results={})
    save_json(args.output/'run.json', metadata)
    try:
        for axis in args.axes:
            quantized.reset_counts()
            conv.forward = quantized.original_forward if axis == 'float_projection' else quantized.forward
            result_label = axis if axis == 'float_projection' else args.integer_label
            result = evaluate_axis(args, model, current, names, result_label, progress_tag='PROJECTION_CONTROL_AEE')
            if axis == 'x12_w8_projection':
                result['projection_quantization'] = dict(quantized.counts,
                    clipped_fraction=quantized.counts['clipped_values']/quantized.counts['input_values'])
                save_json(args.output/(result_label+'_summary.json'), result)
            metadata['results'][result_label] = result
            save_json(args.output/'run.json', metadata)
        metadata['complete'] = True
        save_json(args.output/'run.json', metadata)
    finally:
        hook.remove()
        conv.forward = quantized.original_forward
        conv1.forward, neuron.forward = originals['conv1_forward'], originals['neuron_forward']
    print('DONE', json.dumps(metadata['results']), flush=True)


if __name__ == '__main__':
    main()
