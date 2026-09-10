"""Fixed ordinary Conv2 low-rank controls and one consumer diagnostic.

One Float64 SVD of the captured W2 supplies nested rank16/rank32 factors.
Execution uses saved FP32 factors and the unchanged real BN2/consumers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from capture import BLOCK, PROJECT, CONSUMER_SN, convolution_anchor_mask
from evaluate_branch_control import evaluate_axis

AXES = ('uniform_rank16', 'uniform_rank32', 'nonanchor_rank16_anchor_original',
        'rank16_gate_only_exact_value')


def prepare_parameters(bound_path, output):
    """A weight-only ordinary SVD, with no activation/validation fitting."""
    with np.load(bound_path) as data:
        arrays = {key: data[key].copy() for key in (
            'W2', 'conv2_bias', 'conv2_has_bias', 'conv2_stride',
            'conv2_padding', 'conv2_dilation', 'conv2_groups', 'sn2_theta')}
    w = arrays['W2'].astype(np.float64).reshape(96, 864)
    left, singular, right = np.linalg.svd(w, full_matrices=False)
    scale = np.sqrt(singular[:32])
    first = scale[:, None]*right[:32]
    second = left[:, :32]*scale[None, :]
    arrays.update(first_factor64=first, second_factor64=second,
                  first_factor32=first.astype(np.float32),
                  second_factor32=second.astype(np.float32),
                  singular_values=singular,
                  ranks=np.array([16, 32]),
                  source_theta=arrays['sn2_theta'].copy(),
                  definition=np.array('W[H,C,kh,kw] -> W[96,864]; one FP64 SVD W=L diag(s) R; first=sqrt(s)*R, second=L*sqrt(s); saved FP32 factors execute 3x3 then1x1; original bias after second factor; theta and BN not folded'))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    denom = float(np.square(w).sum())
    summary = dict(source=str(bound_path), parameters=str(output),
        axes=list(AXES), decomposition_dtype='Float64', execution_dtype='Float32',
        factor_layout=dict(first_factor32='[32,864], k=((c*3)+kh)*3+kw',
                           second_factor32='[96,32]'),
        theta='Source remains actual theta*g; theta is not folded into these factors.',
        bias='Original Conv2 bias is applied once after the 1x1 factor; captured Conv2 has no bias.',
        BN='Real fixed BN2 executes after the selected Conv2 output; no BN folding.',
        fitting='Weight-only SVD; no training, activation calibration, or validation-based rank selection.',
        nested='R16 is the first16 latent coordinates of the same saved R32 decomposition.',
        original_weight_count=int(w.size), ranks={})
    for rank in (16, 32):
        reconstructed = second[:, :rank] @ first[:rank]
        rounded = second[:, :rank].astype(np.float32).astype(np.float64) @ first[:rank].astype(np.float32).astype(np.float64)
        summary['ranks'][str(rank)] = dict(
            coefficient_count=int(rank*(96+864)),
            retained_weight_squared_energy=float(np.square(singular[:rank]).sum()/denom),
            relative_Frobenius_error=float(np.linalg.norm(reconstructed-w)/np.sqrt(denom)),
            rounded_factor_relative_Frobenius_error=float(np.linalg.norm(rounded-w)/np.sqrt(denom)))
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
    return summary


class RankConvControl:
    def __init__(self, conv, conv_res, arrays):
        self.original_forward = conv.forward
        device, dtype = conv.weight.device, conv.weight.dtype
        captured = torch.as_tensor(arrays['W2'], device=device, dtype=dtype)
        if not torch.equal(conv.weight.detach(), captured):
            raise ValueError('Live Conv2 weight differs from the captured SVD source.')
        self.first = torch.as_tensor(arrays['first_factor32'], device=device, dtype=dtype).reshape(32, 96, 3, 3)
        self.second = torch.as_tensor(arrays['second_factor32'], device=device, dtype=dtype).reshape(96, 32, 1, 1)
        self.bias = conv.bias
        self.stride, self.padding, self.dilation = conv.stride, conv.padding, conv.dilation
        if conv.groups != 1:
            raise ValueError('This ordinary SVD control uses the actual ungrouped Conv2.')
        self.geometry = dict(kernel=tuple(conv_res.kernel_size), stride=tuple(conv_res.stride),
                             padding=tuple(conv_res.padding), dilation=tuple(conv_res.dilation))
        if tuple(self.geometry.values()) != ((1, 1), (2, 2), (0, 0), (1, 1)):
            raise ValueError('Anchor-original control requires the observed PED1x1/stride2 geometry.')
        self.axis = AXES[0]
        self.masks = {}
        self.checks = {}
        self.frame_cache = {}

    def factor_forward(self, x, rank):
        t, b = x.shape[:2]
        z = F.conv2d(x.flatten(0, 1), self.first[:rank], None,
                     self.stride, self.padding, self.dilation)
        output = F.conv2d(z, self.second[:, :rank], self.bias)
        return output.reshape(t, b, 96, *output.shape[-2:])

    def forward(self, x):
        if self.axis == 'rank16_gate_only_exact_value':
            original = self.original_forward(x)
            self.frame_cache['approximate_raw'] = self.factor_forward(x, 16)
            return original
        rank = 32 if self.axis == 'uniform_rank32' else 16
        output = self.factor_forward(x, rank)
        if self.axis == 'nonanchor_rank16_anchor_original':
            original = self.original_forward(x)
            height, width = output.shape[-2:]
            key = (height, width, output.device)
            if key not in self.masks:
                positions = torch.arange(height*width, device=output.device).reshape(-1, 4)
                self.masks[key] = convolution_anchor_mask(positions, (height, width), **self.geometry).reshape(height, width)
            anchor = self.masks[key]
            output = torch.where(anchor[None, None, None], original, output)
            if self.axis not in self.checks:
                self.checks[self.axis] = dict(anchor_positions=int(anchor.sum()),
                    total_positions=height*width,
                    anchor_value_differences=int(torch.count_nonzero(output[..., anchor]-original[..., anchor])))
        return output


class GateOnlyRouting:
    """Keep the complete PED value path; replace only its neuron input."""
    def __init__(self, control, modules):
        self.control = control
        self.cache = control.frame_cache
        self.norm2_forward = modules[BLOCK+'.norm2'].forward
        self.handles = []
        for name, hook, pre in (
            (BLOCK, self.before_block, True),
            (BLOCK+'.norm2', self.after_norm2, False),
            (BLOCK, self.after_block, False),
            (PROJECT+'.conv_res', self.before_conv_res, True),
            (CONSUMER_SN, self.before_consumer, True),
            (CONSUMER_SN, self.after_consumer, False)):
            module = modules[name]
            self.handles.append(module.register_forward_pre_hook(hook) if pre else
                                module.register_forward_hook(hook))

    def before_block(self, module, inputs):
        self.cache.clear()
        self.cache['identity'] = inputs[0]

    def after_norm2(self, module, inputs, output):
        # Direct forward avoids recursively invoking this wrapper's hook.
        # The actual fixed BN2 runs unchanged on both full and rank16 raw Y.
        raw = self.cache.pop('approximate_raw')
        branch = self.norm2_forward(raw)
        self.cache['approximate_r1out'] = branch+self.cache.pop('identity')

    def after_block(self, module, inputs, output):
        self.cache['original_r1out'] = output

    def before_conv_res(self, module, inputs):
        original = self.cache['original_r1out']
        value = inputs[0].reshape_as(original)
        self.cache['continuous_path_seen'] = True
        if self.control.axis not in self.control.checks:
            self.control.checks[self.control.axis] = dict(
                conv_res_input_value_differences=int(torch.count_nonzero(value-original)),
                conv_res_input_same_storage=value.data_ptr() == original.data_ptr())

    def before_consumer(self, module, inputs):
        if not self.cache.get('continuous_path_seen', False):
            raise RuntimeError('Expected actual PED conv_res before proj.sn; routing is not valid for this execution order.')
        checks = self.control.checks[self.control.axis]
        if 'proj_original_input_value_differences' not in checks:
            checks['proj_original_input_value_differences'] = int(torch.count_nonzero(
                inputs[0]-self.cache['original_r1out']))
            checks['approximate_vs_complete_input_max_abs'] = float((
                self.cache['approximate_r1out']-self.cache['original_r1out']).abs().max())
        return (self.cache['approximate_r1out'], *inputs[1:])

    def after_consumer(self, module, inputs, output):
        checks = self.control.checks[self.control.axis]
        checks['proj_sn_received_rank16_input'] = inputs[0] is self.cache['approximate_r1out']
        self.cache.clear()

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.cache.clear()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--parameters', type=Path)
    parser.add_argument('--bound-parameters', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--split', choices=('diverse', 'valid'), default='diverse')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--axes', nargs='+', choices=AXES, default=list(AXES))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    algorithm = args.root/'algorithm'
    area = algorithm/'patch_probe'
    residual = area/'residual_consumer_probe'
    latent = area/'factor_completion_20260909/latent_stage_train16'
    args.parent = args.parent or latent/'flow_recovery64/preview_only/shared48_u8_vq5.npz'
    args.parameters = args.parameters or residual/'rank_control_parameters.npz'
    args.bound_parameters = args.bound_parameters or residual/'capture_train4_ped/bound_parameters.npz'
    args.output = args.output or residual/f'rank_control_{args.split}{args.count}'
    if not args.parameters.exists():
        prepare_parameters(args.bound_parameters, args.parameters)
    if args.prepare_only:
        print('PARAMETERS_READY', str(args.parameters), flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.parameters) as data:
        arrays = {key: data[key].copy() for key in data.files}
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
    conv2 = modules[BLOCK+'.conv2.0']
    control = RankConvControl(conv2, modules[PROJECT+'.conv_res'], arrays)
    conv2.forward = control.forward
    names = (read_names(args.data, 'valid') if args.split == 'valid' else
             json.loads((algorithm/'samples.json').read_text())['valid'])[:args.count]
    metadata = dict(complete=False, files=names, split=args.split,
        requested_count=args.count, evaluated_count=len(names), selected_axes=args.axes,
        parent=str(args.parent), parameters=str(args.parameters),
        model='Same preview-onlyR32 dequantizedU8/VQ5 Conv1 parent; fixed4patchBN, real Conv2BN2/PED/S2/coarse successors.',
        numeric=str(arrays['definition']),
        axes=dict(uniform_rank16='3x3 to16 continuous latents then1x1 to96 at every position.',
            uniform_rank32='Same saved SVD,32 continuous latents at every position.',
            nonanchor_rank16_anchor_original='Nonanchors select the rank16 output. Anchors select the unmodified original Conv2.forward result, preserving its original numerical value.',
            rank16_gate_only_exact_value='Conv2 and BN2 main path remains original: conv_res receives identity+BN2(originalConv2). Only real proj.sn.spiking_neuron input is replaced with identity+BN2(twoFactorRank16Conv2), at all positions including anchors.'),
        metric='evaluate_branch_control.evaluate_axis, unchanged existing real coarse-head AEE.',
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32,
        TF32_cudnn=torch.backends.cudnn.allow_tf32,
        claim='Fixed ordinary accuracy controls; no training or validation rank selection. Mixed/gate-only software computes both complete routes then selects or routes. No Wres*BNgain*W2 contraction, execution speed, or hardware schedule is claimed.',
        results={})
    save_json(args.output/'run.json', metadata)
    routing = None
    try:
        for axis in args.axes:
            if routing is not None:
                routing.remove()
                routing = None
            control.frame_cache.clear()
            control.axis = axis
            if axis == 'rank16_gate_only_exact_value':
                routing = GateOnlyRouting(control, modules)
            summary = evaluate_axis(args, model, current, names, axis, progress_tag='RANK_CONTROL_AEE')
            if routing is not None:
                routing.remove()
                routing = None
            metadata['results'][axis] = summary
            metadata['numerical_checks'] = control.checks
            save_json(args.output/'run.json', metadata)
        metadata['complete'] = True
        save_json(args.output/'run.json', metadata)
    finally:
        if routing is not None:
            routing.remove()
        control.frame_cache.clear()
        conv2.forward = control.original_forward
        conv1.forward, neuron.forward = originals['conv1_forward'], originals['neuron_forward']
    print('DONE', json.dumps(metadata['results']), flush=True)


if __name__ == '__main__':
    main()
