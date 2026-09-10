"""Read-only r1 residual -> pre-projection PSN boundary and default-gate probe.

Parent: recovered shared48 U8/VQ5 preview-only, dequantized FP32 LatentPair
full mode, four fixed patch BNs and the same real upstream network. Root
launches CUDA; this entry stops after the actual projection neuron output.
Only explicit train16's first four frames and existing64 native P4 are read.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

PATCH = 'sttmultires_unet.encoders.swin3d.patch_embed'
BLOCK = PATCH+'.residual_encoding.resblocks.1'
SOURCE_SN = BLOCK+'.sn1.spiking_neuron'
PROJECT = PATCH+'.proj'
CONSUMER_SN = PROJECT+'.sn.spiking_neuron'


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def description(module):
    return dict(type=type(module).__module__+'.'+type(module).__name__,
        source_file=inspect.getsourcefile(type(module)),
        forward_file=getattr(getattr(module.forward, '__code__', None), 'co_filename', None),
        use_MS=getattr(module, 'use_MS', None), connect_function=getattr(module, 'connect_function', None))


def neuron_parameters(module):
    return dict(A=module.weight.detach().cpu().float(), b=module.bias.detach().cpu().float(),
        theta=module.thresh.detach().cpu().float(), center=module.center.detach().cpu().float(),
        center_mode=module.center_mode, output_mode=module.output_mode,
        threshold_mode=module.threshold_mode, negative_threshold_scale=module.negative_threshold_scale,
        temporal_factor_rank=int(getattr(module, 'temporal_factor_rank', 0)),
        T=int(module.T), activation=getattr(module.act, '__qualname__', str(module.act)))


def membrane(parameters, values, dtype=torch.float32):
    # Same flattened T dimension as native ATLIF; CPU reconstruction is
    # explicitly distinguished from the native full-width CUDA addmm.
    a, b = parameters['A'].to(dtype), parameters['b'].to(dtype).reshape(10, 1)
    result = torch.addmm(b, a, values.to(dtype).reshape(10, -1))
    if parameters['center_mode'] != 'zero':
        result -= parameters['center'].to(dtype).reshape(10, 1)
    return result.reshape_as(values)


def fire(parameters, values):
    theta = parameters['theta'].to(values)
    positive = values.ge(theta)
    if parameters['output_mode'] == 'binary':
        if parameters['threshold_mode'] == 'symmetric_binary_abs':
            positive |= values.le(-theta)
        return positive.to(values.dtype)*theta
    if parameters['output_mode'] == 'ternary':
        scale = 1. if parameters['threshold_mode'] in ('symmetric_bsa_tsn', 'symmetric_target_rate') else parameters['negative_threshold_scale']
        return (positive.to(values.dtype)-values.le(-theta*scale).to(values.dtype))*theta
    raise ValueError('Unsupported actual output definition: '+parameters['output_mode'])


def comparison(default, actual):
    changed = default.ne(actual)
    t, channels, groups, positions = changed.shape
    grouped = changed.reshape(t, channels//8, 8, groups, positions)
    full_word_changed = grouped.any((0, 2, 4))
    each_t_changed = grouped.any((2, 4))
    truth, prediction = actual.ne(0), default.ne(0)
    return dict(gates=changed.numel(), different_gates=int(changed.sum()),
        different_gate_fraction=float(changed.float().mean()), actual_nonzero=int(truth.sum()),
        default_nonzero=int(prediction.sum()), lost_nonzero=int((truth & ~prediction).sum()),
        extra_nonzero=int((~truth & prediction).sum()),
        lost_fraction_of_actual_nonzero=float((truth & ~prediction).sum()/truth.sum().clamp_min(1)),
        position_channel_T10_words=channels*groups*positions,
        position_channel_T10_equal_fraction=float((~changed.any(0)).float().mean()),
        P4_H8_T10_groups=full_word_changed.numel(), P4_H8_T10_equal_groups=int((~full_word_changed).sum()),
        P4_H8_T10_equal_fraction=float((~full_word_changed).float().mean()),
        P4_H8_each_T_groups=each_t_changed.numel(),
        P4_H8_each_T_equal_fraction=float((~each_t_changed).float().mean()))


def convolution_anchor_mask(positions, input_hw, kernel, stride, padding, dilation):
    """Which sampled input positions feed any legal Conv2d output?"""
    height, width = input_hw
    yy, xx = positions//width, positions%width
    output_h = (height+2*padding[0]-dilation[0]*(kernel[0]-1)-1)//stride[0]+1
    output_w = (width+2*padding[1]-dilation[1]*(kernel[1]-1)-1)//stride[1]+1
    used = torch.zeros_like(positions, dtype=torch.bool)
    for kh in range(kernel[0]):
        for kw in range(kernel[1]):
            oy_num = yy+padding[0]-kh*dilation[0]
            ox_num = xx+padding[1]-kw*dilation[1]
            oy, ox = torch.div(oy_num, stride[0], rounding_mode='floor'), torch.div(ox_num, stride[1], rounding_mode='floor')
            used |= ((oy_num % stride[0] == 0) & (ox_num % stride[1] == 0) &
                (oy >= 0) & (oy < output_h) & (ox >= 0) & (ox < output_w))
    return used


def sampled_conv2_source(x, positions, theta):
    """Gather only64 P4 x9 coordinates; no full-frame pad or unfold."""
    t, batch, channels, height, width = x.shape
    if t != 10 or batch != 1:
        raise ValueError('Conv2 source must retain the complete T10, B1 input.')
    yy, xx = positions//width, positions%width
    words = torch.zeros((len(positions), channels, 3, 3, 4), device=x.device, dtype=torch.int16)
    residual = torch.zeros((), device=x.device, dtype=x.dtype)
    sampled_nonzero = torch.zeros((), device=x.device, dtype=torch.int64)
    padding_count = 0
    for kh in range(3):
        for kw in range(3):
            sy, sx = yy+kh-1, xx+kw-1
            valid = (sy >= 0) & (sy < height) & (sx >= 0) & (sx < width)
            values = x.detach()[:, 0, :, sy.clamp(0, height-1), sx.clamp(0, width-1)]
            values = values*valid[None, None]
            active = values.ne(0)
            residual = torch.maximum(residual, (values-active.to(values.dtype)*theta).abs().max())
            sampled_nonzero += active.sum()
            word = torch.zeros((channels, len(positions), 4), device=x.device, dtype=torch.int32)
            for step in range(10):
                word |= active[step].int() << step
            words[:, :, kh, kw] = word.permute(1, 0, 2).short()
            padding_count += int((~valid).sum())*channels*t
    return dict(conv2_source_gate_words=words.reshape(len(positions), channels*9, 4).cpu(),
        conv2_source_theta_g_max_abs=residual.cpu(),
        conv2_source_sampled_nonzero=sampled_nonzero.cpu(),
        conv2_source_padding_scalar_count=torch.tensor(padding_count),
        conv2_source_sampled_scalar_count=torch.tensor(words.numel()*10))


def save_bound_parameters(modules, fixed, pair, output):
    conv = modules[BLOCK+'.conv2.0']
    if (tuple(conv.kernel_size), tuple(conv.stride), tuple(conv.padding), tuple(conv.dilation), conv.groups) != ((3, 3), (1, 1), (1, 1), (1, 1), 1):
        raise ValueError('This exact sampled source layout requires the actual dense stride1 padded3x3 Conv2.')
    bn_names = [name for name in fixed if name.startswith(BLOCK+'.norm2.')]
    if len(bn_names) != 1:
        raise ValueError('Expected one fixed BN2 behind the actual residual branch.')
    bn = modules[bn_names[0]]
    if bn.training or not bn.track_running_stats:
        raise ValueError('Fixed BN2 is required for source-only bound preparation.')
    gamma, beta, mean, var = [getattr(bn, key).detach().cpu().numpy() for key in
        ('weight', 'bias', 'running_mean', 'running_var')]
    gain = gamma.astype(np.float64)/np.sqrt(var.astype(np.float64)+float(bn.eps))
    offset = beta.astype(np.float64)-gain*mean.astype(np.float64)
    arrays = dict(W2=conv.weight.detach().cpu().numpy(),
        conv2_bias=conv.bias.detach().cpu().numpy() if conv.bias is not None else np.zeros(conv.out_channels, np.float32),
        conv2_has_bias=np.array(conv.bias is not None),
        conv2_stride=np.array(conv.stride), conv2_padding=np.array(conv.padding),
        conv2_dilation=np.array(conv.dilation), conv2_groups=np.array(conv.groups),
        bn2_name=np.array(bn_names[0]), bn2_gamma=gamma, bn2_beta=beta,
        bn2_running_mean=mean, bn2_running_var=var, bn2_eps=np.array(bn.eps),
        bn2_gain=gain, bn2_offset=offset, bn2_track_running_stats=np.array(bn.track_running_stats),
        bn2_affine_definition=np.array('gain/offset reconstructed in NumPy Float64 from actual FP32 runtime gamma/beta/running_mean/running_var and eps; CUDA FP32 BN rounding is not assumed identical'),
        sn2_theta=np.array(pair.temporal.theta), sn2_A=pair.temporal.a.detach().cpu().numpy(),
        sn2_b=pair.temporal.b.detach().cpu().numpy(),
        sn2_output_definition=np.array('actual installed LatentPair full: theta_output * (A @ fixedBN1(shared32 preview) + temporal_bias - theta_output >= 0); theta amplitude remains continuous'),
        source_words_layout=np.array('G64,K864,P4 int16 nonnegative10bit; k=((c*3)+kh)*3+kw; bit t; source coordinates output(y+kh-1,x+kw-1); zero outside image'))
    np.savez_compressed(output/'bound_parameters.npz', **arrays)
    return dict(parameters=str(output/'bound_parameters.npz'), source_words_shape=[64, 864, 4],
        source_words_payload_bytes=64*864*4*2, source_theta=pair.temporal.theta,
        source_reconstruction='max |sampled sn2 output - theta*(sampled !=0)| separately reported; nonzero residual invalidates gate-only bounds',
        branch_role='complete norm2_branch is only an answer, not an input to future support-only bounds',
        affine_scope=str(arrays['bn2_affine_definition']))


class Captured(Exception):
    pass


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--train-list', type=Path, required=True)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--bound-context', action='store_true',
        help='also gather sampled Conv2 source neighborhoods and export actual W2/fixed BN2 constants')
    args = parser.parse_args()
    alg = args.root/'algorithm'
    area = alg/'patch_probe'
    latent = area/'factor_completion_20260909/latent_stage_train16'
    args.parent = args.parent or latent/'flow_recovery64/preview_only/shared48_u8_vq5.npz'
    args.output = args.output or area/'residual_consumer_probe'
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(alg/'nrv_cost_probe'))
    sys.path.insert(0, str(alg))
    sys.path.insert(0, str(latent))
    import run_probe as probe
    from run_bn_probe import input_frame
    from adapter import install_latent_factor
    from spikingjelly.activation_based import functional
    raw_names = json.loads(args.train_list.read_text())
    train = raw_names['train'] if isinstance(raw_names, dict) else raw_names
    if len(train) != 16:
        raise ValueError('Expected the explicit approved train16 list, from which only first4 are used.')
    names = train[:4]
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
    required_modules = [PATCH, BLOCK, BLOCK+'.norm2', SOURCE_SN, PROJECT, CONSUMER_SN, PROJECT+'.conv']
    sought = required_modules+[PROJECT+'.conv_res']
    identity = {name: description(modules[name]) if name in modules else None for name in sought}
    run = dict(complete=False, parent=str(args.parent), files=names, train_list=str(args.train_list),
        identity=identity, numeric='dequantized FP32 preview-only LatentPair full, four fixed patch BNs, native projection PSN',
        scope='Source/consumer boundary and sampled numeric opportunity only; no AEE, training, measured skip, cycle or GPU speed claim',
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32, TF32_cudnn=torch.backends.cudnn.allow_tf32,
        rows=[])
    save_json(args.output/'summary.json', run)
    print('BOUNDARY_IDENTITY', json.dumps(identity, ensure_ascii=False), flush=True)
    continuous_projection = modules.get(PROJECT+'.conv_res')
    if any(modules.get(name) is None for name in required_modules) or not (bool(getattr(modules[PROJECT], 'use_MS', False)) or continuous_projection is not None):
        run['stopped'] = 'No verified pre-projection SN route: neither use_MS embedding nor an actual PED continuous residual projection is present.'
        save_json(args.output/'summary.json', run)
        print('STOP', run['stopped'], flush=True)
        return
    if getattr(modules[BLOCK], 'connect_function', None) != 'ADD':
        run['stopped'] = 'The actual residual connection is not additive; identity+branch hypothesis not applied.'
        save_json(args.output/'summary.json', run)
        return
    states = {label: neuron_parameters(modules[name]) for label, name in
        (('r1_sn1', SOURCE_SN), ('proj_sn', CONSUMER_SN))}
    if any(state['temporal_factor_rank'] or state['T'] != 10 for state in states.values()):
        run['stopped'] = 'This explicit full-A reconstruction expects native dense T10 neurons; actual factorized/T identity differs.'
        save_json(args.output/'summary.json', run)
        return
    parameters = {}
    for label, state in states.items():
        parameters.update({label+'_'+key: value.numpy() if torch.is_tensor(value) else np.array(value)
            for key, value in state.items()})
    np.savez_compressed(args.output/'neuron_parameters.npz', **parameters)
    run['neuron_definitions'] = {label: {key: value.tolist() if torch.is_tensor(value) else value
        for key, value in state.items()} for label, state in states.items()}
    print('NEURON_DEFINITIONS', json.dumps({label: {key: value for key, value in state.items()
        if key not in ('A', 'b', 'center')} for label, state in run['neuron_definitions'].items()}, ensure_ascii=False), flush=True)
    pair, originals = install_latent_factor(modules[BLOCK+'.conv1.0'], modules[BLOCK+'.sn2.spiking_neuron'],
        args.parent, conditional=False)
    if args.bound_context:
        run['bound_context'] = save_bound_parameters(modules, fixed, pair, args.output)
    old = torch.load(area/'partial_completion/capture.pt', map_location='cpu', weights_only=False)
    groups, positions = old['groups'].long(), old['positions'].long()
    if continuous_projection is not None:
        conv_res = continuous_projection
        geometry = dict(kernel=tuple(conv_res.kernel_size), stride=tuple(conv_res.stride),
            padding=tuple(conv_res.padding), dilation=tuple(conv_res.dilation))
        anchor_mask = convolution_anchor_mask(positions, (240, 320), **geometry)
        no_anchor_group = ~anchor_mask.any(-1)
        np.savez_compressed(args.output/'projection_parameters.npz',
            conv_res_weight=conv_res.weight.detach().cpu().numpy(),
            conv_res_bias=conv_res.bias.detach().cpu().numpy() if conv_res.bias is not None else np.zeros(conv_res.out_channels, np.float32),
            conv_res_groups=np.array(conv_res.groups), conv_res_kernel_size=np.array(conv_res.kernel_size),
            conv_res_stride=np.array(conv_res.stride), conv_res_padding=np.array(conv_res.padding),
            conv_res_dilation=np.array(conv_res.dilation), anchor_mask=anchor_mask.numpy(),
            positions=positions.numpy(), groups=groups.numpy(),
            anchor_definition=np.array('Input position used by at least one legal conv_res spatial output; before any W-zero refinement. Anchors retain a continuous consumer independently of proj.sn firing.'))
        run['continuous_projection'] = dict(geometry=geometry, parameters=str(args.output/'projection_parameters.npz'),
            sampled_anchor_positions=int(anchor_mask.sum()), sampled_positions=anchor_mask.numel(),
            no_anchor_P4_groups=int(no_anchor_group.sum()), P4_groups=len(groups),
            restriction='proj.sn completion alone cannot cancel Conv2 at positions still consumed by conv_res; oracle coincidence is not permission')
    else:
        anchor_mask = torch.zeros_like(positions, dtype=torch.bool)
        no_anchor_group = torch.ones(len(groups), dtype=torch.bool)
    device = modules[BLOCK+'.conv1.0'].weight.device
    gpu_positions = positions.to(device)
    record, handles, full_ref = {}, [], {}
    def sample(x):
        if tuple(x.shape) != (10, 1, 96, 240, 320):
            raise ValueError('Actual boundary shape differs from original native64 P4: '+str(tuple(x.shape)))
        return x.detach()[:, 0].flatten(2)[:, :, gpu_positions].cpu()
    def before_block(module, inputs):
        record['identity'] = sample(inputs[0])
    def branch(module, inputs, output):
        record['norm2_branch'] = sample(output)
    def after_block(module, inputs, output):
        record['r1out'] = sample(output)
        full_ref['r1out'] = output
    def before_proj(module, inputs):
        record['proj_input'] = sample(inputs[0])
        record['proj_input_same_tensor_r1out'] = inputs[0] is full_ref['r1out']
    def before_conv_res(module, inputs):
        value = inputs[0]
        record['conv_res_input_shape'] = torch.tensor(value.shape, dtype=torch.int32)
        record['conv_res_input'] = sample(value.reshape(10, 1, 96, 240, 320))
        record['conv_res_input_same_storage_r1out'] = value.data_ptr() == full_ref['r1out'].data_ptr()
    def after_conv_res(module, inputs, output):
        record['conv_res_output_shape'] = torch.tensor(output.shape, dtype=torch.int32)
    def before_conv2(module, inputs):
        record.update(sampled_conv2_source(inputs[0], gpu_positions, pair.temporal.theta))
    def before_consumer(module, inputs):
        record['proj_sn_input'] = sample(inputs[0])
        record['proj_sn_input_same_tensor_r1out'] = inputs[0] is full_ref['r1out']
    def after_consumer(module, inputs, output):
        record['proj_sn_output'] = sample(output)
        raise Captured()
    def before_sn1(module, inputs):
        record['r1_sn1_input'] = sample(inputs[0])
    def sn1_output(module, inputs, output):
        record['r1_sn1_output'] = sample(output)
    def observe_proj(h, theta):
        record['proj_native_membrane'] = sample(h.reshape(10, 1, 96, 240, 320))
    def observe_sn1(h, theta):
        record['r1_sn1_native_membrane'] = sample(h.reshape(10, 1, 96, 240, 320))
    for name, hook, pre in (
            (BLOCK, before_block, True), (BLOCK+'.norm2', branch, False),
            (BLOCK, after_block, False), (PROJECT, before_proj, True),
            (CONSUMER_SN, before_consumer, True), (CONSUMER_SN, after_consumer, False),
            (SOURCE_SN, before_sn1, True), (SOURCE_SN, sn1_output, False)):
        handles.append(modules[name].register_forward_pre_hook(hook) if pre else modules[name].register_forward_hook(hook))
    if continuous_projection is not None:
        handles.append(continuous_projection.register_forward_pre_hook(before_conv_res))
        handles.append(continuous_projection.register_forward_hook(after_conv_res))
    if args.bound_context:
        handles.append(modules[BLOCK+'.conv2.0'].register_forward_pre_hook(before_conv2))
    consumer = modules[CONSUMER_SN]
    source_neuron = modules[SOURCE_SN]
    previous_observer = getattr(consumer, '_h9_calibration_observer', None)
    previous_source_observer = getattr(source_neuron, '_h9_calibration_observer', None)
    consumer._h9_calibration_observer = observe_proj
    source_neuron._h9_calibration_observer = observe_sn1
    started = time.monotonic()
    try:
        for index, name in enumerate(names):
            record.clear(); full_ref.clear()
            functional.reset_net(model)
            x, _, _ = input_frame(args.data, name, targets=False)
            try:
                model(x)
            except Captured:
                pass
            else:
                raise RuntimeError('The requested projection-neuron consumer was not reached.')
            full_ref.clear(); del x
            required = ('identity', 'norm2_branch', 'r1out', 'proj_input', 'proj_sn_input', 'proj_sn_output', 'proj_native_membrane', 'r1_sn1_native_membrane')
            if any(key not in record for key in required):
                raise RuntimeError('The actual call order did not provide the requested residual/consumer records.')
            if continuous_projection is not None and 'conv_res_input' not in record:
                raise RuntimeError('PED conv_res was declared but did not consume this r1 output before proj.sn.')
            p = states['proj_sn']
            residual_sum = record['identity']+record['norm2_branch']
            reconstructed = membrane(p, residual_sum)
            reconstructed64 = membrane(p, residual_sum, torch.float64)
            default = fire(p, membrane(p, record['identity']))
            actual = record['proj_sn_output']
            semantic = fire(p, record['proj_native_membrane'])
            checks = dict(identity_plus_branch_max_abs=float((residual_sum-record['r1out']).abs().max()),
                proj_input_vs_r1out_max_abs=float((record['proj_input']-record['r1out']).abs().max()),
                proj_sn_input_vs_r1out_max_abs=float((record['proj_sn_input']-record['r1out']).abs().max()),
                proj_input_same_tensor_r1out=record['proj_input_same_tensor_r1out'],
                proj_sn_input_same_tensor_r1out=record['proj_sn_input_same_tensor_r1out'],
                actual_membrane_output_definition_errors=int(semantic.ne(actual).sum()),
                reconstructed_FP32_membrane_max_abs=float((reconstructed-record['proj_native_membrane']).abs().max()),
                reconstructed_FP32_gate_errors=int(fire(p, reconstructed).ne(actual).sum()),
                reconstructed_FP64_gate_errors=int(fire(p, reconstructed64).ne(actual).sum()),
                r1_sn1_input_vs_identity_max_abs=float((record['r1_sn1_input']-record['identity']).abs().max()),
                r1_sn1_native_membrane_output_definition_errors=int(fire(states['r1_sn1'], record['r1_sn1_native_membrane']).ne(record['r1_sn1_output']).sum()),
                r1_sn1_reconstructed_FP32_membrane_max_abs=float((membrane(states['r1_sn1'], record['identity'])-record['r1_sn1_native_membrane']).abs().max()),
                r1_sn1_reconstructed_FP32_gate_errors=int(fire(states['r1_sn1'], membrane(states['r1_sn1'], record['identity'])).ne(record['r1_sn1_output']).sum()))
            if continuous_projection is not None:
                checks.update(conv_res_input_vs_r1out_max_abs=float((record['conv_res_input']-record['r1out']).abs().max()),
                    conv_res_input_same_storage_r1out=record['conv_res_input_same_storage_r1out'],
                    conv_res_input_shape=record['conv_res_input_shape'].tolist(),
                    conv_res_output_shape=record['conv_res_output_shape'].tolist())
            if args.bound_context:
                checks['conv2_source_theta_g_max_abs'] = float(record['conv2_source_theta_g_max_abs'])
            filename = args.output/(f'{index:02d}_'+Path(name).stem+'.npz')
            np.savez_compressed(filename, **{key: value.numpy() for key, value in record.items() if torch.is_tensor(value)},
                identity_only_default_output=default.numpy(), groups=groups.numpy(), positions=positions.numpy(),
                anchor_mask=anchor_mask.numpy(), no_anchor_P4_group=no_anchor_group.numpy(),
                frame_name=np.array(name), split=np.array('train'), layout=np.array('T10,C96,G64,P4; horizontal native P4'))
            row = dict(file=name, capture=str(filename), checks=checks, identity_only=comparison(default, actual))
            row['identity_only_no_anchor_P4'] = comparison(default[:, :, no_anchor_group], actual[:, :, no_anchor_group]) if no_anchor_group.any() else dict(P4_H8_T10_groups=0, reason='No sampled P4 group is free of the continuous conv_res dependency')
            run['rows'].append(row)
            save_json(args.output/'summary.json', run)
            print('RESIDUAL_CONSUMER', index+1, json.dumps(row, ensure_ascii=False), flush=True)
            if checks['identity_plus_branch_max_abs'] or checks['proj_input_vs_r1out_max_abs'] or checks['proj_sn_input_vs_r1out_max_abs'] or checks['actual_membrane_output_definition_errors'] or checks.get('conv_res_input_vs_r1out_max_abs', 0):
                raise RuntimeError('Actual boundary/output semantics differ; stop the identity+branch hypothesis at this boundary.')
        metrics = [row['identity_only'] for row in run['rows']]
        totals = {key: sum(row[key] for row in metrics) for key in (
            'gates', 'different_gates', 'actual_nonzero', 'default_nonzero', 'lost_nonzero', 'extra_nonzero',
            'P4_H8_T10_groups', 'P4_H8_T10_equal_groups')}
        totals.update(different_gate_fraction=totals['different_gates']/totals['gates'],
            lost_fraction_of_actual_nonzero=totals['lost_nonzero']/max(totals['actual_nonzero'], 1),
            P4_H8_T10_equal_fraction=totals['P4_H8_T10_equal_groups']/totals['P4_H8_T10_groups'])
        if no_anchor_group.any():
            eligible = [row['identity_only_no_anchor_P4'] for row in run['rows']]
            subtotal = {key: sum(row[key] for row in eligible) for key in (
                'gates', 'different_gates', 'actual_nonzero', 'default_nonzero', 'lost_nonzero', 'extra_nonzero',
                'P4_H8_T10_groups', 'P4_H8_T10_equal_groups')}
            subtotal.update(different_gate_fraction=subtotal['different_gates']/subtotal['gates'],
                lost_fraction_of_actual_nonzero=subtotal['lost_nonzero']/max(subtotal['actual_nonzero'], 1),
                P4_H8_T10_equal_fraction=subtotal['P4_H8_T10_equal_groups']/subtotal['P4_H8_T10_groups'])
            run['aggregate_no_anchor_P4'] = subtotal
        run.update(complete=True, wall_seconds=time.monotonic()-started,
            aggregate=totals,
            interpretation='Identity-only is a fixed default gate, not a predictor or an oracle skip permission. It removes the entire normalized branch; zero Conv2 weights would still retain norm2(0). Actual PED conv_res continuously consumes its spatial anchor inputs before proj.sn, so even a proven SN gate does not release those producers. Only64 sampled P4 per frame, not a whole-image skip fraction.',
            numeric_check='Native full-width CUDA membrane/output is authoritative; sampled CPU FP32/FP64 reconstructions are separately counted and do not establish global reassociation bit equality.')
        save_json(args.output/'summary.json', run)
    finally:
        consumer._h9_calibration_observer = previous_observer
        source_neuron._h9_calibration_observer = previous_source_observer
        for handle in handles:
            handle.remove()
        modules[BLOCK+'.conv1.0'].forward = originals['conv1_forward']
        modules[BLOCK+'.sn2.spiking_neuron'].forward = originals['neuron_forward']
    print('DONE', str(args.output/'summary.json'), flush=True)


if __name__ == '__main__':
    main()
