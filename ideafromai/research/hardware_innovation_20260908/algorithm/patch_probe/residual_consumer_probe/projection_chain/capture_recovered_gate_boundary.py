"""Sample the actual recovered r1 -> PED neuron boundary; root launches CUDA.

Use the verified recovery-final TrainableLatentPair forward and exported
RecoveryControl W/L/biases. The first four entries of the original train16 and
the original64 native P4 positions are fixed. No training or whole-frame
capture. Stop only after native proj.sn has produced its actual theta*g.

The common parent deletes r1's entire normalized branch outside even/even
anchors. Save both the unmasked BN answer and the actual masked branch. W2
comes from effective_weight('r1'), and BN2 offset is recomputed from the live,
restored bias. Registered conv.weight and old exported bn_offset are reference
values, not the recovered function. These samples support a later CPU bound
probe; they do not establish exact early termination of the current FP32 graph.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from evaluate_recovered_consumer import load_recovered
from train_consumer_recovery import RecoveryControl, TrainableLatentPair, read_arrays, read_train16
from capture_chain import R1, PROJECT, SPECS, canonical, sample, save_json

CONSUMER_SN = PROJECT+'.sn.spiking_neuron'
AXES = ('ordinary_rank32', 'independent_sparse_L')


def as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


@torch.no_grad()
def bound_arrays(controller, pair):
    """Actual effective Conv2 and restored fixed BN, before spatial deletion."""
    conv, bn = controller.conv['r1'], controller.bn['r1']
    gamma, beta, mean, var = [as_numpy(getattr(bn, name)).copy() for name in
        ('weight', 'bias', 'running_mean', 'running_var')]
    gain = gamma.astype(np.float64)/np.sqrt(var.astype(np.float64)+float(bn.eps))
    offset = beta.astype(np.float64)-gain*mean.astype(np.float64)
    return dict(W2=as_numpy(controller.effective_weight('r1')).copy(),
        conv2_bias=(as_numpy(conv.bias).copy() if conv.bias is not None
                    else np.zeros(conv.out_channels, np.float32)),
        conv2_has_bias=np.array(conv.bias is not None),
        conv2_kernel_size=np.array(conv.kernel_size), conv2_stride=np.array(conv.stride),
        conv2_padding=np.array(conv.padding), conv2_dilation=np.array(conv.dilation),
        conv2_groups=np.array(conv.groups), bn2_name=np.array(SPECS['r1']['norm']),
        bn2_gamma=gamma, bn2_beta=beta, bn2_running_mean=mean, bn2_running_var=var,
        bn2_eps=np.array(bn.eps), bn2_gain=gain, bn2_offset=offset,
        bn2_track_running_stats=np.array(bn.track_running_stats),
        bn2_affine_definition=np.array('NumPy Float64 gain/offset from live restored FP32 gamma/beta/mean/var and eps; native CUDA FP32 BN rounding is not claimed identical'),
        W2_definition=np.array("RecoveryControl.effective_weight('r1'), including recovered W_shadow; not registered conv.weight or original_W"),
        r1_selected_source_mask=np.asarray(controller.arrays['r1_selected_source_mask'], bool).copy(),
        selected_source_mask_definition=np.array('Fixed selected source channels for the pruning comparison. Independent L has these columns masked, but its gate-producing effective W2 generally remains nonzero; this mask is not a W2-zero or gate permission.'),
        branch_domain=np.array('raw Conv2 and pre-delete BN are evaluated by the reference forward on all positions; actual r1 residual branch equals BN output at even/even anchors and zero elsewhere, including its BN offset'),
        sn2_theta=np.array(pair.temporal.theta), sn2_A=as_numpy(pair.temporal.a).copy(),
        sn2_b=as_numpy(pair.temporal.b).copy(),
        sn2_output_definition=np.array('Installed recovery-final TrainableLatentPair, frozen U/V, complete T10 theta-valued hard forward under torch.no_grad'),
        source_words_layout=np.array('G64,K864,P4 int16 nonnegative10bit; k=((c*3)+kh)*3+kw; bit=t; padded3x3 source neighborhood around original output position'))


@torch.no_grad()
def projection_arrays(controller, positions, groups, anchor_mask, model_file):
    conv = controller.projection
    result = dict(U=as_numpy(controller.first[:, :, 0, 0]).copy(),
        V=as_numpy(controller.second[:, :, 0, 0]).copy(),
        projection_base_has_bias=np.array(conv.bias is not None),
        projection_base_bias=(as_numpy(conv.bias).copy() if conv.bias is not None
                              else np.zeros(conv.out_channels, np.float32)),
        projection_bias_delta=as_numpy(controller.projection_bias_delta).copy(),
        conv_res_kernel_size=np.array(conv.kernel_size), conv_res_stride=np.array(conv.stride),
        conv_res_padding=np.array(conv.padding), conv_res_dilation=np.array(conv.dilation),
        conv_res_groups=np.array(conv.groups), positions=as_numpy(positions), groups=as_numpy(groups),
        anchor_mask=as_numpy(anchor_mask), model_file=np.array(str(model_file)),
        registered_conv_res_weight_is_reference_only=np.array(True),
        anchor_definition=np.array('Spatial inputs to the ordinary PED1x1/stride2 projection; actual r1 normalized branch is retained only here. Independent L also consumes source neighborhoods through its explicit corrected continuous path.'),
        forward_definition=np.array('V(U(r1out at anchors)) with original base-bias presence, then separate output delta; independent_sparse_L additionally subtracts actual r0/r1 P*W source terms and adds their independent L terms before V'))
    for label in ('r0', 'r1'):
        result[label+'_P'] = as_numpy(controller.p[label]).copy()
        if label in controller.l_shadow:
            result[label+'_L'] = as_numpy(controller.effective_l(label)).copy()
    return result


def sampled_domain_counts(words, anchor_mask):
    """Neighborhood occurrences, explicitly excluding deleted target positions."""
    lookup = np.array([int(i).bit_count() for i in range(1024)], np.int16)
    active = lookup[as_numpy(words)]
    anchors = as_numpy(anchor_mask).astype(bool)
    count_all = int(active.sum())
    count_anchor = int((active*anchors[:, None, :]).sum())
    return dict(source_nonzero_occurrences_all_sampled_output_neighborhoods=count_all,
        source_nonzero_occurrences_anchor_output_neighborhoods=count_anchor,
        source_nonzero_occurrences_deleted_nonanchor_output_neighborhoods=count_all-count_anchor,
        scope='T10 source occurrences with halo/P4 repeats, before W-zero filtering; neither physical reads nor newly cancellable production. Nonanchor r1 branch deletion already belongs to this parent.')


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--train-list', type=Path, required=True)
    parser.add_argument('--model-files', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    alg = args.root/'algorithm'
    area = alg/'patch_probe'
    residual = area/'residual_consumer_probe'
    chain = residual/'projection_chain'
    latent = area/'factor_completion_20260909/latent_stage_train16'
    args.output = args.output or chain/'recovered_gate_boundary_train4'
    args.output.mkdir(parents=True, exist_ok=True)
    paths = args.model_files or [chain/'consumer_obs_recovery64'/(axis+'.npz') for axis in AXES]
    names = read_train16(args.train_list)[:4]
    for path in (alg, alg/'nrv_cost_probe', latent, residual):
        sys.path.insert(0, str(path))
    import run_probe as probe
    from run_bn_probe import input_frame
    from evaluate_branch_control import mask_nonanchors
    from capture import Captured, convolution_anchor_mask, description, fire, neuron_parameters, sampled_conv2_source
    from spikingjelly.activation_based import functional

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
    conv1, sn2 = modules[R1+'.conv1.0'], modules[R1+'.sn2.spiking_neuron']
    original_conv1, original_sn2 = conv1.forward, sn2.forward
    pair = TrainableLatentPair(read_arrays(parent), conv1.weight.device)
    pair.u.requires_grad_(False)
    pair.v.requires_grad_(False)
    conv1.forward, sn2.forward = pair.conv_forward, pair.neuron_forward
    masks = {}
    def delete_nonanchor(module, inputs, output):
        key = (output.shape[-2:], output.device)
        if key not in masks:
            mask = torch.zeros(output.shape[-2:], device=output.device, dtype=torch.bool)
            mask[::2, ::2] = True
            masks[key] = mask
        return mask_nonanchors(output, masks[key])
    # Register before the capture hook so norm2_branch is the actual output.
    common_hook = modules[R1+'.norm2'].register_forward_hook(delete_nonanchor)
    old = torch.load(area/'partial_completion/capture.pt', map_location='cpu', weights_only=False)
    groups, positions = old['groups'].long(), old['positions'].long()
    del old
    gpu_positions = positions.to(conv1.weight.device)
    projection, consumer = modules[PROJECT+'.conv_res'], modules[CONSUMER_SN]
    geometry = dict(kernel=tuple(projection.kernel_size), stride=tuple(projection.stride),
        padding=tuple(projection.padding), dilation=tuple(projection.dilation))
    anchor_mask = convolution_anchor_mask(positions, (240, 320), **geometry)
    no_anchor_group = ~anchor_mask.any(-1)
    state = neuron_parameters(consumer)
    previous_observer = getattr(consumer, '_h9_calibration_observer', None)
    run = dict(complete=False, parent=str(parent), files=names, train_list=str(args.train_list),
        model_files=[str(path) for path in paths], positions_source=str(area/'partial_completion/capture.pt'),
        layout='T10,C96,G64,P4 real-value samples; source words G64,K864,P4 with bit=t',
        common_parent='Verified recovery-final frozen TrainableLatentPair, four fixed patch BNs, recovered W/L and biases, r1 entire normalized-branch deletion on nonanchors, actual PED and native projection neuron.',
        modules={name: description(modules[name]) for name in (R1, R1+'.norm2', PROJECT, CONSUMER_SN)},
        proj_sn=dict(T=state['T'], A_nonzero=int(torch.count_nonzero(state['A'])),
            A_rank=int(torch.linalg.matrix_rank(state['A'].double())),
            theta=as_numpy(state['theta']).tolist(), output_mode=state['output_mode'],
            threshold_mode=state['threshold_mode'], center_mode=state['center_mode'],
            temporal_factor_rank=state['temporal_factor_rank']),
        anchor_geometry=geometry, sampled_anchor_positions=int(anchor_mask.sum()),
        sampled_positions=int(anchor_mask.numel()),
        scope='Four fixed training frames per actual exported student; sampled numerical boundary only. No AEE, training, full-frame tensor capture, skip permission, current-FP32 exactness or hardware timing result.',
        axes={})
    save_json(args.output/'summary.json', run)
    controller = None
    handles = []
    started = time.monotonic()
    try:
        for path in paths:
            arrays = read_arrays(path)
            runtime = json.loads((path.parent/'result.json').read_text())
            torch.backends.cuda.matmul.allow_tf32 = bool(runtime['TF32_matmul'])
            torch.backends.cudnn.allow_tf32 = bool(runtime['TF32_cudnn'])
            controller = RecoveryControl(modules, arrays)
            loaded = load_recovered(controller, arrays)
            axis = loaded['axis']
            directory = args.output/axis
            directory.mkdir(parents=True, exist_ok=True)
            bound = bound_arrays(controller, pair)
            np.savez_compressed(directory/'bound_parameters.npz', **bound)
            np.savez_compressed(directory/'neuron_parameters.npz',
                **{'proj_sn_'+key: as_numpy(value) for key, value in state.items()})
            np.savez_compressed(directory/'projection_parameters.npz',
                **projection_arrays(controller, positions, groups, anchor_mask, path))
            row = dict(model_file=str(path), loaded=loaded,
                TF32_matmul=torch.backends.cuda.matmul.allow_tf32,
                TF32_cudnn=torch.backends.cudnn.allow_tf32,
                bound_parameters=str(directory/'bound_parameters.npz'),
                W2_effective_vs_registered_reference_max_abs=float(np.abs(bound['W2']-as_numpy(controller.conv['r1'].weight)).max()),
                BN2_beta_vs_original_reference_max_abs=float(np.abs(bound['bn2_beta']-arrays['r1_bn_beta']).max()),
                BN2_offset_vs_stale_export_reference_max_abs=float(np.abs(bound['bn2_offset']-arrays['r1_bn_offset']).max()),
                branch_domain=str(bound['branch_domain']), frames=[])
            run['axes'][axis] = row
            save_json(args.output/'summary.json', run)
            record = {}
            shape = {}
            def before_block(module, inputs):
                record['identity'] = sample(inputs[0], gpu_positions)
            def before_conv2(module, inputs):
                record['conv2_source_shape'] = np.array(inputs[0].shape)
                record.update({key: as_numpy(value) for key, value in
                    sampled_conv2_source(canonical(inputs[0]), gpu_positions, pair.temporal.theta).items()})
            def after_conv2(module, inputs, output):
                record['conv2_raw'] = sample(output, gpu_positions)
            def before_delete(module, inputs, output):
                record['norm2_before_spatial_delete'] = sample(output, gpu_positions)
            def after_delete(module, inputs, output):
                record['norm2_branch'] = sample(output, gpu_positions)
            def after_block(module, inputs, output):
                record['r1out'] = sample(output, gpu_positions)
            def before_consumer(module, inputs):
                shape['consumer'] = tuple(inputs[0].shape)
                record['proj_sn_input'] = sample(inputs[0], gpu_positions)
            def observe_membrane(h, theta):
                record['proj_native_membrane'] = sample(h.reshape(shape['consumer']), gpu_positions)
                record['proj_observed_theta'] = as_numpy(theta).copy()
            def after_consumer(module, inputs, output):
                record['proj_sn_output'] = sample(output, gpu_positions)
                record['proj_gate'] = record['proj_sn_output'] != 0
                raise Captured()
            for module, hook, pre in (
                    (modules[R1], before_block, True),
                    (controller.conv['r1'], before_conv2, True),
                    (controller.conv['r1'], after_conv2, False),
                    (controller.bn['r1'], before_delete, False),
                    (modules[R1+'.norm2'], after_delete, False),
                    (modules[R1], after_block, False),
                    (consumer, before_consumer, True),
                    (consumer, after_consumer, False)):
                handles.append(module.register_forward_pre_hook(hook) if pre else module.register_forward_hook(hook))
            consumer._h9_calibration_observer = observe_membrane
            for index, name in enumerate(names):
                functional.reset_net(model)
                controller.terms.clear()
                record.clear()
                shape.clear()
                x, _, _ = input_frame(args.data, name, targets=False)
                try:
                    model(x)
                except Captured:
                    pass
                finally:
                    controller.terms.clear()
                del x
                native = torch.from_numpy(record['proj_native_membrane'])
                expected_gate = fire(state, native).numpy()
                masked_branch = np.where(anchor_mask.numpy()[None, None],
                                         record['norm2_before_spatial_delete'], 0)
                checks = dict(identity_plus_actual_branch_max_abs=float(np.abs(
                        record['identity']+record['norm2_branch']-record['r1out']).max()),
                    spatial_delete_definition_max_abs=float(np.abs(masked_branch-record['norm2_branch']).max()),
                    proj_sn_input_vs_r1out_max_abs=float(np.abs(record['proj_sn_input']-record['r1out']).max()),
                    native_membrane_to_theta_g_differences=int(np.count_nonzero(expected_gate != record['proj_sn_output'])),
                    source_theta_g_max_abs=float(record['conv2_source_theta_g_max_abs']))
                filename = directory/(f'{index:02d}_'+Path(name).stem+'.npz')
                np.savez_compressed(filename, **record, frame_name=np.array(name), axis=np.array(axis),
                    split=np.array('train'), positions=positions.numpy(), groups=groups.numpy(),
                    anchor_mask=anchor_mask.numpy(), no_anchor_P4_group=no_anchor_group.numpy(),
                    layout=np.array(run['layout']))
                frame = dict(file=name, capture=str(filename), checks=checks,
                    source_domain=sampled_domain_counts(record['conv2_source_gate_words'], anchor_mask),
                    actual_output_nonzero=int(record['proj_gate'].sum()),
                    actual_output_nonzero_at_anchors=int((record['proj_gate']*anchor_mask.numpy()[None, None]).sum()),
                    sample_payload_bytes=int(sum(np.asarray(value).nbytes for value in record.values())))
                row['frames'].append(frame)
                save_json(args.output/'summary.json', run)
                print('RECOVERED_GATE_BOUNDARY', axis, index+1, json.dumps(frame, ensure_ascii=False), flush=True)
            consumer._h9_calibration_observer = previous_observer
            for handle in handles:
                handle.remove()
            handles.clear()
            controller.restore()
            controller = None
            record.clear()
            functional.reset_net(model)
        run.update(complete=True, wall_seconds=time.monotonic()-started)
        save_json(args.output/'summary.json', run)
    finally:
        consumer._h9_calibration_observer = previous_observer
        for handle in handles:
            handle.remove()
        if controller is not None:
            controller.restore()
        common_hook.remove()
        conv1.forward, sn2.forward = original_conv1, original_sn2
        current.pop('flow', None)
    print('DONE', str(args.output/'summary.json'), flush=True)


if __name__ == '__main__':
    main()
