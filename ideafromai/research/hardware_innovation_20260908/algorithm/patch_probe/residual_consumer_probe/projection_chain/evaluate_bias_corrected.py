"""Sequential empirical bias correction for the fixed consumer-pruning controls.

This is the data-based mean-error correction of Nagel et al., ICCV2019,
section4.2 / AppendixD (https://arxiv.org/abs/1906.04721), adapted to the
already fixed pruning students. It is not the complete data-free DFQ method.
Use only the original train16 list's first four full frames. Correct r0 BN2
over all T/spatial positions, then r1 BN2 over actual PED anchors, then the
rank32 conv_res output. Update each bias once; gains, running mean/variance,
U/V, masks, theta and temporal neuron parameters remain unchanged.

Root launches CUDA. This file adds no training, search, 825 evaluation or
performance claim. Calibration exits before the downstream network; valid10
uses the original real successors and AEE implementation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from capture_chain import R0, R1, PROJECT, SPECS, save_json
from evaluate_consumer_pruning import AXES, ConsumerControl


STAGES = ('r0', 'r1', 'projection')
DOMAINS = {
    'r0': 'native[T10,B1,C96,H,W], all time and all spatial positions',
    'r1': 'native[T10,B1,C96,H,W], all time, even/even PED anchors only',
    'projection': 'native[T10,C96,H/2,W/2], all time and output positions',
}


def channel_sum_and_count(output, stage):
    """Reduce only the declared physical domain, retaining native channel order.

    C8 chunks avoid a complete Float64 copy of a C96 activation. A single
    count applies to each channel; frame totals are weighted by this count.
    No input/output tensor is modified.
    """
    value = output.detach()
    native_shape = list(value.shape)
    if stage in ('r0', 'r1'):
        if value.ndim != 5 or tuple(value.shape[:3]) != (10, 1, 96):
            raise ValueError('Expected native T10/B1/C96 norm2 output, got '+str(native_shape))
        if stage == 'r1':
            value = value[..., ::2, ::2]
        channel_axis, dimensions = 2, (0, 1, 3, 4)
    elif stage == 'projection':
        if value.ndim != 4 or tuple(value.shape[:2]) != (10, 96):
            raise ValueError('Expected native T10/C96 projection output, got '+str(native_shape))
        channel_axis, dimensions = 1, (0, 2, 3)
    else:
        raise ValueError(stage)
    pieces = []
    for channel in range(0, 96, 8):
        part = value.narrow(channel_axis, channel, 8)
        pieces.append(part.sum(dim=dimensions, dtype=torch.float64).cpu().numpy())
    return np.concatenate(pieces), int(value.numel()//96), native_shape, list(value.shape)


class CalibrationReached(Exception):
    pass


@torch.no_grad()
def collect_means(args, model, modules, current, controller, names, stages, tag):
    """Each frame stops just after the last requested stage; discard tail state."""
    from run_bn_probe import input_frame
    from spikingjelly.activation_based import functional

    nodes = {'r0': modules[R0+'.norm2'], 'r1': modules[R1+'.norm2'],
             'projection': modules[PROJECT+'.conv_res']}
    rows = {stage: [] for stage in stages}
    frame = {}
    handles = []
    for stage in stages:
        def observe(module, inputs, output, stage=stage):
            total, count, native, selected = channel_sum_and_count(output, stage)
            rows[stage].append(dict(file=frame['name'], sum=total,
                count_per_channel=count, mean=total/count,
                native_shape=native, selected_shape=selected))
            if stage == stages[-1]:
                raise CalibrationReached()
        # The common r1 nonanchor deletion hook was installed first. We still
        # select anchors explicitly; zeros outside that domain are not averaged.
        handles.append(nodes[stage].register_forward_hook(observe))
    controller.counts = {label: [] for label in ('r0', 'r1')}
    started = time.monotonic()
    try:
        for index, name in enumerate(names):
            functional.reset_net(model)
            current.pop('flow', None)
            controller.terms.clear()
            frame['name'] = name
            x, _, _ = input_frame(args.data, name, targets=False)
            try:
                try:
                    model(x)
                except CalibrationReached:
                    pass
                else:
                    raise RuntimeError('The requested calibration boundary was not reached.')
            finally:
                # A norm2 early exit can leave one or both projected-K terms.
                # They must never survive into the next frame or axis.
                controller.terms.clear()
                current.pop('flow', None)
                del x
            print('BIAS_CALIBRATION', tag, index+1, len(names), flush=True)
    finally:
        for handle in handles:
            handle.remove()
        controller.terms.clear()
        controller.counts = {label: [] for label in ('r0', 'r1')}
    result = {}
    for stage, items in rows.items():
        if len(items) != len(names):
            raise RuntimeError('Expected one calibration observation per frame at '+stage)
        count = sum(item['count_per_channel'] for item in items)
        total = np.sum([item['sum'] for item in items], axis=0)
        result[stage] = dict(domain=DOMAINS[stage], mean=total/count,
            sum=total, count_per_channel=count, frames=items,
            sum_dtype='Float64 reduction of actual nativeFP32 outputs',
            wall_seconds=time.monotonic()-started)
    return result


class BiasState:
    """Restore each student's common bias parent; never touch gain/variance."""
    def __init__(self, modules):
        self.bn = {label: modules[SPECS[label]['norm']] for label in ('r0', 'r1')}
        self.original_bn_bias = {label: bn.bias.detach().clone() for label, bn in self.bn.items()}
        self.projection = modules[PROJECT+'.conv_res']
        self.original_projection_parameter = self.projection.bias
        self.original_projection_value = (None if self.projection.bias is None
                                           else self.projection.bias.detach().clone())

    @torch.no_grad()
    def restore(self):
        for label, bn in self.bn.items():
            bn.bias.copy_(self.original_bn_bias[label])
        self.projection.bias = self.original_projection_parameter
        if self.original_projection_value is not None:
            self.projection.bias.copy_(self.original_projection_value)

    @torch.no_grad()
    def apply(self, stage, teacher_mean, current_mean):
        requested = np.asarray(teacher_mean)-np.asarray(current_mean)
        if stage == 'projection':
            # Keep the original bias-free convolution for an exactly zero
            # ordinary correction, rather than introduce a zero-bias operator.
            if self.projection.bias is None and np.any(requested):
                self.projection.bias = torch.nn.Parameter(
                    torch.zeros(96, device=self.projection.weight.device,
                                dtype=self.projection.weight.dtype), requires_grad=False)
            parameter = self.projection.bias
        else:
            parameter = self.bn[stage].bias
        if parameter is None:
            before = after = np.zeros(96, np.float32)
        else:
            before = parameter.detach().clone()
            parameter.add_(torch.as_tensor(requested, device=parameter.device, dtype=parameter.dtype))
            after = parameter.detach().cpu().numpy().copy()
            before = before.cpu().numpy()
        applied = after.astype(np.float64)-before.astype(np.float64)
        return dict(stage=stage, domain=DOMAINS[stage], teacher_mean=teacher_mean,
            current_mean_before=current_mean, requested_delta=requested,
            applied_parameter_delta=applied, bias_before=before, bias_after=after,
            predicted_mean_after_real_arithmetic=np.asarray(current_mean)+applied,
            predicted_mean_scope='Current mean plus actualFP32 bias delta, not a second native-forward measurement',
            requested_delta_max_abs=float(np.max(np.abs(requested))),
            applied_delta_max_abs=float(np.max(np.abs(applied))))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parameters', type=Path)
    parser.add_argument('--train-list', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--axes', nargs='+', choices=AXES, default=list(AXES))
    args = parser.parse_args()
    args.split = 'diverse'
    alg = args.root/'algorithm'
    area = alg/'patch_probe'
    residual = area/'residual_consumer_probe'
    chain = residual/'projection_chain'
    latent = area/'factor_completion_20260909/latent_stage_train16'
    args.parameters = args.parameters or chain/'consumer_pruning_diverse10/parameters.npz'
    args.output = args.output or chain/'consumer_bias_corrected_diverse10'
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.parameters) as z:
        arrays = {key: z[key].copy() for key in z.files}
    train = [str(name) for name in arrays['selected_train_frames']]
    if len(train) != 4:
        raise ValueError('Use the existing parameters with the original train16 first four frames.')
    if args.train_list is not None:
        declared = json.loads(args.train_list.read_text())
        declared = declared['train'] if isinstance(declared, dict) else declared
        if len(declared) != 16 or declared[:4] != train:
            raise ValueError('The explicit train16 list does not match the fixed first four frames.')
    names = json.loads((alg/'samples.json').read_text())['valid'][:10]
    if set(train).intersection(names):
        raise ValueError('Calibration and diverse validation frames overlap.')
    for path in (alg, alg/'nrv_cost_probe', latent, residual):
        sys.path.insert(0, str(path))
    import run_probe as probe
    from adapter import install_latent_factor
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
    conv1, sn2 = modules[R1+'.conv1.0'], modules[R1+'.sn2.spiking_neuron']
    _, originals = install_latent_factor(conv1, sn2, parent, conditional=False)
    projection = modules[PROJECT+'.conv_res']
    if not np.array_equal(projection.weight.detach().cpu().numpy()[:, :, 0, 0].astype(np.float64), arrays['C']):
        raise ValueError('Live original C differs from the fixed consumer-pruning parent.')
    if (tuple(projection.kernel_size), tuple(projection.stride), tuple(projection.padding)) != ((1, 1), (2, 2), (0, 0)):
        raise ValueError('Actual PED continuous projection geometry changed.')
    masks = {}
    def delete_nonanchor(module, inputs, output):
        key = (output.shape[-2:], output.device)
        if key not in masks:
            mask = torch.zeros(output.shape[-2:], device=output.device, dtype=torch.bool)
            mask[::2, ::2] = True
            masks[key] = mask
        return mask_nonanchors(output, masks[key])
    common_hook = modules[R1+'.norm2'].register_forward_hook(delete_nonanchor)
    controller = None
    bias_state = None
    run = dict(complete=False, files=names, split='diverse', count=10,
        calibration_files=train, calibration_count=4,
        calibration_source='Original captured train16 first four identities; new full-frame native forwards, no validation statistics or labels.',
        parent=str(parent), parameters=str(args.parameters), axes=list(args.axes),
        common_parent='Same ordinary rank32, R32 U8/VQ5 preview-only, four fixed patchBNs, r1 nonanchor normalized-branch deletion.',
        method='One sequential empirical mean correction per r0 BN2, r1 BN2 and conv_res, followed by unchanged real-network diverse10.',
        prior=dict(title='Data-Free Quantization Through Weight Equalization and Bias Correction',
            venue='ICCV2019', sections='4.2 and AppendixD, empirical correction only',
            url='https://arxiv.org/abs/1906.04721',
            scope='Data-based ordinary control for pruning; not full data-free DFQ or a new mechanism.'),
        unchanged='BN weights/gains, running means/variances, epsilon, source-group masks, rank32U/V, theta and PSN parameters.',
        projection_only='Receives the same three correction stages; r0/r1 should have zero deltas because their gate-producing W is unchanged.',
        ordinary='Runs the same three-stage calibration as a zero-delta sanity control, then the same diverse10 entrypoint.',
        metric='Unchanged evaluate_branch_control.evaluate_axis: real preds.2, same bilinear480x640 and existing valid GT pixels.',
        numeric='NativeFP32 student withFloat64 observed means andFP32 bias updates; no bit-exact recovery or integer equivalence claim.',
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32,
        TF32_cudnn=torch.backends.cudnn.allow_tf32,
        claim='Calibration/accuracy and source-activity controls only; no training, search, 825 or speed result.',
        calibration={}, results={})
    save_json(args.output/'run.json', run)
    try:
        controller = ConsumerControl(modules, arrays)
        bias_state = BiasState(modules)
        controller.begin('ordinary_rank32')
        teacher = collect_means(args, model, modules, current, controller, train,
                                STAGES, 'teacher')
        save_json(args.output/'teacher_means.json', teacher)
        run['teacher_means_file'] = 'teacher_means.json'
        save_json(args.output/'run.json', run)
        for axis in args.axes:
            bias_state.restore()
            controller.begin(axis)
            record = dict(complete=False, stages=[])
            correction_arrays = {}
            for stage in STAGES:
                # The last pass also records r0/r1 after their bias changes.
                # It still makes only the prescribed four forwards to projection.
                observed_stages = STAGES if stage == 'projection' else (stage,)
                measured = collect_means(args, model, modules, current, controller,
                    train, observed_stages, axis+'/'+stage)
                delta = bias_state.apply(stage, teacher[stage]['mean'], measured[stage]['mean'])
                delta['measurement'] = measured[stage]
                if stage == 'projection':
                    delta['earlier_stages_after_correction'] = {name: measured[name] for name in ('r0', 'r1')}
                record['stages'].append(delta)
                correction_arrays[stage+'_bias'] = delta['bias_after']
                correction_arrays[stage+'_delta'] = delta['applied_parameter_delta']
                save_json(args.output/(axis+'_calibration.json'), record)
            record['complete'] = True
            save_json(args.output/(axis+'_calibration.json'), record)
            np.savez_compressed(args.output/(axis+'_biases.npz'),
                                **correction_arrays, calibration_files=np.asarray(train))
            run['calibration'][axis] = dict(file=axis+'_calibration.json',
                biases=axis+'_biases.npz',
                delta_max_abs={row['stage']: row['applied_delta_max_abs'] for row in record['stages']})
            save_json(args.output/'run.json', run)
            controller.terms.clear()
            controller.counts = {label: [] for label in ('r0', 'r1')}
            result = evaluate_axis(args, model, current, names, axis,
                                   progress_tag='BIAS_CORRECTED_AEE')
            save_json(args.output/(axis+'_source_activity.json'), controller.source_report(names))
            result['actual_source_activity_file'] = axis+'_source_activity.json'
            result['biases_file'] = axis+'_biases.npz'
            run['results'][axis] = result
            save_json(args.output/'run.json', run)
        run['complete'] = True
        save_json(args.output/'run.json', run)
    finally:
        if bias_state is not None:
            bias_state.restore()
        if controller is not None:
            controller.restore()
        common_hook.remove()
        conv1.forward, sn2.forward = originals['conv1_forward'], originals['neuron_forward']
    print('DONE', json.dumps(run['results']), flush=True)


if __name__ == '__main__':
    main()
