"""Ordered three-source capture for the two existing fixed raw students.

One model load, no training or new quantization. Actual fixed helpers and the
existing coarse/AEE evaluator are reused. Packing order is T,C,y,x, little-bit
within each byte; original spatial coordinates are retained. A small native
I24 block is copied from the helper's actual retained input, not requantized.
This trace does not prescribe a hardware schedule or imply GPU speed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
LIFT = HERE.parent
CHAIN = LIFT.parent
RES = CHAIN.parent
PATCH = RES.parent
BASE = PATCH.parent.parent
LATENT = PATCH/'factor_completion_20260909/latent_stage_train16'
for directory in (CHAIN, RES, LATENT, BASE/'algorithm', BASE/'algorithm/nrv_cost_probe'):
    sys.path.insert(0, str(directory))

AXES = ('ordinary', 'lifting_raw')
LABELS = ('sn1', 'sn2', 'proj')
COUNT_KEYS = dict(sn1='sn1_preview_conv1', sn2='sn2_anchor_conv2', proj='proj_sn_spike_conv')


def save_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert)+'\n')


def unpack_gate(data, label):
    shape = tuple(int(x) for x in data[label+'_shape'])
    return np.unpackbits(data[label+'_gate_bits'], axis=1, bitorder='little',
                         count=int(np.prod(shape[1:]))).reshape(shape).astype(bool)


def source_block_reference(data, parameters):
    """Independent int64 execution on the actual small retained I24 block."""
    def write24(n, shift):
        divisor = 1 << int(shift)
        q, r = np.divmod(n, divisor)
        q += (2*r > divisor) | ((2*r == divisor) & ((q & 1) != 0))
        return np.clip(q, -(1 << 23), (1 << 23)-1)
    shape = data['I24_halo'].shape
    identity = data['I24_halo'].astype(np.int64).reshape(10, -1)
    if 'As_q16' in parameters:
        value = write24(parameters['As_q16'].astype(np.int64) @ identity,
                        parameters['As_exponent'])
    else:
        value = identity.copy()
        q12 = parameters['lifting_q12'].astype(np.int64)
        matching = parameters['lifting_matchings']
        for layer in range(4):
            first, second = matching[layer, :, 0], matching[layer, :, 1]
            x, y = value[first].copy(), value[second].copy()
            a = write24((x << 12)+q12[layer, :, 0, None]*y, 12)
            b = write24((y << 12)+q12[layer, :, 1, None]*a, 12)
            value[first], value[second] = a, b
        value = value[parameters['source_permutation']]
    threshold = parameters['source_threshold'][:, None]
    constant = parameters['source_constant'][:, None]
    gate = np.where(constant >= 0, constant.astype(bool),
                    np.where(parameters['source_direction'][:, None] > 0,
                             value >= threshold, value <= threshold))
    y, x = map(int, data['I24_halo_origin_yx'])
    expected = unpack_gate(data, 'sn1')[:, :, y:y+shape[2], x:x+shape[3]].reshape(10, -1)
    differences = int(np.count_nonzero(gate != expected))
    if differences:
        raise ValueError('Actual I24 block does not reproduce source gates')
    return dict(vectors=identity.shape[1], gates=gate.size, gate_differences=differences,
        reference='NumPy int64 complete dense dot or actual forty q12 half-step RNE/saturates, signed inclusive cutoff, source permutation; no real-matrix approximation.')


def packed_source_counts(packed, shape, stride):
    """Independent CPU reconstruction of ordered P2/T10 logical row counts."""
    ticks, channels, height, width = map(int, shape)
    word = np.zeros((channels, height, width), np.uint16)
    plane_bits = channels*height*width
    for t in range(ticks):
        plane = np.unpackbits(packed[t], bitorder='little', count=plane_bits)
        word |= plane.reshape(channels, height, width).astype(np.uint16) << t
    padded = np.pad(word, ((0, 0), (1, 1), (1, 1)))
    pop = np.array([int(i).bit_count() for i in range(1 << ticks)], np.uint8)
    active, rows = np.zeros((channels, 3, 3), np.int64), np.zeros((channels, 3, 3), np.int64)
    for kh in range(3):
        for kw in range(3):
            view = padded[:, kh:kh+height:stride, kw:kw+width:stride]
            active[:, kh, kw] = pop[view].sum((1, 2))
            if view.shape[-1] % 2:
                view = np.pad(view, ((0, 0), (0, 0), (0, 1)))
            rows[:, kh, kw] = ((view[:, :, 0::2] | view[:, :, 1::2]) != 0).sum((1, 2))
    return active.reshape(-1), rows.reshape(-1)


class OrderedCapture:
    def __init__(self, model, modules, helper, theta_sn2, names, directory, continuous_only=False):
        from capture import SOURCE_SN, BLOCK, CONSUMER_SN, PROJECT
        self.helper, self.theta_sn2 = helper, float(theta_sn2)
        self.continuous_only = continuous_only
        self.names, self.directory = list(names), directory
        self.index, self.handles, self.rows = -1, [], []
        self.paths = dict(sn1=SOURCE_SN, sn2=BLOCK+'.sn2.spiking_neuron', proj=CONSUMER_SN)
        self.geometry = {}
        for label, path in dict(sn1=BLOCK+'.conv1.0', sn2=BLOCK+'.conv2.0', proj=PROJECT+'.conv').items():
            conv = modules[path]
            self.geometry[label] = dict(module=path, native_weight_shape=list(conv.weight.shape),
                kernel=list(conv.kernel_size), stride=list(conv.stride), padding=list(conv.padding),
                dilation=list(conv.dilation), groups=int(conv.groups),
                required_output_stride=(1 if label == 'sn1' else 2))
        self.handles.append(model.register_forward_pre_hook(self.begin))
        self.projection_path = PROJECT+'.conv_res'
        self.handles.append(modules[self.projection_path].register_forward_hook(self.observe_continuous))
        if not continuous_only:
            for label in LABELS:
                self.handles.append(modules[self.paths[label]].register_forward_hook(self.observe(label)))

    def begin(self, module, inputs):
        self.index += 1
        if self.index >= len(self.names):
            raise RuntimeError('Unexpected extra network frame')
        self.arrays = dict(frame_name=np.array(self.names[self.index]),
            bit_order=np.array('little'), tensor_order=np.array('T,C,y,x'),
            spatial_origin_yx=np.array([0, 0], np.int32))
        self.order, self.stats = [], {}

    def save_frame(self):
        self.arrays['producer_order'] = np.asarray(self.order)
        filename = f'{self.index:03d}_{Path(self.names[self.index]).stem}.npz'
        np.savez_compressed(self.directory/filename, **self.arrays)
        self.rows.append(dict(file=self.names[self.index], capture=filename,
            producers=self.stats, producer_order=list(self.order)))
        self.arrays = None
        print('CAPTURE_WRITTEN', filename, flush=True)

    def observe_continuous(self, module, inputs, output):
        import torch
        if tuple(output.shape) != (10, 96, 120, 160):
            raise ValueError('Unexpected actual PED continuous output shape')
        bound_to_helper = getattr(module.forward, '__self__', None) is self.helper
        if not bound_to_helper or not self.helper.ready:
            raise ValueError('Continuous output did not execute the fixed helper')
        # Capture the returned module output; helper.continuous is only the
        # independent in-function identity check. Multiplying by2^14 exactly
        # recovers its existing integer quanta, with no new rounding/clamp.
        returned = output[:, :, :4, :4].double()*(1 << 14)
        expected = self.helper.continuous[:, :, :4, :4]
        mismatch = int(torch.count_nonzero(returned != expected))
        if mismatch or not torch.equal(returned, returned.round()):
            raise ValueError('Actual returned projection output differs from fixed q24')
        if int(returned.min()) < -(1 << 23) or int(returned.max()) >= (1 << 23):
            raise ValueError('Fixed continuous output violates signed24')
        self.arrays.update(proj_conv_res_q24=returned.to(torch.int32).cpu().numpy(),
            proj_conv_res_origin_yx=np.array([0, 0], np.int32),
            proj_conv_res_full_shape=np.asarray(output.shape, np.int32),
            proj_conv_res_state_frac=np.array(14, np.int32),
            axis=np.array(self.helper.c.axis),
            proj_conv_res_module_path=np.array(self.projection_path),
            proj_conv_res_forward_owner=np.array(type(self.helper).__name__),
            proj_conv_res_helper_binding=np.array(bound_to_helper),
            proj_conv_res_helper_integer_differences=np.array(mismatch, np.int64))
        self.stats['continuous'] = dict(module=self.projection_path,
            actual_forward_owner=type(self.helper).__name__, helper_binding=bound_to_helper,
            shape=[10, 96, 4, 4], full_shape=list(output.shape), state_frac=14,
            output_grid_origin_yx=[0, 0], input_anchor_yx_step=[2, 2],
            minimum_q24=int(returned.min()), maximum_q24=int(returned.max()),
            returned_output_vs_helper_q24_differences=mismatch,
            definition='Returned actual proj.conv_res output times2^14, copied as int32; fixed helper U/V and bias merge already applied. No FP shadow or new quantization.')
        if self.continuous_only:
            self.order.append('proj_conv_res')
            self.save_frame()

    def observe(self, label):
        def hook(module, inputs, output):
            import torch
            if tuple(output.shape) != (10, 1, 96, 240, 320):
                raise ValueError('Unexpected actual source layout: '+label+' '+str(tuple(output.shape)))
            if label != LABELS[len(self.order)]:
                raise RuntimeError('Changed actual producer order')
            theta = self.theta_sn2 if label == 'sn2' else float(module.thresh)
            local = output[:, 0]
            shape = tuple(local.shape)
            packed, nonzero, amplitude_error = [], 0, 0.0
            with torch.no_grad():
                for t in range(shape[0]):
                    value = local[t]
                    gate = value.ne(0)
                    error = float(torch.where(gate, value-theta, value).abs().max())
                    if not np.isfinite(error):
                        raise ValueError('Nonfinite theta*g source')
                    amplitude_error = max(amplitude_error, error)
                    bits = gate.cpu().numpy()
                    nonzero += int(bits.sum())
                    packed.append(np.packbits(bits.reshape(-1), bitorder='little'))
            if amplitude_error != 0:
                raise ValueError('Packing would change actual amplitude: '+label)
            self.arrays.update({label+'_gate_bits': np.stack(packed),
                label+'_shape': np.asarray(shape, np.int32),
                label+'_native_shape': np.asarray(output.shape, np.int32),
                label+'_theta': np.array(theta, np.float64)})
            self.order.append(label)
            self.stats[label] = dict(module=self.paths[label], nonzero=nonzero,
                shape=shape, theta=theta, theta_g_max_abs_error=amplitude_error,
                packed_bytes=self.arrays[label+'_gate_bits'].nbytes)
            if label == 'sn1':
                # Actual raw helper state, before it is released by proj.sn.
                identity = self.helper.i[:, :, :10, :10]
                if identity is None or not torch.equal(identity, identity.round()):
                    raise ValueError('I24 capture is not the actual integer state')
                self.arrays.update(I24_halo=identity.to(torch.int32).cpu().numpy(),
                    I24_halo_origin_yx=np.array([0, 0], np.int32),
                    I24_core_yxhw=np.array([0, 0, 8, 8], np.int32),
                    I24_fraction_bits=np.array(14, np.int32),
                    I24_padding_note=np.array('Native top-left10x10; core top-left8x8. Coordinates outside the full image are not stored; convolution padding is applied from geometry.'))
            if label == 'proj':
                self.save_frame()
        return hook

    def restore(self):
        for handle in self.handles: handle.remove()
        self.handles.clear()


def compare_constants(helper, filename):
    with np.load(filename, allow_pickle=False) as old:
        actual = helper.export_constants()
        differences = [key for key in old.files
                       if key not in actual or not np.array_equal(old[key], actual[key])]
    if differences:
        raise ValueError('Fixed function differs from saved deployment constants: '+str(differences))
    return dict(reference=str(filename), checked_fields=len(old.files), differences=[])


def verify_saved_capture(directory, rows, previous_counts):
    reference = {r['file']: r['sources'] for r in json.loads(previous_counts.read_text())['frames']}
    with np.load(directory/'parameters.npz') as values:
        parameters = {key: values[key] for key in values.files}
    reports = []
    for row in rows:
        report = dict(file=row['file'], source_count_reference=str(previous_counts), sources={})
        with np.load(directory/row['capture']) as z:
            report['source_block_reference'] = source_block_reference(z, parameters)
            for label in LABELS:
                active, nrv = packed_source_counts(z[label+'_gate_bits'], z[label+'_shape'],
                                                   1 if label == 'sn1' else 2)
                ref = reference[row['file']][COUNT_KEYS[label]] if row['file'] in reference else None
                report['sources'][label] = dict(source_occurrences=int(active.sum()), nrv_rows=int(nrv.sum()),
                    active_columns=active, nrv_rows_by_k=nrv,
                    previous_active_columns_differences=(None if ref is None else
                        int(np.count_nonzero(active != np.asarray(ref['active_columns'])))),
                    previous_nrv_rows_by_k_differences=(None if ref is None else
                        int(np.count_nonzero(nrv != np.asarray(ref['nrv_rows_by_k'])))))
        reports.append(report)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=BASE)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--continuous-only', action='store_true',
                        help='Only capture actual proj.conv_res top-left4x4 q24; no full gate recapture.')
    parser.add_argument('--axes', nargs='+', choices=AXES, default=list(AXES))
    parser.add_argument('--files', nargs='+', default=['zurich_city_09_a_0001.npy'])
    args = parser.parse_args()
    args.output = args.output or HERE/('capture_continuous' if args.continuous_only else 'capture_inputs')
    import torch
    import run_probe as probe
    from adapter import fp32_matmul
    from flow_backward_probe import TrainableLatentPair, read_arrays
    from capture import BLOCK, SOURCE_SN, CONSUMER_SN
    from evaluate_branch_control import evaluate_axis, mask_nonanchors
    from train_shared_temporal_recovery import SharedTemporalControl
    from lifting_temporal_control import LiftingTemporalControl
    from fixed_temporal_coordinates import FixedTemporalForward
    from fixed_lifting_coordinates import FixedLiftingForward
    from spikingjelly.activation_based import functional

    args.output.mkdir(parents=True, exist_ok=True)
    top_output, args.split = args.output, 'diverse'
    system = probe.load_system(args)
    model, modules, _, current, sources, _, _, _ = system
    probe.install_sources(system, sources)
    current['count_codes'] = False
    calibration = torch.load(PATCH/'patch_train_calibration.pt', map_location='cpu', weights_only=False)
    for path, values in calibration.items():
        bn = modules[path]
        bn.track_running_stats = True
        bn.running_mean, bn.running_var = values['mean'].to(bn.weight), values['var'].to(bn.weight)
    model.eval()
    model.requires_grad_(False)
    flags = json.loads((CHAIN/'affine_shared_temporal_control_diverse10/run.json').read_text())
    torch.backends.cuda.matmul.allow_tf32 = bool(flags['TF32_matmul'])
    torch.backends.cudnn.allow_tf32 = bool(flags['TF32_cudnn'])
    parent = LATENT/'flow_recovery64/preview_only/shared48_u8_vq5.npz'
    pair = TrainableLatentPair(read_arrays(parent), modules[SOURCE_SN].weight.device)
    pair.u.requires_grad_(False); pair.v.requires_grad_(False)
    conv1, sn2 = modules[BLOCK+'.conv1.0'], modules[BLOCK+'.sn2.spiking_neuron']
    old_conv1, old_sn2 = conv1.forward, sn2.forward
    conv1.forward, sn2.forward = pair.conv_forward, pair.neuron_forward
    masks = {}
    def nonanchor(module, inputs, output):
        key = (tuple(output.shape[-2:]), output.device)
        if key not in masks:
            mask = torch.zeros(output.shape[-2:], device=output.device, dtype=torch.bool)
            mask[::2, ::2] = True
            masks[key] = mask
        return mask_nonanchors(output, masks[key])
    common_hook = modules[BLOCK+'.norm2'].register_forward_hook(nonanchor)
    common = (modules, read_arrays(RES/'rank_control_parameters.npz'),
              read_arrays(CHAIN/'rank32_diverse10/parameters.npz'))
    run = dict(complete=False, files=args.files, axes={}, parent=str(parent), continuous_only=args.continuous_only,
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32, TF32_cudnn=torch.backends.cudnn.allow_tf32,
        scope='Inference-only capture of existing fixed students. Ordered full source support; no new quantization, training or hardware timing.',
        layout=('proj_conv_res_q24[T10,C96,4,4] int32 carrying signed24/f14, output-grid origin[0,0], input anchors step2; actual helper return, no new quantization.' if args.continuous_only else
            'Each *_gate_bits is [T10,921600] uint8; little-bit T,C,y,x, B1 omitted. Also actual continuous proj_conv_res_q24[10,96,4,4].'),
        capture_scope=('Actual continuous output only; no full gate recapture. Existing coarse evaluator still completes one frame per axis.' if args.continuous_only else
            'Three actual theta*g outputs, retained raw I24 top-left10x10 and actual continuous output top-left4x4. Existing coarse evaluator completes the frame.'),
        common_function='Fixed4patchBN; frozen FP32 U8/VQ5 preview; anchor-only Conv2R16; raw I24; actual PED gate plus continuousU32/V32 consumers; native remaining coarse network.')
    save_json(top_output/'result.json', run)
    try:
        for label in args.axes:
            started = time.monotonic()
            if label == 'ordinary':
                axis = 'identity_permuted_base'
                student_path = CHAIN/'temporal_structured_recovery/stage128x256'/f'{axis}.npz'
                deployed = CHAIN/'temporal_structured_recovery/fixed_sources_diverse10'
                constants = deployed/f'{axis}_coordinate_constants.npz'
                controller = SharedTemporalControl(*common, fit={})
                controller.load_saved(axis, read_arrays(student_path))
                helper = FixedTemporalForward(controller, pair.temporal.theta)
            else:
                axis = 'fast_raw_diagonal'
                student_path = LIFT/'stage320'/f'{axis}.npz'
                deployed = LIFT/'fixed_lifting_diverse10'
                constants = deployed/f'{axis}_fixed_constants.npz'
                init = json.loads((LIFT/'initialization.json').read_text())
                controller = LiftingTemporalControl(*common, source_fit=init['source_fit'],
                    consumer_fits=init['consumer_fits'], basis_lifting=init['basis_lifting'])
                controller.load_saved(axis, read_arrays(student_path))
                helper = FixedLiftingForward(controller, pair.temporal.theta)
            args.output = top_output/label
            args.output.mkdir(parents=True, exist_ok=True)
            observer = None
            try:
                checked = compare_constants(helper, constants)
                np.savez_compressed(args.output/'parameters.npz', **helper.export_constants())
                observer = OrderedCapture(model, modules, helper, pair.temporal.theta, args.files, args.output,
                                          continuous_only=args.continuous_only)
                with torch.no_grad():
                    evaluation = evaluate_axis(args, model, current, args.files, axis, progress_tag='STAGE_B_CAPTURE_AEE')
                save_json(args.output/'ranges.json', helper.range_report(args.files))
                source_checks = ([] if args.continuous_only else
                    verify_saved_capture(args.output, observer.rows, deployed/f'{axis}_sources.json'))
                old_frames = json.loads((deployed/f'{axis}_frames.json').read_text())
                previous = {r['file']: r['AEE'] for r in old_frames}
                new_frames = json.loads((args.output/f'{axis}_frames.json').read_text())
                comparisons = [dict(file=r['file'], actual_AEE=r['AEE'], previous_AEE=previous.get(r['file']),
                    difference=(None if r['file'] not in previous else r['AEE']-previous[r['file']])) for r in new_frames]
                row = dict(complete=True, axis=axis, student=str(student_path), constants=checked,
                    frames=observer.rows, geometry=observer.geometry, source_checks=source_checks,
                    AEE=evaluation, previous_frame_AEE=comparisons, wall_seconds=time.monotonic()-started)
                save_json(args.output/'capture.json', row)
                run['axes'][label] = row
                save_json(top_output/'result.json', run)
            finally:
                if observer is not None: observer.restore()
                helper.restore()
                controller.restore()
                functional.reset_net(model)
                current.pop('flow', None)
                torch.cuda.empty_cache()
        run['complete'] = True
        save_json(top_output/'result.json', run)
    finally:
        common_hook.remove()
        conv1.forward, sn2.forward = old_conv1, old_sn2


if __name__ == '__main__':
    main()
