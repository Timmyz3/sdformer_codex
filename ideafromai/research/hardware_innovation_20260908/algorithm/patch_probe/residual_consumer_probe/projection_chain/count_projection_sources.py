"""Count three actual theta*g sources on the ordinary rank32 PED parent.

Root launches the fixed diverse10 forward. Nine offset views, in C8 chunks,
replace full-frame unfold. An NRV row belongs to one (c,kh,kw) and two adjacent
PED anchors: OR all T10 at both positions. These are source-ready logical rows,
not physical SRAM requests, unique source pixels, or execution cycles.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def valid_anchor_slice(extent, anchors, stride, delta):
    """Indices a with 0 <= stride*a+delta < extent, retaining global parity."""
    first = max(0, -(delta // stride))  # ceil(-delta / stride)
    last = min(anchors, (extent - 1 - delta) // stride + 1)
    return first, last, slice(stride*first + delta, stride*last + delta, stride)


@torch.no_grad()
def count_source(x, effective_stride, theta, selected_channels=None):
    """x is actual T,C,H,W. Padding=1 and kernel=3, then PED anchor grouping.

    Only a single offset's T-OR map is padded to its real anchor coordinates;
    trimming an invalid left border must not change the two-anchor pairing.
    """
    if x.ndim != 4:
        raise ValueError('count_source expects T,C,H,W')
    ticks, channels, height, width = x.shape
    ah = (height + effective_stride - 1) // effective_stride
    aw = (width + effective_stride - 1) // effective_stride
    groups_per_row = (aw + 1) // 2
    active_columns = torch.zeros((channels, 3, 3), device=x.device, dtype=torch.int64)
    nrv_columns = torch.zeros_like(active_columns)
    residual = torch.zeros((), device=x.device, dtype=x.dtype)
    nz_min = torch.full((), float('inf'), device=x.device, dtype=x.dtype)
    nz_max = torch.full((), -float('inf'), device=x.device, dtype=x.dtype)
    theta_tensor = torch.as_tensor(theta, device=x.device, dtype=x.dtype)
    for kh in range(3):
        y0, y1, sy = valid_anchor_slice(height, ah, effective_stride, kh-1)
        for kw in range(3):
            x0, x1, sx = valid_anchor_slice(width, aw, effective_stride, kw-1)
            if y0 >= y1 or x0 >= x1:
                continue
            for c0 in range(0, channels, 8):
                c1 = min(c0+8, channels)
                values = x[:, c0:c1, sy, sx].detach()
                active = values.ne(0)
                active_columns[c0:c1, kh, kw] = active.sum((0, 2, 3))
                spatial = torch.zeros((c1-c0, ah, groups_per_row*2),
                                      device=x.device, dtype=torch.bool)
                spatial[:, y0:y1, x0:x1] = active.any(0)
                rows = spatial.reshape(c1-c0, ah, groups_per_row, 2).any(-1)
                nrv_columns[c0:c1, kh, kw] = rows.sum((1, 2))
                residual = torch.maximum(residual, (values-active.to(values.dtype)*theta_tensor).abs().amax())
                nz_min = torch.minimum(nz_min, torch.where(active, values, float('inf')).amin())
                nz_max = torch.maximum(nz_max, torch.where(active, values, -float('inf')).amax())
    activity = active_columns.cpu().numpy().reshape(channels, 9)
    nrv = nrv_columns.cpu().numpy().reshape(channels, 9)
    result = dict(source_shape=list(x.shape), effective_stride=effective_stride, padding=1,
        anchor_shape=[ah, aw], anchors=ah*aw, T_times_anchors=ticks*ah*aw,
        paired_anchor_groups=ah*groups_per_row, anchor_slots_per_group=2,
        last_group_valid_slots=1 if aw % 2 else 2,
        logical_rows_with_empty_rows=channels*9*ah*groups_per_row,
        source_occurrences=int(activity.sum()), nrv_rows=int(nrv.sum()),
        active_columns=activity.reshape(-1).tolist(), nrv_rows_by_k=nrv.reshape(-1).tolist(),
        theta=float(theta), theta_g_referenced_max_abs=float(residual.cpu()),
        nonzero_amplitude_min=None if not activity.any() else float(nz_min.cpu()),
        nonzero_amplitude_max=None if not activity.any() else float(nz_max.cpu()))
    if selected_channels is not None:
        selected = np.asarray(selected_channels, dtype=bool)
        if selected.shape != (channels,):
            raise ValueError('Selected C8 mask has the wrong source-channel shape')
        result.update(selected_source_occurrences=int(activity[selected].sum()),
            retained_source_occurrences=int(activity[~selected].sum()),
            selected_nrv_rows=int(nrv[selected].sum()), retained_nrv_rows=int(nrv[~selected].sum()))
    return result


def explicit_padded_reference(x, stride):
    """Independent scalar indexing, including the padded member of an odd tail."""
    value = x.numpy()
    ticks, channels, height, width = value.shape
    ah, aw = (height+stride-1)//stride, (width+stride-1)//stride
    activity = np.zeros((channels, 9), np.int64)
    nrv = np.zeros_like(activity)
    for c in range(channels):
        for kh in range(3):
            for kw in range(3):
                k = kh*3+kw
                for ay in range(ah):
                    for group in range((aw+1)//2):
                        row_live = False
                        for p in range(2):
                            ax = group*2+p
                            sy, sx = stride*ay+kh-1, stride*ax+kw-1
                            for t in range(ticks):
                                valid = ax < aw and 0 <= sy < height and 0 <= sx < width
                                live = valid and value[t, c, sy, sx] != 0
                                activity[c, k] += int(live)
                                row_live |= live
                        nrv[c, k] += int(row_live)
    return activity.reshape(-1), nrv.reshape(-1)


def self_test():
    rng = np.random.default_rng(60909)
    cases = 0
    for stride in (2, 4):
        for height, width in ((1, 1), (5, 9), (8, 12), (13, 17)):
            for kind, theta in (('zero', 1.375), ('ones', -0.625), ('random', 1.375)):
                shape = (10, 3, height, width)
                gate = (np.zeros(shape, bool) if kind == 'zero' else
                        np.ones(shape, bool) if kind == 'ones' else rng.random(shape) < 0.19)
                x = torch.from_numpy(gate.astype(np.float32)*theta)
                mask = np.array([True, False, True])
                got = count_source(x, stride, theta, mask)
                a, n = explicit_padded_reference(x, stride)
                assert np.array_equal(got['active_columns'], a), (stride, height, width, kind, 'activity')
                assert np.array_equal(got['nrv_rows_by_k'], n), (stride, height, width, kind, 'NRV')
                assert got['selected_source_occurrences'] == int(a.reshape(3, 9)[mask].sum())
                assert got['selected_nrv_rows'] == int(n.reshape(3, 9)[mask].sum())
                assert got['theta_g_referenced_max_abs'] == 0
                cases += 1
    print(json.dumps(dict(cpu_self_test='PASS', cases=cases,
        checks='explicit padding and C/kh/kw ordering; T10 OR then paired anchors; left-border parity, odd tail, zero/all-live/random, nonunit signed theta, selected/retained masks')))


class SourcesReady(Exception):
    pass


@torch.no_grad()
def run(args):
    alg = args.root/'algorithm'
    area = alg/'patch_probe'
    residual = area/'residual_consumer_probe'
    latent = area/'factor_completion_20260909/latent_stage_train16'
    chain = residual/'projection_chain'
    args.output = args.output or chain/'source_counts_diverse10'
    args.selection = args.selection or chain/'consumer_pruning_diverse10/parameters.npz'
    for path in (alg, alg/'nrv_cost_probe', latent, residual, chain):
        sys.path.insert(0, str(path))
    import run_probe as probe
    from run_bn_probe import input_frame
    from adapter import install_latent_factor
    from capture import BLOCK, PROJECT
    from capture_chain import SPECS
    from evaluate_branch_control import mask_nonanchors
    from spikingjelly.activation_based import functional

    with np.load(args.selection) as z:
        selected = {label: z[label+'_selected_source_mask'].astype(bool) for label in ('r0', 'r1')}
        selected_groups = {label: z[label+'_selected_groups'].tolist() for label in selected}
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
    pair, originals = install_latent_factor(conv1, sn2, parent, conditional=False)
    conv = modules[PROJECT+'.conv_res']
    original_projection_forward = conv.forward
    c = conv.weight.detach()[:, :, 0, 0].double().cpu()
    left, singular, right = torch.linalg.svd(c, full_matrices=False)
    first = right[:32].float().to(conv.weight)[:, :, None, None]
    second = (left[:, :32]*singular[:32]).float().to(conv.weight)[:, :, None, None]
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

    records, pointers, handles = {}, {}, []
    handles.append(modules[BLOCK+'.norm2'].register_forward_hook(delete_nonanchor))
    source_identity = {}
    for label, spec in SPECS.items():
        source_conv = modules[spec['conv']]
        if (tuple(source_conv.kernel_size), tuple(source_conv.stride), tuple(source_conv.padding),
            tuple(source_conv.dilation), source_conv.groups) != ((3, 3), (spec['stride'],)*2, (1, 1), (1, 1), 1):
            raise ValueError('This fixed nine-offset counter does not match '+spec['conv'])
        theta = float(pair.temporal.theta) if label == 'r1' else float(modules[spec['neuron']].thresh.detach())
        effective_stride = spec['stride']*2
        source_identity[label] = dict(producer=spec['producer'], conv=spec['conv'], theta=theta,
            effective_anchor_stride=effective_stride, expected_source_channels=spec['channels'])
        def producer(module, inputs, output, label=label):
            pointers[label] = output.data_ptr()
        def observe(module, inputs, label=label, stride=effective_stride, theta=theta):
            x = inputs[0]
            if x.ndim == 5 and tuple(x.shape[:2]) == (10, 1):
                local = x.flatten(0, 1)
            elif x.ndim == 4 and x.shape[0] == 10:
                local = x
            else:
                raise ValueError('Expected actual T10/B1 source, got '+str(tuple(x.shape)))
            records[label] = count_source(local, stride, theta, selected.get(label))
            records[label]['native_source_shape'] = list(x.shape)
            records[label]['consumer_input_same_storage_as_producer'] = x.data_ptr() == pointers.get(label)
        handles.append(modules[spec['producer']].register_forward_hook(producer))
        handles.append(source_conv.register_forward_pre_hook(observe))
    def stop(module, inputs, output):
        raise SourcesReady()
    handles.append(modules[PROJECT+'.sn.spiking_neuron'].register_forward_hook(stop))

    names = json.loads((alg/'samples.json').read_text())['valid'][:10]
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(complete=False, axis='ordinary_rank32', parent=str(parent), files=names,
        rank=32, fixed_BN_names=list(fixed), selected_mask_file=str(args.selection), selected_groups=selected_groups,
        common_parent='Exactly evaluate_rank32.py: R32 U8/VQ5 preview-only, four fixed patch BNs, r1 nonanchor whole normalized branch deletion, ordinary FP32 U/V rank32 projection.',
        changed_weights_or_theta=False, source_identity=source_identity,
        stop_boundary='After actual proj.sn; conv_res also runs with the ordinary rank32 factors. No flow head or GT load.',
        layout='k=((c*3)+kh)*3+kw; group=(anchor_y, floor(anchor_x/2)); p=anchor_x%2; T10 OR at both p. In native r1 coordinates y=2*anchor_y, x=4*group_x+2*p. Stem source center is twice native r1 coordinates.',
        selection_scope='Existing train4 selected C8 groups partition counts only; source and convolution weights remain ordinary. No validation fitting.',
        count_scope='Complete-frame PED anchor3x3 source occurrences include halo repeats. NRV rows combine T10 and two adjacent anchors but not distinct k or distinct groups. No coefficient-zero filtering, physical word packing, cache, cycles, or PPA.',
        theta_check_scope='Actual values referenced by all PED anchor3x3 windows; no unreferenced-source amplitude claim.',
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32, TF32_cudnn=torch.backends.cudnn.allow_tf32, frames=[])
    save_json(args.output/'result.json', report)
    started = time.monotonic()
    try:
        conv.forward = projected
        for index, name in enumerate(names):
            records.clear(); pointers.clear()
            functional.reset_net(model)
            x, _, _ = input_frame(args.data, name, targets=False)
            try:
                model(x)
            except SourcesReady:
                pass
            else:
                raise RuntimeError('Actual proj.sn stop boundary was not reached')
            if set(records) != set(SPECS):
                raise RuntimeError('A declared actual source did not execute')
            del x
            report['frames'].append(dict(file=name, sources=dict(records)))
            save_json(args.output/'result.json', report)
            print('SOURCE_COUNTS', index+1, name, json.dumps({label: {k: r[k] for k in
                ('source_occurrences', 'nrv_rows', 'T_times_anchors', 'theta_g_referenced_max_abs')}
                for label, r in records.items()}), flush=True)
        totals = {}
        for label in SPECS:
            rows = [frame['sources'][label] for frame in report['frames']]
            fields = ['source_occurrences', 'nrv_rows', 'T_times_anchors', 'anchors',
                      'paired_anchor_groups', 'logical_rows_with_empty_rows']
            if label in selected:
                fields += ['selected_source_occurrences', 'retained_source_occurrences', 'selected_nrv_rows', 'retained_nrv_rows']
            totals[label] = {field: sum(row[field] for row in rows) for field in fields}
            totals[label]['theta_g_referenced_max_abs'] = max(row['theta_g_referenced_max_abs'] for row in rows)
        report.update(complete=True, totals=totals, wall_seconds=time.monotonic()-started)
        save_json(args.output/'result.json', report)
        print('DONE', json.dumps(totals), flush=True)
    finally:
        for handle in handles:
            handle.remove()
        conv.forward = original_projection_forward
        conv1.forward, sn2.forward = originals['conv1_forward'], originals['neuron_forward']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.root is None:
        parser.error('--root is required for the actual diverse10 forward')
    run(args)


if __name__ == '__main__':
    main()
