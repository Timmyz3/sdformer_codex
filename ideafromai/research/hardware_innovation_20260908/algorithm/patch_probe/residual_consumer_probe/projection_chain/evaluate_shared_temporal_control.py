"""Four fixed accuracy controls for ordinary temporal-matrix sharing.

1. Reproduce the saved uniform Conv2R16/full PED projection function.
2. Add the already studied nonanchor whole-normalized-branch deletion.
3. Add the saved ordinary PED rank32 SVD.
4. Copy r1.sn1.A into proj.sn.A; keep each neuron's own bias/center/theta.

These are four predetermined dependencies, not a rank/threshold search. Use
the old preview-only LatentPair parent and original Conv2/BN parameters, not
the later recovered W/L students. Matrix copying changes the student. Both
neurons still execute independently; no membrane reuse or speed is claimed.
Root launches the fixed diverse10 evaluation; this file performs no training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

AXES = (
    'rank16_original_projection',
    'rank16_delete_nonanchor',
    'rank16_delete_nonanchor_ped32',
    'rank16_delete_nonanchor_ped32_tied_A',
)


def read_arrays(path):
    with np.load(path) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def frame_comparison(actual_path, reference_path):
    actual = json.loads(actual_path.read_text())
    reference = json.loads(reference_path.read_text())
    expected = {row['file']: row for row in reference}
    rows = []
    for row in actual:
        if row['file'] in expected:
            old = expected[row['file']]
            rows.append(dict(file=row['file'], valid_pixels_delta=row['valid_pixels']-old['valid_pixels'],
                aee_sum_delta=row['aee_sum']-old['aee_sum'], AEE_delta=row['AEE']-old['AEE']))
    equal = bool(rows) and all(not row['valid_pixels_delta'] and not row['aee_sum_delta']
                              and not row['AEE_delta'] for row in rows)
    return dict(reference=str(reference_path), compared_frames=len(rows),
        same_frame_set={row['file'] for row in actual} == set(expected),
        exact_per_frame_AEE_sum_and_count=equal,
        max_abs_AEE_delta=max((abs(row['AEE_delta']) for row in rows), default=None), rows=rows,
        scope='First axis only: old uniform_rank16, with no nonanchor deletion. The old825 AEE1.2204603075252283 belongs to this function; it is not evidence for axes2-4.')


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.split = 'diverse'
    args.count = 10
    alg = args.root/'algorithm'
    area = alg/'patch_probe'
    residual = area/'residual_consumer_probe'
    chain = residual/'projection_chain'
    latent = area/'factor_completion_20260909/latent_stage_train16'
    args.output = args.output or chain/'shared_temporal_control_diverse10'
    args.output.mkdir(parents=True, exist_ok=True)
    for path in (alg, alg/'nrv_cost_probe', latent, residual):
        sys.path.insert(0, str(path))
    import run_probe as probe
    from run_bn_probe import save_json
    from adapter import install_latent_factor
    from capture import BLOCK, SOURCE_SN, PROJECT, CONSUMER_SN, neuron_parameters
    from evaluate_rank_control import RankConvControl
    from evaluate_branch_control import evaluate_axis, mask_nonanchors

    parent = latent/'flow_recovery64/preview_only/shared48_u8_vq5.npz'
    conv2_path = residual/'rank_control_parameters.npz'
    ped_path = chain/'rank32_diverse10/parameters.npz'
    old_directory = residual/'rank_control_diverse10'
    old_runtime = json.loads((old_directory/'run.json').read_text())
    conv2_arrays, ped = read_arrays(conv2_path), read_arrays(ped_path)
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
    torch.backends.cuda.matmul.allow_tf32 = bool(old_runtime['TF32_matmul'])
    torch.backends.cudnn.allow_tf32 = bool(old_runtime['TF32_cudnn'])
    conv1, sn2 = modules[BLOCK+'.conv1.0'], modules[BLOCK+'.sn2.spiking_neuron']
    # Same constructor as the old rank control. No recovery surrogate adapter
    # or recovered Conv2/BN bias is installed in this experiment.
    _, originals = install_latent_factor(conv1, sn2, parent, conditional=False)
    conv2, projection = modules[BLOCK+'.conv2.0'], modules[PROJECT+'.conv_res']
    source_sn, proj_sn = modules[SOURCE_SN], modules[CONSUMER_SN]
    state_source, state_proj = neuron_parameters(source_sn), neuron_parameters(proj_sn)
    if any(state['T'] != 10 or state['temporal_factor_rank'] != 0 for state in (state_source, state_proj)):
        raise ValueError('This fixed A-sharing probe requires the observed native dense T10 neurons')
    original_A = proj_sn.weight.detach().clone()
    original_projection = projection.forward
    if not torch.equal(projection.weight.detach()[:, :, 0, 0].double().cpu(), torch.from_numpy(ped['C'])):
        raise ValueError('The saved ordinary PED SVD has a different original C')
    first = torch.as_tensor(ped['U'], device=projection.weight.device, dtype=projection.weight.dtype)[:, :, None, None]
    second = torch.as_tensor(ped['V'], device=projection.weight.device, dtype=projection.weight.dtype)[:, :, None, None]
    def projected(x):
        # Exact operation sequence of evaluate_rank32.py; original bias once.
        return F.conv2d(F.conv2d(x[:, :, ::2, ::2], first), second, projection.bias)
    control = RankConvControl(conv2, projection, conv2_arrays)
    control.axis = 'uniform_rank16'
    conv2.forward = control.forward
    current_axis = {'delete_nonanchor': False}
    masks = {}
    def delete_nonanchor(module, inputs, output):
        if not current_axis['delete_nonanchor']:
            return output
        key = (output.shape[-2:], output.device)
        if key not in masks:
            mask = torch.zeros(output.shape[-2:], device=output.device, dtype=torch.bool)
            mask[::2, ::2] = True
            masks[key] = mask
        return mask_nonanchors(output, masks[key])
    hook = modules[BLOCK+'.norm2'].register_forward_hook(delete_nonanchor)
    names = json.loads((alg/'samples.json').read_text())['valid'][:10]
    saved = dict(conv2_first_R16=conv2_arrays['first_factor32'][:16],
        conv2_second_R16=conv2_arrays['second_factor32'][:, :16],
        conv2_has_bias=np.array(conv2.bias is not None),
        conv2_bias=(np.zeros(96, np.float32) if conv2.bias is None else conv2.bias.detach().cpu().numpy()),
        ped_C=ped['C'], ped_U=ped['U'], ped_V=ped['V'], ped_rank=np.array(32),
        ped_has_bias=np.array(projection.bias is not None),
        ped_bias=(np.zeros(96, np.float32) if projection.bias is None else projection.bias.detach().cpu().numpy()),
        proj_tied_A=state_source['A'].numpy().copy(), parent=np.array(str(parent)),
        definition=np.array('Only fourth axis replaces proj.A by a value copy of native r1.sn1.A. Bias, center, theta and output/threshold modes stay local to each neuron; both native forwards still execute.'))
    for label, state in (('r1_sn1', state_source), ('proj_original', state_proj)):
        saved.update({label+'_'+key: value.numpy() if torch.is_tensor(value) else np.asarray(value)
                      for key, value in state.items()})
    np.savez_compressed(args.output/'parameters.npz', **saved)
    definitions = {
        AXES[0]: 'Old uniform Conv2R16 at all spatial positions, real BN2 branch everywhere and original full PED C. Reproduce the sealed ordinary control.',
        AXES[1]: 'Axis1 plus zeroing the entire r1 normalized branch at nonanchors; identity remains.',
        AXES[2]: 'Axis2 plus the saved ordinary PED R32 SVD, evaluated as two FP32 convolutions.',
        AXES[3]: 'Axis3 plus proj.sn.weight <- r1.sn1.weight; keep both original biases, centers, thresholds/amplitudes and full T10 labels.',
    }
    run = dict(complete=False, files=names, fixed_count=10, selected_axes=list(AXES),
        parent=str(parent), conv2_parameters=str(conv2_path), ped_parameters=str(ped_path),
        common_base='Old preview-onlyR32 dequantizedU8/VQ5 LatentPair; original Conv2/BN biases and four fixed patch BN statistics; original S2 and real coarse-head successors. No recovered W/L student.',
        parent_correction='The existing Conv2R16 full825 AEE1.2204603075252283 has no nonanchor branch deletion. Axis2 is a new combination, evaluated separately.',
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32,
        TF32_cudnn=torch.backends.cudnn.allow_tf32,
        temporal={label: dict(T=state['T'], rank=int(torch.linalg.matrix_rank(state['A'].double())),
            nonzero_A=int(torch.count_nonzero(state['A'])),
            condition_number=float(torch.linalg.cond(state['A'].double())),
            theta=state['theta'].tolist(), bias=state['b'].tolist(), center=state['center'].tolist(),
            center_mode=state['center_mode'], output_mode=state['output_mode'], threshold_mode=state['threshold_mode'])
            for label, state in (('r1_sn1', state_source), ('proj_original', state_proj))},
        A_source_vs_original_proj_max_abs=float((state_source['A']-state_proj['A']).abs().max()),
        axis_definitions=definitions,
        metric='Unchanged evaluate_branch_control.evaluate_axis, full actual coarse-flow successors and fixed diverse10 GT.',
        claim='Changed students and ordinary controls only. No training, validation-selected parameters, shared membrane execution, linear-fusion implementation or hardware speed claim.',
        results={})
    save_json(args.output/'run.json', run)
    try:
        for index, axis in enumerate(AXES):
            control.frame_cache.clear()
            current_axis['delete_nonanchor'] = index >= 1
            projection.forward = projected if index >= 2 else original_projection
            proj_sn.weight.copy_(source_sn.weight if index == 3 else original_A)
            run['results'][axis] = evaluate_axis(args, model, current, names, axis,
                progress_tag='SHARED_TEMPORAL_CONTROL_AEE')
            if index == 0:
                run['old_rank16_diverse10_alignment'] = frame_comparison(
                    args.output/(axis+'_frames.json'), old_directory/'uniform_rank16_frames.json')
                print('OLD_RANK16_ALIGNMENT', json.dumps(run['old_rank16_diverse10_alignment']), flush=True)
            save_json(args.output/'run.json', run)
        run['complete'] = True
        save_json(args.output/'run.json', run)
    finally:
        hook.remove()
        proj_sn.weight.copy_(original_A)
        projection.forward = original_projection
        conv2.forward = control.original_forward
        conv1.forward, sn2.forward = originals['conv1_forward'], originals['neuron_forward']
        control.frame_cache.clear()
        current.pop('flow', None)
    print('DONE', json.dumps({name: value['AEE_frame_mean'] for name, value in run['results'].items()}), flush=True)


if __name__ == '__main__':
    main()
