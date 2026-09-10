"""Reload exported consumer-recovery students for diverse10 or valid825 AEE.

No training. Reuse RecoveryControl's actual constrained W/L and explicit
independent-L arithmetic. Load W_shadow/L_shadow, learned BN biases, the
projection base-bias presence/value and its separate learned output increment.
original_W and related arrays are initialization/identity references only.
The default latent parent uses the exact TrainableLatentPair construction and
forward used at the end of recovery, with frozen parameters and no gradients.
--native-latent selects the original LatentPair as a separate control.

Default diverse10 compares each frame's AEE sum/count with this export's own
final recovery evaluation. This tests reloading, not training reproducibility
or a prior run with the same seed. GPU execution is launched by root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

from train_consumer_recovery import AXES, RecoveryControl, LatentPair, TrainableLatentPair, read_arrays
from capture_chain import R1, PROJECT, save_json


@torch.no_grad()
def load_recovered(controller, arrays):
    """Restore the exported function, retaining the original operation order."""
    axis = str(arrays['axis'])
    biases = dict(r0_bias=arrays['r0_bias'], r1_bias=arrays['r1_bias'],
                  projection_bias=arrays['projection_base_bias'])
    controller.initialize_axis(axis, biases)
    for label, shadow in controller.shadow.items():
        shadow.copy_(torch.as_tensor(arrays[label+'_W_shadow'], device=shadow.device, dtype=shadow.dtype))
    for label, shadow in controller.l_shadow.items():
        key = label+'_L_shadow' if label+'_L_shadow' in arrays else label+'_L'
        shadow.copy_(torch.as_tensor(arrays[key], device=shadow.device, dtype=shadow.dtype))
    # Presence is part of the actual numerical path; do not infer it from an
    # all-zero vector or fold the separate increment into this convolution.
    projection = controller.projection
    projection.bias = (nn.Parameter(torch.as_tensor(arrays['projection_base_bias'],
        device=projection.weight.device, dtype=projection.weight.dtype).clone(), requires_grad=False)
        if bool(arrays['projection_base_has_bias']) else None)
    controller.projection_bias_delta.copy_(torch.as_tensor(arrays['projection_bias_delta'],
        device=projection.weight.device, dtype=projection.weight.dtype))
    for parameter in controller.named_trainables().values():
        parameter.requires_grad_(False)
    controller.collect_activity = False
    controller.counts = {'r0': [], 'r1': []}
    controller.terms.clear()
    return dict(axis=axis, W_effective_vs_export_max_abs={label: float((controller.effective_weight(label)
        -torch.as_tensor(arrays[label+'_effective_W'], device=projection.weight.device)).abs().max())
        for label in ('r0', 'r1')},
        L_effective_vs_export_max_abs={label: float((controller.effective_l(label)
            -torch.as_tensor(arrays[label+'_L'], device=projection.weight.device)).abs().max())
            for label in controller.l_shadow},
        projection_base_has_bias=bool(arrays['projection_base_has_bias']),
        projection_output_delta_max_abs=float(controller.projection_bias_delta.abs().max()),
        parameter_counts=controller.parameter_counts(), constraints=controller.constraints())


def compare_frames(actual_path, reference_path):
    """Compare only matching final-evaluation identities; never an initial AEE."""
    if not reference_path.exists():
        return dict(status='reference_not_available', reference=str(reference_path))
    actual = json.loads(actual_path.read_text())
    reference = json.loads(reference_path.read_text())
    by_file = {row['file']: row for row in reference}
    rows = []
    for row in actual:
        if row['file'] not in by_file:
            continue
        expected = by_file[row['file']]
        rows.append(dict(file=row['file'],
            valid_pixels_delta=row['valid_pixels']-expected['valid_pixels'],
            aee_sum_delta=row['aee_sum']-expected['aee_sum'],
            AEE_delta=row['AEE']-expected['AEE']))
    exact = bool(rows) and all(not row['valid_pixels_delta'] and not row['aee_sum_delta']
                               and not row['AEE_delta'] for row in rows)
    return dict(status='exact_on_compared_frames' if exact else 'differences_or_no_overlap',
        reference=str(reference_path), actual_frames=len(actual), reference_frames=len(reference),
        compared_frames=len(rows), all_reference_frames_compared=len(rows) == len(reference),
        same_frame_set={row['file'] for row in actual} == set(by_file),
        max_abs_AEE_delta=max((abs(row['AEE_delta']) for row in rows), default=None),
        max_abs_aee_sum_delta=max((abs(row['aee_sum_delta']) for row in rows), default=None),
        rows=rows, scope='Per-frame valid-pixel counts and AEE sums; no training-reproducibility or all-output bitwise claim.')


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model-files', type=Path, nargs='+')
    parser.add_argument('--split', choices=('diverse', 'valid'), default='diverse')
    parser.add_argument('--count', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reference-directory', type=Path)
    parser.add_argument('--native-latent', action='store_true')
    args = parser.parse_args()
    alg = args.root/'algorithm'
    area = alg/'patch_probe'
    residual = area/'residual_consumer_probe'
    chain = residual/'projection_chain'
    latent = area/'factor_completion_20260909/latent_stage_train16'
    count = args.count if args.count is not None else (10 if args.split == 'diverse' else 825)
    paths = args.model_files or [chain/'consumer_obs_recovery64'/(axis+'.npz') for axis in AXES]
    forward_name = 'native_latent' if args.native_latent else 'training_forward'
    args.output = args.output or chain/('consumer_obs_reload_'+forward_name+'_'+args.split+str(count))
    args.output.mkdir(parents=True, exist_ok=True)
    for path in (alg, alg/'nrv_cost_probe', latent, residual):
        sys.path.insert(0, str(path))
    import run_probe as probe
    from run_bn_probe import read_names
    from evaluate_branch_control import evaluate_axis, mask_nonanchors
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
    parent_arrays = read_arrays(parent)
    conv1, sn2 = modules[R1+'.conv1.0'], modules[R1+'.sn2.spiking_neuron']
    original_conv1, original_sn2 = conv1.forward, sn2.forward
    if args.native_latent:
        pair = LatentPair(parent_arrays, conv1.weight.device, conditional=False)
    else:
        pair = TrainableLatentPair(parent_arrays, conv1.weight.device)
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
    common_hook = modules[R1+'.norm2'].register_forward_hook(delete_nonanchor)
    names = (read_names(args.data, 'valid') if args.split == 'valid' else
             json.loads((alg/'samples.json').read_text())['valid'])[:count]
    run = dict(complete=False, parent=str(parent), model_files=[str(path) for path in paths],
        split=args.split, requested_count=count, files=names,
        loading='RecoveryControl with exported W_shadow/L_shadow, learnedBN biases, explicit base-bias presence plus separate output delta; original_W is reference only.',
        latent_neuron=('Original LatentPair control, completeT10 theta-valued hard forward.' if args.native_latent else
            'Same TrainableLatentPair construction/storage clones/forward as recovery final evaluation; U/V frozen and torch.no_grad, no training.'),
        native_latent=args.native_latent,
        common_parent='Same R32 U8/VQ5 preview-only, four fixedpatchBN, r1 nonanchor normalized-branch deletion, actualPED and realcoarse successors.',
        metric='Unchanged evaluate_branch_control.evaluate_axis and full valid-pixel AEE formula.',
        numeric=getattr(RecoveryControl, 'evaluation_numeric',
            'Same constrainedFloat32 student and independentL subtraction/addition order; no combined-bias reassociation.'),
        claim=getattr(RecoveryControl, 'evaluation_claim',
            'Export/reload and AEE evaluation only; no optimizer, training, source-count or hardware speed result.'), results={})
    save_json(args.output/'run.json', run)
    controller = None
    try:
        for index, path in enumerate(paths):
            arrays = read_arrays(path)
            reference_directory = args.reference_directory or path.parent
            training_run_path = reference_directory/'result.json'
            runtime_source = None
            if training_run_path.exists():
                training_run = json.loads(training_run_path.read_text())
                torch.backends.cuda.matmul.allow_tf32 = bool(training_run['TF32_matmul'])
                torch.backends.cudnn.allow_tf32 = bool(training_run['TF32_cudnn'])
                runtime_source = str(training_run_path)
            controller = RecoveryControl(modules, arrays)
            loaded = load_recovered(controller, arrays)
            axis = loaded['axis']
            tag = f'{index:02d}_{path.stem}'
            row = dict(model_file=str(path), axis=axis, loaded=loaded,
                TF32_matmul=torch.backends.cuda.matmul.allow_tf32,
                TF32_cudnn=torch.backends.cudnn.allow_tf32,
                runtime_flags_from=runtime_source)
            run['results'][tag] = row
            save_json(args.output/'run.json', run)
            row['evaluation'] = evaluate_axis(args, model, current, names, tag,
                                              progress_tag='RECOVERED_CONSUMER_AEE')
            row['final_recovery_frame_alignment'] = compare_frames(
                args.output/(tag+'_frames.json'), reference_directory/(axis+'_frames.json'))
            save_json(args.output/'run.json', run)
            print('RELOAD_ALIGNMENT', tag, json.dumps(row['final_recovery_frame_alignment']), flush=True)
            controller.restore()
            controller = None
            functional.reset_net(model)
        run['complete'] = True
        save_json(args.output/'run.json', run)
    finally:
        if controller is not None:
            controller.restore()
        common_hook.remove()
        conv1.forward, sn2.forward = original_conv1, original_sn2
        current.pop('flow', None)
    print('DONE', json.dumps({key: value['evaluation']['AEE_frame_mean']
                               for key, value in run['results'].items()}), flush=True)


if __name__ == '__main__':
    main()
