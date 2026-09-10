"""Mean-correct the fixed group-OBS initializations on the original train4.

No optimizer updates or validation fitting. Restore the ordinary parent biases
for every axis; then recalibrate r0, r1 anchors, and projection sequentially.
The saved bias files feed the unchanged 64-step real-flow recovery budget.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from capture_chain import R0, R1, PROJECT, save_json
from train_consumer_recovery import AXES, BIAS_AXIS, RecoveryControl, read_arrays
from evaluate_bias_corrected import BiasState, collect_means, STAGES


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--initial-parameters', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.split = 'diverse'
    alg = args.root/'algorithm'
    area = alg/'patch_probe'
    residual = area/'residual_consumer_probe'
    chain = residual/'projection_chain'
    latent = area/'factor_completion_20260909/latent_stage_train16'
    args.initial_parameters = args.initial_parameters or chain/'obs_compensation_train4/initial_parameters.npz'
    args.output = args.output or chain/'obs_bias_calibration_train4'
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = read_arrays(chain/'consumer_pruning_diverse10/parameters.npz')
    initial = read_arrays(args.initial_parameters)
    ordinary_bias = read_arrays(chain/'consumer_bias_corrected_diverse10/ordinary_rank32_biases.npz')
    teacher = json.loads((chain/'consumer_bias_corrected_diverse10/teacher_means.json').read_text())
    names = [str(name) for name in arrays['selected_train_frames']]
    if ordinary_bias['calibration_files'].tolist() != names:
        raise ValueError('Use the same four original teacher calibration frames.')
    for path in (alg, alg/'nrv_cost_probe', latent, residual):
        sys.path.insert(0, str(path))
    import run_probe as probe
    from adapter import install_latent_factor
    from evaluate_branch_control import mask_nonanchors

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
    run = dict(complete=False, initial_parameters=str(args.initial_parameters),
        teacher_means='Existing ordinary-rank32 train4 means; same parent and domains.',
        train_files=names, axes={}, optimizer_updates=0,
        method='Fixed group-OBS weight initialization followed by the same sequential empirical mean correction.',
        unchanged='Source mask, rank32 U/V, BN gain and statistics, theta, T10 and all other network parameters.',
        scope='Calibration only; no held-out fitting or full OBC/variance-correction claim.')
    save_json(args.output/'run.json', run)
    try:
        controller = RecoveryControl(modules, arrays)
        controller.initial_parameters = initial
        bias_state = BiasState(modules)
        for axis in AXES:
            controller.initialize_axis(axis, ordinary_bias)
            stages = []
            for index, stage in enumerate(STAGES):
                measured = collect_means(args, model, modules, current, controller,
                    names, STAGES[:index+1], axis+'/'+stage)
                correction = bias_state.apply(stage, teacher[stage]['mean'], measured[stage]['mean'])
                stages.append(dict(stage=stage, observations_before_correction=measured,
                    correction=correction))
            bias_file = BIAS_AXIS[axis]+'_biases.npz'
            projection = modules[PROJECT+'.conv_res']
            projection_bias = (np.zeros(96, np.float32) if projection.bias is None
                               else projection.bias.detach().cpu().numpy())
            np.savez_compressed(args.output/bias_file,
                r0_bias=controller.bn['r0'].bias.detach().cpu().numpy(),
                r1_bias=controller.bn['r1'].bias.detach().cpu().numpy(),
                projection_bias=projection_bias,
                projection_has_bias=np.array(projection.bias is not None),
                calibration_files=np.asarray(names), axis=np.array(axis))
            save_json(args.output/(axis+'_calibration.json'), dict(complete=True, stages=stages))
            run['axes'][axis] = dict(biases=bias_file,
                calibration=axis+'_calibration.json',
                delta_max_abs={row['stage']:row['correction']['applied_delta_max_abs'] for row in stages},
                constraints=controller.constraints())
            save_json(args.output/'run.json', run)
            print('OBS_CALIBRATION_DONE', axis, json.dumps(run['axes'][axis]['delta_max_abs']), flush=True)
        run['complete'] = True
        save_json(args.output/'run.json', run)
    finally:
        common_hook.remove()
        if bias_state is not None:
            bias_state.restore()
        if controller is not None:
            controller.restore()
        conv1.forward, sn2.forward = originals['conv1_forward'], originals['neuron_forward']
    print('DONE', str(args.output/'run.json'), flush=True)


if __name__ == '__main__':
    main()
