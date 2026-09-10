"""Three fixed diverse10 controls on the ORIGINAL, unrecovered control3.

Read the saved train4-fit A/bias; change only proj.sn A/bias. Original source A,
preview Conv1, Conv2R16, PEDR32, four fixed BNs, theta/center and real successors
stay unchanged. No fitting, training, rank selection, coordinate rewrite or
64/128-frame recovery parameters. Execute native dense A for this accuracy
comparison; a factorized/structured hardware implementation is not measured.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

from evaluate_shared_temporal_control import read_arrays, frame_comparison

AXES = ('ordinary_control3', 'identity_permuted_base', 'identity_permuted_joint_r2')


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parameters', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.split, args.count = 'diverse', 10
    alg, area = args.root/'algorithm', args.root/'algorithm/patch_probe'
    residual = area/'residual_consumer_probe'
    chain = residual/'projection_chain'
    latent = area/'factor_completion_20260909/latent_stage_train16'
    args.parameters = args.parameters or chain/'shared_temporal_lowrank_control.npz'
    args.output = args.output or chain/'temporal_lowrank_control_diverse10'
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = read_arrays(args.parameters)
    previous = chain/'shared_temporal_control_diverse10'
    runtime = json.loads((previous/'run.json').read_text())
    for path in (alg, alg/'nrv_cost_probe', latent, residual):
        sys.path.insert(0, str(path))
    import run_probe as probe
    from adapter import install_latent_factor
    from capture import BLOCK, SOURCE_SN, PROJECT, CONSUMER_SN
    from capture_chain import save_json
    from evaluate_rank_control import RankConvControl
    from evaluate_branch_control import evaluate_axis, mask_nonanchors

    system = probe.load_system(args)
    model, modules, _, current, sources, _, _, _ = system
    probe.install_sources(system, sources)
    current['count_codes'] = False
    fixed = torch.load(area/'patch_train_calibration.pt', map_location='cpu', weights_only=False)
    for name, values in fixed.items():
        bn = modules[name]
        bn.track_running_stats = True
        bn.running_mean, bn.running_var = values['mean'].to(bn.weight), values['var'].to(bn.weight)
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = bool(runtime['TF32_matmul'])
    torch.backends.cudnn.allow_tf32 = bool(runtime['TF32_cudnn'])
    parent = latent/'flow_recovery64/preview_only/shared48_u8_vq5.npz'
    conv1, sn2 = modules[BLOCK+'.conv1.0'], modules[BLOCK+'.sn2.spiking_neuron']
    _, originals = install_latent_factor(conv1, sn2, parent, conditional=False)
    conv2, projection, neuron = modules[BLOCK+'.conv2.0'], modules[PROJECT+'.conv_res'], modules[CONSUMER_SN]
    source = modules[SOURCE_SN]
    if not torch.equal(source.weight.cpu(), torch.from_numpy(arrays['source_A']).float()):
        raise ValueError('Original control3 source A does not match the saved train4-fit parent.')
    old_a, old_b = neuron.weight.clone(), neuron.bias.clone()
    control = RankConvControl(conv2, projection, read_arrays(residual/'rank_control_parameters.npz'))
    control.axis = 'uniform_rank16'
    conv2.forward = control.forward
    original_projection = projection.forward
    ped = read_arrays(chain/'rank32_diverse10/parameters.npz')
    first = torch.as_tensor(ped['U'], device=projection.weight.device, dtype=projection.weight.dtype)[:, :, None, None]
    second = torch.as_tensor(ped['V'], device=projection.weight.device, dtype=projection.weight.dtype)[:, :, None, None]
    projection.forward = lambda x: F.conv2d(F.conv2d(x[:, :, ::2, ::2], first), second, projection.bias)
    masks = {}
    def delete_nonanchor(module, inputs, output):
        key = (output.shape[-2:], output.device)
        if key not in masks:
            mask = torch.zeros(output.shape[-2:], device=output.device, dtype=torch.bool)
            mask[::2, ::2] = True
            masks[key] = mask
        return mask_nonanchors(output, masks[key])
    hook = modules[BLOCK+'.norm2'].register_forward_hook(delete_nonanchor)
    names = json.loads((alg/'samples.json').read_text())['valid'][:10]
    run = dict(complete=False, parent=str(parent), parameters=str(args.parameters), files=names,
        common='Original unrecovered control3: original-sourceA + frozen native-bool preview + Conv2R16 + nonanchor whole BN2 branch deletion + PEDR32; original BN biases/gains and fixed four BN statistics.',
        parameter_keys={AXES[0]: ['original_proj_A', 'original_proj_bias'],
                        AXES[1]: [AXES[1]+'_A', AXES[1]+'_bias'],
                        AXES[2]: [AXES[2]+'_A', AXES[2]+'_bias']},
        numeric='Saved Float64 A/bias round once to the native FP32 parameters; native fullT10 addmm/official theta*g forward. No low-rank factorized evaluation or shared-coordinate claim.',
        theta=dict(source=float(source.thresh), projection=float(neuron.thresh)),
        center_modes=dict(source=source.center_mode, projection=neuron.center_mode),
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32, TF32_cudnn=torch.backends.cudnn.allow_tf32,
        claim='Fixed ordinary accuracy controls only; no new training, fitting, validation choice or hardware cost.', results={})
    save_json(args.output/'run.json', run)
    try:
        for axis in AXES:
            ka, kb = run['parameter_keys'][axis]
            neuron.weight.copy_(torch.as_tensor(arrays[ka], device=neuron.weight.device, dtype=neuron.weight.dtype))
            neuron.bias.copy_(torch.as_tensor(arrays[kb], device=neuron.bias.device, dtype=neuron.bias.dtype).reshape_as(neuron.bias))
            run['results'][axis] = evaluate_axis(args, model, current, names, axis,
                progress_tag='TEMPORAL_LOWRANK_CONTROL_AEE')
            if axis == AXES[0]:
                run['ordinary_control3_alignment'] = frame_comparison(args.output/(axis+'_frames.json'),
                    previous/'rank16_delete_nonanchor_ped32_frames.json')
                run['ordinary_control3_alignment']['scope'] = 'Original unrecovered control3 expected diverse10 AEE1.2363524808; no inherited recovery/825 result.'
            save_json(args.output/'run.json', run)
        run['complete'] = True
        save_json(args.output/'run.json', run)
    finally:
        hook.remove()
        neuron.weight.copy_(old_a)
        neuron.bias.copy_(old_b)
        projection.forward = original_projection
        conv2.forward = control.original_forward
        conv1.forward, sn2.forward = originals['conv1_forward'], originals['neuron_forward']
        current.pop('flow', None)
    print('DONE', json.dumps({axis: value['AEE_frame_mean'] for axis,value in run['results'].items()}), flush=True)


if __name__ == '__main__':
    main()
