"""Four ordinary U-only W4/W8 controls on original R32/CUDA parents."""
from pathlib import Path
import argparse
import csv
import sys
import numpy as np
from run_combinations import Activity, save, rows

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    assert rows(HERE/'combinations/run.json')['complete']
    assert rows(HERE/'representations/aee/run.json')['complete']
    base = args.root; op = base/'open_fusion_execution'; top = HERE/'weight_controls/aee'
    patch = base/'algorithm/patch_probe'; res = patch/'residual_consumer_probe'
    chain = res/'projection_chain'; lift = chain/'fast_temporal_recovery_lifting40'
    latent = patch/'factor_completion_20260909/latent_stage_train16'
    for directory in [chain, res, latent, base/'algorithm', base/'algorithm/nrv_cost_probe', HERE/'weight_controls/package']:
        sys.path.insert(0, str(directory))
    import torch
    import run_probe as probe
    from lowbit_ped_adapter import install, check_fixture
    from flow_backward_probe import TrainableLatentPair, read_arrays
    from capture import BLOCK, SOURCE_SN
    from evaluate_branch_control import evaluate_axis, mask_nonanchors
    from train_shared_temporal_recovery import SharedTemporalControl
    from lifting_temporal_control import LiftingTemporalControl
    from fixed_temporal_coordinates import FixedTemporalForward
    from fixed_lifting_coordinates import FixedLiftingForward
    from spikingjelly.activation_based import functional
    names = rows(base/'algorithm/samples.json')['valid']
    reference = rows(op/'new_interface_selection/aee_rebase/results/run.json')
    with (HERE/'source_nb0_valid825.csv').open() as stream:
        nb0 = {row['file']: row for row in csv.DictReader(stream)}
    args.split = 'diverse'; system = probe.load_system(args)
    model, modules, _, current, sources, _, _, _ = system
    probe.install_sources(system, sources); current['count_codes'] = False
    calibration = torch.load(patch/'patch_train_calibration.pt', map_location='cpu', weights_only=False)
    for path, values in calibration.items():
        bn = modules[path]; bn.track_running_stats = True
        bn.running_mean, bn.running_var = values['mean'].to(bn.weight), values['var'].to(bn.weight)
    model.eval(); model.requires_grad_(False)
    flags = rows(chain/'affine_shared_temporal_control_diverse10/run.json')
    torch.backends.cuda.matmul.allow_tf32 = bool(flags['TF32_matmul'])
    torch.backends.cudnn.allow_tf32 = bool(flags['TF32_cudnn'])
    pair = TrainableLatentPair(read_arrays(latent/'flow_recovery64/preview_only/shared48_u8_vq5.npz'), modules[SOURCE_SN].weight.device)
    pair.u.requires_grad_(False); pair.v.requires_grad_(False)
    conv1, sn2 = modules[BLOCK+'.conv1.0'], modules[BLOCK+'.sn2.spiking_neuron']
    old_conv1, old_sn2 = conv1.forward, sn2.forward
    conv1.forward, sn2.forward = pair.conv_forward, pair.neuron_forward
    masks = {}
    def nonanchor(module, inputs, output):
        key = (tuple(output.shape[-2:]), output.device)
        if key not in masks:
            mask = torch.zeros(output.shape[-2:], device=output.device, dtype=torch.bool)
            mask[::2, ::2] = True; masks[key] = mask
        return mask_nonanchors(output, masks[key])
    hook = modules[BLOCK+'.norm2'].register_forward_hook(nonanchor)
    common = (modules, read_arrays(res/'rank_control_parameters.npz'), read_arrays(chain/'rank32_diverse10/parameters.npz'))
    with np.load(HERE/'weight_controls/package/replay_fixture_probe_only.npz') as z:
        fixture = {key: z[key] for key in z.files}
    activity = Activity(model, modules)
    report = dict(complete=False, training=False, full_valid825=False, files=names, axes={},
        parent='Original R32 plus CUDA BN for each student. R24 and onepass are absent.',
        source_package='stage_20260912/weight_compensation: lowbit_ped_adapter and lowbit_gpu_parameters',
        numeric='U-only code times q16 row scale, exact offline expansion; original U RNE16/sat24, V q16/e15, V RNE15/sat24 and bias unchanged.',
        execution='Functional low-bit-weight check, not a compressed CUDA kernel or hardware cycle measurement.',
        policy='Matched same-population NB0, no +0.005 rule; no automatic825 or training for these controls.',
        config=str(args.config), checkpoint=str(args.checkpoint), new_X=False)
    save(top/'run.json', report)
    try:
        for axis in ('ordinary', 'lifting_raw'):
            if axis == 'ordinary':
                identity = 'identity_permuted_base'
                student = chain/'temporal_structured_recovery/stage128x256'/f'{identity}.npz'
                controller = SharedTemporalControl(*common, fit={})
                controller.load_saved(identity, read_arrays(student))
                helper = FixedTemporalForward(controller, pair.temporal.theta)
            else:
                identity = 'fast_raw_diagonal'; student = lift/'stage320'/f'{identity}.npz'
                init = rows(lift/'initialization.json')
                controller = LiftingTemporalControl(*common, source_fit=init['source_fit'], consumer_fits=init['consumer_fits'], basis_lifting=init['basis_lifting'])
                controller.load_saved(identity, read_arrays(student))
                helper = FixedLiftingForward(controller, pair.temporal.theta)
            original = helper.export_constants()
            with np.load(HERE/'weight_controls/package'/f'{axis}_lowbit_gpu_parameters.npz') as z:
                package = {key: z[key] for key in z.files}
            ar = dict(complete=False, identity=identity, student=str(student), modes={})
            report['axes'][axis] = ar
            try:
                ar['actual_helper_fixtures'] = check_fixture(helper, package, original, fixture, axis)
                print('WEIGHT_FIXTURES_EXACT', axis, flush=True)
                args.output = top/axis/'baseline_check'; args.output.mkdir(parents=True, exist_ok=True)
                with torch.no_grad():
                    evaluate_axis(args, model, current, names[:1], 'original32', progress_tag=f'WEIGHT_BASE {axis}')
                measured = rows(args.output/'original32_frames.json')[0]
                expected = reference['axes'][axis]['modes']['original32']['frames'][0]
                assert measured['AEE'] == expected['AEE'] and measured['valid_pixels'] == expected['valid_pixels']
                ar['baseline_first_exact'] = True
                ar['baseline_diverse10'] = reference['axes'][axis]['modes']['original32']['summary']['AEE_frame_mean']
                for mode in ('W8', 'W4'):
                    install(helper, package, mode, original)
                    helper.frames.clear(); activity.start(names)
                    args.output = top/axis/mode; args.output.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(args.output/'deployed_constants.npz', **helper.export_constants())
                    with torch.no_grad():
                        summary = evaluate_axis(args, model, current, names, mode, progress_tag=f'WEIGHT_AEE {axis}')
                    measured = rows(args.output/(mode+'_frames.json'))
                    assert all(int(row['valid_pixels']) == int(float(nb0[row['file']]['valid_pixels'])) for row in measured)
                    save(args.output/'activity_summary.json', activity.finish(helper.frames, []))
                    mean = float(np.mean([float(nb0[name]['AEE']) for name in names]))
                    ar['modes'][mode] = dict(summary=summary, NB0_AEE=mean,
                        better_than_NB0=summary['AEE_frame_mean'] < mean,
                        delta_NB0=summary['AEE_frame_mean']-mean,
                        delta_parent=summary['AEE_frame_mean']-ar['baseline_diverse10'],
                        holdout9_AEE=float(np.mean([row['AEE'] for row in measured if row['file'] != names[0]])),
                        no_valid825=True, no_hardware_speedup=True)
                    save(top/'run.json', report)
                ar['complete'] = True; save(top/'run.json', report)
            finally:
                activity.active = False; helper.restore(); controller.restore(); functional.reset_net(model)
                current.pop('flow', None); torch.cuda.empty_cache()
        report['complete'] = True; save(top/'run.json', report)
    finally:
        activity.restore(); hook.remove(); conv1.forward, sn2.forward = old_conv1, old_sn2
    print('WEIGHT_CONTROLS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
