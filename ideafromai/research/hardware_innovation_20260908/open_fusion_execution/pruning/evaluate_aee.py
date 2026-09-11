"""No-training A800 accuracy probe for fixed physical-H8 phase pruning.

The real sn2 theta*g return is masked at its original source-pixel phase.
Existing fixed Conv2, raw I24 residual, gate/PED consumers, remaining dynamic
normalization and actual coarse network execute normally. GPU elapsed time is
not a hardware performance measurement. Masks are selected on one old frame;
the existing diverse10 list is reused and its calibration overlap is reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mask', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--axes', nargs='+', choices=['ordinary', 'lifting_raw'],
                        default=['ordinary', 'lifting_raw'])
    parser.add_argument('--modes', nargs='+', default=['unpruned', 'global_group2', 'phase_joint'],
                        help='unpruned and/or keys in the supplied mask JSON.')
    args = parser.parse_args()
    base = args.root
    patch = base/'algorithm/patch_probe'
    res = patch/'residual_consumer_probe'
    chain = res/'projection_chain'
    lift = chain/'fast_temporal_recovery_lifting40'
    stage = lift/'schedule_compare_same_port'
    latent = patch/'factor_completion_20260909/latent_stage_train16'
    for directory in [stage, chain, res, latent, base/'algorithm', base/'algorithm/nrv_cost_probe']:
        sys.path.insert(0, str(directory))
    import torch
    import run_probe as probe
    from capture_inputs import save_json
    from flow_backward_probe import TrainableLatentPair, read_arrays
    from capture import BLOCK, SOURCE_SN
    from evaluate_branch_control import evaluate_axis, mask_nonanchors
    from train_shared_temporal_recovery import SharedTemporalControl
    from lifting_temporal_control import LiftingTemporalControl
    from fixed_temporal_coordinates import FixedTemporalForward
    from fixed_lifting_coordinates import FixedLiftingForward
    from spikingjelly.activation_based import functional

    args.mask = args.mask or base/'open_fusion_execution/review/phase_group8_masks.json'
    args.output = args.output or base/'open_fusion_execution/pruning/aee_results'
    args.output.mkdir(parents=True, exist_ok=True)
    top_output = args.output
    masks = json.loads(args.mask.read_text())
    previous_path = base/'open_fusion_execution/pruning/aee_results/run.json'
    previous = None if 'unpruned' in args.modes else json.loads(previous_path.read_text())
    old_frames = json.loads((chain/'temporal_structured_recovery/fixed_sources_diverse10/identity_permuted_base_frames.json').read_text())
    names = [r['file'] for r in old_frames]
    calibration_frame = 'zurich_city_09_a_0001.npy'
    smoke_names = [next(n for n in names if n != calibration_frame)]
    args.split = 'diverse'
    system = probe.load_system(args)
    model, modules, _, current, sources, _, _, _ = system
    probe.install_sources(system, sources)
    current['count_codes'] = False
    calibration = torch.load(patch/'patch_train_calibration.pt', map_location='cpu', weights_only=False)
    for path, values in calibration.items():
        bn = modules[path]
        bn.track_running_stats = True
        bn.running_mean, bn.running_var = values['mean'].to(bn.weight), values['var'].to(bn.weight)
    model.eval()
    model.requires_grad_(False)
    flags = json.loads((chain/'affine_shared_temporal_control_diverse10/run.json').read_text())
    torch.backends.cuda.matmul.allow_tf32 = bool(flags['TF32_matmul'])
    torch.backends.cudnn.allow_tf32 = bool(flags['TF32_cudnn'])
    parent = latent/'flow_recovery64/preview_only/shared48_u8_vq5.npz'
    pair = TrainableLatentPair(read_arrays(parent), modules[SOURCE_SN].weight.device)
    pair.u.requires_grad_(False)
    pair.v.requires_grad_(False)
    conv1, sn2 = modules[BLOCK+'.conv1.0'], modules[BLOCK+'.sn2.spiking_neuron']
    old_conv1, old_sn2 = conv1.forward, sn2.forward
    conv1.forward, sn2.forward = pair.conv_forward, pair.neuron_forward
    anchor_masks = {}
    def nonanchor(module, inputs, output):
        key = (tuple(output.shape[-2:]), output.device)
        if key not in anchor_masks:
            anchor = torch.zeros(output.shape[-2:], device=output.device, dtype=torch.bool)
            anchor[::2, ::2] = True
            anchor_masks[key] = anchor
        return mask_nonanchors(output, anchor_masks[key])
    common_hook = modules[BLOCK+'.norm2'].register_forward_hook(nonanchor)
    common = (modules, read_arrays(res/'rank_control_parameters.npz'),
              read_arrays(chain/'rank32_diverse10/parameters.npz'))
    report = dict(complete=False, scope=__doc__, axes={}, files=names,
        smoke_files=smoke_names, mask_source=str(args.mask),
        mask_calibration_frame=calibration_frame,
        calibration_overlap=[n for n in names if n == calibration_frame],
        holdout_files=[n for n in names if n != calibration_frame],
        modes=args.modes,
        unpruned_reference=('Measured in this run' if previous is None else str(previous_path)),
        parent=str(parent), training=False, full_valid825=False,
        frozen_ep34_identity=False,
        source_phase='2*(source_y % 2)+(source_x % 2); 1 in mask means zero the complete T10 theta*g output for that channel.',
        preserved='Raw I24; fixed helper signed24 arithmetic; both updated-I24 gate and continuous PED; complete original coarse exit and unchanged AEE formula.',
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32,
        TF32_cudnn=torch.backends.cudnn.allow_tf32)
    save_json(top_output/'run.json', report)
    try:
        for label in args.axes:
            started = time.monotonic()
            if label == 'ordinary':
                identity = 'identity_permuted_base'
                student = chain/'temporal_structured_recovery/stage128x256'/f'{identity}.npz'
                controller = SharedTemporalControl(*common, fit={})
                controller.load_saved(identity, read_arrays(student))
                helper = FixedTemporalForward(controller, pair.temporal.theta)
            else:
                identity = 'fast_raw_diagonal'
                student = lift/'stage320'/f'{identity}.npz'
                init = json.loads((lift/'initialization.json').read_text())
                controller = LiftingTemporalControl(*common, source_fit=init['source_fit'],
                    consumer_fits=init['consumer_fits'], basis_lifting=init['basis_lifting'])
                controller.load_saved(identity, read_arrays(student))
                helper = FixedLiftingForward(controller, pair.temporal.theta)
            axis_report = dict(student=str(student), identity=identity, stages={})
            report['axes'][label] = axis_report
            try:
                for stage_name, selected_names in [('smoke', smoke_names), ('diverse10', names)]:
                    stage_report = {}
                    axis_report['stages'][stage_name] = stage_report
                    for mode in report['modes']:
                        args.output = top_output/label/stage_name/mode
                        args.output.mkdir(parents=True, exist_ok=True)
                        counts = []
                        mask = None if mode == 'unpruned' else torch.as_tensor(
                            masks[label][mode]['mask_uint8'], dtype=torch.bool, device=pair.u.device)
                        def prune(module, inputs, output):
                            # Clone preserves the real producer's own saved activation.
                            # Both downstream consumers read the returned masked value.
                            result = output.clone() if mask is not None else output
                            removed = 0
                            before = int(output.ne(0).sum())
                            if mask is not None:
                                for phase in range(4):
                                    channels = torch.nonzero(mask[phase], as_tuple=False).flatten()
                                    y, x = divmod(phase, 2)
                                    removed += int(result[:, :, channels, y::2, x::2].ne(0).sum())
                                    result[:, :, channels, y::2, x::2] = 0
                            counts.append(dict(nonzero_before=before, nonzero_removed=removed,
                                               nonzero_after=before-removed))
                            return result
                        hook = sn2.register_forward_hook(prune)
                        try:
                            with torch.no_grad():
                                summary = evaluate_axis(args, model, current, selected_names,
                                    mode, progress_tag=f'PHASE_H8_AEE {label} {stage_name}')
                            frames = json.loads((args.output/f'{mode}_frames.json').read_text())
                            for frame, count in zip(frames, counts):
                                frame['sn2_activity'] = count
                            save_json(args.output/'frames_with_activity.json', frames)
                            if mode == 'unpruned':
                                groups = []
                            else:
                                groups = masks[label][mode].get('drop_groups')
                                if groups is None:
                                    full_groups = np.asarray(masks[label][mode]['mask_uint8']).reshape(4, 12, 8).all(2)
                                    groups = [np.flatnonzero(row).tolist() for row in full_groups]
                            stage_report[mode] = dict(summary=summary, frames=frames, drop_groups=groups)
                            save_json(top_output/'run.json', report)
                        finally:
                            hook.remove()
                            functional.reset_net(model)
                            current.pop('flow', None)
                    baseline = (stage_report['unpruned']['frames'] if previous is None else
                                previous['axes'][label]['stages'][stage_name]['unpruned']['frames'])
                    reference_by_name = {row['file']:row for row in baseline}
                    first_by_name = {row['file']:row for row in stage_report[args.modes[0]]['frames']}
                    for mode in report['modes']:
                        rows = stage_report[mode]['frames']
                        paired = [dict(file=row['file'], baseline_AEE=reference_by_name[row['file']]['AEE'], AEE=row['AEE'],
                            delta_AEE=row['AEE']-reference_by_name[row['file']]['AEE'],
                            first_mode=args.modes[0], delta_first_mode_AEE=row['AEE']-first_by_name[row['file']]['AEE']) for row in rows]
                        held = [r for r in paired if r['file'] != calibration_frame]
                        stage_report[mode]['paired'] = paired
                        stage_report[mode]['delta_frame_mean'] = float(np.mean([r['delta_AEE'] for r in paired]))
                        stage_report[mode]['holdout_delta_frame_mean'] = float(np.mean([r['delta_AEE'] for r in held]))
                        stage_report[mode]['delta_first_mode_frame_mean'] = float(np.mean([r['delta_first_mode_AEE'] for r in paired]))
                        stage_report[mode]['holdout_delta_first_mode_frame_mean'] = float(np.mean([r['delta_first_mode_AEE'] for r in held]))
                    save_json(top_output/'run.json', report)
                axis_report['complete'] = True
                axis_report['wall_seconds'] = time.monotonic()-started
            finally:
                helper.restore()
                controller.restore()
                functional.reset_net(model)
                current.pop('flow', None)
                torch.cuda.empty_cache()
            save_json(top_output/'run.json', report)
        report['complete'] = True
        save_json(top_output/'run.json', report)
    finally:
        common_hook.remove()
        conv1.forward, sn2.forward = old_conv1, old_sn2
    print('PHASE_H8_DONE', json.dumps({a:{m:r['summary']['AEE_frame_mean']
        for m,r in v['stages']['diverse10'].items()} for a,v in report['axes'].items()}), flush=True)


if __name__ == '__main__':
    main()
