"""Paired real-GT recovery of the fixed native-T10 factor structure.

Two axes, same source basis and common control3 initialization. Source readout
fits full original train4 I moments; consumer readouts fit the already saved
full control3 x moments. Permutations are assigned once, never from validation.
Forward executes the factors, signed readouts and (for shared) actual inverse.
Fixed64 train16 updates then the existing128-frame/256-update schedule, fresh
Adam1e-4 at each stage. Only stage endpoints are evaluated; no checkpoint pick.
This is a new FP32 student, not a fixed-point or hardware performance result.
--basis-kind lifting40 starts from the saved fixed train4 moment-fit endpoint,
reuses the existing source-I moments and fits each consumer readout once on
the old control3 x moments. It trains forty continuous lifting coefficients
and their exact factorwise reverse, not signs or a dense inverse. Its separate
output directory never overwrites the original fast_sign experiment.
--fixed-lifting is evaluation-only: reload lifting40 and execute the common
signed24/f14 state, signed16/f12 lifting and signed48 integer-dot definition.
It writes separate AEE, activity, actual constants and full-frame ranges.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
RES = HERE.parent
LATENT = RES.parent/'factor_completion_20260909/latent_stage_train16'
for path in (LATENT, RES, HERE):
    sys.path.insert(0, str(path))
from adapter import fp32_matmul
from flow_backward_probe import TrainableLatentPair, read_arrays
from capture import BLOCK, SOURCE_SN, PROJECT, CONSUMER_SN
from capture_chain import save_json
from train_consumer_recovery import read_train16, release_model_graphs
from evaluate_affine_shared_temporal_control import input_moments, merge_moments
from fast_temporal_basis import FastTemporalBasis, fit_readout

AXES = ('fast_raw_diagonal', 'fast_shared')
SEED, LR = 912, 1e-4


class AtSource(Exception):
    pass


@torch.no_grad()
def calibrate_source(model, source, args, train, functional, current, input_frame):
    """One full I-domain moment pass; abort before changing source outputs."""
    moments, rows = None, []

    def observe(module, inputs):
        nonlocal moments
        item = input_moments(inputs[0])
        moments = merge_moments(moments, item)
        rows.append(dict(vectors=item[0], mean=item[1], covariance=item[2]/item[0]))
        raise AtSource()

    handle = source.register_forward_pre_hook(observe)
    try:
        for name in train[:4]:
            functional.reset_net(model)
            current.pop('flow', None)
            x, _, _ = input_frame(args.data, name, targets=False)
            try:
                model(x)
            except AtSource:
                pass
            rows[-1]['file'] = name
            del x
            print('FAST_SOURCE_MOMENTS', name, rows[-1]['vectors'], flush=True)
    finally:
        handle.remove()
    n, mean, moment = moments
    return dict(vectors=n, mean=mean, covariance=moment/n, frames=rows,
                scope='All C96/H240/W320 vectors, all T10; first4 original train16 frames, FP64 centered C8 chunks. No downstream or validation fitting.')


def lifting_initialization(chain, affine):
    """Saved source fit; exactly one scalar LS/assignment per consumer domain."""
    from lifting_temporal_basis import LiftingTemporalBasis
    original = chain/'fast_temporal_recovery/initialization.json'
    old = json.loads(original.read_text())
    parameter_file = chain/'fast_temporal_recovery/block_fit/parameters.npz'
    block = read_arrays(parameter_file)
    coefficients = block['lifting40_block_parameters'].astype(np.float32)
    structure = LiftingTemporalBasis(coefficients)
    # Fit against the real matrix of the actual stored FP32 coefficients.
    # Source gain/P/bias remain the saved endpoint, rounded once for execution.
    basis = structure.dense_matrix(dtype=torch.float64).detach().numpy()
    source_fit = dict(row_gain=block['lifting40_row_gain'],
        row_permutation=block['lifting40_permutation'], bias=block['lifting40_bias'],
        rule='Saved fixed train4 moment-fit final lifting40 readout; no new source fitting, capture or validation selection.')
    consumer_fits = {axis: fit_readout(affine['mean_x'], affine['covariance_x'],
        affine['original_proj_A'], affine['original_proj_bias'],
        basis if axis == 'fast_shared' else np.eye(10)) for axis in AXES}
    return dict(basis_kind='lifting40', basis_lifting=coefficients, initial_basis=basis,
        source_moments=old['source_moments'], source_moments_source=str(original),
        source_fit=source_fit, consumer_fits=consumer_fits,
        fitted_lifting_source=str(parameter_file), initial_conditioning=structure.conditioning(),
        consumer_moments_source=str(chain/'affine_shared_temporal_control_diverse10/parameters.npz'),
        consumer_moments_scope='Original control3 full train4 x-domain, separate from source-I. One scalar LS+assignment for raw I and one for current lifting B; initialization only.',
        numeric='Saved lifting/gains/bias roundFP32 for execution; consumer LS uses F64 real B from these FP32 coefficients. Actual factorwise network AEE must be measured.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--evaluate-saved', type=Path)
    parser.add_argument('--fixed-lifting', action='store_true',
                        help='Evaluation-only lifting40: common fixed24/f14 states and q16/f12 lifting coefficients.')
    parser.add_argument('--basis-kind', choices=('fast_sign', 'lifting40'),
                        help='Default fast_sign for training; infer from saved initialization on reload.')
    parser.add_argument('--split', choices=('diverse', 'valid'), default='diverse')
    parser.add_argument('--count', type=int, default=10)
    args = parser.parse_args()
    alg, area = args.root/'algorithm', args.root/'algorithm/patch_probe'
    residual, latent = area/'residual_consumer_probe', area/'factor_completion_20260909/latent_stage_train16'
    chain = residual/'projection_chain'
    initialization = None
    if args.evaluate_saved is not None:
        initialization = json.loads((args.evaluate_saved.parent/'initialization.json').read_text())
        saved_kind = initialization.get('basis_kind', 'fast_sign')
        if args.basis_kind is not None and args.basis_kind != saved_kind:
            parser.error('--basis-kind differs from the saved factor family.')
        args.basis_kind = saved_kind
    else:
        args.basis_kind = args.basis_kind or 'fast_sign'
    if args.fixed_lifting and (args.evaluate_saved is None or args.check_only or args.basis_kind != 'lifting40'):
        parser.error('--fixed-lifting requires --evaluate-saved lifting40 and cannot train or --check-only.')
    suffix = '_lifting40' if args.basis_kind == 'lifting40' else ''
    args.output = args.output or (args.evaluate_saved.parent/(('fixed_lifting_' if args.fixed_lifting else 'reload_')+args.split+str(args.count))
        if args.evaluate_saved is not None else chain/(('fast_temporal_check' if args.check_only else 'fast_temporal_recovery')+suffix))
    args.output.mkdir(parents=True, exist_ok=True)
    for path in (alg, alg/'nrv_cost_probe', latent, residual):
        sys.path.insert(0, str(path))
    import run_probe as probe
    from run_bn_probe import input_frame, read_names
    from evaluate_stage2_deployment import CoarseReady
    from evaluate_branch_control import evaluate_axis, mask_nonanchors
    from spikingjelly.activation_based import functional
    from fast_temporal_control import FastTemporalControl

    train = read_train16(latent/'flow_train_list.json')
    plan = json.loads((chain/'shared_temporal_recovery128x256/train128.json').read_text())
    rng = np.random.default_rng(SEED)
    schedule64 = [train[int(i)] for i in np.concatenate([rng.permutation(16) for _ in range(4)])]
    schedule256 = plan['train_schedule']
    names = (read_names(args.root.parent/'sdformer_codex/SDformer/data/Datasets/DSEC/saved_flow_data', 'valid')
             if args.split == 'valid' else json.loads((alg/'samples.json').read_text())['valid'])[:args.count]
    if args.evaluate_saved is None and (args.split != 'diverse' or args.count != 10):
        parser.error('Training uses the fixed diverse10 endpoints.')

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
    model.requires_grad_(False)
    affine_run = json.loads((chain/'affine_shared_temporal_control_diverse10/run.json').read_text())
    torch.backends.cuda.matmul.allow_tf32 = bool(affine_run['TF32_matmul'])
    torch.backends.cudnn.allow_tf32 = bool(affine_run['TF32_cudnn'])
    affine = read_arrays(chain/'affine_shared_temporal_control_diverse10/parameters.npz')
    if train[:4] != affine_run['train_files']:
        raise ValueError('The existing consumer moments must use the same first4 training frames.')

    initialization_file = (args.evaluate_saved.parent if args.evaluate_saved is not None else args.output)/'initialization.json'
    if args.evaluate_saved is None:
        if args.basis_kind == 'lifting40':
            initialization = lifting_initialization(chain, affine)
        else:
            moments = calibrate_source(model, modules[SOURCE_SN], args, train, functional, current, input_frame)
            basis = FastTemporalBasis(dtype=torch.float64).dense_matrix().numpy()
            source_fit = fit_readout(moments['mean'], moments['covariance'], affine['source_A'], affine['source_bias'], basis)
            consumer_fits = {axis: fit_readout(affine['mean_x'], affine['covariance_x'],
                affine['original_proj_A'], affine['original_proj_bias'],
                basis if axis == 'fast_shared' else np.eye(10)) for axis in AXES}
            initialization = dict(basis_kind='fast_sign', source_moments=moments, source_fit=source_fit, consumer_fits=consumer_fits,
                consumer_moments_source=str(chain/'affine_shared_temporal_control_diverse10/parameters.npz'),
                consumer_moments_scope='Original control3 full train4 x-domain; source basis changes its descendants, so this is initialization, not restored-network calibration.',
                initial_basis=basis, signs='All+ hard forward; logits0.01 in both axes; no sign/permutation search or validation choice.')
        save_json(initialization_file, initialization)
    source_fit, consumer_fits = initialization['source_fit'], initialization['consumer_fits']

    parent = latent/'flow_recovery64/preview_only/shared48_u8_vq5.npz'
    pair = TrainableLatentPair(read_arrays(parent), modules[SOURCE_SN].weight.device)
    pair.u.requires_grad_(False)
    pair.v.requires_grad_(False)
    conv1, sn2 = modules[BLOCK+'.conv1.0'], modules[BLOCK+'.sn2.spiking_neuron']
    original_conv1, original_sn2 = conv1.forward, sn2.forward
    conv1.forward, sn2.forward = pair.conv_forward, pair.neuron_forward
    core_arguments = (modules, read_arrays(residual/'rank_control_parameters.npz'),
                      read_arrays(chain/'rank32_diverse10/parameters.npz'), source_fit, consumer_fits)
    if args.basis_kind == 'lifting40':
        from lifting_temporal_control import LiftingTemporalControl
        core = LiftingTemporalControl(*core_arguments, basis_lifting=initialization['basis_lifting'])
        basis_parameter = 'basis_lifting'
    else:
        core = FastTemporalControl(*core_arguments)
        basis_parameter = 'basis_sign_logits'
    masks = {}

    def delete_nonanchor(module, inputs, output):
        key = (output.shape[-2:], output.device)
        if key not in masks:
            mask = torch.zeros(output.shape[-2:], device=output.device, dtype=torch.bool)
            mask[::2, ::2] = True
            masks[key] = mask
        return mask_nonanchors(output, masks[key])

    common_hook = modules[BLOCK+'.norm2'].register_forward_hook(delete_nonanchor)

    def preserve_graph(module, inputs, output):
        current['differentiable_flow'] = output.sum(0)

    flow_hook = modules['sttmultires_unet.preds.2'].register_forward_hook(preserve_graph, prepend=True)

    def release():
        core.clear_cached_graphs()
        release_model_graphs(modules, current, core)
        functional.reset_net(model)

    def forward_flow(x):
        functional.reset_net(model)
        current.pop('flow', None)
        current.pop('differentiable_flow', None)
        try:
            model(x)
        except CoarseReady:
            flow = current.pop('differentiable_flow')
            current.pop('flow', None)
            return F.interpolate(flow, (480, 640), mode='bilinear', align_corners=False)
        raise RuntimeError('The actual coarse-flow exit was not reached.')

    def evaluate(axis, directory):
        from shared_temporal_source_counts import SharedTemporalSourceCounts
        counter = SharedTemporalSourceCounts(modules, sn2_theta=pair.temporal.theta).install()
        counter.reset(axis)
        output_before = args.output
        args.output = directory
        try:
            result = evaluate_axis(args, model, current, names, axis, progress_tag='FAST_TEMPORAL_AEE')
            save_json(directory/(axis+'_sources.json'), counter.report(names))
        finally:
            counter.restore()
            args.output = output_before
            release()
        return result

    run = dict(complete=False, check_only=args.check_only, parent=str(parent), axes={},
        axis_family='fast_temporal', basis_kind=args.basis_kind, fixed_lifting=args.fixed_lifting,
        evaluation_files=names, initialization=str(initialization_file),
        common='control3: frozen preview U8/VQ5, Conv2R16, complete nonanchor BN2-branch deletion, continuous PEDR32, real coarse preds.2 successors',
        training=dict(stages=[64, 256], train16=train, extension_plan=str(chain/'shared_temporal_recovery128x256/train128.json'),
            optimizer='Fresh Adam1e-4 at each stage, seed912; 320 total updates, no checkpoint/validation selection',
            loss='Mean sqrt(sum_xy(flow-GT)^2+1e-6) on actual valid GT pixels; no distillation, sparsity/request loss or hyperparameter sweep',
            frozen='theta/center, four patch BN gains/stats, preview factors/PSN, all other network parameters'),
        function='Actual factorwise forward/reverse and independent row gains; source/consumer native ATLIF surrogate and theta*g; helper channel convolutions FP32 without TF32',
        scope='New FP32 training/function and logical source counts. Shadow Conv2/BN are still executed; no hardware state deletion, RTL cycles, PPA or inherited ep34 AEE.')
    if args.fixed_lifting:
        run['function'] = 'Evaluation-only common signed24/f14 completed states; each lifting half-step uses the same signed16/f12 coefficients, signed48 accumulation and RNE/saturate. Actual signed threshold readouts emit theta*g.'
        run['scope'] = 'New fixed-format coordinate function; preview and other layers unchanged. Full-frame saturation/ranges and source activity are measured; shadows remain software only. No native AEE/equality or RTL/physical benefit claim.'
    save_json(args.output/'result.json', run)
    try:
        for axis in AXES:
            torch.manual_seed(SEED)
            if args.evaluate_saved is not None:
                core.load_saved(axis, read_arrays(args.evaluate_saved/(axis+'.npz')))
                fixed_helper = None
                try:
                    if args.fixed_lifting:
                        from fixed_lifting_coordinates import FixedLiftingForward
                        fixed_helper = FixedLiftingForward(core, pair.temporal.theta)
                        np.savez_compressed(args.output/(axis+'_fixed_constants.npz'), **fixed_helper.export_constants())
                        save_json(args.output/(axis+'_fixed_metadata.json'), fixed_helper.metadata)
                    row = dict(evaluation=evaluate(axis, args.output), constraints=core.constraints(), complete=True)
                    if fixed_helper is not None:
                        save_json(args.output/(axis+'_fixed_ranges.json'), fixed_helper.range_report(names))
                        row['fixed_function'] = fixed_helper.metadata
                finally:
                    if fixed_helper is not None:
                        fixed_helper.restore()
                run['axes'][axis] = row
                save_json(args.output/'result.json', run)
                continue
            core.trainable(axis)
            # One actual TRAIN frame: check the new function's differentiable
            # and no_grad modes, not bit equality to the old dense parent.
            x, label, valid = input_frame(args.data, train[0])
            with torch.no_grad():
                expected = forward_flow(x)
            release()
            prediction = forward_flow(x)
            error = prediction.permute(0, 2, 3, 1)[valid]-label.permute(0, 2, 3, 1)[valid]
            loss = torch.sqrt(error.square().sum(-1)+1e-6).mean()
            with fp32_matmul():
                loss.backward()
            gradients = core.gradients()
            check = dict(file=train[0], optimizer_updates=0,
                flow_max_abs=float((prediction.detach()-expected).abs().max()), loss=float(loss.detach()),
                gradients=gradients, constraints=core.constraints())
            new_groups = (basis_parameter, 'source_row_gain', 'consumer_row_gain')
            if args.basis_kind == 'lifting40':
                new_groups += ('source_bias', 'consumer_bias')
            check['active_new_parameter_groups'] = {key: gradients[key].get('nonzero', 0) > 0
                for key in new_groups}
            check['pass'] = (check['flow_max_abs'] == 0
                and all(g.get('present') and g.get('all_finite') for g in gradients.values())
                and all(check['active_new_parameter_groups'].values()))
            row = dict(check=check, total_parameters=sum(p.numel() for p in core.params.values()),
                       parameter_counts={k:p.numel() for k,p in core.params.items()},
                       function=core.metadata, stages={}, complete=False)
            run['axes'][axis] = row
            save_json(args.output/'result.json', run)
            print('FAST_TEMPORAL_CHECK', axis, json.dumps(check), flush=True)
            del x, label, valid, prediction, expected, error, loss
            for p in core.params.values():
                p.grad = None
            release()
            if not check['pass']:
                raise RuntimeError('Actual training-function/gradient check failed for '+axis)
            if args.check_only:
                row['complete'] = True
                continue
            for cumulative, schedule in ((64, schedule64), (320, schedule256)):
                directory = args.output/('stage'+str(cumulative))
                directory.mkdir(exist_ok=True)
                optimizer = torch.optim.Adam(list(core.params.values()), lr=LR)
                stage = dict(history=[], updates=0, complete=False)
                row['stages'][str(cumulative)] = stage
                started = time.monotonic()
                if args.basis_kind == 'fast_sign':
                    stage['hard_sign_flip_events'] = 0
                    previous_signs = core.basis.signs().detach().clone()
                else:
                    initial_lifting = core.basis.lifting.detach().clone()
                    stage['basis_lifting_initial'] = initial_lifting.cpu().tolist()
                for index, filename in enumerate(schedule):
                    x, label, valid = input_frame(args.data, filename)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = forward_flow(x)
                    error = prediction.permute(0, 2, 3, 1)[valid]-label.permute(0, 2, 3, 1)[valid]
                    loss = torch.sqrt(error.square().sum(-1)+1e-6).mean()
                    with fp32_matmul():
                        loss.backward()
                    gradients = core.gradients()
                    if not bool(torch.isfinite(loss)) or not all(g.get('present') and g.get('all_finite') for g in gradients.values()):
                        raise RuntimeError('Nonfinite/missing gradient in fast recovery: '+axis)
                    entry = dict(step=index+1, file=filename, loss_before_update=float(loss.detach()),
                        valid_pixels=int(valid.sum()))
                    optimizer.step()
                    if args.basis_kind == 'fast_sign':
                        current_signs = core.basis.signs().detach()
                        entry['hard_sign_flips'] = int((current_signs != previous_signs).sum())
                        stage['hard_sign_flip_events'] += entry['hard_sign_flips']
                        previous_signs = current_signs.clone()
                    else:
                        entry['basis_lifting_max_abs'] = float(core.basis.lifting.detach().abs().max())
                        entry['basis_lifting_change_from_stage_start_max_abs'] = float(
                            (core.basis.lifting.detach()-initial_lifting).abs().max())
                    stage['history'].append(entry)
                    stage['updates'] = index+1
                    if index == 0 or (index+1) % 8 == 0:
                        save_json(args.output/'result.json', run)
                        print('FAST_TEMPORAL_TRAIN', axis, cumulative, index+1, json.dumps(entry), flush=True)
                    del x, label, valid, prediction, error, loss
                    release()
                stage['training_wall_seconds'] = time.monotonic()-started
                stage['constraints'] = core.constraints()
                arrays = core.export()
                arrays.update(parent=np.array(str(parent)), basis_kind=np.array(args.basis_kind), train_schedule=np.array(schedule),
                              cumulative_steps=np.array(cumulative), steps=np.array(len(schedule)),
                              prior_steps=np.array(cumulative-len(schedule)), seed=np.array(SEED))
                np.savez_compressed(directory/(axis+'.npz'), **arrays)
                stage['evaluation'] = evaluate(axis, directory)
                stage['complete'] = True
                save_json(args.output/'result.json', run)
                del optimizer
            row['complete'] = True
            release()
            gc.collect()
            torch.cuda.empty_cache()
        run['complete'] = True
        save_json(args.output/'result.json', run)
    finally:
        flow_hook.remove()
        common_hook.remove()
        core.restore()
        conv1.forward, sn2.forward = original_conv1, original_sn2
    print('FAST_TEMPORAL_DONE', flush=True)


if __name__ == '__main__':
    main()
