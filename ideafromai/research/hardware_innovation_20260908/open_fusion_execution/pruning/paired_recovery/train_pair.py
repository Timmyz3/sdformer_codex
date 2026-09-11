"""One predeclared paired64-step fixed-forward recovery; no sweep or selection."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--smoke-only', action='store_true')
    args = p.parse_args()
    base = args.root
    patch = base/'algorithm/patch_probe'
    res = patch/'residual_consumer_probe'
    chain = res/'projection_chain'
    latent = patch/'factor_completion_20260909/latent_stage_train16'
    lift = chain/'fast_temporal_recovery_lifting40'
    here = base/'open_fusion_execution/pruning/paired_recovery'
    for d in [chain, res, latent, lift/'schedule_compare_same_port',
              base/'algorithm', base/'algorithm/nrv_cost_probe']:
        sys.path.insert(0, str(d))
    import torch
    import torch.nn.functional as F
    import run_probe as probe
    from adapter import fp32_matmul
    from capture_inputs import save_json
    from flow_backward_probe import TrainableLatentPair, read_arrays, gradient_stats
    from capture import BLOCK, SOURCE_SN, CONSUMER_SN, PROJECT
    from evaluate_branch_control import evaluate_axis, mask_nonanchors
    from train_shared_temporal_recovery import SharedTemporalControl
    from fixed_temporal_coordinates import FixedTemporalForward
    from train_consumer_recovery import read_train16, release_model_graphs
    from run_bn_probe import input_frame, read_names
    from evaluate_stage2_deployment import CoarseReady
    from spikingjelly.activation_based import functional
    from fixed_qat import FixedQATForward

    out = here/('smoke' if args.smoke_only else 'stage64')
    out.mkdir(parents=True, exist_ok=True)
    args.output, args.split = out, 'diverse'
    train = read_train16(latent/'flow_train_list.json')
    rng = np.random.default_rng(912)
    order = np.concatenate([rng.permutation(16) for _ in range(4)])
    schedule = [train[int(i)] for i in order]
    names = [r['file'] for r in json.loads((chain/'temporal_structured_recovery/fixed_sources_diverse10/identity_permuted_base_frames.json').read_text())]
    masks = json.loads((base/'open_fusion_execution/review/phase_group8_masks.json').read_text())['ordinary']
    system = probe.load_system(args)
    model, modules, _, current, sources, _, _, _ = system
    probe.install_sources(system, sources)
    current['count_codes'] = False
    if set(train).intersection(read_names(args.data, 'valid')):
        raise ValueError('The fixed TRAIN list overlaps official validation.')
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
    anchors = {}
    def nonanchor(module, inputs, output):
        key = (tuple(output.shape[-2:]), output.device)
        if key not in anchors:
            mask = torch.zeros(output.shape[-2:], device=output.device, dtype=torch.bool)
            mask[::2, ::2] = True
            anchors[key] = mask
        return mask_nonanchors(output, anchors[key])
    anchor_hook = modules[BLOCK+'.norm2'].register_forward_hook(nonanchor)
    def preserve_graph(module, inputs, output):
        current['differentiable_flow'] = output.sum(0)
    flow_hook = modules['sttmultires_unet.preds.2'].register_forward_hook(preserve_graph, prepend=True)
    common = (modules, read_arrays(res/'rank_control_parameters.npz'), read_arrays(chain/'rank32_diverse10/parameters.npz'))
    original_student = chain/'temporal_structured_recovery/stage128x256/identity_permuted_base.npz'
    initial = read_arrays(original_student)
    report = dict(complete=False, smoke_only=args.smoke_only, axes={},
        student='New ordinary fixed-grid Conv2-recovered identities; source temporal identity frozen.',
        initial=str(original_student), train_files=train, train_schedule=schedule,
        validation_files=names, mask_calibration_frame='zurich_city_09_a_0001.npy',
        budget=dict(steps=0 if args.smoke_only else 64, seed=912, optimizer='Adam', lr=1e-4),
        trainable=['Conv2_U_R16', 'Conv2_V_R16'], trainable_parameters=15360,
        frozen='Source temporal A/bias/theta, preview U/V and sn2, consumer temporal cutoff, PED U/V and bias, BN2 scale/bias/stats, all remaining network parameters.',
        loss='Existing real-GT robust EPE: mean sqrt(sum_xy(error^2)+1e-6); no teacher, no auxiliary term.',
        numeric='Real original fixed helper forward with signed16 coefficients, same fixed exponent, signed48 exact integer arithmetic and each RNE/saturate24. Clip-aware identity STE and triangular PED gate backward only.',
        validation='Only final64 deployed checkpoint, no best-checkpoint/validation selection. Recompile/reload through the original FixedTemporalForward before diverse10.',
        hardware_claim=False, full_valid825=False)
    save_json(out/'run.json', report)
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
        raise RuntimeError('Actual coarse exit was not reached.')
    def loss_for(prediction, label, valid):
        error = prediction.permute(0, 2, 3, 1)[valid]-label.permute(0, 2, 3, 1)[valid]
        return torch.sqrt(error.square().sum(-1)+1e-6).mean()
    observed = {}
    observe_enabled = [False]
    def observe(label):
        def hook(module, inputs, output):
            if observe_enabled[0]:
                observed[label] = output.detach().clone()
        return hook
    obs_hooks = [modules[path].register_forward_hook(observe(label)) for label, path in
                 [('PED_continuous', PROJECT+'.conv_res'), ('projection_gate', CONSUMER_SN)]]
    try:
        for mode in ['global_group2', 'phase_joint']:
            torch.manual_seed(912)
            c = SharedTemporalControl(*common, fit={})
            c.load_saved('identity_permuted_base', initial)
            for parameter in c.params.values():
                parameter.requires_grad_(False)
            mask = torch.as_tensor(masks[mode]['mask_uint8'], device=pair.u.device, dtype=torch.bool)
            def prune(module, inputs, output):
                result = output.clone()
                for phase in range(4):
                    channels = torch.nonzero(mask[phase], as_tuple=False).flatten()
                    y, x = divmod(phase, 2)
                    result[:, :, channels, y::2, x::2] = 0
                return result
            mask_hook = sn2.register_forward_hook(prune)
            row = dict(complete=False, drop_groups=masks[mode]['drop_groups'], history=[], updates=0)
            report['axes'][mode] = row
            helper = None
            try:
                x, label, valid = input_frame(args.data, train[0])
                helper = FixedTemporalForward(c, pair.temporal.theta)
                observe_enabled[0] = True
                with torch.no_grad():
                    expected = forward_flow(x)
                expected_internal = observed.copy()
                observed.clear()
                helper.restore()
                helper = None
                c.conv_u.requires_grad_(True)
                c.conv_v.requires_grad_(True)
                helper = FixedQATForward(c, pair.temporal.theta)
                prediction = forward_flow(x)
                errors = {name: dict(elements=value.numel(), differences=int(torch.count_nonzero(value-observed[name])),
                    max_abs=float((value-observed[name]).abs().max())) for name,value in expected_internal.items()}
                loss = loss_for(prediction, label, valid)
                check = dict(file=train[0], forward_flow_differences=int(torch.count_nonzero(prediction.detach()-expected)),
                    forward_flow_max_abs=float((prediction.detach()-expected).abs().max()),
                    internal=errors, loss_requires_grad=loss.requires_grad, loss=float(loss.detach()))
                if loss.requires_grad:
                    with fp32_matmul():
                        loss.backward()
                check['gradients'] = {'U':gradient_stats(c.conv_u), 'V':gradient_stats(c.conv_v)}
                row['gradient_smoke'] = check
                save_json(out/'run.json', report)
                print('PAIR_FIXED_GRADIENT', mode, json.dumps(check), flush=True)
                if check['forward_flow_differences'] or any(v['differences'] for v in errors.values()) or not all(
                    g.get('present') and g.get('all_finite') and g.get('nonzero') for g in check['gradients'].values()):
                    raise RuntimeError('Exact fixed-forward or nonzero-gradient smoke failed: '+mode)
                observe_enabled[0] = False
                observed.clear()
                del x, label, valid, prediction, expected, expected_internal, loss
                release_model_graphs(modules, current, c)
                c.conv_u.grad = c.conv_v.grad = None
                if args.smoke_only:
                    row['complete'] = True
                    continue
                opt = torch.optim.Adam([c.conv_u, c.conv_v], lr=1e-4)
                started = time.monotonic()
                for step, name in enumerate(schedule):
                    x, label, valid = input_frame(args.data, name)
                    opt.zero_grad(set_to_none=True)
                    prediction = forward_flow(x)
                    loss = loss_for(prediction, label, valid)
                    with fp32_matmul():
                        loss.backward()
                    grads = {'U':gradient_stats(c.conv_u), 'V':gradient_stats(c.conv_v)}
                    if not bool(torch.isfinite(loss)) or not all(g.get('present') and g.get('all_finite') for g in grads.values()):
                        raise RuntimeError('Missing/nonfinite recovery gradient: '+mode)
                    entry = dict(step=step+1, file=name, loss_before_update=float(loss.detach()))
                    if step in (0, 63):
                        entry['gradients'] = grads
                    opt.step()
                    row['history'].append(entry)
                    row['updates'] = step+1
                    if step == 0 or (step+1)%8 == 0:
                        save_json(out/'run.json', report)
                        print('PAIR_FIXED_TRAIN', mode, step+1, entry['loss_before_update'], flush=True)
                    del x, label, valid, prediction, loss
                    release_model_graphs(modules, current, c)
                row['training_wall_seconds'] = time.monotonic()-started
                trained_q = {k:(helper.live_q(k).detach().cpu().numpy(), helper.matrices[k]['exponent'])
                             for k in ['U_conv2_theta', 'F']}
                with torch.no_grad():
                    x, label, valid = input_frame(args.data, train[0])
                    expected = forward_flow(x)
                exported = c.export(original_student, schedule, prior_steps=0)
                exported.update(recovery_mask=np.asarray(mode), source_temporal_frozen=np.asarray(True),
                    factor_numeric=np.asarray('Fixed-grid signed16 forward QAT of Conv2 U/F; source, PED, bias and scales frozen.'))
                np.savez_compressed(out/(mode+'.npz'), **exported)
                helper.restore()
                helper = None
                c.load_saved('identity_permuted_base', read_arrays(out/(mode+'.npz')))
                for parameter in c.params.values():
                    parameter.requires_grad_(False)
                helper = FixedTemporalForward(c, pair.temporal.theta)
                row['deployment_coefficient_check'] = {k:dict(training_exponent=int(v[1]),
                    deployment_exponent=int(helper.matrices[k]['exponent']),
                    coefficient_differences=int(np.count_nonzero(v[0]-helper.matrices[k]['q_numpy'])))
                    for k,v in trained_q.items()}
                if any(v['training_exponent'] != v['deployment_exponent'] or v['coefficient_differences']
                       for v in row['deployment_coefficient_check'].values()):
                    raise RuntimeError('Saved original fixed helper compiles a different grid; do not inherit QAT forward.')
                with torch.no_grad():
                    actual = forward_flow(x)
                row['final_QAT_vs_deployed_max_abs'] = float((actual-expected).abs().max())
                row['final_QAT_vs_deployed_differences'] = int(torch.count_nonzero(actual-expected))
                if row['final_QAT_vs_deployed_differences']:
                    raise RuntimeError('Final fixed deploy reload differs from the QAT hard forward.')
                del x, label, valid, expected, actual
                args.output = out/mode
                args.output.mkdir(parents=True, exist_ok=True)
                with torch.no_grad():
                    row['evaluation'] = evaluate_axis(args, model, current, names, mode, progress_tag='PAIR_FIXED_AEE')
                np.savez_compressed(out/(mode+'_deployed_constants.npz'), **helper.export_constants())
                row['complete'] = True
                save_json(out/'run.json', report)
                del opt
            finally:
                if helper is not None:
                    helper.restore()
                c.restore()
                mask_hook.remove()
                release_model_graphs(modules, current, c)
                functional.reset_net(model)
                gc.collect()
                torch.cuda.empty_cache()
            save_json(out/'run.json', report)
        report['complete'] = True
        save_json(out/'run.json', report)
    finally:
        anchor_hook.remove()
        flow_hook.remove()
        for hook in obs_hooks:
            hook.remove()
        conv1.forward, sn2.forward = old_conv1, old_sn2
    print('PAIR_FIXED_DONE', flush=True)


if __name__ == '__main__':
    main()
