"""Independent stochastic replication of the 2026-09-20 neuron-only repair.

No architecture, T2 layout, BN policy, checkpoint identity, or hardware changes.
Reuse the audited loader, gradients and data preparation from the prior repair.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'atlif_psn_recovery_20260920'))
sys.path.insert(0, str(HERE.parent / 'neuron_repair_20260919'))
import diagnose_threshold_step as bootstrap


def run(model, unused_optimizer, loss_fn, dataset, config, device, transform, args, options):
    import numpy as np
    import torch
    from spikingjelly.activation_based import functional
    from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite
    from models.STSwinNet_SNN.atlif_ternary_psn.atlif_ternary_psn import ATLIFTernaryPSN
    from models.STSwinNet_SNN.h28_optimizer import build_optimizer
    from binary_surrogate import install as set_theta_rule
    from q_support_proxy import install as set_q_bridge, restore as restore_q_bridge
    from recovery_probe import prepare, evaluate

    neurons = {n: m for n, m in model.named_modules() if isinstance(m, ATLIFTernaryPSN)}
    assert len(neurons) == 81
    assert all(m.output_mode == 'binary' and m.threshold_mode == 'official_atlif' and m.sp == 0
               for m in neurons.values())
    acts = {n: m.act for n, m in neurons.items()}
    initial = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
    neuron_ids = {id(p) for m in neurons.values() for p in m.parameters()}
    for p in model.parameters():
        p.requires_grad_(id(p) in neuron_ids)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert all(n.endswith(('.weight', '.bias', '.thresh')) and '.spiking_neuron.' in n for n in trainable)
    def strict_neuron(original):
        def forward(x):
            old_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            try:
                with torch.autocast(device_type=x.device.type, enabled=False):
                    return original(x.float())
            finally:
                torch.backends.cuda.matmul.allow_tf32 = old_tf32
        return forward

    for m in neurons.values():
        m.forward = strict_neuron(m.forward)
    valid = DSECDatasetLite(config, file_list='valid', stereo=False)
    indices = np.linspace(0, len(valid) - 1, 10, dtype=int).tolist()
    batch_size = config['loader']['batch_size']
    starts = np.linspace(0, len(dataset) - batch_size, options.steps, dtype=int).tolist()
    # Preserve the original B2 pairings and training set. Independently vary
    # batch order, augmentation, and dropout; all arms use the same schedule.
    np.random.default_rng(np.random.SeedSequence([options.seed, 0])).shuffle(starts)
    augmentation_seeds = np.random.SeedSequence([options.seed, 1]).generate_state(options.steps).tolist()
    dropout_seeds = np.random.SeedSequence([options.seed, 2]).generate_state(options.steps).tolist()
    report = dict(scope='A100 replication: fixed training set, independent stochastic order/augmentation/dropout',
        seed=options.seed, gpu=torch.cuda.get_device_name(device),
        config_source=str(args.config),
        sampling='Original linspace B2 pairs permuted; no resampling or re-pairing of frames',
        augmentation_seeds=augmentation_seeds, dropout_seeds=dropout_seeds,
        complete=False, checkpoint=str(args.checkpoint), python=sys.version.split()[0], torch=torch.__version__,
        unchanged=['architecture', 'T2 layout', 'backbone', 'BN affine', 'Motion-XOR alpha0.125', 'binary theta output'],
        precision='AMP conv/linear; FP32 PSN forward and FP32 backward matmuls with TF32 disabled; validation AMP off',
        non_neuron_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        bn='Native training batch statistics; evaluation batch1/no_running',
        clip='GradScaler unscale -> clip -> step, retry same batch after overflow',
        eta=0, theta_epsilon=1e-12, successful_steps_per_arm=options.steps,
        batch_size=batch_size, train_indices=[list(range(s, s + batch_size)) for s in starts],
        train_files=[[str(dataset.files.iloc[i, 0]) for i in range(s, s + batch_size)] for s in starts],
        trainable_shapes={n: list(p.shape) for n, p in trainable.items()},
        trainable_parameters=sum(p.numel() for p in trainable.values()),
        diagnostic_valid_indices=indices, arms=[])
    path = args.output / 'RECOVERY.json'

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    report['baseline_diagnostic'] = evaluate(model, valid, indices, device, config)
    save()
    print('[psn-repair baseline]', report['baseline_diagnostic']['aee_frame_mean'], flush=True)
    for rule in options.arms.split(','):
        assert rule in {'official_eps', 'normalized', 'normalized_q'}
        functional.reset_net(model)
        model.load_state_dict(initial, strict=True)
        for n, m in neurons.items():
            m.act = acts[n]; m.update_value = m.act_value = 0.0
        if rule.startswith('normalized'):
            assert set_theta_rule(model, 'normalized') == 81
        q_original, q_counts = (None, None)
        if rule == 'normalized_q':
            q_original, q_counts = set_q_bridge()
        optimizer = build_optimizer(model, config)
        threshold_groups = [g for g in optimizer.param_groups if g.get('name') == 'atlif_threshold']
        assert len(threshold_groups) == 1
        threshold_groups[0]['eps'] = 1e-12
        scaler = torch.cuda.amp.GradScaler(init_scale=options.scale)
        arm = dict(rule=rule, updates=[], diagnostics=[], optimizer=[
            dict(name=g['name'], lr=g['lr'], eps=g['eps'], weight_decay=g['weight_decay'],
                 parameters=sum(p.numel() for p in g['params'])) for g in optimizer.param_groups])
        report['arms'].append(arm)
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        for step, start in enumerate(starts):
            x, gt, mask = prepare(dataset, range(start, start + batch_size), device, transform, augmentation_seeds[step])
            # Retry the identical batch/RNG after overflow. Restore BN buffers
            # as well, so an unsuccessful update has no training-state effect.
            buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
            attempts = []
            for attempt in range(12):
                optimizer.zero_grad(set_to_none=True)
                functional.reset_net(model)
                functional.set_step_mode(model, config['data']['step_mode'])
                if attempt:
                    with torch.no_grad():
                        for n, b in model.named_buffers():
                            b.copy_(buffers[n])
                for m in neurons.values():
                    m.update_value = m.act_value = 0.0
                torch.manual_seed(dropout_seeds[step]); torch.cuda.manual_seed_all(dropout_seeds[step])
                with torch.cuda.amp.autocast():
                    outputs = model(x)['flow']
                    loss = loss_fn(outputs, gt, mask, gamma=config['loss']['gamma'])
                scale = scaler.get_scale()
                old_tf32 = torch.backends.cuda.matmul.allow_tf32
                torch.backends.cuda.matmul.allow_tf32 = False
                try:
                    scaler.scale(loss).backward()
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = old_tf32
                scaler.unscale_(optimizer)
                norm = float(torch.nn.utils.clip_grad_norm_(trainable.values(), config['loss']['clip_grad']))
                scaler.step(optimizer); scaler.update()
                skipped = scaler.get_scale() < scale
                attempts.append(dict(scale=scale, loss=float(loss.detach()), skipped=skipped,
                                     preclip_norm=norm if math.isfinite(norm) else None))
                del outputs, loss
                if not skipped:
                    break
            else:
                arm['failure'] = dict(step=step, reason='No finite step after 12 same-batch AMP backoffs', attempts=attempts)
                save()
                raise RuntimeError(arm['failure'])
            with torch.no_grad():
                for m in neurons.values():
                    m.thresh.clamp_(0.001, 2.0)
            theta_changed = sum(not torch.equal(m.thresh.detach().cpu(), initial[n + '.thresh']) for n, m in neurons.items())
            q_grad_nonzero = sum(m.thresh.grad is not None and bool(m.thresh.grad != 0)
                                 for n, m in neurons.items() if '.sn_q.' in n)
            row = dict(step=step, attempts=attempts, changed_theta=theta_changed, q_nonzero_theta_grad=q_grad_nonzero)
            arm['updates'].append(row)
            if step % 16 == 0 or step + 1 == options.steps:
                save()
                print('[psn-repair step]', rule, step + 1, attempts[-1], 'theta', theta_changed,
                      'Q-gradient', q_grad_nonzero, flush=True)
            del x, gt, mask, buffers
            if (step + 1) % 64 == 0 or step + 1 == options.steps:
                metric = evaluate(model, valid, indices, device, config)
                arm['diagnostics'].append(dict(successful_step=step + 1, validation=metric))
                model.train()
                save()
                print('[psn-repair diagnostic]', rule, step + 1, metric['aee_frame_mean'], flush=True)
        arm['training_seconds_including_diagnostics'] = time.monotonic() - started
        arm['peak_allocated_gib_including_diagnostics'] = torch.cuda.max_memory_allocated(device) / 1024**3
        arm['peak_reserved_gib_including_diagnostics'] = torch.cuda.max_memory_reserved(device) / 1024**3
        arm['successful_updates'] = len(arm['updates'])
        arm['overflow_attempts'] = sum(len(r['attempts']) - 1 for r in arm['updates'])
        arm['theta'] = {n: dict(before=float(initial[n + '.thresh']), after=float(m.thresh.detach())) for n, m in neurons.items()}
        arm['parameter_max_change'] = {n: float((p.detach().cpu() - initial[n]).abs().max()) for n, p in trainable.items()}
        if q_original is not None:
            arm['q_bridge_calls'] = q_counts['calls']
            restore_q_bridge(q_original)
        checkpoint = args.output / f'{rule}_{options.steps}successful.pth'
        torch.save(dict(model_state_dict={n: v.detach().cpu() for n, v in model.state_dict().items()},
                        rule=rule, seed=options.seed, config=config, successful_updates=options.steps,
                        scope='Only PSN weight/bias/theta trainable; forward unchanged; original ep34 preserved'), checkpoint)
        arm['checkpoint'] = str(checkpoint)
        optimizer.zero_grad(set_to_none=True)
        del optimizer, scaler
        gc.collect(); torch.cuda.empty_cache()
        save()

    # Baseline and both fixed endpoints on all valid825. No arm is chosen on the
    # diagnostic ten frames. Reuse one decoded input per frame, no BN batching.
    report['training_complete'] = True
    save()
    if not options.skip_full_validation:
        from validate_candidates import run as validate
        report['full_validation'] = validate(model, initial, neurons, trainable, report['arms'],
                                               valid, config, device, args.output)
    report['complete'] = True
    save()
    print('[psn-repair COMPLETE]', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--steps', type=int, default=256)
    p.add_argument('--arms', default='official_eps,normalized')
    p.add_argument('--scale', type=float, default=8192.)
    p.add_argument('--skip-full-validation', action='store_true')
    options, rest = p.parse_known_args()
    bootstrap.run_probe = lambda *values: run(*values, options)
    sys.argv = [sys.argv[0]] + rest
    bootstrap.main()


if __name__ == '__main__':
    main()
