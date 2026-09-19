"""Same-budget binary recovery: fixed AMP order, three surrogate arms.

64 distinct train batches by default, followed by a preselected valid subset.
This is a smoke evaluation, never a substitute for valid825.
"""
import gc
import json
import random
import time
import numpy as np
import torch
from binary_surrogate import install


def prepare(dataset, indices, device, transform=None, seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    data = [dataset[i] for i in indices]
    chunk, mask, label = [torch.stack([row[j] for row in data]).to(device) for j in range(3)]
    chunk, label, mask = chunk.float(), label.float(), mask.unsqueeze(1)
    if transform is not None:
        chunk, label, mask = transform((chunk, label, mask.float()))
    chunk = torch.stack((torch.relu(chunk), torch.relu(-chunk)), dim=2)
    active = chunk != 0
    lo, hi = chunk[active].min(), chunk[active].max()
    if lo != hi:
        chunk[active] = (chunk[active] - lo) / (hi - lo)
    return chunk, label, mask


def evaluate(model, dataset, indices, device, config):
    from spikingjelly.activation_based import functional
    model.eval()
    bn = [(m, m.track_running_stats, m.running_mean, m.running_var, m.num_batches_tracked)
          for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for m, *_ in bn:
        m.track_running_stats = False
        m.running_mean = m.running_var = m.num_batches_tracked = None
    rows = []
    with torch.no_grad():
        for i in indices:
            functional.reset_net(model)
            functional.set_step_mode(model, config['data']['step_mode'])
            x, gt, mask = prepare(dataset, [i], device)
            pred = model(x)['flow'][-1] * config['metrics']['flow_scaling']
            valid = mask[:, 0].bool()
            e = torch.linalg.vector_norm(pred - gt, dim=1)[valid]
            rows.append(dict(file=str(dataset.files.iloc[i, 0]), pixels=e.numel(),
                             error_sum=float(e.double().sum()), aee=float(e.double().mean())))
    for m, track, mean, var, count in bn:
        m.track_running_stats, m.running_mean, m.running_var, m.num_batches_tracked = track, mean, var, count
    return dict(head='model flow[-1], native output interpolation', amp=False, bn='no_running',
                frames=len(rows), aee_frame_mean=sum(r['aee'] for r in rows) / len(rows),
                aee_pixel_weighted=sum(r['error_sum'] for r in rows) / sum(r['pixels'] for r in rows), rows=rows)


def run(model, loss_function, dataset, config, device, transform, args):
    from spikingjelly.activation_based import functional
    from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite
    from models.STSwinNet_SNN.h28_optimizer import build_optimizer
    from models.STSwinNet_SNN.atlif_ternary_psn.atlif_ternary_psn import ATLIFTernaryPSN

    valid = DSECDatasetLite(config, file_list='valid', stereo=False)
    valid_indices = np.linspace(0, len(valid) - 1, args.valid_samples, dtype=int).tolist()
    modules = {name: m for name, m in model.named_modules() if isinstance(m, ATLIFTernaryPSN)}
    assert all(m.output_mode == 'binary' and m.threshold_mode == 'official_atlif' and m.sp == 0 for m in modules.values())
    acts = {name: m.act for name, m in modules.items()}
    state = {name: v.detach().cpu().clone() for name, v in model.state_dict().items()}
    theta0 = {name: float(m.thresh.detach()) for name, m in modules.items()}
    requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}
    if args.theta_only:
        thresholds = {id(m.thresh) for m in modules.values()}
        for p in model.parameters():
            p.requires_grad_(id(p) in thresholds)
    batch_size = config['loader']['batch_size']
    starts = np.linspace(0, len(dataset) - batch_size, args.recovery_steps, dtype=int).tolist()
    report = dict(scope='Small-budget recovery, not valid825 or a chosen final model',
                  steps=args.recovery_steps, batch_size=batch_size,
                  train_files=[[str(dataset.files.iloc[s + k, 0]) for k in range(batch_size)] for s in starts],
                  valid_indices=valid_indices, python=__import__('sys').version.split()[0],
                  torch=torch.__version__, checkpoint=str(args.checkpoint),
                  clip='unscale then clip at update boundary', eta=0,
                  trainable='theta_only' if args.theta_only else 'whole_network',
                  initial_scale=args.initial_scale,
                  neuron_fp32=args.neuron_fp32,
                  baseline=evaluate(model, valid, valid_indices, device, config), arms=[], complete=False)
    path = args.output / 'RECOVERY.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print('[recovery-baseline]', report['baseline']['aee_frame_mean'], flush=True)
    for rule in args.recovery_arms.split(','):
        assert rule in {'official', 'official_eps12', 'margin', 'normalized'}
        functional.reset_net(model)
        model.load_state_dict(state, strict=True)
        for name, m in modules.items():
            m.act = acts[name]; m.update_value = 0.; m.act_value = 0.
        if rule in {'margin', 'normalized'}:
            assert install(model, rule) == len(modules)
        model.train()
        optimizer = build_optimizer(model, config)
        if rule == 'official_eps12':
            threshold_groups = [g for g in optimizer.param_groups if g.get('name') == 'atlif_threshold']
            assert len(threshold_groups) == 1
            threshold_groups[0]['eps'] = 1e-12
        scaler = torch.cuda.amp.GradScaler(init_scale=args.initial_scale)
        arm = dict(rule=rule, rows=[], optimizer=[
            dict(name=g.get('name'), lr=g['lr'], eps=g['eps'], weight_decay=g['weight_decay'])
            for g in optimizer.param_groups])
        report['arms'].append(arm)
        start_time = time.monotonic()
        for step, start in enumerate(starts):
            chunk, gt, mask = prepare(dataset, range(start, start + batch_size), device, transform, 1900 + step)
            optimizer.zero_grad(set_to_none=True)
            functional.reset_net(model)
            functional.set_step_mode(model, config['data']['step_mode'])
            torch.manual_seed(2900 + step); torch.cuda.manual_seed_all(2900 + step)
            with torch.cuda.amp.autocast():
                out = model(chunk)['flow']
                loss = loss_function(out, gt, mask, gamma=config['loss']['gamma'])
            scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config['loss']['clip_grad']))
            scaler.step(optimizer); scaler.update()
            skipped = scaler.get_scale() < scale
            row = dict(step=step, loss=float(loss.detach()), scale=scale, preclip_norm=norm, skipped=skipped,
                       changed_theta=sum(float(m.thresh.detach()) != theta0[name] for name, m in modules.items()))
            arm['rows'].append(row)
            if step % 8 == 0 or step == len(starts) - 1:
                print('[recovery-step]', rule, row, flush=True)
                path.write_text(json.dumps(report, indent=2) + '\n')
            del out, loss, chunk, gt, mask
        arm['training_seconds'] = time.monotonic() - start_time
        arm['successful_updates'] = sum(not r['skipped'] for r in arm['rows'])
        arm['theta'] = {name: dict(before=theta0[name], after=float(m.thresh.detach())) for name, m in modules.items()}
        arm['validation'] = evaluate(model, valid, valid_indices, device, config)
        arm['delta_aee_frame_mean'] = arm['validation']['aee_frame_mean'] - report['baseline']['aee_frame_mean']
        # Explicit new regular files in a fresh isolated directory, never a
        # source checkpoint path or a symlink to one.
        ckpt = args.output / f'{rule}_{args.recovery_steps}steps.pth'
        torch.save(dict(model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                        rule=rule, config=config, training_attempts=len(starts),
                        optimizer_steps=arm['successful_updates'], neuron_fp32=args.neuron_fp32,
                        theta_only=args.theta_only), ckpt)
        arm['checkpoint'] = str(ckpt)
        path.write_text(json.dumps(report, indent=2) + '\n')
        print('[recovery-valid]', rule, arm['validation']['aee_frame_mean'], arm['delta_aee_frame_mean'], flush=True)
        optimizer.zero_grad(set_to_none=True)
        del optimizer, scaler
        gc.collect()
    model.load_state_dict(state, strict=True)
    for name, m in modules.items():
        m.act = acts[name]
    for name, p in model.named_parameters():
        p.requires_grad_(requires_grad[name])
    report['complete'] = True
    path.write_text(json.dumps(report, indent=2) + '\n')
    print('[recovery] COMPLETE', flush=True)
