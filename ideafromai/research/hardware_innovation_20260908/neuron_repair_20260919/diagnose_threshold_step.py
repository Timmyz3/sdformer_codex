"""Actual network backward + paired scalar AdamW first-step diagnostics.

All alternatives share exactly one forward/backward and its upstream gradients.
Only theta's backward rule changes; no optimizer writes to the network. This is
not a training/AEE result. Run using the sd5ai Python 3.12 environment.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time


def run_probe(model, optimizer, loss_function, train_dataset, config, device, transform_train, args):
    if args.neuron_fp32:
        from neuron_precision import install as install_precision
        print('[neuron-fp32] installed', len(install_precision(model)), flush=True)
    if args.q_probe:
        from q_gradient_probe import run
        return run(model, optimizer, loss_function, train_dataset, config, device, transform_train, args)
    if args.recovery_steps:
        from recovery_probe import run
        return run(model, loss_function, train_dataset, config, device, transform_train, args)
    import numpy as np
    import torch
    from spikingjelly.activation_based import functional
    import models.STSwinNet_SNN.atlif_ternary_psn.atlif_ternary_psn as neuron

    assert config['optimizer']['num_acc'] == 1
    modules = {name: m for name, m in model.named_modules()
               if isinstance(m, neuron.ATLIFTernaryPSN)}
    assert all(m.output_mode == 'binary' and m.threshold_mode == 'official_atlif'
               for m in modules.values())
    theta = {name: m.thresh for name, m in modules.items()}
    by_ptr = {p.data_ptr(): name for name, p in theta.items()}
    param_groups = {id(p): group for group in optimizer.param_groups for p in group['params']}
    original_backward = neuron.OfficialATLIFSurrogate.backward
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    records = {}
    current_scale = 1.0
    rules = ['official', 'official_fp32_mean', 'single_kernel_mean', 'margin_mean', 'margin_sum', 'normalized_sum']

    def observe_backward(ctx, grad_output, dummy):
        h, th = ctx.saved_tensors
        returned = original_backward(ctx, grad_output, dummy)
        name = by_ptr[th.data_ptr()]
        # Gate comparison uses the original forward's dtypes/promotion, NOT a
        # comparison after casting the saved operands to the gradient dtype.
        gate = (h >= th).float()
        fp_gate = h.float() >= th.float()
        t57_gate = h.to(grad_output.dtype) >= th.to(grad_output.dtype)
        p = (1.0 - ((h.float() - th.float()) / th.float()).abs()).clamp_min(0)
        g = grad_output.detach().float() / current_scale
        values = {
            'official': float(returned[1].detach().float().cpu()) / current_scale,
            'official_fp32_mean': float((-g * p * p).mean().cpu()),
            'single_kernel_mean': float((-g * p).mean().cpu()),
            'margin_mean': float((g * (gate - p)).mean().cpu()),
            'margin_sum': float((g * (gate - p)).sum().cpu()),
            'normalized_sum': float((g * (gate - (h.float() / th.float()) * p)).sum().cpu()),
        }
        records[name] = dict(gradient_unscaled=values, elements=h.numel(),
                             h_dtype=str(h.dtype), theta_dtype=str(th.dtype),
                             grad_dtype=str(grad_output.dtype),
                             fp32_comparison_gate_mismatches=int((gate.bool() != fp_gate).sum()),
                             t57_comparison_gate_mismatches=int((gate.bool() != t57_gate).sum()))
        return returned

    def save(report):
        (args.output / 'DIAGNOSIS.json').write_text(json.dumps(report, indent=2) + '\n')

    report = dict(scope='Fresh-optimizer first-step counterfactuals, shared real network backward; no AEE or saved model',
                  checkpoint=str(args.checkpoint), config=str(args.config),
                  python=sys.version.split()[0], torch=torch.__version__,
                  amp=not args.fp32, neurons=len(modules), rules=rules,
                  neuron_fp32=args.neuron_fp32,
                  input_gradient_rule='Unmodified official rule for every arm',
                  parameter_updates_to_network=0, batches=[], complete=False)
    args.output.mkdir(parents=True, exist_ok=True)
    save(report)
    neuron.OfficialATLIFSurrogate.backward = staticmethod(observe_backward)
    model.train()
    batch_size = config['loader']['batch_size']
    # Explicit train-only samples, spread along its fixed CSV. Augmentation is
    # sampled once and then frozen across all counterfactuals and AMP retries.
    starts = np.linspace(0, len(train_dataset) - batch_size, args.batches, dtype=int).tolist()
    for batch_id, start in enumerate(starts):
        random.seed(1900 + batch_id)
        np.random.seed(1900 + batch_id)
        torch.manual_seed(1900 + batch_id)
        torch.cuda.manual_seed_all(1900 + batch_id)
        data = [train_dataset[i] for i in range(start, start + batch_size)]
        chunk, mask, label = [torch.stack([d[j] for d in data]).to(device) for j in range(3)]
        chunk, label = chunk.float(), label.float()
        mask = mask.unsqueeze(1)
        if transform_train is not None:
            chunk, label, mask = transform_train((chunk, label, mask.float()))
        assert config['model']['encoding'] == 'voxel' and config['loader']['polarity']
        chunk = torch.stack((torch.relu(chunk), torch.relu(-chunk)), dim=2)
        active = chunk != 0
        lo, hi = chunk[active].min(), chunk[active].max()
        if lo != hi:
            chunk[active] = (chunk[active] - lo) / (hi - lo)
        batch = dict(index=batch_id, train_indices=list(range(start, start + batch_size)),
                     files=[str(train_dataset.files.iloc[i, 0]) for i in range(start, start + batch_size)],
                     retries=[])
        current_scale = 1.0 if args.fp32 else 65536.0
        for attempt in range(12):
            optimizer.zero_grad(set_to_none=True)
            functional.reset_net(model)
            functional.set_step_mode(model, config['data']['step_mode'])
            with torch.no_grad():
                for name, b in model.named_buffers():
                    b.copy_(buffers[name])
            torch.manual_seed(2900 + batch_id)
            torch.cuda.manual_seed_all(2900 + batch_id)
            records.clear()
            started = time.monotonic()
            with torch.cuda.amp.autocast(enabled=not args.fp32):
                outputs = model(chunk)['flow']
                loss = loss_function(outputs, label, mask, gamma=config['loss']['gamma'])
            (loss * current_scale).backward()
            params = [p for p in model.parameters() if p.grad is not None]
            norms = torch.stack([p.grad.detach().float().norm(2) for p in params])
            scaled_norm = float(norms.norm(2).cpu())
            attempt_row = dict(scale=current_scale, scaled_norm=scaled_norm,
                               loss=float(loss.detach()), seconds=time.monotonic() - started)
            batch['retries'].append(attempt_row)
            print('[neuron-probe]', batch_id, attempt_row, flush=True)
            del outputs, loss
            if math.isfinite(scaled_norm):
                break
            current_scale /= 2
            if args.fp32:
                raise RuntimeError('Nonfinite full precision network gradient')
        else:
            raise RuntimeError('No finite backward after twelve AMP backoffs')

        # All candidate sums are computed from exactly the same upstream grads.
        # Their theta terms replace only the theta contribution to global norm.
        raw_norm = scaled_norm / current_scale
        theta_sq = sum((float(p.grad.detach().float().cpu()) / current_scale) ** 2
                       for p in theta.values() if p.grad is not None)
        rest_sq = max(0.0, raw_norm ** 2 - theta_sq)
        old_threshold = float(config['loss']['clip_grad'])
        batch['observations'] = dict(records)
        batch['unscaled_global_norm'] = raw_norm
        batch['arms'] = []
        for rule in rules:
            new_theta_sq = sum(v['gradient_unscaled'][rule] ** 2 for v in records.values())
            norm = math.sqrt(rest_sq + new_theta_sq)
            for clip_order in ['scaled_then_step_unscale', 'unscale_then_clip']:
                clip_norm = norm * current_scale if clip_order == 'scaled_then_step_unscale' else norm
                coefficient = min(1.0, old_threshold / (clip_norm + 1e-6))
                cloned = {name: torch.nn.Parameter(p.detach().cpu().clone()) for name, p in theta.items()}
                groups = []
                for name, q in cloned.items():
                    source_group = param_groups[id(theta[name])]
                    groups.append(dict(params=[q], lr=source_group['lr'],
                                       betas=source_group['betas'], eps=source_group['eps'],
                                       weight_decay=source_group['weight_decay']))
                    if name in records:
                        q.grad = torch.tensor(records[name]['gradient_unscaled'][rule] * coefficient,
                                              dtype=q.dtype)
                opt = torch.optim.AdamW(groups, foreach=False)
                before = {name: float(p.detach().cpu()) for name, p in theta.items()}
                opt.step()
                details = {}
                for name, q in cloned.items():
                    state = opt.state.get(q, {})
                    group = param_groups[id(theta[name])]
                    grad = float(q.grad) if q.grad is not None else None
                    proposed = (group['lr'] * grad / (abs(grad) + group['eps'])
                                if grad is not None else 0.0)
                    details[name] = dict(before=before[name], after=float(q.detach()),
                                         gradient_after_clip=grad, first_step_proposed_delta=proposed,
                                         exp_avg=float(state['exp_avg']) if state else None,
                                         exp_avg_sq=float(state['exp_avg_sq']) if state else None,
                                         lr=group['lr'], eps=group['eps'])
                arm = dict(rule=rule, clip_order=clip_order, norm=norm, clip_coefficient=coefficient,
                           changed=sum(d['before'] != d['after'] for d in details.values()),
                           nonzero_grad=sum(d['gradient_after_clip'] not in (None, 0.0) for d in details.values()),
                           max_abs_delta=max(abs(d['after'] - d['before']) for d in details.values()),
                           neurons=details)
                batch['arms'].append(arm)
                print('[neuron-arm]', batch_id, rule, clip_order, arm['changed'], coefficient, flush=True)
        report['batches'].append(batch)
        save(report)
        del chunk, label, mask, data, params, norms
        optimizer.zero_grad(set_to_none=True)
    neuron.OfficialATLIFSurrogate.backward = original_backward
    report['complete'] = True
    save(report)
    print('[neuron-probe] COMPLETE', flush=True)
    if args.pilot_steps:
        from short_training_replay import run
        run(model, loss_function, train_dataset, config, device, transform_train,
            args, buffers, current_scale)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batches', type=int, default=2)
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--pilot-steps', type=int, default=0)
    parser.add_argument('--recovery-steps', type=int, default=0)
    parser.add_argument('--valid-samples', type=int, default=10)
    parser.add_argument('--theta-only', action='store_true')
    parser.add_argument('--recovery-arms', default='official,margin,normalized')
    parser.add_argument('--initial-scale', type=float, default=65536.)
    parser.add_argument('--q-probe', action='store_true')
    parser.add_argument('--paired-valid-checkpoint', type=Path)
    parser.add_argument('--neuron-fp32', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import yaml
    cfg = yaml.safe_load(args.config.read_text())
    cfg['data']['path'] = str(args.repo / 'data/Datasets/DSEC/saved_flow_data')
    cfg['loader']['n_workers'] = 0
    cfg['loader']['persistent_workers'] = False
    cfg['runtime']['use_mlflow_model_logging'] = False
    cfg['runtime']['skip_save'] = True
    cfg['runtime']['skip_state_save'] = True
    # No checkpoint write is reached; the baseline training loop is replaced.
    active_config = args.output / 'diagnostic_config.yml'
    active_config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    exp = args.repo / 'neuron_experiments/H9_bipolar_self_attention'
    baseline = args.repo / 'third_party/SDformerFlow'
    for path in [args.repo, baseline, exp / 'overlay']:
        sys.path.insert(0, str(path))
    os.environ['SDFORMER_USE_MLFLOW'] = '0'
    os.environ['SDFORMER_MLFLOW_MODEL_LOGGING'] = '0'
    spec = importlib.util.spec_from_file_location('h9_train_for_probe', exp / 'entrypoints/train.py')
    h9 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h9)
    h9._install_optional_mlflow_stub()
    entry = baseline / 'train_flow_parallel_supervised_SNN.py'
    source = h9._patch_source(entry.read_text(), entry)
    anchor = '    for epoch in range(epoch_initial, config["loader"]["n_epochs"]):'
    assert source.count(anchor) == 1
    source = source.replace(anchor,
        '    _run_neuron_probe(model, optimizer, loss_function, train_dataset, config, device, transform_train)\n'
        '    return\n' + anchor, 1)
    sys.argv = [str(entry), '--config', str(active_config), '--prev_runid', str(args.checkpoint),
                '--save_path', str(args.output / 'UNUSED_checkpoint_{}.pth'), '--finetune', '1']
    os.chdir(baseline)
    exec(compile(source, str(entry), 'exec'),
         {'__name__': '__main__', '__file__': str(entry),
          '_run_neuron_probe': lambda *values: run_probe(*values, args)})


if __name__ == '__main__':
    main()
