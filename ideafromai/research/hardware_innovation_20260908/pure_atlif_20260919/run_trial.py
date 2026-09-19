"""ep34 no-PSN screen and matched short recovery, in an isolated process.

All replacements preserve the existing tensor layouts. In particular, the
T2 attention axis is NOT claimed to be chronological (see migration review).
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'neuron_repair_20260919'))
import diagnose_threshold_step as bootstrap


def run(model, _optimizer, loss_function, train, config, device, transform, args, options):
    import numpy as np
    import torch
    from spikingjelly.activation_based import functional
    from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite
    from models.STSwinNet_SNN.atlif_ternary_psn.atlif_ternary_psn import ATLIFTernaryPSN
    from recovery_probe import prepare, evaluate
    from neuron_precision import install as fp32_neurons
    from pure_atlif import replace, RecurrentATLIF

    originals = {name: m for name, m in model.named_modules() if isinstance(m, ATLIFTernaryPSN)}
    state = {name: v.detach().cpu().clone() for name, v in model.state_dict().items()}
    fp32_neurons(model)
    valid = DSECDatasetLite(config, file_list='valid', stereo=False)
    indices = np.linspace(0, len(valid) - 1, options.frames, dtype=int).tolist()
    report = dict(checkpoint=str(args.checkpoint), complete=False,
                  scope='Current ep34, unchanged network layouts, no-PSN intervention; not valid825',
                  training='matched whole-network recovery, original official surrogate, eta=0',
                  temporal_layout='T2 follows frozen window_partition_v2, not proven chronological',
                  membrane_precision='FP32, including PSN in matched control',
                  validation_indices=indices, steps=options.steps, arms=[])
    outpath = args.output / 'TRIAL.json'

    def save():
        outpath.write_text(json.dumps(report, indent=2) + '\n')

    for arm_name in options.arms.split(','):
        for name, old in originals.items():
            parent_name, _, leaf = name.rpartition('.')
            setattr(model.get_submodule(parent_name), leaf, old)
            old.act_value = old.update_value = 0.0
        functional.reset_net(model)
        model.load_state_dict(state, strict=True)
        if arm_name == 'baseline':
            replaced = []
        elif arm_name in {'t10_tau1', 'all_tau1', 'all_tau09'}:
            replaced = replace(model, 't10' if arm_name.startswith('t10') else 'all',
                               0.9 if arm_name.endswith('09') else 1.0)
        else:
            raise ValueError(arm_name)
        neurons = {name: m for name, m in model.named_modules()
                   if isinstance(m, (ATLIFTernaryPSN, RecurrentATLIF))}
        activity = {}

        def capture(name):
            def hook(module, inputs, out):
                # Counts are output population counts, not elapsed cycles.
                row = activity.setdefault(name, dict(calls=0, elements=0, spike_sum=0.0,
                    T=module.T, recurrent=isinstance(module, RecurrentATLIF),
                    consumer_live=not name.endswith(('.sn2_q', '.attn_sn'))))
                row['calls'] += 1; row['elements'] += out.numel()
                row['spike_sum'] += module.r * out.numel()
            return hook

        hooks = [m.register_forward_hook(capture(name)) for name, m in neurons.items()]
        arm = dict(name=arm_name, installed_replacements=len(replaced),
                   live_replacements=sum(r['consumer_live'] for r in replaced),
                   removed_temporal_parameters=sum(r['removed_parameters'] for r in replaced),
                   replacements=replaced)
        report['arms'].append(arm)
        started = time.monotonic()
        arm['initial'] = evaluate(model, valid, indices, device, config)
        arm['initial_evaluation_seconds'] = time.monotonic() - started
        for hook in hooks:
            hook.remove()
        live = [r for r in activity.values() if r['consumer_live']]
        arm['initial_activity'] = dict(
            runtime_called=len(activity), runtime_live=len(live),
            population_weighted_rate=sum(r['spike_sum'] for r in live) / sum(r['elements'] for r in live),
            by_module=activity)
        save()
        print('[no-psn initial]', arm_name, arm['initial']['aee_frame_mean'],
              'firing', arm['initial_activity']['population_weighted_rate'], flush=True)
        if options.steps and arm_name in options.train_arms.split(','):
            theta_ids = {id(m.thresh) for m in neurons.values()}
            psn_ids = {id(p) for m in neurons.values() if isinstance(m, ATLIFTernaryPSN)
                       for p in m.parameters()}
            buckets = {'backbone': [], 'norm_bias': [], 'psn': [], 'psn_no_decay': [], 'theta': []}
            groups = config['optimizer']['param_groups']
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                key = ('theta' if id(p) in theta_ids
                       else ('psn_no_decay' if p.ndim <= 1 or name.endswith('.bias') else 'psn')
                       if id(p) in psn_ids
                       else 'norm_bias' if p.ndim <= 1 or name.endswith('.bias') else 'backbone')
                buckets[key].append(p)
            specs = dict(backbone=(groups['backbone_lr'], config['optimizer']['wd'], 1e-8),
                         norm_bias=(groups['norm_lr'], 0., 1e-8),
                         psn=(groups['neuron_lr'], config['optimizer']['wd'], 1e-8),
                         psn_no_decay=(groups['neuron_lr'], 0., 1e-8),
                         theta=(groups['threshold_lr'], 0., 1e-12))
            optimizer = torch.optim.AdamW([
                dict(params=ps, name=k, lr=specs[k][0], weight_decay=specs[k][1], eps=specs[k][2])
                for k, ps in buckets.items() if ps])
            scaler = torch.cuda.amp.GradScaler(init_scale=options.initial_scale)
            arm['optimizer'] = [{k: g[k] for k in ['name', 'lr', 'weight_decay', 'eps']}
                                for g in optimizer.param_groups]
            arm['updates'] = []
            batch = options.batch_size
            starts = np.linspace(0, len(train) - batch, options.steps, dtype=int).tolist()
            arm['train_indices'] = [list(range(s, s + batch)) for s in starts]
            model.train()
            started = time.monotonic()
            for step, start in enumerate(starts):
                x, gt, mask = prepare(train, range(start, start + batch), device, transform, 1900 + step)
                optimizer.zero_grad(set_to_none=True)
                functional.reset_net(model)
                functional.set_step_mode(model, config['data']['step_mode'])
                for m in neurons.values():
                    m.act_value = m.update_value = 0.0
                torch.manual_seed(2900 + step); torch.cuda.manual_seed_all(2900 + step)
                with torch.cuda.amp.autocast():
                    flows = model(x)['flow']
                    loss = loss_function(flows, gt, mask, gamma=config['loss']['gamma'])
                scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config['loss']['clip_grad']))
                scaler.step(optimizer); scaler.update()
                with torch.no_grad():
                    for m in neurons.values():
                        m.thresh.clamp_(0.001, 2.0)
                row = dict(step=step, loss=float(loss.detach()), preclip_norm=norm,
                           skipped=scaler.get_scale() < scale, loss_scale=scale)
                arm['updates'].append(row)
                if step % 8 == 0 or step + 1 == options.steps:
                    save(); print('[no-psn train]', arm_name, row, flush=True)
                del x, gt, mask, flows, loss
            arm['training_seconds'] = time.monotonic() - started
            arm['successful_updates'] = sum(not row['skipped'] for row in arm['updates'])
            arm['after'] = evaluate(model, valid, indices, device, config)
            checkpoint = args.output / (arm_name + '_short_recovery.pth')
            torch.save(dict(model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                            arm=arm_name, config=config, steps=options.steps,
                            successful_updates=arm['successful_updates'],
                            note='Instantiate ep34 model, replace neurons using arm, then load. FP32 neuron arithmetic.'),
                       checkpoint)
            arm['checkpoint'] = str(checkpoint)
            print('[no-psn after]', arm_name, arm['after']['aee_frame_mean'], flush=True)
            optimizer.zero_grad(set_to_none=True)
            del optimizer, scaler, buckets
            gc.collect(); torch.cuda.empty_cache()
            save()
    report['complete'] = True
    save()
    print('[no-psn COMPLETE]', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arms', default='baseline,t10_tau1,all_tau1,all_tau09')
    parser.add_argument('--train-arms', default='baseline,all_tau1')
    parser.add_argument('--steps', type=int, default=0)
    parser.add_argument('--frames', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--scale', type=float, default=8192., dest='initial_scale')
    options, rest = parser.parse_known_args()
    bootstrap.run_probe = lambda *values: run(*values, options)
    sys.argv = [sys.argv[0]] + rest
    bootstrap.main()


if __name__ == '__main__':
    main()
