"""Full paired inference after neuron-only training; no source checkpoint writes."""
import json
import time
import torch


def run(model, initial, neurons, trainable, arms, dataset, config, device, output):
    from spikingjelly.activation_based import functional
    from recovery_probe import prepare
    from models.STSwinNet_SNN.atlif_ternary_psn.atlif_ternary_psn import OfficialATLIFSurrogate

    keys = set(trainable)
    weights = {'ep34': {n: initial[n].to(device) for n in keys}}
    checks = {}
    # BN running buffers do not participate in no_running inference. All other
    # tensors outside the explicitly trained neuron set must be unchanged.
    bn_keys = {n + '.' + suffix for n, m in model.named_modules()
               if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
               for suffix in ['running_mean', 'running_var', 'num_batches_tracked']}
    for arm in arms:
        state = torch.load(arm['checkpoint'], map_location='cpu', weights_only=False)['model_state_dict']
        assert set(state) == set(initial)
        different = [n for n in initial if n not in keys and n not in bn_keys
                     and not torch.equal(initial[n], state[n])]
        if different:
            raise RuntimeError('Changes outside neuron parameters: ' + str(different))
        weights[arm['rule']] = {n: state[n].to(device) for n in keys}
        checks[arm['rule']] = dict(non_neuron_tensors_equal=True, changed_neuron_tensors=
            sum(not torch.equal(initial[n], state[n]) for n in keys))
        del state
    model.load_state_dict(initial, strict=True)
    for m in neurons.values():
        m.act = OfficialATLIFSurrogate.apply
        m.update_value = m.act_value = 0.0
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.track_running_stats = False
            m.running_mean = m.running_var = m.num_batches_tracked = None
    current_activity = {}

    def capture(name):
        def hook(module, inputs, output):
            row = current_activity.setdefault(name, dict(elements=0, active_sum=0., calls=0))
            row['elements'] += output.numel()
            row['active_sum'] += module.r * output.numel()
            row['calls'] += 1
        return hook

    hooks = [m.register_forward_hook(capture(n)) for n, m in neurons.items()]
    report = dict(complete=False, frames_total=len(dataset), frames_done=0,
        protocol='AMP off, FP32 tensors, PSN matmul TF32 off; batch1/no_running BN; same voxel and flow[-1]; frame mean AEE',
        non_neuron_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        activity_scope='All elements of 81 live neuron output tensors, including padding; not valid flow pixels',
        checks=checks, arms={name: dict(rows=[], activity={}) for name in weights})
    path = output / 'PAIRED_VALID825.json'
    started = time.monotonic()

    def save():
        for arm in report['arms'].values():
            rows = arm['rows']
            if rows:
                arm['aee_frame_mean'] = sum(r['aee'] for r in rows) / len(rows)
                arm['aee_pixel_weighted'] = sum(r['error_sum'] for r in rows) / sum(r['pixels'] for r in rows)
                activity = arm['activity'].values()
                arm['element_weighted_firing'] = sum(r['active_sum'] for r in activity) / sum(r['elements'] for r in activity)
                arm['zero_firing_modules_observed'] = [n for n, r in arm['activity'].items() if r['active_sum'] == 0.]
        report['elapsed_seconds'] = time.monotonic() - started
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    with torch.no_grad():
        for i in range(len(dataset)):
            x, gt, mask = prepare(dataset, [i], device)
            for name, state in weights.items():
                for key, param in trainable.items():
                    param.copy_(state[key])
                functional.reset_net(model)
                functional.set_step_mode(model, config['data']['step_mode'])
                current_activity.clear()
                pred = model(x)['flow'][-1] * config['metrics']['flow_scaling']
                errors = torch.linalg.vector_norm(pred - gt, dim=1)[mask[:, 0].bool()]
                report['arms'][name]['rows'].append(dict(file=str(dataset.files.iloc[i, 0]),
                    pixels=errors.numel(), error_sum=float(errors.double().sum()), aee=float(errors.double().mean())))
                for neuron_name, values in current_activity.items():
                    activity = report['arms'][name]['activity'].setdefault(neuron_name,
                        dict(elements=0, active_sum=0., calls=0, nonzero_frames=0))
                    for key in ['elements', 'active_sum', 'calls']:
                        activity[key] += values[key]
                    activity['nonzero_frames'] += int(values['active_sum'] > 0.)
                del pred, errors
            del x, gt, mask
            report['frames_done'] = i + 1
            if (i + 1) % 25 == 0 or i + 1 == len(dataset):
                save()
                print('[psn-repair fullvalid]', i + 1,
                      {n: a['aee_frame_mean'] for n, a in report['arms'].items()}, flush=True)
    report['complete'] = True
    for hook in hooks:
        hook.remove()
    save()
    return dict(path=str(path), frames=report['frames_done'], metrics={
        n: {k: v for k, v in a.items() if k not in {'rows', 'activity'}} for n, a in report['arms'].items()})
