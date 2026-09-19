"""Full FP32 no_running evaluation of a verified theta-only checkpoint pair."""
import json
import time
import torch
from recovery_probe import prepare


def run(model, config, device, args):
    from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite
    from spikingjelly.activation_based import functional
    candidate = torch.load(args.paired_valid_checkpoint, map_location='cpu')
    candidate_state = candidate['model_state_dict']
    parameters = dict(model.named_parameters())
    names = [n for n in parameters if n.endswith('.thresh')]
    # This inexpensive direct comparison is essential: otherwise switching just
    # theta would not evaluate the checkpoint named in the report.
    unequal = [n for n, p in parameters.items()
               if n not in names and not torch.equal(p.detach().cpu(), candidate_state[n])]
    if unequal:
        raise ValueError(f'Theta-only checkpoint has {len(unequal)} other changed parameters: {unequal[:3]}')
    bn = [(m, m.track_running_stats, m.running_mean, m.running_var, m.num_batches_tracked)
          for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    ignored_bn = {id(v) for m, *_ in bn for v in [m.running_mean, m.running_var, m.num_batches_tracked]
                  if v is not None}
    non_bn_buffer_changes = [n for n, v in model.named_buffers() if id(v) not in ignored_bn
                            and not torch.equal(v.detach().cpu(), candidate_state[n])]
    assert not non_bn_buffer_changes, non_bn_buffer_changes[:3]
    baseline_theta = [parameters[n].detach().clone() for n in names]
    candidate_theta = [candidate_state[n].to(device) for n in names]
    del candidate_state, candidate
    model.eval()
    for m, *_ in bn:
        m.track_running_stats = False
        m.running_mean = m.running_var = m.num_batches_tracked = None
    valid = DSECDatasetLite(config, file_list='valid', stereo=False)
    assert len(valid) == 825
    report = dict(scope='Paired complete valid825, no training or hardware timing claim',
                  checkpoint=str(args.checkpoint), candidate_checkpoint=str(args.paired_valid_checkpoint),
                  precision='FP32', batch_size=1, bn='no_running', head='flow[-1]',
                  non_theta_parameters_equal=True, non_bn_buffers_equal=True,
                  changed_theta=sum(not torch.equal(x, y) for x, y in zip(baseline_theta, candidate_theta)),
                  python=__import__('sys').version.split()[0], torch=torch.__version__,
                  rows=[], complete=False)
    path = args.output / 'PAIRED_VALID825.json'
    start = time.monotonic()

    def save():
        rows = report['rows']
        if rows:
            report['frames_completed'] = len(rows)
            report['pixels'] = sum(r['pixels'] for r in rows)
            report['baseline_aee_frame_mean'] = sum(r['baseline_aee'] for r in rows) / len(rows)
            report['candidate_aee_frame_mean'] = sum(r['candidate_aee'] for r in rows) / len(rows)
            report['baseline_aee_pixel_weighted'] = sum(r['baseline_error_sum'] for r in rows) / report['pixels']
            report['candidate_aee_pixel_weighted'] = sum(r['candidate_error_sum'] for r in rows) / report['pixels']
            report['delta_aee_frame_mean'] = report['candidate_aee_frame_mean'] - report['baseline_aee_frame_mean']
        report['elapsed_seconds'] = time.monotonic() - start
        path.write_text(json.dumps(report, indent=2) + '\n')

    with torch.no_grad():
        for i in range(len(valid)):
            x, gt, mask = prepare(valid, [i], device)
            row = dict(index=i, file=str(valid.files.iloc[i, 0]), pixels=int(mask[:, 0].bool().sum()))
            for label, values in [('baseline', baseline_theta), ('candidate', candidate_theta)]:
                for n, value in zip(names, values):
                    parameters[n].copy_(value)
                functional.reset_net(model)
                functional.set_step_mode(model, config['data']['step_mode'])
                pred = model(x)['flow'][-1] * config['metrics']['flow_scaling']
                errors = torch.linalg.vector_norm(pred - gt, dim=1)[mask[:, 0].bool()]
                row[label + '_error_sum'] = float(errors.double().sum())
                row[label + '_aee'] = row[label + '_error_sum'] / row['pixels']
            report['rows'].append(row)
            if (i + 1) % 25 == 0:
                save()
                print('[paired-valid825]', i + 1, report['baseline_aee_frame_mean'],
                      report['candidate_aee_frame_mean'], report['elapsed_seconds'], flush=True)
        for n, value in zip(names, baseline_theta):
            parameters[n].copy_(value)
    for m, track, mean, var, count in bn:
        m.track_running_stats, m.running_mean, m.running_var, m.num_batches_tracked = track, mean, var, count
    report['complete'] = True
    save()
    print('[paired-valid825] COMPLETE', report['baseline_aee_frame_mean'],
          report['candidate_aee_frame_mean'], report['delta_aee_frame_mean'], flush=True)
