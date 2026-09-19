"""Post-hoc numerical check of the frame dominating paired mean improvement.

No training. This single-frame diagnostic never replaces valid825.
"""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'neuron_repair_20260919'))
import diagnose_threshold_step as bootstrap


def run(model, optimizer, loss_fn, train_dataset, config, device, transform, args):
    import torch
    from spikingjelly.activation_based import functional
    from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite
    from models.STSwinNet_SNN.atlif_ternary_psn.atlif_ternary_psn import ATLIFTernaryPSN
    from recovery_probe import prepare

    training = json.loads((HERE / 'neuron256/RECOVERY.json').read_text())
    paired = json.loads((HERE / 'neuron256/PAIRED_VALID825.json').read_text())
    assert training['complete'] and paired['complete']
    neuron_ids = {id(p) for m in model.modules() if isinstance(m, ATLIFTernaryPSN)
                  for p in m.parameters()}
    params = {n: p for n, p in model.named_parameters() if id(p) in neuron_ids}
    states = {'ep34': {n: p.detach().clone() for n, p in params.items()}}
    for arm in training['arms']:
        state = torch.load(arm['checkpoint'], map_location='cpu', weights_only=False)['model_state_dict']
        states[arm['rule']] = {n: state[n].to(device) for n in params}
        del state
    policy = {'tf32': False}

    def wrap(original):
        def forward(x):
            previous = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = policy['tf32']
            try:
                with torch.autocast(device_type=x.device.type, enabled=False):
                    return original(x.float())
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous
        return forward

    for m in model.modules():
        if isinstance(m, ATLIFTernaryPSN):
            m.forward = wrap(m.forward)
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.track_running_stats = False
            m.running_mean = m.running_var = m.num_batches_tracked = None
    dataset = DSECDatasetLite(config, file_list='valid', stereo=False)
    filename = 'zurich_city_09_a_0541.npy'
    index = next(i for i in range(len(dataset)) if str(dataset.files.iloc[i, 0]) == filename)
    x, gt, mask = prepare(dataset, [index], device)
    report = dict(scope='Post-hoc one-frame repeat and PSN precision sensitivity; not quality validation',
                  file=filename, non_neuron_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                  cudnn_tf32=torch.backends.cudnn.allow_tf32, rows=[], complete=False)
    with torch.no_grad():
        for allow_tf32 in (False, True):
            policy['tf32'] = allow_tf32
            for name, state in states.items():
                for n, p in params.items():
                    p.copy_(state[n])
                predictions, aees = [], []
                for repeat in range(2):
                    functional.reset_net(model)
                    functional.set_step_mode(model, config['data']['step_mode'])
                    pred = model(x)['flow'][-1] * config['metrics']['flow_scaling']
                    errors = torch.linalg.vector_norm(pred - gt, dim=1)[mask[:, 0].bool()]
                    aees.append(float(errors.double().mean()))
                    predictions.append(pred.cpu())
                previous = next(r['aee'] for r in paired['arms'][name]['rows'] if r['file'] == filename)
                row = dict(arm=name, psn_tf32=allow_tf32, aees=aees,
                           repeat_output_max_abs_difference=float((predictions[1] - predictions[0]).abs().max()),
                           delta_from_full_valid_strict=aees[0] - previous)
                report['rows'].append(row)
                print('[sensitive-frame]', row, flush=True)
    report['complete'] = True
    (args.output / 'FRAME_PRECISION_REPLAY.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    bootstrap.run_probe = run
    bootstrap.main()
