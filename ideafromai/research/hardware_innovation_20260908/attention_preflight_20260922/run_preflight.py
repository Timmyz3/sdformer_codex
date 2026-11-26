"""Ten full-resolution frames; run only AFTER the neuron replication queue.

No training, new checkpoint, sparse kernel, or measured acceleration. Replaces
only stage2/block0's carrier, using one fixed official_eps endpoint for all arms.
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'neuron_repair_20260919'))
import diagnose_threshold_step as bootstrap


def run(model, optimizer, loss, dataset, config, device, transform, args):
    import numpy as np
    import torch
    from DSEC_dataloader.DSEC_dataset_lite import DSECDatasetLite
    from spikingjelly.activation_based import functional
    from models.STSwinNet_SNN.atlif_ternary_psn.atlif_ternary_psn import ATLIFTernaryPSN
    from recovery_probe import prepare
    from adaptive_spike_probe import AttentionProbe

    neurons = [m for m in model.modules() if isinstance(m, ATLIFTernaryPSN)]
    assert len(neurons) == 81
    def strict(original):
        def forward(x):
            previous = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            try:
                with torch.autocast(device_type=x.device.type, enabled=False):
                    return original(x.float())
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous
        return forward
    for m in neurons:
        m.forward = strict(m.forward)
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.track_running_stats = False
            m.running_mean = m.running_var = m.num_batches_tracked = None
    matches = [(n, m) for n, m in model.named_modules() if n.endswith('layers.2.swin_blocks.0.attn')]
    assert len(matches) == 1, [n for n, _ in matches]
    name, attention = matches[0]
    assert attention._h9_shiftmax_cfg.mode == 'h60'
    valid = DSECDatasetLite(config, file_list='valid', stereo=False)
    indices = np.linspace(0, len(valid)-1, 10, dtype=int).tolist()
    report = dict(complete=False, checkpoint=str(args.checkpoint), module=name,
        frozen_neurons=81, training_steps=0, frames=10, validation_indices=indices,
        precision='AMPoff, strict FP32 PSN/no TF32, B1/no_running BN',
        non_neuron_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        scope='ZERO-SHOT interface/quality probe; not trained SAFformer, not full valid825',
        inference_only=True, guide_training_implemented=False,
        original_motion_computation_still_executed=True,
        hardware_speedup=None, arms={})
    path = args.output / 'PREFLIGHT.json'
    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    def forward(x):
        functional.reset_net(model)
        functional.set_step_mode(model, config['data']['step_mode'])
        return model(x)['flow'][-1] * config['metrics']['flow_scaling']
    with torch.no_grad():
        # Instrumentation must be exactly invisible for A0 on identical input.
        x, gt, mask = prepare(valid, [indices[0]], device)
        reference = forward(x)
        probe = AttentionProbe(attention)
        observed = forward(x)
        report['A0_hook_bit_exact'] = torch.equal(reference, observed)
        assert report['A0_hook_bit_exact']
        del reference, observed, x, gt, mask
        for mode in ['motion', 'q_presence_probe', 'saf_fixed_half_probe']:
            probe.mode = mode
            arm = dict(rows=[])
            report['arms'][mode] = arm
            for i in indices:
                x, gt, mask = prepare(valid, [i], device)
                probe.rows.clear()
                pred = forward(x)
                errors = torch.linalg.vector_norm(pred-gt, dim=1)[mask[:, 0].bool()]
                assert bool(torch.all(torch.isfinite(errors)))
                arm['rows'].append(dict(file=str(valid.files.iloc[i, 0]), pixels=errors.numel(),
                    error_sum=float(errors.double().sum()), aee=float(errors.double().mean()),
                    selection=list(probe.rows)))
                arm['aee_frame_mean'] = sum(r['aee'] for r in arm['rows']) / len(arm['rows'])
                arm['aee_pixel_weighted'] = sum(r['error_sum'] for r in arm['rows']) / sum(r['pixels'] for r in arm['rows'])
                save()
                print('[attention-preflight]', mode, len(arm['rows']), arm['aee_frame_mean'], flush=True)
                del x, gt, mask, pred, errors
        probe.close()
    previous = json.loads((args.checkpoint.parent / 'PAIRED_VALID825.json').read_text())
    reference_rows = {r['file']: r for r in previous['arms']['official_eps']['rows']}
    differences = []
    for row in report['arms']['motion']['rows']:
        old = reference_rows[row['file']]
        assert row['pixels'] == old['pixels']
        differences.append(dict(file=row['file'], aee_delta=row['aee']-old['aee']))
    report['endpoint_reload_vs_original_valid825'] = dict(rows=differences,
        max_abs_aee_difference=max(abs(r['aee_delta']) for r in differences))
    report['complete'] = True
    save()


if __name__ == '__main__':
    bootstrap.run_probe = run
    bootstrap.main()
