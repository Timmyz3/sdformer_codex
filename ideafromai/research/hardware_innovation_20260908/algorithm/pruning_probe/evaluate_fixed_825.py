"""Evaluate two already fixed one-layer pruning checkpoints on full valid825.

No training, calibration, mask selection, new source capture or code arrays.
The saved integer bits3 parent and its existing full825 readout are reused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

import run_pruning_probe as pruning


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    probe = pruning.probe
    from run_bn_probe import input_frame, read_names
    from evaluate_stage2_deployment import CoarseReady, summarize
    from spikingjelly.activation_based import functional

    out = args.root/'algorithm/pruning_probe/valid825'
    out.mkdir(parents=True, exist_ok=True)
    parent = args.root/'algorithm/direct_code_integer/deployment'
    parent_run = json.loads((parent/'valid825_run.json').read_text())
    parent_summary = json.loads((parent/'valid825_summary.json').read_text())
    system = probe.load_system(args)
    model, _, _, current, sources, _, _, _ = system
    names = read_names(args.data, 'valid')
    assert len(names) == 825 and names == parent_run['frames']
    assert parent_summary['complete'] and parent_summary['frames'] == 825
    probe.install_sources(system, sources)
    current['count_codes'] = False
    current['save_codes_dir'] = None
    axes = ('hidden_H_half_trained', 'row_2of4_trained')
    run = {'checkpoints': list(axes), 'module': pruning.PREFIX,
           'source_parent': 'algorithm/direct_code_integer/parameters.pt',
           'parent_full825': parent_summary, 'same_parent_frame_order': True,
           'preserved': parent_run['preserved'], 'readout': parent_run['readout'],
           'mask_and_training': 'fixed final train32/256-step checkpoints; no new fitting',
           'continuous_theta': 'unchanged; tau remains a separate decision threshold',
           'allow_tf32_matmul': bool(torch.backends.cuda.matmul.allow_tf32),
           'allow_tf32_cudnn': bool(torch.backends.cudnn.allow_tf32),
           'codes_capture': False,
           'prior_parent_check': 'unpruned one-layer wrapper exactly reproduced saved parent valid10 AEE and unchanged W/tau',
           'claim': 'full825 availability of ordinary pruning controls; not a new hardware mechanism or hardware speedup'}
    probe.save_json(out/'run.json', run)
    summaries = {}
    for axis in axes:
        record = torch.load(out.parent/(axis+'.pt'), map_location='cpu', weights_only=False)
        pruning.install_pruned_consumer(system, record)
        rows = []
        start = time.monotonic()
        with torch.no_grad():
            for index, filename in enumerate(names):
                functional.reset_net(model)
                x, label, mask = input_frame(args.data, filename)
                try:
                    model(x)
                except CoarseReady:
                    pred = F.interpolate(current.pop('flow'), (480, 640), mode='bilinear', align_corners=False)
                error = torch.linalg.vector_norm(pred.permute(0, 2, 3, 1)[mask]
                                                -label.permute(0, 2, 3, 1)[mask], dim=1)
                total, pixels = float(error.double().sum()), error.numel()
                rows.append({'file': filename, 'valid_pixels': pixels,
                             'aee_sum': total, 'AEE': total/pixels})
                count = index+1
                if count == 25 or count % 50 == 0 or count == len(names):
                    elapsed = time.monotonic()-start
                    result = summarize(rows, count == len(names))
                    result.update(remaining_H=record['remaining_H'], nonzero_W=record['nonzero_W'],
                                  evaluation_wall_seconds=elapsed,
                                  estimated_remaining_seconds=elapsed*(len(names)-count)/count)
                    if result['complete']:
                        assert result['valid_pixels'] == parent_summary['valid_pixels']
                        result['frame_AEE_delta_vs_parent'] = result['AEE_frame_mean']-parent_summary['AEE_frame_mean']
                        result['pixel_AEE_delta_vs_parent'] = result['AEE_pixel_mean']-parent_summary['AEE_pixel_mean']
                    summaries[axis] = result
                    probe.save_json(out/(axis+'_frames.json'), rows)
                    probe.save_json(out/'summary.json', summaries)
                    print('PROGRESS_825', axis, count, json.dumps(result), flush=True)
                del x, label, mask, pred, error
        print('COMPLETE_825', axis, json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
