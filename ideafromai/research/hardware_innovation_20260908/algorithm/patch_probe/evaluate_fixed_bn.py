"""Evaluate fixed train32 patch BN constants on full valid825, without fitting."""
import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from run_patch_probe import probe, RES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    system = probe.load_system(args)
    from run_bn_probe import input_frame, read_names
    from evaluate_stage2_deployment import CoarseReady, summarize
    from spikingjelly.activation_based import functional
    model, modules, _, current, sources, _, _, _ = system
    probe.install_sources(system, sources)
    current['count_codes'] = False
    out = args.root/'algorithm/patch_probe'
    stats = torch.load(out/'patch_train_calibration.pt', map_location='cpu', weights_only=False)
    for name, values in stats.items():
        m = modules[name]
        m.track_running_stats = True
        m.running_mean = values['mean'].to(m.weight)
        m.running_var = values['var'].to(m.weight)
    names = read_names(args.data, 'valid')
    parent = json.loads((args.root/'algorithm/direct_code_integer/deployment/valid825_summary.json').read_text())
    rows, started = [], time.monotonic()
    with torch.no_grad():
        for i, name in enumerate(names):
            functional.reset_net(model)
            x, label, mask = input_frame(args.data, name)
            try:
                model(x)
            except CoarseReady:
                pred = F.interpolate(current.pop('flow'), (480,640), mode='bilinear', align_corners=False)
            error = torch.linalg.vector_norm(pred.permute(0,2,3,1)[mask]-label.permute(0,2,3,1)[mask], dim=1)
            total, pixels = float(error.double().sum()), error.numel()
            rows.append(dict(file=name, valid_pixels=pixels, aee_sum=total, AEE=total/pixels))
            if (i+1)%50 == 0 or i+1 == len(names):
                result = dict(**summarize(rows, i+1==len(names)), wall_seconds=time.monotonic()-started,
                    changed_BN=list(stats), parent_full825=parent,
                    fitting='none; constants already fitted to train32',
                    claim='algorithm AEE only, no hardware speedup')
                if result['complete']:
                    result['delta_frame_AEE_vs_parent'] = result['AEE_frame_mean']-parent['AEE_frame_mean']
                probe.save_json(out/'fixed_bn_valid825_summary.json', result)
                probe.save_json(out/'fixed_bn_valid825_frames.json', rows)
                print('FULL825', i+1, json.dumps(result), flush=True)
            del x, label, mask, pred, error


if __name__ == '__main__':
    main()
