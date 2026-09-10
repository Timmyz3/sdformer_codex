"""Full valid825 evaluation of already fixed grouping/deletion checkpoints."""
import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from run_group_pruning_probe import base, probe, install


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--record', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    system = probe.load_system(args)
    from run_bn_probe import input_frame, read_names
    from evaluate_stage2_deployment import CoarseReady, summarize
    from spikingjelly.activation_based import functional
    model, _, _, current, sources, _, _, _ = system
    probe.install_sources(system, sources)
    current['count_codes'] = False
    names = read_names(args.data, 'valid')
    parent = args.root/'algorithm/direct_code_integer/deployment'
    parent_run = json.loads((parent/'valid825_run.json').read_text())
    parent_result = json.loads((parent/'valid825_summary.json').read_text())
    assert names == parent_run['frames'] and len(names) == 825
    out = args.root/'algorithm/group_pruning_probe/valid825'
    out.mkdir(exist_ok=True)
    summaries = json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else {}
    probe.save_json(out/'run.json', dict(records=[str(p) for p in args.record],
        parent_full825=parent_result, frames=names, readout=parent_run['readout'],
        fitting='none; checkpoints fixed before this evaluation',
        claim='algorithm-only full-network AEE; no hardware speedup',
        allow_tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
        allow_tf32_cudnn=bool(torch.backends.cudnn.allow_tf32)))
    for filename in args.record:
        record = torch.load(filename, map_location='cpu', weights_only=False)
        (install if 'physical_to_logical_h' in record else base.install_pruned_consumer)(system, record)
        rows, started = [], time.monotonic()
        with torch.no_grad():
            for index, name in enumerate(names):
                functional.reset_net(model)
                x, label, mask = input_frame(args.data, name)
                try:
                    model(x)
                except CoarseReady:
                    pred = F.interpolate(current.pop('flow'), (480, 640), mode='bilinear', align_corners=False)
                error = torch.linalg.vector_norm(pred.permute(0, 2, 3, 1)[mask]-label.permute(0, 2, 3, 1)[mask], dim=1)
                total, pixels = float(error.double().sum()), error.numel()
                rows.append(dict(file=name, valid_pixels=pixels, aee_sum=total, AEE=total/pixels))
                count = index+1
                if count == 25 or count % 50 == 0 or count == len(names):
                    result = summarize(rows, count == len(names))
                    result.update(remaining_H=record['remaining_H'], nonzero_W=record['nonzero_W'],
                                  wall_seconds=time.monotonic()-started)
                    if result['complete']:
                        assert result['valid_pixels'] == parent_result['valid_pixels']
                        result['frame_AEE_delta_vs_parent'] = result['AEE_frame_mean']-parent_result['AEE_frame_mean']
                    summaries[filename.stem] = result
                    probe.save_json(out/(filename.stem+'_frames.json'), rows)
                    probe.save_json(out/'summary.json', summaries)
                    print('FULL825', filename.stem, count, json.dumps(result), flush=True)
                del x, label, mask, pred, error


if __name__ == '__main__':
    main()
