"""Paired ten-frame controls: bypass selected MLP residual branches.

No weights are trained.  Both the final head and the already-supervised
penultimate head are evaluated; this is an algorithm necessity test, not
a hardware speed estimate.
"""
import argparse
import json
from pathlib import Path
import types

import numpy as np
import torch
import torch.nn.functional as F

from run_bn_probe import build_model, input_frame, save_json, set_bn_mode, MLPS, SHALLOW
from run_integer_fc1_probe import install_integer_consumers


def main():
    parser = argparse.ArgumentParser()
    for name in ('code-root', 'config', 'checkpoint', 'data', 'calibration', 'samples', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = json.loads(args.samples.read_text())['valid'][:10]
    scopes = {'integer_teacher': [], 'drop_s0_block0': MLPS[:1],
              'drop_s0_both': MLPS[:2], 'drop_all_mlp': MLPS}
    rows = []
    for mode, prefixes in scopes.items():
        model, _, _, _ = build_model(args)
        from spikingjelly.activation_based import functional
        stats = torch.load(args.calibration, map_location='cpu', weights_only=False)
        set_bn_mode(model, stats, SHALLOW)
        install_integer_consumers(model, stats)
        modules = dict(model.named_modules())

        def zero_branch(self, x):
            return torch.zeros_like(x)

        for prefix in prefixes:
            branch = modules[prefix.rstrip('.')]
            branch.forward = types.MethodType(zero_branch, branch)
        holder = {}

        def coarse_hook(module, inputs, output):
            holder['flow'] = output.detach().sum(0)

        handle = modules['sttmultires_unet.preds.2'].register_forward_hook(coarse_hook)
        with torch.no_grad():
            for name in names:
                functional.reset_net(model)
                x, label, mask = input_frame(args.data, name)
                final = model(x)['flow'][-1]
                coarse = F.interpolate(holder.pop('flow'), size=(480, 640),
                                       mode='bilinear', align_corners=False)
                record = {'mode': mode, 'file': name, 'valid_pixels': int(mask.sum())}
                for key, flow in (('final', final), ('coarse_bilinear', coarse)):
                    error = (flow-label).square().sum(1).sqrt()
                    record[key] = float(error[mask].sum())/record['valid_pixels']
                rows.append(record)
                print('FRAME', json.dumps(record), flush=True)
                del x, label, mask, final, coarse, error
        handle.remove()
        del model, modules
        torch.cuda.empty_cache()
        save_json(args.output/'frames.json', rows)
    summary = {'frames_per_mode': len(names),
               'AEE': {mode: {key: float(np.mean([r[key] for r in rows if r['mode']==mode]))
                              for key in ('final', 'coarse_bilinear')} for mode in scopes},
               'scope': 'zero-training residual branch deletion, paired ten-frame algorithm control; no full825 or hardware claim'}
    save_json(args.output/'summary.json', summary)
    print('COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
