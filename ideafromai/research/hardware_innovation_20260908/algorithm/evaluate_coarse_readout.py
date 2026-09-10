"""Zero-training penultimate optical-flow readout, full official validation."""
import argparse
import json
from pathlib import Path

from run_bn_probe import build_model, input_frame, read_names, save_json
import numpy as np
import torch
import torch.nn.functional as F


class CoarseReady(Exception):
    pass


def main():
    p = argparse.ArgumentParser()
    for name in ('code-root', 'config', 'checkpoint', 'data', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model, _, _, _ = build_model(args)
    from spikingjelly.activation_based import functional
    coarse = dict(model.named_modules())['sttmultires_unet.preds.2']
    current = {}

    def stop(module, inputs, output):
        current['flow'] = output.detach().sum(0)
        raise CoarseReady()

    handle = coarse.register_forward_hook(stop)
    rows = []
    with torch.no_grad():
        for i, name in enumerate(read_names(args.data, 'valid')):
            functional.reset_net(model)
            x, label, mask = input_frame(args.data, name)
            try:
                model(x)
            except CoarseReady:
                low = current.pop('flow')
            values = {'file': name, 'valid_pixels': int(mask.sum())}
            for mode in ('nearest', 'bilinear'):
                kwargs = {'align_corners': False} if mode == 'bilinear' else {}
                pred = F.interpolate(low, size=(480, 640), mode=mode, **kwargs)
                error = (pred-label).square().sum(1).sqrt()
                total = float(error[mask].sum())
                values[mode] = {'AEE': total/values['valid_pixels'], 'sum': total}
            rows.append(values)
            if (i+1) % 50 == 0:
                save_json(args.output/'frames.json', rows)
                print('VALID', i+1, {m: float(np.mean([r[m]['AEE'] for r in rows]))
                                    for m in ('nearest', 'bilinear')}, flush=True)
            del x, label, mask, low, pred, error
    handle.remove()
    save_json(args.output/'frames.json', rows)
    summary = {'frames': len(rows), 'valid_pixels': sum(r['valid_pixels'] for r in rows),
               'AEE': {m: float(np.mean([r[m]['AEE'] for r in rows])) for m in ('nearest', 'bilinear')},
               'model_change': 'penultimate preds.2 time sum; stop before last decoder; no weight/BN/quantization changes',
               'flow_units': 'full-image pixels; no multiplication on interpolation',
               'scope': 'new readout algorithm baseline, not RTL speed or hardware innovation'}
    save_json(args.output/'summary.json', summary)
    print('COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
