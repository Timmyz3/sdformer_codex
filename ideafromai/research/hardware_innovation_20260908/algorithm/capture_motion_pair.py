"""Capture four adjacent native-ep34 sources and causal previous-frame flow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_bn_probe import build_model, input_frame, save_json
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    for name in ('code-root', 'config', 'checkpoint', 'data', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    model, cfg, _, _ = build_model(args)
    from spikingjelly.activation_based import functional
    prefix = 'sttmultires_unet.encoders.swin3d.patch_embed.residual_encoding.resblocks.1.'
    modules = dict(model.named_modules())
    theta = float(modules[prefix+'sn1.spiking_neuron'].thresh)
    conv = modules[prefix+'conv1.0']
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/'operator.npz', weight=conv.weight.detach().cpu().numpy(),
        bias=conv.bias.detach().cpu().numpy() if conv.bias is not None else np.zeros(conv.out_channels),
        source_theta=np.array(theta), stride=np.array(conv.stride), padding=np.array(conv.padding))
    timestamps_path = args.data.parent/'train_optical_flow/zurich_city_09_a/flow/forward_timestamps.txt'
    timestamps = np.loadtxt(timestamps_path, delimiter=',', dtype=np.int64)
    rows = []
    current = {}

    def source_hook(module, inputs):
        z = inputs[0].detach()
        active = z != 0
        residual = float((z-active.to(z.dtype)*theta).abs().max())
        record = {'source_shape': list(z.shape), 'source_theta': theta,
                  'nonzero_count': int(active.sum()), 'max_value_residual_to_theta_g': residual}
        path = args.output/(current['stem']+'_source.npz')
        if residual == 0:
            np.savez_compressed(path,
                source_bits=np.packbits(active.cpu().numpy().reshape(-1), bitorder='little'),
                source_shape=np.array(z.shape), source_theta=np.array(theta))
            record['format'] = 'little_endian_gate_bits_plus_continuous_theta'
        else:
            np.savez_compressed(path, source=z.cpu().numpy())
            record['format'] = 'full_fp32_source'
        current.update(record)

    handle = conv.register_forward_pre_hook(source_hook)
    with torch.no_grad():
        for i in range(1, 5):
            name = f'zurich_city_09_a_{i:04d}.npy'
            current = {'file': name, 'stem': name[:-4],
                       'timestamp_start_us': int(timestamps[i-1, 0]),
                       'timestamp_end_us': int(timestamps[i-1, 1])}
            functional.reset_net(model)
            x, _, _ = input_frame(args.data, name, targets=False)
            flow = model(x)['flow'][-1]
            if i < 4:
                np.savez_compressed(args.output/(current['stem']+'_flow.npz'),
                                    flow=flow.detach().cpu().numpy())
            rows.append(dict(current))
            print('CAPTURE', json.dumps(current), flush=True)
            del x, flow
    handle.remove()
    save_json(args.output/'frames.json', {'module': prefix+'conv1.0', 'frames': rows,
        'timestamp_source': str(timestamps_path),
        'protocol': 'Native ep34 dynamic BN, full resolution; only an earlier completed prediction may schedule its successor',
        'prediction_endpoint': "model(x)['flow'][-1]", 'weight_file': 'operator.npz'})


if __name__ == '__main__':
    main()
