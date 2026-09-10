"""Export complete T10 hidden supports for a bounded encoder-query study."""
import argparse
import json
from pathlib import Path

from run_bn_probe import build_model, input_frame, save_json, set_bn_mode, MLPS, SHALLOW
from run_integer_fc1_probe import install_integer_consumers
import numpy as np
import torch


class HiddenReady(Exception):
    pass


def main():
    p = argparse.ArgumentParser()
    for name in ('code-root', 'config', 'checkpoint', 'data', 'samples', 'calibration', 'cache', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model, _, _, _ = build_model(args)
    from spikingjelly.activation_based import functional
    stats = torch.load(args.calibration, map_location='cpu', weights_only=False)
    set_bn_mode(model, stats, SHALLOW)
    parameters, _ = install_integer_consumers(model, stats)
    par = parameters[MLPS[0]]
    aq = par['temporal_int16'].cuda().double()
    tau = par['threshold_int64'].cuda()
    positive = par['positive_gain'].cuda()
    constant = par['constant_channels'].cuda()
    fixed = par['constant_gate'].cuda()
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    training = []
    with torch.no_grad():
        for row in cache:
            y = row['yi'].cuda().double()
            u = (aq @ y.reshape(10, -1)).reshape_as(y)
            g = torch.where(positive[None, None], u >= tau[:, None], u <= tau[:, None])
            g = torch.where(constant[None, None], fixed[:, None], g)
            training.append(g.cpu().numpy())
    train = np.concatenate(training, axis=1)
    np.savez_compressed(args.output/'train_hidden.npz',
                        gate_bits=np.packbits(train, axis=-1, bitorder='little'),
                        shape=np.array(train.shape))
    modules = dict(model.named_modules())
    sn = modules[MLPS[0]+'sn2.spiking_neuron']
    current = {}

    def capture(module, inputs, output):
        current['g'] = output.detach().ne(0).reshape(10, -1, 384).cpu().numpy()
        raise HiddenReady()

    handle = sn.register_forward_hook(capture)
    names = json.loads(args.samples.read_text())['valid'][:10]
    with torch.no_grad():
        for name in names:
            functional.reset_net(model)
            x, _, _ = input_frame(args.data, name, targets=False)
            try:
                model(x)
            except HiddenReady:
                g = current.pop('g')
                np.savez_compressed(args.output/(name[:-4]+'_hidden.npz'),
                                    gate_bits=np.packbits(g, axis=-1, bitorder='little'),
                                    shape=np.array(g.shape))
            del x
            print('HIDDEN', name, flush=True)
    handle.remove()
    np.save(args.output/'fc2_weight.npy', modules[MLPS[0]+'fc2'].weight.detach().cpu().numpy())
    save_json(args.output/'run.json', {'module': MLPS[0], 'training_files': [r['file'] for r in cache],
              'training_positions_per_file': 512, 'validation_files': names,
              'theta': float(par['theta_output']), 'numeric_path': 'integer S0 teacher',
              'purpose': 'tree encoder query-union and reconstruction probe; not an AEE or hardware-speed result'})


if __name__ == '__main__':
    main()
