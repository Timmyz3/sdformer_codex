"""Fit and evaluate train-only hidden-query readout controls.

Borrow the depth-four tree constraint and lambda=1 ridge prototype step from
MADDNESS.  The fixed-index and node-specific trees come from the same binary
SSE fitting procedure.  Ridge-fitting FC2 partial contributions directly is
algebraically equivalent to ridge-fitting the complete hidden vector and
then multiplying by fixed W2.  This probe uses FP32 tables, not MADDNESS's
quantized aggregation or a measured hardware implementation.

The narrow and fixed-tree modes actually produce only their common 96 hidden
channels.  Dynamic trees currently evaluate the complete hidden tensor on
the GPU; separate request traces are needed for their hardware cost.
"""
import argparse
import json
from pathlib import Path
import types

import numpy as np
import torch
import torch.nn.functional as F

from run_bn_probe import build_model, input_frame, read_names, save_json, set_bn_mode, MLPS, SHALLOW
from run_integer_fc1_probe import install_integer_consumers


def tree_tensors(groups, mode, compact=False):
    features, leaves = [], []
    for group in groups:
        tree = group['trees'][mode]
        f = np.asarray(tree['features_heap']).copy()
        active = f >= 0
        if compact:
            common = tree['common_indices_by_level']
            f[active] = [common.index(int(j)) for j in f[active]]
            f[active] += group['group']*4
        else:
            f[active] += group['group']*16
        features.append(f)
        leaves.append(tree['leaf_index_heap'])
    return (torch.tensor(np.array(features), device='cuda', dtype=torch.long),
            torch.tensor(np.array(leaves), device='cuda', dtype=torch.long))


def encode(x, features, leaves):
    """Four actual feature queries per path; no nearest-code search."""
    flat = x.reshape(-1, x.shape[-1])
    n = flat.shape[0]
    group = torch.arange(24, device=x.device)[None, :].expand(n, -1)
    row = torch.arange(n, device=x.device)[:, None]
    node = torch.zeros((n, 24), dtype=torch.long, device=x.device)
    for _ in range(4):
        feature = features[group, node]
        active = feature >= 0
        branch = flat[row, feature.clamp_min(0)].ne(0).long()
        node = torch.where(active, node*2+1+branch, node)
    return leaves[group, node]


def narrow_producer(modules, params, indices):
    prefix = MLPS[0]
    record = params[prefix]
    w = record['weight_int8'].to('cuda', torch.float32)[indices]
    aq = record['temporal_int16'].to('cuda', torch.float64)
    tau = record['threshold_int64'].to('cuda', torch.float64)[:, indices]
    positive = record['positive_gain'].to('cuda')[indices]
    variable = ~record['constant_channels'].to('cuda')[indices]
    fixed = record['constant_gate'].to('cuda')[:, indices]
    theta_in = record['theta_source'].to('cuda', torch.float32)
    theta_out = record['theta_output'].to('cuda', torch.float32)

    def linear(self, x):
        return F.linear(x/theta_in, w)

    def temporal(self, x):
        shape = x.shape
        u = (aq @ x.double().reshape(shape[0], -1)).reshape(shape[0], -1, shape[-1])
        gate = torch.where(positive[None, None, :], u >= tau[:, None, :], u <= tau[:, None, :])
        gate = torch.where(variable[None, None, :], gate, fixed[:, None, :])
        return gate.reshape(shape).float()*theta_out

    modules[prefix+'fc1'].forward = types.MethodType(linear, modules[prefix+'fc1'])
    sn = modules[prefix+'sn2.spiking_neuron']
    sn.forward = types.MethodType(temporal, sn)


def main():
    parser = argparse.ArgumentParser()
    for name in ('code-root', 'config', 'checkpoint', 'data', 'calibration', 'samples', 'hidden-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--full-valid', action='store_true')
    parser.add_argument('--mode', default='all', choices=['all', 'narrow96', 'fixed_level_index', 'node_specific_index'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tree_file = json.loads((args.hidden_root/'hidden_sse_trees.json').read_text())
    groups = tree_file['groups']
    indices = torch.tensor([g['group']*16+j for g in groups
                            for j in g['trees']['fixed_level_index']['common_indices_by_level']],
                           device='cuda', dtype=torch.long)
    np.save(args.output/'narrow_indices.npy', indices.cpu().numpy())
    run = json.loads((args.hidden_root/'run.json').read_text())
    with np.load(args.hidden_root/'train_hidden.npz') as capture:
        train_bits = np.unpackbits(capture['gate_bits'], axis=-1, bitorder='little')
    theta = float(run['theta'])
    train = torch.from_numpy(train_bits.reshape(-1, 384)).to('cuda', torch.float32)*theta
    del train_bits
    w2 = torch.from_numpy(np.load(args.hidden_root/'fc2_weight.npy')).to('cuda', torch.float32)
    target = train @ w2.T
    modes = ['narrow96', 'fixed_level_index', 'node_specific_index'] if args.mode == 'all' else [args.mode]
    fit_records, fitted = {}, {}
    # The ridge solve is deliberately shared.  No validation selects lambda.
    for mode in modes:
        if mode == 'narrow96':
            design = torch.cat((train[:, indices], torch.ones((len(train), 1), device='cuda')), 1)
        else:
            features, leaves = tree_tensors(groups, mode)
            codes = encode(train, features, leaves)
            design = F.one_hot(codes, num_classes=16).reshape(len(train), 384).float()
        gram = (design.T.double() @ design.double())
        rhs = design.T.double() @ target.double()
        gram.diagonal().add_(1.0)
        table = torch.linalg.solve(gram, rhs).float()
        fit = design @ table
        fitted[mode] = table.detach()
        fit_records[mode] = {'lambda': 1.0, 'training_rows': len(train),
                            'parameters': table.numel(), 'FP32_parameter_bytes': table.numel()*4,
                            'training_relative_l2': float((fit-target).norm()/target.norm()),
                            'fit_target': 'original complete FC2 theta*g*W2 contribution',
                            'producer': '96 exact hidden channels' if mode != 'node_specific_index'
                                        else 'GPU full hidden; hardware demand trace evaluated separately'}
        torch.save(table.cpu(), args.output/(mode+'_readout.pt'))
        print('FIT', mode, json.dumps(fit_records[mode]), flush=True)
        del design, gram, rhs, fit
    del train, target, w2
    torch.cuda.empty_cache()
    save_json(args.output/'fit.json', fit_records)
    names = read_names(args.data, 'valid') if args.full_valid else json.loads(args.samples.read_text())['valid'][:10]
    rows = []
    for mode in modes:
        model, _, _, _ = build_model(args)
        from spikingjelly.activation_based import functional
        stats = torch.load(args.calibration, map_location='cpu', weights_only=False)
        set_bn_mode(model, stats, SHALLOW)
        params, _ = install_integer_consumers(model, stats)
        modules = dict(model.named_modules())
        if mode != 'node_specific_index':
            narrow_producer(modules, params, indices)
        fc2 = modules[MLPS[0]+'fc2']
        fc_bias = fc2.bias.detach() if fc2.bias is not None else torch.zeros(96, device='cuda')
        table = fitted[mode]
        if mode == 'narrow96':
            def forward(self, x):
                return F.linear(x, table[:-1].T, table[-1]+fc_bias)
        else:
            features, leaves = tree_tensors(groups, mode, compact=mode=='fixed_level_index')
            grouped_table = table.reshape(24, 16, 96)
            group_ids = torch.arange(24, device='cuda')[None, :]

            def forward(self, x):
                ids = encode(x, features, leaves)
                out = grouped_table[group_ids, ids].sum(1)+fc_bias
                return out.reshape(*x.shape[:-1], 96)
        fc2.forward = types.MethodType(forward, fc2)
        if not args.full_valid:
            with np.load(args.hidden_root/(Path(names[0]).stem+'_hidden.npz')) as reference:
                expected = torch.from_numpy(np.unpackbits(reference['gate_bits'], axis=-1,
                                           bitorder='little')).to('cuda', torch.float32)*theta
            if mode != 'node_specific_index':
                expected = expected[:, :, indices]

            def check_hidden(module, inputs):
                actual = inputs[0].reshape(10, -1, expected.shape[-1])
                mismatches = int(actual.ne(expected).sum())
                fit_records[mode]['first_frame_hidden_values_checked'] = expected.numel()
                fit_records[mode]['first_frame_hidden_mismatches'] = mismatches
                if mismatches:
                    raise RuntimeError(f'{mode}: compact integer hidden differs from original capture')
                hidden_check.remove()

            hidden_check = fc2.register_forward_pre_hook(check_hidden)
        holder = {}

        def coarse_hook(module, inputs, output):
            holder['flow'] = output.detach().sum(0)

        handle = modules['sttmultires_unet.preds.2'].register_forward_hook(coarse_hook)
        with torch.no_grad():
            for i, name in enumerate(names):
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
                if not args.full_valid or (i+1) % 50 == 0:
                    print('VALID', mode, i+1, {k: float(np.mean([r[k] for r in rows if r['mode']==mode]))
                                              for k in ('final', 'coarse_bilinear')}, flush=True)
                    save_json(args.output/'frames.json', rows)
                del x, label, mask, final, coarse, error
        handle.remove()
        del model, modules, fc2
        torch.cuda.empty_cache()
    save_json(args.output/'frames.json', rows)
    summary = {'frames_per_mode': len(names), 'fit': fit_records,
               'AEE': {mode: {k: float(np.mean([r[k] for r in rows if r['mode']==mode]))
                              for k in ('final', 'coarse_bilinear')} for mode in modes},
               'scope': 'new-model FP32 readout after integer shallow FC1; real BN2, shortcut and remaining network',
               'limits': 'no quantized LUT aggregation, selective dynamic GPU producer, RTL or hardware speed claim',
               'prior': 'MADDNESS ICML2021, binary-input SSE tree transfer plus lambda=1 ridge table fitting'}
    save_json(args.output/'summary.json', summary)
    print('COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
