"""Two ordinary raw-coordinate temporal controls, fixed64+256 GT updates.

Stage1 starts at the original control3 (not a recovered/shared student), using
the saved permuted diagonal or jointly fitted diagonal+rank2 Ap. Stage2 starts
at each own64-step endpoint and reuses the existing shared train128.json list
and256-step order verbatim, with a fresh Adam1e-4. Fixed P, rank, parameters,
surrogates, parent and loss; only each stage's final diverse10 is evaluated.

This is a thin entry to the existing controller/trainer, not a second network
loader. Root launches GPU. --self-check is CPU-only; --check-only runs the
first TRAIN-frame zero-update dense-reference/gradient check on both axes.
--evaluate-saved reloads the same parameterization/forward for diverse/valid.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize

from train_shared_temporal_recovery import (
    HERE, RAW_AXES, SEED, LR, RawTemporal, raw_initializations,
    read_arrays, gradient_stats, main as run_stage,
)


def self_check(path):
    torch.set_num_threads(2)
    arrays = read_arrays(path)
    initial = raw_initializations(arrays)
    rng = torch.Generator().manual_seed(SEED)
    x, target = torch.randn(10, 29, generator=rng), torch.randn(10, 29, generator=rng)
    report = {}
    for axis in RAW_AXES:
        source = nn.Parameter(torch.from_numpy(arrays['source_A'].astype(np.float32)).clone())
        consumer = nn.Linear(10, 10, bias=False)
        consumer.weight.requires_grad_(False)
        mapping = RawTemporal(consumer.weight, initial[axis], axis == RAW_AXES[1])
        parametrize.register_parametrization(consumer, 'weight', mapping)
        dense = consumer.weight.detach().clone()
        mismatch = int(torch.count_nonzero(consumer.weight@x-dense@x))
        source_gradient = torch.autograd.grad((consumer.weight@x).square().sum(), source,
                                             allow_unused=True)[0]
        loss = (source@x-target).square().mean()+(consumer.weight@x+target).square().mean()
        loss.backward()
        parameters = dict(source_A=source, e=mapping.e)
        if mapping.residual:
            parameters.update(L2=mapping.left, R2=mapping.right)
        gradients = {name: gradient_stats(value) for name, value in parameters.items()}
        torch.optim.Adam(list(parameters.values()), lr=LR).step()
        actual = consumer.weight.detach()
        expected = mapping.dense_double().detach().float()
        assert mismatch == 0 and torch.equal(actual, expected) and source_gradient is None
        assert all(g['present'] and g['all_finite'] and g['nonzero'] for g in gradients.values())
        assert not consumer.parametrizations.weight.original.requires_grad
        report[axis] = dict(initial_dense_forward_differences=mismatch,
            initial_saved_A32_max_abs=float((dense-torch.from_numpy(initial[axis]['saved_dense_A'])).abs().max()),
            raw_A_has_no_source_A_gradient=True, after_update_constraint_max_abs=float((actual-expected).abs().max()),
            permutation=mapping.permutation.tolist(), gradients=gradients,
            raw_projection_trainable_coefficients=sum(p.numel() for n,p in parameters.items() if n != 'source_A'),
            parameter_dtype=str(mapping.e.dtype))
    print('RAW_TEMPORAL_CPU', json.dumps(dict(pass_check=True, axes=report,
        scope='Actual saved matrices and small CPU gradients only; full-network TRAIN-frame check and AEE remain for root.'), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--evaluate-saved', type=Path)
    parser.add_argument('--split', choices=('diverse', 'valid'), default='diverse')
    parser.add_argument('--count', type=int, default=10)
    args = parser.parse_args()
    chain = (args.root/'algorithm/patch_probe/residual_consumer_probe/projection_chain'
             if args.root is not None else HERE)
    parameters = chain/'shared_temporal_lowrank_control.npz'
    if args.self_check:
        self_check(parameters)
        return
    if args.root is None:
        parser.error('--root is required for real-network execution.')
    common = ['--root', str(args.root), '--raw-structured-parameters', str(parameters)]
    if args.evaluate_saved is not None:
        if args.check_only:
            parser.error('--evaluate-saved and --check-only are separate modes.')
        argv = common+['--evaluate-saved', str(args.evaluate_saved),
                       '--split', args.split, '--count', str(args.count)]
        if args.output is not None:
            argv += ['--output', str(args.output)]
        run_stage(argv)
        return
    if args.split != 'diverse' or args.count != 10:
        parser.error('Training is fixed to end-of-stage diverse10; split/count are evaluation-only.')
    output = args.output or chain/'temporal_structured_recovery'
    if args.check_only:
        run_stage(common+['--output', str(output/'check'), '--check-only'])
        return
    run_stage(common+['--output', str(output/'stage64')])
    gc.collect()
    torch.cuda.empty_cache()
    run_stage(common+['--extend-saved', str(output/'stage64'),
        '--extension-plan', str(chain/'shared_temporal_recovery128x256/train128.json'),
        '--output', str(output/'stage128x256')])


if __name__ == '__main__':
    main()
