"""Matched-marginal local-NR4 versus shuffled-NR4 bounded follow-up.

Both axes restart from the original integer bits3 source and use the same
256 updates, train32/P4/C384 cache, native teacher labels, minibatches, fixed
consumer, quantization, and lambda. Only the SOURCE projections/decision
biases train. The target eight-class marginal for a minibatch is the r1
distillation source evaluated on exactly that training minibatch. No valid
labels, costs, or fitted constants enter training.

The NR objective is soft distinct / soft nonzero after current-hard stable
NRV packing. The shuffled control permutes complete source-channel rows,
jointly across the four spatial positions, before applying the SAME packing.
Its permutation is redrawn each minibatch using a separate fixed RNG. The
marginal penalty uses hard forward counts with straight-through gradients.
This is a penalty, not an assertion of exact matched marginals: actual train
and valid deviations are reported, and must be checked before attribution.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.nn import functional as F

import run_probe as probe

AXES = ('local_nr4_marginal', 'shuffled_nr4_marginal')


def nr_ratio(probability, codes):
    grouped, _ = probe.pack_groups(probability, codes)
    nonzero = grouped[..., 1:].sum((1, 2, 3, 4)).mean()
    distinct = (1 - (1 - grouped[..., 1:]).prod(3)).sum((1, 2, 3)).mean()
    return distinct / nonzero.clamp_min(1), distinct, nonzero


def shuffled_groups(probability, codes, generator):
    b, p, c, _ = probability.shape
    order = torch.stack([torch.randperm(c, generator=generator, device=codes.device) for _ in range(b)])
    return (probability.gather(2, order[:, None, :, None].expand(b, p, c, 8)),
            codes.gather(2, order[:, None].expand(b, p, c)))


def self_check():
    gen = torch.Generator().manual_seed(9231)
    codes = torch.randint(0, 8, (4, 4, 12), generator=gen)
    codes[0, :, :5] = 0
    probability = F.one_hot(codes, 8).float().requires_grad_()
    shuffled, scodes = shuffled_groups(probability, codes, gen)
    assert torch.equal(probability.sum((0, 1, 2)), shuffled.sum((0, 1, 2)))
    assert torch.equal((codes != 0).any(1).sum(1), (scodes != 0).any(1).sum(1))
    ratio, distinct, nonzero = nr_ratio(probability, codes)
    expected_nonzero = codes.ne(0).sum() / len(codes)
    assert float(nonzero) == float(expected_nonzero)
    assert 0 <= float(ratio) <= 1
    ratio.backward()
    assert torch.isfinite(probability.grad).all()
    print('SELF_CHECK_PASS marginal/row-live preservation and exact onehot ratio', flush=True)


def measure_training(producer, xq, dictionary, reference_hist):
    histogram = torch.zeros(8, dtype=torch.int64, device='cuda')
    total_distinct = total_nonzero = 0
    with torch.no_grad():
        for frame in xq:
            _, code, _ = producer(frame)
            histogram += torch.bincount(code.flatten(), minlength=8)
            onehot = F.one_hot(code, 8).float()
            _, distinct, nonzero = nr_ratio(onehot, code)
            total_distinct += round(float(distinct) * len(frame))
            total_nonzero += round(float(nonzero) * len(frame))
    marginal = histogram.double() / histogram.sum()
    target = reference_hist.double() / reference_hist.sum()
    difference = marginal - target
    return {'class_histogram': histogram.cpu().tolist(), 'class_marginal': marginal.cpu().tolist(),
            'source_nonzero': total_nonzero, 'source_nonzero_rate': float(1-marginal[0]),
            'true_nr4_distinct': total_distinct, 'distinct_per_nonzero': total_distinct / max(total_nonzero, 1),
            'spike_rate': float((marginal * dictionary.double().sum(1)).sum()/10),
            'marginal_total_variation_from_train_distill': float(difference.abs().sum()/2),
            'marginal_max_absolute_deviation': float(difference.abs().max()),
            'marginal_difference': difference.cpu().tolist(),
            'scope': 'All 32 captured train inputs, P4 spatial blocks and full C384; upstream held at original integer student during capture'}


def fit(args, system, out):
    from evaluate_stage2_deployment import S2, tag
    from train_mfpsn_probe import ATanSpike
    _, modules, gpu, _, initial, dictionaries, _, _ = system
    source = out.parent
    r1 = torch.load(source/'parameters.pt', map_location='cpu', weights_only=False)['distill']
    old_norm = json.loads((source/'normalization.json').read_text())
    parameters = {axis: {} for axis in AXES}
    history, train_stats, normalization = {}, {}, {}
    for mi, p in enumerate(S2):
        name = tag(p)
        cache = torch.load(source/(name+'_train.pt'), map_location='cpu', weights_only=False)
        xq = cache['xq'].cuda().float()
        labels = cache['target'].cuda().long()
        base = {k: v.cuda() if torch.is_tensor(v) else v for k, v in initial[p].items()}
        ref_record = {k: v.cuda() if torch.is_tensor(v) else v for k, v in r1[p].items()}
        scale = torch.tensor(old_norm[name]['margin_std'], device='cuda')
        starting = probe.QuantizedProducer(base, scale).cuda()
        reference = probe.QuantizedProducer(ref_record, scale).cuda()
        with torch.no_grad():
            reference_codes = torch.cat([reference(frame)[1][None] for frame in xq], 0)
            reference_hist = torch.bincount(reference_codes.flatten(), minlength=8)
            reference_marginal = reference_hist.double()/reference_hist.sum()
            check = torch.cat([starting(frame)[1][None] for frame in xq], 0)
            assert torch.equal(check.cpu().byte(), cache['start_codes'])
            pp, cc, _ = starting(xq[:4].reshape(-1, 4, 384, 10))
            ratio_scale = float(nr_ratio(pp, cc)[0])
            del check, pp, cc
        b = (gpu[p]['B'].float() @ gpu[p]['E'].T)[:, 1:]
        w, tau = gpu[p]['w'].float(), gpu[p]['tau'].float()
        w2 = modules[p+'fc2'].weight.detach()
        dictionary = dictionaries[p]
        with torch.no_grad():
            sums = torch.zeros((10, w.shape[0]), dtype=torch.float64, device='cuda')
            squares = torch.zeros_like(sums)
            count = 0
            for frame_codes in labels:
                partial = F.one_hot(frame_codes.reshape(-1, 384), 8)[..., 1:].float().permute(0, 2, 1) @ w.T
                margin = torch.einsum('tk,pkh->tph', b, partial)-tau[:, None]
                sums += margin.double().sum(1)
                squares += margin.double().square().sum(1)
                count += margin.shape[1]
            sigma = (squares/count-(sums/count).square()).clamp_min(1).sqrt().float()[:, None]
            del sums, squares, partial, margin
        normalization[name] = {'train_distill_target_histogram': reference_hist.cpu().tolist(),
                               'train_distill_target_marginal': reference_marginal.cpu().tolist(),
                               'initial_soft_ratio_scale': ratio_scale, 'margin_std': scale.cpu().tolist()}
        history[name], train_stats[name] = {}, {}
        train_stats[name]['target_distill'] = measure_training(reference, xq, dictionary, reference_hist)
        train_stats[name]['start_integer'] = measure_training(starting, xq, dictionary, reference_hist)
        print('START_R2', name, json.dumps(normalization[name]), flush=True)
        for axis in AXES:
            producer = copy.deepcopy(starting)
            optimizer = torch.optim.Adam(producer.parameters(), lr=args.lr)
            minibatch_rng = torch.Generator(device='cuda').manual_seed(9210+mi)
            shuffle_rng = torch.Generator(device='cuda').manual_seed(9410+mi)
            trace = []
            for step in range(args.steps):
                frame = step % xq.shape[0]
                selected = torch.randperm(xq.shape[1], generator=minibatch_rng, device='cuda')[:4]
                inputs, targets = xq[frame, selected], labels[frame, selected]
                probability, codes, _ = producer(inputs)
                ce = F.nll_loss(probability.clamp_min(1e-12).log().reshape(-1, 8), targets.reshape(-1))
                assignment = F.one_hot(codes, 8).float()+(probability-probability.detach())
                partial = assignment.reshape(-1, 384, 8)[..., 1:].permute(0, 2, 1) @ w.T
                margin = torch.einsum('tk,pkh->tph', b, partial)-tau[:, None]
                with torch.no_grad():
                    target_partial = F.one_hot(targets.reshape(-1, 384), 8)[..., 1:].float().permute(0, 2, 1) @ w.T
                    target_margin = torch.einsum('tk,pkh->tph', b, target_partial)-tau[:, None]
                    target_fc2 = F.linear(target_margin.ge(0).float()*gpu[p]['theta_output'], w2)
                    target_marginal = F.one_hot(reference_codes[frame, selected], 8).float().mean((0, 1, 2))
                normalized = margin/sigma
                consumer = F.mse_loss(normalized, target_margin/sigma)
                hidden = ATanSpike.apply(normalized)*gpu[p]['theta_output']
                fc2 = F.mse_loss(F.linear(hidden, w2), target_fc2)/target_fc2.var(unbiased=False).clamp_min(.01)
                marginal = assignment.mean((0, 1, 2))
                marginal_cost = ((marginal-target_marginal).square()/target_marginal.clamp_min(.005)).sum()
                if axis == 'shuffled_nr4_marginal':
                    nr_prob, nr_codes = shuffled_groups(probability, codes, shuffle_rng)
                else:
                    nr_prob, nr_codes = probability, codes
                ratio, soft_distinct, soft_nonzero = nr_ratio(nr_prob, nr_codes)
                loss = ce+.25*consumer+.1*fc2+args.regularizer_weight*ratio/ratio_scale+args.marginal_weight*marginal_cost
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if step % 64 == 0 or step+1 == args.steps:
                    with torch.no_grad():
                        true_hard = nr_ratio(F.one_hot(codes, 8).float(), codes)
                    row = {'step': step+1, 'loss': float(loss), 'ce': float(ce), 'consumer_mse': float(consumer),
                           'fc2_mse': float(fc2), 'marginal_cost': float(marginal_cost),
                           'marginal_TV_batch': float((marginal-target_marginal).abs().sum()/2),
                           'soft_group_ratio': float(ratio), 'soft_group_distinct': float(soft_distinct),
                           'soft_group_nonzero': float(soft_nonzero), 'true_hard_ratio': float(true_hard[0]),
                           'true_hard_distinct': float(true_hard[1]), 'true_hard_nonzero': float(true_hard[2])}
                    trace.append(row)
                    print('FIT_R2', name, axis, json.dumps(row), flush=True)
            parameters[axis][p] = producer.export(initial[p])
            history[name][axis] = trace
            train_stats[name][axis] = measure_training(producer, xq, dictionary, reference_hist)
            print('TRAIN_R2', name, axis, json.dumps(train_stats[name][axis]), flush=True)
            torch.save(parameters, out/'parameters.pt')
            probe.save_json(out/'training.json', history)
            probe.save_json(out/'train_stats.json', train_stats)
            probe.save_json(out/'normalization.json', normalization)
            del producer, optimizer
        del xq, labels, cache, starting, reference, reference_codes
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--lr', type=float, default=.002)
    parser.add_argument('--regularizer-weight', type=float, default=.1)
    parser.add_argument('--marginal-weight', type=float, default=20)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.root is None:
        parser.error('--root is required')
    out = args.root/'algorithm/nrv_cost_probe/r2_matched'
    out.mkdir(parents=True, exist_ok=True)
    args.blocks, args.save_codes, args.skip_start = 32, True, True
    probe.save_json(out/'run.json', {'axes': list(AXES), 'steps_each_module': args.steps, 'lr': args.lr,
                    'same_integer_start': 'algorithm/direct_code_integer/parameters.pt',
                    'NR_ratio_weight': args.regularizer_weight, 'marginal_weight': args.marginal_weight,
                    'marginal_weight_selection': 'fixed before training, no validation tuning or sweep',
                    'marginal_target': 'r1 distill final parameters on same train32 captured inputs; matching same minibatch marginal',
                    'marginal_penalty': 'sum((hard_STE_class_marginal-target)^2/max(target,.005)); all eight classes include zero',
                    'NR_cost': 'soft distinct / soft nonzero; both axes share initial true-local normalization',
                    'shuffled_control': 'fresh random channel-row permutation per P4 block, shared across four positions, then current-hard stable NRV and NR4',
                    'minibatches': 'identical fixed-seed four P4 blocks; complete C384/T10; shuffle uses separate RNG',
                    'train_scope': 'only source W3x10 and source decision biases; consumer B/W/tau, dictionary/map and theta fixed',
                    'interpretation_gate': 'do not claim matched-marginal causal benefit unless actual train/valid marginal differences are small; report all differences',
                    'evaluation': 'same valid10 only, complete real network and full source code capture; not cycles or full825'})
    system = probe.load_system(args)
    fit(args, system, out)
    probe.evaluate(args, system, out)


if __name__ == '__main__':
    main()
