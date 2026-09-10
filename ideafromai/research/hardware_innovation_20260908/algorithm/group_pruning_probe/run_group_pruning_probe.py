"""Matched C16/H8 pruning probe on the existing full-T10 integer student.

The reference implements the output-grouping branch of HiNM/Gyro, specialized
to common physical source blocks. This is not a complete HiNM, FlexHiNM or
VENOM reproduction: no input permutation, inner 2:4, OBS or region search.
The experimental axis changes the assignment objective to exact within-group
T10 gate / FC2 distortion. Inter-group cross terms are checked after grouping,
and complete network evaluation retains the real FC2, BN2 and shortcut.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'pruning_probe'))
import run_pruning_probe as base
from hinm_reference import gyro_group, shared_keep

probe, PREFIX = base.probe, base.PREFIX
AXES = ('consecutive_w2', 'gyro_membrane', 'gyro_fc2_separable', 'gyro_fc2_exact_diagonal', 'gyro_fc2_joint')


def theta_vector(theta, h):
    value = torch.as_tensor(theta, device='cuda', dtype=torch.float32).flatten()
    return value.expand(h).clone() if value.numel() == 1 else value.clone()


@torch.no_grad()
def prepare(system, codes, out):
    _, modules, gpu, _, _, _, _, _ = system
    q = gpu[PREFIX]
    w, tau, b, e = q['w'].clone(), q['tau'].float().clone(), q['B'].float().clone(), q['E'].clone()
    h, c = w.shape
    theta = theta_vector(q['theta_output'], h)
    fc2 = modules[PREFIX+'fc2']
    # All 32 train frames and 32 genuine contiguous P4 blocks per frame.
    flat = codes.reshape(-1, c)
    z = torch.einsum('tr,pcr->tpc', b, e[flat])
    margin = z @ w.T-tau[:, None]
    factorized = torch.einsum('tr,prh->tph', b.double(),
                             (e[flat[:16]].permute(0, 2, 1) @ w.T).double())-tau[:, None].double()
    assert torch.equal(margin[:, :16].double(), factorized), 'full-C integer factorization'
    sigma = margin.double().var(dim=1, unbiased=False).clamp_min(1).sqrt().float()
    standardized = margin/sigma[:, None]
    derivative = 1/(1+(torch.pi*standardized).square())
    column_norm = fc2.weight.square().mean(0)*theta.square()
    membrane_scores, consumer_scores = [], []
    for block in range(c//16):
        sl = slice(16*block, 16*(block+1))
        delta = (z[..., sl] @ w[:, sl].T)/sigma[:, None]
        membrane_scores.append(delta.square().mean((0, 1)))
        consumer_scores.append((delta*derivative).square().mean((0, 1))*column_norm)
    # Eight fixed uniformly spaced P4 blocks in EVERY training frame. This
    # covers the image instead of accidentally calibrating only its corner.
    spatial = torch.linspace(0, codes.shape[1]-1, 8, device='cuda').round().long()
    selection = torch.arange(codes.shape[0], device='cuda')[:, None, None]*(codes.shape[1]*codes.shape[2])
    selection = (selection+spatial[None, :, None]*codes.shape[2]
                 +torch.arange(codes.shape[2], device='cuda')[None, None]).flatten()
    small_z, small_margin = z[:, selection], margin[:, selection]
    block_u = torch.stack([(small_z[..., j:j+16] @ w[:, j:j+16].T).permute(2, 0, 1).flatten(1)
                           for j in range(0, c, 16)], dim=1).contiguous()
    target_hidden = small_margin.ge(0).float()*theta
    target_fc2 = F.linear(target_hidden, fc2.weight, fc2.bias)
    scales = {'fc2_variance': float(target_fc2.var(unbiased=False).clamp_min(.01)),
              'fc2_outputs': fc2.weight.shape[0]}
    prepared = dict(w=w, tau=tau, b=b, e=e, sigma=sigma, theta=theta,
                    membrane=torch.stack(membrane_scores, 1).cpu().numpy(),
                    consumer=torch.stack(consumer_scores, 1).cpu().numpy(),
                    block_u=block_u, margin=small_margin.permute(2, 0, 1).flatten(1).contiguous(),
                    gram=fc2.weight.T @ fc2.weight, scales=scales,
                    z=small_z, fc2_weight=fc2.weight, fc2_bias=fc2.bias)
    probe.save_json(out/'preparation.json', dict(
        training_frames=codes.shape[0], positions_per_frame=codes.shape[1]*codes.shape[2],
        saliency_positions=len(flat), joint_positions=len(selection), timesteps=10,
        joint_spatial_blocks=spatial.tolist(),
        H=h, C=c, factor_rank=b.shape[1], theta_min=float(theta.min()), theta_max=float(theta.max()),
        score='block Gauss-Newton reconstruction, ATan derivative alpha2; no zero-loss empirical Fisher',
        fc2_scores='per-h squared Jacobian; within-block cross terms retained',
        joint='exact hard gates and actual FC2 column Gram within each H8; no inter-group cross terms in search',
        sigma_min=float(sigma.min()), sigma_max=float(sigma.max()), **scales))
    print('PREPARED', json.dumps(json.loads((out/'preparation.json').read_text())), flush=True)
    return prepared


class JointCost:
    def __init__(self, prepared, saliency, diagonal=False, keep_blocks=12):
        self.p, self.saliency, self.calls = prepared, saliency, 0
        self.diagonal = diagonal
        self.keep_blocks = keep_blocks

    @torch.no_grad()
    def __call__(self, groups):
        p = self.p
        index = torch.as_tensor(groups, device='cuda', dtype=torch.long)
        removed = torch.as_tensor(~shared_keep(self.saliency, groups, self.keep_blocks), device='cuda')
        contribution = p['block_u'][index]
        delta = (contribution*removed[:, None, :, None]).sum(2)
        margin = p['margin'][index]
        dg = (margin.sub(delta).ge(0).float()-margin.ge(0).float())*p['theta'][index, None]
        covariance = (dg @ dg.transpose(1, 2))/dg.shape[-1]
        gram = p['gram'][index[:, :, None], index[:, None, :]]
        product = covariance*gram
        result = (product.diagonal(dim1=1, dim2=2).sum(1) if self.diagonal else product.sum((1, 2)))
        result = result/(p['scales']['fc2_outputs']*p['scales']['fc2_variance'])
        self.calls += len(groups)
        if self.calls % 32768 < len(groups):
            print('JOINT_CANDIDATES', self.calls, flush=True)
        return result.clamp_min(0).cpu().numpy().astype(np.float64)


@torch.no_grad()
def local_metrics(p, physical_weight, physical_tau, perm):
    z = p['z']
    margin = z @ physical_weight.T-physical_tau[:, None]
    theta = p['theta'][perm]
    gates = margin.ge(0).float()
    target_gate = p['margin'].reshape(len(perm), 10, -1).permute(1, 2, 0).ge(0).float()
    y = F.linear(gates*theta, p['fc2_weight'][:, perm], p['fc2_bias'])
    target_y = F.linear(target_gate*p['theta'], p['fc2_weight'], p['fc2_bias'])
    return dict(gate_disagreement=float((gates != target_gate[..., perm]).float().mean()),
                complete_fc2_mse=float(F.mse_loss(y, target_y)),
                complete_fc2_normalized_mse=float(F.mse_loss(y, target_y)/p['scales']['fc2_variance']),
                activity=float(gates.mean()))


def make_record(consumer, p, perm, axis):
    record = consumer.export(p['b'], p['e'], p['w'][perm])
    record['physical_to_logical_h'] = perm.cpu()
    record['theta_output_physical'] = p['theta'][perm].cpu()
    record['mask_rule'] = axis
    record['complete_T10_and_real_FC2'] = True
    return record


def search(args, p, out):
    results, records = {}, {}
    summaries = json.loads((out/'search.json').read_text()) if (out/'search.json').exists() else {}
    h, c = p['w'].shape
    original_groups = np.arange(h).reshape(-1, 8)
    w2 = p['w'].double().square().reshape(h, c//16, 16).sum(-1).cpu().numpy()
    for axis in args.axes:
        started = time.monotonic()
        saliency = (w2 if axis == 'consecutive_w2' else
                    p['membrane'] if axis == 'gyro_membrane' else p['consumer'])
        joint = (JointCost(p, saliency, diagonal=axis.endswith('diagonal'), keep_blocks=args.keep_blocks)
                 if axis in ('gyro_fc2_joint', 'gyro_fc2_exact_diagonal') else None)
        if axis == 'consecutive_w2':
            groups = original_groups.copy()
            block_mask = np.zeros((h, c//16), bool)
            block_mask[groups] = shared_keep(saliency, groups, args.keep_blocks)[:, None]
            result = dict(groups=groups, perm=groups.ravel(), mask_original=np.repeat(block_mask, 16, 1),
                          history=[], objective='consecutive_W_squared', group_evaluations=0)
        else:
            result = gyro_group(saliency, keep_blocks=args.keep_blocks, group_cost=joint, callback_batch=args.callback_batch,
                                stop_on_nonimprove=False, seed=9523)
        perm = torch.as_tensor(result['perm'], device='cuda', dtype=torch.long)
        mask = torch.as_tensor(result['mask_original'], device='cuda')[perm]
        consumer = base.PrunedConsumer(p['w'][perm], p['tau'][:, perm], p['sigma'][:, perm],
                                       mask, torch.ones(h, dtype=torch.bool, device='cuda')).cuda()
        record = make_record(consumer, p, perm, axis)
        name = axis+'_initial'
        records[name] = record
        base.save_record(out, name, record)
        metrics = local_metrics(p, record['weight_int8'].cuda().float(),
                                record['threshold_int32'].cuda().float(), perm)
        summaries[axis] = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
        summaries[axis].update(seconds=time.monotonic()-started, **metrics)
        np.savez_compressed(out/(axis+'_grouping.npz'),
                            **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
        results[axis] = result
        probe.save_json(out/'search.json', summaries)
        print('SEARCH_DONE', axis, json.dumps(summaries[axis]), flush=True)
    return records


def train(args, system, p, codes, initials, out):
    from train_mfpsn_probe import ATanSpike
    variants = {}
    histories = json.loads((out/'training.json').read_text()) if (out/'training.json').exists() else {}
    h = p['w'].shape[0]
    for axis in args.axes:
        rec = initials[axis+'_initial']
        perm = rec['physical_to_logical_h'].cuda()
        mask = rec['weight_mask'].cuda()
        consumer = base.PrunedConsumer(p['w'][perm], p['tau'][:, perm], p['sigma'][:, perm],
                                       mask, torch.ones(h, device='cuda', dtype=torch.bool)).cuda()
        optimizer = torch.optim.Adam([{'params': [consumer.weight], 'lr': args.weight_lr},
                                      {'params': [consumer.offset], 'lr': args.threshold_lr}])
        generator = torch.Generator(device='cuda').manual_seed(9523)
        trace = []
        for step in range(args.steps):
            frame = step % codes.shape[0]
            blocks = torch.randperm(codes.shape[1], generator=generator, device='cuda')[:4]
            inputs = codes[frame, blocks].reshape(-1, p['w'].shape[1])
            membership = p['e'][inputs].permute(0, 2, 1)
            with torch.no_grad():
                target_margin = torch.einsum('tr,prh->tph', p['b'], membership @ p['w'].T)-p['tau'][:, None]
                target_hidden = target_margin.ge(0).float()*p['theta']
                target_fc2 = F.linear(target_hidden, p['fc2_weight'], p['fc2_bias'])
            w, tau = consumer.quantized()
            margin = torch.einsum('tr,prh->tph', p['b'], membership @ w.T)-tau[:, None]
            norm = margin/p['sigma'][:, None, perm]
            membrane = F.mse_loss(norm, target_margin[:, :, perm]/p['sigma'][:, None, perm])
            hidden = ATanSpike.apply(norm)*p['theta'][perm]
            prediction = F.linear(hidden, p['fc2_weight'][:, perm], p['fc2_bias'])
            downstream = F.mse_loss(prediction, target_fc2)/target_fc2.var(unbiased=False).clamp_min(.01)
            loss = membrane+.1*downstream
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                consumer.weight.mul_(mask).clamp_(-128, 127)
            if step % 64 == 0 or step+1 == args.steps:
                row = dict(step=step+1, loss=float(loss), membrane_mse=float(membrane),
                           fc2_normalized_mse=float(downstream),
                           gate_disagreement=float((hidden.ne(0) != target_hidden[..., perm].ne(0)).float().mean()))
                trace.append(row)
                print('FIT_GROUP', axis, json.dumps(row), flush=True)
        record = make_record(consumer, p, perm, axis)
        name = axis+'_trained'
        variants[name] = record
        base.save_record(out, name, record)
        histories[axis] = dict(trace=trace, final_local=local_metrics(p, record['weight_int8'].cuda().float(),
                              record['threshold_int32'].cuda().float(), perm))
        probe.save_json(out/'training.json', histories)
        del consumer, optimizer
    return variants


def install(system, record):
    _, modules, gpu, _, _, _, _, _ = system
    q = gpu[PREFIX]
    q['w'] = record['weight_int8'].cuda().float()
    q['tau'] = record['threshold_int32'].cuda().double()
    q['physical_to_logical_h'] = record['physical_to_logical_h'].cuda()
    q['theta_physical'] = record['theta_output_physical'].cuda()

    def forward(self, x, compiled=q):
        if self.norm_layer in ('LN', 'GN'):
            x = self.norm(x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        source = self.drop1(self.sn1(x))
        shape = source.shape
        shifts = torch.arange(10, device=x.device).reshape(10, *([1]*(source.ndim-1)))
        raw = ((source != 0).long() << shifts).sum(0)
        codes = compiled['map'][raw].reshape(-1, compiled['w'].shape[1])
        partial = compiled['E'][codes].permute(0, 2, 1) @ compiled['w'].T
        u = torch.einsum('tr,prh->tph', compiled['B'], partial.double())
        hidden = (u.ge(compiled['tau'][:, None]).float()*compiled['theta_physical']).reshape(*shape[:-1], -1)
        out = F.linear(self.drop2(hidden), self.fc2.weight[:, compiled['physical_to_logical_h']], self.fc2.bias)
        if self.norm_layer in ('BN', 'BNTT', 'tdBN', 'IN'):
            out = self.bn2(out.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        return out

    mlp = modules[PREFIX.rstrip('.')]
    mlp.forward = types.MethodType(forward, mlp)


def evaluate(args, system, variants, out):
    # Retain existing complete network validation and actual downstream captures.
    base.install_pruned_consumer = install
    summaries = json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else {}
    base.evaluate(args, system, variants, out, summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('all', 'search', 'fit_eval', 'eval'), default='all')
    parser.add_argument('--axes', nargs='+', choices=AXES, default=list(AXES))
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--weight-lr', type=float, default=.03)
    parser.add_argument('--threshold-lr', type=float, default=.003)
    parser.add_argument('--callback-batch', type=int, default=128)
    parser.add_argument('--keep-blocks', type=int, default=12)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    out = args.out or args.root/'algorithm/group_pruning_probe'
    out.mkdir(parents=True, exist_ok=True)
    system = probe.load_system(args)
    network_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    cache = torch.load(args.root/'algorithm/nrv_cost_probe/s2b3_train.pt', map_location='cpu', weights_only=False)
    codes = cache['start_codes'].cuda().long()
    probe.save_json(out/'run.json', dict(module=PREFIX, axes=args.axes, steps=args.steps,
        budget=f'each H8 keeps {args.keep_blocks} of24 common C16 blocks; no hidden deletion',
        search_schedule=[8, 1, 4, 1, 2, 1, 1], search_stop='matched fixed-budget, best-so-far',
        fit='same 256 minibatches, surviving W8 + tau24 only, same membrane + .1 FC2 loss',
        identity='existing integer bits3 student; theta*g and full noncausal T10; not native ep34',
        training_matmul_tf32=False, inference_matmul_tf32=network_tf32,
        comparison='restricted output-grouping migration, not complete HiNM/VENOM/FlexHiNM',
        strong_external_controls='pruning_probe hidden50 / row2:4 / W2-consecutive; same parent and fit budget'))
    if args.stage in ('all', 'search', 'fit_eval'):
        torch.backends.cuda.matmul.allow_tf32 = False
        p = prepare(system, codes, out)
        if args.stage != 'fit_eval':
            initial = search(args, p, out)
        else:
            initial = {axis+'_initial': torch.load(out/(axis+'_initial.pt'), map_location='cpu', weights_only=False)
                       for axis in args.axes}
        if args.stage == 'search':
            return
        trained = train(args, system, p, codes, initial, out)
        # Unpruned permutation must preserve full-T10 gates and the FC2 mapping.
        for axis in args.axes:
            perm = initial[axis+'_initial']['physical_to_logical_h'].cuda()
            check = local_metrics(p, p['w'][perm], p['tau'][:, perm], perm)
            assert check['gate_disagreement'] == 0 and check['complete_fc2_normalized_mse'] < 1e-9, check
        print('PERMUTATION_CHECK_PASS complete_T10_theta_tau_FC2', flush=True)
        del p, initial
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = network_tf32
        evaluate(args, system, trained, out)
    else:
        trained = {axis+'_trained': torch.load(out/(axis+'_trained.pt'), map_location='cpu', weights_only=False)
                   for axis in args.axes}
        evaluate(args, system, trained, out)


if __name__ == '__main__':
    main()
