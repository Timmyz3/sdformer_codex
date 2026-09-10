"""One-layer S2b3 50% pruning probe with same fixed integer-code student.

Masks: none, rowwise top2-of4, shared eight-H by C16 half, or top-half hidden
channels. Masks derive only from original FC1 W^2, with lower-index stable
ties. Only surviving W8 and per(t,h) tau24 train; B, dictionary, source,
theta, FC2 weights, BN2 and all other layers remain fixed. Hidden pruning
deletes the selected neurons and corresponding FC2 input columns, rather
than leaving zero-weight neurons with tau-induced nonzero outputs.

This is bounded mask+local-distillation screening, not a reproduction of a
complete pruning paper. Real network validation is ten fixed frames only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'nrv_cost_probe'))
import run_probe as probe

PREFIX = 'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.3.mlp.'
AXES = ('original', 'row_2of4', 'broadcast8_C16_half', 'hidden_H_half')


def masks_from_weight(weight):
    w = np.asarray(weight, dtype=np.int64)
    h, c = w.shape
    assert h % 8 == 0 and c % 32 == 0
    w2 = w*w
    order = np.argsort(-w2.reshape(h, c//4, 4), axis=-1, kind='stable')[..., :2]
    row = np.zeros((h, c//4, 4), dtype=bool)
    np.put_along_axis(row, order, True, axis=-1)
    scores = w2.reshape(h//8, 8, c//16, 16).sum((1, 3))
    chosen = np.argsort(-scores, axis=-1, kind='stable')[:, :c//32]
    blocks = np.zeros_like(scores, dtype=bool)
    np.put_along_axis(blocks, chosen, True, axis=-1)
    shared = np.broadcast_to(blocks[:, None, :, None], (h//8, 8, c//16, 16)).reshape(h, c)
    selected_h = np.argsort(-w2.sum(1), kind='stable')[:h//2]
    hidden = np.zeros(h, dtype=bool)
    hidden[selected_h] = True
    ones = np.ones(h, dtype=bool)
    return {'original': (np.ones_like(w, dtype=bool), ones),
            'row_2of4': (row.reshape(h, c), ones),
            'broadcast8_C16_half': (shared.copy(), ones),
            'hidden_H_half': (np.broadcast_to(hidden[:, None], (h, c)).copy(), hidden)}


class PrunedConsumer(nn.Module):
    def __init__(self, weight, tau, sigma, mask, keep):
        super().__init__()
        self.weight = nn.Parameter(weight.clone()*mask)
        self.offset = nn.Parameter(torch.zeros_like(tau))
        self.register_buffer('reference_tau', tau.clone())
        self.register_buffer('sigma', sigma.clone())
        self.register_buffer('mask', mask.bool())
        self.register_buffer('keep', keep.bool())

    def quantized(self):
        weight = probe.ste_round(self.weight).clamp(-128, 127)*self.mask
        raw_tau = self.reference_tau-self.offset*self.sigma
        tau = raw_tau+(raw_tau.ceil()-raw_tau).detach()
        tau = tau.clamp(-(1 << 23), (1 << 23)-1)
        return weight, tau

    def export(self, b, e, original):
        with torch.no_grad():
            w, tau = self.quantized()
            w, tau = w.long(), tau.long()
            assert not torch.any(w[~self.mask])
            per_h = w.abs().sum(1)
            bound = b.long().abs().sum(1)[:, None]*per_h[None]
            assert int(bound.max()) < (1 << 23), 'S/U prefix bound must fit Acc24'
            record = {'weight_int8': w.cpu().to(torch.int8),
                      'threshold_int32': tau.cpu().to(torch.int32),
                      'hidden_keep': self.keep.cpu(), 'weight_mask': self.mask.cpu(),
                      'B_int8': b.cpu().to(torch.int8), 'E_int8': e.cpu().to(torch.int8),
                      'hidden_indices': torch.nonzero(self.keep).flatten().cpu(),
                      'accumulator_abs_bound_max': int(bound.max()),
                      'fixed_B_and_theta': True,
                      'remaining_H': int(self.keep.sum()),
                      'nonzero_W': int(w.ne(0).sum()), 'W_slots': w.numel(),
                      'mask_slots': int(self.mask.sum()),
                      'actual_W_squared_vs_original': float(w.double().square().sum()/original.double().square().sum())}
            return record


def self_check():
    w = np.ones((16, 32), dtype=np.int64)
    masks = masks_from_weight(w)
    for name, (mask, keep) in masks.items():
        assert mask.sum() == (w.size if name == 'original' else w.size//2)
        if name == 'row_2of4':
            assert (mask.reshape(16, 8, 4).sum(-1) == 2).all()
            assert mask[0, :4].tolist() == [True, True, False, False]
        if name == 'broadcast8_C16_half':
            assert mask[:, :16].all() and not mask[:, 16:].any()
    # A negative tau would emit a spike even with a zero W row. Deleting that
    # hidden channel must instead remove its FC2 contribution altogether.
    gate = torch.zeros(2).ge(torch.tensor([-1., 1.])).float()
    keep = torch.tensor([False, True])
    fc2 = torch.tensor([[3., 7.]])
    assert float(F.linear(gate, fc2)) == 3
    assert float(F.linear(gate[keep], fc2[:, keep])) == 0
    print('SELF_CHECK_PASS exact mask budgets/ties and hidden-deletion semantics', flush=True)


def install_pruned_consumer(system, record):
    _, modules, gpu, current, _, _, _, _ = system
    q = gpu[PREFIX]
    q['w'] = record['weight_int8'].cuda().float()
    q['tau'] = record['threshold_int32'].cuda().double()
    keep = record['hidden_indices'].cuda()
    q['hidden_indices'] = keep
    assert torch.equal(q['B'].cpu().to(torch.int8), record['B_int8'])

    def forward(self, x, compiled=q):
        if self.norm_layer in ('LN', 'GN'):
            x = self.norm(x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        source = self.drop1(self.sn1(x))
        shape = source.shape
        shifts = torch.arange(10, device=x.device).reshape(10, *([1]*(source.ndim-1)))
        raw = ((source != 0).long() << shifts).sum(0)
        codes = compiled['map'][raw].reshape(-1, compiled['w'].shape[1])
        selected = compiled['hidden_indices']
        partial = compiled['E'][codes].permute(0, 2, 1) @ compiled['w'][selected].T
        u = torch.einsum('tr,prh->tph', compiled['B'], partial.double())
        hidden = u.ge(compiled['tau'][:, None, selected]).float()*compiled['theta_output']
        hidden = hidden.reshape(*shape[:-1], len(selected))
        if len(selected) == compiled['w'].shape[0]:
            out = self.fc2(self.drop2(hidden))
        else:
            out = F.linear(self.drop2(hidden), self.fc2.weight[:, selected], self.fc2.bias)
        if self.norm_layer in ('BN', 'BNTT', 'tdBN', 'IN'):
            out = self.bn2(out.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        return out

    mlp = modules[PREFIX.rstrip('.')]
    mlp.forward = types.MethodType(forward, mlp)


def save_record(out, name, record):
    torch.save(record, out/(name+'.pt'))
    np.savez_compressed(out/(name+'.npz'), **{k: v.numpy() for k, v in record.items() if torch.is_tensor(v)})


def evaluate(args, system, variants, out, summaries):
    from run_bn_probe import input_frame
    from evaluate_stage2_deployment import CoarseReady, summarize
    from spikingjelly.activation_based import functional
    model, _, _, current, initial, _, _, _ = system
    probe.install_sources(system, initial)
    current['count_codes'] = True
    names = json.loads((args.root/'algorithm/samples.json').read_text())['valid'][:10]
    for axis, record in variants.items():
        install_pruned_consumer(system, record)
        rows = []
        with torch.no_grad():
            for i, filename in enumerate(names):
                functional.reset_net(model)
                current['code_counts'] = {}
                current['save_codes_dir'] = out/'codes'/axis/Path(filename).stem
                x, label, mask = input_frame(args.data, filename)
                try:
                    model(x)
                except CoarseReady:
                    pred = F.interpolate(current.pop('flow'), (480, 640), mode='bilinear', align_corners=False)
                error = torch.linalg.vector_norm(pred.permute(0, 2, 3, 1)[mask]-label.permute(0, 2, 3, 1)[mask], dim=1)
                total, pixels = float(error.double().sum()), error.numel()
                rows.append({'file': filename, 'valid_pixels': pixels, 'aee_sum': total,
                             'AEE': total/pixels, 'sources': current['code_counts']})
                probe.save_json(out/(axis+'_valid10_frames.json'), rows)
                print('VALID_PRUNE', axis, i+1, total/pixels, flush=True)
                del x, label, mask, pred, error
        result = summarize(rows, True)
        result.update(remaining_H=record['remaining_H'], nonzero_W=record['nonzero_W'],
                      fixed_B=True, consumer_tau_min=int(record['threshold_int32'].min()),
                      consumer_tau_max=int(record['threshold_int32'].max()),
                      source='same fixed integer bits3 source; downstream source codes may change through the pruned network')
        summaries[axis] = result
        probe.save_json(out/'summary.json', summaries)
        print('COMPLETE_PRUNE', axis, json.dumps(result), flush=True)


def train(args, system, out, initial_records, w0, tau0, b, e, sigma, codes):
    from train_mfpsn_probe import ATanSpike
    _, modules, gpu, _, _, _, _, _ = system
    q = gpu[PREFIX]
    fc2 = modules[PREFIX+'fc2']
    variants, history = {}, {}
    for axis in AXES:
        initial = initial_records[axis+'_initial']
        mask, keep = initial['weight_mask'].cuda(), initial['hidden_keep'].cuda()
        selected = torch.nonzero(keep).flatten()
        consumer = PrunedConsumer(w0, tau0, sigma, mask, keep).cuda()
        optimizer = torch.optim.Adam([{'params': [consumer.weight], 'lr': args.weight_lr},
                                      {'params': [consumer.offset], 'lr': args.threshold_lr}])
        generator = torch.Generator(device='cuda').manual_seed(9523)
        trace = []
        for step in range(args.steps):
            frame = step % codes.shape[0]
            blocks = torch.randperm(codes.shape[1], generator=generator, device='cuda')[:4]
            inputs = codes[frame, blocks].reshape(-1, 384)
            membership = e[inputs].permute(0, 2, 1)
            with torch.no_grad():
                target_partial = membership @ w0.T
                target_margin = torch.einsum('tr,prh->tph', b, target_partial)-tau0[:, None]
                target_hidden = target_margin.ge(0).float()*q['theta_output']
                target_fc2 = F.linear(target_hidden, fc2.weight, fc2.bias)
            w, tau = consumer.quantized()
            partial = membership @ w[selected].T
            margin = torch.einsum('tr,prh->tph', b, partial)-tau[:, None, selected]
            norm = margin/sigma[:, None, selected]
            membrane = F.mse_loss(norm, target_margin[:, :, selected]/sigma[:, None, selected])
            hidden = ATanSpike.apply(norm)*q['theta_output']
            predicted_fc2 = F.linear(hidden, fc2.weight[:, selected], fc2.bias)
            downstream = F.mse_loss(predicted_fc2, target_fc2)/target_fc2.var(unbiased=False).clamp_min(.01)
            loss = membrane+.1*downstream
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                consumer.weight.mul_(mask).clamp_(-128, 127)
            if step % 64 == 0 or step+1 == args.steps:
                row = {'step': step+1, 'loss': float(loss), 'retained_membrane_mse': float(membrane),
                       'complete_FC2_mse': float(downstream),
                       'retained_gate_disagreement': float((hidden.ne(0) != target_hidden[:, :, selected].ne(0)).float().mean()),
                       'student_retained_activity': float(hidden.ne(0).float().mean()),
                       'teacher_retained_activity': float(target_hidden[:, :, selected].ne(0).float().mean())}
                trace.append(row)
                print('FIT_PRUNE', axis, json.dumps(row), flush=True)
        record = consumer.export(b, e, w0)
        record['mask_rule'] = initial['mask_rule']
        name = axis+'_trained'
        variants[name], history[axis] = record, trace
        save_record(out, name, record)
        probe.save_json(out/'training.json', history)
        del consumer, optimizer
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--weight-lr', type=float, default=.03)
    parser.add_argument('--threshold-lr', type=float, default=.003)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.root is None:
        parser.error('--root is required')
    out = args.root/'algorithm/pruning_probe'
    out.mkdir(parents=True, exist_ok=True)
    probe.save_json(out/'run.json', {'module': PREFIX, 'mask_axes': list(AXES), 'steps_each': args.steps,
                   'mask_budget': '50% original FC1 coefficient slots; hidden control additionally deletes half neurons/FC2 columns',
                   'mask_selection': 'original W squared only; lower index stable ties; no validation fit',
                   'source_and_network_start': 'saved integer bits3, original power2-trained S2 consumers',
                   'train_cache': 'nrv_cost_probe/s2b3_train.pt start_codes: train32, P4 x32 blocks/frame, all C384',
                   'train_parameters': 'only surviving signed8 FC1 W and signed24 per(t,h) tau; B/dictionary/theta/FC2/BN2 fixed',
                   'loss': 'standardized retained-neuron membrane MSE + .1 complete FC2 output MSE; no minibatch BN approximation',
                   'optimizer': {'name': 'Adam', 'weight_lr': args.weight_lr, 'threshold_offset_lr': args.threshold_lr},
                   'validation': 'same ten frames, true BN2/shortcut; both initial and final weights; no 825',
                   'claim': 'simple static-mask and local fine-tuning screen, not full QP-SNN or hardware speedup'})
    system = probe.load_system(args)
    _, _, gpu, _, _, _, _, _ = system
    q = gpu[PREFIX]
    w0, tau0, b, e = q['w'].clone(), q['tau'].float().clone(), q['B'].float().clone(), q['E'].clone()
    cache = torch.load(args.root/'algorithm/nrv_cost_probe/s2b3_train.pt', map_location='cpu', weights_only=False)
    codes = cache['start_codes'].cuda().long()
    with torch.no_grad():
        sums = torch.zeros_like(tau0, dtype=torch.float64)
        squares = torch.zeros_like(sums)
        count = 0
        for frame in codes:
            members = e[frame.reshape(-1, 384)].permute(0, 2, 1)
            partial = members @ w0.T
            margin = torch.einsum('tr,prh->tph', b, partial)-tau0[:, None]
            sums += margin.double().sum(1); squares += margin.double().square().sum(1)
            count += margin.shape[1]
        sigma = (squares/count-(sums/count).square()).clamp_min(1).sqrt().float()
    masks = masks_from_weight(w0.cpu().numpy())
    variants, metadata = {}, {}
    rules = {'original': 'all slots', 'row_2of4': 'each h and C4 retain highest two W^2',
             'broadcast8_C16_half': 'each consecutive H8 retain 12 of24 common C16 blocks by sum W^2',
             'hidden_H_half': 'retain top768 of1536 h by row sum W^2; physically compact h and FC2 input columns'}
    for axis, (mask_array, keep_array) in masks.items():
        mask, keep = torch.from_numpy(mask_array).cuda(), torch.from_numpy(keep_array).cuda()
        consumer = PrunedConsumer(w0, tau0, sigma, mask, keep).cuda()
        record = consumer.export(b, e, w0)
        record['mask_rule'] = rules[axis]
        variants[axis+'_initial'] = record
        save_record(out, axis+'_initial', record)
        metadata[axis] = {k: v for k, v in record.items() if not torch.is_tensor(v)}
    probe.save_json(out/'initial_masks.json', metadata)
    summaries = {}
    evaluate(args, system, variants, out, summaries)
    trained = train(args, system, out, variants, w0, tau0, b, e, sigma, codes)
    evaluate(args, system, trained, out, summaries)
    torch.save({**variants, **trained}, out/'parameters.pt')
    probe.save_json(out/'final_parameters.json', {name: {k: v for k, v in rec.items() if not torch.is_tensor(v)}
                                               for name, rec in trained.items()})


if __name__ == '__main__':
    main()
