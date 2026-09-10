"""One-block, bounded support-shaping probe using cached teacher features.

This is local teacher distillation, not end-to-end flow training. The exact
dictionary student always emits its own hard theta*g, including misses.
The forced-code student is a separate lossy control. Neither implies RTL speed.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
import types

from run_bn_probe import (build_model, input_frame, read_names, representative_train,
                          save_json, set_bn_mode, MLPS, SHALLOW)
from run_integer_fc1_probe import install_integer_consumers
import numpy as np
import torch
import torch.nn.functional as F


class CaptureComplete(Exception):
    pass


def hard_dictionary(bits, count=16):
    """Training data only. Zero plus frequent patterns with actual reuse."""
    grouped = bits.reshape(-1, 6, 16).to(torch.int64).cpu()
    powers = 2**torch.arange(16, dtype=torch.int64)
    packed = (grouped*powers).sum(-1)
    codes = []
    for j in range(6):
        freq = torch.bincount(packed[:, j], minlength=65536)
        words = torch.arange(65536, dtype=torch.int64)
        pc = ((words[:, None] >> torch.arange(16)) & 1).sum(-1)
        benefit = freq*torch.clamp(pc-1, min=0)
        benefit[0] = -1
        best = torch.argsort(benefit, descending=True, stable=True)[:count-1]
        chosen = torch.cat((torch.zeros(1, dtype=torch.int64), best))
        codes.append(((chosen[:, None] >> torch.arange(16)) & 1).float())
    return torch.stack(codes)


def nearest_code(gate, dictionary):
    shaped = gate.reshape(-1, 6, 16)
    # Binary Euclidean distance equals Hamming distance; no float centroids.
    distance = (shaped.sum(-1, keepdim=True)
                + dictionary.sum(-1)[None]
                - 2*torch.einsum('ngc,gkc->ngk', shaped, dictionary))
    index = distance.argmin(-1)
    selected = dictionary[torch.arange(6, device=gate.device)[None], index]
    return selected.reshape_as(gate), distance.amin(-1), index


class Student(torch.nn.Module):
    def __init__(self, source, fc, params, dictionary, kind):
        super().__init__()
        self.kind = kind
        self.A = torch.nn.Parameter(source.weight.detach().clone())
        self.bias = torch.nn.Parameter(source.bias.detach().clone())
        self.W = torch.nn.Parameter(fc.weight.detach().clone())
        self.register_buffer('center', source.center.detach().clone() if source.center_mode != 'zero'
                             else torch.zeros_like(source.bias))
        self.register_buffer('theta', source.thresh.detach().clone())
        self.register_buffer('scale', params['weight_row_scale'].to(self.W.device, torch.float32))
        self.register_buffer('dictionary', dictionary.to(self.W.device))

    def source(self, x, differentiable=False):
        shape = x.shape
        h = torch.addmm(self.bias, self.A, x.reshape(10, -1))-self.center
        gate = (h >= self.theta).float().reshape(shape)
        soft = torch.sigmoid((h-self.theta)/0.25).reshape(shape)
        if differentiable:
            emitted = gate+soft-soft.detach()
        else:
            emitted = gate
        if self.kind == 'forced_code':
            selected, _, _ = nearest_code(gate, self.dictionary)
            emitted = selected+(emitted-emitted.detach()) if differentiable else selected
        return emitted, gate, soft

    def quantized_weight(self, differentiable=False):
        w = (self.W*self.theta/self.scale[:, None]).clamp(-127, 127)
        return w+(w.round()-w).detach() if differentiable else w.round()

    def forward(self, x):
        emitted, gate, soft = self.source(x, differentiable=True)
        return F.linear(emitted, self.quantized_weight(True)), emitted, gate, soft


def gate_metrics(gate, dictionary):
    g = gate.reshape(-1, 6, 16)
    _, distance, _ = nearest_code(gate, dictionary)
    n = g.sum(-1)
    exact = distance == 0
    useful = exact & (n >= 2)
    return {'groups': int(n.numel()), 'active_bits': int(n.sum()),
            'exact_groups': int(exact.sum()), 'useful_exact_groups': int(useful.sum()),
            'saved_vector_adds_unpriced': int(torch.where(useful, n-1, 0).sum()),
            'firing_rate': float(g.mean()), 'exact_rate': float(exact.float().mean()),
            'useful_exact_rate': float(useful.float().mean())}


def main():
    parser = argparse.ArgumentParser()
    for name in ('code-root', 'config', 'checkpoint', 'data', 'samples', 'calibration', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--train-frames', type=int, default=32)
    parser.add_argument('--positions', type=int, default=512)
    parser.add_argument('--updates', type=int, default=32)
    parser.add_argument('--max-valid', type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model, _, _, _ = build_model(args)
    from spikingjelly.activation_based import functional
    modules = dict(model.named_modules())
    prefix = MLPS[0]
    source = modules[prefix+'sn1.spiking_neuron']
    fc = modules[prefix+'fc1']
    stats = torch.load(args.calibration, map_location='cpu', weights_only=False)
    set_bn_mode(model, stats, SHALLOW)
    params, _ = install_integer_consumers(model, stats)
    integer = params[prefix]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    teacher_source = source.forward
    teacher_fc = fc.forward

    train = representative_train(read_names(args.data, 'train'), args.train_frames)
    valid = json.loads(args.samples.read_text())['valid'][:args.max_valid]
    cache_file = args.output/'teacher_cache.pt'
    if cache_file.exists():
        cache = torch.load(cache_file, map_location='cpu', weights_only=False)
    else:
        cache = []
        current = {}

        def source_capture(module, inputs):
            xx = inputs[0].detach().reshape(10, -1, 96)
            positions = torch.linspace(0, xx.shape[1]-1, args.positions,
                                       device=xx.device).round().long()
            current['positions'] = positions
            current['x'] = xx[:, positions].cpu()

        def fc_capture(module, inputs, output):
            positions = current.pop('positions')
            current['gate'] = (inputs[0].detach().reshape(10, -1, 96)[:, positions]/source.thresh).cpu()
            current['yi'] = output.detach().reshape(10, -1, 384)[:, positions].cpu()
            raise CaptureComplete()

        handles = [source.register_forward_pre_hook(source_capture),
                   fc.register_forward_hook(fc_capture)]
        with torch.no_grad():
            for name in train:
                functional.reset_net(model)
                x, _, _ = input_frame(args.data, name, targets=False)
                try:
                    model(x)
                except CaptureComplete:
                    cache.append({'file': name, **current})
                    current = {}
                del x
                print('TRAIN_CAPTURE', name, len(cache), flush=True)
        for handle in handles:
            handle.remove()
        torch.save(cache, cache_file)
    dictionary = hard_dictionary(torch.cat([r['gate'] for r in cache], dim=1))
    np.save(args.output/'dictionary.npy', dictionary.numpy().astype(bool))
    gpu_cache = [{k: v.cuda() if torch.is_tensor(v) else v for k, v in r.items()} for r in cache]
    row_scale = integer['weight_row_scale'].cuda().float()
    std = (stats[prefix+'bn1.norm_layer']['var'].cuda()+1e-5).sqrt()
    loss_scale = row_scale/std
    students = {}
    history = []
    # Equal update budget. Sparsity is a necessary control, not yet a matched-rate
    # control; report its actual rate rather than assert rates are matched.
    for kind in ('sparse', 'exact_dictionary', 'forced_code'):
        student = Student(source, fc, integer, dictionary, kind).cuda()
        optimizer = torch.optim.Adam([{'params': [student.A, student.bias], 'lr': 1e-3},
                                      {'params': [student.W], 'lr': 2e-4}])
        for step in range(args.updates):
            row = gpu_cache[step % len(gpu_cache)]
            optimizer.zero_grad(set_to_none=True)
            yi, emitted, gate, soft = student(row['x'])
            distill = ((yi-row['yi'])*loss_scale).square().mean()
            if kind == 'sparse':
                regularizer = soft.mean()
            elif kind == 'exact_dictionary':
                target, _, _ = nearest_code(gate.detach(), student.dictionary)
                regularizer = (soft-target).square().mean()
            else:
                regularizer = distill.new_zeros(())
            loss = distill+0.5*regularizer
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            record = {'variant': kind, 'step': step+1, 'train_file': row['file'],
                      'distill_mse_bn_units': float(distill.detach()),
                      'regularizer': float(regularizer.detach())}
            history.append(record)
            if (step+1) % 8 == 0:
                print('TRAIN', json.dumps(record), flush=True)
        students[kind] = student.eval()
        torch.save(student.state_dict(), args.output/(kind+'_parameters.pt'))
    save_json(args.output/'training.json', history)

    run = {'kind': 'exploratory_local_teacher_distillation', 'train_files': train,
           'validation_files': valid, 'sampled_positions_per_frame': args.positions,
           'updates_per_student': args.updates, 'trainable': ['source A', 'source bias', 'FC1 QAT W'],
           'fixed': ['theta', 'BN', 'dyadic row_scale', 'downstream Aq and integer thresholds'],
           'dictionary': 'six groups, 16 binary supports each; zero plus train-frequency times max(popcount-1,0)',
           'loss': 'FC1 Yi distillation normalized by fixed BN std + 0.5 auxiliary',
           'limitations': 'local feature supervision, no end-to-end flow loss, unmatched firing-rate control; 10 frames are not valid825 admission'}
    save_json(args.output/'run.json', run)
    results = []
    for kind in ('teacher', 'sparse', 'exact_dictionary', 'forced_code'):
        current = {}
        if kind == 'teacher':
            source.forward = teacher_source
            fc.forward = teacher_fc
        else:
            student = students[kind]

            def source_forward(self, x, selected=student):
                gate, _, _ = selected.source(x, differentiable=False)
                return gate*selected.theta

            qw = student.quantized_weight(False).detach()

            def linear_forward(self, x, weight=qw, theta=student.theta):
                return F.linear(x/theta, weight)

            source.forward = types.MethodType(source_forward, source)
            fc.forward = types.MethodType(linear_forward, fc)
            torch.save(qw.cpu().to(torch.int8), args.output/(kind+'_weight_int8.pt'))

        def valid_capture(module, inputs):
            current['gate'] = (inputs[0].detach()/source.thresh).reshape(10, -1, 96)

        handle = fc.register_forward_pre_hook(valid_capture)
        with torch.no_grad():
            for name in valid:
                functional.reset_net(model)
                x, label, mask = input_frame(args.data, name)
                start = time.monotonic()
                prediction = model(x)['flow'][-1]
                error = (prediction-label).square().sum(1).sqrt()
                gate = current.pop('gate')
                record = {'variant': kind, 'file': name, 'valid_pixels': int(mask.sum()),
                          'AEE': float(error[mask].mean()), 'wall_s': time.monotonic()-start,
                          **gate_metrics(gate, dictionary.cuda())}
                results.append(record)
                packed = np.packbits(gate.cpu().numpy().astype(bool), axis=-1, bitorder='little')
                np.savez_compressed(args.output/(kind+'_'+name[:-4]+'_source.npz'),
                                    gate_bits=packed, shape=np.array(gate.shape))
                print('VALID', json.dumps(record), flush=True)
                del x, label, mask, prediction, error, gate
        handle.remove()
        save_json(args.output/'frames.json', results)
    summaries = {}
    for kind in ('teacher', 'sparse', 'exact_dictionary', 'forced_code'):
        rows = [r for r in results if r['variant'] == kind]
        groups = sum(r['groups'] for r in rows)
        summaries[kind] = {'frames': len(rows), 'AEE': float(np.mean([r['AEE'] for r in rows])),
                           'firing_rate': sum(r['active_bits'] for r in rows)/(groups*16),
                           'exact_rate': sum(r['exact_groups'] for r in rows)/groups,
                           'useful_exact_rate': sum(r['useful_exact_groups'] for r in rows)/groups,
                           'active_bits': sum(r['active_bits'] for r in rows),
                           'saved_vector_adds_unpriced': sum(r['saved_vector_adds_unpriced'] for r in rows)}
    save_json(args.output/'summary.json', summaries)
    print('COMPLETE', json.dumps(summaries), flush=True)


if __name__ == '__main__':
    main()
