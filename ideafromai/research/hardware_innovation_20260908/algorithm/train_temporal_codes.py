"""Bounded temporal-code producer pilot, with the full PSN consumer retained.

New lossy model: each source theta*g trajectory is projected onto a train-only
eight-word T10 dictionary (including silence).  Train the source PSN and FC1
weight using the same local teacher objective as the earlier support pilot.
This file also checks the integer identity A*(g*W) == (A*D)*(onehot(code)*W)
for this new model.  It reports no hardware cycles or PPA.
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


class TemporalStudent(torch.nn.Module):
    def __init__(self, source, fc, params, dictionary):
        super().__init__()
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
        h = (torch.addmm(self.bias, self.A, x.reshape(10, -1))-self.center)
        raw = (h >= self.theta).float()
        soft = torch.sigmoid((h-self.theta)/0.25)
        words = raw.T
        # Squared distance is Hamming for binary words; first index breaks ties.
        distance = words.sum(1, keepdim=True)+self.dictionary.sum(1)[None, :]-2*(words @ self.dictionary.T)
        code = distance.argmin(1)
        projected = self.dictionary[code].T
        emitted = projected+soft-soft.detach() if differentiable else projected
        return emitted.reshape(shape), code.reshape(shape[1:]), raw.reshape(shape)

    def weight(self, differentiable=False):
        continuous = (self.W*self.theta/self.scale[:, None]).clamp(-127, 127)
        return continuous+(continuous.round()-continuous).detach() if differentiable else continuous.round()


def choose_dictionary(cache):
    gates = torch.cat([row['gate'] for row in cache], dim=1).numpy().astype(np.uint16)
    words = np.sum(gates << np.arange(10, dtype=np.uint16)[:, None, None], axis=0)
    counts = np.bincount(words.reshape(-1).astype(np.int64), minlength=1024)
    order = sorted(range(1, 1024), key=lambda word: (-int(counts[word]), word))
    selected = np.array([0]+order[:7], np.uint16)
    dictionary = ((selected[:, None] >> np.arange(10)) & 1).astype(np.float32)
    return torch.from_numpy(dictionary), {'words': selected.tolist(),
        'train_counts': [int(counts[w]) for w in selected],
        'training_trajectories': int(counts.sum()),
        'training_exact_coverage': float(counts[selected].sum()/counts.sum()),
        'rule': 'zero plus seven most frequent nonzero train-only words; no validation selection'}


def main():
    parser = argparse.ArgumentParser()
    for name in ('code-root', 'config', 'checkpoint', 'data', 'calibration', 'samples', 'teacher-cache', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--updates', type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model, _, _, _ = build_model(args)
    from spikingjelly.activation_based import functional
    modules = dict(model.named_modules())
    prefix = MLPS[0]
    source, fc = modules[prefix+'sn1.spiking_neuron'], modules[prefix+'fc1']
    stats = torch.load(args.calibration, map_location='cpu', weights_only=False)
    set_bn_mode(model, stats, SHALLOW)
    params, _ = install_integer_consumers(model, stats)
    record = params[prefix]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    cache = torch.load(args.teacher_cache, map_location='cpu', weights_only=False)
    dictionary, description = choose_dictionary(cache)
    np.save(args.output/'dictionary.npy', dictionary.numpy().astype(bool))
    gpu_cache = [{k: v.cuda() if torch.is_tensor(v) else v for k, v in row.items()} for row in cache]
    scale = record['weight_row_scale'].cuda().float()/(stats[prefix+'bn1.norm_layer']['var'].cuda()+1e-5).sqrt()
    students = {}
    history = []
    for variant in ('untrained_code8', 'trained_code8'):
        student = TemporalStudent(source, fc, record, dictionary).cuda()
        if variant == 'trained_code8':
            optimizer = torch.optim.Adam([{'params': [student.A, student.bias], 'lr': 1e-3},
                                          {'params': [student.W], 'lr': 2e-4}])
            for step in range(args.updates):
                row = gpu_cache[step % len(gpu_cache)]
                optimizer.zero_grad(set_to_none=True)
                emitted, _, _ = student.source(row['x'], True)
                yi = F.linear(emitted, student.weight(True))
                loss = ((yi-row['yi'])*scale).square().mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                history.append({'step': step+1, 'file': row['file'], 'loss': float(loss.detach())})
                if (step+1) % 8 == 0:
                    print('TRAIN', json.dumps(history[-1]), flush=True)
        students[variant] = student.eval()
        torch.save(student.state_dict(), args.output/(variant+'_parameters.pt'))
        torch.save(student.weight(False).cpu().to(torch.int8), args.output/(variant+'_weight_int8.pt'))
    save_json(args.output/'training.json', history)
    aq = record['temporal_int16'].cuda().double()
    transformed = aq @ dictionary.cuda().double().T
    np.save(args.output/'transformed_temporal_int32.npy', transformed.cpu().numpy().astype(np.int32))
    check = []
    with torch.no_grad():
        for variant, student in students.items():
            g, code, _ = student.source(gpu_cache[0]['x'][:, :6])
            w = student.weight(False).double()
            direct = aq @ F.linear(g.double(), w).reshape(10, -1)
            membership = F.one_hot(code, num_classes=8).double().permute(0, 2, 1)
            partials = membership @ w.T
            fused = torch.einsum('tk,pkh->tph', transformed, partials).reshape(10, -1)
            check.append({'variant': variant, 'integer_U_values': direct.numel(),
                          'mismatches': int(direct.ne(fused).sum()),
                          'max_abs_B': int(transformed.abs().max()),
                          'B_nonzero_columns': int(transformed.ne(0).any(0).sum()),
                          'B_nonzero_terms': int(transformed.ne(0).sum())})
    save_json(args.output/'integer_identity.json', check)
    names = json.loads(args.samples.read_text())['valid'][:10]
    rows = []
    for variant, student in students.items():
        qw = student.weight(False).detach()
        current = {}
        numeric = {'values': 0, 'mismatches': None}
        tau = record['threshold_int64'].cuda().double()
        positive = record['positive_gain'].cuda()
        variable = ~record['constant_channels'].cuda()
        fixed = record['constant_gate'].cuda()
        theta_out = record['theta_output'].cuda().float()

        def source_forward(self, x):
            emitted, code, raw = student.source(x, False)
            current['gpu_code'] = code.reshape(-1, 96)
            current['code'] = code.detach().reshape(-1, 96).cpu().numpy().astype(np.uint8)
            current['firing'] = float(emitted.mean())
            current['raw_firing'] = float(raw.mean())
            return emitted*student.theta

        def linear(self, x):
            return F.linear(x/student.theta, qw)

        def fused_mlp(self, x):
            source_value = self.sn1(x)
            shape = source_value.shape
            code = current.pop('gpu_code')
            membership = F.one_hot(code, num_classes=8)[:, :, 1:].float().permute(0, 2, 1)
            partials = membership @ qw.T
            u = torch.einsum('tk,pkh->tph', transformed[:, 1:], partials.double())
            if numeric['mismatches'] is None:
                yi = F.linear(source_value/student.theta, qw)
                direct = (aq @ yi.double().reshape(10, -1)).reshape_as(u)
                numeric['values'] = u.numel()
                numeric['mismatches'] = int(u.ne(direct).sum())
                if numeric['mismatches']:
                    raise RuntimeError('Full-frame fused temporal U differs from direct integer U')
            gate = torch.where(positive[None, None, :], u >= tau[:, None, :], u <= tau[:, None, :])
            gate = torch.where(variable[None, None, :], gate, fixed[:, None, :])
            hidden = gate.reshape(*shape[:-1], 384).float()*theta_out
            out = self.fc2(hidden)
            return self.bn2(out.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)

        def coarse_hook(module, inputs, output):
            current['flow'] = output.detach().sum(0)

        source.forward = types.MethodType(source_forward, source)
        fc.forward = types.MethodType(linear, fc)
        mlp = modules[prefix.rstrip('.')]
        mlp.forward = types.MethodType(fused_mlp, mlp)
        handle = modules['sttmultires_unet.preds.2'].register_forward_hook(coarse_hook)
        with torch.no_grad():
            for name in names:
                functional.reset_net(model)
                x, label, mask = input_frame(args.data, name)
                final = model(x)['flow'][-1]
                coarse = F.interpolate(current.pop('flow'), size=(480, 640), mode='bilinear', align_corners=False)
                row = {'variant': variant, 'file': name, 'valid_pixels': int(mask.sum()),
                       'firing': current['firing'], 'raw_firing': current['raw_firing']}
                for key, flow in (('final', final), ('coarse_bilinear', coarse)):
                    err = (flow-label).square().sum(1).sqrt()
                    row[key] = float(err[mask].sum())/row['valid_pixels']
                rows.append(row)
                np.savez_compressed(args.output/(variant+'_'+Path(name).stem+'_codes.npz'),
                                    codes=current.pop('code'), shape=np.array([10, 19200, 96]))
                print('FRAME', json.dumps(row), flush=True)
                del x, label, mask, final, coarse, err
        handle.remove()
        check.append({'variant': variant, 'full_first_frame_U_values': numeric['values'],
                      'full_first_frame_U_mismatches': numeric['mismatches']})
    save_json(args.output/'frames.json', rows)
    summary = {'frames_per_variant': len(names), 'dictionary': description,
               'training': {'frames': len(cache), 'spatial_positions_per_frame': 512, 'updates': args.updates,
                            'objective': 'local teacher FC1 Yi in fixed BN units; not end-to-end flow training'},
               'AEE': {variant: {key: float(np.mean([r[key] for r in rows if r['variant']==variant]))
                                 for key in ('final', 'coarse_bilinear', 'firing', 'raw_firing')}
                       for variant in students},
               'theta_source': float(source.thresh), 'integer_identity': check,
               'scope': 'new source temporal-code model; seven partial-sum classes are consumed by A*D directly, original tau and real FC2/BN2/remaining network; no hardware speed claim',
               'time_rank_limit': 'shared zero+7-word input dictionary spans at most seven time dimensions; keeping original A does not preserve all original temporal freedom'}
    save_json(args.output/'summary.json', summary)
    print('COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
