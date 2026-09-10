"""MFPSN method migration, as a strong *prior* for the S2 temporal-code study.

Paper: NeurIPS 2025, Multiplication-Free Parallelizable Spiking Neurons with
Efficient Spatio-Temporal Dynamics, section4 and appendixC. This implements
causal channelwise kernels, sawtooth dilations, two-pass BN-fused training,
whole-quantizer STE, and fixed-statistic inference. It is local teacher/FC2
distillation, not the authors' full-network training or GPU autoselect.

Six S2 FC1 consumers use k3,d=1,2,3,1,2,3. The old BN1/sn2 are replaced;
raw integer FC1 Y is scaled exactly by 2^-8 before the new neuron. Output
amplitude remains the checkpoint theta. No original-FP32 equivalence claim.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import types

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from run_bn_probe import ALL_FC1, build_model, input_frame, read_names, save_json, set_bn_mode
from evaluate_stage2_deployment import S2, CoarseReady, install_saved_consumers, summarize, tag


def causal_charge(x, weight, dilation):
    """x[T,P,C], weight[C,k]; no right/future input and no temporal wrap."""
    out = x * weight[:, -1]
    for j in range(weight.shape[1]-1):
        lag = (weight.shape[1]-1-j)*dilation
        if lag < x.shape[0]:
            part = x[:-lag]*weight[:, j]
            out = out + F.pad(part, (0, 0, 0, 0, lag, 0))
    return out


class Power2STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.where(x == 0, 0, x.sign()*torch.exp2(torch.round(torch.log2(x.abs().clamp_min(2.0**-60)))))

    @staticmethod
    def backward(ctx, grad):
        return grad


class ATanSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        # SpikingJelly ATan(alpha=2), as in the official DVS implementation.
        return grad/(1+(math.pi*x).square())


class MFPSN(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(channels, 3).uniform_(-1/math.sqrt(3), 1/math.sqrt(3)))
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(-torch.ones(channels))
        self.register_buffer('running_mean', torch.zeros(channels))
        self.register_buffer('running_var', torch.ones(channels))
        self.eps = 1.0e-5

    def forward(self, x):
        if self.training:
            raw = causal_charge(x, self.weight, self.dilation)
            var, mean = torch.var_mean(raw, dim=(0, 1), unbiased=False)
            # Keep gradients through both statistics, as in the authors' code.
            with torch.no_grad():
                count = raw.shape[0]*raw.shape[1]
                self.running_mean.lerp_(mean.detach(), .1)
                self.running_var.lerp_(var.detach()*count/(count-1), .1)
        else:
            mean, var = self.running_mean, self.running_var
        gain = self.gamma*torch.rsqrt(var+self.eps)
        fused_weight = Power2STE.apply(self.weight*gain[:, None])
        fused_bias = self.beta-gain*mean
        return causal_charge(x, fused_weight, self.dilation)+fused_bias

    @torch.no_grad()
    def compile(self, source_bounds):
        gain = self.gamma*torch.rsqrt(self.running_var+self.eps)
        w = Power2STE.apply(self.weight*gain[:, None]).double()
        bias = (self.beta-gain*self.running_mean).double()
        # x=Y/256. Use a channel-local binary denominator to preserve all bits.
        nonzero = w != 0
        exponent = torch.where(nonzero, torch.round(torch.log2(w.abs().clamp_min(2.0**-60))).long()-8, 0)
        frac = (-exponent.amin(1)).clamp_min(0)
        integer_w = w.sign().long() * torch.pow(2, exponent+frac[:, None]).long()
        tau = torch.ceil(-bias*torch.exp2(frac.double()))
        bound = integer_w.abs().sum(1)*source_bounds
        assert int(bound.max()) < 2**52, 'FP64 integer emulation range exceeded'
        tau = torch.maximum(torch.minimum(tau, bound.double()+1), -bound.double()-1).long()
        return {'weight_int64': integer_w.cpu(), 'threshold_int64': tau.cpu(),
                'weight_power2': w.cpu(), 'fused_bias': bias.cpu(),
                'fractional_bits_per_channel': frac.cpu(),
                'signed_exponents_on_raw_Y': exponent.cpu(),
                'accumulator_abs_bound_per_channel': bound.cpu(),
                'dilation': self.dilation, 'kernel_size': 3}


def target_margin(y, q, scale):
    u = torch.einsum('ij,jpc->ipc', q['a'], y.double())
    sign = q['positive'].float()*2-1
    return ((u-q['tau'][:, None])*sign).float()/scale


@torch.no_grad()
def warm_start(student, cache, q):
    """Training-only channelwise ridge fit; differs from official random init."""
    channels = cache.shape[-1]
    positions = torch.linspace(0, cache.shape[2]-1, 128, device='cuda').round().long()
    first, second, count = torch.zeros(channels, device='cuda'), torch.zeros(channels, device='cuda'), 0
    for frame in cache:
        y = frame[:, positions].float()
        m = target_margin(y, q, 1.0)
        first += m.sum((0, 1)); second += m.square().sum((0, 1)); count += m.shape[0]*m.shape[1]
    scale = (second/count-(first/count).square()).clamp_min(1.0).sqrt()
    gram = torch.zeros(channels, 4, 4, dtype=torch.float64, device='cuda')
    rhs = torch.zeros(channels, 4, dtype=torch.float64, device='cuda')
    for frame in cache:
        y = frame[:, positions].float()
        x = y/256.0
        columns = []
        for j in range(3):
            lag = (2-j)*student.dilation
            columns.append(x if not lag else F.pad(x[:-lag], (0, 0, 0, 0, lag, 0)))
        columns.append(torch.ones_like(x))
        design = torch.stack(columns, -1).permute(2, 0, 1, 3).reshape(channels, -1, 4).double()
        target = target_margin(y, q, scale).permute(2, 0, 1).reshape(channels, -1).double()
        gram += design.transpose(1, 2) @ design
        rhs += (design*target[:, :, None]).sum(1)
    reg = torch.eye(4, device='cuda', dtype=torch.float64)[None]*gram.diagonal(dim1=1, dim2=2).mean(1)[:, None, None]*1e-5
    solution = torch.linalg.solve(gram+reg, rhs).float()
    student.weight.copy_(solution[:, :3])
    raw = causal_charge(cache[0, :, positions].float()/256, student.weight, student.dilation)
    var, mean = torch.var_mean(raw, dim=(0, 1), unbiased=True)
    student.running_mean.copy_(mean); student.running_var.copy_(var)
    student.gamma.copy_((var+student.eps).sqrt())
    student.beta.copy_(mean+solution[:, 3])
    return scale


def install_compiled(model, compiled, gpu):
    modules = dict(model.named_modules())
    for prefix in S2:
        record = compiled[prefix]
        weight = record['weight_int64'].cuda().double()
        tau = record['threshold_int64'].cuda().double()
        dilation = record['dilation']
        theta = gpu[prefix]['theta_output']

        def neuron(self, y, w=weight, threshold=tau, d=dilation, amplitude=theta):
            shape = y.shape
            u = causal_charge(y.reshape(shape[0], -1, shape[-1]).double(), w, d)
            return (u >= threshold).reshape(shape).float()*amplitude
        modules[prefix+'sn2.spiking_neuron'].forward = types.MethodType(neuron, modules[prefix+'sn2.spiking_neuron'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--full-valid', action='store_true')
    args = parser.parse_args()
    alg = args.root/'algorithm'
    out = alg/'mfpsn_probe'
    out.mkdir(parents=True, exist_ok=True)
    old = json.loads((alg/'stage2_temporal_codes/run.json').read_text())
    args.code_root = args.root/'code/SDformer'
    args.config, args.checkpoint = Path(old['config']), Path(old['checkpoint'])
    args.data = args.root.parent/'sdformer_codex/SDformer/data/Datasets/DSEC/saved_flow_data'
    train = old['train_codebook_frames']
    names = read_names(args.data, 'valid') if args.full_valid else json.loads((alg/'samples.json').read_text())['valid'][:10]
    model, cfg, installed, attention = build_model(args)
    model.requires_grad_(False)
    from spikingjelly.activation_based import functional
    modules = dict(model.named_modules())
    stats = torch.load(alg/'valid825_cal32/train_calibration.pt', map_location='cpu', weights_only=False)
    params = torch.load(alg/'stage2_temporal_codes/integer_parameters.pt', map_location='cpu', weights_only=False)
    set_bn_mode(model, stats, ALL_FC1)
    gpu = install_saved_consumers(model, params)
    current = {}

    def stop(module, inputs, output):
        current['flow'] = output.detach().sum(0)
        raise CoarseReady()
    modules['sttmultires_unet.preds.2'].register_forward_hook(stop)
    run = {'kind': 'MFPSN_method_migration_local_distillation_and_real_downstream_validation',
           'train': train, 'train_frames': len(train), 'steps_per_module': args.steps,
           'kernel_size': 3, 'dilation': [1, 2, 3, 1, 2, 3], 'modules': S2,
           'input': 'raw integer FC1 Y / 256, dyadic rescale with no discarded bits',
           'output': 'original theta_output times hard gate; threshold is a separate learned quantity',
           'method': 'paper Eq10/11/13/16 and appendixC two-pass BN-fused quantizer; ATan alpha2',
           'adaptations': 'replace only S2 BN1/sn2; train-only ridge warm start; local margin and FC2 distillation; no whole-network task fine-tuning or authors GPU autoselect',
           'source': 'https://github.com/PengXue0812/Multiplication-Free-Parallelizable-Spiking-Neurons-with-Efficient-Spatio-Temporal-Dynamics',
           'scope': 'new model and arithmetic implementation, not original ep34 equivalence, RTL or PPA'}
    if not args.evaluate:
        save_json(out/'run.json', run)
        cached = {p: [] for p in S2}
        handles = []
        for prefix in S2:
            def capture(module, inputs, name=prefix):
                y = inputs[0].detach().reshape(10, -1, inputs[0].shape[-1])
                assert y.abs().max() <= 32767 and torch.equal(y, y.round())
                cached[name].append(y.cpu().to(torch.int16))
            handles.append(modules[prefix+'sn2.spiking_neuron'].register_forward_pre_hook(capture))
        with torch.no_grad():
            for i, name in enumerate(train):
                functional.reset_net(model)
                x, _, _ = input_frame(args.data, name, targets=False)
                try:
                    model(x)
                except CoarseReady:
                    current.pop('flow')
                del x
                print('TRAIN_CAPTURE', i+1, name, flush=True)
        for handle in handles:
            handle.remove()
        compiled, training, state = {}, {}, {}
        for index, prefix in enumerate(S2):
            cache = torch.stack(cached.pop(prefix)).cuda()
            q = gpu[prefix]
            student = MFPSN(cache.shape[-1], index%3+1).cuda()
            scale = warm_start(student, cache, q)
            optimizer = torch.optim.Adam(student.parameters(), lr=.003)
            rows = []
            w2 = modules[prefix+'fc2'].weight.detach()
            with torch.no_grad():
                theta = q['theta_output']
            student.train()
            generator = torch.Generator(device='cuda').manual_seed(730+index)
            for step in range(args.steps):
                f = step % cache.shape[0]
                pos = torch.randperm(cache.shape[2], device='cuda', generator=generator)[:128]
                y = cache[f, :, pos].float()
                with torch.no_grad():
                    target = target_margin(y, q, scale)
                    gate = (target >= 0).float()
                    consumer = F.linear(gate*theta, w2)
                    denom = consumer.var(unbiased=False).clamp_min(.01)
                margin = student(y/256.0)
                spike = ATanSpike.apply(margin)
                membrane_loss = F.mse_loss(margin, target)
                consumer_loss = F.mse_loss(F.linear(spike*theta, w2), consumer)/denom
                loss = membrane_loss + .1*consumer_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if step % 32 == 0 or step+1 == args.steps:
                    row = {'step': step+1, 'loss': float(loss), 'membrane': float(membrane_loss),
                           'consumer': float(consumer_loss), 'gate_disagreement': float((spike != gate).float().mean()),
                           'teacher_activity': float(gate.mean()), 'student_activity': float(spike.mean())}
                    rows.append(row)
                    print('FIT', tag(prefix), json.dumps(row), flush=True)
            student.eval()
            compiled[prefix] = student.compile(params[prefix]['weight_int8'].long().abs().sum(1).cuda())
            state[prefix] = {k: v.cpu() for k, v in student.state_dict().items()}
            training[prefix] = rows
            record = compiled[prefix]
            print('COMPILED', tag(prefix), 'acc_bound', int(record['accumulator_abs_bound_per_channel'].max()),
                  'exponents', int(record['signed_exponents_on_raw_Y'].min()), int(record['signed_exponents_on_raw_Y'].max()), flush=True)
            del cache, student, optimizer
        torch.save({'compiled': compiled, 'state_dicts': state}, out/'student.pt')
        save_json(out/'training.json', training)
    else:
        compiled = torch.load(out/'student.pt', map_location='cpu', weights_only=False)['compiled']
    install_compiled(model, compiled, gpu)
    rows = []
    with torch.no_grad():
        for i, name in enumerate(names):
            functional.reset_net(model)
            x, label, mask = input_frame(args.data, name)
            try:
                model(x)
            except CoarseReady:
                pred = F.interpolate(current.pop('flow'), size=(480, 640), mode='bilinear', align_corners=False)
            error = torch.linalg.vector_norm(pred.permute(0, 2, 3, 1)[mask]-label.permute(0, 2, 3, 1)[mask], dim=1)
            total, pixels = float(error.double().sum()), error.numel()
            row = {'file': name, 'valid_pixels': pixels, 'aee_sum': total, 'AEE': total/pixels}
            rows.append(row)
            print('FRAME', i+1, json.dumps(row), flush=True)
            if (i+1)%25 == 0 or i+1 == len(names):
                stem = 'valid825' if args.full_valid else 'valid10'
                save_json(out/(stem+'_frames.json'), rows)
                save_json(out/(stem+'_summary.json'), summarize(rows, len(rows) == len(names)))
            del x, label, mask, pred, error
    print('COMPLETE', json.dumps(summarize(rows, True)), flush=True)


if __name__ == '__main__':
    main()
