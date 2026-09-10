"""Deploy two calibrated shallow FC1 consumers with explicit integer arithmetic.

This is a new quantized model, not frozen-ep34 FP32 equivalence or RTL timing.
The GPU uses exact-range FP32/FP64 products to emulate the integer dot products.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import types

from run_bn_probe import (build_model, input_frame, read_names, save_json,
                          set_bn_mode, MLPS, SHALLOW)
import numpy as np
import torch


def install_integer_consumers(model, stats, fractional_bits=14, capture_dir=None, prefixes=None):
    modules = dict(model.named_modules())
    exported = {}
    descriptions = []
    scale_a = float(2**fractional_bits)
    for prefix in (MLPS[:2] if prefixes is None else prefixes):
        fc = modules[prefix+'fc1']
        bn = modules[prefix+'bn1.norm_layer']
        sn = modules[prefix+'sn2.spiking_neuron']
        source = modules[prefix+'sn1.spiking_neuron']
        theta_source = source.thresh.detach().double()
        theta_output = sn.thresh.detach().double()
        effective_w = fc.weight.detach().double()*theta_source
        max_w = effective_w.abs().amax(1)
        row_exponent = torch.ceil(torch.log2(max_w.clamp_min(2**-40)/127.0)).to(torch.int32)
        row_scale = torch.ldexp(torch.ones_like(max_w), row_exponent)
        wq = torch.round(effective_w/row_scale[:, None]).clamp(-127, 127)
        a = sn.weight.detach().double()
        aq = torch.round(a*scale_a).clamp(-32768, 32767)
        a_quant = aq/scale_a
        bias = sn.bias.detach().double().reshape(-1)
        center = (sn.center.detach().double().reshape(-1) if sn.center_mode != 'zero'
                  else torch.zeros_like(bias))
        s = stats[prefix+'bn1.norm_layer']
        mean = s['mean'].to(a.device, torch.float64)
        var = s['var'].to(a.device, torch.float64)
        gamma = bn.weight.detach().double()
        beta = bn.bias.detach().double()
        fc_bias = (fc.bias.detach().double() if fc.bias is not None
                   else torch.zeros_like(mean))
        gain = gamma/(var+bn.eps).sqrt()
        offset = beta+gain*(fc_bias-mean)
        gain = gain*row_scale
        row_sum = a_quant.sum(1)
        nonzero_gain = gain != 0
        raw_tau = ((theta_output+center-bias)[:, None]-row_sum[:, None]*offset[None, :])
        raw_tau = raw_tau*scale_a/torch.where(nonzero_gain, gain, torch.ones_like(gain))[None, :]
        # The source is theta*g. This static absolute bound covers signed g as
        # well; thresholds outside it have a constant decision and need no
        # arbitrarily wide threshold register.
        u_bound = aq.abs().sum(1)[:, None]*wq.abs().sum(1)[None, :]
        raw_tau = torch.maximum(torch.minimum(raw_tau, u_bound+1), -u_bound-1)
        positive_gain = gain >= 0
        threshold = torch.where(positive_gain[None, :], raw_tau.ceil(), raw_tau.floor())
        constant_gate = ((row_sum[:, None]*offset[None, :]+bias[:, None]-center[:, None]) >= theta_output)
        record = {
            'weight_int8': wq.cpu().to(torch.int8),
            'weight_row_scale': row_scale.cpu(),
            'temporal_int16': aq.cpu().to(torch.int16),
            'temporal_fractional_bits': fractional_bits,
            'threshold_int64': threshold.cpu().to(torch.int64),
            'positive_gain': positive_gain.cpu(),
            'constant_channels': (~nonzero_gain).cpu(),
            'constant_gate': constant_gate.cpu(),
            'theta_source': theta_source.cpu(), 'theta_output': theta_output.cpu(),
        }
        exported[prefix] = record
        desc = {'module': prefix, 'W_shape': list(wq.shape), 'A_shape': list(aq.shape),
                'theta_source': float(theta_source), 'theta_output': float(theta_output),
                'weight_scale_range': [float(row_scale.min()), float(row_scale.max())],
                'A_clipped_coefficients': int(((torch.round(a*scale_a) < -32768) | (torch.round(a*scale_a) > 32767)).sum()),
                'Y_accumulator_abs_bound': int(wq.abs().sum(1).max()),
                'PSN_accumulator_abs_bound': int(u_bound.max()),
                'threshold_min': int(threshold.min()), 'threshold_max': int(threshold.max()),
                'weight_zero_fraction': float((wq == 0).double().mean()),
                'source_unit_residual_first_frame': None}
        descriptions.append(desc)

        label = 'block'+prefix.split('.swin_blocks.')[1].split('.')[0]

        def linear(self, x, qw=wq.float(), source_theta=theta_source.float(), report=desc, tag=label):
            unit = x/source_theta
            if report['source_unit_residual_first_frame'] is None:
                report['source_unit_residual_first_frame'] = float((unit-unit.round()).abs().max())
                report['source_unit_abs_max_first_frame'] = float(unit.abs().max())
                if capture_dir:
                    capture_dir.mkdir(parents=True, exist_ok=True)
                    g = unit.ne(0).reshape(unit.shape[0], -1, unit.shape[-1]).cpu().numpy()
                    np.savez_compressed(capture_dir/(tag+'_source.npz'),
                        gate_bits=np.packbits(g, axis=-1, bitorder='little'), shape=np.array(g.shape))
            return torch.nn.functional.linear(unit, qw)

        def passthrough(self, x):
            return x

        def temporal(self, x, qa=aq, tau=threshold, positive=positive_gain,
                     variable=nonzero_gain, fixed=constant_gate, output_theta=theta_output.float(),
                     captured=[False], tag=label):
            shape = x.shape
            u = (qa @ x.double().reshape(shape[0], -1)).reshape(shape[0], -1, shape[-1])
            g = torch.where(positive[None, None, :], u >= tau[:, None, :], u <= tau[:, None, :])
            g = torch.where(variable[None, None, :], g, fixed[:, None, :])
            if capture_dir and not captured[0]:
                yy = x.reshape(shape[0], -1, shape[-1])
                positions = torch.linspace(0, yy.shape[1]-1, 128, device=x.device).round().long()
                channels = torch.linspace(0, yy.shape[2]-1, 32, device=x.device).round().long()
                idx = (positions[:, None]*yy.shape[2]+channels[None, :]).flatten()
                np.savez_compressed(capture_dir/(tag+'_psn.npz'),
                    x_int=yy.reshape(shape[0], -1)[:, idx].to(torch.int32).cpu().numpy(),
                    output_gate=g.reshape(shape[0], -1)[:, idx].cpu().numpy(),
                    indices=idx.cpu().numpy(), channels=(idx % yy.shape[2]).cpu().numpy(),
                    original_shape=np.array(shape))
                captured[0] = True
            return g.reshape(shape).float()*output_theta

        fc.forward = types.MethodType(linear, fc)
        bn.forward = types.MethodType(passthrough, bn)
        sn.forward = types.MethodType(temporal, sn)
    return exported, descriptions


def main():
    p = argparse.ArgumentParser()
    for name in ('code-root', 'config', 'checkpoint', 'data', 'samples', 'output', 'calibration'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--full-valid', action='store_true')
    p.add_argument('--max-valid', type=int, default=10)
    p.add_argument('--capture-first', action='store_true')
    args = p.parse_args()
    model, cfg, installed, attention = build_model(args)
    from spikingjelly.activation_based import functional
    stats = torch.load(args.calibration, map_location='cpu', weights_only=False)
    set_bn_mode(model, stats, SHALLOW)
    params, descriptions = install_integer_consumers(model, stats,
        capture_dir=args.output/'capture' if args.capture_first else None)
    names = read_names(args.data, 'valid') if args.full_valid else json.loads(args.samples.read_text())['valid'][:args.max_valid]
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(params, args.output/'integer_parameters.pt')
    rows = []
    with torch.no_grad():
        for i, name in enumerate(names):
            functional.reset_net(model)
            x, label, mask = input_frame(args.data, name)
            start = time.monotonic()
            pred = model(x)['flow'][-1]
            error = (pred-label).square().sum(1).sqrt()
            n, total = int(mask.sum()), float(error[mask].sum())
            row = {'file': name, 'valid_pixels': n, 'aee_sum': total, 'AEE': total/n,
                   'emulator_wall_s': time.monotonic()-start}
            rows.append(row)
            print('FRAME', json.dumps(row), flush=True)
            if (i+1) % 25 == 0 or i+1 == len(names):
                save_json(args.output/'frames.json', rows)
            del x, label, mask, pred, error
    summary = {'kind': 'new_quantized_model_gpu_integer_dot_emulation', 'frames': len(rows),
        'valid_pixels': sum(r['valid_pixels'] for r in rows),
        'AEE_frame_mean': float(np.mean([r['AEE'] for r in rows])),
        'AEE_pixel_mean': sum(r['aee_sum'] for r in rows)/sum(r['valid_pixels'] for r in rows),
        'modules': descriptions, 'checkpoint': str(args.checkpoint),
        'calibration': str(args.calibration),
        'capture_sample': names[0] if args.capture_first else None,
        'numeric_path': 'theta*g -> INT8 weights/INT24 accumulation -> signed Q14 temporal coefficients/INT48 accumulation -> integer threshold -> theta*g',
        'claim': 'GPU arithmetic emulation; no RTL speed, original FP32 equivalence or ASIC PPA claim'}
    save_json(args.output/'summary.json', summary)
    print('COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
