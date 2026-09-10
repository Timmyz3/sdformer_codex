"""Patch zero-column consumer opportunity and its fixed-BN strong control.

Raw Y masks are derived causally from the actual 3x3 source footprint. The
Float64 algebra check charges the full BN population and full T10 PSN; it is
not a bit-exact FP32 deployment or a cycle simulation.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from run_patch_probe import probe, RES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    system = probe.load_system(args)
    from run_bn_probe import input_frame, Calibration
    from evaluate_stage2_deployment import CoarseReady, summarize
    from spikingjelly.activation_based import functional
    model, modules, _, current, sources, _, _, _ = system
    probe.install_sources(system, sources)
    current['count_codes'] = False
    out = args.root/'algorithm/patch_probe'
    train = json.loads((args.root/'algorithm/direct_code_integer/run.json').read_text())['train']
    targets = [RES+f'{b}.norm{j}.norm_layer' for b in (0, 1) for j in (1, 2)]
    cal = Calibration(model, targets)
    with torch.no_grad():
        for i,name in enumerate(train):
            functional.reset_net(model)
            x, _, _ = input_frame(args.data, name, targets=False)
            try:
                model(x)
            except CoarseReady:
                current.pop('flow')
            print('CAL', i+1, flush=True)
    stats = cal.finish()
    torch.save(stats, out/'patch_train_calibration.pt')
    scopes = {'parent': [], 'r1_norm1_fixed': [RES+'1.norm1.norm_layer'],
              'both_norm1_fixed': [RES+f'{b}.norm1.norm_layer' for b in (0, 1)],
              'all_patch_residual_norm_fixed': targets}
    names = json.loads((args.root/'algorithm/samples.json').read_text())['valid'][:10]
    measurements, scratch = [], {}
    capture = False
    filename = ''
    handles = []
    for b in (0, 1):
        prefix = RES+str(b)+'.'
        def preconv(m, inputs, k=b):
            if capture:
                x = inputs[0].detach()
                t, batch, c, h, w = x.shape
                live = F.max_pool2d(x.ne(0).any(2).reshape(t*batch, 1, h, w).float(),
                                    3, stride=1, padding=1).bool().reshape(t, h*w)
                scratch[k] = dict(live=live)
        def convout(m, inputs, y, k=b):
            if capture:
                scratch[k]['y'] = y.detach()
        def normout(m, inputs, y, k=b):
            if capture:
                scratch[k]['bn_y'] = y.detach()
        def snout(m, inputs, output, k=b, pre=prefix):
            if not capture:
                return
            y = scratch[k]['y'].squeeze(1).reshape(10, 96, -1)
            live = scratch[k]['live']
            h, w = output.shape[-2:]
            # Entire raw-Y matrix is used for true full-domain BN moments.
            yd = y.double()
            mean = yd.mean((0, 2))
            var = yd.var((0, 2), unbiased=False)
            norm = modules[pre+'norm1.norm_layer']
            a = norm.weight.double()/torch.sqrt(var+norm.eps)
            bias = norm.bias.double()-a*mean
            neuron = modules[pre+'sn2.spiking_neuron']
            A = neuron.weight.double()
            center = (neuron.center.double() if neuron.center_mode != 'zero' else
                      torch.zeros_like(neuron.bias, dtype=torch.float64))
            nbias = (neuron.bias.double()-center).reshape(10, 1, 1)
            # Spatially stratified positions include truly empty and live bins;
            # all C and T preserved. Float64 GEMM avoids TF32 re-association.
            pos = torch.linspace(0, h*w-1, 512, device=y.device).round().long()
            small = yd[:, :, pos]
            direct = (A @ (small*a[None, :, None]+bias[None, :, None]).reshape(10, -1)).reshape_as(small)+nbias
            sparse = small*live[:, pos, None].permute(0, 2, 1)
            rearranged = (A @ sparse.reshape(10, -1)).reshape_as(small)*a[None, :, None]
            rearranged += A.sum(1)[:, None, None]*bias[None, :, None]+nbias
            theta = neuron.thresh.double()
            default = A.sum(1)[:, None]*bias[None, :]+nbias[:, :, 0]
            original_g = output.detach().squeeze(1).reshape(10, 96, -1)[:, :, pos].ne(0)
            # The exact empty footprint is shared across H, no per-h oracle.
            known_zero_values = int((~live).sum())*96
            empty_positions = int((~live.any(0)).sum())
            raw_values = y.numel()
            total_columns = live.numel()
            dense_psn_mac = 10*raw_values
            sparse_psn_mac = 10*(raw_values-known_zero_values)
            # Width sweep is representation accounting, not an admitted fixed point.
            widths = {}
            for bits in (16, 24, 32):
                raw_bytes = (raw_values*bits+7)//8
                comp_bytes = ((raw_values-known_zero_values)*bits+7)//8
                valid_bytes = (total_columns+7)//8
                widths[str(bits)] = dict(dense_Y_bytes=raw_bytes,
                    sparse_Y_payload_bytes=comp_bytes, shared_validity_bytes=valid_bytes,
                    prefix_u32_per_spatial_row_bytes=4*(h+1),
                    sparse_Y_total_bytes=comp_bytes+valid_bytes+4*(h+1))
            item = dict(file=filename, block=k, Y_shape=[10,96,h,w],
                zero_column_fraction=float((~live).float().mean()),
                zero_full_T10_fraction=empty_positions/(h*w),
                raw_Y_values=raw_values, proved_zero_Y_values=known_zero_values,
                default_T10_nonzero_gates=int((default >= theta).sum()), default_T10_gate_count=960,
                dense_PSN_MAC=dense_psn_mac, skip_zero_column_PSN_MAC=sparse_psn_mac,
                default_affine_setup_multiplies=960,
                dense_BN_population_per_channel=10*h*w,
                raw_Y_on_empty_footprint_max=float(y.masked_select((~live)[:, None, :]).abs().max()) if known_zero_values else 0.,
                algebra_samples=direct.numel(), algebra_max_abs=float((direct-rearranged).abs().max()),
                float64_gate_mismatches=int(((direct >= theta)!=(rearranged >= theta)).sum()),
                reference_FP32_gate_vs_float64_mismatches=int((original_g!=(direct>=theta)).sum()),
                theta=float(theta), representation_width_sweep=widths,
                omitted_costs=['source mask generation','compaction/popcount and address generation',
                    'memory ports and stalls','BN moment accumulation/reduction',
                    'nonzero Y affine application and final gate comparisons',
                    'conv2, norm2 and shortcut timeline'])
            measurements.append(item)
            scratch.pop(k)
        handles.extend([modules[prefix+'conv1.0'].register_forward_pre_hook(preconv),
                        modules[prefix+'conv1.0'].register_forward_hook(convout),
                        modules[prefix+'norm1'].register_forward_hook(normout),
                        modules[prefix+'sn2'].register_forward_hook(snout)])
    summaries = {}
    with torch.no_grad():
        for scope, selected in scopes.items():
            for name in targets:
                m = modules[name]
                m.track_running_stats = name in selected
                m.running_mean = stats[name]['mean'].to(m.weight) if name in selected else None
                m.running_var = stats[name]['var'].to(m.weight) if name in selected else None
            capture = scope == 'parent'
            rows = []
            started = time.monotonic()
            for name in names:
                filename = name
                functional.reset_net(model)
                x, label, mask = input_frame(args.data, name)
                try:
                    model(x)
                except CoarseReady:
                    pred = F.interpolate(current.pop('flow'), (480,640), mode='bilinear', align_corners=False)
                error = torch.linalg.vector_norm(pred.permute(0,2,3,1)[mask]-label.permute(0,2,3,1)[mask], dim=1)
                total,pixels = float(error.double().sum()), error.numel()
                rows.append(dict(file=name, valid_pixels=pixels, aee_sum=total, AEE=total/pixels))
            summaries[scope] = dict(**summarize(rows, False), wall_seconds=time.monotonic()-started)
            probe.save_json(out/'fixed_bn_valid10_summary.json', summaries)
            probe.save_json(out/(scope+'_bn_valid10_frames.json'), rows)
            probe.save_json(out/'zero_column_consumer.json', measurements)
            print('RESULT', scope, json.dumps(summaries[scope]), flush=True)
    probe.save_json(out/'consumer_run.json', dict(train=train, valid=names,
        claim='zero-column Float64 algebra and state/operation counts, not RTL speedup',
        strongest_required_control='same compressed raw Y plus full-domain BN compensation and PSN zero-column skip',
        fixed_BN='train32 calibration only, no fitting or use of validation statistics'))


if __name__ == '__main__':
    main()
