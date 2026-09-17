"""Evaluate literal frozen source/FC1 variants; no fitting or threshold export.

GPU execution is explicit through main(). Importing this module does not load Torch.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import types
import numpy as np

MODES = ('integer_teacher', 'fp32_forced_code', 'integer_source_original',
         'integer_source_adapt', 'integer_source_zero', 'pure_ep34')


def install_frozen(model, bundle, mode, prefixes, capture=None):
    import torch
    modules = dict(model.named_modules())
    device = next(model.parameters()).device
    t = {k: torch.as_tensor(v.copy(), device=device) for k, v in bundle.items()}
    observed = {'source_X_q16_abs_max': 0, 'source_U_q28_abs_max': 0,
                'FC1_Y_abs_max': [0, 0], 'PSN_U_abs_max': [0, 0],
                'source_unit_max_residual': [0.0, 0.0]}
    captured = {}

    def remember(key, value):
        if capture is not None and key not in captured:
            a = value.reshape(value.shape[0], -1, value.shape[-1])
            ix = torch.linspace(0, a.shape[1]-1, min(32, a.shape[1]), device=device).round().long()
            captured[key] = a[:, ix].detach().cpu().numpy()
            captured[key+'_positions'] = ix.cpu().numpy()

    if mode not in ('integer_teacher', 'pure_ep34'):
        source = modules[prefixes[0]+'sn1.spiking_neuron']
        D = t['dictionary'].float()
        groups = torch.arange(6, device=device)[None, :]

        def source_forward(self, x):
            if x.shape[0] != 10 or x.shape[-1] != 96:
                raise RuntimeError(f'Unexpected source shape {x.shape}')
            remember('source_X_fp32', x)
            if mode == 'fp32_forced_code':
                # Preserve the trained source's actual FP32 addmm operation.
                h = torch.addmm(t['source_bias_fp32'], t['source_A_fp32'], x.reshape(10, -1))
                gate = (h-t['source_center_fp32'] >= t['source_theta_fp32']).reshape_as(x)
            else:
                q = torch.round(x.double()*65536)
                if not bool(torch.isfinite(q).all()) or bool(((q < -(1 << 23)) | (q >= (1 << 23))).any()):
                    raise RuntimeError('Source input is outside signed24 Q16; no saturation is permitted')
                u = t['source_A_q12'].double() @ q.reshape(10, -1)
                observed['source_X_q16_abs_max'] = max(observed['source_X_q16_abs_max'], int(q.abs().max()))
                observed['source_U_q28_abs_max'] = max(observed['source_U_q28_abs_max'], int(u.abs().max()))
                gate = (u >= t['source_tau_q28'][:, None]).reshape_as(x)
                remember('source_X_q16', q.to(torch.int32))
                remember('source_U_q28', u.reshape_as(x).to(torch.int64))
            shaped = gate.float().reshape(-1, 6, 16)
            distance = shaped.sum(-1, keepdim=True)+D.sum(-1)[None]-2*torch.einsum('ngc,gkc->ngk', shaped, D)
            # Torch argmin returns the first index: the frozen lowest-index tie rule.
            index = distance.argmin(-1)
            emitted = D[groups, index].reshape_as(x)
            remember('source_raw_gate', gate.to(torch.uint8))
            remember('source_projected_gate', emitted.to(torch.uint8))
            return emitted*t['source_theta_fp32']

        source.forward = types.MethodType(source_forward, source)

    for i, prefix in enumerate(prefixes[:2]):
        p = 'consumer'+str(i)+'_'
        w = t[p+'weight_int8']
        if i == 0 and mode != 'integer_teacher':
            variant = {'fp32_forced_code': 'original', 'integer_source_original': 'original',
                       'integer_source_adapt': 'adapt', 'integer_source_zero': 'zero'}[mode]
            w = t['fc1_'+variant+'_W_int8']
        qw = w.double()
        qa = t[p+'temporal_int16'].double()
        tau = t[p+'threshold_int64']
        positive = t[p+'positive_gain']
        constant = t[p+'constant_channels']
        fixed = t[p+'constant_gate']
        source_theta = (t['source_theta_fp32'].double() if i == 0 and mode != 'integer_teacher'
                        else t[p+'theta_source'].double())
        output_theta = t[p+'theta_output'].float()

        def linear(self, x, qw=qw, theta=source_theta, idx=i):
            unit = x.double()/theta
            residual = float((unit-unit.round()).abs().max())
            observed['source_unit_max_residual'][idx] = max(observed['source_unit_max_residual'][idx], residual)
            if residual != 0 or not bool(torch.isfinite(unit).all()) or bool((unit.abs() > 1).any()):
                raise RuntimeError(f'Consumer {idx} source is not a signed unit gate: residual={residual}')
            y = torch.nn.functional.linear(unit, qw)
            observed['FC1_Y_abs_max'][idx] = max(observed['FC1_Y_abs_max'][idx], int(y.abs().max()))
            remember(f'consumer{idx}_input_gate', unit.to(torch.int8))
            remember(f'consumer{idx}_Y', y.to(torch.int32))
            # All Y values are exact integers < 2^24; retain the model's FP32 edge.
            return y.float()

        def passthrough(self, x):
            return x

        def temporal(self, x, qa=qa, tau=tau, pos=positive, const=constant,
                     fixed=fixed, theta=output_theta, idx=i):
            shape = x.shape
            u = (qa @ x.double().reshape(shape[0], -1)).reshape(shape[0], -1, shape[-1])
            gate = torch.where(pos[None, None, :], u >= tau[:, None, :], u <= tau[:, None, :])
            gate = torch.where(const[None, None, :], fixed[:, None, :], gate)
            observed['PSN_U_abs_max'][idx] = max(observed['PSN_U_abs_max'][idx], int(u.abs().max()))
            remember(f'consumer{idx}_U', u.to(torch.int64))
            remember(f'consumer{idx}_gate', gate.to(torch.uint8))
            return gate.reshape(shape).float()*theta

        modules[prefix+'fc1'].forward = types.MethodType(linear, modules[prefix+'fc1'])
        modules[prefix+'bn1.norm_layer'].forward = types.MethodType(passthrough, modules[prefix+'bn1.norm_layer'])
        modules[prefix+'sn2.spiking_neuron'].forward = types.MethodType(temporal, modules[prefix+'sn2.spiking_neuron'])
    return observed, captured


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('algorithm-root', 'code-root', 'config', 'checkpoint', 'data', 'calibration', 'samples', 'nb0', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--bundle', type=Path, default=Path(__file__).with_name('frozen_bundle.npz'))
    p.add_argument('--mode', choices=MODES, required=True)
    p.add_argument('--full-valid', action='store_true')
    p.add_argument('--max-valid', type=int, default=10)
    p.add_argument('--capture-first', action='store_true')
    args = p.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.algorithm_root))
    import torch
    from run_bn_probe import build_model, input_frame, read_names, set_bn_mode, MLPS, SHALLOW
    model, cfg, installed, attention = build_model(args)
    from spikingjelly.activation_based import functional
    bundle = dict(np.load(args.bundle, allow_pickle=False))
    if int(bundle['schema_version']) != 1:
        raise RuntimeError('Unknown frozen bundle schema')
    if args.mode != 'pure_ep34':
        stats = torch.load(args.calibration, map_location='cpu', weights_only=False)
        set_bn_mode(model, stats, SHALLOW)
        observed, captured = install_frozen(model, bundle, args.mode, MLPS,
                                           capture=args.output if args.capture_first else None)
    else:
        observed, captured = {}, {}
    names = (read_names(args.data, 'valid') if args.full_valid
             else json.loads(args.samples.read_text())['valid'][:args.max_valid])
    if len(names) != len(set(names)):
        raise RuntimeError('Repeated evaluation frame')
    nb0 = json.loads(args.nb0.read_text())
    nb_rows = {r['file']: r for r in nb0['rows']}
    if any(name not in nb_rows for name in names):
        raise RuntimeError('NB0 is missing a selected frame')
    runtime = dict(python=sys.version.split()[0], torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                   TF32_matmul=torch.backends.cuda.matmul.allow_tf32,
                   TF32_cudnn=torch.backends.cudnn.allow_tf32,
                   cudnn_benchmark=torch.backends.cudnn.benchmark,
                   window=cfg['swin_transformer']['window_size'])
    # Model/BN/ATLIF/checkpoint differences are intentional arms; runtime differences are not.
    flags = {k: runtime[k] == (nb0[k].split()[0] if k == 'python' else nb0[k])
             for k in ('python', 'torch', 'gpu', 'TF32_matmul', 'TF32_cudnn', 'window')}
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    report = dict(mode=args.mode, complete=False, training=False, new_quantization=False,
                  reestimated_thresholds=False, head="original complete model flow[-1]",
                  runtime=runtime, nb0_runtime_match=flags, nb0_path=str(args.nb0),
                  configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  metric='FP32 vector_norm on valid pixels, FP64 summation; same NB0 evaluator',
                  frozen_consumers=MLPS[:2] if args.mode != 'pure_ep34' else [],
                  width_bound_file='BUNDLE.json', observed=observed, rows=rows)
    def save():
        (args.output/'summary.json').write_text(json.dumps(report, separators=(',', ':'))+'\n')
    save()
    started = time.monotonic()
    with torch.no_grad(), (args.output/'frames.jsonl').open('w') as out:
        for i, name in enumerate(names):
            functional.reset_net(model)
            x, label, mask = input_frame(args.data, name)
            pred = model(x)['flow'][-1]
            error = torch.linalg.vector_norm(pred.permute(0, 2, 3, 1)[mask]-label.permute(0, 2, 3, 1)[mask], dim=1)
            n, total = error.numel(), float(error.double().sum())
            if n != nb_rows[name]['valid_pixels'] or not np.isfinite(total):
                raise RuntimeError('Metric mask differs from NB0 or prediction is nonfinite')
            row = dict(file=name, valid_pixels=n, aee_sum=total, AEE=total/n,
                       NB0_AEE=nb_rows[name]['AEE'], delta_AEE_vs_NB0=total/n-nb_rows[name]['AEE'])
            rows.append(row)
            out.write(json.dumps(row, separators=(',', ':'))+'\n'); out.flush()
            print('FRAME', i+1, json.dumps(row), flush=True)
            if i == 0 and captured:
                np.savez_compressed(args.output/'capture_first.npz', frame_file=np.array(name), **captured)
            if (i+1) % 25 == 0:
                save()
            del x, label, mask, pred, error
    report.update(complete=True, frames=len(rows), valid_pixels=sum(r['valid_pixels'] for r in rows),
                  AEE_frame_mean=float(np.mean([r['AEE'] for r in rows])),
                  AEE_pixel_mean=sum(r['aee_sum'] for r in rows)/sum(r['valid_pixels'] for r in rows),
                  NB0_paired_frame_mean=float(np.mean([r['NB0_AEE'] for r in rows])),
                  paired_delta_AEE=float(np.mean([r['delta_AEE_vs_NB0'] for r in rows])),
                  wall_seconds=time.monotonic()-started)
    report['better_than_paired_NB0'] = report['paired_delta_AEE'] < 0
    report['quality_gate_comparable'] = all(flags.values())
    save()
    print('COMPLETE', json.dumps({k: v for k, v in report.items() if k not in ('rows', 'configuration')}), flush=True)


if __name__ == '__main__':
    main()
