"""Real flow evaluation of the fitted two-factor Conv1 controls.

The unfactored control receives the same saved temporal neuron and four fixed
patch BNs. All paths execute real Conv2, BN2, shortcut and the existing coarse
flow successor. Dense CUDA execution checks accuracy, not sparse speed.
"""
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
sys.path.insert(0, str(HERE.parent/'joint_completion_20260909'))
from evaluate_network import BLOCK, TARGET, full_precision_matmul, save_json
from train_factors import install_conv1
from conditional_adapter import install_conditional_factor


class TemporalNeuron:
    def __init__(self, arrays, device):
        self.a = torch.as_tensor(arrays['a'], device=device).float()
        self.b = torch.as_tensor(arrays['temporal_bias'], device=device).float().reshape(10)
        self.theta = float(arrays['theta_output'])

    def __call__(self, x):
        t, batch, channels, height, width = x.shape
        assert (t, batch, channels) == (10, 1, 96)
        output = torch.empty_like(x)
        # Work on bounded spatial groups, with the same explicit FP32
        # temporal arithmetic as the fitted controls and their teacher.
        with full_precision_matmul(x.device):
            for first in range(0, height, 8):
                last = min(first+8, height)
                local = x[:, 0, :, first:last, :].permute(1, 2, 3, 0)
                margin = local @ self.a.T+self.b-self.theta
                output[:, 0, :, first:last, :] = (
                    (margin >= 0).to(x.dtype)*self.theta).permute(3, 0, 1, 2)
        return output


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model-files', type=Path, nargs='+', required=True)
    p.add_argument('--split', choices=('diverse', 'valid'), default='diverse')
    p.add_argument('--count', type=int, default=10)
    p.add_argument('--modes', nargs='+', choices=('full', 'conditional'), default=['full'])
    p.add_argument('--skip-unfactored', action='store_true')
    p.add_argument('--latent-stages', action='store_true',
                   help='use the disjoint shared/private latent adapter for both full and conditional modes')
    p.add_argument('--integer-factors', action='store_true',
                   help='evaluate the separate compiled U8/dyadic V/Aq14 student with integer threshold comparisons')
    p.add_argument('--capture', action='store_true',
                   help='with --latent-stages, save first four frames: full gate/demand bits and five small real-valued windows')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.capture and not args.latent_stages:
        p.error('--capture is defined only for --latent-stages, not old time-prefix demands')
    if args.integer_factors and args.capture:
        p.error('The FP latent-window capture does not represent integer nodes; run integer precision evaluation separately')
    sys.path.insert(0, str(args.root/'algorithm/nrv_cost_probe'))
    sys.path.insert(0, str(args.root/'algorithm'))
    import run_probe as probe
    from run_bn_probe import input_frame, read_names
    from evaluate_stage2_deployment import CoarseReady, summarize
    from spikingjelly.activation_based import functional
    system = probe.load_system(args)
    model, modules, _, current, sources, _, _, _ = system
    probe.install_sources(system, sources)
    current['count_codes'] = False
    fixed = torch.load(args.root/'algorithm/patch_probe/patch_train_calibration.pt',
                       map_location='cpu', weights_only=False)
    for name, values in fixed.items():
        bn = modules[name]
        bn.track_running_stats = True
        bn.running_mean = values['mean'].to(bn.weight)
        bn.running_var = values['var'].to(bn.weight)
    conv, neuron = modules[BLOCK+'.conv1.0'], modules[TARGET]
    original_conv, original_neuron = conv.forward, neuron.forward
    assert neuron.center_mode == 'zero'
    names = (read_names(args.data, 'valid') if args.split == 'valid' else
             json.loads((args.root/'algorithm/samples.json').read_text())['valid'])
    names = names[:args.count]
    arrays = []
    for file in args.model_files:
        with np.load(file) as z:
            arrays.append({key: z[key].copy() for key in z.files})
    for a in arrays:
        assert float(a['theta_source']) == float(modules[BLOCK+'.sn1.spiking_neuron'].thresh)
        assert float(a['theta_output']) == float(neuron.thresh)
        assert np.array_equal(a['a'], arrays[0]['a'])
        assert np.array_equal(a['temporal_bias'], arrays[0]['temporal_bias'])
    temporal_arrays = dict(arrays[0])
    if args.integer_factors:
        temporal_arrays['a'] = arrays[0]['integer_a_q14'].astype(np.float32)/16384
    temporal = TemporalNeuron(temporal_arrays, conv.weight.device)
    neuron.forward = types.MethodType(lambda self, x: temporal(x), neuron)
    args.output.mkdir(parents=True, exist_ok=True)
    run = dict(complete=False, split=args.split, files=names,
        parent='saved integer-S2-source/coarse-head student; four fixed patch BN',
        model_files=[str(f) for f in args.model_files],
        unfactored_control='original Conv1 with exactly the same saved A/bias/theta',
        numeric=('distinct deployed student: theta folded into U8; integer Z; aligned dyadic V; Aq14; BN folded into integer thresholds; FP64 exact-range integer reference'
                 if args.integer_factors else 'original FP32 Conv1 or two FP32 factors; continuous latent Z; fixed outer BN and T10 neuron'),
        temporal_tf32=False, convolution_cudnn_tf32=torch.backends.cudnn.allow_tf32,
        modes=args.modes,
        phase_axis=('disjoint latent factors, each complete T10' if args.latent_stages or args.integer_factors else 'time prefix and tail'),
        consumer='real Conv2, BN2, shortcut, downstream network and preds.2 coarse flow',
        claim='flow AEE; no physical source, weight, latent or output work skipped on GPU', results={})
    if args.capture:
        from latent_capture import LatentFrameCapture, save_latent_consumer
        save_latent_consumer(modules, fixed, args.output/'capture')
        run['capture'] = dict(first_frames=min(4, len(names)),
            axes='saved latent factor students only; full and conditional separately',
            gate_layout='T,C,H,W little bitpack; full mode accepted=0',
            demand_layout='G,S,P,J2_tail; actual A/V dependencies and source-empty metadata',
            values='five common border/interior 5x5 windows only; no full FP32 identity/Y/Z',
            actual_weights='each axis/student_parameters.npz; global unfactored_reference_W1 is not executed',
            full_preview='capture-only CPU window diagnostic, not a full-mode permission')
    save_json(args.output/'run.json', run)
    axes = [] if args.skip_unfactored else [('unfactored_same_temporal', None, 'full')]
    axes += [(f'{i:02d}_{f.parent.name}_{f.stem}'+('_conditional' if mode == 'conditional' else ''), f, mode)
             for i, f in enumerate(args.model_files) for mode in args.modes]
    capture = None
    try:
        for axis, file, mode in axes:
            conv.forward = original_conv
            neuron.forward = types.MethodType(lambda self, x: temporal(x), neuron)
            adapter = None
            if file is not None:
                if args.integer_factors:
                    from latent_stage_train16.integer_adapter import install_integer_factor
                    adapter, _ = install_integer_factor(conv, neuron, file, conditional=(mode == 'conditional'))
                elif args.latent_stages:
                    from latent_stage_train16.adapter import install_latent_factor
                    if args.capture:
                        with np.load(file) as data:
                            captured_arrays = {key: data[key].copy() for key in data.files}
                        capture = LatentFrameCapture(modules, args.output/'capture'/axis,
                                                     captured_arrays, mode == 'conditional')
                    adapter, _ = install_latent_factor(conv, neuron, file, conditional=(mode == 'conditional'),
                                                       capture_callback=capture)
                elif mode == 'conditional':
                    adapter, _ = install_conditional_factor(conv, neuron, file)
                else:
                    adapter, _ = install_conv1(conv, file)
                    adapter.eval()
            rows, started = [], time.monotonic()
            for i, name in enumerate(names):
                functional.reset_net(model)
                if capture is not None:
                    capture.begin(i, name)
                x, label, valid = input_frame(args.data, name)
                try:
                    model(x)
                except CoarseReady:
                    pred = F.interpolate(current.pop('flow'), (480, 640), mode='bilinear', align_corners=False)
                error = torch.linalg.vector_norm(pred.permute(0, 2, 3, 1)[valid]
                         -label.permute(0, 2, 3, 1)[valid], dim=1)
                total, count = float(error.double().sum()), error.numel()
                rows.append(dict(file=name, valid_pixels=count, aee_sum=total, AEE=total/count))
                if mode == 'conditional':
                    rows[-1]['completion'] = dict(adapter.last_counts)
                if capture is not None:
                    capture.finish()
                if i < 4 or (i+1) % 25 == 0 or i+1 == len(names):
                    summary = dict(**summarize(rows, i+1 == len(names)),
                                   axis=axis, mode=mode, wall_seconds=time.monotonic()-started)
                    if mode == 'conditional':
                        summary['completion_table'] = (adapter.table_metadata if args.integer_factors else adapter.temporal.table_metadata)
                    if args.integer_factors and adapter is not None:
                        summary['numeric_checks'] = dict(adapter.numeric_checks)
                    save_json(args.output/(axis+'_frames.json'), rows)
                    save_json(args.output/(axis+'_summary.json'), summary)
                    print('FACTOR_AEE', axis, i+1, summary['AEE_frame_mean'], flush=True)
                del x, label, valid, pred, error
            run['results'][axis] = summary
            save_json(args.output/'run.json', run)
            if capture is not None:
                capture.close()
                capture = None
            del adapter
        run['complete'] = True
        save_json(args.output/'run.json', run)
    finally:
        if capture is not None:
            capture.close()
        conv.forward, neuron.forward = original_conv, original_neuron
    print('DONE', json.dumps(run['results']), flush=True)


if __name__ == '__main__':
    main()
