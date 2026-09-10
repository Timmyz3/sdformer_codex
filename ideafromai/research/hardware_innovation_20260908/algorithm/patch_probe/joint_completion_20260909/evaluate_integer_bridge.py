"""Real-network AEE for the NEW common-W8 joint-completion students.

Source theta*g is checked and converted to g.  Conv1 computes signed W8 sums
Zi, the norm1 wrapper is bypassed, and saved full/prefix integer thresholds
already contain its BN offset/gain and the learned per-(h,t) predictor.
The emitter keeps theta_amplitude.  Real Conv2, fixed BN2, shortcut, and the
existing CoarseReady successor all execute.  --quantize-conv2 is a separate
ordinary control that also folds and quantizes W2/BN2 with the same per-H
dyadic W8 rule, returning an affine FP32 branch value before the real shortcut.
Quantizing W1 (or W1+W2) changes the model;
this is neither the earlier integer model nor FP-student equivalence.

The GPU reference uses FP32 integer-valued W8 accumulation (within INT24),
then FP64 integer-valued Aq14 accumulation (within INT48).  TF32 is disabled
only inside these integer kernels.  Spatial stripes bound Conv1 unfold memory.
Both exact and conditional modes still produce complete Conv1/PSN values in
this reference; no physical skipping or cycle result is claimed.

--capture follows evaluate_network.py's gate/window schema, with explicit
virtual folded-convolution fields: conv1_raw=scale*Zi, norm1_Y=scale*Zi+offset,
and an additional exact conv1_Z_int32 window.  parameters.npz exports the
equivalent folded W1/affine BN1 as well as actual W1_int8/scale.  The integer
threshold arrays, not a Float64 window re-evaluation, define decisions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import types

sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from compile_integer_bridge import integer_convolution, integer_gates
from evaluate_network import (BLOCK, TARGET, BlockReady, FrameCapture,
                              full_precision_matmul, save_consumer, save_json,
                              window_snapshot)


class IntegerKernel:
    def __init__(self, path, device='cpu'):
        self.path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            self.arrays = {key: np.array(z[key], copy=True) for key in z.files}
        a = self.arrays
        self.source_theta = float(a['theta_source'])
        self.theta = float(a['theta_amplitude'])
        self.prefix = a['prefix'].astype(int).tolist()
        self.weight = torch.from_numpy(a['weight_int8'].reshape(96, 864)).to(device).float()
        self.a = torch.from_numpy(a['temporal_int16']).to(device).double()
        self.full_threshold = torch.from_numpy(a['full_threshold'].T.copy()).to(device).double()
        self.positive = torch.from_numpy(a['prefix_threshold_positive'].T.copy()).to(device).double()
        self.negative = torch.from_numpy(a['prefix_threshold_negative'].T.copy()).to(device).double()
        self.known = torch.zeros(10, device=device, dtype=torch.bool)
        self.known[self.prefix] = True
        self.support = self.a.ne(0)
        self.channels = torch.arange(96, device=device).reshape(12, 8)
        self.numeric_checks = {}
        self.frame_counts = {}

    @torch.no_grad()
    def convolution(self, x, stripe_rows=16):
        t, batch, c, h, w = x.shape
        assert (t, batch, c) == (10, 1, 96)
        unit = x[:, 0].ne(0).float()
        first_check = 'conv' not in self.numeric_checks
        if first_check:
            payload_error = float((x[:, 0]-unit*self.source_theta).abs().max())
            assert payload_error == 0, 'source does not have the saved theta*g amplitude'
        padded = F.pad(unit, (1, 1, 1, 1))
        out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
        residual = torch.zeros((), device=x.device)
        lo = torch.full((96,), float('inf'), device=x.device)
        hi = torch.full((96,), -float('inf'), device=x.device)
        with full_precision_matmul(x.device):
            for first in range(0, h, stripe_rows):
                last = min(first+stripe_rows, h)
                columns = F.unfold(padded[..., first:last+2, :], kernel_size=3)
                z = torch.matmul(self.weight[None], columns)
                if first_check:
                    residual = torch.maximum(residual, (z-z.round()).abs().amax())
                    lo = torch.minimum(lo, z.amin((0, 2)))
                    hi = torch.maximum(hi, z.amax((0, 2)))
                out[:, 0, :, first:last, :] = z.reshape(10, 96, last-first, w)
        if first_check:
            assert float(residual) == 0
            assert bool((lo >= torch.from_numpy(self.arrays['Y_lower']).to(lo)).all())
            assert bool((hi <= torch.from_numpy(self.arrays['Y_upper']).to(hi)).all())
            self.numeric_checks['conv'] = dict(source_theta_residual=payload_error,
                                               integer_residual=float(residual),
                                               observed_min=float(lo.min()), observed_max=float(hi.max()),
                                               arithmetic='explicit im2col FP32 GEMM; integer terms and sums within 24 bits')
        return out

    def context_decisions(self, z, channels, mode):
        full_u = z.double() @ self.a.T
        full_gate = full_u >= self.full_threshold[channels][:, :, None, :]
        if mode == 'exact':
            return full_gate, None, full_gate, full_u
        prefix_u = (z.double()*self.known) @ self.a.T
        positive = prefix_u >= self.positive[channels][:, :, None, :]
        negative = prefix_u <= self.negative[channels][:, :, None, :]
        accepted = positive | negative
        return torch.where(accepted, positive, full_gate), accepted, full_gate, full_u

    @torch.no_grad()
    def network_output(self, x, mode, chunk_groups=256, decision_sink=None):
        t, batch, c, h, w = x.shape
        assert (t, batch, c) == (10, 1, 96) and w % 4 == 0
        ngroups = h*(w//4)
        source = x[:, 0].reshape(10, 96, ngroups, 4)
        output = torch.empty(x.shape, device=x.device, dtype=torch.float32)
        destination = output[:, 0].reshape(10, 96, ngroups, 4)
        counts = torch.zeros(3, device=x.device, dtype=torch.int64)
        first_check = 'neuron' not in self.numeric_checks
        u_residual = torch.zeros((), device=x.device, dtype=torch.float64)
        u_min = torch.full((), float('inf'), device=x.device, dtype=torch.float64)
        u_max = torch.full((), -float('inf'), device=x.device, dtype=torch.float64)
        with full_precision_matmul(x.device):
            for first in range(0, ngroups, chunk_groups):
                last = min(first+chunk_groups, ngroups)
                n = last-first
                z = source[:, :, first:last].permute(2, 1, 3, 0).reshape(n*12, 8, 4, 10)
                channels = self.channels.repeat(n, 1)
                gate, accepted, full, u = self.context_decisions(z, channels, mode)
                if first_check:
                    u_residual = torch.maximum(u_residual, (u-u.round()).abs().amax())
                    u_min = torch.minimum(u_min, u.amin())
                    u_max = torch.maximum(u_max, u.amax())
                counts[0] += gate.sum()
                counts[1] += (gate != full).sum()
                if accepted is not None:
                    counts[2] += accepted.sum()
                if decision_sink is not None:
                    if accepted is None:
                        need = torch.ones((n*12, 10), device=x.device, dtype=torch.bool)
                    else:
                        need = ((~accepted)[..., None] & self.support).any(-2).any((1, 2)) | self.known
                    decision_sink(first, last, ngroups, need, accepted)
                destination[:, :, first:last] = (gate.float()*self.theta).reshape(n, 96, 4, 10).permute(3, 1, 0, 2)
        if first_check:
            assert float(u_residual) == 0
            assert float(u_min) >= int(self.arrays['U_full_lower'].min())
            assert float(u_max) <= int(self.arrays['U_full_upper'].max())
            self.numeric_checks['neuron'] = dict(integer_residual=float(u_residual),
                                                 observed_min=float(u_min), observed_max=float(u_max),
                                                 arithmetic='FP64 exact-range integer Aq14 x Zi; integer threshold comparisons')
        values = counts.cpu().tolist()
        self.frame_counts = dict(gates=output.numel(), positive_gates=values[0],
                                 difference_from_own_full=values[1], accepted=values[2])
        return output


class DyadicConv2Control:
    """Ordinary fixed-BN2 folding/QDQ; no learned mask or changed shortcut."""

    # The same exact-range im2col engine is used by both convolutions.
    convolution = IntegerKernel.convolution

    def __init__(self, weight, conv_bias, gamma, beta, mean, variance, eps,
                 source_theta, device='cpu'):
        self.source_theta = float(source_theta)
        gain = np.asarray(gamma, dtype=np.float64)/np.sqrt(
            np.asarray(variance, dtype=np.float64)+float(eps))
        bias = (np.asarray(beta, dtype=np.float64)
                + gain*(np.asarray(conv_bias, dtype=np.float64)-np.asarray(mean, dtype=np.float64)))
        folded = np.asarray(weight, dtype=np.float64)*self.source_theta*gain[:, None, None, None]
        peak = np.abs(folded).reshape(96, -1).max(1)
        exponent = np.zeros(96, dtype=np.int32)
        nonzero = peak != 0
        exponent[nonzero] = np.ceil(np.log2(peak[nonzero]/127.)).astype(np.int32)
        scale = np.ldexp(np.ones(96), exponent)
        wq = np.clip(np.rint(folded/scale[:, None, None, None]), -127, 127).astype(np.int8)
        flat = wq.reshape(96, 864).astype(np.int64)
        self.arrays = dict(weight_int8=wq, weight_scale=scale,
                           weight_scale_exponent=exponent, bn_bias_fp64=bias,
                           bn_gain_fp64=gain, theta_source=np.asarray(self.source_theta),
                           Y_lower=np.minimum(flat, 0).sum(1),
                           Y_upper=np.maximum(flat, 0).sum(1))
        assert max(np.abs(self.arrays['Y_lower']).max(), np.abs(self.arrays['Y_upper']).max()) < 2**23
        self.weight = torch.from_numpy(wq.reshape(96, 864)).to(device).float()
        self.scale = torch.from_numpy(scale).to(device)[None, None, :, None, None]
        self.bias = torch.from_numpy(bias).to(device)[None, None, :, None, None]
        self.numeric_checks = {}

    @classmethod
    def from_modules(cls, conv, bn, source_theta):
        def cpu(value):
            return value.detach().cpu().numpy()
        conv_bias = cpu(conv.bias) if conv.bias is not None else np.zeros(96)
        return cls(cpu(conv.weight), conv_bias, cpu(bn.weight), cpu(bn.bias),
                   cpu(bn.running_mean), cpu(bn.running_var), bn.eps,
                   source_theta, conv.weight.device)

    @classmethod
    def from_parameters(cls, path, source_theta, device='cpu'):
        with np.load(path, allow_pickle=False) as z:
            return cls(z['W2'], z['conv2_bias'], z['bn2_gamma'], z['bn2_beta'],
                       z['bn2_mean'], z['bn2_var'], z['bn2_eps'], source_theta, device)

    def affine(self, z):
        # The constant is retained even when every source gate is zero.
        # One explicit Float64 affine evaluation followed by FP32 storage
        # defines this new numerical control; it is not original BN equality.
        return (z.double()*self.scale+self.bias).float()

    def description(self):
        a = self.arrays
        return dict(weight_zero_fraction=float((a['weight_int8'] == 0).mean()),
                    weight_scale_min=float(a['weight_scale'].min()),
                    weight_scale_max=float(a['weight_scale'].max()),
                    integer_lower=int(a['Y_lower'].min()), integer_upper=int(a['Y_upper'].max()),
                    source_theta=self.source_theta,
                    normalization='fixed BN2 gain and theta folded into W2; per-H dyadic W8; Z2*scale+BN2_offset rounded to FP32 before original shortcut',
                    zero_source='branch remains BN2_offset, not zero',
                    numeric_checks=self.numeric_checks)


class IntegerCapture(FrameCapture):
    def __init__(self, *args, kernel, conv2_control=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel = kernel
        self.conv2_control = conv2_control

    def conv1(self, module, inputs, output):
        if self.dense_active:
            z, _ = window_snapshot(output)
            self.record['conv1_Z_int32'] = z.astype(np.int32)
            scale = self.kernel.arrays['weight_scale'][None, None, :, None, None]
            self.record['conv1_raw'] = z.astype(np.float64)*scale

    def norm1(self, module, inputs):
        if self.dense_active:
            bias = self.kernel.arrays['bn_bias_fp64'][None, None, :, None, None]
            self.record['norm1_Y'] = self.record['conv1_raw']+bias

    def conv2(self, module, inputs, output):
        if self.conv2_control is None:
            super().conv2(module, inputs, output)
        elif self.dense_active:
            z, _ = window_snapshot(output)
            self.record['conv2_Z_int32'] = z.astype(np.int32)
            scale = self.conv2_control.arrays['weight_scale'][None, None, :, None, None]
            self.record['conv2_raw'] = z.astype(np.float64)*scale


def save_integer_consumer(modules, fixed, kernel, output, conv2_control=None):
    save_consumer(modules, fixed, output)
    path = output/'parameters.npz'
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: np.array(z[k], copy=True) for k in z.files}
    a = kernel.arrays
    arrays['W1_parent_fp32'] = arrays['W1']
    arrays['W1'] = (a['weight_int8'].astype(np.float64)*a['weight_scale'][:, None, None, None]
                    /kernel.source_theta)
    arrays['W1_int8'] = a['weight_int8']
    arrays['W1_dyadic_scale'] = a['weight_scale']
    arrays['conv1_bias'] = np.zeros(96)
    for field in ('gamma', 'beta', 'mean', 'var', 'eps'):
        arrays['bn1_parent_'+field] = arrays['bn1_'+field]
    arrays.update(bn1_gamma=np.ones(96), bn1_beta=a['bn_bias_fp64'],
                  bn1_mean=np.zeros(96), bn1_var=np.ones(96), bn1_eps=np.asarray(0.),
                  theta_source=np.asarray(kernel.source_theta), theta_output=np.asarray(kernel.theta),
                  integer_bridge=np.asarray(str(kernel.path)),
                  W1_semantics=np.asarray('virtual folded W1 for theta*g; actual primitive is g x W1_int8 -> Zi'),
                  norm1_semantics=np.asarray('virtual scale*Zi+offset; runtime BN bypassed and constants compiled into neuron thresholds'))
    arrays['quantize_conv2'] = np.asarray(conv2_control is not None)
    if conv2_control is not None:
        c = conv2_control.arrays
        arrays['W2_parent_fp32'] = arrays['W2']
        arrays['W2'] = c['weight_int8'].astype(np.float64)*c['weight_scale'][:, None, None, None]/kernel.theta
        arrays['W2_int8'] = c['weight_int8']
        arrays['W2_dyadic_scale'] = c['weight_scale']
        arrays['conv2_bias'] = np.zeros(96)
        for field in ('gamma', 'beta', 'mean', 'var', 'eps'):
            arrays['bn2_parent_'+field] = arrays['bn2_'+field]
        arrays.update(bn2_gamma=np.ones(96), bn2_beta=c['bn_bias_fp64'],
                      bn2_mean=np.zeros(96), bn2_var=np.ones(96), bn2_eps=np.asarray(0.),
                      W2_semantics=np.asarray('virtual folded W2 for theta*g; actual primitive is g x W2_int8 -> Z2'),
                      norm2_semantics=np.asarray('scale*Z2+BN2_offset, rounded to FP32, then original shortcut'))
    np.savez(path, **arrays)


@torch.no_grad()
def local_check(args):
    torch.set_num_threads(args.threads)
    paths = sorted(args.source_directory.rglob('sampled_source.npz'))[:4]
    assert paths
    rows = []
    for path in args.model_files:
        kernel = IntegerKernel(path)
        conv2_control = (DyadicConv2Control.from_parameters(args.conv2_parameters, kernel.theta)
                         if args.quantize_conv2 else None)
        conv2_check = None
        mismatch = dict(native_exact=0, native_conditional=0, accepted=0, samples=0)
        for source_file in paths:
            with np.load(source_file, allow_pickle=False) as z:
                zi = integer_convolution(z['source_gate_words'], kernel.arrays['weight_int8'])
            expected, conditional, accepted, _, _ = integer_gates(zi, kernel.arrays)
            # Real sampled values retain T/C/P4 identity; arranging groups
            # along one image row is a tensor-layout check, not a new conv.
            native = torch.from_numpy(zi).float().reshape(10, 1, 96, 1, -1)
            got = kernel.network_output(native, 'exact', chunk_groups=7)
            got_c = kernel.network_output(native, 'conditional', chunk_groups=7)
            mismatch['native_exact'] += int(np.count_nonzero(got.ne(0).numpy().reshape(zi.shape) != expected))
            mismatch['native_conditional'] += int(np.count_nonzero(got_c.ne(0).numpy().reshape(zi.shape) != conditional))
            y = torch.from_numpy(zi).float().permute(2, 1, 3, 0).reshape(-1, 8, 4, 10)
            h = kernel.channels.repeat(zi.shape[2], 1)
            _, got_accept, _, _ = kernel.context_decisions(y, h, 'conditional')
            got_accept = got_accept.reshape(zi.shape[2], 96, 4, 10).permute(3, 1, 0, 2).numpy()
            mismatch['accepted'] += int(np.count_nonzero(got_accept != accepted))
            mismatch['samples'] += int(expected.size)
            if conv2_control is not None and conv2_check is None:
                # The sampled P4 groups form only a layout-test domain here.
                # It is not a claim about real intervening Conv2 neighbors.
                z2 = conv2_control.convolution(got_c, stripe_rows=1)
                expected_z2 = F.conv2d(got_c[:, 0].double()/kernel.theta,
                                      conv2_control.weight.double().reshape(96, 96, 3, 3), padding=1)
                branch = conv2_control.affine(z2)
                expected_branch = (expected_z2[:, None]*conv2_control.scale+conv2_control.bias).float()
                zero_branch = conv2_control.affine(torch.zeros_like(z2))
                zero_expected = conv2_control.bias.float().expand_as(z2)
                conv2_check = dict(integer_sum_mismatches=int((z2[:, 0].double() != expected_z2).sum()),
                                   affine_output_mismatches=int((branch != expected_branch).sum()),
                                   zero_source_bias_mismatches=int((zero_branch != zero_expected).sum()),
                                   elements=z2.numel(), **conv2_control.description())
                assert all(conv2_check[key] == 0 for key in
                           ('integer_sum_mismatches', 'affine_output_mismatches', 'zero_source_bias_mismatches'))
        assert mismatch['native_exact'] == mismatch['native_conditional'] == mismatch['accepted'] == 0
        rows.append(dict(model=str(path), prefix=kernel.prefix, conv2_check=conv2_check, **mismatch))
        print('LOCAL_INTEGER_LAYOUT', json.dumps(rows[-1]), flush=True)
    print('LOCAL_DONE', json.dumps(dict(scope='CPU true captured integer Zi only; no network AEE', models=rows)), flush=True)


@torch.no_grad()
def run_network(args):
    algorithm = args.root/'algorithm'
    sys.path.insert(0, str(algorithm))
    sys.path.insert(0, str(algorithm/'nrv_cost_probe'))
    import run_probe as probe
    system = probe.load_system(args)
    from run_bn_probe import input_frame, read_names
    from evaluate_stage2_deployment import CoarseReady, summarize
    from spikingjelly.activation_based import functional
    model, modules, _, current, sources, _, _, _ = system
    probe.install_sources(system, sources)
    current['count_codes'] = False
    fixed = torch.load(algorithm/'patch_probe/patch_train_calibration.pt',
                       map_location='cpu', weights_only=False)
    for name, values in fixed.items():
        bn = modules[name]
        bn.track_running_stats = True
        bn.running_mean = values['mean'].to(bn.weight)
        bn.running_var = values['var'].to(bn.weight)
    conv1, norm1, neuron = (modules[BLOCK+'.conv1.0'], modules[BLOCK+'.norm1'], modules[TARGET])
    conv2, norm2 = modules[BLOCK+'.conv2.0'], modules[BLOCK+'.norm2']
    assert neuron.center_mode == 'zero'
    original = conv1.forward, norm1.forward, neuron.forward, conv2.forward, norm2.forward
    conv2_control = None
    if args.quantize_conv2:
        bn2_names = [name for name in fixed if name.startswith(BLOCK+'.norm2')]
        assert len(bn2_names) == 1
        conv2_control = DyadicConv2Control.from_modules(conv2, modules[bn2_names[0]], float(neuron.thresh))
        conv2.forward = types.MethodType(lambda self, x: conv2_control.convolution(x, args.stripe_rows), conv2)
        norm2.forward = types.MethodType(lambda self, z: conv2_control.affine(z), norm2)
    names = (json.loads((algorithm/'samples.json').read_text())['valid']
             if args.split == 'diverse' else read_names(args.data, 'valid'))
    if args.count:
        names = names[:args.count]
    args.output.mkdir(parents=True, exist_ok=True)
    run = dict(complete=False, files=names, modes=args.modes,
               model_files=[str(p) for p in args.model_files], split=args.split,
               numeric='NEW common dyadic W8 / integer Zi / Q14 A / compiled integer full+prefix predicates',
               parent='saved integer-S2-source/coarse student with four fixed patch BN',
               norm1='gain folded in W1, offset in compiled thresholds; wrapper bypassed',
               successor=('ordinary dyadic W8 Conv2/fixed-BN2 affine, then real shortcut and all downstream layers through CoarseReady'
                          if args.quantize_conv2 else
                          'real unquantized Conv2, fixed BN2, shortcut and all downstream layers through CoarseReady'),
               quantize_conv2=args.quantize_conv2,
               claim='new-model algorithm AEE only; dense reference computes complete Conv1 and fallback PSN',
               parent_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
               parent_tf32_cudnn=torch.backends.cudnn.allow_tf32,
               integer_kernel_tf32=False, stripe_rows=args.stripe_rows,
               chunk_groups=args.chunk_groups, stop_at_block=args.stop_at_block, results={})
    save_json(args.output/'run.json', run)
    if conv2_control is not None:
        np.savez_compressed(args.output/'conv2_integer_control.npz', **conv2_control.arrays)
        run['conv2_integer_control'] = conv2_control.description()
    common = None
    try:
        for index, path in enumerate(args.model_files):
            kernel = IntegerKernel(path, conv1.weight.device)
            if common is None:
                common = kernel.arrays
            else:
                for key in ('weight_int8', 'weight_scale', 'bn_bias_fp64', 'theta_source'):
                    assert np.array_equal(kernel.arrays[key], common[key]), 'the two students must share the same W8 producer'
            assert kernel.source_theta == float(modules[BLOCK+'.sn1.spiking_neuron'].thresh)
            assert kernel.theta == float(neuron.thresh)
            assert float(kernel.arrays['theta_decision']) == float(neuron.thresh)
            conv1.forward = types.MethodType(lambda self, x, k=kernel: k.convolution(x, args.stripe_rows), conv1)
            norm1.forward = types.MethodType(lambda self, x: x, norm1)
            label = f'{index:02d}_{path.stem}'
            if args.quantize_conv2:
                label += '_w1w2'
            if args.capture:
                save_integer_consumer(modules, fixed, kernel, args.output/'capture'/label, conv2_control)
            for mode in args.modes:
                axis = label+'_'+mode
                capture = (IntegerCapture(modules, args.output/'capture'/axis,
                                          kernel.source_theta, kernel.theta, dense=args.capture,
                                          prefix=kernel.prefix, stop_at_block=args.stop_at_block,
                                          kernel=kernel, conv2_control=conv2_control)
                           if args.capture or args.stop_at_block else None)
                def replacement(self, x, k=kernel, m=mode):
                    sink = capture.decisions if capture and capture.dense_active else None
                    return k.network_output(x, m, args.chunk_groups, sink)
                neuron.forward = types.MethodType(replacement, neuron)
                rows, started = [], time.monotonic()
                try:
                    for i, name in enumerate(names):
                        functional.reset_net(model)
                        if capture:
                            capture.begin(i, name)
                        x, label_flow, mask = input_frame(args.data, name, targets=not args.stop_at_block)
                        try:
                            model(x)
                        except BlockReady:
                            assert args.stop_at_block
                            pred = error = None
                        except CoarseReady:
                            pred = F.interpolate(current.pop('flow'), (480, 640), mode='bilinear', align_corners=False)
                        else:
                            raise RuntimeError('expected the existing block/coarse completion hook')
                        if args.stop_at_block:
                            row = dict(file=name, AEE_evaluated=False, block_completed=True)
                        else:
                            error = torch.linalg.vector_norm(pred.permute(0, 2, 3, 1)[mask]
                                      - label_flow.permute(0, 2, 3, 1)[mask], dim=1)
                            total, pixels = float(error.double().sum()), error.numel()
                            row = dict(file=name, valid_pixels=pixels, aee_sum=total, AEE=total/pixels)
                        row.update(kernel.frame_counts)
                        rows.append(row)
                        if capture:
                            capture.finish()
                        complete = len(rows) == len(names)
                        summary = (dict(frames=len(rows), complete=complete, AEE_evaluated=False)
                                   if args.stop_at_block else summarize(rows, complete))
                        summary.update(model_file=str(path), prefix=kernel.prefix, mode=mode,
                                       theta_amplitude=kernel.theta,
                                       theta_decision=float(kernel.arrays['theta_decision']),
                                       numeric_checks=kernel.numeric_checks,
                                       conv2_integer_control=(conv2_control.description() if conv2_control else None),
                                       wall_seconds=time.monotonic()-started)
                        if i < 4 or (i+1) % 10 == 0 or complete:
                            save_json(args.output/(axis+'_frames.json'), rows)
                            save_json(args.output/(axis+'_summary.json'), summary)
                            print('INTEGER_PROGRESS', axis, i+1, '/', len(names),
                                  json.dumps(summary, ensure_ascii=False), flush=True)
                        del x, label_flow, mask, pred, error
                    run['results'][axis] = summary
                    save_json(args.output/'run.json', run)
                finally:
                    if capture:
                        capture.close()
        run['complete'] = True
        save_json(args.output/'run.json', run)
    finally:
        conv1.forward, norm1.forward, neuron.forward, conv2.forward, norm2.forward = original
    print('DONE', json.dumps(run['results'], ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model-files', type=Path, nargs='+', required=True)
    p.add_argument('--modes', nargs='+', choices=('exact', 'full', 'conditional'), default=['exact', 'conditional'])
    p.add_argument('--split', choices=('diverse', 'valid'), default='diverse')
    p.add_argument('--count', type=int, default=10, help='0 uses the whole selected split')
    p.add_argument('--output', type=Path)
    p.add_argument('--capture', action='store_true')
    p.add_argument('--quantize-conv2', action='store_true',
                   help='separate new-student control: fold theta and fixed BN2 into per-H dyadic W8 Conv2 for both structures')
    p.add_argument('--stop-at-block', action='store_true')
    p.add_argument('--stripe-rows', type=int, default=16)
    p.add_argument('--chunk-groups', type=int, default=256)
    p.add_argument('--local-check', action='store_true')
    p.add_argument('--source-directory', type=Path)
    p.add_argument('--conv2-parameters', type=Path,
                   help='only for --local-check --quantize-conv2: captured real W2/BN2 parameters.npz')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    args.modes = list(dict.fromkeys('exact' if mode == 'full' else mode for mode in args.modes))
    if args.stripe_rows < 1 or args.chunk_groups < 1 or args.count < 0:
        p.error('stripe-rows/chunk-groups must be positive; count must be nonnegative')
    if args.local_check:
        if args.source_directory is None:
            p.error('--source-directory is required for local captured-Zi checks')
        if args.quantize_conv2 and args.conv2_parameters is None:
            p.error('--conv2-parameters is required for the optional local Conv2 control check')
        local_check(args)
    else:
        if args.output is None:
            p.error('--output is required for real-network evaluation')
        run_network(args)


if __name__ == '__main__':
    main()
