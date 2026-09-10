"""Evaluate saved joint-completion students on the real FP-Conv1 network.

Only patch residual block 1's sn2 is replaced.  Its input remains the real
FP32 Conv1 -> train-calibrated norm1 output used by train_local.py.  All four
patch BNs use the same saved calibration; Conv2, BN2, shortcut and the existing
CoarseReady successor execute normally.  The S2 integer-source/class student
is installed by the existing nrv_cost_probe.load_system, without changing the
patch Conv1 into an integer convolution.

"exact" (alias "full") computes the saved student's full T10 neuron.  In
"conditional", only the saved prefix enters the predictor, then a complete
neuron result supplies fallback.  This is a dense numerical reference, not a
GPU implementation or timing claim for canceled convolution work.  A is Q14;
Y, bias, train moments and radius remain FP32.  theta is the output amplitude
and the saved neuron's decision threshold; it is not a production-mask cutoff.

The replacement's FP32 GEMMs disable TF32 locally to match CPU training.
The parent's Conv1/cuDNN and all other modules retain their runtime settings.
--local-check requires only CPU Torch and writes nothing.  It checks real
captured Y against train_local.Completion and the saved validation decisions,
including conversion between context/H8/P4/T and native T/N/C/H/W layouts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time
import types

sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
RES = 'sttmultires_unet.encoders.swin3d.patch_embed.residual_encoding.resblocks.'
BLOCK = RES + '1'
TARGET = BLOCK + '.sn2.spiking_neuron'


@contextmanager
def full_precision_matmul(device):
    """Scope the replacement's numeric choice; never change Conv1 settings."""
    if torch.device(device).type != 'cuda':
        yield
        return
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


class CompletionKernel:
    """Saved single-prefix predictor, with a complete same-student fallback."""

    def __init__(self, path, device='cpu'):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as z:
            arrays = {k: np.array(z[k], copy=True) for k in z.files}
        self.theta = float(arrays['theta_output'])
        self.source_theta = float(arrays['theta_source'])
        self.prefix = arrays['prefix'].astype(int).tolist()
        self.support_cpu = torch.from_numpy(arrays['support']).bool()
        self.a_cpu = torch.from_numpy(arrays['temporal_q14']).float()/16384
        self.b_cpu = torch.from_numpy(arrays['temporal_bias']).float().reshape(10)
        self.mean_cpu = torch.from_numpy(arrays['mean']).float()
        self.covariance_cpu = torch.from_numpy(arrays['covariance']).float()
        self.gamma_cpu = torch.from_numpy(arrays['gamma']).float().reshape(10)
        assert self.a_cpu.shape == (10, 10)
        assert self.mean_cpu.shape == (96, 10)
        assert self.covariance_cpu.shape == (96, 10, 10)
        assert not (self.a_cpu.ne(0) & ~self.support_cpu).any()
        self.known_cpu = torch.zeros(10)
        self.known_cpu[self.prefix] = 1
        remaining = self.a_cpu*(1-self.known_cpu)
        # Compile the saved CPU-trained constants once, with the exact same
        # contraction and clamping as train_local.Completion.forward.
        offset = self.mean_cpu @ remaining.T+self.b_cpu-self.theta
        variance = torch.einsum('ts,csu,tu->ct', remaining,
                               self.covariance_cpu, remaining)
        radius = variance.clamp_min(1e-6).sqrt()*self.gamma_cpu
        self.a = self.a_cpu.to(device)
        self.b = self.b_cpu.to(device)
        self.known = self.known_cpu.to(device)
        self.offset = offset.to(device)
        self.radius = radius.to(device)
        self.support = self.support_cpu.to(device)
        self.channel_groups = torch.arange(96, device=device).reshape(12, 8)

    def margins(self, y, channels):
        """Input: context,H8,P4,T; channel IDs: context,H8."""
        full = y @ self.a.T+self.b-self.theta
        predicted = ((y*self.known) @ self.a.T
                     + self.offset[channels][:, :, None, :])
        accepted = predicted.abs() >= self.radius[channels][:, :, None, :]
        return full, predicted, accepted

    def context_gates(self, y, channels, mode):
        if mode in ('exact', 'full'):
            return (y @ self.a.T+self.b-self.theta) >= 0
        full, predicted, accepted = self.margins(y, channels)
        return torch.where(accepted, predicted >= 0, full >= 0)

    @torch.no_grad()
    def network_output(self, x, mode, chunk_groups=256, decision_sink=None):
        """Evaluate native T/N/C/H/W by bounded groups; all T labels survive.

        A horizontal P4 group is never confused with four channels or four
        unrelated pixels.  Only the small current P4 chunk is transposed.
        The final theta*g output is required by the real Conv2 consumer.
        """
        t, batch, channels, height, width = x.shape
        assert (t, batch, channels) == (10, 1, 96) and width % 4 == 0
        assert x.dtype == torch.float32
        total_groups = height*(width//4)
        source = x[:, 0].reshape(10, 96, total_groups, 4)
        output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        destination = output[:, 0].reshape(10, 96, total_groups, 4)
        with full_precision_matmul(x.device):
            for first in range(0, total_groups, chunk_groups):
                last = min(first+chunk_groups, total_groups)
                n = last-first
                y = source[:, :, first:last, :].permute(2, 1, 3, 0)
                y = y.reshape(n*12, 8, 4, 10).contiguous()
                h = self.channel_groups.repeat(n, 1)
                if mode == 'conditional':
                    full, predicted, accepted = self.margins(y, h)
                    gate = torch.where(accepted, predicted >= 0, full >= 0)
                else:
                    gate = self.context_gates(y, h, mode)
                    accepted = None
                if decision_sink is not None:
                    if accepted is None:
                        need = torch.ones((n*12, 10), dtype=torch.bool, device=x.device)
                    else:
                        need = ((~accepted)[..., None] & self.support).any(-2).any((1, 2))
                        need = need | self.known.bool()
                    decision_sink(first, last, total_groups, need, accepted)
                payload = gate.to(x.dtype)*self.theta
                destination[:, :, first:last, :] = payload.reshape(
                    n, 96, 4, 10).permute(3, 1, 0, 2)
        return output

    def description(self):
        return dict(model_file=str(self.path), prefix=self.prefix,
                    theta_output=self.theta, theta_source=self.source_theta,
                    gamma=self.gamma_cpu.tolist(),
                    temporal_q14_nonzeros=int(self.a_cpu.ne(0).sum()),
                    numeric='FP32 Y, Aq/16384, bias and compiled predictor constants; no integer Conv1')


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


@torch.no_grad()
def local_check(args):
    """Real capture comparison, no synthetic network and no AEE claim."""
    sys.path.insert(0, str(HERE))
    from train_local import Completion
    patch = args.root/'algorithm/patch_probe'
    cap = torch.load(patch/'partial_completion/capture.pt', map_location='cpu',
                     weights_only=False)
    assert cap['metadata']['center_mode'] == 'zero'
    valid = [row for row in cap['samples'] if row['split'] == 'valid']
    torch.set_num_threads(args.threads)
    rows = []
    for model_file in args.model_files:
        kernel = CompletionKernel(model_file)
        reference = Completion(
            dict(weight=kernel.a_cpu, bias=kernel.b_cpu), kernel.mean_cpu,
            kernel.covariance_cpu, kernel.theta, prefix=kernel.prefix)
        reference.mask.copy_(kernel.support_cpu)
        reference.log_gamma.copy_(kernel.gamma_cpu.log())
        decisions_file = Path(model_file).with_name(Path(model_file).stem+'_valid_decisions.npz')
        decisions = None
        if decisions_file.exists():
            with np.load(decisions_file, allow_pickle=False) as z:
                decisions = {k: np.array(z[k], copy=True) for k in z.files}
        result = dict(model_file=str(model_file), prefix=kernel.prefix,
                      gates=0, context_vs_training_exact=0,
                      context_vs_training_conditional=0,
                      native_layout_exact=0, native_layout_conditional=0,
                      saved_gate_mismatches=0 if decisions else None,
                      saved_need_mismatches=0 if decisions else None,
                      prediction_suffix_change_max=0.)
        all_gates, all_need = [], []
        for row in valid:
            captured = row['Y'].float()  # T,C,native P4 group,within P4
            group_count = captured.shape[2]
            y = captured.permute(2, 1, 3, 0).reshape(-1, 8, 4, 10).contiguous()
            h = kernel.channel_groups.repeat(group_count, 1)
            full, pred, _, accepted = reference(y, h)
            expected_exact = full >= 0
            expected_conditional = torch.where(accepted, pred >= 0, full >= 0)
            got_exact = kernel.context_gates(y, h, 'exact')
            got_conditional = kernel.context_gates(y, h, 'conditional')
            result['gates'] += got_exact.numel()
            result['context_vs_training_exact'] += int((got_exact != expected_exact).sum())
            result['context_vs_training_conditional'] += int((got_conditional != expected_conditional).sum())
            # Arrange the real sampled P4 groups along one test image row.
            # This verifies tensor addressing only; it is not a conv replay.
            native = captured.reshape(10, 1, 96, 1, group_count*4)
            for mode, expected in [('exact', got_exact), ('conditional', got_conditional)]:
                native_result = kernel.network_output(native, mode, chunk_groups=7)
                native_gates = native_result[:, 0].reshape(10, 96, group_count, 4)
                native_gates = native_gates.permute(2, 1, 3, 0).reshape(-1, 8, 4, 10).ne(0)
                result['native_layout_'+mode] += int((native_gates != expected).sum())
                assert torch.equal(native_result,
                                   native_result.ne(0).to(native_result.dtype)*kernel.theta)
            _, _, accepted_direct = kernel.margins(y, h)
            need = ((~accepted_direct)[..., None] & kernel.support_cpu).any(-2)
            need = need | kernel.known_cpu.bool()
            all_need.append(need.any((1, 2)).numpy())
            all_gates.append(got_conditional.numpy())
            # The early predicate may not inspect an unproduced time column.
            _, p0, a0 = kernel.margins(y[:1], h[:1])
            changed = y[:1].clone()
            changed[..., ~kernel.known_cpu.bool()] = 123.25
            _, p1, a1 = kernel.margins(changed, h[:1])
            result['prediction_suffix_change_max'] = max(
                result['prediction_suffix_change_max'], float((p0-p1).abs().max()))
            assert torch.equal(a0, a1)
        if decisions is not None:
            result['saved_gate_mismatches'] = int(np.count_nonzero(
                np.concatenate(all_gates) != decisions['gate']))
            result['saved_need_mismatches'] = int(np.count_nonzero(
                np.concatenate(all_need) != decisions['need_column']))
        mismatch_fields = ('context_vs_training_exact', 'context_vs_training_conditional',
                           'native_layout_exact', 'native_layout_conditional',
                           'saved_gate_mismatches', 'saved_need_mismatches',
                           'prediction_suffix_change_max')
        result['passed'] = all(result[k] in (None, 0) for k in mismatch_fields)
        rows.append(result)
        print('LOCAL_CHECK', json.dumps(result, ensure_ascii=False), flush=True)
    report = dict(complete=all(r['passed'] for r in rows),
                  scope='CPU FP32 real capture only; no CUDA or real-network AEE run',
                  capture_parent=cap['metadata']['parent'],
                  capture_runtime=dict(tf32_matmul=cap['metadata']['tf32_matmul'],
                                       tf32_cudnn=cap['metadata']['tf32_cudnn']),
                  frames=[r['file'] for r in valid], models=rows)
    print('LOCAL_DONE', json.dumps(report, ensure_ascii=False), flush=True)
    if not report['complete']:
        raise RuntimeError('local neuron/capture comparison differs')
    return report


def packed_gate(x):
    """T,1,C,H,W -> flat [T,C,H,W] C-order little-endian bits.

    Only one time slice of Boolean data is copied to CPU at once.  Full
    FP32 identity/membrane tensors are never copied for these captures.
    """
    t, batch, c, h, w = x.shape
    assert batch == 1 and c % 8 == 0
    assert (c*h*w) % 8 == 0
    packed = np.empty((t, c*h*w//8), dtype=np.uint8)
    for tick in range(t):
        bits = x[tick, 0].detach().ne(0).cpu().numpy().reshape(-1)
        packed[tick] = np.packbits(bits, bitorder='little')
    return packed.reshape(-1)


def window_snapshot(x):
    """Small true border and interior windows, preserving all T/C entries."""
    h, w = x.shape[-2:]
    origins = [(0, 0), (0, max(w-5, 0)), (max(h-5, 0), 0),
               (max(h-5, 0), max(w-5, 0)), (h//2-2, w//2-2)]
    origins = [(max(y, 0), max(z, 0)) for y, z in origins]
    windows = torch.stack([x[:, 0, :, y:y+5, z:z+5].detach()
                           for y, z in origins]).cpu().numpy()
    return windows, np.asarray(origins, dtype=np.int32)


class BlockReady(Exception):
    """The requested real block output has completed; later layers not needed."""


class FrameCapture:
    def __init__(self, modules, output, source_theta, target_theta,
                 dense=False, sampled=None, stop_at_block=False, prefix=(),
                 reference_capture=None):
        self.output = Path(output)
        self.source_theta, self.target_theta = source_theta, target_theta
        self.dense = dense
        self.sampled = sampled
        self.stop_at_block = stop_at_block
        self.prefix = list(prefix)
        self.reference_capture = reference_capture
        self.y_check = None
        self.active = False
        self.dense_active = False
        self.record = {}
        self.sampled_record = {}
        self.accepted_full = None
        self.handles = [
            modules[BLOCK].register_forward_pre_hook(self.before_block),
            modules[BLOCK].register_forward_hook(self.after_block),
            modules[BLOCK+'.conv1.0'].register_forward_pre_hook(self.source),
            modules[BLOCK+'.conv1.0'].register_forward_hook(self.conv1),
            modules[TARGET].register_forward_pre_hook(self.norm1),
            modules[BLOCK+'.conv2.0'].register_forward_pre_hook(self.sn2),
            modules[BLOCK+'.conv2.0'].register_forward_hook(self.conv2),
        ]

    def before_block(self, module, inputs):
        if self.dense_active:
            self.record['identity'], self.record['window_origins_yx'] = window_snapshot(inputs[0])
            self.record['native_shape_TCHW'] = np.asarray(
                (inputs[0].shape[0], *inputs[0].shape[2:]), dtype=np.int32)

    def after_block(self, module, inputs, output):
        if self.dense_active:
            self.record['block_output'], _ = window_snapshot(output)
        if self.stop_at_block:
            raise BlockReady()

    def source(self, module, inputs):
        x = inputs[0]
        if self.dense_active:
            self.record['source_gate_bits'] = packed_gate(x)
            self.record['source_gate_shape'] = np.asarray((x.shape[0], *x.shape[2:]), dtype=np.int32)
        if self.sampled is not None:
            # Only the selected real 3x3 neighborhoods are materialized.  P4
            # members remain separate; zero padding is explicitly applied.
            h, w = x.shape[-2:]
            group_y = self.sampled['group_y'].to(x.device).long()
            group_x = self.sampled['group_x_start'].to(x.device).long()
            yy = group_y[:, None].expand(-1, 4)
            xx = group_x[:, None]+torch.arange(4, device=x.device)[None]
            bits = (1 << torch.arange(10, device=x.device, dtype=torch.int32))[:, None, None, None]
            words = torch.empty((len(group_y), 96, 3, 3, 4), dtype=torch.int16, device=x.device)
            for kh in range(3):
                for kw in range(3):
                    sy, sx = yy+kh-1, xx+kw-1
                    inside = (sy >= 0) & (sy < h) & (sx >= 0) & (sx < w)
                    live = x[:, 0, :, sy.clamp(0, h-1), sx.clamp(0, w-1)].ne(0)
                    live = live & inside[None, None]
                    packed = (live.to(torch.int32)*bits).sum(0, dtype=torch.int32)
                    words[:, :, kh, kw] = packed.permute(1, 0, 2).short()
            self.sampled_record = dict(
                source_gate_words=words.reshape(len(group_y), 864, 4).cpu().numpy(),
                group_ids=self.sampled['groups'].cpu().numpy(),
                positions=self.sampled['positions'].cpu().numpy(),
                group_y=self.sampled['group_y'].cpu().numpy(),
                group_x_start=self.sampled['group_x_start'].cpu().numpy())

    def conv1(self, module, inputs, output):
        if self.dense_active:
            self.record['conv1_raw'], _ = window_snapshot(output)

    def norm1(self, module, inputs):
        if self.dense_active:
            self.record['norm1_Y'], _ = window_snapshot(inputs[0])
        observed = None
        if self.sampled is not None:
            x = inputs[0]
            positions = self.sampled['positions'].to(x.device).long()
            observed = x[:, 0].flatten(2)[:, :, positions].detach().cpu()
            self.sampled_record['Y'] = observed.numpy()
        if self.index == 0 and self.reference_capture is not None:
            reference = self.reference_capture['rows'].get(self.name)
            if reference is not None:
                if observed is None:
                    x = inputs[0]
                    positions = self.reference_capture['positions'].to(x.device).long()
                    observed = x[:, 0].flatten(2)[:, :, positions].detach().cpu()
                error = observed-reference['Y']
                self.y_check = dict(file=self.name, elements=error.numel(),
                                    shape=list(observed.shape),
                                    max_abs=float(error.abs().max()),
                                    mean_abs=float(error.abs().mean()),
                                    differing_elements=int(error.ne(0).sum()),
                                    source='same 64 P4 groups as saved real FP norm1 capture')
                print('CAPTURE_Y_CHECK', json.dumps(self.y_check), flush=True)

    def sn2(self, module, inputs):
        x = inputs[0]
        if self.dense_active:
            self.record['output_gate_bits'] = packed_gate(x)
            self.record['output_gate_shape'] = np.asarray((x.shape[0], *x.shape[2:]), dtype=np.int32)
            gate_windows, _ = window_snapshot(x)
            self.record['sn2_gate'] = gate_windows != 0

    def conv2(self, module, inputs, output):
        if self.dense_active:
            self.record['conv2_raw'], _ = window_snapshot(output)

    def begin(self, index, name):
        self.dense_active = self.dense and index < 4
        self.active = self.dense_active or self.sampled is not None
        self.record = {}
        self.sampled_record = {}
        self.accepted_full = None
        self.y_check = None
        self.name = name
        self.index = index

    def decisions(self, first, last, total_groups, need, accepted):
        if 'need_columns' not in self.record:
            self.record['need_columns'] = np.empty((total_groups, 12, 10), dtype=np.bool_)
            self.record['accepted_gate_counts_H_T'] = np.zeros((96, 10), dtype=np.int64)
        self.record['need_columns'][first:last] = need.reshape(-1, 12, 10).cpu().numpy()
        if accepted is not None:
            counts = accepted.reshape(last-first, 12, 8, 4, 10).sum((0, 3))
            self.record['accepted_gate_counts_H_T'] += counts.reshape(96, 10).cpu().numpy()
            if first == 0:
                self.accepted_full = np.empty((10, 96, total_groups, 4), dtype=np.bool_)
            self.accepted_full[:, :, first:last] = accepted.reshape(
                last-first, 96, 4, 10).permute(3, 1, 0, 2).cpu().numpy()

    def finish(self):
        if not self.active:
            return
        directory = self.output/(f'{self.index:03d}_'+Path(self.name).stem)
        directory.mkdir(parents=True, exist_ok=True)
        if self.dense_active:
            gates = {k: self.record.pop(k) for k in
                     ('source_gate_bits', 'source_gate_shape', 'output_gate_bits', 'output_gate_shape')}
            h, w = gates['source_gate_shape'][-2:]
            gates['need_column_bits'] = np.packbits(self.record.pop('need_columns').reshape(-1), bitorder='little')
            gates['need_column_shape'] = np.asarray((h, w//4, 12, 10), dtype=np.int32)
            gates['accepted_gate_counts_H_T'] = self.record.pop('accepted_gate_counts_H_T')
            gates['prefix_acceptance_evaluated'] = np.asarray(self.accepted_full is not None)
            if self.accepted_full is not None:
                accepted_bits = np.packbits(self.accepted_full.reshape(-1), bitorder='little')
                gates['accepted_gate_bits'] = accepted_bits
                gates['accepted_gate_shape'] = gates['output_gate_shape']
                # Retain the integer capture's existing small-file interface.
                # This is the same predicate, not a second neuron evaluation.
                np.savez_compressed(directory/'accepted.npz', accepted_bits=accepted_bits,
                         accepted_shape=gates['output_gate_shape'], bitorder=np.asarray('little'),
                         layout=np.asarray('flat T,C,H,W C-order'), prefix=np.asarray(self.prefix),
                         prefix_acceptance_evaluated=np.asarray(True))
            np.savez(directory/'gates.npz', **gates,
                     theta_source=np.asarray(self.source_theta), theta_output=np.asarray(self.target_theta),
                     frame_name=np.asarray(self.name), bitorder=np.asarray('little'),
                     prefix=np.asarray(self.prefix, dtype=np.int32),
                     need_semantics=np.asarray('OR of unresolved output dependencies across actual P4/H8, plus prefix; exact mode is all T'),
                     layout=np.asarray('gate: flat T,C,H,W; need_column: H,W/4,12,T; C order'))
            np.savez(directory/'consumer_windows.npz', **self.record,
                     frame_name=np.asarray(self.name),
                     windows_layout=np.asarray('window,T,C,local_y,local_x'),
                     sn2_gate_semantics=np.asarray('Boolean gate; payload is theta_output*sn2_gate'),
                     identity_semantics=np.asarray('real MS_ResBlock input, added after norm2'),
                     output_semantics=np.asarray('real norm2(conv2(theta*g)) + identity'))
        if self.sampled is not None:
            np.savez(directory/'sampled_source.npz', **self.sampled_record,
                     frame_name=np.asarray(self.name), theta_source=np.asarray(self.source_theta),
                     order=np.asarray('group,k,p; k=((c*3)+kh)*3+kw; bit t; no OR across p'),
                     Y_layout=np.asarray('T,C,group,within-P4; same-forward FP32 norm1 output'),
                     padding=np.asarray('zero outside source H,W'))
        self.record.clear()
        self.sampled_record.clear()
        self.accepted_full = None
        self.active = False
        self.dense_active = False

    def close(self):
        for handle in self.handles:
            handle.remove()


def save_consumer(modules, fixed, output):
    arrays = {}
    for i in (1, 2):
        conv = modules[BLOCK+f'.conv{i}.0']
        arrays[f'W{i}'] = conv.weight.detach().cpu().numpy()
        arrays[f'conv{i}_bias'] = (conv.bias.detach().cpu().numpy() if conv.bias is not None
                                  else np.zeros(conv.out_channels, dtype=np.float32))
        for name in ('stride', 'padding', 'dilation', 'groups'):
            arrays[f'conv{i}_'+name] = np.asarray(getattr(conv, name))
        bn_names = [name for name in fixed if name.startswith(BLOCK+f'.norm{i}')]
        assert len(bn_names) == 1
        bn = modules[bn_names[0]]
        arrays[f'bn{i}_name'] = np.asarray(bn_names[0])
        arrays[f'bn{i}_eps'] = np.asarray(bn.eps)
        arrays[f'bn{i}_track_running_stats'] = np.asarray(bn.track_running_stats)
        for source, target in [('weight', 'gamma'), ('bias', 'beta'),
                               ('running_mean', 'mean'), ('running_var', 'var')]:
            arrays[f'bn{i}_'+target] = getattr(bn, source).detach().cpu().numpy()
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output/'parameters.npz', **arrays)


@torch.no_grad()
def evaluate_network(args):
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
    patch = algorithm/'patch_probe'
    calibration = patch/'patch_train_calibration.pt'
    fixed = torch.load(calibration, map_location='cpu', weights_only=False)
    for name, values in fixed.items():
        bn = modules[name]
        bn.track_running_stats = True
        bn.running_mean = values['mean'].to(bn.weight)
        bn.running_var = values['var'].to(bn.weight)
    neuron = modules[TARGET]
    assert neuron.center_mode == 'zero'
    assert neuron.threshold_mode == 'official_atlif' and neuron.output_mode == 'binary'
    source_theta = float(modules[BLOCK+'.sn1.spiking_neuron'].thresh)
    original_forward = neuron.forward
    sampled = None
    capture_metadata = None
    reference_capture = None
    if args.capture or args.capture_sampled_source or args.capture_train_source or args.split == 'train':
        old_capture = torch.load(patch/'partial_completion/capture.pt', map_location='cpu', weights_only=False)
        capture_metadata = old_capture['metadata']
        reference_capture = dict(positions=old_capture['positions'],
                                 rows={row['file']: dict(Y=row['Y']) for row in old_capture['samples']})
        if args.capture_sampled_source or args.capture_train_source:
            sampled = {key: old_capture[key] for key in ('groups', 'positions', 'group_y', 'group_x_start')}
        del old_capture
    if args.split == 'train':
        names = capture_metadata['train']
    elif args.split == 'diverse':
        names = json.loads((algorithm/'samples.json').read_text())['valid']
    else:
        names = read_names(args.data, 'valid')
    if args.count:
        names = names[:args.count]
    assert names
    args.output.mkdir(parents=True, exist_ok=True)
    run = dict(complete=False, split=args.split, files=names,
               model_files=[str(p) for p in args.model_files], modes=args.modes,
               parent='saved integer-S2-source/coarse student with four fixed patch BN; FP patch Conv1',
               target=TARGET, checkpoint=str(args.checkpoint), config=str(args.config),
               calibration=str(calibration), fixed_patch_BN=list(fixed),
               fp_conv1_dtype=str(modules[BLOCK+'.conv1.0'].weight.dtype),
               parent_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
               parent_tf32_cudnn=torch.backends.cudnn.allow_tf32,
               parent_cudnn_benchmark=torch.backends.cudnn.benchmark,
               replacement_tf32_matmul=False, chunk_groups=args.chunk_groups,
               numeric='Aq14/16384, real FP32 norm1 Y, floating bias/radius; dense same-student fallback',
               successor='real Conv2, fixed BN2, original shortcut, original downstream model to preds.2; bilinear 480x640',
               claim=('real block capture only; no network AEE' if args.stop_at_block else
                      'network AEE only; no physical Conv/PSN work is canceled by this reference'),
               stop_at_block=args.stop_at_block,
               appended_train_source=(capture_metadata['train'] if args.capture_train_source else None),
               sampled_source=(dict(frames=len(names), shape=[64, 864, 4],
                                    k='((c*3)+kh)*3+kw', time='bit t', across_p='no OR',
                                    positions='same real groups/positions as partial_completion/capture.pt')
                               if sampled is not None else None),
               capture=(dict(first_frames=min(4, len(names)), gates='complete sn1 and sn2; flat T,C,H,W little bitpack',
                             raw_bytes_each_at_240x320=9216000,
                             windows='five 5x5 border/interior windows of real identity and block output',
                             full_identity_saved=False) if args.capture else None),
               results={})
    save_json(args.output/'run.json', run)
    if args.capture or args.capture_sampled_source or args.capture_train_source:
        save_consumer(modules, fixed, args.output/'capture')
    device = modules[BLOCK+'.conv1.0'].weight.device
    try:
        for model_index, model_file in enumerate(args.model_files):
            kernel = CompletionKernel(model_file, device)
            assert kernel.theta == float(neuron.thresh)
            assert kernel.source_theta == source_theta
            label = f'{model_index:02d}_{Path(model_file).parent.name}_{Path(model_file).stem}'
            for mode in args.modes:
                axis = label+'_'+mode

                def replacement(self, x, k=kernel, current_mode=mode):
                    sink = capture.decisions if capture and capture.dense_active else None
                    return k.network_output(x, current_mode, args.chunk_groups, decision_sink=sink)

                neuron.forward = types.MethodType(replacement, neuron)
                capture = (FrameCapture(modules, args.output/'capture'/axis,
                                        source_theta, kernel.theta, dense=args.capture,
                                        sampled=sampled if args.capture_sampled_source else None,
                                        stop_at_block=args.stop_at_block,
                                        prefix=kernel.prefix, reference_capture=reference_capture)
                           if args.capture or args.capture_sampled_source or args.stop_at_block else None)
                rows, y_checks, started = [], [], time.monotonic()
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
                            pred = None
                        except CoarseReady:
                            pred = F.interpolate(current.pop('flow'), (480, 640),
                                                 mode='bilinear', align_corners=False)
                        else:
                            raise RuntimeError('expected the existing preds.2 CoarseReady hook')
                        if args.stop_at_block:
                            error = None
                            rows.append(dict(file=name, block_completed=True, AEE_evaluated=False))
                        else:
                            error = torch.linalg.vector_norm(pred.permute(0, 2, 3, 1)[mask]
                                      - label_flow.permute(0, 2, 3, 1)[mask], dim=1)
                            total, pixels = float(error.double().sum()), error.numel()
                            rows.append(dict(file=name, valid_pixels=pixels,
                                             aee_sum=total, AEE=total/pixels))
                        if capture:
                            capture.finish()
                            if capture.y_check is not None:
                                y_checks.append(capture.y_check)
                        complete = len(rows) == len(names)
                        summary = (dict(frames=len(rows), complete=complete, AEE_evaluated=False)
                                   if args.stop_at_block else summarize(rows, complete))
                        summary.update(axis=axis, mode=mode, **kernel.description(),
                                       wall_seconds=time.monotonic()-started,
                                       reference_norm1_Y_checks=y_checks)
                        if i < 4 or (i+1) % 10 == 0 or complete:
                            save_json(args.output/(axis+'_frames.json'), rows)
                            save_json(args.output/(axis+'_summary.json'), summary)
                            print('PROGRESS', axis, i+1, '/', len(names),
                                  json.dumps(summary, ensure_ascii=False), flush=True)
                        del x, label_flow, mask, pred, error
                    run['results'][axis] = summary
                    save_json(args.output/'run.json', run)
                finally:
                    if capture:
                        capture.close()
        if args.capture_train_source:
            # Reuse the loaded model.  The selected sn2 student is downstream
            # of this source and cannot change it.  Complete real r1 consumers,
            # then stop; no training-set AEE or parameter update is performed.
            capture = FrameCapture(modules, args.output/'capture'/'train_sampled_source',
                                   source_theta, kernel.theta, sampled=sampled,
                                   stop_at_block=True, prefix=kernel.prefix,
                                   reference_capture=reference_capture)
            train_rows = []
            try:
                for i, name in enumerate(capture_metadata['train']):
                    functional.reset_net(model)
                    capture.begin(i, name)
                    x, _, _ = input_frame(args.data, name, targets=False)
                    try:
                        model(x)
                    except BlockReady:
                        capture.finish()
                    else:
                        raise RuntimeError('train source capture expected real r1 output')
                    train_rows.append(dict(file=name, split='train', block_completed=True,
                                           reference_norm1_Y_check=capture.y_check))
                    print('TRAIN_SOURCE_CAPTURE', i+1, '/', len(capture_metadata['train']), name, flush=True)
                    del x
            finally:
                capture.close()
            run['train_source_capture'] = dict(complete=True, frames=train_rows,
                                               directory=str(args.output/'capture'/'train_sampled_source'))
            save_json(args.output/'train_source_frames.json', train_rows)
        run['complete'] = True
        save_json(args.output/'run.json', run)
    finally:
        neuron.forward = original_forward
    print('DONE', json.dumps(run['results'], ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model-files', type=Path, nargs='+', required=True)
    parser.add_argument('--modes', nargs='+', choices=('exact', 'full', 'conditional'),
                        default=['exact', 'conditional'])
    parser.add_argument('--split', choices=('diverse', 'valid', 'train'), default='diverse')
    parser.add_argument('--count', type=int, default=10,
                        help='0 evaluates every frame of the selected list; default diverse10')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--capture-sampled-source', action='store_true',
                        help='save each selected real P4 member source word at the original 64 capture groups')
    parser.add_argument('--capture-train-source', action='store_true',
                        help='after evaluation, reuse this model load to capture all saved train16 sources; stop at real r1 output')
    parser.add_argument('--stop-at-block', action='store_true',
                        help='stop after real r1 Conv2/BN2/shortcut, write no AEE; automatic for train split')
    parser.add_argument('--chunk-groups', type=int, default=256,
                        help='native horizontal P4 groups per replacement call chunk')
    parser.add_argument('--local-check', action='store_true',
                        help='CPU real-capture function/layout comparison only, no output files')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.split == 'train':
        args.stop_at_block = True
    args.modes = list(dict.fromkeys('exact' if m == 'full' else m for m in args.modes))
    if args.count < 0 or args.chunk_groups < 1:
        parser.error('count must be nonnegative and chunk-groups must be positive')
    if args.local_check:
        local_check(args)
    else:
        if args.output is None:
            parser.error('--output is required for network evaluation')
        evaluate_network(args)


if __name__ == '__main__':
    main()
