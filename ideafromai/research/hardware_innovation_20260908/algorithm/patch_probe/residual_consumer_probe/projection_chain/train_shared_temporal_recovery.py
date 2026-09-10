"""Fixed64-step real-GT recovery of independent versus shared temporal A.

Common control3: saved preview Conv1 U8/VQ5, ordinary Conv2R16, nonanchor
whole BN2-branch deletion, and ordinary PEDR32. Both axes train the two Conv2
factors, two PED factors, r1 BN2 beta, PED output bias, and both neuron biases.
Independent A has two100-coefficient matrices. Shared A has one100-coefficient
source matrix and10 signed slopes: proj.A[t] = d[t]*source.A[P[t]] on EVERY
forward. P is the exact train4-only one-to-one scalar-regression assignment;
output time labels never move. No validation fitting or permutation sweep.

Only the frozen preview's sn2 backward is replaced by the existing triangular
surrogate, identically in both axes. The two trained native ATLIFs retain their
own official surrogate, theta and center. No threshold-update installer runs.
The preview remains frozen but propagates gradients to r1.sn1.

Root launches CUDA. --check-only compares native initialized controls against
the training forwards on the first TRAIN frame, then checks gradients without
an optimizer update. --self-check performs a small CPU parameterization check
and the fixed assignment calculation using the existing train4 moments.
Exactly64 Adam1e-4 updates, seed912 train16 permutations, robust real-flow EPE;
initial/final diverse10 are reported, never used to choose the checkpoint.
These remain two separately executed PSNs; this is not a membrane-reuse RTL.
--evaluate-saved DIR reloads both final NPZ students in the identical training
parameter layout/forward and evaluates --split diverse/valid --count N only.
It performs no assignment fitting, gradient check or optimizer update.
--extend-saved DIR starts from each saved64-step endpoint, fixes its P, selects
128 representative train frames across the sorted actual18 sequences and runs
two seed912 permutations (256 updates) with a fresh Adam1e-4 per axis. All
parameter/freeze/numeric rules are the original training rules, not coordinates.
"""
from __future__ import annotations

import argparse
import gc
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import parametrize

HERE = Path(__file__).resolve().parent
RES = HERE.parent
LATENT = RES.parent/'factor_completion_20260909/latent_stage_train16'
sys.path.insert(0, str(LATENT))
sys.path.insert(0, str(RES))
from flow_backward_probe import TrainableLatentPair, gradient_stats, read_arrays
from adapter import LatentPair, fp32_matmul
from capture import BLOCK, SOURCE_SN, PROJECT, CONSUMER_SN, neuron_parameters
from capture_chain import save_json
from evaluate_rank_control import RankConvControl
from evaluate_shared_temporal_control import frame_comparison
from train_consumer_recovery import read_train16, release_model_graphs

AXES = ('ordinary_independent_A', 'shared_assigned_affine_A')
RAW_AXES = ('identity_permuted_base', 'identity_permuted_joint_r2')
STEPS, LR, SEED = 64, 1e-4, 912


def assignment_fit(arrays):
    """1024-state exact assignment of source rows to labeled output rows.

    Cost is centered scalar-regression MSE on the saved full-domain train4
    moments. Zero-variance regressors are constants. No ridge or AEE search.
    """
    mean, cov = arrays['mean_x'], arrays['covariance_x']
    source = arrays['source_A'].astype(np.float64)
    original = arrays['original_proj_A'].astype(np.float64)
    offset = arrays['original_proj_bias'].reshape(10)-arrays['effective_proj_center']
    mu_m, mu_h = source@mean, original@mean+offset
    vm = np.maximum(np.einsum('ti,ij,tj->t', source, cov, source), 0)
    vh = np.maximum(np.einsum('ti,ij,tj->t', original, cov, original), 0)
    cross = original@cov@source.T  # [labeled consumer t, source row j]
    slopes = np.divide(cross, vm[None], out=np.zeros_like(cross), where=vm[None] > 0)
    cost = np.maximum(vh[:, None]+slopes*slopes*vm[None]-2*slopes*cross, 0)
    best = np.full(1 << 10, np.inf)
    previous = np.full(1 << 10, -1, np.int16)
    choice = np.full(1 << 10, -1, np.int16)
    best[0] = 0
    for mask in range((1 << 10)-1):
        t = mask.bit_count()
        for j in range(10):
            if mask & (1 << j):
                continue
            nxt = mask | (1 << j)
            value = best[mask]+cost[t, j]
            if value < best[nxt]:
                best[nxt], previous[nxt], choice[nxt] = value, mask, j
    permutation = np.empty(10, np.int64)
    mask = (1 << 10)-1
    for t in range(9, -1, -1):
        permutation[t] = choice[mask]
        mask = int(previous[mask])
    d = slopes[np.arange(10), permutation]
    intercept = mu_h-d*mu_m[permutation]
    denominator = np.sqrt(vh[:, None]*vm[None])
    correlations = np.divide(cross, denominator, out=np.zeros_like(cross), where=denominator > 0)
    return dict(permutation=permutation, d=d, intercept=intercept,
        native_A=(d[:, None]*source[permutation]).astype(np.float32),
        native_bias=(intercept+arrays['effective_proj_center']).astype(np.float32).reshape(10, 1),
        assignment_cost_matrix=cost, correlation_matrix=correlations,
        selected_correlation=correlations[np.arange(10), permutation],
        selected_residual_variance=cost[np.arange(10), permutation],
        total_residual_variance=float(best[-1]),
        same_row_total_residual_variance=float(np.trace(cost)),
        teacher_total_variance=float(vh.sum()), zero_variance_source_rows=vm == 0,
        scope='Static one-to-one source-row assignment from existing train4 identity moments; no output-label permutation and no validation choice.')


class RowAffine(nn.Module):
    """An external source Parameter is referenced, never copied/detached."""
    def __init__(self, source_weight, slopes, permutation):
        super().__init__()
        # Do not register the same source Parameter again under proj.sn.
        object.__setattr__(self, 'source_weight', source_weight)
        self.d = nn.Parameter(torch.as_tensor(slopes, device=source_weight.device, dtype=torch.float64).clone())
        self.register_buffer('permutation', torch.as_tensor(permutation, device=source_weight.device, dtype=torch.long))

    def forward(self, unused_original):
        selected = self.source_weight.index_select(0, self.permutation)
        # Same Float64 multiply -> Float32 rounding as the closed-form initial A.
        return (self.d[:, None]*selected.double()).to(selected.dtype)


def raw_initializations(arrays):
    """Read the two already fitted structures; no new fitting or P selection."""
    return {axis: dict(raw_e=arrays[axis+'_slope'],
        raw_permutation=arrays[axis+'_permutation'],
        raw_L2=arrays[axis+'_residual_left'], raw_R2=arrays[axis+'_residual_right'],
        consumer_bias=arrays[axis+'_bias'].astype(np.float32).reshape(10, 1),
        saved_dense_A=arrays[axis+'_A'].astype(np.float32)) for axis in RAW_AXES}


class RawTemporal(nn.Module):
    """Ap=diag(e)P(+L2R2) in raw input coordinates, independent of source A.

    Train/store the fitted factors in Float64; assemble in Float64 and round
    the whole matrix to the native weight dtype on each forward. The unused
    original dense weight stays frozen and is not an optimization parameter.
    """
    def __init__(self, original_weight, initial, residual):
        super().__init__()
        device = original_weight.device
        self.e = nn.Parameter(torch.as_tensor(initial['raw_e'], device=device, dtype=torch.float64).clone())
        self.register_buffer('permutation', torch.as_tensor(initial['raw_permutation'], device=device, dtype=torch.long).clone())
        self.register_buffer('permutation_matrix', torch.eye(10, device=device, dtype=torch.float64)[self.permutation])
        self.residual = residual
        if residual:
            self.left = nn.Parameter(torch.as_tensor(initial['raw_L2'], device=device, dtype=torch.float64).clone())
            self.right = nn.Parameter(torch.as_tensor(initial['raw_R2'], device=device, dtype=torch.float64).clone())

    def dense_double(self):
        matrix = self.e[:, None]*self.permutation_matrix
        return matrix+self.left@self.right if self.residual else matrix

    def forward(self, unused_original):
        return self.dense_double().to(unused_original.dtype)


def clone_strided_parameter(value):
    """Retain the old R16 view's physical stride (second factor has stride32)."""
    out = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device=value.device)
    with torch.no_grad():
        out.copy_(value)
    return nn.Parameter(out)


class SharedTemporalControl:
    def __init__(self, modules, conv_arrays, ped_arrays, fit, raw_initial=None):
        self.source, self.consumer = modules[SOURCE_SN], modules[CONSUMER_SN]
        self.conv, self.projection = modules[BLOCK+'.conv2.0'], modules[PROJECT+'.conv_res']
        self.bn = modules[BLOCK+'.norm2.norm_layer']
        self.rank = RankConvControl(self.conv, self.projection, conv_arrays)
        self.rank.axis = 'uniform_rank16'
        self.original_projection_forward = self.projection.forward
        self.fit = fit
        self.raw_initial = raw_initial or {}
        device, dtype = self.projection.weight.device, self.projection.weight.dtype
        self.ped_u = torch.as_tensor(ped_arrays['U'], device=device, dtype=dtype)[:, :, None, None]
        self.ped_v = torch.as_tensor(ped_arrays['V'], device=device, dtype=dtype)[:, :, None, None]
        self.original = dict(source_A=self.source.weight, source_bias=self.source.bias,
            consumer_A=self.consumer.weight, consumer_bias=self.consumer.bias,
            bn_bias=self.bn.bias, projection_bias=self.projection.bias)
        self.source_state, self.consumer_state = neuron_parameters(self.source), neuron_parameters(self.consumer)
        self.terms = {}  # Same release helper as the preceding GT recovery.
        self.shared_parameterization = None
        self.raw_parameterization = None
        self.params = {}

    def restore_native_parameters(self):
        if parametrize.is_parametrized(self.consumer, 'weight'):
            parametrize.remove_parametrizations(self.consumer, 'weight', leave_parametrized=False)
        self.source.weight, self.source.bias = self.original['source_A'], self.original['source_bias']
        self.consumer.weight, self.consumer.bias = self.original['consumer_A'], self.original['consumer_bias']
        self.bn.bias, self.projection.bias = self.original['bn_bias'], self.original['projection_bias']
        self.shared_parameterization = None
        self.raw_parameterization = None

    def reference(self, axis):
        """Old operation/layout path; shared uses its new fixed-P native A."""
        self.restore_native_parameters()
        if axis == AXES[1]:
            self.consumer.weight = nn.Parameter(torch.as_tensor(self.fit['native_A'],
                device=self.consumer.weight.device).clone(), requires_grad=False)
            self.consumer.bias = nn.Parameter(torch.as_tensor(self.fit['native_bias'],
                device=self.consumer.bias.device).clone(), requires_grad=False)
        elif axis in RAW_AXES:
            initial = self.raw_initial[axis]
            mapping = RawTemporal(self.consumer.weight, initial, axis == RAW_AXES[1])
            with torch.no_grad():
                actual = mapping(self.consumer.weight)
            self.consumer.weight = nn.Parameter(actual.clone(), requires_grad=False)
            self.consumer.bias = nn.Parameter(torch.as_tensor(initial['consumer_bias'],
                device=self.consumer.bias.device).clone(), requires_grad=False)
        self.conv.forward = self.rank.forward
        self.projection.forward = lambda x: F.conv2d(F.conv2d(x[:, :, ::2, ::2], self.ped_u),
                                                        self.ped_v, self.original['projection_bias'])

    def trainable(self, axis):
        self.axis = axis
        self.restore_native_parameters()
        self.conv_u = clone_strided_parameter(self.rank.first[:16])
        self.conv_v = clone_strided_parameter(self.rank.second[:, :16])
        self.u, self.v = clone_strided_parameter(self.ped_u), clone_strided_parameter(self.ped_v)
        self.source.weight = clone_strided_parameter(self.original['source_A'])
        self.source.bias = clone_strided_parameter(self.original['source_bias'])
        self.consumer.bias = clone_strided_parameter(self.original['consumer_bias'])
        self.bn.bias = clone_strided_parameter(self.original['bn_bias'])
        self.projection_bias_delta = nn.Parameter(torch.zeros(96, device=self.v.device, dtype=self.v.dtype))
        self.params = dict(conv2_U_R16=self.conv_u, conv2_V_R16=self.conv_v,
            ped_U_R32=self.u, ped_V_R32=self.v, r1_BN2_bias=self.bn.bias,
            ped_output_bias_delta=self.projection_bias_delta,
            source_A=self.source.weight, source_bias=self.source.bias, consumer_bias=self.consumer.bias)
        if axis == AXES[0]:
            self.consumer.weight = clone_strided_parameter(self.original['consumer_A'])
            self.params['consumer_A'] = self.consumer.weight
        elif axis == AXES[1]:
            with torch.no_grad():
                self.consumer.bias.copy_(torch.as_tensor(self.fit['native_bias'], device=self.consumer.bias.device))
            self.consumer.weight = self.original['consumer_A']
            mapping = RowAffine(self.source.weight, self.fit['d'], self.fit['permutation'])
            parametrize.register_parametrization(self.consumer, 'weight', mapping)
            self.shared_parameterization = mapping
            self.params['shared_d'] = mapping.d
        elif axis in RAW_AXES:
            initial = self.raw_initial[axis]
            with torch.no_grad():
                self.consumer.bias.copy_(torch.as_tensor(initial['consumer_bias'], device=self.consumer.bias.device))
            self.consumer.weight = self.original['consumer_A']
            self.consumer.weight.requires_grad_(False)
            mapping = RawTemporal(self.consumer.weight, initial, axis == RAW_AXES[1])
            parametrize.register_parametrization(self.consumer, 'weight', mapping)
            self.raw_parameterization = mapping
            self.params['raw_e'] = mapping.e
            if mapping.residual:
                self.params.update(raw_L2=mapping.left, raw_R2=mapping.right)
        else:
            raise ValueError('Unknown temporal parameterization: '+axis)
        self.conv.forward, self.projection.forward = self.conv_forward, self.projection_forward

    def conv_forward(self, x):
        z = F.conv2d(x.flatten(0, 1), self.conv_u, None, self.rank.stride,
                     self.rank.padding, self.rank.dilation)
        out = F.conv2d(z, self.conv_v, self.rank.bias)
        return out.reshape(x.shape[0], x.shape[1], 96, *out.shape[-2:])

    def projection_forward(self, x):
        out = F.conv2d(F.conv2d(x[:, :, ::2, ::2], self.u), self.v, self.original['projection_bias'])
        return out+self.projection_bias_delta[None, :, None, None]

    def gradients(self):
        return {name: gradient_stats(parameter) for name, parameter in self.params.items()}

    @torch.no_grad()
    def load_saved(self, axis, arrays):
        if axis == AXES[1]:
            self.fit = dict(native_bias=arrays['consumer_bias'], d=arrays['shared_d'],
                            permutation=arrays['shared_permutation'])
        elif axis in RAW_AXES:
            self.raw_initial[axis] = dict(raw_e=arrays['raw_e'],
                raw_permutation=arrays['raw_permutation'], raw_L2=arrays['raw_L2'],
                raw_R2=arrays['raw_R2'], consumer_bias=arrays['consumer_bias'],
                saved_dense_A=arrays['consumer_A'])
        self.trainable(axis)
        fields = dict(r1_BN2_bias='bn2_beta', ped_output_bias_delta='projection_bias_delta')
        for name, parameter in self.params.items():
            value = arrays[fields.get(name, name)]
            parameter.copy_(torch.as_tensor(value, device=parameter.device,
                                           dtype=parameter.dtype).reshape(parameter.shape))
        # The frozen BN gain/stats, theta/center, Conv2 bias and PED base bias
        # remain those of the same parent. Only the exported bias delta trains.
        # A live shared A must be reconstructed from sourceA/d/P, not copied
        # into a second independent matrix.
        self.saved_consumer_A_max_abs = float((self.consumer.weight-
            torch.as_tensor(arrays['consumer_A'], device=self.consumer.weight.device)).abs().max())

    def constraints(self):
        with torch.no_grad():
            result = dict(source_rank=int(torch.linalg.matrix_rank(self.source.weight.double())),
                consumer_rank=int(torch.linalg.matrix_rank(self.consumer.weight.double())),
                conv2_factor_strides=[list(self.conv_u.stride()), list(self.conv_v.stride())],
                source_theta=float(self.source.thresh), consumer_theta=float(self.consumer.thresh))
            if self.shared_parameterization is not None:
                p = self.shared_parameterization
                expected = (p.d[:, None]*self.source.weight[p.permutation].double()).float()
                result.update(shared_relation_max_abs=float((self.consumer.weight-expected).abs().max()),
                    permutation=p.permutation.cpu().numpy(), d=p.d.cpu().numpy())
            if self.raw_parameterization is not None:
                p = self.raw_parameterization
                result.update(raw_relation_max_abs=float((self.consumer.weight-p.dense_double().float()).abs().max()),
                    raw_permutation=p.permutation.cpu().numpy(), raw_e=p.e.cpu().numpy(),
                    raw_residual_factor_rank=2 if p.residual else 0,
                    raw_dense_weight_is_frozen=not self.consumer.parametrizations.weight.original.requires_grad,
                    raw_factors_dtype=str(p.e.dtype), source_A_is_independent=True,
                    difference_from_loaded_dense_A=float((self.consumer.weight-
                        torch.as_tensor(self.raw_initial[self.axis]['saved_dense_A'],
                                        device=self.consumer.weight.device)).abs().max()))
            return result

    def export(self, parent, schedule, prior_steps=0):
        def cpu(value):
            return value.detach().cpu().numpy()
        bias = self.original['projection_bias']
        arrays = dict(axis=np.array(self.axis), parent=np.array(str(parent)),
            conv2_U_R16=cpu(self.conv_u), conv2_V_R16=cpu(self.conv_v),
            conv2_has_bias=np.array(self.rank.bias is not None),
            conv2_bias=cpu(self.rank.bias) if self.rank.bias is not None else np.zeros(96, np.float32),
            ped_U_R32=cpu(self.u[:, :, 0, 0]), ped_V_R32=cpu(self.v[:, :, 0, 0]),
            projection_base_has_bias=np.array(bias is not None),
            projection_base_bias=cpu(bias) if bias is not None else np.zeros(96, np.float32),
            projection_bias_delta=cpu(self.projection_bias_delta),
            source_A=cpu(self.source.weight), consumer_A=cpu(self.consumer.weight),
            source_bias=cpu(self.source.bias), consumer_bias=cpu(self.consumer.bias),
            source_theta=cpu(self.source.thresh), consumer_theta=cpu(self.consumer.thresh),
            source_center=cpu(self.source.center), consumer_center=cpu(self.consumer.center),
            source_center_mode=np.array(self.source.center_mode),
            consumer_center_mode=np.array(self.consumer.center_mode),
            source_output_mode=np.array(self.source.output_mode), consumer_output_mode=np.array(self.consumer.output_mode),
            source_threshold_mode=np.array(self.source.threshold_mode), consumer_threshold_mode=np.array(self.consumer.threshold_mode),
            bn2_gamma=cpu(self.bn.weight), bn2_beta=cpu(self.bn.bias), bn2_mean=cpu(self.bn.running_mean),
            bn2_var=cpu(self.bn.running_var), bn2_eps=np.array(self.bn.eps),
            train_schedule=np.array(schedule), seed=np.array(SEED), steps=np.array(len(schedule)),
            prior_steps=np.array(prior_steps), cumulative_steps=np.array(prior_steps+len(schedule)),
            nonanchor_whole_bn_branch_deleted=np.array(True),
            factor_numeric=np.array('Actual FP32 trained factors; original preview U8/VQ5 stays frozen. Conv2 secondR16 has saved runtime stride32; two independent native PSNs still execute.'),
            definition=np.array('Shared consumer A is roundFP32(d64[:,None]*sourceA32[P]); d/P address source rows, not output labels. Native output remains theta*g; center/threshold amplitude fixed.'))
        if self.shared_parameterization is not None:
            arrays.update(shared_d=cpu(self.shared_parameterization.d),
                          shared_permutation=cpu(self.shared_parameterization.permutation))
        else:
            arrays.update(shared_d=np.empty(0, np.float64), shared_permutation=np.empty(0, np.int64))
        if self.raw_parameterization is not None:
            p = self.raw_parameterization
            arrays.update(raw_e=cpu(p.e), raw_permutation=cpu(p.permutation),
                raw_L2=cpu(p.left) if p.residual else np.empty((10, 0), np.float64),
                raw_R2=cpu(p.right) if p.residual else np.empty((0, 10), np.float64),
                raw_residual_rank=np.array(2 if p.residual else 0),
                definition=np.array('Independent raw-coordinate consumer: roundFP32(diag(e64)*P [+ L2_64@R2_64]); fixed P, sourceA independently trains only its source neuron. No trainable dense consumer A. Native theta*g, center and theta frozen.'))
        return arrays

    def restore(self):
        self.restore_native_parameters()
        self.conv.forward, self.projection.forward = self.rank.original_forward, self.original_projection_forward


def self_check(affine_path):
    """Real train4 assignment plus small live-parameter/stride gradient checks."""
    torch.set_num_threads(2)
    arrays = read_arrays(affine_path)
    fit = assignment_fit(arrays)
    source = nn.Parameter(torch.from_numpy(arrays['source_A'].astype(np.float32)))
    consumer = nn.Linear(10, 10, bias=False)
    consumer.weight.requires_grad_(False)
    mapping = RowAffine(source, fit['d'], fit['permutation'])
    parametrize.register_parametrization(consumer, 'weight', mapping)
    initial_error = float((consumer.weight.detach()-torch.from_numpy(fit['native_A'])).abs().max())
    rng = torch.Generator().manual_seed(SEED)
    x = torch.randn(10, 23, generator=rng)
    desired = torch.randn(10, 23, generator=rng)
    # Both the source's own use and the consumer link contribute real gradients.
    loss = (source@x-desired).square().mean()+(consumer.weight@x+desired).square().mean()
    loss.backward()
    grads = dict(source=gradient_stats(source), d=gradient_stats(mapping.d))
    optimizer = torch.optim.Adam([source, mapping.d], lr=LR)
    optimizer.step()
    expected = (mapping.d[:, None]*source[mapping.permutation].double()).float()
    relation_error = float((consumer.weight-expected).detach().abs().max())
    base = torch.randn(96, 32, 1, 1, generator=rng)
    copied = clone_strided_parameter(base[:, :16])
    assert initial_error == 0 and relation_error == 0 and copied.stride() == base[:, :16].stride()
    assert all(g['all_finite'] and g['nonzero'] for g in grads.values())
    # An independent exhaustive assignment check on a 4x4 subproblem.
    import itertools
    small = fit['assignment_cost_matrix'][:4, :4]
    exhaustive = min(sum(small[t, p[t]] for t in range(4)) for p in itertools.permutations(range(4)))
    dp = {0: 0.}
    for mask in range(15):
        for j in range(4):
            if not mask & (1 << j):
                nxt = mask | (1 << j)
                dp[nxt] = min(dp.get(nxt, np.inf), dp[mask]+small[mask.bit_count(), j])
    assert abs(dp[15]-exhaustive) < 1e-12
    result = dict(CPU='PASS', real_train4_permutation=fit['permutation'],
        train4_residual_variance_sum=fit['total_residual_variance'],
        previous_same_row_sum=fit['same_row_total_residual_variance'],
        teacher_variance_sum=fit['teacher_total_variance'], native_A_initial_max_abs=initial_error,
        after_update_shared_relation_max_abs=relation_error, gradients=grads,
        preserved_R16_second_stride=list(copied.stride()),
        scope='No real network forward, GT recovery or AEE has run here; root must run --check-only.')
    print('SHARED_TEMPORAL_CPU', json.dumps(result, default=lambda v: v.tolist()), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--train-list', type=Path)
    parser.add_argument('--affine-parameters', type=Path)
    parser.add_argument('--raw-structured-parameters', type=Path,
                        help='Use the two saved raw-coordinate fixed-P diag / diag+R2 controls; source A is independent. End-of-stage diverse10 only.')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--evaluate-saved', type=Path)
    parser.add_argument('--extend-saved', type=Path,
                        help='Fixed128-frame/256-update extension from the saved64-step students; new optimizer, fixed P.')
    parser.add_argument('--extension-plan', type=Path,
                        help='Reuse an already fixed train128.json list AND schedule verbatim in the256-update extension.')
    parser.add_argument('--coordinate-mode', choices=('native', 'q_ui', 'single_q', 'raw_factorized', 'fixed_temporal'),
                        help='Optional new coordinate-function evaluation; only with --evaluate-saved.')
    parser.add_argument('--coordinate-fp32-channel', action='store_true',
                        help='Only with coordinate-mode: disable cuDNN TF32 for every convolution inside that helper, in all three modes.')
    parser.add_argument('--projection-branch-off', action='store_true',
                        help='Evaluation only: zero the complete PED spike branch after its BN (or conv when no BN), including offset.')
    parser.add_argument('--count-sources', action='store_true',
                        help='Evaluation only: also count actual three-source halo occurrences and T10/P2 logical NRV rows; use a separate --output.')
    parser.add_argument('--preview-time-first', action='store_true',
                        help='Evaluation only: ordinary fixed-BN compilation moves the preview PSN before V32; use a separate --output and new AEE.')
    parser.add_argument('--preview-gate-fixed', action='store_true',
                        help='Evaluation only: signed24/f14 AZ, exact integer dyadic V and compiled theta gate; capture fixed spatial samples for strict digit certificates.')
    parser.add_argument('--preview-az-frac', type=int, default=14,
                        help='Fixed preview AZ fractional bits; 6 is the predeclared unconditional coarse-precision control, a different numeric function.')
    parser.add_argument('--capture-consumers', action='store_true',
                        help='Coordinate evaluation only: save64 fixed T10/P2 groups of actual consumer inputs and outputs per frame.')
    parser.add_argument('--split', choices=('diverse', 'valid', 'train'), default='diverse',
                        help='Evaluation-only split; training stays fixed diverse10.')
    parser.add_argument('--count', type=int, default=10,
                        help='Evaluation-only frame count; training stays fixed diverse10.')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args(argv)
    if args.self_check:
        self_check(args.affine_parameters or HERE/'affine_shared_temporal_control_diverse10/parameters.npz')
        return
    if args.root is None:
        parser.error('--root is required for the real network check/recovery')
    if args.evaluate_saved is not None and args.check_only:
        parser.error('--evaluate-saved does not run --check-only')
    if args.extend_saved is not None and (args.evaluate_saved is not None or args.check_only):
        parser.error('--extend-saved is the fixed256-update extension, separate from evaluate/check modes.')
    if args.extension_plan is not None and args.extend_saved is None:
        parser.error('--extension-plan requires --extend-saved.')
    if args.coordinate_mode is not None and args.evaluate_saved is None:
        parser.error('--coordinate-mode is evaluation-only; the training function is unchanged.')
    if args.coordinate_fp32_channel and args.coordinate_mode is None:
        parser.error('--coordinate-fp32-channel requires --coordinate-mode.')
    if args.projection_branch_off and (args.evaluate_saved is None or args.coordinate_mode is not None):
        parser.error('--projection-branch-off requires --evaluate-saved and excludes --coordinate-mode.')
    if args.count_sources and args.evaluate_saved is None:
        parser.error('--count-sources is evaluation-only.')
    if args.preview_time_first and args.evaluate_saved is None:
        parser.error('--preview-time-first is evaluation-only.')
    if args.preview_gate_fixed and (args.evaluate_saved is None or args.preview_time_first):
        parser.error('--preview-gate-fixed is a separate evaluation function; do not combine with --preview-time-first.')
    if args.capture_consumers and (args.evaluate_saved is None or args.coordinate_mode is None):
        parser.error('--capture-consumers needs a saved student and a coordinate function.')
    if args.evaluate_saved is None and (args.split != 'diverse' or args.count != 10):
        parser.error('Training/check mode has fixed diverse10; --split/--count are evaluation-only.')
    alg, area = args.root/'algorithm', args.root/'algorithm/patch_probe'
    residual, latent = area/'residual_consumer_probe', area/'factor_completion_20260909/latent_stage_train16'
    chain = residual/'projection_chain'
    args.train_list = args.train_list or latent/'flow_train_list.json'
    args.affine_parameters = args.affine_parameters or chain/'affine_shared_temporal_control_diverse10/parameters.npz'
    saved_directory = args.evaluate_saved or args.extend_saved
    saved_run = None if saved_directory is None else json.loads((saved_directory/'result.json').read_text())
    raw_mode = args.raw_structured_parameters is not None or (saved_run or {}).get('axis_family') == 'raw_structured'
    axes = RAW_AXES if raw_mode else AXES
    if raw_mode:
        args.raw_structured_parameters = args.raw_structured_parameters or chain/'shared_temporal_lowrank_control.npz'
        if args.coordinate_mode not in (None, 'native', 'raw_factorized', 'fixed_temporal'):
            parser.error('Raw-coordinate controls support native, raw_factorized or fixed_temporal execution.')
        if args.extend_saved is not None and args.extension_plan is None:
            parser.error('The raw controls must reuse the saved shared train128.json through --extension-plan.')
    elif args.coordinate_mode == 'raw_factorized':
        parser.error('raw_factorized requires the raw-structured saved students.')
    args.output = args.output or (args.evaluate_saved.with_name(
        args.evaluate_saved.name+('_projection_branch_off_' if args.projection_branch_off else
            '_preview_gate_fixed_' if args.preview_gate_fixed else
            '_reload_' if args.coordinate_mode is None else '_coordinates_'+args.coordinate_mode+
            ('_fp32_channel_' if args.coordinate_fp32_channel else '_'))
        +args.split+str(args.count)) if args.evaluate_saved is not None else
        (chain/'temporal_structured_recovery'/('stage128x256' if args.extend_saved is not None else
            'check' if args.check_only else 'stage64') if raw_mode else
         chain/('shared_temporal_recovery128x256' if args.extend_saved is not None else
                'shared_temporal_recovery_check' if args.check_only else 'shared_temporal_recovery64')))
    if saved_directory is not None and args.output.resolve() == saved_directory.resolve():
        parser.error('Output must be separate from the supplied training result directory.')
    args.output.mkdir(parents=True, exist_ok=True)
    saved_arrays = None
    raw_arrays = read_arrays(args.raw_structured_parameters) if raw_mode else None
    raw_initial = raw_initializations(raw_arrays) if raw_mode else None
    if saved_directory is not None:
        affine_run = saved_run
        saved_arrays = {axis: read_arrays(saved_directory/(axis+'.npz')) for axis in axes}
        if raw_mode:
            fit = {}
        else:
            shared = saved_arrays[AXES[1]]
            fit = dict(native_bias=shared['consumer_bias'], d=shared['shared_d'],
                       permutation=shared['shared_permutation'])
        train, schedule = [], []
    else:
        affine = raw_arrays if raw_mode else read_arrays(args.affine_parameters)
        fit = {} if raw_mode else assignment_fit(affine)
        affine_run = json.loads(args.affine_parameters.with_name('run.json').read_text())
        train = read_train16(args.train_list)
        if affine_run['train_files'] != train[:4]:
            raise ValueError('Existing affine calibration is not the first4 of this fixed train16.')
        generator = np.random.default_rng(SEED)
        order = []
        while len(order) < STEPS:
            order.extend(generator.permutation(16).tolist())
        schedule = [train[index] for index in order[:STEPS]]
    for path in (alg, alg/'nrv_cost_probe', latent, residual):
        sys.path.insert(0, str(path))
    import run_probe as probe
    from run_bn_probe import input_frame, read_names, representative_train
    from evaluate_stage2_deployment import CoarseReady
    from evaluate_branch_control import evaluate_axis, mask_nonanchors
    from spikingjelly.activation_based import functional

    system = probe.load_system(args)
    model, modules, _, current, sources, _, _, _ = system
    probe.install_sources(system, sources)
    current['count_codes'] = False
    selection = None
    if args.extend_saved is not None:
        from collections import Counter
        all_train = sorted(read_names(args.data, 'train'))
        counts = Counter(name.rsplit('_', 1)[0] for name in all_train)
        if args.extension_plan is not None:
            selection = json.loads(args.extension_plan.read_text())
            train, schedule = selection['train'], selection['train_schedule']
            if len(schedule) != 256 or not set(train).issubset(all_train) or any(
                    Counter(schedule[start:start+128]) != Counter(train) for start in (0, 128)):
                raise ValueError('The existing extension plan must contain two permutations of the actual128 train frames.')
            selection = dict(selection, reused_plan=str(args.extension_plan))
        else:
            train = representative_train(all_train, 128)
            generator = np.random.default_rng(SEED)
            order = np.concatenate((generator.permutation(128), generator.permutation(128)))
            schedule = [train[int(index)] for index in order]
        selected_counts = Counter(name.rsplit('_', 1)[0] for name in train)
        if len(all_train) != 7345 or len(counts) != 18 or len(set(train)) != 128:
            raise ValueError('This fixed extension expects the actual7345-frame/18-sequence training split and128 distinct selections.')
        if args.extension_plan is None:
            selection = dict(source=str(args.data/'sequence_lists/train_split_seq.csv'),
                source_frames=len(all_train), source_sequences=dict(counts),
                selected_sequences=dict(selected_counts), train=train, train_schedule=schedule,
                rule='Sort frame filenames within each sequence (global lexicographic sort), then original representative_train(names,128); two successive seed912 permutations; no validation selection.')
        save_json(args.output/'train128.json', selection)
    if args.evaluate_saved is None and set(train).intersection(read_names(args.data, 'valid')):
        raise ValueError('The fixed training list overlaps official validation.')
    fixed = torch.load(area/'patch_train_calibration.pt', map_location='cpu', weights_only=False)
    for name, values in fixed.items():
        bn = modules[name]
        bn.track_running_stats = True
        bn.running_mean, bn.running_var = values['mean'].to(bn.weight), values['var'].to(bn.weight)
    model.eval()
    model.requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32 = bool(affine_run['TF32_matmul'])
    torch.backends.cudnn.allow_tf32 = bool(affine_run['TF32_cudnn'])
    parent = latent/'flow_recovery64/preview_only/shared48_u8_vq5.npz'
    parent_arrays = read_arrays(parent)
    conv1, sn2 = modules[BLOCK+'.conv1.0'], modules[BLOCK+'.sn2.spiking_neuron']
    original_conv1, original_sn2 = conv1.forward, sn2.forward
    controller = SharedTemporalControl(modules, read_arrays(residual/'rank_control_parameters.npz'),
        read_arrays(chain/'rank32_diverse10/parameters.npz'), fit, raw_initial=raw_initial)
    source, consumer = controller.source, controller.consumer
    if saved_directory is None and (not torch.equal(source.weight.detach().cpu(), torch.from_numpy(affine['source_A']).float()) or not torch.equal(
            consumer.weight.detach().cpu(), torch.from_numpy(affine['original_proj_A']).float())):
        raise ValueError('The live native temporal matrices do not match the common calibration parent.')
    masks = {}
    def delete_nonanchor(module, inputs, output):
        key = (output.shape[-2:], output.device)
        if key not in masks:
            mask = torch.zeros(output.shape[-2:], device=output.device, dtype=torch.bool)
            mask[::2, ::2] = True
            masks[key] = mask
        return mask_nonanchors(output, masks[key])
    common_hook = modules[BLOCK+'.norm2'].register_forward_hook(delete_nonanchor)
    projection_branch_hook = None
    projection_branch_target = None
    if args.projection_branch_off:
        ped = modules[PROJECT]
        projection_branch_target = 'norm_layer' if getattr(ped, 'norm', None) is not None else 'conv'
        branch_output = getattr(ped, projection_branch_target)
        projection_branch_hook = branch_output.register_forward_hook(
            lambda module, inputs, output: torch.zeros_like(output))
    def preserve_graph(module, inputs, output):
        current['differentiable_flow'] = output.sum(0)
    flow_hook = modules['sttmultires_unet.preds.2'].register_forward_hook(preserve_graph, prepend=True)
    names = (read_names(args.data, 'valid') if args.split == 'valid' else
             read_train16(args.train_list) if args.split == 'train' else
             json.loads((alg/'samples.json').read_text())['valid'])[:args.count]
    native_metadata = {label: dict(class_name=type(module).__module__+'.'+type(module).__name__,
        forward_file=inspect.getsourcefile(type(module)), activation=getattr(module.act, '__qualname__', str(module.act)),
        T=int(module.T), output_mode=module.output_mode, threshold_mode=module.threshold_mode,
        center_mode=module.center_mode, theta=float(module.thresh), temporal_factor_rank=int(module.temporal_factor_rank))
        for label, module in (('source', source), ('consumer', consumer))}
    run = dict(complete=False, check_only=args.check_only, parent=str(parent),
        affine_parameters=str(args.affine_parameters), train_files=train, train_schedule=schedule,
        assignment=fit, assignment_calibration_files=affine_run['train_files'], evaluation_files=names,
        fixed=dict(updates=0 if args.check_only else len(schedule), lr=LR, optimizer='Adam', seed=SEED,
            loss='mean sqrt(sum_xy(flow-GT)^2+1e-6), actual valid GT pixels; no distillation/request regularizer',
            trainable_common='Conv2 U16/V16, PED U32/V32, r1 BN2 beta96, separate PED outputbias96, both native neuron bias10',
            independent='sourceA100 + consumerA100', shared='sourceA100 + signed d10 with fixed P',
            frozen='theta, center, four BN gains/stats, preview Conv1 factors/PSN, all remaining parameters'),
        common_model='Original control3: Conv2R16 + nonanchor whole BN2 branch deletion + PEDR32; no recovered W/L model.',
        native_neurons=native_metadata,
        backward='Native official ATLIF derivatives for trained sn1/proj; frozen preview sn2 has existing same-hard-forward triangle in BOTH axes; no threshold installer update.',
        numeric='NativeFP32 operators; d is trained/stored Float64 to preserve closed-form multiply then FP32 parameter rounding. P only indexes sourceA rows.',
        check='First TRAIN frame; native initial function versus differentiable training function, sn1/sn2/proj hard gates and flow, each parameter gradient and source-output gradient. Zero optimizer updates.',
        initial_AEE='Measured for the exact training forward before any update. Fixed assigned-P shared is a new initialization, not the old same-t affine AEE.',
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32, TF32_cudnn=torch.backends.cudnn.allow_tf32,
        preview_temporal_and_backward_matmul_TF32=False,
        native_neuron_forward_matmul_TF32=torch.backends.cuda.matmul.allow_tf32, axes={})
    if args.extend_saved is not None:
        run.update(extend_saved=str(args.extend_saved), selection=selection,
            optimizer_restart='Fresh Adam1e-4 per axis from its own64-step final parameters; no optimizer-state inheritance.',
            assignment='Use saved shared P and d/sourceA; no further assignment or affine fitting.',
            assignment_calibration_files=affine_run.get('assignment_calibration_files', []),
            initial_AEE='Same native training arithmetic from each64-step endpoint; measure initial diverse10 and compare to that endpoint, never select a checkpoint.',
            check='First selected TRAIN frame: same saved training-forward/storage under no_grad versus differentiable mode, then the existing parameter/source-gradient check. No original-bool-adapter substitution.')
    if args.evaluate_saved is not None:
        run = dict(complete=False, evaluate_saved=str(args.evaluate_saved), parent=str(parent),
            split=args.split, count=args.count, coordinate_mode=args.coordinate_mode,
            coordinate_fp32_channel=args.coordinate_fp32_channel,
            count_sources=args.count_sources,
            preview_gate_fixed=args.preview_gate_fixed,
            projection_branch_off=args.projection_branch_off,
            projection_branch_target=None if projection_branch_target is None else PROJECT+'.'+projection_branch_target,
            evaluation_files=names, optimizer_updates=0,
            forward=('Explicit coordinate/FP32-time control on the same saved students, under no_grad; see per-axis numeric definition. New AEE required, not training-final bit equivalence.'
                if args.coordinate_mode is not None else
                'Same SharedTemporalControl and TrainableLatentPair as training final evaluation, under no_grad; no assignment fit or gradient check.'),
            storage='Copy NPZ values into original training strides, including Conv2 V stride32 and live sourceA/d/P parameterization; preserve frozen parent and base projection bias plus separate delta.',
            TF32_matmul=torch.backends.cuda.matmul.allow_tf32, TF32_cudnn=torch.backends.cudnn.allow_tf32,
            axes={})
        if args.projection_branch_off:
            run['forward'] += ' Complete PED spike-branch output is then zeroed after its norm_layer (including BN offset), or after conv if no norm; only the real conv_res branch reaches the PED addition.'
            run['activity_scope'] = 'The software still evaluates proj.sn/conv/BN for existing activity hooks, but their output has no consumer. These diagnostic counts neither require hardware execution nor measure saved cycles.'
    run['axis_family'] = 'raw_structured' if raw_mode else 'independent_and_shared'
    if raw_mode:
        run.update(raw_structured_parameters=str(args.raw_structured_parameters),
            numeric='Train/store e and optional L2/R2 in Float64; each forward assembles diag(e)*P [+ L2@R2] in Float64 and rounds the whole Ap to native Float32. SourceA is a separate trainable Float32 matrix. No free dense Ap.',
            assignment='Use each saved raw-coordinate permutation unchanged; no assignment, regression or validation fitting in this run.',
            raw_coordinate_initialization='The first TRAIN-frame reference uses this parameterization\'s actual assembled Ap as a native dense weight. Differences from the old saved rounded Ap are reported; old AEE is not borrowed.')
        if args.evaluate_saved is None:
            run['fixed'].pop('independent')
            run['fixed'].pop('shared')
            run['fixed'].update(raw_base='independent sourceA100 + e10 with fixed raw-coordinate P',
                raw_r2='independent sourceA100 + e10 + L2[10,2]/R2[2,10], fixed raw-coordinate P')
            run['initial_AEE'] = 'No initial validation pass in this fixed control: first TRAIN-frame hard-forward/gradient check only; diverse10 is evaluated once at each stage endpoint.'
    save_json(args.output/'result.json', run)
    if saved_directory is None and not raw_mode:
        np.savez_compressed(args.output/'assignment_initialization.npz', **{key: np.asarray(value) for key, value in fit.items()})

    def forward_flow(x):
        functional.reset_net(model)
        current.pop('flow', None)
        current.pop('differentiable_flow', None)
        try:
            model(x)
        except CoarseReady:
            flow = current.pop('differentiable_flow')
            current.pop('flow', None)
            return F.interpolate(flow, (480, 640), mode='bilinear', align_corners=False)
        raise RuntimeError('The actual coarse-flow exit was not reached.')

    def evaluate_with_activity(tag, progress):
        # Retain scalar/vector counts, never a full source tensor trace.
        activity = {key: [] for key in ('sn1', 'sn2', 'proj')}
        def count(label):
            def hook(module, inputs, output):
                activity[label].append(dict(nonzero_gates=int(output.ne(0).sum()),
                    total_elements=output.numel(), theta_amplitude=float(module.thresh),
                    gate_has_consumer=not (args.projection_branch_off and label == 'proj')))
            return hook
        handles = [module.register_forward_hook(count(label)) for label, module in
                   (('sn1', source), ('sn2', sn2), ('proj', consumer))]
        source_counts = None
        if args.count_sources:
            from shared_temporal_source_counts import SharedTemporalSourceCounts
            source_counts = SharedTemporalSourceCounts(modules, sn2_theta=student.temporal.theta).install()
            source_counts.reset(tag)
        try:
            evaluation = evaluate_axis(args, model, current, names, tag, progress_tag=progress)
        finally:
            for handle in handles:
                handle.remove()
            if source_counts is not None:
                source_counts.restore()
        if source_counts is not None:
            save_json(args.output/(tag+'_sources.json'), source_counts.report(names))
        report = dict(scope='Actual new-student hard gate counts; theta amplitude separate, not inherited old-source/NRV statistics.',
            frames=[dict(file=filename, **{key: activity[key][i] for key in activity})
                    for i, filename in enumerate(names)],
            totals={key:dict(nonzero_gates=sum(row['nonzero_gates'] for row in rows),
                total_elements=sum(row['total_elements'] for row in rows),
                theta_amplitude=rows[0]['theta_amplitude'],
                gate_has_consumer=rows[0]['gate_has_consumer']) for key,rows in activity.items()})
        if args.projection_branch_off:
            report['scope'] += ' proj gates are software diagnostics only: the whole normalized spike branch is discarded and has no consumer; no measured execution saving is asserted.'
        save_json(args.output/(tag+'_activity.json'), report)
        return evaluation, tag+'_activity.json'

    gate_handles = []
    try:
        if args.evaluate_saved is not None:
            for axis in axes:
                student = TrainableLatentPair(parent_arrays, conv1.weight.device)
                student.u.requires_grad_(False)
                student.v.requires_grad_(False)
                controller.load_saved(axis, saved_arrays[axis])
                conv1.forward, sn2.forward = student.conv_forward, student.neuron_forward
                row = dict(complete=False, parameters_file=str(args.evaluate_saved/(axis+'.npz')),
                    saved_consumer_A_max_abs=controller.saved_consumer_A_max_abs,
                    constraints=controller.constraints())
                run['axes'][axis] = row
                coordinates = None
                preview_coordinates = None
                consumer_capture = None
                if args.coordinate_mode is not None:
                    if args.coordinate_mode == 'fixed_temporal':
                        from fixed_temporal_coordinates import FixedTemporalForward
                        coordinates = FixedTemporalForward(controller, sn2_theta=student.temporal.theta)
                    elif args.coordinate_mode == 'raw_factorized':
                        from raw_temporal_coordinates import RawTemporalForward
                        coordinates = RawTemporalForward(controller,
                            fp32_channel=args.coordinate_fp32_channel)
                    else:
                        from shared_temporal_coordinates import CoordinateForward
                        coordinates = CoordinateForward(controller, args.coordinate_mode,
                            fp32_channel=args.coordinate_fp32_channel)
                    row['coordinate_function'] = coordinates.metadata
                    np.savez_compressed(args.output/(axis+'_coordinate_constants.npz'), **coordinates.export_constants())
                if args.preview_time_first:
                    from preview_temporal_coordinates import PreviewTemporalForward
                    preview_coordinates = PreviewTemporalForward(student, conv1, sn2,
                        modules[BLOCK+'.norm1.norm_layer'], fp32_channel=args.coordinate_fp32_channel)
                    row['preview_coordinate_function'] = preview_coordinates.metadata
                    np.savez_compressed(args.output/(axis+'_preview_constants.npz'), **preview_coordinates.export_constants())
                if args.preview_gate_fixed:
                    from preview_gate_fixed import PreviewGateFixed
                    preview_coordinates = PreviewGateFixed(student, conv1, sn2,
                        modules[BLOCK+'.norm1.norm_layer'],
                        output_directory=args.output/(axis+'_preview_capture'),
                        fractional_bits=args.preview_az_frac)
                    row['preview_coordinate_function'] = preview_coordinates.metadata
                    np.savez_compressed(args.output/(axis+'_preview_constants.npz'), **preview_coordinates.export_constants())
                if args.capture_consumers:
                    from temporal_consumer_capture import TemporalConsumerCapture
                    consumer_capture = TemporalConsumerCapture(controller, coordinates,
                        modules, args.output/(axis+'_capture'))
                try:
                    row['evaluation'], row['activity_file'] = evaluate_with_activity(axis, 'SHARED_TEMPORAL_RELOAD_AEE')
                    if args.count_sources:
                        row['source_counts_file'] = axis+'_sources.json'
                    if coordinates is not None:
                        row['coordinate_ranges_file'] = axis+'_coordinate_ranges.json'
                        save_json(args.output/row['coordinate_ranges_file'], coordinates.range_report(names))
                    if preview_coordinates is not None:
                        row['preview_ranges_file'] = axis+'_preview_ranges.json'
                        save_json(args.output/row['preview_ranges_file'], preview_coordinates.range_report(names))
                    if consumer_capture is not None:
                        consumer_capture.save(names)
                        row['consumer_capture_directory'] = axis+'_capture'
                finally:
                    if consumer_capture is not None:
                        consumer_capture.restore()
                    if preview_coordinates is not None:
                        preview_coordinates.restore()
                    if coordinates is not None:
                        coordinates.restore()
                row['training_final_alignment'] = frame_comparison(args.output/(axis+'_frames.json'),
                    args.evaluate_saved/(axis+'_frames.json'))
                row['training_final_alignment']['scope'] = ('Complete PED spike-branch deletion versus the saved full student; changed AEE is the ablation result, not a reload failure.' if args.projection_branch_off else
                    'New coordinate/FP32-time function versus old training-final arithmetic; differences are expected, not an exact reload failure.'
                    if coordinates is not None or preview_coordinates is not None else 'Only frame identities common to this reload and training-final diverse10; full825 has a different frame set.')
                row['complete'] = True
                save_json(args.output/'result.json', run)
                release_model_graphs(modules, current, controller)
                functional.reset_net(model)
                gc.collect()
                torch.cuda.empty_cache()
            run['complete'] = True
            save_json(args.output/'result.json', run)
            print('RELOAD_DONE', json.dumps({axis: row['evaluation']['AEE_frame_mean']
                                            for axis,row in run['axes'].items()}), flush=True)
            return
        for axis in axes:
            torch.manual_seed(SEED)
            if args.extend_saved is None:
                reference = LatentPair(parent_arrays, conv1.weight.device, conditional=False)
            else:
                reference = TrainableLatentPair(parent_arrays, conv1.weight.device)
                reference.u.requires_grad_(False)
                reference.v.requires_grad_(False)
            student = TrainableLatentPair(parent_arrays, conv1.weight.device)
            student.u.requires_grad_(False)
            student.v.requires_grad_(False)
            observed, source_gradient = {}, {}
            def observer(label):
                def collect(module, inputs, output):
                    observed[label] = dict(shape=list(output.shape),
                        bits=np.packbits(output.detach().ne(0).cpu().numpy(), bitorder='little'))
                    if label == 'sn1' and output.requires_grad:
                        def source_grad(gradient):
                            source_gradient.update(present=True, all_finite=bool(torch.isfinite(gradient).all()),
                                nonzero=int(torch.count_nonzero(gradient)),
                                l2=float(torch.linalg.vector_norm(gradient.double())))
                        output.register_hook(source_grad)
                return collect
            gate_handles = [module.register_forward_hook(observer(label)) for label, module in
                            (('sn1', source), ('sn2', sn2), ('proj', consumer))]
            x, label, valid = input_frame(args.data, train[0])
            if args.extend_saved is None:
                controller.reference(axis)
            else:
                controller.load_saved(axis, saved_arrays[axis])
            conv1.forward, sn2.forward = reference.conv_forward, reference.neuron_forward
            with torch.no_grad():
                expected = forward_flow(x)
            expected_gates = observed.copy()
            observed.clear()
            if args.extend_saved is None:
                controller.trainable(axis)
            conv1.forward, sn2.forward = student.conv_forward, student.neuron_forward
            prediction = forward_flow(x)
            different = (prediction.detach()-expected).abs()
            error = prediction.permute(0, 2, 3, 1)[valid]-label.permute(0, 2, 3, 1)[valid]
            loss = torch.sqrt(error.square().sum(-1)+1e-6).mean()
            check = dict(file=train[0], optimizer_updates=0,
                gates={key:dict(shape=observed[key]['shape'], different=int(np.unpackbits(
                    observed[key]['bits']^expected_gates[key]['bits']).sum())) for key in observed},
                flow_different=int(torch.count_nonzero(different)), flow_max_abs=float(different.max()),
                loss=float(loss.detach()), loss_requires_grad=loss.requires_grad, valid_pixels=int(valid.sum()))
            if loss.requires_grad:
                with fp32_matmul():
                    loss.backward()
            check['gradients'], check['source_output_gradient'] = controller.gradients(), source_gradient
            check['pass'] = (all(not item['different'] for item in check['gates'].values())
                and not check['flow_different']
                and all(g.get('present') and g.get('all_finite') and g.get('nonzero', 0) > 0
                        for g in check['gradients'].values())
                and source_gradient.get('all_finite', False) and source_gradient.get('nonzero', 0) > 0)
            row = dict(complete=False, updates=0, check=check,
                parameter_counts={key:p.numel() for key,p in controller.params.items()},
                total_parameters=sum(p.numel() for p in controller.params.values()),
                initial_constraints=controller.constraints(), history=[])
            prior_steps = 0 if args.extend_saved is None else int(saved_arrays[axis].get(
                'cumulative_steps', saved_arrays[axis]['steps']))
            row['prior_steps'] = prior_steps
            run['axes'][axis] = row
            save_json(args.output/'result.json', run)
            print('SHARED_TEMPORAL_CHECK', axis, json.dumps(check), flush=True)
            for handle in gate_handles:
                handle.remove()
            gate_handles = []
            del x, label, valid, expected, expected_gates, observed, prediction, different, error, loss
            for parameter in controller.params.values():
                parameter.grad = None
            release_model_graphs(modules, current, controller)
            if not check['pass']:
                raise RuntimeError('Training-frame hard-forward/gradient check failed; no optimizer update for '+axis)
            if args.check_only:
                row['complete'] = True
                save_json(args.output/'result.json', run)
                continue
            if not raw_mode:
                row['initial_evaluation'], row['initial_activity_file'] = evaluate_with_activity(
                    axis+'_initial', 'SHARED_TEMPORAL_INITIAL_AEE')
                if args.extend_saved is not None:
                    alignment = frame_comparison(args.output/(axis+'_initial_frames.json'),
                        args.extend_saved/(axis+'_frames.json'))
                    alignment['scope'] = 'Same axis64-step endpoint reloaded into original training forward; no coordinate rewrite.'
                    row['initial_endpoint_alignment'] = alignment
                elif axis == AXES[0]:
                    alignment = frame_comparison(args.output/(axis+'_initial_frames.json'),
                        args.affine_parameters.with_name('ordinary_control3_frames.json'))
                    alignment['scope'] = 'Original control3 under the new training-forward/storage path; no inherited825.'
                    row['initial_ordinary_alignment'] = alignment
            release_model_graphs(modules, current, controller)
            save_json(args.output/'result.json', run)
            optimizer = torch.optim.Adam(list(controller.params.values()), lr=LR)
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            for step, filename in enumerate(schedule):
                x, label, valid = input_frame(args.data, filename)
                optimizer.zero_grad(set_to_none=True)
                prediction = forward_flow(x)
                error = prediction.permute(0, 2, 3, 1)[valid]-label.permute(0, 2, 3, 1)[valid]
                loss = torch.sqrt(error.square().sum(-1)+1e-6).mean()
                with fp32_matmul():
                    loss.backward()
                gradients = controller.gradients()
                if not bool(torch.isfinite(loss)) or not all(g.get('present') and g.get('all_finite') for g in gradients.values()):
                    raise RuntimeError('Nonfinite/missing gradient in the fixed recovery: '+axis)
                entry = dict(step=step+1, file=filename, loss_before_update=float(loss.detach()),
                    AEE_before_update=float(torch.linalg.vector_norm(error.detach(), dim=-1).double().mean()),
                    valid_pixels=int(valid.sum()))
                if step in (0, len(schedule)-1):
                    entry['gradients'] = gradients
                optimizer.step()
                row['history'].append(entry)
                row['updates'] = step+1
                if step == 0 or (step+1)%8 == 0:
                    save_json(args.output/'result.json', run)
                    print('SHARED_TEMPORAL_RECOVERY', axis, step+1, json.dumps(entry), flush=True)
                del x, label, valid, prediction, error, loss
                release_model_graphs(modules, current, controller)
            row['training_wall_seconds'] = time.monotonic()-started
            row['training_peak_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
            row['final_constraints'] = controller.constraints()
            row['cumulative_steps'] = prior_steps+len(schedule)
            np.savez_compressed(args.output/(axis+'.npz'), **controller.export(parent, schedule, prior_steps=prior_steps))
            row['parameters_file'] = axis+'.npz'
            row['evaluation'], row['activity_file'] = evaluate_with_activity(axis, 'SHARED_TEMPORAL_FINAL_AEE')
            row['complete'] = True
            save_json(args.output/'result.json', run)
            del optimizer
            release_model_graphs(modules, current, controller)
            functional.reset_net(model)
            gc.collect()
            torch.cuda.empty_cache()
        run['complete'] = True
        save_json(args.output/'result.json', run)
    finally:
        for handle in gate_handles:
            handle.remove()
        flow_hook.remove()
        common_hook.remove()
        if projection_branch_hook is not None:
            projection_branch_hook.remove()
        controller.restore()
        conv1.forward, sn2.forward = original_conv1, original_sn2
        current.pop('flow', None)
        current.pop('differentiable_flow', None)
    print('DONE', json.dumps({axis:dict(updates=row['updates'], check_pass=row['check']['pass'])
                              for axis,row in run['axes'].items()}), flush=True)


if __name__ == '__main__':
    main()
