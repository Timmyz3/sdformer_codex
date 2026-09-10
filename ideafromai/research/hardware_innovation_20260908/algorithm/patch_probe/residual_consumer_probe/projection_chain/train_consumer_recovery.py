"""Four fixed64-step real-GT recovery controls; root launches CUDA.

Same rank32 U/V, fixed6/12 source-C8 groups, train16, seed912 schedule,
Adam1e-4, real coarse-flow robust EPE. The starting biases are the already
completed sequential empirical-correction results. Train the two Conv2 W and
the same288 output biases in every axis; independent_sparse_L additionally
trains two sparse32x864 projected kernels. No distillation/request loss,
gain/variance changes, new surrogate for other integer paths, or sweep.

Only r1.sn2 changes its backward to the existing TrainableLatentPair triangle;
its theta-valued hard forward and latent U/V remain fixed. --check-only runs
one training-frame native-bool versus surrogate-forward/gradient check per
axis, with zero optimizer updates. --self-check is a small CPU algebra check.

The independent-L forward explicitly subtracts current P W source products
and adds L source products before V. It is an accuracy implementation, not a
deployment cost model. Its L never follows W after initialization. Both its
selected L columns and source_group_zero W columns have hard masks. Nullspace
W uses Float64 fixed P/P-dagger projection then nativeFloat32 rounding, not an
exact finite-precision zero claim. This is a recovery probe, not full
EigenDamage or UPSCALE and not a new mechanism.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
LATENT = HERE.parent.parent/'factor_completion_20260909/latent_stage_train16'
sys.path.insert(0, str(LATENT))
from flow_backward_probe import TrainableLatentPair, gradient_stats, read_arrays
from adapter import LatentPair, fp32_matmul
from capture_chain import R0, R1, PROJECT, SPECS, save_json
from evaluate_consumer_pruning import ConsumerControl

AXES = ('ordinary_rank32', 'source_group_zero', 'consumer_nullspace', 'independent_sparse_L')
BIAS_AXIS = {axis: axis for axis in AXES}
BIAS_AXIS['independent_sparse_L'] = 'projection_only_group_zero'
STEPS, LR, SEED = 64, 1e-4, 912


def nullspace_weight(shadow, p, pinverse, selected_columns):
    """Selected columns = (I-P^dagger P)Z; keep original source coordinates."""
    flat = shadow.flatten(1)
    selected = flat.index_select(1, selected_columns).double()
    projected = selected-pinverse@(p@selected)
    return flat.index_copy(1, selected_columns, projected.to(flat.dtype)).reshape_as(shadow)


def canonical_source(x):
    if x.ndim == 5 and tuple(x.shape[:3]) == (10, 1, 96):
        return x.flatten(0, 1)
    if x.ndim == 4 and tuple(x.shape[:2]) == (10, 96):
        return x
    raise ValueError('Expected actual T10/B1/C96 Conv2 source: '+str(tuple(x.shape)))


class RecoveryControl(ConsumerControl):
    def __init__(self, modules, arrays):
        super().__init__(modules, arrays)
        device = self.projection.weight.device
        self.original_conv_forward = {label: conv.forward for label, conv in self.conv.items()}
        self.bn = {label: modules[SPECS[label]['norm']] for label in ('r0', 'r1')}
        self.original_bn_parameter = {label: bn.bias for label, bn in self.bn.items()}
        self.original_projection_parameter = self.projection.bias
        self.p = {label: torch.as_tensor(arrays[label+'_P'], device=device, dtype=torch.float64)
                  for label in ('r0', 'r1')}
        self.pinverse = {label: torch.as_tensor(np.linalg.pinv(arrays[label+'_P'], rcond=1e-12),
                            device=device, dtype=torch.float64) for label in ('r0', 'r1')}
        self.flatmask = {label: torch.as_tensor(np.repeat(arrays[label+'_selected_source_mask'], 9),
                                              device=device, dtype=torch.bool) for label in ('r0', 'r1')}
        self.selected_columns = {label: mask.nonzero().flatten() for label, mask in self.flatmask.items()}
        self.shadow, self.l_shadow = {}, {}
        self.collect_activity = False
        self.projection_bias_delta = None
        self.bias_files = None
        self.initial_parameters = None
        for label, conv in self.conv.items():
            conv.forward = lambda x, label=label: self.conv_forward(label, x)

    def initialize_axis(self, axis, biases):
        with torch.no_grad():
            super().begin('ordinary_rank32' if axis == 'independent_sparse_L' else axis)
        self.axis = axis
        self.shadow, self.l_shadow = {}, {}
        self.collect_activity = False
        self.terms.clear()
        for label, conv in self.conv.items():
            # Null Z starts at the original W, so the constrained forward
            # implements the same original-coordinate projection as selection.
            weights = self.arrays[label+'_zero_W'] if axis == 'source_group_zero' else self.arrays[label+'_original_W']
            self.shadow[label] = nn.Parameter(torch.as_tensor(weights, device=conv.weight.device,
                                                               dtype=conv.weight.dtype).clone())
            self.bn[label].bias = nn.Parameter(torch.as_tensor(biases[label+'_bias'],
                device=conv.weight.device, dtype=conv.weight.dtype).clone())
            if axis == 'independent_sparse_L':
                l = torch.as_tensor(self.arrays[label+'_K'], device=conv.weight.device,
                                    dtype=conv.weight.dtype).clone()
                l[:, self.flatmask[label]] = 0
                self.l_shadow[label] = nn.Parameter(l)
        base = torch.as_tensor(biases['projection_bias'], device=self.projection.weight.device,
                               dtype=self.projection.weight.dtype).clone()
        # Keep ordinary's original bias-free convolution when correction is0.
        self.projection.bias = (None if self.original_projection_parameter is None and not bool(base.ne(0).any())
                                else nn.Parameter(base, requires_grad=False))
        self.projection_bias_delta = nn.Parameter(torch.zeros_like(base))
        if self.initial_parameters is not None:
            with torch.no_grad():
                for label, parameter in self.shadow.items():
                    parameter.copy_(torch.as_tensor(self.initial_parameters[axis+'__'+label+'_W_shadow'],
                        device=parameter.device, dtype=parameter.dtype))
                for label, parameter in self.l_shadow.items():
                    parameter.copy_(torch.as_tensor(self.initial_parameters[axis+'__'+label+'_L'],
                        device=parameter.device, dtype=parameter.dtype))

    def effective_weight(self, label):
        weight = self.shadow[label]
        if self.axis == 'source_group_zero':
            return weight.flatten(1).masked_fill(self.flatmask[label][None], 0).reshape_as(weight)
        if self.axis == 'consumer_nullspace':
            return nullspace_weight(weight, self.p[label], self.pinverse[label], self.selected_columns[label])
        return weight

    def effective_l(self, label):
        return self.l_shadow[label].masked_fill(self.flatmask[label][None], 0)

    def conv_forward(self, label, x):
        local = canonical_source(x)
        value = F.conv2d(local, self.effective_weight(label), self.conv[label].bias, padding=1)
        return value.reshape(10, 1, 96, *value.shape[-2:]) if x.ndim == 5 else value

    def observe(self, label, x):
        if self.collect_activity:
            # The inherited observer only creates old projected-K terms for
            # projection_only_group_zero, which is not a recovery axis.
            super().observe(label, x)
            from count_projection_sources import count_source
            rows = count_source(canonical_source(x).detach(), 2,
                float(self.arrays[label+'_theta']), self.arrays[label+'_selected_source_mask'])
            self.counts[label][-1].update(paired_anchor_nrv_rows=rows['nrv_rows'],
                selected_paired_anchor_nrv_rows=rows['selected_nrv_rows'],
                retained_paired_anchor_nrv_rows=rows['retained_nrv_rows'],
                paired_anchor_nrv_rows_by_k=rows['nrv_rows_by_k'])
        if self.axis == 'independent_sparse_L':
            local = canonical_source(x)
            weight = self.effective_weight(label)
            pw = (self.p[label]@weight.flatten(1).double()).to(weight.dtype).reshape(32, 96, 3, 3)
            l = self.effective_l(label).reshape(32, 96, 3, 3)
            # Do not detach current W, L, or the actual source. In particular,
            # r0 changes can affect r1's new source through the original gates.
            self.terms[label] = (F.conv2d(local, pw, stride=2, padding=1),
                                 F.conv2d(local, l, stride=2, padding=1))

    def projected(self, x):
        value = F.conv2d(x[:, :, ::2, ::2], self.first)
        if self.axis == 'independent_sparse_L':
            r0_pw, r0_l = self.terms.pop('r0')
            r1_pw, r1_l = self.terms.pop('r1')
            value = value-r0_pw-r1_pw+r0_l+r1_l
        value = F.conv2d(value, self.second, self.projection.bias)
        # An independent zero-initialized output increment preserves the old
        # convolution's bias/no-bias choice at initialization. Train288 biases
        # in every axis without changing BN gains or the fixed nullspace P.
        return value+self.projection_bias_delta[None, :, None, None]

    def named_trainables(self):
        result = {label+'_W': value for label, value in self.shadow.items()}
        result.update({label+'_L': value for label, value in self.l_shadow.items()})
        result.update({label+'_BN_bias': bn.bias for label, bn in self.bn.items()})
        result['projection_bias_delta'] = self.projection_bias_delta
        return result

    def source_report(self, names):
        report = super().source_report(names)
        report['nrv_scope'] = ('OR allT10 and two adjacent PED anchors per original horizontalP4; '
            'source-ready logical rows, not physical requests or cycles.')
        for label, branch in report['branches'].items():
            for key in ('paired_anchor_nrv_rows', 'selected_paired_anchor_nrv_rows',
                        'retained_paired_anchor_nrv_rows'):
                branch[key] = sum(row[key] for row in self.counts[label])
        return report

    def gradients(self):
        result = {}
        for name, parameter in self.named_trainables().items():
            allowed = None
            label = name[:2]
            if name.endswith('_L'):
                allowed = (~self.flatmask[label])[None].expand_as(parameter)
            elif name.endswith('_W') and self.axis == 'source_group_zero':
                allowed = (~self.flatmask[label])[None].expand(96, -1).reshape_as(parameter)
            result[name] = gradient_stats(parameter, allowed)
        return result

    @torch.no_grad()
    def constraints(self):
        result = {}
        for label in ('r0', 'r1'):
            weight = self.effective_weight(label).flatten(1)
            selected = weight[:, self.flatmask[label]]
            row = dict(effective_W_nonzeros=int(torch.count_nonzero(weight)),
                       selected_W_nonzeros=int(torch.count_nonzero(selected)))
            if self.axis == 'consumer_nullspace':
                row['selected_PW_max_abs_after_Float32_W'] = float((self.p[label]@selected.double()).abs().max())
                row['zero_claim'] = 'No exact Float32 projected-zero or legal source-skip claim.'
            if self.axis == 'independent_sparse_L':
                l = self.effective_l(label)
                row.update(effective_L_nonzeros=int(torch.count_nonzero(l)),
                           selected_L_nonzeros=int(torch.count_nonzero(l[:, self.flatmask[label]])))
            result[label] = row
        return result

    @torch.no_grad()
    def parameter_counts(self):
        bias_count = 288
        w_slots = sum(parameter.numel() for parameter in self.shadow.values())
        l_slots = sum(parameter.numel() for parameter in self.l_shadow.values())
        if self.axis == 'source_group_zero':
            w_dof = sum(96*int((~mask).sum()) for mask in self.flatmask.values())
        elif self.axis == 'consumer_nullspace':
            w_dof = sum(96*int((~self.flatmask[label]).sum())
                        +(96-int(np.linalg.matrix_rank(self.arrays[label+'_P'])))*int(self.flatmask[label].sum())
                        for label in ('r0', 'r1'))
        else:
            w_dof = w_slots
        l_dof = sum(32*int((~mask).sum()) for mask in self.flatmask.values()) if l_slots else 0
        return dict(optimizer_shadow_scalars=w_slots+l_slots+bias_count,
            W_shadow_scalars=w_slots, W_real_constraint_degrees_of_freedom=w_dof,
            L_shadow_scalars=l_slots, L_unmasked_scalars=l_dof, shared_trainable_biases=bias_count,
            total_real_constraint_degrees_of_freedom=w_dof+l_dof+bias_count,
            exported_effective_W_dense_FP32_bytes=w_slots*4,
            independent_L_unmasked_FP32_coefficient_bytes=l_dof*4,
            independent_L_dense_NPZ_scalar_slots=l_slots,
            common_projection_UV_fixed_scalars=int(self.first.numel()+self.second.numel()),
            deployment_scope='Coefficient counts only. Independent L needs extra sparse kernels and dispatch; temporary PW reconstruction in this accuracy forward is not its deployment cost.')

    @torch.no_grad()
    def export(self, biases, schedule):
        result = {key: value.copy() for key, value in self.arrays.items()}
        for label in ('r0', 'r1'):
            result[label+'_effective_W'] = self.effective_weight(label).detach().cpu().numpy()
            result[label+'_W_shadow'] = self.shadow[label].detach().cpu().numpy()
            result[label+'_bias'] = self.bn[label].bias.detach().cpu().numpy()
            if self.axis == 'independent_sparse_L':
                result[label+'_L'] = self.effective_l(label).detach().cpu().numpy()
                result[label+'_L_shadow'] = self.l_shadow[label].detach().cpu().numpy()
        base = (np.zeros(96, np.float32) if self.projection.bias is None
                else self.projection.bias.detach().cpu().numpy())
        delta = self.projection_bias_delta.detach().cpu().numpy()
        result.update(axis=np.array(self.axis), projection_base_bias=base,
            projection_base_has_bias=np.array(self.projection.bias is not None),
            projection_bias_delta=delta, projection_bias_combined=base+delta,
            projection_bias_evaluation=np.array('F.conv2d with original corrected base bias/no-bias, then learned output delta; combined array is algebraic, not a bitwise reassociation guarantee'),
            calibration_files=biases['calibration_files'], recovery_schedule=np.asarray(schedule),
            recovery_steps=np.array(STEPS), recovery_lr=np.array(LR), recovery_seed=np.array(SEED),
            recovery_loss=np.array('real coarse GT mean sqrt(sum_xy(error^2)+1e-6), valid pixels; W/L and288bias only'),
            reference_arrays=np.array('original_W,zero_W,nullspace_W,K are unchanged initialization/reference arrays; use effective_W and L for this recovered student'),
            numeric_scope=np.array('NativeFP32 accuracy student; fixedP nullspace projection usesFloat64 thenFloat32, not an integer or exact source-pruning contract'))
        return result

    def restore(self):
        for label, conv in self.conv.items():
            conv.forward = self.original_conv_forward[label]
            self.bn[label].bias = self.original_bn_parameter[label]
        self.projection.bias = self.original_projection_parameter
        super().restore()


def read_train16(path):
    text = Path(path).read_text()
    if Path(path).suffix == '.json':
        names = json.loads(text)
        names = names['train'] if isinstance(names, dict) else names
    else:
        names = [line.strip() for line in text.splitlines() if line.strip()]
    if len(names) != 16 or len(set(names)) != 16:
        raise ValueError('Use the existing explicit16 distinct training-frame identities.')
    return names


def release_model_graphs(modules, current, controller):
    controller.terms.clear()
    current.pop('flow', None)
    current.pop('differentiable_flow', None)
    for module in modules.values():
        value = getattr(module, 'act_value', None)
        if torch.is_tensor(value):
            module.act_value = value.detach()


def self_check():
    """Small CPU constraint/gradient check, including independent-L cancellation."""
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(SEED)
    p = torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 1.]], dtype=torch.float64)
    pinv = torch.linalg.pinv(p)
    columns = torch.tensor([0, 2])
    z = nn.Parameter(torch.randn(4, 4, generator=generator, dtype=torch.float64))
    w = nullspace_weight(z, p, pinv, columns)
    loss = (w*torch.randn(w.shape, generator=generator, dtype=torch.float64)).sum()
    loss.backward()
    projected_max = float((p@w[:, columns]).detach().abs().max())
    gradient_max = float((p@z.grad[:, columns]).abs().max())
    source = torch.randn(4, 5, generator=generator, dtype=torch.float64)
    gate_w = nn.Parameter(torch.randn(4, 4, generator=generator, dtype=torch.float64))
    l = nn.Parameter(torch.randn(2, 4, generator=generator, dtype=torch.float64))
    mask = torch.tensor([True, False, True, False])
    l_eff = l.masked_fill(mask[None], 0)
    ordinary = p@(gate_w@source)
    independent = ordinary-(p@gate_w)@source+l_eff@source
    expected = l_eff@source
    forward_error = float((independent-expected).detach().abs().max())
    independent.square().sum().backward()
    cancel_gradient = float(gate_w.grad.abs().max())
    masked_gradient = int(torch.count_nonzero(l.grad[:, mask]))
    if projected_max > 1e-12 or gradient_max > 1e-12 or forward_error > 1e-12 or cancel_gradient > 1e-11 or masked_gradient:
        raise RuntimeError('CPU algebra check failed.')
    print('CPU_CONSTRAINT_CHECK', json.dumps(dict(null_selected_PW_max=projected_max,
        null_shadow_gradient_PG_max=gradient_max, independent_L_forward_max=forward_error,
        isolated_continuous_W_gradient_max=cancel_gradient,
        forbidden_L_gradient_nonzeros=masked_gradient,
        note='The full network still has W-dependent gate/producer paths; this toy only checks direct continuous-term cancellation.')), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--train-list', type=Path)
    parser.add_argument('--parameters', type=Path)
    parser.add_argument('--bias-directory', type=Path)
    parser.add_argument('--initial-parameters', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.root is None or args.train_list is None:
        parser.error('--root and the existing explicit --train-list are required')
    args.split = 'diverse'
    alg, area = args.root/'algorithm', args.root/'algorithm/patch_probe'
    residual = area/'residual_consumer_probe'
    chain = residual/'projection_chain'
    latent = area/'factor_completion_20260909/latent_stage_train16'
    args.parameters = args.parameters or chain/'consumer_pruning_diverse10/parameters.npz'
    args.bias_directory = args.bias_directory or chain/'consumer_bias_corrected_diverse10'
    args.output = args.output or chain/('consumer_recovery_check' if args.check_only else 'consumer_recovery64')
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = read_arrays(args.parameters)
    initial_parameters = None if args.initial_parameters is None else read_arrays(args.initial_parameters)
    train = read_train16(args.train_list)
    if train[:4] != arrays['selected_train_frames'].tolist():
        raise ValueError('The first four train16 identities differ from the fixed selection/correction frames.')
    bias_arrays = {axis: read_arrays(args.bias_directory/(BIAS_AXIS[axis]+'_biases.npz')) for axis in AXES}
    for axis, biases in bias_arrays.items():
        if biases['calibration_files'].tolist() != train[:4]:
            raise ValueError('Bias calibration identities differ for '+axis)
    generator = np.random.default_rng(SEED)
    order = []
    while len(order) < STEPS:
        order.extend(generator.permutation(16).tolist())
    schedule = [train[index] for index in order[:STEPS]]
    for path in (alg, alg/'nrv_cost_probe', latent, residual):
        sys.path.insert(0, str(path))
    import run_probe as probe
    from run_bn_probe import input_frame, read_names
    from evaluate_stage2_deployment import CoarseReady
    from evaluate_branch_control import evaluate_axis, mask_nonanchors
    from spikingjelly.activation_based import functional

    system = probe.load_system(args)
    model, modules, _, current, sources, _, _, _ = system
    probe.install_sources(system, sources)
    current['count_codes'] = False
    if set(train).intersection(read_names(args.data, 'valid')):
        raise ValueError('Recovery train16 overlaps official validation.')
    fixed = torch.load(area/'patch_train_calibration.pt', map_location='cpu', weights_only=False)
    for name, values in fixed.items():
        bn = modules[name]
        bn.track_running_stats = True
        bn.running_mean = values['mean'].to(bn.weight)
        bn.running_var = values['var'].to(bn.weight)
    model.eval()
    parent = latent/'flow_recovery64/preview_only/shared48_u8_vq5.npz'
    parent_arrays = read_arrays(parent)
    conv1, sn2 = modules[R1+'.conv1.0'], modules[R1+'.sn2.spiking_neuron']
    original_conv1, original_sn2 = conv1.forward, sn2.forward
    masks = {}
    def delete_nonanchor(module, inputs, output):
        key = (output.shape[-2:], output.device)
        if key not in masks:
            mask = torch.zeros(output.shape[-2:], device=output.device, dtype=torch.bool)
            mask[::2, ::2] = True
            masks[key] = mask
        return mask_nonanchors(output, masks[key])
    common_hook = modules[R1+'.norm2'].register_forward_hook(delete_nonanchor)
    def preserve_graph(module, inputs, output):
        current['differentiable_flow'] = output.sum(0)
    flow_hook = modules['sttmultires_unet.preds.2'].register_forward_hook(preserve_graph, prepend=True)
    names = json.loads((alg/'samples.json').read_text())['valid'][:10]
    run = dict(complete=False, check_only=args.check_only, parent=str(parent),
        parameters=str(args.parameters), bias_directory=str(args.bias_directory),
        initial_parameters=None if args.initial_parameters is None else str(args.initial_parameters),
        train_files=train, train_schedule=schedule, check_frame=train[0], evaluation_files=names,
        fixed=dict(updates=0 if args.check_only else STEPS, optimizer='Adam', lr=LR, seed=SEED,
            batch_frames=1, loss='mean sqrt(sum_xy(flow-GT)^2+1e-6), actual valid GT pixels',
            trainable='two Conv2 W and288 common biases; independent_sparse_L additionally two masked32x864L',
            gain_variance_P_UV_theta_T='fixed', distillation=False, request_loss=False),
        backward='Only r1.sn2 uses existing TrainableLatentPair triangular surrogate; its U/V frozen, all other native/integer derivatives unchanged.',
        independent_L=('Initial W/L come from the explicitly supplied fixed group-OBS parameters and recalibrated biases.'
            if args.initial_parameters is not None else
            'Initialized from projection_only bias correction, original gate W, L=PW with selected columns0.')
            +' CurrentPW subtracts actual current-source products; L is independent after initialization.',
        nullspace='Selected columns=(I-PdaggerP)Z in originalW coordinates, staticFloat64P, thenFloat32. Same real feasible set as a constrained projectedL; parameterization is not a new mechanism.',
        common_biases='r0/r1 BN.beta96 each plus separate projection output delta96; original bias-free ordinary convolution preserved at initialization.',
        forward_check='Same initialized effective student: native boolean LatentPair versus TrainableLatentPair, one real TRAIN frame, no update.',
        source_statistics='Initial and final diverse10 evaluations collect fresh actual r0/r1 source occurrences and T10/two-anchor NRV; no accumulating training-source trace.',
        numeric='NativeFP32 network; explicit independentL subtraction/addition is an accuracy-only reassociation, not a deployed sparse schedule.',
        prior_scope='Fixed recovery control, not full EigenDamage/UPSCALE, no new mechanism or speed claim.',
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32, TF32_cudnn=torch.backends.cudnn.allow_tf32,
        temporal_and_backward_matmul_TF32=False, axes={})
    save_json(args.output/'result.json', run)
    controller = None
    gate_handle = None
    def forward_flow(x):
        functional.reset_net(model)
        current.pop('flow', None)
        current.pop('differentiable_flow', None)
        controller.terms.clear()
        try:
            try:
                model(x)
            except CoarseReady:
                flow = current.pop('differentiable_flow')
                current.pop('flow', None)
                return F.interpolate(flow, (480, 640), mode='bilinear', align_corners=False)
            raise RuntimeError('Expected the existing real coarse-flow exit.')
        finally:
            controller.terms.clear()
    try:
        controller = RecoveryControl(modules, arrays)
        controller.initial_parameters = initial_parameters
        for axis in AXES:
            torch.manual_seed(SEED)
            controller.initialize_axis(axis, bias_arrays[axis])
            reference = LatentPair(parent_arrays, conv1.weight.device, conditional=False)
            student = TrainableLatentPair(parent_arrays, conv1.weight.device)
            student.u.requires_grad_(False)
            student.v.requires_grad_(False)
            row = dict(complete=False, initial_bias_axis=BIAS_AXIS[axis], updates=0,
                       parameter_counts=controller.parameter_counts(), history=[])
            with torch.no_grad():
                key = 'zero_W' if axis == 'source_group_zero' else 'nullspace_W' if axis == 'consumer_nullspace' else 'original_W'
                row['initial_W_vs_saved_max_abs'] = {label: float((controller.effective_weight(label)
                    -torch.as_tensor(arrays[label+'_'+key], device=conv1.weight.device)).abs().max()) for label in ('r0', 'r1')}
            run['axes'][axis] = row
            observed_gate = {}
            def gate_observer(module, inputs, output):
                value = output.detach()
                observed_gate['shape'] = list(value.shape)
                observed_gate['bits'] = np.packbits(value.ne(0).cpu().numpy(), bitorder='little')
                observed_gate['theta'] = float(student.temporal.theta)
            gate_handle = sn2.register_forward_hook(gate_observer)
            x, label, valid = input_frame(args.data, train[0])
            conv1.forward, sn2.forward = reference.conv_forward, reference.neuron_forward
            with torch.no_grad():
                expected = forward_flow(x)
            expected_gate = observed_gate.pop('bits')
            conv1.forward, sn2.forward = student.conv_forward, student.neuron_forward
            prediction = forward_flow(x)
            gate_xor = observed_gate.pop('bits')^expected_gate
            gate_differences = int(np.unpackbits(gate_xor).sum())
            difference = (prediction.detach()-expected).abs()
            error = prediction.permute(0, 2, 3, 1)[valid]-label.permute(0, 2, 3, 1)[valid]
            loss = torch.sqrt(error.square().sum(-1)+1e-6).mean()
            check = dict(file=train[0], optimizer_updates=0, gate_shape=observed_gate['shape'],
                theta=observed_gate['theta'], gate_differences=gate_differences,
                flow_differences=int(torch.count_nonzero(difference)), flow_max_abs=float(difference.max()),
                loss=float(loss.detach()), loss_requires_grad=loss.requires_grad, valid_pixels=int(valid.sum()))
            if loss.requires_grad:
                with fp32_matmul():
                    loss.backward()
            check['gradients'] = controller.gradients()
            check['pass'] = (not check['gate_differences'] and not check['flow_differences']
                and all(g.get('present') and g.get('all_finite') and g.get('nonzero', 0)>0 for g in check['gradients'].values()))
            row['check'] = check
            row['initial_constraints'] = controller.constraints()
            save_json(args.output/'result.json', run)
            print('CONSUMER_RECOVERY_CHECK', axis, json.dumps(check), flush=True)
            gate_handle.remove()
            gate_handle = None
            del x, label, valid, expected, expected_gate, prediction, difference, error, loss
            for parameter in controller.named_trainables().values():
                parameter.grad = None
            release_model_graphs(modules, current, controller)
            if not check['pass']:
                raise RuntimeError('Real training-frame forward/gradient check failed; no update applied for '+axis)
            if args.check_only:
                row['complete'] = True
                save_json(args.output/'result.json', run)
                continue
            # In particular independentL's full-PW subtraction and sparseL
            # addition have their own floating-point order. Measure this exact
            # initialized function rather than reuse the old projection-only
            # zero-step AEE. These results never select a checkpoint or budget.
            initial_name = axis+'_initial'
            controller.collect_activity = True
            controller.counts = {name: [] for name in ('r0', 'r1')}
            row['initial_evaluation'] = evaluate_axis(args, model, current, names, initial_name,
                progress_tag='CONSUMER_RECOVERY_INITIAL_AEE')
            initial_source = initial_name+'_source_activity.json'
            save_json(args.output/initial_source, controller.source_report(names))
            row['initial_source_activity_file'] = initial_source
            controller.collect_activity = False
            controller.counts = {name: [] for name in ('r0', 'r1')}
            release_model_graphs(modules, current, controller)
            save_json(args.output/'result.json', run)
            optimizer = torch.optim.Adam(list(controller.named_trainables().values()), lr=LR)
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            for step, filename in enumerate(schedule):
                x, label, valid = input_frame(args.data, filename)
                optimizer.zero_grad(set_to_none=True)
                prediction = forward_flow(x)
                error = prediction.permute(0, 2, 3, 1)[valid]-label.permute(0, 2, 3, 1)[valid]
                loss = torch.sqrt(error.square().sum(-1)+1e-6).mean()
                if not loss.requires_grad or not bool(torch.isfinite(loss)):
                    raise RuntimeError('Real recovery loss has no finite graph: '+axis)
                with fp32_matmul():
                    loss.backward()
                gradients = controller.gradients()
                if not all(g.get('present') and g.get('all_finite') for g in gradients.values()):
                    raise RuntimeError('Missing/nonfinite recovery gradient: '+axis)
                entry = dict(step=step+1, file=filename, loss_before_update=float(loss.detach()),
                    AEE_before_update=float(torch.linalg.vector_norm(error.detach(), dim=-1).double().mean()),
                    valid_pixels=int(valid.sum()))
                if step in (0, STEPS-1):
                    entry['gradients'] = gradients
                optimizer.step()
                row['history'].append(entry)
                row['updates'] = step+1
                if step == 0 or (step+1)%8 == 0:
                    save_json(args.output/'result.json', run)
                    print('CONSUMER_RECOVERY', axis, step+1, json.dumps(entry), flush=True)
                del x, label, valid, prediction, error, loss
                release_model_graphs(modules, current, controller)
            row['training_wall_seconds'] = time.monotonic()-started
            row['training_peak_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
            row['final_constraints'] = controller.constraints()
            np.savez_compressed(args.output/(axis+'.npz'), **controller.export(bias_arrays[axis], schedule))
            row['parameters_file'] = axis+'.npz'
            controller.collect_activity = True
            controller.counts = {name: [] for name in ('r0', 'r1')}
            row['evaluation'] = evaluate_axis(args, model, current, names, axis,
                                              progress_tag='CONSUMER_RECOVERY_AEE')
            save_json(args.output/(axis+'_source_activity.json'), controller.source_report(names))
            controller.collect_activity = False
            row['source_activity_file'] = axis+'_source_activity.json'
            row['complete'] = True
            save_json(args.output/'result.json', run)
            del optimizer, reference, student
            release_model_graphs(modules, current, controller)
            functional.reset_net(model)
            gc.collect()
            torch.cuda.empty_cache()
        run['complete'] = True
        save_json(args.output/'result.json', run)
    finally:
        if gate_handle is not None:
            gate_handle.remove()
        flow_hook.remove()
        common_hook.remove()
        if controller is not None:
            controller.restore()
        conv1.forward, sn2.forward = original_conv1, original_sn2
        current.pop('flow', None)
        current.pop('differentiable_flow', None)
    print('DONE', json.dumps({axis: dict(updates=row['updates'], check_pass=row['check']['pass'])
                              for axis, row in run['axes'].items()}), flush=True)


if __name__ == '__main__':
    main()
