"""Fixed train4 affine calibration of the shared temporal matrix; no SGD.

All axes use the preceding control3: original preview-only parent, Conv2R16,
nonanchor normalized-branch deletion and saved ordinary PEDR32. Full-domain
train4 moments of the actual proj.sn input x fit h=A_proj*x+b-center against
m=A_sn1*x, separately for each output time. Compare original, slope1 with
mean correction, and signed fitted slope with intercept. Preserve theta and
center. Replacing the native FP32 matrix by rounded diag(d)*A_sn1 is a changed
student, not a proof of exact shared-membrane execution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_shared_temporal_control import read_arrays, frame_comparison

AXES = ('ordinary_control3', 'tied_A_bias_corrected', 'row_affine_shared_A')


def merge_moments(left, right):
    """Chan merge of (count, mean, unnormalized centered second moment)."""
    if left is None:
        return right
    n, mean, moment = left
    m, other, another = right
    delta = other-mean
    return n+m, mean+delta*(m/(n+m)), moment+another+np.outer(delta, delta)*(n*m/(n+m))


@torch.no_grad()
def input_moments(value):
    """All C/spatial positions, all T10 coordinates; C8 FP64 chunks."""
    if value.ndim != 5 or tuple(value.shape[:2]) != (10, 1):
        raise ValueError('Expected native proj.sn input [T10,B1,C,H,W]')
    combined = None
    for channel in range(0, value.shape[2], 8):
        block = value.detach()[:, :, channel:channel+8].double().reshape(10, -1)
        mean = block.mean(1)
        centered = block-mean[:, None]
        moment = centered@centered.T
        combined = merge_moments(combined, (block.shape[1], mean.cpu().numpy(), moment.cpu().numpy()))
    return combined


def fit_rows(mean, covariance, source_A, original_A, original_bias, effective_center):
    """Exact unregularized scalar least squares from sufficient statistics."""
    source_A, original_A = np.asarray(source_A, np.float64), np.asarray(original_A, np.float64)
    teacher_offset = np.asarray(original_bias, np.float64).reshape(-1)-np.asarray(effective_center).reshape(-1)
    mu_m = source_A@mean
    mu_h = original_A@mean+teacher_offset
    vm = np.einsum('ti,ij,tj->t', source_A, covariance, source_A)
    vh = np.einsum('ti,ij,tj->t', original_A, covariance, original_A)
    cross = np.einsum('ti,ij,tj->t', original_A, covariance, source_A)
    # No low-variance threshold or ridge sweep. Negative roundoff variance is
    # represented as zero; an exactly constant m uses the constant predictor.
    vm, vh = np.maximum(vm, 0), np.maximum(vh, 0)
    constant = vm == 0
    slope = np.divide(cross, vm, out=np.zeros_like(vm), where=~constant)
    intercept = mu_h-slope*mu_m
    bias_intercept = mu_h-mu_m
    correlation = np.divide(cross, np.sqrt(vm*vh), out=np.zeros_like(vm), where=(vm*vh) > 0)
    residual_affine = np.maximum(vh+slope*slope*vm-2*slope*cross, 0)
    residual_bias = np.maximum(vh+vm-2*cross, 0)
    return dict(mean_m=mu_m, mean_teacher_membrane=mu_h,
        variance_m=vm, variance_teacher_membrane=vh, covariance_h_m=cross,
        correlation=correlation, correlation_defined=(vm*vh) > 0,
        zero_variance_constant_rows=constant, fitted_slope=slope, fitted_intercept=intercept,
        slope1_intercept=bias_intercept, affine_residual_variance=residual_affine,
        slope1_residual_variance=residual_bias,
        affine_residual_RMSE=np.sqrt(residual_affine), slope1_residual_RMSE=np.sqrt(residual_bias),
        old_unadjusted_tied_A_membrane_MSE=residual_bias+(mu_h-mu_m-teacher_offset)**2,
        slope1_native_bias=bias_intercept+effective_center,
        affine_native_bias=intercept+effective_center,
        affine_native_A=slope[:, None]*source_A)


def self_check():
    """Domain/count and independent direct-regression checks, entirely CPU."""
    torch.set_num_threads(2)
    rng = np.random.default_rng(961014)
    blocks = [torch.from_numpy(rng.normal(size=(10, 1, 3, 4, 5))),
              torch.from_numpy(rng.normal(2, 3, size=(10, 1, 3, 4, 7)))]
    count, mean, moment = merge_moments(input_moments(blocks[0]), input_moments(blocks[1]))
    x = np.concatenate([block.numpy().reshape(10, -1) for block in blocks], axis=1)
    assert count == x.shape[1]
    assert np.allclose(mean, x.mean(1), rtol=0, atol=2e-15)
    cov = moment/count
    assert np.allclose(cov, np.cov(x, bias=True), rtol=0, atol=1e-13)
    source = rng.normal(size=(10, 10)); source[0] = 0
    original = rng.normal(size=(10, 10)); original[1] = -2*source[1]
    bias, center = rng.normal(size=10), rng.normal(size=10)
    fit = fit_rows(mean, cov, source, original, bias, center)
    m, teacher = source@x, original@x+(bias-center)[:, None]
    errors = []
    for t in range(10):
        expected = np.linalg.lstsq(np.column_stack((m[t], np.ones(count))), teacher[t], rcond=None)[0]
        errors.extend(np.abs(expected-np.array([fit['fitted_slope'][t], fit['fitted_intercept'][t]])))
    assert max(errors) < 1e-12
    assert fit['fitted_slope'][0] == 0 and abs(fit['fitted_slope'][1]+2) < 1e-12
    assert np.allclose(fit['affine_native_bias']-center, fit['fitted_intercept'])
    print(json.dumps(dict(CPU_check='PASS', vectors=count, direct_lstsq_max_abs=float(max(errors)),
        cases='Full C/spatial/T moments, unequal-count merge, signed slope, constant row, bias/center conversion; synthetic math only, no real network calibration.')))


class AtProjection(Exception):
    pass


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--train-list', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.root is None:
        parser.error('--root is required for network calibration/evaluation')
    args.split, args.count = 'diverse', 10
    alg, area = args.root/'algorithm', args.root/'algorithm/patch_probe'
    residual = area/'residual_consumer_probe'
    chain = residual/'projection_chain'
    latent = area/'factor_completion_20260909/latent_stage_train16'
    args.train_list = args.train_list or latent/'flow_train_list.json'
    train = json.loads(args.train_list.read_text())
    train = (train['train'] if isinstance(train, dict) else train)[:4]
    args.output = args.output or chain/'affine_shared_temporal_control_diverse10'
    args.output.mkdir(parents=True, exist_ok=True)
    for path in (alg, alg/'nrv_cost_probe', latent, residual):
        sys.path.insert(0, str(path))
    import run_probe as probe
    from run_bn_probe import input_frame
    from adapter import install_latent_factor
    from capture import BLOCK, SOURCE_SN, PROJECT, CONSUMER_SN, neuron_parameters
    from capture_chain import save_json
    from evaluate_rank_control import RankConvControl
    from evaluate_branch_control import evaluate_axis, mask_nonanchors
    from spikingjelly.activation_based import functional

    system = probe.load_system(args)
    model, modules, _, current, sources, _, _, _ = system
    probe.install_sources(system, sources)
    current['count_codes'] = False
    fixed = torch.load(area/'patch_train_calibration.pt', map_location='cpu', weights_only=False)
    for name, values in fixed.items():
        bn = modules[name]
        bn.track_running_stats = True
        bn.running_mean, bn.running_var = values['mean'].to(bn.weight), values['var'].to(bn.weight)
    model.eval()
    previous = chain/'shared_temporal_control_diverse10'
    runtime = json.loads((previous/'run.json').read_text())
    torch.backends.cuda.matmul.allow_tf32 = bool(runtime['TF32_matmul'])
    torch.backends.cudnn.allow_tf32 = bool(runtime['TF32_cudnn'])
    parent = latent/'flow_recovery64/preview_only/shared48_u8_vq5.npz'
    conv1, sn2 = modules[BLOCK+'.conv1.0'], modules[BLOCK+'.sn2.spiking_neuron']
    _, originals = install_latent_factor(conv1, sn2, parent, conditional=False)
    conv2, projection, neuron = modules[BLOCK+'.conv2.0'], modules[PROJECT+'.conv_res'], modules[CONSUMER_SN]
    control = RankConvControl(conv2, projection, read_arrays(residual/'rank_control_parameters.npz'))
    control.axis = 'uniform_rank16'
    conv2.forward = control.forward
    original_projection = projection.forward
    ped = read_arrays(chain/'rank32_diverse10/parameters.npz')
    first = torch.as_tensor(ped['U'], device=projection.weight.device, dtype=projection.weight.dtype)[:, :, None, None]
    second = torch.as_tensor(ped['V'], device=projection.weight.device, dtype=projection.weight.dtype)[:, :, None, None]
    projection.forward = lambda x: F.conv2d(F.conv2d(x[:, :, ::2, ::2], first), second, projection.bias)
    masks = {}
    def delete_nonanchor(module, inputs, output):
        key = (output.shape[-2:], output.device)
        if key not in masks:
            mask = torch.zeros(output.shape[-2:], device=output.device, dtype=torch.bool)
            mask[::2, ::2] = True
            masks[key] = mask
        return mask_nonanchors(output, masks[key])
    delete_hook = modules[BLOCK+'.norm2'].register_forward_hook(delete_nonanchor)
    source_state, state = neuron_parameters(modules[SOURCE_SN]), neuron_parameters(neuron)
    source_A = source_state['A'].numpy().astype(np.float64)
    original_A, original_bias = neuron.weight.detach().clone(), neuron.bias.detach().clone()
    center = state['center'].numpy().reshape(-1).astype(np.float64)
    if state['center_mode'] == 'zero':
        center = np.zeros(10, np.float64)
    names = json.loads((alg/'samples.json').read_text())['valid'][:10]
    run = dict(complete=False, parent=str(parent), fixed_control='Previous control3: Conv2R16 + nonanchor whole BN branch deletion + PEDR32, original proj A/bias.',
        train_files=train, evaluation_files=names, train_list=str(args.train_list),
        calibration='Full actual proj.sn input, all C96 and H240xW320, jointly allT10 coordinates. FP64 centered C8 chunk moments, count-weighted across all4 frames. No validation fitting, ridge, SGD or low-variance threshold.',
        teacher='Mathematical FP64 A_proj*x+b_proj-effective_center on native FP32 x; not an assertion of native CUDA sum-order equality.',
        axis_definitions={AXES[0]:'Original control3 A/bias unchanged.',
            AXES[1]:'A_sn1 with d=1 and per-t train4 mean-error correction.',
            AXES[2]:'diag(d_t)*A_sn1 plus intercept from per-t signed scalar least squares; zero-variance m is a constant prediction.'},
        shared_amplitude='Source theta*g is unchanged; proj theta and stored center remain unchanged. Only proj A and bias change.',
        execution='All axes run native proj.sn. FP32 diag-scaled matrix is not proven equivalent to reused M plus threshold/compare; negative d changes comparison direction and d=0 is constant.',
        hardware_claim='None. Membrane state, source reuse and signed/dyadic threshold representation remain to be tested. No inherited825 result.',
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32, TF32_cudnn=torch.backends.cudnn.allow_tf32,
        moments_frames=[], results={})
    save_json(args.output/'run.json', run)
    frame_moments = []
    def observe(module, inputs):
        frame_moments.append(input_moments(inputs[0]))
        raise AtProjection()
    capture_hook = neuron.register_forward_pre_hook(observe)
    try:
        combined = None
        for index, filename in enumerate(train):
            functional.reset_net(model)
            current.pop('flow', None)
            x, _, _ = input_frame(args.data, filename, targets=False)
            try:
                model(x)
            except AtProjection:
                pass
            del x
            item = frame_moments[-1]
            combined = merge_moments(combined, item)
            run['moments_frames'].append(dict(file=filename, vectors=item[0], mean=item[1], covariance=item[2]/item[0]))
            print('AFFINE_CALIBRATION', index+1, filename, item[0], flush=True)
        capture_hook.remove()
        capture_hook = None
        count, mean, moment = combined
        covariance = moment/count
        fit = fit_rows(mean, covariance, source_A, original_A.cpu().numpy(), original_bias.cpu().numpy(), center)
        effective = {
            AXES[0]: (original_A.cpu().numpy(), original_bias.cpu().numpy()),
            AXES[1]: (source_A.astype(np.float32), fit['slope1_native_bias'].astype(np.float32).reshape(original_bias.shape)),
            AXES[2]: (fit['affine_native_A'].astype(np.float32), fit['affine_native_bias'].astype(np.float32).reshape(original_bias.shape)),
        }
        saved = dict(mean_x=mean, covariance_x=covariance, vectors=np.array(count),
            source_A=source_A, original_proj_A=original_A.cpu().numpy(), original_proj_bias=original_bias.cpu().numpy(),
            source_bias=source_state['b'].numpy(), source_center=source_state['center'].numpy(),
            source_theta=source_state['theta'].numpy(), source_center_mode=np.array(source_state['center_mode']),
            proj_center=state['center'].numpy(), effective_proj_center=center, proj_theta=state['theta'].numpy(),
            **fit)
        for axis, (a, b) in effective.items():
            saved[axis+'_A32'], saved[axis+'_bias32'] = a, b
        np.savez_compressed(args.output/'parameters.npz', **saved)
        run['fit'] = fit
        run['calibration_vectors'] = count
        run['calibration_scalar_inputs'] = count*10
        run['actual_parameter_ranks'] = {axis:int(np.linalg.matrix_rank(a.astype(np.float64))) for axis, (a, _) in effective.items()}
        save_json(args.output/'run.json', run)
        for axis, (a, b) in effective.items():
            neuron.weight.copy_(torch.as_tensor(a, device=neuron.weight.device))
            neuron.bias.copy_(torch.as_tensor(b, device=neuron.bias.device))
            run['results'][axis] = evaluate_axis(args, model, current, names, axis, progress_tag='AFFINE_SHARED_AEE')
            if axis == AXES[0]:
                run['ordinary_control3_alignment'] = frame_comparison(args.output/(axis+'_frames.json'),
                    previous/'rank16_delete_nonanchor_ped32_frames.json')
                run['ordinary_control3_alignment']['scope'] = 'Same preceding control3; no claim about the separate old R16/fullC825.'
            save_json(args.output/'run.json', run)
        run['complete'] = True
        save_json(args.output/'run.json', run)
    finally:
        if capture_hook is not None:
            capture_hook.remove()
        delete_hook.remove()
        neuron.weight.copy_(original_A)
        neuron.bias.copy_(original_bias)
        projection.forward = original_projection
        conv2.forward = control.original_forward
        conv1.forward, sn2.forward = originals['conv1_forward'], originals['neuron_forward']
        current.pop('flow', None)
    print('DONE', json.dumps({axis: value['AEE_frame_mean'] for axis, value in run['results'].items()}), flush=True)


if __name__ == '__main__':
    main()
