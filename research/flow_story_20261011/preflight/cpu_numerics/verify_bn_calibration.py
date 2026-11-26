#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import copy
import hashlib
import json
from pathlib import Path
import torch
from bn_calibration import ChannelMoments, BNCalibrationCapture
from native_mlp_probe import make_mlp, no_running, actual_local_forward, expanded_mask, masks_for


def reject(fn, exceptions=(ValueError, RuntimeError)):
    try:
        fn()
    except exceptions:
        return True
    raise AssertionError('Expected rejection: ' + repr(fn))


def snapshot(model):
    return {
        'state': {name: value.detach().clone() for name, value in model.state_dict().items()},
        'modes': {name: module.training for name, module in model.named_modules()},
        'bn': {name: (module.track_running_stats, module.running_mean is None,
                      module.running_var is None, module.num_batches_tracked is None)
               for name, module in model.named_modules()
               if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)}}


def unchanged(model, before):
    after = snapshot(model)
    return (after['modes'] == before['modes'] and after['bn'] == before['bn'] and
            after['state'].keys() == before['state'].keys() and
            all(torch.equal(value, after['state'][name]) for name, value in before['state'].items()))


def no_hooks(model):
    return all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def ordinary():
    return torch.nn.Sequential(torch.nn.BatchNorm2d(3)).eval()


def capture_one(model):
    with BNCalibrationCapture(model, ['train0'], ['eval0'], sample_batch_axis=0) as capture:
        capture.begin_sample('train0')
        model(torch.randn(1, 3, 4, 5))
    return capture


def guard_checks():
    checks = {}
    checks['evaluation_set_overlap_rejected'] = reject(lambda: BNCalibrationCapture(
        ordinary(), ['train0', 'eval0'], ['eval0'], sample_batch_axis=0))
    checks['batch_axis_must_be_explicit'] = reject(
        lambda: BNCalibrationCapture(ordinary(), ['train0'], ['eval0']), (TypeError,))
    for bad_id in ('eval0', 'unknown'):
        model = ordinary()
        with BNCalibrationCapture(model, ['train0'], ['eval0'], sample_batch_axis=0) as cap:
            reject(lambda: cap.begin_sample(bad_id))
        checks[bad_id + '_sample_rejected_and_install_blocked'] = cap.failed and reject(
            lambda: cap.install(copy.deepcopy(model)))
    model = ordinary()
    with BNCalibrationCapture(model, ['train0'], ['eval0'], sample_batch_axis=0) as cap:
        reject(lambda: model(torch.randn(1, 3, 2, 2)))
    checks['forward_without_begin_rejected'] = cap.failed
    model = ordinary()
    with BNCalibrationCapture(model, ['train0'], ['eval0'], sample_batch_axis=0) as cap:
        cap.begin_sample('train0')
        model(torch.randn(1, 3, 2, 2))
        reject(lambda: model(torch.randn(1, 3, 2, 2)))
    checks['unlabelled_second_forward_rejected'] = cap.failed and cap.moments['0'].calls == 1
    checks['failed_second_forward_blocks_install'] = reject(lambda: cap.install(copy.deepcopy(model)))
    model = ordinary()
    with BNCalibrationCapture(model, ['train0'], ['eval0'], sample_batch_axis=0) as cap:
        cap.begin_sample('train0')
        model(torch.randn(1, 3, 2, 2))
        reject(lambda: cap.begin_sample('train0'))
    checks['repeated_target_rejected'] = cap.failed
    model = ordinary()
    with BNCalibrationCapture(model, ['train0', 'train1'], ['eval0'], sample_batch_axis=0) as cap:
        cap.begin_sample('train0')
        reject(lambda: cap.begin_sample('train1'))
    checks['overwriting_pending_target_rejected'] = cap.failed
    model = ordinary()
    with BNCalibrationCapture(model, ['train0'], ['eval0'], sample_batch_axis=0) as cap:
        cap.begin_sample('train0')
    checks['begin_without_forward_blocks_install'] = cap.failed and reject(lambda: cap.install(copy.deepcopy(model)))
    model = ordinary()
    with BNCalibrationCapture(model, ['train0'], ['eval0'], sample_batch_axis=0) as cap:
        cap.begin_sample('train0')
        reject(lambda: model(torch.randn(2, 3, 2, 2)))
    checks['batch2_rejected_before_BN_capture'] = cap.failed and cap.moments['0'].calls == 0
    model = ordinary()
    with BNCalibrationCapture(model, ['train0'], ['eval0'], sample_batch_axis=0) as cap:
        cap.begin_sample('train0')
        reject(lambda: model[0](torch.randn(1, 3, 2, 2)))
    checks['direct_child_BN_outside_root_rejected'] = cap.failed and cap.moments['0'].calls == 0

    class FailingRoot(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm2d(3)
        def forward(self, x):
            self.bn(x)
            raise RuntimeError('Synthetic downstream failure')
    model = FailingRoot().eval()
    with BNCalibrationCapture(model, ['train0'], ['eval0'], sample_batch_axis=0) as cap:
        cap.begin_sample('train0')
        reject(lambda: model(torch.randn(1, 3, 2, 2)))  # exception caught INSIDE context
    checks['caught_forward_exception_blocks_install'] = (cap.failed and not cap.seen and
        cap.moments['bn'].calls == 1 and reject(lambda: cap.install(copy.deepcopy(model))))
    checks['exception_cleanup_removes_hooks'] = no_hooks(model)

    class SkippingRoot(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn0, self.bn1 = torch.nn.BatchNorm2d(3), torch.nn.BatchNorm2d(3)
        def forward(self, x):
            return self.bn0(x)
    model = SkippingRoot().eval()
    cap = capture_one(model)
    checks['missing_BN_execution_blocks_install'] = cap.failed and reject(lambda: cap.install(copy.deepcopy(model)))
    model = torch.nn.Sequential(torch.nn.BatchNorm2d(3), torch.nn.BatchNorm1d(3)).eval()
    cap = BNCalibrationCapture(model, ['train0'], ['eval0'], sample_batch_axis=0)
    reject(cap.__enter__)
    checks['unsupported_later_BN_leaves_no_hooks'] = no_hooks(model)
    model = torch.nn.Sequential(torch.nn.BatchNorm2d(3), torch.nn.BatchNorm2d(3)).eval()
    original_register = model[1].register_forward_pre_hook
    def fail_register(*args, **kwargs):
        raise RuntimeError('Synthetic registration failure')
    model[1].register_forward_pre_hook = fail_register
    cap = BNCalibrationCapture(model, ['train0'], ['eval0'], sample_batch_axis=0)
    reject(cap.__enter__)
    model[1].register_forward_pre_hook = original_register
    checks['partial_registration_failure_leaves_no_hooks'] = no_hooks(model)
    model = torch.nn.Sequential(torch.nn.BatchNorm2d(3)).train()
    cap = BNCalibrationCapture(model, ['train0'], ['eval0'], sample_batch_axis=0)
    checks['source_running_stats_mutation_policy_rejected'] = reject(cap.__enter__) and no_hooks(model)
    model = ordinary()
    cap = capture_one(model)
    before = snapshot(model)
    checks['install_source_rejected_without_mutation'] = reject(lambda: cap.install(model)) and unchanged(model, before)
    shared = torch.nn.Sequential(model[0])
    checks['install_shared_BN_rejected_without_mutation'] = reject(lambda: cap.install(shared)) and unchanged(model, before)
    target = copy.deepcopy(model)
    target.add_module('extra', torch.nn.BatchNorm2d(3))
    before = snapshot(target)
    checks['extra_deployment_BN_rejected_atomically'] = reject(lambda: cap.install(target)) and unchanged(target, before)
    target = copy.deepcopy(model)
    target[0].eps *= 2
    before = snapshot(target)
    checks['changed_eps_rejected_atomically'] = reject(lambda: cap.install(target)) and unchanged(target, before)
    model = torch.nn.Sequential(torch.nn.BatchNorm2d(3),
                                torch.nn.BatchNorm2d(3, affine=False, track_running_stats=False)).eval()
    cap = capture_one(model)
    target = copy.deepcopy(model)
    before = snapshot(target)
    checks['late_nonaffine_stateless_target_rejected_atomically'] = reject(
        lambda: cap.install(target)) and unchanged(target, before)
    model = ordinary()
    cap = capture_one(model)
    target = copy.deepcopy(model)
    before = snapshot(target)
    def fail_eval():
        target[0].training = True
        raise RuntimeError('Synthetic commit failure')
    target.eval = fail_eval
    checks['commit_exception_rolls_back_buffers_and_modes'] = reject(
        lambda: cap.install(target)) and unchanged(target, before)
    model = ordinary()
    cap = capture_one(model)
    cap.moments['0'].mean = torch.full_like(cap.moments['0'].mean, float('inf'))
    target = copy.deepcopy(model)
    before = snapshot(target)
    checks['nonfinite_installed_statistics_rejected_atomically'] = reject(
        lambda: cap.install(target)) and unchanged(target, before)
    for bad_value, label in ((float('nan'), 'NaN'), (float('inf'), 'Inf')):
        model = ordinary()
        with BNCalibrationCapture(model, ['train0'], ['eval0'], sample_batch_axis=0) as cap:
            cap.begin_sample('train0')
            reject(lambda: model(torch.full((1, 3, 2, 2), bad_value)))
        checks[label + '_input_blocks_install'] = cap.failed and reject(lambda: cap.install(copy.deepcopy(model)))
    assert all(checks.values()), checks
    return checks


def main():
    torch.set_num_threads(2)
    torch.manual_seed(31001)
    # Different counts AND means make naive averaging of batch variances fail.
    a = torch.randn(2, 3, 4, 5, dtype=torch.float64) - 100
    b = torch.randn(1, 3, 2, 7, dtype=torch.float64) + 70
    cap = ChannelMoments(max_chunk_elements=21)
    cap.update(a, 1)
    cap.update(b, 1)
    whole = torch.cat([a.movedim(1, 0).reshape(3, -1), b.movedim(1, 0).reshape(3, -1)], 1)
    mean_error = (cap.mean - whole.mean(1)).abs().max().item()
    variance_error = (cap.population_variance() - whole.var(1, correction=0)).abs().max().item()
    assert mean_error < 1e-10 and variance_error < 1e-10
    assert cap.calls == 2 and cap.chunks > cap.calls and cap.max_chunk_elements_seen <= 21
    naive_variance = (a.movedim(1, 0).reshape(3, -1).var(1, correction=0) +
                      b.movedim(1, 0).reshape(3, -1).var(1, correction=0)) / 2
    naive_error = (naive_variance - whole.var(1, correction=0)).abs().max().item()
    assert naive_error > 1000
    # Non-contiguous NativeSJ T,B,C,H,W, including T in the reduction.
    native = torch.randn(10, 1, 11, 13, 7).permute(0, 1, 4, 2, 3)
    assert not native.is_contiguous()
    native_cap = ChannelMoments(max_chunk_elements=101)
    native_cap.update(native, 2)
    native_mean_error = (native_cap.mean - native.double().mean((0, 1, 3, 4))).abs().max().item()
    native_var_error = (native_cap.population_variance() - native.double().var((0, 1, 3, 4), correction=0)).abs().max().item()
    assert native_mean_error < 1e-12 and native_var_error < 1e-12
    assert native_cap.count == 10 * 1 * 11 * 13 and native_cap.calls == 1
    assert native_cap.max_chunk_elements_seen <= 101 and native_cap.chunks > 1
    before_moments = (cap.count, cap.calls, cap.mean.clone(), cap.m2.clone())
    broken = a.clone()
    broken[:, :, -1, -1] = float('nan')
    reject(lambda: cap.update(broken, 1))
    atomic_update = (cap.count == before_moments[0] and cap.calls == before_moments[1] and
                     torch.equal(cap.mean, before_moments[2]) and torch.equal(cap.m2, before_moments[3]))
    assert atomic_update
    reject(lambda: ChannelMoments(max_chunk_elements=2).update(a, 1))
    with torch.inference_mode():
        guards = guard_checks()
        # P uses CURRENT full-spatial statistics, as native no_running eval does.
        reference = no_running(make_mlp(96, torch.float64))
        deployment = copy.deepcopy(reference)
        source_before = snapshot(reference)
        x = torch.randn(10, 1, 18, 21, 96, dtype=torch.float64)
        p_before = reference(x)
        sample_ids = ['synthetic_train_0', 'synthetic_train_1', 'synthetic_train_2']
        with BNCalibrationCapture(reference, sample_ids, ['synthetic_eval'],
                                  sample_batch_axis=1, max_chunk_elements=65536) as capture:
            for i, target_id in enumerate(sample_ids):
                capture.begin_sample(target_id)
                reference(torch.randn(10, 1, 18, 21, 96, dtype=torch.float64) + i * .2)
        calibration_report = capture.install(deployment)
        p_after, d_output = reference(x), deployment(x)
        source_unchanged = unchanged(reference, source_before) and torch.equal(p_before, p_after)
        assert source_unchanged and no_hooks(reference)
        assert torch.isfinite(d_output).all()
        gate = masks_for(18, 21, 9)['checkerboard']
        full_masked = x + d_output * expanded_mask(x, gate, 9)
        local, counts = actual_local_forward(deployment, x, gate, 9)
        tile_error = (full_masked - local).abs().max().item()
        assert torch.allclose(full_masked, local, atol=1e-11, rtol=1e-11)
        p_full_masked = x + p_after * expanded_mask(x, gate, 9)
        p_local, _ = actual_local_forward(reference, x, gate, 9)
        current_stat_tile_error = (p_full_masked - p_local).abs().max().item()
        assert current_stat_tile_error > 1e-4
        p_d_difference = {'max_abs_output_difference': (p_after - d_output).abs().max().item(),
                          'mean_abs_output_difference': (p_after - d_output).abs().mean().item(),
                          'interpretation': 'Synthetic MLP outputs only; not flow, AEE, or an accuracy gate'}
        for layer in calibration_report['layers'].values():
            assert layer['calls'] == 3 and layer['values_per_channel'] == 3 * 10 * 18 * 21
            assert layer['max_chunk_elements_seen'] <= 65536
            assert not layer['source_bn_policy']['track_running_stats']
        native_bad = no_running(make_mlp(96, torch.float32))
        with BNCalibrationCapture(native_bad, ['train0'], ['eval0'], sample_batch_axis=1) as bad_cap:
            bad_cap.begin_sample('train0')
            reject(lambda: native_bad(torch.randn(10, 2, 3, 3, 96)))
        guards['native_TB_B2_rejected_before_BN_capture'] = bad_cap.failed and all(
            moment.calls == 0 for moment in bad_cap.moments.values())
        assert all(guards.values())
    here = Path(__file__).parent
    result = {
        'scope': 'Synthetic CPU implementation checks; no dataset, training, or AEE',
        'status': 'PASS', 'torch_version': torch.__version__,
        'script_sha256': {name: hashlib.sha256((here / name).read_bytes()).hexdigest()
                          for name in ('bn_calibration.py', 'verify_bn_calibration.py')},
        'pooled_mean_max_error': mean_error, 'pooled_population_variance_max_error': variance_error,
        'naive_average_variance_max_error': naive_error,
        'chunked_native_noncontiguous': {
            'input_shape': list(native.shape), 'input_elements': native.numel(),
            'chunk_limit': 101, 'max_chunk_elements_seen': native_cap.max_chunk_elements_seen,
            'chunks': native_cap.chunks, 'BN_calls': native_cap.calls,
            'values_per_channel': native_cap.count, 'mean_max_error': native_mean_error,
            'population_variance_max_error': native_var_error},
        'failed_chunk_update_preserves_existing_moments': atomic_update,
        'guards': guards, 'source_current_stat_graph_unchanged': source_unchanged,
        'source_current_stat_tile_negative_control_max_error': current_stat_tile_error,
        'P_to_D_synthetic_output_difference': p_d_difference,
        'calibrated_native_mlp_tile_max_error': tile_error, 'actual_fc_calls': counts,
        'calibration_report': calibration_report, 'cuda_initialized': torch.cuda.is_initialized()}
    assert not result['cuda_initialized']
    (here / 'bn_calibration_test_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('calibration_report', 'guards')}))
    print('guard_checks_passed=' + str(len(guards)))


if __name__ == '__main__':
    main()
