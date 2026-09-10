"""CPU-only relations between r1.sn1 and the actual proj.sn time matrices.

Static-A least squares and explicitly in-sample train4 identity least squares
are separate axes. No Conv2/residual or actual post-residual answer selects M.
All M are fixed across frames, channels and space. Native sn1 membranes are
used with folded bias/center corrections. Identity-only gate agreement is
diagnostic, never a skip permission. The SpikingPEDLayer continuous even/even
consumer remains live regardless of the projection-neuron gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def parameters(path, label):
    with np.load(path) as data:
        prefix = label+'_'
        result = {key[len(prefix):]: data[key].copy() for key in data.files if key.startswith(prefix)}
    result['A'] = result['A'].astype(np.float64)
    result['b'] = result['b'].astype(np.float64).reshape(10)
    center = np.asarray(result['center'], np.float64).reshape(-1)
    result['effective_center'] = np.broadcast_to(center, (10,)).copy() if str(result['center_mode']) != 'zero' else np.zeros(10)
    result['offset'] = result['b']-result['effective_center']
    result['theta'] = float(result['theta'])
    if int(result['temporal_factor_rank']) or int(result['T']) != 10:
        raise ValueError('This probe is for the captured dense T10 neurons.')
    return result


def fire(p, membrane):
    positive = membrane >= p['theta']
    if str(p['output_mode']) == 'binary':
        if str(p['threshold_mode']) == 'symmetric_binary_abs':
            positive |= membrane <= -p['theta']
        return positive.astype(np.int8)
    scale = 1. if str(p['threshold_mode']) in ('symmetric_bsa_tsn', 'symmetric_target_rate') else float(p['negative_threshold_scale'])
    return positive.astype(np.int8)-(membrane <= -p['theta']*scale).astype(np.int8)


def flatten(values):
    return values.reshape(10, -1).astype(np.float64)


def records(folder):
    result = []
    for filename in sorted(folder.glob('[0-9][0-9]_*.npz')):
        with np.load(filename) as data:
            selected = {key: data[key].copy() for key in (
                'identity', 'r1_sn1_native_membrane', 'r1_sn1_input',
                'r1_sn1_output', 'identity_only_default_output', 'positions', 'anchor_mask', 'frame_name', 'split')}
        selected['file'] = selected.pop('frame_name')
        if str(selected['split']) != 'train':
            raise ValueError('Train-identity fit accepts explicit train captures only.')
        if selected['identity'].shape != (10, 96, 64, 4):
            raise ValueError('Expected actual native T10,C96,G64,P4 capture.')
        selected['path'] = str(filename)
        result.append(selected)
    if len(result) != 4:
        raise ValueError('Expected exactly the declared first four train16 frames.')
    return result


def fit_matrices(a1, ap, second_moment):
    """Origin-preserving LS for linear parts; known offsets are never fit."""
    cross = ap @ second_moment @ a1.T
    denominator = np.einsum('is,st,it->i', a1, second_moment, a1)
    target_energy = np.einsum('is,st,it->i', ap, second_moment, ap)
    scales = np.divide(cross, denominator[None], out=np.zeros_like(cross), where=denominator[None] > 0)
    losses = np.maximum(target_energy[:, None]-2*scales*cross+scales**2*denominator[None], 0)
    best = np.argmin(losses, axis=1)
    selected = np.zeros((10, 10))
    selected[np.arange(10), best] = scales[np.arange(10), best]
    diagonal = np.diag(np.diag(scales))
    return dict(identity=np.eye(10), diagonal_scale=diagonal, one_row_scalar=selected)


def route_cost(m, offset):
    nz = m != 0
    row_counts = nz.sum(1)
    kept = int(nz.any(0).sum())
    return dict(nonzero_coefficients=int(nz.sum()),
        plain_nontrivial_scalar_multiplications=int((nz & (np.abs(m) != 1)).sum()),
        unit_negative_sign_changes=int((m == -1).sum()),
        reduction_additions=int(np.maximum(row_counts-1, 0).sum()),
        folded_offset_additions=int((offset != 0).sum()),
        maximum_matrix_row_sources=int(row_counts.max()),
        retained_native_sn1_rows=kept, maximum_source_row_fanout=int(nz.sum(0).max()),
        FP32_native_m1_payload_P4_H96_bytes=kept*4*96*4,
        FP32_native_m1_payload_P4_H8_bytes=kept*4*8*4,
        identity_full_T10_P4_H96_FP32_bytes=10*4*96*4,
        coefficient_FP32_payload_bytes=int(nz.sum())*4,
        offset_FP32_payload_bytes=40,
        coefficient_scope='M only; selector indices/ports/routing and source/identity lifecycle are extra. Exact coefficient 0/±1 simplification allowed. Ordinary static row sharing gets identical rights.',
        instruction_scope='Counts per position/channel, not cycles. Dense inverse pays a complete10x10 transform:100 scalar multiplications and90 reduction additions before offset (subject to any literal zeros/units).')


def metrics(error, target, predicted_gates, target_gates, anchor):
    n = error.size
    difference = predicted_gates != target_gates
    positive, actual_positive = predicted_gates != 0, target_gates != 0
    grouped = difference.reshape(10, 12, 8, 64, 4)
    strata = {}
    for name, mask in (('continuous_anchor_positions', anchor), ('neuron_only_positions', ~anchor)):
        select = np.broadcast_to(mask[None, None], difference.shape)
        strata[name] = dict(gates=int(select.sum()), gate_differences=int((difference & select).sum()))
    return dict(values=n, error_sum=float(error.sum()), error_squared_sum=float(np.square(error).sum()),
        target_squared_sum=float(np.square(target).sum()), max_absolute_error=float(np.abs(error).max()),
        row_error_squared_sum=np.square(error).reshape(10, -1).sum(1).tolist(),
        row_target_squared_sum=np.square(target).reshape(10, -1).sum(1).tolist(),
        row_max_absolute_error=np.abs(error).reshape(10, -1).max(1).tolist(),
        gate_differences=int(difference.sum()), target_positive=int(actual_positive.sum()),
        false_positive=int((positive & ~actual_positive).sum()), false_negative=int((~positive & actual_positive).sum()),
        row_gate_differences=difference.reshape(10, -1).sum(1).tolist(),
        P4_H8_T10_groups=12*64,
        P4_H8_T10_equal_groups=int((~grouped.any((0, 2, 4))).sum()), strata=strata)


def summarize(rows):
    summed = ('values', 'error_sum', 'error_squared_sum', 'target_squared_sum', 'gate_differences',
        'target_positive', 'false_positive', 'false_negative', 'P4_H8_T10_groups', 'P4_H8_T10_equal_groups')
    result = {key: sum(row[key] for row in rows) for key in summed}
    result['max_absolute_error'] = max(row['max_absolute_error'] for row in rows)
    for key in ('row_error_squared_sum', 'row_target_squared_sum', 'row_gate_differences'):
        result[key] = np.asarray([row[key] for row in rows]).sum(0).tolist()
    result['row_max_absolute_error'] = np.asarray([row['row_max_absolute_error'] for row in rows]).max(0).tolist()
    result['RMSE'] = (result['error_squared_sum']/result['values'])**.5
    result['mean_error'] = result['error_sum']/result['values']
    result['relative_RMS_error'] = (result['error_squared_sum']/max(result['target_squared_sum'], 1e-300))**.5
    result['gate_difference_fraction'] = result['gate_differences']/result['values']
    result['false_negative_fraction_of_target_positive'] = result['false_negative']/max(result['target_positive'], 1)
    result['always_zero_default_gate_differences'] = result['target_positive']
    result['P4_H8_T10_equal_fraction'] = result['P4_H8_T10_equal_groups']/result['P4_H8_T10_groups']
    result['per_output_RMSE'] = np.sqrt(np.asarray(result['row_error_squared_sum'])/(result['values']/10)).tolist()
    result['strata'] = {name: {key: sum(row['strata'][name][key] for row in rows) for key in ('gates', 'gate_differences')}
        for name in rows[0]['strata']}
    return result


def analyze_route(m, label, fit_scope, a1, ap, p1, pp, captures):
    matrix_error = ap-m @ a1
    folded_offset = pp['offset']-m @ p1['offset']
    row_relative = np.linalg.norm(matrix_error, axis=1)/np.maximum(np.linalg.norm(ap, axis=1), 1e-300)
    rows = []
    for capture in captures:
        shape = capture['identity'].shape
        x = flatten(capture['identity'])
        source_native = flatten(capture['r1_sn1_native_membrane'])
        target = (ap @ x+pp['offset'][:, None]).reshape(shape)
        prediction = (m @ source_native+folded_offset[:, None]).reshape(shape)
        predicted_gates, target_gates = fire(pp, prediction), fire(pp, target)
        position = capture['positions']
        anchor = ((position//320) % 2 == 0) & ((position % 320) % 2 == 0)
        row = metrics(prediction-target, target, predicted_gates, target_gates, anchor)
        row.update(file=str(capture['file']),
            gate_differences_vs_captured_CPU_FP32_identity_only=int(np.count_nonzero(predicted_gates != np.sign(capture['identity_only_default_output']))),
            source_native_roundoff_contribution_max_abs=float(np.max(np.abs(m @ (source_native-(a1 @ x+p1['offset'][:, None]))))))
        rows.append(row)
    aggregate = summarize(rows)
    aggregate['gate_differences_vs_captured_CPU_FP32_identity_only'] = sum(row['gate_differences_vs_captured_CPU_FP32_identity_only'] for row in rows)
    return dict(label=label, fit_scope=fit_scope, M=m.tolist(), folded_offset=folded_offset.tolist(),
        matrix_difference_frobenius=float(np.linalg.norm(matrix_error)),
        matrix_relative_frobenius_error=float(np.linalg.norm(matrix_error)/np.linalg.norm(ap)),
        matrix_relative_frobenius_squared_error=float(np.square(matrix_error).sum()/np.square(ap).sum()),
        matrix_per_output_relative_error=row_relative.tolist(),
        selected_source_rows=[np.flatnonzero(row).tolist() for row in m],
        maximum_absolute_matrix_coefficient_error=float(np.abs(matrix_error).max()),
        arithmetic_and_state=route_cost(m, folded_offset), frames=rows, aggregate=aggregate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, default=HERE/'capture_train4_ped')
    parser.add_argument('--output', type=Path, default=HERE/'relation_result.json')
    args = parser.parse_args()
    p1 = parameters(args.capture/'neuron_parameters.npz', 'r1_sn1')
    pp = parameters(args.capture/'neuron_parameters.npz', 'proj_sn')
    a1, ap = p1['A'], pp['A']
    captured = records(args.capture)
    second = np.zeros((10, 10))
    columns = 0
    native_checks = []
    for record in captured:
        x = flatten(record['identity'])
        second += x @ x.T
        columns += x.shape[1]
        native = record['r1_sn1_native_membrane'].astype(np.float64)
        error = (a1 @ x+p1['offset'][:, None]).reshape(native.shape)-native
        source_codes = fire(p1, native)
        target_default = fire(pp, (ap @ x+pp['offset'][:, None]).reshape(native.shape))
        positions = record['positions']
        anchor = ((positions//320) % 2 == 0) & ((positions % 320) % 2 == 0)
        native_checks.append(dict(file=str(record['file']), split=str(record['split']),
            identity_equals_native_sn1_input=bool(np.array_equal(record['identity'], record['r1_sn1_input'])),
            FP64_reconstruction_vs_native_sn1_max_abs=float(np.abs(error).max()),
            native_sn1_membrane_to_output_gate_differences=int(np.count_nonzero(source_codes != np.sign(record['r1_sn1_output']))),
            native_sn1_output_theta_amplitude_differences=int(np.count_nonzero(source_codes*p1['theta'] != record['r1_sn1_output'])),
            FP64_identity_only_vs_captured_CPU_FP32_gate_differences=int(np.count_nonzero(target_default != np.sign(record['identity_only_default_output']))),
            sampled_positions=int(positions.size), continuous_anchor_positions=int(anchor.sum()),
            derived_vs_captured_anchor_differences=int(np.count_nonzero(anchor != record['anchor_mask'])),
            continuous_anchor_fraction=float(anchor.mean())))
    second /= columns
    routes = {}
    for scope, covariance in [('static_A_unweighted', np.eye(10)), ('train4_identity_second_moment', second)]:
        for name, m in fit_matrices(a1, ap, covariance).items():
            # Identity has no fitted parameter: retain one result only.
            if name == 'identity' and scope != 'static_A_unweighted':
                continue
            label = scope+'/'+name
            fit_scope = ('A matrices only, unweighted coefficient LS; no data fitting' if scope == 'static_A_unweighted' else
                'In-sample first4 approved training identities, global origin-preserving LS for linear parts. No residual/post-residual output enters selection; no held-out generalization claim.')
            routes[label] = analyze_route(m, label, fit_scope, a1, ap, p1, pp, captured)
    rank = int(np.linalg.matrix_rank(a1))
    dense = np.linalg.solve(a1.T, ap.T).T if rank == 10 else ap @ np.linalg.pinv(a1)
    routes['dense_static_inverse_control'] = analyze_route(dense, 'dense_static_inverse_control',
        'Static mathematical reachability control, complete dense10x10 transform charged; inverse' if rank == 10 else
        'Static pseudoinverse least-squares control; source matrix is not fullrank and exact reachability is not assumed',
        a1, ap, p1, pp, captured)
    result = dict(complete=True, scope='CPU relation diagnostics only. No training, GPU, AEE, skip permission or cycles.',
        input=str(args.capture), source_files=[record['path'] for record in captured],
        parameter_source=str(args.capture/'neuron_parameters.npz'),
        source_rank=rank, source_condition_number=float(np.linalg.cond(a1)), target_rank=int(np.linalg.matrix_rank(ap)),
        source_A=a1.tolist(), target_A=ap.tolist(),
        source_bias=p1['b'].tolist(), source_effective_center=p1['effective_center'].tolist(),
        target_bias=pp['b'].tolist(), target_effective_center=pp['effective_center'].tolist(),
        source_theta=p1['theta'], target_theta=pp['theta'],
        train_identity_vectors=columns, train_identity_second_moment=second.tolist(),
        native_checks=native_checks, routes=routes,
        target_definition='Counterfactual identity-only proj membrane Ap*x+bp-effective_centerp, computed FP64 from actual x; not captured post-residual proj membrane.',
        prediction_definition='M*native_sn1_membrane + [bp-effective_centerp - M*(b1-effective_center1)]. Known offsets are folded offline, never fitted. Theta is retained only for the actual firing rule/output amplitude.',
        numeric_boundary='Native CUDA sn1 may differ from FP64 reassociation. Dense mathematical matrix equality does not promise bit equality after native FP32 rounding. All gate results refer to identity-only defaults, not residual-completion permits.',
        graph_boundary='Actual SpikingPEDLayer consumes r1out also via1x1stride2pad0 continuous projection. Full-grid even/even fraction is1/4; actual sampled anchor fraction is separately recorded. These positions remain continuous consumers and parity grouping is also an ordinary compiler control.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    lines = ['# 两个真实 T10 神经元的廉价关系诊断', '',
        '只分析 A 与 identity-only 默认膜；没有拟合 Conv2 残差，没有许可或硬件加速比。train4 拟合结果是同样本诊断，不能当验证精度。', '',
        '| 拟合/映射 | 矩阵相对 Frobenius 差 | native 膜重用 RMSE | 默认门差 | 丢失目标非零比例 | 保存膜行数 |',
        '|---|---:|---:|---:|---:|---:|']
    for name, row in routes.items():
        metrics_ = row['aggregate']
        lines.append(f"| {name} | {row['matrix_relative_frobenius_error']:.6g} | {metrics_['RMSE']:.6g} | {metrics_['gate_differences']}/{metrics_['values']} | {metrics_['false_negative_fraction_of_target_positive']:.4%} | {row['arithmetic_and_state']['retained_native_sn1_rows']} |")
    lines.extend(['', f"源 A 秩 {rank}，条件数 {result['source_condition_number']:.6g}；目标 A 秩 {result['target_rank']}。稠密逆映射需完整 100 个标量乘法及归约/偏置，不能算便宜变换。", '',
        '预测显式使用 native sn1 膜以及折合的 bias/center 修正，目标是 CPU FP64 的 Aproj(identity) 默认膜；与真实 residual 后的 proj 输出不是同一个对象。全部来源和 native 数值误差见 JSON。', '',
        f"目标默认门中 {next(iter(routes.values()))['aggregate']['target_positive']} 个非零；全零默认本来就会得到很高的总体一致率。因此低总门差不能掩盖丢失大多数非零门。当前固定廉价映射没有显示可直接代替 Ap(identity) 的关系；若后续联合约束两套 A，属于新的训练学生，仍需精度、许可和费用验证。", '',
        '实际 SpikingPEDLayer 的偶数行/偶数列还有连续消费者；即使默认门相同也不构成取消 Conv2 的许可。必须再证明残差界、依赖、相同存储/端口下的净费用及新学生 AEE。'])
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    for name, row in routes.items():
        print(name, 'matrix_relative', row['matrix_relative_frobenius_error'], 'native_RMSE', row['aggregate']['RMSE'],
              'default_gate_diff', row['aggregate']['gate_differences'], flush=True)
    print('DONE', args.output, flush=True)


if __name__ == '__main__':
    main()
