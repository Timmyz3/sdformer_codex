"""One fixed all-layers gate-collapse arithmetic screen; no GPU or search.

Actual q12 shears are composed with Fraction, without internal RNE/saturation.
The actual IEEE source gain and permutation are then folded into this rational
10x10 matrix. One matrix-wide signed16 dyadic quantization follows the existing
As rule. Official da4ml compiles the resulting integer matrix as a whole.

This approximate matrix is NOT the deployed lifting/RNE function. All graph
checks below prove only its own q16 integer dot. Guard, error-bound compares,
fallback and extra storage are optimistically absent in this first screen.
"""
from __future__ import annotations

import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'DA_DEFAULT_THREADS'):
    os.environ[key] = '1'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from fractions import Fraction
from collections import Counter
from importlib.metadata import version
import json
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ordinary_source_cmvm import (
    solve, flatten, precise_domain, normalized_interpreter, eval_graph,
    graph_order_account, signed_width, save, LOW, HIGH,
)


def pow2(exponent):
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def rational_rne(value):
    quotient, remainder = divmod(value.numerator, value.denominator)
    return quotient+int(2*remainder > value.denominator or
                        (2*remainder == value.denominator and quotient & 1))


def matrix_exponent(maximum):
    ratio = Fraction(32767)/maximum
    exponent = ratio.numerator.bit_length()-ratio.denominator.bit_length()
    while pow2(exponent) > ratio:
        exponent -= 1
    while pow2(exponent+1) <= ratio:
        exponent += 1
    return exponent


def rational_matrix(student, fixed):
    q = fixed['lifting_q12'].astype(np.int64)
    rebuilt = np.asarray([rational_rne(Fraction.from_float(float(v))*4096)
                          for v in student['basis_lifting'].flat], np.int64).reshape(4, 5, 2)
    assert np.array_equal(q, rebuilt)
    matching = fixed['lifting_matchings'].astype(np.int64)
    assert np.array_equal(matching, student['basis_matchings'])
    basis = [[Fraction(int(i == j)) for j in range(10)] for i in range(10)]
    for layer in range(4):
        for pair, (first, second) in enumerate(matching[layer]):
            a = Fraction(int(q[layer, pair, 0]), 4096)
            b = Fraction(int(q[layer, pair, 1]), 4096)
            basis[first] = [x+a*y for x, y in zip(basis[first], basis[second])]
            basis[second] = [y+b*x for x, y in zip(basis[first], basis[second])]
    permutation = student['source_row_permutation'].astype(np.int64)
    gains = [Fraction.from_float(float(v)) for v in student['source_row_gain']]
    matrix = [[gain*v for v in basis[int(p)]] for p, gain in zip(permutation, gains)]
    return basis, matrix, q, gains, permutation


def pack_fraction_matrix(matrix):
    return [[[v.numerator, v.denominator] for v in row] for row in matrix]


def graph_stats(graph):
    ops = graph['nodes'][graph['n_input']:]
    return dict(add_sub_nodes=len(ops), additions=sum(not n['subtract'] for n in ops),
        subtractions=sum(n['subtract'] for n in ops),
        shared_direct_nodes=sum(n['distinct_consumers'] > 1 for n in ops),
        multiple_terminal_nodes=sum(len(n['downstream_output_rows']) > 1 for n in ops),
        terminal_count_histogram=dict(Counter(len(n['downstream_output_rows']) for n in ops)),
        sum_node_bits=sum(n['signed_bits'] for n in ops),
        sum_operator_bits=sum(n['conservative_operator_bits'] for n in ops),
        node_width_histogram=dict(Counter(n['signed_bits'] for n in ops)),
        operator_width_histogram=dict(Counter(n['conservative_operator_bits'] for n in ops)),
        maximum_node_or_output_bits=max(n['signed_bits'] for n in graph['nodes']+graph['outputs']),
        maximum_operator_bits=max(n['conservative_operator_bits'] for n in ops),
        maximum_depth=max(n['depth'] for n in graph['nodes']),
        maximum_fanout=max(n['fanout_edge_count'] for n in graph['nodes']),
        positive_operand_shifts=sum(n[key] > 0 for n in ops for key in ('lhs_shift', 'rhs_shift')),
        operand_shift_histogram=dict(Counter(int(n[key]) for n in ops for key in ('lhs_shift', 'rhs_shift') if n[key])),
        exact_output_shifts=[o['shift'] for o in graph['outputs']],
        output_sign_aliases=[o['sign'] for o in graph['outputs']],
        output_sign_cost='A final sign alias can be absorbed into the gate cutoff and inclusive comparison direction; no materialized continuous output is required here. This does not prove the original RNE function equivalent.',
        units='Add/sub count and widths of the actual exact q16 dot DAG. Neither adder-area bits nor scheduled cycles.')


def lifting_control():
    bundle = json.loads((HERE/'constant_compilation_graphs.json').read_text())
    result = json.loads((HERE/'constant_compilation_result.json').read_text())
    raw = result['models']['fast_raw_diagonal']['directions']['forward']
    records = [bundle['whole_halfstage_graphs'][f'fast_raw_diagonal/forward/{i}'] for i in range(8)]
    operator_bits = node_bits = 0
    for record in records:
        graph = record['whole_five_pairs_graph']
        for node in graph['nodes'][10:]:
            node_bits += node['signed_bits']
            # Same metric as precise_domain: cover each shifted operand and
            # the result. No gratuitous extra carry bit for one contender.
            operator_bits += max(node['signed_bits'], node['lhs_shifted_signed_bits'],
                                 node['rhs_shifted_signed_bits'])
    q = np.asarray(result['models']['fast_raw_diagonal']['coefficients_q12'])
    mandatory = [raw['boundary_costs']['details'][i]
                 for i in range(40) if not (i//10 == 3 and i % 2 == 1)]
    assert len(mandatory) == 35
    assert all(int(row['q']) == int(q.flat[i]) for i, row in enumerate(raw['boundary_costs']['details']))
    return dict(add_sub_nodes=159, sum_node_bits=node_bits, sum_operator_bits=operator_bits,
        intermediate_RNE=35, final_five_RNE_folded_into_gate=True,
        round_increment_operations=35,
        round_increment_operand_result_bits=sum(row['conditional_increment_bits'] for row in mandatory),
        round_guard_bits=35,
        round_sticky_OR_input_bits=sum(row['sticky_OR_input_bits'] for row in mandatory),
        signed_saturation_selections=35,
        arithmetic_plus_round_increment_operations=159+35,
        arithmetic_plus_round_increment_bits=operator_bits+sum(row['conditional_increment_bits'] for row in mandatory),
        scope='Existing exact lifting raw graph. 35 RNE increments are an explicit component count, not complete synthesized RNE/saturation cost; range observation does not force RF writes. Both alternatives owe final ten gate comparisons.')


def main():
    student_path = HERE/'stage320/fast_raw_diagonal.npz'
    fixed_path = HERE/'fixed_lifting_diverse10/fast_raw_diagonal_fixed_constants.npz'
    student = dict(np.load(student_path, allow_pickle=False))
    fixed = dict(np.load(fixed_path, allow_pickle=False))
    basis, exact, q12, gains, permutation = rational_matrix(student, fixed)
    maximum = max(abs(v) for row in exact for v in row)
    exponent = matrix_exponent(maximum)
    rounded = [[rational_rne(v*pow2(exponent)) for v in row] for row in exact]
    matrix = np.asarray(rounded, np.int64)
    assert np.all((matrix >= -32768) & (matrix <= 32767))
    error = [[v-Fraction(int(q))*pow2(-exponent) for v, q in zip(row, quant)]
             for row, quant in zip(exact, matrix)]
    options = dict(method0='wmc', method1='auto', hard_dc=-1, decompose_dc=-2,
                   qintervals=[(float(LOW), float(HIGH), 1.)]*10,
                   adder_size=1, carry_size=1, search_all_decompose_dc=True)
    pipeline = solve(np.ascontiguousarray(matrix.T, dtype=np.float32), **options)
    assert np.array_equal(pipeline.kernel, matrix.T)
    graph = flatten(pipeline, matrix, -LOW, np.array([LOW]), np.array([HIGH]))
    precise_domain(graph)
    graph.update(instance='One channel/position with ten continuous I24 inputs; all ten source readouts compiled jointly.',
        output_domain='Only the new approximate q16 dot, with real scale 2^(-exponent-14). No equivalence to the original per-half-step RNE/saturation function.',
        gate_scope='Gain and P are inside the matrix. A guard/error-bound/fallback certificate has NOT been constructed or verified.')
    interpreter = normalized_interpreter(graph)
    rng = np.random.default_rng(912)
    corners = np.where(((np.arange(1024)[:, None] >> np.arange(10)) & 1) != 0, HIGH, LOW)
    random = rng.integers(LOW, HIGH+1, size=(1024, 10), dtype=np.int64)
    basis_inputs = np.concatenate([np.eye(10, dtype=np.int64), -np.eye(10, dtype=np.int64), np.zeros((1, 10), np.int64)])
    x = np.concatenate([corners, random, basis_inputs]).T
    observed = np.column_stack([np.full(len(graph['nodes']), np.iinfo(np.int64).max, np.int64),
                                np.full(len(graph['nodes']), np.iinfo(np.int64).min, np.int64)])
    actual, reference = eval_graph(graph, x, observed), matrix@x
    official = interpreter.predict(np.ascontiguousarray(x.T, dtype=np.float64), n_threads=1).T
    assert np.array_equal(actual, reference) and np.array_equal(official, reference)
    stats = graph_stats(graph)
    orders = {}
    for name, pressure in (('official', False), ('fixed_last_use_pressure', True)):
        account = graph_order_account(graph, pressure=pressure)
        account.pop('temporary_peak_bytes_96lane', None)
        account.update(resident_input_bits=240, final_gate_bits=10,
            two_operand_register_upper_bits=2*stats['maximum_node_or_output_bits'],
            scope='Fixed static topological order with last-use reclamation, two forwarding registers and accepting output sink. Input I24 is retained separately. No port schedule, retiming or proof of minimum RF.')
        orders[name] = account
    control = lifting_control()
    summary = dict(complete=True, source_student=str(student_path), deployed_coefficients=str(fixed_path),
        source_stage=320, actual_q12=q12, source_permutation=permutation,
        gain_exact=[[g.numerator, g.denominator] for g in gains],
        rational_B=pack_fraction_matrix(basis), rational_gain_P_B=pack_fraction_matrix(exact),
        construction='Exact Fraction product of actual q12 shears without intermediate RNE/saturation, then exact IEEE source gains and actual P. Exported FP32 source_A is not read.',
        quantization=dict(rule='One matrix-wide exponent=floor(log2(32767/maxabs)); exact rational RNE to signed16, same fixed rule as old As; no exponent/partition search.',
            exponent=exponent, matrix_q16=matrix, nonzero=int(np.count_nonzero(matrix)), clips=0,
            integer_range=[int(matrix.min()), int(matrix.max())],
            max_abs_rational_coefficient_error=float(max(abs(v) for row in error for v in row)),
            omitted_halfstep_rounding_error='Not bounded in this first arithmetic screen; the rational B is not the deployed rounding/saturation function.'),
        official_version=version('da4ml'), official_options=options,
        official_method='One complete10x10 weighted graph/decomposition solve; not separate row/CSD compilation.',
        statistics=stats, static_orders=orders, exact_lifting_control=control,
        comparison=dict(collapse_vs_159_add_sub_ratio=stats['add_sub_nodes']/159,
            collapse_vs_159_plus_35_increment_ops_ratio=stats['add_sub_nodes']/control['arithmetic_plus_round_increment_operations'],
            collapse_vs_operator_plus_round_increment_bit_ratio=stats['sum_operator_bits']/control['arithmetic_plus_round_increment_bits'],
            assumptions='Collapse guard, error-bound comparisons, fallback and extra storage optimistically zero. No source/header/packing or variable-precision advantage added to either. Final ten output comparisons common. Raw guard/sticky/saturation separate, not synthesized.'),
        verification=dict(vectors=x.shape[1], output_values=int(reference.size),
            symbolic_all_signed24_inputs_equal_to_new_q16_matrix=True,
            integer_DAG_errors=0, official_DAIS_errors=0,
            original_lifting_gate_equivalence_tested=False, network_AEE_tested=False),
        decision='Stop this fixed full-collapse layout before certificate replay: no arithmetic-operation/bit-work margin in the optimistic first screen. Not a physical dominance proof; RNE/saturation and routing lack mapped cost. Do not reject partial collapse or the general gate-certificate family.',
        graph=graph)
    save(HERE/'gate_collapse_probe.json', summary)
    pressure = orders['fixed_last_use_pressure']
    official_order = orders['official']
    text = f'''# 固定全合并纯门图：第一次 CPU 费用筛查

仅使用 lifting raw 的 stage320：从实际 q12 半步以 Python Fraction 组成无 RNE 的 B，再折入真实 IEEE source gain 和 P；没有读取 export FP32 source_A。按既定每矩阵 signed16 规则得到 exponent={exponent}、100 个非零系数、范围 [{matrix.min()},{matrix.max()}]，无裁剪。该近似矩阵不是已验证825的逐步 RNE/saturation 函数，不能继承其 AEE。

| 每 T10 向量 | 原精确 lifting raw | 固定全合并近似图 |
|---|---:|---:|
| 完整 da4ml 加减 | 159 | {stats['add_sub_nodes']}（add {stats['additions']} / sub {stats['subtractions']}） |
| 中间 RNE 条件增量 | 35；末5已折门 | 乐观按0 |
| 精确节点位宽和 | {control['sum_node_bits']} | {stats['sum_node_bits']} |
| 覆盖移位操作数/结果的算术位和 | {control['sum_operator_bits']} | {stats['sum_operator_bits']} |
| 加上原35个RNE增量的算术位和 | {control['arithmetic_plus_round_increment_bits']} | {stats['sum_operator_bits']} |
| 原 guard/sticky/饱和 | 35 guard、{control['round_sticky_OR_input_bits']} sticky输入位、35饱和选择 | 额外guard/半径比较/fallback均按0 |

完整合并图有 {stats['shared_direct_nodes']} 个直接共享节点、{stats['multiple_terminal_nodes']} 个跨最终输出祖先节点；最长加减依赖 {stats['maximum_depth']}，最大实际算术位宽 {stats['maximum_operator_bits']}。唯一末端负号可静态翻转门比较方向和cutoff，无须物化连续负输出。完整官方顺序/固定 last-use 压力顺序的临时峰值分别为 {official_order['temporary_peak_scalar_bits']}/{pressure['temporary_peak_scalar_bits']} bit（{official_order['temporary_peak_vectors']}/{pressure['temporary_peak_vectors']} 个数），另有共同 I24 的240bit和两个至多{stats['maximum_node_or_output_bits']}bit转发寄存器；输出sink假定及时接收。这是静态压力统计，未闭端口或峰值物理RF，更不是周期。

全合并图单看算术就比原159+35次增量多 {100*(stats['add_sub_nodes']/194-1):.2f}%；同口径算术位和比原加减及RNE增量之和多 {100*(stats['sum_operator_bits']/control['arithmetic_plus_round_increment_bits']-1):.2f}%。故本次停止**这个固定全合并布局**，不追加误差界/回退扫描。尚未综合原RNE的guard/sticky/saturation，不能将上述比值当成面积或物理严格支配证明；部分合并/其他纯门证书仍未被该结果否定。

新 q16 图在完整符号系数、全部1024个signed24域角点、1024个固定整数向量及21个基向量/零向量下，与独立整数dot和官方DAIS共 {reference.size} 个输出零差。这里的零差只对应新矩阵，未与原lifting门比较；没有网络实验、GPU、RTL、格式或分组搜索。
'''
    (HERE/'gate_collapse_probe.md').write_text(text)
    print(json.dumps(dict(complete=True, exponent=exponent, statistics=stats,
        raw_control=control, temporary_peak_bits={k:v['temporary_peak_scalar_bits'] for k,v in orders.items()},
        integer_output_values=int(reference.size), mismatches=0), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
