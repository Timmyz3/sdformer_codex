"""One official da4ml0.6.0 whole-As control for the ordinary fixed825 student.

Compile the saved 10x10 As_q16 once with wmc/auto and no hard delay limit.
Full signed24 input -> exact signed48 dot -> original RNE/saturate24 and
source threshold. Also compile the exactly equivalent gate-only dot cutoff.
No GPU, new model, compiler changes, individual-row compilation or RTL.
The graph is a different student/function from lifting40, not a speed ratio.
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
import math
import time

import numpy as np

HERE = Path(__file__).resolve().parent
CHAIN = HERE.parent
ROOT = CHAIN.parents[3]
sys.path.insert(0, str(ROOT/'psn/cmvm_20260909'))
sys.path.insert(0, str(CHAIN.parent/'projection_cmvm'))
from compile_psn import solve, flatten, signed_width, csd_digits, eval_graph, graph_order_account
from compile_projection import normalized_interpreter, save

LOW, HIGH = -(1 << 23), (1 << 23)-1


def domain(coeff):
    pos = sum(int(c) for c in coeff if c > 0)
    neg = sum(int(c) for c in coeff if c < 0)
    return LOW*pos+HIGH*neg, HIGH*pos+LOW*neg


def precise_domain(graph):
    consumers = [set() for _ in graph['nodes']]
    for out in graph['outputs']:
        lo, hi = domain(out['coeff'])
        out.update(static_min=lo, static_max=hi, signed_bits=signed_width(lo, hi))
        consumers[out['node']].add(out['t'])
    for node in reversed(graph['nodes']):
        if node['kind'] != 'input':
            for parent in (node['lhs'], node['rhs']):
                consumers[parent].update(consumers[node['id']])
    for node in graph['nodes']:
        lo, hi = domain(node['coeff'])
        node.update(static_min=lo, static_max=hi, signed_bits=signed_width(lo, hi),
                    downstream_output_rows=sorted(consumers[node['id']]))
        for key in ('binary_source_global_min', 'binary_source_global_max', 'binary_source_max_lane_bits'):
            node.pop(key, None)
        if node['kind'] != 'input':
            operands = []
            for edge, shift in (('lhs', 'lhs_shift'), ('rhs', 'rhs_shift')):
                parent = graph['nodes'][node[edge]]
                operands.append(domain([int(v) << node[shift] for v in parent['coeff']]))
            op_lo = min(lo, *(v[0] for v in operands))
            op_hi = max(hi, *(v[1] for v in operands))
            node.update(shifted_operand_ranges=operands,
                        conservative_operator_bits=signed_width(op_lo, op_hi))
        assert max(abs(lo), abs(hi)) < 1 << 47
    graph.update(input_domain=[LOW, HIGH], input_signed_bits=24,
        instance='One spatial/channel context: ten continuous I24 temporal samples -> ten source dots.',
        static_domain='Independent signed24 box; exact Python-integer coefficients, no assumed source sparsity.',
        output_domain='Signed48 dot, followed by RNE/saturate24 and source theta gate; exact gate-only cutoff exported separately.')


def rne_integer(value, fraction):
    """Signed RNE using integer quotient/remainder; no floating approximation."""
    divisor = 1 << fraction
    q, remainder = np.divmod(np.asarray(value, np.int64), divisor)
    increment = (2*remainder > divisor) | ((2*remainder == divisor) & ((q & 1) != 0))
    return q+increment


def state_gate(q, parameters):
    direction = parameters['source_direction'][:, None]
    constant = parameters['source_constant'][:, None]
    cut = parameters['source_threshold'][:, None]
    return np.where(constant >= 0, constant.astype(bool), np.where(direction > 0, q >= cut, q <= cut))


def compile_postprocess(parameters, matrix, exponent):
    divisor = 1 << exponent
    rows = []
    boundary_mismatches = 0
    boundary_values = 0
    for t, (k, direction, constant) in enumerate(zip(parameters['source_threshold'],
            parameters['source_direction'], parameters['source_constant'])):
        k, direction, constant = int(k), int(direction), int(constant)
        lo, hi = domain(matrix[t])
        rounded_lo, rounded_hi = (int(v) for v in rne_integer([lo, hi], exponent))
        state_lo, state_hi = max(LOW, rounded_lo), min(HIGH, rounded_hi)
        cutoff = 0
        if constant < 0:
            if direction > 0:
                if k <= LOW: constant = 1
                elif k > HIGH: constant = 0
                else: cutoff = k*divisor-divisor//2+(k & 1)
            else:
                if k >= HIGH: constant = 1
                elif k < LOW: constant = 0
                else: cutoff = k*divisor+divisor//2-(k & 1)
            if constant < 0:
                if (direction > 0 and cutoff <= lo) or (direction < 0 and cutoff >= hi): constant = 1
                elif (direction > 0 and cutoff > hi) or (direction < 0 and cutoff < lo): constant = 0
        # Clamp begins here, including odd/even tie behavior at both ends.
        high_first = (HIGH+1)*divisor-divisor//2+((HIGH+1) & 1)
        low_last = (LOW-1)*divisor+divisor//2-((LOW-1) & 1)
        targets = [lo, hi, 0, cutoff, high_first, low_last]
        for n in (-3, -2, -1, 0, 1, 2, 3, k-1, k, k+1):
            targets.extend([n*divisor-divisor//2, n*divisor+divisor//2])
        values = np.unique([int(z)+d for z in targets for d in (-2, -1, 0, 1, 2)])
        values = values[(values >= lo) & (values <= hi)]
        rounded = rne_integer(values, exponent)
        clipped = rounded.clip(LOW, HIGH)
        reference = (np.full(len(values), bool(parameters['source_constant'][t]))
            if parameters['source_constant'][t] >= 0 else
            clipped >= k if direction > 0 else clipped <= k)
        direct = (np.full(len(values), bool(constant)) if constant >= 0 else
                  values >= cutoff if direction > 0 else values <= cutoff)
        mismatch = int(np.count_nonzero(reference != direct))
        boundary_mismatches += mismatch; boundary_values += len(values)
        assert np.array_equal(rounded, np.rint(values.astype(np.float64)/divisor).astype(np.int64))
        rows.append(dict(t=t, dot_domain=[lo, hi], dot_signed_bits=signed_width(lo, hi),
            rne_shift=exponent, rounded_unclipped_domain=[rounded_lo, rounded_hi],
            q24_domain=[state_lo, state_hi], q24_source_threshold=k, direction=direction,
            direct_dot_cutoff=cutoff, constant_gate=constant,
            first_high_saturating_dot=high_first, last_low_saturating_dot=low_last,
            boundary_test_values=len(values), boundary_mismatches=mismatch))
    assert boundary_mismatches == 0
    return dict(rows=rows, source_theta=float(parameters['source_theta']),
        source_tau_real=parameters['source_tau_real'],
        fixed_function='dot=Aq@I24; q24=clamp(RNE(dot/2^exponent),signed24); then original inclusive source cutoff; output actual theta*g.',
        preimage_proof='For integer k inside the state domain and f>=1: RNE(n/2^f)>=k iff n>=k*2^f-2^(f-1)+(k mod2); <=k iff n<=k*2^f+2^(f-1)-(k mod2). Saturation is monotone; out-of-domain cutoffs are constants.',
        ordinary_gate_only_permission='Source Q has no continuous consumer in this raw student, so its RNE/saturate can be composed into the dot comparison. This exact same-function simplification is available to ordinary CMVM. Persistent raw I is still needed.',
        all_legal_integer_dot_preimage_equal=True, boundary_test_values=boundary_values,
        boundary_mismatches=boundary_mismatches,
        excluded_cost='RNE/saturation hardware if materialized, or dot cutoff compare plus theta gate if gate-only; none are counted as CMVM add nodes.')


def main():
    started = time.monotonic()
    assert version('da4ml') == '0.6.0'
    stem = HERE/'ordinary_source_cmvm'
    source = CHAIN/'temporal_structured_recovery/fixed_valid825/identity_permuted_base_coordinate_constants.npz'
    parameters = dict(np.load(source))
    student_path = CHAIN/'temporal_structured_recovery/stage128x256/identity_permuted_base.npz'
    student = dict(np.load(student_path, allow_pickle=True))
    a, exponent = parameters['As_q16'].astype(np.int64), int(parameters['As_exponent'])
    assert a.shape == (10, 10) and exponent >= 1
    rebuilt = np.clip(np.rint(np.ldexp(student['source_A'].astype(np.float64), exponent)), -32768, 32767).astype(np.int64)
    assert np.array_equal(a, rebuilt)
    assert exponent == math.floor(math.log2(32767/np.max(np.abs(student['source_A']))))
    bias = student['source_bias'].reshape(10)
    center = (np.broadcast_to(student['source_center'].reshape(-1), (10,))
              if str(student['source_center_mode']) != 'zero' else np.zeros(10))
    tau = [Fraction.from_float(float(student['source_theta']))-Fraction.from_float(float(b))+Fraction.from_float(float(c)) for b,c in zip(bias, center)]
    rebuilt_cut = np.array([-((-v.numerator*16384)//v.denominator) for v in tau])
    assert np.array_equal(parameters['source_threshold'], rebuilt_cut)
    options = dict(method0='wmc', method1='auto', hard_dc=-1, decompose_dc=-2,
        qintervals=[(float(LOW), float(HIGH), 1.)]*10, adder_size=1, carry_size=1,
        search_all_decompose_dc=True)
    pipeline = solve(np.ascontiguousarray(a.T, dtype=np.float32), **options)
    assert np.array_equal(pipeline.kernel, a.T)
    pipeline.save(stem.with_suffix('.official_pipeline.json'))
    graph = flatten(pipeline, a, -LOW, np.array([LOW]), np.array([HIGH]))
    precise_domain(graph)
    interpreter = normalized_interpreter(graph)
    interpreter.save_binary(stem.with_suffix('.dais'))
    graph.update(official_version=version('da4ml'), official_options=options,
        source_parameter_file=str(source), source_exponent=exponent,
        symbolic_final_coefficients_equal=True,
        aliases='Exact signs and power-of-two shifts retained; negative shifts must divide every integer coefficient. No RNE inside the adder graph.')
    save(stem.with_suffix('.integer_dag.json'), graph)
    rng = np.random.default_rng(912)
    corners = np.where(((np.arange(1024)[:, None] >> np.arange(10)) & 1) != 0, HIGH, LOW)
    random = rng.integers(LOW, HIGH+1, size=(4096, 10), dtype=np.int64)
    basis = np.concatenate([np.eye(10, dtype=np.int64), -np.eye(10, dtype=np.int64), np.zeros((1, 10), np.int64)])
    x = np.concatenate([corners, random, basis]).T
    observed = np.zeros((len(graph['nodes']), 2), np.int64)
    got, expected = eval_graph(graph, x, observed), a@x
    assert np.array_equal(got, expected)
    official_x = np.ascontiguousarray(x.T, dtype=np.float64)
    official = interpreter.predict(official_x, n_threads=1).T
    assert np.array_equal(official, expected)
    q = rne_integer(got, exponent).clip(LOW, HIGH)
    assert np.array_equal(q, np.rint(expected.astype(np.float64)/(1 << exponent)).clip(LOW, HIGH))
    postprocess = compile_postprocess(parameters, a, exponent)
    direct = np.stack([np.full(x.shape[1], bool(row['constant_gate'])) if row['constant_gate'] >= 0 else
                       got[t] >= row['direct_dot_cutoff'] if row['direction'] > 0 else
                       got[t] <= row['direct_dot_cutoff'] for t,row in enumerate(postprocess['rows'])])
    assert np.array_equal(direct, state_gate(q, parameters))
    ops = graph['nodes'][10:]
    stats = dict(add_sub_nodes=len(ops), additions=sum(not n['subtract'] for n in ops),
        subtractions=sum(n['subtract'] for n in ops), stage_nodes=[len(s['arithmetic_nodes']) for s in graph['stage_info']],
        shared_direct_arithmetic_nodes=sum(n['distinct_consumers'] > 1 for n in ops),
        nodes_with_multiple_final_outputs=sum(len(n['downstream_output_rows']) > 1 for n in ops),
        max_add_depth=max(n['depth'] for n in graph['nodes']),
        output_add_depths=[graph['nodes'][o['node']]['depth'] for o in graph['outputs']],
        maximum_fanout=max(n['fanout_edge_count'] for n in graph['nodes']),
        node_width_histogram=dict(Counter(n['signed_bits'] for n in ops)),
        operator_width_histogram=dict(Counter(n['conservative_operator_bits'] for n in ops)),
        sum_arithmetic_node_bits=sum(n['signed_bits'] for n in ops),
        sum_conservative_operator_bits=sum(n['conservative_operator_bits'] for n in ops),
        maximum_node_or_output_bits=max(n['signed_bits'] for n in graph['nodes']+graph['outputs']),
        DAIS_outward_interval_wider_nodes=[n['id'] for n in graph['nodes']
            if n['DAIS_declared_signed_bits'] > n['signed_bits']],
        DAIS_scope='Official interval endpoints are Float32, conservatively rounded outward and sign-corrected. Their interpreter widths are distinct from exact mathematical widths in this report.',
        independent_CSD_digits=sum(csd_digits(v) for v in a.flat),
        independent_CSD_adds=sum(csd_digits(v) for v in a.flat)-int(np.count_nonzero(np.any(a, axis=1))),
        nonzero_matrix_coefficients=int(np.count_nonzero(a)),
        official_abstract_bit_cost=pipeline.cost, official_abstract_delay=pipeline.latency,
        official_cost_scope='FPGA-oriented compiler proxies, not ASIC area/delay or cycles.')
    orders = {name: graph_order_account(graph, pressure=pressure) for name,pressure in (('official',False),('pressure_heuristic',True))}
    for row in orders.values():
        row.pop('temporary_peak_bytes_96lane', None)
        row.update(resident_input_bits=240, final_gate_bits=10, optional_collect_dot_bits=480,
            optional_materialized_q24_bits=240,
            scope='One T10 scalar/channel context; accepting output sink and two-operand forwarding. No finite-port latency or input/next-layer overlap closure. No free pressure-order selection by validation.')
    summary = json.loads((source.parent/'identity_permuted_base_summary.json').read_text())
    result = dict(complete=True, source_file=str(source), saved_student=str(student_path),
        source_A_quantization_mismatches=0, source_threshold_recompile_mismatches=0,
        source_matrix=a, As_exponent=exponent, fixed_student_full825=summary,
        official_version=version('da4ml'), official_options=options,
        official_method='Complete graph-based matrix decomposition plus weighted signed-shift cross-output CSE; one full10x10 solve, not10 row solves or independent CSD.',
        official_sources=['https://github.com/calad0i/da4ml', 'https://calad0i.github.io/da4ml/cmvm.html'],
        statistics=stats, postprocess=postprocess, static_orders=orders,
        verification=dict(whole_domain_symbolic_coefficients_equal=True, signed24_input=True,
            all_normalized_nodes_and_outputs_fit_signed48=True, corner_vectors=1024, seeded_uniform_vectors=4096,
            identity_negative_identity_zero_vectors=21, total_output_values=int(expected.size),
            integer_DAG_mismatches=0, official_DAIS_mismatches=0, RNE_state_mismatches=0,
            direct_gate_mismatches=0, observed_node_ranges=observed,
            scope='Legal-domain CPU checks and full symbolic equality; no new real frame capture, no full-network rerun. Existing825 belongs to unchanged integer As/RNE/gate function.'),
        boundary='Lifting40 is a separately trained different function, with half-step RNE and its own AEE. Node ratios are not time/energy gains; this fills an ordinary complete-CMVM baseline only.',
        missing='Resource assignment, coefficient/operand supply, buffering/retiming/fanout, actual finite-port schedule, input zero/bit-demand dynamic costs and full-chain area/energy remain unmeasured.',
        elapsed_seconds=time.monotonic()-started)
    save(stem.with_suffix('.json'), result)
    text = f'''# 普通 dense source PSN 的完整 CMVM 控制

直接读取 fixed_valid825 的 identity_permuted_base 导出；与 stage128x256 的 source_A 逐元素再量化、τ再编译均0差。该普通学生既有825 AEE={summary['AEE_frame_mean']:.12f}，不是 lifting 的函数或精度。

官方 da4ml0.6.0，固定 wmc/auto、不限延迟，一次完整10×10矩阵分解+CSE：**{stats['add_sub_nodes']}加减节点**，{stats['shared_direct_arithmetic_nodes']}个直接共享节点，最长加法依赖{stats['max_add_depth']}，逐输出深度{stats['output_add_depths']}。逐系数CSD仅作为弱算术参照：{stats['independent_CSD_adds']}加减。节点精确box宽度最高{stats['maximum_node_or_output_bits']}位；还分别记录移位输入所需的保守算术端口宽度，不把输出位宽当免费窄加法器。

输入为真实signed24非对称合法域，As exponent={exponent}，dot48→RNE{exponent}→saturate24。10维节点系数和最终A全域符号核验通过；全部1024域角点、4096固定随机向量和21基向量/零向量，共{expected.size}个输出，与独立整数dot及官方DAIS均0差。RNE状态和最终门0差；另有{postprocess['boundary_test_values']}个正负tie／截断／阈值边界测试0差。这不是实际图像再捕获。

普通强控制也获得静态门编译：source Q24只有门消费者，RNE/clamp与阈值可精确合成dot48 cutoff，包含奇偶tie；无需为这个门出口物化Q24，仍须保存raw I。JSON保留每行完整合法dot/RNE/饱和域及cutoff，不省略τ与θ的区别。

官方顺序与简单压力顺序的存取、临时状态另列，输出sink假定可接收；它们尚不是端口／流水／扇出闭合。官方cost、节点数、逻辑深度均不能称周期、面积或功耗；不与lifting40半步回写混成同函数比较。此项补齐A，尚未证明新的X。

来源：[官方算法](https://calad0i.github.io/da4ml/cmvm.html)、[官方实现](https://github.com/calad0i/da4ml)。产物是同前缀 .py/.json/.integer_dag.json/.official_pipeline.json/.dais，旧helper、训练与官方源码未修改。
'''
    stem.with_suffix('.md').write_text(text)
    print(json.dumps(dict(complete=True, statistics=stats, output_values=int(expected.size),
        boundary_values=postprocess['boundary_test_values'], mismatches=0,
        elapsed_seconds=result['elapsed_seconds']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
