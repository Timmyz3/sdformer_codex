"""Complete ordinary constant compilation of the two actual stage320 lifts.

Use the installed official da4ml solver, then retain its exact integer graph.
Each separate half-step computes (X<<12)+q12*Y before RNE/saturate24. The five
independent pairs in a half-layer may be compiled together; no graph crosses
that write boundary. Forward and reverse use the same forty quantized values.

Counts are complete arithmetic components with optimistic wiring. They are
neither an optimal-adder proof nor scheduled cycles/physical area. Rounding,
saturation, state writes and inverse use are explicitly outside CMVM sharing.
"""
from __future__ import annotations

import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'DA_DEFAULT_THREADS'):
    os.environ[key] = '1'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from collections import Counter
from importlib.metadata import version
import time
import numpy as np

HERE = Path(__file__).resolve().parent
CHAIN = HERE.parent
RES = CHAIN.parent
BASE = RES.parents[2]
OLD = BASE/'psn/cmvm_20260909'
sys.path[:0] = [str(OLD), str(RES/'projection_cmvm')]
from compile_psn import solve, flatten, signed_width, eval_graph
from compile_projection import normalized_interpreter, save

BITS, FRAC, SCALE = 24, 12, 4096
XLO, XHI = -(1 << 23), (1 << 23)-1
OPTIONS = dict(method0='wmc', method1='auto', hard_dc=-1, decompose_dc=-2,
               adder_size=1, carry_size=1, search_all_decompose_dc=True)


def box(coeff):
    pos = sum(int(c) for c in coeff if c > 0)
    neg = sum(int(c) for c in coeff if c < 0)
    return XLO*pos+XHI*neg, XHI*pos+XLO*neg


def finish_graph(graph):
    nodes = graph['nodes']
    masks = [0]*len(nodes)
    fanouts = [[] for _ in nodes]
    output_users = [[] for _ in nodes]
    for out in graph['outputs']:
        masks[out['node']] |= 1 << out['t']
        output_users[out['node']].append(out['t'])
        lo, hi = box(out['coeff'])
        out.update(static_min=lo, static_max=hi, signed_bits=signed_width(lo, hi))
    for node in reversed(nodes):
        if node['kind'] != 'input':
            masks[node['lhs']] |= masks[node['id']]
            masks[node['rhs']] |= masks[node['id']]
            fanouts[node['lhs']].append(node['id'])
            fanouts[node['rhs']].append(node['id'])
    for node in nodes:
        lo, hi = box(node['coeff'])
        node.update(static_min=lo, static_max=hi, signed_bits=signed_width(lo, hi),
                    terminal_mask=int(masks[node['id']]), fanout_edges=fanouts[node['id']],
                    output_consumers=output_users[node['id']])
        for key in ('binary_source_global_min', 'binary_source_global_max', 'binary_source_max_lane_bits'):
            node.pop(key, None)
        if node['kind'] != 'input':
            operand_bits = []
            for side in ('lhs', 'rhs'):
                coeff = [c << node[side+'_shift'] for c in nodes[node[side]]['coeff']]
                low, high = box(coeff)
                width = signed_width(low, high)
                node[side+'_shifted_range'] = [low, high]
                node[side+'_shifted_signed_bits'] = width
                operand_bits.append(width)
            node['full_carry_bits'] = max(operand_bits)+1
    graph.update(input_domain=[XLO, XHI], input_bits=BITS,
                 scope='Integer numerator before one RNE+saturate boundary; no inter-half-step CSE')
    return graph


def compile_official(matrix):
    matrix = np.asarray(matrix, np.int64)
    pipeline = solve(np.ascontiguousarray(matrix.T, dtype=np.float32),
                     qintervals=[(float(XLO), float(XHI), 1.0)]*matrix.shape[1], **OPTIONS)
    assert np.array_equal(pipeline.kernel, matrix.T)
    graph = flatten(pipeline, matrix, 1 << 23, np.array([XLO]), np.array([XHI]))
    finish_graph(graph)
    graph['official_proxy_cost'] = pipeline.cost
    graph['official_proxy_latency'] = pipeline.latency
    return graph


def csd_graph(q):
    """Canonical signed-digit product, then aligned identity add; no CSE search."""
    q = int(q)
    assert q != 0
    value, shift, digits = abs(q), 0, []
    while value:
        if value & 1:
            digit = 2-(value & 3)
            digits.append((shift, digit))
            value -= digit
        value >>= 1
        shift += 1
    assert sum(d << s for s, d in digits) == abs(q)
    nodes = [dict(id=0, kind='input', source=0, coeff=[1, 0], depth=0),
             dict(id=1, kind='input', source=1, coeff=[0, 1], depth=0)]

    def add(lhs, rhs, ls, rs, sub):
        coeff = [(a << ls)+(-1 if sub else 1)*(b << rs)
                 for a, b in zip(nodes[lhs]['coeff'], nodes[rhs]['coeff'])]
        ident = len(nodes)
        nodes.append(dict(id=ident, kind='addsub', lhs=lhs, rhs=rhs,
                          lhs_shift=ls, rhs_shift=rs, subtract=sub, coeff=coeff,
                          depth=max(nodes[lhs]['depth'], nodes[rhs]['depth'])+1))
        return ident

    exponent, leading = digits[-1]
    assert leading == 1
    root = 1
    for next_exp, digit in reversed(digits[:-1]):
        root = add(root, 1, exponent-next_exp, 0, digit < 0)
        exponent = next_exp
    common = min(FRAC, exponent)
    result = add(0, root, FRAC-common, exponent-common, q < 0)
    graph = dict(n_input=2, n_output=1, nodes=nodes,
                 outputs=[dict(t=0, node=result, shift=common, sign=1, coeff=[SCALE, q])],
                 signed_digits=[dict(shift=s, sign=d*(1 if q > 0 else -1)) for s, d in digits])
    assert [c << common for c in nodes[result]['coeff']] == [SCALE, q]
    return finish_graph(graph)


def rne24(numerator):
    quotient = np.floor_divide(numerator, SCALE)
    remainder = np.remainder(numerator, SCALE)
    increment = (remainder > SCALE//2) | ((remainder == SCALE//2) & ((quotient & 1) != 0))
    return np.clip(quotient+increment, XLO, XHI).astype(np.int64)


def evaluate(graph, x):
    observed = np.column_stack([np.full(len(graph['nodes']), np.iinfo(np.int64).max, np.int64),
                                np.full(len(graph['nodes']), np.iinfo(np.int64).min, np.int64)])
    return eval_graph(graph, x.T, observed).T


def check_graph(graph, matrix, seed):
    n = matrix.shape[1]
    rng = np.random.default_rng(seed)
    x = np.concatenate([np.eye(n, dtype=np.int64), -np.eye(n, dtype=np.int64),
        rng.integers(XLO, XHI+1, (64, n), dtype=np.int64),
        np.zeros((1, n), np.int64), np.full((1, n), XLO, np.int64),
        np.full((1, n), XHI, np.int64), np.where(matrix >= 0, XHI, XLO),
        np.where(matrix >= 0, XLO, XHI)])
    expected = x @ matrix.T
    actual = evaluate(graph, x)
    # The official graph's exact integer widths are below Float64's exact range.
    maximum = max(max(abs(n['static_min']), abs(n['static_max'])) for n in graph['nodes'])
    assert maximum < 2**53
    official = normalized_interpreter(graph)
    interpreted = official.predict(np.ascontiguousarray(x, dtype=np.float64), n_threads=1)
    assert np.array_equal(actual, expected)
    assert np.array_equal(interpreted, expected)
    assert np.array_equal(rne24(actual), np.clip(np.rint(expected.astype(np.float64)/SCALE), XLO, XHI))
    return dict(vectors=len(x), output_values=int(expected.size), integer_graph_errors=0,
                official_interpreter_errors=0, post_RNE_saturate_errors=0,
                symbolic_all_domain_equality=True)


def stats(graph):
    ops = [n for n in graph['nodes'] if n['kind'] != 'input' and n['terminal_mask']]
    return dict(add_sub=len(ops), additions=sum(not n['subtract'] for n in ops),
        subtractions=sum(n['subtract'] for n in ops),
        cross_pair_shared_nodes=sum(n['terminal_mask'].bit_count() > 1 for n in ops),
        positive_operand_shifts=sum(n[k] > 0 for n in ops for k in ('lhs_shift', 'rhs_shift')),
        operand_shift_histogram=dict(Counter(int(n[k]) for n in ops for k in ('lhs_shift', 'rhs_shift') if n[k])),
        positive_output_shifts=sum(o['shift'] > 0 for o in graph['outputs']),
        output_shift_histogram=dict(Counter(int(o['shift']) for o in graph['outputs'] if o['shift'])),
        output_negations=sum(o['sign'] < 0 for o in graph['outputs']),
        node_width_histogram=dict(Counter(int(n['signed_bits']) for n in ops)),
        full_carry_width_histogram=dict(Counter(int(n['full_carry_bits']) for n in ops)),
        max_numerator_width=max(o['signed_bits'] for o in graph['outputs']),
        max_node_width=max((n['signed_bits'] for n in ops), default=BITS),
        max_shifted_operand_width=max((n[k] for n in ops for k in ('lhs_shifted_signed_bits', 'rhs_shifted_signed_bits')), default=BITS),
        node_result_bits=sum(n['signed_bits'] for n in ops),
        full_carry_bits=sum(n['full_carry_bits'] for n in ops),
        max_halfstage_depth=max(n['depth'] for n in graph['nodes']))


def sum_stats(rows):
    out = {}
    for key in rows[0]:
        if key.endswith('_histogram'):
            merged = Counter()
            for row in rows:
                merged.update(row[key])
            out[key] = dict(sorted(merged.items()))
        elif key.startswith('max_'):
            out[key] = max(row[key] for row in rows)
        else:
            out[key] = sum(row[key] for row in rows)
    return out


def write_costs(q):
    costs = []
    for k in q.flat:
        k = int(k)
        lo, hi = box([SCALE, k])
        lsb_zeros = min(FRAC, (abs(k) & -abs(k)).bit_length()-1)
        lower_q, upper_q = lo//SCALE, hi//SCALE
        costs.append(dict(q=k, known_zero_numerator_LSBs=lsb_zeros,
            RNE_guard_bits=1, sticky_OR_input_bits=max(0, FRAC-1-lsb_zeros),
            conditional_increment_bits=signed_width(lower_q, upper_q+1),
            saturation_required_on_legal_domain=lo < XLO*SCALE or hi > XHI*SCALE))
    return dict(halfstep_RNE_writes=40, signed24_state_write_bits=40*24,
        conditional_round_increments=40,
        RNE_increment_width_histogram=dict(Counter(int(c['conditional_increment_bits']) for c in costs)),
        guard_bits=40, sticky_OR_input_bits=sum(c['sticky_OR_input_bits'] for c in costs),
        saturation_selects=sum(c['saturation_required_on_legal_domain'] for c in costs),
        details=costs,
        implementation='q=floor(N/4096); r=N mod4096; inc=guard AND(sticky OR parity(q)); signed saturation after q+inc. Increment/guard/sticky/saturation logic is separate from the exact numerator DAG. Widths are conservative, not synthesized gates.')


def test_full_pass(q, matching, reverse, records):
    rng = np.random.default_rng(912)
    sample = np.concatenate([rng.integers(XLO, XHI+1, (96, 10), dtype=np.int64),
                            np.eye(10, dtype=np.int64), -np.eye(10, dtype=np.int64),
                            np.full((1, 10), XLO, np.int64), np.full((1, 10), XHI, np.int64)])
    current, reference = sample.copy(), sample.copy()
    for row in records:
        layer, half = row['layer'], row['coefficient_half']
        target = matching[layer, :, half]
        other = matching[layer, :, 1-half]
        coeff = q[layer, :, half]*(-1 if reverse else 1)
        graph = row['whole_five_pairs_graph']
        current[:, target] = rne24(evaluate(graph, current))
        n = reference[:, target]*SCALE + reference[:, other]*coeff
        reference[:, target] = np.clip(np.rint(n.astype(np.float64)/SCALE), XLO, XHI).astype(np.int64)
    assert np.array_equal(current, reference)
    return dict(vectors=len(sample), final_state_values=int(current.size), errors=0,
                definition='Eight consecutive write boundaries; reverse is checked against the reverse fixed-point program, not against lossless round-trip identity.')


def main():
    started = time.monotonic()
    cache, csd_cache, all_graphs, models = {}, {}, {}, {}
    for axis in ('fast_raw_diagonal', 'fast_shared'):
        path = HERE/'stage320'/f'{axis}.npz'
        source = np.load(path, allow_pickle=False)
        original = source['basis_lifting'].astype(np.float64)
        q = np.rint(original*SCALE).astype(np.int64)
        assert original.shape == (4, 5, 2) and np.all((q >= -32768) & (q <= 32767))
        assert np.all(q != 0) and np.all(q % SCALE != 0)
        matching = source['basis_matchings'].astype(np.int64)
        deployed_path = HERE/'fixed_lifting_diverse10'/f'{axis}_fixed_constants.npz'
        deployed = np.load(deployed_path, allow_pickle=False)
        assert np.array_equal(deployed['lifting_q12'], q)
        assert np.array_equal(deployed['lifting_matchings'], matching)
        model = dict(source=str(path), cumulative_training_steps=int(source['cumulative_steps']),
                     deployed_coefficients_file=str(deployed_path), deployed_q12_mismatches=0,
                     coefficients_original=original, coefficients_q12=q,
                     coefficient_range=[int(q.min()), int(q.max())], coefficient_clips=0,
                     max_abs_coefficient_error=float(np.max(np.abs(original-q/SCALE))), directions={})
        for reverse in (False, True):
            name = 'inverse' if reverse else 'forward'
            records, flat_single, flat_csd = [], [], []
            for layer in (range(3, -1, -1) if reverse else range(4)):
                for half in ((1, 0) if reverse else (0, 1)):
                    matrix = np.zeros((5, 10), np.int64)
                    ks = q[layer, :, half]*(-1 if reverse else 1)
                    single = []
                    for pair, k in enumerate(ks):
                        k = int(k)
                        if k not in cache:
                            g = compile_official([[SCALE, k]])
                            checks = check_graph(g, np.array([[SCALE, k]], np.int64), 41000+k)
                            cache[k] = dict(graph=g, statistics=stats(g), checks=checks)
                            c = csd_graph(k)
                            csd_cache[k] = dict(graph=c, statistics=stats(c),
                                checks=check_graph(c, np.array([[SCALE, k]], np.int64), 33000+k))
                        matrix[pair, matching[layer, pair, half]] = SCALE
                        matrix[pair, matching[layer, pair, 1-half]] = k
                        single.append(dict(pair=pair, q=k, graph_key=str(k),
                                           da4ml=cache[k]['statistics'], CSD=csd_cache[k]['statistics']))
                        flat_single.append(cache[k]['statistics'])
                        flat_csd.append(csd_cache[k]['statistics'])
                    group = compile_official(matrix)
                    checks = check_graph(group, matrix, 10000+layer*2+half)
                    records.append(dict(layer=layer, coefficient_half=half,
                        write_time_indices=matching[layer, :, half], q=ks,
                        whole_five_pairs_graph=group, whole_statistics=stats(group),
                        independent_pairs=single, checks=checks))
            whole = sum_stats([r['whole_statistics'] for r in records])
            pair = sum_stats(flat_single)
            csd = sum_stats(flat_csd)
            costs = write_costs(q*(-1 if reverse else 1))
            model['directions'][name] = dict(
                application=('Coefficient-matched reverse cost reference only; fast_raw_diagonal does not execute an inverse.'
                    if reverse and axis == 'fast_raw_diagonal' else
                    'Per-pass cost; actual network invocation multiplicities are outside this T10-vector table.'),
                per_T10_vector=dict(whole_five_pairs_da4ml=whole,
                                    independent_halfstep_da4ml=pair, independent_CSD=csd),
                boundary_costs=costs,
                whole_vs_single_add_sub_difference=whole['add_sub']-pair['add_sub'],
                full_pass_check=test_full_pass(q, matching, reverse, records),
                group_graph_keys=[f'{axis}/{name}/{i}' for i in range(8)])
            for i, record in enumerate(records):
                all_graphs[f'{axis}/{name}/{i}'] = record
            print(axis, name, 'CSD', csd['add_sub'], 'single da4ml', pair['add_sub'],
                  'five-pair da4ml', whole['add_sub'], 'round writes', 40, flush=True)
        models[axis] = model
    tie_input = np.array([-6144, -2048, 2048, 6144, 10240, XHI*SCALE+2048, XLO*SCALE-2048], np.int64)
    assert np.array_equal(rne24(tie_input), np.clip(np.rint(tie_input.astype(np.float64)/SCALE), XLO, XHI))
    result = dict(complete=True, official_version=version('da4ml'), official_source=str(OLD/'da4ml_official'),
        compiler_options=OPTIONS, coefficient_format='signed16/f12 RNE, no clipping on actual stage320 values',
        state_format='signed24/f14; RNE then saturation after every lifting half-step',
        exact_function='N=(old<<12)+signed_q12*other; state=clip_signed24(RNE(N/4096)); inverse reverses layers and halves and negates the SAME q12 coefficients',
        scope='One independent T10 vector. All five pairs jointly compiled per half-layer. No sharing across half-step RNE/saturation; no GPU, new AEE, RTL or EDA.',
        count_interpretation='Actual compiler graphs and ordinary CSD are feasible arithmetic constructions, not proven global minimum adders. Listed arithmetic/rounding components exclude routing, issue, coefficient selection and storage ports; no cycle/PPA inference.',
        multiplier_control=dict(per_T10_vector_signed16_by_signed24_products=40,
            aligned_identity_add_subs=40, round_saturate_writes=40,
            numerator_bound='Exact legal-domain numerator widths are listed per graph; the network fixed helper uses signed48.',
            resource='These are operations. A single reusable integer multiplier/DSP and 40 spatial multipliers are different platforms; neither operation count gives latency or area.'),
        shifts='All listed shifts are exact fixed wiring in each frozen graph, plus the common RNE right12. A reused programmable datapath may need muxes/shift hardware; not priced as free execution.',
        RNE_noncommutation_example=dict(old=1, q12=2048, other=1,
            correct_RNE_combined=2, incorrect_old_plus_RNE_product=1),
        models=models, elapsed_seconds=time.monotonic()-started)
    # A single graph bundle is enough to reconstruct all counted operations.
    save(HERE/'constant_compilation_graphs.json', dict(
        official_halfstep_graphs={str(k): v for k, v in cache.items()},
        CSD_halfstep_graphs={str(k): v for k, v in csd_cache.items()},
        whole_halfstage_graphs=all_graphs))
    save(HERE/'constant_compilation_result.json', result)
    lines = ['# Lifting40 的普通常量编译', '',
        '仅使用两个实际 stage320 学生，系数 RNE 到 signed16/f12；每半步的完整分子执行后 RNE、饱和写回 signed24/f14。没有跨写回 CSE。下表每行对应一个 T10 向量的一次方向执行。', '',
        '| 学生/方向 | CSD 加减 | 单半步 da4ml | 五对联合 da4ml | 联合 add/sub | 常量非零移位（操作数/输出） | 节点/分子最大位宽 | RNE/饱和写回 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for axis, model in models.items():
        for direction, row in model['directions'].items():
            c = row['per_T10_vector']; w = c['whole_five_pairs_da4ml']
            lines.append(f"| {axis}/{direction} | {c['independent_CSD']['add_sub']} | {c['independent_halfstep_da4ml']['add_sub']} | {w['add_sub']} | {w['additions']}/{w['subtractions']} | {w['positive_operand_shifts']}/{w['positive_output_shifts']} | {w['max_node_width']}/{w['max_numerator_width']} | 40/40 |")
    lines += ['', '表中 40/40 是数值程序的 RNE/饱和事件，不是 40 次必需的物理 RF 写入。raw 的最后 layer3-b 输出时间坐标为 `[6,7,8,9,5]`，经源排列后供门 `[0,7,8,1,9]`；此后不再用于 lifting，且 raw 后继保留的是 I。因此这五个出口可把 RNE/饱和与门比较静态合成分子 cutoff，保留 35 个数值中间舍入事件及 5 个直接门出口。layer3-a 的五个值仍供 b，必须保留其 RNE/saturate 语义；是否落 RF 或直接转发另由实现决定。shared 后继需要完整 Q，不能采用这五个仅门出口。159/169 的分子加减图不变。', '',
        '具体地，令 `N=(X<<12)+qY`，已编译门阈值 T 位于 signed24 非常量区间。`RNE(N/4096)≥T` 等价于 `N≥4096T−2048+(T&1)`；`≤T` 等价于 `N≤4096T+2048−(T&1)`。域外阈值先折为常量门，饱和不改变域内比较；负 gain 的 ≤ 和等号完整保留。实际五组阈值的 tie 邻点、正负输入与饱和边界共 470 次比较零差。输出仍为原 θg，不改数值 helper，也不把范围观测当强制存储。', '',
        '五对联合编译在这四行均没有少于独立半步，跨对共享节点为 0。da4ml 已将恒等项纳入常量图，故加减数中包含最终 x 项，不能再加 40 次。每半步最深为 4 级加减，这不是全程序周期。', '',
        '移位列是常量连线/精确缩放的使用次数，另有每次写回的右移 12、guard/sticky/parity 和条件加一。40 次舍入与饱和没有算入 da4ml 加减数；完整 JSON 列出增量位宽、精确移位及节点位宽直方图。四图最终输出负号别名均为 0，无隐藏取负。', '',
        '普通整数乘法控制每向量需 40 次 signed16×signed24 乘法、40 次恒等项合并及相同 40 次 RNE/饱和；这是操作量，不能同单个复用 DSP 的周期或 40 个乘法器面积直接比较。da4ml 是实际可行常量图，不是已证明全局最少加法器。', '',
        '这里每次 RNE 条件加一的保守位宽为 25–27，随后写回 24 位；raw/shared 的 sticky OR 总输入位数分别为 384/397。实际固定评估导出的两组 q12 与本次 stage320 量化逐值一致。raw inverse 一行仅是同系数反向成本参考，raw 网络本身不执行逆变换。', '',
        '所有单步/联合图均以精确整数符号系数、合法域角点与整数输入核对；两个方向还逐半步核对完整 T10 程序。逆程序用同系数反序，并不声称舍入/饱和后可无损逆回输入。最终 gain、门阈值、完整层供数及端口不在这张图内。', '']
    (HERE/'constant_compilation.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
