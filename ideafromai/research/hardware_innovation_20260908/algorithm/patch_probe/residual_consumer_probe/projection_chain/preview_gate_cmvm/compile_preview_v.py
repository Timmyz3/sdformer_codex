"""Compile the real preview V32->96 with the complete official da4ml solver.

One context is one time/position with 32 coupled continuous latent inputs.
The fixed interface is signed24 Zq in a common unit delta_Z. No new V
quantization: V consists of exact powers of two, and K=2**15 * V.T is integer.
Y_num=K@Zq, Y=delta_Z*2**-15*Y_num; that scale is not a truncating shift.

Only static disconnection of the 16 zero V rows is removed. The whole 96x32
matrix is compiled once, with the official internal decomposition search.
Existing adapters normalize exact stage scale/sign aliases and export the
official graph; they do not replace the compiler's sharing algorithm.

The 24-bit input interface is not a claim that the captured FP32 first factor
already supplies these common-scale integers. Per-latent U scales, alignment,
A-before-V, BN/bias/theta decisions and finite-resource execution remain outside.
No GPU, training, RTL, EDA, hash, AEE or cycle claim.
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
import argparse
import time
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent.parent
BASE = RES.parents[2]
OLD = BASE / 'psn/cmvm_20260909'
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(RES / 'projection_cmvm'))
from compile_psn import solve, flatten, signed_width, eval_graph
from compile_projection import save, normalized_interpreter, order_account

DEFAULT_PARENT = (RES.parent / 'factor_completion_20260909/latent_stage_train16'
                  / 'flow_recovery64/preview_only/shared48_u8_vq5.npz')
BITS = 24
XLO, XHI = -(1 << (BITS-1)), (1 << (BITS-1))-1


def box(coeff):
    positive = sum(int(x) for x in coeff if x > 0)
    negative = sum(int(x) for x in coeff if x < 0)
    return XLO*positive + XHI*negative, XHI*positive + XLO*negative


def annotate(graph, live_latents):
    """Exact integer ranges and transitive final-output masks on the same DAG."""
    terminal_masks = [0] * len(graph['nodes'])
    for out in graph['outputs']:
        out['h'] = out['t']
        terminal_masks[out['node']] |= 1 << out['h']
        lo, hi = box(out['coeff'])
        out.update(static_min=lo, static_max=hi, signed_bits=signed_width(lo, hi))
    for node in reversed(graph['nodes']):
        if node['kind'] != 'input':
            terminal_masks[node['lhs']] |= terminal_masks[node['id']]
            terminal_masks[node['rhs']] |= terminal_masks[node['id']]
    maximum_operand = 0
    for node in graph['nodes']:
        lo, hi = box(node['coeff'])
        node.update(static_min=lo, static_max=hi, signed_bits=signed_width(lo, hi))
        for key in ('binary_source_global_min', 'binary_source_global_max',
                    'binary_source_max_lane_bits'):
            node.pop(key, None)
        mask = terminal_masks[node['id']]
        output_h = [h for h in range(graph['n_output']) if mask >> h & 1]
        node.update(terminal_output_mask_hex=hex(mask), terminal_output_count=len(output_h),
                    downstream_h=output_h, downstream_H8_groups=sorted({h//8 for h in output_h}))
        if node['kind'] == 'input':
            node['original_latent'] = int(live_latents[node['source']])
        else:
            operand_bits = []
            for side in ('lhs', 'rhs'):
                parent = graph['nodes'][node[side]]
                shift = node[side+'_shift']
                operand = [int(c) << shift for c in parent['coeff']]
                low, high = box(operand)
                bits = signed_width(low, high)
                node[side+'_shifted_range'] = [low, high]
                node[side+'_shifted_signed_bits'] = bits
                operand_bits.append(bits)
                maximum_operand = max(maximum_operand, abs(low), abs(high))
            node['conservative_full_add_sub_bits'] = max(operand_bits)+1
    graph.update(input_domain=[XLO, XHI], input_signed_bits=BITS,
                 original_input_latents=live_latents.tolist(),
                 instance='one time/position; 32 jointly consumed latent component integers',
                 terminal_bitmap_semantics='bit h means final V output h transitively consumes this node; not a PSN gate certificate',
                 width_semantics='exact per-node independent signed24 input-box range; shifted operands and full carry width separately listed')
    return maximum_operand


def verify(graph, official, K, V):
    n = K.shape[1]
    basis = np.eye(n, dtype=np.int64)
    rng = np.random.default_rng(20260910)
    random = rng.integers(XLO, XHI+1, size=(128, n), dtype=np.int64)
    corners = np.concatenate([
        np.zeros((1, n), np.int64),
        np.full((1, n), XLO, np.int64),
        np.full((1, n), XHI, np.int64),
        np.where(K >= 0, XHI, XLO),
        np.where(K >= 0, XLO, XHI),
    ])
    x = np.concatenate([basis, random, corners])
    expected = K @ x.T
    observed = np.column_stack([
        np.full(len(graph['nodes']), np.iinfo(np.int64).max, np.int64),
        np.full(len(graph['nodes']), np.iinfo(np.int64).min, np.int64)])
    got = eval_graph(graph, x.T, observed)
    official_x = np.ascontiguousarray(x, dtype=np.float64)
    official_y = official.predict(official_x, n_threads=1).T
    # All integer partial sums lie below 2**53, so this direct Float64 dyadic
    # relation is exact on the declared integer component interface.
    real_reference = V.T @ x.T.astype(np.float64)
    real_got = np.ldexp(got.astype(np.float64), -15)
    counts = dict(integer_DAG_mismatches=int(np.count_nonzero(got != expected)),
                  official_DAIS_mismatches=int(np.count_nonzero(official_y != expected)),
                  original_V_dyadic_relation_mismatches=int(np.count_nonzero(real_reference != real_got)))
    assert all(v == 0 for v in counts.values()), counts
    assert np.array_equal(real_got[:, :n], V.T)
    return dict(**counts, input_vectors=len(x), basis_vectors=n,
                full_domain_random_integer_vectors=len(random), directed_domain_corners=len(corners),
                tested_output_values=int(expected.size),
                tested_numerator_range=[int(expected.min()), int(expected.max())],
                symbolic_all_legal_integer_inputs_equal=True,
                source='basis, fixed-seed integer vectors and signed-domain extrema; not network capture')


def statistics(graph, pipeline):
    nodes = graph['nodes']
    ops = nodes[graph['n_input']:]
    live_ops = [n for n in ops if n['terminal_output_count']]
    exclusive = [n for n in live_ops if n['terminal_output_count'] == 1]
    one_H8 = [n for n in live_ops if len(n['downstream_H8_groups']) == 1]
    account = order_account(graph)
    account['read_model'] = 'one whole graph/context; resident input/intermediate RF and two initially empty value-forwarding registers'
    per_output = []
    for h in range(graph['n_output']):
        ancestors = [n for n in live_ops if h in n['downstream_h']]
        per_output.append(dict(h=h, output_node=graph['outputs'][h]['node'],
                               depth=nodes[graph['outputs'][h]['node']]['depth'],
                               signed_bits=graph['outputs'][h]['signed_bits'],
                               arithmetic_ancestors=len(ancestors),
                               exclusive_arithmetic_ancestors=sum(n['terminal_output_count'] == 1 for n in ancestors)))
    return dict(
        complete_matrix_add_sub_nodes=len(ops), reachable_add_sub_nodes=len(live_ops),
        dead_arithmetic_nodes=len(ops)-len(live_ops),
        additions=sum(not n['subtract'] for n in live_ops),
        subtractions=sum(n['subtract'] for n in live_ops),
        exact_operand_shift_histogram=dict(Counter(
            s for n in live_ops for s in (n['lhs_shift'], n['rhs_shift']) if s)),
        exact_output_shift_histogram=dict(Counter(o['shift'] for o in graph['outputs'])),
        exact_shift_semantics='constant wiring/scale aliases with full integer numerator precision; no rounding or dynamic barrel-shifter claim',
        maximum_add_sub_depth=max(n['depth'] for n in nodes),
        mathematical_node_width_histogram=dict(Counter(n['signed_bits'] for n in live_ops)),
        safe_full_adder_width_histogram=dict(Counter(n['conservative_full_add_sub_bits'] for n in live_ops)),
        maximum_output_signed_bits=max(o['signed_bits'] for o in graph['outputs']),
        maximum_immediate_fanout=max(n['fanout_edge_count'] for n in nodes),
        arithmetic_terminal_consumer_histogram=dict(Counter(n['terminal_output_count'] for n in live_ops)),
        arithmetic_terminal_H8_histogram=dict(Counter(len(n['downstream_H8_groups']) for n in live_ops)),
        exclusive_single_output_nodes=len(exclusive),
        single_H8_only_nodes=len(one_H8),
        multiple_final_output_nodes=len(live_ops)-len(exclusive),
        multiple_H8_nodes=len(live_ops)-len(one_H8),
        per_output=per_output,
        official_stage_node_counts=[len(s['arithmetic_nodes']) for s in graph['stage_info']],
        official_abstract_bit_cost=pipeline.cost,
        official_abstract_delay=pipeline.latency,
        official_cost_scope='FPGA-oriented solver proxies, not ASIC PPA or execution cycles',
        official_interpreter_wider_domain_nodes=sum(
            n['DAIS_declared_signed_bits'] > n['signed_bits'] for n in nodes),
        declared_order_account=account,
        cancellation_scope='Terminal-mask structure only. No actual gates/strict-bound checkpoints, cancellation simulation or savings measured; a node can be canceled only before issue and after all terminal outputs no longer need it.',
        counter_scope='No consumer counters/mask updates or synchronization included in solver node count. Static mask and immediate edges are exported for a future paid implementation.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, default=DEFAULT_PARENT)
    args = parser.parse_args()
    start = time.monotonic()
    parent = np.load(args.parent, allow_pickle=False)
    v_all = parent['v'].astype(np.float64)
    live = np.flatnonzero(np.any(v_all != 0, axis=1))
    assert v_all.shape == (48, 96) and np.array_equal(live, np.arange(32))
    V = v_all[live]
    assert np.count_nonzero(V) == 3072
    K = np.rint(np.ldexp(V.T, 15)).astype(np.int64)
    assert np.array_equal(np.ldexp(K.astype(np.float64), -15), V.T)
    # Complete official solve consumes C-order input-by-output storage.
    options = dict(method0='wmc', method1='auto', hard_dc=-1, decompose_dc=-2,
                   qintervals=[(float(XLO), float(XHI), 1.0)]*len(live),
                   adder_size=1, carry_size=1, search_all_decompose_dc=True)
    print('COMPILING whole preview V32->96, exact denominator 32768, signed24 components', flush=True)
    pipeline = solve(np.ascontiguousarray(K.T, dtype=np.float32), **options)
    assert np.array_equal(pipeline.kernel, K.T)
    graph = flatten(pipeline, K, 1 << (BITS-1), np.array([XLO]), np.array([XHI]))
    maximum_operand = annotate(graph, live)
    maximum_node = max(max(abs(n['static_min']), abs(n['static_max'])) for n in graph['nodes'])
    maximum_output = max(max(abs(o['static_min']), abs(o['static_max'])) for o in graph['outputs'])
    assert max(maximum_operand, maximum_node, maximum_output) < 2**53
    graph.update(original_V_input_by_output=V, numerator_K_output_by_input=K,
                 output_denominator=32768, physical_input_unit='delta_Z, common across 32 component lanes',
                 output_function='Y_num=K@Zq; Y=delta_Z*Y_num/32768. The final scale is exact, not an integer truncation.',
                 compiler_options=options, official_version=version('da4ml'))
    interpreter = normalized_interpreter(graph)
    checks = verify(graph, interpreter, K, V)
    stats = statistics(graph, pipeline)
    pipeline.save(HERE/'whole_official_pipeline.json')
    save(HERE/'whole_integer_dag.json', graph)
    result = dict(
        complete=True, parent=str(args.parent), official_source=str(OLD/'da4ml_official'),
        official_version=version('da4ml'), solver_options=options,
        original_V_shape=list(v_all.shape), live_latents=live,
        compiled_V_shape=list(V.shape), numerator_K_shape=list(K.shape),
        V_exact_nonzeros=3072,
        V_exact_power2_exponent_histogram=dict(Counter(int(x) for x in np.log2(np.abs(V)).astype(int).flat)),
        input_signed_bits=BITS, input_integer_domain=[XLO, XHI],
        output_denominator=32768, new_weight_quantization=False,
        independent_shift_term_count=3072, independent_output_reduction_additions=96*(32-1),
        input_scope='signed24 common-unit component interface, not a claim that original FP32 latents or differently scaled raw U accumulators already match it',
        source_theta=float(parent['theta_source']), output_theta=float(parent['theta_output']),
        original_U_per_latent_scale_exponents=parent['u_scale_exponent'][live],
        maximum_exact_node_abs_bound=maximum_node, maximum_shifted_operand_abs_bound=maximum_operand,
        maximum_output_numerator_abs_bound=maximum_output,
        exact_intermediates_below_2pow53=True,
        checks=checks, statistics=stats,
        graph_file='whole_integer_dag.json', official_pipeline_file='whole_official_pipeline.json',
        excluded=['source U production and per-latent scale alignment',
                  'temporal A-before-V production; later PSN consumer and bias/BN/threshold',
                  'rounding to a deployed latent/output format and FP32 network equivalence',
                  'consumer certificates/counters, finite ports, backpressure, wire/fanout and pipeline alignment'],
        no_AEE_or_timing_claim=True, wall_seconds=time.monotonic()-start)
    save(HERE/'result.json', result)
    print('PASS nodes', stats['complete_matrix_add_sub_nodes'],
          'depth', stats['maximum_add_sub_depth'],
          'single_output', stats['exclusive_single_output_nodes'],
          'cross_H8', stats['multiple_H8_nodes'],
          'seconds', round(result['wall_seconds'], 3), flush=True)


if __name__ == '__main__':
    main()
