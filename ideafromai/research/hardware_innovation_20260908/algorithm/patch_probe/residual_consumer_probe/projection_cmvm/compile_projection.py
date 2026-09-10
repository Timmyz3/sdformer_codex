"""Complete official da4ml baseline for the actual continuous PED projection.

One predeclared solver setting: wmc/auto, unrestricted delay, official internal
decomposition search. Compile Wq[96,96] as a whole and as twelve output-H8
matrices. This is not a collection of independent CSD multipliers. No training,
new capture, RTL generation, EDA, or changes to the old/compiler source tree.

Xq is the real continuous residual input rounded/clamped to signed X12;
Wq is per-output-row dyadic INT8. The graph implements Wq @ Xq exactly.
input_scale * row_scale and original projection bias stay outside this integer
graph. It does not equate the new quantized student with FP32 network arithmetic.

Reuse the previously verified pipeline flattening, C-contiguous matrix binding,
and scale/sign-alias normalization. Here the X12 interval is asymmetric, so
the DAIS stored-sign interval is corrected explicitly. All symbolic coefficients
and exact box-domain widths use Python integers. The official interpreter gets
a conservatively rounded interval if a float32 endpoint cannot represent an
integer. Its declared interval/width is recorded separately from the minimum
mathematical width; no hidden interpreter widening is called a circuit result.

One graph instance is ONE (spatial position, time) vector of 96 input channels.
Do not multiply its temporary state by '96 independent H lanes' as for the old
10x10 PSN graph. Report scalar/context bits, an explicit P4 replication example,
resident X, temporary nodes, two operand registers, and output collection apart.
Output streaming is conditional on the sink accepting it; this is not a finite
port timing model. Whole vs H8 are independently compiled graphs, so their node
count difference is not itself an isolated attribution to cross-H8 CSE.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'DA_DEFAULT_THREADS'):
    os.environ[_key] = '1'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from collections import Counter, defaultdict
from importlib.metadata import version
import argparse
import json
import math
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = ROOT / 'psn/cmvm_20260909'
sys.path.insert(0, str(OLD))
from compile_psn import solve, flatten, signed_width, csd_digits, eval_graph
from da4ml.types import CombLogic, Op, QInterval

XLO, XHI = -2048, 2047


def save(path, obj):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value))
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=convert) + '\n')


def interval(coeff):
    pos = sum(int(c) for c in coeff if c > 0)
    neg = sum(int(c) for c in coeff if c < 0)
    return XLO * pos + XHI * neg, XHI * pos + XLO * neg


def fix_domain(graph, output_h):
    """Replace old symmetric-Y/sparse-source metadata by the true X12 box."""
    groups = [0] * len(graph['nodes'])
    outsets = [set() for _ in graph['nodes']]
    for out, h in zip(graph['outputs'], output_h):
        out['local_output'] = out.pop('t')
        out['h'] = int(h)
        lo, hi = interval(out['coeff'])
        out.update(static_min=lo, static_max=hi, signed_bits=signed_width(lo, hi))
        groups[out['node']] |= 1 << (int(h) // 8)
        outsets[out['node']].add(int(h))
    # The inherited independent evaluator uses 't' for the local output index.
    # Retain this compatibility alias, while all scope reports use explicit h.
    for out in graph['outputs']:
        out['t'] = out['local_output']
    for node in reversed(graph['nodes']):
        if node['kind'] != 'input':
            for parent in (node['lhs'], node['rhs']):
                groups[parent] |= groups[node['id']]
                outsets[parent].update(outsets[node['id']])
    for node in graph['nodes']:
        lo, hi = interval(node['coeff'])
        node.update(static_min=lo, static_max=hi, signed_bits=signed_width(lo, hi),
                    downstream_h=sorted(outsets[node['id']]),
                    downstream_H8_groups=[g for g in range(12) if groups[node['id']] >> g & 1])
        for key in ('binary_source_global_min', 'binary_source_global_max', 'binary_source_max_lane_bits'):
            node.pop(key, None)
    graph.update(input_domain=[XLO, XHI], input_signed_bits=12,
                 input_labels=list(range(96)), output_h=[int(h) for h in output_h],
                 input_semantics='continuous r1out quantized to X12; not theta*g',
                 exact_domain_method='independent X12 box; exact signed coefficient extrema',
                 instance='one (spatial position,time), with 96 coupled channel inputs')


def outward_float32(value, lower):
    result = np.float32(value)
    if (lower and float(result) > value) or (not lower and float(result) < value):
        result = np.nextafter(result, np.float32(-np.inf if lower else np.inf))
    return float(result)


def normalized_interpreter(graph):
    """Same official add/sub graph, exact integer alias/sign normalization."""
    ops, signs = [], []
    for node in graph['nodes']:
        if node['kind'] == 'input':
            left, right, opcode, shift, sign = node['source'], -1, -1, 0, 1
        else:
            assert min(node['lhs_shift'], node['rhs_shift']) == 0
            terms = [(node['lhs'], node['lhs_shift'], signs[node['lhs']]),
                     (node['rhs'], node['rhs_shift'], signs[node['rhs']] * (-1 if node['subtract'] else 1))]
            if terms[0][1] != 0:
                terms.reverse()
            (left, _, sign), (right, shift, other_sign) = terms
            opcode = int(sign != other_sign)
        lo, hi = node['static_min'], node['static_max']
        stored_lo, stored_hi = (lo, hi) if sign > 0 else (-hi, -lo)
        qlo, qhi = outward_float32(stored_lo, True), outward_float32(stored_hi, False)
        node['DAIS_stored_sign'] = sign
        node['DAIS_declared_integer_interval'] = [qlo, qhi]
        node['DAIS_declared_signed_bits'] = signed_width(math.floor(qlo), math.ceil(qhi))
        ops.append(Op(left, right, opcode, shift, QInterval(qlo, qhi, 1.0),
                      node.get('official_abstract_arrival', 0.0),
                      node.get('official_abstract_bit_cost', 0.0)))
        signs.append(sign)
    outputs = graph['outputs']
    return CombLogic((graph['n_input'], graph['n_output']), [0] * graph['n_input'],
                     [o['node'] for o in outputs], [o['shift'] for o in outputs],
                     [o['sign'] * signs[o['node']] < 0 for o in outputs], ops, 1, 1)


def order_account(graph):
    """Official topological order with last-use freeing and two-value forwarding.

    Persistent X is separate; completed outputs may stream to an accepting sink.
    The temporary peak includes a new result before its operands are freed.
    A delayed sink instead needs the explicitly reported output collection state.
    """
    nodes = graph['nodes']
    uses = [len(n['fanout_edges']) for n in nodes]
    live, held = set(), []
    live_bits = peak_bits = peak_values = reads = input_reads = writes = 0
    read_bits = write_bits = 0
    for node in nodes[graph['n_input']:]:
        ident = node['id']
        for parent in (node['lhs'], node['rhs']):
            if parent not in held:
                reads += 1
                input_reads += parent < graph['n_input']
                read_bits += nodes[parent]['signed_bits']
                if len(held) == 2:
                    held.pop(0)
            else:
                held.remove(parent)
            held.append(parent)
        peak_bits = max(peak_bits, live_bits + node['signed_bits'])
        peak_values = max(peak_values, len(live) + 1)
        for parent in (node['lhs'], node['rhs']):
            uses[parent] -= 1
            if uses[parent] == 0 and parent in live:
                live.remove(parent)
                live_bits -= nodes[parent]['signed_bits']
        if uses[ident]:
            live.add(ident)
            live_bits += node['signed_bits']
            writes += 1
            write_bits += node['signed_bits']
        if len(held) == 2:
            held.pop(0)
        held.append(ident)
    operand_bits = 2 * max(n['signed_bits'] for n in nodes)
    input_bits = sum(n['signed_bits'] for n in nodes[:graph['n_input']])
    output_bits = sum(o['signed_bits'] for o in graph['outputs'])
    return dict(order='official topological node-id order',
                temporary_peak_bits_per_context=peak_bits,
                temporary_peak_values=peak_values,
                two_operand_register_bits_per_context=operand_bits,
                resident_input_bits_per_context=input_bits,
                streamed_output_collection_bits=0,
                collect_all_outputs_bits_per_context=output_bits,
                P4_example_temporary_bits=4 * peak_bits,
                P4_example_input_bits=4 * input_bits,
                RF_reads=reads, RF_input_reads=input_reads, RF_intermediate_reads=reads-input_reads,
                RF_writes=writes, RF_read_bits=read_bits, RF_write_bits=write_bits,
                read_model='resident input/intermediate RF, two-value forwarding, new H8 graph starts with empty forwarding registers',
                state_excludes='pipeline alignment, retiming copies, fanout buffers, addresses/control, scale/bias units and blocked-output buffering',
                timing_claim=False)


def canonical_form(coeff):
    nonzero = [abs(int(c)) for c in coeff if c]
    if not nonzero:
        return tuple(coeff)
    shift = min((v & -v).bit_length()-1 for v in nonzero)
    sign = 1 if next(c for c in coeff if c) > 0 else -1
    return tuple((int(c) >> shift) * sign for c in coeff)


def graph_statistics(graph, pipe):
    ops = graph['nodes'][graph['n_input']:]
    multi_group = [n for n in ops if len(n['downstream_H8_groups']) > 1]
    return dict(add_sub_nodes=len(ops), additions=sum(not n['subtract'] for n in ops),
                subtractions=sum(n['subtract'] for n in ops),
                stages=[len(s['arithmetic_nodes']) for s in graph['stage_info']],
                maximum_add_depth=max(n['depth'] for n in graph['nodes']),
                output_depths=[graph['nodes'][o['node']]['depth'] for o in graph['outputs']],
                mathematical_width_histogram=dict(Counter(n['signed_bits'] for n in ops)),
                arithmetic_node_bits=sum(n['signed_bits'] for n in ops),
                maximum_fanout_edges=max(n['fanout_edge_count'] for n in graph['nodes']),
                shared_arithmetic_nodes=sum(n['distinct_consumers'] > 1 for n in ops),
                arithmetic_nodes_with_multiple_final_H8_consumers=len(multi_group),
                multiple_H8_node_ids=[n['id'] for n in multi_group],
                multiple_H8_node_bits=sum(n['signed_bits'] for n in multi_group),
                DAIS_interpreter_nodes_with_wider_declared_domain=sum(
                    n['DAIS_declared_signed_bits'] > n['signed_bits'] for n in graph['nodes']),
                official_abstract_bit_cost=pipe.cost, official_abstract_delay=pipe.latency,
                official_cost_domain='compiler FPGA-oriented proxy; not ASIC PPA or cycle counts',
                order_account=order_account(graph))


def check_vectors(graph, interpreter, matrix, actual, corners):
    observed = np.column_stack((np.full(len(graph['nodes']), np.iinfo(np.int64).max, np.int64),
                                np.full(len(graph['nodes']), np.iinfo(np.int64).min, np.int64)))
    reference = matrix @ actual.T.astype(np.int64)
    got = eval_graph(graph, actual.T.astype(np.int64), observed)
    mismatch = int(np.count_nonzero(reference != got))
    for i in range(graph['n_input']):
        observed[i] = [int(actual[:, i].min()), int(actual[:, i].max())]
    corner_observed = np.zeros_like(observed)
    expected_corner = matrix @ corners.T.astype(np.int64)
    corner_got = eval_graph(graph, corners.T.astype(np.int64), corner_observed)
    corner_mismatch = int(np.count_nonzero(corner_got != expected_corner))
    pick = np.unique(np.linspace(0, len(actual)-1, min(64, len(actual)), dtype=int))
    official_x = np.ascontiguousarray(np.concatenate([actual[pick], corners]), dtype=np.float64)
    official_y = interpreter.predict(official_x, n_threads=1)
    official_ref = official_x @ matrix.T.astype(np.float64)
    official_mismatch = int(np.count_nonzero(official_y != official_ref))
    assert mismatch == corner_mismatch == official_mismatch == 0
    assert max(max(abs(n['static_min']), abs(n['static_max'])) for n in graph['nodes']) < 2**53
    return dict(actual_vectors=len(actual), actual_integer_output_values=int(reference.size),
                actual_integer_DAG_mismatches=mismatch,
                actual_output_range=[int(reference.min()), int(reference.max())],
                exact_domain_corner_vectors=len(corners), domain_corner_mismatches=corner_mismatch,
                official_actual_vector_indices=pick, official_actual_vectors=len(pick),
                official_domain_corner_vectors=len(corners), official_output_values=int(official_ref.size),
                official_DAIS_mismatches=official_mismatch,
                all_legal_integer_inputs_symbolic_matrix_equal=True), observed


def comparison(cases, graphs):
    whole = cases['whole']['statistics']
    parts = [cases[f'h8_{g:02d}']['statistics'] for g in range(12)]
    forms = defaultdict(set)
    for g in range(12):
        graph = graphs[f'h8_{g:02d}']
        for node in graph['nodes'][96:]:
            forms[canonical_form(node['coeff'])].add(g)
    repeated = [sorted(v) for v in forms.values() if len(v) > 1]
    return dict(
        whole_nodes=whole['add_sub_nodes'],
        sum_twelve_H8_nodes=sum(p['add_sub_nodes'] for p in parts),
        whole_depth=whole['maximum_add_depth'],
        largest_H8_depth=max(p['maximum_add_depth'] for p in parts),
        whole_cross_H8_arithmetic_nodes=whole['arithmetic_nodes_with_multiple_final_H8_consumers'],
        repeated_signed_power2_equivalent_intermediate_forms_across_H8=len(repeated),
        repeated_H8_form_group_histogram=dict(Counter(len(v) for v in repeated)),
        whole_order=whole['order_account'],
        sequential_H8=dict(
            resident_input_bits_per_context=96*12,
            temporary_peak_bits_per_context=max(p['order_account']['temporary_peak_bits_per_context'] for p in parts),
            two_operand_register_bits_per_context=max(p['order_account']['two_operand_register_bits_per_context'] for p in parts),
            collect_all_96_outputs_bits_per_context=sum(p['order_account']['collect_all_outputs_bits_per_context'] for p in parts),
            RF_reads=sum(p['order_account']['RF_reads'] for p in parts),
            RF_input_reads=sum(p['order_account']['RF_input_reads'] for p in parts),
            RF_intermediate_reads=sum(p['order_account']['RF_intermediate_reads'] for p in parts),
            RF_writes=sum(p['order_account']['RF_writes'] for p in parts)),
        interpretation='Independent official compiles: node difference includes decomposition choices, not a pure cross-H8-CSE ablation. Repeated forms are an opportunity count only. No port/timing/PPA comparison.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameters', type=Path, default=HERE.parent/'projection_control_parameters.npz')
    args = parser.parse_args()
    started = time.monotonic()
    parameters = np.load(args.parameters, allow_pickle=False)
    W = parameters['Wq'].astype(np.int64)
    actual = parameters['calibration_Xq'].astype(np.int64)
    assert W.shape == (96, 96) and np.all((W >= -127) & (W <= 127))
    assert actual.ndim == 2 and actual.shape[1] == 96 and len(actual) > 0
    assert np.all((actual >= XLO) & (actual <= XHI))
    # Signed-domain output extrema catch sign/shift/interval mistakes. These are
    # directed arithmetic checks, separate from the real captured-vector results.
    corners = np.concatenate([
        np.zeros((1, 96), np.int64), np.full((1, 96), XLO, np.int64), np.full((1, 96), XHI, np.int64),
        np.where(W >= 0, XHI, XLO), np.where(W >= 0, XLO, XHI)], axis=0)
    options = dict(method0='wmc', method1='auto', hard_dc=-1, decompose_dc=-2,
                   qintervals=[(float(XLO), float(XHI), 1.0)]*96,
                   adder_size=1, carry_size=1, search_all_decompose_dc=True)
    result = dict(complete=False, kind='official complete CMVM, integer proof and captured arithmetic; not timing/PPA',
                  parameters=str(args.parameters), official_version=version('da4ml'), official_source=str(OLD/'da4ml_official'),
                  integer_function='output_integer[h] = sum_c Wq[h,c] * Xq[c]',
                  matrix_shape=list(W.shape), input_integer_domain=[XLO, XHI], input_signed_bits=12,
                  input_scale=float(parameters['input_scale']), row_scale=parameters['row_scale'],
                  output_scale=float(parameters['input_scale'])*parameters['row_scale'],
                  output_bias=parameters['original_bias'],
                  output_scale_bias_claim='external to integer DAG; decode output_integer*output_scale+original_bias, no FP/network equivalence claim',
                  actual_vectors=len(actual), actual_input_values=int(actual.size),
                  actual_Xq_zero_values=int(np.count_nonzero(actual == 0)),
                  Wq_nonzeros=int(np.count_nonzero(W)), independent_CSD_digits=sum(csd_digits(v) for v in W.flat),
                  independent_CSD_add_subs=sum(csd_digits(v) for v in W.flat)-int(np.count_nonzero(np.any(W, axis=1))),
                  compiler_options=options, cases={},
                  scope='parameter-package calibration_Xq: four captured train frames, all T10 and the 64 real anchors/frame; not full-frame or valid825 performance',
                  excluded=['upstream source production', 'source/weight SRAM service', 'scale/bias service',
                            'clock, wire/shift/fanout and pipeline alignment', 'finite ports, backpressure and complete PED schedule'])
    save(HERE/'result.json', result)
    graphs = {}
    scopes = [('whole', np.arange(96))] + [(f'h8_{g:02d}', np.arange(8*g, 8*g+8)) for g in range(12)]
    for label, output_h in scopes:
        before = time.monotonic()
        A = np.ascontiguousarray(W[output_h])
        print('COMPILING', label, A.shape, flush=True)
        # Official native binding consumes raw C-order input-by-output storage.
        pipeline = solve(np.ascontiguousarray(A.T, dtype=np.float32), **options)
        assert np.array_equal(pipeline.kernel, A.T)
        pipeline.save(HERE/f'{label}_official_pipeline.json')
        graph = flatten(pipeline, A, 2048, np.array([XLO]), np.array([XHI]))
        fix_domain(graph, output_h)
        graph.update(official_options=options, official_version=version('da4ml'),
                     method='official matrix decomposition + weighted cross-output signed-shift CSE')
        interpreter = normalized_interpreter(graph)
        interpreter.save_binary(HERE/f'{label}.dais')
        stats = graph_statistics(graph, pipeline)
        save(HERE/f'{label}_integer_dag.json', graph)
        checks, observed = check_vectors(graph, interpreter, A, actual, corners)
        save(HERE/f'{label}_observed_node_ranges.json', dict(source='real calibration Xq only', ranges=observed))
        graphs[label] = graph
        result['cases'][label] = dict(output_h=output_h, statistics=stats, checks=checks,
                                     wall_seconds=time.monotonic()-before,
                                     graph_file=f'{label}_integer_dag.json', DAIS_file=f'{label}.dais')
        result['wall_seconds'] = time.monotonic()-started
        save(HERE/'result.json', result)
        print('PASS', label, 'nodes', stats['add_sub_nodes'], 'depth', stats['maximum_add_depth'],
              'tmp_bits', stats['order_account']['temporary_peak_bits_per_context'],
              'seconds', round(time.monotonic()-before, 2), flush=True)
    result['comparison'] = comparison(result['cases'], graphs)
    result['complete'] = True
    result['wall_seconds'] = time.monotonic()-started
    save(HERE/'result.json', result)
    print('FINISHED', round(result['wall_seconds'], 2), flush=True)


if __name__ == '__main__':
    main()
