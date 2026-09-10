"""Official da4ml whole-matrix baseline for the existing S0 support student.

No training, new network capture, RTL generation, EDA, or production edits.
Input is the recorded forced-support theta*g plus its actual INT8 effective W.
Yi is reconstructed exactly on CPU; the downstream Aq/thresholds are those of
the fixed integer consumer, NOT the trainable upstream source A in the student
state dictionary. Q14 is already included in Aq and tau; there is no right shift
before the integer comparison. The output payload remains theta_output*g.

da4ml is used as its complete two-stage CMVM solver, not emulated with separate
CSD multipliers. Two predeclared official modes are emitted: unrestricted and
minimum compiler-delay constraint. The compiler's internal decomposition search
is part of its published algorithm. Its cost/delay are abstract FPGA-oriented
proxies, never ASIC area, ns, or measured cycles.

DAIS can use fractional intermediate scales. This adapter removes only exact
power-of-two aliases and normalizes every stored node to an integer numerator.
The complete ten-dimensional coefficient vector of every node is retained;
each final coefficient is checked against Aq with Python integers. Exact static
node ranges, instead of rounded float32 interval endpoints, determine our widths.

All old files are imported read-only. Numerical verification streams all 10
existing captures, all P=19200/H=384/T=10 values. BLAS reconstruction of integer
Yi is exact because all products and every partial sum lie below 2**53. The
independent graph evaluator and U reference use INT64. Official DAIS interpreter
is additionally checked on 32 fixed spatial positions per frame (all H/T).

This first delivery includes static graph and read/write/liveness attribution.
Port lower bounds and serialized register traffic are NOT a closed global
FC1/PSN schedule. Ordinary strict bit-demand execution remains a listed gap;
the adapter does not invent free interval evaluation or claim a speedup.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'DA_DEFAULT_THREADS'):
    os.environ[_key] = '1'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
from collections import Counter
from fractions import Fraction
from importlib.metadata import version
import json
import math
import time

import numpy as np
from da4ml.cmvm import solve
from da4ml.types import CombLogic, Op, QInterval

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT/'bn_state'))
from support_service_model import read_torch, CAP, ALG, psn_schedule

PREFIX = 'sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.'
T, H, P = 10, 384, 19200
MODES = {'whole_cmvm': -1, 'whole_cmvm_min_delay': 0}


def save(name, data):
    def convert(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, Path):
            return str(v)
        raise TypeError(type(v))
    (HERE/name).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=convert)+'\n')


def signed_width(lo, hi):
    bits = 1
    while lo < -(1 << (bits-1)) or hi > (1 << (bits-1))-1:
        bits += 1
    return bits


def scaled_coeff(coef, shift, sign=1):
    if shift >= 0:
        return [int(c)*sign*(1 << shift) for c in coef]
    denom = 1 << -shift
    assert all(int(c) % denom == 0 for c in coef)
    return [int(c)*sign//denom for c in coef]


def flatten(pipeline, A, y_bound, y_low_h, y_high_h):
    """Return pure integer DAG with exact scale/sign output aliases."""
    nodes = []
    n_input, n_output = A.shape[1], A.shape[0]
    for t in range(n_input):
        coeff = [int(t == i) for i in range(n_input)]
        nodes.append(dict(id=t, kind='input', source=t, coeff=coeff, depth=0))
    previous = [(i, 0, 1) for i in range(n_input)]  # id, exponent, sign
    stage_info = []
    for stage, sol in enumerate(pipeline.solutions):
        refs = []
        stage_ids = []
        for local, op in enumerate(sol.ops):
            if op.opcode == -1:
                ident, shift, sign = previous[op.id0]
                refs.append((ident, shift+int(sol.inp_shifts[op.id0]), sign))
                continue
            assert op.opcode in (0, 1), op
            id0, shift0, sign0 = refs[op.id0]
            id1, shift1, sign1 = refs[op.id1]
            shift1 += int(op.data)
            sign1 *= -1 if op.opcode else 1
            common_shift = min(shift0, shift1)
            left_shift, right_shift = shift0-common_shift, shift1-common_shift
            subtract = sign0 != sign1
            c0, c1 = nodes[id0]['coeff'], nodes[id1]['coeff']
            coeff = [(a << left_shift) + (-1 if subtract else 1)*(b << right_shift)
                     for a, b in zip(c0, c1)]
            ident = len(nodes)
            nodes.append(dict(id=ident, kind='addsub', stage=stage, local_op=local,
                              lhs=id0, rhs=id1, lhs_shift=left_shift, rhs_shift=right_shift,
                              subtract=subtract, coeff=coeff,
                              depth=max(nodes[id0]['depth'], nodes[id1]['depth'])+1,
                              official_abstract_bit_cost=float(op.cost),
                              official_abstract_arrival=float(op.latency),
                              official_qinterval=list(op.qint),
                              raw_value_scale_exponent=common_shift, raw_value_sign=sign0))
            refs.append((ident, common_shift, sign0))
            stage_ids.append(ident)
        previous = []
        for idx, sh, neg in zip(sol.out_idxs, sol.out_shifts, sol.out_negs):
            assert idx >= 0
            ident, shift, sign = refs[idx]
            previous.append((ident, shift+int(sh), sign*(-1 if neg else 1)))
        stage_info.append(dict(stage=stage, shape=sol.shape, arithmetic_nodes=stage_ids,
                               matrix_input_by_output=sol.kernel,
                               exact_input_shifts=sol.inp_shifts,
                               exact_output_shifts=sol.out_shifts,
                               exact_output_negations=sol.out_negs))
    outputs = []
    for t, (ident, shift, sign) in enumerate(previous):
        actual = scaled_coeff(nodes[ident]['coeff'], shift, sign)
        assert actual == A[t].tolist(), (t, actual, A[t].tolist())
        outputs.append(dict(t=t, node=ident, shift=shift, sign=sign, coeff=actual))
    fanouts = [[] for _ in nodes]
    output_users = [[] for _ in nodes]
    for node in nodes[n_input:]:
        fanouts[node['lhs']].append(node['id'])
        fanouts[node['rhs']].append(node['id'])
    for out in outputs:
        output_users[out['node']].append(out['t'])
    for node in nodes:
        c = node['coeff']
        bound = sum(abs(v) for v in c)*y_bound
        # The signed source superset matches the previous legal Yi13/U29 audit.
        node['static_min'] = -bound
        node['static_max'] = bound
        node['signed_bits'] = signed_width(-bound, bound)
        cn = np.asarray(c, np.int64)
        lo = np.maximum(cn, 0).sum()*y_low_h + np.minimum(cn, 0).sum()*y_high_h
        hi = np.maximum(cn, 0).sum()*y_high_h + np.minimum(cn, 0).sum()*y_low_h
        node['binary_source_global_min'] = int(lo.min())
        node['binary_source_global_max'] = int(hi.max())
        node['binary_source_max_lane_bits'] = max(signed_width(int(l), int(h)) for l, h in zip(lo, hi))
        node['fanout_edges'] = fanouts[node['id']]
        node['fanout_edge_count'] = len(fanouts[node['id']])+len(output_users[node['id']])
        node['distinct_consumers'] = len(set(fanouts[node['id']]))+len(output_users[node['id']])
        node['output_consumers'] = output_users[node['id']]
    return dict(n_input=n_input,n_output=n_output,nodes=nodes, outputs=outputs, stage_info=stage_info,
                exact_symbolic_matrix=A, symbolic_all_integer_inputs_equal=True)


def csd_digits(value):
    v = abs(int(value))
    terms = 0
    while v:
        if v & 1:
            digit = 2-(v & 3)
            v -= digit
            terms += 1
        v >>= 1
    return terms


def normalized_dais(graph):
    """One official DAIS block with exact integer scales across stage aliases.

    solve() 0.6.0 stage-1 output shifts are not included in the qintervals of
    stage-2 copy inputs. Calling predict() separately on these stages therefore
    wraps valid integer data. This adapter leaves every solved add/sub intact,
    removes stage-boundary aliases, and gives the official interpreter corrected
    exact widths. It is not a replacement CSE algorithm.
    """
    ops, signs = [], []
    for n in graph['nodes']:
        qint = QInterval(float(n['static_min']),float(n['static_max']),1.0)
        if n['kind'] == 'input':
            ops.append(Op(n['source'],-1,-1,0,qint,0.0,0.0))
            signs.append(1)
        else:
            # Both normalized shifts can be nonzero only if common factors
            # survived normalization, which does not occur for the imported
            # integer matrices. Use a signed relative shift and output scale
            # if necessary rather than silently allocate another arithmetic op.
            assert min(n['lhs_shift'],n['rhs_shift']) == 0
            terms = [(n['lhs'],n['lhs_shift'],signs[n['lhs']]),
                     (n['rhs'],n['rhs_shift'],signs[n['rhs']]*(-1 if n['subtract'] else 1))]
            if terms[0][1] != 0:
                terms.reverse()
            (left,_,sgn0),(right,shift,sgn1) = terms
            ops.append(Op(left,right,int(sgn0 != sgn1),shift,qint,
                          n['official_abstract_arrival'],n['official_abstract_bit_cost']))
            signs.append(sgn0)
    return CombLogic((graph['n_input'],graph['n_output']),[0]*graph['n_input'],[o['node'] for o in graph['outputs']],
                     [o['shift'] for o in graph['outputs']],
                     [o['sign']*signs[o['node']] < 0 for o in graph['outputs']],
                     ops,1,1)


def graph_order_account(graph, pressure=False):
    """A legal static DAG order, last-use storage, and two-register traffic.

    All input Y values are already in the persistent tile RF and excluded from
    temporary liveness. Outputs may retire to the existing output register at
    completion. Read forwarding through two operand registers is explicitly
    simulated. This is a traffic/order model, not a pipelined cycle scheduler.
    """
    nodes, outputs = graph['nodes'], graph['outputs']
    n_input = graph['n_input']
    users = {n['id']: len(n['fanout_edges']) for n in nodes}
    pending = set(range(n_input, len(nodes)))
    ready_values = set(range(n_input))
    live = set()
    order, peak_bits, peak_values, reads, y_reads, writes = [], 0, 0, 0, 0, 0
    held = []
    read_bits = write_bits = 0
    while pending:
        ready = [i for i in sorted(pending) if nodes[i]['lhs'] in ready_values and nodes[i]['rhs'] in ready_values]
        assert ready
        def key(i):
            op = nodes[i]
            consumed = Counter((op['lhs'], op['rhs']))
            freed = sum(nodes[j]['signed_bits'] for j, n in consumed.items()
                        if j in live and users[j] == n)
            return (freed-op['signed_bits'], len(op['output_consumers']), -i)
        ident = max(ready, key=key) if pressure else ready[0]
        op = nodes[ident]
        for src in (op['lhs'], op['rhs']):
            if src not in held:
                reads += 1
                y_reads += src < n_input
                read_bits += nodes[src]['signed_bits']
                if len(held) == 2:
                    held.pop(0)
                held.append(src)
            else:
                held.remove(src)
                held.append(src)
        # The output must exist before last-read operands can be reused; count
        # an explicit result vector in addition to all live RF entries.
        peak_bits = max(peak_bits, sum(nodes[i]['signed_bits'] for i in live)+op['signed_bits'])
        peak_values = max(peak_values, len(live)+1)
        for src in (op['lhs'], op['rhs']):
            users[src] -= 1
            if users[src] == 0:
                live.discard(src)
        if users[ident]:
            live.add(ident)
            writes += 1
            write_bits += op['signed_bits']
        # A result forwarding register can replace the older operand register;
        # count it within the same two held values, rather than add a free cache.
        if len(held) == 2:
            held.pop(0)
        held.append(ident)
        ready_values.add(ident)
        pending.remove(ident)
        order.append(ident)
    return dict(order=order, temporary_peak_scalar_bits=peak_bits,
                temporary_peak_vectors=peak_values, temporary_peak_bytes_96lane=peak_bits*12,
                RF_read_vectors=reads, RF_Y_read_vectors=y_reads,
                RF_intermediate_read_vectors=reads-y_reads, RF_write_vectors=writes,
                RF_read_scalar_bits=read_bits, RF_write_scalar_bits=write_bits,
                two_operand_register_bits=2*max(n['signed_bits'] for n in nodes),
                one_read_port_issue_lower_bound=max(len(order), reads),
                two_read_port_issue_lower_bound=max(len(order), math.ceil(reads/2)),
                no_latency_or_overlap_claim=True)


def specialize_known_zero(graph, pattern):
    """Ordinary static propagation from known-zero Y rows, no output oracle.

    Counts adds that remain after constant-zero propagation and shift/sign
    aliases. Does not re-run CMVM on each pattern, and is an arithmetic lower
    bound until the routing/control implementation is supplied.
    """
    n_input=graph['n_input']
    refs = [(i, 0, 1) if pattern >> i & 1 else None for i in range(n_input)]
    ops = 0
    for n in graph['nodes'][n_input:]:
        a, b = refs[n['lhs']], refs[n['rhs']]
        if a is None and b is None:
            refs.append(None)
        elif a is None:
            refs.append((b[0], b[1]+n['rhs_shift'], b[2]*(-1 if n['subtract'] else 1)))
        elif b is None:
            refs.append((a[0], a[1]+n['lhs_shift'], a[2]))
        else:
            ops += 1
            refs.append((n['id'], 0, 1))
    return ops


def eval_graph(graph, x, observed):
    """INT64 streaming SSA evaluation; x is (T,N)."""
    nodes = graph['nodes']
    users = np.array([len(n['fanout_edges'])+len(n['output_consumers']) for n in nodes], np.int64)
    n_input=graph['n_input']
    values = {t: x[t] for t in range(n_input)}
    for n in nodes[n_input:]:
        ident, lhs, rhs = n['id'], n['lhs'], n['rhs']
        value = (values[lhs] << n['lhs_shift']) + (-1 if n['subtract'] else 1)*(values[rhs] << n['rhs_shift'])
        lo, hi = int(value.min()), int(value.max())
        assert lo >= n['static_min'] and hi <= n['static_max']
        observed[ident, 0] = min(observed[ident, 0], lo)
        observed[ident, 1] = max(observed[ident, 1], hi)
        values[ident] = value
        for src in (lhs, rhs):
            users[src] -= 1
            if users[src] == 0:
                del values[src]
    out = []
    for o in graph['outputs']:
        value = values[o['node']]*o['sign']
        if o['shift'] >= 0:
            value = value << o['shift']
        else:
            denom = 1 << -o['shift']
            assert np.all(value % denom == 0)
            value = value//denom
        out.append(value)
    return np.stack(out)


def gate(u, params, channel_indices):
    tau = params['threshold_int64'][:, channel_indices]
    positive = params['positive_gain'][channel_indices][None]
    const = params['constant_channels'][channel_indices][None]
    fixed = params['constant_gate'][:, channel_indices]
    return np.where(const, fixed, np.where(positive, u >= tau, u <= tau))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=10)
    parser.add_argument('--chunk', type=int, default=128)
    args = parser.parse_args()
    started = time.monotonic()
    param_path = ALG/'integer_s0_valid825/integer_parameters.pt'
    params = read_torch(param_path)[PREFIX]
    W = read_torch(CAP/'forced_code_weight_int8.pt').astype(np.int64)
    A = params['temporal_int16'].astype(np.int64)
    assert A.shape == (10, 10) and W.shape == (384, 96)
    y_bound = int(np.abs(W).sum(1).max())
    y_low_h = np.minimum(W, 0).sum(1)
    y_high_h = np.maximum(W, 0).sum(1)
    assert y_bound*np.abs(A).sum(1).max() < 2**53
    pipelines, graphs, descriptions, interpreters, api_checks = {}, {}, {}, {}, {}
    for label, hard_dc in MODES.items():
        options = dict(method0='wmc', method1='auto', hard_dc=hard_dc,
                       decompose_dc=-2, qintervals=[(-float(y_bound), float(y_bound), 1.0)]*T,
                       adder_size=1, carry_size=1, search_all_decompose_dc=True)
        # Official binding consumes raw C-order memory; do not pass an F-order
        # transpose here. kernel is input-by-output, while local A is t-by-s.
        pipe = solve(np.ascontiguousarray(A.T, dtype=np.float32), **options)
        assert np.array_equal(pipe.kernel, A.T)
        pipe.save(HERE/f'{label}_official_pipeline.json')
        graph = flatten(pipe, A, y_bound, y_low_h, y_high_h)
        graphs[label], pipelines[label] = graph, pipe
        graph['official_version'] = version('da4ml')
        graph['official_options'] = options
        graph['method'] = 'official matrix decomposition + weighted cross-output signed-shift CSE'
        save(f'{label}_integer_dag.json', graph)
        interpreter = normalized_dais(graph)
        interpreter.save_binary(HERE/f'{label}_normalized.dais')
        interpreters[label] = interpreter
        example = np.ascontiguousarray(np.arange(T,dtype=np.float64)[None])
        raw = example
        for stage in pipe.solutions:
            raw = stage.predict(raw,n_threads=1)
        fixed = interpreter.predict(example,n_threads=1)
        expected = example @ A.T
        assert np.array_equal(fixed,expected)
        api_checks[label] = dict(input=example, direct_reference=expected,
                                raw_stagewise_predict=raw, normalized_DAIS_predict=fixed,
                                raw_stagewise_mismatches=int(np.count_nonzero(raw != expected)),
                                normalized_mismatches=0)
        nodes = graph['nodes']
        descriptions[label] = dict(
            additions=len(nodes)-T, add_count=sum(not n['subtract'] for n in nodes[T:]),
            sub_count=sum(n['subtract'] for n in nodes[T:]),
            stages=[len(s['arithmetic_nodes']) for s in graph['stage_info']],
            output_add_depths=[nodes[o['node']]['depth'] for o in graph['outputs']],
            max_add_depth=max(n['depth'] for n in nodes),
            normalized_node_width_histogram=dict(Counter(n['signed_bits'] for n in nodes[T:])),
            all_node_signed_bits=sum(n['signed_bits'] for n in nodes),
            arithmetic_output_bits=sum(n['signed_bits'] for n in nodes[T:]),
            peak_fanout_edges=max(n['fanout_edge_count'] for n in nodes),
            input_fanout_edges=[n['fanout_edge_count'] for n in nodes[:T]],
            shared_nodes=sum(n['distinct_consumers'] > 1 for n in nodes[T:]),
            official_abstract_bit_cost=pipe.cost, official_abstract_delay=pipe.latency,
            official_topological_order=graph_order_account(graph),
            pressure_aware_ordinary_order=graph_order_account(graph, pressure=True),
            known_zero_pattern_adds=[specialize_known_zero(graph, p) for p in range(1024)])
        print('COMPILED', label, descriptions[label]['additions'], descriptions[label]['max_add_depth'], flush=True)
    save('official_interface_notes.json',dict(
        version=version('da4ml'),
        kernel_layout='C++ binding uses raw C-order memory; an F-contiguous transposed array must first be made C-contiguous.',
        stage_boundary='Raw stage-2 input intervals omit stage-1 output shifts; standalone stage predict wraps valid values.',
        adapter='Same official graph, exact integer scale aliases merged into one DAIS block; symbolic coefficients and independent integer evaluator check the graph.',
        examples=api_checks))
    digit_count = sum(csd_digits(v) for v in A.flat)
    input_info = dict(
        student='existing forced_code S0 block0; fixed downstream integer consumer',
        params_file=param_path, params_key=PREFIX, source_dir=CAP,
        weight_file=CAP/'forced_code_weight_int8.pt',
        Yi_status='CPU exact reconstruction from recorded post-projection source bits and actual W',
        graph_input_qinterval=[-y_bound, y_bound, 1], graph_input_signed_bits=signed_width(-y_bound, y_bound),
        binary_source_min=int(y_low_h.min()), binary_source_max=int(y_high_h.max()),
        original_physical_formats=dict(Y=24, A=16, U=48),
        legal_common_formats=dict(Y=signed_width(-y_bound, y_bound),
                                  U=signed_width(-int(y_bound*np.abs(A).sum(1).max()),int(y_bound*np.abs(A).sum(1).max()))),
        theta_source=float(params['theta_source']), theta_output=float(params['theta_output']),
        tau_range=[int(params['threshold_int64'].min()),int(params['threshold_int64'].max())],
        threshold_semantics='constant-channel bypass then gain-sign >= or <=, no intermediate requantization',
        quantization_fractional_bits=int(params['temporal_fractional_bits']),
        distinct_from_source_A='forced_code_parameters.pt:A is upstream sn1, not this sn2 consumer')
    save('compiled_summary.json', dict(kind='whole CMVM graph + exact static cost attribution, not timing/PPA',
                                      inputs=input_info, matrix=A,
                                      naive=dict(matrix_nonzeros=int(np.count_nonzero(A)), csd_digits=digit_count,
                                                 separate_constant_adds=digit_count-int(np.count_nonzero(A)),
                                                 separate_output_reduce_adds=int(np.count_nonzero(A))-T,
                                                 separate_CSD_total_adds=digit_count-T), graphs=descriptions))
    run = json.loads((CAP/'run.json').read_text())
    frame_names = [Path(f).stem for f in run['validation_files'][:args.frames]]
    observed = {label: np.zeros((len(g['nodes']), 2), np.int64) for label, g in graphs.items()}
    frames = []
    schedules = psn_schedule(A)
    saved_result = json.loads((ROOT/'bn_state/support_service_result.json').read_text())
    old_frames = {f['frame']: f for f in saved_result['frames'] if f['variant'] == 'forced_code'}
    for frame in frame_names:
        fstart = time.monotonic()
        with np.load(CAP/f'forced_code_{frame}_source.npz') as packed:
            assert tuple(packed['shape']) == (T, P, 96)
            bits = np.unpackbits(packed['gate_bits'], axis=-1, bitorder='little')
        # Preserve the source-valid-only oracle used by the old schedule; actual
        # zero Yi cancellation is recorded separately and never used as a free
        # control signal to reduce model work.
        live = bits.any(-1)
        patterns = (live.T*(1 << np.arange(T))).sum(1)
        hist = np.bincount(patterns, minlength=1024)
        metrics = dict(frame=frame, vectors=P*H, Yi_values=P*H*T, U_values=P*H*T,
                       Yi_range=[0, 0], U_range=[0, 0], real_zero_Y_values=0,
                       source_known_zero_Y_values=int((~live).sum())*H,
                       source_live_pattern_histogram=hist,
                       official_interpreter_positions=np.linspace(0,P-1,32,dtype=int).tolist(),
                       official_interpreter_U_values=0, checks={label:dict(U_mismatches=0,gate_mismatches=0,
                           official_interpreter_mismatches=0) for label in MODES})
        official_positions = set(metrics['official_interpreter_positions'])
        W_double = np.ascontiguousarray(W.T, dtype=np.float64)
        for start in range(0, P, args.chunk):
            count = min(args.chunk, P-start)
            block = bits[:, start:start+count]
            y_double = np.ascontiguousarray(block.reshape(-1,96), dtype=np.float64) @ W_double
            assert np.all(y_double == np.rint(y_double))
            y = y_double.astype(np.int64).reshape(T,count,H).reshape(T,-1)
            assert np.all(y >= -y_bound) and np.all(y <= y_bound)
            u = A @ y
            ch = np.tile(np.arange(H), count)
            expected_gate = gate(u, params, ch)
            metrics['Yi_range'][0] = min(metrics['Yi_range'][0],int(y.min()))
            metrics['Yi_range'][1] = max(metrics['Yi_range'][1],int(y.max()))
            metrics['U_range'][0] = min(metrics['U_range'][0],int(u.min()))
            metrics['U_range'][1] = max(metrics['U_range'][1],int(u.max()))
            metrics['real_zero_Y_values'] += int(np.count_nonzero(y == 0))
            local = [p-start for p in sorted(official_positions) if start <= p < start+count]
            idx = np.concatenate([np.arange(p*H,(p+1)*H) for p in local]) if local else None
            for label, graph in graphs.items():
                got = eval_graph(graph, y, observed[label])
                metrics['checks'][label]['U_mismatches'] += int(np.count_nonzero(got != u))
                metrics['checks'][label]['gate_mismatches'] += int(np.count_nonzero(gate(got,params,ch) != expected_gate))
                if idx is not None:
                    official = np.ascontiguousarray(y[:,idx].T, dtype=np.float64)
                    official = interpreters[label].predict(official,n_threads=1)
                    metrics['checks'][label]['official_interpreter_mismatches'] += int(np.count_nonzero(official.T != u[:,idx]))
            if idx is not None:
                metrics['official_interpreter_U_values'] += T*len(idx)
        metrics['baseline_MAC_vector_issues'] = int(sum(hist[p]*schedules[p]['vector_MACs']*4 for p in range(1024)))
        metrics['baseline_PSN_service_beats'] = int(sum(hist[p]*schedules[p]['beats']*4 for p in range(1024)))
        assert metrics['baseline_MAC_vector_issues']*96 == old_frames[frame]['modes']['exact_L16']['PSN_MAC_scalar_issues']
        assert metrics['baseline_PSN_service_beats'] == old_frames[frame]['modes']['exact_L16']['PSN_beats_including_output']
        metrics['same_frontend_old_exact_L16'] = {k:old_frames[frame]['modes']['exact_L16'][k] for k in
                 ('FC1_beats','warm_chain_beats','coefficient_read_words','PSN_beats_including_output',
                  'threshold_and_zero_template_read_beats')}
        metrics['graph_arithmetic_counts_not_cycles'] = {}
        for label, desc in descriptions.items():
            metrics['graph_arithmetic_counts_not_cycles'][label] = dict(
                full_graph_vector_adds=int(desc['additions']*P*4),
                known_zero_pruned_vector_adds=int(sum(hist[p]*desc['known_zero_pattern_adds'][p]*4 for p in range(1024))),
                full_scalar_bit_add_proxy=int(desc['arithmetic_output_bits']*P*H))
        metrics['elapsed_s'] = time.monotonic()-fstart
        frames.append(metrics)
        save('numeric_result.json',dict(kind='integer functional replay and graph attribution, no cycle/PPA claim',
                    inputs=input_info, frames=frames, completed_frames=len(frames), requested_frames=len(frame_names),
                    all_integer_values_checked=sum(f['U_values'] for f in frames),
                    all_gate_values_checked=sum(f['U_values'] for f in frames),
                    official_interpreter_values_per_graph=sum(f['official_interpreter_U_values'] for f in frames),
                    total_elapsed_s=time.monotonic()-started))
        print('FRAME',frame,round(metrics['elapsed_s'],2),metrics['checks'],flush=True)
        assert all(not any(v.values()) for v in metrics['checks'].values())
    save('observed_node_ranges.json',{label:values for label,values in observed.items()})
    print('FINISHED',len(frames),round(time.monotonic()-started,2),flush=True)


if __name__ == '__main__':
    main()
