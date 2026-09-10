"""Ordinary exact input-digit certificates for the complete preview V DAG.

Trace NPZ: az_q[S,32] signed24 common-unit component integers, time[S],
threshold[10,96] int64 in K@az_q numerator units, sense[96] in {-1,0,+1}.
For sense=0 supply constant_gate[96] or [10,96]. Optional expected_gate[S,96]
is checked only as a label. K=32768*V.T comes from whole_integer_dag.json;
there is no new BN folding, division, input quantization or weight quantization.

At d known bits, r=24-d and d>=1:
  low=(az_q >> r)<<r; high=low+2**r-1.
The right shift is arithmetic, hence sign-extends the known MSB prefix.
At d=0 the entire signed24 box is used, without observing the sign.
For each row, exact extrema are low@K.T plus the positive/negative coefficient
suffix ranges. Earliest true OR false gate certificates use the original
inclusive >= / <= semantics. Full dot products are labels, never permissions.

--variable-width gives the ordinary producer an exact shared width header W
for its completed 32-component vector, and calls the same certificate function
with bits=W. Its signed-W box is public metadata, not an output oracle. Header
detection, buffering and transport are paid outside the precision-demand proxy.
This does not cancel production of the full input needed to construct W.

A node's maximum depth over its final output consumers is only a logical
input-precision-demand proxy. It is not a schedule, a width for internal online
arithmetic, a bit-operation count, or cycles. FGIE/BitSET-style ordinary early
termination baselines receive the same permission; this is not a new X.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path
import sys
import time as clock
import numpy as np

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
BITS = 24


def save(path, value):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x))
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert)+'\n')


def prefix_bounds(x, K, depth, bits):
    """Exact box extrema after ONLY the given MSB prefixes are known."""
    if depth == 0:
        low = np.full_like(x, -(1 << (bits-1)))
        suffix = (1 << bits)-1
    else:
        remaining = bits-depth
        low = (x >> remaining) << remaining
        suffix = (1 << remaining)-1
    base = low @ K.T
    negative = np.minimum(K, 0).sum(axis=1)
    positive = np.maximum(K, 0).sum(axis=1)
    return base+suffix*negative[None, :], base+suffix*positive[None, :]


def certify(x, K, tau, sense, constant, bits=BITS):
    """Return earliest prefix depths and certified gates; full U is labels only."""
    full = x @ K.T
    truth = np.where(sense[None, :] > 0, full >= tau, full <= tau)
    truth = np.where(sense[None, :] == 0, constant, truth)
    depths = np.full(truth.shape, -1, dtype=np.int16)
    gate = np.zeros(truth.shape, dtype=bool)
    is_constant = np.broadcast_to(sense[None, :] == 0, truth.shape)
    depths[is_constant] = 0
    gate[is_constant] = constant[is_constant]
    wrong_retirements = 0
    last_depth = 0
    for depth in range(bits+1):
        if np.all(depths >= 0):
            break
        lower, upper = prefix_bounds(x, K, depth, bits)
        # sense +1: true lower>=tau; false upper<tau.
        # sense -1: true upper<=tau; false lower>tau.
        positive_gate = np.where(sense[None, :] > 0, lower >= tau, upper <= tau)
        negative_gate = np.where(sense[None, :] > 0, upper < tau, lower > tau)
        assert not np.any(positive_gate & negative_gate)
        ready = (depths < 0) & (positive_gate | negative_gate)
        gate[ready] = positive_gate[ready]
        depths[ready] = depth
        wrong_retirements += int(np.count_nonzero(gate[ready] != truth[ready]))
        if depth == bits:
            assert np.array_equal(lower, full) and np.array_equal(upper, full)
        last_depth = depth
    assert np.all(depths >= 0) and np.array_equal(gate, truth)
    assert wrong_retirements == 0
    return depths.astype(np.uint8), gate, dict(
        wrong_retirements=wrong_retirements, full_dot_gate_mismatches=0,
        positive_full_gates=int(truth.sum()), label_min=int(full.min()),
        label_max=int(full.max()), deepest_prefix_executed=last_depth)


def summarize(hist, bits=BITS):
    hist = np.asarray(hist, dtype=np.int64)
    count = int(hist.sum())
    total = int(hist @ np.arange(len(hist), dtype=np.int64))
    present = np.flatnonzero(hist)
    return dict(count=count, histogram=hist, total_requested_depth=total,
                mean_required_depth=total/count if count else 0.0,
                maximum_observed_depth=int(present[-1]) if len(present) else 0,
                full_width_baseline=bits*count,
                ratio_to_full_width=total/(bits*count) if count else 0.0)


def minimum_signed_width(x):
    """Shared exact two-complement width; uses producer input, never a gate."""
    magnitude = np.where(x < 0, ~x, x).max(axis=1)
    return np.asarray([int(v).bit_length()+1 for v in magnitude], np.uint8)


def summarize_widths(hist, full_baseline):
    result = summarize(hist)
    result['fixed24_baseline'] = result['full_width_baseline']
    result['ratio_to_fixed24_baseline'] = result['ratio_to_full_width']
    result['full_width_baseline'] = int(full_baseline)
    result['ratio_to_full_width'] = (result['total_requested_depth']/full_baseline
                                     if full_baseline else 0.0)
    result['baseline_definition'] = 'sum of each vector shared W, repeated once per represented output/group/node'
    return result


def hand_check():
    # First sample gives equal-threshold true on BOTH comparison directions.
    K = np.array([[2,-1],[-1,2],[1,1],[1,0],[-1,0],[-1,0],[1,1]], np.int64)
    x = np.array([[-3,2],[3,-2],[-8,7],[7,-8],[0,0],[-1,1]], np.int64)
    tau = np.tile(np.array([-8,7,0,0,0,3,42], np.int64), (len(x),1))
    sense = np.array([1,-1,1,1,1,-1,0], np.int8)
    constant = np.full(tau.shape, -1, np.int8)
    constant[:,6] = 1
    depths, gates, checks = certify(x, K, tau, sense, constant, bits=4)
    enumerated = 0
    first_ranges = []
    for d in range(5):
        lower, upper = prefix_bounds(x, K, d, bits=4)
        for i, sample in enumerate(x):
            if d == 0:
                low, high = [-8,-8], [7,7]
            else:
                r=4-d
                low=((sample >> r) << r).tolist()
                high=[v+(1 << r)-1 for v in low]
            possible = np.array(list(product(
                range(low[0], high[0]+1), range(low[1], high[1]+1))), np.int64)
            values = possible @ K.T
            assert np.array_equal(values.min(axis=0), lower[i])
            assert np.array_equal(values.max(axis=0), upper[i])
            enumerated += len(possible)
            if i == 0:
                first_ranges.append(dict(known_bits=d, input_low=low, input_high=high,
                                         output_low=lower[i], output_high=upper[i]))
    assert (gates[0,:2] == [True,True]).all()
    assert (depths[0,:2] == [4,4]).all()
    assert depths[0,3] == 1 and not gates[0,3]  # negative input sign known
    assert depths[0,4] == 1 and gates[0,4]      # negative weight
    assert np.all(depths[:,6] == 0) and np.all(gates[:,6])
    assert np.array_equal(depths[:,[0,3]].max(axis=1),
                          np.maximum(depths[:,0], depths[:,3]))
    width_x = np.array([[-1,0],[-2,1],[-3,2],[-8,7],[7,-8],[127,-128],[128,-129]],np.int64)
    expected_width = np.array([1,2,3,4,4,8,9],np.uint8)
    assert np.array_equal(minimum_signed_width(width_x),expected_width)
    for width in np.unique(minimum_signed_width(x)):
        selected = minimum_signed_width(x) == width
        wd,wg,_ = certify(x[selected],K,tau[selected],sense,constant[selected],bits=int(width))
        assert np.array_equal(wg,gates[selected]) and np.all(wd<=width)
    return dict(complete=True, bits=4, matrix=K, inputs=x, threshold=tau[0],
                sense=sense, constant_gate=True, depths=depths, gates=gates,
                dynamic_constant_sentinel=-1, signed_width_inputs=width_x,
                signed_width_results=expected_width, width_header_function_equal=True,
                first_negative_input_prefix_ranges=first_ranges,
                exhaustive_completions_checked=enumerated,
                checks=checks,
                meaning='Small signed two-complement / mixed-weight / both equality / constant and terminal-max checks; not a network trace.')


def scan(paths, graph, chunk_size, variable_width=False):
    K = np.asarray(graph['numerator_K_output_by_input'], dtype=np.int64)
    assert K.shape == (96,32) and graph['input_signed_bits'] == BITS
    assert all(np.array_equal(np.asarray(out['coeff'],np.int64), K[h])
               for h,out in enumerate(graph['outputs']))
    lo, hi = -(1 << (BITS-1)), (1 << (BITS-1))-1
    assert max(sum(abs(int(v)) for v in row) for row in K)*max(abs(lo),abs(hi)) < 2**63
    groups = defaultdict(list)
    for node in graph['nodes']:
        mask = int(node['terminal_output_mask_hex'],16)
        groups[mask].append(node['id'])
    grouped = [(np.array(ids,np.int64), [h for h in range(96) if mask >> h & 1])
               for mask,ids in groups.items()]
    node_hist = np.zeros((len(graph['nodes']), BITS+1), np.int64)
    output_hist = np.zeros((96,BITS+1),np.int64)
    h8_hist = np.zeros((12,BITS+1),np.int64)
    time_hist = np.zeros((10,BITS+1),np.int64)
    word_hist = np.zeros(BITS+1,np.int64)
    width_hist = np.zeros(BITS+1,np.int64)
    time_width_sum = np.zeros(10,np.int64)
    trace_results = []
    sampled = 0
    all_wrong = 0
    for path in paths:
        a=np.load(path,allow_pickle=False)
        x=a['az_q']
        times=a['time']
        threshold=a['threshold']
        sense=a['sense']
        assert x.ndim==2 and x.shape[1]==32 and np.issubdtype(x.dtype,np.integer)
        assert times.shape==(len(x),) and np.issubdtype(times.dtype,np.integer)
        assert threshold.shape==(10,96) and np.issubdtype(threshold.dtype,np.integer)
        assert sense.shape==(96,) and np.isin(sense,[-1,0,1]).all()
        assert len(x)>0 and np.all((x>=lo)&(x<=hi)) and np.all((times>=0)&(times<10))
        constant=a['constant_gate'] if 'constant_gate' in a else None
        if np.any(sense==0):
            assert constant is not None and constant.shape in ((96,),(10,96))
            assert np.isin(constant[...,sense==0],[0,1]).all()
        expected=a['expected_gate'] if 'expected_gate' in a else None
        if expected is not None:
            assert expected.shape==(len(x),96) and np.isin(expected,[0,1]).all()
        file_hist=np.zeros(BITS+1,np.int64)
        file_h8=np.zeros(BITS+1,np.int64)
        file_word=np.zeros(BITS+1,np.int64)
        wrong=0
        expected_errors=0
        full_positive=0
        labels=[2**63-1,-2**63]
        file_width_sum=0
        for first in range(0,len(x),chunk_size):
            end=min(first+chunk_size,len(x))
            xb=x[first:end].astype(np.int64)
            tb=times[first:end].astype(np.int64)
            taub=threshold[tb].astype(np.int64)
            constants=np.zeros(taub.shape,bool)
            if constant is not None:
                constants=np.broadcast_to(constant,taub.shape) if constant.ndim==1 else constant[tb]
            widths=minimum_signed_width(xb) if variable_width else np.full(len(xb),BITS,np.uint8)
            width_hist+=np.bincount(widths,minlength=BITS+1)
            file_width_sum+=int(widths.sum())
            depths=np.empty(taub.shape,np.uint8)
            gates=np.empty(taub.shape,bool)
            for width in np.unique(widths):
                selected=widths==width
                wd,wg,checks=certify(xb[selected],K,taub[selected],sense,constants[selected],bits=int(width))
                depths[selected],gates[selected]=wd,wg
                wrong+=checks['wrong_retirements']
                full_positive+=checks['positive_full_gates']
                labels=[min(labels[0],checks['label_min']),max(labels[1],checks['label_max'])]
            if expected is not None:
                expected_errors+=int(np.count_nonzero(gates != expected[first:end]))
            hist=np.bincount(depths.ravel(),minlength=BITS+1)
            file_hist+=hist
            h8=depths.reshape(-1,12,8).max(axis=2)
            word=depths.max(axis=1)
            file_h8+=np.bincount(h8.ravel(),minlength=BITS+1)
            file_word+=np.bincount(word,minlength=BITS+1)
            for h in range(96):
                output_hist[h]+=np.bincount(depths[:,h],minlength=BITS+1)
            for group in range(12):
                h8_hist[group]+=np.bincount(h8[:,group],minlength=BITS+1)
            for t in np.unique(tb):
                time_hist[t]+=np.bincount(depths[tb==t].ravel(),minlength=BITS+1)
                time_width_sum[t]+=int(widths[tb==t].sum())
            for ids,terminals in grouped:
                demand=depths[:,terminals].max(axis=1) if terminals else np.zeros(len(depths),np.uint8)
                node_hist[ids]+=np.bincount(demand,minlength=BITS+1)[None,:]
        assert wrong==0 and expected_errors==0
        sampled+=len(x)
        all_wrong+=wrong
        word_hist+=file_word
        trace_results.append(dict(trace=str(path),samples=len(x),positive_full_gates=full_positive,
                                  full_dot_range=labels,wrong_retirements=wrong,
                                  optional_expected_gate_present=expected is not None,
                                  optional_expected_gate_mismatches=expected_errors,
                                  sum_vector_widths=file_width_sum,
                                  single_output=summarize_widths(file_hist,96*file_width_sum),
                                  H8_joint=summarize_widths(file_h8,12*file_width_sum),
                                  all96_joint=summarize_widths(file_word,file_width_sum)))
        print('PASS',Path(path).name,'samples',len(x),'mean output/H8/all96',
              *(round(summarize(h)['mean_required_depth'],4) for h in [file_hist,file_h8,file_word]),
              flush=True)
    arithmetic=np.array([n['kind']!='input' for n in graph['nodes']])
    width_sum=int(width_hist@np.arange(BITS+1,dtype=np.int64))
    per_node=[]
    for node,hist in zip(graph['nodes'],node_hist):
        per_node.append(dict(id=node['id'],kind=node['kind'],
                             terminal_output_mask_hex=node['terminal_output_mask_hex'],
                             terminal_output_count=node['terminal_output_count'],
                             **summarize_widths(hist,width_sum)))
    return dict(
        complete=True, scope='ordinary signed input-prefix interval certificate, not a new execution mechanism',
        graph=str(HERE/'whole_integer_dag.json'),samples=sampled,output_gates=sampled*96,
        precision_mode='exact_per_vector_width_header' if variable_width else 'fixed24',
        input_signed_bits=BITS,integer_function='K=32768*V.T; compare K@az_q directly to trace threshold',
        thresholds_scope='already-compiled supplied threshold/sense; no BN, theta, gain, rounding or output-scale change',
        depth0='constant gates and certificates from the signed-W box announced by the exact header; sign is unknown' if variable_width else 'constant gates and certificates from the full signed24 legal domain; sign is unknown',
        comparison='sense+1: lower>=tau or upper<tau; sense-1: upper<=tau or lower>tau; inclusive equal gate retained',
        wrong_retirements=all_wrong,full_dot_gate_mismatches=0,
        single_output=summarize_widths(output_hist.sum(axis=0),96*width_sum),
        H8_joint=summarize_widths(h8_hist.sum(axis=0),12*width_sum),all96_joint=summarize_widths(word_hist,width_sum),
        per_output=[dict(h=h,**summarize_widths(hist,width_sum)) for h,hist in enumerate(output_hist)],
        per_H8=[dict(group=g,**summarize_widths(hist,width_sum)) for g,hist in enumerate(h8_hist)],
        per_time=[dict(t=t,**summarize_widths(hist,96*int(time_width_sum[t]))) for t,hist in enumerate(time_hist)],
        DAG_arithmetic_node_demand=summarize_widths(node_hist[arithmetic].sum(axis=0),int(arithmetic.sum())*width_sum),
        DAG_input_node_demand=summarize_widths(node_hist[~arithmetic].sum(axis=0),int((~arithmetic).sum())*width_sum),
        per_node=per_node,traces=trace_results,
        baseline='sum of exact vector widths W per output or graph node; ordinary header compression and all domain certificates receive the same permission' if variable_width else '24 input-prefix bits per output or graph node; constants/domain certificates are also available to the ordinary baseline',
        width_header=dict(enabled=variable_width,width_histogram=width_hist,sum_vector_widths=width_sum,
                          mean_vector_width=width_sum/sampled,raw_fixed24_input_payload_bits=32*BITS*sampled,
                          exact_width_input_payload_bits=32*width_sum,
                          fixed_length_header_bits_per_vector=5 if variable_width else 0,
                          total_header_bits=5*sampled if variable_width else 0,
                          header_rule='W=1+bit_length(max_j(x_j if x_j>=0 else ~x_j)); no rounding/clamping',
                          domain_scope='signed-W box only; no output labels or identity of a maximum-width component used',
                          unpaid_cost='32 component leading-sign/zero detectors and maximum reduction, width header transport, waiting/buffering until full vector width known, packing and finite ports; not included in precision ratios',
                          bit_payload_scope='exact source encoding only; do not add its bits to the separate DAG precision proxy or call it a cycle reduction'),
        not_measured=['internal carry/online arithmetic precision or valid digit schedule',
                      'interval bound hardware cost and input-digit delivery',
                      'terminal-mask counters, dependency updates, queues and bank/port service',
                      'actual native FP32 equivalence, network AEE, energy or cycles'],
        demand_interpretation='max certified output prefix depth over each node terminal mask. This is a logical input precision request proxy ONLY: it does not prove that an internal node can run for that many bits or stop then.',
        strong_control='Ordinary FGIE/BitSET-style input-digit early stopping is given this same permission.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--trace',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,default=HERE/'certificate_scan.json')
    parser.add_argument('--chunk',type=int,default=1024)
    parser.add_argument('--variable-width',action='store_true')
    parser.add_argument('--fixed-reference',type=Path)
    args=parser.parse_args()
    start=clock.monotonic()
    check=hand_check()
    save(HERE/'hand_check.json',check)
    print('hand-check PASS',flush=True)
    if args.trace:
        graph=json.loads((HERE/'whole_integer_dag.json').read_text())
        result=scan(args.trace,graph,args.chunk,variable_width=args.variable_width)
        if args.fixed_reference:
            fixed=json.loads(args.fixed_reference.read_text())
            assert fixed['samples']==result['samples']
            assert sorted(t['trace'] for t in fixed['traces'])==sorted(str(p) for p in args.trace)
            result['fixed24_reference_file']=str(args.fixed_reference)
            result['fixed24_comparison']={key:dict(
                fixed24_mean=fixed[key]['mean_required_depth'],
                header_mean=result[key]['mean_required_depth'],
                fixed24_demand=fixed[key]['total_requested_depth'],
                header_demand=result[key]['total_requested_depth'],
                actual_sum_W_baseline=result[key]['full_width_baseline'],
                fixed24_demand_div_sum_W=fixed[key]['total_requested_depth']/result[key]['full_width_baseline'],
                header_demand_div_sum_W=result[key]['ratio_to_full_width'])
                for key in ('single_output','H8_joint','all96_joint','DAG_arithmetic_node_demand')}
        result['CPU_wall_seconds']=clock.monotonic()-start
        result['wall_time_scope']='host implementation elapsed time, not accelerator performance'
        save(args.output,result)
        print('SAVED',args.output,flush=True)


if __name__=='__main__':
    main()
