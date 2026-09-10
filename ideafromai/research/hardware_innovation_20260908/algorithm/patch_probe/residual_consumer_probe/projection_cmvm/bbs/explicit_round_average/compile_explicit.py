"""Explicit ordinary round-average decomposition; no new quantization or RTL.

Six C16 sums are produced once, then twelve independently compiled H8 graphs
consume [X96, S6]. Official da4ml supplies its complete two-stage solver.
Every node is also expressed over the original independent X96 domain, so
the correlation S_g=sum16 X is retained when deriving exact integer widths.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','DA_DEFAULT_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from collections import Counter
from importlib.metadata import version
import time
import json
import numpy as np

HERE=Path(__file__).resolve().parent
BBS=HERE.parent
PROJECTION=BBS.parent
sys.path.insert(0,str(PROJECTION))
from compile_projection import (solve,flatten,normalized_interpreter,eval_graph,
    signed_width,interval,save,XLO,XHI,OLD)
from audit_physical_gap import order_for,account


def extend(x):
    return np.concatenate([x,x.reshape(-1,6,16).sum(axis=2)],axis=1)


def correlate_graph(graph, binding, output_h, expected):
    """Keep official 102-D coefficients and add exact original 96-D forms."""
    groups=[0]*len(graph['nodes'])
    for out,h in zip(graph['outputs'],output_h):
        out['local_output']=out['t']
        out['h']=int(h)
        original=np.asarray(out['coeff'],np.int64)@binding
        assert np.array_equal(original,expected[out['t']])
        out['original_coeff']=original.tolist()
        lo,hi=interval(original)
        out.update(static_min=lo,static_max=hi,signed_bits=signed_width(lo,hi))
        groups[out['node']]|=1<<(int(h)//8)
    for node in reversed(graph['nodes'][graph['n_input']:]):
        for parent in (node['lhs'],node['rhs']):groups[parent]|=groups[node['id']]
    for node in graph['nodes']:
        original=np.asarray(node['coeff'],np.int64)@binding
        if node['kind']!='input':
            left=np.asarray(graph['nodes'][node['lhs']]['coeff'],np.int64)<<node['lhs_shift']
            right=np.asarray(graph['nodes'][node['rhs']]['coeff'],np.int64)<<node['rhs_shift']
            assert np.array_equal(left-right if node['subtract'] else left+right,node['coeff'])
        lo,hi=interval(original)
        node.update(original_coeff=original.tolist(),static_min=lo,static_max=hi,
            signed_bits=signed_width(lo,hi),
            downstream_H8_groups=[i for i in range(12) if groups[node['id']]>>i&1])
        for key in ('binary_source_global_min','binary_source_global_max','binary_source_max_lane_bits'):
            node.pop(key,None)
    assert [n['signed_bits'] for n in graph['nodes'][:102]]==[12]*96+[16]*6
    graph.update(input_labels=[f'X{c}' for c in range(96)]+[f'S{g}' for g in range(6)],
        input_domains=[[XLO,XHI]]*96+[[16*XLO,16*XHI]]*6,
        input_binding_to_original_X96=binding,
        output_h=output_h,
        instance='one (p,t), correlated X96/S6, 8 output channels',
        strict_domain='independent original X96 in [-2048,2047]; S_g=sum(X_16g:16g+16)',
        output_matches_RA_Wq_for_every_legal_X96=True)


def checks(graph,interpreter,W,actual,corners):
    def integer_eval(x):
        observed=np.column_stack([np.full(len(graph['nodes']),np.iinfo(np.int64).max,np.int64),
            np.full(len(graph['nodes']),np.iinfo(np.int64).min,np.int64)])
        got=eval_graph(graph,extend(x).T,observed)
        reference=W@x.T
        mismatch=int(np.count_nonzero(got!=reference))
        assert mismatch==0
        return mismatch
    actual_errors=integer_eval(actual)
    corner_errors=integer_eval(corners)
    pick=np.unique(np.linspace(0,len(actual)-1,min(64,len(actual)),dtype=int))
    chosen=np.concatenate([actual[pick],corners])
    got=interpreter.predict(np.ascontiguousarray(extend(chosen),dtype=np.float64),n_threads=1)
    expected=chosen@W.T
    official_errors=int(np.count_nonzero(got!=expected))
    assert official_errors==0
    return dict(actual_vectors=len(actual),actual_output_values=len(actual)*len(W),
        actual_integer_DAG_mismatches=actual_errors,corner_vectors=len(corners),
        corner_mismatches=corner_errors,official_actual_vectors=len(pick),
        official_corner_vectors=len(corners),official_DAIS_mismatches=official_errors,
        symbolic_full_original_X96_equal=True)


def comparison(result):
    records=[c['ordinary_DFS'] for c in result['cases'].values()]
    sumgen=result['sum_generation']
    explicit=dict(add_sub_nodes=sumgen['additions']+sum(c['add_sub_nodes'] for c in result['cases'].values()),
        generation_additions=sumgen['additions'],
        local_graph_add_sub_nodes=sum(c['add_sub_nodes'] for c in result['cases'].values()),
        input_reads=sumgen['input_reads']+sum(r['input_reads'] for r in records),
        X_reads=sumgen['input_reads']+sum(r['X_reads'] for r in records),
        group_sum_reads=sum(r['group_sum_reads'] for r in records),
        group_sum_writes=6,
        temporary_reads=sum(r['temporary_reads'] for r in records),
        temporary_writes=sum(r['temporary_writes'] for r in records),
        peak_RF_values=max(r['peak_RF_values'] for r in records),
        peak_RF_bits=max(r['peak_RF_bits'] for r in records),
        peak_live_plus_result_bits=max(r['peak_live_plus_result_bits'] for r in records),
        resident_input_bits=96*12,resident_group_sum_bits=6*16,
        fixed_25bit_scratch_upper_bits=max(r['peak_RF_values'] for r in records)*25,
        actual_output_values=sum(c['checks']['actual_output_values'] for c in result['cases'].values()),
        actual_integer_DAG_mismatches=sum(c['checks']['actual_integer_DAG_mismatches'] for c in result['cases'].values()),
        official_DAIS_mismatches=sum(c['checks']['official_DAIS_mismatches'] for c in result['cases'].values()))
    explicit['all_logical_reads_writes']=sum(explicit[k] for k in ('input_reads','temporary_reads','temporary_writes','group_sum_writes'))
    direct=json.loads((BBS/'round_average/result.json').read_text())
    comparisons=dict(explicit_S6_serial_H8=explicit)
    for mode in ('whole','serial_H8'):
        a=direct['ordinary_DFS_comparison'][mode]
        if mode=='whole':
            nodes=direct['cases']['whole']['statistics']['add_sub_nodes']
        else:
            nodes=sum(direct['cases'][f'h8_{i:02d}']['statistics']['add_sub_nodes'] for i in range(12))
        comparisons['direct_RA_'+mode]=dict(add_sub_nodes=nodes,
            input_reads=a['input_reads'],temporary_reads=a['temporary_reads'],temporary_writes=a['temporary_writes'],
            all_logical_reads_writes=a['input_reads']+a['temporary_reads']+a['temporary_writes'],
            peak_RF_values=a['peak_RF_values'],peak_RF_bits=a['peak_RF_bits'],
            peak_live_plus_result_bits=a['peak_live_plus_result_bits'],
            resident_input_bits=1152,resident_group_sum_bits=0,
            fixed_25bit_scratch_upper_bits=a['peak_RF_values']*25)
    return comparisons


def input_kind_reads(graph,order):
    """Same two-register replacement as account, for X versus S read attribution."""
    uses=Counter(i for ident in order for i in (graph['nodes'][ident]['lhs'],graph['nodes'][ident]['rhs']))
    cache=[]
    x_reads=s_reads=0
    def add(i):
        if i in cache:cache.remove(i)
        elif len(cache)==2:
            dead=[v for v in cache if v>=102 and uses[v]==0]
            cache.remove(dead[0] if dead else cache[0])
        cache.append(i)
    for ident in order:
        n=graph['nodes'][ident]
        operands=(n['lhs'],n['rhs'])
        for src in dict.fromkeys(operands):
            if src not in cache:
                x_reads+=src<96
                s_reads+=96<=src<102
            add(src)
        for src in operands:uses[src]-=1
        if uses[ident]:add(ident)
    return dict(X_reads=int(x_reads),group_sum_reads=int(s_reads))


def main():
    started=time.monotonic()
    z=np.load(BBS/'round_average.npz')
    W=z['Wq'].astype(np.int64)
    actual=z['calibration_Xq'].astype(np.int64)
    pruned=z['pruned_rows'].astype(int)
    sensitive=z['sensitive_rows'].astype(int)
    constants=z['group_constant'].reshape(len(pruned),6).astype(np.int64)
    low=z['group_low_bit_width'].reshape(len(pruned),6).astype(np.int64)
    D=np.zeros((96,6),np.int64)
    D[pruned]=constants
    core=W-np.repeat(D,16,axis=1)
    assert np.all(core[pruned].reshape(len(pruned),6,16)%(1<<low[:,:,None])==0)
    assert not np.any(D[sensitive]) and np.array_equal(core[sensitive],W[sensitive])
    binding=np.concatenate([np.eye(96,dtype=np.int64),np.repeat(np.eye(6,dtype=np.int64),16,axis=1)])
    M=np.concatenate([core,D],axis=1)
    assert np.array_equal(M@binding,W)
    sums=actual.reshape(-1,6,16).sum(axis=2)
    assert np.array_equal(actual@core.T+sums@D.T,actual@W.T)
    np.savez(HERE/'decomposition.npz',Wq=W,core=core,D=D,
        input_binding_to_original_X96=binding,pruned_rows=pruned,
        sensitive_rows=sensitive,group_low_bit_width=low,
        output_scale=float(z['input_scale'])*z['row_scale'],output_bias=z['original_bias'])
    corners=np.concatenate([np.zeros((1,96),np.int64),np.full((1,96),XLO,np.int64),
        np.full((1,96),XHI,np.int64),np.where(W>=0,XHI,XLO),np.where(W>=0,XLO,XHI)])
    options=dict(method0='wmc',method1='auto',hard_dc=-1,decompose_dc=-2,
        qintervals=[(float(XLO),float(XHI),1.0)]*96+[(float(16*XLO),float(16*XHI),1.0)]*6,
        adder_size=1,carry_size=1,search_all_decompose_dc=True)
    result=dict(complete=False,kind='ordinary explicit round-average S6 plus full official local CMVM',
        parameters=str(BBS/'round_average.npz'),official_source=str(OLD/'da4ml_official'),
        official_version=version('da4ml'),compiler_options=options,
        same_restored_weight_matrix=True,protected32_rows_D_zero=True,
        exact_original_X96_binding=True,core_range=[int(core.min()),int(core.max())],
        D_range=[int(D.min()),int(D.max())],local_matrix_shape=[8,102],
        input_domains=dict(X96=[XLO,XHI],S6=[16*XLO,16*XHI]),
        sum_generation=dict(groups=6,group_size=16,additions=90,input_reads=96,
            input_read_bits=96*12,persistent_sum_writes=6,persistent_sum_write_bits=6*16,
            persistent_sum_bits=6*16,work_accumulator_bits=16,
            folded_execution='one group sequentially; first X seeds, 15 adds, write S; retain S until its last H8 consumer',
            generation_not_free=True),
        numeric_scope='same RA quantized integer matrix, original independent X12; not original W8/FP equivalence',
        schedule_scope='twelve H8 graphs sequential, same output DFS and two bypass registers; six S reads/writes included; no finite-port cycles/PPA',
        exclusions=['input capture/quantization','source DMA','uop/address/shift control memory','RF latency/port contention',
            'output scale/bias and sink service','unrolled routing/retiming','ASIC PPA'],cases={})
    save(HERE/'result.json',result)
    for g in range(12):
        before=time.monotonic()
        rows=np.arange(g*8,g*8+8)
        A=np.ascontiguousarray(M[rows])
        print('COMPILING',g,A.shape,flush=True)
        pipe=solve(np.ascontiguousarray(A.T,dtype=np.float32),**options)
        assert np.array_equal(pipe.kernel,A.T)
        pipe.save(HERE/f'h8_{g:02d}_official_pipeline.json')
        graph=flatten(pipe,A,32768,np.array([-32768]),np.array([32752]))
        correlate_graph(graph,binding,rows,W[rows])
        interpreter=normalized_interpreter(graph)
        interpreter.save_binary(HERE/f'h8_{g:02d}.dais')
        widths=[n['signed_bits'] for n in graph['nodes']]
        masks=[sum(1<<h for h in n['downstream_H8_groups']) for n in graph['nodes']]
        order=order_for(graph,widths,'output_group_first')
        traffic=account(graph,widths,masks,order)
        traffic.update(input_kind_reads(graph,order))
        assert traffic['input_reads']==traffic['X_reads']+traffic['group_sum_reads']
        numerical=checks(graph,interpreter,W[rows],actual,corners)
        save(HERE/f'h8_{g:02d}_integer_dag.json',graph)
        result['cases'][f'h8_{g:02d}']=dict(add_sub_nodes=len(graph['nodes'])-102,
            max_local_add_depth=max(n['depth'] for n in graph['nodes']),
            mathematical_width_histogram=dict(Counter(widths[102:])),
            maximum_node_bits=max(widths),ordinary_DFS=traffic,checks=numerical,
            graph_file=f'h8_{g:02d}_integer_dag.json',wall_seconds=time.monotonic()-before)
        save(HERE/'result.json',result)
        print('PASS',g,'adds',len(graph['nodes'])-102,'RFwords',traffic['peak_RF_values'],flush=True)
    result.update(complete=True,comparison=comparison(result),wall_seconds=time.monotonic()-started)
    save(HERE/'result.json',result)
    print('FINISHED',result['wall_seconds'],flush=True)


if __name__=='__main__':main()
