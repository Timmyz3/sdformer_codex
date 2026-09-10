"""Compile restored BBS/control matrices with the same complete da4ml baseline.

All three use official matrix decomposition and cross-output signed-shift CSE,
one wmc/auto setting, unrestricted delay, whole96 and twelve fixed H8 graphs.
Also grant each graph the ordinary output-dependency DFS, two bypass registers
and lazy RF writeback already supplied to the original W8 baseline.

This compiles the complete restored integer matrix. It is NOT the additional
strong BitVert control that explicitly forms six shared sum(X_C16) values and
separately compiles low-bit correction plus group constants. The NPZ contains
that exact decomposition information for a later equally applied comparison.
Neither this graph compiler nor its liveness counts are a finite-port schedule,
bit-serial BitVert simulation, ASIC PPA, or a new hardware mechanism.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','DA_DEFAULT_THREADS'):
    os.environ[_key]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from importlib.metadata import version
import json
import time
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from compile_projection import (solve, flatten, fix_domain, normalized_interpreter,
    graph_statistics, check_vectors, comparison, csd_digits, save, XLO, XHI, OLD)
from audit_physical_gap import order_for, account


def dfs_account(graph):
    widths=[n['signed_bits'] for n in graph['nodes']]
    group_masks=[sum(1<<g for g in n['downstream_H8_groups']) for n in graph['nodes']]
    order=order_for(graph,widths,'output_group_first')
    result=account(graph,widths,group_masks,order)
    created={ident:i for i,ident in enumerate(order)}
    last={ident:i for i,ident in enumerate(order)}
    for i,ident in enumerate(order):
        for parent in (graph['nodes'][ident]['lhs'],graph['nodes'][ident]['rhs']):
            if parent in last: last[parent]=i
    spans=[(last[n]-created[n],widths[n]) for n in order if group_masks[n].bit_count()>1]
    result.update(order='ordinary output dependency DFS; same rule as original W8 audit',
        sum_cross_H8_node_spans=sum(s for s,w in spans),
        sum_cross_H8_width_times_issued_node_span=sum(s*w for s,w in spans),
        span_scope='distance in this legal scalar issue order, not cycles or wire delay',
        source_cache_bits=96*12)
    return result


def compile_variant(name):
    started=time.monotonic()
    path=HERE/f'{name}.npz'
    z=np.load(path)
    W=z['Wq'].astype(np.int64)
    actual=z['calibration_Xq'].astype(np.int64)
    assert W.shape==(96,96) and np.all((actual>=XLO)&(actual<=XHI))
    corners=np.concatenate([np.zeros((1,96),np.int64),np.full((1,96),XLO,np.int64),
        np.full((1,96),XHI,np.int64),np.where(W>=0,XHI,XLO),np.where(W>=0,XLO,XHI)],axis=0)
    options=dict(method0='wmc',method1='auto',hard_dc=-1,decompose_dc=-2,
        qintervals=[(float(XLO),float(XHI),1.0)]*96,
        adder_size=1,carry_size=1,search_all_decompose_dc=True)
    out=HERE/name
    out.mkdir(exist_ok=True)
    result=dict(complete=False,variant=name,parameters=str(path),
        kind='complete official da4ml restored-weight CMVM; no timing/PPA',
        official_version=version('da4ml'),official_source=str(OLD/'da4ml_official'),
        compiler_options=options,input_integer_domain=[XLO,XHI],input_signed_bits=12,
        weight_integer_range=[int(W.min()),int(W.max())],
        output_scale=float(z['input_scale'])*z['row_scale'],output_bias=z['original_bias'],
        independent_CSD_digits=sum(csd_digits(v) for v in W.flat),
        independent_CSD_add_subs=sum(csd_digits(v) for v in W.flat)-int(np.any(W,axis=1).sum()),
        actual_vectors=len(actual),actual_input_values=int(actual.size),cases={},
        strong_control_gap='explicit six C16 sum(X) plus BBS constant/correction two-stage compile not implemented here; restored-matrix CMVM alone is not the complete BitVert execution baseline')
    save(out/'result.json',result)
    graphs={}
    for label,rows in [('whole',np.arange(96))]+[(f'h8_{g:02d}',np.arange(g*8,g*8+8)) for g in range(12)]:
        before=time.monotonic()
        A=np.ascontiguousarray(W[rows])
        print('COMPILING',name,label,flush=True)
        pipeline=solve(np.ascontiguousarray(A.T,dtype=np.float32),**options)
        assert np.array_equal(pipeline.kernel,A.T)
        pipeline.save(out/f'{label}_official_pipeline.json')
        graph=flatten(pipeline,A,2048,np.array([XLO]),np.array([XHI]))
        fix_domain(graph,rows)
        graph.update(official_options=options,official_version=version('da4ml'),
            method='official matrix decomposition + weighted cross-output signed-shift CSE',variant=name)
        interpreter=normalized_interpreter(graph)
        interpreter.save_binary(out/f'{label}.dais')
        stats=graph_statistics(graph,pipeline)
        stats['ordinary_DFS']=dfs_account(graph)
        checks,observed=check_vectors(graph,interpreter,A,actual,corners)
        save(out/f'{label}_integer_dag.json',graph)
        save(out/f'{label}_observed_node_ranges.json',dict(source='real calibration Xq only',ranges=observed))
        result['cases'][label]=dict(output_h=rows,statistics=stats,checks=checks,
            wall_seconds=time.monotonic()-before,graph_file=f'{label}_integer_dag.json',DAIS_file=f'{label}.dais')
        graphs[label]=graph
        save(out/'result.json',result)
        print('PASS',name,label,'nodes',stats['add_sub_nodes'],'depth',stats['maximum_add_depth'],
            'DFS_RF_bits',stats['ordinary_DFS']['peak_RF_bits'],'seconds',round(time.monotonic()-before,2),flush=True)
    result['comparison']=comparison(result['cases'],graphs)
    ds=[result['cases'][f'h8_{g:02d}']['statistics']['ordinary_DFS'] for g in range(12)]
    result['ordinary_DFS_comparison']=dict(whole=result['cases']['whole']['statistics']['ordinary_DFS'],
        serial_H8=dict(input_reads=sum(d['input_reads'] for d in ds),
            temporary_reads=sum(d['temporary_reads'] for d in ds),temporary_writes=sum(d['temporary_writes'] for d in ds),
            peak_RF_bits=max(d['peak_RF_bits'] for d in ds),peak_live_plus_result_bits=max(d['peak_live_plus_result_bits'] for d in ds),
            peak_RF_values=max(d['peak_RF_values'] for d in ds),source_cache_bits=96*12))
    result.update(complete=True,wall_seconds=time.monotonic()-started)
    save(out/'result.json',result)
    return result


def compact(result):
    w=result['cases']['whole']['statistics']
    return dict(whole_add_subs=w['add_sub_nodes'],sum_H8_add_subs=result['comparison']['sum_twelve_H8_nodes'],
        whole_depth=w['maximum_add_depth'],max_H8_depth=result['comparison']['largest_H8_depth'],
        whole_node_bits=w['arithmetic_node_bits'],cross_H8_nodes=w['arithmetic_nodes_with_multiple_final_H8_consumers'],
        cross_H8_node_bits=w['multiple_H8_node_bits'],maximum_fanout=w['maximum_fanout_edges'],
        independent_CSD_add_subs=result['independent_CSD_add_subs'],
        ordinary_DFS=result['ordinary_DFS_comparison'],
        verification=dict(actual_output_values_per_organization=2560*96,
            actual_DAG_mismatches=sum(c['checks']['actual_integer_DAG_mismatches'] for c in result['cases'].values()),
            DAIS_mismatches=sum(c['checks']['official_DAIS_mismatches'] for c in result['cases'].values())))


def main():
    # The original W8 is already completely compiled. Give it precisely the
    # same ordinary DFS accounting without rerunning or editing its compiler.
    original=json.loads((HERE.parent/'result.json').read_text())
    audit=json.loads((HERE.parent/'audit_physical_gap.json').read_text())
    original['ordinary_DFS_comparison']=dict(whole=audit['full']['orders']['output_group_first'],
        serial_H8=audit['independent_H8_folded_comparison']['output_group_first'])
    # Span totals were not part of the old audit; compute on its fixed graph.
    original['ordinary_DFS_comparison']['whole']=dfs_account(json.loads((HERE.parent/'whole_integer_dag.json').read_text()))
    report=dict(complete=False,configuration='fixed group16, three pruned columns,32 protected rows, ZPS6bit constant',
        source_metadata='parameters_result.json',original_W8=compact(original),variants={},
        not_claimed=['new mechanism','cycles','ASIC PPA','complete BitVert execution',
            'all calibrated PTQ alternatives','whole-network AEE from local arithmetic'],
        interpretation='Cross-H8 spans/peaks are logical state burdens for the same ordinary order. They do not establish that the variant saves finite-port service; total source and temporary accesses are reported alongside.')
    save(HERE/'compilation_result.json',report)
    for name in ('round_average','zero_point_shift','uniform_group5'):
        result=compile_variant(name)
        report['variants'][name]=compact(result)
        save(HERE/'compilation_result.json',report)
    report['complete']=True
    save(HERE/'compilation_result.json',report)
    print('ALL_COMPLETE',flush=True)


if __name__=='__main__': main()
