"""Apply the same complete official CMVM compiler to both patch students.

The two scopes are complete Aq and Aq[:, [2,3,7]] only. The latter produces ten
partial U values, NOT complete gates (except rows with no other dependency).
Existing integer_valid10 captures supply real Yi and exact full-student gates.
No prediction thresholds, masks, training parameters, or original files change.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','DA_DEFAULT_THREADS'):
    os.environ[_key]='1'
import sys
sys.dont_write_bytecode=True
from collections import Counter
import time
import numpy as np
from compile_psn import (HERE,ROOT,MODES,solve,flatten,normalized_dais,graph_order_account,
                         csd_digits,signed_width,eval_graph,save)

PATCH=ROOT/'algorithm/patch_probe/partial_completion'
PUBLIC=[2,3,7]


def main():
    started=time.monotonic()
    cases={}
    for name in ('row34','common3_diagonal_34'):
        path=PATCH/'integer_deployment'/f'{name}.npz'
        params=dict(np.load(path))
        fullA=params['temporal_int16'].astype(np.int64)
        W=params['weight_int8'].astype(np.int64).reshape(96,-1)
        assert np.array_equal(params['Y_abs_bound'],np.abs(W).sum(1))
        bound=int(params['Y_abs_bound'].max())
        assert np.all(params['positive_gain'] == 1)
        tau=params['threshold_positive'][params['full_entry']]
        fullcase=dict(parameter_file=str(path),theta_source=float(params['theta_source']),
                      theta_output=float(params['theta_output']),
                      quantization='INT8 effective W (theta and fixed BN gain folded), Aq Q14, fixed integer full predicate',
                      Y_signed_bound=bound,Y_legal_signed_bits=signed_width(-bound,bound),
                      U_signed_bound=int(params['U_abs_bound'].max()),
                      U_legal_signed_bits=signed_width(-int(params['U_abs_bound'].max()),int(params['U_abs_bound'].max())),
                      full_integer_tau_range=[int(tau.min()),int(tau.max())],scopes={})
        for scope,columns in [('full',list(range(10))),('public_prefix',PUBLIC)]:
            A=np.ascontiguousarray(fullA[:,columns])
            record=dict(source_columns=columns,output_rows=list(range(10)),matrix=A,
                        nonzero_coefficients=int(np.count_nonzero(A)),
                        CSD_digits=sum(csd_digits(v) for v in A.flat),
                        independent_CSD_total_adds=sum(csd_digits(v) for v in A.flat)-int(np.count_nonzero(np.any(A,axis=1))),
                        prefix_can_finish_output_rows=[t for t in range(10) if not np.any(np.delete(fullA[t],columns))],
                        graphs={})
            for mode,hard_dc in MODES.items():
                label=f'patch_{name}_{scope}_{mode}'
                opts=dict(method0='wmc',method1='auto',hard_dc=hard_dc,decompose_dc=-2,
                          qintervals=[(-float(bound),float(bound),1.)]*len(columns),
                          adder_size=1,carry_size=1,search_all_decompose_dc=True)
                pipeline=solve(np.ascontiguousarray(A.T,dtype=np.float32),**opts)
                assert np.array_equal(pipeline.kernel,A.T)
                pipeline.save(HERE/f'{label}_official_pipeline.json')
                graph=flatten(pipeline,A,bound,np.minimum(W,0).sum(1),np.maximum(W,0).sum(1))
                graph.update(input_time_labels=columns,official_options=opts,student=name,scope=scope)
                save(f'{label}_integer_dag.json',graph)
                interpreter=normalized_dais(graph)
                interpreter.save_binary(HERE/f'{label}.dais')
                obs=np.zeros((len(graph['nodes']),2),np.int64)
                nodeops=graph['nodes'][len(columns):]
                stats=dict(additions=len(nodeops),stage_additions=[len(s['arithmetic_nodes']) for s in graph['stage_info']],
                           max_add_depth=max(n['depth'] for n in graph['nodes']),
                           output_add_depths=[graph['nodes'][o['node']]['depth'] for o in graph['outputs']],
                           node_width_histogram=dict(Counter(n['signed_bits'] for n in nodeops)),
                           sum_arithmetic_node_bits=sum(n['signed_bits'] for n in nodeops),
                           shared_arithmetic_nodes=sum(n['distinct_consumers']>1 for n in nodeops),
                           peak_fanout_edges=max(n['fanout_edge_count'] for n in graph['nodes']),
                           official_abstract_cost=pipeline.cost,official_abstract_delay=pipeline.latency,
                           official_topological_order=graph_order_account(graph),
                           pressure_aware_order=graph_order_account(graph,pressure=True),frames=[])
                stats['P4_H8_graph_state_only']={
                    order:dict(temporary_value_bytes=stats[order]['temporary_peak_scalar_bits']*4,
                               two_operand_register_bytes=stats[order]['two_operand_register_bits']*4,
                               input_Y_bytes=len(columns)*16*4,
                               excludes='tau, source/coefficient buffers, controls, interval certificates and pipeline alignment')
                    for order in ('official_topological_order','pressure_aware_order')}
                for capture_path in sorted((PATCH/'integer_valid10').glob('capture_*.npz')):
                    sample=np.load(capture_path)
                    y=sample['Yi'].astype(np.int64).transpose(1,0,2).reshape(10,-1)
                    assert np.max(np.abs(y))<=bound
                    selected=y[columns]
                    ref=A@selected
                    got=eval_graph(graph,selected,obs)
                    official=interpreter.predict(np.ascontiguousarray(selected.T,dtype=np.float64),n_threads=1).T
                    row=dict(capture=capture_path.name,file=str(sample['file']),
                             input_vectors=selected.shape[1],output_values=int(ref.size),
                             U_mismatches=int(np.count_nonzero(got!=ref)),
                             official_DAIS_mismatches=int(np.count_nonzero(official!=ref)),
                             observed_U_range=[int(ref.min()),int(ref.max())])
                    if scope=='full':
                        channels=np.tile(np.repeat(np.arange(96),4),64)
                        predicted=got>=tau[:,channels]
                        expected=sample[f'gate_{name}_exact'].transpose(1,0,2).reshape(10,-1)
                        row['captured_gate_mismatches']=int(np.count_nonzero(predicted!=expected))
                    assert row['U_mismatches']==row['official_DAIS_mismatches']==row.get('captured_gate_mismatches',0)==0
                    stats['frames'].append(row)
                stats['observed_node_ranges']=obs
                record['graphs'][mode]=stats
                print(label,'adds',stats['additions'],'depth',stats['max_add_depth'],'PASS',flush=True)
            fullcase['scopes'][scope]=record
        cases[name]=fullcase
        save('patch_result.json',dict(kind='whole-matrix exact compiler and captured arithmetic checks; no finite-port timing/PPA',
                                     scope='four existing captures, 64 P4/frame, all H96/T10; scope is not full825 timing',
                                     cases=cases,elapsed_s=time.monotonic()-started,
                                     exclusions=['Decision-threshold generation, interval certificates and feedback are not CMVM nodes.',
                                                 'Partial prefix U cannot be used as a complete row34 gate.',
                                                 'Conv1/Conv2 fanout, data supply and actual finite-port execution not closed here.']))
    print('PATCH FINISHED',round(time.monotonic()-started,2),flush=True)


if __name__=='__main__':
    main()
