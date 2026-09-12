from pathlib import Path
import json
import numpy as np
HERE=Path(__file__).resolve().parent
reports={name:json.loads((HERE/path/'run.json').read_text()) for name,path in
    [('cuda','baseline_recheck'),('centered','centered_results'),('onepass','paired_onepass')]}
keys=dict(cuda='original_cuda_bn',centered='centered_engine',onepass='onepass')
r=dict(complete=all(x['complete'] for x in reports.values()),scope='Three matched numerical functions on two unchanged students and diverse10; same original metric and final GPU flow.',
    changed_module=reports['onepass']['changed_module'],training=False,new_X=False,valid825=False,
    interpretation='Both Engine functions change original CUDA BN arithmetic. Onepass minus centered isolates the complete onepass-statistics choice with the shared seed/Newton/affine boundary fixed, not individual stripe or variance-formula effects.',
    historical_relative_AEE_gate=.005,axes={})
for axis in ['ordinary','lifting_raw']:
    stats={};flows={};names=None
    for method,report in reports.items():
        v=report['axes'][axis]
        with np.load(v['actual_flow_capture']['path']) as z:
            flows[method]=z['predictions'];nm=z['files'].tolist()
        if names is None:names=nm
        assert names==nm
        stats[method]=dict(frame_mean=v[keys[method]]['AEE_frame_mean'],frames={x['file']:x['AEE'] for x in v['paired']},
            complete_frame_GPU_vs_Engine=v['complete_frame_GPU_vs_Engine'],
            first_runtime_CPU_GPU_exact=(v['calls'][0].get('runtime_GPU_vs_CPU_exact') if v['calls'] else None))
    out=dict(methods=stats,comparisons={})
    for candidate,baseline in [('centered','cuda'),('onepass','cuda'),('onepass','centered')]:
        rows=[]
        for i,name in enumerate(names):
            b=flows[baseline][i];x=flows[candidate][i];d=x.astype(np.float64)-b.astype(np.float64)
            e=np.sqrt(np.square(d).sum(axis=1))
            rows.append(dict(file=name,baseline_AEE=stats[baseline]['frames'][name],candidate_AEE=stats[candidate]['frames'][name],
                delta_AEE=stats[candidate]['frames'][name]-stats[baseline]['frames'][name],flow_values=int(d.size),
                flow_component_bit_differences=int(np.count_nonzero(x.view(np.uint32)!=b.view(np.uint32))),
                flow_changed_pixels=int(np.count_nonzero(np.any(d!=0,axis=1))),
                flow_max_component_abs=float(np.max(np.abs(d))),flow_L2_mean_all_pixels=float(e.mean()),flow_L2_max_all_pixels=float(e.max())))
        delta=stats[candidate]['frame_mean']-stats[baseline]['frame_mean']
        out['comparisons'][candidate+'_vs_'+baseline]=dict(delta_frame_mean=delta,legacy_within_plus_0_005=delta<=.005,
            changed_flow_frames=sum(x['flow_component_bit_differences']>0 for x in rows),
            flow_domain='Actual GPU1x2x480x640 prediction before valid-GT mask; flow distances cover all307200 pixels, not AEE denominator.',rows=rows)
    r['axes'][axis]=out
r['current_admission_policy']='Former +0.005 is historical only. Active early accuracy comparison is verified same-diverse10 SDformerFlow NB0 fullres ep29 at 1.45460286107; no candidate valid825 claim.'
(HERE/'three_arms_summary.json').write_text(json.dumps(r,indent=2)+'\n')
for axis,v in r['axes'].items():
    print(axis,{k:x['frame_mean'] for k,x in v['methods'].items()})
    print({k:(x['delta_frame_mean'],x['changed_flow_frames']) for k,x in v['comparisons'].items()})
