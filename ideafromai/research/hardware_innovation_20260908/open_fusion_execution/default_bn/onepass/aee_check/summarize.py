from pathlib import Path
import json
import numpy as np
HERE=Path(__file__).resolve().parent
initial=json.loads((HERE/'results/run.json').read_text())
base=json.loads((HERE/'baseline_recheck/run.json').read_text())
new=json.loads((HERE/'paired_onepass/run.json').read_text())
r=dict(complete=bool(initial['complete'] and base['complete'] and new['complete']),
    evidence='Same fixed students, full diverse10 original CUDA BN and specified onepass BN, actual final 480x640 GPU flow comparison. No training or valid825.',
    changed_module=initial['changed_module'],new_X=False,training=False,full_valid825=False,
    historical_relative_AEE_gate=.005,baseline_source='baseline_recheck/run.json',candidate_source='paired_onepass/run.json',
    initial_candidate_source='results/run.json',axes={})
for axis in ['ordinary','lifting_raw']:
    b=base['axes'][axis];v=new['axes'][axis]
    rows=[]
    with np.load(b['actual_flow_capture']['path']) as z:bf=z['predictions'];bnames=z['files'].tolist()
    with np.load(v['actual_flow_capture']['path']) as z:vf=z['predictions'];vnames=z['files'].tolist()
    assert bnames==vnames==base['files']==new['files']
    oldb={x['file']:x for x in b['paired']};oldv={x['file']:x for x in v['paired']}
    old_initial={x['file']:x for x in initial['axes'][axis]['paired']}
    for i,name in enumerate(bnames):
        d=vf[i].astype(np.float64)-bf[i].astype(np.float64)
        e=np.sqrt(np.square(d).sum(axis=1))
        bits=bf[i].view(np.uint32)!=vf[i].view(np.uint32)
        rows.append(dict(file=name,baseline_AEE=oldb[name]['AEE'],onepass_AEE=oldv[name]['AEE'],
            delta_AEE=oldv[name]['AEE']-oldb[name]['AEE'],baseline_delta_old=oldb[name]['delta_AEE'],
            onepass_delta_initial=oldv[name]['AEE']-old_initial[name]['AEE'],
            flow_values=int(d.size),flow_component_bit_differences=int(np.count_nonzero(bits)),
            flow_changed_pixels=int(np.count_nonzero(np.any(d!=0,axis=1))),
            flow_max_component_abs=float(np.max(np.abs(d))),
            flow_L2_mean_all_pixels=float(e.mean()),flow_L2_max_all_pixels=float(e.max())))
    bm=float(np.mean([x['baseline_AEE'] for x in rows]));vm=float(np.mean([x['onepass_AEE'] for x in rows]))
    r['axes'][axis]=dict(baseline_frame_mean=bm,onepass_frame_mean=vm,delta_frame_mean=vm-bm,
        legacy_within_plus_0_005=vm-bm<=.005,
        baseline_reproduces_old_every_frame=all(x['baseline_delta_old']==0 for x in rows),
        onepass_reproduces_initial_every_frame=all(x['onepass_delta_initial']==0 for x in rows),
        complete_frame_GPU_vs_Engine=v['complete_frame_GPU_vs_Engine'],
        min_variance=min(x['min_variance'] for x in v['calls']),
        flow_domain='Actual 1x2x480x640 final prediction before GT valid-pixel mask; L2 fields cover all307200 pixels, distinct from AEE.',
        actual_flow_capture_baseline=b['actual_flow_capture'],actual_flow_capture_candidate=v['actual_flow_capture'],rows=rows)
r['current_admission_policy']='Former +0.005 is historical only. Active early accuracy comparison is verified same-diverse10 SDformerFlow NB0 fullres ep29 at 1.45460286107; no candidate valid825 claim.'
(HERE/'paired_summary.json').write_text(json.dumps(r,indent=2)+'\n')
for axis,v in r['axes'].items():
    print(axis,'base',v['baseline_frame_mean'],'onepass',v['onepass_frame_mean'],'delta',v['delta_frame_mean'],'historical_plus005',v['legacy_within_plus_0_005'])
    for row in v['rows']:
        print(row['file'],row['delta_AEE'],row['flow_changed_pixels'],row['flow_L2_mean_all_pixels'],row['flow_L2_max_all_pixels'])
