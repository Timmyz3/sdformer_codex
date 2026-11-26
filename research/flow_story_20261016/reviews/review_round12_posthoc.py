#!/usr/bin/env python3
"""Post-result independent check of descriptive input/direction statistics only."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent/'ordinary_recovery_round12'
OUT=Path(__file__).with_name('ROUND12_POSTHOC_INDEPENDENT_REVIEW.json')
D=ROOT/'POSTHOC_INPUT_DIAGNOSTIC.json'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(v,m):
    if not v:raise AssertionError(m)

checks=0;maximum=0.
def same(x,y,label):
    global checks,maximum
    checks+=1
    if isinstance(x,(int,np.integer)):
        need(type(y) is int and int(x)==y,label)
    elif isinstance(x,(list,tuple,np.ndarray)):
        need(len(x)==len(y),label)
        for a,b in zip(x,y):same(a,b,label)
    else:
        need(type(y) in (float,int) and math.isfinite(y),label)
        maximum=max(maximum,abs(float(x)-y))
        need(math.isclose(float(x),y,rel_tol=1e-12,abs_tol=1e-12),label)

def describe(values,saved,label):
    same(len(values),saved['count'],label+'/count')
    same(values.mean(),saved['mean'],label+'/mean')
    for q in (0.,.5,.9,.95,.99,1.):
        same(np.quantile(values,q,method='linear'),saved['quantiles'][str(q)],label+'/quantile/'+str(q))

need(not OUT.exists(),'Fresh output')
d=json.loads(D.read_text());need(d['status']=='COMPLETE_DESCRIPTIVE_ONLY','Scope')
need(d['script_sha256']==sha(ROOT/'posthoc_input_diagnostic.py'),'Producer source SHA')
raw=json.loads((ROOT/'recover_native_v1/results.json').read_text())
need(d['complete_result_SHA']['native']==sha(ROOT/'recover_native_v1/results.json'),'Native result SHA')
by_phase={phase:{f['target_id']:f for f in raw['evaluations'][phase]['frames']} for phase in ('initial','endpoint')}
verified_files=0;verified_predictions=0;all_stats={};worst=None
for group in ('check8','fit20'):
    rows=d[group];need([r['target_id'] for r in rows]==d['member_order_'+group], 'All ordered group members')
    expected=8 if group=='check8' else 20;need(len(rows)==expected,'Group size')
    folder=ROOT/('metric_input_snapshot_v1' if group=='check8' else 'fit_gt_snapshot_v1')
    stats=[]
    for row in rows:
        t=row['target_id'];arrays={}
        for kind,item in row['source_arrays'].items():
            p=folder/item['relative_path'];need(sha(p)==item['sha256'] and p.stat().st_size==item['bytes'],'Source integrity')
            verified_files+=1;arrays[kind]=np.load(p,allow_pickle=False)
            evidence=by_phase['initial'][t]['input_evidence'] if group=='check8' else raw['training_input_evidence'][t]
            need(item['sha256']==evidence[kind],'Accepted source binding')
        mask=arrays['mask'];g=arrays['gt'].astype(np.float64)
        magnitude=np.sqrt(g[0]*g[0]+g[1]*g[1]);gm=magnitude[mask]
        same(int(mask.sum()),row['GT']['valid_pixels'],'Valid GT count')
        describe(gm,row['GT']['magnitude_pixels'],'GT distribution')
        vector=[g[0][mask].mean(),g[1][mask].mean()]
        same(vector,row['GT']['mean_vector_xy_pixels'],'GT mean vector')
        same(math.hypot(*vector)/gm.mean(),row['GT']['mean_vector_norm_divided_by_mean_magnitude'],'GT coherence')
        stats.append({'mean':float(gm.mean()),'median':float(np.median(gm)),'maximum':float(gm.max())})
        if group=='fit20':
            same(sum(s['target_id']==t for s in raw['steps']),row['training_visits_each_arm'],'Training visits')
            need(row['voxel_support'] is None and not row['fit_events_downloaded'],'Fit event limitation disclosed')
            continue
        event=arrays['events'];active=(event!=0).any(axis=0);v=row['voxel_support']
        same(int(np.count_nonzero(event)),v['nonzero_bin_entries'],'Nonzero bins')
        same(int(active.sum()),v['pixels_with_any_nonzero_bin'],'Nonzero spatial pixels')
        same(int((active&mask).sum()),v['valid_pixels_with_any_nonzero_bin'],'Nonzero valid pixels')
        same(active[mask].mean(),v['any_nonzero_fraction_within_valid_GT'],'Voxel support fraction')
        for phase,label in [('initial','initial_native'),('endpoint','recovered_N_native')]:
            frame=by_phase[phase][t];path=ROOT/'recover_native_v1'/frame['prediction_file']
            need(sha(path)==frame['prediction_file_sha256'],'NPZ SHA')
            with np.load(path,allow_pickle=False) as payload:pred32=payload['native']
            need(hashlib.sha256(pred32.tobytes()).hexdigest()==frame['arms']['native']['prediction_sha256'],'Prediction SHA')
            verified_predictions+=1;p=pred32.astype(np.float64);pm=np.sqrt(p[0]*p[0]+p[1]*p[1])
            s=row['predictions'][label]['valid_GT_support']
            describe(pm[mask],s['magnitude_pixels'],'Prediction magnitude')
            same([p[0][mask].mean(),p[1][mask].mean()],s['mean_vector_xy_pixels'],'Prediction mean vector')
            same([(p[0]-g[0])[mask].mean(),(p[1]-g[1])[mask].mean()],s['mean_component_error_xy_pixels'],'Mean component error')
            same((pm-magnitude)[mask].mean(),s['mean_predicted_minus_GT_magnitude_pixels'],'Mean magnitude difference')
            ok=mask & (pm>0) & (magnitude>0)
            cosine=(p[0][ok]*g[0][ok]+p[1][ok]*g[1][ok])/(pm[ok]*magnitude[ok])
            same(int(ok.sum()),s['direction_cosine_nonzero_vector_pixels'],'Direction defined support')
            same(cosine.mean(),s['mean_direction_cosine_nonzero_vectors'],'Mean pixel cosine')
        if t=='zurich_city_05_a_0235':worst=row
    all_stats[group]=stats
need(verified_files==64 and verified_predictions==16,'All snapshot inputs and both native predictions')
fit_max={k:max(r[k] for r in all_stats['fit20']) for k in ('mean','median','maximum')}
need(worst['GT']['magnitude_pixels']['mean']>fit_max['mean'] and worst['GT']['magnitude_pixels']['quantiles']['0.5']>fit_max['median'],'Hard frame bulk-magnitude observation')
need(worst['GT']['magnitude_pixels']['quantiles']['1.0']<fit_max['maximum'],'Fit20 does contain larger isolated magnitude')
report={'schema':'r12_posthoc_independent_review_v1','status':'PASS_DESCRIPTIVE_ONLY','script_sha256':sha(Path(__file__)),
        'diagnostic_sha256':sha(D),'diagnostic_script_sha256':d['script_sha256'],'all28_GT_frames_checked':True,
        'source_files_checked':verified_files,'native_prediction_arrays_checked':verified_predictions,'comparison_count':checks,
        'maximum_absolute_numeric_difference':maximum,'fit20_maximum_frame_mean_median_and_pixel_max':fit_max,
        'scope':'Independent original-array recomputation of all8/check and fit20 GT magnitude/vector summaries, check voxel counts and initial/post N-native vector/cosine descriptors. Not new quality evidence, gate, selection, causal inference or flow-specific contribution.'}
OUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2))
