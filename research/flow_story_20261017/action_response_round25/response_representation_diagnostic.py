"""Post-hoc representation diagnostic on exposed R24, no training or policy release.

The exact and pooled response oracles use final outputs; gain additionally uses GT.
Neither oracle is a legal runtime scorer. Does not reopen the stopped action family.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
R24=HERE.parent/'two_slot_action_round24'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def read(p): return json.loads(Path(p).read_text())
def load(p,n):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def rank(v):
    v=np.asarray(v);r=np.empty(len(v),dtype=np.float64)
    for x in np.unique(v):
        ix=np.flatnonzero(v==x);r[ix]=np.count_nonzero(v<x)+(len(ix)-1)/2
    return r

def statistics(truth,score):
    truth,score=np.asarray(truth),np.asarray(score)
    a,b=rank(truth),rank(score)
    corr=float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None
    p=int(np.argmax(score));best=int(np.argmax(truth))
    return dict(spearman=corr,selected_index=p,oracle_best_index=best,
                top1_match=p==best,true_gain_at_selected=float(truth[p]),
                regret_AEE=float(truth[best]-truth[p]),
                gain_mae=float(np.abs(truth-score).mean()))

def run():
    reducer=load(R24/'independent_review/reduce_raw_v1.py','r24_diagnostic_sources')
    result_dir=R24/'actual_v1/run_v1';rp=result_dir/'results.json';result=read(rp)
    audit=read(R24/'independent_execution_audit/ACTUAL_EXECUTION_AUDIT_V1.json')
    reducer.validate_complete(result,audit)
    assert audit['source_result_sha256']==sha(rp)
    quality=read(R24/'independent_review/ACTUAL_RAW_QUALITY_V1.json')
    assert quality['source_result_sha256']==sha(rp)
    members=read(reducer.H8)['members'];ids=[m['target_id'] for m in members]
    metric,_=reducer.frozen_math()
    cache,evidence=metric.load_snapshot(reducer.SNAPSHOT,reducer.SNAPSHOT/'receipt.json',ids)
    region=read(reducer.REGIONS)
    predictions={};files=[]
    for r in result['forward_records']:
        if not r['quality_endpoint'] or r['arm']=='FF': continue
        p=result_dir/r['prediction_file'];assert sha(p)==r['prediction_file_sha256']
        assert r['input_evidence']==evidence[r['target_id']]
        with np.load(p,allow_pickle=False) as z:
            a=z['head3'];assert a.shape==(1,2,480,640) and a.dtype==np.float32
            assert hashlib.sha256(a.tobytes()).hexdigest()==r['head_records'][3]['sha256']
            predictions[r['target_id'],r['arm']]=a[0].copy()
        files.append(dict(file=p.name,sha256=sha(p)))
    arms=tuple(f'fine_{r}_{c}' for r in range(3) for c in range(4))
    frames=[]
    for tid in ids:
        src=cache[tid];gt=src['gt'].astype(np.float64)
        masks=metric.masks_from_original(src['gt'],src['mask'],src['events'],region['boundary_threshold_pixels'],region['large_displacement_threshold_pixels'])
        bg=predictions[tid,'T2'];base=metric.measure(bg,src['gt'],masks)
        e=bg.astype(np.float64)-gt;en=np.linalg.norm(e,axis=0)
        rows=[]
        for arm in arms:
            pred=predictions[tid,arm];d=pred-bg
            grid=d.reshape(2,30,16,40,16).mean((2,4),dtype=np.float32)
            expand=grid.repeat(16,axis=1).repeat(16,axis=2)
            reconstructed=bg+expand
            actual=metric.measure(pred,src['gt'],masks)
            approximate=metric.measure(reconstructed,src['gt'],masks)
            full_reconstructed=metric.measure(bg+d,src['gt'],masks)
            # First-order response with GT residual direction: oracle, not runtime.
            first_order=-np.sum(e*d.astype(np.float64),axis=0)/np.maximum(en,1e-12)
            regs={}
            for name,mask in masks.items():
                n=int(mask.sum())
                if not n:
                    regs[name]=dict(valid_pixels=0,true_gain=None,pooled_oracle_gain=None,full_oracle_gain=None,GT_direction_taylor_gain=None,response_magnitude=None);continue
                regs[name]=dict(valid_pixels=n,true_gain=base[name]['AEE']-actual[name]['AEE'],
                   pooled_oracle_gain=base[name]['AEE']-approximate[name]['AEE'],
                   full_oracle_gain=base[name]['AEE']-full_reconstructed[name]['AEE'],
                   GT_direction_taylor_gain=float(first_order[mask].mean()),
                   response_magnitude=float(np.linalg.norm(d.astype(np.float64),axis=0)[mask].mean()))
            rows.append(dict(arm=arm,regions=regs,all_pixel_response_magnitude=float(np.linalg.norm(d.astype(np.float64),axis=0).mean())))
        diagnostics={}
        for name,mask in masks.items():
            if not mask.any():diagnostics[name]=None;continue
            true=[r['regions'][name]['true_gain'] for r in rows]
            scores={k:[r['regions'][name][k] for r in rows] for k in ('pooled_oracle_gain','full_oracle_gain','GT_direction_taylor_gain','response_magnitude')}
            scores['all_pixel_response_magnitude']=[r['all_pixel_response_magnitude'] for r in rows]
            diagnostics[name]={k:statistics(true,v) for k,v in scores.items()}
        frames.append(dict(target_id=tid,actions=rows,diagnostics=diagnostics))
    summaries={}
    for name in metric.REGIONS:
        good=[f['diagnostics'][name] for f in frames if f['diagnostics'][name] is not None]
        summaries[name]={}
        if not good:continue
        for k in good[0]:
            values=[v[k] for v in good]
            corrs=[v['spearman'] for v in values if v['spearman'] is not None]
            summaries[name][k]=dict(frames=len(values),top1_matches=sum(v['top1_match'] for v in values),
                mean_spearman=float(np.mean(corrs)) if corrs else None,
                mean_regret_AEE=float(np.mean([v['regret_AEE'] for v in values])),
                mean_gain_mae=float(np.mean([v['gain_mae'] for v in values])))
    return dict(schema='flow_r25_posthoc_r24_response_representation_v1',status='COMPLETE_OFFLINE_ONLY',
       source_sha256=sha(__file__),result_sha256=sha(rp),quality_sha256=sha(R24/'independent_review/ACTUAL_RAW_QUALITY_V1.json'),
       actual_new_model_forwards=0,optimizer_steps=0,runtime_policy=False,R24_STOP_remains=True,
       design='Post-hoc exposed H8; fixed R25 FP32 delta then16x16mean andnearest expand; no learned PCA or predictor.',
       score_direction='All scores maximized for restore action; large response magnitude is a heuristic, not signed gain.',
       oracle_warning='Every delta uses an executed final counterfactual; pooled gain and Taylor also use GT. No deployable score or heldout claim.',
       magnitude_mae_warning='Magnitude scores are not gain estimates; their MAE is scale diagnostic only.',
       archives=files,frames=frames,summaries=summaries)

if __name__=='__main__':
    out=HERE/'RESPONSE_REPRESENTATION_DIAGNOSTIC_V1.json'
    assert not out.exists(),'Keep prior outputs immutable'
    x=run();out.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
    print(json.dumps(x['summaries']['all_valid'],indent=2))
