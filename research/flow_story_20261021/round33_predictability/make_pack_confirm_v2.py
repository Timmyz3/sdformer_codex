"""CPU label pack from source-bound R33 outputs; role-specific files only."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE/'algorithm'));import predictors as models
sys.path.insert(0,str(HERE.parent/'independent_review'));import reduce_e1_quality_v1 as old
raw,mc=old.raw,old.mathcore
def read(p):return json.loads(Path(p).read_text())
def sha(p):return old.sha(p)
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--predictors-sha256',required=True);a=ap.parse_args()
    assert not a.output.exists() and sha(models.__file__)==a.predictors_sha256
    memberpath=HERE/'data/MEMBERSHIP_V1.json';assert sha(memberpath)=='d860612a3ae3e4620fa2a9a616df9f8e93774efadcf7dc564881e27b7e7da8cc'
    members=read(memberpath);result=read(a.run/'results.json')
    assert result['phase']=='confirm' and result['native_verification_sha256']=='769671e6e3754f068587eaed576ce51c0d0d6f939aeabf033a006f9aec9e6c5c'
    assert result['status']=='R33_COMPLETE' and result['schema']=='flow_r33_collector_v1'
    assert result['membership_sha256']==sha(memberpath) and result['runner_sha256']==sha(HERE/'collect_confirm_v2.py')
    assert result['completed_optimizer_steps']==0 and result['final_state_sha256']=='7be9fed7c55332a150b4c9f67e863d5e68db2bb0d1829a5344703427a48ff018'
    assert not result['GT_arrays_read'] and not result['training_performed'] and not result['quality_computed']
    roles=['fit','tune'] if result['phase']=='fit_tune' else ['confirm'];rows=[r for r in members['records'] if r['role'] in roles];ids=[r['target_id'] for r in rows]
    assert ids==result['selected_ids'] and result['model_forwards']==7*len(rows)+(6 if result['phase']=='fit_tune' else 0)
    assert result['formal_model_forwards']==7*len(rows) and result['engineering_model_forwards']==(6 if result['phase']=='fit_tune' else 0)
    if result['phase']=='fit_tune':assert len(result['parity'])==6 and all(p['comparison']['pass'] for p in result['parity'].values())
    regionspath=ROOT/'research/flow_story_20261016/coverage_recovery_round13/preparation/regions_v1.json';assert sha(regionspath)==old.REGIONS_SHA
    regions=read(regionspath);actions=['background_Q4']+[f'upgrade_{i}' for i in range(6)]
    predictions={};sources={}
    # All file bytes are bound before numerical GT/label construction.
    for rec in rows:
        sources[rec['target_id']]={}
        for f in rec['source_files']:
            p=old.child(a.source,f['relative_path']);assert sha(p)==f['sha256'] and p.stat().st_size==int(f['bytes'])
            sources[rec['target_id']][f['category']]=p
    for state in actions:
        ev=result['evaluations'][state];assert [f['target_id'] for f in ev['frames']]==ids
        workhash=hashlib.sha256(json.dumps(ev['first_frame_full_work'],sort_keys=True).encode()).hexdigest()
        for frame in ev['frames']:
            assert len(frame['actual_BN_calls'])==78 and set(frame['actual_BN_calls'].values())=={1}
            assert len(frame['actual_affine_call_counts'])==28 and set(frame['actual_affine_call_counts'].values())=={1}
            assert frame['work_sha256']==workhash,'Value-independent action geometry/work should be unchanged'
            predictions[state,frame['target_id']]=old.prediction(a.run,frame)
    features=[];labels=[];frames=[]
    for rec in rows:
        tid=rec['target_id'];arrays={k:np.load(p,allow_pickle=False) for k,p in sources[tid].items()}
        masks=raw.region_masks(arrays['gt'],arrays['mask'],arrays['input'],regions['boundary_threshold_pixels'],regions['large_displacement_threshold_pixels'])
        pred=[]
        for state in actions:
            with np.load(predictions[state,tid],allow_pickle=False) as z:pred.append(z['head3'].copy())
        measured={s:raw.measure(f,arrays['gt'],masks) for s,f in zip(actions,pred)}
        features.append(models.voxel_features(arrays['input']))
        label=models.make_labels(pred[0],np.stack(pred[1:]),arrays['gt'],masks['all_valid']);labels.append(label)
        assert label['valid_pixels']==measured['background_Q4']['all_valid']['valid_pixels']
        assert np.allclose(label['delta_aee'],[measured[s]['all_valid']['AEE']-measured['background_Q4']['all_valid']['AEE'] for s in actions[1:]],rtol=1e-9,atol=1e-9)
        frames.append(dict(target_id=tid,sequence=rec['sequence'],role=rec['role'],states=measured))
        print(tid,'packed',flush=True)
    a.output.mkdir(parents=True)
    payload=dict(ids=np.asarray(ids),groups=np.asarray([r['sequence'] for r in rows]),roles=np.asarray([r['role'] for r in rows]),x=np.stack(features))
    for k in ['valid_pixels','background_AEE','delta_aee','aux_g1','aux_g4','ideal_signed_g1','ideal_signed_g4','ideal_signed_full','support_counts_g1','support_counts_g4']:payload[k]=np.asarray([l[k] for l in labels])
    for role in roles:
        idx=np.flatnonzero(payload['roles']==role);np.savez_compressed(a.output/(role+'.npz'),**{k:v[idx] for k,v in payload.items()})
    write(a.output/'QUALITY_V1.json',dict(schema='flow_r33_source_bound_raw_quality_v1',status='COMPLETE',phase=result['phase'],source_result_sha256=sha(a.run/'results.json'),membership_sha256=sha(memberpath),runner_sha256=sha(__file__),predictors_sha256=sha(models.__file__),raw_math_sha256=sha(raw.__file__),scalar_math_sha256=sha(mc.__file__),regions_sha256=sha(regionspath),source_arrays=len(rows)*3,source_predictions=len(rows)*7,states=actions,frames=frames,offline_GT_and_counterfactuals_used=True))
    write(a.output/'PACK_FREEZE_V1.json',dict(schema='flow_r33_pack_freeze_v1',phase=result['phase'],files={p.name:sha(p) for p in a.output.iterdir() if p.is_file()},model_forwards=0,GPU_used=False))
if __name__=='__main__':main()
