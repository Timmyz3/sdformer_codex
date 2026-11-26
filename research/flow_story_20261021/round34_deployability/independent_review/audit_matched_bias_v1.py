"""Independent all-action original-array and fixed-policy bias comparison."""
from pathlib import Path
import json,os
import numpy as np
import audit_bias_v1 as H
N=H.N;M=H.M;B=N.B;R33=N.R33;D=B/'bias_control/extension_v1';A=D/'actual_v1'
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';out=B/'independent_review/MATCHED_BIAS_INDEPENDENT_AUDIT_V1.json';assert not out.exists()
    ba=N.read(B/'independent_review/BIAS_INDEPENDENT_AUDIT_V1.json');assert ba['status']=='PASS'
    freeze=N.read(D/'SOURCE_FREEZE_V1.json')
    for p,h in freeze['files'].items():assert N.sha(D/p)==h
    for f in freeze['bound'].values():assert N.sha(f['path'])==f['sha256']
    bp=B/'bias_control/actual_v1/BIAS_MODEL_V1.json';bm=N.read(bp);assert ba['bias_model_sha256']==N.sha(bp);bias=np.asarray(bm['bias'])
    frozen=N.read(A/'FROZEN_POLICY_CHOICES_V1.json');assert frozen['bias_model_sha256']==N.sha(bp) and not frozen['corrected_quality_computed']
    bindings=N.read(A/'ALL_ACTION_INPUT_BINDINGS_V1.json');members=N.read(R33/'data/MEMBERSHIP_V1.json')['records']
    region=N.read(R33.parents[1]/'flow_story_20261016/coverage_recovery_round13/preparation/regions_v1.json')
    counts=dict(source_predictions=0,raw_action_region_values=0,policy_frame_regions=0,sequence_region_values=0,uniform_risk_frames=0);tables={}
    for role in ['fit','tune','confirm']:
        fc=frozen['roles'][role]
        for p,h in fc['source_hashes'].items():assert N.sha(p)==h
        ids=fc['ids'];choices={}
        for dim in [128,160]:
            pred=N.read(B/f'numeric/actual_v1/{role}_{dim}/PREDICTIONS_V1.json');assert pred['ids']==ids
            for label,key in [('six','fixed_six_choices'),('allow_Q4','allow_Q4_choices')]:
                for method,v in pred['results']['fp64'][key].items():choices[f'X{dim}__{method}__{label}']=v
            arr=N.load(B/f'numeric/ordinary_extension_actual_v1/{role}_{dim}_ARRAYS_V1.npz')
            for label,key in [('six','fixed_six'),('allow_Q4','allow_Q4')]:choices[f'X{dim}__calibrated_ordinary__{label}']=arr['results__fp64__'+key].tolist()
        arr=N.load(B/f'numeric/folded_constant_v/actual_v1/{role}_ARRAYS_V1.npz')
        for label,key in [('six','fixed_six'),('allow_Q4','allow_Q4')]:choices['X128__folded_constant_v__'+label]=arr['choices__unfolded_fp64__'+key].tolist()
        mm=N.read(R33/'fitted/voxel128_v1/MODEL_V1.json')['metadata']
        choices['Q4']=[-1]*len(ids);choices['fit_static__six']=[mm['fit_static_six_action']]*len(ids);choices['fit_static__allow_Q4']=[mm['fit_static_allow_Q4_action']]*len(ids)
        assert choices==fc['choices'] and len(choices)==37
        rows=[r for r in members if r['role']==role];assert ids==[r['target_id'] for r in rows];groups=np.array([r['sequence'] for r in rows])
        phase='confirm' if role=='confirm' else 'fit_tune';run=R33/f'actual/{phase}_v1';result=N.read(run/'results.json')
        states=['background_Q4']+[f'upgrade_{i}' for i in range(6)];lookup={s:{r['target_id']:r for r in result['evaluations'][s]['frames']} for s in states}
        raw=N.read(A/f'{role}_ALL_ACTION_QUALITY_V1.json');original={r['target_id']:r for r in N.read(R33/f'packs/{phase}_128_v1/QUALITY_V1.json')['frames']}
        allzero=[];allcorrected=[];numbers=[]
        for i,r in enumerate(rows):
            tid=r['target_id'];src={}
            for f in r['source_files']:
                p=R33/f'data/{phase}_source_snapshot'/f['relative_path'];assert N.sha(p)==f['sha256'];src[f['category']]=np.load(p,allow_pickle=False)
            masks=M.masks_from_sources(src['gt'],src['mask'],src['input'],region['boundary_threshold_pixels'],region['large_displacement_threshold_pixels']);names=list(masks)
            numbers.append([int(m.sum()) for m in masks.values()]);zero=[];correct=[]
            for s in states:
                rr=lookup[s][tid];p=run/rr['prediction_file'];assert N.sha(p)==rr['prediction_file_sha256']==bindings['predictions'][tid+'/'+s]['sha256']
                pred=N.load(p)['head3'];changed=pred.astype(float)-bias[:,None,None]
                assert N.hashlib.sha256(np.ascontiguousarray(changed).tobytes()).hexdigest()==raw['corrected_payloads'][i]['corrected_payload_sha256'][s]
                ze=np.sqrt(np.sum((pred.astype(float)-src['gt'].astype(float))**2,axis=0));ce=np.sqrt(np.sum((changed-src['gt'].astype(float))**2,axis=0));zs=[];cs=[]
                for name,m in masks.items():
                    n=int(m.sum());z=float(ze[m].sum());c=float(ce[m].sum());zs.append(z);cs.append(c)
                    for condition,es in [('uncorrected',z),('corrected',c)]:
                        stored=raw[condition][i]['states'][s][name];assert stored['valid_pixels']==n;N.close(stored['error_sum'],es)
                        if n:N.close(stored['AEE'],es/n)
                        else:assert stored['AEE'] is None
                        counts['raw_action_region_values']+=1
                    assert raw['uncorrected'][i]['states'][s][name]==original[tid]['states'][s][name]
                zero.append(zs);correct.append(cs);counts['source_predictions']+=1
            allzero.append(zero);allcorrected.append(correct)
        ns=np.asarray(numbers);ai=names.index('all_valid');tables[role]={}
        for condition,error in [('uncorrected',np.asarray(allzero)),('same_bias_corrected',np.asarray(allcorrected))]:
            ev=N.read(A/f'{role}_{condition}_EVALUATION_V1.json');assert ev['unchanged_observable_policy_choices']==choices
            policies={k:np.asarray(v) for k,v in choices.items()};policies['uniform_six_expectation']=None
            delta=(error[:,1:,ai]-error[:,:1,ai])/ns[:,ai,None]
            policies['oracle__six']=delta.argmin(axis=1);policies['oracle__allow_Q4']=np.where(delta.min(axis=1)>=0,-1,delta.argmin(axis=1));assert len(policies)==40
            tables[role][condition]={}
            for name,a in policies.items():
                selected=error[:,1:].mean(axis=1) if a is None else error[np.arange(len(ids)),a+1];tails=[];seqd={};leave={}
                for i,tid in enumerate(ids):
                    saved=ev['frames'][i]['policies'][name];assert saved['action']==(None if a is None else int(a[i]))
                    for j,rn in enumerate(names):
                        v=saved['regions'][rn];assert v['valid_pixels']==ns[i,j];N.close(v['error_sum'],selected[i,j]);counts['policy_frame_regions']+=1
                    for rn in ['all_valid','B','L']:
                        j=names.index(rn)
                        if ns[i,j]>=256 and (selected[i,j]>1.2*error[i,0,j] or selected[i,j]-error[i,0,j]>ns[i,j]):tails.append((tid,rn))
                if name!='Q4':assert tails==[(r['target_id'],r['region']) for r in ev['comparisons'][name+'__vs_Q4']['high_support_tail_review']]
                for seq in ['overall']+list(dict.fromkeys(groups.tolist())):
                    ix=np.ones(len(ids),bool) if seq=='overall' else groups==seq
                    for j,rn in enumerate(names):
                        n=int(ns[ix,j].sum());es=float(selected[ix,j].sum());v=ev['summaries'][name][seq][rn];assert v['valid_pixels']==n;N.close(v['error_sum'],es)
                        if n:N.close(v['pixel_weighted_AEE'],es/n)
                        else:assert v['pixel_weighted_AEE'] is None
                        counts['sequence_region_values']+=1
                    if seq!='overall':
                        seqd[seq]=float((selected[ix,ai]-error[ix,0,ai]).sum()/ns[ix,ai].sum())
                        leave[seq]=float((selected[~ix,ai]-error[~ix,0,ai]).sum()/ns[~ix,ai].sum())
                        for rn in ['all_valid','B','L']:
                            j=names.index(rn);den=int(ns[~ix,j].sum());candidate=selected[~ix,j].sum()/den;base=error[~ix,0,j].sum()/den
                            saved=ev['leave_one_sequence'][name][seq][rn];assert saved['valid_pixels']==den;N.close(saved['candidate_AEE'],candidate);N.close(saved['Q4_AEE'],base);N.close(saved['delta'],candidate-base);N.close(saved['relative'],candidate/base-1)
                aee=float(selected[:,ai].sum()/ns[:,ai].sum());baseaee=float(error[:,0,ai].sum()/ns[:,ai].sum())
                tables[role][condition][name]=dict(AEE=aee,relative_vs_Q4=aee/baseaee-1,upgraded_expected=float(len(ids) if a is None else np.sum(a>=0)),sequence_delta_vs_Q4=seqd,
                    leave_one_sequence_delta_vs_Q4=leave,high_support_tail_items=tails,mean_tail_only_if_uniform=a is None,
                    regions={rn:dict(AEE=float(selected[:,j].sum()/ns[:,j].sum()),relative_vs_Q4=float(selected[:,j].sum()/error[:,0,j].sum()-1)) if ns[:,j].sum() else None for j,rn in enumerate(names)})
            risk=ev['uniform_six_actual_tail_risk'];expected=0.
            for i,row in enumerate(risk['frames']):
                assert row['target_id']==ids[i];union=set()
                for rn in ['all_valid','B','L']:
                    j=names.index(rn);bad=[]
                    if ns[i,j]>=256:
                        bad=[a for a in range(6) if error[i,a+1,j]>1.2*error[i,0,j] or error[i,a+1,j]-error[i,0,j]>ns[i,j]]
                    assert row['regions'][rn]['triggering_actions']==bad;N.close(row['regions'][rn]['trigger_probability'],len(bad)/6);union.update(bad)
                assert row['any_primary_region_triggering_actions']==sorted(union);N.close(row['any_primary_region_trigger_probability'],len(union)/6);expected+=len(union)/6;counts['uniform_risk_frames']+=1
            tables[role][condition]['uniform_six_expectation']['realized_expected_any_primary_bad_frames']=expected
    final=dict(schema='flow_r34_matched_bias_independent_audit_v1',status='PASS',auditor_sha256=N.sha(__file__),mask_helper_sha256=N.sha(Path(M.__file__)),
        source_freeze_sha256=N.sha(D/'SOURCE_FREEZE_V1.json'),bias_model_sha256=N.sha(bp),summary_sha256=N.sha(A/'SUMMARY_V1.json'),counts=counts,independent_policy_quality=tables,
        scope='All840 corrected real outputs independently recomputed, observable policies unchanged. Same output correction for every compared method. Only offline oracle reselects. Exposed-data diagnostic; fit resubstitution; postprocessing/prefix cycles not established here.',GPU_calls=0,new_fit_or_policy_selection=False)
    out.write_text(json.dumps(final,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':'PASS','counts':counts,'confirm_corrected':{k:v['AEE'] for k,v in tables['confirm']['same_bias_corrected'].items() if k in ['Q4','X128__folded_constant_v__six','X128__calibrated_ordinary__six','X160__signed_g1__six']}}))
if __name__=='__main__':main()
