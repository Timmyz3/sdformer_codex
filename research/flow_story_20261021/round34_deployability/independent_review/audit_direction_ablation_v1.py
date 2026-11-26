"""Independent three-readout attribution and fixed-direction linearity check."""
from pathlib import Path
import json,os
import numpy as np
import audit_ordinary_v1 as H
B=H.B;R33=H.R33;A=B/'ordinary/direction_ablation_v1'

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';out=B/'independent_review/DIRECTION_ABLATION_INDEPENDENT_AUDIT_V1.json';assert not out.exists()
    freeze=H.read(B/'ordinary/DIRECTION_ABLATION_FREEZE_V1.json')
    for name,h in freeze['files'].items():assert H.sha(B/'ordinary'/name)==h
    saved=H.read(A/'RESULTS_V1.json');pred=H.read(A/'PREDICTIONS_BEFORE_QUALITY_V1.json')
    assert saved['status']=='COMPLETE' and saved['only_fixed_six'] and saved['new_fits']==0 and saved['new_model_forwards']==0 and not saved['GPU_used']
    assert saved['source_sha256']==pred['source_sha256']==H.sha(B/'ordinary/run_direction_ablation_v1.py')
    assert saved['summarizer_source_sha256']==H.sha(B/'ordinary/run_calibration_v1.py')
    table={};equivalence={};count=0
    for dim,branch in [(128,'voxel128_v1'),(160,'directional160_v1')]:
        mp=R33/f'fitted/{branch}/MODEL_V1.json';m=H.read(mp);aux=m['auxiliary']['g1'];assert H.sha(mp)==saved['sources'][f'model_{dim}']
        d=H.load(R33/f'packs/fit_tune_{dim}_v1/fit.npz');w=d['valid_pixels']/sum(d['valid_pixels'])
        v=np.sum(w[:,None]*d['aux_g1'][:,:2],axis=0);H.close(v,np.asarray(aux['intercept'])[:2])
        coeff=np.asarray(aux['coef'])[:,2:].reshape(dim,6,2)
        folded_coef=np.einsum('dao,o->da',coeff,v)
        folded_intercept=np.asarray(aux['intercept'])[2:].reshape(6,2)@v
        yproj=np.einsum('nao,o->na',d['aux_g1'][:,2:].reshape(64,6,2),v)
        x=(d['x']-np.asarray(aux['mean']))/np.asarray(aux['scale'])
        gradient=x.T@(w[:,None]*(x@folded_coef+folded_intercept-yproj))+.1*folded_coef
        kkt=float(np.max(np.abs(gradient)));assert kkt<1e-9
        H.close(folded_intercept,np.sum(w[:,None]*yproj,axis=0))
        equivalence[str(dim)]=dict(fit_constant_direction=v.tolist(),ridge_alpha=.1,folded_input_dimension=dim,folded_output_dimension=6,
            projected_label='fit_weighted_mean_v dot true_mean_valid_response_d_a',weighted_stationarity_max=kkt,
            shared_fit_standardization=True,independent_equivalence='A single ordinary ridge on these projected-response labels has the same unique optimum.',
            coefficients_sha256=H.hashlib.sha256(folded_coef.tobytes()).hexdigest())
        for role,folder in [('tune','fit_tune'),('old_confirm','confirm')]:
            key=f'{dim}_{role}';packpath=R33/f'packs/{folder}_{dim}_v1/{"confirm" if role=="old_confirm" else role}.npz'
            q=H.load(packpath);assert H.sha(packpath)==saved['sources'][key+'_pack']
            qp=packpath.parent/'QUALITY_V1.json';assert H.sha(qp)==saved['sources'][key+'_quality']
            pa=H.read(R33/'independent_review'/('CONFIRM_PACK_INDEPENDENT_AUDIT_V1.json' if role=='old_confirm' else 'FIT_TUNE_PACK_INDEPENDENT_AUDIT_V1.json'))
            assert pa['status']=='PASS' and pa['quality_sha256']==H.sha(qp)
            ids=q['ids'].tolist();p=pred['predictions'][key];assert ids==p['ids'];H.close(p['constant_direction'],v)
            z=H.predict(aux,q['x']);response=z[:,2:].reshape(len(ids),6,2)
            scores={'predicted_direction':np.einsum('no,nao->na',z[:,:2],response),
                'fit_constant_direction':np.einsum('o,nao->na',v,response),
                'predicted_response_squared_norm':np.sum(response**2,axis=2)}
            folded=((q['x']-np.asarray(aux['mean']))/np.asarray(aux['scale']))@folded_coef+folded_intercept
            H.close(folded,scores['fit_constant_direction']);equivalence[str(dim)][role+'_max_fold_score_difference']=float(np.max(np.abs(folded-scores['fit_constant_direction'])))
            raw={r['target_id']:r for r in H.read(qp)['frames']};regs=list(raw[ids[0]]['states']['background_Q4']);assert len(regs)==16
            ns=np.array([[raw[i]['states']['background_Q4'][r]['valid_pixels'] for r in regs] for i in ids])
            base=np.array([[raw[i]['states']['background_Q4'][r]['error_sum'] for r in regs] for i in ids]);ai=regs.index('all_valid');table[key]={}
            for name,s in scores.items():
                H.close(s,p['scores'][name]);actions=s.argmin(axis=1);stored=saved['results'][key][name];assert stored['choices']==actions.tolist()
                errors=np.array([[raw[i]['states'][f'upgrade_{a}'][r]['error_sum'] for r in regs] for i,a in zip(ids,actions)]);tails=[]
                for n,tid in enumerate(ids):
                    assert stored['frames'][n]['target_id']==tid and stored['frames'][n]['action']==actions[n]
                    for j,r in enumerate(regs):
                        rr=stored['frames'][n]['regions'][r];assert rr['valid_pixels']==ns[n,j];H.close(rr['error_sum'],errors[n,j]);count+=1
                    for r in ['all_valid','B','L']:
                        j=regs.index(r)
                        if ns[n,j]>=256 and (errors[n,j]>base[n,j]*1.2 or errors[n,j]-base[n,j]>ns[n,j]):tails.append((tid,r))
                assert tails==[(r['target_id'],r['region']) for r in stored['high_support_tail_review']]
                seqrows={};leave={}
                for seq in ['overall']+list(dict.fromkeys(q['groups'].tolist())):
                    ix=np.ones(len(ids),bool) if seq=='overall' else q['groups']==seq
                    for j,r in enumerate(regs):
                        n=int(ns[ix,j].sum());e=float(errors[ix,j].sum());rr=stored['summaries'][seq][r]
                        assert rr['valid_pixels']==n;H.close(rr['error_sum'],e)
                        if n:H.close(rr['pixel_weighted_AEE'],e/n)
                        else:assert rr['pixel_weighted_AEE'] is None
                    if seq!='overall':
                        seqrows[seq]=float((errors[ix,ai]-base[ix,ai]).sum()/ns[ix,ai].sum())
                        leave[seq]=float((errors[~ix,ai]-base[~ix,ai]).sum()/ns[~ix,ai].sum())
                aee=float(errors[:,ai].sum()/ns[:,ai].sum());q4=float(base[:,ai].sum()/ns[:,ai].sum())
                regional={r:dict(AEE=float(errors[:,j].sum()/ns[:,j].sum()),relative_vs_Q4=float(errors[:,j].sum()/base[:,j].sum()-1)) if ns[:,j].sum() else None for j,r in enumerate(regs)}
                table[key][name]=dict(AEE=aee,relative_vs_Q4=aee/q4-1,regions=regional,sequence_delta_vs_Q4=seqrows,
                    leave_one_sequence_delta_vs_Q4=leave,high_support_tail_items=tails,choices=actions.tolist())
    report=dict(schema='flow_r34_direction_ablation_independent_audit_v1',status='PASS',auditor_sha256=H.sha(__file__),helper_sha256=H.sha(Path(H.__file__)),
        source_freeze_sha256=H.sha(B/'ordinary/DIRECTION_ABLATION_FREEZE_V1.json'),results_sha256=H.sha(A/'RESULTS_V1.json'),
        frame_region_values_checked=count,fixed_direction_equivalence=equivalence,independent_quality=table,
        scope='Posthoc attribution on already exposed tune/confirm. Fixed direction is fit-only, not runtime GT. Exact ordinary projected-response-label ridge equivalence; no new fitting or new novelty claim.',GPU_calls=0)
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'PASS','frame_region_values':count,'equivalence':equivalence}))

if __name__=='__main__':main()
