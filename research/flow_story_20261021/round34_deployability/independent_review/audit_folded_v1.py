"""Independent exact fold, fixed-format trace and selected quality verification."""
from pathlib import Path
import json,os
import numpy as np
import audit_numeric_v1 as N
from audit_ordinary_numeric_v1 import affine
B=N.B;R33=N.R33;D=B/'numeric/folded_constant_v';A=D/'actual_v1'
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';out=B/'independent_review/FOLDED_CONSTANT_INDEPENDENT_AUDIT_V1.json';assert not out.exists()
    freeze=N.read(D/'SOURCE_FREEZE_V1.json')
    for p,h in freeze['files'].items():assert N.sha(D/p)==h
    for b in freeze['bound'].values():assert N.sha(b['path'])==b['sha256']
    oldpath=R33/'fitted/voxel128_v1/MODEL_V1.json';old=N.read(oldpath)['auxiliary']['g1'];saved=N.read(A/'FOLDED_MODEL_V1.json')
    v=np.asarray(old['intercept'])[:2];coef=np.asarray(old['coef'])[:,2:].reshape(128,6,2);bias=np.asarray(old['intercept'])[2:].reshape(6,2)
    independent=dict(mean=old['mean'],scale=old['scale'],coef=coef[:,:,0]*v[0]+coef[:,:,1]*v[1],intercept=bias[:,0]*v[0]+bias[:,1]*v[1])
    assert saved['original_model_sha256']==N.sha(oldpath) and saved['no_refit'] and saved['coefficient_count']==768 and saved['intercept_count']==6
    N.eq(v,saved['constant_direction'])
    for k in independent:N.eq(independent[k],saved['ridge'][k])
    cal=N.read(A/'CALIBRATION_V1.json');cf=N.read(A/'CALIBRATION_FREEZE_V1.json');features=N.load(B/'numeric/actual_v1/FEATURES_FP32_V1.npz')
    assert cal['fit_only'] and cal['fit_ids']==features['fit_ids'].tolist() and cal['folded_model_sha256']==N.sha(A/'FOLDED_MODEL_V1.json')
    assert cal['numerical_source_sha256']==N.sha(B/'numeric/core.py') and cf['calibration_sha256']==N.sha(A/'CALIBRATION_V1.json') and not cf['tune_confirm_inference_started']
    report={};counts=dict(affine_traces=0,frame_region_metrics=0,sequence_region_metrics=0);maxfold=0.
    for role in ['fit','tune','confirm']:
        folder='confirm' if role=='confirm' else 'fit_tune';p=R33/f'packs/{folder}_128_v1/{role}.npz';pack=N.load(p);ids=pack['ids'].tolist()
        a=N.load(A/f'{role}_ARRAYS_V1.npz');N.eq(a['ids'],ids);q=N.read(A/f'{role}_QUALITY_V1.json')
        assert q['fit_is_resubstitution']==(role=='fit') and q['allow_Q4_is_new_diagnostic_extension']
        trace={}
        for k,value in a.items():
            if k.startswith('fp32_trace__'):trace['trace__fp32__fold__'+k[len('fp32_trace__'):]]=value
            if k.startswith('mixed_trace__'):trace['trace__mixed_int16__fold__'+k[len('mixed_trace__'):]]=value
        x=pack['x'];aux=((x-np.asarray(old['mean']))/np.asarray(old['scale']))@np.asarray(old['coef'])+np.asarray(old['intercept'])
        unfolded=(aux[:,2:].reshape(len(x),6,2)*v).sum(axis=2)
        modes={'unfolded_fp64':unfolded,'folded_fp64':affine(independent,x,'fp64',None,trace,'fold',False)}
        maxfold=max(maxfold,float(np.max(np.abs(unfolded-modes['folded_fp64']))))
        for mode in ['fp32','mixed_int16']:
            modes[mode]=affine(independent,features[role+'_x160'][:,:128],mode,cal['calibration'],trace,'fold',role=='fit');counts['affine_traces']+=1
        raw={r['target_id']:r for r in N.read(p.parent/'QUALITY_V1.json')['frames']};regs=list(raw[ids[0]]['states']['background_Q4'])
        ns=np.array([[raw[i]['states']['background_Q4'][r]['valid_pixels'] for r in regs] for i in ids]);states=['background_Q4']+[f'upgrade_{i}' for i in range(6)]
        errors=np.array([[[raw[i]['states'][s][r]['error_sum'] for r in regs] for s in states] for i in ids]);ai=regs.index('all_valid');report[role]={}
        for mode,s in modes.items():
            if mode in ['fp32','mixed_int16']:N.eq(a['scores__'+mode],s)
            else:N.close(a['scores__'+mode],s)
            for allow,label in [(False,'fixed_six'),(True,'allow_Q4')]:
                choices=s.argmin(axis=1);ref=unfolded.argmin(axis=1)
                if allow:choices=np.where(s.min(axis=1)>=0,-1,choices);ref=np.where(unfolded.min(axis=1)>=0,-1,ref)
                N.eq(a[f'choices__{mode}__{label}'],choices);assert q['bridge'][mode][label+'_flips']==int(np.sum(choices!=ref))
                name=mode+'__'+label;ee=errors[np.arange(len(ids)),choices+1];tails=[]
                for i,tid in enumerate(ids):
                    stored=q['frames'][i]['policies'][name];assert stored['action']==choices[i]
                    for j,r in enumerate(regs):
                        rr=stored['regions'][r];assert rr['valid_pixels']==ns[i,j];N.close(rr['error_sum'],ee[i,j]);counts['frame_region_metrics']+=1
                    for r in ['all_valid','B','L']:
                        j=regs.index(r)
                        if ns[i,j]>=256 and (ee[i,j]>1.2*errors[i,0,j] or ee[i,j]-errors[i,0,j]>ns[i,j]):tails.append((tid,r))
                assert tails==[(r['target_id'],r['region']) for r in q['comparisons'][name+'__vs_Q4']['high_support_tail_review']]
                seqrows={};leave={}
                for seq in ['overall']+sorted(set(pack['groups'])):
                    ix=np.ones(len(ids),bool) if seq=='overall' else pack['groups']==seq
                    for j,r in enumerate(regs):
                        n=int(ns[ix,j].sum());e=float(ee[ix,j].sum());rr=q['summaries'][name][seq][r];assert rr['valid_pixels']==n;N.close(rr['error_sum'],e)
                        if n:N.close(rr['pixel_weighted_AEE'],e/n)
                        else:assert rr['pixel_weighted_AEE'] is None
                        counts['sequence_region_metrics']+=1
                    if seq!='overall':
                        seqrows[seq]=float((ee[ix,ai]-errors[ix,0,ai]).sum()/ns[ix,ai].sum())
                        leave[seq]=float((ee[~ix,ai]-errors[~ix,0,ai]).sum()/ns[~ix,ai].sum())
                report[role][name]=dict(AEE=float(ee[:,ai].sum()/ns[:,ai].sum()),upgraded=int((choices>=0).sum()),flips_to_unfolded=int(np.sum(choices!=ref)),high_support_tails=tails,sequence_delta_vs_Q4=seqrows,leave_one_sequence_delta_vs_Q4=leave)
    final=dict(schema='flow_r34_folded_constant_independent_audit_v1',status='PASS',auditor_sha256=N.sha(__file__),numeric_helper_sha256=N.sha(Path(N.__file__)),
        affine_helper_sha256=N.sha(B/'independent_review/audit_ordinary_numeric_v1.py'),source_freeze_sha256=N.sha(D/'SOURCE_FREEZE_V1.json'),
        folded_model_sha256=N.sha(A/'FOLDED_MODEL_V1.json'),results_sha256=N.sha(A/'SUMMARY_V1.json'),counts=counts,max_absolute_fold_fp64_difference=maxfold,results=report,
        scope='Exact deterministic linear fold; mixed readout only. Same128-to6 function class as ordinary direct prediction; supervision provenance differs. All data exposed; allowQ4 new diagnostic, fit resubstitution.',GPU_calls=0)
    out.write_text(json.dumps(final,indent=2)+'\n');print(json.dumps({'status':'PASS','counts':counts,'max_fold_difference':maxfold}))
if __name__=='__main__':main()
