"""Independently verify the fixed-format extension for calibrated controls."""
from pathlib import Path
import json,os
import numpy as np
import audit_numeric_v1 as N
B=N.B;R33=N.R33;A=B/'numeric/ordinary_extension_actual_v1';F=np.float32
def affine(m,x,mode,c,trace,key,fit):
    dtype=float if mode=='fp64' else F;r={k:np.asarray(m[k],dtype) for k in ['mean','scale','intercept','coef']}
    z=(x-r['mean'])/r['scale']
    if mode=='fp64':return z@r['coef']+r['intercept']
    N.eq(trace[f'trace__{mode}__{key}__standardized'],z)
    if mode=='fp32':
        out=np.zeros((len(z),len(r['intercept'])),F)
        for k in range(z.shape[1]):out=np.add(out,np.multiply(z[:,k,None],r['coef'][k]))
    else:
        sx=F(c['input_scale']);sw=np.asarray(c['weight_scale'],F);qw,ws=N.quant(r['coef'],sw)
        N.eq(qw,c['weight_q']);N.eq(ws,c['weight_saturation']);N.eq(sw,N.scale_for(np.max(np.abs(r['coef']),axis=0)))
        if fit:N.eq(sx,N.scale_for(np.max(np.abs(z))))
        qx,sat=N.quant(z,sx);acc=np.zeros((len(z),qw.shape[1]),np.int64)
        for k in range(z.shape[1]):acc+=qx[:,k,None].astype(np.int64)*qw[k].astype(np.int64)
        for name,v in [('input_q',qx),('input_saturation',sat),('int64_accumulator',acc)]:N.eq(trace[f'trace__{mode}__{key}__{name}'],v)
        out=np.multiply(np.multiply(acc.astype(F),sx),sw)
    out=np.add(out,r['intercept']);N.eq(trace[f'trace__{mode}__{key}__output'],out)
    return out

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';out=B/'independent_review/ORDINARY_NUMERIC_INDEPENDENT_AUDIT_V1.json';assert not out.exists()
    primary=N.read(B/'independent_review/NUMERIC_INDEPENDENT_AUDIT_V1.json');assert primary['status']=='PASS'
    freeze=N.read(B/'numeric/ORDINARY_EXTENSION_FREEZE_V1.json')
    for name,h in freeze['files'].items():assert N.sha(B/'numeric'/name)==h
    for r in freeze['bound'].values():assert N.sha(r['path'])==r['sha256']
    c=N.read(A/'CALIBRATION_V1.json');cf=N.read(A/'CALIBRATION_FREEZE_V1.json')
    assert cf['calibration_sha256']==N.sha(A/'CALIBRATION_V1.json') and not cf['tune_confirm_inference_started']
    fs=N.load(B/'numeric/actual_v1/FEATURES_FP32_V1.npz');assert c['fit_ids']==fs['fit_ids'].tolist() and c['fit_only']
    counts=dict(traces=0,frame_region_metrics=0,sequence_region_metrics=0,decisions=0);results={}
    for dim in [128,160]:
        mp=B/f'ordinary/actual_v1/MODEL_{dim}_V1.json';m=N.read(mp);assert c['models'][str(dim)]==N.sha(mp)
        residual=m['family']['residual'][str(m['selection']['alpha'])];results[str(dim)]={}
        for role in ['fit','tune','confirm']:
            folder='confirm' if role=='confirm' else 'fit_tune';pack=N.load(R33/f'packs/{folder}_{dim}_v1/{role}.npz')
            ids=pack['ids'].tolist();x32=fs[role+'_x160'][:,:dim];arr=N.load(A/f'{role}_{dim}_ARRAYS_V1.npz');N.eq(arr['ids'],ids)
            q=N.read(A/f'{role}_{dim}_QUALITY_V1.json');assert q['fit_is_resubstitution']==(role=='fit')
            raw={r['target_id']:r for r in N.read(R33/f'packs/{folder}_{dim}_v1/QUALITY_V1.json')['frames']}
            regs=list(raw[ids[0]]['states']['background_Q4']);states=['background_Q4']+[f'upgrade_{i}' for i in range(6)]
            nn=np.array([[raw[i]['states'][states[0]][r]['valid_pixels'] for r in regs] for i in ids])
            ee=np.array([[[raw[i]['states'][s][r]['error_sum'] for r in regs] for s in states] for i in ids]);ai=regs.index('all_valid');rr={}
            reference={}
            for mode in ['fp64','fp32','mixed_int16']:
                x=pack['x'] if mode=='fp64' else x32;cal=c['scales'][str(dim)]
                aux=affine(m['family']['auxiliary'],x,mode,cal['aux'],arr,'aux',role=='fit')
                signed=N.signed(aux,1,np.float64 if mode=='fp64' else F)
                aug=np.concatenate([x,aux,N.parts(aux,1).transpose(0,2,1).reshape(len(x),12)],axis=1)
                resid=affine(residual,aug,mode,cal['residual'],arr,'residual',role=='fit')
                s=signed+(float(m['selection']['shrink']) if mode=='fp64' else F(m['selection']['shrink']))*resid
                if mode=='fp64':N.close(arr[f'results__{mode}__scores'],s)
                else:N.eq(arr[f'results__{mode}__scores'],s);counts['traces']+=2
                for allow,label in [(False,'fixed_six'),(True,'allow_Q4')]:
                    a=s.argmin(axis=1)
                    if allow:a=np.where(s.min(axis=1)>=0,-1,a)
                    N.eq(arr[f'results__{mode}__{label}'],a);counts['decisions']+=len(a)
                    if mode=='fp64':reference[label]=a
                    else:assert q['bridges'][mode][label+'_flips']==int(np.sum(a!=reference[label]))
                    selected=ee[np.arange(len(ids)),a+1];name=mode+'__'+label;tails=[]
                    for i,tid in enumerate(ids):
                        saved=q['frames'][i]['policies'][name];assert saved['action']==a[i]
                        for j,r in enumerate(regs):
                            v=saved['regions'][r];assert v['valid_pixels']==nn[i,j];N.close(v['error_sum'],selected[i,j]);counts['frame_region_metrics']+=1
                        for r in ['all_valid','B','L']:
                            j=regs.index(r)
                            if nn[i,j]>=256 and (selected[i,j]>1.2*ee[i,0,j] or selected[i,j]-ee[i,0,j]>nn[i,j]):tails.append((tid,r))
                    assert tails==[(t['target_id'],t['region']) for t in q['comparisons'][name+'__vs_Q4']['high_support_tail_review']]
                    for seq in ['overall']+sorted(set(pack['groups'])):
                        ix=np.ones(len(ids),bool) if seq=='overall' else pack['groups']==seq
                        for j,r in enumerate(regs):
                            number=int(nn[ix,j].sum());error=float(selected[ix,j].sum());v=q['summaries'][name][seq][r]
                            assert v['valid_pixels']==number;N.close(v['error_sum'],error)
                            if number:N.close(v['pixel_weighted_AEE'],error/number)
                            else:assert v['pixel_weighted_AEE'] is None
                            counts['sequence_region_metrics']+=1
                    rr[name]={'AEE':float(selected[:,ai].sum()/nn[:,ai].sum()),'upgraded':int((a>=0).sum()),'flips_to_FP64':int(np.sum(a!=reference[label])),'tail_items':tails}
            results[str(dim)][role]=rr
    report=dict(schema='flow_r34_ordinary_numeric_independent_audit_v1',status='PASS',auditor_sha256=N.sha(__file__),helper_sha256=N.sha(Path(N.__file__)),
        original_numeric_audit_sha256=N.sha(B/'independent_review/NUMERIC_INDEPENDENT_AUDIT_V1.json'),extension_freeze_sha256=N.sha(B/'numeric/ORDINARY_EXTENSION_FREEZE_V1.json'),
        summary_sha256=N.sha(A/'SUMMARY_V1.json'),counts=counts,results=results,
        scope='Same fixed numeric contract applied to two fit-selected ordinary controls. Fit final-model resubstitution is distinct from nested OOF. No new fit/format/identity choice.',GPU_calls=0)
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'PASS','counts':counts}))

if __name__=='__main__':main()
