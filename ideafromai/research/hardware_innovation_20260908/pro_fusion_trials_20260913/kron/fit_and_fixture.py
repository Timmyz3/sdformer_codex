"""One fixed layout, 1/2 terms; real inputs; no training or parameter sweep."""
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT/'open_fusion_execution/breadth_20260912/algorithm/hardware_exports/dense'

def rne24(x, shift=15):
    x=np.asarray(x,dtype=np.int64)
    if shift:
        q=x >> shift; rem=x-(q << shift); half=1 << (shift-1)
        x=q+((rem>half)|((rem==half)&((q&1)!=0)))
    return np.clip(x,-(1<<23),(1<<23)-1).astype(np.int64)

def diff(a,b):
    d=a.astype(np.float64)-b
    return dict(elements=int(d.size),different=int(np.count_nonzero(d)),
        max_abs=float(abs(d).max()),rmse=float(np.sqrt(np.mean(d*d))),
        nrmse=float(np.linalg.norm(d)/np.linalg.norm(b)))

def write_ints(name,x):
    (HERE/name).write_text('\n'.join(str(int(i)) for i in np.asarray(x).ravel())+'\n')

def kron_fit(w,terms):
    # W[a*24+b,u*4+v] = sum_r A[r,a,u] B[r,b,v].
    z=w.reshape(4,24,6,4).transpose(0,2,1,3).reshape(24,96)
    u,s,v=np.linalg.svd(z,full_matrices=False)
    a=(u[:,:terms]*np.sqrt(s[:terms])).T.reshape(terms,4,6)
    b=(v[:terms]*np.sqrt(s[:terms,None])).reshape(terms,24,4)
    return a,b,sum(np.kron(x,y) for x,y in zip(a,b))

def main():
    q=dict(np.load(SRC/'deployed_constants.npz'))
    data=dict(np.load(SRC/'000_zurich_city_09_a_0001.npz'))
    geo=json.loads(str(data['window_geometry_json']))
    inputs=[];checks={};windows=[]
    for label in ('corner','interior'):
        g=geo[label]; a=data[label+'_updated_I24'].astype(np.int64)
        y0=2*g['output_origin'][0]-g['gate_origin'][0]
        x0=2*g['output_origin'][1]-g['gate_origin'][1]
        real=a[:,:,y0:y0+8:2,x0:x0+8:2].transpose(1,0,2,3).reshape(96,-1)
        latent=rne24(q['U_ped_q16'].astype(np.int64)@real,int(q['U_ped_exponent']))
        ped=rne24(rne24(q['V_ped_q16'].astype(np.int64)@latent,15)+q['PED_bias_q24'][:,None],0)
        captured=data[label+'_continuous_q24'].transpose(1,0,2,3).reshape(96,-1)
        checks[label]=diff(ped,captured); assert checks[label]['different']==0
        inputs.append(latent);windows.append(label)
    x=np.concatenate(inputs,axis=1); wq=q['V_ped_q16'].astype(np.int64);w=wq/32768.
    ref=rne24(wq@x,15);bias=q['PED_bias_q24'].astype(np.int64)
    uf,sf,vf=np.linalg.svd(w,full_matrices=False)
    rows=[]; arrays={'input_q24':x.T,'original_wq15':wq,'bias_q24':bias,'original_output_q24':rne24(ref+bias[:,None],0).T}
    for terms in (1,2):
        a,b,approx=kron_fit(w,terms)
        # Fixed split eA=7, eB=8; no rounding or clipping at contraction boundary.
        aq=np.rint(a*128).astype(np.int64);bq=np.rint(b*256).astype(np.int64)
        expanded=sum(np.kron(aa,bb) for aa,bb in zip(aq,bq))
        assert abs(expanded).max() <= 32767
        z=np.einsum('rau,uvn->ravn',aq,x.reshape(6,4,-1),optimize=False)
        accum=np.einsum('rbv,ravn->abn',bq,z,optimize=False).reshape(96,-1)
        assert np.array_equal(accum,expanded@x)
        out=rne24(rne24(accum,15)+bias[:,None],0)
        lr=(uf[:,:terms]*sf[:terms])@vf[:terms]
        la=uf[:,:terms]*np.sqrt(sf[:terms]);lb=np.sqrt(sf[:terms,None])*vf[:terms]
        lq=np.rint(la*128).astype(np.int64)@np.rint(lb*256).astype(np.int64)
        for name,ap,apq in [('kron',approx,expanded),('lowrank',lr,lq)]:
            row=dict(name=name,terms=terms,parameters=120*terms,
                weight_float=diff(ap,w),weight_quantized=diff(apq/32768.,w),
                latent_float=diff(ap@x,w@x),post_V_rne=diff(rne24(apq@x),ref),
                post_bias=diff(rne24(rne24(apq@x)+bias[:,None],0),arrays['original_output_q24'].T))
            rows.append(row)
        arrays[f'k{terms}_a_q7']=aq;arrays[f'k{terms}_b_q8']=bq
        arrays[f'k{terms}_expanded_wq15']=expanded;arrays[f'k{terms}_output_q24']=out.T
        assert abs(z).max()<2**47 and abs(accum).max()<2**63
        rows[-2]['intermediate_max_abs']=int(abs(z).max())
        rows[-2]['accumulator_max_abs']=int(abs(accum).max())
    np.savez(HERE/'fixture.npz',**arrays)
    write_ints('inputs.txt',x.T);write_ints('bias.txt',bias)
    for mode,wgt in [('original',wq),('expanded_k1',arrays['k1_expanded_wq15']),('expanded_k2',arrays['k2_expanded_wq15'])]:
        # physical128-bit CR word: eight adjacent output coefficients for one input.
        packed=wgt.reshape(12,8,24).transpose(0,2,1).reshape(-1,8)
        write_ints(mode+'_coeff.txt',packed)
        key='original_output_q24' if mode=='original' else mode.replace('expanded_','')+'_output_q24'
        write_ints(mode+'_gold.txt',arrays[key])
    for n in (1,2):
        words=[]
        for a,b in zip(arrays[f'k{n}_a_q7'],arrays[f'k{n}_b_q8']):
            words.extend(np.pad(a.T,((0,0),(0,4)))) # six A words, four coefs each.
            words.extend(b.reshape(3,8,4).transpose(0,2,1).reshape(12,8))
        write_ints(f'kron{n}_coeff.txt',np.array(words))
        write_ints(f'kron{n}_gold.txt',arrays[f'k{n}_output_q24'])
    profile=json.load((ROOT/'open_fusion_execution/major_operator_fusions_20260913/root_owned/profile.json').open())
    target=[r for r in profile['rows'] if r.get('left_shape')==[96,24] and 'patch_embed' in r['module']]
    result=dict(source=str(SRC),frame=str(data['frame_name']),windows=windows,
        vectors=int(x.shape[1]),matrix_shape=list(w.shape),source_replay=checks,
        input_range=[int(x.min()),int(x.max())],input_nonzero_fraction=float(np.count_nonzero(x)/x.size),
        layout={'output':[4,24],'input':[6,4],'formula':'W[a*24+b,u*4+v] = sum_r A[r,a,u]*B[r,b,v]'},
        rounding='Existing U RNE16/sat24 retained. No intermediate A rounding; final V RNE15/sat24 then original bias sat24. A q7, B q8.',
        profile=target,results=rows,AEE_evaluated=False,parameter_sweep=False)
    (HERE/'fit_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
