#!/usr/bin/env python3
"""Independent NumPy remeasurement of fixed R16 V2 synthetic CPU arrays.

Does not import the implementation, runner, comparison function or model.
Recomputes stated gates; extra real-arithmetic BN reconstruction is diagnostic.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

FREEZE_SHA = 'daad159375734f47dfe1cead72e58a8abdeadd1fbea50723f74533c1213df5da'
CONTINUOUS = ['fc1_input','fc1_output','fc2_input','fc2_output','sn1_input',
              'sn1_prethreshold','sn2_input','sn2_prethreshold','bn1_output',
              'bn2_output','bn1_mean','bn1_population_variance',
              'bn2_mean','bn2_population_variance','branch','residual']

def require(x, msg):
    if not x:
        raise AssertionError(msg)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()

def expected_cases():
    for seed, dtype, hw, mask, scenario in itertools.product(
            (17,29,43), ('float64','float32'), ((9,9),(11,14),(3,4)),
            ('FF','CC','checkerboard'), ('random','constant','low_variance','near_sn1','near_sn2')):
        yield seed, dtype, *hw, mask, scenario, 1
    for dtype, mask in itertools.product(('float64','float32'),('FF','CC','checkerboard')):
        yield 17, dtype, 11, 14, mask, 'random', 2

def cover(h, w, mask):
    cells=[]
    for ty, tx in np.ndindex(mask.shape):
        stop_y, stop_x = min((ty+1)*9,h), min((tx+1)*9,w)
        stride=1 if mask[ty,tx] else 3
        for y in range(ty*9,stop_y,stride):
            for x in range(tx*9,stop_x,stride):
                ey,ex=min(y+stride,stop_y),min(x+stride,stop_x)
                cells.append((y,ey,x,ex,(ey-y)*(ex-x)))
    visited=np.zeros((h,w),dtype=np.int64)
    for y,ey,x,ex,area in cells:
        visited[y:ey,x:ex]+=1
        require(area==(ey-y)*(ex-x),'Cell area')
    require(np.all(visited==1),'Every real pixel exactly once')
    return np.asarray(cells,dtype=np.int64)

def expand(a,cells,h,w):
    out=np.empty((a.shape[0],a.shape[1],h,w,a.shape[-1]),dtype=a.dtype)
    require(a.shape[2:4]==(len(cells),1),'Packed shape')
    for j,(y,ey,x,ex,_) in enumerate(cells):
        out[:,:,y:ey,x:ex,:]=a[:,:,j:j+1,:,:]
    return out

def compare(a,b,tol):
    require(a.shape==b.shape and a.dtype==b.dtype,'Same numeric shape/dtype')
    err=np.abs(a-b)
    bad=(err>tol+tol*np.abs(b))|~np.isfinite(err)
    return {'passed':bool(np.isfinite(a).all() and np.isfinite(b).all() and not bad.any()),
            'bad_elements':int(bad.sum()),'max_abs_error':float(err.max()),'numel':int(a.size)}

def run(root):
    freeze=root/'freeze_v2.json'; require(sha(freeze)==FREEZE_SHA,'Original freeze')
    fr=read(freeze)
    for name,digest in fr['source_sha256'].items(): require(sha(root/name)==digest,'Frozen source '+name)
    require(sha(root/'runtime_environment_sources_v1.json')=='419ee5e0ab8f6f639091db44c5f8c7a873079e419a37e8b80bd5b38e92ef0a29','Same runtime source receipt')
    require(sha(Path(__file__).parent/'CPU_RAW_ARRAY_AUDIT_V1.json')=='c301e86f908e1d57e8328801a056d8ae4159daaca545f452aec16179874cea56','Preserved independent V1 result')
    out=root/'actual_v2'; result=read(out/'results.json')
    require(result['status']=='CPU_MATRIX_COMPLETE' and result['cases_complete']==276,'Complete fixed matrix')
    require(result['freeze_sha256']==FREEZE_SHA,'Result source freeze')
    require(len(result['case_records'])==276,'Unique complete cases')
    failed=[]; rows=Counter(); calls=Counter(); by_group=Counter(); outcomes=[]
    mismatch_checks=0; payloads=0; mathematical_checks=0
    for i,spec in enumerate(expected_cases(),1):
        index=result['case_records'][i-1]; p=out/index['path']
        require(index['index']==i and sha(p)==index['sha256'],'Case receipt binding')
        r=read(p); seed,dtype,h,w,mask_name,scenario,b=spec
        require(tuple(r[k] for k in ('seed','dtype','height','width','mask','scenario','batch'))==spec,'Fixed matrix identity')
        v1=read(root/'actual_v1'/f'case_{i:03d}.json')
        require(r['state_sha256']==v1['state_sha256'],'Unchanged actual synthetic model')
        require(r['BN_mode']=='NATIVE_DENSE_CURRENT','Explicit V2 execution contract')
        for key in v1['array_archive']['payloads']:
            if key in ('u','mask','cells') or key.startswith('state__'):
                require(r['array_archive']['payloads'][key]==v1['array_archive']['payloads'][key],'Unchanged input/model/cell '+key)
        archive=out/r['array_archive']['path'];require(sha(archive)==r['array_archive']['sha256'],'Actual array archive')
        with np.load(archive,allow_pickle=False) as z:
            arrays={k:z[k] for k in z.files}
        require(set(arrays)==set(r['array_archive']['payloads']),'All raw payloads')
        for k,a in arrays.items():
            d=r['array_archive']['payloads'][k]
            require(d['shape']==list(a.shape) and d['dtype']==str(a.dtype) and d['sha256']==array_sha(a),'Raw payload '+k)
            require(np.isfinite(a).all(),'Finite raw '+k);payloads+=1
        mask=np.asarray([[mask_name=='FF' or (mask_name=='checkerboard' and (y+x)%2==0)
                          for x in range((w+8)//9)] for y in range((h+8)//9)],dtype=bool)
        require(np.array_equal(mask,arrays['mask']),'Fixed whole-tile mask')
        cells=cover(h,w,mask);require(np.array_equal(cells,arrays['cells']),'Independent cover')
        require(np.array_equal(cells[:,4],arrays['compact__areas']),'Actual area weights')
        require(r['BN_population']==10*b*h*w and r['barriers']==2,'Population/barriers')
        tol=1e-10 if dtype=='float64' else 2e-5
        band=1e-12 if dtype=='float64' else 1e-6
        methods=('compact','expanded','direct') if mask_name=='FF' else ('compact','expanded')
        for mode in methods:
            actual=r['actual_calls'][mode]; expected=10*b*(len(cells) if mode=='compact' else h*w)
            require(actual['fc1_rows']==actual['fc2_rows']==expected,'Independent FC rows')
            require(actual.get('mlp',0)==(mode!='compact'),'Actual native method')
            require(all(actual[k]==1 for k in ('fc1','fc2','sn1','sn2','bn1','bn2')),'One actual producer and native BN each')
            rows[mode]+=expected;calls[mode]+=1
            for sn in ('sn1','sn2'):
                spikes=arrays[mode+'__'+sn+'_output']; pre=arrays[mode+'__'+sn+'_prethreshold']
                require(np.array_equal(spikes,pre>=0),'Actual native Heaviside sign');mathematical_checks+=1
            require(np.array_equal(arrays[mode+'__residual'], arrays['u']+arrays[mode+'__branch']),'Recorded residual')
        cell_exact={}
        for n in (1,2):
            dense_in=arrays[f'compact__bn{n}_dense_input'];dense_out=arrays[f'compact__bn{n}_dense_output']
            raw_input=arrays[f'compact__bn{n}_input'].transpose(0,1,3,4,2)
            require(np.array_equal(dense_in,raw_input),'Actual BN input view')
            require(np.array_equal(dense_in,expand(arrays[f'compact__fc{n}_output'],cells,h,w)),'Actual dense FC expansion')
            int_dtype=np.uint32 if dtype=='float32' else np.uint64
            gathered=[];equal=True
            for y,ey,x,ex,_ in cells:
                cell=dense_out[:,:,y:ey,x:ex,:]
                point=dense_out[:,:,y:y+1,x:x+1,:]
                equal=bool(np.array_equal(cell.view(int_dtype),np.broadcast_to(point,cell.shape).view(int_dtype))) and equal
                gathered.append(point[:,:,0,0,:])
            packed=np.stack(gathered,axis=2)[:,:,:,None,:]
            require(np.array_equal(packed.view(int_dtype),arrays[f'compact__bn{n}_output'].view(int_dtype)),'Exact top-left all-axis gather')
            require(equal==r['cell_exact'][f'BN{n}_cell_bitwise_equal'],'Cell bitwise gate')
            cell_exact[str(n)]=equal;mathematical_checks+=3
        require(np.array_equal(arrays['compact__bn2_dense_output'],arrays['compact__branch']),'Direct dense BN2 branch')
        require(all(cell_exact.values())==r['cell_exact']['each_cell_BN1_and_BN2_bitwise_equal'],'Combined exact gate')
        resource_expected={'full_domain_BN_elements':10*b*h*w*20,'expand_written_elements':10*b*h*w*20,
                           'BN_output_dense_elements':10*b*h*w*20,'production_BN1_gather_read_elements':10*b*len(cells)*16,
                           'production_BN1_gather_written_elements':10*b*len(cells)*16,'diagnostic_BN2_gather_read_elements':10*b*len(cells)*4,
                           'cell_exact_audit_read_elements':2*10*b*h*w*20}
        require(r['resources']==resource_expected,'Dense BN and diagnostic resource elements')
        # High-precision reconstruction from actual representatives, not an implementation call.
        moments={}
        for n in (1,2):
            z=arrays[f'compact__fc{n}_output'].astype(np.float64)
            weights=cells[:,4][None,None,:,None,None]
            denominator=10*b*h*w
            mu=np.sum(z*weights,axis=(0,1,2,3))/denominator
            var=np.sum((z-mu)**2*weights,axis=(0,1,2,3))/denominator
            actual_mu=arrays[f'compact__bn{n}_mean'].astype(np.float64)
            actual_var=arrays[f'compact__bn{n}_population_variance'].astype(np.float64)
            # Formula check uses the predeclared dtype tolerance, not a new admission gate.
            require(np.all(np.abs(actual_mu-mu)<=tol+tol*np.abs(mu)),'Population mean formula')
            require(np.all(np.abs(actual_var-var)<=tol+tol*np.abs(var)),'Population variance formula')
            moments[str(n)]={'mean_max_abs_to_fp64':float(np.abs(actual_mu-mu).max()),
                             'variance_max_abs_to_fp64':float(np.abs(actual_var-var).max())}
            mathematical_checks+=2
        pairs=[('compact','expanded')]
        if mask_name=='FF':pairs += [('compact','direct'),('expanded','direct')]
        gates={};reason=[]
        for left,right in pairs:
            label=left+'_vs_'+('direct_native' if right=='direct' else right)
            reported=r['comparisons'][label]
            def get(mode,k):
                a=arrays[mode+'__'+k]
                return expand(a,cells,h,w) if mode=='compact' and a.ndim==5 and k not in ('branch','residual') else a
            continuous={k:compare(get(left,k),get(right,k),tol) for k in CONTINUOUS}
            for k,check in continuous.items():
                prior=reported['continuous'][k]
                for field in ('passed','bad_elements','numel','max_abs_error'):
                    require(check[field]==prior[field],f'Independent comparison mismatch {i}/{label}/{k}/{field}')
                    mismatch_checks+=1
                if not check['passed']:reason.append(label+':'+k)
            flip={}
            for sn in ('sn1','sn2'):
                a,bb=get(left,sn+'_output'),get(right,sn+'_output'); bad=a!=bb
                ah,bh=get(left,sn+'_prethreshold'),get(right,sn+'_prethreshold')
                summary={'count':int(bad.sum()),'exact_passed':not bool(bad.any()),
                         'either_in_band':int(((np.abs(ah)<=band)|(np.abs(bh)<=band))[bad].sum()),
                         'both_in_band':int(((np.abs(ah)<=band)&(np.abs(bh)<=band))[bad].sum())}
                for k,v in summary.items():require(v==reported['PSN_flips'][sn][k],f'Independent flip mismatch {i}/{label}/{sn}/{k}');mismatch_checks+=1
                require(np.array_equal(np.argwhere(bad),np.asarray(reported['PSN_flips'][sn]['indices']).reshape(-1,5)),'Flip locations')
                require(np.array_equal(ah[bad],reported['PSN_flips'][sn]['actual_h']) and np.array_equal(bh[bad],reported['PSN_flips'][sn]['reference_h']),'Actual flip margins')
                flip[sn]=summary
                if summary['count']:reason.append(label+':'+sn+'_flip')
            passed=all(v['passed'] for v in continuous.values()) and all(v['exact_passed'] for v in flip.values())
            require(passed==reported['passed'],'Pair gate')
            gates[label]={'passed':passed,'flips':flip,'failed_continuous':[k for k,v in continuous.items() if not v['passed']]}
        passed=all(v['passed'] for v in gates.values()) and all(cell_exact.values()); require(passed==r['passed']==index['passed'],'Case gate')
        if not passed:
            failed.append(i);by_group[(dtype,scenario,mask_name)]+=1
        outcomes.append({'index':i,'seed':seed,'dtype':dtype,'geometry':[h,w],'batch':b,'mask':mask_name,
                         'scenario':scenario,'passed':passed,'comparisons':gates,'moments':moments,'cell_exact':cell_exact,'reasons':reason})
    require(failed==result['failed_cases'],'Original failures preserved')
    require(dict(rows)=={'compact':152260,'expanded':240780,'direct':80260},'Whole-matrix row budgets')
    require(sum(calls.values())==644 and calls['expanded']+calls['direct']==368,'Whole matrix method budgets')
    require(result['admission']==('NOT_ADMITTED' if failed else 'ADMITTED_FINITE_SYNTHETIC_CPU_ONLY'),'Fixed strict gate')
    return {'schema':'flow_r16_independent_cpu_raw_array_audit_v2','status':'PASS_RAW_RECOMPUTATION',
            'source_result_sha256':sha(out/'results.json'),'freeze_sha256':FREEZE_SHA,'auditor_sha256':sha(__file__),
            'new_numerical_model_calls':0,'GPU_used':False,'GT_or_real_W_read':False,
            'cases':276,'logical_MLP_calls_observed':sum(calls.values()),'actual_FC_rows_per_layer_by_path':dict(rows),
            'payloads_verified':payloads,'reported_numeric_fields_recomputed':mismatch_checks,
            'additional_sign_population_formula_cell_checks':mathematical_checks,'report_mismatches':0,
            'strict_admission':result['admission'],'failed_cases':failed,
            'failures_by_dtype_scenario_mask':[dict(dtype=k[0],scenario=k[1],mask=k[2],count=v) for k,v in sorted(by_group.items())],
            'outcomes':outcomes,
            'scope':'Independent V2 raw array gate/coverage/sign/population-moment/cell-gather remeasurement; not a second model run or a cycle/quality measurement.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--implementation',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();require(not a.output.exists(),'New output only')
    result=run(a.implementation)
    with a.output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('outcomes','failed_cases','failures_by_dtype_scenario_mask')},indent=2))
