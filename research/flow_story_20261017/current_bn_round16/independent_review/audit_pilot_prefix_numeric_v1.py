#!/usr/bin/env python3
"""Independent NumPy audit of stopped five-forward pilot; no GT or quality.

Reads only result metadata, source identities, and actual branch/spike/head
arrays. Does not import a model, rerun inference, or measure optical-flow AEE.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
R16=HERE.parent
EXPECTED=[('native','native'),('FF','compact'),('FF','expanded'),('CC','compact'),('CC','expanded')]
TARGET_ID='zurich_city_07_a_0095'
RUNNER_SHA='eed9a9467304cf141818d914cb4f72163fd24d3b307c8aa9db8f051bcee23cb0'
CORE_SHA='e874e976ce4833a0638092ca78612e8224139eb1cc3b49d9ce0098488a3bb731'
ADMISSION_SHA='533090a30fc541db42d2f9d0e53b16e337bd0d708b005afdc7ff48c9c58ec48d'

def require(ok,msg):
    if not ok:raise AssertionError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def payload(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def compare(a,b):
    require(a.shape==b.shape and a.dtype==b.dtype,'Same shape/dtype')
    require(np.isfinite(a).all() and np.isfinite(b).all(),'Finite arrays')
    d=np.abs(a.astype(np.float64)-b.astype(np.float64));bad=d>2e-5+2e-5*np.abs(b.astype(np.float64))
    return {'pass':not bool(bad.any()),'max_abs':float(d.max()),'mean_abs':float(d.mean()),
            'outside_tolerance':int(bad.sum()),'different_elements':int((d!=0).sum()),
            'elements':int(d.size),'atol':2e-5,'rtol':2e-5}

def audit(root):
    require(sha(R16/'pilot/run_spatial_pilot.py')==RUNNER_SHA,'Final frozen runner')
    require(sha(R16/'implementation/native_dense_bn_v2.py')==CORE_SHA,'Final V2 primitive')
    require(sha(HERE/'PILOT123_ADMISSION_V1.json')==ADMISSION_SHA,'Only finite123 admission')
    path=root/'results.json';r=read(path)
    require(r['schema']=='flow_r16_spatial_pilot_v1' and r['status']=='INCOMPLETE_OR_NUMERICAL_FAILURE','Actual stopped prefix')
    require(r['full_forward_count']==r['full_forward_attempts']==5,'Exactly five actual forwards')
    require(r['error']=='Predeclared numerical contract failed; no action-value interpretation','Original fail-fast reason')
    require(r['optimizer_steps']==r['calibration_forwards']==0 and r['training_performed'] is False,'No training/calibration')
    b=r['binding'];require(b['plan']=='pilot123' and b['BN_mode']=='NATIVE_DENSE_CURRENT','Original declared pilot')
    require(b['source_freeze_sha256']==sha(R16/'pilot/source_freeze_v1.json'),'Original root source freeze')
    require(b['independent_admission_sha256']==ADMISSION_SHA and b['primitive_sha256']==CORE_SHA,'Original source admission')
    require(len(r['forward_records'])==5 and len(r['numerical_checks'])==3,'Complete stopped prefix records')
    arrays=[];payload_count=0;fc1=fc2=bn=0;artifacts=[]
    for i,(record,expected) in enumerate(zip(r['forward_records'],EXPECTED),1):
        require(record['forward_id']==i and record['target_id']==TARGET_ID and tuple(record[k] for k in ('arm','execution'))==expected,'Fixed first-H8 prefix')
        relative=Path(record['prediction_file']);require(len(relative.parts)==1 and relative.suffix=='.npz','Safe artifact')
        artifact=root/relative;require(sha(artifact)==record['prediction_file_sha256'],'Actual archive SHA')
        with np.load(artifact,allow_pickle=False) as z:data={k:z[k] for k in z.files}
        require(set(data)=={'u','branch','sn1','sn2','head0','head1','head2','head3'},'All eight raw payloads')
        for key,value in data.items():
            require(value.dtype==np.float32 and np.isfinite(value).all(),'Finite actual FP32 arrays')
            meta=record['head_records'][int(key[-1])] if key.startswith('head') else record['tensor_records'][key]
            require(list(value.shape)==meta['shape'] and payload(value)==meta['sha256'],'Actual raw payload SHA/shape')
            payload_count+=1
        require(data['u'].shape==data['branch'].shape==data['sn1'].shape==(10,1,30,40,384),'Actual complete-T target')
        require(data['sn2'].shape==(10,1,30,40,1536),'Actual full hidden width')
        require(all(data[f'head{j}'].shape==(1,2,480,640) for j in range(4)),'Four complete output heads')
        for sn in ('sn1','sn2'):require(np.isin(data[sn],(0.,1.)).all(),'Actual binary PSN output')
        require(record['state_and_BN_unchanged'] and record['original_modules_restored_and_hooks_removed'],'Recorded per-forward closure')
        require(len(record['native_BN_calls'])==78 and all(v==1 for v in record['native_BN_calls'].values()),'Recorded78 BN calls')
        expected_rows=547400 if i==4 else 558000
        require(record['summed_fc1_rows']==record['summed_fc2_rows']==expected_rows,'Exact prefix FC rows')
        fc1+=record['summed_fc1_rows'];fc2+=record['summed_fc2_rows'];bn+=sum(record['native_BN_calls'].values())
        arrays.append(data);artifacts.append({'forward_id':i,'file':relative.name,'sha256':sha(artifact)})
    require(fc1==fc2==2779400 and bn==390 and payload_count==40,'Complete prefix counts')
    require(all(np.array_equal(d['u'].view(np.uint32),arrays[0]['u'].view(np.uint32)) for d in arrays),'All actions exact same u')
    recomputed=[];numeric_fields=0
    for (actual,reference),old in zip(((1,0),(1,2),(3,4)),r['numerical_checks']):
        a,b=arrays[actual],arrays[reference]
        continuous={k:compare(a[k],b[k]) for k in ('branch','head0','head1','head2','head3')}
        flips={sn:int(np.count_nonzero(a[sn]!=b[sn])) for sn in ('sn1','sn2')}
        passed=all(v['pass'] for v in continuous.values()) and not any(flips.values())
        require(passed==old['pass'] and flips==old['PSN_flips'] and old['same_u_exact'],'Original numerical decisions')
        for key,c in continuous.items():
            for field,value in c.items():
                reported=old['continuous'][key][field]
                if field=='mean_abs':require(math.isclose(value,reported,abs_tol=1e-14,rel_tol=1e-13),'Independent mean reduction')
                else:require(value==reported,'Independent elementwise numeric field')
                numeric_fields+=1
        recomputed.append({'actual_forward':actual+1,'reference_forward':reference+1,'kind':old['kind'],'arm':old['arm'],
                           'pass':passed,'same_u_exact':True,'continuous':continuous,'PSN_flips':flips})
    require([x['pass'] for x in recomputed]==[True,True,False],'Original numerical stop reproduced')
    return {'schema':'flow_r16_independent_prefix_raw_numeric_audit_v1','status':'PASS_PREFIX_NUMERICAL_RECOMPUTATION',
            'admission':'NOT_ADMITTED_NUMERICAL_FAILURE','quality_analysis_allowed':False,
            'source_result_sha256':sha(path),'source_freeze_sha256':sha(R16/'pilot/source_freeze_v1.json'),
            'auditor_sha256':sha(__file__),'full_forwards':5,'new_forward_calls':0,'raw_payloads':40,
            'numeric_fields_recomputed':numeric_fields,'report_mismatches':0,
            'recorded_FC1_rows':fc1,'recorded_FC2_rows':fc2,'recorded_BN_calls':bn,
            'artifacts':artifacts,'numerical_checks':recomputed,
            'GT_or_mask_arrays_read':False,'flow_quality_metrics_computed':False,'GPU_used':False,
            'scope':'Independent existing-array numeric audit only; full source/state/execution reconstruction remains the separate hardware reviewer audit. No AEE, spatial action-value, predictor, or hardware-benefit claim.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'New audit output only');result=audit(a.result_dir)
    with a.output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('status','admission','quality_analysis_allowed','full_forwards','raw_payloads','numeric_fields_recomputed','report_mismatches')},indent=2))
