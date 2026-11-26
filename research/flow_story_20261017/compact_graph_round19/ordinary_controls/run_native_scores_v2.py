#!/usr/bin/env python3
"""Thin R19-only native-cache IO; unchanged frozen R18 descriptor mathematics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import time
sys.dont_write_bytecode=True
import numpy as np
HERE=Path(__file__).resolve().parent
RUN=HERE.parent
DATE=RUN.parent
MATH=DATE/'ordinary_controls_round18/implementation/g0_core.py'
MATH_SHA='01d54124dfecec70cffa9efaa2a8e88579ecb14477ae891d46643b140f15216d'
GRAPH='r19_W271482_all78current_s2b2_R16V2_compact'
H8_SHA='1083d2a3f9b940aee0ebf271cb30b062eeed7bc4bcbf51429d573172573569f5'
STATE_SHA='b721031729a9a5f69d03037be53e62ebf8c1536dad0ad8a8f8e1601fe9fc2798'
OWN=('run_native_scores_v2.py','test_native_scores_v2_cpu.py','PROTOCOL_V1.md')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def require(ok,message):
    if not ok:raise ValueError(message)


def math_module():
    require(sha(MATH)==MATH_SHA,'Unchanged frozen R18 pure descriptor math')
    spec=importlib.util.spec_from_file_location('_r19_original_r18_score_math',MATH)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    require(Path(m.__file__).resolve()==MATH.resolve(),'Actual math module path')
    return m


def validate_admission(result,audit,result_sha,root_freeze_sha,core):
    require(result['schema']=='flow_r19_compact_graph_v1' and result['status']=='COMPACT_GRAPH122_COMPLETE', 'Complete R19 graph only')
    require(result['graph_id']==result['binding']['graph_id']==GRAPH, 'Own graph ID')
    require(audit['schema']=='flow_r19_compact_graph_independent_execution_audit_v1' and
            audit['status']=='PASS' and audit['quality_analysis_allowed'] is True and
            audit['admission']=='ADMITTED_NEW_COMPACT_GRAPH120_ENDPOINTS_ONLY',
            'Complete independent own-graph execution audit')
    require(audit['source_result_sha256']==result_sha and
            audit['source_freeze_sha256']==result['binding']['source_freeze_sha256']==root_freeze_sha,'Bound actual graph/source evidence')
    want=dict(full_forwards=122,native=8,compact=114,expanded=0,quality_endpoints=120,determinism_replays=2,
              FC1_rows=67021520,FC2_rows=67021520,BN_calls=9516,archives=122,head_payloads=488,capture_payloads=98,
              optimizer_steps=0,calibration_forwards=0)
    require(all(audit['counts'][k]==v for k,v in want.items()),'Complete fixed122 audit population')
    require(result['full_forward_count']==result['full_forward_attempts']==122 and
            result['initial_allcurrent_state_sha256']==result['final_state_sha256']==STATE_SHA,'Actual state and count closure')
    require(all(result[k] is True for k in ('all_states_unchanged','all_hooks_removed','original_modules_restored')),'Actual closure flags')
    require(result['binding']['h8_sha256']==H8_SHA and result['binding']['BN_mode']=='NATIVE_DENSE_CURRENT','Fixed H8/native BN')
    checks=audit['numerical_checks']
    require(len(checks)==10 and all(x['pass'] is True for x in checks) and
            sum(x['kind']=='allfine_vs_native' for x in checks)==8 and
            sum(x['kind']=='CC_exact_replay' for x in checks)==2,'All8 FF and two replay gates')
    expected=[]
    for i,tid in enumerate(core.IDS):
        for arm in ('native','FF','CC')+tuple('fine_%d_%d'%p for p in core.TILES):
            expected.append((tid,arm,True))
            if i==0 and arm=='CC':expected.extend((tid,'CC_repeat%d'%j,False) for j in (1,2))
    records=result['forward_records']
    require(len(records)==122 and [(r['target_id'],r['arm'],r['quality_endpoint']) for r in records]==expected,'Own fixed identities/endpoint split')
    require(len(audit['predictions'])==122,'Full audit archive binding')
    for r,a in zip(records,audit['predictions']):
        require(all(r[k]==a[k] for k in ('forward_id','target_id','arm','execution','quality_endpoint','endpoint_arm')) and
                r['prediction_file_sha256']==a['file_sha256'],'Each actual endpoint bound to audit')
    natives=[r for r in records if r['arm']=='native']
    require([r['target_id'] for r in natives]==list(core.IDS),'Exact8 new-native inputs')
    by={r['target_id']:r for r in natives}
    require(all(r['tensor_records']['u']['sha256']==by[r['target_id']]['tensor_records']['u']['sha256'] for r in records),'Exact predecision prefix across actions')
    return natives


def load_native(path,record,core):
    require(Path(path).name==record['prediction_file'] and sha(path)==record['prediction_file_sha256'],'Own actual native archive')
    with np.load(path,allow_pickle=False) as z:
        require(set(z.files)=={'head0','head1','head2','head3','u','branch','sn1','sn2'},'Own native capture schema')
        arrays={name:z[name] for name in ('u','sn1')}
    for name,x in arrays.items():
        meta=record['tensor_records'][name]
        require(x.dtype==np.float32 and x.shape==(10,1,30,40,384) and
                list(x.shape)==meta['shape'] and meta['dtype']=='torch.float32' and
                core.payload_sha(x)==meta['sha256'] and np.isfinite(x).all(),'Own actual raw tensor '+name)
    require(((arrays['sn1']==0)|(arrays['sn1']==1)).all(),'Own original binary PSN')
    return arrays['u'],arrays['sn1']


def metadata_preflight(result_path,audit_path,result_sha,audit_sha):
    """Complete production admission; reads JSON/source bytes, never NPZ/GT/loss."""
    require(sha(result_path)==result_sha and sha(audit_path)==audit_sha,'Actual execution/audit bytes')
    exfreeze=RUN/'independent_execution_audit/PRE_RESULT_FREEZE_V1.json'
    rootfreeze=RUN/'source_freeze_v1.json';auditor=RUN/'independent_execution_audit/audit_compact.py'
    ex=read(exfreeze);audit=read(audit_path);result=read(result_path)
    for path,digest in ex['source_sha256'].items():require(sha(path)==digest,'Frozen independent execution auditor source')
    require(audit['auditor_sha256']==sha(auditor) and audit['pre_result_freeze_sha256']==sha(exfreeze),'Actual frozen auditor identity')
    # The frozen auditor checks result and binding graph_id but does not export
    # its own top-level graph_id. Bind its exact result hash and R19 admission.
    core=math_module();natives=validate_admission(result,audit,result_sha,sha(rootfreeze),core)
    return core,natives,sha(rootfreeze)


def main():
    p=argparse.ArgumentParser()
    for key in ('result','audit','freeze','review'):p.add_argument('--'+key,type=Path,required=True);p.add_argument('--'+key+'-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--wall-seconds',type=int,default=300)
    a=p.parse_args();require(not a.output.exists() and 0<a.wall_seconds<=600,'Fresh bounded CPU output')
    for key in ('result','audit','freeze','review'):require(sha(getattr(a,key))==getattr(a,key+'_sha256'),'Actual '+key+' bytes')
    own={name:sha(HERE/name) for name in OWN};freeze=read(a.freeze);review=read(a.review)
    require(freeze['schema']=='flow_r19_native_score_freeze_v2' and freeze['source_sha256']==own and
            freeze['math_sha256']==MATH_SHA and freeze['cpu_sha256']==sha(HERE/'CPU_TESTS_V2.json'),'Own IO/math/CPU freeze')
    require(review['schema']=='flow_r19_native_score_source_review_v2' and review['status']=='SOURCE_REVIEW_PASS' and
            review['source_sha256']==own and review['math_sha256']==MATH_SHA,'Independent thin IO source review')
    core,natives,root_freeze_sha=metadata_preflight(a.result,a.audit,a.result_sha256,a.audit_sha256)
    def stop(sig,frame):raise TimeoutError('Bounded native score run')
    signal.signal(signal.SIGALRM,stop);signal.alarm(a.wall_seconds);start=time.monotonic()
    a.output.mkdir(parents=True)
    out=dict(schema='flow_r19_g0_cached_ordinary_v1',IO_revision=2,graph_id=GRAPH,status='RUNNING',source_sha256=own,
             math_sha256=MATH_SHA,freeze_sha256=a.freeze_sha256,independent_source_review_sha256=a.review_sha256,
             R19_result_sha256=a.result_sha256,R19_execution_audit_sha256=a.audit_sha256,root_source_freeze_sha256=root_freeze_sha,
             h8_sha256=H8_SHA,frames=[],archive_reads=[],score_names=list(core.SCORES),tiles=[list(t) for t in core.TILES],
             precision='Unchanged R18: FP32 source, float64 u math, integer binary PSN counts then float64 division',
             model_forwards=0,PSN_calls=0,FC_calls=0,GPU_calls=0,optimizer_steps=0,GT_loaded=False,quality_loaded=False,
             scope='Own compact graph native inputs; H8 exposed development; no R18 quality transfer',
             policy_count=20,score_policy_count=7,runtime=dict(numpy=np.__version__,python=sys.version))
    dest=a.output/'scores.json'
    def save():dest.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    save()
    try:
        for r in natives:
            require(Path(r['prediction_file']).name==r['prediction_file'],'Archive basename only')
            path=a.result.parent/r['prediction_file'];u,s=load_native(path,r,core)
            frame=core.score_frame(u,s);frame.update(target_id=r['target_id'],input_archive=r['prediction_file'],
                  input_archive_sha256=r['prediction_file_sha256'],input_tensor_records={k:r['tensor_records'][k] for k in ('u','sn1')})
            out['frames'].append(frame)
            out['archive_reads'].append(dict(path=str(path),sha256=r['prediction_file_sha256'],archive_bytes=path.stat().st_size,
                 payload_names_loaded=['u','sn1'],uncompressed_payload_bytes=u.nbytes+s.nbytes))
            save();del u,s
        require(len(out['frames'])==8,'All new-native frames')
        out.update(status='G0_CPU_COMPLETE',score_values=672,score_selections=56,policy_frame_cells=160,
                   selector_score_comparisons=616,total_loaded_payload_bytes=sum(x['uncompressed_payload_bytes'] for x in out['archive_reads']),
                   wall_seconds=time.monotonic()-start);save()
    except BaseException as error:out.update(status='FAILED',error=repr(error));save();raise
    finally:signal.alarm(0)
    print(json.dumps(dict(status=out['status'],graph_id=GRAPH,output_sha256=sha(dest),wall_seconds=out['wall_seconds'])))


if __name__=='__main__':main()
