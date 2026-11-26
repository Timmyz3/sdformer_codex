#!/usr/bin/env python3
"""Label-free R18 CPU cache runner; cannot load model/GT/quality as a score input."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
sys.dont_write_bytecode=True
import numpy as np
import g0_core as core

HERE=Path(__file__).resolve().parent
R17=HERE.parents[1]/'dense_reference_round17'
RESULT_SHA='0b4ef2744c9da8fd3513b65141b4133f381cf267db68083e4bb854165ae07a2e'
AUDIT_SHA='6ed07d00330df7824e4f8cc754a1c735bded4b3bf5bf89565f9114506594276f'
H8_SHA='1083d2a3f9b940aee0ebf271cb30b062eeed7bc4bcbf51429d573172573569f5'
OWN=('g0_core.py','run_g0.py','test_g0_cpu.py','PROTOCOL_V1.md')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def validate_contract(result,audit):
    core.require(result['schema']=='flow_r17_dense_reference_v1' and result['status']=='DENSE_REFERENCE120_COMPLETE', 'Complete R17')
    core.require(audit['schema']=='flow_r17_dense_reference_independent_execution_audit_v1' and audit['status']=='PASS', 'Independent R17 execution PASS')
    core.require(audit['source_result_sha256']==RESULT_SHA and audit['counts']['full_forwards']==120 and
                 audit['counts']['archives']==120 and audit['counts']['capture_payloads']==64, 'Complete actual source audit')
    core.require(result['full_forward_count']==result['full_forward_attempts']==120 and
                 result['binding']['h8_sha256']==H8_SHA and result['binding']['BN_mode']=='REFERENCE_DENSE_CURRENT', 'Fixed H8/graph')
    for key in ('all_states_unchanged','all_hooks_removed','original_modules_restored'):
        core.require(result[key] is True, 'Source closure '+key)
    records=result['forward_records']
    arms=('native','FF','CC')+tuple('fine_%d_%d'%p for p in core.TILES)
    core.require(len(records)==120 and [(r['target_id'],r['arm']) for r in records]==[(t,a) for t in core.IDS for a in arms], 'All120 fixed identities/order')
    for r in records:
        core.require(r['tensor_records']['u']['sha256']==records[(r['forward_id']-1)//15*15]['tensor_records']['u']['sha256'], 'Exact common u')
    return [r for r in records if r['arm']=='native']


def load_capture(path,record):
    core.require(Path(path).name==record['prediction_file'] and sha(path)==record['prediction_file_sha256'], 'Actual source archive hash')
    # np.load(NPZ) is lazy. No flow, branch, PSN2, GT or metric payload is loaded.
    with np.load(path,allow_pickle=False) as archive:
        core.require(set(archive.files)=={'head0','head1','head2','head3','u','branch','sn1','sn2'}, 'Native capture archive schema')
        u=archive['u']; spikes=archive['sn1']
    for name,x in [('u',u),('sn1',spikes)]:
        descriptor=record['tensor_records'][name]
        core.require(x.dtype==np.float32 and x.shape==(10,1,30,40,384) and
                     list(x.shape)==descriptor['shape'] and core.payload_sha(x)==descriptor['sha256'], 'Actual capture tensor '+name)
        core.require(np.isfinite(x).all(), 'Finite complete capture including tails')
    core.require(((spikes==0)|(spikes==1)).all(), 'Binary complete original PSN capture')
    return u,spikes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--freeze-sha256',required=True)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--review-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wall-seconds',type=int,default=300)
    a=parser.parse_args()
    core.require(not a.output.exists() and 0<a.wall_seconds<=600, 'Fresh output and finite CPU budget')
    core.require(sha(a.freeze)==a.freeze_sha256 and sha(a.review)==a.review_sha256, 'Freeze/review bytes')
    freeze,review=read(a.freeze),read(a.review)
    sources={name:sha(HERE/name) for name in OWN}
    core.require(freeze['schema']=='flow_r18_g0_freeze_v1' and freeze['source_sha256']==sources and
                 freeze['cpu_receipt_sha256']==sha(HERE/'CPU_TESTS_V1.json'), 'Source and CPU freeze')
    core.require(review['schema']=='flow_r18_g0_source_review_v1' and review['status']=='SOURCE_REVIEW_PASS' and
                 review['source_sha256']==sources, 'Independent source review')
    source=R17/'actual_v1/run_v1/results.json'; audit_path=R17/'independent_execution_audit/ACTUAL_EXECUTION_AUDIT_V1.json'
    core.require(sha(source)==RESULT_SHA and sha(audit_path)==AUDIT_SHA, 'Pinned R17 actual execution evidence')
    result=read(source); records=validate_contract(result,read(audit_path))
    a.output.mkdir(parents=True)
    def expired(signum,frame): raise TimeoutError('Bounded CPU score run expired')
    signal.signal(signal.SIGALRM,expired); signal.alarm(a.wall_seconds)
    start=time.monotonic()
    output=dict(schema='flow_r18_g0_cached_ordinary_v1',status='RUNNING',source_sha256=sources,
                freeze_sha256=a.freeze_sha256,independent_source_review_sha256=a.review_sha256,
                R17_result_sha256=RESULT_SHA,R17_execution_audit_sha256=AUDIT_SHA,h8_sha256=H8_SHA,
                score_names=list(core.SCORES),tiles=[list(p) for p in core.TILES],frames=[],
                precision='FP32 original inputs; all u arithmetic/reductions float64; binary sn1 int64 counts then float64 ratios',
                PSN_semantics='cached full-native PSN1; not an exact reproduction of old per-tile FP32 PSN calls',
                model_forwards=0,PSN_calls=0,FC_calls=0,GPU_calls=0,optimizer_steps=0,GT_loaded=False,quality_loaded=False,
                scope='All H8 exposed development; no flow-specific superiority or hardware-cycle claim',
                runtime=dict(python=sys.version,numpy=np.__version__,cpu_only=True),
                archive_reads=[],policy_count=20,score_policy_count=7)
    dest=a.output/'scores.json'
    def save(): dest.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    save()
    try:
        for record in records:
            path=source.parent/record['prediction_file']
            core.require(path.parent==source.parent and Path(record['prediction_file']).name==record['prediction_file'], 'Basename archive only')
            u,spikes=load_capture(path,record)
            frame=core.score_frame(u,spikes)
            frame.update(target_id=record['target_id'],input_archive=record['prediction_file'],
                         input_archive_sha256=record['prediction_file_sha256'],
                         input_tensor_records={k:record['tensor_records'][k] for k in ('u','sn1')})
            output['frames'].append(frame)
            output['archive_reads'].append(dict(path=str(path),sha256=sha(path),archive_bytes=path.stat().st_size,
                      payload_names_loaded=['u','sn1'],uncompressed_payload_bytes=u.nbytes+spikes.nbytes,
                      payload_elements_each=u.size,
                      note='Archive bytes hashed twice for integrity; NPZ extraction is additional offline I/O, not runtime DRAM traffic'))
            save()
            del u,spikes
        core.require(len(output['frames'])==8, 'Complete fixed H8')
        output.update(status='G0_CPU_COMPLETE',score_values=8*12*7,score_selections=8*7,
                      policy_frame_cells=8*20,selector_score_comparisons=8*7*11,
                      total_loaded_payload_bytes=sum(x['uncompressed_payload_bytes'] for x in output['archive_reads']),
                      wall_seconds=time.monotonic()-start)
        save()
    except BaseException as error:
        output.update(status='FAILED',error=repr(error),wall_seconds=time.monotonic()-start);save();raise
    finally: signal.alarm(0)
    print(json.dumps({'status':output['status'],'frames':8,'scores':672,'selections':56,
                      'output_sha256':sha(dest),'wall_seconds':output['wall_seconds']}))


if __name__=='__main__':main()
