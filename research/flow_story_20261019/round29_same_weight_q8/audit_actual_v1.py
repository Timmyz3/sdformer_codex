#!/usr/bin/env python3
"""Independent thin audit: same Q4 final W, Q8-only evaluation, no GT values."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
R27=HERE.parent/'round27'
R28=HERE.parent/'round28'
RUNNER_SHA='6b730696940018d40412054cbba1c76434d1b1986342d2b0e4e62fb037ab37e3'
MEMBERS_SHA='40eba2514b56fd33a910fda4e37d4f86e80431ffe8149efc5f79c18827991e7b'

def require(ok,message):
    if not ok:raise ValueError(message)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def module(p,name,expected):
    require(sha(p)==expected,'Fixed audit/helper source '+str(p))
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def child(folder,name):
    p=folder/name;require(p.resolve().is_relative_to(folder.resolve()),'Artifact within actual run');return p

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();require(not a.output.exists(),'Fresh audit result');torch.set_num_threads(2)
    old=module(R27/'audit_actual_recovery.py','r29_old_audit','14d39dc77e890dca215b17810d163609aed11e6a967e784f0fe4fd211bf982d0')
    q4=module(R28/'audit_actual_recovery_v1.py','r29_q4_audit','eb433542a13b6f2df185b503b922ab98fe648d9e69c9d51b956eb5e8654df71b')
    runner=module(HERE/'implementation/run_same_weight_q8.py','r29_runtime',RUNNER_SHA)
    rpath=a.run/'results.json';r=read(rpath);auditpath=R28/'actual_execution_audit_v1.json';prior=read(auditpath)
    require(prior['status']=='PASS' and prior['schema']=='flow_r28_independent_actual_execution_audit_v1','Actual completed Q4 source audit')
    require(prior['auditor_sha256']==runner.AUDITOR_SHA and prior['counts']==dict(optimizer_updates=512,quality_predictions=128,quality_archives=128,formal_model_forwards=1155),'Fixed source audit identity and execution counts')
    weights=R28/'actual_v1/formal_Q4_v2/recovered_step512.pt'
    require(sha(weights)==prior['arms']['Q4']['checkpoint_sha256'],'Independently reloaded actual Q4 source checkpoint bytes')
    cp=torch.load(weights,map_location='cpu',weights_only=True);state=cp['model_state_dict'];digest=old.state_digest(state)
    require(cp['arm']=='Q4' and cp['completed_optimizer_steps']==512 and len(state)==477 and digest==cp['final_state_sha256'],'Actual unique Q4 final weights')
    psns={'PSN::'+k.removesuffix('.weight') for k in state if k.endswith('.spiking_neuron.weight')};require(len(psns)==105,'Original105 PSN names')
    del cp,state
    require(r['schema']==runner.SCHEMA and r['graph_id']==runner.GRAPH and r['status']=='SAME_Q4_WEIGHT_Q8_EVALUATION_COMPLETE','Complete same-W graph')
    require(r['arm']=='Q8' and r['state_label']=='Q8_on_recovered_Q4','Single fixed Q8 mode')
    require(r['model_forwards']==r['expected_model_forwards']==67 and r['completed_optimizer_steps']==0 and r['wall_cap_minutes']==20,'67 forwards and zero updates within fixed cap')
    for k in ('training_performed','teacher_constructed','optimizer_constructed','optimizer_restored','development_GT_read','quality_computed','quality_selection'):
        require(r[k] is False,'No new '+k)
    require(not r['runtime']['AMP'] and not r['runtime']['TF32'],'Original numerical execution settings')
    require(r['sources']==dict(runner=RUNNER_SHA,smoke_helpers=runner.SMOKE_SHA,evaluation_helpers=runner.EVAL_SHA,core=runner.CORE_SHA,members=MEMBERS_SHA,R28_auditor=runner.AUDITOR_SHA),'Exact executed source contract')
    for f,h in (('run_smoke_v2.py',runner.SMOKE_SHA),('run_recovery_v2.py',runner.EVAL_SHA),('recovery_integer_core.py',runner.CORE_SHA)):
        require(sha(R28/'implementation'/f)==h,'Actual frozen helper bytes')
    source=r['source_Q4_checkpoint']
    require(source['sha256']==sha(weights) and source['state_sha256']==digest and source['audit_sha256']==sha(auditpath),'Same audited Q4 weights and audit')
    require(source['source_arm']=='Q4' and source['source_updates']==512 and source['source_result_sha256']==prior['source_results']['Q4'],'Same Q4 provenance')
    require(r['state_sha256']==r['initial_state_sha256']==r['final_state_sha256']==digest and r['original_graph_state_before_final_reload']==runner.ORIGINAL_STATE,'All forwards after strict reload, no state changes')
    mp=R27/'independent_review/membership_v1.json';require(sha(mp)==MEMBERS_SHA,'Fixed members');m=read(mp)
    records={x['target_id']:x for x in m['records']}
    def evidence(tid):return {('events' if x['category']=='input' else x['category']):dict(relative_path=x['relative_path'],sha256=x['sha256']) for x in records[tid]['source_files']}
    first=m['train_order'][0]['target_id']
    require(r['parity_input']==dict(target_id=first,sequence=records[first]['sequence'],input_evidence=evidence(first),full_geometry=True),'Fixed full training input for parity only')
    require(set(r['evaluations'])=={'diagnostic'},'Only declared64 evaluation')
    ev=r['evaluations']['diagnostic'];require(ev['model_forwards']==64 and ev['GT_arrays_read'] is False and ev['state_sha256']==digest,'Same-W64 evaluation')
    require([(x['target_id'],x['sequence']) for x in ev['frames']]==[(x['target_id'],x['sequence']) for x in m['dev_members']],'Exact fixed development order')
    q4result=R28/'actual_v1/formal_Q4_v2/results.json';require(sha(q4result)==prior['source_results']['Q4'],'Original actual source result')
    bn_names=set(read(q4result)['evaluations']['initial']['frames'][0]['actual_BN_calls']);require(len(bn_names)==78,'Same original78 BN names')
    predictions=[];work_sum=dict(row_planes=0,kernel_calls=0,explicit_padding_rows=0)
    for frame in ev['frames']:
        require(frame['input_evidence']==evidence(frame['target_id']),'Same original data provenance')
        require(set(frame['actual_BN_calls'])==bn_names and set(frame['actual_BN_calls'].values())=={1},'All78 BN each forward')
        work=old.check_calls(frame['actual_affine_calls'],frame['work'],'Q8',(480,640))
        for k,v in work.items():work_sum[k]+=v
        require(set(frame['heads'])=={'head0','head1','head2','head3'},'All4 full outputs recorded')
        p=child(a.run,frame['prediction_file']);require(sha(p)==frame['prediction_file_sha256'],'Actual prediction archive')
        with np.load(p,allow_pickle=False) as z:
            require(z.files==['head3'],'One actual quality head');array=z['head3']
        require(array.shape==(2,480,640) and array.dtype==np.float32 and np.isfinite(array).all(),'Full finite pixel flow')
        d=hashlib.sha256(array.tobytes()).hexdigest();require(d==frame['prediction_sha256']==frame['heads']['head3']['sha256'],'Actual prediction payload')
        predictions.append(dict(state='Q8_on_recovered_Q4',target_id=frame['target_id'],sequence=frame['sequence'],file=str(p.resolve()),file_sha256=sha(p),head3_sha256=d,input_evidence=frame['input_evidence']))
    for key in ('final_training_deployment_parity','final_dense_integer_parity'):q4.parity(r[key],True,psns)
    pars=[r['final_training_deployment_parity'][k] for k in ('reference_record','actual_record')]+[r['final_dense_integer_parity']['dense_record']]
    for p,(ste,dense) in zip(pars,((True,False),(False,False),(False,True))):
        require(set(p['actual_BN_calls'])==bn_names and set(p['actual_BN_calls'].values())=={1},'Actual parity BN calls')
        require(set(p['actual_target_calls'])==old.TARGETS and set(p['actual_target_calls'].values())=={1},'All28 actual parity targets')
        require((p['PSN_count'],p['integer_count'],p['head_count'])==(105,28,4) and p['ste_entry'] is ste and p['dense_integer_reference'] is dense,'Distinct three parity paths')
    for p in pars[:2]:
        w=old.check_calls(ev['frames'][0]['actual_affine_calls'],p['work'],'Q8',(480,640))
        for k,v in w.items():work_sum[k]+=v
        for x in p['work'].values():
            limit=max(24,min(65536,8*1024*1024//x['Cout']));limit=limit//8*8
            expected=[]
            for bit in range(8):
                for start in range(0,x['rows'],limit):
                    n=min(limit,x['rows']-start)
                    expected.append(dict(kind='binary_weight_plane',bit=bit,rows=n,padding_rows=max(24,((n+7)//8)*8)-n,scope='all',coefficient=-128 if bit==7 else 1<<bit))
            require(x['kernel_calls']==expected,'Exact Q8 bit/chunk/row/padding ledger')
    q4.dense_reference_work(pars[2]['work'],pars[0]['work'])
    require(work_sum['row_planes']==2211264000 and sum(x['rows'] for x in pars[2]['work'].values())==4188000,'66 true8-plane forwards and1 dense reference')
    pf=child(a.run,r['parity_heads_file']);require(sha(pf)==r['parity_heads_sha256'],'Actual parity4 archive')
    with np.load(pf,allow_pickle=False) as z:
        require(set(z.files)=={'head0','head1','head2','head3'},'Four saved parity heads')
        for k in z.files:
            v=z[k];require(v.shape==(1,2,480,640) and v.dtype==np.float32 and np.isfinite(v).all(),'Finite full parity heads')
            require(hashlib.sha256(v.tobytes()).hexdigest()==r['final_training_deployment_parity']['checks'][k]['actual_sha256'],'Independent actual parity bytes')
    out=dict(schema='flow_r29_independent_actual_execution_audit_v1',status='PASS',quality_analysis_allowed=True,auditor_sha256=sha(__file__),source_result_sha256=sha(rpath),Q4_source_checkpoint_sha256=sha(weights),source_state_sha256=digest,R28_execution_audit_sha256=sha(auditpath),counts=dict(optimizer_updates=0,quality_predictions=64,formal_model_forwards=67),predictions=predictions,serial_work=work_sum,GT_values_read=False,scope='Same-W Q8 compatibility diagnostic only; runtime PSN/integer bytewise receipts plus independently reloaded64 predictions and4 parity heads. No task score or hardware benefit claim.')
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('status','counts','source_state_sha256')}))

if __name__=='__main__':main()
