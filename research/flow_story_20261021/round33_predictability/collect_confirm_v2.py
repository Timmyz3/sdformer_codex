"""Round33 confirmation v2: complete accepted source inventory, same execution.

Old E3 contract is used only for its validated geometries/masks/endpoint. Its
old16 identity list is NOT the identity authority for this new experiment.
"""
import argparse,copy,fcntl,hashlib,importlib.util,json,signal,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def load(p,n):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def prepare(a):
    import torch
    assert not torch.cuda.is_initialized() and not a.output.exists()
    assert sha(__file__)==a.runner_sha256
    assert sha(a.new_members)== 'd860612a3ae3e4620fa2a9a616df9f8e93774efadcf7dc564881e27b7e7da8cc'
    assert sha(a.e3_root/'probe_contract.py')=='63861615f1b93f06e26862d846b9bf8b7d4b66a2a13335f2e44d0c166cadb24e'
    assert sha(a.e3_root/'collect_probe.py')=='ab4222e012857275e82e7cf0b848d1ca8bca1dee4d007069ebabe702cc236077'
    sys.path.insert(0,str(a.e3_root)); import probe_contract as ct
    oldcollect=load(a.e3_root/'collect_probe.py','_r33_old_geometry_validation_only')
    assert sha(a.e3_root/'ENDPOINT_CONTRACT_V1.json')=='1be30c237a24fd7173b6c5678e41aa56093301ad5f76f57b96a5a55f82df8718'
    core=ct.load_core(a.e0_core);contract=ct.validate(read(a.e3_root/'ENDPOINT_CONTRACT_V1.json'),core)
    assert sha(a.weights)==contract['endpoint']['weights_sha256']
    base=a.r28_root/'implementation'
    for n,h in [('run_smoke_v2.py',ct.SMOKE_SHA),('run_recovery_v2.py',ct.HELPER_SHA),('recovery_integer_core.py',ct.R28_CORE_SHA)]:assert sha(base/n)==h
    smoke=load(base/'run_smoke_v2.py','_r33_smoke'); helper=load(base/'run_recovery_v2.py','_r33_helper')
    old=copy.copy(a);old.runner_sha256=ct.SMOKE_SHA
    support,_,_,_=smoke.prepare(old)
    members=read(a.new_members)
    assert a.phase=='confirm', 'This version only admits frozen confirmation identities'
    verification_path=a.new_members.parent/'native_verification_files_v1.json'
    assert sha(verification_path)=='769671e6e3754f068587eaed576ce51c0d0d6f939aeabf033a006f9aec9e6c5c'
    verified={f['relative_path']:f for f in read(verification_path)['files']}
    roles=['confirm']
    rows=[r for r in members['records'] if r['role'] in roles]
    assert len(rows)==(80 if a.phase=='fit_tune' else 40)
    native,_,metrics,_,hashes,*_=support['context']
    canonical=metrics.canonical_rows(support['b'].canonical)
    canonical={r['target_id']:r for r in canonical} if isinstance(canonical,list) else canonical
    for rec in rows:
        assert rec['canonical']==canonical[rec['target_id']]
        expected=native.target_files(rec['canonical'])
        for f in rec['source_files']:
            assert f['relative_path']==expected[f['category']]
            v=verified[f['relative_path']]
            assert v['passed'] is True and v['all_elements_finite'] is True
            assert v['sha256']==f['sha256'] and v['bytes']==int(f['bytes'])
            if f['relative_path'] in hashes:assert f['sha256']==hashes[f['relative_path']]['sha256']
            p=native.checked_path(support['b'].data_root,f['relative_path'])
            assert sha(p)==f['sha256'] and p.stat().st_size==int(f['bytes'])
    if a.phase=='confirm':
        assert a.heads_freeze and a.heads_freeze_sha256 and sha(a.heads_freeze)==a.heads_freeze_sha256
        hf=read(a.heads_freeze)
        assert hf['schema']=='flow_r33_heads_freeze_v1' and hf['membership_sha256']==sha(a.new_members)
        assert hf['fit_ids']==[r['target_id'] for r in members['records'] if r['role']=='fit']
        assert hf['confirm_ids']==[r['target_id'] for r in rows]
        assert not set(hf['fit_ids'])&set(hf['confirm_ids'])
        for name,digest in hf['files'].items():
            assert Path(name).name==name and sha(a.heads_freeze.parent/name)==digest
        assert {'MODEL_V1.json','FIT_RECEIPT_V1.json'}.issubset(hf['files'])
        model=read(a.heads_freeze.parent/'MODEL_V1.json')
        assert model['schema']=='flow_r33_voxel_small_predictors_v1' and model['metadata']['fit_ids']==hf['fit_ids']
        assert model['alpha']==0.1 and model['metadata']['alpha_search'] is False
        receipt=read(a.heads_freeze.parent/'FIT_RECEIPT_V1.json')
        assert receipt['schema']=='flow_r33_fit_receipt_v1' and receipt['membership_sha256']==sha(a.new_members)
        assert receipt['model_sha256']==hf['files']['MODEL_V1.json'] and receipt['fit_ids']==hf['fit_ids']
        assert receipt['alpha']==0.1 and receipt['tune_or_confirm_read'] is False
    return ct,oldcollect,core,contract,smoke,helper,support,rows
def execute(a,p,result):
    import numpy as np,torch
    ct,oldcollect,core,c,smoke,helper,support,rows=p
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261019)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.set_float32_matmul_precision('highest')
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.manual_seed_all(20261019);torch.cuda.reset_peak_memory_stats()
    start=time.monotonic(); model,functional,bn=smoke.make_model(support,'full9',device,torch)
    cp=torch.load(a.weights,map_location='cpu',weights_only=False)
    state=c['endpoint']['state_sha256'];assert cp['final_state_sha256']==state and cp['completed_optimizer_steps']==512
    model.load_state_dict(cp['model_state_dict'],strict=True);del cp;model.eval().requires_grad_(False)
    assert support['modules'][0].state_digest(model)==state
    bnsig=support['base'].bn_signature(model,torch)
    result.update(status='RUNNING',BN_scope=bn,model_forwards=0,formal_model_forwards=0,engineering_model_forwards=0,evaluations={},parity={},runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),AMP=False,TF32=False))
    write(a.output/'results.json',result)
    for action in range(-1,6):
        label='background_Q4' if action<0 else f'upgrade_{action}'
        bound=ct.BoundCore(core,c,action);folder=a.output/'predictions'/label;folder.mkdir(parents=True)
        frames=[];tic=time.monotonic()
        for index,rec in enumerate(rows):
            tid=rec['target_id'];source=next(f for f in rec['source_files'] if f['category']=='input')
            path=support['b'].data_root/source['relative_path'];assert sha(path)==source['sha256']
            events=np.load(path,allow_pickle=False);assert events.shape==(10,480,640) and np.isfinite(events).all()
            chunk=support['base'].prepare_input(events,device,torch);del events
            inputrecord=helper.tensor_record(chunk)
            if action>=0 and index==0 and a.phase=='fit_tune':
                actual,ar=smoke.capture_full(model,functional,chunk,bound.mode,bound,torch,ste=False)
                result['model_forwards']+=1;result['formal_model_forwards']+=1
                dense,dr=smoke.capture_full(model,functional,chunk,bound.mode,bound,torch,ste=False,reference=True)
                result['model_forwards']+=1;result['engineering_model_forwards']+=1
                cmp=smoke.compare_full(actual,dense,torch);assert cmp['pass']
                result['parity'][label]=dict(target_id=tid,comparison=cmp,actual_record=ar,dense_record=dr)
                heads={f'head{i}':helper.tensor_record(actual[f'head{i}'][0]) for i in range(4)}
                prediction=actual['head3'][0].numpy().copy();work,bns,calls=ar['work'],ar['actual_BN_calls'],ar['actual_target_calls'];del actual,dense
            else:
                functional.reset_net(model)
                with bound.installed(model,bound.mode,ste=False) as wraps:
                    with helper.observed(model,bound,torch) as (bns,measured):
                        with torch.no_grad():flows=model(chunk)['flow']
                    work={n:copy.deepcopy(w.last_audit) for n,w in wraps.items()}
                result['model_forwards']+=1;result['formal_model_forwards']+=1
                assert len(flows)==4
                heads={f'head{i}':helper.tensor_record(v[0]) for i,v in enumerate(flows)}
                prediction=flows[3][0].detach().cpu().numpy().copy();calls={n:len(v) for n,v in measured.items()};del flows
            assert prediction.shape==(2,480,640) and prediction.dtype==np.float32 and np.isfinite(prediction).all()
            checks=oldcollect.validate_work(work,calls,bns,c,core,action)
            workdigest=hashlib.sha256(json.dumps(work,sort_keys=True).encode()).hexdigest()
            if index==0:exemplar=work;exemplar_digest=workdigest
            else:assert workdigest==exemplar_digest,'Same action must have identical value-independent work ledger'
            file=folder/f'{tid}.npz';np.savez_compressed(file,head3=prediction)
            frames.append(dict(target_id=tid,sequence=rec['sequence'],role=rec['role'],action=action,input_source=source,actual_input=inputrecord,
                prediction_file=str(file.relative_to(a.output)),prediction_file_sha256=sha(file),prediction_sha256=heads['head3']['sha256'],heads=heads,
                actual_BN_calls=bns,actual_affine_call_counts=calls,work_checks=checks,
                work_sha256=workdigest))
            del chunk,prediction
            write(a.output/'progress.json',dict(action=label,identities_completed=index+1,formal_model_forwards=result['formal_model_forwards'],engineering_model_forwards=result['engineering_model_forwards']))
            if (index+1)%8==0:print(label,index+1,flush=True)
        assert support['modules'][0].state_digest(model)==state and support['base'].bn_signature(model,torch)==bnsig
        result['evaluations'][label]=dict(action=action,mask_layout=bound.mask_record,frames=frames,state_sha256=state,model_forwards=len(rows),GT_arrays_read=False,first_frame_full_work=exemplar,wall_seconds=time.monotonic()-tic)
        write(a.output/'results.json',result)
    assert result['formal_model_forwards']==len(rows)*7 and result['engineering_model_forwards']==(6 if a.phase=='fit_tune' else 0)
    result.update(status='R33_COMPLETE',final_state_sha256=state,completed_optimizer_steps=0,wall_seconds=time.monotonic()-start,max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved())
def main():
    ap=argparse.ArgumentParser()
    for n in ['e0-core','e3-root','r28-root','r26-root','members','new-members','weights','output']:ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--phase',choices=['fit_tune','confirm'],required=True);ap.add_argument('--runner-sha256',required=True)
    ap.add_argument('--heads-freeze',type=Path);ap.add_argument('--heads-freeze-sha256')
    ap.add_argument('--preflight',type=Path);ap.add_argument('--execute',action='store_true');a=ap.parse_args();p=prepare(a)
    result=dict(schema='flow_r33_collector_v1',native_verification_sha256='769671e6e3754f068587eaed576ce51c0d0d6f939aeabf033a006f9aec9e6c5c',data_admission='complete_accepted_inventory_plus_optional_legacy_subset_crosscheck',status='CPU_PREFLIGHT_PASS',phase=a.phase,runner_sha256=sha(__file__),membership_sha256=sha(a.new_members),endpoint=p[3]['endpoint'],selected_ids=[r['target_id'] for r in p[-1]],roles=[r['role'] for r in p[-1]],action_contract_sha256=sha(a.e3_root/'ENDPOINT_CONTRACT_V1.json'),
        old_action_contract_used_for='Frozen masks, geometry and endpoint only; new identity authority is new_members.',expected_formal_forwards=len(p[-1])*7,expected_engineering_forwards=6 if a.phase=='fit_tune' else 0,
        training_performed=False,GT_arrays_read=False,quality_computed=False,resume_allowed=False,heads_freeze_sha256=a.heads_freeze_sha256,wall_cap_minutes=75 if a.phase=='fit_tune' else 40)
    if not a.execute:
        assert a.preflight and not a.preflight.exists();write(a.preflight,result);return
    lock=p[-2]['modules'][3].GPU_LOCK.open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);a.output.mkdir(parents=True,exist_ok=False)
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('R33 wall cap')));signal.alarm(result['wall_cap_minutes']*60)
        try:execute(a,p,result)
        except BaseException as err:
            result.update(status='R33_PARTIAL_FAILED',error=str(err),traceback=traceback.format_exc());write(a.output/'results.json',result);raise
        finally:signal.alarm(0)
        write(a.output/'results.json',result)
    finally:lock.close()
if __name__=='__main__':main()
