"""One static360 action,120 formal+4 engineering forwards. No training/GT decode."""
import argparse,copy,fcntl,importlib.util,json,signal,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
import static_contract as sc
sha,read,require=sc.sha,sc.read,sc.require
def write(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def prepare(a):
    import torch
    require(not torch.cuda.is_initialized() and not a.output.exists(),'CPU-only preflight/fresh output')
    require(sha(__file__)==a.runner_sha256 and sha(a.source_freeze)==a.source_freeze_sha256,'Frozen collector/manifest')
    for name,h in read(a.source_freeze)['files'].items():require(Path(name).name==name and sha(Path(__file__).parent/name)==h,'Frozen local source '+name)
    require(sha(a.contract)==a.contract_sha256,'New360 contract bytes')
    require(sha(a.e3_root/'probe_contract.py')=='63861615f1b93f06e26862d846b9bf8b7d4b66a2a13335f2e44d0c166cadb24e','Original geometry helper')
    ct=load(a.e3_root/'probe_contract.py','_r35_old_pinned_helper');core=ct.load_core(a.e0_core)
    c=sc.validate(read(a.contract),core)
    require(sha(a.e3_root/'ENDPOINT_CONTRACT_V1.json')==c['old_geometry_contract_sha256'],'Original geometry authority')
    oldc=ct.validate(read(a.e3_root/'ENDPOINT_CONTRACT_V1.json'),core)
    require(c['geometries']==oldc['geometries'] and c['endpoint']==oldc['endpoint'],'Same geometry and endpoint')
    require(sha(a.weights)==c['endpoint']['weights_sha256'],'Actual frozen checkpoint bytes')
    base=a.r28_root/'implementation'
    for name,h in [('run_smoke_v2.py',ct.SMOKE_SHA),('run_recovery_v2.py',ct.HELPER_SHA),('recovery_integer_core.py',ct.R28_CORE_SHA)]:require(sha(base/name)==h,'Pinned helper '+name)
    smoke=load(base/'run_smoke_v2.py','_r35_smoke');helper=load(base/'run_recovery_v2.py','_r35_helper')
    old=copy.copy(a);old.runner_sha256=ct.SMOKE_SHA;support,_,_,_=smoke.prepare(old)
    require(sha(a.new_members)==c['membership_sha256'],'Same120 membership')
    rows=read(a.new_members)['records'];require([r['target_id'] for r in rows]==c['target_ids'],'Exact120 role ordering')
    verpath=a.new_members.parent/'native_verification_files_v1.json'
    require(sha(verpath)==c['native_verification_sha256'],'Complete source admission inventory')
    verified={f['relative_path']:f for f in read(verpath)['files']}
    native,_,metrics,_,hashes,*_=support['context'];can=metrics.canonical_rows(support['b'].canonical)
    can={r['target_id']:r for r in can} if isinstance(can,list) else can
    for rec in rows:
        require(rec['canonical']==can[rec['target_id']],'Native canonical identity')
        expected=native.target_files(rec['canonical'])
        for f in rec['source_files']:
            require(f['relative_path']==expected[f['category']],'Native file mapping')
            v=verified[f['relative_path']];require(v['passed'] and v['all_elements_finite'] and v['sha256']==f['sha256'] and v['bytes']==int(f['bytes']),'Accepted raw source')
            if f['relative_path'] in hashes:require(f['sha256']==hashes[f['relative_path']]['sha256'],'Legacy subset source')
            path=native.checked_path(support['b'].data_root,f['relative_path'])
            require(sha(path)==f['sha256'] and path.stat().st_size==int(f['bytes']),'Actual data bytes')
    refs=read(Path(__file__).parent/'Q4_PARITY_REFERENCES_V1.json')
    require(sha(Path(__file__).parent/'Q4_PARITY_REFERENCES_V1.json')==c['Q4_reference_sha256'],'Original four-head replay refs')
    for role in ('fit','tune','confirm'):
        tid=next(r['target_id'] for r in rows if r['role']==role)
        require(refs['roles'][role]['target_id']==tid,'First identity per role')
    masks,record=sc.materialize(c,core,True)
    return core,c,smoke,helper,support,rows,refs,record

def execute(a,p,result):
    import numpy as np,torch
    core,c,smoke,helper,support,rows,refs,maskrecord=p
    require(not torch.cuda.is_initialized(),'GPU lock precedes CUDA')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261019)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.set_float32_matmul_precision('highest');device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.manual_seed_all(20261019);torch.cuda.reset_peak_memory_stats()
    start=time.monotonic();model,functional,bn=smoke.make_model(support,'full9',device,torch)
    cp=torch.load(a.weights,map_location='cpu',weights_only=False);state=c['endpoint']['state_sha256']
    require(cp['schema']=='flow_e0_rne4_joint_checkpoint_v1' and cp['completed_optimizer_steps']==512 and cp['final_state_sha256']==state,'Actual endpoint checkpoint metadata')
    model.load_state_dict(cp['model_state_dict'],strict=True);del cp;model.eval().requires_grad_(False)
    require(support['modules'][0].state_digest(model)==state,'Actual endpoint model digest')
    psn_names=[n for n,m in model.named_modules() if type(m).__name__=='PSN'];require(len(psn_names)==105,'105 original PSN modules')
    bnsig=support['base'].bn_signature(model,torch)
    result.update(status='RUNNING',BN_scope=bn,model_forwards=0,forward_attempts=0,formal_model_forwards=0,engineering_model_forwards=0,
        evaluations={},parity={},runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),AMP=False,TF32=False))
    def begin(kind):
        require(result['forward_attempts']<124,'Hard124 attempts');result['forward_attempts']+=1;result['current_forward_kind']=kind
    def complete(kind):
        result['model_forwards']+=1;result['formal_model_forwards' if kind=='formal' else 'engineering_model_forwards']+=1
    def input_for(rec):
        source=next(f for f in rec['source_files'] if f['category']=='input');path=support['b'].data_root/source['relative_path'];require(sha(path)==source['sha256'],'Actual event bytes')
        events=np.load(path,allow_pickle=False);require(events.shape==(10,480,640) and events.dtype==np.float32 and np.isfinite(events).all(),'Actual T10 event shape')
        chunk=support['base'].prepare_input(events,device,torch);return chunk,source,helper.tensor_record(chunk)
    def normal(chunk,bound,kind):
        functional.reset_net(model);psns={n:0 for n in psn_names};handles=[]
        with bound.installed(model,bound.mode,ste=False) as wraps:
            with helper.observed(model,bound,torch) as (bns,measured):
                try:
                    for name in psn_names:
                        def count(m,ins,out,name=name):psns[name]+=1
                        handles.append(model.get_submodule(name).register_forward_hook(count))
                    begin(kind)
                    with torch.no_grad():flows=model(chunk)['flow']
                    complete(kind)
                finally:
                    for h in handles:h.remove()
            work={n:copy.deepcopy(w.last_audit) for n,w in wraps.items()}
        require(all(v==1 for v in psns.values()),'105 actual one-time PSN calls')
        require(len(flows)==4,'Four output heads');heads={f'head{i}':helper.tensor_record(v[0]) for i,v in enumerate(flows)}
        prediction=flows[3][0].detach().cpu().numpy().copy();del flows
        return prediction,heads,work,bns,{n:len(v) for n,v in measured.items()},psns
    # Replay three oldQ4 references before spending120 new action forwards.
    q4=sc.BoundCore(core,c,False)
    for role in ('fit','tune','confirm'):
        rec=next(r for r in rows if r['role']==role);chunk,source,inputrec=input_for(rec)
        prediction,heads,work,bns,calls,psns=normal(chunk,q4,'engineering')
        require(all(heads[k]['sha256']==refs['roles'][role]['heads'][k]['sha256'] for k in heads),'Four-head Q4 byte parity '+role)
        checks=sc.validate_work(work,calls,bns,c,core,False)
        result['parity']['Q4_'+role]=dict(target_id=rec['target_id'],heads=heads,reference=refs['roles'][role],all_four_head_hashes_exact=True,
            actual_input=inputrec,work_checks=checks,actual_BN_calls=bns,actual_affine_call_counts=calls,actual_PSN_calls=psns)
        del chunk,prediction;write(a.output/'results.json',result)
    bound=sc.BoundCore(core,c,True);frames=[];folder=a.output/'predictions/static360';folder.mkdir(parents=True)
    result['evaluations']['static360']=dict(action='static360',mask_layout=bound.mask_record,frames=frames,GT_arrays_read=False,state_sha256=state)
    for index,rec in enumerate(rows):
        chunk,source,inputrec=input_for(rec)
        if index==0:
            begin('formal');actual,ar=smoke.capture_full(model,functional,chunk,bound.mode,bound,torch,ste=False);complete('formal')
            begin('engineering');dense,dr=smoke.capture_full(model,functional,chunk,bound.mode,bound,torch,ste=False,reference=True);complete('engineering')
            require(ar['PSN_count']==dr['PSN_count']==105,'105 parity PSN actual captures')
            comparison=smoke.compare_full(actual,dense,torch);require(comparison['pass'],'Full static360 dense integer/PSN/four-head parity')
            result['parity']['static360_dense']=dict(target_id=rec['target_id'],comparison=comparison,actual_record=ar,dense_record=dr)
            heads={f'head{i}':helper.tensor_record(actual[f'head{i}'][0]) for i in range(4)};prediction=actual['head3'][0].numpy().copy()
            work,bns,calls=ar['work'],ar['actual_BN_calls'],ar['actual_target_calls'];psns={n:1 for n in psn_names};del actual,dense
        else:prediction,heads,work,bns,calls,psns=normal(chunk,bound,'formal')
        require(prediction.shape==(2,480,640) and prediction.dtype==np.float32 and np.isfinite(prediction).all(),'Finite actualhead3')
        checks=sc.validate_work(work,calls,bns,c,core,True);digest=__import__('hashlib').sha256(json.dumps(work,sort_keys=True).encode()).hexdigest()
        if index==0:result['evaluations']['static360']['first_frame_full_work']=work;firstdigest=digest
        else:require(digest==firstdigest,'Value-independent fixed360 work')
        path=folder/(rec['target_id']+'.npz');np.savez_compressed(path,head3=prediction)
        frame=dict(target_id=rec['target_id'],sequence=rec['sequence'],role=rec['role'],action='static360',input_source=source,actual_input=inputrec,
            prediction_file=str(path.relative_to(a.output)),prediction_file_sha256=sha(path),prediction_sha256=heads['head3']['sha256'],heads=heads,
            actual_BN_calls=bns,actual_affine_call_counts=calls,actual_PSN_calls=psns,work_checks=checks,work_sha256=digest)
        frames.append(frame);write(a.output/'records'/(rec['target_id']+'.json'),frame);del chunk,prediction
        write(a.output/'progress.json',dict(identities_completed=index+1,formal_model_forwards=result['formal_model_forwards'],engineering_model_forwards=result['engineering_model_forwards'],forward_attempts=result['forward_attempts']))
        if (index+1)%8==0:print('STATIC360',index+1,flush=True);write(a.output/'results.json',result)
    require(result['model_forwards']==result['forward_attempts']==124 and result['formal_model_forwards']==120 and result['engineering_model_forwards']==4,'Exact completed120+4')
    require(support['modules'][0].state_digest(model)==state and support['base'].bn_signature(model,torch)==bnsig,'Weights andBN unchanged')
    result['evaluations']['static360']['model_forwards']=120
    result.update(status='R35_STATIC360_COMPLETE',final_state_sha256=state,completed_optimizer_steps=0,wall_seconds=time.monotonic()-start,
        max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved())

def main():
    ap=argparse.ArgumentParser()
    for n in ['e0-core','e3-root','r28-root','r26-root','members','new-members','weights','output','contract','source-freeze']:ap.add_argument('--'+n,type=Path,required=True)
    for n in ['runner-sha256','contract-sha256','source-freeze-sha256']:ap.add_argument('--'+n,required=True)
    ap.add_argument('--preflight',type=Path);ap.add_argument('--execute',action='store_true');a=ap.parse_args();p=prepare(a)
    result=dict(schema='flow_r35_static_neighbor_collector_v1',status='CPU_PREFLIGHT_PASS',runner_sha256=sha(__file__),source_freeze_sha256=sha(a.source_freeze),
        action_contract_sha256=sha(a.contract),membership_sha256=sha(a.new_members),endpoint=p[1]['endpoint'],selected_ids=[r['target_id'] for r in p[5]],
        roles=[r['role'] for r in p[5]],mask_record=p[7],expected_formal_forwards=120,expected_engineering_forwards=4,max_forward_attempts=124,
        wall_cap_minutes=20,training_performed=False,GT_arrays_read=False,quality_computed=False,resume_allowed=False)
    if not a.execute:
        require(a.preflight and not a.preflight.exists(),'Fresh explicit preflight destination');write(a.preflight,result);return
    lock=p[4]['modules'][3].GPU_LOCK.open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);a.output.mkdir(parents=True,exist_ok=False)
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('R35 hard20min execution cap')));signal.alarm(1200)
        try:execute(a,p,result)
        except BaseException as err:
            result.update(status='R35_PARTIAL_FAILED',error=str(err),traceback=traceback.format_exc());write(a.output/'results.json',result);raise
        finally:signal.alarm(0)
        write(a.output/'results.json',result)
    finally:lock.close()

if __name__=='__main__':main()
