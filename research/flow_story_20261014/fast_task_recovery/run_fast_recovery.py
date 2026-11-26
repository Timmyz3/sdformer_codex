#!/usr/bin/env python3
"""Fixed R0 feature-MSE versus R1 native-task recovery of one warmed fast MLP.

Default is CPU provenance preflight. CUDA work requires --execute and the existing
shared GPU lock. Every phase writes a NEW directory. No old files are changed.
"""
import argparse
import contextlib
import copy
import fcntl
import hashlib
import inspect
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PARENT_SHA='564d70a02b65d00c1aec40eacf069babb38f600c5a553a5ceb2a263630f3c389'
FIT_SHA='29ec7ba9c9faca74c62c3e170c9d3c70855a4605a7eab31f9331c519f56f8915'
FIT_IDS_SHA='99c2409ca95b8b96903a9449767435df59f6e9a845c95c9b35eb97c250be1011'
ADAPTER_SHA='75c97bb1e35e31a4e7b0fb97adefe434e6346dfa156ee6e9c297547efdbe30d7'
ADAPTER_PROVENANCE_SHA='4971687125cd6df959d2314e71e3b44c8a4cbb5514694fb10f0bf126fe3a29c1'
CHECKPOINT_SHA='d5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30'
GPU_LOCK=Path('/data/flow_a/runs/.flow_a_gpu0.lock')
RECIPE={'seed':20261014,'steps':200,'fit_targets':20,'passes':10,'batch':1,'geometry':'full9','T':10,
        'lr':.001,'weight_decay':.0001,'optimizer':'AdamW','betas':[.9,.999],'eps':1e-8,
        'precision':'FP32, TF32 off, no AMP','gradient_clipping':None,'BN_scope':'MLP_CMA176',
        'fast':'original per-position LN(C384)+Linear(C384,C384)','mask':'fixed cap900; per-target common reachable actual points; identical frozen mask for R0/R1',
        'backbone':'eval, all original parameters frozen; current-stat external BN retained',
        'augmentation':'none; same full image, fixed fit20 order repeated10',
        'R0':'mean squared teacher branch error over all fast positions/T/channels',
        'R1':"native flow_loss_supervised(config,device)(output['flow'],label,mask,gamma=None)",
        'budget_scope':'equal target exposures/updates, NOT equal backward FLOPs or wall time'}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temp,path)


def load_fast_class(adapter_dir):
    """Load only the pinned topology; CPU fitter has import-time device policy."""
    directory=Path(adapter_dir).resolve()
    require(sha(directory/'adapter.py')==ADAPTER_SHA and
            sha(directory/'source_provenance.json')==ADAPTER_PROVENANCE_SHA,'Borrowed FastBranch source differs')
    before=os.environ.get('CUDA_VISIBLE_DEVICES')
    name='_flow_fast_task_recovery_pinned_adapter'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,directory/'adapter.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        try:spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    module=sys.modules[name]
    require(Path(module.__file__).resolve()==directory/'adapter.py','Borrowed FastBranch import path differs')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==before,'Fast topology import changed CUDA visibility')
    return module.FastBranch


def membership(fit,canonical,roles,dev):
    require(fit.get('schema')=='flow_recovery_svs_fit20_v1' and fit.get('checkpoint_sha256')==CHECKPOINT_SHA,
            'Wrong fit20/checkpoint identity')
    ids=fit['target_ids'];byid={r['target_id']:r for r in canonical}
    require(len(ids)==len(set(ids))==20 and digest(ids)==FIT_IDS_SHA==fit['target_ids_sha256'],'Fixed ordered fit20 required')
    rows=[byid[t] for t in ids]
    require(all(r['protocol_split']=='train' and roles.get(r['target_id'])=='gate_fit' for r in rows),'Fit target outside gate_fit')
    counts={s:sum(r['sequence']==s for r in rows) for s in {r['sequence'] for r in rows}}
    require(len(counts)==10 and set(counts.values())=={2} and counts==fit['sequence_counts'],'Fit sequence coverage differs')
    require(len(dev)==8 and len({r['target_id'] for r in dev})==8 and
            all(r['protocol_split']=='train' and roles.get(r['target_id'])=='gate_development' for r in dev),
            'Fixed eight gate-development targets required')
    require(not set(ids)&{r['target_id'] for r in dev} and not set(counts)&{r['sequence'] for r in dev},'Fit/dev overlap')
    return rows


def source_digest(model):
    h=hashlib.sha256()
    for name,v in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def validate_bank(bank,binding,fit_rows,dev_rows):
    require(bank.get('schema')=='flow_fast_recovery_mask_bank_v1' and bank.get('status')=='MASK_BANK_COMPLETE',
            'Complete mask bank required')
    require(bank.get('binding')==binding,'Mask bank provenance/recipe differs')
    require(bank.get('fit_ids')==[r['target_id'] for r in fit_rows] and bank.get('dev_ids')==[r['target_id'] for r in dev_rows],
            'Mask bank population/order differs')
    frames=bank['frames'];expected=fit_rows+dev_rows
    require(len(frames)==28 and [f['target_id'] for f in frames]==[r['target_id'] for r in expected],
            'Mask bank frame identities differ')
    for index,(f,row) in enumerate(zip(frames,expected)):
        require(f['role']==('fit' if index<20 else 'dev') and f['sequence']==row['sequence'],'Mask role/sequence differs')
        mask=f['keep'];require(len(mask)==4 and all(len(r)==5 and all(type(v)is bool for v in r) for r in mask),'Invalid bank mask')
        count=sum(min(9,30-y*9)*min(9,40-x*9) for y in range(4) for x in range(5) if mask[y][x])
        require(0<count<=900 and f['selector']['policy']=='independent' and not f['selector']['GT_or_final_flow_used'],
                'Mask is not the fixed-cap runtime independent control')
        require(type(f.get('slow_points')) is int and type(f.get('target_cost')) is int and
                count==f['slow_points']==f['target_cost']==f['selector']['selection']['slow_service_cost']==
                f['selector']['common_budget']['target_cost'] and
                f['selector']['nominal_slow_cap']==f['selector']['common_budget']['cap']==900 and
                f['selector']['selection']['slow_mask']==mask,
                'Mask/target/common-reachable-budget evidence differs')
        require(len(f['u_sha256'])==64 and f['u_sha256']==f['selector']['input_sha256'],'Input binding missing')
    return {f['target_id']:f for f in frames}


def validate_smoke(smoke,arm,binding,bank_sha):
    require(smoke.get('schema')=='flow_fast_task_recovery_results_v1' and smoke.get('status')=='SMOKE_COMPLETE'
            and smoke.get('arm')==arm and smoke.get('completed_optimizer_steps')==1,'Wrong/incomplete smoke arm')
    require(smoke.get('binding')==binding and smoke.get('mask_bank_sha256')==bank_sha,'Smoke source/mask differs')
    require(smoke.get('original_model_unchanged') is True and smoke.get('BN_buffers_unchanged') is True
            and smoke['initial_numerical_pair']['all_flow_outputs_exact'] is True
            and smoke['final_numerical_pair']['all_flow_outputs_exact'] is True,'Smoke numerical/frozen-state guards missing')


def parse_recovery_args(parent_args,output,extended):
    # The inherited parser rejects an existing output during parsing itself.
    # Replace in an independent argument list BEFORE invoking that parser.
    argv=list(parent_args)
    require(argv.count('--output')==1,'Parent output argument missing/ambiguous')
    argv[argv.index('--output')+1]=str(output)
    return extended.parse_extended(argv)


def prepare(a):
    require(a.parent_contract_sha256==PARENT_SHA==sha(a.parent_contract),'Parent runtime contract differs')
    require(a.fit_manifest_sha256==FIT_SHA==sha(a.fit_manifest),'Fixed fit20 source differs')
    require(not a.output_dir.exists(),'New output directory required')
    c=json.loads(a.parent_contract.read_text())
    require(c['schema']=='flow_fast_runtime_contract_v1' and c['target_count']==4,'Parent scope differs')
    for path,pin in c['source_pins'].items():require(sha(path)==pin,'Pinned source differs: '+path)
    for key in ('head_weights','fast_weights'):require(sha(c[key])==c[key+'_sha256'],'Pinned weights differ: '+key)
    runner=[Path(p) for p in c['source_pins'] if Path(p).name=='run_fast_runtime.py']
    require(len(runner)==1,'Missing frozen fast root')
    sys.path[:0]=[str(runner[0].parent),c['motion_root'],c['breadth_root']]
    import evaluate_breadth as base
    import evaluate_breadth_extended as extended
    import run_bn_scope as scope
    import run_same_shape_reference as loader
    import intervention_core as core
    b=parse_recovery_args(c['base_args'],a.output_dir/'results.json',extended)
    require(b.geometry=='full9' and b.checkpoint_sha256==CHECKPOINT_SHA and not b.run_inference,'Fixed full9 checkpoint required')
    b.output=a.output_dir/'results.json';b.device='cuda:0';b.wall_minutes=30
    context=extended.extended_preflight(b)
    native,trainer,metrics,dev32,hashes,config,saved,side,regions,plan=context
    require(side['completed_optimizer_steps']==135930,'New final baseline cannot replace fixed interim weights')
    canonical=metrics.canonical_rows(b.canonical);roles=metrics.auxiliary_roles(b.roles,canonical)
    dev=core.choose_targets(dev32,8)
    fit=membership(json.loads(a.fit_manifest.read_text()),canonical,roles,dev)
    binding={'parent_contract_sha256':PARENT_SHA,'fit_manifest_sha256':FIT_SHA,'checkpoint_sha256':CHECKPOINT_SHA,
        'fast_weights_sha256':c['fast_weights_sha256'],'head_weights_sha256':c['head_weights_sha256'],
        'entry_sha256':sha(__file__),'adapter_sha256':sha(HERE/'trainable_fast_tile.py'),
        'recipe':RECIPE,'device':'cuda:0','gpu_lock':str(GPU_LOCK),
        'fit_ids_sha256':digest([r['target_id'] for r in fit]),'dev_ids_sha256':digest([r['target_id'] for r in dev])}
    bank=None
    if a.mode!='prepare':
        require(a.mask_bank and a.mask_bank_sha256==sha(a.mask_bank),'Explicit mask bank SHA required')
        bank=json.loads(a.mask_bank.read_text());validate_bank(bank,binding,fit,dev)
    if a.mode in ('R0','R1'):
        require(a.smoke_result and a.smoke_result_sha256==sha(a.smoke_result),'Matching one-step GPU smoke receipt required')
        smoke=json.loads(a.smoke_result.read_text())
        validate_smoke(smoke,a.mode,binding,a.mask_bank_sha256)
    return c,b,context,base,extended,scope,loader,fit,dev,binding,bank


def execute(a,p,result):
    import numpy as np
    import torch
    import coarse_head
    import runtime_group_adapter as policy_module
    import trainable_fast_tile as tile
    c,b,context,base,extended,scope,loader,fit,dev,binding,bank=p
    fast_root=next(Path(path).parent for path in c['source_pins'] if Path(path).name=='run_fast_runtime.py')
    FastBranch=load_fast_class(c['adapter_dir'])
    require(not torch.cuda.is_initialized(),'CUDA initialized before shared GPU lock')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(RECIPE['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.manual_seed_all(RECIPE['seed'])
    runtime={'torch':str(torch.__version__),'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
        'device_name':torch.cuda.get_device_name(device),'device':'cuda:0','threads':2,'TF32':False,'AMP':False}
    started=time.monotonic()
    def deadline():require(time.monotonic()-started<1800,'Phase wall budget exhausted; no complete result')
    def sync():torch.cuda.synchronize(device)
    native,trainer,metrics,rows,hashes,config,saved,side,regions,plan=context
    model,functional=loader.load_model(b,context,base,torch,device)
    bn=scope.apply_scope(model,'MLP_CMA176',b.bn_payload,torch)
    model.eval().requires_grad_(False)
    before=source_digest(model);bn_before=base.bn_signature(model,torch)
    result.update(status='RUNNING',runtime=runtime,BN_scope=bn,original_model_state_sha256=before,steps=[])
    weights=torch.load(c['fast_weights'],map_location='cpu',weights_only=True)
    require(weights['schema']=='flow_fast_warmstart_weights_v1' and
            weights['provenance']['cache_sha256']==c['fast_cache_sha256'] and
            weights['provenance']['checkpoint_sha256']==CHECKPOINT_SHA and
            weights['provenance']['BN_scope']=='MLP_CMA176' and weights['recipe']['steps']==500 and
            weights['recipe']['seed']==20261014 and weights['provenance']['fit_script_sha256']==sha(fast_root/'fit_fast_cache.py'),
            'Warmstart provenance differs')
    fast=FastBranch(384).to(device)
    fast.load_state_dict(weights['stages']['2']['state_dict'],strict=True);fast.eval().requires_grad_(True)
    initial=source_digest(fast)
    result['initial_fast_state_sha256']=initial

    def read(row,events_only=False):
        paths=native.target_files(row);expected={}
        for kind,rel in paths.items():
            native.checked_path(b.data_root,rel);expected['events' if kind=='input' else kind]=hashes[rel]['sha256']
        if events_only:
            payload=native.checked_path(b.data_root,paths['input']).read_bytes()
            require(hashlib.sha256(payload).hexdigest()==expected['events'],'Event bytes differ')
            event=np.load(io.BytesIO(payload),allow_pickle=False)
            require(event.shape==(10,480,640) and event.dtype==np.float32 and np.isfinite(event).all(),'Invalid event array')
            return {'events':event},expected
        evidence={};arrays,_=metrics.load_arrays(b.data_root/'saved_flow_data',row,evidence=evidence,expected_hashes=expected)
        return arrays,evidence

    if a.mode=='prepare':
        hw=torch.load(c['head_weights'],map_location='cpu',weights_only=True)
        require(hw['provenance']['cache_sha256']==c['head_fit_cache_sha256'] and hw['flow_contract']==c['flow_contract'],
                'Coarse head provenance differs')
        head=coarse_head.RuntimeCoarseFlowHead(384,flow_contract=c['flow_contract'])
        head.load_state_dict(hw['stages']['2']['methods']['C16_GELU2']['state_dict'],strict=True)
        head.eval().requires_grad_(False)
        bank={'schema':'flow_fast_recovery_mask_bank_v1','status':'MASK_BANK_RUNNING','binding':binding,
              'runtime':runtime,'fit_ids':[r['target_id'] for r in fit],'dev_ids':[r['target_id'] for r in dev],
              'frames':[],'GT_used_by_selector':False,'final_flow_used_by_selector':False,
              'scope':'Frozen backbone pre-MLP u on CUDA; same CPU head/four-policy exact-budget planner; no tuning'}
        for i,row in enumerate(fit+dev):
            deadline();arrays,evidence=read(row,True);chunk=base.prepare_input(arrays['events'],device,torch)
            captured=[]
            def hook(module,inputs):
                cpu_u=inputs[0].detach().cpu()
                policy=policy_module.RuntimeGroupPolicy(head,'independent')
                mask=policy(tile.LAYER,cpu_u);record=policy.records[0]
                actual=record['selection']['slow_service_cost']
                require(type(actual) is int and 0<actual<=900 and actual==record['common_budget']['target_cost'],
                        'Invalid per-target common-reachable budget')
                captured.append({'keep':mask.tolist(),'u_sha256':tile.tensor_sha(inputs[0]),'selector':record,
                                 'slow_points':actual,'target_cost':actual})
            functional.reset_net(model)
            handle=model.get_submodule(tile.LAYER).register_forward_pre_hook(hook)
            try:
                with torch.no_grad():out=model(chunk)
            finally:handle.remove()
            require(len(captured)==1 and all(bool(torch.isfinite(x).all()) for x in out['flow']),'Invalid prepare forward')
            bank['frames'].append({'target_id':row['target_id'],'sequence':row['sequence'],'role':'fit' if i<20 else 'dev',
                'input_evidence':evidence,**captured[0]})
            write(a.output_dir/'mask_bank.json',bank)
            print(json.dumps({'prepare_target':row['target_id'],'complete':i+1}),flush=True)
            del chunk,out
        require(source_digest(model)==before and base.bn_signature(model,torch)==bn_before,'Prepare changed native model')
        bank['status']='MASK_BANK_COMPLETE';validate_bank(bank,binding,fit,dev)
        write(a.output_dir/'mask_bank.json',bank)
        result.update(status='MASK_BANK_COMPLETE',mask_bank_sha256=sha(a.output_dir/'mask_bank.json'),
                      complete_targets=28,training_performed=False,wall_seconds=time.monotonic()-started)
        return

    require(runtime==bank['runtime'],'Mask bank numerical runtime differs')
    entries=validate_bank(bank,binding,fit,dev)
    from loss.flow_supervised import flow_loss_supervised
    require(Path(inspect.getfile(flow_loss_supervised)).resolve().is_relative_to(b.source_root),'Unpinned loss import')
    loss_fn=flow_loss_supervised(config,device)
    result['native_loss']={'file_sha256':sha(inspect.getfile(flow_loss_supervised)),'gamma':None,
        'flow_scaling':config['metrics']['flow_scaling'],'lambda_mod':config['loss']['lambda_mod'],
        'all_native_output_flows_used':True,'GT_used_for_optimization':a.mode in ('R1','smoke_R1')}
    fit_cache={r['target_id']:read(r) for r in fit}
    result['training_input_evidence']={key:v[1] for key,v in fit_cache.items()}
    optimizer=torch.optim.AdamW(fast.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8)
    require(len(optimizer.param_groups)==1 and sum(p.numel() for p in fast.parameters())==148608,'Unexpected trainable parameters')

    def forward(row,arrays,mode,active_fast):
        deadline();functional.reset_net(model)
        entry=entries[row['target_id']];chunk=base.prepare_input(arrays['events'],device,torch)
        keep=torch.tensor(entry['keep'],dtype=torch.bool)
        with tile.replace(model,active_fast,keep,mode,entry['u_sha256']) as wrapper:
            output=model(chunk)
            require(len(wrapper.records)==1 and all(bool(torch.isfinite(x).all()) for x in output['flow']),
                    'Invalid native outputs/wrapper invocation')
            rec=wrapper.records[0];mse=wrapper.mse
        return output,mse,rec

    def numerical_pair(row,arrays,active_fast):
        with torch.no_grad():
            actual,_,ra=forward(row,arrays,'actual',active_fast)
            dual,_,rd=forward(row,arrays,'dual',active_fast)
            exact=all(torch.equal(x,y) for x,y in zip(actual['flow'],dual['flow']))
            require(len(actual['flow'])==len(dual['flow']) and exact and ra['output_sha256']==rd['output_sha256'],
                    'Actual branch evaluation differs from dual training reference')
            return {'all_flow_outputs_exact':exact,'branch_output_exact':True,'actual_service':ra,'dual_service':rd}

    result['initial_numerical_pair']=numerical_pair(fit[0],fit_cache[fit[0]['target_id']][0],fast)
    smoke=a.mode.startswith('smoke_');arm=a.mode.removeprefix('smoke_');steps=1 if smoke else 200
    torch.cuda.reset_peak_memory_stats(device);sync();train_start=time.monotonic()
    for step in range(steps):
        deadline();row=fit[step%20];arrays,_=fit_cache[row['target_id']]
        optimizer.zero_grad(set_to_none=True);sync();begin=time.monotonic()
        output,mse,rec=forward(row,arrays,'dual',fast)
        require(mse is not None and mse.requires_grad and rec['MSE_elements']==(1200-rec['slow_points'])*10*384,'Teacher supervision differs')
        label=torch.from_numpy(arrays['gt']).unsqueeze(0).to(device=device,dtype=torch.float32)
        mask=torch.from_numpy(arrays['mask']).unsqueeze(0).unsqueeze(0).to(device=device,dtype=torch.float32)
        require(int(mask.sum())>0,'Empty GT frame: no resampling')
        task_loss=loss_fn(output['flow'],label,mask,gamma=None)
        loss=mse if arm=='R0' else task_loss
        require(loss.requires_grad and bool(torch.isfinite(loss)) and bool(torch.isfinite(task_loss)),'Invalid recovery loss/gradient path')
        flow_details=[{'shape':list(x.shape),'requires_grad':x.requires_grad} for x in output['flow']]
        sync();forward_seconds=time.monotonic()-begin;back_begin=time.monotonic()
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in fast.parameters()),'Missing/nonfinite fast gradient')
        require(all(p.grad is None for p in model.parameters()),'Frozen backbone received parameter gradients')
        grad_norm=float(torch.sqrt(sum(p.grad.double().square().sum() for p in fast.parameters())))
        if step==0:require(grad_norm>0,'Zero first-step fast gradient: recovery path not qualified')
        optimizer.step()
        require(all(bool(torch.isfinite(p).all()) for p in fast.parameters()) and
                all(not isinstance(v,torch.Tensor) or bool(torch.isfinite(v).all()) for state in optimizer.state.values() for v in state.values()),
                'Nonfinite optimizer state/weight')
        sync();back_seconds=time.monotonic()-back_begin
        record={'step':step+1,'target_id':row['target_id'],'epoch0':step//20,'objective':arm,
                'optimized_loss':float(loss.detach()),'teacher_MSE':float(mse.detach()),'native_task_loss':float(task_loss.detach()),
                'fast_grad_L2':grad_norm,'forward_seconds':forward_seconds,'backward_optimizer_seconds':back_seconds,
                'native_output_flows':flow_details,'branch_service':rec,'valid_GT_pixels':int(mask.sum())}
        result['steps'].append(record)
        del output,mse,task_loss,loss,label,mask
        functional.reset_net(model)
        if step==0 or (step+1)%25==0:
            write(a.output_dir/'results.json',result);print(json.dumps({'arm':arm,'step':step+1,'loss':record['optimized_loss']}),flush=True)
    result.update(completed_optimizer_steps=steps,training_performed=True,training_seconds=time.monotonic()-train_start,
        max_allocated_bytes=torch.cuda.max_memory_allocated(device),max_reserved_bytes=torch.cuda.max_memory_reserved(device))
    result['final_numerical_pair']=numerical_pair(fit[0],fit_cache[fit[0]['target_id']][0],fast)
    if not smoke:
        warm=FastBranch(384).to(device)
        warm.load_state_dict(weights['stages']['2']['state_dict'],strict=True);warm.eval().requires_grad_(False)
        require(source_digest(warm)==initial,'Warm comparator initialization differs')
        meters={'warmstart':metrics.MetricAccumulator(),'recovered':metrics.MetricAccumulator()};eval_records=[]
        with torch.no_grad():
            for row in dev:
                arrays,evidence=read(row);masks=metrics.partitions(arrays['gt'],arrays['mask'],arrays['events'],regions)
                frame={'target_id':row['target_id'],'sequence':row['sequence'],'input_evidence':evidence,'arms':{}}
                predictions={}
                for name,active in [('warmstart',warm),('recovered',fast)]:
                    output,_,rec=forward(row,arrays,'actual',active)
                    pred=output['flow'][-1][0].detach().cpu().numpy().copy();predictions[name]=pred
                    frame['arms'][name]={'metrics':meters[name].update(row['target_id'],row['sequence'],pred,arrays['gt'],masks),
                        'execution':rec,'prediction_sha256':hashlib.sha256(pred.tobytes()).hexdigest()}
                np.savez_compressed(a.output_dir/('dev_'+row['target_id']+'.npz'),**predictions)
                eval_records.append(frame);del output
        result['endpoint_dev8']={'scope':'Fixed training-side, backbone-seen dev8; no checkpoint/step selection; not formal test',
            'frames':eval_records,'summaries':{name:m.summary() for name,m in meters.items()}}
    require(source_digest(model)==before and base.bn_signature(model,torch)==bn_before,'Native weights/BN changed')
    require(source_digest(fast)!=initial,'No fast parameter update occurred')
    payload={'schema':'flow_fast_task_recovery_weights_v1','arm':arm,'smoke_only':smoke,'completed_optimizer_steps':steps,
        'stages':{'2':{'state_dict':{k:v.detach().cpu() for k,v in fast.state_dict().items()}}},
        'optimizer_state_dict':optimizer.state_dict(),'binding':binding,'mask_bank_sha256':a.mask_bank_sha256,
        'initial_fast_state_sha256':initial,'runtime':runtime,'recipe':RECIPE}
    torch.save(payload,a.output_dir/'weights.pt')
    result.update(status='SMOKE_COMPLETE' if smoke else 'FAST_TASK_RECOVERY_COMPLETE',arm=arm,
        original_model_unchanged=True,BN_buffers_unchanged=True,weights_sha256=sha(a.output_dir/'weights.pt'),
        final_fast_state_sha256=source_digest(fast),wall_seconds=time.monotonic()-started,
        full_DynamicSwin_reproduction_claim=False,flow_specific_increment_claim=False,hardware_benefit_claim=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-contract',type=Path,required=True);parser.add_argument('--parent-contract-sha256',required=True)
    parser.add_argument('--fit-manifest',type=Path,required=True);parser.add_argument('--fit-manifest-sha256',required=True)
    parser.add_argument('--mode',choices=['prepare','smoke_R0','smoke_R1','R0','R1'],required=True)
    parser.add_argument('--mask-bank',type=Path);parser.add_argument('--mask-bank-sha256')
    parser.add_argument('--smoke-result',type=Path);parser.add_argument('--smoke-result-sha256')
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--preflight-output',type=Path)
    a=parser.parse_args();p=prepare(a)
    result={'schema':'flow_fast_task_recovery_results_v1','status':'PREFLIGHT_PASS_NO_INFERENCE','mode':a.mode,
        'binding':p[-2],'recipe':RECIPE,'native_preflight':p[2][-1],
        'mask_bank_file':str(a.mask_bank) if a.mask_bank else None,'mask_bank_sha256':a.mask_bank_sha256,
        'smoke_result_sha256':a.smoke_result_sha256,
        'scope':'Fixed intermediate checkpoint ordinary fast recovery; no new gate, no finalvalid, no method failure/ASIC claim',
        'training_performed':False,'inference_executed':False}
    if not a.execute:
        require(a.preflight_output and not a.preflight_output.exists(),'New preflight output required')
        write(a.preflight_output,result);print(result['status']);return
    require(GPU_LOCK.parent.is_dir(),'Existing shared GPU lock directory required')
    with GPU_LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # No device/model work occurs before the shared lock has been acquired.
        a.output_dir.mkdir(parents=True,exist_ok=False)
        try:
            result['inference_executed']=True;execute(a,p,result)
        except BaseException as exc:
            result.update(status='INCOMPLETE_NO_SCIENTIFIC_CONCLUSION',error=type(exc).__name__+': '+str(exc),
                          traceback=traceback.format_exc());raise
        finally:write(a.output_dir/'results.json',result)


if __name__=='__main__':main()
