#!/usr/bin/env python3
"""Four fair ordinary layouts, separate 5-step smoke or200-step recovery. Default preflight."""
import argparse
import contextlib
import copy
import fcntl
import hashlib
import inspect
import json
from pathlib import Path
import signal
import sys
import time
import traceback
sys.dont_write_bytecode = True
from source_contract import HERE, require, sha, source_path, verify_all, load_source
import trainable_representative_mlp as tile

ARMS = tile.ARMS
LOSS_SHA = '92aa1451469690ad89c16b6954c972d3c460689f5cb765662b4eff7fc7dec122'
MEMBER_SHA = 'cfc84c52158213599717d1aaafb19c73506ff08edbf54831d2b839bfa9da75d5'
SELECTION_SHA = 'a6c6e7fb8daabacfa705a40d8fb7fbf11fcdb934f23f919a5917bfd26b5b0df1'
PARAMETERS = 7252064
SCHEMA = 'flow_round12_fair_recovery_v1'
ROWS = {'native':558000, 'FF':558000, 'CC':153600, 'fine_s2b2':164200}
OWN = ('run_recovery.py', 'trainable_representative_mlp.py', 'source_contract.py', 'test_recovery_cpu.py')
RECIPE = {'seed':20261016, 'smoke_steps':5, 'recovery_steps':200, 'fit_targets':20, 'batch':1, 'T':10,
    'input_hw':[480,640], 'augmentation':'none', 'fit_order':'original fixed fit20 repeated10; smoke first5 only',
    'optimizer':'AdamW', 'lr':0.00025, 'weight_decay':0.01, 'betas':[0.9,0.999], 'eps':1e-8,
    'clip_grad_norm':100., 'scheduler':None, 'precision':'FP32 TF32off noAMP',
    'trainable':'all parameters in exactly stage0 two +stage2 six nativeMLPs, shared fine/coarse',
    'BN':'MLP_CMA176 private24 fixed buffers/affine trainable in selected8; external54 eval/current-stat',
    'loss':'native four flows equally averaged all-valid supervised AEE; gamma=None,flow_scaling1,lambda_mod1',
    'check_used_for_training_or_checkpoint_selection':False}


def load_runtime():
    verify_all()
    collector = load_source('collector', '_r12_frozen_collector')
    probe = collector.load_probe(); parent = probe.load_scope_runner()
    r9 = load_source('r9_runner', '_r12_frozen_anchor_adapter'); r8 = r9.load_r8()
    require(sha(r8.__file__) == sha(source_path('r8_runner')), 'R8 helper differs')
    return collector, probe, parent, r9, r8


def state_digest(model, exclude=()):
    excluded = set(exclude); digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name in excluded: continue
        value = value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode()); digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def select_parameters(model, targets, expected=PARAMETERS):
    model.eval().requires_grad_(False)
    names = [name for name in targets if tile.is_selected_layer(name)]
    require(len(names) == 8, 'Exactly8 stage02 MLPs required')
    allowed = {name+'.'+key for name in names for key, _ in targets[name].named_parameters()}
    selected = {name:p for name,p in model.named_parameters() if name in allowed}
    require(set(selected) == allowed and sum(p.numel() for p in selected.values()) == expected,
            'Eight-MLP trainable name/count whitelist differs')
    for p in selected.values(): p.requires_grad_(True)
    require({n for n,p in model.named_parameters() if p.requires_grad} == allowed, 'Extra trainable parameter')
    return selected


def expected_rows(shapes, arm, dual=False):
    import math
    rows=0
    for name,(t,b,h,w,c) in shapes.items():
        if arm == 'native' or not tile.is_selected_layer(name): rows += t*b*h*w; continue
        reps=sum(math.ceil(min(9,h-y)/3)*math.ceil(min(9,w-x)/3) for y in range(0,h,9) for x in range(0,w,9))
        rows += t*b*(h*w+reps if dual else h*w if tile.fine_layer(arm,name) else reps)
    return rows


def forward_train(model, functional, targets, shapes, chunk, arm, *, capture=False, mode='actual'):
    """Autograd is controlled by caller; outputs retain the complete downstream graph."""
    import torch
    functional.reset_net(model)
    observed = {n:{'fc1_rows':0,'fc2_rows':0} for n in targets}
    handles=[]; branches={}; inputs={}
    try:
        for name,module in targets.items():
            for fc in ('fc1','fc2'):
                def count(m,args,n=name,key=fc+'_rows'): observed[n][key] += args[0].numel()//args[0].shape[-1]
                handles.append(getattr(module,fc).register_forward_pre_hook(count))
        with tile.wrapped(model,targets,shapes,arm,mode) as wrappers:
            selected = {n:m for n,m in targets.items() if tile.is_selected_layer(n)} if arm == 'native' else wrappers
            if capture:
                for name,module in selected.items():
                    def store(m,args,out,n=name):
                        require(n not in branches, 'One branch per audit root call')
                        branches[n]=out.detach().cpu().clone(); inputs[n]=hashlib.sha256(args[0].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                    handles.append(module.register_forward_hook(store))
            output=model(chunk)
            require(isinstance(output['flow'],(list,tuple)) and len(output['flow'])>0 and
                    all(bool(torch.isfinite(x).all()) for x in output['flow']), 'Finite complete flow list required')
            records={n:w.records[0] for n,w in wrappers.items()}
        require(all(observed[n]['fc1_rows']==observed[n]['fc2_rows'] for n in observed), 'FC1/FC2 rows differ')
        total=sum(x['fc1_rows'] for x in observed.values())
        require(total==expected_rows(shapes,arm,mode=='dual_reference'), 'Independent differentiable FC rows differ')
        for name,record in records.items():
            require(observed[name]['fc1_rows']==record['executed_fc_rows_per_layer'], 'Observed wrapper rows differ')
        return output, {'arm':arm,'mode':mode,'layers':records,'all12_observed_fc_rows':observed,
            'summed_fc1_rows':total,'summed_fc2_rows':total,'capture_input_sha256':inputs,
            'gradient_enabled':torch.is_grad_enabled(),'full_flow_count':len(output['flow'])}, branches
    finally:
        for h in handles:h.remove()


@contextlib.contextmanager
def frozen_parameters(model):
    old={p:p.requires_grad for p in model.parameters()}
    try:
        for p in old:p.requires_grad_(False)
        yield
    finally:
        for p,value in old.items():p.requires_grad_(value)


def reference_forward(r9,r8,parent,model,functional,targets,shapes,chunk,arm,merge,torch,**kwargs):
    branches={}; inputs={}; handles=[]
    try:
        if arm=='native':
            for name,module in targets.items():
                if not tile.is_selected_layer(name):continue
                def store(m,args,out,n=name):
                    require(n not in branches,'Repeated native reference branch')
                    branches[n]=out.detach().cpu().clone();inputs[n]=merge.tensor_sha(args[0])
                handles.append(module.register_forward_hook(store))
        with frozen_parameters(model),torch.no_grad():
            flows,execution,captured=r9.forward_arm(r8,parent,model,functional,targets,shapes,chunk,arm,merge,torch,
                                                   capture=arm!='native',**kwargs)
        if arm!='native':
            branches=captured;inputs={n:r['input_sha256'] for n,r in execution['layers'].items()}
        require(execution['summed_fc1_rows']==expected_rows(shapes,arm,kwargs.get('mode')=='dual_reference'), 'Reference rows differ')
        return flows,execution,branches,inputs
    finally:
        for h in handles:h.remove()


def compare_all(flows,branches,inputs,ref,torch):
    rflows,rex,rbranches,rinputs=ref
    require(len(flows)==len(rflows) and len(flows)>0 and all(torch.equal(x.detach().cpu(),y) for x,y in zip(flows,rflows)),
            'Full flow list numerical mismatch')
    require(set(branches)==set(rbranches)==set(inputs)==set(rinputs) and len(branches)==8,'All8 branch/input audit required')
    require(inputs==rinputs and all(torch.equal(branches[n],rbranches[n]) for n in branches),'Branch/input numerical mismatch')
    return {'all_flow_outputs_exact':True,'flow_count':len(flows),'all8_inputs_exact':True,'all8_branches_exact':True}


def gradient_audit(model,selected,torch):
    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in selected.values()), 'Missing/nonfinite allowed gradient')
    require(all(p.grad is None for n,p in model.named_parameters() if n not in selected),'Frozen parameter gradient is notNone')
    groups={'FC':[],'PSN':[],'BN_affine':[]}; per_layer={}
    for name,p in selected.items():
        category='FC' if '.fc1.' in name or '.fc2.' in name else 'PSN' if '.sn1.' in name or '.sn2.' in name else 'BN_affine'
        norm2=float(p.grad.detach().double().square().sum())
        groups[category].append((name,norm2));layer=name.split('.mlp.')[0]+'.mlp';per_layer[layer]=per_layer.get(layer,0.)+norm2
    require(set(groups)=={'FC','PSN','BN_affine'} and all(v for v in groups.values()),'Expected gradient categories missing')
    return {'all_allowed_gradients_present_finite':True,'frozen_gradients_None':True,
        'category_squared_L2':{k:sum(x[1] for x in v) for k,v in groups.items()},
        'category_nonzero_tensors':{k:sum(x[1]>0 for x in v) for k,v in groups.items()},'per_layer_squared_L2':per_layer}


def gradient_coverage(steps, selected_layers, require_complete=False):
    layers={n:sum(s['gradient_audit']['per_layer_squared_L2'][n]>0 for s in steps) for n in selected_layers}
    categories={n:sum(s['gradient_audit']['category_squared_L2'][n]>0 for s in steps) for n in ('FC','PSN','BN_affine')}
    complete=all(layers.values()) and all(categories.values())
    if require_complete:require(complete,'Five-step gradient coverage incomplete; diagnostic only, no quality conclusion')
    return {'layer_steps_nonzero':layers,'category_steps_nonzero':categories,'complete':bool(complete),
            'per_step_zero_is_allowed':True,'scope':'At least one nonzero step per selected layer and parameter category'}


def validate_smokes(paths,binding):
    require(len(paths)==4,'Recovery requires four completed fixed smoke receipts')
    seen=set()
    for path in paths:
        value=json.loads(path.read_text());arm=value.get('arm');require(arm in ARMS and arm not in seen,'Duplicate/unknown smoke arm');seen.add(arm)
        require(value.get('schema')==SCHEMA and value.get('status')=='ROUND12_SMOKE_COMPLETE' and
                value.get('completed_optimizer_steps')==5 and value.get('training_performed') is True,'Incomplete smoke')
        fields=('runner_sha256','wrapper_sha256','runner_checks_sha256','source_pins_sha256','checkpoint_sha256',
                'loss_sha256','membership_sha256','selection_sha256','allowed_parameter_count')
        require(all(value['binding'][key]==binding[key] for key in fields) and value['binding']['recipe']==RECIPE
                and value['binding']['arm']==arm and value['binding']['mode']=='smoke','Smoke source/recipe differs')
        require(value['frozen_state_unchanged'] and value['BN_buffers_unchanged'] and value['original_modules_restored']
                and value['all_hooks_removed'] and value['smoke_weights_discarded'],'Smoke closure missing')
        require(value['initial_numerical_pair']['all_flow_outputs_exact'] and value['final_numerical_pair']['all_flow_outputs_exact'],'Smoke numerics missing')
        require(value['gradient_coverage']['complete'],'Smoke gradient coverage missing')
    return {p.name+':'+json.loads(p.read_text())['arm']:sha(p) for p in paths}


def prepare(a,collector,probe,parent):
    require(0<a.wall_minutes<=30,'Fixed30minute cap')
    require(sha(HERE/'check_membership_v1.json')==MEMBER_SHA and sha(a.selection_manifest)==a.selection_manifest_sha256==SELECTION_SHA,
            'Frozen R12 membership/selection differs')
    member=json.loads((HERE/'check_membership_v1.json').read_text())
    manifest=json.loads(a.selection_manifest.read_text());require(manifest['groups']==member['groups'],'Selection view differs')
    inherited=argparse.Namespace(**vars(a));inherited.group='fit' if a.mode=='smoke' else 'all';inherited.runner_checks=a.collector_runner_checks
    prepared=collector.prepare(inherited,probe,parent)
    fit=[r for g,r in prepared[6] if g=='fit'];checks=[r for g,r in prepared[6] if g=='check']
    require([r['target_id'] for r in fit]==member['groups']['fit'] and len(fit)==20,'Exact original fit20 order required')
    require(not checks if a.mode=='smoke' else [r['target_id'] for r in checks]==member['groups']['check'],'Evaluation group differs')
    receipt=parent.checked_cpu_receipt(a.runner_checks,[HERE/n for n in OWN])
    b=prepared[1];source=json.loads(b.source_manifest.read_text())
    require(source['loss/flow_supervised.py']==LOSS_SHA and sha(b.source_root/'loss/flow_supervised.py')==LOSS_SHA,'Actual deployed loss/source manifest differs')
    binding={'runner_sha256':sha(__file__),'wrapper_sha256':sha(HERE/'trainable_representative_mlp.py'),
        'runner_checks_sha256':receipt,'source_pins_sha256':sha(HERE/'source_pins.json'),
        'checkpoint_sha256':collector.CHECKPOINT_SHA,'membership_sha256':MEMBER_SHA,'selection_sha256':SELECTION_SHA,
        'recipe':RECIPE,'loss_sha256':LOSS_SHA,'allowed_parameter_count':PARAMETERS,'arm':a.arm,'mode':a.mode,
        'fit_ids':[r['target_id'] for r in fit],'check_ids':[] if a.mode=='smoke' else [r['target_id'] for r in checks],
        'inherited_data_and_numerical_contract':prepared[7],'GPU_lock':str(parent.GPU_LOCK),'wall_minutes':a.wall_minutes,
        'expected_training_forwards':5 if a.mode=='smoke' else 200,
        'expected_audit_forwards':4 if a.arm=='native' else 6,'expected_evaluation_forwards':0 if a.mode=='smoke' else 40,
        'scope':'Fair ordinary shared-weight representative-MLP recovery; no flow-specific mechanism, selection tuning or hardware claim'}
    if a.mode=='recover':binding['smoke_receipts_sha256']=validate_smokes(a.smoke_receipt,binding)
    return prepared,fit,checks,binding


def execute(a,prepared,result,collector,probe,parent,r9,r8):
    import numpy as np
    import torch
    inherited,fit,checks,binding=prepared
    c,b,context,base,scope,loader,chosen,oldbinding,oldhead=inherited
    require(not torch.cuda.is_initialized(),'CUDA before shared GPU lock')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(RECIPE['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.manual_seed_all(RECIPE['seed'])
    started=time.monotonic()
    def deadline():require(time.monotonic()-started<a.wall_minutes*60,'Fixed arm wall budget exhausted')
    def sync():torch.cuda.synchronize(device)
    native,trainer,metrics,dev32,hashes,config,saved,side,regions,plan=context
    model,functional=loader.load_model(b,context,base,torch,device)
    bn_info=scope.apply_scope(model,'MLP_CMA176',b.bn_payload,torch)
    require(bn_info['scope']=='MLP_CMA176' and bn_info['BN_count']==78 and bn_info['MLP_BN_count']==24
            and len(bn_info['frozen_BN_names'])==24 and len(bn_info['current_stat_BN_names'])==54
            and set(bn_info['frozen_BN_names'])==set(bn_info['MLP_BN_names'])
            and not(set(bn_info['frozen_BN_names']) & set(bn_info['current_stat_BN_names'])),'BN scope differs')
    bn_info['execution']='Eval/dropout off, private24 running buffers fixed; external54 current-stat. Onlystage02 eightMLP parameters trainable.'
    targets,shapes=parent.target_layers(model);merge=parent.load_frozen_merge()
    selected=select_parameters(model,targets);initial=state_digest(model);frozen_before=state_digest(model,selected);bn_before=base.bn_signature(model,torch)
    parameter_initial={n:hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for n,p in selected.items()}
    from loss.flow_supervised import flow_loss_supervised
    require(Path(inspect.getfile(flow_loss_supervised)).resolve()==(b.source_root/'loss/flow_supervised.py').resolve()
            and sha(inspect.getfile(flow_loss_supervised))==LOSS_SHA,'Wrong imported loss')
    require(config['metrics']['flow_scaling']==1 and config['loss']['lambda_mod']==1 and config['loss']['gamma'] is None,'Native loss config differs')
    loss_fn=flow_loss_supervised(config,device)
    optimizer=torch.optim.AdamW(list(selected.values()),lr=RECIPE['lr'],weight_decay=RECIPE['weight_decay'],
                                betas=tuple(RECIPE['betas']),eps=RECIPE['eps'])
    result.update(status='RUNNING',arm=a.arm,mode=a.mode,training_performed=False,completed_optimizer_steps=0,
        initial_model_state_sha256=initial,frozen_state_before_sha256=frozen_before,BN_scope=bn_info,
        allowed_parameter_names=list(selected),allowed_parameter_count=sum(p.numel() for p in selected.values()),
        full_forward_count=0,forward_records=[],steps=[],evaluations={},runtime={'torch':str(torch.__version__),
        'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'NumPy':np.__version__,
        'device_name':torch.cuda.get_device_name(device),'device_index':torch.cuda.current_device(),
        'TF32':False,'AMP':False,'threads':2,'grad_enabled_training':True},native_loss={'sha256':LOSS_SHA,'gamma':None,
        'flow_scaling':1,'lambda_mod':1,'all_four_flows_used':True,'GT_event_filter':False})
    def read(row):
        expected={};evidence={}
        for key,relative in native.target_files(row).items():
            expected['events' if key=='input' else key]=hashes[relative]['sha256']
        arrays,_=metrics.load_arrays(b.data_root/'saved_flow_data',row,evidence=evidence,expected_hashes=expected)
        return arrays,evidence
    train_rows=fit[:5] if a.mode=='smoke' else fit
    cache={r['target_id']:read(r) for r in train_rows}
    result['training_input_evidence']={k:v[1] for k,v in cache.items()}
    def log_forward(purpose,row,execution):
        result['full_forward_count']+=1
        result['forward_records'].append({'forward_id':result['full_forward_count'],'purpose':purpose,
            'target_id':row['target_id'],'execution':execution})
        cap=sum(binding[k] for k in ('expected_training_forwards','expected_audit_forwards','expected_evaluation_forwards'))
        require(result['full_forward_count']<=cap,'Full-forward accounting cap exceeded')
    def numerical_pair(row,arrays,label):
        deadline();chunk=base.prepare_input(arrays['events'],device,torch)
        with torch.no_grad():
            output,ex,branches=forward_train(model,functional,targets,shapes,chunk,a.arm,capture=True)
            flows=[x.detach().cpu().clone() for x in output['flow']];del output
            require(len(flows)==4,'Exactly4 native flow heads required')
            log_forward('audit_new_'+label,row,ex)
            ref=reference_forward(r9,r8,parent,model,functional,targets,shapes,chunk,a.arm,merge,torch,deadline=deadline)
            log_forward('audit_old_actual_'+label,row,ref[1]);comparison=compare_all(flows,branches,ex['capture_input_sha256'],ref,torch)
            rec={'new_execution':ex,'old_actual_execution':ref[1],**comparison}
            del ref
            if a.arm!='native':
                dual=reference_forward(r9,r8,parent,model,functional,targets,shapes,chunk,a.arm,merge,torch,
                                       mode='dual_reference',deadline=deadline)
                log_forward('audit_old_dual_'+label,row,dual[1]);compare_all(flows,branches,ex['capture_input_sha256'],dual,torch)
                rec.update(old_dual_execution=dual[1],actual_dual_all_flows_branches_exact=True)
                del dual
            else:rec['native_has_no_tiled_dual']=True
        require(all(p.requires_grad for p in selected.values()),'Audit failed to restore gradient flags')
        require(base.bn_signature(model,torch)==bn_before,'Audit changed BN buffers')
        del chunk,branches,flows
        return rec
    def evaluate(label,modes):
        frames=[];meters={arm:metrics.MetricAccumulator() for arm in modes}
        for row in checks:
            deadline();arrays,evidence=read(row);masks=metrics.partitions(arrays['gt'],arrays['mask'],arrays['events'],regions)
            chunk=base.prepare_input(arrays['events'],device,torch);frame={'target_id':row['target_id'],'sequence':row['sequence'],'input_evidence':evidence,'arms':{}}
            predictions={}
            for arm in modes:
                ref=reference_forward(r9,r8,parent,model,functional,targets,shapes,chunk,arm,merge,torch,deadline=deadline)
                log_forward('evaluation_'+label,row,ref[1]);prediction=ref[0][-1][0].numpy().copy();predictions[arm]=prediction
                frame['arms'][arm]={'metrics':meters[arm].update(row['target_id'],row['sequence'],prediction,arrays['gt'],masks),
                    'prediction_sha256':hashlib.sha256(prediction.tobytes()).hexdigest(),'execution':ref[1]}
                del ref
            path=a.output_dir/(label+'_'+row['target_id']+'.npz');np.savez_compressed(path,**predictions)
            frame.update(prediction_file=path.name,prediction_file_sha256=sha(path));frames.append(frame)
            del chunk,predictions
        result['evaluations'][label]={'frames':frames,'summaries':{arm:m.summary() for arm,m in meters.items()}}
    result['initial_numerical_pair']=numerical_pair(fit[0],cache[fit[0]['target_id']][0],'initial')
    if a.mode=='recover':evaluate('initial',[a.arm])
    result['initial_audit_peak_bytes']={'allocated':torch.cuda.max_memory_allocated(device),
                                       'reserved':torch.cuda.max_memory_reserved(device)}
    torch.cuda.reset_peak_memory_stats(device);sync();train_start=time.monotonic()
    steps=5 if a.mode=='smoke' else 200
    for index in range(steps):
        deadline();row=train_rows[index%len(train_rows)];arrays,_=cache[row['target_id']]
        optimizer.zero_grad(set_to_none=True);chunk=base.prepare_input(arrays['events'],device,torch)
        label=torch.from_numpy(arrays['gt']).unsqueeze(0).to(device=device,dtype=torch.float32)
        mask=torch.from_numpy(arrays['mask']).unsqueeze(0).unsqueeze(0).to(device=device,dtype=torch.float32)
        require(int(mask.sum())>0,'No empty-GT resampling')
        sync();begin=time.monotonic()
        with torch.enable_grad():
            output,execution,_=forward_train(model,functional,targets,shapes,chunk,a.arm)
            log_forward('training',row,execution)
            require(len(output['flow'])==4 and all(x.requires_grad for x in output['flow']),'Missing full task gradient path')
            loss=loss_fn(output['flow'],label,mask,gamma=None)
            require(loss.requires_grad and bool(torch.isfinite(loss)),'Invalid task loss')
            sync();forward_time=time.monotonic()-begin;back_start=time.monotonic()
            loss.backward();grad=gradient_audit(model,selected,torch)
            preclip=float(torch.nn.utils.clip_grad_norm_(list(selected.values()),100.,error_if_nonfinite=True))
            optimizer.step()
        require(all(bool(torch.isfinite(p).all()) for p in selected.values()),'Nonfinite trained parameter')
        require(all(not isinstance(v,torch.Tensor) or bool(torch.isfinite(v).all()) for state in optimizer.state.values() for v in state.values()),'Nonfinite optimizer state')
        sync();back_time=time.monotonic()-back_start
        result['steps'].append({'step':index+1,'target_id':row['target_id'],'task_loss':float(loss.detach()),
            'valid_GT_pixels':int(mask.sum()),'forward_seconds':forward_time,'backward_optimizer_seconds':back_time,
            'synchronized_train_step_seconds':time.monotonic()-begin,'preclip_grad_norm':preclip,'gradient_audit':grad,
            'service':execution,'max_allocated_bytes_so_far':torch.cuda.max_memory_allocated(device),
            'max_reserved_bytes_so_far':torch.cuda.max_memory_reserved(device)})
        result.update(training_performed=True,completed_optimizer_steps=index+1)
        del output,loss,label,mask,chunk
        optimizer.zero_grad(set_to_none=True);functional.reset_net(model)
        if index==0 or (index+1)%25==0 or index==steps-1:
            parent.write(a.output_dir/'results.json',result)
            print(json.dumps({'arm':a.arm,'mode':a.mode,'step':index+1,'loss':result['steps'][-1]['task_loss']}),flush=True)
    result.update(training_seconds=time.monotonic()-train_start,max_allocated_bytes=torch.cuda.max_memory_allocated(device),
        max_reserved_bytes=torch.cuda.max_memory_reserved(device),
        peak_fields_scope='Training segment only; input preparation and telemetry are included in training_seconds, not synchronized_train_step_seconds')
    result['gradient_coverage']=gradient_coverage(result['steps'],[n for n in targets if tile.is_selected_layer(n)],a.mode=='smoke')
    result['final_numerical_pair']=numerical_pair(fit[0],cache[fit[0]['target_id']][0],'final')
    if a.mode=='recover':evaluate('endpoint',ARMS)
    result['full_entry_peak_bytes']={
        'allocated':max(result['initial_audit_peak_bytes']['allocated'],torch.cuda.max_memory_allocated(device)),
        'reserved':max(result['initial_audit_peak_bytes']['reserved'],torch.cuda.max_memory_reserved(device))}
    require(state_digest(model,selected)==frozen_before,'Frozen-state complement changed')
    require(base.bn_signature(model,torch)==bn_before,'Private/current BN buffers changed')
    changed=[n for n,p in selected.items() if hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()!=parameter_initial[n]]
    require(changed and all(any(n.startswith(layer+'.') for n in changed) for layer in targets if tile.is_selected_layer(layer)),
            'At least one selectedMLP did not update')
    require(all(model.get_submodule(n) is m for n,m in targets.items()),'Original modules not restored')
    require(all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules()),'Audit hooks remain')
    require(result['full_forward_count']==sum(binding[k] for k in ('expected_training_forwards','expected_audit_forwards','expected_evaluation_forwards')),
            'Complete fixed forward count differs')
    if a.mode=='recover':
        path=a.output_dir/'weights.pt'
        torch.save({'schema':'flow_round12_recovered_shared_native_state_v1','model_state_dict':{n:v.detach().cpu() for n,v in model.state_dict().items()},
            'optimizer_state_dict':optimizer.state_dict(),'binding':binding,'completed_optimizer_steps':200,
            'trainable_names':list(selected),'initial_state_sha256':initial,'final_state_sha256':state_digest(model),
            'BN_scope':'MLP_CMA176','BN_buffers_unchanged':True,'rng_state':{'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all()}},path)
        result.update(weights_file=path.name,weights_sha256=sha(path),smoke_weights_discarded=False)
    else:result['smoke_weights_discarded']=True
    result.update(status='ROUND12_SMOKE_COMPLETE' if a.mode=='smoke' else 'ROUND12_FAIR_RECOVERY_COMPLETE',
        frozen_state_unchanged=True,BN_buffers_unchanged=True,original_modules_restored=True,all_hooks_removed=True,
        updated_parameter_names=changed,final_state_sha256=state_digest(model),wall_seconds=time.monotonic()-started,
        hardware_benefit_claim=False,flow_specific_increment_claim=False)


def main(argv=None):
    collector,probe,parent,r9,r8=load_runtime()
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent-contract','identity-bank','prior-screen-results','prior-scope-results','selection-manifest','fit-membership','prior-check-membership'):
        parser.add_argument('--'+name,type=Path,required=True);parser.add_argument('--'+name+'-sha256',required=True)
    for name,default in (('module-checks',probe.BASE/'cpu_results.json'),('scope-runner-checks',probe.BASE/'scope_extension_v1/runner_cpu_results.json'),
                          ('probe-runner-checks',probe.HERE/'runner_cpu_results.json'),('collector-runner-checks',source_path('collector_receipt')),
                          ('runner-checks',HERE/'runner_cpu_results.json')):parser.add_argument('--'+name,type=Path,default=default)
    parser.add_argument('--mode',choices=('smoke','recover'),required=True);parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--smoke-receipt',type=Path,action='append',default=[])
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--preflight-output',type=Path)
    parser.add_argument('--wall-minutes',type=float,default=30);parser.add_argument('--execute',action='store_true')
    a=parser.parse_args(argv);prepared=prepare(a,collector,probe,parent)
    result={'schema':SCHEMA,'status':'PREFLIGHT_PASS_NO_INFERENCE','binding':prepared[-1],'training_performed':False,'inference_executed':False}
    if not a.execute:
        require(a.preflight_output and not a.preflight_output.exists(),'Fresh preflight output required')
        parent.write(a.preflight_output,result);print(result['status']);return
    require(parent.GPU_LOCK.parent.is_dir(),'Existing shared lock directory required')
    with parent.GPU_LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);a.output_dir.mkdir(parents=True,exist_ok=False)
        previous=signal.getsignal(signal.SIGALRM)
        def timeout(signum,frame):raise TimeoutError('R12 fixed arm budget exhausted')
        signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,a.wall_minutes*60)
        try:
            result['inference_executed']=True;execute(a,prepared,result,collector,probe,parent,r9,r8)
        except BaseException as exc:
            result.update(status='INCOMPLETE_NO_SCIENTIFIC_CONCLUSION',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous);parent.write(a.output_dir/'results.json',result)


if __name__=='__main__':main()
