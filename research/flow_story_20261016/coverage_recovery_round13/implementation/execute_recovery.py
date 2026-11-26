"""R12 execute loop adapted only for explicit R13 order, binding and new BN.

Source provenance is frozen in inherited_pins.json. Shared mathematical helpers
are imported from the immutable R12 runner by run_coverage_recovery.py.
"""
import hashlib
import inspect
import json
from pathlib import Path
import time

def execute(a,prepared,result,collector,probe,parent,r9,r8):
    import numpy as np
    import torch
    inherited,fit,checks,binding,manifest=prepared
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
    bn_info=apply_fresh_bn(model,a.bn_buffers,a.bn_receipt,manifest,binding['membership_sha256'],scope,base,torch)
    require(bn_info['scope']=='MLP_R13_COMMON_CMA176' and bn_info['BN_count']==78 and bn_info['MLP_BN_count']==24
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
    result.update(status='RUNNING',arm=a.arm,mode=a.mode,condition=a.condition,training_performed=False,completed_optimizer_steps=0,
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
    cache={}
    # Cache each identity once; fit is already the exact frozen 200-step order.
    for row in train_rows:
        if row['target_id'] not in cache:cache[row['target_id']]=read(row)
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
        deadline();row=train_rows[index];arrays,_=cache[row['target_id']]
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
        result['steps'].append({'step':index+1,'sequence':row['sequence'],'target_id':row['target_id'],'task_loss':float(loss.detach()),
            'valid_GT_pixels':int(mask.sum()),'forward_seconds':forward_time,'backward_optimizer_seconds':back_time,
            'synchronized_train_step_seconds':time.monotonic()-begin,'preclip_grad_norm':preclip,'gradient_audit':grad,
            'service':execution,'max_allocated_bytes_so_far':torch.cuda.max_memory_allocated(device),
            'max_reserved_bytes_so_far':torch.cuda.max_memory_reserved(device)})
        result.update(training_performed=True,completed_optimizer_steps=index+1)
        del output,loss,label,mask,chunk
        optimizer.zero_grad(set_to_none=True);functional.reset_net(model)
        if index==0 or (index+1)%25==0 or index==steps-1:
            parent.write(a.output_dir/'results.json',result)
            print(json.dumps({'condition':a.condition,'arm':a.arm,'mode':a.mode,'step':index+1,'loss':result['steps'][-1]['task_loss']}),flush=True)
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
        torch.save({'schema':'flow_round13_recovered_shared_native_state_v1','model_state_dict':{n:v.detach().cpu() for n,v in model.state_dict().items()},
            'optimizer_state_dict':optimizer.state_dict(),'binding':binding,'completed_optimizer_steps':200,
            'trainable_names':list(selected),'initial_state_sha256':initial,'final_state_sha256':state_digest(model),
            'BN_scope':'MLP_R13_COMMON_CMA176','BN_buffers_unchanged':True,'rng_state':{'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all()}},path)
        result.update(weights_file=path.name,weights_sha256=sha(path),smoke_weights_discarded=False)
    else:result['smoke_weights_discarded']=True
    result.update(status='ROUND13_SMOKE_COMPLETE' if a.mode=='smoke' else 'ROUND13_COVERAGE_RECOVERY_COMPLETE',
        frozen_state_unchanged=True,BN_buffers_unchanged=True,original_modules_restored=True,all_hooks_removed=True,
        updated_parameter_names=changed,final_state_sha256=state_digest(model),wall_seconds=time.monotonic()-started,
        hardware_benefit_claim=False,flow_specific_increment_claim=False)

