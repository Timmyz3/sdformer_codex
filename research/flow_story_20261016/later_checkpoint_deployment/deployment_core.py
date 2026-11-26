"""Small zero-optimizer helpers for the fixed later-checkpoint diagnostic."""
import contextlib
import hashlib
import types


def require(ok,message):
    if not ok:raise ValueError(message)


def plan(members):
    require(len(members['bn_members'])==176 and len(members['check_members'])==8,'Fixed176+8 required')
    out=[{'phase':'calibration','target_id':r['target_id'],'layout':'native','mode':'actual'} for r in members['bn_members']]
    for i,row in enumerate(members['check_members']):
        for layout in ('native','FF'):
            item={'phase':'evaluation','target_id':row['target_id'],'layout':layout,'mode':'actual'};out.append(item)
            if i==0 and layout=='FF':out.append(dict(item,mode='dual_reference'))
    require(len(out)==193,'193 forwards only')
    return [dict(v,forward_id=i+1) for i,v in enumerate(out)]


def native_calibration_forward(model,functional,targets,shapes,chunk,torch):
    """Only native call; BN training is allowed here, no tiled wrapper involved."""
    observed={n:{'fc1_rows':0,'fc2_rows':0} for n in targets};handles=[]
    try:
        for name,module in targets.items():
            for field in ('fc1','fc2'):
                def count(layer,args,n=name,key=field+'_rows'):observed[n][key]+=args[0].numel()//args[0].shape[-1]
                handles.append(getattr(module,field).register_forward_pre_hook(count))
        functional.reset_net(model)
        with torch.inference_mode():
            flows=model(chunk)['flow']
            require(len(flows)==4 and all(bool(torch.isfinite(v).all()) for v in flows),'Four finite calibration heads required')
            heads=[{'shape':list(v.shape),'dtype':str(v.dtype),'finite':True} for v in flows]
        for name,(t,b,h,w,c) in shapes.items():
            require(observed[name]=={'fc1_rows':t*b*h*w,'fc2_rows':t*b*h*w},'Calibration actual native rows differ')
        total=sum(v['fc1_rows'] for v in observed.values())
        return {'all12_observed_fc_rows':observed,'summed_fc1_rows':total,'summed_fc2_rows':total,
                'head_records':heads,'layout':'native','BN_mode':'all78_reset_CMA_train_nonBN_eval','gradient_enabled':False}
    finally:
        for handle in handles:handle.remove()


def finish_calibration(model,all_bn,mlp_bn,cma,non_bn_before,torch):
    require(len(all_bn)==78 and len(mlp_bn)==24,'Exact78/24 BN required')
    require(cma.non_bn_digest(model,all_bn)==non_bn_before,'Calibration changed non-BN state')
    values={}
    for name,layer in all_bn.items():
        require(layer.training and layer.track_running_stats and layer.momentum is None,'Native CMA reset contract changed')
        require(int(layer.num_batches_tracked)==176 and bool(torch.isfinite(layer.running_mean).all())
                and bool(torch.isfinite(layer.running_var).all()) and bool((layer.running_var>=0).all()),'Incomplete/nonfinite native CMA')
        if name in mlp_bn:
            for key in ('running_mean','running_var','num_batches_tracked'):values[name+'.'+key]=getattr(layer,key).detach().cpu().numpy().copy()
    require(len(values)==72,'Exactly72 new retained buffers')
    # No imported BN payload is accepted. These values come only from this model.
    model.eval().requires_grad_(False)
    for name,layer in all_bn.items():
        if name not in mlp_bn:
            layer.track_running_stats=False;layer.running_mean=layer.running_var=layer.num_batches_tracked=None
    require(all(not layer.training for layer in model.modules()),'Deployment must be eval')
    require(cma.non_bn_digest(model,all_bn)==non_bn_before,'Deployment switch changed non-BN state')
    return values


def check_closure(model,targets,state_before,digest,bn_before,bn_signature,torch):
    require(digest(model)==state_before,'Deployment inference changed full state')
    require(bn_signature(model,torch)==bn_before,'Deployment inference changed BN')
    require(all(model.get_submodule(n) is layer for n,layer in targets.items()),'Original MLP not restored')
    require(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()),'Hooks remain')
    require(all(not p.requires_grad and p.grad is None for p in model.parameters()),'Unexpected training/gradient state')
    return True


def forward_with_raw_inputs(h0,runtime,model,functional,targets,shapes,chunk,merge,torch,mode,deadline):
    """Same pinned FF math, extra whole-wrapper input CPU capture; no extra call."""
    probe=runtime[2];probe_fn=h0.adapted_probe(probe,'FF');original_wrap=probe_fn.__globals__['wrap'];captured={}
    @contextlib.contextmanager
    def wrapped(*args,**kwargs):
        handles=[]
        with original_wrap(*args,**kwargs) as wrappers:
            try:
                for name,wrapper in wrappers.items():
                    def save(module,inputs,n=name):
                        require(n not in captured,'Duplicate whole-MLP raw input')
                        captured[n]=inputs[0].detach().cpu().clone()
                    handles.append(wrapper.register_forward_pre_hook(save))
                yield wrappers
            finally:
                for handle in handles:handle.remove()
    namespace=dict(probe_fn.__globals__,wrap=wrapped)
    captured_probe=types.FunctionType(probe_fn.__code__,namespace,probe_fn.__name__,probe_fn.__defaults__,probe_fn.__closure__)
    def adapt(unused_probe,anchor):
        require(anchor=='FF','Raw dual capture only FF');return captured_probe
    call=types.FunctionType(h0.forward.__code__,dict(h0.forward.__globals__,adapted_probe=adapt),h0.forward.__name__,h0.forward.__defaults__)
    result=call(runtime,model,functional,targets,shapes,chunk,'FF',merge,torch,mode=mode,deadline=deadline)
    require(set(captured)==set(result[2]) and len(captured)==8,'Eight raw whole inputs required')
    require(all(merge.tensor_sha(value)==result[1]['branch_records'][n]['input_sha256'] for n,value in captured.items()),'Captured raw input source differs')
    return result,captured


def save_raw_pair_side(path,forward,inputs,np,sha):
    flows,execution,branches,_=forward;arrays={};records={};names=sorted(branches)
    for i,tensor in enumerate(flows):arrays['flow_'+str(i)]=tensor.numpy()
    for i,name in enumerate(names):
        arrays['input_'+str(i)]=inputs[name].numpy();arrays['branch_'+str(i)]=branches[name].numpy()
    require(len(flows)==4 and len(names)==8 and len(arrays)==20,'Four flow/eight input/eight branch raw arrays required')
    for key,value in arrays.items():
        require(value.dtype==np.float32 and np.isfinite(value).all(),'Raw audit tensor dtype/finite differs')
        records[key]={'shape':list(value.shape),'dtype':str(value.dtype),'payload_sha256':hashlib.sha256(value.tobytes()).hexdigest()}
    for i,name in enumerate(names):
        require(records['input_'+str(i)]['payload_sha256']==execution['branch_records'][name]['input_sha256'] and
                records['branch_'+str(i)]['payload_sha256']==execution['branch_records'][name]['output_sha256'],'Raw branch/input audit binding differs')
    require(all(records['flow_'+str(i)]['payload_sha256']==execution['head_records'][i]['sha256'] for i in range(4)),'Raw head audit binding differs')
    np.savez_compressed(path,**arrays)
    return {'file':path.name,'file_sha256':sha(path),'arrays':records,'branch_names_in_index_order':names,
            'scope':'Lossless CPU capture from the existing first FF forward; no extra model call.'}
