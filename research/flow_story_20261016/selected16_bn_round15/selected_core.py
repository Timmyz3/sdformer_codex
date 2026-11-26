"""One original W, its audited own-CMA subset, and eight native calls only."""
import hashlib
import types

STEP='271482'
BN_SCOPE='SELECTED16_FIXED_62_CURRENT'
BUFFERS=('running_mean','running_var','num_batches_tracked')
CHECKPOINT_SHA='1e72cca3946f40e24a1575f3348f9a596d744460808ad910a616ba713b5e4977'
SOURCE_SHA='ad14a7d45091aa75d64df83e1a71bd89b35dfbbf49c9ab84861d5ae0aa270560'
NON_BN_SHA='b721031729a9a5f69d03037be53e62ebf8c1536dad0ad8a8f8e1601fe9fc2798'
K_RESULT_SHA='8e0dafe172a27b6d56b484bada005ec18d15e62ae18afdc7b0bff7870ba1a841'
K_AUDIT_SHA='7455e073fcfb1ff981e228807cf917a4e4657a7ef988d072d7a16cdcbb77e9bd'
CURRENT_RESULT_SHA='c9ead13e411512f570fc6ca926fcfaaa67c2f93e04c700bdd7799c85fcb98404'
CURRENT_AUDIT_SHA='6bdcd565ccff753447f9becb7066f6670c7216841d644b174998a0a77960da71'
OWN_BN_SHA='4d887a8fce408e3a33cb95bd120d2ed3d865541a39c034205f4172ab7bdd30dc'
OWN_RECEIPT_SHA='24d33ba100150775a733fe6ecfabd4e88a1c152ca080adbf1f387a30ce574b9c'
SELECTED_STATE_SHA='6b4be9b8f136c4f978c892fff5532723828654f203c2d468e837a70ad0fd4451'


def require(ok,message):
    if not ok:raise ValueError(message)


def mlp_bn_names(stages=(0,1,2,3)):
    return {f'sttmultires_unet.encoders.swin3d.layers.{s}.swin_blocks.{b}.mlp.bn{bn}.norm_layer'
            for s,d in ((0,2),(1,2),(2,6),(3,2)) if s in stages for b in range(d) for bn in (1,2)}


def plan(members):
    rows=members['check_members']
    require(len(rows)==len({r['target_id'] for r in rows})==8,'Exactly eight unique fixed checks')
    return [dict(forward_id=i+1,checkpoint_step=271482,target_id=r['target_id'],layout='native',BN_scope=BN_SCOPE,mode='actual') for i,r in enumerate(rows)]


def validate_own_receipt(receipt,members,common):
    require(receipt['schema']=='flow_later_checkpoint_own_bn_v1' and receipt['status']=='OWN_BN_COMPLETE','Audited K own-CMA receipt required')
    require(receipt['checkpoint_step']==271482 and receipt['checkpoint_sha256']==CHECKPOINT_SHA,'Same original W for own BN')
    require(receipt['buffers_sha256']==OWN_BN_SHA and receipt['source_checkpoint_state_sha256']==SOURCE_SHA,'Own source identity')
    require(receipt['non_bn_state_before_sha256']==receipt['non_bn_state_after_sha256']==NON_BN_SHA and receipt['non_bn_state_unchanged'] is True,'Own calibration non-BN closure')
    require(receipt['bn_target_ids']==[r['target_id'] for r in members['bn_members']],'Exact original176 legal identities')
    for key in ('membership_sha256','runtime_manifest_sha256'):require(receipt[key]==common[key],'Own calibration input contract '+key)
    require(receipt['calibration_definition']=='all78_reset_CMA_train_nonBN_eval' and receipt['calibration_forwards']==176
            and receipt['calibration_bn_count']==78 and receipt['retained_mlp_bn_count']==24,'Original own calibration definition')
    require(receipt['old_CMA_loaded'] is False and receipt['GT_used'] is False and receipt['optimizer_steps']==0,'No legacy/GT/optimizer BN source')
    require(receipt['retained_names']==sorted(mlp_bn_names()),'All original24 private BN inventory')


def validate_buffers(values,source_state,np):
    private=mlp_bn_names();selected=mlp_bn_names((0,2))
    require(set(values)=={n+'.'+s for n in private for s in BUFFERS},'Validate all72 own arrays before selecting48')
    for name,value in values.items():
        require(name in source_state and tuple(value.shape)==tuple(source_state[name].shape),'Own BN source shape '+name)
        counter=name.endswith('.num_batches_tracked')
        require(value.dtype==np.dtype('int64' if counter else 'float32') and np.isfinite(value).all(),'Own BN dtype/finite '+name)
        require(int(value)==176 if counter else bool((value>=0).all()) if name.endswith('.running_var') else True,'Own BN variance/count '+name)
    return {k:v.copy() for k,v in values.items() if k.rsplit('.',1)[0] in selected}


def construct_state(source_state,values,digest,torch,np,strict=True):
    chosen=validate_buffers(values,source_state,np)
    without={n:v for n,v in source_state.items() if not n.endswith(tuple('.'+s for s in BUFFERS))}
    if strict:
        require(len(source_state)==711 and digest(source_state)==SOURCE_SHA,'Original711 checkpoint state')
        require(len(without)==477 and digest(without)==NON_BN_SHA,'Unchanged actual477 non-BN state')
    state=dict(without)
    state.update({n:torch.from_numpy(v.copy()) for n,v in chosen.items()})
    require(len(chosen)==48 and len(state)==len(without)+48,'Exactly48 selected BN buffers')
    if strict:require(len(state)==525 and sum(v.numel() for v in state.values())==54944214 and digest(state)==SELECTED_STATE_SHA,'Independently reconstructed525 state anchor')
    record={'source_state_sha256':digest(source_state),'non_BN_state_sha256':digest(without),'selected16_state_sha256':digest(state),
            'selected16_tensor_count':len(state),'selected16_elements':sum(v.numel() for v in state.values()),
            'retained_buffer_count':48,'retained_buffer_elements':sum(v.size for v in chosen.values()),
            'retained_buffer_tensor_sha256':digest({n:torch.from_numpy(v) for n,v in chosen.items()}),
            'retained_buffer_payloads':{n:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(v.tobytes()).hexdigest()} for n,v in sorted(chosen.items())},
            'frozen_BN_names':sorted(mlp_bn_names((0,2))),'private_current_BN_names':sorted(mlp_bn_names((1,3)))}
    return state,chosen,record


def install(model,values,scope,base,digest,torch,np,expected_state_sha,strict=True):
    all_bn,private=scope.bn_inventory(model,torch)
    require(set(private)==mlp_bn_names() and len(all_bn)==78,'Exact native78/private24 names')
    original={n:v.detach().cpu() for n,v in model.state_dict().items()}
    expected,chosen,record=construct_state(original,values,digest,torch,np,strict)
    require(record['selected16_state_sha256']==expected_state_sha,'CPU selected16 expected state')
    frozen=mlp_bn_names((0,2));model.eval().requires_grad_(False)
    with torch.no_grad():
        for name,bn in all_bn.items():
            if name in frozen:
                require(bn.track_running_stats and all(getattr(bn,s) is not None for s in BUFFERS),'Install only into original source BN')
                for s in BUFFERS:getattr(bn,s).copy_(torch.from_numpy(chosen[name+'.'+s]).to(getattr(bn,s)))
            else:
                bn.track_running_stats=False;bn.running_mean=bn.running_var=bn.num_batches_tracked=None
    actual=model.state_dict()
    require(set(actual)==set(expected) and digest(actual)==expected_state_sha,'Actual selected16 full state differs')
    require(all(torch.equal(actual[n].detach().cpu(),v) for n,v in expected.items()),'Actual selected16 tensors differ')
    require(all(not m.training for m in model.modules()),'Eval/dropout-off only')
    signature=base.bn_signature(model,torch)
    require(len(frozen)==16 and len(set(all_bn)-frozen)==62,'Selected16/current62 count')
    return {'scope':BN_SCOPE,'BN_count':78,'private_BN_frozen':16,'private_BN_current':8,'external_BN_current':54,
            'frozen_BN_names':sorted(frozen),'current_stat_BN_names':sorted(set(all_bn)-frozen),
            'BN_signature_before_inference':signature,'non_BN_parameters_and_affine_unchanged':True,
            'calibration_performed':False,'own_CMA_source_loaded':True,'retained_buffer_count':48,
            'source_unselected_BN_buffers_removed_before_inference':True},chosen,record


def native_with_bn_records(call,model,torch,h0):
    """Same frozen native math; its hook guard allows only our exact live hooks."""
    observed={};handles=[];allowed={};guard_seen=[]
    try:
        for name,bn in model.named_modules():
            if not isinstance(bn,torch.nn.modules.batchnorm._BatchNorm):continue
            observed[name]={'calls':0,'input_shapes':[],'current_stat':not bn.track_running_stats}
            def count(module,args,n=name):
                observed[n]['calls']+=1;observed[n]['input_shapes'].append(list(args[0].shape))
            handle=bn.register_forward_pre_hook(count);handles.append(handle);allowed[bn]={handle.id}
        require(len(observed)==78,'Observe exactly78 BN')
        def checked_require(ok,message):
            if message=='Unremoved forward hooks':
                # The frozen helper closes its own temporary hooks before this
                # point. Only the outer78 observers may still exist here.
                require(all(not m._forward_hooks and set(m._forward_pre_hooks)==allowed.get(m,set()) for m in model.modules()),'Only known78 observer hooks may remain')
                guard_seen.append(True)
            else:h0.require(ok,message)
        forward=types.FunctionType(h0.forward.__code__,dict(h0.forward.__globals__,require=checked_require),
                                   '_selected16_observed_native',h0.forward.__defaults__,h0.forward.__closure__)
        result=call(forward)
        require(all(x['calls']==1 for x in observed.values()),'Every BN exactly once per native forward')
        require(sum(x['current_stat'] for x in observed.values())==62,'Actual current62 graph')
        require(len(guard_seen)==1,'Exactly one frozen native hook-closure guard')
        return result,observed
    finally:
        for handle in handles:handle.remove()


def current_reference(result,audit,members,common):
    require(result['schema']=='flow_bn_conversion_round14_v1' and result['status']=='ROUND14_BN_CONVERSION_COMPLETE','Complete R14 source')
    require(audit['schema']=='flow_bn_conversion_round14_independent_execution_audit_v1' and audit['status']=='PASS'
            and audit['source_result_sha256']==CURRENT_RESULT_SHA,'Actual fixed R14 independent audit')
    require(audit['full_forwards']==16 and audit['quality_prediction_count']==audit['quality_archive_count']==16
            and audit['optimizer_updates']==audit['calibration_forwards']==audit['dual_forwards']==0,'Actual16/zero training calibration dual source')
    for key in ('membership_sha256','runtime_manifest_sha256','region_sha256'):require(result['binding'][key]==common[key],'Current reference '+key)
    for key in ('all_states_unchanged','all_BN_buffers_removed','all_hooks_removed','original_modules_restored'):
        require(result[key] is True and audit[key] is True,'Current reference closure '+key)
    state=result['checkpoints'][STEP];audited=audit['source_states'][STEP]
    require(state['checkpoint_sha256']==audited['checkpoint_sha256']==CHECKPOINT_SHA and state['source_state_sha256']==audited['source_state_sha256']==SOURCE_SHA,'Current reference original W')
    require(state['initial_allcurrent_state_sha256']==state['final_allcurrent_state_sha256']==audited['allcurrent_state_sha256']==NON_BN_SHA,'Actual current reference unchanged477')
    require(result['runtime']['AMP'] is False and result['runtime']['TF32'] is False and state['BN_scope']['scope']=='ALL78_CURRENT','Current reference policy')
    frames=state['evaluations']['endpoint']['frames'];ids=[r['target_id'] for r in members['check_members']]
    require(len(frames)==8 and [f['target_id'] for f in frames]==ids,'Exact current check8 order')
    predictions=[p for p in audit['quality_predictions'] if p['checkpoint_step']==271482]
    require([(p['target_id'],p['layout']) for p in predictions]==[(tid,'native') for tid in ids],'Audited current8 order')
    for f,p,row in zip(frames,predictions,members['check_members']):
        require(f['sequence']==row['sequence'] and f['arms']['native']['prediction_sha256']==p['array_sha256'],'Current native prediction binding')
    return {'cache_phase':'endpoint','cache_result_sha256':CURRENT_RESULT_SHA,'execution_audit_sha256':CURRENT_AUDIT_SHA,
            'checkpoint_sha256':CHECKPOINT_SHA,'cached_deployment_state_sha256':NON_BN_SHA,'cached_BN_buffers_sha256':None,
            'cached_native_predictions':[{k:f[k] for k in ('target_id','sequence','input_evidence','prediction_file','prediction_file_sha256')}|
                {'prediction_sha256':f['arms']['native']['prediction_sha256']} for f in frames],
            'cache_not_loaded_as_model':True,'preselected_W_not_quality_ranked':True}
