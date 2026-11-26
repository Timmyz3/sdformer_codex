"""Fixed two-source BN conversion contracts; no optimizer or model math changes."""
CHECKPOINTS=('135930','271482')
CACHE_SHA={'135930':'254fb4ec5f18d30e4245b4c277ee1c5fb53beaf7fb3cf42cf60d0a4ee4390d0f',
           '271482':'8e0dafe172a27b6d56b484bada005ec18d15e62ae18afdc7b0bff7870ba1a841'}
AUDIT_SHA={'135930':'eb77f69e87d80192157b1b3fd458289cdff56ea07a134bb1738076f9325d9a61',
           '271482':'7455e073fcfb1ff981e228807cf917a4e4657a7ef988d072d7a16cdcbb77e9bd'}
CHECKPOINT_SHA={
 '135930':'d5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30',
 '271482':'1e72cca3946f40e24a1575f3348f9a596d744460808ad910a616ba713b5e4977'}
SOURCE_STATE_SHA={
 '135930':'2bd43221e02d376df89e2b90ddce5a14f05301f45dd5ce871304b2a71b8c2831',
 '271482':'ad14a7d45091aa75d64df83e1a71bd89b35dfbbf49c9ab84861d5ae0aa270560'}
CURRENT_STATE_SHA={
 '135930':'8d4b5141731b0bed7bd12da67217ee5f68ad72288d03fd6c56726d8b6acd9ef5',
 '271482':'b721031729a9a5f69d03037be53e62ebf8c1536dad0ad8a8f8e1601fe9fc2798'}
BUFFERS=('running_mean','running_var','num_batches_tracked')


def require(value,message):
    if not value:raise ValueError(message)


def plan(members):
    rows=members['check_members']
    require(len(rows)==8 and len({r['target_id'] for r in rows})==8,'Fixed8 unique checks')
    return [dict(forward_id=i*8+j+1,checkpoint_step=int(step),target_id=row['target_id'],layout='native',BN_scope='ALL78_CURRENT',mode='actual')
            for i,step in enumerate(CHECKPOINTS) for j,row in enumerate(rows)]


def strip_bn_state(state,digest,expected_source=None,expected_current=None):
    keys={k for k in state if k.endswith(tuple('.'+v for v in BUFFERS))}
    names={k.rsplit('.',1)[0] for k in keys}
    require(len(state)==711 and len(names)==78 and keys=={n+'.'+v for n in names for v in BUFFERS},'Actual711 source/234BN keys required')
    current={k:v for k,v in state.items() if k not in keys}
    source_sha=digest(state);current_sha=digest(current)
    require(len(current)==477 and sum(x.numel() for x in current.values())==54919238,'True current477 state inventory')
    if expected_source is not None:require(source_sha==expected_source,'Original source state anchor differs')
    if expected_current is not None:require(current_sha==expected_current,'All-current/nonBN state anchor differs')
    return {'source_state_sha256':source_sha,'allcurrent_state_sha256':current_sha,'source_tensor_count':711,
            'removed_BN_buffers':234,'allcurrent_tensor_count':477,'allcurrent_elements':54919238,'BN_names':sorted(names)}


def switch_allcurrent(model,base,scope,digest,torch,expected_current,expected_count=477):
    all_bn,private=scope.bn_inventory(model,torch)
    require(len(all_bn)==78 and len(private)==24,'True native78/24 inventory')
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    require(base.apply_bn(model,'P_paper',torch)==78,'Exactly78 current BN switches')
    model.requires_grad_(False)
    require(all(not m.training for m in model.modules()),'Eval/dropout-off required')
    require(all(not b.track_running_stats and all(getattr(b,k) is None for k in BUFFERS) for b in all_bn.values()),'All running buffers must be removed')
    require(len(model.state_dict())==expected_count and digest(model)==expected_current,'Actual all-current full state differs')
    require(all(torch.equal(p.detach(),before[n]) for n,p in model.named_parameters()),'BN switch modified parameter/affine')
    return {'scope':'ALL78_CURRENT','BN_count':78,'private_BN_current':24,'external_BN_current':54,
            'frozen_BN_names':[],'current_stat_BN_names':sorted(all_bn),'BN_signature_before_inference':base.bn_signature(model,torch),
            'non_BN_parameters_and_affine_unchanged':True,'calibration_performed':False,'external_BN_artifact_loaded':False,
            'source_BN_buffers_removed_before_inference':True}


def reference_contract(step,result,result_sha,audit,members,common):
    """Metadata-only admission; no AEE or quality gate affects this experiment."""
    require(step in CHECKPOINTS,'Only two predeclared checkpoints')
    b=result['binding'];rows=members['check_members'];scope=result['BN_scope']
    require(b['checkpoint_sha256']==CHECKPOINT_SHA[step],'Cached same-W source differs')
    for k in ('membership_sha256','runtime_manifest_sha256','region_sha256'):
        require(b[k]==common[k],'Cached common input contract differs: '+k)
    require(result['runtime']['AMP'] is False and result['runtime']['TF32'] is False,'Cached precision differs')
    require(scope['BN_count']==78 and scope['MLP_BN_count']==24 and len(scope['frozen_BN_names'])==len(set(scope['frozen_BN_names']))==24
            and len(scope['current_stat_BN_names'])==len(set(scope['current_stat_BN_names']))==54 and not set(scope['frozen_BN_names'])&set(scope['current_stat_BN_names']),
            'Cached24 private/54 external scope required')
    require(all(result[k] is True for k in ('BN_buffers_unchanged','original_modules_restored','all_hooks_removed')),'Cached execution not closed')
    require(audit['status']=='PASS','Independent cache audit required')
    if step=='135930':
        require(result['schema']=='flow_round13_coverage_recovery_v1' and result['status']=='ROUND13_COVERAGE_RECOVERY_COMPLETE','R13 cache complete schema')
        require((b['condition'],b['arm'],b['mode'])==('diversity','native','recover'),'Only fixed R13 diversity/native initial cache')
        require(audit['schema']=='flow_round13_independent_execution_audit_v1' and audit['scope']=={'conditions':['reference','diversity'],'modes':['smoke','recover'],'runs':16},'Full16 R13 audit required')
        require(len(audit['results'])==16 and {(v['condition'],v['mode'],v['arm']) for v in audit['results']}==
                {(c,m,a) for c in ('reference','diversity') for m in ('smoke','recover') for a in ('native','FF','CC','fine_s2b2')},'Complete unique16 audit entries')
        matches=[v for v in audit['results'] if (v['condition'],v['mode'],v['arm'])==('diversity','recover','native')]
        require(len(matches)==1 and matches[0]['result_sha256']==result_sha,'R13 actual cache SHA not audited')
        require(result['initial_model_state_sha256']==matches[0]['initial_model_state_sha256']==audit['common_initial_model_sha256'],'R13 initial state not paired')
        require(scope['scope']=='MLP_R13_COMMON_CMA176' and scope['non_bn_state_sha256']==CURRENT_STATE_SHA[step],'R13 original nonBN source differs')
        phase='initial';predictions=matches[0]['predictions'];prediction_lookup={(v['target_id'],v['layout']):v['array_sha256'] for v in predictions if v['label']=='initial'}
        state=result['initial_model_state_sha256'];bn_sha=b['BN_buffers_sha256']
        require(bn_sha==audit['new_BN_actual_sha256'],'R13 audited commonBN bytes')
    else:
        require(result['schema']=='flow_later_checkpoint_deployment_v1' and result['status']=='LATER_CHECKPOINT_193_FORWARD_COMPLETE','K cache complete schema')
        require(audit['schema']=='flow_later_checkpoint_independent_execution_audit_v1' and audit['source_result_sha256']==result_sha,'K actual cache SHA not audited')
        require(audit['full_forwards']==193 and audit['optimizer_updates']==0 and audit['calibration_forwards']==176,'K actual fixed inference audit')
        require(audit['source_non_BN_state_sha256']==CURRENT_STATE_SHA[step] and audit['source_state_sha256']==SOURCE_STATE_SHA[step],'K original source independently audited')
        require(scope['scope']=='MLP_271482_OWN_CMA176' and result['non_bn_state_unchanged'] is True,'K own deployment BN source')
        require(result['deployment_initial_state_sha256']==result['deployment_final_state_sha256']==audit['deployment_state_sha256'],'K complete state cache not paired')
        phase='endpoint';prediction_lookup={(v['target_id'],v['layout']):v['array_sha256'] for v in audit['quality_predictions']}
        state=result['deployment_final_state_sha256'];bn_sha=result['own_BN_buffers']['sha256']
        require(audit['BN_buffers_sha256']==bn_sha,'K audited ownBN bytes')
    frames=result['evaluations'][phase]['frames']
    require([f['target_id'] for f in frames]==[v['target_id'] for v in rows] and len(frames)==8,'Cached ordered check8 differs')
    for f,row in zip(frames,rows):
        require(f['sequence']==row['sequence'] and f['arms']['native']['prediction_sha256']==prediction_lookup[(row['target_id'],'native')],'Cached native payload not independently audited')
    return {'cache_phase':phase,'cache_result_sha256':result_sha,'cached_deployment_state_sha256':state,'cached_BN_buffers_sha256':bn_sha,
            'checkpoint_sha256':CHECKPOINT_SHA[step],'allcurrent_state_sha256':CURRENT_STATE_SHA[step],
            'cached_native_predictions':[{k:f[k] for k in ('target_id','sequence','input_evidence','prediction_file','prediction_file_sha256')}|
                {'prediction_sha256':f['arms']['native']['prediction_sha256']} for f in frames],
            'cache_not_loaded_as_model':True,'no_quality_based_admission':True}
