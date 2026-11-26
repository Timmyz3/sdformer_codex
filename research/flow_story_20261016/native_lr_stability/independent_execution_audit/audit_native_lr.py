#!/usr/bin/env python3
"""Native lower-LR independent execution audit. No runner imports, GPU, SSH or AEE reduction.

Adapted from frozen independent R13 auditor dd57f9f315e1c5f38961208496a37f46db577c47b8507029a55b5bc216d8c8e8.
The narrow schema/recipe/schedule changes are explicit here; original remains unchanged.

Prepare pins before result reading. Audit expands a protocol-derived expected
forward schedule, reconstructs every FC row count from native geometry, checks
real NPZ/PT bytes, and optionally reconstructs the common initial state directly
from the original checkpoint plus fresh BN NPZ. It never uses quality to decide
which arms or predictions to retain.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
RUNROOT=HERE.parent
ROOT=RUNROOT.parent/'coverage_recovery_round13'
R12=ROOT.parent/'ordinary_recovery_round12'
ARMS=('native','FF')
CONDS=('diversity',)
BASE='sttmultires_unet.encoders.swin3d.layers.'
ORIGINAL_SHA='d5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30'
LOSS_SHA='92aa1451469690ad89c16b6954c972d3c460689f5cb765662b4eff7fc7dec122'
GEOMETRY={BASE+str(s)+'.swin_blocks.'+str(b)+'.mlp':(10,1,h,w,c)
          for s,(d,h,w,c) in enumerate([(2,120,160,96),(2,60,80,192),(6,30,40,384),(2,15,20,768)]) for b in range(d)}
SELECTED={n for n in GEOMETRY if '.layers.0.' in n or '.layers.2.' in n}
TARGET=BASE+'2.swin_blocks.2.mlp'
SUFFIX=('fc1.weight','bn1.norm_layer.weight','bn1.norm_layer.bias','sn1.spiking_neuron.weight','sn1.spiking_neuron.bias',
        'fc2.weight','bn2.norm_layer.weight','bn2.norm_layer.bias','sn2.spiking_neuron.weight','sn2.spiking_neuron.bias')
ALLOWED={n+'.'+suffix for n in SELECTED for suffix in SUFFIX}


def require(value,message):
    if not value: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'))
def sig(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def hexdigest(value): return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def finite(value): return isinstance(value,(int,float)) and math.isfinite(value)


def pins():
    paths=[ROOT/'ANALYSIS_PROTOCOL.md',ROOT/'data_protocol/manifest_v1.json',ROOT/'inputs_v1.json',ROOT/'runtime_v1.json',
           ROOT/'preparation/bn_receipt.json',ROOT/'preparation/regions_v1.json']
    paths += [ROOT/'implementation'/n for n in ('run_coverage_recovery.py','execute_recovery.py','runner_cpu_results.json','inherited_pins.json','test_coverage_cpu.py')]
    paths += [R12/n for n in ('run_recovery.py','trainable_representative_mlp.py','source_contract.py','source_pins.json')]
    return {str(p):sha(p) for p in paths}


def memberships(m):
    require(m['checkpoint_sha256']==ORIGINAL_SHA,'manifest original checkpoint')
    train=set(m['split_sequences']['train']);check=set(m['split_sequences']['check'])
    require(len(train)==8 and len(check)==2 and not train&check,'8/2 sequence exclusion')
    require([r['sequence'] for r in m['train_order_reference']]==[r['sequence'] for r in m['train_order_diversity']],'sequence schedule pairing')
    require(set(r['target_id'] for r in m['train_members_reference'])<=set(r['target_id'] for r in m['train_members_diversity']),'nested coverage')
    for cond,count,repeat in [('reference',40,5),('diversity',200,1)]:
        order=m['train_order_'+cond];members=m['train_members_'+cond]
        require(len(order)==200 and [r['step'] for r in order]==list(range(1,201)),'fixed200 schedule')
        require(len(members)==count and Counter(r['target_id'] for r in order)==Counter({r['target_id']:repeat for r in members}),'fixed identity repetitions')
        require(Counter(r['sequence'] for r in order)==Counter({s:25 for s in train}),'fixed per-sequence visits')
    require(Counter(r['sequence'] for r in m['bn_members'])==Counter({s:22 for s in train}),'new BN identities')
    require(len(m['check_members'])==8 and set(r['sequence'] for r in m['check_members'])==check,'check membership')
    sets=[set(r['target_id'] for r in m[n]) for n in ('train_members_diversity','bn_members','selector_members','check_members')]
    require(all(not x&y for i,x in enumerate(sets) for y in sets[i+1:]),'role overlaps')


def expected_layer(name,arm,dual=False):
    t,b,h,w,c=GEOMETRY[name]
    fine=arm=='FF' or (arm=='fine_s2b2' and name==TARGET)
    tiles=[]
    for y in range(0,h,9):
        for x in range(0,w,9):
            hh=min(9,h-y);ww=min(9,w-x)
            tiles.append((hh,ww,(hh+2)//3,(ww+2)//3))
    fpoints=h*w; rpoints=sum(a*b for _,_,a,b in tiles)
    if arm=='native' or name not in SELECTED: rows=t*b*fpoints
    elif dual: rows=t*b*(fpoints+rpoints)
    else: rows=t*b*(fpoints if fine else rpoints)
    return rows,fine,tiles


def check_execution(ex,arm,dual,purpose):
    require(set(ex['all12_observed_fc_rows'])==set(GEOMETRY),'all12 hook coverage')
    total=0;wrapped=0;native_calls=0
    for name in GEOMETRY:
        expected,fine,tiles=expected_layer(name,arm,dual);obs=ex['all12_observed_fc_rows'][name]
        require(obs=={'fc1_rows':expected,'fc2_rows':expected},'per-layer FC rows '+name)
        total+=expected
        if arm=='native' or name not in SELECTED: continue
        wrapped+=1;layer=ex['layers'][name];t,b,h,w,c=GEOMETRY[name]
        require(layer['shape_TBHWC']==list(GEOMETRY[name]),'whole T10/B1/shape')
        require(layer['mode']==('dual_reference' if dual else 'actual'),'producer mode')
        require(layer['fine_mask']==[[fine]*((w+8)//9) for _ in range((h+8)//9)],'whole fixed-layer mask')
        require(layer['executed_fc_rows_per_layer']==expected,'wrapper row independent reconstruction')
        require(layer['temporal_pooling'] is False and layer['external_residual_added_here'] is False,'preserved graph')
        if 'native_mlp_calls' in layer:
            require(layer['native_mlp_calls']==len(tiles)*(2 if dual else 1),'differentiable producer calls')
            native_calls+=layer['native_mlp_calls']
        else:
            obs=layer['observed']; require(layer['merge_stride']==3,'representative stride')
            for branch,enabled in [('fine',dual or fine),('coarse',dual or not fine)]:
                shapes=[[10,1,hh if branch=='fine' else rh,ww if branch=='fine' else rw,c] for hh,ww,rh,rw in tiles] if enabled else []
                require(obs['native_input_shapes'][branch]==shapes,'actual producer input shape sequence')
                rows=sum(a[0]*a[1]*a[2]*a[3] for a in shapes)
                expected_obs={'native_mlp_calls':len(shapes),'fc1_rows':rows,'fc2_rows':rows,
                              'sn1_elements':rows*c,'sn2_elements':rows*4*c,'bn1_elements':rows*4*c,'bn2_elements':rows*c}
                require(obs[branch]==expected_obs,'actual producer FC/PSN/BN element evidence')
                native_calls+=len(shapes)
            coarse=dual or not fine;reps=sum(rh*rw for hh,ww,rh,rw in tiles)
            for key,wanted in {'executed_pool_input_elements':10*c*h*w if coarse else 0,
                               'executed_pool_reduction_additions_logical':10*c*(h*w-reps) if coarse else 0,
                               'executed_pool_divisions_logical':10*c*reps if coarse else 0,
                               'executed_repeat_write_elements':10*c*h*w if coarse else 0}.items():
                require(layer[key]==wanted,'actual pooling/repeat '+key)
    require(set(ex['layers'])==(set() if arm=='native' else SELECTED),'wrapped layer set')
    require(ex['summed_fc1_rows']==ex['summed_fc2_rows']==total,'summed rows vs actual all12')
    if 'gradient_enabled' in ex:
        require(ex['gradient_enabled']==(purpose=='training'),'autograd mode')
        require(ex['full_flow_count']==4 and ex['arm']==arm,'all four output heads')
        require(ex['mode']=='actual','new wrapper actual execution')
    return {'FC1_rows':total,'FC2_rows':total,'wrapped_layers':wrapped,'native_calls_in_wrappers':native_calls,'observed_FC_modules':24}


def expected_forwards(condition,mode,arm,m):
    order=m['train_order_'+condition]; first=order[0]['target_id']; checks=m['check_members'];out=[]
    def audit(which):
        out.append(('audit_new_'+which,first,arm,False));out.append(('audit_old_actual_'+which,first,arm,False))
        if arm!='native': out.append(('audit_old_dual_'+which,first,arm,True))
    audit('initial')
    if mode=='recover':out.extend(('evaluation_initial',r['target_id'],arm,False) for r in checks)
    out.extend(('training',r['target_id'],arm,False) for r in order[:5 if mode=='smoke' else 200])
    audit('final')
    if mode=='recover':out.extend(('evaluation_endpoint',r['target_id'],a,False) for r in checks for a in ARMS)
    return out


def verify_binding(d,pinned,m,inputs):
    b=d['binding'];mode=d['mode'];base=read(BASELINE)['binding']
    require((b['condition'],b['arm'],b['mode'])==('diversity','native',mode),'fixed D/native binding')
    expected={'runner_sha256':sha(RUNROOT/'run_native_lr.py'),'execute_sha256':sha(RUNROOT/'execute_recovery.py'),
              'runner_checks_sha256':sha(RUNROOT/'runner_cpu_results.json'),'source_pins_sha256':sha(RUNROOT/'inherited_pins.json')}
    for k,v in expected.items(): require(b[k]==v,'new-source binding '+k)
    common=('checkpoint_sha256','wrapper_sha256','membership_sha256','runtime_manifest_sha256','BN_buffers_sha256',
            'BN_receipt_sha256','loss_sha256','allowed_parameter_count','region_sha256','membership_audit')
    for k in common:require(b[k]==base[k],'paired baseline common binding '+k)
    require(b['BN_buffers_sha256']==inputs['bn_buffers_sha256'] and b['BN_receipt_sha256']==inputs['bn_receipt_sha256'],'local BN inputs')
    require(b['membership_sha256']==sha(ROOT/'data_protocol/manifest_v1.json'),'actual manifest SHA')
    require(b['runtime_manifest_sha256']==sha(ROOT/'runtime_v1.json'),'actual runtime SHA')
    recipe=dict(base['recipe']);recipe['lr']=2.5e-5
    require(base['recipe']['lr']==2.5e-4 and b['recipe']==recipe,'only LR changed exactly')
    require(b['fit_ids']==[r['target_id'] for r in m['train_order_diversity']],'fixed D200 once order')
    require(b['check_ids']==([] if mode=='smoke' else [r['target_id'] for r in m['check_members']]),'fixed check8')
    nsteps=5 if mode=='smoke' else 200;neval=0 if mode=='smoke' else 24
    require((b['expected_training_forwards'],b['expected_audit_forwards'],b['expected_evaluation_forwards'],b['total_full_forward_budget'])
            ==(nsteps,4,neval,nsteps+4+neval),'fixed 9/228 budget')
    require(b['fresh_W0_and_optimizer'] is True and b['experiment']=='native_lr_stability','fresh state provenance')
    require(b['initial_layouts']==([] if mode=='smoke' else ['native']) and b['endpoint_layouts']==([] if mode=='smoke' else ['native','FF']),'fixed layouts')
    require(b['reused_prepare_source_sha256']==base['runner_sha256'] and b['reused_execute_parent_sha256']==base['execute_sha256'],'inherited source SHA')
    original=read(BASELINE);pred={r['target_id']:r['arms']['native']['prediction_sha256'] for r in original['evaluations']['initial']['frames']}
    comparison={'result_sha256':sha(BASELINE),'initial_state_sha256':original['initial_model_state_sha256'],
        'initial_native_prediction_sha256':pred,'baseline_lr':2.5e-4,'diagnostic_lr':2.5e-5,'only_recipe_difference':['lr'],
        'R13_check8_now_development':True,'recovered_baseline_weights_used_as_initialization':False}
    require(b['baseline_result']['sha256']==sha(BASELINE) and b['baseline_comparison']==comparison,'exact baseline state/prediction comparison')
    require(d['paired_initial_state_exact'] is True and d['initial_model_state_sha256']==comparison['initial_state_sha256'],'actual paired initial state')
    require(d['actual_optimizer_hyperparameters']=={'lr':2.5e-5,'weight_decay':.01,'betas':[.9,.999],'eps':1e-8},'actual new optimizer recipe')
    require(0<b['wall_minutes']<=30 and 0<d['wall_seconds']<=1800,'bounded wall limit')
    return expected

def bn_payload(path,m,inputs):
    import numpy as np
    receipt=read(ROOT/'preparation/bn_receipt.json')
    require(sha(path)==inputs['bn_buffers_sha256']==receipt['buffers_sha256'],'BN actual NPZ SHA')
    require(sha(ROOT/'preparation/bn_receipt.json')==inputs['bn_receipt_sha256'],'BN receipt SHA')
    require(receipt['bn_target_ids']==[r['target_id'] for r in m['bn_members']],'BN ordered176')
    require(receipt['calibration_forwards']==176 and receipt['calibration_bn_count']==78 and receipt['retained_mlp_bn_count']==24,'BN count')
    require(receipt['old_CMA_loaded'] is False and receipt['GT_used'] is False and receipt['optimizer_steps']==0,'BN input-only from reset')
    require(receipt['calibration_definition']=='all78_reset_CMA_train_nonBN_eval','BN reset semantics')
    require(receipt['non_bn_state_unchanged'] is True and receipt['non_bn_state_before_sha256']==receipt['non_bn_state_after_sha256'],'BN nonbuffer unchanged')
    names={n+'.bn'+str(i)+'.norm_layer' for n in GEOMETRY for i in (1,2)}
    require(set(receipt['retained_names'])==names,'all24 private names')
    with np.load(path,allow_pickle=False) as z:
        require(set(z.files)=={n+'.'+k for n in names for k in ('running_mean','running_var','num_batches_tracked')},'exact72 BN keys')
        values={k:z[k].copy() for k in z.files}
    for n in names:
        base=n.split('.bn')[0];c=GEOMETRY[base][-1];width=4*c if '.bn1.' in n else c
        for suffix in ('running_mean','running_var','num_batches_tracked'):
            a=values[n+'.'+suffix];counter=suffix=='num_batches_tracked'
            require(a.shape==(() if counter else (width,)) and a.dtype==np.dtype('int64' if counter else 'float32'),'BN dtype/native shape')
            require(np.isfinite(a).all(),'BN finite')
            if counter:require(int(a)==176,'BN CMA176 counter')
            if suffix=='running_var':require((a>=0).all(),'BN nonnegative variance')
    return values,receipt


def tensor_digest(state,exclude=()):
    h=hashlib.sha256();exclude=set(exclude)
    for name in sorted(state):
        if name in exclude:continue
        t=state[name].detach().cpu().contiguous()
        h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
    return h.hexdigest()


def reconstruct_initial(path,bn):
    import torch
    require(sha(path)==ORIGINAL_SHA,'Original checkpoint actual SHA')
    p=torch.load(path,map_location='cpu',weights_only=True)
    require(p['training']['completed_optimizer_steps']==135930,'original completed step')
    s=p['model_state_dict'];require(all(isinstance(t,torch.Tensor) and bool(torch.isfinite(t).all()) for t in s.values()),'original finite tensors')
    suffixes=('.running_mean','.running_var','.num_batches_tracked')
    private=set(bn);external=[n for n in s if n.endswith(suffixes) and n not in private]
    require(len(external)==162,'external54 buffer removal')
    for name in external:del s[name]
    for name,value in bn.items():
        require(name in s and tuple(s[name].shape)==value.shape,'BN replacement native shape')
        s[name]=torch.from_numpy(value.copy())
    return s


def checkpoint(path,d,bn,initial):
    import torch
    require(not torch.cuda.is_initialized(),'CPU audit must not initialize CUDA')
    require(sha(path)==d['weights_sha256'],'weights actual bytes SHA')
    p=torch.load(path,map_location='cpu',weights_only=True);s=p['model_state_dict'];names=p['trainable_names']
    require(p['schema']=='flow_native_lr_stability_recovered_state_v1','weight schema')
    require(p['binding']==d['binding'],'checkpoint exact execution binding')
    require(p['completed_optimizer_steps']==200 and p['initial_state_sha256']==d['initial_model_state_sha256'],'checkpoint step/initial')
    require(p['BN_scope']=='MLP_R13_COMMON_CMA176' and p['BN_buffers_unchanged'] is True,'checkpoint BN')
    require(names==d['allowed_parameter_names'] and set(names)==ALLOWED,'checkpoint exact allowed names')
    require(all(isinstance(t,torch.Tensor) and t.device.type=='cpu' and bool(torch.isfinite(t).all()) for t in s.values()),'finite CPU complete state')
    require(sum(s[n].numel() for n in names)==7252064,'actual trainable tensor elements')
    final=tensor_digest(s);frozen=tensor_digest(s,names)
    require(final==d['final_state_sha256']==p['final_state_sha256'],'independent final state digest')
    require(frozen==d['frozen_state_before_sha256'],'independent frozen complement')
    require(all(torch.equal(s[n],torch.from_numpy(v)) for n,v in bn.items()),'actual72 BN buffers unchanged')
    extern=d['BN_scope']['current_stat_BN_names']
    require(all(n+'.'+key not in s for n in extern for key in ('running_mean','running_var','num_batches_tracked')),'no external running buffers')
    if initial is not None:
        require(set(s)==set(initial),'same full state keys as reconstructed initial')
        check_drift(s,initial,d['parameter_drift'])
        require(tensor_digest(initial)==d['initial_model_state_sha256'],'actual reconstructed common initial digest')
        changed={n for n in s if not torch.equal(s[n],initial[n])}
        require(changed==set(d['updated_parameter_names']) and changed<=ALLOWED,'actual changed parameters')
    opt=p['optimizer_state_dict'];require(len(opt['param_groups'])==1,'one fresh optimizer group')
    group=opt['param_groups'][0];ids=group['params'];require(len(ids)==80 and len(set(ids))==80 and set(ids)==set(opt['state']),'80 optimizer states')
    for key,wanted in [('lr',.000025),('weight_decay',.01),('eps',1e-8)]:require(group[key]==wanted,'optimizer '+key)
    require(list(group['betas'])==[.9,.999],'optimizer betas')
    for index,name in zip(ids,names):
        st=opt['state'][index];require(set(st)=={'step','exp_avg','exp_avg_sq'},'AdamW state fields')
        require(float(st['step'])==200 and all(bool(torch.isfinite(v).all()) for v in st.values()),'optimizer finite200')
        require(st['exp_avg'].shape==s[name].shape and st['exp_avg_sq'].shape==s[name].shape,'moment shape')
    require(set(p['rng_state'])=={'torch_cpu','torch_cuda'} and len(p['rng_state']['torch_cuda'])==1,'RNG snapshot')
    return {'file':str(path),'sha256':sha(path),'bytes':path.stat().st_size,'state_tensors':len(s),
            'state_elements':sum(t.numel() for t in s.values()),'optimizer_tensors':240,
            'final_state_independently_recomputed':True,'frozen_complement_independently_recomputed':True,
            'initial_state_reconstructed':initial is not None,'BN72_actual_values_match':True}


def check_run(path,m,inputs,bn,initial,pinned):
    import numpy as np
    d=read(path);cond=d['condition'];mode=d['mode'];arm=d['arm']
    require(d['schema']=='flow_native_lr_stability_v1' and cond in CONDS and mode in ('smoke','recover') and arm=='native','actual run schema')
    require(d['status']==('NATIVE_LR_SMOKE_COMPLETE' if mode=='smoke' else 'NATIVE_LR_RECOVERY_COMPLETE'),'complete run required')
    require(d['training_performed'] is True and d['inference_executed'] is True,'actual execution flags')
    verify_binding(d,pinned,m,inputs)
    require(set(d['allowed_parameter_names'])==ALLOWED and len(d['allowed_parameter_names'])==80 and d['allowed_parameter_count']==7252064,'trainable whitelist')
    require(set(d['updated_parameter_names'])<=ALLOWED and all(any(n.startswith(l+'.') for n in d['updated_parameter_names']) for l in SELECTED),'all8 MLP updates')
    for flag in ('frozen_state_unchanged','BN_buffers_unchanged','original_modules_restored','all_hooks_removed'):require(d[flag] is True,'closed execution '+flag)
    require(d['hardware_benefit_claim'] is False and d['flow_specific_increment_claim'] is False,'no unsupported scientific claim')
    require(d['runtime']['TF32'] is False and d['runtime']['AMP'] is False and d['runtime']['grad_enabled_training'] is True,'FP32 runtime')
    require(d['native_loss']=={'sha256':LOSS_SHA,'gamma':None,'flow_scaling':1,'lambda_mod':1,'all_four_flows_used':True,'GT_event_filter':False},'native allGT four-flow loss')
    scope=d['BN_scope'];private={n.rsplit('.',1)[0] for n in bn}
    require(scope['scope']=='MLP_R13_COMMON_CMA176' and scope['BN_count']==78 and scope['MLP_BN_count']==24,'new BN scope')
    require(scope['old_CMA_loaded'] is False and scope['new_buffers_sha256']==inputs['bn_buffers_sha256'] and scope['new_receipt_sha256']==inputs['bn_receipt_sha256'],'new BN actual source binding')
    require(set(scope['frozen_BN_names'])==set(scope['MLP_BN_names'])==private,'private24 set')
    require(len(scope['current_stat_BN_names'])==54 and not private&set(scope['current_stat_BN_names']),'external54 set')
    signatures=scope['BN_signature_before_inference'];require(set(signatures)==private|set(scope['current_stat_BN_names']),'BN signature78')
    for name,signature in signatures.items():
        require(signature['training'] is False and signature['tracking']==(name in private),'BN eval/current-stat flags')
        for key in ('running_mean','running_var','num_batches_tracked'):
            expected=hashlib.sha256(bn[name+'.'+key].tobytes()).hexdigest() if name in private else None
            require(signature[key]==expected,'BN starting array content SHA')
    require(scope['non_bn_state_unchanged'] is True,'new BN installation preserved nonbuffer')
    if initial is not None:
        require(d['initial_model_state_sha256']==tensor_digest(initial),'original+freshBN reconstructed common initial')
        require(d['frozen_state_before_sha256']==tensor_digest(initial,ALLOWED),'original+freshBN frozen complement')
    planned=expected_forwards(cond,mode,arm,m);actual=d['forward_records'];totals=Counter();purposes=Counter()
    require(len(actual)==len(planned)==d['full_forward_count'],'explicit forward schedule length')
    lookup={}
    for i,(record,expect) in enumerate(zip(actual,planned),1):
        purpose,target,layout,dual=expect
        require(record['forward_id']==i and record['purpose']==purpose and record['target_id']==target,'exact ordered forward record')
        totals.update(check_execution(record['execution'],layout,dual,purpose));purposes[purpose]+=1
        lookup.setdefault((purpose,target,layout),[]).append(record['execution'])
    steps=d['steps'];nsteps=5 if mode=='smoke' else 200;order=m['train_order_'+cond][:nsteps]
    require(len(steps)==nsteps==d['completed_optimizer_steps'],'fixed update count')
    category=Counter();layers=Counter();valid_pixels=[]
    for i,(step,wanted) in enumerate(zip(steps,order),1):
        require(step['step']==i and step['target_id']==wanted['target_id'] and step['sequence']==wanted['sequence'],'exact step identity and sequence')
        service=lookup[('training',wanted['target_id'],arm)].pop(0)
        require(step['service']==service,'step service equals independent forward record')
        require(step['valid_GT_pixels']>0 and finite(step['task_loss']) and finite(step['preclip_grad_norm']),'finite training and no emptyGT substitute')
        g=step['gradient_audit'];require(g['all_allowed_gradients_present_finite'] is True and g['frozen_gradients_None'] is True,'gradient boundary')
        require(set(g['per_layer_squared_L2'])==SELECTED and set(g['category_squared_L2'])=={'FC','PSN','BN_affine'},'gradient set coverage')
        for name,value in g['per_layer_squared_L2'].items():
            require(finite(value) and value>=0,'finite layer gradient');layers[name]+=value>0
        for name,value in g['category_squared_L2'].items():
            require(finite(value) and value>=0,'finite category gradient');category[name]+=value>0
        valid_pixels.append(step['valid_GT_pixels'])
    require(d['gradient_coverage']['layer_steps_nonzero']==dict(layers) and d['gradient_coverage']['category_steps_nonzero']==dict(category),'independent gradient coverage reduction')
    require(d['gradient_coverage']['complete']==(all(layers.values()) and all(category.values())),'gradient completeness verdict')
    require(all(layers.values()) and all(category.values()),'all layers/categories exercised')
    input_rows={r['target_id']:r for r in d['binding']['data_evidence']}
    require(set(d['training_input_evidence'])==set(r['target_id'] for r in order),'exact unique cached training identities')
    for target,evidence in d['training_input_evidence'].items():
        expect=input_rows[target]['files'];require(evidence=={'events':expect['input'],'gt':expect['gt'],'mask':expect['mask']},'cached input source SHA')
    first=order[0]['target_id']
    for position in ('initial','final'):
        pair=d[position+'_numerical_pair']
        require(pair['flow_count']==4 and all(pair[k] is True for k in ('all_flow_outputs_exact','all8_inputs_exact','all8_branches_exact')),'complete four-head/eight-branch numeric verdict')
        for key,prefix in [('new_execution','audit_new_'),('old_actual_execution','audit_old_actual_')]:
            require(pair[key]==lookup[(prefix+position,first,arm)][0],'numerical audit actual forward binding')
        new_inputs=pair['new_execution']['capture_input_sha256']
        require(set(new_inputs)==SELECTED and all(hexdigest(v) for v in new_inputs.values()),'all8 new input capture')
        if arm=='native':require(pair['native_has_no_tiled_dual'] is True,'native no fictitious dual')
        else:
            require(pair['actual_dual_all_flows_branches_exact'] is True,'actual/dual numerical verdict')
            require(pair['old_dual_execution']==lookup[('audit_old_dual_'+position,first,arm)][0],'dual actual forward binding')
            for ex in (pair['old_actual_execution'],pair['old_dual_execution']):
                require({n:r['input_sha256'] for n,r in ex['layers'].items()}==new_inputs,'all8 input hashes actual/dual/new')
    archives=[];preds=[];weight=None
    if mode=='smoke':
        require(d['smoke_weights_discarded'] is True and not d['evaluations'] and 'weights_file' not in d,'smoke discarded/no check inference')
        require(not list(path.parent.glob('*.npz')) and not list(path.parent.glob('*.pt')),'smoke has no retained prediction or weight files')
    else:
        require(d['smoke_weights_discarded'] is False and set(d['evaluations'])=={'initial','endpoint'},'complete evaluation sets')
        require(d['paired_initial_all_eight_native_predictions_exact'] is True,'eight actual initial predictions paired')
        for label,modes in [('initial',(arm,)),('endpoint',ARMS)]:
            frames=d['evaluations'][label]['frames']
            require([f['target_id'] for f in frames]==[r['target_id'] for r in m['check_members']],'all8 ordered checks')
            for f,row in zip(frames,m['check_members']):
                require(f['sequence']==row['sequence'] and set(f['arms'])==set(modes),'check scope/layout coverage')
                name=f['prediction_file'];require(Path(name).name==name,'local archive basename')
                archive=path.parent/name;require(sha(archive)==f['prediction_file_sha256'],'raw archive bytes SHA')
                with np.load(archive,allow_pickle=False) as z:
                    require(set(z.files)==set(modes),'archive exact layout keys')
                    for layout in modes:
                        a=z[layout];payload=f['arms'][layout]
                        require(a.dtype==np.float32 and a.shape==(2,480,640) and np.isfinite(a).all(),'raw prediction dtype/shape/finite')
                        digest=hashlib.sha256(a.tobytes()).hexdigest();require(digest==payload['prediction_sha256'],'raw prediction array SHA')
                        if label=='initial':require(digest==d['binding']['baseline_comparison']['initial_native_prediction_sha256'][row['target_id']],'actual raw initial prediction equals baseline')
                        require(payload['execution']==lookup[('evaluation_'+label,row['target_id'],layout)][0],'prediction bound to actual forward')
                        preds.append({'label':label,'target_id':row['target_id'],'layout':layout,'array_sha256':digest})
                archives.append({'path':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size})
        require(set(p.name for p in path.parent.glob('*.npz'))=={Path(v['path']).name for v in archives},'no unreported NPZ files')
        weight_name=d['weights_file'];require(Path(weight_name).name==weight_name,'local weight basename')
        weight=checkpoint(path.parent/weight_name,d,bn,initial)
    return {'condition':cond,'mode':mode,'arm':arm,'result_path':str(path),'result_sha256':sha(path),
            'forward_count_reconstructed':len(actual),'optimizer_updates_reconstructed':len(steps),
            'forward_purposes':dict(purposes),'service_totals_reconstructed':dict(totals),
            'initial_model_state_sha256':d['initial_model_state_sha256'],'frozen_state_sha256':d['frozen_state_before_sha256'],
            'final_state_sha256':d['final_state_sha256'],'BN_signature_sha256':sig(signatures),
            'recipe_sha256':sig(d['binding']['recipe']),'archives':archives,'predictions':preds,'weights':weight,
            'unique_training_identities':len(set(r['target_id'] for r in order)),
            'smoke_receipts_sha256':d['binding'].get('smoke_receipts_sha256'),
            'numerical_audit_scope':'source and actual-forward receipts verified; all4 flow/eight branch equality is runner asserted, unretained tensors not reexecuted',
            'quality_metrics_read_or_ranked':False,'resource_ledger':check_resources(d)}


BASELINE=ROOT/'raw/experiments_v1/diversity/recover/native/results.json'
OLD_AUDIT=ROOT/'independent_execution_audit/ACTUAL_EXECUTION_AUDIT_V1.json'
OLD_AUDITOR=ROOT/'independent_execution_audit/audit_execution.py'
ROOT_FREEZE=RUNROOT/'EXECUTION_FREEZE_V1.json'
ROOT_FREEZE_SHA='44b0101857918d3b655bf30fdf8f7d8735958e229d77196f1bc414c2d857a572'
OLD_AUDIT_SHA='eb77f69e87d80192157b1b3fd458289cdff56ea07a134bb1738076f9325d9a61'
BASELINE_SHA='254fb4ec5f18d30e4245b4c277ee1c5fb53beaf7fb3cf42cf60d0a4ee4390d0f'


def source_pins():
    require(sha(ROOT_FREEZE)==ROOT_FREEZE_SHA,'Root fixed execution freeze unchanged')
    require(sha(OLD_AUDIT)==OLD_AUDIT_SHA and sha(BASELINE)==BASELINE_SHA,'Original full audit/baseline source unchanged')
    require(sha(OLD_AUDITOR)=='dd57f9f315e1c5f38961208496a37f46db577c47b8507029a55b5bc216d8c8e8','Independent inherited auditor unchanged')
    rootpins=read(ROOT_FREEZE)['source_sha256']
    for p,digest in rootpins.items():require(sha(p)==digest,'Frozen input changed '+p)
    out=pins();out.update(rootpins)
    out.update({str(p):sha(p) for p in (ROOT_FREEZE,OLD_AUDIT,OLD_AUDITOR,BASELINE,RUNROOT/'INDEPENDENT_IMPLEMENTATION_REVIEW.md')})
    audit=read(OLD_AUDIT)
    require(audit['status']=='PASS' and audit['scope']['runs']==16 and audit['weights_count']==8 and audit['prediction_count']==320,'Full original execution audit')
    matches=[r for r in audit['results'] if (r['condition'],r['mode'],r['arm'])==('diversity','recover','native')]
    require(len(matches)==1 and matches[0]['result_sha256']==BASELINE_SHA,'Baseline in independent full16 audit')
    return out


def check_drift(state,initial,reported):
    groups={k:{'initial_squared_L2':0.,'delta_squared_L2':0.,'max_abs_delta':0.,'tensors':0} for k in ('FC','PSN','BN_affine')};layers={}
    for name in sorted(ALLOWED):
        before=initial[name].double();delta=state[name].double()-before
        category='FC' if name.endswith(('.fc1.weight','.fc2.weight')) else 'BN_affine' if '.norm_layer.' in name else 'PSN'
        layer=name.split('.mlp.')[0]+'.mlp'
        if layer not in layers:layers[layer]={'initial_squared_L2':0.,'delta_squared_L2':0.,'max_abs_delta':0.,'tensors':0}
        for record in (groups[category],layers[layer]):
            record['initial_squared_L2']+=float(before.square().sum());record['delta_squared_L2']+=float(delta.square().sum())
            record['max_abs_delta']=max(record['max_abs_delta'],float(delta.abs().max()));record['tensors']+=1
    for record in list(groups.values())+list(layers.values()):
        record['relative_L2']=math.sqrt(record['delta_squared_L2']/record['initial_squared_L2']) if record['initial_squared_L2'] else None
    for key,records in [('categories',groups),('layers',layers)]:
        require(set(reported[key])==set(records),'drift coverage')
        for name,record in records.items():
            require(set(reported[key][name])==set(record),'drift field coverage')
            for field,value in record.items():
                actual=reported[key][name][field]
                require(actual is None if value is None else finite(actual) and math.isclose(actual,value,rel_tol=1e-12,abs_tol=1e-12),'actual FP64 drift '+key+'/'+name+'/'+field)
    return True


def check_resources(d):
    steps=d['steps'];last_alloc=0;last_reserved=0;total=0.
    for step in steps:
        alloc=step['max_allocated_bytes_so_far'];reserved=step['max_reserved_bytes_so_far']
        require(isinstance(alloc,int) and isinstance(reserved,int) and 0<alloc<=reserved and alloc>=last_alloc and reserved>=last_reserved,'monotonic training memory peaks')
        last_alloc,last_reserved=alloc,reserved
        forward=step['forward_seconds'];back=step['backward_optimizer_seconds'];duration=step['synchronized_train_step_seconds']
        require(all(finite(x) and x>=0 for x in (forward,back,duration)) and forward+back<=duration+1e-6,'step clock nesting')
        total+=duration
    require(d['max_allocated_bytes']==last_alloc and d['max_reserved_bytes']==last_reserved,'training peak reduction')
    require(total<=d['training_seconds']+1e-6 and d['training_seconds']<=d['wall_seconds'],'training/wall clock nesting')
    full=d['full_entry_peak_bytes'];initial=d['initial_audit_peak_bytes']
    require(0<initial['allocated']<=initial['reserved'] and 0<full['allocated']<=full['reserved'],'initial/full peak consistency')
    require(full['allocated']>=max(initial['allocated'],last_alloc) and full['reserved']>=max(initial['reserved'],last_reserved),'full peak includes initial/training')
    return {'device_name':d['runtime']['device_name'],'torch':d['runtime']['torch'],'training_seconds':d['training_seconds'],
        'wall_seconds':d['wall_seconds'],'full_entry_peak_bytes':full,'training_step_count':len(steps),
        'interpretation':'Runner recorded PyTorch allocator peaks/host synchronized clocks, checked on CPU; not chip total memory or hardware cycles.'}


def prepare_receipt():
    m=read(ROOT/'data_protocol/manifest_v1.json');memberships(m)
    require(sum(expected_layer(n,'native')[0] for n in GEOMETRY)==558000,'independent native geometry')
    require(sum(expected_layer(n,'FF')[0] for n in GEOMETRY)==558000,'independent FF geometry')
    lengths={mode:len(expected_forwards('diversity',mode,'native',m)) for mode in ('smoke','recover')}
    require(lengths=={'smoke':9,'recover':228},'fixed schedule lengths')
    return {'schema':'flow_native_lr_independent_execution_freeze_v1','status':'PREPARED_BEFORE_NEW_RESULT_READ','auditor_sha256':sha(__file__),
        'source_sha256':source_pins(),'forward_counts':lengths,'FC_rows_each':{'smoke':5022000,'recover':127224000},
        'combined_forwards':237,'combined_updates':205,'new_BN_calibration_forwards':0,'prediction_payloads':24,'prediction_archives':16,
        'runner_modules_imported':False,'quality_metrics_read_or_ranked':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('prepare','audit'),default='prepare')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--freeze',type=Path)
    p.add_argument('--smoke-result',type=Path);p.add_argument('--formal-result',type=Path)
    a=p.parse_args();require(not a.output.exists(),'Refuse overwriting audit evidence')
    if a.stage=='prepare':result=prepare_receipt()
    else:
        require(a.freeze and a.smoke_result,'Audit needs freeze and smoke result')
        frozen=read(a.freeze);require(frozen['auditor_sha256']==sha(__file__) and frozen['source_sha256']==source_pins(),'Frozen independent audit/source unchanged')
        m=read(ROOT/'data_protocol/manifest_v1.json');memberships(m);inputs=read(ROOT/'inputs_v1.json')
        bn,bnreceipt=bn_payload(ROOT/'preparation/bn_buffers.npz',m,inputs)
        initial=reconstruct_initial(ROOT/'source_snapshot/checkpoint_step_000135930.pt',bn)
        paths=[a.smoke_result]+([a.formal_result] if a.formal_result else [])
        results=[check_run(path,m,inputs,bn,initial,frozen) for path in paths]
        require([r['mode'] for r in results]==(['smoke','recover'] if a.formal_result else ['smoke']),'CLI actual ordered modes')
        if a.formal_result:
            require(results[1]['smoke_receipts_sha256']=={'native':sha(a.smoke_result)},'Formal exact one new smoke receipt')
            require(results[0]['initial_model_state_sha256']==results[1]['initial_model_state_sha256'],'fresh W0 after smoke')
            require(results[0]['final_state_sha256']!=results[1]['initial_model_state_sha256'],'smoke endpoint not formal initial')
        totals=Counter()
        for r in results:totals.update(r['service_totals_reconstructed'])
        result={'schema':'flow_native_lr_independent_execution_audit_v1','status':'PASS','auditor_sha256':sha(__file__),
            'freeze_sha256':sha(a.freeze),'root_execution_freeze_sha256':ROOT_FREEZE_SHA,'original_full16_audit_sha256':OLD_AUDIT_SHA,
            'scope':{'runs':len(results),'modes':[r['mode'] for r in results]},'results':results,
            'full_forwards':sum(r['forward_count_reconstructed'] for r in results),'optimizer_updates':sum(r['optimizer_updates_reconstructed'] for r in results),
            'service_totals_from_actual_records':dict(totals),'archive_count':sum(len(r['archives']) for r in results),
            'prediction_count':sum(len(r['predictions']) for r in results),'weights_count':sum(r['weights'] is not None for r in results),
            'common_initial_model_sha256':tensor_digest(initial),'common_frozen_complement_sha256':tensor_digest(initial,ALLOWED),
            'new_BN_calibration_forwards':0,'GPU_used':False,'SSH_used':False,'quality_metrics_read_or_ranked':False,
            'scope_limitations':['Training arrays not reread; accepted input SHA bindings checked.',
              'Unretained numerical-audit intermediate tensors not recomputed; actual forward records and source assertions checked.',
              'Optimizer trajectory not reexecuted; final state, moments and counters checked when formal PT is present.',
              'Smoke has no retained endpoint PT; its endpoint state closure is runtime evidence.',
              'Row/host-time/allocator bookkeeping is not hardware-cycle or benefit evidence.']}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ('status','schema')},sort_keys=True))


if __name__=='__main__':main()
