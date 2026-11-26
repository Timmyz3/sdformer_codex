#!/usr/bin/env python3
"""R13 independent execution audit. No runner imports, GPU, SSH or AEE reduction.

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
ROOT=HERE.parent
R12=ROOT.parent/'ordinary_recovery_round12'
ARMS=('native','FF','CC','fine_s2b2')
CONDS=('reference','diversity')
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
    b=d['binding'];cond=d['condition'];mode=d['mode'];arm=d['arm']
    require(b['condition']==cond and b['arm']==arm and b['mode']==mode,'result/binding arm')
    expected={'runner_sha256':sha(ROOT/'implementation/run_coverage_recovery.py'),
              'execute_sha256':sha(ROOT/'implementation/execute_recovery.py'),
              'wrapper_sha256':sha(R12/'trainable_representative_mlp.py'),
              'runner_checks_sha256':sha(ROOT/'implementation/runner_cpu_results.json'),
              'source_pins_sha256':sha(ROOT/'implementation/inherited_pins.json'),
              'checkpoint_sha256':ORIGINAL_SHA,'loss_sha256':LOSS_SHA,'allowed_parameter_count':7252064,
              'membership_sha256':inputs['membership_sha256'],'runtime_manifest_sha256':inputs['runtime_manifest_sha256'],
              'BN_buffers_sha256':inputs['bn_buffers_sha256'],'BN_receipt_sha256':inputs['bn_receipt_sha256'],
              'region_sha256':read(ROOT/'runtime_v1.json')['region_sha256']}
    for k,v in expected.items():require(b[k]==v,'binding '+k)
    require(b['fit_ids']==[r['target_id'] for r in m['train_order_'+cond]],'binding fit ordered200')
    require(b['check_ids']==([] if mode=='smoke' else [r['target_id'] for r in m['check_members']]),'binding check')
    recipe=b['recipe']
    required={'seed':20261016,'batch':1,'T':10,'input_hw':[480,640],'augmentation':'none','optimizer':'AdamW','lr':.00025,
              'weight_decay':.01,'betas':[.9,.999],'eps':1e-8,'clip_grad_norm':100.,'scheduler':None,
              'precision':'FP32 TF32off noAMP','smoke_steps':5,'recovery_steps':200,
              'fit_targets':{'reference':40,'diversity':200},'check_used_for_training_or_checkpoint_selection':False}
    for k,v in required.items():require(recipe[k]==v,'fixed recipe '+k)
    require(0<b['wall_minutes']<=30,'bounded wall limit')
    require(d['wall_seconds']<=1800,'actual arm wall limit')
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
    require(p['schema']=='flow_round13_recovered_shared_native_state_v1','weight schema')
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
        require(tensor_digest(initial)==d['initial_model_state_sha256'],'actual reconstructed common initial digest')
        changed={n for n in s if not torch.equal(s[n],initial[n])}
        require(changed==set(d['updated_parameter_names']) and changed<=ALLOWED,'actual changed parameters')
    opt=p['optimizer_state_dict'];require(len(opt['param_groups'])==1,'one fresh optimizer group')
    group=opt['param_groups'][0];ids=group['params'];require(len(ids)==80 and len(set(ids))==80 and set(ids)==set(opt['state']),'80 optimizer states')
    for key,wanted in [('lr',.00025),('weight_decay',.01),('eps',1e-8)]:require(group[key]==wanted,'optimizer '+key)
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
    require(d['schema']=='flow_round13_coverage_recovery_v1' and cond in CONDS and mode in ('smoke','recover') and arm in ARMS,'actual run schema')
    require(d['status']==('ROUND13_SMOKE_COMPLETE' if mode=='smoke' else 'ROUND13_COVERAGE_RECOVERY_COMPLETE'),'complete run required')
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
            'quality_metrics_read_or_ranked':False}


def prepare_receipt():
    m=read(ROOT/'data_protocol/manifest_v1.json');memberships(m)
    static={arm:sum(expected_layer(n,arm)[0] for n in GEOMETRY) for arm in ARMS}
    require(static=={'native':558000,'FF':558000,'CC':153600,'fine_s2b2':164200},'independent native shape rows')
    require(sum(expected_layer(n,'CC',True)[0] for n in GEOMETRY)==609600,'dual geometry')
    schedules={cond:{mode:{arm:len(expected_forwards(cond,mode,arm,m)) for arm in ARMS} for mode in ('smoke','recover')} for cond in CONDS}
    return {'schema':'flow_round13_independent_execution_audit_freeze_v1','status':'PREPARED_BEFORE_EXECUTION_RESULT_READ',
            'auditor_sha256':sha(__file__),'source_sha256':pins(),'membership_sha256':sha(ROOT/'data_protocol/manifest_v1.json'),
            'FC_rows_per_full_forward_each':static,'dual_FC_rows_each':609600,'forward_schedules':schedules,
            'required_run_count':16,'smoke_updates':40,'formal_updates':1600,'smoke_forwards':84,'formal_forwards':1964,
            'calibration_forwards':176,'expected_all_full_forwards_including_calibration':2224,
            'quality_metrics_read_or_ranked':False,'runner_modules_imported':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('prepare','audit'),default='prepare')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--freeze',type=Path)
    p.add_argument('--results-root',type=Path)
    p.add_argument('--bn-buffers',type=Path,default=ROOT/'preparation/bn_buffers.npz')
    p.add_argument('--original-checkpoint',type=Path,default=ROOT/'source_snapshot/checkpoint_step_000135930.pt')
    p.add_argument('--conditions',nargs='+',choices=CONDS,default=list(CONDS))
    p.add_argument('--modes',nargs='+',choices=('smoke','recover'),default=['smoke','recover'])
    a=p.parse_args();require(not a.output.exists(),'Refuse overwriting audit evidence')
    if a.stage=='prepare':result=prepare_receipt()
    else:
        require(a.freeze and a.results_root,'Audit requires freeze and actual results root')
        frozen=read(a.freeze);require(frozen['auditor_sha256']==sha(__file__),'Auditor changed after freeze')
        require(pins()==frozen['source_sha256'],'Frozen source/protocol inputs changed')
        require(len(a.conditions)==len(set(a.conditions)) and len(a.modes)==len(set(a.modes)),'no duplicate requested scope')
        m=read(ROOT/'data_protocol/manifest_v1.json');memberships(m);inputs=read(ROOT/'inputs_v1.json')
        bn,receipt=bn_payload(a.bn_buffers,m,inputs)
        initial=reconstruct_initial(a.original_checkpoint,bn) if a.original_checkpoint.is_file() else None
        require(initial is not None,'Original checkpoint needed for complete independent initial-state reconstruction')
        results=[]
        for cond in a.conditions:
            for mode in a.modes:
                for arm in ARMS:
                    path=a.results_root/cond/mode/arm/'results.json'
                    results.append(check_run(path,m,inputs,bn,initial,frozen))
        by_key={(r['condition'],r['mode'],r['arm']):r for r in results}
        for r in results:
            if r['mode']=='recover':
                needed={arm:sha(a.results_root/r['condition']/'smoke'/arm/'results.json') for arm in ARMS}
                require(r['smoke_receipts_sha256']==needed,'formal uses own four exact smoke receipts')
        require(len({r['initial_model_state_sha256'] for r in results})==1,'all runs same reconstructed initial state')
        require(len({r['frozen_state_sha256'] for r in results})==1,'all runs same frozen complement')
        require(len({r['BN_signature_sha256'] for r in results})==1,'all runs same BN signature')
        require(len({r['recipe_sha256'] for r in results})==1,'all runs same recipe')
        initial_predictions_equal=None
        if set(a.conditions)==set(CONDS) and 'recover' in a.modes:
            for arm in ARMS:
                left=[v for v in by_key[('reference','recover',arm)]['predictions'] if v['label']=='initial']
                right=[v for v in by_key[('diversity','recover',arm)]['predictions'] if v['label']=='initial']
                require(left==right,'paired conditions same initial own-layout prediction bytes')
            initial_predictions_equal=True
        aggregate=Counter()
        for r in results:aggregate.update(r['service_totals_reconstructed'])
        result={'schema':'flow_round13_independent_execution_audit_v1','status':'PASS',
                'scope':{'conditions':a.conditions,'modes':a.modes,'runs':len(results)},
                'auditor_sha256':sha(__file__),'pre_result_freeze_sha256':sha(a.freeze),
                'original_checkpoint_sha256':sha(a.original_checkpoint),'new_BN_actual_sha256':sha(a.bn_buffers),
                'common_initial_model_sha256':tensor_digest(initial),'common_frozen_complement_sha256':tensor_digest(initial,ALLOWED),
                'paired_initial_predictions_equal':initial_predictions_equal,
                'optimizer_updates':sum(r['optimizer_updates_reconstructed'] for r in results),
                'full_forwards':sum(r['forward_count_reconstructed'] for r in results),
                'calibration_forwards_separate':176,'service_totals_from_actual_records':dict(aggregate),
                'archive_count':sum(len(r['archives']) for r in results),
                'prediction_count':sum(len(r['predictions']) for r in results),
                'weights_count':sum(r['weights'] is not None for r in results),
                'results':results,'quality_metrics_read_or_ranked':False,'GPU_used':False,'SSH_used':False,
                'scope_limitations':['Actual training arrays not reread by this audit; recorded accepted input SHA bindings checked.',
                  'Unretained intermediate numerical-audit tensors not recomputed; source and two full audit record sets checked.',
                  'Fresh optimizer source and final moments/step checked; training trajectory not rerun.',
                  'Forward/FC rows do not establish hardware cycles or benefit.']}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ('status','schema')},sort_keys=True))

if __name__=='__main__':main()
