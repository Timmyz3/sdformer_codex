#!/usr/bin/env python3
"""Independent CPU audit of completed R28 Q4 execution; never reads GT values."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
sys.dont_write_bytecode=True

HERE=Path(__file__).resolve().parent
CHECKPOINT_SHA='1e72cca3946f40e24a1575f3348f9a596d744460808ad910a616ba713b5e4977'
INITIAL_STATE_SHA='b721031729a9a5f69d03037be53e62ebf8c1536dad0ad8a8f8e1601fe9fc2798'
MEMBERS_SHA='40eba2514b56fd33a910fda4e37d4f86e80431ffe8149efc5f79c18827991e7b'
CORE_SHA='f4e4dae18a7e59fab286f2e59f23f72d20583a2b8de22dee4c78204bbfd2fe9a'
SMOKE_SHA='6dadb497bf97fc33929209a3feed0acedcac241036d27bbcda502d8f35244d83'
RUNNER_SHA='5a5bf2abaca8fe23fc269e4dc830acbd3ee6537e52958c29c1520543756a4f95'
GRAPH='flow_r28_all12mlp_patch4conv_integer_Q4_complete9_tailQ8_all78current_v1'
R27=HERE.parent/'round27'
PREFIX='sttmultires_unet.encoders.swin3d.'
LINEARS={PREFIX+f'layers.{s}.swin_blocks.{b}.mlp.{fc}' for s,d in enumerate((2,2,6,2)) for b in range(d) for fc in ('fc1','fc2')}
CONVS={PREFIX+f'patch_embed.residual_encoding.resblocks.{b}.conv{j}.0' for b in (0,1) for j in (1,2)}
TARGETS=LINEARS|CONVS
NONE={PREFIX+f'layers.{s}.swin_blocks.{b}.attn.attn_sn.spiking_neuron.{p}' for s,d in enumerate((2,2,6,2)) for b in range(d) for p in ('weight','bias')}
EXPECTED_RECIPE=dict(updates=512,seed=20261019,optimizer='AdamW',lr=2.5e-5,weight_decay=.01,
    betas=[.9,.999],eps=1e-8,clip_grad_norm=100.,teacher_weight=.1,all_parameters=477,
    quantized_affine_modules=28,flow_scaling=1.,crop=[288,384],augmentation='fixed_crop_only_no_flip_or_rotation',
    teacher='fixed_original_W271482_native_all78current',BN='all78current',dropout_mode='model.eval()',
    formal_resume_from_smoke=False,train_members=256,passes=2,development_members=64,
    checkpoint_selection='fixed_final_step512_only',training_mode='eval_with_grad_currentBN_no_stochastic_depth',
    scope='bounded_ordinary_recovery_not_converged_original_training')


def require(ok,msg):
    if not ok:raise ValueError(msg)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())


def state_digest(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        t=t.detach().cpu().contiguous()
        for part in (name.encode(),str(t.dtype).encode(),str(tuple(t.shape)).encode(),t.numpy().tobytes()):h.update(part)
    return h.hexdigest()


def check_calls(calls,work,arm,geometry):
    require(arm in ('native','Q4'),'Only native teacher or fixed Q4 student')
    require(set(calls)==TARGETS,'Exactly24 FC and4 conv callables')
    require(set(work)==(set() if arm=='native' else TARGETS),'Work only for actual integer targets')
    totals=dict(row_planes=0,kernel_calls=0,explicit_padding_rows=0)
    for name,records in calls.items():
        require(len(records)==1,'Every target called once')
        r=records[0];ins,outs=r['input_shape'],r['output_shape']
        require(len(ins)==len(outs)==5 and ins[0]==outs[0]==10 and ins[1]==outs[1]==1,'Native T10/B1')
        if name in LINEARS:
            t,b,h,w,ci=ins;co=outs[-1];require(outs[:4]==ins[:4],'Full FC grid retained');k=ci
            stage=int(name.split('.layers.',1)[1].split('.')[0]);channels=96*2**stage
            eh,ew=geometry[0]//(4*2**stage),geometry[1]//(4*2**stage)
            eci,eco=(channels,4*channels) if name.endswith('.fc1') else (4*channels,channels)
            require((h,w,ci,co)==(eh,ew,eci,eco),'Exact native stage geometry/channels')
        else:
            t,b,ci,h,w=ins;co=outs[2];require(outs[:2]+outs[3:]==ins[:2]+ins[3:],'Full conv grid retained');k=ci*9
            require((h,w,ci,co)==(geometry[0]//2,geometry[1]//2,96,96),'Exact original patch conv geometry/channels')
        if arm=='native':continue
        require(r['module_class']=='STEAffine','Actual installed integer wrapper callable')
        a=work[name];rows=t*b*h*w;tail=t*b*(h*w-(h//9*9)*(w//9*9))
        require(a['mode']==arm and not a['reference'] and a['rows']==rows and a['tail_rows']==tail and a['complete_rows']==rows-tail,'Actual original tile coverage')
        require(a['Cin']==k and a['Cout']==co and a['binary_verified'] and a['int32_accumulation'] and a['scale_applications']==1,'Binary/int32/scaling contract')
        require(a['low_bit_rule']=='arithmetic_shift_drop4_no_reclamp_no_offset' and a['software_plane_dtype']=='int8'
                and a['quantization_device']=='cpu' and a['spatial_rows_retained'] and a['full_T_and_channels'] and not a['cycle_claim'],'Declared actual drop4 full-grid implementation')
        planes=4*rows+4*tail
        require(a['logical_row_plane_work']==planes,'Independent logical plane work')
        limit=max(24,min(65536,8*1024*1024//co));limit=limit//8*8
        expected_calls=0;expected_padding=0;expected_ledger=[]
        for bit in range(8):
            count=tail if bit<4 else rows
            for start in range(0,count,limit):
                n=min(limit,count-start);expected_calls+=1;expected_padding+=max(24,math.ceil(n/8)*8)-n
                expected_ledger.append(dict(kind='binary_weight_plane',bit=bit,rows=n,padding_rows=max(24,math.ceil(n/8)*8)-n,
                                           scope='tail' if bit<4 else 'all',coefficient=-128 if bit==7 else 1<<bit))
        require(a['actual_integer_kernel_calls']==expected_calls and a['explicit_padded_rows']==expected_padding,'Actual chunk count and explicit zero padding')
        if 'kernel_calls' in a:
            require(a['kernel_calls']==expected_ledger,'Every actual plane/chunk/sign/tail scope/padded-row record')
        totals['row_planes']+=planes;totals['kernel_calls']+=expected_calls;totals['explicit_padding_rows']+=expected_padding
    return totals


def dense_reference_work(reference,serial):
    require(set(reference)==set(serial)==TARGETS,'All28 dense integer reference records')
    for name,a in reference.items():
        s=serial[name]
        for key in ('mode','rows','Cin','Cout','tail_rows','complete_rows','binary_verified','int32_accumulation',
                    'scale_applications','spatial_rows_retained','full_T_and_channels','low_bit_rule','quantization_device','software_plane_dtype','cycle_claim'):
            require(a[key]==s[key],'Dense reference same actual domain and encoding')
        require(a['reference'] is True and a['logical_row_plane_work'] is None,'Actual dense integer reference')
        limit=max(24,min(65536,8*1024*1024//a['Cout']));limit=limit//8*8;expected=[]
        for scope,nrows in (('complete',a['complete_rows']),('tail',a['tail_rows'])):
            for start in range(0,nrows,limit):
                n=min(limit,nrows-start)
                expected.append(dict(kind='dense_integer_reference',bit=None,rows=n,padding_rows=max(24,math.ceil(n/8)*8)-n,scope=scope,coefficient=None))
        require(a['kernel_calls']==expected and a['actual_integer_kernel_calls']==len(expected)
                and a['explicit_padded_rows']==sum(v['padding_rows'] for v in expected),'Dense reference actual chunks and padding')


def parity(p,integer,psn_names=None):
    require(p['pass'] is True and all(v['bitwise_equal'] and v['different_bytes']==0 and v['actual_sha256']==v['reference_sha256'] for v in p['checks'].values()),'Every stored full-graph comparison exact')
    keys=set(p['checks']);require(len([k for k in keys if k.startswith('PSN::')])==105,'105 actual PSN captures, including12 diagnostic')
    if psn_names is not None:require({k for k in keys if k.startswith('PSN::')}==psn_names,'All exact source PSN names')
    require({k for k in keys if k.startswith('integer::')}==({'integer::'+k for k in TARGETS} if integer else set()),'Exact original28 integer captures')
    require({k for k in keys if k.startswith('head')}=={'head0','head1','head2','head3'},'All4 heads')


def main():
    p=argparse.ArgumentParser();p.add_argument('--actual-root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    require(not a.output.exists(),'Fresh independent audit output')
    audit_freeze=read(HERE/'AUDITOR_FREEZE_V1.json')
    require(all(sha(HERE/name)==digest for name,digest in audit_freeze['files'].items()),'Pre-quality auditor and protocol source freeze')
    members_path=R27/'independent_review/membership_v1.json';require(sha(members_path)==MEMBERS_SHA,'Pre-result members')
    m=read(members_path);freeze=R27/'independent_review/ANALYSIS_FREEZE_V1.json'
    require(sha(a.checkpoint)==CHECKPOINT_SHA,'Original W271482 bytes')
    source=torch.load(a.checkpoint,map_location='cpu',weights_only=True)['model_state_dict']
    initial={k:v for k,v in source.items() if not k.endswith(('.running_mean','.running_var','.num_batches_tracked'))}
    require(len(source)==711 and len(initial)==477 and state_digest(initial)==INITIAL_STATE_SHA,'Original711 to allcurrent477 only')
    bn_names={k.removesuffix('.running_mean') for k in source if k.endswith('.running_mean')}
    psn_names={'PSN::'+k.removesuffix('.weight') for k in initial if k.endswith('.spiking_neuron.weight')}
    require(len(bn_names)==78 and len(psn_names)==105,'Exact original78 BN and105 PSN identities')
    del source
    records={r['target_id']:r for r in m['records']};dev=[r['target_id'] for r in m['dev_members']]
    predictions=[];summaries={};source_results={};common_recipe=None;total_calls=0
    for arm in ('Q4',):
        folder=a.actual_root/('formal_'+arm+'_v2');path=folder/'results.json';r=read(path);source_results[arm]=sha(path)
        require(r['schema']=='flow_r28_bounded_recovery_v1' and r['graph_id']==GRAPH and r['status']=='FIXED512_AND128_DEVELOPMENT_PREDICTIONS_COMPLETE' and r['arm']==arm,'Complete formal arm')
        require(r['sources']['core']==CORE_SHA and r['sources']['smoke_helpers']==SMOKE_SHA and r['sources']['members']==MEMBERS_SHA,'Executed fixed source versions')
        require(r['sources']['runner']==RUNNER_SHA==sha(HERE/'implementation/run_recovery_v2.py'),'Actual formal source bytes')
        require(r['completed_optimizer_steps']==512 and r['model_forwards']==r['expected_model_forwards']==(1154 if arm=='native' else 1155),'Exact fixed updates and forwards')
        require(r['initial_state_sha256']==r['teacher_initial_state_sha256']==r['teacher_final_state_sha256']==INITIAL_STATE_SHA,'Same original anchor/fixed teacher')
        require(not r['development_GT_read'] and not r['quality_computed'] and not r['quality_selection'] and not r['formal_resume_from_smoke'],'No development selection or smoke resume')
        require(r['wall_cap_minutes']==60 and not r['runtime']['AMP'] and not r['runtime']['TF32'],'Predeclared cap and numerical precision')
        require(r['recipe']==EXPECTED_RECIPE,'Unchanged exact original R27 ordinary recovery recipe');common_recipe=r['recipe']
        require(len(r['trainable_names'])==477 and set(r['trainable_names'])==set(initial),'All477 allowed')
        require(set(r['no_gradient_names'])==NONE and r['actual_optimizer_parameter_count']==453,'Actual453 active and24 diagnostic unused parameters')
        ledger_path=folder/r['training_ledger_file'];require(sha(ledger_path)==r['training_ledger_sha256'],'Ledger bytes')
        require(ledger_path.resolve().is_relative_to(folder.resolve()),'Ledger belongs to current arm')
        steps=[json.loads(line) for line in ledger_path.read_text().splitlines()];require(len(steps)==512,'512 complete updates')
        work_totals=dict(row_planes=0,kernel_calls=0,explicit_padding_rows=0)
        for i,(row,order) in enumerate(zip(steps,m['train_order']),1):
            require(all(row[k]==order[k] for k in ('step','target_id','sequence','pass_index','crop')) and row['step']==i,'Identical frozen identities/order/crops')
            expected={('events' if x['category']=='input' else x['category']):x['sha256'] for x in records[row['target_id']]['source_files']}
            require(row['input_evidence']==expected,'Actual accepted training input SHA')
            require(row['gradients_finite'] and row['gradient_parameter_count']==row['actual_optimizer_parameter_count']==453 and set(row['no_gradient_names'])==NONE,'Finite matched actual update scope')
            require(row['teacher_actual_BN_count']==row['student_actual_BN_count']==78,'All78 current BN calls each teacher/student')
            if i in (1,512):
                for label in ('teacher_BN_calls','student_BN_calls'):
                    require(set(row[label])==bn_names and set(row[label].values())=={1},'Exact78 BN first/last names')
                require(set(row['gradient_norms'])==set(initial)-NONE and all(math.isfinite(v) and v>=0 for v in row['gradient_norms'].values()),'All453 actual first/last gradient norms')
                require(all('kernel_calls' in w for w in row['work'].values()),'First/last detailed actual integer work')
            require(all(math.isfinite(row[k]) and row[k]>=0 for k in ('supervised_loss_px','teacher_mean_L1_px','combined_loss','preclip_grad_norm')),'Finite ordinary loss/gradient trace')
            require(math.isclose(row['combined_loss'],row['supervised_loss_px']+.1*row['teacher_mean_L1_px'],rel_tol=2e-6,abs_tol=1e-7),'Fixed pixel-unit ordinary objective')
            check_calls(row['teacher_affine_calls'],{},'native',(288,384));w=check_calls(row['student_affine_calls'],row['work'],arm,(288,384))
            for k,v in w.items():work_totals[k]+=v
        if arm=='native':require(steps[0]['teacher_mean_L1_px']==0,'Same native teacher/student at step1')
        weights=folder/r['weights_file'];require(sha(weights)==r['weights_sha256'],'Actual final checkpoint bytes')
        require(weights.resolve().is_relative_to(folder.resolve()),'Weights belong to current arm')
        cp=torch.load(weights,map_location='cpu',weights_only=True);state=cp['model_state_dict']
        require(set(state)==set(initial) and state_digest(state)==r['final_state_sha256']==r['training_state_sha256']==cp['final_state_sha256'],'Exact final477 state')
        require(cp['completed_optimizer_steps']==512 and cp['arm']==arm and cp['members_sha256']==MEMBERS_SHA and cp['recipe']==r['recipe'],'Fixed final checkpoint provenance')
        require(cp['schema']=='flow_r28_fixed512_checkpoint_v1' and cp['initial_state_sha256']==INITIAL_STATE_SHA and cp['sources']==r['sources']
                and cp['fixed_endpoint_no_development_selection'] is True,'Checkpoint exact initial/source/fixed endpoint binding')
        require(all(state[k].shape==v.shape and state[k].dtype==v.dtype and bool(torch.isfinite(state[k]).all()) for k,v in initial.items()),'Finite actual matching tensors')
        changed=[k for k in state if not torch.equal(state[k],initial[k])]
        require(changed and not set(changed)&NONE,'Unused parameters unchanged; actual training changed student')
        opt=cp['optimizer_state_dict'];require(len(opt['param_groups'])==1 and len(opt['param_groups'][0]['params'])==477 and len(opt['state'])==453,'Real full optimizer with453 active states')
        group=opt['param_groups'][0]
        require(group['lr']==2.5e-5 and group['weight_decay']==.01 and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8,'Fixed optimizer hyperparameters')
        for j,name in zip(group['params'],r['trainable_names']):
            require((j not in opt['state'])==(name in NONE),'Optimizer membership matches actual parameter names')
            if j not in opt['state']:continue
            st=opt['state'][j];require(float(st['step'])==512,'Every active optimizer state reached512')
            require(all(st[k].shape==state[name].shape and bool(torch.isfinite(st[k]).all()) for k in ('exp_avg','exp_avg_sq')),'Actual finite Adam moments')
        del cp,state,opt
        parity(r['final_training_deployment_parity'],True,psn_names)
        parity(r['final_dense_integer_parity'],True,psn_names)
        parity_records=[r['final_training_deployment_parity'][k] for k in ('reference_record','actual_record')]+[r['final_dense_integer_parity']['dense_record']]
        for q,(ste,reference) in zip(parity_records,((True,False),(False,False),(False,True))):
            require(set(q['actual_BN_calls'])==bn_names and set(q['actual_BN_calls'].values())=={1},'All78 parity BN names/calls')
            require(set(q['actual_target_calls'])==TARGETS and set(q['actual_target_calls'].values())=={1},'All28 parity target calls')
            require((q['PSN_count'],q['integer_count'],q['head_count'])==(105,28,4) and q['ste_entry'] is ste and q['dense_integer_reference'] is reference,'Three distinct parity entry contracts')
        full_calls=r['evaluations']['initial']['frames'][0]['actual_affine_calls']
        for q in parity_records[:2]:check_calls(full_calls,q['work'],'Q4',(480,640))
        dense_reference_work(parity_records[2]['work'],parity_records[0]['work'])
        parity_file=folder/'final_parity_heads.npz';require(sha(parity_file)==r['parity_heads_sha256'],'Actual final parity raw archive')
        with np.load(parity_file,allow_pickle=False) as archive:
            require(set(archive.files)=={'head0','head1','head2','head3'},'Four archived full parity heads')
            for name in archive.files:
                array=archive[name];check=r['final_training_deployment_parity']['checks'][name]
                require(array.dtype==np.float32 and array.shape==(1,2,480,640) and list(array.shape)==check['shape'] and np.isfinite(array).all(),'Actual parity head shape/dtype/finite')
                require(hashlib.sha256(array.tobytes()).hexdigest()==check['actual_sha256']==check['reference_sha256'],'Actual archived head equals runtime compared head')
        for stage in ('initial','endpoint'):
            ev=r['evaluations'][stage];require(ev['model_forwards']==64 and ev['GT_arrays_read'] is False and [x['target_id'] for x in ev['frames']]==dev,'Same64 ordered development predictions')
            require(ev['state_sha256']==(INITIAL_STATE_SHA if stage=='initial' else r['final_state_sha256']),'Exact evaluated weight state')
            state_name=('original_native' if arm=='native' else 'initial_'+arm) if stage=='initial' else 'recovered_'+arm
            for f in ev['frames']:
                require(f['sequence']==records[f['target_id']]['sequence'],'Identity sequence')
                require(set(f['actual_BN_calls'])==bn_names and set(f['actual_BN_calls'].values())=={1},'All78 actual evaluation BN')
                check_calls(f['actual_affine_calls'],f['work'],arm,(480,640))
                require(set(f['heads'])=={'head0','head1','head2','head3'} and all(v['shape']==[2,480,640] and v['dtype']=='torch.float32' for v in f['heads'].values()),'All4 actual full-resolution head records')
                expected={('events' if x['category']=='input' else x['category']):dict(relative_path=x['relative_path'],sha256=x['sha256']) for x in records[f['target_id']]['source_files']}
                require(f['input_evidence']==expected,'Bound development sources')
                pred=folder/f['prediction_file'];require(pred.resolve().is_relative_to(folder.resolve()) and sha(pred)==f['prediction_file_sha256'],'Actual prediction file')
                with np.load(pred,allow_pickle=False) as archive:
                    require(set(archive.files)=={'head3'},'Only declared head3 raw');array=archive['head3']
                require(array.shape==(2,480,640) and array.dtype==np.float32 and np.isfinite(array).all(),'Full finite pixel flow')
                digest=hashlib.sha256(array.tobytes()).hexdigest();require(digest==f['prediction_sha256']==f['heads']['head3']['sha256'],'Actual head3 bytes')
                predictions.append(dict(state=state_name,target_id=f['target_id'],sequence=f['sequence'],file=str(pred.resolve()),file_sha256=sha(pred),head3_sha256=digest,input_evidence=expected))
        summaries[arm]=dict(result_sha256=sha(path),checkpoint_sha256=sha(weights),ledger_sha256=sha(ledger_path),changed_parameter_tensors=len(changed),
                            updates=512,model_forwards=r['model_forwards'],actual_optimizer_parameters=453,allowed_parameters=477,training_work=work_totals,
                            wall_seconds=r['wall_seconds'],peak_allocated_bytes=r['max_allocated_bytes'])
        total_calls+=r['model_forwards']
    require(len(predictions)==128 and total_calls==1155,'Complete fixed single Q4 arm')
    out=dict(schema='flow_r28_independent_actual_execution_audit_v1',status='PASS',quality_analysis_allowed=True,
             auditor_sha256=sha(__file__),auditor_freeze_sha256=sha(HERE/'AUDITOR_FREEZE_V1.json'),pre_result_freeze_sha256=sha(freeze),source_results=source_results,predictions=predictions,arms=summaries,
             counts=dict(optimizer_updates=512,quality_predictions=128,quality_archives=128,formal_model_forwards=1155),
             GT_values_read=False,hardware_cycle_claim=False,
             graph_id=GRAPH,entry_scope_correction_sha256=sha(HERE/'ENTRY_SCOPE_CORRECTION_V2.md'),
             pre_result_freeze_source=str(freeze.resolve()),
             parity_evidence='Independent source review and runtime bytewise comparison receipts for all PSN/integer tensors; independently reloaded actual4 final heads only, intermediate arrays not persisted',
             scope='Completed fixed512 ordinary recovery and exact integer execution; no task-specific scoring victory')
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(status='PASS',counts=out['counts'],arms=summaries)))


if __name__=='__main__':main()
