#!/usr/bin/env python3
"""Independent CPU audit of completed R27 execution; never reads GT values."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
CHECKPOINT_SHA='1e72cca3946f40e24a1575f3348f9a596d744460808ad910a616ba713b5e4977'
INITIAL_STATE_SHA='b721031729a9a5f69d03037be53e62ebf8c1536dad0ad8a8f8e1601fe9fc2798'
MEMBERS_SHA='40eba2514b56fd33a910fda4e37d4f86e80431ffe8149efc5f79c18827991e7b'
CORE_SHA='19623636229b4dc5a85132e4acc8bf217f9f62a5e51fcab0fdd7394ee186f788'
SMOKE_SHA='48563d36f1b448a66d18278e0448deee02a7650583182a628dcb3865923c6c5c'
PREFIX='sttmultires_unet.encoders.swin3d.'
LINEARS={PREFIX+f'layers.{s}.swin_blocks.{b}.mlp.{fc}' for s,d in enumerate((2,2,6,2)) for b in range(d) for fc in ('fc1','fc2')}
CONVS={PREFIX+f'patch_embed.residual_encoding.resblocks.{b}.conv{j}.0' for b in (0,1) for j in (1,2)}
TARGETS=LINEARS|CONVS
NONE={PREFIX+f'layers.{s}.swin_blocks.{b}.attn.attn_sn.spiking_neuron.{p}' for s,d in enumerate((2,2,6,2)) for b in range(d) for p in ('weight','bias')}


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
        a=work[name];rows=t*b*h*w;tail=t*b*(h*w-(h//9*9)*(w//9*9))
        require(a['mode']==arm and not a['reference'] and a['rows']==rows and a['tail_rows']==tail and a['complete_rows']==rows-tail,'Actual original tile coverage')
        require(a['Cin']==k and a['Cout']==co and a['binary_verified'] and a['int32_accumulation'] and a['scale_applications']==1,'Binary/int32/scaling contract')
        planes=8*rows if arm=='Q8' else 6*rows+2*tail
        require(a['logical_row_plane_work']==planes,'Independent logical plane work')
        limit=max(24,min(65536,8*1024*1024//co));limit=limit//8*8
        expected_calls=0;expected_padding=0
        for bit in range(8):
            count=tail if arm=='Q6' and bit<2 else rows
            for start in range(0,count,limit):
                n=min(limit,count-start);expected_calls+=1;expected_padding+=max(24,math.ceil(n/8)*8)-n
        require(a['actual_integer_kernel_calls']==expected_calls and a['explicit_padded_rows']==expected_padding,'Actual chunk count and explicit zero padding')
        if 'kernel_calls' in a:
            require(len(a['kernel_calls'])==expected_calls and sum(v['rows'] for v in a['kernel_calls'])==planes,'Detailed kernel ledger')
            require(all(v['kind']=='binary_weight_plane' and v['coefficient']==(-128 if v['bit']==7 else 1<<v['bit']) for v in a['kernel_calls']),'Signed bitplane coefficients')
        totals['row_planes']+=planes;totals['kernel_calls']+=expected_calls;totals['explicit_padding_rows']+=expected_padding
    return totals


def parity(p,integer):
    require(p['pass'] is True and all(v['bitwise_equal'] and v['different_bytes']==0 and v['actual_sha256']==v['reference_sha256'] for v in p['checks'].values()),'Every stored full-graph comparison exact')
    keys=set(p['checks']);require(len([k for k in keys if k.startswith('PSN::')])==105,'105 actual PSN captures, including12 diagnostic')
    require({k for k in keys if k.startswith('integer::')}==({'integer::'+k for k in TARGETS} if integer else set()),'Exact original28 integer captures')
    require({k for k in keys if k.startswith('head')}=={'head0','head1','head2','head3'},'All4 heads')


def main():
    p=argparse.ArgumentParser();p.add_argument('--actual-root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    audit_freeze=read(HERE/'AUDITOR_FREEZE_V1.json')
    require(all(sha(HERE/name)==digest for name,digest in audit_freeze['files'].items()),'Pre-quality auditor and protocol source freeze')
    members_path=HERE/'independent_review/membership_v1.json';require(sha(members_path)==MEMBERS_SHA,'Pre-result members')
    m=read(members_path);freeze=HERE/'independent_review/ANALYSIS_FREEZE_V1.json'
    require(sha(a.checkpoint)==CHECKPOINT_SHA,'Original W271482 bytes')
    source=torch.load(a.checkpoint,map_location='cpu',weights_only=True)['model_state_dict']
    initial={k:v for k,v in source.items() if not k.endswith(('.running_mean','.running_var','.num_batches_tracked'))}
    require(len(source)==711 and len(initial)==477 and state_digest(initial)==INITIAL_STATE_SHA,'Original711 to allcurrent477 only')
    del source
    records={r['target_id']:r for r in m['records']};dev=[r['target_id'] for r in m['dev_members']]
    predictions=[];summaries={};source_results={};common_recipe=None;total_calls=0
    for arm in ('native','Q8','Q6'):
        folder=a.actual_root/('formal_'+arm+'_v1');path=folder/'results.json';r=read(path);source_results[arm]=sha(path)
        require(r['schema']=='flow_r27_bounded_recovery_v1' and r['status']=='FIXED512_AND128_DEVELOPMENT_PREDICTIONS_COMPLETE' and r['arm']==arm,'Complete formal arm')
        require(r['sources']['core']==CORE_SHA and r['sources']['smoke_helpers']==SMOKE_SHA and r['sources']['members']==MEMBERS_SHA,'Executed fixed source versions')
        require(r['sources']['runner']==sha(HERE/'implementation/run_recovery.py'),'Actual formal source bytes')
        require(r['completed_optimizer_steps']==512 and r['model_forwards']==r['expected_model_forwards']==(1154 if arm=='native' else 1155),'Exact fixed updates and forwards')
        require(r['initial_state_sha256']==r['teacher_initial_state_sha256']==r['teacher_final_state_sha256']==INITIAL_STATE_SHA,'Same original anchor/fixed teacher')
        require(not r['development_GT_read'] and not r['quality_computed'] and not r['quality_selection'] and not r['formal_resume_from_smoke'],'No development selection or smoke resume')
        require(r['wall_cap_minutes']==60 and not r['runtime']['AMP'] and not r['runtime']['TF32'],'Predeclared cap and numerical precision')
        require(common_recipe is None or r['recipe']==common_recipe,'Matched ordinary recovery recipes');common_recipe=r['recipe']
        require(len(r['trainable_names'])==477 and set(r['trainable_names'])==set(initial),'All477 allowed')
        require(set(r['no_gradient_names'])==NONE and r['actual_optimizer_parameter_count']==453,'Actual453 active and24 diagnostic unused parameters')
        ledger_path=folder/r['training_ledger_file'];require(sha(ledger_path)==r['training_ledger_sha256'],'Ledger bytes')
        steps=[json.loads(line) for line in ledger_path.read_text().splitlines()];require(len(steps)==512,'512 complete updates')
        work_totals=dict(row_planes=0,kernel_calls=0,explicit_padding_rows=0)
        for i,(row,order) in enumerate(zip(steps,m['train_order']),1):
            require(all(row[k]==order[k] for k in ('step','target_id','sequence','pass_index','crop')) and row['step']==i,'Identical frozen identities/order/crops')
            expected={('events' if x['category']=='input' else x['category']):x['sha256'] for x in records[row['target_id']]['source_files']}
            require(row['input_evidence']==expected,'Actual accepted training input SHA')
            require(row['gradients_finite'] and row['gradient_parameter_count']==row['actual_optimizer_parameter_count']==453 and set(row['no_gradient_names'])==NONE,'Finite matched actual update scope')
            require(row['teacher_actual_BN_count']==row['student_actual_BN_count']==78,'All78 current BN calls each teacher/student')
            require(all(math.isfinite(row[k]) and row[k]>=0 for k in ('supervised_loss_px','teacher_mean_L1_px','combined_loss','preclip_grad_norm')),'Finite ordinary loss/gradient trace')
            require(math.isclose(row['combined_loss'],row['supervised_loss_px']+.1*row['teacher_mean_L1_px'],rel_tol=2e-6,abs_tol=1e-7),'Fixed pixel-unit ordinary objective')
            check_calls(row['teacher_affine_calls'],{},'native',(288,384));w=check_calls(row['student_affine_calls'],row['work'],arm,(288,384))
            for k,v in w.items():work_totals[k]+=v
        if arm=='native':require(steps[0]['teacher_mean_L1_px']==0,'Same native teacher/student at step1')
        weights=folder/r['weights_file'];require(sha(weights)==r['weights_sha256'],'Actual final checkpoint bytes')
        cp=torch.load(weights,map_location='cpu',weights_only=True);state=cp['model_state_dict']
        require(set(state)==set(initial) and state_digest(state)==r['final_state_sha256']==r['training_state_sha256']==cp['final_state_sha256'],'Exact final477 state')
        require(cp['completed_optimizer_steps']==512 and cp['arm']==arm and cp['members_sha256']==MEMBERS_SHA and cp['recipe']==r['recipe'],'Fixed final checkpoint provenance')
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
        parity(r['final_training_deployment_parity'],arm!='native')
        if arm!='native':parity(r['final_dense_integer_parity'],True)
        parity_file=folder/'final_parity_heads.npz';require(sha(parity_file)==r['parity_heads_sha256'],'Actual final parity raw archive')
        with np.load(parity_file,allow_pickle=False) as archive:
            require(set(archive.files)=={'head0','head1','head2','head3'},'Four archived full parity heads')
            for name in archive.files:
                array=archive[name];check=r['final_training_deployment_parity']['checks'][name]
                require(array.dtype==np.float32 and list(array.shape)==check['shape'] and np.isfinite(array).all(),'Actual parity head shape/dtype/finite')
                require(hashlib.sha256(array.tobytes()).hexdigest()==check['actual_sha256']==check['reference_sha256'],'Actual archived head equals runtime compared head')
        for stage in ('initial','endpoint'):
            ev=r['evaluations'][stage];require(ev['model_forwards']==64 and ev['GT_arrays_read'] is False and [x['target_id'] for x in ev['frames']]==dev,'Same64 ordered development predictions')
            require(ev['state_sha256']==(INITIAL_STATE_SHA if stage=='initial' else r['final_state_sha256']),'Exact evaluated weight state')
            state_name=('original_native' if arm=='native' else 'initial_'+arm) if stage=='initial' else 'recovered_'+arm
            for f in ev['frames']:
                require(f['sequence']==records[f['target_id']]['sequence'],'Identity sequence')
                require(len(f['actual_BN_calls'])==78 and set(f['actual_BN_calls'].values())=={1},'All78 actual evaluation BN')
                check_calls(f['actual_affine_calls'],f['work'],arm,(480,640))
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
    require(len(predictions)==384 and total_calls==3464,'Complete3-arm execution matrix')
    out=dict(schema='flow_r27_independent_actual_execution_audit_v1',status='PASS',quality_analysis_allowed=True,
             auditor_sha256=sha(__file__),auditor_freeze_sha256=sha(HERE/'AUDITOR_FREEZE_V1.json'),pre_result_freeze_sha256=sha(freeze),source_results=source_results,predictions=predictions,arms=summaries,
             counts=dict(optimizer_updates=1536,quality_predictions=384,quality_archives=384,formal_model_forwards=3464),
             GT_values_read=False,hardware_cycle_claim=False,
             parity_evidence='Independent source review and runtime bytewise comparison receipts for all PSN/integer tensors; independently reloaded actual4 final heads only, intermediate arrays not persisted',
             scope='Completed fixed512 ordinary recovery and exact integer execution; no task-specific scoring victory')
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(status='PASS',counts=out['counts'],arms=summaries)))


if __name__=='__main__':main()
