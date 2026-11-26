#!/usr/bin/env python3
"""Independent actual E0 RNE evidence audit; no GT values or quality selection."""
import argparse,collections,hashlib,importlib.util,json,math,sys
from pathlib import Path
import numpy as np
import torch
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import audit_e0_work_v1 as mw
BASE=HERE.parent/'e0_ordinary'
OLD=HERE.parents[1]/'flow_story_20261019'
ORIGINAL='b721031729a9a5f69d03037be53e62ebf8c1536dad0ad8a8f8e1601fe9fc2798'
CHECKPOINT_SHA='1e72cca3946f40e24a1575f3348f9a596d744460808ad910a616ba713b5e4977'
MEMBERS_SHA='40eba2514b56fd33a910fda4e37d4f86e80431ffe8149efc5f79c18827991e7b'
require=mw.require
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def child(folder,name):
    p=folder/name;require(p.resolve().is_relative_to(folder.resolve()),'Artifact belongs to actual run');return p
def bind_source():
    frozen=read(BASE/'SOURCE_FREEZE_V1.json')
    require(all(sha(BASE/p)==h for p,h in frozen['files'].items()),'Actual frozen E0 source bytes')
    p=OLD/'round28/audit_actual_recovery_v1.py'
    require(sha(p)=='eb433542a13b6f2df185b503b922ab98fe648d9e69c9d51b956eb5e8654df71b','Frozen generic state/parity helper')
    spec=importlib.util.spec_from_file_location('_old_actual_audit',p);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    return old

def representation_hashes(state):
    out={}
    for name in mw.TARGETS:
        w=state[name+'.weight'].detach().cpu().numpy().reshape(state[name+'.weight'].shape[0],-1)
        maximum=np.max(np.abs(w),axis=1);s8=maximum/np.float32(127);s4=maximum/np.float32(7)
        q8=np.rint(w/np.where(s8==0,np.float32(1),s8)[:,None]).clip(-127,127).astype(np.int8)
        q4=np.rint(w/np.where(s4==0,np.float32(1),s4)[:,None]).clip(-8,7).astype(np.int8)
        low=(q4.astype(np.int16)<<4).astype(np.int8)
        out[name]=dict(q8_sha256=mw.digest(q8),s8_sha256=mw.digest(s8),q4_sha256=mw.digest(q4),s4_sha256=mw.digest(s4),low_q_sha256=mw.digest(low),low_scale_sha256=mw.digest(s4/np.float32(16)))
    return out

def check_representations(work,expected,mode):
    for name,w in work.items():
        keys=('q8_sha256','s8_sha256') if mode=='Q8' else tuple(expected[name])
        require(all(w['representation'][k]==expected[name][k] for k in keys),'Independent actual checkpoint quantizer hashes: '+name)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--mask-contract',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args();require(not a.output.exists(),'Fresh audit output');torch.set_num_threads(2);old=bind_source()
    rp=a.run/'results.json';r=read(rp);contract=read(a.mask_contract);n=1 if a.smoke else 512;nf=5 if a.smoke else 1353
    require(r['schema']=='flow_e0_ordinary_quantizer_v1' and r['graph_id']=='flow_e0_rne4_independent_scales_all12mlp_patch4_currentBN_v1' and r['arm']=='joint','Unique actual graph')
    require(r['engineering_smoke'] is a.smoke and r['status']==('MIXED_ONE_UPDATE_AND_PARITY_COMPLETE' if a.smoke else 'RNE4_JOINT512_AND320_DEVELOPMENT_PREDICTIONS_COMPLETE'),'Complete declared execution')
    require(r['model_forwards']==r['expected_model_forwards']==nf and r['completed_optimizer_steps']==n,'Exact actually completed forward/update budget')
    require(r['wall_cap_minutes']==(20 if a.smoke else 60) and not r['runtime']['AMP'] and not r['runtime']['TF32'],'Fixed run cap and numerics')
    require(all(r[k] is False for k in ('development_GT_read','quality_computed','quality_selection','formal_resume_from_smoke')),'No dev GT/selection or smoke resume')
    files=dict(runner=BASE/'run_joint_recovery.py',core=BASE/'mixed_integer_core.py',mask_helper=BASE/'mask_contract.py',smoke_helpers=OLD/'round28/implementation/run_smoke_v2.py',formal_helpers=OLD/'round28/implementation/run_recovery_v2.py',members=OLD/'round27/independent_review/membership_v1.json',mask_contract=a.mask_contract)
    require(r['sources']=={k:sha(v) for k,v in files.items()},'Actual exact executed production sources')
    require(r['sources']['members']==MEMBERS_SHA and r['joint_contract']['mask_contract']==contract and r['joint_contract']['mask_contract_sha256']==sha(a.mask_contract),'Exact external non-GT mask contract')
    require(r['joint_contract']['seed']==20261019 and r['joint_contract']['smoke_state_never_resumed'] is True,'Fixed seed and fresh formal start')
    modes={'M':1} if a.smoke else {'Q4':256,'Q8':128,'M':128}
    require(r['joint_contract']['mode_counts']==r['actual_student_mode_counts']==modes,'Fixed student mode counts')
    require(r['recipe']==old.EXPECTED_RECIPE,'Unchanged original ordinary recovery recipe; smoke is one engineering update only')
    require(sha(a.checkpoint)==CHECKPOINT_SHA,'Actual original source checkpoint')
    allstate=torch.load(a.checkpoint,map_location='cpu',weights_only=True)['model_state_dict']
    initial={k:v for k,v in allstate.items() if not k.endswith(('.running_mean','.running_var','.num_batches_tracked'))}
    bn={k.removesuffix('.running_mean') for k in allstate if k.endswith('.running_mean')}
    psn={'PSN::'+k.removesuffix('.weight') for k in initial if k.endswith('.spiking_neuron.weight')}
    require(len(allstate)==711 and len(initial)==477 and len(bn)==78 and len(psn)==105 and old.state_digest(initial)==ORIGINAL,'Actual original state/domain')
    require(r['initial_state_sha256']==r['teacher_initial_state_sha256']==r['teacher_final_state_sha256']==ORIGINAL,'Original start and unchanged actual teacher')
    require(set(r['trainable_names'])==set(initial) and len(r['trainable_names'])==477 and set(r['no_gradient_names'])==old.NONE and r['actual_optimizer_parameter_count']==453,'Allowed477/active453/diagnostic24')
    m=read(files['members']);records={x['target_id']:x for x in m['records']};dev=m['dev_members'];sequences=m['sequences']
    def evidence(tid,nested=True):
        return {('events' if x['category']=='input' else x['category']):(dict(relative_path=x['relative_path'],sha256=x['sha256']) if nested else x['sha256']) for x in records[tid]['source_files']}
    def layout(record,geometry,step=None):
        chosen=mw.expected_selections(contract,geometry,step)
        require(record['geometry']==geometry and record['step']==step and record['selected_tiles']==chosen,'Explicit deterministic non-GT mask schedule')
        for group,cells in chosen.items():
            q=contract['geometries'][geometry][group];z=np.zeros((q['H'],q['W']),dtype=bool)
            for y,x in cells:z[y*9:y*9+9,x*9:x*9+9]=True
            require(record['mask_sha256'][group]==mw.digest(z),'Independent actual spatial mask bytes')
        require(set(record['mask_sha256'])==mw.GROUPS,'Every mask group bound')
    lp=child(a.run,r['training_ledger_file']);require(sha(lp)==r['training_ledger_sha256'],'Actual immutable training ledger')
    ledger=[json.loads(x) for x in lp.read_text().splitlines()];require(len(ledger)==n,'All actual updates present')
    totals=dict(row_planes=0,kernel_calls=0,explicit_padding_rows=0);counts=collections.Counter();seen=set()
    for row,order in zip(ledger,m['train_order']):
        step=order['step'];mode='M' if a.smoke else ('Q4','Q4','Q8','M')[(order['sequence_visit_index']-1+sequences.index(order['sequence'])+2*(order['pass_index']-1))%4]
        require(all(row[k]==order[k] for k in ('step','target_id','sequence','pass_index','crop')) and row['mode']==mode and row['input_evidence']==evidence(row['target_id'],False),'Actual scheduled identity/crop/mode/input')
        if mode=='M':layout(row['mask_layout'],'crop_center9',step)
        else:require(row['mask_layout'] is None,'No hidden uniform-mode mask')
        detailed=mode not in seen or step==n;seen.add(mode);counts[mode]+=1
        require(row['detailed_kernel_records'] is detailed,'First each mode and final step full ledger')
        require(row['gradients_finite'] and row['gradient_parameter_count']==row['actual_optimizer_parameter_count']==453 and set(row['no_gradient_names'])==old.NONE,'Actual finite active updates')
        require(row['teacher_actual_BN_count']==row['student_actual_BN_count']==78,'All current BN in teacher and student')
        if detailed:
            require(all(set(row[k])==bn and set(row[k].values())=={1} for k in ('teacher_BN_calls','student_BN_calls')),'Exact BN names')
            require(set(row['gradient_norms'])==set(initial)-old.NONE and all(math.isfinite(x) and x>=0 for x in row['gradient_norms'].values()),'All active actual gradient norms')
            require(all('kernel_calls' in w for w in row['work'].values()),'Full actual mode kernel ledger')
        require(all(math.isfinite(row[k]) and row[k]>=0 for k in ('supervised_loss_px','teacher_mean_L1_px','combined_loss','preclip_grad_norm')),'Finite actual loss/gradient trace')
        require(math.isclose(row['combined_loss'],row['supervised_loss_px']+.1*row['teacher_mean_L1_px'],rel_tol=2e-6,abs_tol=1e-7),'Unchanged loss units and coefficients')
        mw.check_calls(row['teacher_affine_calls'],{},'native',(288,384),contract)
        work=mw.check_calls(row['student_affine_calls'],row['work'],mode,(288,384),contract,step=step if mode=='M' else None)
        for k,v in work.items():totals[k]+=v
    require(dict(counts)==modes,'Independent actual mixed mode totals')
    wp=child(a.run,r['weights_file']);require(sha(wp)==r['weights_sha256'],'Actual final checkpoint bytes')
    cp=torch.load(wp,map_location='cpu',weights_only=True);state=cp['model_state_dict'];final=old.state_digest(state)
    require(cp['schema']=='flow_e0_rne4_joint_checkpoint_v1' and cp['arm']=='joint' and cp['completed_optimizer_steps']==n and cp['initial_state_sha256']==ORIGINAL,'Actual fixed final checkpoint origin')
    require(set(state)==set(initial) and final==cp['final_state_sha256']==r['final_state_sha256']==r['training_state_sha256'] and final!=ORIGINAL,'Same changed477 final state')
    require(cp['sources']==r['sources'] and cp['recipe']==r['recipe'] and cp['joint_contract']==r['joint_contract'] and cp['members_sha256']==MEMBERS_SHA and cp['fixed_endpoint_no_development_selection'] is True and cp['formal_resume_allowed'] is False,'Checkpoint actual run/source/schedule binding')
    require(all(x.shape==state[k].shape and x.dtype==state[k].dtype and bool(torch.isfinite(state[k]).all()) for k,x in initial.items()),'Finite matching final tensors')
    changed=[k for k in initial if not torch.equal(initial[k],state[k])];require(changed and not(set(changed)&old.NONE),'Real parameter changes, unused parameters unchanged')
    opt=cp['optimizer_state_dict'];require(len(opt['param_groups'])==1 and len(opt['state'])==453,'Actual optimizer state count')
    group=opt['param_groups'][0];require(len(group['params'])==477 and group['lr']==2.5e-5 and group['weight_decay']==.01 and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8,'Actual optimizer scope/hyperparameters')
    for idx,name in zip(group['params'],r['trainable_names']):
        require((idx not in opt['state'])==(name in old.NONE),'Actual named optimizer membership')
        if idx not in opt['state']:continue
        st=opt['state'][idx];require(float(st['step'])==n and all(st[k].shape==state[name].shape and bool(torch.isfinite(st[k]).all()) for k in ('exp_avg','exp_avg_sq')),'Every actual finite Adam state at final step')
    representations=dict(initial=representation_hashes(initial),final=representation_hashes(state))
    require(cp['quantizer']=='independent_maxabs_rne4' and cp['checkpoint_role']=='endpoint_only_not_eligible_for_training_resume','Actual RNE endpoint role')
    del cp,state,opt,initial,allstate
    evaluations={} if a.smoke else r['evaluations'];stages={'initial_Q4':('Q4','initial_rne_Q4'),'initial_M':('M','initial_rne_M'),**{'endpoint_'+mode:(mode,'recovered_rne_'+mode) for mode in ('Q4','Q8','M')}}
    require(set(evaluations)==(set() if a.smoke else set(stages)),'Only predeclared320 quality evaluations')
    predictions=[];full_calls=None
    for stage,ev in evaluations.items():
        mode,label=stages[stage];require(ev['model_forwards']==64 and ev['GT_arrays_read'] is False and ev['state_sha256']==(ORIGINAL if stage.startswith('initial_') else final),'Correct initial or final64 state')
        require([(x['target_id'],x['sequence']) for x in ev['frames']]==[(x['target_id'],x['sequence']) for x in dev],'Same64 fixed dev order')
        if mode=='M':layout(ev['mask_layout'],'full9')
        else:require(ev['mask_layout'] is None,'Uniform endpoint no mask')
        for f in ev['frames']:
            require(f['input_evidence']==evidence(f['target_id']) and set(f['actual_BN_calls'])==bn and set(f['actual_BN_calls'].values())=={1},'Actual dev data provenance and all BN')
            mw.check_calls(f['actual_affine_calls'],f['work'],mode,(480,640),contract);full_calls=f['actual_affine_calls']
            check_representations(f['work'],representations['initial' if stage.startswith('initial_') else 'final'],mode)
            require(set(f['heads'])=={'head0','head1','head2','head3'} and all(v['shape']==[2,480,640] and v['dtype']=='torch.float32' for v in f['heads'].values()),'Four full finite head records')
            pp=child(a.run,f['prediction_file']);require(sha(pp)==f['prediction_file_sha256'],'Actual prediction archive bytes')
            with np.load(pp,allow_pickle=False) as z:
                require(set(z.files)=={'head3'},'Only declared quality head');arr=z['head3']
            require(arr.shape==(2,480,640) and arr.dtype==np.float32 and np.isfinite(arr).all() and mw.digest(arr)==f['prediction_sha256']==f['heads']['head3']['sha256'],'Actual finite prediction payload')
            predictions.append(dict(state=label,target_id=f['target_id'],sequence=f['sequence'],file=str(pp.resolve()),file_sha256=sha(pp),head3_sha256=mw.digest(arr),input_evidence=f['input_evidence']))
    require(set(r['final_parity'])==({'M'} if a.smoke else {'M','Q4','Q8'}),'All declared final modes independently compared')
    if full_calls is None:
        # Original graph shapes; not model outputs. Production wrappers keep all positions.
        source=read(HERE.parents[1]/'flow_story_20261020/round30/INITIAL_REFERENCES_LOCAL_V1.json')['states']['initial_Q4']
        require(sha(source['result_path'])==source['result_sha256'],'Frozen original shape provenance for engineering smoke')
        full_calls=read(source['result_path'])['evaluations']['initial']['frames'][0]['actual_affine_calls']
    for mode,p in r['final_parity'].items():
        first=m['train_order'][0]['target_id'];require(p['mode']==mode and p['input_target_id']==first and p['input_evidence']==evidence(first),'Fixed parity input only')
        if mode=='M':layout(p['mask_layout'],'full9')
        else:require(p['mask_layout'] is None,'Uniform parity no mask')
        for key in ('training_deployment','dense_integer'):old.parity(p[key],True,psn)
        ps=[p['training_deployment'][k] for k in ('reference_record','actual_record')]+[p['dense_integer']['dense_record']]
        for rec,(ste,dense) in zip(ps,((True,False),(False,False),(False,True))):
            require(set(rec['actual_BN_calls'])==bn and set(rec['actual_BN_calls'].values())=={1} and set(rec['actual_target_calls'])==mw.TARGETS and set(rec['actual_target_calls'].values())=={1},'All actual parity target calls')
            require((rec['PSN_count'],rec['integer_count'],rec['head_count'])==(105,28,4) and rec['ste_entry'] is ste and rec['dense_integer_reference'] is dense,'Three complete parity routes')
            mw.check_calls(full_calls,rec['work'],mode,(480,640),contract,dense=dense)
            check_representations(rec['work'],representations['final'],mode)
        pp=child(a.run,p['heads_file']);require(sha(pp)==p['heads_sha256'],'Actual parity archive')
        with np.load(pp,allow_pickle=False) as z:
            require(set(z.files)=={'head0','head1','head2','head3'},'All four saved parity heads')
            for k in z.files:require(z[k].shape==(1,2,480,640) and z[k].dtype==np.float32 and np.isfinite(z[k]).all() and mw.digest(z[k])==p['training_deployment']['checks'][k]['actual_sha256'],'Actual parity head payload')
    if not a.smoke:
        refs={'initial_Q8':read(HERE.parents[1]/'flow_story_20261020/round30/INITIAL_REFERENCES_LOCAL_V1.json')['states']['initial_Q8']};require(set(r['original_initial_references'])==set(refs),'Declared original reference states')
        for state,d in refs.items():
            actual=r['original_initial_references'][state]
            require(sha(d['result_path'])==actual['result_sha256']==d['result_sha256'] and sha(d['audit_path'])==actual['audit_sha256']==d['audit_sha256'] and actual['state_sha256']==ORIGINAL and actual['new_forwards']==0,'Old original reference bytes explicitly reused')
    execution_counts=dict(optimizer_updates=n,quality_predictions=len(predictions),**({'engineering_model_forwards':nf} if a.smoke else {'formal_model_forwards':nf}))
    out=dict(schema='flow_e0_independent_actual_execution_audit_v1',status='PASS',engineering_smoke=a.smoke,quality_analysis_allowed=not a.smoke,auditor_sha256=sha(__file__),production_source_freeze_sha256=sha(BASE/'SOURCE_FREEZE_V1.json'),work_auditor_sha256=sha(HERE/'audit_e0_work_v1.py'),source_result_sha256=sha(rp),source_mask_contract_sha256=sha(a.mask_contract),checkpoint_sha256=sha(wp),final_state_sha256=final,graph_id=r['graph_id'],counts=execution_counts,changed_parameter_tensors=len(changed),predictions=predictions,training_work=totals,student_mode_counts=dict(counts),GT_values_read=False,hardware_cycle_claim=False,scope='Actual independent-RNE4 ordinary joint-mode recovery and independently reloaded artifacts; final/initial quantizer hashes independently reconstructed from weights;  runtime PSN/integer parity receipts, not separately persisted intermediate arrays; no task-specific scoring victory.')
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({k:out[k] for k in ('status','counts','student_mode_counts','changed_parameter_tensors')}))
if __name__=='__main__':main()
