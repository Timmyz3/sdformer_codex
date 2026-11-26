#!/usr/bin/env python3
"""CPU-only independent inspection of four completed R12 saved states.

No model construction, data loading or GPU. Loads only SHA-bound weights_only PT.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import torch
import audit_recovery_services as service


def require(ok,message):
    if not ok:raise ValueError(message)


def digest_state(state,exclude=()):
    excluded=set(exclude);digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        if name in excluded:continue
        value=value.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(value.dtype).encode());digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def audit(results,digests,output,partial=False):
    require(len(results)==len(digests) and (1<=len(results)<4 if partial else len(results)==4) and not output.exists(),
            'Explicit raws and fresh output; partial accepts one to three arms only')
    service.preflight();torch.set_num_threads(2)
    records={};baseline=None
    for path,expected in zip(results,digests):
        require(service.sha(path)==expected,'Explicit completed raw SHA')
        raw=json.loads(path.read_text());arm=raw['arm']
        require(arm in service.ARMS and arm not in records and raw['status']=='ROUND12_FAIR_RECOVERY_COMPLETE' and
                raw['completed_optimizer_steps']==200,'Complete fixed recovery only')
        weights=path.parent/raw['weights_file']
        require(weights.resolve().parent==path.parent.resolve() and service.sha(weights)==raw['weights_sha256'],'Complete PT bytes')
        saved=torch.load(weights,map_location='cpu',weights_only=True)
        require(not torch.cuda.is_initialized(),'No CUDA audit')
        require(saved['schema']=='flow_round12_recovered_shared_native_state_v1' and saved['completed_optimizer_steps']==200 and
                saved['binding']==raw['binding'] and saved['trainable_names']==raw['allowed_parameter_names'] and
                saved['initial_state_sha256']==raw['initial_model_state_sha256'] and saved['final_state_sha256']==raw['final_state_sha256'] and
                saved['BN_scope']=='MLP_CMA176' and saved['BN_buffers_unchanged'],'Serialized state contract')
        state=saved['model_state_dict'];names=saved['trainable_names'];selected=set(names)
        require(len(state)==549 and all(isinstance(v,torch.Tensor) and v.device.type=='cpu' and
                (not v.is_floating_point() or bool(torch.isfinite(v).all())) for v in state.values()),'Finite CPU state tensors')
        require(len(names)==len(selected)==80 and selected<=set(state) and sum(state[n].numel() for n in names)==7252064,'Saved exact80 trainable tensors')
        require(digest_state(state)==saved['final_state_sha256']==raw['final_state_sha256'],'Recomputed full saved state digest')
        require(digest_state(state,selected)==raw['frozen_state_before_sha256'],'Saved frozen-state complement equals initial')
        require(saved['final_state_sha256']!=saved['initial_state_sha256'],'Recovery did not change state')
        bn=raw['BN_scope'];sig=bn['BN_signature_before_inference'];frozen=set(bn['frozen_BN_names'])
        for name,initial in sig.items():
            for field in ('running_mean','running_var','num_batches_tracked'):
                key=name+'.'+field
                if name in frozen:require(key in state and tensor_hash(state[key])==initial[field],'Saved private BN buffer changed')
                else:require(key not in state and initial[field] is None,'External current BN gained buffer')
        opt=saved['optimizer_state_dict'];groups=opt['param_groups'];require(len(groups)==1,'One fresh optimizer group')
        group=groups[0];recipe=raw['binding']['recipe'];ids=group['params']
        require(len(ids)==len(set(ids))==80 and set(opt['state'])==set(ids),'All80 optimizer states')
        require(group['lr']==recipe['lr'] and group['weight_decay']==recipe['weight_decay'] and
                tuple(group['betas'])==tuple(recipe['betas']) and group['eps']==recipe['eps'] and not group['amsgrad'],
                'Saved optimizer common hyperparameters')
        optimizer_tensors=0
        for index,name in zip(ids,names):
            entry=opt['state'][index]
            require(set(entry)=={'step','exp_avg','exp_avg_sq'},'AdamW state keys')
            require(isinstance(entry['step'],torch.Tensor) and entry['step'].numel()==1 and float(entry['step'])==200,'All parameter steps200')
            for key in ('exp_avg','exp_avg_sq'):
                value=entry[key]
                require(isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.shape==state[name].shape and
                        value.dtype==state[name].dtype and bool(torch.isfinite(value).all()),'Finite shape-matched optimizer moment')
                if key=='exp_avg_sq':require(bool((value>=0).all()),'Second moment negative')
            optimizer_tensors+=3
        rng=saved['rng_state']
        require(set(rng)=={'torch_cpu','torch_cuda'} and isinstance(rng['torch_cpu'],torch.Tensor) and
                rng['torch_cpu'].device.type=='cpu' and rng['torch_cpu'].dtype==torch.uint8 and
                isinstance(rng['torch_cuda'],list) and len(rng['torch_cuda'])==1 and
                rng['torch_cuda'][0].device.type=='cpu' and rng['torch_cuda'][0].dtype==torch.uint8,'Stored CPU/CUDA RNG bytes')
        common=dict(initial=saved['initial_state_sha256'],frozen=raw['frozen_state_before_sha256'],names=names,
                    state_geometry={n:[list(v.shape),str(v.dtype)] for n,v in state.items()},recipe=recipe)
        if baseline is None:baseline=common
        else:require(common==baseline,'Four saved states share architecture/start/parameter contract')
        records[arm]=dict(raw_sha256=expected,weights_path=str(weights),weights_sha256=raw['weights_sha256'],
            weights_bytes=weights.stat().st_size,state_tensor_count=len(state),state_elements=sum(v.numel() for v in state.values()),
            state_all_CPU_finite=True,full_state_digest=saved['final_state_sha256'],frozen_complement_digest=raw['frozen_state_before_sha256'],
            private_BN_buffer_tensors=72,external_BN_buffers_absent=162,trainable_tensors=80,trainable_elements=7252064,
            optimizer_state_tensors=optimizer_tensors,every_optimizer_step=200,torch_rng_bytes_present=True)
        del saved,state,opt
    require(partial or set(records)==set(service.ARMS),'Four completed checkpoints')
    receipt=dict(schema='flow_round12_independent_checkpoint_CPU_audit_v1',
        status='PASS_PARTIAL_SAVED_STATE_AND_OPTIMIZER_INSPECTION' if partial else 'PASS_SAVED_STATE_AND_OPTIMIZER_INSPECTION',partial=partial,
        source_sha256={str(Path(__file__)):service.sha(__file__),str(Path(service.__file__)):service.sha(service.__file__),
                       str(service.HERE/'freeze_smoke_v1.json'):service.sha(service.HERE/'freeze_smoke_v1.json')},
        torch=str(torch.__version__),device='cpu',GPU_initialized=torch.cuda.is_initialized(),records=records,
        full_model_constructed=False,model_forward_or_AEE_executed=False,optimizer_resume_step_tested=False,
        native_FP32_GPU_equivalence_proved=False,
        scope='Independent CPU weights_only loads, state bytes/modes and80 optimizer moments/steps. Not a resume-transition or quality validation.')
    output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    return dict(status=receipt['status'],checkpoints=len(records),GPU_initialized=False,output=str(output))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,action='append',required=True)
    parser.add_argument('--results-sha256',action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--partial',action='store_true',help='Audit one to three checkpoints without asserting four-arm closure')
    args=parser.parse_args();print(json.dumps(audit(args.results,args.results_sha256,args.output,args.partial)))


if __name__=='__main__':main()
