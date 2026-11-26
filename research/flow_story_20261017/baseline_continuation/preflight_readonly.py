#!/usr/bin/env python3
"""Read-only strict-resume preparation. No training entry main/GPU invocation.

Default: existing dedicated A100 paths, CPU checkpoint load + native metadata
preflight + read-only GPU/lock observation. Local mode audits only the supplied
existing checkpoint snapshot and contract, never claims remote readiness.
All output is stdout; source/run/data/checkpoint files are never modified.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import argparse
import datetime
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
sys.dont_write_bytecode=True

ROOT=Path('/data/flow_a')
RUN=ROOT/'runs/native_crop9_fresh_20261012'
CODE=ROOT/'code/native_transfer_run_20261012/baseline'
GPU_LOCK=ROOT/'runs/.flow_a_gpu0.lock'
CHECKPOINT_SHA='1e72cca3946f40e24a1575f3348f9a596d744460808ad910a616ba713b5e4977'
CONTRACT_SHA='961f1700e76f3427c74d4e9af9799a684433ff00449b8386be6c1f9cdd75e934'
SOURCES={'train_native_baseline.py':'88b2febd21be4f941118b7339d7ec8275d9840530a2a359a25f206a57a62b154',
 'native_data_acceptance.py':'8c3c6f5dc3264116429b08f4de32b2eab0ba062d91e5f4d3d50903597fe81b1f',
 'train_baseline.py':'9628cad1c7dc7e20416ec9465303692f1b3860aaf6c6f3413345cc243b9e11e0'}
BN_SUFFIXES=('.running_mean','.running_var','.num_batches_tracked')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()


def resume_arguments():
    return ['--source-root',str(ROOT/'code/SDformerFlow'),'--data-root',str(ROOT/'data/DSEC_native_v1'),
      '--run-dir',str(RUN),'--manifest',str(ROOT/'setup/common_targets.json'),
      '--source-manifest',str(ROOT/'setup/native_source_sha256.json'),
      '--archive-audit',str(ROOT/'setup/archive_data_audit.json'),
      '--data-acceptance',str(ROOT/'setup/native_cpu_pipeline_20261012/acceptance/acceptance.json'),
      '--acceptance-sha256','b1d87de8fd48b710fe5a49dd37d6d64427d2fe111df249efa73ba2299f454540',
      '--phase','continue','--max-steps','480000','--wall-hours','11.5','--checkpoint-steps','250',
      '--keep-last','2','--log-every','10','--workers','2','--cpu-threads','2','--seed','0','--device','0',
      '--resume',str(RUN/'checkpoint_step_000271482.pt'),
      '--pilot-marker',str(RUN/'pilot_step_000000500.json'),
      '--pilot-marker-sha256','15af8442f99eb91a9b4327436fa6df33cc284c8cea78dbb7b2c5d787c37b6b96']


def checkpoint_audit(checkpoint,contract_file):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    require(not torch.cuda.is_initialized(),'CPU-only checkpoint audit')
    require(sha(checkpoint)==CHECKPOINT_SHA,'Exact original271482 checkpoint bytes')
    run=json.loads(Path(contract_file).read_text());contract=run['contract']
    require(run['contract_sha256']==digest(contract)==CONTRACT_SHA,'Fixed original contract')
    value=torch.load(checkpoint,map_location='cpu',weights_only=True)
    require(value['schema_version']=='flow_a_native_encoding_v2_checkpoint_v1' and value['contract']==contract
            and value['contract_sha256']==CONTRACT_SHA,'Safe checkpoint schema/contract identity')
    p=value['provenance'];require(p['created_by']=='flow_a_20261012_native_encoding_baseline'
        and p['random_init'] is True and p['old_checkpoint_loaded'] is False
        and p['BN_policy']=='native_train_tracked' and p['precision']=='fp32','Original training provenance')
    t=value['training'];require(t['completed_optimizer_steps']==271482 and t['epoch']==45 and t['next_index']==1482
        and t['completed_optimizer_steps']==6000*t['epoch']+t['next_index'],'Fixed committed step/cursor')
    require(value['stop_reason']=='wall_budget','Original completed segment stop')
    require(value['epoch_order']==torch.randperm(6000,generator=torch.Generator().manual_seed(1000003*45)).tolist(),
            'Original6000 epoch permutation')
    state=value['model_state_dict'];params={k:v for k,v in state.items() if not k.endswith(BN_SUFFIXES)}
    require(len(state)==711 and sum(v.numel() for v in state.values())==55001588
        and len(params)==477 and sum(v.numel() for v in params.values())==54919238,'Original full model inventory')
    require(all(v.device.type=='cpu' and bool(torch.isfinite(v).all()) for v in state.values()),'Model tensor CPU/finite')
    opt=value['optimizer_state_dict'];require(len(opt['param_groups'])==1,'Original AdamW group')
    group=opt['param_groups'][0];ids=group['params']
    require(ids==list(range(477)) and len(opt['state'])==453 and set(opt['state']).issubset(ids),'477 original parameter IDs /453 initialized states')
    require(group['lr']==.001*.5**4 and group['weight_decay']==.01 and tuple(group['betas'])==(.9,.999)
        and group['eps']==1e-8 and group['amsgrad'] is False and group['maximize'] is False,'Original AdamW recipe/LR')
    shapes=[tuple(v.shape) for v in params.values()]
    for index,s in opt['state'].items():
        require(set(s)=={'step','exp_avg','exp_avg_sq'} and float(s['step'])==271482,'Original optimizer step/moments')
        require(tuple(s['exp_avg'].shape)==tuple(s['exp_avg_sq'].shape)==shapes[index],'Optimizer moment parameter shape')
        require(all(v.device.type=='cpu' and bool(torch.isfinite(v).all()) for v in s.values()),'Optimizer tensors CPU/finite')
    sched=value['scheduler_state_dict']
    require(sched['last_epoch']==45 and sched['step_size']==10 and sched['gamma']==.5
        and sched['base_lrs']==[.001] and sched['_last_lr']==[.001*.5**4],'Original StepLR state')
    rng=value['rng_state'];require(set(rng)=={'python','numpy','torch_cpu','torch_cuda'},'Complete RNG fields')
    random.Random().setstate(rng['python']);n=rng['numpy'];np.random.RandomState().set_state(
        (n['name'],np.asarray(n['keys'],dtype=np.uint32),n['pos'],n['has_gauss'],n['cached_gaussian']))
    torch.Generator(device='cpu').set_state(rng['torch_cpu'])
    require(len(rng['torch_cuda'])==1 and rng['torch_cuda'][0].dtype==torch.uint8
            and rng['torch_cuda'][0].device.type=='cpu' and rng['torch_cuda'][0].ndim==1
            and rng['torch_cuda'][0].numel()>0,'Saved single-device CUDA RNG present; no CUDA restore here')
    require(not torch.cuda.is_initialized(),'Audit did not initialize CUDA')
    return {'checkpoint_sha256':CHECKPOINT_SHA,'contract_sha256':CONTRACT_SHA,'training':t,
       'remaining_to_fixed_480000':208518,'model_tensors':711,'model_elements':55001588,
       'original_BN_buffers_retained':234,'parameter_tensors':477,'initialized_optimizer_states':453,
       'optimizer_steps_all':271482,'lr':group['lr'],'scheduler':sched,
       'epoch_order_exact':True,'python_numpy_torchCPU_RNG_local_restore_pass':True,
       'saved_CUDA_RNG_tensors':1,'CUDA_RNG_restored':False,'model_instantiated':False,
       'optimizer_instantiated':False,'training_updates':0,'GPU_model_calls':0},contract['runtime']


def readonly_lock(path):
    if not path.is_file():return 'MISSING_NOT_CREATED'
    with path.open('rb') as f:
        try:fcntl.flock(f,fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError:return 'EXCLUSIVE_OWNER_PRESENT'
        fcntl.flock(f,fcntl.LOCK_UN);return 'NO_EXCLUSIVE_OWNER_AT_OBSERVATION'


def remote_preflight():
    for name,expected in SOURCES.items():require(sha(CODE/name)==expected,'Original source '+name)
    latest=json.loads((RUN/'latest.json').read_text());side=json.loads((RUN/'checkpoint_step_000271482.json').read_text())
    require(latest==side and latest['completed_optimizer_steps']==271482 and latest['checkpoint_sha256']==CHECKPOINT_SHA,
            'Explicit271482 remains latest; do not silently select another checkpoint')
    # Direct read-only function use: old main writes preflight.json and locks.
    sys.path.insert(0,str(CODE));spec=importlib.util.spec_from_file_location('_native_readonly_prepare',CODE/'train_native_baseline.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    a=module.parse_args(resume_arguments());require(a.run_gpu is False,'No GPU flag in read-only preflight')
    module.guard_run_directory(a.run_dir);rows,hashes,sources,plan=module.preflight(a)
    require(plan['GPU_executed'] is False and len(rows)==6000 and len(hashes)==18000 and len(sources)==63,
            'Native complete read-only contract preflight')
    audit,expected_runtime=checkpoint_audit(a.resume,a.run_dir/'run_contract.json')
    import torch
    actual_runtime={'python':sys.version.split()[0],'torch':str(torch.__version__),
       'torchvision':importlib.metadata.version('torchvision'),'cuda':torch.version.cuda,
       'cudnn':torch.backends.cudnn.version(),
       'packages':{n:importlib.metadata.version(n) for n in ('numpy','spikingjelly','timm','einops','PyYAML')}}
    for key,value in actual_runtime.items():require(expected_runtime[key]==value,'Same runtime before GPU: '+key)
    locks={str(p):readonly_lock(p) for p in (GPU_LOCK,RUN/'.run.lock')}
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip()
    require(not torch.cuda.is_initialized(),'No CUDA context from read-only runtime checks')
    return {'status':'READ_ONLY_CONTRACT_PREFLIGHT_PASS','checkpoint':audit,'native_preflight':plan,
       'original_trainer_source_sha256':SOURCES,'runtime_nondevice_exact':actual_runtime,
       'recorded_runtime_device':{k:expected_runtime[k] for k in ('device_count','device_name')},
       'current_nvidia_smi':gpu,'current_compute_apps':apps,'lock_observations':locks,
       'available_at_observation':not apps and all(v=='NO_EXCLUSIVE_OWNER_AT_OBSERVATION' for v in locks.values()),
       'CUDA_device_runtime_and_RNG_restore':'Must still be checked by unchanged native run_training under exclusive GPU0 lock, before first update.',
       'training_argv_template':[str(ROOT/'envs/v2/bin/python'),'-B',str(CODE/'train_native_baseline.py')]+resume_arguments()+['--run-gpu'],
       'this_script_can_launch_training':False,'root_decision_required_for_new_segment':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--local-checkpoint',type=Path);parser.add_argument('--local-contract',type=Path)
    a=parser.parse_args();require(bool(a.local_checkpoint)==bool(a.local_contract),'Local paths must be supplied together')
    if a.local_checkpoint:
        audit,runtime=checkpoint_audit(a.local_checkpoint,a.local_contract)
        result={'status':'LOCAL_CPU_CHECKPOINT_CONTENT_ONLY','checkpoint':audit,'recorded_runtime':runtime,
                'remote_data_source_runtime_lock_readiness_checked':False,'this_script_can_launch_training':False}
    else:result=remote_preflight()
    result.update(schema='flow_crop80_continuation_readonly_preflight_v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        preflight_source_sha256=sha(__file__),files_written_by_script=0,training_started=False)
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
