#!/usr/bin/env python3
"""Actual original271482+own72 CPU reconstruction; no model/GPU/quality reads."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import torch
import selected_core as core
HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for v in iter(lambda:f.read(4*1024*1024),b''):h.update(v)
    return h.hexdigest()


def digest(state):
    # Frozen H0 h0_core.py:202 convention, copied only for CPU source checking.
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        value=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=HERE.parent/'baseline_checkpoint_verification/source_snapshot/checkpoint_step_000271482.pt')
    p.add_argument('--bn-buffers',type=Path,default=HERE.parent/'later_checkpoint_deployment/run_v1/bn_buffers.npz')
    p.add_argument('--output',type=Path,default=HERE/'expected_state.json');a=p.parse_args()
    torch.set_num_threads(2)
    core.require(not a.output.exists() and not torch.cuda.is_initialized(),'Fresh CPU-only receipt')
    core.require(sha(a.checkpoint)==core.CHECKPOINT_SHA and sha(a.bn_buffers)==core.OWN_BN_SHA,'Actual original source bytes')
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    core.require(payload['training']['completed_optimizer_steps']==271482 and payload['schema_version']=='flow_a_native_encoding_v2_checkpoint_v1','Original native source contract')
    with np.load(a.bn_buffers,allow_pickle=False) as z:values={n:z[n].copy() for n in z.files}
    state,chosen,record=core.construct_state(payload['model_state_dict'],values,digest,torch,np)
    core.require(not torch.cuda.is_initialized(),'No GPU initialized')
    out={'schema':'flow_selected16_expected_state_v1','status':'PASS','checkpoint_sha256':sha(a.checkpoint),'own_BN_buffers_sha256':sha(a.bn_buffers),
         'selected_state':record,'CPU_only':True,'model_forwards':0,'optimizer_steps':0,'calibration_forwards':0,
         'source_sha256':{'source_state_cpu.py':sha(__file__),'selected_core.py':sha(HERE/'selected_core.py')},
         'scope':'Actual source tensors and own72 array bytes; no model evaluation or quality inspection.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:record[k] for k in ('selected16_state_sha256','selected16_tensor_count','selected16_elements','retained_buffer_count')}))


if __name__=='__main__':main()
