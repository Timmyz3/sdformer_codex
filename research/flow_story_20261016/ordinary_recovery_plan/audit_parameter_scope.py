#!/usr/bin/env python3
"""CPU native-MLP construction only, plus documentary evidence inventory; no full model/checkpoint/GPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
factory=ROOT/'research/flow_story_20261011/preflight/cpu_numerics/native_mlp_probe.py'
spec=importlib.util.spec_from_file_location('_ordinary_recovery_native_factory',factory)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
import torch
sources=[factory,
 ROOT/'research/flow_story_20261011/preflight/reproduction_protocol/official_git/models/STSwinNet_SNN/Spiking_swin_transformer3D.py',
 ROOT/'research/flow_story_20261011/preflight/reproduction_protocol/official_git/loss/flow_supervised.py',
 ROOT/'research/flow_story_20261014/structured_merge/structured_merge_mlp.py',
 ROOT/'research/flow_story_20261014/fast_task_recovery/run_fast_recovery.py',
 ROOT/'research/flow_story_20261014/fast_task_recovery/trainable_fast_tile.py',
 ROOT/'research/flow_story_20261014/fast_task_recovery/R0_v3/results.json',
 ROOT/'research/flow_story_20261014/fast_task_recovery/R1_v3/results.json',
 ROOT/'research/flow_story_20261013/breadth/remote_snapshot/checkpoint_step_000135930.json',
 ROOT/'research/flow_story_20261013/recovery/results_20261013/baseline_yield.json',
 ROOT/'research/flow_story_20261013/recovery/results_20261013/baseline_remainder_launch.json',
 ROOT/'research/flow_story_20261014/fast_warmstart/receipts/resource_snapshot_20261014_0241.json']
rows=[]
for channels,count in [(96,2),(384,6)]:
 native=m.make_mlp(channels,torch.float32)
 named={name:int(p.numel()) for name,p in native.named_parameters()}
 rows.append({'channels':channels,'native_blocks':count,'parameters_per_block':sum(named.values()),'named_parameters':named})
result={'schema':'flow_ordinary_recovery_scope_inventory_v1','CPU_only':True,'GPU_initialized':torch.cuda.is_initialized(),
 'full_model_loaded':False,'trained_checkpoint_loaded':False,'named_MLP_parameter_counts':rows,
 'stage02_eight_MLP_total_parameters':sum(row['native_blocks']*row['parameters_per_block'] for row in rows),
 'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
 'scope':'Constructs only two native MLPs with synthetic weights. Documentary checkpoint sidecars/source records are not current remote checkpoint existence/hash verification.'}
assert not result['GPU_initialized']
assert result['stage02_eight_MLP_total_parameters']==7252064
(HERE/'parameter_scope_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'CPU_only':True,'parameters':result['stage02_eight_MLP_total_parameters'],'blocks':8}))
