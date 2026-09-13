"""Unmodified official CPU relation AND outer path on two full source layers."""
import sys
sys.dont_write_bytecode=True
import json,os
from dataclasses import asdict
import torch
from execute import HERE,HW,load_official_api,run_official_fc
torch.set_num_threads(1);torch.set_num_interop_threads(1)
_,FC,_,create_network=load_official_api()
repo=HW/'third_party/Prosperity'
previous=os.getcwd()
try:
    os.chdir(repo)
    network=create_network('lenet5','data/lenet5_mnist.pkl')
finally:os.chdir(previous)
out={'official_repo':str(repo),'runner':str(HW/'scripts/run_prosperity_official_probe.py'),
     'scope':'Two complete official LeNet5 MNIST FC operators, original CPU relation and full outer model, no accelerated callback.',
     'source_bit_counters_are_bits_not_bytes':True,'runs':[]}
for op in network:
    if isinstance(op,FC):
        for product in (False,True):
            result=asdict(run_official_fc(op,product))
            out['runs'].append(result);print(op.name,product,result['total_cycles'],flush=True)
            (HERE/'official_reference.json').write_text(json.dumps(out,indent=2)+'\n')
