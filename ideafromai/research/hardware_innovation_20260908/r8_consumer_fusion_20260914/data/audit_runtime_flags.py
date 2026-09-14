"""Configuration-only audit, no input forward or evaluation."""
import json
from model_access import HERE,load_parent
import torch
args,net=load_parent(HERE/'runtime_flags')
rows=[dict(path=n,type=type(m).__module__+'.'+type(m).__name__) for n,m in net.model.named_modules() if m.training]
r=dict(complete=True,no_forward=True,training_flags_true=rows,BN_training_true=[n for n,m in net.model.named_modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm) and m.training],dropout_training_true=[n for n,m in net.model.named_modules() if isinstance(m,torch.nn.modules.dropout._DropoutNd) and m.training],model_training=net.model.training,all_parameters_frozen=all(not p.requires_grad for p in net.model.parameters()))
(HERE/'runtime_flags.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);net.close()
