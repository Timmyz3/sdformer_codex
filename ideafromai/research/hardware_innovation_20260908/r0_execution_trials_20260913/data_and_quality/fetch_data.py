import json,subprocess,shlex
from pathlib import Path
from model_access import HERE,BASE
names=json.loads((BASE/'algorithm/samples.json').read_text())['valid'][:10]
remote='root@ssh.sd5ai.scnet.cn'
socket='/tmp/codex_followthrough_a800_20260913.sock'
root='/root/private_data/work/sdformer_codex/SDformer'
base=root+'/data/Datasets/DSEC/saved_flow_data'
scp=['scp','-o','ControlPath='+socket,'-o','BatchMode=yes','-P','10037']
requests=[]
for seq in sorted(set(n.rsplit('_',1)[0] for n in names)):
 out=HERE/'data_mirror/event_tensors/10bins/left'/seq;out.mkdir(parents=True,exist_ok=True)
 files=[remote+':'+base+'/event_tensors/10bins/left/'+seq+'/'+n for n in names if n.rsplit('_',1)[0]==seq]
 subprocess.run(scp+files+[str(out)],check=True);requests+=files
for kind in ['gt_tensors','mask_tensors']:
 out=HERE/'data_mirror'/kind;out.mkdir(parents=True,exist_ok=True)
 files=[remote+':'+base+'/'+kind+'/'+n for n in names]
 subprocess.run(scp+files+[str(out)],check=True);requests+=files
# Existing local files are not presumed identical to remote archived evaluation inputs.
import numpy as np
local=HERE.parents[4]/'SDformer/data/Datasets/DSEC/saved_flow_data'
comparisons=[]
for name in names:
 seq=name.rsplit('_',1)[0]
 for rel in [Path('event_tensors/10bins/left')/seq/name,Path('gt_tensors')/name,Path('mask_tensors')/name]:
  if (local/rel).exists():
   a=np.load(local/rel);b=np.load(HERE/'data_mirror'/rel)
   comparisons.append(dict(file=str(rel),array_equal=bool(np.array_equal(a,b)),local_shape=list(a.shape),remote_shape=list(b.shape),max_abs=float(np.max(np.abs(a.astype(float)-b.astype(float))))))
(HERE/'data_transfer.json').write_text(json.dumps(dict(complete=True,source=base,files=names,count_files=len(requests),local_comparisons=comparisons),indent=2)+'\n')
checkpoint=root+'/neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_NB0_equal_plus10_ep40_20260805/checkpoint_epoch29.pth'
subprocess.run(scp+[remote+':'+checkpoint,str(HERE/'nb0_checkpoint_epoch29.pth')],check=True)
print('DATA_AND_NB0_FETCHED',flush=True)
