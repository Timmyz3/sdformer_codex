"""One predeclared independent training frame, all 336 fixed grid windows."""
import json
import numpy as np
from model_access import HERE,BASE,load_parent
class Captured(Exception):pass

def main():
 import torch
 import torch.nn.functional as F
 args,net=load_parent(HERE/'capture_runtime')
 from run_bn_probe import input_frame,read_names
 name='thun_00_a_0002.npy';train=read_names(args.data,'train');valid=read_names(args.data,'valid')
 assert name in train and name not in valid
 origins=np.array([(y,x) for y in list(range(0,240,16))+[238] for x in list(range(0,320,16))+[318]],np.int32)
 assert len(origins)==336
 r0name=net.BLOCK.rsplit('.',1)[0]+'.0';r0=net.modules[r0name];conv=net.modules[r0name+'.conv2.0'];state={}
 flat=lambda x:x[:,0] if x.ndim==5 else x
 def pre(m,ins):state['identity']=flat(ins[0]).detach()
 def forward(x):
  g=flat(x);assert tuple(g.shape)==(10,96,240,320) and torch.equal(g,g.ne(0).to(g))
  gp=F.pad(g,(1,1,1,1));id=state.pop('identity')
  source=np.stack([gp[:,:,y:y+4,z:z+4].cpu().numpy().astype(bool) for y,z in origins])
  identity=np.stack([id[:,:,y:y+2,z:z+2].cpu().numpy() for y,z in origins])
  J=np.rint(identity.astype(np.float64)*2**20).clip(-2**31,2**31-1).astype(np.int32)
  np.savez_compressed(HERE/'calibration.npz',source_bits=source,identity_fp32=identity,identity_q20=J,output_origin_yx=origins,input_origin_yx=origins-1,frame=np.array(name),theta=np.array(1.0))
  meta=dict(complete=True,frame=name,belongs_train=True,overlap_valid825=False,grid_windows=336,source_shape=list(source.shape),identity_shape=list(identity.shape),source_spikes=int(source.sum()),all_layer_spikes=int(g.ne(0).sum()),method='fixed regular output grid, no activity selection; actual r0 conv2 source and identity from same prefix',torch=torch.__version__,gpu=torch.cuda.get_device_name(0))
  (HERE/'calibration.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True);raise Captured()
 hook=r0.register_forward_pre_hook(pre);old=conv.forward;conv.forward=forward
 try:
  with torch.no_grad():
   x,_,_=input_frame(args.data,name,targets=False)
   try:net.model(x)
   except Captured:pass
   else:raise AssertionError('capture failed')
 finally:conv.forward=old;hook.remove();net.close()
if __name__=='__main__':main()
