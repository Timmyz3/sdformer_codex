"""Current matched-dense executed arithmetic inventory, not latency/PPA.

Observe actual ATen matmul/conv calls, including overridden functional paths.
No old ep34 share is used as the current denominator.
"""
from pathlib import Path
import argparse
from collections import defaultdict
import json
import sys
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    algorithm=args.root/'open_fusion_execution/breadth_20260912/algorithm'
    sys.path.insert(0,str(algorithm))
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from parent_network import ParentNetwork,arrays
    net=ParentNetwork(args)
    from fixed_structure import LiteralForward
    from evaluate_branch_control import evaluate_axis
    net.install('ordinary');net.helper.restore()
    q=arrays(algorithm/'matched_training/dense/stage320/deployed_constants.npz')
    net.helper=LiteralForward(net.controller,net.pair.temporal.theta,q,'dense')
    stack=[];records=defaultdict(lambda:dict(calls=0,MACs=0,input_elements=0,input_nonzero=0))
    class Inventory(TorchDispatchMode):
        enabled=True
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            out=func(*args,**(kwargs or {}))
            if not self.enabled:return out
            kind=str(func);mac=0;source=None;weight=None
            if kind=='aten.convolution.default':
                a,w=args[:2];transposed=bool(args[6]);groups=int(args[8])
                mac=(a.numel()*w.shape[1]*int(np.prod(w.shape[2:])) if transposed
                     else out.numel()*w.shape[1]*int(np.prod(w.shape[2:])))
                source,weight=a,w
            elif kind=='aten.mm.default':
                a,w=args[:2];mac=out.numel()*a.shape[-1];source,weight=a,w
            elif kind=='aten.addmm.default':
                a,w=args[1:3];mac=out.numel()*a.shape[-1];source,weight=a,w
            elif kind=='aten.bmm.default':
                a,w=args[:2];mac=out.numel()*a.shape[-1];source,weight=a,w
            if mac:
                name=stack[-1] if stack else '<functional>'
                key=(name,kind,tuple(source.shape),tuple(weight.shape),tuple(out.shape))
                r=records[key];r['calls']+=1;r['MACs']+=int(mac)
                # Diagnostic density of the left ATen operand only; for PSN
                # addmm this may be A, so never call MAC*density saved work.
                r['input_elements']+=source.numel()
                r['input_nonzero']+=int(torch.count_nonzero(source))
            return out
    inv=Inventory();hooks=[]
    def leave_module(module,inputs,output):
        stack.pop()
    for name,module in net.model.named_modules():
        hooks.append(module.register_forward_pre_hook(lambda m,a,n=name:stack.append(n)))
        hooks.append(module.register_forward_hook(leave_module))
    r0=net.BLOCK.rsplit('.',1)[0]+'.0'
    wanted={r0+'.conv1.0',r0+'.conv2.0'}
    ff=[name for name,m in net.modules.items() if isinstance(m,torch.nn.Linear)
        and m.in_features==384 and m.out_features==1536]
    wanted.update(ff[:2]);captures={}
    def capture(name,module,inputs,output):
        if name in captures:return
        inv.enabled=False
        try:
            a=inputs[0].detach();y=output.detach()
            entry=dict(weight=module.weight.detach().cpu().float().numpy())
            if module.bias is not None:entry['bias']=module.bias.detach().cpu().float().numpy()
            if isinstance(module,torch.nn.Conv2d):
                if a.ndim==5:a=a.flatten(0,1);y=y.flatten(0,1)
                assert a.ndim==4 and module.groups==1
                b,c,ih,iw=a.shape;oh,ow=y.shape[-2:]
                indices=torch.linspace(0,oh*ow-1,64,device=a.device).long()
                yy,xx=indices//ow,indices%ow
                parts=[]
                for ky in range(module.kernel_size[0]):
                    row=[]
                    for kx in range(module.kernel_size[1]):
                        iy=yy*module.stride[0]-module.padding[0]+ky*module.dilation[0]
                        ix=xx*module.stride[1]-module.padding[1]+kx*module.dilation[1]
                        valid=(iy>=0)&(iy<ih)&(ix>=0)&(ix<iw)
                        z=a[:,:,iy.clamp(0,ih-1),ix.clamp(0,iw-1)]*valid[None,None]
                        row.append(z)
                    parts.append(torch.stack(row,2))
                patches=torch.stack(parts,2).permute(0,4,1,2,3).reshape(b,64,-1)
                sample_y=y[:,:,yy,xx].permute(0,2,1)
                entry.update(input=patches.cpu().float().numpy(),output=sample_y.cpu().float().numpy(),
                             positions=indices.cpu().numpy())
            else:
                flat=a.reshape(-1,a.shape[-1]);flat_y=y.reshape(-1,y.shape[-1])
                indices=torch.linspace(0,len(flat)-1,640,device=a.device).long()
                entry.update(input=flat[indices].cpu().float().numpy(),output=flat_y[indices].cpu().float().numpy())
            file=name.replace('.','_')+'.npz';np.savez_compressed(args.output/file,**entry)
            captures[name]=dict(file=file,input_shape=list(a.shape),output_shape=list(y.shape),
                                weight_shape=list(module.weight.shape),kind=type(module).__name__,
                                sample_rows=int(np.prod(entry['input'].shape[:-1])))
        finally:inv.enabled=True
    for name in sorted(wanted):
        m=net.modules[name]
        hooks.append(m.register_forward_hook(lambda m,a,o,n=name:capture(n,m,a,o)))
    names=json.loads((args.root/'algorithm/samples.json').read_text())['valid'][:1]
    try:
        with inv:
            summary=evaluate_axis(args,net.model,net.current,names,'profile',progress_tag='MAJOR_PROFILE')
        rows=[]
        for (module,op,a,w,o),r in records.items():
            rows.append(dict(module=module,op=op,left_shape=a,right_shape=w,output_shape=o,**r))
        rows.sort(key=lambda r:r['MACs'],reverse=True)
        total=sum(r['MACs'] for r in rows)
        for r in rows:r['dense_MAC_share']=r['MACs']/total
        report=dict(parent='matched dense stage320',frames=names,actual_forward=True,
                    counts='Actual ATen dense arithmetic extents. Conv counts include nominal padded/transpose-zero taps; no sparsity or ASIC latency claim.',
                    omitted='Non-matmul elementwise, comparisons, Motion-XOR boolean/popcount, memory/layout and NumPy onepass BN work not part of MAC denominator.',
                    total_dense_MACs=total,rows=rows,captures=captures,AEE=summary)
        (args.output/'profile.json').write_text(json.dumps(report,indent=2)+'\n')
        print('MAJOR_PROFILE_COMPLETE',total,len(rows),flush=True)
        for r in rows[:12]:print(r['module'],r['op'],r['MACs'],flush=True)
    finally:
        for h in hooks:h.remove()
        net.close()


if __name__=='__main__':main()
