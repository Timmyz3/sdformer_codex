"""Frozen patch integer functions in the existing factor_parent ep34 background.

No training/quantization. Root owns GPU scheduling. The parent axis is original
Conv1 with the SAME frozen A14/16384 temporal neuron, S2 integer/class/coarse
successors and four fixed patch BNs, exactly the old integer-factor evaluation
background. This is neither bare ep34 nor the later matched LiteralForward net.
"""
from pathlib import Path
from contextlib import contextmanager
import argparse,json,sys,time,types
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent


def arrays(path):
    with np.load(path) as d:return {k:d[k].copy() for k in d.files}


def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


@contextmanager
def exact_conv(torch):
    old=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    try:
        # Avoid cuDNN FFT/Winograd roundoff on integer-valued convolution.
        # FP32 gate*U8 and every partial sum fit exactly; V/A use FP64.
        with torch.backends.cudnn.flags(enabled=False):yield
    finally:torch.backends.cuda.matmul.allow_tf32=old


def int_reference(data,words,groups):
    u=data['u'].astype(np.int64);v=data['v'].astype(np.int64);a=data['a'].astype(np.int64)
    x=((words[...,None].astype(np.int64)>>np.arange(10))&1).transpose(0,3,2,1)
    mask=data['masks'][groups//2400].repeat(4,axis=1)
    z=(x@u)*mask[:,None,None,:];y=z@v;q=np.einsum('ts,gspr->gtpr',a,z)
    out=np.einsum('ts,gsph->gtph',a,y);assert np.array_equal(out,q@v)
    return dict(z=z,y=y,q=q,u=out,gate=out>=data['tau'][None,:,None,:])


class FixedPatchInteger:
    def __init__(self,data,device,chunk_rows=8):
        import torch
        self.torch=torch;self.data=data;self.rows=chunk_rows
        self.theta_source=float(data['theta_source']);self.theta=float(data['theta_output'])
        self.rank=data['u'].shape[1]
        self.u=torch.as_tensor(data['u'].copy(),device=device,dtype=torch.float32).T.reshape(self.rank,96,3,3)
        self.v=torch.as_tensor(data['v'].copy(),device=device,dtype=torch.float64)
        self.a=torch.as_tensor(data['a'].copy(),device=device,dtype=torch.float64)
        self.tau=torch.as_tensor(data['tau'].copy(),device=device,dtype=torch.float64)
        self.scale=torch.exp2(torch.as_tensor(data['y_exponent'].copy(),device=device,dtype=torch.float64))
        self.masks=torch.as_tensor(data['masks'].repeat(4,axis=1).copy(),device=device,dtype=torch.bool)
        self.pending=None;self.audit=None;self.frames=0;self.numeric_checks=dict(frames=0,Z_integer_residual=0.,Y_integer_residual=0.,U_integer_residual=0.)

    def calculate_chunk(self,gate,first,last):
        import torch.nn.functional as F
        torch=self.torch;H=gate.shape[-2]
        local=gate[:,0,:,max(first-1,0):min(last+1,H),:].float()
        local=F.pad(local,(1,1,int(first==0),int(last==H)))
        with exact_conv(torch):z=F.conv2d(local,self.u)
        row_region=(torch.arange(first,last,device=z.device)//30).long()
        z=z*self.masks[row_region].T[None,:,:,None]
        values=z.permute(0,2,3,1).double();y=values@self.v
        out=torch.einsum('ts,syxh->tyxh',self.a,y)
        return values,y,out,out>=self.tau[:,None,None,:]

    def conv_forward(self,x):
        torch=self.torch
        if tuple(x.shape)!=(10,1,96,240,320):raise ValueError('Fixed RTL domain is T10/B1/C96/H240/W320; regional bands are global y//30.')
        gate=x.ne(0)
        if bool(torch.where(gate,x!=self.theta_source,x!=0).any()):raise ValueError('Live source is not {0,theta_source}; theta must not be folded twice.')
        display=torch.empty_like(x);self.pending=torch.empty_like(x,dtype=torch.bool)
        for first in range(0,240,self.rows):
            last=min(first+self.rows,240);z,y,out,g=self.calculate_chunk(gate,first,last)
            if self.frames==0:
                for key,val,limit in [('Z',z,2**15),('Y',y,2**31),('U',out,2**47)]:
                    residual=float((val-val.round()).abs().max());self.numeric_checks[key+'_integer_residual']=max(self.numeric_checks[key+'_integer_residual'],residual)
                    if residual or bool((val.abs()>=limit).any()):raise ValueError(key+' integer/backend range admission failed')
            display[:,0,:,first:last,:]=(y*self.scale).permute(0,3,1,2).to(x.dtype)
            self.pending[:,0,:,first:last,:]=g.permute(0,3,1,2)
            if self.audit is not None:self.audit.capture(first,last,z,y,out,g,self.a)
        self.frames+=1;self.numeric_checks['frames']=self.frames
        return display

    def neuron_forward(self,displayed_bn_y):
        if self.pending is None:raise RuntimeError('Missing current Conv1 integer gate state')
        # norm1 runs on the displayed real Y, but its affine is already present
        # exactly once in frozen tau. Never feed displayed_bn_y into another A.
        output=self.pending.to(displayed_bn_y.dtype)*self.theta;self.pending=None
        if self.audit is not None:self.audit.finish_integer()
        return output


class LiveAudit:
    def __init__(self,torch,base,output):
        self.torch=torch;self.output=Path(output);self.old={};self.rows=[];self.context=None
        for f in sorted((base/'algorithm/patch_probe/partial_completion/integer_valid10').glob('capture_*.npz'))[:4]:
            d=arrays(f);self.old[str(d['file'])]=d
        if len(self.old)!=4:raise ValueError('All four old source captures are required')
        self.groups=next(iter(self.old.values()))['group_ids'].astype(np.int64)
        self.current_name='';self.current_axis='';self.enabled=False;self.data=None

    def begin(self,axis,name,index,data):
        self.current_name=name;self.current_axis=axis;self.enabled=index<4;self.data=data;self.context=None

    def source_hook(self,module,inputs):
        if not self.enabled:return
        torch=self.torch;x=inputs[0]
        if tuple(x.shape)!=(10,1,96,240,320):raise ValueError('Live source geometry')
        theta=float(self.data['theta_source'])
        if bool(torch.where(x.ne(0),x!=theta,x!=0).any()):raise ValueError('Full-image sn1 theta mismatch')
        dev=x.device;gid=torch.as_tensor(self.groups,device=dev);k=torch.arange(864,device=dev)
        yy=gid[:,None,None]//80+(k//3%3)[None,:,None]-1
        xx=(gid[:,None,None]%80)*4+torch.arange(4,device=dev)[None,None,:]+(k%3)[None,:,None]-1
        valid=(yy>=0)&(yy<240)&(xx>=0)&(xx<320)
        selected=x[:,0,k[None,:,None]//9,yy.clamp(0,239),xx.clamp(0,319)].ne(0)&valid[None]
        words=(selected.long()*(1<<torch.arange(10,device=dev))[:,None,None,None]).sum(0).cpu().numpy().astype(np.uint16)
        old=self.old.get(self.current_name);equal=old is not None and np.array_equal(old['group_ids'],self.groups) and np.array_equal(old['source_gate_words'],words)
        row=dict(axis=self.current_axis,file=self.current_name,groups=len(self.groups),source_words=words.size,source_theta_residual=0,old_source_equal=bool(equal))
        self.rows.append(row);save(self.output/'live_audit.json',self.rows)
        if not equal:
            self.output.mkdir(parents=True,exist_ok=True)
            np.savez(self.output/(self.current_axis+'_source_mismatch.npz'),file=self.current_name,group_ids=self.groups,source_gate_words=words)
            raise ValueError('Live source differs from RTL workload; saved new source, stop before reusing old cycles.')
        if self.current_axis!='factor_parent_same_A14':
            ref=int_reference(self.data,words,self.groups)
            self.context=dict(reference=ref,observed={k:np.empty_like(v) for k,v in ref.items()},seen=np.zeros(len(self.groups),bool),row=row)

    def capture(self,first,last,z,y,out,gate,a):
        if self.context is None:return
        pick=np.flatnonzero((self.groups//80>=first)&(self.groups//80<last))
        if not len(pick):return
        torch=self.torch;dev=z.device;gy=torch.as_tensor(self.groups[pick]//80-first,device=dev)
        gx=torch.as_tensor((self.groups[pick]%80)[:,None]*4+np.arange(4)[None,:],device=dev)
        data={'z':z[:,gy[:,None],gx,:].permute(1,0,2,3),'y':y[:,gy[:,None],gx,:].permute(1,0,2,3),
              'u':out[:,gy[:,None],gx,:].permute(1,0,2,3),'gate':gate[:,gy[:,None],gx,:].permute(1,0,2,3)}
        data['q']=torch.einsum('ts,gspr->gtpr',a,data['z']) # audit-only AV counterpart, not the executed network order
        for key,value in data.items():self.context['observed'][key][pick]=value.detach().cpu().numpy()
        self.context['seen'][pick]=True

    def finish_integer(self):
        if self.context is None:return
        c=self.context
        if not c['seen'].all():raise ValueError('Missing sampled live intermediate rows')
        for key,value in c['reference'].items():
            errors=int(np.count_nonzero(c['observed'][key]!=value));c['row'][key+'_errors']=errors
            if errors:save(self.output/'live_audit.json',self.rows);raise ValueError('Live integer '+key+' differs from independent source oracle')
        c['row']['Q_scope']='sample-only AV identity check; network executes VA'
        save(self.output/'live_audit.json',self.rows);self.context=None


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',type=Path,default=HERE.parents[1]);ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--data-root',type=Path);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--split',choices=['diverse','valid'],default='diverse');ap.add_argument('--count',type=int)
    ap.add_argument('--axes',nargs='+',choices=['factor_parent_same_A14','ordinary_r32','regional_r96_a48'],default=['factor_parent_same_A14','ordinary_r32','regional_r96_a48'])
    ap.add_argument('--chunk-rows',type=int,default=8)
    opt=ap.parse_args();opt.output.mkdir(parents=True,exist_ok=True)
    import torch
    import torch.nn.functional as F
    import runpy
    base=opt.base.resolve();factor=base/'algorithm/patch_probe/factor_completion_20260909'
    for path in [base/'algorithm',base/'algorithm/nrv_cost_probe',factor,factor.parent/'joint_completion_20260909']:sys.path.insert(0,str(path))
    import run_bn_probe
    import run_probe as probe
    from evaluate_factors_network import TemporalNeuron,BLOCK,TARGET
    from evaluate_stage2_deployment import CoarseReady,summarize
    from spikingjelly.activation_based import functional
    # Keep the old loader's code/config/checkpoint/S2 parent. Only fix data root
    # explicitly rather than depending on a workstation directory spelling.
    old_build=run_bn_probe.build_model
    def build(args):
        args.data=opt.data_root or opt.repo/'data/Datasets/DSEC/saved_flow_data'
        if 'epoch34' not in str(args.checkpoint):raise ValueError('Expected the same ep34 checkpoint')
        return old_build(args)
    run_bn_probe.build_model=build
    args=types.SimpleNamespace(root=base,output=opt.output,split=opt.split)
    try:system=probe.load_system(args)
    finally:run_bn_probe.build_model=old_build
    model,modules,_,current,sources,_,_,_=system;probe.install_sources(system,sources);current['count_codes']=False
    fixed=torch.load(base/'algorithm/patch_probe/patch_train_calibration.pt',map_location='cpu',weights_only=False)
    for name,val in fixed.items():
        bn=modules[name];bn.track_running_stats=True;bn.running_mean=val['mean'].to(bn.weight);bn.running_var=val['var'].to(bn.weight)
    model.eval();model.requires_grad_(False)
    data={name:arrays(HERE/(name+'.npz')) for name in ['ordinary_r32','regional_r96_a48']}
    for key in ['a','fp_a','temporal_bias','theta_source','theta_output','bn_scale','bn_bias']:
        if not np.array_equal(data['ordinary_r32'][key],data['regional_r96_a48'][key]):raise ValueError('Different frozen shared temporal/BN contract: '+key)
    conv,neuron=modules[BLOCK+'.conv1.0'],modules[TARGET]
    if tuple(conv.stride)!=(1,1) or tuple(conv.padding)!=(1,1) or tuple(conv.kernel_size)!=(3,3):raise ValueError('Conv1 native source window mismatch')
    if neuron.center_mode not in ['zero','none']:raise ValueError('No dynamic centering in compiled tau contract')
    for d in data.values():
        if float(modules[BLOCK+'.sn1.spiking_neuron'].thresh)!=float(d['theta_source']) or float(neuron.thresh)!=float(d['theta_output']):raise ValueError('Live theta differs from frozen integer factors')
    native_conv,native_neuron=conv.forward,neuron.forward
    parent=dict(data['ordinary_r32']);parent['a']=parent['a'].astype(np.float32)/16384
    temporal=TemporalNeuron(parent,conv.weight.device)
    names=(run_bn_probe.read_names(args.data,'valid') if opt.split=='valid' else json.loads((base/'algorithm/samples.json').read_text())['valid'])
    names=names[:(opt.count if opt.count is not None else (825 if opt.split=='valid' else 10))]
    audit=LiveAudit(torch,base,opt.output/'audit');hook=conv.register_forward_pre_hook(audit.source_hook)
    record=dict(complete=False,split=opt.split,files=names,checkpoint=str(args.checkpoint),config=str(args.config),code_root=str(args.code_root),data=str(args.data),
        parent='ep34 + existing S2 integer source/class/coarse consumers + four fixed patch BNs; original Conv1 plus common A14 temporal parent',
        modern_matched_LiteralForward=False,axes=opt.axes,executed_integer_order='VA',chunk_rows=opt.chunk_rows,
        source_scope='live full-image sn1 theta*g; first four frames fixed P4x64 compared to exact RTL source captures',
        preserved='original factor-parent Conv2, BN2, shortcut/identity, PED/residual branches, S2 and real preds.2 coarse exit',
        numeric='gate*U8 direct FP32 exact-range; no cuDNN/TF32 locally; integer V and A in exact-range FP64; frozen tau once; no new quantization',
        result_scope='fresh network AEE; wall time is not RTL performance',results={})
    save(opt.output/'run.json',record)
    try:
        for axis in opt.axes:
            conv.forward=native_conv;neuron.forward=types.MethodType(lambda self,x:temporal(x),neuron);adapter=None
            params=data['ordinary_r32'] if axis=='factor_parent_same_A14' else data[axis]
            if axis!='factor_parent_same_A14':
                adapter=FixedPatchInteger(params,conv.weight.device,opt.chunk_rows);adapter.audit=audit
                conv.forward,neuron.forward=adapter.conv_forward,adapter.neuron_forward
            rows=[];started=time.monotonic()
            for i,name in enumerate(names):
                functional.reset_net(model);current.pop('flow',None);audit.begin(axis,name,i,params)
                x,label,valid=run_bn_probe.input_frame(args.data,name)
                with torch.no_grad():
                    try:model(x)
                    except CoarseReady:pred=F.interpolate(current.pop('flow'),(480,640),mode='bilinear',align_corners=False)
                    else:raise RuntimeError('Real coarse consumer exit missing')
                    error=torch.linalg.vector_norm(pred.permute(0,2,3,1)[valid]-label.permute(0,2,3,1)[valid],dim=1)
                n=error.numel();total=float(error.double().sum());rows.append(dict(file=name,valid_pixels=n,aee_sum=total,AEE=total/n))
                summary=dict(**summarize(rows,i+1==len(names)),axis=axis,wall_seconds=time.monotonic()-started)
                if adapter is not None:summary['numeric_checks']=adapter.numeric_checks.copy()
                save(opt.output/(axis+'_frames.json'),rows);save(opt.output/(axis+'_summary.json'),summary)
                if i<4 or (i+1)%25==0 or i+1==len(names):print('PATCH_INTEGER_AEE',axis,i+1,summary['AEE_frame_mean'],flush=True)
                del x,label,valid,pred,error
            record['results'][axis]=summary;save(opt.output/'run.json',record)
        record['complete']=True;save(opt.output/'run.json',record)
    finally:hook.remove();conv.forward,neuron.forward=native_conv,native_neuron
    print('DONE',json.dumps(record['results']),flush=True)

if __name__=='__main__':main()
