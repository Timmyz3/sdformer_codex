"""Bounded authorized tail audit: one fixed worst frame, two functions twice each."""
import json,sys
import numpy as np
from model_access import HERE,BASE,load_parent

def main():
    import torch,cupy
    import torch.nn.functional as F
    from spikingjelly.activation_based import functional
    args,net=load_parent(HERE/'tail_diagnostic')
    from run_bn_probe import input_frame
    from evaluate_stage2_deployment import CoarseReady
    f=np.load(HERE/'factors.npz');c=np.load(HERE/'consumer_coefficients.npz')
    q1=torch.as_tensor(f['q1'].reshape(8,96,3,3),device='cuda',dtype=torch.float64)
    q2=torch.as_tensor(f['q2'].reshape(96,8,1,1),device='cuda',dtype=torch.float64)
    scale=torch.as_tensor(f['output_scale'][None,:,None,None],device='cuda',dtype=torch.float64)
    a=torch.as_tensor(c['a_q40'].astype(np.int64)[None,:,None,None],device='cuda');b=torch.as_tensor(c['b_q20'].astype(np.int64)[None,:,None,None],device='cuda')
    r0name=net.BLOCK.rsplit('.',1)[0]+'.0';r0=net.modules[r0name];conv=net.modules[r0name+'.conv2.0'];original=conv.forward
    state={};ref={};mode_ref={};rows=[]
    flat=lambda x:x[:,0] if x.ndim==5 else x
    def pre(m,ins):
        identity=flat(ins[0]).detach();state['identity']=identity
        if 'identity' not in ref:ref['identity']=identity.clone()
        state['record']['identity_equal_first']=bool(torch.equal(identity,ref['identity']))
    def forward(x):
        g=flat(x);z=F.conv2d(g.double(),q1,padding=1);p=F.conv2d(z,q2)
        assert torch.equal(z,z.round()) and torch.equal(p,p.round())
        p=p.to(torch.int64);state['p']=p
        if 'g' not in ref:ref['g']=g.ne(0).clone();ref['p']=p.clone()
        state['record']['source_equal_first']=bool(torch.equal(g.ne(0),ref['g']))
        state['record']['p_equal_first']=bool(torch.equal(p,ref['p']))
        return (p.double()*scale).to(x.dtype)[:,None]
    def post(m,ins,out):
        identity=state.pop('identity');p=state.pop('p');old=(flat(out).double()*16384).round().clamp(-2**23,2**23-1).to(torch.int32)
        if state['mode']=='raw':I=old
        else:
            j=(identity.double()*2**20).round().clamp(-2**31,2**31-1).to(torch.int64)
            wide=p*a+((j+b)<<20);q=torch.div(wide,2**26,rounding_mode='floor');rem=wide-q*2**26
            q+=((rem>2**25)|((rem==2**25)&((q&1)!=0))).to(torch.int64);I=q.clamp(-2**23,2**23-1).to(torch.int32)
        if 'I24' not in ref:ref['I24']=I.clone()
        state['record'].update(I24_diff_vs_first_raw=int((I!=ref['I24']).sum()),I24_max_delta_vs_first_raw=int((I.to(torch.int64)-ref['I24']).abs().max()),I24_total=I.numel())
        state['expected']=I
        if state['mode']=='deployed':return (I.float()/16384)[:,None]
    def reader(m,ins,out):
        I=state.pop('expected');assert torch.equal(net.helper.i,I.double())
        gate=flat(out).ne(0)
        if 'gate' not in ref:ref['gate']=gate.clone()
        state['record'].update(r1_source_gate_difference=int((gate!=ref['gate']).sum()),r1_source_gate_total=gate.numel())
        key=state['mode']
        if key not in mode_ref:mode_ref[key]={'I24':I.clone(),'gate':gate.clone()}
        else:
            state['record']['I24_equal_same_function_previous']=bool(torch.equal(I,mode_ref[key]['I24']))
            state['record']['gate_equal_same_function_previous']=bool(torch.equal(gate,mode_ref[key]['gate']))
    hooks=[r0.register_forward_pre_hook(pre),r0.register_forward_hook(post),net.helper.source.register_forward_hook(reader)];conv.forward=forward
    frame='zurich_city_09_a_0541.npy';report=dict(complete=False,frame=frame,scope='only worst absolute paired AEE delta; raw and deployed twice; no format change',
        python=sys.version,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),all_modules_eval=all(not m.training for m in net.model.modules()),
        all_parameters_frozen=all(not p.requires_grad for p in net.model.parameters()),seed=0,rows=rows)
    def save():(HERE/'worst_frame_repeats.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        with torch.no_grad():
            x,label,valid=input_frame(args.data,frame)
            for mode in ('raw','deployed','raw','deployed'):
                functional.reset_net(net.model);net.current.pop('flow',None);state['mode']=mode;state['record']={'mode':mode}
                try:net.model(x)
                except CoarseReady:pred=F.interpolate(net.current.pop('flow'),(480,640),mode='bilinear',align_corners=False)
                else:raise AssertionError('coarse exit missing')
                err=torch.linalg.vector_norm(pred.permute(0,2,3,1)[valid]-label.permute(0,2,3,1)[valid],dim=1)
                rec=state.pop('record');rec.update(valid_pixels=err.numel(),AEE=float(err.double().sum())/err.numel())
                if 'pred' not in mode_ref[mode]:mode_ref[mode]['pred']=pred.clone()
                else:rec.update(flow_equal_same_function_previous=bool(torch.equal(pred,mode_ref[mode]['pred'])),flow_max_delta_same_function=float((pred-mode_ref[mode]['pred']).abs().max()))
                rows.append(rec);save();print('WORST_FRAME',json.dumps(rec),flush=True)
        report['complete']=True;save()
    except Exception as e:report['error']=repr(e);save();raise
    finally:
        conv.forward=original
        for h in hooks:h.remove()
        net.close()
if __name__=='__main__':main()
