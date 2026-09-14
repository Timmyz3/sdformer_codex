"""Fixed R8 official825, with actual first-frame BN/add/I24 boundary capture."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from model_access import HERE,BASE,load_parent
ORIGINS=np.array([(0,0),(0,318),(238,0),(238,318),(32,48),(60,80),(120,160),(180,240)],dtype=np.int32)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--count',type=int,default=825);ap.add_argument('--data-root');opt=ap.parse_args()
    import torch,cupy
    import torch.nn.functional as F
    f=np.load(HERE/'factors.npz');q1=f['q1'].astype(np.int64);q2=f['q2'].astype(np.int64);scale=f['output_scale'].astype(np.float64)
    assert np.array_equal(q2@q1,f['expanded']) and float(f['theta'])==1 and not f['bias'].any()
    args,net=load_parent(HERE/'quality',opt.data_root)
    from run_bn_probe import read_names
    from evaluate_branch_control import evaluate_axis
    from fixed_temporal_coordinates import STATE_SCALE,STATE_MIN,STATE_MAX
    r0name=net.BLOCK.rsplit('.',1)[0]+'.0';r0=net.modules[r0name];r1=net.modules[net.BLOCK]
    conv=net.modules[r0name+'.conv2.0'];norm=net.modules[r0name+'.norm2'];bn=norm.norm_layer
    def arr(x):return None if x is None else x.detach().cpu().numpy()
    bnmeta=dict(module=r0name+'.norm2.norm_layer',type=type(bn).__module__+'.'+type(bn).__name__,training=bn.training,
        track_running_stats=bn.track_running_stats,eps=bn.eps,step_mode=getattr(bn,'step_mode',None),
        running_mean_present=bn.running_mean is not None,running_var_present=bn.running_var is not None,
        actual_forward='MS_ResBlock sn1/conv1/norm1/sn2/conv2/norm2 then ADD identity; direct r1.sn1 input')
    (HERE/'bn_runtime.json').write_text(json.dumps(bnmeta,indent=2)+'\n');print('R0_BN_RUNTIME',json.dumps(bnmeta),flush=True)
    q1t=torch.as_tensor(q1.reshape(8,96,3,3),device='cuda',dtype=torch.float64)
    q2t=torch.as_tensor(q2.reshape(96,8,1,1),device='cuda',dtype=torch.float64)
    st=torch.as_tensor(scale.reshape(1,96,1,1),device='cuda',dtype=torch.float64)
    original=conv.forward;captured=[];state={};integer_checks=[]
    def flatten(x):return x[:,0] if x.ndim==5 else x
    def tiles(x):
        x=flatten(x);return np.stack([arr(x[:,:,y:y+2,z:z+2]) for y,z in ORIGINS])
    def r0pre(m,inputs):
        if not captured:
            x=inputs[0];state['identity_gpu']=x.detach()
            state['identity_fp32']=tiles(x)
            state['identity_range']=[float(x.min()),float(x.max())]
    def forward(x):
        g=flatten(x);assert tuple(g.shape)==(10,96,240,320)
        assert torch.equal(g,g.ne(0).to(g))
        z=F.conv2d(g.double(),q1t,padding=1);p=F.conv2d(z,q2t)
        assert torch.equal(z,z.round()) and float(z.abs().max())<=2592
        assert torch.equal(p,p.round()) and float(p.abs().max())<=679477248
        y=(p*st).to(x.dtype)
        integer_checks.append(dict(frame_index=len(integer_checks),z_min=int(z.min()),z_max=int(z.max()),p_min=int(p.min()),p_max=int(p.max())))
        if not captured:
            gp=F.pad(g,(1,1,1,1))
            state['source_bits']=np.stack([arr(gp[:,:,oy:oy+4,ox:ox+4]).astype(bool) for oy,ox in ORIGINS])
            state['z_int']=tiles(z).astype(np.int16);state['p_int']=tiles(p).astype(np.int32)
            state['linear_fp32']=tiles(y)
            # Full native source in compact time-word format; no patch assembly.
            bits=arr(g.ne(0));words=np.zeros((96,240,320),dtype=np.uint16)
            for t in range(10):words|=bits[t].astype(np.uint16)<<t
            np.save(HERE/'first_source_words.npy',words)
            yd=y.double();mu=yd.mean((0,2,3));var=((yd-mu[None,:,None,None])**2).mean((0,2,3))
            state['actual_frame_mean_fp64']=arr(mu);state['actual_frame_var_biased_fp64']=arr(var)
            if not bn.training and bn.running_mean is not None and bn.running_var is not None:
                usedmu=bn.running_mean.double();usedvar=bn.running_var.double();state['BN_statistics_kind']='fixed_eval_running'
            else:usedmu=mu;usedvar=var;state['BN_statistics_kind']='dynamic_frame_TBHW_biased'
            gain=bn.weight.double()/torch.sqrt(usedvar+bn.eps);offset=bn.bias.double()-gain*usedmu
            state['bn_gain_fp64']=arr(gain);state['bn_offset_fp64']=arr(offset)
            state['BN_used_mean_fp64']=arr(usedmu);state['BN_used_var_fp64']=arr(usedvar)
            state['BN_affine_reference_gpu']=yd*gain[None,:,None,None]+offset[None,:,None,None]
        return y[:,None] if x.ndim==5 else y
    def bnhook(m,inputs,out):
        if not captured:
            state['bn2_fp32']=tiles(out);state['bn_gpu']=out.detach()
            state['bn_affine_fp64_max_error']=float((flatten(out).double()-state.pop('BN_affine_reference_gpu')).abs().max())
    def r0hook(m,inputs,out):
        if not captured:
            assert torch.equal(out,state['identity_gpu']+state['bn_gpu'])
            state['r0_output_fp32']=tiles(out);state['r0_output_range']=[float(out.min()),float(out.max())]
            state['r0_gpu']=out.detach()
    def r1pre(m,inputs):
        if not captured:assert torch.equal(inputs[0],state['r0_gpu']);state['r1_input_fp32']=tiles(inputs[0])
    def sourcepost(m,inputs,out):
        if captured:return
        expected=(flatten(state['r0_gpu']).double()*STATE_SCALE).round().clamp(STATE_MIN,STATE_MAX)
        assert torch.equal(net.helper.i,expected)
        state['I24']=tiles(net.helper.i).astype(np.int32);state['r1_source_gate']=tiles(out)
        meta=dict(complete=True,frame='zurich_city_09_a_0001.npy',BN=bnmeta,BN_statistics_kind=state['BN_statistics_kind'],
            identity_range=state['identity_range'],r0_output_range=state['r0_output_range'],
            bn_affine_fp64_max_error=state['bn_affine_fp64_max_error'],r0_add_exact=True,r1_input_equal_r0=True,
            I24_exact_check=True,I24_fraction=14,I24_signed_bits=24,I24_clip_counts=net.helper.frame['clip_counts']['I24'],
            integer_function='z=Q1*g;p=Q2*z;y32=FP32(p*output_scale64); actual BN2 float32; ADD identity float32; I24=RNE/clip(y*16384)',
            coefficient_folding_is_not_FP32_bittrue=True,source_shape=[8,10,96,4,4],output_shape=[8,10,96,2,2],
            output_origin_yx=ORIGINS.tolist(),input_origin_yx=(ORIGINS-1).tolist(),torch=torch.__version__,gpu=torch.cuda.get_device_name(0))
        payload={k:v for k,v in state.items() if isinstance(v,np.ndarray)}
        payload.update(q1=q1.astype(np.int8),q2=q2.astype(np.int16),output_scale=scale,bn_gamma=arr(bn.weight),bn_beta=arr(bn.bias),
            bn_running_mean=arr(bn.running_mean) if bn.running_mean is not None else np.empty(0),
            bn_running_var=arr(bn.running_var) if bn.running_var is not None else np.empty(0),bn_eps=np.array(bn.eps),
            output_origin_yx=ORIGINS,input_origin_yx=ORIGINS-1,theta=f['theta'])
        np.savez_compressed(HERE/'consumer_first8.npz',**payload)
        (HERE/'consumer_contract.json').write_text(json.dumps(meta,indent=2)+'\n');print('CONSUMER_CONTRACT_READY',json.dumps(meta),flush=True)
        state.clear();captured.append(True)
    hooks=[r0.register_forward_pre_hook(r0pre),norm.register_forward_hook(bnhook),r0.register_forward_hook(r0hook),r1.register_forward_pre_hook(r1pre),net.helper.source.register_forward_hook(sourcepost)]
    conv.forward=forward
    names=read_names(args.data,'valid')[:opt.count];assert names[0]=='zurich_city_09_a_0001.npy'
    if opt.count==825:assert len(names)==len(set(names))==825
    for n in names:
        for path in (args.data/'event_tensors/10bins/left'/n.rsplit('_',1)[0]/n,args.data/'gt_tensors'/n,args.data/'mask_tensors'/n):assert path.exists(),path
    report=dict(complete=False,frames=names,requested_count=opt.count,python=sys.version,torch=torch.__version__,cupy=cupy.__version__,gpu=torch.cuda.get_device_name(0),
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32,integer_function='samefactors as previous stage; FP64 z,p then FP64scale to originalfloat32 consumers',
        numerical_consumer='original BN2/add and current literal I24; no new fixedpointconsumer yet',
        historical_NB0_valid825=1.44535253468097,NB0_rerun=False,training=False,format_change=False,fullnet_bittrue=False,integer_checks=integer_checks)
    save=lambda:(HERE/'r8_valid825.json').write_text(json.dumps(report,indent=2)+'\n')
    save();args.split='valid'
    try:
        result=evaluate_axis(args,net.model,net.current,names,'integer_r8',progress_tag='R8_VALID825')
        assert len(integer_checks)==len(names) and captured
        report.update(complete=True,result=result,below_historical_NB0=result['AEE_frame_mean']<1.44535253468097);save()
    except Exception as e:report['error']=repr(e);save();raise
    finally:
        conv.forward=original
        for h in hooks:h.remove()
        net.close()
if __name__=='__main__':main()
