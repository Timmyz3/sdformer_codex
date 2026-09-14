"""Exact signed64 consumer endpoint, independently evaluated through current I24 reader."""
import argparse,json,sys
import numpy as np
from model_access import HERE,BASE,load_parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=('diverse','valid'),required=True);o=ap.parse_args()
    import torch,cupy
    import torch.nn.functional as F
    from spikingjelly.activation_based import functional
    f=np.load(HERE/'factors.npz');c=np.load(HERE/'consumer_coefficients.npz');ref=np.load(HERE/'consumer_first8.npz')
    args,net=load_parent(HERE/('deployed_'+o.split));args.split=o.split
    from run_bn_probe import read_names
    from evaluate_branch_control import evaluate_axis
    r0name=net.BLOCK.rsplit('.',1)[0]+'.0';r0=net.modules[r0name];conv=net.modules[r0name+'.conv2.0'];bn=net.modules[r0name+'.norm2.norm_layer']
    assert not bn.training and bn.track_running_stats
    for key,tensor in [('bn_gamma',bn.weight),('bn_beta',bn.bias),('bn_running_mean',bn.running_mean),('bn_running_var',bn.running_var)]:assert np.array_equal(ref[key],tensor.detach().cpu().numpy()),key
    q1=torch.as_tensor(f['q1'].reshape(8,96,3,3),device='cuda',dtype=torch.float64)
    q2=torch.as_tensor(f['q2'].reshape(96,8,1,1),device='cuda',dtype=torch.float64)
    scale=torch.as_tensor(f['output_scale'][None,:,None,None],device='cuda',dtype=torch.float64)
    a=torch.as_tensor(c['a_q40'].astype(np.int64)[None,:,None,None],device='cuda')
    b=torch.as_tensor(c['b_q20'].astype(np.int64)[None,:,None,None],device='cuda')
    rows=[];state={};original=conv.forward
    def flat(x):return x[:,0] if x.ndim==5 else x
    def pre(m,inputs):state['identity']=flat(inputs[0]).detach()
    def forward(x):
        g=flat(x);assert torch.equal(g,g.ne(0).to(g))
        z=F.conv2d(g.double(),q1,padding=1);p=F.conv2d(z,q2)
        assert torch.equal(z,z.round()) and z.abs().max()<=2592
        assert torch.equal(p,p.round()) and p.abs().max()<=679477248
        state['p']=p.to(torch.int64)
        if not rows and o.split=='diverse':
            bits=g.ne(0).cpu().numpy();words=np.zeros((96,240,320),np.uint16)
            for t in range(10):words|=bits[t].astype(np.uint16)<<t
            state['source_equal']=bool(np.array_equal(words,np.load(HERE/'first_source_words.npy')))
        y=(p*scale).to(x.dtype)
        return y[:,None] if x.ndim==5 else y
    def post(m,inputs,original_output):
        identity=state.pop('identity');p=state.pop('p')
        jr=(identity.double()*2**20).round();j=jr.clamp(-2**31,2**31-1).to(torch.int64)
        wide=p*a+((j+b)<<20)
        q=torch.div(wide,1<<26,rounding_mode='floor');rem=wide-q*(1<<26)
        rounded=q+((rem>(1<<25))|((rem==(1<<25))&((q&1)!=0))).to(torch.int64)
        I24=rounded.clamp(-2**23,2**23-1)
        rec=dict(frame_index=len(rows),identity_saturations=int((jr!=j).sum()),I24_low=int((rounded<-(2**23)).sum()),I24_high=int((rounded>2**23-1).sum()),
            identity_min=float(identity.min()),identity_max=float(identity.max()),wide_min=int(wide.min()),wide_max=int(wide.max()))
        if not rows:
            old=(flat(original_output).double()*16384).round().clamp(-2**23,2**23-1).to(torch.int64)
            rec.update(different_original_I24=int((old!=I24).sum()),max_original_delta=int((old-I24).abs().max()))
            if o.split=='diverse':
                rec['source_equal_baseline_capture']=state.pop('source_equal')
                assert rec['source_equal_baseline_capture']
                for name,x in [('raw_p',p),('identity_q20',j),('i24_new',I24)]:
                    ar=x.cpu().numpy().astype('<i4')
                    tile=ar.reshape(10,96,120,2,160,2).transpose(2,4,0,1,3,5).reshape(19200,10,96,2,2)
                    np.save(HERE/(name+'_first64.npy'),tile[128:192])
                    np.save(HERE/(name+'_full.npy'),tile)
                print('FULL_CONSUMER_ARRAYS_READY',json.dumps(rec),flush=True)
        state['expected_I24']=I24;rows.append(rec)
        return (I24.float()/16384)[:,None]
    def reader(m,inputs,out):
        assert torch.equal(net.helper.i,state.pop('expected_I24').double()),'Current reader did not receive exact new I24'
    hooks=[r0.register_forward_pre_hook(pre),r0.register_forward_hook(post),net.helper.source.register_forward_hook(reader)]
    conv.forward=forward
    names=read_names(args.data,'valid') if o.split=='valid' else json.loads((BASE/'algorithm/samples.json').read_text())['valid'][:10]
    assert len(names)==(825 if o.split=='valid' else 10)
    report=dict(complete=False,split=o.split,frames=names,python=sys.version,torch=torch.__version__,cupy=cupy.__version__,gpu=torch.cuda.get_device_name(0),
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32,
        function='signed64 p*aQ40+((JQ20+bQ20)<<20), one signedtie-even shift26 then saturateI24',
        J_rule='RNE identityfloat32*2^20 then signed32sat',old_FP_BN_add_executed_as_ignored_shadow=True,
        reader='new I24/16384 exactfloat32 reinjection; currentLiteralForward I24 checked equal everyframe',
        training=False,format_search=False,fullnet_bittrue=False,per_frame_integer_checks=rows)
    save=lambda:(HERE/('deployed_'+o.split+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        result=evaluate_axis(args,net.model,net.current,names,'integer_consumer',progress_tag='DEPLOYED_CONSUMER')
        threshold=1.44535253468097 if o.split=='valid' else 1.45460286107
        report.update(complete=True,result=result,NB0_historical=threshold,below_historical_NB0=result['AEE_frame_mean']<threshold);save()
    except Exception as e:report['error']=repr(e);save();raise
    finally:
        conv.forward=original
        for h in hooks:h.remove()
        net.close()
if __name__=='__main__':main()
