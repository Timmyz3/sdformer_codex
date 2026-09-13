"""One fixed low-rank integer chain; no TF32 inside z/p, no parameter search."""
import json,sys,time
import numpy as np
from model_access import HERE,BASE,load_parent

def main():
    import torch,cupy
    import torch.nn.functional as F
    f=np.load(HERE/'integer_factors.npz')
    q1=f['q1'].astype(np.int64);q2=f['q2'].astype(np.int64);scale=f['output_scale'].astype(np.float64)
    assert q1.shape==(8,864) and q2.shape==(96,8) and scale.shape==(96,)
    assert np.array_equal(q2@q1,f['expanded']) and float(f['theta'])==1.0 and not f['bias'].any()
    args,net=load_parent(HERE/'integer_factor_aee')
    from evaluate_branch_control import evaluate_axis
    m=net.modules[net.BLOCK.rsplit('.',1)[0]+'.0.conv2.0'];original_forward=m.forward
    q1t=torch.as_tensor(q1.reshape(8,96,3,3),device='cuda',dtype=torch.float64)
    q2t=torch.as_tensor(q2.reshape(96,8,1,1),device='cuda',dtype=torch.float64)
    st=torch.as_tensor(scale.reshape(1,96,1,1),device='cuda',dtype=torch.float64)
    ext=torch.as_tensor(f['expanded'].reshape(96,96,3,3),device='cuda',dtype=torch.float64)
    rows=[]
    origins=np.array([(0,0),(0,318),(238,0),(238,318),(32,48),(60,80),(120,160),(180,240)])
    def integer_forward(a):
        shape=a.shape
        if a.ndim==5:
            assert tuple(shape[:3])==(10,1,96);x=a.flatten(0,1)
        else:assert a.ndim==4;x=a
        assert tuple(x.shape)==(10,96,240,320)
        assert torch.equal(x,x.ne(0).to(x)), 'Source amplitude is not0/1'
        g=x.to(torch.float64)
        z=F.conv2d(g,q1t,padding=1)
        assert torch.equal(z,z.round()) and int(z.abs().max())<=2592
        p=F.conv2d(z,q2t)
        assert torch.equal(p,p.round()) and int(p.abs().max())<=679477248
        rec=dict(frame_index=len(rows),source_spikes=int(x.ne(0).sum()),z_min=int(z.min()),z_max=int(z.max()),p_min=int(p.min()),p_max=int(p.max()),integer_checks=True)
        if not rows:
            # Full first frame independent expanded operator, same exact integer function.
            direct=F.conv2d(g,ext,padding=1)
            assert torch.equal(p,direct);rec['full_expanded_integer_equal']=True
            zp=np.stack([z[:,:,oy:oy+2,ox:ox+2].cpu().numpy().astype(np.int16) for oy,ox in origins])
            pp=np.stack([p[:,:,oy:oy+2,ox:ox+2].cpu().numpy().astype(np.int32) for oy,ox in origins])
            gp=F.pad(x,(1,1,1,1));sp=np.stack([gp[:,:,oy:oy+4,ox:ox+4].bool().cpu().numpy() for oy,ox in origins])
            np.savez_compressed(HERE/'integer_factor_first8.npz',source_bits=sp,z=zp,p=pp,output_origin_yx=origins,output_scale=scale)
        rows.append(rec)
        out=(p*st).to(a.dtype)
        return out.reshape(shape) if a.ndim==5 else out
    m.forward=integer_forward
    names=json.loads((BASE/'algorithm/samples.json').read_text())['valid'][:10]
    report=dict(complete=False,python=sys.version,torch=torch.__version__,cupy=cupy.__version__,gpu=torch.cuda.get_device_name(0),
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32,
        frames=names,factors='integer_factors.npz',rank=8,definition='z=Q1*g; p=Q2*z; y=p*output_scale; nointermediateRNE',
        integer_zp_exact_FP64=True,output_scaling='FP64scale then originalconsumer float32',
        bound_z=2592,bound_p=679477248,training=False,format_search=False,fullnet_bittrue=False,valid825=False,
        NB0_diverse10_historical=1.45460286107,NB0_rerun=False,
        environment='same A800/env312 as currentthree-arm825 anddenseprotocol10',per_frame_integer_checks=rows)
    save=lambda:(HERE/'integer_factor_diverse10.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        report['result']=evaluate_axis(args,net.model,net.current,names,'integer_q1_q2',progress_tag='INTEGER_FACTOR_AEE')
        assert len(rows)==10 and report['result']['frames']==10
        report.update(complete=True,below_historical_NB0=report['result']['AEE_frame_mean']<1.45460286107);save()
    except Exception as e:report['error']=repr(e);save();raise
    finally:m.forward=original_forward;net.close()
    print('INTEGER_FACTOR_COMPLETE',json.dumps(report['result']),flush=True)
if __name__=='__main__':main()
