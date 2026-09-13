"""Fresh real contiguous r0 conv2 T10/C96 source, one matched-dense frame."""
from pathlib import Path
import json
import sys
import numpy as np
from model_access import HERE,BASE,ALGORITHM,load_parent

# Chosen before source values: four corners and four distinct interior tiles.
ORIGINS=np.array([(0,0),(0,318),(238,0),(238,318),(32,48),(60,80),(120,160),(180,240)],dtype=np.int32)

def main():
    args,net=load_parent(HERE/'capture_evaluation')
    import torch
    from evaluate_branch_control import evaluate_axis
    r0=net.BLOCK.rsplit('.',1)[0]+'.0';name=r0+'.conv2.0'
    module=net.modules[name];sn=net.modules[r0+'.sn2.spiking_neuron']
    frame=json.loads((BASE/'algorithm/samples.json').read_text())['valid'][0]
    manifest=dict(complete=False,python=sys.version,python_executable=sys.executable,torch_version=torch.__version__,source_snapshot='matched-dense stage320',frame=frame,module=name,
        capture_kind='fresh hook of native contiguous tensor; not assembled from64 sampled patches',
        source_axes='tile,T,C,local_y,local_x',output_axes='tile,T,N,phase_y,phase_x',
        source_shape=[8,10,96,4,4],output_shape=[8,10,96,2,2],
        output_origin_yx=ORIGINS.tolist(),input_origin_yx=(ORIGINS-1).tolist(),
        kernel=[3,3],stride=[1,1],padding=[1,1],dilation=[1,1],
        scope='Eight spatial F(2x2,3x3) tiles. Complete T10,C96,N96; no full layer timing.',
        diagnostic_integer='raw sum of source_bits * round_even(theta*FP32_W*2^16); original bias stored separately; Q16 is not inherited validated model.',
        upstream_checkpoint=str(args.checkpoint),upstream_config=str(args.config),
        parent_constants=str(ALGORITHM/'matched_training/dense/stage320/deployed_constants.npz'),
        source_code='capture_native.py + read-only ParentNetwork/profile_current import path')
    (HERE/'capture_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    captured=[]
    def hook(m,inputs,output):
        if captured:return
        a=inputs[0].detach();y=output.detach()
        original_shape=list(a.shape)
        if a.ndim==5:
            assert tuple(a.shape[:2])==(10,1);a=a[:,0];y=y[:,0]
        assert tuple(a.shape)==(10,96,240,320) and tuple(y.shape)==(10,96,240,320)
        assert tuple(m.kernel_size)==(3,3) and tuple(m.stride)==(1,1) and tuple(m.padding)==(1,1)
        theta=float(sn.thresh.detach().reshape(-1)[0])
        assert sn.thresh.numel()==1
        err=float((a-a.ne(0).to(a)*theta).abs().max())
        assert err==0., f'Not fixed binary amplitude: {err}'
        blocks=[];outputs=[];valid=[]
        for oy,ox in ORIGINS:
            sy,sx=int(oy)-1,int(ox)-1
            block=torch.zeros((10,96,4,4),dtype=a.dtype,device=a.device)
            v=np.zeros((4,4),dtype=np.bool_)
            y0,y1=max(sy,0),min(sy+4,240);x0,x1=max(sx,0),min(sx+4,320)
            block[:,:,y0-sy:y1-sy,x0-sx:x1-sx]=a[:,:,y0:y1,x0:x1]
            v[y0-sy:y1-sy,x0-sx:x1-sx]=True
            blocks.append(block.cpu().float().numpy());valid.append(v)
            outputs.append(y[:,:,oy:oy+2,ox:ox+2].cpu().float().numpy())
        source=np.stack(blocks);bits=source!=0;w=m.weight.detach().cpu().float().numpy()
        reference=BASE/'open_fusion_execution/major_operator_fusions_20260913/root_owned'/ (name.replace('.','_')+'.npz')
        with np.load(reference) as old:
            same_weight=bool(np.array_equal(w,old['weight']))
        has_bias=m.bias is not None
        bias=m.bias.detach().cpu().float().numpy() if has_bias else np.zeros(96,dtype=np.float32)
        q=np.rint(w.astype(np.float64)*theta*65536).astype(np.int64)
        assert q.min()>=-32768 and q.max()<=32767
        gold=np.empty((8,10,96,2,2),np.int64)
        for py in range(2):
            for px in range(2):
                gold[:,:,:,py,px]=np.einsum('ptcij,ncij->ptn',bits[:,:,:,py:py+3,px:px+3].astype(np.int64),q)
        np.savez_compressed(HERE/'r0_contiguous_t10.npz',source_fp32=source,source_bits=bits,
            source_valid_yx=np.stack(valid),output_fp32=np.stack(outputs),weight_fp32=w,
            weight_q16=q.astype(np.int16),bias_fp32=bias,bias_present=np.array(has_bias),
            theta=np.array(theta),weight_exponent=np.array(16),golden_accum=gold,
            output_origin_yx=ORIGINS,input_origin_yx=ORIGINS-1,
            frame=np.array(frame),module=np.array(name),source_axes=np.array('tile,T,C,y,x'),
            output_axes=np.array('tile,T,N,y,x'))
        manifest.update(capture_written=True,npz='r0_contiguous_t10.npz',actual_input_shape=original_shape,
            reference_weight_path=str(reference),reference_weight_array_equal=same_weight,
            theta=theta,amplitude_error_max=err,bias_present=has_bias,
            weight_q16_range=[int(q.min()),int(q.max())],source_nonzero=int(bits.sum()),
            real_source_nonzero_full_domain=int(torch.count_nonzero(a)),real_source_elements_full_domain=a.numel(),
            source_layer_type=type(sn).__module__+'.'+type(sn).__name__,
            output_mode=getattr(sn,'output_mode',None),threshold_mode=getattr(sn,'threshold_mode',None))
        (HERE/'capture_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        captured.append(True);print('CONTIGUOUS_CAPTURE_WRITTEN',json.dumps(manifest),flush=True)
    h=module.register_forward_hook(hook)
    try:
        summary=evaluate_axis(args,net.model,net.current,[frame],'unmodified_capture',progress_tag='R0_CAPTURE')
        assert captured
        manifest.update(complete=True,unmodified_frame_AEE=summary)
        (HERE/'capture_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    finally:h.remove();net.close()

if __name__=='__main__':main()
