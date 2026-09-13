"""One real full-layer capture. No synthetic patch assembly or integer golden yet."""
import json,sys,time
import numpy as np
from model_access import HERE,BASE,load_parent
FRAME='zurich_city_09_a_0001.npy'
PREV=BASE/'r0_execution_trials_20260913/data_and_quality'
CONTROLS=BASE/'r0_source_retirement_20260913/data'

def main():
    args,net=load_parent(HERE/'capture_aee',PREV/'data_mirror')
    import torch
    from evaluate_branch_control import evaluate_axis
    name=net.BLOCK.rsplit('.',1)[0]+'.0.conv2.0';m=net.modules[name]
    sn=net.modules[net.BLOCK.rsplit('.',1)[0]+'.0.sn2.spiking_neuron']
    report=dict(complete=False,frame=FRAME,module=name,python=sys.version,executable=sys.executable,
        source_shape=[10,96,240,320],source_axes='T,C,y,x',source_words_axes='C,y,x',
        source_words_dtype='uint16 little-endian, low10bit: bit t equals g[t]',
        gold_axes='tile,T,N,phase_y,phase_x',gold_shape=[19200,10,96,2,2],
        tile_order='row-major 120x160, output_origin=(2*(id//160),2*(id%160)), input_origin=output_origin-1',
        padding='SV/fixture masks global coordinates outside H240,W320 to0',
        first64_tile_ids=list(range(128,192)),kernel=[3,3],stride=[1,1],padding_size=[1,1],
        checkpoint=str(args.checkpoint),config=str(args.config),capture_snapshot='matched-dense stage320',
        integer_function='sum g*round_even(theta*W*65536), signed32 exact completeK864; no bias',
        fullnet_bittrue=False,gold_complete=False)
    captured=[]
    def hook(module,inputs,output):
        if captured:return
        a=inputs[0].detach();y=output.detach()
        report['original_shape']=list(a.shape)
        if a.ndim==5:assert tuple(a.shape[:2])==(10,1);a=a[:,0];y=y[:,0]
        assert tuple(a.shape)==(10,96,240,320)
        assert sn.thresh.numel()==1
        theta=float(sn.thresh.detach());err=float((a-a.ne(0).to(a)*theta).abs().max())
        assert err==0 and module.bias is None
        bits=a.ne(0).cpu().numpy();w=module.weight.detach().cpu().numpy()
        q=np.rint(w.astype(np.float64)*theta*65536).astype(np.int64)
        assert q.min()>=-32768 and q.max()<=32767
        old=np.load(PREV/'r0_contiguous_t10.npz')
        assert np.array_equal(q,old['weight_q16']) and np.array_equal(w,old['weight_fp32'])
        old_tiles=[]
        padded=np.pad(bits,((0,0),(0,0),(1,1),(1,1)))
        for oy,ox in old['output_origin_yx']:old_tiles.append(padded[:,:,oy:oy+4,ox:ox+4])
        assert np.array_equal(np.stack(old_tiles),old['source_bits'])
        np.save(HERE/'source_bits.npy',bits)
        words=np.zeros((96,240,320),dtype='<u2')
        for t in range(10):words|=bits[t].astype(np.uint16)<<t
        np.save(HERE/'source_words.npy',words)
        # Original full FP32 linear output kept as a distinct reference function.
        np.save(HERE/'original_output_fp32.npy',y.cpu().float().numpy())
        arms={}
        for arm in ('dense_q16','block_magnitude25','cin_fullcost25'):
            z=np.load(CONTROLS/(arm+'.npz'));live=z['live'];qw=z['weight_q16']
            assert np.array_equal(qw,q*live.repeat(8,0).repeat(4,1)[:,:,None,None])
            np.savez(HERE/(arm+'_weights.npz'),weight_q16=qw.astype('<i2'),live=live,
                weight_fp32=w,theta=np.array(theta),exponent=np.array(16),frame=np.array(FRAME))
            arms[arm]=dict(weights=arm+'_weights.npz',deleted_blocks=int((~live).sum()))
        report.update(source_complete=True,theta=theta,amplitude_error=err,bias_present=False,
            old_weight_array_equal=True,old8_source_array_equal=True,spikes=int(bits.sum()),elements=int(bits.size),
            q_range=[int(q.min()),int(q.max())],arms=arms,source_bits='source_bits.npy',source_words='source_words.npy',
            torch=torch.__version__,gpu=torch.cuda.get_device_name(0))
        (HERE/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        captured.append(True);print('FULL_SOURCE_READY',report['spikes'],flush=True)
    h=m.register_forward_hook(hook)
    try:
        report['capture_AEE']=evaluate_axis(args,net.model,net.current,[FRAME],'parent_capture',progress_tag='STREAM_CAPTURE')
        assert captured;report['complete']=True
        (HERE/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:h.remove();net.close()
if __name__=='__main__':main()
