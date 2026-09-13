"""Five actual full-model AEE arms; Q16 r0 coefficients, floating consumers."""
import json,sys,time,argparse
import numpy as np
from model_access import HERE,BASE,load_parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=10);p.add_argument('--output-label',default='quality');p.add_argument('--data-root')
    options=p.parse_args();assert options.count in (1,10)
    args,net=load_parent(HERE/(options.output_label+'_aee'),options.data_root)
    import torch
    import torch.nn.functional as F
    import cupy
    from evaluate_branch_control import evaluate_axis
    name=net.BLOCK.rsplit('.',1)[0]+'.0.conv2.0'
    m=net.modules[name];original_w=m.weight.detach().clone();original_forward=m.forward
    names=json.loads((BASE/'algorithm/samples.json').read_text())['valid'][:options.count]
    record=dict(complete=False,python=sys.version,executable=sys.executable,torch=torch.__version__,
        cupy=cupy.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0),
        frames=names,data_root=str(args.data),requested_count=options.count,diverse10_complete=False,parent='fresh relocated matched-dense stage320; old AEE differs and is not inherited',
        module=name,metric='real preds.2 summed over T; bilinear480x640 align_corners=False; original valid GT mask',
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32,
        NB0_diverse10=1.45460286107,NB0_valid825=1.44535253468097,
        NB0_in_this_environment_rerun=False,calibration_overlaps_frame0=True,valid825_run=False,training=False,
        numeric_boundary='Exact same Q16 coefficient arrays as RTL decoded tofloat, floating CUDA conv and consumers; not fullnetbittrue. Winograd E4 is evaluated without leaf RNE2; RTL rounds E4/4 to Q16.',results={})
    def save(): (HERE/(options.output_label+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    save()
    try:
        for axis in ['parent_fp32','dense_q16','physical25_q16','magnitude25_q16','coordinate25_q16']:
            m.forward=original_forward
            with torch.no_grad():m.weight.copy_(original_w)
            if axis=='coordinate25_q16':
                z=np.load(HERE/(axis+'.npz'))
                kernel=torch.from_numpy(z['phase_kernel4_q16'].astype(np.float32)/(4*65536)).to(original_w)
                def phase_forward(x):
                    shape=x.shape; assert shape[-2:] == (240,320)
                    flat=x.flatten(0,1) if x.ndim==5 else x
                    y=F.pixel_shuffle(F.conv2d(flat,kernel,stride=2,padding=1),2)
                    return y.reshape(*shape[:-3],96,240,320)
                m.forward=phase_forward
                with torch.no_grad():
                    source=torch.from_numpy(z['source_bits'].astype(np.float32)).to(original_w).flatten(0,1)
                    local=F.conv2d(source,kernel).reshape(8,10,96,2,2).cpu().numpy()
                    ideal=z['golden_accum4'].astype(np.float64)/(4*65536)
                    rne=z['golden_rne_q16'].astype(np.float64)/65536
                record['coordinate_cal_leaf']=dict(float_vs_unrounded_integer_max_abs=float(np.max(np.abs(local-ideal))),
                    float_vs_rtl_rne_max_abs=float(np.max(np.abs(local-rne))),
                    output_phase_order_checked=True,RTL_RNE2_implemented_in_SV=True)
            elif axis!='parent_fp32':
                z=np.load(HERE/(axis+'.npz'))
                with torch.no_grad():m.weight.copy_(torch.from_numpy(z['weight_q16'].astype(np.float32)/65536).to(original_w))
            summary=evaluate_axis(args,net.model,net.current,names,axis,progress_tag='R0_QUALITY')
            rows=json.loads((args.output/(axis+'_frames.json')).read_text())
            holdout9=float(np.mean([r['AEE'] for r in rows[1:]])) if len(rows)>1 else None
            record['results'][axis]=dict(**summary,holdout9_frame_mean=holdout9,
                below_historical_NB0_gate=summary['AEE_frame_mean']<1.45460286107 if len(rows)==10 else None,
                evaluation_complete=True,fullnet_bittrue=False)
            save()
        record['complete']=True;record['diverse10_complete']=options.count==10;save()
    except Exception as e:
        record['error']=repr(e);save();raise
    finally:
        m.forward=original_forward
        with torch.no_grad():m.weight.copy_(original_w)
        net.close()
    print('QUALITY_COMPLETE',json.dumps(record),flush=True)
if __name__=='__main__':main()
