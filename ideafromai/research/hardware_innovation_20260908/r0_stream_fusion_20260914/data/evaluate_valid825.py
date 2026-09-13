"""Fixed three-arm official valid825 on the existing remote Python3.12 environment."""
import json,sys,time
from pathlib import Path
import numpy as np
from model_access import HERE,BASE,load_parent
ARMS=('dense_q16','block_magnitude25','cin_fullcost25')

def main():
    args,net=load_parent(HERE/'quality825')
    import torch,cupy
    from run_bn_probe import read_names
    from evaluate_branch_control import evaluate_axis
    names=read_names(args.data,'valid');assert len(names)==825 and len(set(names))==825
    missing=[]
    for n in names:
        for p in (args.data/'event_tensors/10bins/left'/n.rsplit('_',1)[0]/n,args.data/'gt_tensors'/n,args.data/'mask_tensors'/n):
            if not p.exists():missing.append(str(p))
    assert not missing,missing
    report=dict(complete=False,python=sys.version,executable=sys.executable,torch=torch.__version__,cupy=cupy.__version__,numpy=np.__version__,
        gpu=torch.cuda.get_device_name(0),TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32,
        official_list=str(args.data/'sequence_lists/valid_split_seq.csv'),frames=names,requested_count=825,all2475files_present=True,
        snapshot='matched-dense stage320',checkpoint=str(args.checkpoint),config=str(args.config),
        training=False,mask_reselection=False,calibration_frame='thun_00_a_0002.npy',
        fullnet_bittrue=False,historical_NB0_valid825=1.44535253468097,NB0_rerun=False,
        environment_changed_from_previous_3090=True,metric='existing evaluate_axis realpreds.2 sumT; bilinear480x640 align_cornersFalse; frameequalAEE',
        numerical='unchanged fixed Q16 arrays decoded tofloat, original floating/TF32 consumers',results={})
    save=lambda:(HERE/'valid825.json').write_text(json.dumps(report,indent=2)+'\n')
    m=net.modules[net.BLOCK.rsplit('.',1)[0]+'.0.conv2.0'];original=m.weight.detach().clone()
    weights={a:np.load(HERE/(a+'_weights.npz')) for a in ARMS}
    theta=float(weights[ARMS[0]]['theta']);assert theta==1.0
    original_q=np.rint(original.cpu().numpy().astype(np.float64)*theta*65536).astype(np.int16)
    assert np.array_equal(original_q,weights[ARMS[0]]['weight_q16'])
    for a,z in weights.items():assert np.array_equal(z['weight_q16'],original_q*z['live'].repeat(8,0).repeat(4,1)[:,:,None,None])
    save()
    try:
        # Exact previous diverse10 names offer a protocol comparison in this A800 environment.
        probe_names=json.loads((BASE/'algorithm/samples.json').read_text())['valid'][:10]
        with torch.no_grad():m.weight.copy_(torch.from_numpy(weights['dense_q16']['weight_q16'].astype(np.float32)/65536).to(original))
        report['dense_diverse10_protocol_probe']=evaluate_axis(args,net.model,net.current,probe_names,'dense_protocol10',progress_tag='STREAM_PROTOCOL')
        elapsed=report['dense_diverse10_protocol_probe']['wall_seconds']
        report['estimated_three_arm_seconds_from_probe']=elapsed/10*825*3
        print('ACTUAL_ESTIMATE_SECONDS',report['estimated_three_arm_seconds_from_probe'],flush=True);save()
        args.split='valid'
        for arm in ARMS:
            with torch.no_grad():m.weight.copy_(torch.from_numpy(weights[arm]['weight_q16'].astype(np.float32)/65536).to(original))
            result=evaluate_axis(args,net.model,net.current,names,arm,progress_tag='STREAM_VALID825')
            assert result['complete'] and result['frames']==825
            rows=json.loads((args.output/(arm+'_frames.json')).read_text())
            assert [r['file'] for r in rows]==names
            report['results'][arm]=dict(**result,below_historical_NB0=result['AEE_frame_mean']<1.44535253468097)
            save()
        report['complete']=True;save()
    except Exception as e:report['error']=repr(e);save();raise
    finally:
        with torch.no_grad():m.weight.copy_(original)
        net.close()
    print('VALID825_COMPLETE',json.dumps(report['results']),flush=True)
if __name__=='__main__':main()
