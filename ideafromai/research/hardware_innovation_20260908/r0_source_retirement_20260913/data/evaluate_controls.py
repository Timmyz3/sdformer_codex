"""Six fixed paired diverse10 arms, same local Python3.12 environment."""
import argparse,json,sys
import numpy as np
from model_access import HERE,BASE,load_parent
PREVIOUS=BASE/'r0_execution_trials_20260913/data_and_quality'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fusion-npz',required=True);options=parser.parse_args()
    fixtures={'dense_q16':HERE/'dense_q16.npz','block_magnitude25':HERE/'block_magnitude25.npz',
        'cin_magnitude25':HERE/'cin_magnitude25.npz','cin_fullcost25':HERE/'cin_fullcost25.npz',
        'fusion72':__import__('pathlib').Path(options.fusion_npz)}
    base=np.load(PREVIOUS/'r0_contiguous_t10.npz')['weight_q16'].astype(np.int64)
    for name,path in fixtures.items():
        z=np.load(path);live=z['live'];assert live.shape==(12,24)
        assert int((~live).sum())==(0 if name=='dense_q16' else 72)
        expected=base*live.repeat(8,0).repeat(4,1)[:,:,None,None]
        assert np.array_equal(z['weight_q16'],expected),(name,'different masked integer function')
    args,net=load_parent(HERE/'diverse10_aee',PREVIOUS/'data_mirror')
    import torch,cupy
    from evaluate_branch_control import evaluate_axis
    m=net.modules[net.BLOCK.rsplit('.',1)[0]+'.0.conv2.0'];original=m.weight.detach().clone()
    names=json.loads((BASE/'algorithm/samples.json').read_text())['valid'][:10]
    nb=json.loads((PREVIOUS/'nb0_diverse10.json').read_text());assert nb['complete']
    selection=json.loads((HERE/'calibration_selection.json').read_text())
    assert selection['frame'] not in names
    assert selection['frame'].rsplit('_',1)[0] not in {n.rsplit('_',1)[0] for n in names}
    report=dict(complete=False,python=sys.version,executable=sys.executable,torch=torch.__version__,cupy=cupy.__version__,
        numpy=np.__version__,gpu=torch.cuda.get_device_name(0),TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32,
        frames=names,calibration_frame=selection['frame'],calibration_source='calibration_grid_t10.npz',calibration_tiles=336,
        calibration_frame_overlap=False,calibration_sequence_overlap=False,training=False,rate_sweep=False,
        NB0_historical=1.45460286107,NB0_same_environment=nb['AEE_frame_mean'],NB0_reused_from=str(PREVIOUS/'nb0_diverse10.json'),
        NB0_rerun_this_stage=False,NB0_environment_unchanged=True,fullnet_bittrue=False,valid825=False,
        numerical='same Q16 weight arrays asRTL decoded tofloat; originalTF32 CUDA consumers; no integer whole-network rounding claim',
        metric='same realpreds.2 summed overT,bilinear480x640 align_cornersFalse,originalvalidGT,frameequalmean',results={})
    def save(): (HERE/'diverse10.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        for name in ['parent_fp32']+list(fixtures):
            with torch.no_grad():
                if name=='parent_fp32':m.weight.copy_(original)
                else:m.weight.copy_(torch.from_numpy(np.load(fixtures[name])['weight_q16'].astype(np.float32)/65536).to(original))
            summary=evaluate_axis(args,net.model,net.current,names,name,progress_tag='CIN_RETIRE_AEE')
            rows=json.loads((args.output/(name+'_frames.json')).read_text())
            assert [r['file'] for r in rows]==[r['file'] for r in nb['rows']]
            assert [r['valid_pixels'] for r in rows]==[r['valid_pixels'] for r in nb['rows']]
            assert abs(np.mean([r['aee_sum']/r['valid_pixels'] for r in rows])-summary['AEE_frame_mean'])<1e-12
            report['results'][name]=dict(**summary,below_historical_NB0=summary['AEE_frame_mean']<1.45460286107,
                below_same_environment_NB0=summary['AEE_frame_mean']<nb['AEE_frame_mean'],fixture=str(fixtures.get(name,'')))
            save()
        report['complete']=True;save()
    except Exception as e:report['error']=repr(e);save();raise
    finally:
        with torch.no_grad():m.weight.copy_(original)
        net.close()
    print('SIX_ARM_COMPLETE',json.dumps(report),flush=True)
if __name__=='__main__':main()
