"""Same diverse10: fixed PED R24 bases on both existing fixed students.

Only continuous PED U/V signed16 matrices are replaced. Existing preview,
sn1/sn2/proj gates, fixed BN2, original dynamic proj BN, previous nonanchor
branch deletion, full coarse successor and metric remain unchanged. No train.
"""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
from ped_rebase_adapter import MODES,install,check_fixture
HERE=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--axes',nargs='+',choices=['ordinary','lifting_raw'],default=['ordinary','lifting_raw'])
    args=ap.parse_args();base=args.root;args.output=args.output or HERE/'results'
    args.output.mkdir(parents=True,exist_ok=True);top=args.output
    patch=base/'algorithm/patch_probe';res=patch/'residual_consumer_probe';chain=res/'projection_chain'
    lift=chain/'fast_temporal_recovery_lifting40';stage=lift/'schedule_compare_same_port'
    latent=patch/'factor_completion_20260909/latent_stage_train16'
    for directory in [stage,chain,res,latent,base/'algorithm',base/'algorithm/nrv_cost_probe']:
        sys.path.insert(0,str(directory))
    import torch
    import run_probe as probe
    from capture_inputs import save_json
    from flow_backward_probe import TrainableLatentPair,read_arrays
    from capture import BLOCK,SOURCE_SN,PROJECT
    from evaluate_branch_control import evaluate_axis,mask_nonanchors
    from train_shared_temporal_recovery import SharedTemporalControl
    from lifting_temporal_control import LiftingTemporalControl
    from fixed_temporal_coordinates import FixedTemporalForward
    from fixed_lifting_coordinates import FixedLiftingForward
    from spikingjelly.activation_based import functional
    old=json.loads((base/'open_fusion_execution/pruning/aee_results/run.json').read_text())
    names=[r['file'] for r in old['axes']['ordinary']['stages']['diverse10']['unpruned']['frames']]
    calibration_frame='zurich_city_09_a_0001.npy'
    args.split='diverse';system=probe.load_system(args)
    model,modules,_,current,sources,_,_,_=system
    probe.install_sources(system,sources);current['count_codes']=False
    calibration=torch.load(patch/'patch_train_calibration.pt',map_location='cpu',weights_only=False)
    for path,values in calibration.items():
        bn=modules[path];bn.track_running_stats=True
        bn.running_mean,bn.running_var=values['mean'].to(bn.weight),values['var'].to(bn.weight)
    model.eval();model.requires_grad_(False)
    flags=json.loads((chain/'affine_shared_temporal_control_diverse10/run.json').read_text())
    torch.backends.cuda.matmul.allow_tf32=bool(flags['TF32_matmul'])
    torch.backends.cudnn.allow_tf32=bool(flags['TF32_cudnn'])
    parent=latent/'flow_recovery64/preview_only/shared48_u8_vq5.npz'
    pair=TrainableLatentPair(read_arrays(parent),modules[SOURCE_SN].weight.device)
    pair.u.requires_grad_(False);pair.v.requires_grad_(False)
    conv1,sn2=modules[BLOCK+'.conv1.0'],modules[BLOCK+'.sn2.spiking_neuron']
    old_conv1,old_sn2=conv1.forward,sn2.forward
    conv1.forward,sn2.forward=pair.conv_forward,pair.neuron_forward
    anchor_masks={}
    def nonanchor(module,inputs,output):
        key=(tuple(output.shape[-2:]),output.device)
        if key not in anchor_masks:
            anchor=torch.zeros(output.shape[-2:],device=output.device,dtype=torch.bool);anchor[::2,::2]=True
            anchor_masks[key]=anchor
        return mask_nonanchors(output,anchor_masks[key])
    common_hook=modules[BLOCK+'.norm2'].register_forward_hook(nonanchor)
    common=(modules,read_arrays(res/'rank_control_parameters.npz'),read_arrays(chain/'rank32_diverse10/parameters.npz'))
    with np.load(HERE/'cpu_fixture.npz') as z:fixture={k:z[k] for k in z.files}
    report=dict(complete=False,scope=__doc__,training=False,full_valid825=False,new_X=False,
        changed=PROJECT+'.conv_res fixed-helper U_ped/V_ped only',
        fixed_contract='Actual saved signed16 U exponent16, V exponent15; signed48 dot, original two RNE/sat24 and same bias/sat24.',
        files=names,calibration_frame=calibration_frame,
        calibration_overlap=[n for n in names if n==calibration_frame],
        holdout_files=[n for n in names if n!=calibration_frame],
        modes=list(MODES),baseline='Original R32 re-evaluated for all diverse10 in this same batch, independently per student.',
        identity='Derived ordinary/lifting students; not frozen ep34 and not full decoder completion.',
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32,axes={})
    save_json(top/'run.json',report)
    try:
        for axis in args.axes:
            began=time.monotonic()
            if axis=='ordinary':
                identity='identity_permuted_base';student=chain/'temporal_structured_recovery/stage128x256'/f'{identity}.npz'
                controller=SharedTemporalControl(*common,fit={});controller.load_saved(identity,read_arrays(student))
                helper=FixedTemporalForward(controller,pair.temporal.theta)
            else:
                identity='fast_raw_diagonal';student=lift/'stage320'/f'{identity}.npz'
                init=json.loads((lift/'initialization.json').read_text())
                controller=LiftingTemporalControl(*common,source_fit=init['source_fit'],consumer_fits=init['consumer_fits'],basis_lifting=init['basis_lifting'])
                controller.load_saved(identity,read_arrays(student));helper=FixedLiftingForward(controller,pair.temporal.theta)
            original=helper.export_constants()
            with np.load(HERE.parent/(axis+'_rebase_parameters.npz')) as z:q={k:z[k] for k in z.files}
            with np.load(stage/'full_chain/capture'/axis/'parameters.npz') as z:captured={k:z[k] for k in z.files}
            for key in ('U_ped_q16','V_ped_q16','U_ped_exponent','V_ped_exponent','PED_bias_q24'):
                assert np.array_equal(original[key],captured[key]),(axis,key,'original parameter drift')
            ar=dict(student=str(student),identity=identity,modes={});report['axes'][axis]=ar
            try:
                ar['GPU_wrapper_exact']=check_fixture(helper,q,original,fixture,axis)
                print('PED_REBASE_GPU_WRAPPER_EXACT',axis,flush=True)
                save_json(top/'run.json',report)
                for mode in MODES:
                    install(helper,q,mode,original)
                    helper.frames.clear()
                    args.output=top/axis/mode;args.output.mkdir(parents=True,exist_ok=True)
                    with torch.no_grad():
                        summary=evaluate_axis(args,model,current,names,mode,progress_tag=f'PED_REBASE_AEE {axis}')
                    rows=json.loads((args.output/(mode+'_frames.json')).read_text())
                    assert len(rows)==len(helper.frames)==len(names)
                    ref=rows if mode=='original32' else ar['modes']['original32']['frames']
                    refmap={r['file']:r for r in ref}
                    paired=[dict(file=r['file'],baseline_AEE=refmap[r['file']]['AEE'],AEE=r['AEE'],delta_AEE=r['AEE']-refmap[r['file']]['AEE']) for r in rows]
                    held=[r for r in paired if r['file']!=calibration_frame]
                    gates=[{k:f[k] for k in ('source_gate','consumer_gate')} for f in helper.frames]
                    if mode!='original32':
                        assert gates==ar['modes']['original32']['gate_counts'],(axis,mode,'upstream gate counts changed')
                    mr=dict(summary=summary,frames=rows,paired=paired,
                        delta_frame_mean=float(np.mean([r['delta_AEE'] for r in paired])),
                        holdout_delta_frame_mean=float(np.mean([r['delta_AEE'] for r in held])),
                        holdout_frame_mean=float(np.mean([r['AEE'] for r in held])),
                        within_plus_0_005=float(np.mean([r['delta_AEE'] for r in paired]))<=.005,
                        gate_counts=gates,
                        PED_clip_counts=[{k:v for k,v in f['clip_counts'].items() if k.startswith('PED_')} for f in helper.frames])
                    if mode=='original32':
                        oldrows=old['axes'][axis]['stages']['diverse10']['unpruned']['frames']
                        oldmap={r['file']:r for r in oldrows}
                        mr['old_run_max_abs_AEE_difference']=max(abs(r['AEE']-oldmap[r['file']]['AEE']) for r in rows)
                    ar['modes'][mode]=mr
                    save_json(top/'run.json',report)
                    functional.reset_net(model);current.pop('flow',None)
                ar['complete']=True;ar['wall_seconds']=time.monotonic()-began
            finally:
                helper.restore();controller.restore();functional.reset_net(model)
                current.pop('flow',None);torch.cuda.empty_cache()
            save_json(top/'run.json',report)
        report['complete']=True;save_json(top/'run.json',report)
    finally:
        common_hook.remove();conv1.forward,sn2.forward=old_conv1,old_sn2
    print('PED_REBASE_DONE',json.dumps({a:{m:dict(AEE=r['summary']['AEE_frame_mean'],delta=r['delta_frame_mean']) for m,r in ar['modes'].items()} for a,ar in report['axes'].items()}),flush=True)


if __name__=='__main__':main()
