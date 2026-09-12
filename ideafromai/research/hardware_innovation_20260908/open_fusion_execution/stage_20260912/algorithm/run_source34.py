"""One ordinary contiguous3/3/4 source control on the actual R24+onepass parent."""
from pathlib import Path
import argparse
import csv
import sys
import numpy as np
from run_combinations import Activity, save, rows

HERE=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    args=ap.parse_args();base=args.root;op=base/'open_fusion_execution';top=HERE/'source34/aee'
    assert rows(HERE/'weight_controls/aee/run.json')['complete']
    combo=rows(HERE/'combinations/run.json');assert combo['complete']
    patch=base/'algorithm/patch_probe';res=patch/'residual_consumer_probe';chain=res/'projection_chain'
    latent=patch/'factor_completion_20260909/latent_stage_train16'
    for p in [chain,res,latent,base/'algorithm',base/'algorithm/nrv_cost_probe',op/'new_interface_selection/aee_rebase',HERE/'onepass_math',HERE/'source34/package']:
        sys.path.insert(0,str(p))
    import torch
    import run_probe as probe
    from numeric import Arithmetic
    from ped_rebase_adapter import install as install_R24
    from source34_adapter import install as install_source34
    from flow_backward_probe import TrainableLatentPair,read_arrays
    from capture import BLOCK,SOURCE_SN,PROJECT
    from evaluate_branch_control import evaluate_axis,mask_nonanchors
    from train_shared_temporal_recovery import SharedTemporalControl
    from fixed_temporal_coordinates import FixedTemporalForward
    from spikingjelly.activation_based import functional
    names=rows(base/'algorithm/samples.json')['valid']
    with (HERE/'source_nb0_valid825.csv').open() as f:nb0={r['file']:r for r in csv.DictReader(f)}
    args.split='diverse';system=probe.load_system(args);model,modules,_,current,sources,_,_,_=system
    probe.install_sources(system,sources);current['count_codes']=False
    calibration=torch.load(patch/'patch_train_calibration.pt',map_location='cpu',weights_only=False)
    for path,values in calibration.items():
        bn=modules[path];bn.track_running_stats=True
        bn.running_mean,bn.running_var=values['mean'].to(bn.weight),values['var'].to(bn.weight)
    model.eval();model.requires_grad_(False)
    flags=rows(chain/'affine_shared_temporal_control_diverse10/run.json')
    torch.backends.cuda.matmul.allow_tf32=bool(flags['TF32_matmul']);torch.backends.cudnn.allow_tf32=bool(flags['TF32_cudnn'])
    pair=TrainableLatentPair(read_arrays(latent/'flow_recovery64/preview_only/shared48_u8_vq5.npz'),modules[SOURCE_SN].weight.device)
    pair.u.requires_grad_(False);pair.v.requires_grad_(False)
    conv1,sn2=modules[BLOCK+'.conv1.0'],modules[BLOCK+'.sn2.spiking_neuron']
    old_conv1,old_sn2=conv1.forward,sn2.forward;conv1.forward,sn2.forward=pair.conv_forward,pair.neuron_forward
    masks={}
    def nonanchor(module,inputs,output):
        key=(tuple(output.shape[-2:]),output.device)
        if key not in masks:
            m=torch.zeros(output.shape[-2:],device=output.device,dtype=torch.bool);m[::2,::2]=True;masks[key]=m
        return mask_nonanchors(output,masks[key])
    hook=modules[BLOCK+'.norm2'].register_forward_hook(nonanchor)
    student=chain/'temporal_structured_recovery/stage128x256/identity_permuted_base.npz'
    controller=SharedTemporalControl(modules,read_arrays(res/'rank_control_parameters.npz'),read_arrays(chain/'rank32_diverse10/parameters.npz'),fit={})
    controller.load_saved('identity_permuted_base',read_arrays(student));helper=FixedTemporalForward(controller,pair.temporal.theta)
    with np.load(op/'new_interface_selection/ordinary_rebase_parameters.npz') as z:package={k:z[k] for k in z.files}
    install_R24(helper,package,'original_ordered24',helper.export_constants())
    bn=modules[PROJECT+'.norm_layer'];old_bn=bn.forward
    gamma=bn.weight.detach().cpu().numpy().copy();beta=bn.bias.detach().cpu().numpy().copy();math=Arithmetic();calls=[]
    def onepass(x):
        p=x.detach().permute(0,2,3,1).contiguous().cpu().numpy();st=math.statistics(p,gamma,beta,bn.eps)
        y=x*torch.as_tensor(st[3],device=x.device).reshape(1,96,1,1)
        y=y+torch.as_tensor(st[4],device=x.device).reshape(1,96,1,1)
        calls.append(dict(min_variance=float(st[1].min())));return y
    bn.forward=onepass;activity=Activity(model,modules)
    report=dict(complete=False,training=False,full_valid825=False,new_X=False,files=names,
        parent='ordinary original_ordered24 + onepass BN',
        source_package='stage_20260912/hardware/structured_source_control',
        intervention='Only As_q16: fixed contiguous3/3/4 support, 34 original coefficients; original exponent15, thresholds, source theta and every consumer retained.',
        policy='Matched NB0 only. One new fixed layout, no training, scan or automatic825.')
    save(top/'run.json',report)
    try:
        args.output=top/'baseline_check';args.output.mkdir(parents=True,exist_ok=True)
        with torch.no_grad():evaluate_axis(args,model,current,names[:1],'parent',progress_tag='SOURCE34_BASE')
        observed=rows(args.output/'parent_frames.json')[0]
        old=rows(HERE/'combinations/ordinary/diverse10/combo_frames.json')
        assert observed['AEE']==old[0]['AEE'];report['baseline_first_exact']=True
        report['baseline_diverse10']=float(np.mean([r['AEE'] for r in old]))
        report['adapter']=install_source34(helper,HERE/'source34/package/deployed_constants.npz')
        with np.load(HERE/'source34/package/fixture.npz') as fixture:
            checks=[]
            for window in ['corner','interior']:
                helper.frame=dict(clip_counts={},state_ranges={},accumulator_ranges={})
                i=torch.as_tensor(fixture[window+'_I24'],device=helper.device,dtype=torch.float64)
                q=helper.time_dot('As',i,'source34_fixture_Q24')
                gate=helper.compare('source',q)
                assert np.array_equal(q.cpu().numpy(),fixture[window+'_Q24'])
                assert np.array_equal(gate.cpu().numpy(),fixture[window+'_gate'])
                checks.append(dict(window=window,Q24_values=q.numel(),gate_values=gate.numel(),differences=0))
        report['actual_helper_fixtures']=checks;helper.ready=False
        print('SOURCE34_FIXTURES_EXACT',flush=True)
        np.savez_compressed(top/'deployed_constants.npz',**helper.export_constants())
        helper.frames.clear();calls.clear();activity.start(names)
        args.output=top/'diverse10';args.output.mkdir(parents=True,exist_ok=True)
        with torch.no_grad():summary=evaluate_axis(args,model,current,names,'source34',progress_tag='SOURCE34_AEE')
        measured=rows(args.output/'source34_frames.json')
        assert all(int(r['valid_pixels'])==int(float(nb0[r['file']]['valid_pixels'])) for r in measured)
        save(args.output/'activity_summary.json',activity.finish(helper.frames,calls))
        baseline=float(np.mean([float(nb0[n]['AEE']) for n in names]))
        report.update(complete=True,summary=summary,NB0_AEE=baseline,
            better_than_NB0=summary['AEE_frame_mean']<baseline,
            delta_NB0=summary['AEE_frame_mean']-baseline,
            delta_parent=summary['AEE_frame_mean']-report['baseline_diverse10'],
            holdout9_AEE=float(np.mean([r['AEE'] for r in measured if r['file']!=names[0]])))
        save(top/'run.json',report)
    finally:
        activity.restore();bn.forward=old_bn;helper.restore();controller.restore();functional.reset_net(model)
        hook.remove();conv1.forward,sn2.forward=old_conv1,old_sn2
    print('SOURCE34_COMPLETE',flush=True)


if __name__=='__main__':main()
