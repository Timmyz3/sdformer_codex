"""No-training diverse10 accuracy of the common one-pass proj BN function.

Only PROJECT.norm_layer changes. Fixed ordinary/lifting students, raw I24,
PED and later network remain unchanged. This is not valid825 or a new X.
"""
from pathlib import Path
import argparse,json,sys,time,gzip
import numpy as np
from numeric import Arithmetic,difference
HERE=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--axes',nargs='+',default=['ordinary','lifting_raw'])
    ap.add_argument('--centered',action='store_true',help='Evaluate the existing four-stripe centered Engine function.')
    ap.add_argument('--capture-flow',action='store_true',help='Save actual resized GPU flow for a paired check, outside the repository.')
    ap.add_argument('--original-cuda',action='store_true',help='Re-evaluate the unchanged original BN control into a separate directory.')
    args=ap.parse_args();base=args.root;args.output=args.output or HERE/('baseline_recheck' if args.original_cuda else 'centered_results' if args.centered else 'results')
    result_key='original_cuda_bn' if args.original_cuda else 'centered_engine' if args.centered else 'onepass'
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
    math=Arithmetic(centered=args.centered)
    old=json.loads((base/'open_fusion_execution/pruning/aee_results/run.json').read_text())
    names=[r['file'] for r in old['axes']['ordinary']['stages']['diverse10']['unpruned']['frames']]
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
    report=dict(complete=False,scope=__doc__,training=False,full_valid825=False,new_X=False,
        changed_module=(None if args.original_cuda else PROJECT+'.norm_layer'),evaluated_function=result_key,unchanged='Student weights, source/sn2 gates, raw I24, PED, original coarse exit, metric and precision flags.',
        baseline_source=str(base/'open_fusion_execution/pruning/aee_results/run.json'),files=names,axes={},
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32,
        statistic_function=('Original unmodified CUDA BatchNorm2d; separate helper sanity does not describe this method.' if args.original_cuda else 'FP32 four stripes; centered two-pass paired256 tree; original seed+3Newton; affine separate MUL then ADD.' if args.centered else 'FP32 two sum plus two FMA-square stripes; paired256 tree; variance E[x²]−fl(mean²); original seed+3Newton; affine separate MUL then ADD.'),
        numerical_claim=('Unchanged CUDA BN control; independent helper sanity is separate.' if args.original_cuda else 'Changed BN function; exact implementation agreement is checked independently, not identity to CUDA BN.'),
        current_admission_policy='Former +0.005 is historical only. Active early accuracy comparison is verified same-diverse10 SDformerFlow NB0 fullres ep29 at 1.45460286107; no candidate valid825 claim.')
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
            bn=modules[PROJECT+'.norm_layer'];original_bn=bn.forward
            axis_report=dict(student=str(student),identity=identity,calls=[])
            report['axes'][axis]=axis_report
            try:
                # Before network evaluation, compare actual CUDA elementwise output
                # directly with the complete local Engine result on its saved frame.
                capture=stage/'full_chain/capture_full_producers'/axis/'000_zurich_city_09_a_0001.npz'
                with np.load(capture) as z:
                    inp=z['proj_bn_full_input_fp32'].transpose(0,2,3,1).reshape(-1,96).copy()
                with np.load(capture.parent/'live_parameters.npz') as z:
                    gamma=z['proj_bn_gamma'];beta=z['proj_bn_beta'];eps=float(z['proj_bn_eps'])
                stats=math.statistics(inp,gamma,beta,eps)
                gpu=torch.as_tensor(inp,device=pair.u.device)
                scaled=gpu*torch.as_tensor(stats[3],device=gpu.device)
                result=(scaled+torch.as_tensor(stats[4],device=gpu.device)).cpu().numpy()
                with gzip.open(HERE/'engine_reference'/f'{"centered_" if args.centered else ""}{axis}_output.f32.gz','rb') as f:
                    reference=np.frombuffer(f.read(),np.float32).reshape(-1,96)
                exact=difference(result,reference);assert exact['bit_differences']==0,exact
                axis_report['complete_frame_GPU_vs_Engine']=None if args.original_cuda else exact
                if args.original_cuda:axis_report['independent_onepass_helper_check_not_CUDA_BN']=exact
                del gpu,scaled,result,reference,inp
                print('ONEPASS_GPU_ENGINE_EXACT',axis,exact,flush=True)
                # One baseline frame detects configuration drift without rerunning
                # all historical baseline frames when it reproduces exactly.
                args.output=top/axis/'baseline_first';args.output.mkdir(parents=True,exist_ok=True)
                with torch.no_grad():
                    axis_report['baseline_first']=evaluate_axis(args,model,current,names[:1],'baseline_first',progress_tag=f'ONEPASS_BASE {axis}')
                actual=json.loads((args.output/'baseline_first_frames.json').read_text())[0]
                oldrows=old['axes'][axis]['stages']['diverse10']['unpruned']['frames']
                ref=next(r for r in oldrows if r['file']==actual['file'])
                axis_report['baseline_first_delta_AEE']=actual['AEE']-ref['AEE']
                assert actual['AEE']==ref['AEE'],('baseline configuration differs',axis,actual,ref)
                gamma=bn.weight.detach().cpu().numpy().copy();beta=bn.bias.detach().cpu().numpy().copy()
                def forward(x):
                    assert x.dtype==torch.float32 and tuple(x.shape)==(10,96,120,160),tuple(x.shape)
                    payload=x.detach().permute(0,2,3,1).contiguous().cpu().numpy()
                    stats=math.statistics(payload,gamma,beta,bn.eps)
                    scale=torch.as_tensor(stats[3],device=x.device).reshape(1,96,1,1)
                    offset=torch.as_tensor(stats[4],device=x.device).reshape(1,96,1,1)
                    multiplied=x*scale
                    out=multiplied+offset
                    row=dict(domain=192000,min_variance=float(stats[1].min()))
                    if not axis_report['calls']:
                        cpu=math.output(payload,stats)
                        check=difference(out.detach().permute(0,2,3,1).contiguous().cpu().numpy().reshape(-1,96),cpu)
                        row['runtime_GPU_vs_CPU_exact']=check
                        assert check['bit_differences']==0,check
                    axis_report['calls'].append(row)
                    return out
                bn.forward=original_bn if args.original_cuda else forward
                args.output=top/axis/result_key;args.output.mkdir(parents=True,exist_ok=True)
                flow_capture=[]
                def observe_flow(module,inputs,output):
                    coarse=output.detach().sum(0)
                    final=torch.nn.functional.interpolate(coarse,(480,640),mode='bilinear',align_corners=False)
                    flow_capture.append(final.cpu().numpy())
                flow_hook=(modules['sttmultires_unet.preds.2'].register_forward_hook(observe_flow,prepend=True) if args.capture_flow else None)
                with torch.no_grad():
                    axis_report[result_key]=evaluate_axis(args,model,current,names,result_key,progress_tag=f'ONEPASS_AEE {axis}')
                if flow_hook is not None:
                    flow_hook.remove()
                    flow_dir=Path('/tmp/onepass_aee_flow')/result_key;flow_dir.mkdir(parents=True,exist_ok=True)
                    assert len(flow_capture)==len(names)
                    flow_path=flow_dir/f'{axis}.npz'
                    np.savez_compressed(flow_path,files=np.asarray(names),predictions=np.stack(flow_capture))
                    axis_report['actual_flow_capture']=dict(path=str(flow_path),shape=list(np.stack(flow_capture).shape),domain='Actual GPU bilinear 480x640 prediction; all pixels, before valid-GT selection.')
                rows=json.loads((args.output/(result_key+'_frames.json')).read_text())
                reference={r['file']:r for r in oldrows}
                axis_report['paired']=[dict(file=r['file'],baseline_AEE=reference[r['file']]['AEE'],AEE=r['AEE'],delta_AEE=r['AEE']-reference[r['file']]['AEE']) for r in rows]
                axis_report['baseline_frame_mean']=float(np.mean([r['AEE'] for r in oldrows]))
                axis_report['delta_frame_mean']=axis_report[result_key]['AEE_frame_mean']-axis_report['baseline_frame_mean']
                axis_report['max_abs_frame_delta']=max(abs(r['delta_AEE']) for r in axis_report['paired'])
                axis_report['legacy_within_plus_0_005']=axis_report['delta_frame_mean']<=.005
                axis_report['complete']=True;axis_report['wall_seconds']=time.monotonic()-began
            finally:
                bn.forward=original_bn;helper.restore();controller.restore();functional.reset_net(model)
                current.pop('flow',None);torch.cuda.empty_cache()
            save_json(top/'run.json',report)
        report['complete']=True;save_json(top/'run.json',report)
    finally:
        common_hook.remove();conv1.forward,sn2.forward=old_conv1,old_sn2
    print('ONEPASS_DONE',json.dumps({a:dict(AEE=v[result_key]['AEE_frame_mean'],delta=v['delta_frame_mean']) for a,v in report['axes'].items()}),flush=True)

if __name__=='__main__':main()
