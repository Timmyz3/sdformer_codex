"""Actual R24+onepass BN combinations, diverse10 then automatic full valid825.

Existing numerical functions/parameters are reused. All new artifacts belong
to this directory. GPU elapsed time is never a hardware speed measurement.
"""
from pathlib import Path
import argparse
import csv
import gzip
import json
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def rows(path):
    return json.loads(path.read_text())


class Activity:
    def __init__(self,model,modules):
        self.active=False;self.handles=[model.register_forward_pre_hook(self.begin)]
        for name,module in modules.items():
            if type(module).__name__=='ATLIFTernaryPSN':
                self.handles.append(module.register_forward_hook(self.observe(name)))
    def start(self,names):
        self.names=names;self.index=-1;self.frames=[];self.layers={};self.active=True
    def begin(self,module,inputs):
        if not self.active:return
        self.index+=1
        self.frames.append(dict(file=self.names[self.index],calls=0,nonzero=0,elements=0))
    def observe(self,name):
        def hook(module,inputs,output):
            if not self.active:return
            values=dict(calls=1,nonzero=int(output.count_nonzero()),elements=output.numel())
            layer=self.layers.setdefault(name,dict(calls=0,nonzero=0,elements=0))
            for key,value in values.items():layer[key]+=value;self.frames[-1][key]+=value
        return hook
    def finish(self,helper_frames,bn_calls):
        self.active=False
        clips={};ranges={};gates={}
        for frame in helper_frames:
            for k,v in frame['clip_counts'].items():
                a=clips.setdefault(k,dict(elements=0,low=0,high=0))
                for sub in a:a[sub]+=v[sub]
            for k,v in frame['state_ranges'].items():
                a=ranges.setdefault(k,dict(integer_min=v['integer_min'],integer_max=v['integer_max']))
                a['integer_min']=min(a['integer_min'],v['integer_min']);a['integer_max']=max(a['integer_max'],v['integer_max'])
            for key in ('source_gate','consumer_gate'):
                if key in frame:
                    a=gates.setdefault(key,dict(nonzero=0,elements=0))
                    for sub in a:a[sub]+=frame[key][sub]
        return dict(frames=self.frames,ATLIF_layers=self.layers,
            ATLIF_hook_scope='Actual called ATLIFTernaryPSN module outputs through coarse exit, including dead-result calls; not all installed105.',
            fixed_helper_gate_totals=gates,clip_counts=clips,state_ranges=ranges,
            onepass_calls=len(bn_calls),onepass_min_variance=min((x['min_variance'] for x in bn_calls),default=None),
            actual_helper_frames=len(helper_frames),hardware_cycles=False)
    def restore(self):
        for h in self.handles:h.remove()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--axes',nargs='+',choices=['ordinary','lifting_raw'],default=['ordinary','lifting_raw'])
    ap.add_argument('--diverse-only',action='store_true')
    args=ap.parse_args();base=args.root;op=base/'open_fusion_execution'
    top=args.output or HERE/'combinations';top.mkdir(parents=True,exist_ok=True)
    patch=base/'algorithm/patch_probe';res=patch/'residual_consumer_probe';chain=res/'projection_chain'
    lift=chain/'fast_temporal_recovery_lifting40';stage=lift/'schedule_compare_same_port';full=stage/'full_chain'
    latent=patch/'factor_completion_20260909/latent_stage_train16'
    for directory in [full,stage,chain,res,latent,base/'algorithm',base/'algorithm/nrv_cost_probe',op/'new_interface_selection/aee_rebase',HERE/'onepass_math']:
        sys.path.insert(0,str(directory))
    import torch
    import run_probe as probe
    from numeric import Arithmetic,difference
    from ped_rebase_adapter import install,check_fixture
    from flow_backward_probe import TrainableLatentPair,read_arrays
    from capture import BLOCK,SOURCE_SN,PROJECT
    from evaluate_branch_control import evaluate_axis,mask_nonanchors
    from train_shared_temporal_recovery import SharedTemporalControl
    from lifting_temporal_control import LiftingTemporalControl
    from fixed_temporal_coordinates import FixedTemporalForward
    from fixed_lifting_coordinates import FixedLiftingForward
    from run_bn_probe import read_names
    from capture_full_producers import FullCapture
    from spikingjelly.activation_based import functional

    names=rows(base/'algorithm/samples.json')['valid']
    reference=rows(op/'new_interface_selection/aee_rebase/results/run.json')
    with (HERE/'source_nb0_valid825.csv').open() as f:nb0={r['file']:r for r in csv.DictReader(f)}
    baseline10=float(np.mean([float(nb0[n]['AEE']) for n in names]));baseline825=float(np.mean([float(r['AEE']) for r in nb0.values()]))
    args.split='diverse';system=probe.load_system(args)
    model,modules,_,current,sources,_,_,_=system
    probe.install_sources(system,sources);current['count_codes']=False
    calibration=torch.load(patch/'patch_train_calibration.pt',map_location='cpu',weights_only=False)
    for path,values in calibration.items():
        bn=modules[path];bn.track_running_stats=True
        bn.running_mean,bn.running_var=values['mean'].to(bn.weight),values['var'].to(bn.weight)
    model.eval();model.requires_grad_(False)
    flags=rows(chain/'affine_shared_temporal_control_diverse10/run.json')
    torch.backends.cuda.matmul.allow_tf32=bool(flags['TF32_matmul']);torch.backends.cudnn.allow_tf32=bool(flags['TF32_cudnn'])
    parent=latent/'flow_recovery64/preview_only/shared48_u8_vq5.npz'
    pair=TrainableLatentPair(read_arrays(parent),modules[SOURCE_SN].weight.device)
    pair.u.requires_grad_(False);pair.v.requires_grad_(False)
    conv1,sn2=modules[BLOCK+'.conv1.0'],modules[BLOCK+'.sn2.spiking_neuron']
    old_conv1,old_sn2=conv1.forward,sn2.forward;conv1.forward,sn2.forward=pair.conv_forward,pair.neuron_forward
    anchor_masks={}
    def nonanchor(module,inputs,output):
        key=(tuple(output.shape[-2:]),output.device)
        if key not in anchor_masks:
            mask=torch.zeros(output.shape[-2:],device=output.device,dtype=torch.bool);mask[::2,::2]=True;anchor_masks[key]=mask
        return mask_nonanchors(output,anchor_masks[key])
    common_hook=modules[BLOCK+'.norm2'].register_forward_hook(nonanchor)
    common=(modules,read_arrays(res/'rank_control_parameters.npz'),read_arrays(chain/'rank32_diverse10/parameters.npz'))
    math=Arithmetic();activity=Activity(model,modules)
    with np.load(op/'new_interface_selection/aee_rebase/cpu_fixture.npz') as z:fixture={k:z[k] for k in z.files}
    report=dict(complete=False,training=False,new_X=False,scope=__doc__,NB0_diverse10=baseline10,NB0_valid825=baseline825,
        files=names,policy='Strictly lower than same-population NB0; no +0.005 gate.',
        candidate_axes=dict(ordinary='original_ordered24 + onepass BN',lifting_raw='activation_whitened24 + onepass BN'),
        common_parent=str(parent),config=str(args.config),checkpoint=str(args.checkpoint),
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32,
        output_head='Real coarse preds.2 sum(T), bilinear480x640 align_corners=False; NB0 reference uses original final head.',axes={})
    save(top/'run.json',report)
    try:
        for axis in args.axes:
            begin=time.monotonic()
            if axis=='ordinary':
                identity='identity_permuted_base';student=chain/'temporal_structured_recovery/stage128x256'/f'{identity}.npz';mode='original_ordered24'
                controller=SharedTemporalControl(*common,fit={});controller.load_saved(identity,read_arrays(student));helper=FixedTemporalForward(controller,pair.temporal.theta)
            else:
                identity='fast_raw_diagonal';student=lift/'stage320'/f'{identity}.npz';mode='activation_whitened24'
                init=rows(lift/'initialization.json');controller=LiftingTemporalControl(*common,source_fit=init['source_fit'],consumer_fits=init['consumer_fits'],basis_lifting=init['basis_lifting'])
                controller.load_saved(identity,read_arrays(student));helper=FixedLiftingForward(controller,pair.temporal.theta)
            bn=modules[PROJECT+'.norm_layer'];old_bn=bn.forward
            ar=dict(complete=False,student=str(student),identity=identity,PED_mode=mode,stages={});report['axes'][axis]=ar
            original=helper.export_constants()
            with np.load(op/'new_interface_selection'/(axis+'_rebase_parameters.npz')) as z:q={k:z[k] for k in z.files}
            try:
                ar['R24_GPU_vs_CPU_fixtures']=check_fixture(helper,q,original,fixture,axis)
                print('COMBO_R24_FIXTURE_EXACT',axis,flush=True)
                capture_path=full/'capture_full_producers'/axis/'000_zurich_city_09_a_0001.npz'
                with np.load(capture_path) as z:inp=z['proj_bn_full_input_fp32'].transpose(0,2,3,1).reshape(-1,96).copy()
                with np.load(capture_path.parent/'live_parameters.npz') as z:gamma=z['proj_bn_gamma'];beta=z['proj_bn_beta'];eps=float(z['proj_bn_eps'])
                st=math.statistics(inp,gamma,beta,eps);gpu=torch.as_tensor(inp,device=pair.u.device)
                result=(gpu*torch.as_tensor(st[3],device=gpu.device)+torch.as_tensor(st[4],device=gpu.device)).cpu().numpy()
                with gzip.open(op/'default_bn/onepass/aee_check/engine_reference'/f'{axis}_output.f32.gz','rb') as f:gold=np.frombuffer(f.read(),np.float32).reshape(-1,96)
                ar['onepass_existing_Engine_agreement']=difference(result,gold)
                assert ar['onepass_existing_Engine_agreement']['bit_differences']==0
                del gpu,result,gold,inp
                for control in ('original32',mode):
                    install(helper,q,control,original);args.output=top/axis/('check_'+control);args.output.mkdir(parents=True,exist_ok=True)
                    with torch.no_grad():s=evaluate_axis(args,model,current,names[:1],control,progress_tag=f'COMBO_CONTROL {axis}')
                    value=rows(args.output/(control+'_frames.json'))[0]
                    expected=reference['axes'][axis]['modes'][control]['frames'][0]
                    assert value['AEE']==expected['AEE'] and value['valid_pixels']==expected['valid_pixels'],(axis,control,value,expected)
                    ar['check_'+control]=dict(AEE=value['AEE'],old_AEE=expected['AEE'],exact=True)
                install(helper,q,mode,original)
                axis_dir=top/axis;axis_dir.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(axis_dir/'deployed_constants.npz',**helper.export_constants())
                np.savez_compressed(axis_dir/'student_parameters.npz',**read_arrays(student))
                gamma=bn.weight.detach().cpu().numpy().copy();beta=bn.bias.detach().cpu().numpy().copy();calls=[];latest={}
                def forward(x):
                    assert x.dtype==torch.float32 and tuple(x.shape)==(10,96,120,160)
                    payload=x.detach().permute(0,2,3,1).contiguous().cpu().numpy()
                    st=math.statistics(payload,gamma,beta,bn.eps);latest['stats']=st
                    out=x*torch.as_tensor(st[3],device=x.device).reshape(1,96,1,1)
                    out=out+torch.as_tensor(st[4],device=x.device).reshape(1,96,1,1)
                    call=dict(domain=192000,min_variance=float(st[1].min()))
                    if not ar.get('runtime_combo_BN_exact'):
                        check=difference(out.detach().permute(0,2,3,1).contiguous().cpu().numpy().reshape(-1,96),math.output(payload,st))
                        assert check['bit_differences']==0;ar['runtime_combo_BN_exact']=check
                    calls.append(call);return out
                bn.forward=forward
                np.savez_compressed(axis_dir/'BN_parameters.npz',gamma=gamma,beta=beta,eps=np.asarray(bn.eps))
                # One actual first-frame full-producer archive, kept in ignored captures/.
                class ComboCapture(FullCapture):
                    def output_values(self,name,final=False):
                        old=super().output_values(name,final)
                        def hook(module,inputs,output):
                            old(module,inputs,output)
                            if name=='proj_norm_fp32':
                                for key in ('mean','var'):
                                    source='proj_bn_actual_domain_'+key
                                    if source in self.arrays:self.arrays['proj_bn_torch_diagnostic_'+key]=self.arrays.pop(source)
                                self.arrays['proj_bn_onepass_statistics']=latest['stats'].copy()
                                self.arrays['proj_bn_function']=np.asarray('onepass paired256, original rsqrt seed+3Newton, separate affine MUL/ADD')
                        return hook
                capdir=HERE/'captures'/axis;capdir.mkdir(parents=True,exist_ok=True)
                observer=ComboCapture(model,modules,helper,pair.temporal.theta,names[:1],capdir,True)
                try:
                    args.output=top/axis/'capture_check';args.output.mkdir(parents=True,exist_ok=True)
                    with torch.no_grad():ar['capture_check']=evaluate_axis(args,model,current,names[:1],'combo',progress_tag=f'COMBO_CAPTURE {axis}')
                    ar['capture']=dict(directory=str(capdir),rows=observer.rows,large_files_ignored_by_git=True,
                        geometry='Existing corner output(0,0), interior(60,80), plus full T,C,H,W producers; identical original windows.')
                finally:observer.restore()
                for split,selected in [('diverse10',names),('valid825',read_names(args.data,'valid'))]:
                    if split=='valid825':
                        if args.diverse_only:break
                        if not ar['stages']['diverse10']['better_than_NB0']:break
                    assert (len(selected)==10 if split=='diverse10' else len(selected)==825 and set(selected)==set(nb0))
                    helper.frames.clear();calls.clear();activity.start(selected)
                    args.split='diverse' if split=='diverse10' else 'valid';args.output=axis_dir/split;args.output.mkdir(parents=True,exist_ok=True)
                    with torch.no_grad():s=evaluate_axis(args,model,current,selected,'combo',progress_tag=f'COMBO_{split} {axis}')
                    measured=rows(args.output/'combo_frames.json')
                    assert len(helper.frames)==len(measured)==len(selected)
                    assert all(int(r['valid_pixels'])==int(float(nb0[r['file']]['valid_pixels'])) for r in measured)
                    act=activity.finish(helper.frames,calls);save(args.output/'activity_summary.json',act)
                    ref_mean=baseline10 if split=='diverse10' else baseline825
                    paired=[dict(file=r['file'],valid_pixels=r['valid_pixels'],AEE=r['AEE'],NB0_AEE=float(nb0[r['file']]['AEE']),delta_AEE=r['AEE']-float(nb0[r['file']]['AEE'])) for r in measured]
                    save(args.output/'paired_NB0.json',paired)
                    sr=dict(summary=s,NB0_AEE=ref_mean,delta_NB0=s['AEE_frame_mean']-ref_mean,better_than_NB0=s['AEE_frame_mean']<ref_mean,
                        same_frame_set=True,same_per_frame_valid_pixels=True,activity='activity_summary.json',parameters='../deployed_constants.npz')
                    if split=='diverse10':
                        sr['single_item_controls']={m:reference['axes'][axis]['modes'][m]['summary']['AEE_frame_mean'] for m in ['original32',mode]}
                        bn_old=rows(HERE/'single_item_controls.json')
                        sr['single_item_controls']['R32_onepass']=bn_old['axes'][axis]['methods']['onepass']['frame_mean']
                        sr['holdout9_AEE']=float(np.mean([r['AEE'] for r in measured if r['file']!=names[0]]))
                    ar['stages'][split]=sr;save(top/'run.json',report)
                    print('COMBO_STAGE_DONE',axis,split,json.dumps(sr),flush=True)
                ar['complete']=True;ar['wall_seconds']=time.monotonic()-begin;save(top/'run.json',report)
            finally:
                activity.active=False;bn.forward=old_bn;helper.restore();controller.restore();functional.reset_net(model)
                current.pop('flow',None);torch.cuda.empty_cache()
        report['complete']=True;save(top/'run.json',report)
    finally:
        activity.restore();common_hook.remove();conv1.forward,sn2.forward=old_conv1,old_sn2
    print('COMBINATIONS_DONE',flush=True)


if __name__=='__main__':main()
