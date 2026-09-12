"""Bounded three-representation AEE probe, after the R24+onepass825 queue.

Only continuous PED input changes. Its actual same-producer projection gate
is reproduced from the same fixed comparison and checked against the emitted
gate; the gate consumer and its different3x3 weights are never substituted.
"""
from pathlib import Path
import argparse,csv,json,sys,time
import numpy as np
from run_combinations import Activity,save,rows
HERE=Path(__file__).resolve().parent
MODES=('i24_q8_s11','Dg_q8_s11','Dg_only')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--axes',nargs='+',choices=['ordinary','lifting_raw'],default=['ordinary','lifting_raw'])
    args=ap.parse_args();base=args.root;op=base/'open_fusion_execution';top=HERE/'representations/aee';top.mkdir(parents=True,exist_ok=True)
    combo=rows(HERE/'combinations/run.json')
    assert combo['complete'],'The complete combination GPU queue must finish first'
    patch=base/'algorithm/patch_probe';res=patch/'residual_consumer_probe';chain=res/'projection_chain';lift=chain/'fast_temporal_recovery_lifting40';stage=lift/'schedule_compare_same_port'
    latent=patch/'factor_completion_20260909/latent_stage_train16'
    for d in [stage,chain,res,latent,base/'algorithm',base/'algorithm/nrv_cost_probe',op/'new_interface_selection/aee_rebase',HERE/'onepass_math']:
        sys.path.insert(0,str(d))
    import torch
    import run_probe as probe
    from numeric import Arithmetic,difference
    from ped_rebase_adapter import install
    from flow_backward_probe import TrainableLatentPair,read_arrays
    from capture import BLOCK,SOURCE_SN,PROJECT,CONSUMER_SN
    from evaluate_branch_control import evaluate_axis,mask_nonanchors
    from train_shared_temporal_recovery import SharedTemporalControl
    from lifting_temporal_control import LiftingTemporalControl
    from fixed_temporal_coordinates import FixedTemporalForward
    from fixed_lifting_coordinates import FixedLiftingForward
    from spikingjelly.activation_based import functional
    names=rows(base/'algorithm/samples.json')['valid']
    with (HERE/'source_nb0_valid825.csv').open() as f:nb0={r['file']:r for r in csv.DictReader(f)}
    args.split='diverse';system=probe.load_system(args);model,modules,_,current,sources,_,_,_=system
    probe.install_sources(system,sources);current['count_codes']=False
    calibration=torch.load(patch/'patch_train_calibration.pt',map_location='cpu',weights_only=False)
    for path,values in calibration.items():
        b=modules[path];b.track_running_stats=True;b.running_mean,b.running_var=values['mean'].to(b.weight),values['var'].to(b.weight)
    model.eval();model.requires_grad_(False)
    flags=rows(chain/'affine_shared_temporal_control_diverse10/run.json')
    torch.backends.cuda.matmul.allow_tf32=bool(flags['TF32_matmul']);torch.backends.cudnn.allow_tf32=bool(flags['TF32_cudnn'])
    pair=TrainableLatentPair(read_arrays(latent/'flow_recovery64/preview_only/shared48_u8_vq5.npz'),modules[SOURCE_SN].weight.device)
    pair.u.requires_grad_(False);pair.v.requires_grad_(False)
    conv1,sn2=modules[BLOCK+'.conv1.0'],modules[BLOCK+'.sn2.spiking_neuron'];old_conv1,old_sn2=conv1.forward,sn2.forward
    conv1.forward,sn2.forward=pair.conv_forward,pair.neuron_forward
    masks={}
    def nonanchor(module,inputs,output):
        key=(tuple(output.shape[-2:]),output.device)
        if key not in masks:
            m=torch.zeros(output.shape[-2:],device=output.device,dtype=torch.bool);m[::2,::2]=True;masks[key]=m
        return mask_nonanchors(output,masks[key])
    common_hook=modules[BLOCK+'.norm2'].register_forward_hook(nonanchor)
    common=(modules,read_arrays(res/'rank_control_parameters.npz'),read_arrays(chain/'rank32_diverse10/parameters.npz'))
    arithmetic=Arithmetic();activity=Activity(model,modules)
    report=dict(complete=False,training=False,full_valid825=False,new_X=False,files=names,modes=list(MODES),axes={},
        policy='Matched NB0 AEE only; existing fixedR24+onepass is the ordinary quantization/control baseline.',
        scope=__doc__,calibration_frame=names[0],decoder_calibration='Existing256 anchors in this old valid frame; not a training-only fit.',
        gate_execution='Software probe computes the exact same producer predicate before PED and checks the later actual gate. Hardware must explicitly retime/forward it or pay duplicate work; no free availability or cycle saving is assumed.',
        numeric='Q8 RNE/signed8 saturation with step2048 in I24 units; reconstruct/saturate24; original actual R24 U/RNE/V/RNE/bias preserved.',
        no_hardware_acceleration_claim=True)
    save(top/'run.json',report)
    try:
        for axis in args.axes:
            if axis=='ordinary':
                identity='identity_permuted_base';student=chain/'temporal_structured_recovery/stage128x256'/f'{identity}.npz';mode24='original_ordered24'
                controller=SharedTemporalControl(*common,fit={});controller.load_saved(identity,read_arrays(student));helper=FixedTemporalForward(controller,pair.temporal.theta)
            else:
                identity='fast_raw_diagonal';student=lift/'stage320'/f'{identity}.npz';mode24='activation_whitened24'
                init=rows(lift/'initialization.json');controller=LiftingTemporalControl(*common,source_fit=init['source_fit'],consumer_fits=init['consumer_fits'],basis_lifting=init['basis_lifting'])
                controller.load_saved(identity,read_arrays(student));helper=FixedLiftingForward(controller,pair.temporal.theta)
            with np.load(op/'new_interface_selection'/(axis+'_rebase_parameters.npz')) as z:q={k:z[k] for k in z.files}
            install(helper,q,mode24,helper.export_constants())
            with np.load(HERE/'representations'/(axis+'_decoder.npz')) as z:Dnp=z['D_integer'];cnp=z['c_integer']
            D=torch.as_tensor(Dnp,device=helper.device,dtype=torch.float64);c=torch.as_tensor(cnp,device=helper.device,dtype=torch.float64)
            permutation=helper.permutation if axis=='ordinary' else helper.consumer_permutation
            bn=modules[PROJECT+'.norm_layer'];old_bn=bn.forward;gamma=bn.weight.detach().cpu().numpy().copy();beta=bn.bias.detach().cpu().numpy().copy();bn_calls=[]
            def bn_forward(x):
                p=x.detach().permute(0,2,3,1).contiguous().cpu().numpy();st=arithmetic.statistics(p,gamma,beta,bn.eps)
                out=x*torch.as_tensor(st[3],device=x.device).reshape(1,96,1,1);out=out+torch.as_tensor(st[4],device=x.device).reshape(1,96,1,1)
                bn_calls.append(dict(min_variance=float(st[1].min())));return out
            bn.forward=bn_forward
            old_channel=helper.channel_dot;active={'mode':None,'rows':[],'gate':None};gate_hook=None
            ar=dict(complete=False,parent_combo=str(HERE/'combinations'/axis),PED_mode=mode24,modes={});report['axes'][axis]=ar
            def replace_source(key,x,name):
                if key!='U_ped' or active['mode'] is None:return old_channel(key,x,name)
                g=helper.compare('consumer',x.index_select(0,permutation))
                active['gate']=g
                b=(D@g.double().reshape(10,-1)+c[:,None]).reshape_as(x)
                residual=x if active['mode']=='i24_q8_s11' else x-b
                rounded=torch.round(residual/2048)
                quant=torch.zeros_like(x) if active['mode']=='Dg_only' else rounded.clamp(-128,127)
                prediction=(torch.zeros_like(b) if active['mode']=='i24_q8_s11' else b)+quant*2048
                estimate=helper.write24('PED_reconstructed_source24',prediction)
                groups=quant.reshape(10,96,120,80,2)
                active['rows'].append(dict(elements=x.numel(),quant_zero=int((quant==0).sum()),
                    quant_clipped=(0 if active['mode']=='Dg_only' else int(((rounded< -128)|(rounded>127)).sum())),
                    empty_P2_T10_source_words=int((~groups.ne(0).any(dim=4).any(dim=0)).sum()),
                    P2_T10_source_words=96*120*80,input_physical_MAE=float((estimate-x).abs().mean())/(1<<14),
                    same_actual_proj_gate=False))
                return old_channel(key,estimate,name)
            def check_emitted_gate(module,inputs,output):
                if active['mode'] is None:return
                actual=output[:,0,:,::2,::2].ne(0)
                assert torch.equal(active['gate'],actual),'Dg used a different gate producer or time order'
                active['rows'][-1]['same_actual_proj_gate']=True;active['gate']=None
            helper.channel_dot=replace_source;gate_hook=modules[CONSUMER_SN].register_forward_hook(check_emitted_gate)
            try:
                args.output=top/axis/'baseline_check';args.output.mkdir(parents=True,exist_ok=True)
                with torch.no_grad():evaluate_axis(args,model,current,names[:1],'baseline',progress_tag=f'REPR_BASE {axis}')
                old=rows(HERE/'combinations'/axis/'diverse10/combo_frames.json')
                observed=rows(args.output/'baseline_frames.json')[0]
                assert observed['AEE']==old[0]['AEE'];ar['baseline_first_exact']=True
                ar['baseline_diverse10']=float(np.mean([r['AEE'] for r in old]))
                # Independent CPU signed-integer arithmetic for all three fixed
                # representations on the actual saved corner values.
                with np.load(HERE/'captures'/axis/'000_zurich_city_09_a_0001.npz') as z:
                    xn=z['full_updated_I24'][:,:,:8:2,:8:2].astype(np.int64)
                    wn=z['full_proj_words'][:,:8:2,:8:2]
                gn=((wn[None]>>np.arange(10)[:,None,None,None])&1).astype(np.int64)
                xt=torch.as_tensor(xn,device=helper.device,dtype=torch.float64)
                gt=helper.compare('consumer',xt.index_select(0,permutation))
                assert np.array_equal(gt.cpu().numpy(),gn)
                ar['same_producer_fixture_gate_exact']=True
                def rne(n,shift):
                    if shift:
                        divisor=1<<shift;q,r=np.divmod(n,divisor);q+=((2*r>divisor)|((2*r==divisor)&((q&1)!=0)));n=q
                    return np.clip(n,-(1<<23),(1<<23)-1)
                original=helper.export_constants();un=original['U_ped_q16'].astype(np.int64);vn=original['V_ped_q16'].astype(np.int64);bias=original['PED_bias_q24']
                fixture=[]
                for mode in MODES:
                    b=(Dnp@gn.reshape(10,-1)+cnp[:,None]).reshape(xn.shape);r=xn if mode=='i24_q8_s11' else xn-b
                    floor,rem=np.divmod(r,2048);floor+=((2*rem>2048)|((2*rem==2048)&((floor&1)!=0)))
                    quant=np.zeros_like(xn) if mode=='Dg_only' else np.clip(floor,-128,127)
                    estimate=rne((np.zeros_like(b) if mode=='i24_q8_s11' else b)+quant*2048,0)
                    flat=estimate.transpose(1,0,2,3).reshape(96,-1);z=rne(un@flat,16);pred=rne(rne(vn@z,15)+bias[:,None],0)
                    helper.frame=dict(clip_counts={},state_ranges={},accumulator_ranges={})
                    bt=(D@gt.double().reshape(10,-1)+c[:,None]).reshape_as(xt);rt=xt if mode=='i24_q8_s11' else xt-bt
                    qt=torch.zeros_like(xt) if mode=='Dg_only' else torch.round(rt/2048).clamp(-128,127)
                    et=helper.write24('fixture_source24',(torch.zeros_like(bt) if mode=='i24_q8_s11' else bt)+qt*2048)
                    lt=old_channel('U_ped',et,'fixtureU24');vt=old_channel('V_ped',lt,'fixtureV24');out=helper.write24('fixture_bias',vt+helper.projection_bias[None,:,None,None])
                    outn=out.cpu().numpy().transpose(1,0,2,3).reshape(96,-1)
                    assert np.array_equal(outn,pred);fixture.append(dict(mode=mode,values=pred.size,differences=0))
                ar['independent_integer_fixtures']=fixture;helper.ready=False
                for mode in MODES:
                    active.update(mode=mode,rows=[],gate=None);helper.frames.clear();bn_calls.clear();activity.start(names)
                    args.output=top/axis/mode;args.output.mkdir(parents=True,exist_ok=True)
                    with torch.no_grad():s=evaluate_axis(args,model,current,names,mode,progress_tag=f'REPR_AEE {axis}')
                    measured=rows(args.output/(mode+'_frames.json'));assert len(active['rows'])==len(names)
                    assert all(r['same_actual_proj_gate'] for r in active['rows'])
                    assert all(int(r['valid_pixels'])==int(float(nb0[r['file']]['valid_pixels'])) for r in measured)
                    save(args.output/'representation_activity.json',active['rows'])
                    save(args.output/'activity_summary.json',activity.finish(helper.frames,bn_calls))
                    ref=float(np.mean([float(nb0[n]['AEE']) for n in names]))
                    ar['modes'][mode]=dict(summary=s,NB0_AEE=ref,delta_NB0=s['AEE_frame_mean']-ref,
                        better_than_NB0=s['AEE_frame_mean']<ref,delta_to_fixed_R24_onepass=s['AEE_frame_mean']-ar['baseline_diverse10'],
                        holdout9_AEE=float(np.mean([r['AEE'] for r in measured if r['file']!=names[0]])),
                        same_actual_proj_gate=True,representation_activity='representation_activity.json')
                    save(top/'run.json',report)
                ar['complete']=True;save(top/'run.json',report)
            finally:
                activity.active=False
                if gate_hook is not None:gate_hook.remove()
                helper.channel_dot=old_channel;bn.forward=old_bn;helper.restore();controller.restore();functional.reset_net(model)
                current.pop('flow',None);torch.cuda.empty_cache()
        report['complete']=True;save(top/'run.json',report)
    finally:
        activity.restore();common_hook.remove();conv1.forward,sn2.forward=old_conv1,old_sn2
    print('REPRESENTATIONS_DONE',flush=True)


if __name__=='__main__':main()
