"""Two forward passes, one fixed frame: locate first post-projection ATLIF flip.
No algorithm edits. Existing calibration observer exposes the actual membrane.
Only packed reference gates and sparse near-threshold observations stay in RAM;
only first-difference statistics/examples are written, not whole-net tensors.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
from numeric import Arithmetic
HERE=Path(__file__).resolve().parent
FRAME='zurich_city_07_a_0001.npy'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
    base=args.root;patch=base/'algorithm/patch_probe';res=patch/'residual_consumer_probe';chain=res/'projection_chain'
    lift=chain/'fast_temporal_recovery_lifting40';stage=lift/'schedule_compare_same_port'
    latent=patch/'factor_completion_20260909/latent_stage_train16'
    for p in [stage,chain,res,latent,base/'algorithm',base/'algorithm/nrv_cost_probe']:sys.path.insert(0,str(p))
    import torch
    import torch.nn.functional as F
    import run_probe as probe
    from capture_inputs import save_json
    from flow_backward_probe import TrainableLatentPair,read_arrays
    from capture import BLOCK,SOURCE_SN,PROJECT,CONSUMER_SN
    from evaluate_branch_control import evaluate_axis,mask_nonanchors
    from train_shared_temporal_recovery import SharedTemporalControl
    from fixed_temporal_coordinates import FixedTemporalForward
    args.split='diverse';args.output=HERE/'first_flip';args.output.mkdir(exist_ok=True);top=args.output
    system=probe.load_system(args);model,modules,_,current,sources,_,_,_=system
    probe.install_sources(system,sources);current['count_codes']=False
    calibration=torch.load(patch/'patch_train_calibration.pt',map_location='cpu',weights_only=False)
    for path,v in calibration.items():
        m=modules[path];m.track_running_stats=True;m.running_mean,m.running_var=v['mean'].to(m.weight),v['var'].to(m.weight)
    model.eval();model.requires_grad_(False)
    flags=json.loads((chain/'affine_shared_temporal_control_diverse10/run.json').read_text())
    torch.backends.cuda.matmul.allow_tf32=bool(flags['TF32_matmul']);torch.backends.cudnn.allow_tf32=bool(flags['TF32_cudnn'])
    pair=TrainableLatentPair(read_arrays(latent/'flow_recovery64/preview_only/shared48_u8_vq5.npz'),modules[SOURCE_SN].weight.device)
    conv1,sn2=modules[BLOCK+'.conv1.0'],modules[BLOCK+'.sn2.spiking_neuron'];conv1.forward,sn2.forward=pair.conv_forward,pair.neuron_forward
    anchor=torch.zeros((240,320),device=pair.u.device,dtype=torch.bool);anchor[::2,::2]=True
    modules[BLOCK+'.norm2'].register_forward_hook(lambda m,i,o:mask_nonanchors(o,anchor))
    controller=SharedTemporalControl(modules,read_arrays(res/'rank_control_parameters.npz'),read_arrays(chain/'rank32_diverse10/parameters.npz'),fit={})
    controller.load_saved('identity_permuted_base',read_arrays(chain/'temporal_structured_recovery/stage128x256/identity_permuted_base.npz'))
    helper=FixedTemporalForward(controller,pair.temporal.theta)
    bn=modules[PROJECT+'.norm_layer'];original_bn=bn.forward;arith=Arithmetic()
    gamma=bn.weight.detach().cpu().numpy();beta=bn.bias.detach().cpu().numpy()
    def onepass(x):
        p=x.permute(0,2,3,1).contiguous().cpu().numpy();s=arith.statistics(p,gamma,beta,bn.eps)
        a=torch.as_tensor(s[3],device=x.device).reshape(1,96,1,1);b=torch.as_tensor(s[4],device=x.device).reshape(1,96,1,1)
        y=x*a;return y+b
    context={};reference=[];case_stats={};first={};pending_input={};pending_h={};old_observers={}
    def projection_gate(m,i,o):
        x=o[:,0] if o.ndim==5 else o
        active=x.ne(0).any(1).float().unsqueeze(1)
        conv=modules[PROJECT+'.conv']
        assert conv.bias is None and conv.groups==1
        context['producer_empty']=F.max_pool2d(active,conv.kernel_size,conv.stride,conv.padding).squeeze(1).eq(0).cpu().numpy()
    modules[CONSUMER_SN].register_forward_hook(projection_gate)
    def bn_done(m,i,o):
        raw=i[0];z=raw.eq(0).all(1).cpu().numpy()
        pe=context['producer_empty'];assert z.shape==pe.shape
        context.update(started=True,bn_zero=z)
        case_stats[context['mode']]=dict(raw_zero_vectors=int(z.sum()),producer_empty_vectors=int(pe.sum()),
            producer_empty_all_raw_zero=bool(z[pe].all()),raw_zeros_without_producer_empty=int((z&~pe).sum()),
            true_zero_default_PED_vectors=int((pe&context['ped_zero']).sum()),domain_vectors=int(z.size))
    bn.register_forward_hook(bn_done)
    def ped_done(m,i,o):
        x=o[:,0] if o.ndim==5 else o
        ped=x.eq(0).all(1).cpu().numpy();context['ped_zero']=ped
    modules[PROJECT+'.conv_res'].register_forward_hook(ped_done)
    def attention_pre(path,m,inputs):
        if not context.get('started') or 'attention_path' in context:return
        context['attention_path']=path
        block=modules[path.rsplit('.attn',1)[0]]
        x=inputs[0];ws=(int(x.shape[0]),int(x.shape[2]),int(x.shape[3]))
        part=block.SSA.__func__.__globals__['window_partition_v2']
        ids=torch.arange(10*120*160,device=x.device).reshape(1,10,120,160,1)
        pd=(ws[0]-10%ws[0])%ws[0];ph=(ws[1]-120%ws[1])%ws[1];pw=(ws[2]-160%ws[2])%ws[2]
        ids=F.pad(ids,(0,0,0,pw,0,ph,0,pd),value=-1)
        shifts=tuple(int(a) for a in block.shift_size)
        if any(shifts):ids=torch.roll(ids,shifts=tuple(-a for a in shifts),dims=(1,2,3))
        mapped=part(ids,ws).squeeze(-1).cpu().numpy()
        assert tuple(mapped.shape)==tuple(x.shape[:-1])
        context['source_ids']=mapped
        context['map_description']='Actual first-block window_partition_v2 including its existing reshape order, padding and roll; no assumed temporal/window order.'
    for path,m in modules.items():
        if hasattr(m,'linear_q') and hasattr(m,'sn_q') and path.endswith('.attn'):
            m.register_forward_pre_hook(lambda m,i,path=path:attention_pre(path,m,i))
    def pre(path,m,inputs):
        if context.get('started') and not (context['mode']=='onepass' and first):pending_input[path]=inputs[0].detach()
    def observe(path,m,h,theta):
        if not context.get('started') or (context['mode']=='onepass' and first):return
        pending_h[path]=(h.detach(),theta.detach())
    def post(path,m,inputs,output):
        if not context.get('started') or (context['mode']=='onepass' and first):return
        x=pending_input.pop(path,None);obs=pending_h.pop(path,None)
        gate=output.detach().ne(0).cpu().numpy();flat=gate.reshape(-1)
        if context['mode']=='cuda':
            rec=dict(path=path,shape=list(gate.shape),bits=np.packbits(flat),nonzeros=int(flat.sum()),near=None)
            if obs is not None:
                h,theta=obs;hf=h.flatten();margin=hf-theta
                # This radius only bounds diagnostic storage. It never changes an activation.
                ids=torch.nonzero(margin.abs()<=1e-4,as_tuple=False).flatten()
                cols=ids%h.shape[1]
                rec['near']=dict(ids=ids.cpu().numpy(),h=hf.index_select(0,ids).cpu().numpy(),
                    inputs=x.flatten(1).index_select(1,cols).cpu().numpy(),theta=float(theta))
            reference.append(rec);return
        index=context['index'];context['index']+=1;rec=reference[index]
        assert path==rec['path'] and list(gate.shape)==rec['shape'],(index,path,rec['path'])
        old=np.unpackbits(rec['bits'],count=flat.size).astype(bool)
        changed=np.flatnonzero(old!=flat)
        if not changed.size:return
        first.update(module=path,post_BN_call_index=index,shape=list(gate.shape),flips=int(changed.size),denominator=int(flat.size),
            zero_to_one=int((~old[changed]&flat[changed]).sum()),one_to_zero=int((old[changed]&~flat[changed]).sum()),
            baseline_nonzeros=rec['nonzeros'],candidate_nonzeros=int(flat.sum()),threshold_mode=getattr(m,'threshold_mode',None),
            output_mode=getattr(m,'output_mode',None),actual_membrane_observer=obs is not None,
            equal_preceding_neuron_modules=[r['path'] for r in reference[:index]])
        coords=np.stack(np.unravel_index(changed,gate.shape),axis=1)
        first['flips_per_T']=np.bincount(coords[:,0],minlength=gate.shape[0]).tolist()
        if obs is not None:
            h,theta=obs;ids=torch.as_tensor(changed,device=h.device);cols=ids%h.shape[1]
            ch=h.flatten().index_select(0,ids).cpu().numpy();ci=x.flatten(1).index_select(1,cols).cpu().numpy()
            near=rec['near'];pos=np.searchsorted(near['ids'],changed) if near is not None else np.full(changed.size,-1)
            matched=np.zeros(changed.size,bool)
            if near is not None:
                valid=pos<len(near['ids']);matched[valid]=near['ids'][pos[valid]]==changed[valid]
            first['membrane']=dict(theta=float(theta),baseline_sparse_record_radius=1e-4,
                baseline_flip_margins_available=int(matched.sum()),candidate_margin_min=float((ch-float(theta)).min()),
                candidate_margin_max=float((ch-float(theta)).max()),all_candidate_flip_temporal_inputs_zero=int(np.all(ci==0,axis=0).sum()))
            if matched.any():
                bh=near['h'][pos[matched]];first['membrane'].update(baseline_margin_min=float((bh-near['theta']).min()),
                    baseline_margin_max=float((bh-near['theta']).max()),max_flip_membrane_delta=float(np.max(np.abs(ch[matched]-bh))))
            examples=[]
            for j in range(min(16,changed.size)):
                row=dict(flat_index=int(changed[j]),coordinates=coords[j].tolist(),cuda_gate=int(old[changed[j]]),onepass_gate=int(flat[changed[j]]),
                    candidate_h=float(ch[j]),candidate_input_T=ci[:,j].tolist(),candidate_margin=float(ch[j]-float(theta)))
                if matched[j]:row.update(baseline_h=float(near['h'][pos[j]]),baseline_margin=float(near['h'][pos[j]]-near['theta']),baseline_input_T=near['inputs'][:,pos[j]].tolist())
                examples.append(row)
            first['examples']=examples
            uniq,counts=np.unique(ci.T,axis=0,return_counts=True)
            classes=[]
            for j in np.argsort(-counts)[:8]:
                vector=torch.as_tensor(uniq[j],device=x.device)
                multiplicity=int(x.flatten(1).eq(vector[:,None]).all(0).sum())
                classes.append(dict(actual_input_T=uniq[j].tolist(),flips_sharing_input=int(counts[j]),leaf_columns_sharing_input=multiplicity))
            first['candidate_repeated_input_classes']=classes
        mapped=context.get('source_ids');attn=context.get('attention_path')
        if mapped is not None and path.startswith(attn+'.') and tuple(gate.shape[:-1])==tuple(mapped.shape):
            positions=(changed//gate.shape[-1]);source_id=mapped.reshape(-1)[positions];valid=source_id>=0
            masks={}
            for key,src in [('producer_empty',context['producer_empty']),('raw_zero',context['bn_zero']),('PED_zero',context['ped_zero']),('producer_empty_and_PED_zero',context['producer_empty']&context['ped_zero'])]:
                w=np.zeros(mapped.shape,bool);real=mapped>=0;w[real]=src.reshape(-1)[mapped[real]]
                pair=np.broadcast_to(w.all(0,keepdims=True),w.shape)
                masks[key]=dict(flips_at_current_position=int(w.reshape(-1)[positions].sum()),
                    flips_with_all_PSN_input_T_positions=int(pair.reshape(-1)[positions].sum()),
                    valid_source_vectors=int(src.sum()),source_vector_domain=int(src.size))
            first['source_default_distribution']=dict(mapping=context['map_description'],valid_position_flips=int(valid.sum()),padding_flips=int((~valid).sum()),classes=masks)
            for j,row in enumerate(first.get('examples',[])):
                sid=int(source_id[j]);row['source_T_y_x']=(list(map(int,np.unravel_index(sid,(10,120,160)))) if sid>=0 else None)
        else:first['source_default_distribution']=dict(mapping_available=False,reason='First differing leaf is beyond the first attention direct-position interface; no unsupported source-default mapping is inferred.')
    for path,m in modules.items():
        if type(m).__name__!='ATLIFTernaryPSN':continue
        old_observers[path]=getattr(m,'_h9_calibration_observer',None)
        m._h9_calibration_observer=lambda h,t,path=path,m=m:observe(path,m,h,t)
        m.register_forward_pre_hook(lambda m,i,path=path:pre(path,m,i))
        m.register_forward_hook(lambda m,i,o,path=path:post(path,m,i,o))
    report=dict(scope=__doc__,student='ordinary unchanged',frame=FRAME,network_forwards=2,training=False,
        changed_module=PROJECT+'.norm_layer',method='Original CUDA BN versus specified onepass Engine function; no threshold/epsilon/sample sweep.',cases={})
    try:
        for mode in ['cuda','onepass']:
            context.clear();context.update(mode=mode,started=False,index=0)
            bn.forward=original_bn if mode=='cuda' else onepass
            args.output=top/mode;args.output.mkdir(exist_ok=True)
            with torch.no_grad():report['cases'][mode]=evaluate_axis(args,model,current,[FRAME],mode,progress_tag='FIRST_POST_BN_FLIP')
            print('FIRST_FLIP_CASE',mode,'observed_reference_leaves',len(reference),'first',first.get('module'),flush=True)
        report.update(first_difference=first,projection_default_counts=case_stats,
            reference_storage='Packed gate bits plus only abs(h−theta)<=1e−4 sparse actual-membrane/input samples in RAM. No all-network tensor archive. This storage radius is not an execution threshold.',
            scope_limit='Source-default membership is correlation at the mapped first attention input; only complete temporal default/PED-zero and actual repeated-input counts support a repeated-class interpretation.')
        save_json(top/'result.json',report)
        print('FIRST_FLIP_DONE',json.dumps(first),flush=True)
    finally:
        bn.forward=original_bn
        for path,old in old_observers.items():modules[path]._h9_calibration_observer=old
        helper.restore();controller.restore()

if __name__=='__main__':main()
