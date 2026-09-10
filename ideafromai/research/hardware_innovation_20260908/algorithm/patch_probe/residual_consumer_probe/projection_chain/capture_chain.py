"""Prepare sampled three-source reconstruction of the continuous PED branch.

Root launches CUDA. Keep the R32 parent and its four fixed patch BNs, run all
ordinary gates up to proj.sn, and capture only the existing 64 native P4 groups.
No full-frame unfold, X12 rounding, training, or hardware-cycle claims.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

PATCH='sttmultires_unet.encoders.swin3d.patch_embed'
R0=PATCH+'.residual_encoding.resblocks.0'
R1=PATCH+'.residual_encoding.resblocks.1'
PROJECT=PATCH+'.proj'
SPECS={
    'stem':dict(producer=PATCH+'.head',neuron=PATCH+'.head.sn.spiking_neuron',
        conv=PATCH+'.conv.conv.0',norm=PATCH+'.conv.norm_layer.norm_layer',stride=2,channels=48),
    'r0':dict(producer=R0+'.sn2',neuron=R0+'.sn2.spiking_neuron',
        conv=R0+'.conv2.0',norm=R0+'.norm2.norm_layer',stride=1,channels=96),
    'r1':dict(producer=R1+'.sn2',neuron=R1+'.sn2.spiking_neuron',
        conv=R1+'.conv2.0',norm=R1+'.norm2.norm_layer',stride=1,channels=96),
}


def save_json(path,data):
    def convert(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        if isinstance(value,Path):return str(value)
        raise TypeError(type(value))
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,default=convert)+'\n')


def canonical(x):
    if x.ndim==4 and x.shape[0]==10:return x.reshape(10,1,*x.shape[1:])
    if x.ndim==5 and tuple(x.shape[:2])==(10,1):return x
    raise ValueError('This capture requires T10/B1 tensors: '+str(tuple(x.shape)))


def sample(x,positions):
    x=canonical(x)
    if tuple(x.shape[2:])!=(96,240,320):
        raise ValueError('Expected a sampled C96/240x320 residual value, got '+str(tuple(x.shape)))
    return x.detach()[:,0].flatten(2)[:,:,positions].cpu().numpy()


def gather_source(x,positions,conv,theta):
    """Consumer input coordinates, C/kh/kw order; sample before any projected K."""
    x=canonical(x).detach()
    t,_,channels,height,width=x.shape
    yy,xx=positions//320,positions%320
    words=torch.zeros((len(positions),channels,3,3,4),device=x.device,dtype=torch.int32)
    active_count=0
    padding_count=0
    residual=0.
    nzlo,nzhi=float('inf'),float('-inf')
    coordinates=[]
    for kh in range(3):
        for kw in range(3):
            sy=yy*conv.stride[0]+kh*conv.dilation[0]-conv.padding[0]
            sx=xx*conv.stride[1]+kw*conv.dilation[1]-conv.padding[1]
            valid=(sy>=0)&(sy<height)&(sx>=0)&(sx<width)
            value=x[:,0,:,sy.clamp(0,height-1),sx.clamp(0,width-1)]*valid[None,None]
            active=value.ne(0)
            active_count+=int(active.sum())
            residual=max(residual,float((value-active.to(value.dtype)*theta).abs().max()))
            if active.any():
                nonzero=value[active]
                nzlo=min(nzlo,float(nonzero.min()));nzhi=max(nzhi,float(nonzero.max()))
            packed=torch.zeros_like(value[0],dtype=torch.int32)
            for tick in range(t):packed|=active[tick].int()<<tick
            words[:,:,kh,kw]=packed.permute(1,0,2)
            padding_count+=int((~valid).sum())*channels*t
            coordinates.append((sy*width+sx)[valid])
    unique=torch.unique(torch.cat(coordinates))
    unique_values=x[:,0].flatten(2)[:,:,unique]
    return words.reshape(len(positions),channels*9,4).short().cpu().numpy().astype(np.uint16),dict(
        input_shape=list(x.shape),sampled_scalar_entries_with_halo_repeats=int(words.numel()*t),
        sampled_nonzero_entries_with_halo_repeats=active_count,
        padding_zero_scalar_entries=padding_count,
        unique_valid_spatial_coordinates=int(len(unique)),
        unique_sampled_scalar_inputs=int(unique_values.numel()),
        unique_sampled_nonzero_inputs=int(unique_values.ne(0).sum()),
        theta=float(theta),theta_g_sample_max_abs=residual,
        nonzero_amplitude_min=None if nzlo==float('inf') else nzlo,
        nonzero_amplitude_max=None if nzhi==float('-inf') else nzhi,
        layout='G64,K(C,kh,kw),P4 uint16; bit=t, little logical time order; padding zero')


def bn_parameters(module):
    channels=module.num_features
    gamma=module.weight.detach().cpu().numpy() if module.weight is not None else np.ones(channels,np.float32)
    beta=module.bias.detach().cpu().numpy() if module.bias is not None else np.zeros(channels,np.float32)
    batch=bool(module.training or module.running_mean is None or module.running_var is None)
    result=dict(gamma=gamma,beta=beta,eps=np.array(module.eps),uses_batch_statistics=np.array(batch))
    if not batch:
        mean=module.running_mean.detach().cpu().numpy()
        var=module.running_var.detach().cpu().numpy()
        gain=gamma.astype(np.float64)/np.sqrt(var.astype(np.float64)+float(module.eps))
        result.update(mean=mean,var=var,gain=gain,offset=beta.astype(np.float64)-gain*mean)
    return result


def actual_bn_measurement(module,raw,output,positions,full_statistics):
    """Observed-domain Float64 statistics; compare against actual native output.

    No claim to expose cuDNN's internally rounded saved mean/invstd. C8 chunks
    avoid allocating a complete Float64 copy of the full activation tensor.
    """
    x,y=canonical(raw).detach(),canonical(output).detach()
    params=bn_parameters(module)
    gamma=torch.as_tensor(params['gamma'],device=x.device,dtype=torch.float64)
    beta=torch.as_tensor(params['beta'],device=x.device,dtype=torch.float64)
    measured_mean=[];measured_var=[]
    affine_mean=[];affine_var=[];gains=[];offsets=[]
    maximum=0.;square_sum=0;elements=0
    if full_statistics:
        for c in range(0,x.shape[2],8):
            value=x[:,:,c:c+8].double()
            var,mean=torch.var_mean(value,dim=(0,1,3,4),correction=0)
            measured_mean.append(mean.cpu().numpy());measured_var.append(var.cpu().numpy())
            if bool(params['uses_batch_statistics']):used_mean,used_var=mean,var
            else:
                used_mean=torch.as_tensor(params['mean'][c:c+8],device=x.device,dtype=torch.float64)
                used_var=torch.as_tensor(params['var'][c:c+8],device=x.device,dtype=torch.float64)
            gain=gamma[c:c+8]/torch.sqrt(used_var+float(module.eps))
            offset=beta[c:c+8]-gain*used_mean
            error=value*gain[None,None,:,None,None]+offset[None,None,:,None,None]-y[:,:,c:c+8].double()
            maximum=max(maximum,float(error.abs().max()));square_sum+=float(error.square().sum());elements+=error.numel()
            affine_mean.append(used_mean.cpu().numpy());affine_var.append(used_var.cpu().numpy())
            gains.append(gain.cpu().numpy());offsets.append(offset.cpu().numpy())
        params.update(measured_domain_mean=np.concatenate(measured_mean),measured_domain_var=np.concatenate(measured_var),
            mean=np.concatenate(affine_mean),var=np.concatenate(affine_var),
            gain=np.concatenate(gains),offset=np.concatenate(offsets))
        region='complete T10/B1/H240/W320 per-channel domain; biased population variance'
    else:
        if bool(params['uses_batch_statistics']):raise ValueError('Residual BN2 must use the four fixed-BN parent.')
        value=sample(x,positions).astype(np.float64)
        native=sample(y,positions).astype(np.float64)
        error=value*params['gain'][None,:,None,None]+params['offset'][None,:,None,None]-native
        maximum=float(np.abs(error).max());square_sum=float(np.square(error).sum());elements=error.size
        region='64 native P4 sampled positions, all T10/C96'
    return params,dict(region=region,elements=int(elements),FP64_affine_vs_native_max_abs=maximum,
        FP64_affine_vs_native_RMSE=float(np.sqrt(square_sum/elements)),
        statistics_scope='Float64 reduction of actual BN input, not native cuDNN saved-stat bit equivalence',
        uses_batch_statistics=bool(params['uses_batch_statistics']))


def gate_values(words,theta):
    return ((words[None]>>np.arange(10,dtype=np.uint16)[:,None,None,None])&1).astype(np.float64)*theta


def reconstruct(words,theta,conv_weight,conv_bias,gain,offset,C):
    values=gate_values(words,theta)
    W=conv_weight.astype(np.float64).reshape(96,-1)
    raw=np.einsum('hk,tgkp->thgp',W,values,optimize=True)+conv_bias[None,:,None,None]
    affine=raw*gain[None,:,None,None]+offset[None,:,None,None]
    K=(C*gain[None,:])@W
    constant=C@(gain*conv_bias+offset)
    projected=np.einsum('ok,tgkp->togp',K,values,optimize=True)+constant[None,:,None,None]
    return raw,affine,K,constant,projected


def opportunity(words,K,anchor_mask):
    active=((words[None]>>np.arange(10,dtype=np.uint16)[:,None,None,None])&1).astype(np.int64)
    column_nonzero=np.count_nonzero(K,axis=0)
    full=active.sum(axis=(0,1,3))
    anchors=(active*anchor_mask[None,:,None,:]).sum(axis=(0,1,3))
    return dict(K_shape=list(K.shape),K_FP64_bytes=int(K.nbytes),K_nonzero_coefficients=int(np.count_nonzero(K)),
        source_sample_nonzero_with_halo_repeats=int(full.sum()),source_anchor_nonzero_with_halo_repeats=int(anchors.sum()),
        extra_projected_weighted_terms_all_sampled_P4=int(full@column_nonzero),
        extra_projected_weighted_terms_actual_PED_anchors=int(anchors@column_nonzero),
        interpretation='Additional projected theta*g weighted terms including first valid assignment; ordinary gate-producing convolutions/neurons remain. These are neither cycles nor SRAM transactions.')


def errors(actual,reconstructed):
    e=actual.astype(np.float64)-reconstructed
    return dict(values=int(e.size),max_abs=float(np.abs(e).max()),RMSE=float(np.sqrt(np.square(e).mean())))


class Captured(Exception):pass


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--train-list',type=Path,required=True)
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    alg=args.root/'algorithm';area=alg/'patch_probe';residual=area/'residual_consumer_probe'
    latent=area/'factor_completion_20260909/latent_stage_train16'
    args.parent=args.parent or latent/'flow_recovery64/preview_only/shared48_u8_vq5.npz'
    args.output=args.output or residual/'projection_chain/capture_train4'
    args.output.mkdir(parents=True,exist_ok=True)
    for path in (alg,alg/'nrv_cost_probe',latent,residual):sys.path.insert(0,str(path))
    import run_probe as probe
    from run_bn_probe import input_frame
    from adapter import install_latent_factor
    from capture import convolution_anchor_mask,description
    from spikingjelly.activation_based import functional
    train=json.loads(args.train_list.read_text())
    train=train['train'] if isinstance(train,dict) else train
    if len(train)!=16:raise ValueError('Use the existing explicit train16 list; capture its first four frames only.')
    with np.load(args.parent) as p:
        if int(p['shared_rank'])!=32 or np.any(p['v'][32:]):raise ValueError('Expected the R32 preview-only parent.')
    system=probe.load_system(args)
    model,modules,_,current,sources,_,_,_=system
    probe.install_sources(system,sources);current['count_codes']=False
    fixed=torch.load(area/'patch_train_calibration.pt',map_location='cpu',weights_only=False)
    for name,values in fixed.items():
        bn=modules[name];bn.track_running_stats=True
        bn.running_mean=values['mean'].to(bn.weight);bn.running_var=values['var'].to(bn.weight)
    model.eval()
    if any(getattr(modules[name],'connect_function',None)!='ADD' for name in (R0,R1)):
        raise ValueError('The three-source reconstruction requires the actual additive r0/r1 residuals.')
    pair,originals=install_latent_factor(modules[R1+'.conv1.0'],modules[R1+'.sn2.spiking_neuron'],args.parent,conditional=False)
    positions=torch.load(area/'partial_completion/capture.pt',map_location='cpu',weights_only=False)['positions'].long()
    if tuple(positions.shape)!=(64,4):raise ValueError('Expected the existing64 native P4 positions.')
    conv_res=modules[PROJECT+'.conv_res']
    if (tuple(conv_res.kernel_size),tuple(conv_res.stride),tuple(conv_res.padding),conv_res.groups)!=((1,1),(2,2),(0,0),1):
        raise ValueError('This capture requires the actual 1x1 stride2 PED continuous projection.')
    geometry=dict(kernel=tuple(conv_res.kernel_size),stride=tuple(conv_res.stride),padding=tuple(conv_res.padding),dilation=tuple(conv_res.dilation))
    anchor=convolution_anchor_mask(positions,(240,320),**geometry).numpy()
    gpu_positions=positions.to(conv_res.weight.device)
    C=conv_res.weight.detach().cpu().numpy()[:,:,0,0].astype(np.float64)
    Cbias=conv_res.bias.detach().cpu().numpy().astype(np.float64) if conv_res.bias is not None else np.zeros(96,np.float64)
    parameters=dict(C=C,original_C_float32=conv_res.weight.detach().cpu().numpy()[:,:,0,0],
        projection_bias=Cbias,projection_has_bias=np.array(conv_res.bias is not None),
        positions=positions.numpy(),anchor_mask=anchor,projection_stride=np.array(conv_res.stride))
    info={};theta={};conv_data={};handles=[];record={};source_meta={};bn_stats={};bn_checks={};producer_pointers={}
    for label,spec in SPECS.items():
        conv=modules[spec['conv']]
        if (tuple(conv.kernel_size),tuple(conv.stride),tuple(conv.padding),tuple(conv.dilation),conv.groups,conv.in_channels,conv.out_channels)!=((3,3),(spec['stride'],spec['stride']),(1,1),(1,1),1,spec['channels'],96):
            raise ValueError('Actual convolution differs from the declared sampler: '+spec['conv'])
        theta[label]=float(pair.temporal.theta) if label=='r1' else float(modules[spec['neuron']].thresh.detach())
        W=conv.weight.detach().cpu().numpy()
        bias=conv.bias.detach().cpu().numpy().astype(np.float64) if conv.bias is not None else np.zeros(96,np.float64)
        conv_data[label]=(W,bias)
        parameters.update({label+'_W':W,label+'_bias':bias,label+'_has_bias':np.array(conv.bias is not None),
            label+'_theta':np.array(theta[label]),label+'_stride':np.array(conv.stride),
            label+'_padding':np.array(conv.padding),label+'_dilation':np.array(conv.dilation)})
        if label!='stem':
            static=bn_parameters(modules[spec['norm']])
            if bool(static['uses_batch_statistics']):raise ValueError('r0/r1 BN2 must be fixed.')
            parameters.update({label+'_bn_'+k:v for k,v in static.items()})
            parameters[label+'_K_float64']=(C*static['gain'][None,:])@W.reshape(96,-1).astype(np.float64)
            parameters[label+'_constant_float64']=C@(static['gain']*bias+static['offset'])
        info[label]={key:description(modules[spec[key]]) for key in ('producer','conv','norm')}
        def producer(module,inputs,output,label=label):producer_pointers[label]=output.data_ptr()
        def before_conv(module,inputs,label=label):
            words,meta=gather_source(inputs[0],gpu_positions,module,theta[label])
            record[label+'_source_gate_words']=words
            meta['consumer_input_same_storage_as_producer']=inputs[0].data_ptr()==producer_pointers.get(label)
            source_meta[label]=meta
        def after_conv(module,inputs,output,label=label):record[label+'_conv_raw']=sample(output,gpu_positions)
        def after_bn(module,inputs,output,label=label):
            record[label+'_bn_output']=sample(output,gpu_positions)
            bn_stats[label],bn_checks[label]=actual_bn_measurement(module,inputs[0],output,gpu_positions,label=='stem')
        handles.extend([modules[spec['producer']].register_forward_hook(producer),
            conv.register_forward_pre_hook(before_conv),conv.register_forward_hook(after_conv),
            modules[spec['norm']].register_forward_hook(after_bn)])
    for name,label in ((R0,'r0_output'),(R1,'r1_output')):
        handles.append(modules[name].register_forward_hook(lambda m,i,o,label=label:record.update({label:sample(o,gpu_positions)})))
    def projection(module,inputs,output):
        raw=canonical(output)[:,0].flatten(2)
        selected=gpu_positions[torch.as_tensor(anchor,device=gpu_positions.device)]
        outpos=(selected//320//2)*160+(selected%320//2)
        record['projection_native_anchor_output']=raw[:,:,outpos].detach().cpu().numpy()
        record['projection_input']=sample(inputs[0],gpu_positions)
    def stop(module,inputs,output):
        record['proj_sn_output']=sample(output,gpu_positions)
        raise Captured()
    handles.append(conv_res.register_forward_hook(projection))
    handles.append(modules[PROJECT+'.sn.spiking_neuron'].register_forward_hook(stop))
    np.savez_compressed(args.output/'parameters.npz',**parameters)
    run=dict(complete=False,parent=str(args.parent),files=train[:4],identity=info,
        fixed_BN_names=list(fixed),sampled_P4_groups=64,sampled_positions=256,sampled_PED_anchors=int(anchor.sum()),
        numeric_scope='R32 dequantized FP32 parent, four fixed patch BNs, original stem BN; no X12/weight quantization applied',
        stop_boundary='after real proj.sn; ordinary source/gate chain retained, proj.conv downstream is outside capture',
        opportunity_scope='Additional sparse projected paths over actual theta*g; real-arithmetic reassociation differs from native FP32, new-student opportunity only',
        source_layout='G64,K(C,kh,kw),P4; bit t; stem source center=(2y,2x), r0/r1=(y,x)',
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32,rows=[])
    save_json(args.output/'summary.json',run)
    started=time.monotonic()
    try:
        for index,name in enumerate(train[:4]):
            record.clear();source_meta.clear();bn_stats.clear();bn_checks.clear();producer_pointers.clear()
            functional.reset_net(model)
            x,_,_=input_frame(args.data,name,targets=False)
            try:model(x)
            except Captured:pass
            else:raise RuntimeError('The requested real proj.sn boundary was not reached.')
            del x
            projected=[];chain_checks={};opportunities={}
            for label in SPECS:
                W,bias=conv_data[label];stats=bn_stats[label]
                raw,affine,K,constant,proj=reconstruct(record[label+'_source_gate_words'],theta[label],W,bias,stats['gain'],stats['offset'],C)
                chain_checks[label]=dict(conv_from_theta_g_vs_native=errors(record[label+'_conv_raw'],raw),
                    affine_conv_vs_native_BN=errors(record[label+'_bn_output'],affine))
                opportunities[label]=opportunity(record[label+'_source_gate_words'],K,anchor)
                projected.append(proj)
                if label=='stem':
                    record['stem_K_float64']=K;record['stem_constant_float64']=constant
                    record.update({'stem_bn_'+key:value for key,value in stats.items()})
            affine_sum=record['stem_bn_output'].astype(np.float64)+record['r0_bn_output']+record['r1_bn_output']
            candidate=sum(projected)+Cbias[None,:,None,None]
            direct_C=np.einsum('oh,thgp->togp',C,record['r1_output'].astype(np.float64),optimize=True)+Cbias[None,:,None,None]
            native=record['projection_native_anchor_output']
            chain_checks.update(three_native_BN_outputs_vs_native_r1=errors(record['r1_output'],affine_sum),
                projection_input_vs_r1=errors(record['projection_input'],record['r1_output']),
                direct_C_FP64_vs_native_projection=errors(native,direct_C[:,:,anchor]),
                three_K_FP64_vs_direct_C_FP64=errors(direct_C[:,:,anchor],candidate[:,:,anchor]),
                three_K_FP64_vs_native_projection=errors(native,candidate[:,:,anchor]))
            record.update(three_K_FP64_anchor_output=candidate[:,:,anchor],direct_C_FP64_anchor_output=direct_C[:,:,anchor],
                positions=positions.numpy(),anchor_mask=anchor,frame_name=np.array(name),split=np.array('train'))
            out=args.output/(f'{index:02d}_'+Path(name).stem+'.npz')
            np.savez_compressed(out,**record)
            row=dict(file=name,capture=str(out),source=source_meta.copy(),BN_checks=bn_checks.copy(),checks=chain_checks,
                additional_projection_terms=opportunities,
                direct_C_anchor_continuous_products=int(10*anchor.sum()*np.count_nonzero(C)),
                dynamic_stem_K_generation=dict(dense_C_gain_products=96*96,dense_matrix_products=96*96*48*9,
                    per_frame=True,scope='Exact current-stem-stat K cannot be assumed static or free; generation/data movement is not scheduled'),
                constants_and_merge='Three projected affine constants plus projection bias once. Branch accumulations/merge still require ports and state; no gate-chain work subtracted.')
            run['rows'].append(row);save_json(args.output/'summary.json',run)
            print('CHAIN_CAPTURE',index+1,name,json.dumps(dict(source_nonzero={k:v['sampled_nonzero_entries_with_halo_repeats'] for k,v in source_meta.items()},
                extra_anchor_terms={k:v['extra_projected_weighted_terms_actual_PED_anchors'] for k,v in opportunities.items()},checks=chain_checks)),flush=True)
        run.update(complete=True,wall_seconds=time.monotonic()-started,
            claim='Sampled source, affine re-association and extra projection workload only; no GPU speed, RTL cycles, whole-frame/whole-network reduction or PPA.')
        save_json(args.output/'summary.json',run)
    finally:
        for handle in handles:handle.remove()
        modules[R1+'.conv1.0'].forward=originals['conv1_forward']
        modules[R1+'.sn2.spiking_neuron'].forward=originals['neuron_forward']


if __name__=='__main__':main()
