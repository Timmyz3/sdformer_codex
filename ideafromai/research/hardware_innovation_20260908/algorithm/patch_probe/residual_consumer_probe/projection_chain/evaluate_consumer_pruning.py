"""One fixed four-axis consumer-pruning accuracy ablation; root launches CUDA.

Common loading/parent/metric follow evaluate_rank32.py exactly. Two source-C8
groupsets (6 of12 per branch) are chosen once from captured train4 anchors using
activity-weighted projected-column energy. No training, validation selection,
rank/mask/precision sweep or hardware speed claim. A changed r0 can change the
actual r1 source; full-frame anchor source counts are collected anew per axis.

Nullspace projection is computed in Float64 and its W is then cast to native
Float32. It is a changed numerical student, not a bit-exact invariant. The
projection-only control retains native W and subtracts deleted projected-source
contributions at conv_res, before V. This deliberately expensive accuracy
implementation does not model a deployed sparse projection.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F

from capture_chain import R0, R1, PROJECT, SPECS, save_json

AXES=('ordinary_rank32','source_group_zero','consumer_nullspace','projection_only_group_zero')


def rank32_arrays(parameters, existing=None):
    C=parameters['C'].astype(np.float64)
    if existing is not None and existing.exists():
        with np.load(existing) as z:
            if not np.array_equal(C,z['C']):
                raise ValueError('Existing rank32 C differs from the captured original projection.')
            U,V=z['U'].copy(),z['V'].copy()
            singular=z['singular_values'].copy()
        origin=str(existing)
    else:
        # Same CPU Float64 torch.linalg.svd and factor rounding as rank32 control.
        left,singular,right=torch.linalg.svd(torch.from_numpy(C),full_matrices=False)
        U=right[:32].float().numpy()
        V=(left[:,:32]*singular[:32]).float().numpy()
        singular=singular.numpy()
        origin='Same unweighted CPU Float64 torch SVD as evaluate_rank32.py; factors rounded toFloat32.'
    if U.shape!=(32,96) or V.shape!=(96,32):
        raise ValueError('This fixed ablation requires ordinary rank32 factors.')
    return U,V,singular,origin


def prepare(capture,output,rank_parameters=None):
    with np.load(capture/'parameters.npz') as z:
        p={k:z[k].copy() for k in z.files}
    U,V,singular,rank_origin=rank32_arrays(p,rank_parameters)
    files=sorted(capture.glob('[0-9][0-9]_*.npz'))
    if len(files)!=4:
        raise ValueError('Exactly the existing four captured training frames are required.')
    pop=np.array([i.bit_count() for i in range(1024)],np.int64)
    counts={label:np.zeros(864,np.int64) for label in ('r0','r1')}
    names=[]
    for file in files:
        with np.load(file) as z:
            if str(z['split'])!='train':raise ValueError('Group selection must not read validation data.')
            if not np.array_equal(z['positions'],p['positions']):raise ValueError('Train P4 positions differ.')
            anchor=z['anchor_mask'].astype(bool)
            names.append(str(z['frame_name']))
            for label in counts:
                words=z[label+'_source_gate_words']
                if words.shape!=(64,864,4) or words.max()>1023:raise ValueError('Expected G64/K864/P4 T10 words.')
                counts[label]+=(pop[words]*anchor[:,None,:]).sum(axis=(0,2))
    data=dict(C=p['C'],U=U,V=V,singular_values=singular,rank=np.array(32),
        projection_bias=p['projection_bias'],projection_has_bias=p['projection_has_bias'],
        selected_train_frames=np.asarray(names),selection_anchor_mask=p['anchor_mask'],
        selection_positions=p['positions'])
    description=dict(rank=32,rank_origin=rank_origin,training_frames=names,
        source_scope='existing64 native P4/frame; only actual PED anchors; allT10 and3x3 offsets, not full training frames',
        score='sum_k n_active_anchor[k] * ||U diag(fixed_BN2_gain) W[:,k]||_2^2',
        grouping='12 consecutive input-channel groups of8; all9 kernel offsets selected together',
        selected_per_branch=6,selection_tie='group index ascending',
        theta_rule='source amplitude remains actual theta*g; each branch scalar theta is unchanged and its common theta^2 would not alter within-branch ranking',
        nullspace_numeric='Float64 Moore-Penrose pinv, fixed rcond1e-12, followed by nativeFloat32 W; no exact Float32 cancellation claim',
        axes=dict(zip(AXES,[
            'Original two Conv2 W; same ordinary rank32 projection.',
            'Set the same selected source-channel W columns to0 in both Conv2 layers; biases and BN offsets remain.',
            'Replace only selected W columns by W-P^+P W, P=U diag(fixedBNgain); retain native nonlinear consumers.',
            'Native W and gate chain unchanged. Subtract both actual sn2 selected-K convolutions from U(r1out), then V and original bias once.'])),
        branches={})
    for label in ('r0','r1'):
        W=p[label+'_W'].astype(np.float64).reshape(96,864)
        gain=p[label+'_bn_gain'].astype(np.float64)
        if bool(p[label+'_bn_uses_batch_statistics']):raise ValueError('Both residual BN2 layers must be fixed.')
        P=U.astype(np.float64)*gain[None,:]
        K=P@W
        score_columns=counts[label]*np.sum(K*K,axis=0)
        score=score_columns.reshape(12,8,9).sum(axis=(1,2))
        groups=np.sort(np.lexsort((np.arange(12),score))[:6])
        channels=(groups[:,None]*8+np.arange(8)[None,:]).reshape(-1)
        mask=np.zeros(96,bool);mask[channels]=True
        flatmask=np.repeat(mask,9)
        zero=W.copy();zero[:,flatmask]=0
        null=W.copy();null[:,flatmask]-=np.linalg.pinv(P,rcond=1e-12)@K[:,flatmask]
        null32=null.astype(np.float32)
        K_deleted=K.copy();K_deleted[:,~flatmask]=0
        projected64=P@null
        projected32=P@null32.astype(np.float64)
        data.update({label+'_original_W':W.reshape(96,96,3,3).astype(np.float32),
            label+'_zero_W':zero.reshape(96,96,3,3).astype(np.float32),
            label+'_nullspace_W':null32.reshape(96,96,3,3),
            label+'_nullspace_W_float64':null.reshape(96,96,3,3),
            label+'_P':P,label+'_K':K,label+'_K_deleted':K_deleted.reshape(32,96,3,3),
            label+'_selected_groups':groups,label+'_selected_channels':channels,
            label+'_selected_source_mask':mask,label+'_group_scores':score,
            label+'_train_active_columns':counts[label],
            label+'_theta':p[label+'_theta'],label+'_bias':p[label+'_bias'],
            label+'_bn_gain':gain,label+'_bn_offset':p[label+'_bn_offset'],
            label+'_bn_gamma':p[label+'_bn_gamma'],label+'_bn_beta':p[label+'_bn_beta'],
            label+'_bn_mean':p[label+'_bn_mean'],label+'_bn_var':p[label+'_bn_var'],
            label+'_bn_eps':p[label+'_bn_eps']})
        description['branches'][label]=dict(selected_groups=groups,selected_channels=channels,
            group_scores=score,train_anchor_active_columns=counts[label],
            train_anchor_active_sum=int(counts[label].sum()),
            train_selected_anchor_active_sum=int(counts[label][flatmask].sum()),
            projection_rank=int(np.linalg.matrix_rank(P)),
            projected_selected_original_max_abs=float(np.abs(K[:,flatmask]).max()),
            projected_selected_after_Float64_max_abs=float(np.abs(projected64[:,flatmask]).max()),
            projected_selected_after_Float32_W_max_abs=float(np.abs(projected32[:,flatmask]).max()),
            original_selected_W_nonzeros=int(np.count_nonzero(W[:,flatmask])),
            nullspace_selected_W_nonzeros=int(np.count_nonzero(null32[:,flatmask])),
            original_W_norm=float(np.linalg.norm(W)),nullspace_W_norm=float(np.linalg.norm(null32)),
            unchanged_columns_max_error=float(np.max(np.abs(null32[:,~flatmask]-W[:,~flatmask]))),
            physical_skip_claim='No: small Float32 projected residue is not an exact zero coefficient contract.')
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'parameters.npz',**data)
    save_json(output/'selection.json',description)
    return data,description


def anchor_activity_columns(x):
    """Actual source occurrences needed at stride2 anchors of a3x3/pad1 conv.

    Nine strided views, no complete unfold and no full-frame source capture.
    Counts include temporal/nonzero source occurrences and spatial halo reuse;
    they are not SRAM requests or unique source reads.
    """
    if x.ndim!=4 or x.shape[:2]!=(10,96):
        raise ValueError('Expected the actual flattened T10/B1,C96 Conv2 input.')
    h,w=x.shape[-2:]
    oh,ow=(h+1)//2,(w+1)//2
    result=torch.zeros((96,3,3),device=x.device,dtype=torch.int64)
    for kh in range(3):
        dy=kh-1
        y0=max(0,(-dy+1)//2);y1=min(oh,(h-dy+1)//2)
        sy=slice(2*y0+dy,2*y1+dy,2)
        for kw in range(3):
            dx=kw-1
            x0=max(0,(-dx+1)//2);x1=min(ow,(w-dx+1)//2)
            sx=slice(2*x0+dx,2*x1+dx,2)
            result[:,kh,kw]=torch.count_nonzero(x[:,:,sy,sx],dim=(0,2,3))
    return result.reshape(-1).cpu().numpy()


class ConsumerControl:
    def __init__(self,modules,arrays):
        self.modules,self.arrays=modules,arrays
        self.projection=modules[PROJECT+'.conv_res']
        self.original_projection_forward=self.projection.forward
        self.conv={label:modules[SPECS[label]['conv']] for label in ('r0','r1')}
        self.original_weights={label:conv.weight.detach().clone() for label,conv in self.conv.items()}
        self.original_bias={label:None if conv.bias is None else conv.bias.detach().clone() for label,conv in self.conv.items()}
        self.first=torch.as_tensor(arrays['U'],device=self.projection.weight.device,dtype=self.projection.weight.dtype)[:,:,None,None]
        self.second=torch.as_tensor(arrays['V'],device=self.projection.weight.device,dtype=self.projection.weight.dtype)[:,:,None,None]
        self.selected={label:torch.as_tensor(arrays[label+'_selected_channels'],device=self.projection.weight.device) for label in ('r0','r1')}
        self.kernel={label:torch.as_tensor(arrays[label+'_K_deleted'][:,arrays[label+'_selected_channels']],
            device=self.projection.weight.device,dtype=self.projection.weight.dtype) for label in ('r0','r1')}
        self.axis=None;self.terms={};self.counts={};self.handles=[]
        for label,conv in self.conv.items():
            if (tuple(conv.kernel_size),tuple(conv.stride),tuple(conv.padding),tuple(conv.dilation),conv.groups)!=(
                    (3,3),(1,1),(1,1),(1,1),1):raise ValueError('The actual residual Conv2 geometry changed.')
            expected=torch.as_tensor(arrays[label+'_original_W'],device=conv.weight.device,dtype=conv.weight.dtype)
            if not torch.equal(conv.weight.detach(),expected):raise ValueError('Live W differs from captured original '+label)
            bn=modules[SPECS[label]['norm']]
            if bn.training or bn.running_mean is None or bn.running_var is None:
                raise ValueError('Residual BN2 must remain on fixed statistics.')
            for key,actual in [('gamma',bn.weight),('beta',bn.bias),('mean',bn.running_mean),('var',bn.running_var)]:
                ref=torch.as_tensor(arrays[label+'_bn_'+key],device=actual.device,dtype=actual.dtype)
                if not torch.equal(actual.detach(),ref):raise ValueError('Live fixed BN2 differs: '+label+'/'+key)
            self.handles.append(conv.register_forward_pre_hook(
                lambda module,inputs,label=label:self.observe(label,inputs[0])))
        self.projection.forward=self.projected

    def begin(self,axis):
        self.axis=axis;self.terms.clear();self.counts={label:[] for label in self.conv}
        for label,conv in self.conv.items():
            if axis=='source_group_zero':weights=self.arrays[label+'_zero_W']
            elif axis=='consumer_nullspace':weights=self.arrays[label+'_nullspace_W']
            else:weights=self.arrays[label+'_original_W']
            conv.weight.copy_(torch.as_tensor(weights,device=conv.weight.device,dtype=conv.weight.dtype))

    def observe(self,label,x):
        native_shape=list(x.shape)
        # SpikingJelly multi-step Conv2d sees native[T,B,C,H,W] in its pre-hook.
        # Use a local view only; never replace the real module input tuple.
        if x.ndim==5 and tuple(x.shape[:3])==(10,1,96):
            local=x.flatten(0,1)
        elif x.ndim==4 and tuple(x.shape[:2])==(10,96):
            local=x
        else:
            raise ValueError('Expected actual T10/B1/C96 residual source, got '+str(native_shape))
        columns=anchor_activity_columns(local.detach())
        selected=np.repeat(self.arrays[label+'_selected_source_mask'],9)
        self.counts[label].append(dict(source_shape=native_shape,local_conv2d_shape=list(local.shape),
            active_columns=columns,anchor_source_active_terms=int(columns.sum()),
            selected_anchor_source_active_terms=int(columns[selected].sum()),
            group_active_terms=columns.reshape(12,8,9).sum(axis=(1,2))))
        if self.axis=='projection_only_group_zero':
            # Actual current source: r1 is observed after the actual r0 forward.
            self.terms[label]=F.conv2d(local.index_select(1,self.selected[label]),self.kernel[label],
                                     bias=None,stride=2,padding=1)

    def projected(self,x):
        value=F.conv2d(x[:,:,::2,::2],self.first)
        if self.axis=='projection_only_group_zero':
            if set(self.terms)!=set(self.conv):raise RuntimeError('Both real Conv2 sources must precede conv_res.')
            value=value-self.terms.pop('r0')-self.terms.pop('r1')
        return F.conv2d(value,self.second,self.projection.bias)

    def source_report(self,names):
        report=dict(scope='Fresh actual full-frame anchor3x3 source occurrences per axis; includesT10 and halo repeats, not cycles/physical requests.',branches={})
        for label,rows in self.counts.items():
            if len(rows)!=len(names):raise RuntimeError('Expected one '+label+' source call per evaluated frame.')
            for name,row in zip(names,rows):row['file']=name
            report['branches'][label]=dict(frames=rows,
                anchor_source_active_terms=sum(r['anchor_source_active_terms'] for r in rows),
                selected_anchor_source_active_terms=sum(r['selected_anchor_source_active_terms'] for r in rows),
                group_active_terms=np.sum([r['group_active_terms'] for r in rows],axis=0))
        return report

    def restore(self):
        for handle in self.handles:handle.remove()
        self.handles.clear();self.terms.clear()
        for label,conv in self.conv.items():
            conv.weight.copy_(self.original_weights[label])
            if conv.bias is not None:conv.bias.copy_(self.original_bias[label])
        self.projection.forward=self.original_projection_forward


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--capture',type=Path)
    parser.add_argument('--rank-parameters',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();args.split='diverse'
    alg=args.root/'algorithm';area=alg/'patch_probe';residual=area/'residual_consumer_probe'
    chain=residual/'projection_chain';latent=area/'factor_completion_20260909/latent_stage_train16'
    args.capture=args.capture or chain/'capture_train4'
    args.output=args.output or chain/'consumer_pruning_diverse10'
    rank_parameters=args.rank_parameters or chain/'rank32_diverse10/parameters.npz'
    arrays,selection=prepare(args.capture,args.output,rank_parameters)
    if args.prepare_only:
        print('PREPARED',json.dumps({k:v['selected_groups'].tolist() for k,v in selection['branches'].items()}),flush=True)
        return
    for path in (alg,alg/'nrv_cost_probe',latent,residual):sys.path.insert(0,str(path))
    import run_probe as probe
    from adapter import install_latent_factor
    from evaluate_branch_control import evaluate_axis,mask_nonanchors
    system=probe.load_system(args)
    model,modules,_,current,sources,_,_,_=system
    probe.install_sources(system,sources);current['count_codes']=False
    fixed=torch.load(area/'patch_train_calibration.pt',map_location='cpu',weights_only=False)
    for name,values in fixed.items():
        bn=modules[name];bn.track_running_stats=True
        bn.running_mean=values['mean'].to(bn.weight);bn.running_var=values['var'].to(bn.weight)
    model.eval()
    parent=latent/'flow_recovery64/preview_only/shared48_u8_vq5.npz'
    conv1,sn2=modules[R1+'.conv1.0'],modules[R1+'.sn2.spiking_neuron']
    _,originals=install_latent_factor(conv1,sn2,parent,conditional=False)
    projection=modules[PROJECT+'.conv_res']
    if not np.array_equal(projection.weight.detach().cpu().numpy()[:,:,0,0].astype(np.float64),arrays['C']):
        raise ValueError('Live original C differs from the selection parent.')
    if (tuple(projection.kernel_size),tuple(projection.stride),tuple(projection.padding))!=((1,1),(2,2),(0,0)):
        raise ValueError('Actual PED continuous projection geometry changed.')
    masks={}
    def delete_nonanchor(module,inputs,output):
        key=(output.shape[-2:],output.device)
        if key not in masks:
            mask=torch.zeros(output.shape[-2:],device=output.device,dtype=torch.bool)
            mask[::2,::2]=True;masks[key]=mask
        return mask_nonanchors(output,masks[key])
    common_hook=modules[R1+'.norm2'].register_forward_hook(delete_nonanchor)
    controller=None
    names=json.loads((alg/'samples.json').read_text())['valid'][:10]
    run=dict(complete=False,files=names,split='diverse',count=10,parent=str(parent),
        common_parent='Same evaluate_rank32 R32 U8/VQ5 preview-only, four fixed patchBNs, r1 nonanchor entire normalized branch deletion.',
        axes=list(AXES),selection_file='selection.json',parameters_file='parameters.npz',
        metric='Unchanged evaluate_branch_control.evaluate_axis coarse-head AEE and valid pixel count.',
        numeric='NativeFP32 network. Float64 mask projection cast toFP32; no X12/INT8 bridge or floating reassociation equivalence claim.',
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32,
        scope='One fixed50% source-group selection per residual, zero training, same10 diverse frames; no825 or speed claim.',
        results={})
    save_json(args.output/'run.json',run)
    try:
        controller=ConsumerControl(modules,arrays)
        for axis in AXES:
            controller.begin(axis)
            result=evaluate_axis(args,model,current,names,axis,progress_tag='CONSUMER_PRUNING_AEE')
            source=controller.source_report(names)
            save_json(args.output/(axis+'_source_activity.json'),source)
            result['actual_source_activity_file']=axis+'_source_activity.json'
            run['results'][axis]=result;save_json(args.output/'run.json',run)
        run['complete']=True;save_json(args.output/'run.json',run)
    finally:
        if controller is not None:controller.restore()
        common_hook.remove()
        conv1.forward,sn2.forward=originals['conv1_forward'],originals['neuron_forward']
    print('DONE',json.dumps(run['results']),flush=True)


if __name__=='__main__':main()
