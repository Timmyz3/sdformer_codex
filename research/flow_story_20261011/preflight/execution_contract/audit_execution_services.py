#!/usr/bin/env python3
"""CPU-only structural producer and configurable issue-loop audit for v2.

Not a calibrated cycle simulator: no checkpoint/spike-rate/data trace, SRAM
stalls, DMA timing, synthesis, or trained predictor. Original source is read
through the prior AST layout adapter; all new files remain in this directory.
Run with /opt/anaconda3/envs/pytorch310_cpu/bin/python for ConvTranspose probes.
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
import pathlib
import sys
import types

sys.dont_write_bytecode = True
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PREFLIGHT = HERE.parent
ROOT = PREFLIGHT.parents[2]
SOURCE = ROOT/'SDformer/third_party/SDformerFlow'
layout_path = PREFLIGHT/'audit_v2_window_layout.py'
layout = types.ModuleType('layout_audit')
layout.__file__ = str(layout_path)
exec(compile(layout_path.read_text(),str(layout_path),'exec'),layout.__dict__)

T = 10
DEPTHS = [2,2,6,2]
CHANNELS = [96,192,384,768]
HEADS = [3,6,12,24]
MLP_TILE = 9  # Independent of the paper full-resolution attention window 15.


def ceildiv(a,b): return (int(a)+int(b)-1)//int(b)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


class Engine:
    """A stated rectangular dot engine, not a characterized RTL datapath.

    For M independent rows, K input/reduction entries and N output channels:
    M * ceil(K/Pk) * ceil(N/Pn) issue steps. Conv loops restart K per valid
    spatial tap. PSN maps W[T,T] @ X[T,F] as M=T,K=T,N=F. Native binary spikes
    do not become one cycle each: they occupy reduction/output lane groups.
    All engines are serialized for this compute-service budget; no overlap or
    data-dependent zero skipping is credited. Affine frozen BN is folded into
    preceding weights/bias. Accumulator bias init is assumed included.
    """
    def __init__(self,pk=16,pn=16,real_ii=1,spike_ii=1,pack_attention_components=False):
        self.pk,self.pn,self.real_ii,self.spike_ii=pk,pn,real_ii,spike_ii
        self.pack_attention_components=pack_attention_components

    def dot(self,name,m,k,n,kind):
        ii=self.spike_ii if kind=='binary_weighted_AC' else self.real_ii
        return {'name':name,'kind':kind,'M':int(m),'K':int(k),'N':int(n),
                'scalar_products':int(m*k*n),
                'static_issue_steps':int(m*ceildiv(k,self.pk)*ceildiv(n,self.pn)*ii),
                'output_elements':int(m*n)}

    def vector(self,name,elements,kind='real_add'):
        return {'name':name,'kind':kind,'scalar_products':0,
                'scalar_element_ops':int(elements),'output_elements':int(elements),
                'static_issue_steps':ceildiv(elements,self.pn)}

    def psn(self,name,t,features,input_kind='real_MAC'):
        return [self.dot(name+'.temporal',t,t,features,input_kind),
                self.vector(name+'.threshold',t*features,'threshold')]

    def reduction(self,name,gates,width):
        return {'name':name,'kind':'binary_sum_to_uint6','scalar_products':0,
                'scalar_element_ops':int(gates*(width-1)),
                'static_issue_steps':int(ceildiv(gates,self.pn)*ceildiv(width,self.pk)),
                'output_elements':int(gates)}

    def spec(self):
        return {'Pk_reduction_lanes':self.pk,'Pn_output_lanes':self.pn,
                'dot_lanes':self.pk*self.pn,'real_MAC_issue_interval':self.real_ii,
                'binary_weighted_AC_issue_interval':self.spike_ii,
                'vector_lanes':self.pn,'frozen_BN_folded':True,
                'attention_across_component_tail_packing':self.pack_attention_components,
                'data_dependent_zero_skip':False,'calibrated_cycle_model':False,
                'precision':'native FP code; semantic binary/real split only, fixed-point widths below are accounting assumptions',
                'weight_bits_accounting':8,'private_accumulator_bits_accounting':32,
                'residual_bits_accounting':16,'spike_bits_accounting':1,
                'q_sum_bits_accounting':6,'logical_word_bits':128}


def total(rows,key='static_issue_steps'):
    return sum(int(x.get(key,0)) for x in rows)


def stats(rows):
    bykind=collections.defaultdict(lambda:{'scalar_products':0,'static_issue_steps':0,'scalar_element_ops':0})
    for r in rows:
        for k in bykind[r['kind']]:bykind[r['kind']][k]+=int(r.get(k,0))
    return {'static_issue_steps':total(rows),'scalar_products':total(rows,'scalar_products'),
            'by_kind':dict(bykind)}


def runs(ids):
    ids=np.unique(np.asarray(ids,dtype=np.int64))
    ids=ids[ids>=0]
    if not len(ids):return {'count':0,'token_count':0,'first_runs':[]}
    starts=np.r_[0,np.nonzero(np.diff(ids)!=1)[0]+1]
    ends=np.r_[starts[1:],len(ids)]
    return {'count':int(len(starts)),'token_count':int(len(ids)),
            'first_runs':[{'first_token':int(ids[s]),'tokens':int(e-s)}
                          for s,e in zip(starts[:12],ends[:12])],
            'first_runs_truncated':bool(len(starts)>12),
            'layout':'B=1, contiguous T,H,W,C; token run is C adjacent elements; actual chosen SRAM/DMA mapping not yet measured'}


def dependency_graph(H,W,C,heads,win,shifted):
    ws=(2,win,win)
    shift_req=(1,win//2,win//2) if shifted else (0,0,0)
    ws,shift=layout.window_size((T,H,W),ws,shift_req)
    dp,hp,wp=[ceildiv(n,w)*w for n,w in zip((T,H,W),ws)]
    ids=np.arange(T*H*W,dtype=np.int64).reshape(1,T,H,W,1)
    padded=np.pad(ids,((0,0),(0,dp-T),(0,hp-H),(0,wp-W),(0,0)),constant_values=-1)
    part=layout.partition(layout.Tensor(np.roll(padded,tuple(-v for v in shift),axis=(1,2,3))),ws)
    t,n,wh,ww,_=part.shape
    pairs=n*wh*ww
    pair=np.broadcast_to(np.arange(pairs).reshape(1,n,wh,ww,1),(t,n,wh,ww,heads)).copy()
    deps=layout.reverse_to_original(layout.qk_probe(layout.Tensor(pair),heads),1,T,H,W,ws,shift,(dp,hp,wp))[0]
    # Independently expose Q->sum->sn2_q->gate path through the native forward
    # AST. K=1; paired-T IDs are equal across the two input lanes. Swapping the
    # two sn2_q temporal lanes probes off-diagonal temporal contributions. With
    # the proven head_dim quotient=1, sum keeps a label unchanged. This is a
    # dependency-label test, not a native trained PSN numerical forward.
    identity=lambda value:value
    qobj=types.SimpleNamespace(num_heads=heads,norm_layer=None,
        positional_encoding=layout.Tensor(np.zeros((1,heads,t*wh*ww,1),dtype=np.int64)),
        proj_sn=identity,linear_q=identity,sn_q=identity,
        linear_k=lambda value:layout.Tensor(np.ones(value.shape,dtype=np.int64)),sn_k=identity,
        sn2_q=lambda value:layout.Tensor(value.a[::-1].copy()),
        attn_drop=identity,attn_sn=identity,proj=identity)
    qdeps=layout.reverse_to_original(layout.native_qk(qobj,layout.Tensor(pair))[0],1,T,H,W,ws,shift,(dp,hp,wp))[0]
    assert np.array_equal(deps,qdeps)
    yy,xx=np.meshgrid(np.arange(H),np.arange(W),indexing='ij')
    ny,nx=hp//wh,wp//ww
    grid=((yy-shift[1])%hp//wh)*nx+((xx-shift[2])%wp//ww)
    group=np.broadcast_to(grid[None,:,:,None],deps.shape)
    ng=ny*nx
    code=np.unique(deps.reshape(-1)*ng+group.reshape(-1))
    dsu=layout.DSU(ng);last=-1;first=-1
    for value in code:
        p,g=divmod(int(value),ng)
        if p!=last:last,first=p,g
        else:dsu.union(first,g)
    components=collections.defaultdict(list)
    for g in range(ng):components[dsu.root(g)].append(g)
    comps=sorted(components.values(),key=lambda z:z[0])
    comp_for_group=np.empty(ng,dtype=np.int64)
    for ci,gs in enumerate(comps):comp_for_group[gs]=ci
    spatial_comp=comp_for_group[grid]
    producer_owner=np.full(pairs,-1,dtype=np.int64)
    for value in code:
        p,g=divmod(int(value),ng);owner=int(comp_for_group[g])
        assert producer_owner[p] in (-1,owner)
        producer_owner[p]=owner
    # Original input IDs of each paired-T Q/K producer; -1 means padding.
    pair_inputs=part.a[:,:,:,:,0].reshape(2,pairs)
    assert np.all(producer_owner[np.unique(deps)]>=0)
    return {'H':H,'W':W,'C':C,'heads':heads,'win':win,'ws':ws,'shift':shift,
            'padded':(dp,hp,wp),'pairs_total':pairs,'pair_inputs':pair_inputs,
            'deps':deps,'grid':grid,'ng':ng,'components':comps,
            'spatial_comp':spatial_comp,'producer_owner':producer_owner,
            'all_live_pairs':np.unique(deps),'input_ids':ids[0,:,:,:,0]}


def attention_services(g,keep_pixels,engine):
    """Keep producers used by any retained real output; dead padding output
    projections are omitted in BOTH baseline and candidate. Private Q/K
    producer pairs remain full C even if only one consumer channel is needed.
    """
    C,h=g['C'],g['heads']
    mask=np.broadcast_to(keep_pixels[None,:,:,None],g['deps'].shape)
    live=np.unique(g['deps'][mask])
    rows=[]
    # A fixed component descriptor schedule: no free repacking of PSN gate tails
    # across different legal components. Same grouping in old/new baselines.
    component_sizes=[]
    for ci in range(len(g['components'])):
        count=int(np.sum(g['producer_owner'][live]==ci))
        if not count:continue
        component_sizes.append(count)
    issue_sizes=([sum(component_sizes)] if component_sizes else []) if engine.pack_attention_components else component_sizes
    for count in issue_sizes:
        rows+=engine.psn('attn.input_PSN',2,count*C)
        rows.append(engine.dot('attn.Q',2*count,C,C,'binary_weighted_AC'))
        rows+=engine.psn('attn.Q_PSN',2,count*C)
        rows.append(engine.dot('attn.K',2*count,C,C,'binary_weighted_AC'))
        rows.append(engine.vector('attn.K_position_add',2*count*C))
        rows+=engine.psn('attn.K_PSN',2,count*C)
        rows.append(engine.reduction('attn.Q_head_sum',2*count*h,32))
        rows+=engine.psn('attn.Q_gate_PSN',2,count*h,'count_times_weight_MAC')
    retained_rows=T*int(keep_pixels.sum())
    rows.append(engine.vector('attn.binary_AND',retained_rows*C,'binary_AND'))
    rows.append(engine.dot('attn.output_projection',retained_rows,C,C,'binary_weighted_AC'))
    rows.append(engine.vector('attn.residual_add',retained_rows*C))
    # Logical materializations, no promise that each becomes a physical SRAM IO.
    npair=len(live)
    private_bits=(2*npair*C*(1+32+1+32+1) # input spike,Q accum,Q spike,K accum,K spike
                  +2*npair*h*(6+1) # Q count and gate spike
                  +retained_rows*C*(1+32)) # AND and output accumulator
    valid_input=g['pair_inputs'][:,live].reshape(-1)
    valid_input=np.unique(valid_input[valid_input>=0])
    skipped_spatial=(~keep_pixels).reshape(-1)
    shared=int(np.sum(skipped_spatial[valid_input%(g['H']*g['W'])])) if len(valid_input) else 0
    padding_values=int(np.sum(g['pair_inputs'][:,live]<0))
    weight_bits=(3*C*C*8+3*C*32+4*4*16+2*g['ws'][1]*g['ws'][2]*C*16) if npair else 0
    detail={
        'QK_paired_time_producers_kept':npair,
        'QK_projection_rows_kept':2*npair,
        'output_projection_rows_kept':retained_rows,
        'private_padding_input_lanes_still_needed':padding_values,
        'unique_real_attention_input_tokens_still_read':int(len(valid_input)),
        'attention_input_tokens_from_bypassed_output_regions_still_needed':shared,
        'residual_full_grid_tokens_still_available':T*g['H']*g['W'],
        'potential_private_materialized_write_bits':int(private_bits),
        'ideal_packed_private_write_words128':ceildiv(private_bits,128),
        'selected_layer_weight_payload_bits_accounting':int(weight_bits),
        'weight_caveat':'selected payload estimate, not exact parameter bytes: includes FC/folded-affine bias, temporal matrices and position table, excludes PSN bias parameters; partial groups do not remove layer-shared weights; widths and caching are unvalidated accounting choices',
        'input_address_runs':runs(valid_input),
        'bypassed_output_address_runs':runs(g['input_ids'][:,~keep_pixels].reshape(-1)),
        'active_descriptor_components':len(component_sizes),
        'diagnostic_attn_sn':'omitted from both baselines because normal flow forward discards its returned diagnostic tensor',
    }
    return rows,detail,live


def mlp_services(H,W,C,keep_pixels,engine):
    pix=int(keep_pixels.sum());rows=[]
    rows+=engine.psn('mlp.sn1',T,pix*C)
    rows.append(engine.dot('mlp.fc1',T*pix,C,4*C,'binary_weighted_AC'))
    rows+=engine.psn('mlp.sn2',T,pix*4*C)
    rows.append(engine.dot('mlp.fc2',T*pix,4*C,C,'binary_weighted_AC'))
    rows.append(engine.vector('mlp.residual_add',T*pix*C))
    bits=T*pix*(C+4*C*32+4*C+C*32)
    ids=np.arange(T*H*W).reshape(T,H,W)
    return rows,{'spatial_sites_kept':pix,'time_rows_kept':T*pix,
                 'potential_private_materialized_write_bits':int(bits),
                 'ideal_packed_private_write_words128':ceildiv(bits,128),
                 'selected_layer_weight_payload_bits_accounting':int((8*C*C*8+5*C*32+2*T*T*16) if pix else 0),
                 'weight_caveat':'selected payload estimate, excludes PSN biases; layer-shared FC/BN-folded/temporal parameters retained for every partial mask; no measured memory traffic',
                 'bypassed_output_address_runs':runs(ids[:,~keep_pixels].reshape(-1)),
                 'bypass_value':'u=x+attention(x), not pre-attention x; joint uses x instead',
                 'all_T10_inputs_required_for_retained_sites':True}


def valid_conv_taps(h,w,k=3,stride=1,pad=1):
    oh=(h+2*pad-k)//stride+1;ow=(w+2*pad-k)//stride+1
    vh=sum(sum(0<=y*stride-pad+j<h for j in range(k)) for y in range(oh))
    vw=sum(sum(0<=x*stride-pad+j<w for j in range(k)) for x in range(ow))
    return oh,ow,vh*vw


def valid_transpose_taps(h,w,k=3,stride=2,pad=1,outpad=1):
    oh=(h-1)*stride-2*pad+k+outpad;ow=(w-1)*stride-2*pad+k+outpad
    vh=sum(sum(0<=y*stride-pad+j<oh for j in range(k)) for y in range(h))
    vw=sum(sum(0<=x*stride-pad+j<ow for j in range(k)) for x in range(w))
    return oh,ow,vh*vw


def retained_network_services(H,W,engine):
    """Reconstruct uncancelled front/merge/bottleneck/decoder/head shapes from
    true constructors, not legacy .flops() or scaling the old crop denominator.
    """
    rows=[];operators=[]
    def conv(name,h,w,cin,cout,k=3,s=1,p=1,kind='binary_weighted_AC',transpose=False):
        oh,ow,taps=(valid_transpose_taps(h,w,k,s,p,1) if transpose else valid_conv_taps(h,w,k,s,p))
        r=engine.dot(name,T*taps,cin,cout,kind);rows.append(r)
        operators.append({'name':name,'input_HWC':[h,w,cin],'output_HWC':[oh,ow,cout],
                          'kernel':k,'stride':s,'valid_spatial_taps':taps,
                          'full_kernel_padding_tap_proxy':(h*w if transpose else oh*ow)*k*k,
                          'operation': 'ConvTranspose2d' if transpose else 'Conv2d',**r})
        return oh,ow
    # Head has continuous voxel input, THEN PSN; following MS conv first_layer=True
    # receives head spikes and does not instantiate an additional neuron.
    conv('patch.head',H,W,2,48,kind='real_MAC')
    rows+=engine.psn('patch.head_PSN',T,H*W*48)
    h,w=conv('patch.conv',H,W,48,96,s=2)
    for b in range(2):
        for j in range(2):
            rows+=engine.psn('patch.res{}.sn{}'.format(b,j),T,h*w*96)
            conv('patch.res{}.conv{}'.format(b,j),h,w,96,96)
        rows.append(engine.vector('patch.res{}.add'.format(b),T*h*w*96))
    conv('patch.PED.conv_res',h,w,96,96,k=1,s=2,p=0,kind='real_MAC')
    rows+=engine.psn('patch.PED.PSN',T,h*w*96)
    h,w=conv('patch.PED.conv_spike',h,w,96,96,s=2)
    rows.append(engine.vector('patch.PED.add',T*h*w*96))
    for si in range(3):
        sh,sw=H//(4*2**si),W//(4*2**si);c=CHANNELS[si]
        oh,ow=ceildiv(sh,2),ceildiv(sw,2)
        rows+=engine.psn('merge{}.PSN'.format(si),T,oh*ow*4*c)
        rows.append(engine.dot('merge{}.reduction'.format(si),T*oh*ow,4*c,2*c,'binary_weighted_AC'))
    bh,bw=H//32,W//32
    for b in range(2):
        for j in range(2):
            rows+=engine.psn('bottleneck.res{}.sn{}'.format(b,j),T,bh*bw*768)
            conv('bottleneck.res{}.conv{}'.format(b,j),bh,bw,768,768)
        rows.append(engine.vector('bottleneck.res{}.add'.format(b),T*bh*bw*768))
    dh,dw=bh,bw
    for di,(ci,co) in enumerate(zip([1536,770,386,194],[384,192,96,96])):
        rows+=engine.psn('decoder{}.PSN'.format(di),T,dh*dw*ci)
        dh,dw=conv('decoder{}.deconv'.format(di),dh,dw,ci,co,s=2,transpose=True)
        rows+=engine.psn('pred{}.PSN'.format(di),T,dh*dw*co)
        conv('pred{}.conv'.format(di),dh,dw,co,2,k=1,p=0)
        rows.append(engine.vector('pred{}.time_sum'.format(di),(T-1)*dh*dw*2))
    return rows,operators


def closure_mask(g,component_ids):
    return np.isin(g['spatial_comp'],np.asarray(component_ids,dtype=np.int64))


def aggregate_ops(rows):
    out={}
    for r in rows:
        name=r['name']
        if name not in out:out[name]={'kind':r['kind'],'scalar_products':0,'static_issue_steps':0,'scalar_element_ops':0,'output_elements':0}
        for key in ('scalar_products','static_issue_steps','scalar_element_ops','output_elements'):
            out[name][key]+=int(r.get(key,0))
    return out


def mask_record(g,name,bypass,engine,attention_enabled=True):
    keep=~bypass
    att,ad,live=attention_services(g,keep if attention_enabled else np.ones_like(keep),engine)
    mlp,md=mlp_services(g['H'],g['W'],g['C'],keep,engine)
    baseatt,basead,_=attention_services(g,np.ones_like(keep),engine)
    basemlp,basemd=mlp_services(g['H'],g['W'],g['C'],np.ones_like(keep),engine)
    touched=np.unique(g['spatial_comp'][bypass]) if bypass.any() else []
    closed=all(np.all(bypass[g['spatial_comp']==ci]) for ci in touched)
    cancelled=np.setdiff1d(g['all_live_pairs'],live,assume_unique=True)
    assert not np.isin(g['deps'][np.broadcast_to(keep[None,:,:,None],g['deps'].shape)],cancelled).any() if attention_enabled else True
    if not bypass.any():assert total(att+mlp)==total(baseatt+basemlp)
    if bypass.all() and attention_enabled:assert total(att+mlp)==0
    return {'mask':name,'contract':'joint' if attention_enabled else 'MLP_only',
            'real_spatial_sites_bypassed':int(bypass.sum()),
            'fraction_of_real_spatial_sites_bypassed':float(bypass.mean()),
            'whole_joint_producer_component_closed':bool(closed),
            'closed_component_ids_touched':[int(x) for x in touched],
            'zero_shared_cancelled_producers_used_by_kept_attention':True,
            'cancelled_QK_paired_time_producers':int(len(cancelled)),
            'attention':ad,'MLP':md,
            'kept_operator_services':aggregate_ops(att+mlp),
            'baseline_static_issue_steps':total(baseatt+basemlp),
            'kept_static_issue_steps':total(att+mlp),
            'saved_static_issue_steps':total(baseatt+basemlp)-total(att+mlp),
            'potential_private_write_bits_saved':basead['potential_private_materialized_write_bits']+basemd['potential_private_materialized_write_bits']-ad['potential_private_materialized_write_bits']-md['potential_private_materialized_write_bits'],
            'boundary':'no predictor, DMA stalls, memory port conflicts, frequency or AEE included'}


def torch_transpose_check():
    import torch
    torch.set_num_threads(2)
    assert not torch.cuda.is_available()
    out=[]
    for h,w in [(2,3),(3,4),(9,12),(15,20)]:
        x=torch.ones(1,2,h,w,dtype=torch.float64)
        m=torch.nn.ConvTranspose2d(2,3,3,stride=2,padding=1,output_padding=1,bias=False,dtype=torch.float64)
        with torch.no_grad():m.weight.fill_(1); y=m(x)
        oh,ow,taps=valid_transpose_taps(h,w)
        assert y.shape[-2:]==(oh,ow)
        assert int(y.sum())==taps*2*3
        # Explicit scatter using real Torch tensor values and weights, independent
        # of the convolution backend; binary exact ones first, random FP second.
        torch.manual_seed(20261011)
        xr=torch.randn_like(x);wr=torch.randn_like(m.weight)
        with torch.no_grad():m.weight.copy_(wr);yr=m(xr)
        reference=torch.zeros_like(yr)
        for iy in range(h):
            for ix in range(w):
                for ky in range(3):
                    for kx in range(3):
                        oy,ox=2*iy-1+ky,2*ix-1+kx
                        if 0<=oy<oh and 0<=ox<ow:
                            reference[0,:,oy,ox]+=xr[0,:,iy,ix]@wr[:,:,ky,kx]
        err=float((reference-yr).abs().max())
        assert err<1e-12
        out.append({'input_HW':[h,w],'output_HW':[oh,ow],'Cin':2,'Cout':3,
                    'torch_ones_output_sum':int(y.sum()),'valid_scalar_products':taps*6,
                    'old_output_HW_times_9_proxy':oh*ow*9*6,
                    'random_FP64_scatter_max_error':err})
    return {'torch_version':torch.__version__,'device':'CPU','cases':out}


def config_audit(label,H,W,win,engine,include_mask_cases=True):
    retained,ret_ops=retained_network_services(H,W,engine)
    stage_data=[];graphs={};base_service=total(retained);base_products=total(retained,'scalar_products')
    packed_engine=Engine(engine.pk,engine.pn,engine.real_ii,engine.spike_ii,True)
    packed_baseline=total(retained)
    base_by_part={'uncancelled_front_merge_decoder_heads':stats(retained)}
    for si,(c,h,dep) in enumerate(zip(CHANNELS,HEADS,DEPTHS)):
        sh,sw=H//(4*2**si),W//(4*2**si)
        variants=[]
        for shifted in [False,True]:
            g=dependency_graph(sh,sw,c,h,win,shifted);graphs[(si,shifted)]=g
            ones=np.ones((sh,sw),dtype=bool)
            a,ad,_=attention_services(g,ones,engine);m,md=mlp_services(sh,sw,c,ones,engine)
            packed_a,_,_=attention_services(g,ones,packed_engine)
            local=(len(g['components'])>1)
            masks=[]
            if include_mask_cases:
                ncomp=len(g['components'])
                defs=[('all_open',np.zeros_like(ones)),('all_closed_diagnostic',ones),
                      ('one_legal_group',closure_mask(g,[0])),
                      ('alternate_legal_groups',closure_mask(g,list(range(0,ncomp,2)))),
                      ('one_nominal_tile_not_promised_closed',g['grid']==0)]
                for n,b in defs:masks.append(mask_record(g,n,b,engine,True))
                yy,xx=np.meshgrid(np.arange(sh),np.arange(sw),indexing='ij')
                # MLP-only uses physical unshifted 9x9 tiles, all T10,
                # independently of attention window9/window15.
                mlpgrid=(yy//MLP_TILE)*ceildiv(sw,MLP_TILE)+(xx//MLP_TILE)
                for n,b in [('all_open',np.zeros_like(ones)),('all_closed',ones),
                            ('one_spatial_tile',mlpgrid==0),('alternate_spatial_tiles',mlpgrid%2==0)]:
                    masks.append(mask_record(g,n,b,engine,False))
            variant={'shifted':shifted,'actual_shift':list(g['shift']),
                     'padded_DHW':list(g['padded']),'real_output_time_rows':T*sh*sw,
                     'padded_partition_time_rows':2*g['pairs_total'],
                     'space_windows':g['ng'],'joint_components':g['components'],
                     'joint_component_sizes':dict(collections.Counter(len(z) for z in g['components'])),
                     'local_joint_allowed_in_candidate_policy':local,
                     'baseline_attention_producers':ad,'baseline_MLP':md,
                     'baseline_attention_services':aggregate_ops(a),'baseline_MLP_services':aggregate_ops(m),
                     'baseline_attention_summary':stats(a),'baseline_MLP_summary':stats(m),
                     'cross_group_packed_dense_attention_issue_steps':total(packed_a),
                     'Q_gate_off_diagonal_T_dependency_labels_match_K_path':True,
                     'mask_cases':masks}
            variants.append(variant)
        blocks=[]
        for bi in range(dep):
            variant=variants[bi%2]
            sv=variant['baseline_attention_summary']['static_issue_steps']+variant['baseline_MLP_summary']['static_issue_steps']
            products=variant['baseline_attention_summary']['scalar_products']+variant['baseline_MLP_summary']['scalar_products']
            base_service+=sv;base_products+=products
            packed_baseline+=variant['cross_group_packed_dense_attention_issue_steps']+variant['baseline_MLP_summary']['static_issue_steps']
            blocks.append({'block':bi,'variant_index':bi%2,'static_issue_steps':sv,'scalar_products':products})
        stage_data.append({'stage':si,'feature_HWC':[sh,sw,c],'heads':h,'depth':dep,'blocks':blocks,'variants':variants})

    policies={}
    fractions=[0,.1,.25,.5,.75,1.0]
    for policy in ['joint_local_only','MLP_only_all_stages','hybrid_joint_local_MLP_other_stages']:
        points=[]
        for fraction in fractions:
            kept=total(retained);cancelled_pixels=0;candidate_pixels=0;mask_units=0;segments=0
            kept_products=total(retained,'scalar_products')
            for si,(c,heads,dep) in enumerate(zip(CHANNELS,HEADS,DEPTHS)):
                for bi in range(dep):
                    g=graphs[(si,bool(bi%2))];ones=np.ones((g['H'],g['W']),dtype=bool)
                    islocal=len(g['components'])>1
                    usejoint=(policy!='MLP_only_all_stages' and islocal)
                    usemlp=(policy=='MLP_only_all_stages' or (policy=='hybrid_joint_local_MLP_other_stages' and not islocal))
                    if usejoint:
                        n=len(g['components']);k=min(n,int(math.floor(fraction*n+1e-12)))
                        # Stable evenly spaced groups; not an accuracy-aware policy.
                        picks=np.linspace(0,n-1,k,dtype=int).tolist() if k else []
                        bypass=closure_mask(g,picks);mask_units+=n
                        a,ad,_=attention_services(g,~bypass,engine)
                        m,md=mlp_services(g['H'],g['W'],c,~bypass,engine)
                        cancelled_pixels+=int(bypass.sum());candidate_pixels+=g['H']*g['W']
                        segments+=ad['bypassed_output_address_runs']['count']
                    elif usemlp:
                        yy,xx=np.meshgrid(np.arange(g['H']),np.arange(g['W']),indexing='ij')
                        grid=yy//MLP_TILE*ceildiv(g['W'],MLP_TILE)+xx//MLP_TILE;n=int(grid.max())+1
                        k=min(n,int(math.floor(fraction*n+1e-12)))
                        picks=np.linspace(0,n-1,k,dtype=int).tolist() if k else []
                        bypass=np.isin(grid,picks);mask_units+=n
                        a,ad,_=attention_services(g,ones,engine)
                        m,md=mlp_services(g['H'],g['W'],c,~bypass,engine)
                        cancelled_pixels+=int(bypass.sum());candidate_pixels+=g['H']*g['W']
                        segments+=md['bypassed_output_address_runs']['count']
                    else:
                        a,ad,_=attention_services(g,ones,engine)
                        m,md=mlp_services(g['H'],g['W'],c,ones,engine)
                    kept+=total(a+m);kept_products+=total(a+m,'scalar_products')
            saved=base_service-kept
            points.append({'nominal_group_fraction_to_bypass':fraction,
                'actual_fraction_of_candidate_spatial_sites_bypassed':cancelled_pixels/float(candidate_pixels) if candidate_pixels else 0,
                'kept_static_issue_steps':kept,'saved_static_issue_steps':saved,
                'compute_service_reduction_fraction':saved/float(base_service),
                'reduction_vs_cross_group_packed_dense_baseline':(packed_baseline-kept)/float(packed_baseline),
                'saved_semantic_scalar_products':base_products-kept_products,
                'predictor_plus_extra_control_headroom_for_10pct_issue_reduction':saved-.1*base_service,
                'mask_units_to_score_or_decode':mask_units,
                'bypassed_NHWC_token_run_segments':segments,
                'scope':'compute issue template only; headroom is not a measured scheduler budget and excludes memory timing'})
        maximum=points[-1]['compute_service_reduction_fraction']
        policies[policy]={'points':points,'ideal_zero_overhead_max_issue_reduction_fraction':maximum,
            'max_reduction_vs_cross_group_packed_dense_baseline':points[-1]['reduction_vs_cross_group_packed_dense_baseline'],
            'ideal_at_least_fraction_of_cancellable_service_to_remove_for_10pct':.1/maximum if maximum else None,
            'theoretical_10pct_possible_in_this_issue_template':maximum>=.1,
            'not_a_quality_result':True}
    return {'name':label,'input_HWT':[H,W,T],'batch':1,'window':[2,win,win],
            'MLP_only_spatial_tile':[MLP_TILE,MLP_TILE],
            'closure_scope':'sufficient producer closure for full-channel paired-T Q/K cancellation; does not prove impossibility of finer legal schedules',
            'window_status':'paper crop recipe' if H==288 else ('local configured full-shape alternative; not claimed paper full recipe' if win==9 else 'paper full-resolution fine-tuning window contract; checkpoint not loaded'),
            'engine':engine.spec(),'baseline_static_issue_steps':base_service,
            'cross_group_packed_dense_baseline_issue_steps':packed_baseline,
            'dense_baseline_caveat':'component-reset baseline isolates the same map; cross-group packed comparator removes attention component tail overhead only and is not a proof of globally optimal dense hardware',
            'baseline_semantic_scalar_products':base_products,
            'retained_whole_network_ops':aggregate_ops(retained),
            'retained_conv_geometry':ret_ops,'retained_summary':stats(retained),
            'stages':stage_data,'policies':policies,
            'outside_compute_template':['predictor/risk score','mask index/control beyond listed units','SRAM/DRAM ports and stalls','clock frequency/RTL timing','input voxelization/normalization outside model','final nearest-neighbor output materialization and other tensor copies'],
            'dead_padding_baseline_rule':'both schedules omit output projections discarded by unpad; QK padded input lanes remain whenever any real output consumes them',
            'current_stat_BN':'all local whole-producer cancellation results are inapplicable without an independently validated normalization change'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pk',type=int,default=16)
    parser.add_argument('--pn',type=int,default=16)
    parser.add_argument('--real-ii',type=int,default=1)
    parser.add_argument('--spike-ii',type=int,default=1)
    parser.add_argument('--output',default=str(HERE/'execution_services_results.json'))
    args=parser.parse_args()
    eng=Engine(args.pk,args.pn,args.real_ii,args.spike_ii)
    convcheck=torch_transpose_check()
    results=[]
    for label,H,W,window in [('crop_288x384_window9',288,384,9),
                              ('full_480x640_window9_current_config',480,640,9),
                              ('full_480x640_window15_paper_finetune',480,640,15)]:
        result=config_audit(label,H,W,window,eng)
        results.append(result)
        print(label,'baseline issue steps',result['baseline_static_issue_steps'],flush=True)
        for name,p in result['policies'].items():
            print(' ',name,p['ideal_zero_overhead_max_issue_reduction_fraction'],flush=True)
    # Resource sensitivity regenerates real shape counts (no frequency claims).
    sensitivities=[]
    for pk,pn,rii in [(8,16,1),(32,16,1),(16,16,2)]:
        e=Engine(pk,pn,rii,1)
        r=config_audit('paper_full15_sensitivity',480,640,15,e,False)
        sensitivities.append({'engine':e.spec(),'baseline_static_issue_steps':r['baseline_static_issue_steps'],
                              'policies':r['policies']})
    source_files=['models/STSwinNet_SNN/Spiking_modules.py',
                  'models/STSwinNet_SNN/Spiking_swin_transformer3D.py',
                  'models/STSwinNet_SNN/Spiking_submodules.py',
                  'models/STSwinNet_SNN/SNN_models.py',
                  'models/STSwinNet_SNN/Spiking_STSwinNet.py',
                  'models/STSwinNet/STSwinNet.py',
                  'configs/train_DSEC_supervised_SDformerFlow_en4.yml']
    result={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'CPU_only':True,'no_model_training':True,'no_checkpoint':True,'old_code_modified':False,
        'script_sha256':sha(pathlib.Path(__file__)),'AST_layout_adapter_sha256':sha(layout_path),
        'source_sha256':{rel:sha(SOURCE/rel) for rel in source_files},
        'paper_window_source':'https://arxiv.org/html/2409.04082v1#S3.SS3.SSS2',
        'provenance_reference':'../v2_window_layout_results.json',
        'ConvTranspose2d_CPU_validation':convcheck,'configurations':results,
        'paper_full15_resource_sensitivity':sensitivities,
        'top_level_warning':'STATIC ISSUE-LOOP TEMPLATE, NOT A CALIBRATED CYCLE SIMULATOR, GPU BENCHMARK, RTL RESULT OR AEE RESULT'}
    pathlib.Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Wrote',args.output,flush=True)


if __name__=='__main__':main()
