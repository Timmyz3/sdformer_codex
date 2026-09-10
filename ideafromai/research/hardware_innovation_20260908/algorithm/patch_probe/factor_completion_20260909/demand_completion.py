"""One common prefix completion policy plus actual latent dependencies.

Statistics come from each factor student's own train outputs. Future Y only
defines that student's complete label/fallback, never its prefix decision.
Counts are scalar operations / logical J4 coefficient requests, not cycles.
"""
from __future__ import annotations

import torch

from train_factors import batch, forward_membrane

PREFIX = (2,3,7)
GAMMA = 3.0


@torch.no_grad()
def calibrate(model, data, constants, device, source_theta):
    was_training = model.training
    model.eval()
    values = []
    for start in range(0,len(data['y']),32):
        ids = torch.arange(start,min(start+32,len(data['y'])))
        item = batch(data,ids,device,source_theta)
        y,_ = forward_membrane(model,item,constants)
        values.append(y)
    y = torch.cat(values)
    mean = y.mean((0,2))  # S,H
    centered = y-mean[None,:,None,:]
    covariance = torch.einsum('gsph,guph->hsu',centered,centered)/(y.shape[0]*y.shape[2])
    model.train(was_training)
    return dict(mean=mean.detach(), covariance=covariance.detach())


def completion(model,item,constants,moments,soft=False):
    y,full = forward_membrane(model,item,constants)
    device = y.device
    prefix = torch.zeros(10,dtype=torch.bool,device=device)
    prefix[list(PREFIX)] = True
    empty = item['source'].eq(0).all(-1)  # G,S,P; known without producing Y
    masks = model.masks(soft)[item['regions']]
    scalar_masks = masks.repeat_interleave(model.latent_tile,dim=1)
    hard_masks = model.masks()[item['regions']]
    live_h = (hard_masks.repeat_interleave(model.latent_tile,dim=1) @ model.connectivity.float()).gt(0)
    known = empty | prefix[None,:,None]
    # Read actual Y only at the already-produced prefix. Empty-source Y is
    # the original fixed BN bias; no dense future reference is consulted.
    nominal = torch.where(prefix[None,:,None,None], y,
        torch.where(empty[...,None],constants['bn_bias'],moments['mean'][None,:,None,:]))
    predicted = torch.einsum('ts,gsph->gtph',constants['a'],nominal)+constants['b'][None,:,None,None]-constants['theta']
    remaining = constants['a'][None,None,:,:]*(~known).permute(0,2,1)[:,:,None,:]
    variance = torch.einsum('gpts,hsu,gptu->gtph',remaining,moments['covariance'],remaining).clamp_min(0.)
    sigma = variance.sqrt()
    # Entire inactive H channels are compile-time constants for a region;
    # this ordinary simplification is granted to every layout.
    constant = constants['a'].sum(1)[:,None]*constants['bn_bias'][None,:]+constants['b'][:,None]-constants['theta']
    predicted = torch.where(live_h[:,None,None,:],predicted,constant[None,:,None,:])
    sigma = torch.where(live_h[:,None,None,:],sigma,torch.zeros_like(sigma))
    accept = predicted.abs() >= GAMMA*sigma
    probability = torch.sigmoid((predicted.abs()-GAMMA*sigma)/(0.25*sigma).clamp_min(0.025))
    probability = torch.where(sigma.eq(0),torch.ones_like(probability),probability)
    continuation = 1-probability if soft else (~accept).float()
    e = constants['a'].ne(0)
    # G,t,P,H -> G,s,P,H. Max is a differentiable OR relaxation in training.
    need_y_tail = (continuation[...,None]*e[None,:,None,None,:]).amax(1).permute(0,3,1,2)
    need_y_tail = need_y_tail*(~prefix)[None,:,None,None]*(~empty)[...,None]*live_h[:,None,None,:]
    need_y_prefix = prefix[None,:,None,None]*(~empty)[...,None]*live_h[:,None,None,:]
    # Each J4 tile has a common support in all layouts in this experiment.
    support = model.connectivity.reshape(-1,model.latent_tile,96).any(1)
    need_z_tail = (need_y_tail[...,None,:]*support[None,None,None,:,:]).amax(-1)*masks[:,None,None,:]
    need_z_prefix = (need_y_prefix[...,None,:]*support[None,None,None,:,:]).amax(-1)*masks[:,None,None,:]
    gate = torch.where(accept,predicted.ge(0),full.ge(0))
    return dict(y=y,full=full,predicted=predicted,accept=accept,accept_probability=probability,gate=gate,
        empty=empty,masks=masks,scalar_masks=scalar_masks,live_h=live_h,
        z_prefix=need_z_prefix,z_tail=need_z_tail,y_prefix=need_y_prefix,y_tail=need_y_tail)


def coefficient_requests(source,need,u_tile_nonzero):
    """Actual OR over T/P for each G,J4,K; soft max only during training."""
    g,_,_,k = source.shape
    source = source.ne(0).reshape(g,-1,k)
    need = need.reshape(g,-1,need.shape[-1]).transpose(1,2)
    # K chunks bound the temporary; they do not imply hardware scheduling.
    result = []
    for start in range(0,k,108):
        end = min(start+108,k)
        value = (need[...,None]*source[:,None,:,start:end]).amax(2)
        result.append(value*u_tile_nonzero[None,:,start:end])
    return torch.cat(result,-1)


def request_loss(model,item,state):
    nz = model.u.detach().ne(0).T.reshape(-1,model.latent_tile,model.u.shape[0]).any(1)
    prefix = coefficient_requests(item['source'],state['z_prefix'],nz)
    tail = coefficient_requests(item['source'],state['z_tail'],nz)
    # All models have 48 live scalar latents = 12 J4 slots. Denominator is
    # one ordinary all-T scan, including actual source-empty suppression.
    denominator = item['source'].ne(0).any((1,2)).sum()*model.keep
    return (prefix.sum()+tail.sum())/denominator.clamp_min(1)


@torch.no_grad()
def measure(model,data,constants,moments,device,source_theta):
    model.eval()
    sums = dict(gates=0,teacher_positive=0,full_false_positive=0,full_false_negative=0,
        early_false_positive=0,early_false_negative=0,early_vs_own_full=0,accepted=0,
        prefix_u_J4_requests=0,tail_u_J4_requests=0,full_u_J4_requests=0,
        prefix_u_scalar_terms=0,tail_u_scalar_terms=0,full_u_scalar_terms=0,
        prefix_v_continuous_terms=0,tail_v_continuous_terms=0,full_v_continuous_terms=0,
        full_latent_time_position_slots=0,needed_latent_time_position_slots=0,
        prefix_source_descriptors=0,tail_source_descriptors=0,full_source_descriptors=0,
        predicate_comparisons=0,full_psn_nonzero_terms=0,early_psn_nonzero_terms=0)
    u_nz = model.u.ne(0).float()
    u_tile = u_nz.T.reshape(-1,model.latent_tile,model.u.shape[0]).any(1)
    v_nz = (model.v*model.connectivity).ne(0).float()
    a_nz = constants['a'].ne(0)
    prefix_cols = torch.tensor(PREFIX,device=device)
    tail_counts = a_nz.sum(1)-a_nz[:,prefix_cols].sum(1)
    for start in range(0,len(data['y']),8):
        ids = torch.arange(start,min(start+8,len(data['y'])))
        item = batch(data,ids,device,source_theta)
        state = completion(model,item,constants,moments)
        source = item['source'].ne(0)
        truth = item['target'].ge(0)
        full_gate = state['full'].ge(0)
        sums['gates'] += truth.numel()
        sums['teacher_positive'] += int(truth.sum())
        sums['full_false_positive'] += int((full_gate&~truth).sum())
        sums['full_false_negative'] += int((~full_gate&truth).sum())
        sums['early_false_positive'] += int((state['gate']&~truth).sum())
        sums['early_false_negative'] += int((~state['gate']&truth).sum())
        sums['early_vs_own_full'] += int((state['gate']!=full_gate).sum())
        sums['accepted'] += int(state['accept'].sum())
        sums['predicate_comparisons'] += truth.numel()+int((~state['accept']).sum())
        full_z = state['masks'][:,None,None,:]*(~state['empty'])[...,None]
        phase_z = [('prefix',state['z_prefix']),('tail',state['z_tail']),('full',full_z)]
        z = (item['source']@model.u)*state['scalar_masks'][:,None,None,:]
        # A sparse source can cancel exactly in a continuous Z; its zero is
        # available only after producing that Z, and benefits every V path.
        z_nz = z.ne(0).float()
        source_products = source.float()@u_nz
        for phase,need in phase_z:
            requests = coefficient_requests(item['source'],need,u_tile)
            sums[phase+'_u_J4_requests'] += int(requests.sum())
            scalar_need = need.repeat_interleave(model.latent_tile,dim=-1)
            sums[phase+'_u_scalar_terms'] += int((source_products*scalar_need).sum(dtype=torch.float64))
            yn = (state['y_'+phase] if phase!='full' else
                (~state['empty'])[...,None]*state['live_h'][:,None,None,:])
            sums[phase+'_v_continuous_terms'] += int(torch.einsum('gspr,rh,gsph->',z_nz*scalar_need,v_nz,yn.float()))
            any_need = need.amax(-1)
            descriptor = (source*any_need[...,None]).any((1,2))
            sums[phase+'_source_descriptors'] += int(descriptor.sum())
        sums['full_latent_time_position_slots'] += int(full_z.sum())*model.latent_tile
        sums['needed_latent_time_position_slots'] += int((state['z_prefix']+state['z_tail']).sum())*model.latent_tile
        positions = truth.shape[0]*truth.shape[2]*truth.shape[3]
        sums['full_psn_nonzero_terms'] += positions*int(a_nz.sum())
        # Keep the scalar addition outside a conditional expression: one
        # prefix projection and actual unresolved tail rows, once each.
        sums['early_psn_nonzero_terms'] += positions*int(a_nz[:,prefix_cols].sum())+int(((~state['accept'])*tail_counts[None,:,None,None]).sum())
    result = dict(**sums,
        own_full_error=(sums['full_false_positive']+sums['full_false_negative'])/sums['gates'],
        mixed_teacher_error=(sums['early_false_positive']+sums['early_false_negative'])/sums['gates'],
        mixed_vs_own_full=sums['early_vs_own_full']/sums['gates'],
        accepted_fraction=sums['accepted']/sums['gates'],
        U_request_ratio=(sums['prefix_u_J4_requests']+sums['tail_u_J4_requests'])/sums['full_u_J4_requests'],
        U_scalar_ratio=(sums['prefix_u_scalar_terms']+sums['tail_u_scalar_terms'])/sums['full_u_scalar_terms'],
        V_continuous_ratio=(sums['prefix_v_continuous_terms']+sums['tail_v_continuous_terms'])/max(sums['full_v_continuous_terms'],1),
        latent_slot_ratio=sums['needed_latent_time_position_slots']/sums['full_latent_time_position_slots'],
        count_scope='sampled P4; J4 logical U coefficient requests after source/W-zero OR, separate prefix/tail scans; no cycles or DRAM claim')
    return result
