"""Fixed disjoint-latent preview/conditional-tail experiment, CPU reference.

The two R56 layouts both execute shared32 first, then at most the disjoint
remaining24. Source theta is retained; BN affine and full common3 T10 are
unchanged. Statistical completion is lossy, and dense training is not a
hardware timing model. Only this directory is written by this experiment.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
FACTOR = HERE.parent
sys.path.insert(0, str(FACTOR))
from factor_reference import svd_factors
from train_factors import load_data, batch, PARTIAL
from demand_completion import coefficient_requests

SHARED = 32
J = 2
GAMMA = 3.
AXES = ('shared56', 'shared32_private2')


class LatentModel(nn.Module):
    def __init__(self, u, v, connectivity, structure):
        super().__init__()
        self.u = nn.Parameter(torch.as_tensor(u, dtype=torch.float32).clone())
        self.v = nn.Parameter(torch.as_tensor(v, dtype=torch.float32).clone())
        self.register_buffer('connectivity', torch.as_tensor(connectivity, dtype=torch.bool).clone())
        self.structure = structure

    def parts(self, source):
        z = source @ self.u
        v = self.v*self.connectivity
        shared = z[..., :SHARED] @ v[:SHARED]
        tail = z[..., SHARED:] @ v[SHARED:]
        return z, shared, tail

    def export(self):
        r = self.u.shape[1]
        # All columns exist everywhere: no learned or position-specific mask.
        return dict(u=self.u.detach().numpy(), v=(self.v*self.connectivity).detach().numpy(),
            connectivity=self.connectivity.numpy(), structure=np.array(self.structure),
            shared_rank=np.array(SHARED), latent_tile=np.array(J), active_tiles=np.array(r//J),
            masks=np.ones((1, r//J), dtype=bool),
            stage_rule=np.array('all T10 shared32; disjoint tail columns gated by unresolved consumers'))


def initialize(weight, axis):
    if axis == 'shared56':
        u, v = svd_factors(weight, 56)
        return LatentModel(u, v, np.ones_like(v, bool), axis)
    u = np.zeros((864, 56), np.float64)
    v = np.zeros((56, 96), np.float64)
    support = np.zeros_like(v, bool)
    u[:, :SHARED], v[:SHARED] = svd_factors(weight, SHARED)
    support[:SHARED] = True
    residual = weight-u[:, :SHARED]@v[:SHARED]
    for h8 in range(12):
        hs, js = slice(h8*8, h8*8+8), slice(SHARED+2*h8, SHARED+2*h8+2)
        u[:, js], v[js, hs] = svd_factors(residual[:, hs], 2)
        support[js, hs] = True
    return LatentModel(u, v, support, axis)


def read_model(filename, structure=None):
    with np.load(filename) as p:
        return LatentModel(p['u'], p['v'], p['connectivity'],
            structure or str(p['structure'].item()))


def forward(model, item, constants):
    z, shared, tail = model.parts(item['source'])
    # Canonical complete student: sum both raw Conv factors, then original
    # fixed BN, then the same saved A/b/theta. This is a new FP32 function.
    y = (shared+tail)*constants['bn_scale']+constants['bn_bias']
    full = torch.einsum('ts,gsph->gtph', constants['a'], y)
    full = full+constants['b'][None, :, None, None]-constants['theta']
    return z, shared, tail, y, full


@torch.no_grad()
def calibrate(model, data, constants, source_theta):
    model.eval()
    residuals = []
    for start in range(0, len(data['y']), 32):
        item = batch(data, torch.arange(start, min(start+32, len(data['y']))), 'cpu', source_theta)
        _, _, tail = model.parts(item['source'])
        residuals.append(tail*constants['bn_scale'])
    residual = torch.cat(residuals)
    mean = residual.mean((0, 2))
    centered = residual-mean[None, :, None, :]
    covariance = torch.einsum('gsph,guph->hsu', centered, centered)/(residual.shape[0]*4)
    return dict(mean=mean, covariance=covariance)


def predict(shared, empty, constants, moments):
    # This function has no private value argument. Empty-source private
    # residual is exactly zero; fixed BN bias belongs only to the preview.
    preview_y = shared*constants['bn_scale']+constants['bn_bias']
    base = torch.einsum('ts,gsph->gtph', constants['a'], preview_y)
    base = base+constants['b'][None, :, None, None]-constants['theta']
    nominal = moments['mean'][None, :, None, :]*(~empty)[..., None]
    predicted = base+torch.einsum('ts,gsph->gtph', constants['a'], nominal)
    remaining = constants['a'][None, None]*(~empty).permute(0, 2, 1)[:, :, None, :]
    variance = torch.einsum('gpts,hsu,gptu->gtph', remaining,
                            moments['covariance'], remaining).clamp_min(0.)
    sigma = variance.sqrt()
    accept = predicted.abs() >= GAMMA*sigma
    probability = torch.sigmoid((predicted.abs()-GAMMA*sigma)/(0.25*sigma).clamp_min(.025))
    probability = torch.where(sigma.eq(0), torch.ones_like(probability), probability)
    return base, predicted, accept, probability


def completion(model, item, constants, moments, soft=False):
    z, shared, tail, y, full = forward(model, item, constants)
    empty = item['source'].eq(0).all(-1)
    base, predicted, accept, probability = predict(shared, empty, constants, moments)
    unresolved = 1-probability if soft else (~accept).float()
    e = constants['a'].ne(0)
    need_y = (unresolved[..., None]*e[None, :, None, None, :]).amax(1).permute(0, 3, 1, 2)
    need_y = need_y*(~empty)[..., None]
    support = (model.v.detach()*model.connectivity).ne(0)[SHARED:].reshape(-1, J, 96).any(1)
    need_z = (need_y[..., None, :]*support[None, None, None]).amax(-1)
    gate = torch.where(accept, predicted.ge(0), full.ge(0))
    return dict(z=z, shared=shared, tail=tail, y=y, full=full, base=base,
        predicted=predicted, accept=accept, accept_probability=probability,
        gate=gate, empty=empty, need_y=need_y, need_z=need_z, unresolved=unresolved)


def request_loss(model, item, state):
    # Prefix and tail are disjoint U columns. Each J2 coefficient vector
    # is requested at most once over all source times and native P4.
    nz = model.u.detach().ne(0).T.reshape(-1, J, 864).any(1)
    base_need = (~state['empty'])[..., None].expand(-1, -1, -1, SHARED//J)
    prefix = coefficient_requests(item['source'], base_need, nz[:SHARED//J])
    tail = coefficient_requests(item['source'], state['need_z'], nz[SHARED//J:])
    full_need = (~state['empty'])[..., None].expand(-1, -1, -1, len(nz))
    denominator = coefficient_requests(item['source'], full_need, nz).sum()
    return (prefix.sum()+tail.sum())/denominator.clamp_min(1)


@torch.no_grad()
def measure(model, data, constants, moments, source_theta):
    model.eval()
    r = model.u.shape[1]
    sums = {key: 0 for key in (
        'gates teacher_positive own_full_FP own_full_FN mixed_FP mixed_FN mixed_vs_own_full '
        'accepted U_shared_requests U_tail_requests U_full_requests U_shared_adds U_tail_adds U_full_adds '
        'full_source_live_words shared_source_live_words tail_source_live_words '
        'tail_used_source_descriptors full_source_dense_words shared_source_dense_words tail_source_dense_words '
        'full_Z_values shared_Z_values tail_Z_values preview_table_pairs preview_comparisons final_comparisons '
        'dependency_A_edges dependency_V_edges').split()}
    orders = {order: {key: 0 for key in ('shared_V tail_V full_V shared_A tail_A full_A').split()}
              for order in ('V_then_A', 'A_then_V')}
    u_nz = model.u.ne(0).float()
    u_word_nz = u_nz.T.reshape(-1, J, 864).any(1)
    v = model.v*model.connectivity
    v_nz = v.ne(0).float()
    a, e = constants['a'], constants['a'].ne(0)
    snapshots = dict(full=[], mixed=[], accepted=[])
    peak = dict(tail_z_values=0, unresolved_gates=0, tail_U_J2_words=0)
    for start in range(0, len(data['y']), 8):
        item = batch(data, torch.arange(start, min(start+8, len(data['y']))), 'cpu', source_theta)
        state = completion(model, item, constants, moments)
        source = item['source'].ne(0)
        truth, full_gate, gate = item['target'].ge(0), state['full'].ge(0), state['gate']
        sums['gates'] += truth.numel()
        sums['teacher_positive'] += int(truth.sum())
        for name, value in (('own_full', full_gate), ('mixed', gate)):
            sums[name+'_FP'] += int((value&~truth).sum())
            sums[name+'_FN'] += int((~value&truth).sum())
        sums['mixed_vs_own_full'] += int((gate!=full_gate).sum())
        sums['accepted'] += int(state['accept'].sum())
        sums['preview_table_pairs'] += truth.numel()
        sums['preview_comparisons'] += 2*truth.numel()
        sums['final_comparisons'] += int((~state['accept']).sum())
        base_need = (~state['empty'])[..., None].expand(-1, -1, -1, SHARED//J)
        full_need = (~state['empty'])[..., None].expand(-1, -1, -1, r//J)
        needs = (('shared', base_need, slice(0, SHARED)),
                 ('tail', state['need_z'], slice(SHARED, r)),
                 ('full', full_need, slice(0, r)))
        for phase, need, rs in needs:
            req = coefficient_requests(item['source'], need, u_word_nz[rs.start//J:rs.stop//J])
            sums['U_'+phase+'_requests'] += int(req.sum())
            scalar_need = need.repeat_interleave(J, -1)
            adds = source.float()@u_nz[:, rs]
            sums['U_'+phase+'_adds'] += int((adds*scalar_need).sum(dtype=torch.float64))
            sums[phase+'_Z_values'] += int(scalar_need.sum())
            if phase == 'tail':
                peak['tail_z_values'] = max(peak['tail_z_values'], int(scalar_need.sum((1,2,3)).max()))
                peak['tail_U_J2_words'] = max(peak['tail_U_J2_words'], int(req.sum((1,2)).max()))
        live_words = source.any((1,2))
        tail_any = state['need_z'].bool().any((1,2,3))
        used = (source*state['need_z'].bool().any(-1)[..., None]).any((1,2))
        sums['full_source_live_words'] += int(live_words.sum())
        sums['shared_source_live_words'] += int(live_words.sum())
        # A sparse K directory can skip all-zero words. It cannot know that
        # a live word misses the new tail demand without reading that word.
        sums['tail_source_live_words'] += int((live_words*tail_any[:, None]).sum())
        sums['tail_used_source_descriptors'] += int(used.sum())
        sums['full_source_dense_words'] += source.shape[0]*864
        sums['shared_source_dense_words'] += source.shape[0]*864
        sums['tail_source_dense_words'] += int(tail_any.sum())*864
        peak['unresolved_gates'] = max(peak['unresolved_gates'], int((~state['accept']).sum((1,2,3)).max()))
        z_nz = state['z'].ne(0).float()
        remaining = (~state['accept']).float()
        full_y_need = (~state['empty'])[..., None].float().expand(-1,-1,-1,96)
        # V -> A: tail Y is materialized only where an unresolved row needs
        # that source-time/output channel. Exact zeros after Z/V help all axes.
        for phase, rs, y_need, row_need in (
            ('shared', slice(0,SHARED), full_y_need, torch.ones_like(remaining)),
            ('tail', slice(SHARED,r), state['need_y'], remaining),
            ('full', slice(0,r), full_y_need, torch.ones_like(remaining))):
            orders['V_then_A'][phase+'_V'] += int(torch.einsum('gspr,rh,gsph->', z_nz[...,rs],v_nz[rs],y_need))
            raw = state['shared'] if phase=='shared' else state['tail'] if phase=='tail' else state['shared']+state['tail']
            orders['V_then_A'][phase+'_A'] += int(torch.einsum('gsph,ts,gtph->',raw.ne(0).float(),e.float(),row_need))
            # A -> V is also legal for this linear fragment. Its latent
            # temporal transform may densify Q even when original Z is zero.
            q = torch.einsum('ts,gspr->gtpr', a, state['z'][...,rs])
            q_need = (row_need[...,None,:]*v_nz[rs][None,None,None]).amax(-1)
            orders['A_then_V'][phase+'_A'] += int(torch.einsum('gspr,ts,gtpr->',z_nz[...,rs],e.float(),q_need))
            orders['A_then_V'][phase+'_V'] += int(torch.einsum('gtpr,rh,gtph->',q.ne(0).float(),v_nz[rs],row_need))
        sums['dependency_A_edges'] += int((remaining*e.sum(1)[None,:,None,None]).sum())
        sums['dependency_V_edges'] += int(torch.einsum('gsph,rh->',state['need_y'],v_nz[SHARED:]))
        for key, value in (('full',full_gate),('mixed',gate),('accepted',state['accept'])):
            snapshots[key].append(value.cpu().numpy())
    sums.update(full_gate_error=(sums['own_full_FP']+sums['own_full_FN'])/sums['gates'],
        mixed_gate_error=(sums['mixed_FP']+sums['mixed_FN'])/sums['gates'],
        full_FN_rate=sums['own_full_FN']/max(sums['teacher_positive'],1),
        mixed_FN_rate=sums['mixed_FN']/max(sums['teacher_positive'],1),
        mixed_vs_own_full_rate=sums['mixed_vs_own_full']/sums['gates'],
        acceptance=sums['accepted']/sums['gates'],
        U_request_ratio=(sums['U_shared_requests']+sums['U_tail_requests'])/sums['U_full_requests'],
        U_add_ratio=(sums['U_shared_adds']+sums['U_tail_adds'])/sums['U_full_adds'],
        source_word_ratio=(sums['shared_source_live_words']+sums['tail_source_live_words'])/sums['full_source_live_words'],
        U_request_bytes=(sums['U_shared_requests']+sums['U_tail_requests'])*J*4,
        U_full_request_bytes=sums['U_full_requests']*J*4,
        source_word_read_bytes=(sums['shared_source_live_words']+sums['tail_source_live_words'])*8,
        arithmetic_orders=orders, actual_maximum_per_native_P4=peak)
    return sums, {key:np.concatenate(value) for key,value in snapshots.items()}


def storage(model, a):
    r = model.u.shape[1]
    per_row_entries = (2**a.ne(0).sum(1)).tolist()
    return dict(U_coefficients_fp32_bytes=864*r*4,
        V_nonzero_coefficients_fp32_bytes=int((model.v*model.connectivity).ne(0).sum())*4,
        V_dense_slots_fp32_bytes=r*96*4,
        threshold_mean_radius_table_fp32_bytes=sum(per_row_entries)*96*2*4,
        threshold_entries_per_row=per_row_entries,
        source_P4_allT_packed_bytes=864*4*10//8,
        source_P4_allT_64bit_words_bytes=864*8,
        source_live_K_directory_bytes=864//8,
        A_then_V_P4_H96_execution=dict(
            shared_margin_fp32_bytes=10*4*96*4,
            current_J2_Z_plus_Q_fp32_bytes=2*10*4*J*4,
            output_and_unresolved_bits_bytes=2*10*4*96//8,
            tail_need_J2_bits_bytes=10*4*((r-SHARED)//J)//8,
            empty_time_bits_bytes=10*4//8,
            directory_bytes=864//8,
            total_without_coefficients_tables_or_source_cache_bytes=(10*4*96*4+2*10*4*J*4+
                2*10*4*96//8+10*4*((r-SHARED)//J)//8+10*4//8+864//8),
            lifetime='One J2 complete T10 Z and Q at a time; retain H96 preview base through private stage. Accepted gates release base values, but capacity is the worst case.',
            ports='FP32 register/SRAM capacity illustration, no port timing or macro mapping.'),
        V_then_A_P4_H96_execution=dict(
            conservative_two_Y_margin_arrays_fp32_bytes=2*10*4*96*4,
            current_J2_Z_fp32_bytes=10*4*J*4,
            note='Can reduce with a proven in-place A schedule; not silently credited here.'),
        H8_alternative='H8 preview alone is 1280B. Keeping U once across H96 additionally needs shared Z/Q cache (5120B for R32), or explicit U replay across output groups; no free H96 broadcast with only H8 state.',
        input_scan='Directory construction/common source delivery not removed. Without a resident 6912B source-word buffer, used tail groups reread sparse-directory words; used-descriptor intersections are not word-read savings.')


@torch.no_grad()
def functional_check(model, data, constants, moments, source_theta):
    item = batch(data, torch.arange(8), 'cpu', source_theta)
    state = completion(model,item,constants,moments)
    # Produce only allowed private Z. Required outputs are exact same dense
    # reference values; all unneeded private coordinates can be absent.
    z_tail = state['z'][...,SHARED:]*state['need_z'].repeat_interleave(J,-1)
    raw = state['shared']+z_tail@(model.v*model.connectivity)[SHARED:]
    y = raw*constants['bn_scale']+constants['bn_bias']
    partial = torch.einsum('ts,gsph->gtph',constants['a'],y)+constants['b'][None,:,None,None]-constants['theta']
    actual = torch.where(state['accept'],state['predicted'].ge(0),partial.ge(0))
    # Counterfactual private values do not occur in the predictor interface.
    changed = copy.deepcopy(model)
    changed.u[:,SHARED:].mul_(3.)
    other = completion(changed,item,constants,moments)
    q = torch.einsum('ts,gspr->gtpr',constants['a'],state['z'])
    av = q@(model.v*model.connectivity)
    av = av*constants['bn_scale']
    av += (constants['a'].sum(1)[:,None]*constants['bn_bias'][None,:]+
           constants['b'][:,None]-constants['theta'])[None,:,None,:]
    return dict(gates=actual.numel(),
        conditional_missing_tail_gate_differences=int((actual!=state['gate']).sum()),
        acceptance_changed_by_mutating_private_U=int((other['accept']!=state['accept']).sum()),
        predicted_margin_max_change_by_private_U=float((other['predicted']-state['predicted']).abs().max()),
        A_V_reassociation_FP32_gate_differences=int((av.ge(0)!=state['full'].ge(0)).sum()),
        A_V_reassociation_max_margin_difference=float((av-state['full']).abs().max()),
        meaning='CPU FP32 sampled functional checks, not frozen FP32, integer deployment, RTL, or a strict bound proof')


def balanced_losses(state, item, constants):
    expected = item['target'].ge(0)
    weights = torch.where(expected,.5/constants['rate'],.5/(1-constants['rate']))
    temperature = (.25*constants['margin_scale']).clamp_min(.025)
    full = (F.binary_cross_entropy_with_logits(state['full']/temperature,expected.float(),reduction='none')*weights).mean()
    prob = state['accept_probability']*torch.sigmoid(state['predicted']/temperature)
    prob = prob+(1-state['accept_probability'])*torch.sigmoid(state['full']/temperature)
    mixed = (F.binary_cross_entropy(prob.clamp(1e-6,1-1e-6),expected.float(),reduction='none')*weights).mean()
    y = ((state['y']-item['y'])/constants['y_scale']).square().mean()
    return y, full, mixed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=FACTOR.parent/'joint_completion_20260909/full_capture4/capture')
    p.add_argument('--capture',type=Path,default=PARTIAL/'capture.pt')
    p.add_argument('--valid-source',type=Path,default=PARTIAL/'integer_valid10')
    p.add_argument('--operator',type=Path,default=PARTIAL/'shared_column_deployment_source.npz')
    p.add_argument('--temporal',default='common3',choices=['common3'])
    p.add_argument('--steps',type=int,default=256)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--output',type=Path,default=HERE)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(909)
    data,op,a,b,theta,yscale,mscale,rate,groups = load_data(args)
    constants = dict(a=a,b=b,theta=theta,bn_scale=torch.tensor(op['bn_scale'],dtype=torch.float32),
        bn_bias=torch.tensor(op['bn_bias'],dtype=torch.float32),y_scale=yscale,margin_scale=mscale,rate=rate)
    source_theta = float(op['source_theta'])
    args.output.mkdir(parents=True,exist_ok=True)
    result = dict(scope='native64 P4 per frame, train16/valid4; local factor/PSN gates, not network AEE',
        structures=dict(shared56='ordinary all-shared R56, prefix32/tail24',
            shared32_private2='shared32 plus two residual latents for each of twelve H8 groups',
            shared_compact48='unchanged old stage1 R48; ordinary prefix32/tail16; 256 total updates'),
        fixed=dict(shared_rank=32,gamma=GAMMA,latent_word=J,
            A='same common3 full T10 matrix',theta_source=source_theta,theta_output=theta,
            no_spatial_mask=True,rank_sweep=False,gamma_sweep=False),
        training=dict(stage1_steps=args.steps,stage2_steps=args.steps,stage2_request_weights=[0.,.1],
            batch_native_P4=8,Adam_lr=.002,stage1_seed=909,stage2_seed=910,
            stage1_loss='normalized Y MSE + .25 balanced full gate BCE',
            stage2_loss='normalized Y MSE + .125 full balanced BCE + .125 mixed balanced BCE + lambda U_J2_requests/full',
            update='U,V only, allowed V connectivity projected after each step; no A/b/theta optimization',
            statistics='each axis private post-BN contribution train16 mean/covariance, detached and refreshed at 0/64/128/192/end',
            targets='same captured Y and common3 gate; no real Conv2 loss or whole-network GT recovery',
            no_validation_fitting=True),
        policy='All T10 shared preview, one per-gate statistical decision; private need propagated through actual A/V support before first private U request.',
        numeric='new CPU FP32 factor+fixedBN+T10; statistical completion is lossy; alternative A/V work orders are algebraic counts, not identical FP32 rounding or cycles',
        scope_limits='no full network AEE, halo-wide Conv2 work, finite-port service, or PPA',
        source_counter='P4 fullT40bit packed words; 64bit alignment. All-zero K directory charged as metadata; tail used descriptors do not cancel pre-read examination.',
        train=data['train']['files'],valid=data['valid']['files'],groups=groups.tolist(),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
        stage1={},stage2={},reference={})
    (args.output/'definition.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    started=time.monotonic()
    weight=op['weight'].reshape(96,864).T
    batches1=torch.randint(len(data['train']['y']),(args.steps,8),generator=torch.Generator().manual_seed(909))
    batches2=torch.randint(len(data['train']['y']),(args.steps,8),generator=torch.Generator().manual_seed(910))
    def deliver(model,name,stage,history):
        moments=calibrate(model,data['train'],constants,source_theta)
        metrics,snapshots=measure(model,data['valid'],constants,moments,source_theta)
        checks=functional_check(model,data['valid'],constants,moments,source_theta)
        if checks['conditional_missing_tail_gate_differences'] or checks['acceptance_changed_by_mutating_private_U']:
            raise AssertionError('Conditional execution/predictor isolation failed: '+json.dumps(checks))
        arrays=model.export()
        arrays.update(a=a.numpy(),temporal_bias=b.numpy(),theta_source=np.array(source_theta),
            theta_output=np.array(theta),bn_scale=op['bn_scale'],bn_bias=op['bn_bias'],
            completion_mean=moments['mean'].numpy(),completion_covariance=moments['covariance'].numpy(),
            gamma=np.array(GAMMA))
        np.savez_compressed(args.output/(name+'.npz'),**arrays)
        np.savez_compressed(args.output/(name+'_valid_gates.npz'),
            **{key:np.packbits(value,bitorder='little') for key,value in snapshots.items()},
            shape=np.array(snapshots['full'].shape),groups=groups,files=np.array(data['valid']['files']))
        result[stage][name]=dict(valid=metrics,checks=checks,storage=storage(model,a),history=history)
        (args.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print('RESULT',name,json.dumps(dict(gate=metrics['mixed_gate_error'],FN=metrics['mixed_FN_rate'],
              U=metrics['U_request_ratio'],Uadd=metrics['U_add_ratio'],source=metrics['source_word_ratio'])),flush=True)
    # Preserve the original R48 weights while granting the new ordinary
    # disjoint-stage execution. This is a 256-update, not 512-update model.
    compact=read_model(FACTOR/'fit_train16/shared_compact.npz','shared_compact48')
    deliver(compact,'shared_compact48','reference',[])
    for axis in AXES:
        model=initialize(weight,axis)
        optimizer=torch.optim.Adam(model.parameters(),lr=.002)
        history=[]
        for step,ids in enumerate(batches1):
            model.train()
            item=batch(data['train'],ids,'cpu',source_theta)
            _,_,_,y,margin=forward(model,item,constants)
            fit=((y-item['y'])/constants['y_scale']).square().mean()
            expected=item['target'].ge(0)
            weights=torch.where(expected,.5/rate,.5/(1-rate))
            logits=margin/(.25*mscale).clamp_min(.025)
            bce=(F.binary_cross_entropy_with_logits(logits,expected.float(),reduction='none')*weights).mean()
            loss=fit+.25*bce
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            with torch.no_grad(): model.v.mul_(model.connectivity)
            if step%64==0 or step==args.steps-1:
                row=dict(step=step+1,loss=float(loss.detach()),Y=float(fit.detach()),BCE=float(bce.detach()))
                history.append(row);print('RECON',axis,json.dumps(row),flush=True)
        deliver(model,axis+'_reconstruction','stage1',history)
    for axis in AXES:
        for lam in (0.,.1):
            model=read_model(args.output/(axis+'_reconstruction.npz'),axis)
            optimizer=torch.optim.Adam(model.parameters(),lr=.002)
            name=axis+('_lambda0' if lam==0 else '_lambda01')
            history=[]
            for step,ids in enumerate(batches2):
                if step%64==0: moments=calibrate(model,data['train'],constants,source_theta)
                model.train()
                item=batch(data['train'],ids,'cpu',source_theta)
                state=completion(model,item,constants,moments,soft=True)
                fit,full_bce,mixed_bce=balanced_losses(state,item,constants)
                work=request_loss(model,item,state)
                loss=fit+.125*full_bce+.125*mixed_bce+lam*work
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                with torch.no_grad(): model.v.mul_(model.connectivity)
                if step%64==0 or step==args.steps-1:
                    row=dict(step=step+1,loss=float(loss.detach()),Y=float(fit.detach()),full_BCE=float(full_bce.detach()),
                        mixed_BCE=float(mixed_bce.detach()),U_request_ratio=float(work.detach()))
                    history.append(row);print('MIXED',name,json.dumps(row),flush=True)
            deliver(model,name,'stage2',history)
    result['complete']=True
    result['wall_seconds']=time.monotonic()-started
    (args.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('DONE',result['wall_seconds'],flush=True)


if __name__=='__main__':
    main()
