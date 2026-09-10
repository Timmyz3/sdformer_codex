"""Fixed train-moment expressivity diagnostic for three T10 factor families.

No network training/validation/capture. The source P is fixed to the original
all-plus initialization assignment. All cases fit centered covariance loss,
Float64 CPU, Adam lr=.01 for exactly1024 updates, no restarts/best selection.
Each initial B gets its own fixed-P scalar-LS gain; gain then trains jointly.
Bias always analytically matches the original teacher input mean.

Orthogonal case: [[cos(a),sin(a)],[sin(a),-cos(a)]], a0=pi/4, det=-1.
Full case: unconstrained W/sqrt(2), W0=[[1,1],[1,-1]].
Lifting: y0=x0+a*x1; y1=x1+b*y0, a0=b0=0, det=1.
All use exactly the existing four layers of five fixed pairs. Canonical B does
not include source row gains/P, hence zero readout gain does not destroy B.
"""
from __future__ import annotations

import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import math
import time
import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
CHAIN=HERE.parent.parent
sys.path.insert(0,str(CHAIN))
from fast_temporal_basis import MATCHINGS, fit_readout

STEPS=1024
LR=.01
KINDS=('orthogonal20','full_blocks80','lifting40')
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)


def save(path,value):
    def convert(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        if isinstance(x,Path): return str(x)
        raise TypeError(type(x))
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=convert)+'\n')


class Factors(nn.Module):
    def __init__(self,kind):
        super().__init__()
        self.kind=kind
        if kind=='orthogonal20':
            values=torch.full((4,5),math.pi/4)
        elif kind=='full_blocks80':
            values=torch.tensor([[1.,1.],[1.,-1.]]).expand(4,5,2,2).clone()
        else:
            values=torch.zeros((4,5,2))
        self.block_parameters=nn.Parameter(values)
        self.row_gain=nn.Parameter(torch.ones(10))
        self.register_buffer('pairs',torch.tensor(MATCHINGS,dtype=torch.long))

    def blocks(self):
        v=self.block_parameters
        if self.kind=='orthogonal20':
            c,s=torch.cos(v),torch.sin(v)
            return torch.stack([torch.stack([c,s],-1),torch.stack([s,-c],-1)],-2)
        if self.kind=='full_blocks80':
            return v/math.sqrt(2)
        a,b=v[...,0],v[...,1]
        one=torch.ones_like(a)
        return torch.stack([torch.stack([one,a],-1),torch.stack([b,one+a*b],-1)],-2)

    def matrix(self):
        result=torch.eye(10)
        blocks=self.blocks()
        for layer,pair in enumerate(self.pairs):
            i,j=pair[:,0],pair[:,1]
            x,y=result[i],result[j]
            if self.kind=='lifting40':
                a,b=self.block_parameters[layer,:,0,None],self.block_parameters[layer,:,1,None]
                u=x+a*y
                v=y+b*u
            else:
                block=blocks[layer]
                u=block[:,0,0,None]*x+block[:,0,1,None]*y
                v=block[:,1,0,None]*x+block[:,1,1,None]*y
            result=torch.empty_like(result).index_copy(0,i,u).index_copy(0,j,v)
        return result


def inverse_by_factors(kind,parameters,blocks):
    value=np.eye(10)
    for layer in range(3,-1,-1):
        for pair,(i,j) in enumerate(MATCHINGS[layer]):
            x,y=value[i].copy(),value[j].copy()
            if kind=='lifting40':
                a,b=parameters[layer,pair]
                recovered_y=y-b*x
                recovered_x=x-a*recovered_y
                value[i],value[j]=recovered_x,recovered_y
            else:
                inverse=blocks[layer,pair].T if kind=='orthogonal20' else np.linalg.inv(blocks[layer,pair])
                value[i]=inverse[0,0]*x+inverse[0,1]*y
                value[j]=inverse[1,0]*x+inverse[1,1]*y
    return value


def arithmetic(kind,parameters,blocks,gains):
    if kind=='lifting40':
        mul=40
        add=40
        coefficients=parameters.ravel()
        inverse_note='reverse layers and two subtracting lifting steps; same a,b; no online reciprocal'
    else:
        mul=80
        add=40
        coefficients=blocks.ravel()
        inverse_note=('reverse layers, transposed orthogonal blocks'
                      if kind=='orthogonal20' else
                      'reverse layers; precompute each full 2x2 inverse offline, including reciprocal determinant; no runtime division for frozen weights')
    return dict(
        unit='one ten-dimensional vector, frozen real-valued coefficients; no timing/PPA',
        actual_direct_factor_program_forward=dict(scalar_products=mul,add_subtracts=add),
        actual_direct_factor_program_inverse=dict(scalar_products=mul,add_subtracts=add),
        nonzero_nonunit_forward_coefficient_occurrences=int(np.count_nonzero((coefficients!=0)&(np.abs(coefficients)!=1))),
        readout_separate=dict(row_gain_products=int(np.count_nonzero((gains!=0)&(np.abs(gains)!=1))),
                              bias_additions=10,threshold_comparisons=10,
                              note='canonical inverse never divides by these gains; signed/zero-gain threshold folding is a separate compiled control'),
        inverse_rule=inverse_note,
        ordinary_three_product_orthogonal_rewrite=(
            dict(scalar_products=60,add_subtracts=60,
                 rule='per block t=c*(x+y), out0=t+(s-c)*y, out1=(s+c)*x-t; c±s precomputed',
                 numeric_scope='real algebra; new floating/fixed rounding order would need its own functional check')
            if kind=='orthogonal20' else None),
        excludes=['pair-address movement, coefficient fetch, stage/consumer state and ports',
                  'quantization/rounding/peak-bit growth','trigonometric/offline coefficient generation',
                  'continuous-channel consumers and whole network'])


def fit(kind,mean,cov,teacher,bias,P,initial_basis):
    begin=time.monotonic()
    model=Factors(kind)
    B0=model.matrix().detach().numpy()
    initial=fit_readout(mean,cov,teacher,bias,B0,row_permutation=P)
    with torch.no_grad():
        model.row_gain.copy_(torch.from_numpy(initial['row_gain']))
    if kind!='lifting40':
        assert np.max(np.abs(B0-initial_basis))<1e-14
    sigma=torch.from_numpy(cov)
    A=torch.from_numpy(teacher)
    order=torch.tensor(P,dtype=torch.long)
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    initial_loss=float(initial['sum_MSE'])
    last_loss_before_update=None
    for step in range(STEPS):
        optimizer.zero_grad()
        B=model.matrix()
        predicted=model.row_gain[:,None]*B[order]
        difference=predicted-A
        loss=torch.einsum('ti,ij,tj->',difference,sigma,difference)
        loss.backward()
        optimizer.step()
        last_loss_before_update=float(loss.detach())
    B=model.matrix().detach().numpy()
    gain=model.row_gain.detach().numpy()
    blocks=model.blocks().detach().numpy()
    parameters=model.block_parameters.detach().numpy()
    fitted_A=gain[:,None]*B[P]
    fitted_bias=teacher@mean+bias-fitted_A@mean
    difference=fitted_A-teacher
    row_mse=np.einsum('ti,ij,tj->t',difference,cov,difference)
    total=float(row_mse.sum())
    teacher_variance=float(np.einsum('ti,ij,tj->',teacher,cov,teacher))
    determinants=np.linalg.det(blocks)
    singular_values=np.linalg.svd(B,compute_uv=False)
    try:
        inverse=inverse_by_factors(kind,parameters,blocks)
        inverse_check=float(np.max(np.abs(inverse@B-np.eye(10))))
        inverse_l1=float(np.abs(inverse).sum(axis=1).max())
    except np.linalg.LinAlgError:
        inverse=np.full((10,10),np.nan)
        inverse_check=None
        inverse_l1=None
    data=dict(
        kind=kind,initial_sum_MSE=initial_loss,initial_residual_fraction=initial_loss/teacher_variance,
        final_sum_MSE=total,final_residual_fraction=total/teacher_variance,
        per_row_MSE=row_mse,teacher_centered_variance_sum=teacher_variance,
        final_after_exact_updates=STEPS,last_preupdate_loss=last_loss_before_update,
        row_permutation=P,row_gain=gain,mean_corrected_bias=fitted_bias,
        mean_error_max=float(np.max(np.abs(teacher@mean+bias-(fitted_A@mean+fitted_bias)))),
        B_condition_number=float(np.linalg.cond(B)),B_rank=int(np.linalg.matrix_rank(B)),
        B_singular_values=singular_values,B_max_row_L1=float(np.abs(B).sum(axis=1).max()),
        inverse_B_max_row_L1=inverse_l1,inverse_factor_identity_max_abs_error=inverse_check,
        per_block_determinants=determinants,minimum_absolute_block_determinant=float(np.abs(determinants).min()),
        B_orthogonality_max_error=float(np.max(np.abs(B.T@B-np.eye(10)))),
        block_parameter_count=int(parameters.size),row_gain_parameter_count=10,
        arithmetic=arithmetic(kind,parameters,blocks,gain),CPU_wall_seconds=time.monotonic()-begin,
        interpretation='One fixed local moment-loss optimization from the stated initialization. Residual is not a global optimum or gate/flow/AEE measurement; no rank or condition regularization.')
    arrays=dict(B=B,inverse_B=inverse,blocks=blocks,block_parameters=parameters,
                row_gain=gain,permutation=P,A=fitted_A,bias=fitted_bias,initial_B=B0)
    print(kind,'finalMSE',f'{total:.9f}','fraction',f'{total/teacher_variance:.6%}',
          'cond',f'{data["B_condition_number"]:.6g}','minDet',f'{data["minimum_absolute_block_determinant"]:.6g}',flush=True)
    return data,arrays


def main():
    started=time.monotonic()
    init_path=CHAIN/'fast_temporal_recovery/initialization.json'
    teacher_path=CHAIN/'affine_shared_temporal_control_diverse10/parameters.npz'
    initialization=json.loads(init_path.read_text())
    moment=initialization['source_moments']
    mean=np.asarray(moment['mean'],np.float64)
    cov=np.asarray(moment['covariance'],np.float64)
    old=np.load(teacher_path,allow_pickle=False)
    teacher=old['source_A'].astype(np.float64)
    bias=old['source_bias'].astype(np.float64).reshape(10)
    P=np.asarray(initialization['source_fit']['row_permutation'],np.int64)
    B0=np.asarray(initialization['initial_basis'],np.float64)
    teacher_variance=float(np.einsum('ti,ij,tj->',teacher,cov,teacher))
    initial=fit_readout(mean,cov,teacher,bias,B0,row_permutation=P)
    assert abs(float(initial['sum_MSE'])-initialization['source_fit']['sum_MSE'])<1e-10
    result=dict(complete=False,scope='source temporal expressivity on the original train4 moment domain only',
                inputs=dict(initialization=str(init_path),teacher=str(teacher_path)),
                moments=moment,teacher_centered_variance_sum=teacher_variance,
                covariance_eigenvalues=np.linalg.eigvalsh(cov),
                fixed_P=P,optimizer='Adam default beta/eps, F64 CPU',steps=STEPS,learning_rate=LR,
                endpoint_selection='final after update1024, never best, no restarts/order/seed/width/rank search',
                bias_rule='teacher_A@mean+teacher_bias-fitted_A@mean; center and output theta unchanged',
                original_hard_allplus_fixed_gain_fit=dict(
                    sum_MSE=float(initial['sum_MSE']),residual_fraction=float(initial['sum_MSE'])/teacher_variance,
                    B_condition_number=float(np.linalg.cond(B0)),
                    per_forward_or_inverse_add_subtracts=40,
                    general_products_per_B_pass=0,
                    final_scale='divide4 exact real/dyadic scale; fixed-point rounding outside'),
                teacher_dense=dict(B_condition_number=float(np.linalg.cond(teacher)),
                                   general_products=100,add_subtracts=90,
                                   inverse_general_products=100,inverse_add_subtracts=90),
                cases={},no_network_training_or_AEE=True,
                exclusions='No validation values, flow, classification margins, input peaks or deployed bitwidth measured. Orthogonality does not preserve gate outcomes or bound observed peak by cond alone.')
    arrays=dict(source_mean=mean,source_covariance=cov,teacher_A=teacher,teacher_bias=bias,
                source_center=old['source_center'],source_theta=old['source_theta'])
    for kind in KINDS:
        data,values=fit(kind,mean,cov,teacher,bias,P,B0)
        result['cases'][kind]=data
        arrays.update({kind+'_'+name:value for name,value in values.items()})
    result['complete']=True
    result['CPU_wall_seconds']=time.monotonic()-started
    save(HERE/'result.json',result)
    np.savez(HERE/'parameters.npz',**arrays)
    print('COMPLETE',round(result['CPU_wall_seconds'],3),'CPU seconds',flush=True)


if __name__=='__main__':
    main()

