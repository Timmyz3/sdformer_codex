"""Literal signed24 deployment and the same hard function with STE training.

Only backward differs. There is no FP shadow substitution for source, preview
input, completed latent, either consumer, or the onepass BN hard forward.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from fixed_temporal_coordinates import FixedTemporalForward,STATE_SCALE,STATE_MIN,STATE_MAX

MATRICES=('As','U_conv2_theta','F','U_ped','V_ped')


class RoundClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,lo,hi):
        ctx.save_for_backward(x);ctx.lo=lo;ctx.hi=hi
        return x.round().clamp(lo,hi)
    @staticmethod
    def backward(ctx,dy):
        x,=ctx.saved_tensors
        return dy*((x>=ctx.lo)&(x<=ctx.hi)),None,None


class Bits(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,theta):ctx.theta=theta;return x.ne(0).double()
    @staticmethod
    def backward(ctx,dy):return dy/ctx.theta,None


class Gate(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,cutoff,direction,constant,theta):
        shape=(10,)+(1,)*(value.ndim-1);d=direction.reshape(shape)
        c=cutoff.reshape(shape);k=torch.where(d>0,torch.ceil(c*STATE_SCALE),torch.floor(c*STATE_SCALE))
        margin=(value/STATE_SCALE-c)*d
        ctx.save_for_backward(margin,d,constant.reshape(shape));ctx.theta=abs(theta)
        live=torch.where(d>0,value>=k,value<=k)
        return torch.where(constant.reshape(shape)>=0,constant.reshape(shape).bool(),live).double()
    @staticmethod
    def backward(ctx,dy):
        margin,d,constant=ctx.saved_tensors
        grad=dy*(1-margin.abs()/max(ctx.theta,1e-12)).clamp_min(0)*d*(constant<0)
        return grad/STATE_SCALE,-grad.flatten(1).sum(1),None,None,None


class OnepassSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,math,gamma,beta,eps):
        payload=x.detach().permute(0,2,3,1).contiguous().cpu().numpy()
        st=math.statistics(payload,gamma,beta,eps)
        mu=torch.as_tensor(st[0],device=x.device).reshape(1,96,1,1)
        inv=torch.as_tensor(st[2],device=x.device).reshape(1,96,1,1)
        scale=torch.as_tensor(st[3],device=x.device).reshape(1,96,1,1)
        offset=torch.as_tensor(st[4],device=x.device).reshape(1,96,1,1)
        ctx.save_for_backward(x,mu,inv,scale)
        return x*scale+offset
    @staticmethod
    def backward(ctx,dy):
        x,mu,inv,scale=ctx.saved_tensors
        normalized=(x-mu)*inv;dims=(0,2,3)
        # Common smooth BN derivative; discrete forward remains the exact
        # onepass function and is separately compared against deployment.
        return scale*(dy-dy.mean(dims,keepdim=True)-normalized*(dy*normalized).mean(dims,keepdim=True)),None,None,None,None


class LiteralForward(FixedTemporalForward):
    def __init__(self,controller,sn2_theta,constants,structure):
        super().__init__(controller,sn2_theta)
        self.structure=structure;self.sn2_theta=float(sn2_theta)
        self.base_constants={k:np.asarray(v).copy() for k,v in constants.items()}
        for key in MATRICES:
            q=constants[key+'_q16'].astype(np.int64);e=int(constants[key+'_exponent'])
            pos=np.maximum(q,0).sum(1);neg=np.minimum(q,0).sum(1)
            lo=pos*STATE_MIN+neg*STATE_MAX;hi=pos*STATE_MAX+neg*STATE_MIN
            bound=self.prove48(key,lo,hi)
            self.matrices[key]=dict(q=torch.as_tensor(q,device=self.device,dtype=torch.float64),q_numpy=q,exponent=e,lower=lo,upper=hi)
            self.matrix_metadata[key]=dict(shape=list(q.shape),exponent=e,integer_nonzero=int(np.count_nonzero(q)),dot_abs_bound=bound,fits_signed48=True,input_domain='signed24 bound (conservative for spike input)')
            self.constants[key+'_q16']=q.astype(np.int16);self.constants[key+'_exponent']=np.asarray(e)
        for name in ['source','consumer']:
            for key in ['threshold','direction','constant']:
                value=constants[name+'_'+key]
                self.comparisons[name][key]=torch.as_tensor(value,device=self.device)
                self.constants[name+'_'+key]=np.asarray(value)
        self.c_bn=torch.as_tensor(constants['BN2_constant_q24'],device=self.device,dtype=torch.float64)
        self.projection_bias=torch.as_tensor(constants['PED_bias_q24'],device=self.device,dtype=torch.float64)
        self.constants['BN2_constant_q24']=constants['BN2_constant_q24'];self.constants['PED_bias_q24']=constants['PED_bias_q24']
        self.metadata.update(mode='literal_compiled_structure',structure=structure,
            coefficient_format='Common parent exponent grids frozen; live signed16 coefficients and signed24 writes use RNE/saturation.',
            comparisons={name:dict(threshold=constants[name+'_threshold'].tolist(),direction=constants[name+'_direction'].tolist(),constant=constants[name+'_constant'].tolist()) for name in ['source','consumer']},
            compile_numeric='Literal exported coefficient/threshold arrays are authoritative; original native bias/readout metadata is not reused as a trained parameter mapping.')
        if structure=='lifting40':
            self.lifting_q12=torch.as_tensor(constants['lifting_q12'],device=self.device,dtype=torch.float64)
            self.matchings=torch.as_tensor(constants['lifting_matchings'],device=self.device,dtype=torch.long)
            self.source_permutation=torch.as_tensor(constants['source_permutation'],device=self.device,dtype=torch.long)
            for k in ['lifting_q12','lifting_matchings','source_permutation','lifting_fraction_bits']:self.constants[k]=constants[k]

    def lifting(self,value):
        result=value
        coefficients=self.lift_q()
        shape=(5,)+(1,)*(value.ndim-1)
        for layer in range(4):
            first,second=self.matchings[layer,:,0],self.matchings[layer,:,1]
            x0,x1=result.index_select(0,first),result.index_select(0,second)
            a,b=(coefficients[layer,:,h].reshape(shape) for h in (0,1))
            y0=self.write24('lift'+str(layer)+'a',4096*x0+a*x1,12)
            y1=self.write24('lift'+str(layer)+'b',4096*x1+b*y0,12)
            result=result.index_copy(0,first,y0).index_copy(0,second,y1)
        return result

    def lift_q(self):return self.lifting_q12

    def source_forward(self,x):
        self.frame=dict(clip_counts={},state_ranges={},accumulator_ranges={});self.ready=False
        identity=self.write24('I24',x[:,0].double()*STATE_SCALE)
        q=(self.lifting(identity).index_select(0,self.source_permutation) if self.structure=='lifting40'
           else self.time_dot('As',identity,'As_I_Q24'))
        self.i,self.q=identity,None
        gate=self.compare('source',q)
        self.frame['source_gate']=dict(nonzero=int(gate.detach().sum()),elements=gate.numel())
        return self.emit(self.source,gate)


class QATForward(LiteralForward):
    def __init__(self,controller,sn2_theta,constants,structure):
        super().__init__(controller,sn2_theta,constants,structure)
        self.parameters=nn.ParameterDict()
        for key in MATRICES:
            if key=='As' and structure=='lifting40':continue
            self.parameters[key]=nn.Parameter(self.matrices[key]['q'].float()/(2**self.matrices[key]['exponent']))
        if structure=='lifting40':self.parameters['lifting']=nn.Parameter(self.lifting_q12.float()/4096)
        for name in ['source','consumer']:
            self.parameters[name+'_cutoff']=nn.Parameter(self.comparisons[name]['threshold'].float()/STATE_SCALE)
        self.parameters['BN2_constant']=nn.Parameter(self.c_bn.float()/STATE_SCALE)
        self.parameters['PED_bias']=nn.Parameter(self.projection_bias.float()/STATE_SCALE)
        self.source_mask=torch.as_tensor(constants.get('source_mask',np.ones((10,10))),device=self.device,dtype=torch.float64)

    def live_matrix(self,key):
        value=self.parameters[key].double()
        if key=='As':value=value*self.source_mask
        return RoundClip.apply(value*(2**self.matrices[key]['exponent']),-32768,32767)
    def lift_q(self):return RoundClip.apply(self.parameters['lifting'].double()*4096,-32768,32767)
    def write24(self,name,value,shift=0):return RoundClip.apply(value/(2**shift),STATE_MIN,STATE_MAX)
    def observe(self,*args):pass
    def time_dot(self,key,value,name,complete=True):
        q=self.live_matrix(key);acc=(q@value.reshape(value.shape[0],-1)).reshape(q.shape[0],*value.shape[1:])
        return self.write24(name,acc,self.matrices[key]['exponent']) if complete else acc
    def channel_dot(self,key,value,name):
        t,_,h,w=value.shape;q=self.live_matrix(key);flat=value.permute(1,0,2,3).reshape(value.shape[1],-1)
        acc=(q@flat).reshape(q.shape[0],t,h,w).permute(1,0,2,3)
        return self.write24(name,acc,self.matrices[key]['exponent'])
    def compare(self,name,value):
        c=self.comparisons[name];theta=float((self.source if name=='source' else self.consumer).thresh)
        return Gate.apply(value,self.parameters[name+'_cutoff'].double(),c['direction'],c['constant'],theta)
    def conv_forward(self,x):
        bits=Bits.apply(x.flatten(0,1).double(),self.sn2_theta)
        acc=F.conv2d(bits,self.live_matrix('U_conv2_theta').reshape_as(self.c.conv_u),None,self.c.rank.stride,self.c.rank.padding,self.c.rank.dilation)
        self.z=self.write24('Z24',acc,self.matrices['U_conv2_theta']['exponent']-14)
        with torch.no_grad():return self.original['conv'](x)
    def finish(self):
        if self.ready:return
        branch=self.channel_dot('F',self.z[:,:,::2,::2],'F_Z_residual24')
        bias=self.write24('BN2_constant',self.parameters['BN2_constant'].double()*STATE_SCALE)
        self.updated=self.i.clone()
        self.updated[:,:,::2,::2]=self.write24('updated_anchor24',self.i[:,:,::2,::2]+branch+bias[None,:,None,None])
        z=self.channel_dot('U_ped',self.updated[:,:,::2,::2],'PED_U24')
        y=self.channel_dot('V_ped',z,'PED_V24')
        bias=self.write24('PED_bias',self.parameters['PED_bias'].double()*STATE_SCALE)
        self.continuous=self.write24('PED_bias_output24',y+bias[None,:,None,None]);self.ready=True
    def projection_forward(self,x):self.finish();return self.continuous.float()/STATE_SCALE
    def consumer_forward(self,x):
        self.finish();gate=self.compare('consumer',self.updated.index_select(0,self.permutation))
        self.frame['consumer_gate']=dict(nonzero=int(gate.detach().sum()),elements=gate.numel())
        result=self.emit(self.consumer,gate);self.frames.append(self.frame)
        self.i=self.q=self.z=self.updated=self.continuous=None
        return result
    @torch.no_grad()
    def literal_constants(self):
        result={k:np.asarray(v).copy() for k,v in self.base_constants.items()}
        for key in MATRICES:
            if key!='As' or self.structure!='lifting40':result[key+'_q16']=self.live_matrix(key).cpu().numpy().astype(np.int16)
        if self.structure=='lifting40':result['lifting_q12']=self.lift_q().cpu().numpy().astype(np.int16)
        for name in ['source','consumer']:
            value=self.parameters[name+'_cutoff'].double()*STATE_SCALE;d=self.comparisons[name]['direction']
            result[name+'_threshold']=torch.where(d>0,value.ceil(),value.floor()).cpu().numpy().astype(np.int64)
        for key in ['BN2_constant','PED_bias']:
            result[key+'_q24']=self.write24(key,self.parameters[key].double()*STATE_SCALE).cpu().numpy().astype(np.int32)
        result['new_compiled_cutoff_parameterization']=np.asarray(True)
        return result
