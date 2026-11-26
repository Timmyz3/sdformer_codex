"""Action-response distillation math. No real data, GT paths or model executor.

Local mechanism adaptation; these routines do not establish quality or cycles.
"""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SEED=20261017
TILES=tuple((r,c) for r in range(3) for c in range(4))
ALL_TILES=tuple((r,c) for r in range(4) for c in range(5))

def require(ok,msg):
    if not ok: raise ValueError(msg)

def h1_expand(x):
    x=np.asarray(x,dtype=np.float32)
    require(x.shape[-3:-1]==(9,9),'Complete tile with channels last')
    out=np.empty_like(x)
    for y in range(0,9,3):
        for z in range(0,9,3):
            for a,b in ((y,y+1),(y+1,y+3)):
                out[...,a:b,z:z+3,:]=x[...,a:b,z:z+3,:].mean((-3,-2),keepdims=True,dtype=np.float32)
    return out

def features(prefix):
    """Input is actual FP32 full-T mean, N x30x40x384, without GT/mask."""
    x=np.asarray(prefix,dtype=np.float32)
    require(x.ndim==4 and x.shape[1:]==(30,40,384) and np.isfinite(x).all(),'Finite prefix means')
    grouped=x.reshape(len(x),30,40,16,24).mean(-1,dtype=np.float32)
    context=np.concatenate((grouped.mean((1,2)),grouped.std((1,2))),-1)
    rows=[]
    for i in range(len(x)):
        for r,c in TILES:
            a=x[i,9*r:9*r+9,9*c:9*c+9]
            residual=a-h1_expand(a)
            rows.append(np.r_[a.mean((0,1)),a.std((0,1)),np.sqrt(np.square(residual).mean((0,1))),
                              context[i],np.float32(r/2),np.float32(c/3)])
    out=np.asarray(rows,dtype=np.float32)
    require(out.shape==(12*len(x),1186) and np.isfinite(out).all(),'Shared input features')
    return out

def response_grid(action,background):
    """Offline output response, all pixels, never sparse GT or region masks."""
    a,b=np.asarray(action),np.asarray(background)
    require(a.shape==b.shape==(2,480,640) and a.dtype==b.dtype==np.float32,'Actual head3 geometry')
    d=a-b
    require(np.isfinite(d).all(),'Finite signed response')
    return d.reshape(2,30,16,40,16).mean((2,4),dtype=np.float32).transpose(1,2,0)

def scaler(x):
    x=np.asarray(x,dtype=np.float32)
    mean,std=x.mean(0),x.std(0)
    return {'mean':mean,'std':np.where(std>1e-6,std,np.float32(1))}

def scale(x,s):
    return ((x-s['mean'])/s['std']).astype(np.float32)

def fit_response(x,d,rank=8,alpha=10.):
    x=np.asarray(x,dtype=np.float64);d=np.asarray(d,dtype=np.float64)
    require(x.ndim==2 and x.shape[1]==1186 and d.shape==(len(x),30,40,2) and len(x)>rank,
            'Fit-side response matrices')
    require(np.isfinite(x).all() and np.isfinite(d).all(),'Finite fit matrix')
    mean,std=x.mean(0),x.std(0);std=np.where(std>1e-6,std,1.)
    z=(x-mean)/std
    y=d.reshape(len(d),-1);ym=y.mean(0);center=y-ym
    _,s,basis=np.linalg.svd(center,full_matrices=False);basis=basis[:rank]
    for row in basis:
        if row[np.argmax(np.abs(row))]<0:row*=-1
    coefficients=center@basis.T
    beta=z.T@np.linalg.solve(z@z.T+float(alpha)*np.eye(len(z)),coefficients)
    weight=beta/std[:,None];bias=-mean@weight
    return dict(weight=weight.astype(np.float32),bias=bias.astype(np.float32),
                basis=basis.astype(np.float32),response_mean=ym.astype(np.float32),
                rank=rank,alpha=float(alpha),fit_rows=len(x),
                retained_variance=float(np.square(s[:rank]).sum()/np.square(s).sum()) if np.square(s).sum()>0 else 1.)

def predict_response(model,x):
    x=np.asarray(x,dtype=np.float32)
    y=(x@model['weight']+model['bias'])@model['basis']+model['response_mean']
    require(y.shape==(len(x),2400) and np.isfinite(y).all(),'Finite predicted response')
    return y.reshape(len(x),30,40,2).astype(np.float32)

def motion_scaler(d):
    d=np.asarray(d,dtype=np.float32)
    mean=d.mean((0,1,2),keepdims=True)
    std=np.float32(np.sqrt(np.square(d-mean).mean(dtype=np.float32)))
    return {'mean':mean,'std':np.float32(std if std>1e-6 else 1.)}

class TilePool(nn.Module):
    def forward(self,x):
        return torch.stack([x[:,:,9*r:min(9*r+9,30),9*c:min(9*c+9,40)].mean((2,3))
                            for r,c in ALL_TILES],-1).flatten(1)

class GenericResponse(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(2,8,3,padding=1),nn.ReLU(),nn.Conv2d(8,8,3,padding=1),nn.ReLU())
        self.pool=TilePool()
    def forward(self,x):return self.pool(self.body(x))

class StructuredResponse(nn.Module):
    def __init__(self):
        super().__init__();self.node=nn.Conv2d(2,4,1);self.edge=nn.Linear(8,4);self.pool=TilePool()
        src=[];dst=[];direction=[];group=[]
        for y in range(30):
            for x in range(40):
                for dy,dx in ((0,1),(1,0)):
                    if y+dy<30 and x+dx<40:
                        src.append(y*40+x);dst.append((y+dy)*40+x+dx)
                        direction.append((dx,dy));group.append((y//9)*5+x//9)
        self.register_buffer('src',torch.tensor(src,dtype=torch.long))
        self.register_buffer('dst',torch.tensor(dst,dtype=torch.long))
        self.register_buffer('direction',torch.tensor(direction,dtype=torch.float32))
        for i in range(20):self.register_buffer('group_'+str(i),torch.tensor([k for k,v in enumerate(group) if v==i],dtype=torch.long))
    def forward(self,x):
        raw=x.flatten(2).transpose(1,2);p,q=raw[:,self.src],raw[:,self.dst]
        e=F.relu(self.edge(torch.cat((p,q,p-q,self.direction.expand(len(x),-1,-1)),-1)))
        pooled=torch.stack([e[:,getattr(self,'group_'+str(i))].mean(1) for i in range(20)],-1).flatten(1)
        return torch.cat((self.pool(F.relu(self.node(x))),pooled),1)

class GainHead(nn.Module):
    def __init__(self,kind):
        super().__init__();require(kind in ('direct','generic_response','structured_response'),'Known head')
        self.body=None if kind=='direct' else GenericResponse() if kind=='generic_response' else StructuredResponse()
        self.head=nn.Sequential(nn.Linear(1186+(0 if self.body is None else 160),16),nn.ReLU(),nn.Linear(16,1))
    def forward(self,x,d):
        return self.head(x if self.body is None else torch.cat((x,self.body(d)),1)).squeeze(1)

def fit_head(kind,x,d,target,steps=400):
    torch.manual_seed(SEED);torch.set_num_threads(2)
    xs,ds=scaler(x),motion_scaler(d)
    xx=torch.from_numpy(scale(x,xs));dd=torch.from_numpy(scale(d,ds).transpose(0,3,1,2))
    target=np.asarray(target,dtype=np.float32);ym,ys=np.float32(target.mean()),np.float32(target.std())
    require(target.shape==(len(x),) and np.isfinite(target).all() and ys>1e-10,'Nonconstant finite signed target')
    yy=torch.from_numpy((target-ym)/ys)
    model=GainHead(kind);initial_state_sha=state_sha(model)
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.001)
    loss_record=[];updates=0
    for i in range(steps):
        opt.zero_grad(set_to_none=True);loss=F.mse_loss(model(xx,dd),yy)
        require(bool(torch.isfinite(loss)),'Finite training loss');loss.backward()
        grads=[p.grad for p in model.parameters()]
        require(all(g is not None and bool(torch.isfinite(g).all()) for g in grads),'All actual parameter gradients finite')
        norm=float(torch.sqrt(sum(g.detach().square().sum() for g in grads)))
        opt.step();updates+=1
        if i in (0,99,199,299,399):loss_record.append({'step':i+1,'normalized_fit_MSE':float(loss.detach()),'gradient_l2':norm})
    model.eval()
    final_state_sha=state_sha(model)
    require(updates==steps and final_state_sha!=initial_state_sha,'Actual optimizer updates changed model')
    return dict(model=model,x_scaler=xs,d_scaler=ds,target_mean=ym,target_std=ys,steps=steps,seed=SEED,
                parameter_count=sum(p.numel() for p in model.parameters()),loss_record=loss_record,
                actual_optimizer_updates=updates,all_parameter_gradients_checked=True,
                initial_state_sha256=initial_state_sha,final_state_sha256=final_state_sha)

def infer_head(fitted,x,d):
    with torch.inference_mode():
        score=fitted['model'](torch.from_numpy(scale(x,fitted['x_scaler'])),torch.from_numpy(scale(d,fitted['d_scaler']).transpose(0,3,1,2)))
    out=score.numpy()*fitted['target_std']+fitted['target_mean']
    require(np.isfinite(out).all(),'Finite gain predictions');return out

def state_sha(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().numpy().tobytes())
    return h.hexdigest()
