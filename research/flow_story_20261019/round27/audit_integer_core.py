#!/usr/bin/env python3
"""Independent R27 integer/STE contract audit; synthetic CPU only."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import torch
from torch.nn import functional as F


def main():
    p=argparse.ArgumentParser(); p.add_argument('--core',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(271019)
    spec=importlib.util.spec_from_file_location('audit_subject',a.core);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    results=[]
    for kind in ('linear','conv'):
        for mode in ('Q8','Q6'):
            shape=(10,1,9,11,8) if kind=='linear' else (10,1,8,9,11)
            x=torch.randint(2,shape).float().requires_grad_()
            source=torch.nn.Linear(8,16,bias=False) if kind=='linear' else torch.nn.Conv2d(8,16,3,padding=1,bias=False)
            with torch.no_grad(): source.weight.copy_(torch.randn_like(source.weight)*.17)
            wrap=m.STEAffine(source,kind,mode);wrap.capture_integer=True
            y=wrap(x)
            # Quantization reproduced without calling subject.quantize/effective_weights.
            w=source.weight.detach().reshape(16,-1);s=w.abs().max(1).values/127
            q=torch.round(w/s[:,None]).clamp(-127,127).to(torch.int64)
            qlow=q if mode=='Q8' else torch.div(q,4,rounding_mode='floor')*4
            yy,xx=torch.meshgrid(torch.arange(9),torch.arange(11),indexing='ij')
            tail=((yy>=9)|(xx>=9)).expand(10,1,9,11).reshape(-1)
            rows=x.detach().reshape(-1,8) if kind=='linear' else F.unfold(x.detach().reshape(10,8,9,11),3,padding=1).transpose(1,2).reshape(-1,72)
            z=torch.empty(len(rows),16,dtype=torch.int64)
            z[tail]=rows[tail].to(torch.int64)@q.T
            z[~tail]=rows[~tail].to(torch.int64)@qlow.T
            expected=z.float()*s
            if kind=='linear':expected=expected.reshape(10,1,9,11,16)
            else:expected=expected.reshape(10,1,9,11,16).permute(0,1,4,2,3).contiguous()
            assert torch.equal(wrap.last_integer,z.to(torch.int32))
            assert torch.equal(y,expected)
            g=torch.randn_like(y);y.backward(g)
            # Independent differentiable surrogate expression: integer-grid W for
            # activation derivative, original input and identity dW for W derivative.
            xr=x.detach().clone().requires_grad_();wr=source.weight.detach().clone().requires_grad_()
            wf=(q.float()*s[:,None]).reshape_as(wr);wl=(qlow.float()*s[:,None]).reshape_as(wr)
            wf=wr+(wf-wr).detach();wl=wr+(wl-wr).detach()
            if kind=='linear':
                ref=torch.where(tail.reshape(10,1,9,11,1),F.linear(xr,wf),F.linear(xr,wl))
            else:
                xf=xr.reshape(10,8,9,11)
                ref=torch.where(tail.reshape(10,1,9,11),F.conv2d(xf,wf,padding=1),F.conv2d(xf,wl,padding=1)).reshape(10,1,16,9,11)
            ref.backward(g)
            dx=float((x.grad-xr.grad).abs().max());dw=float((source.weight.grad-wr.grad).abs().max())
            torch.testing.assert_close(x.grad,xr.grad,rtol=3e-5,atol=3e-5)
            torch.testing.assert_close(source.weight.grad,wr.grad,rtol=3e-5,atol=1e-4)
            audit=wrap.last_audit
            assert audit['logical_row_plane_work']==(10*9*11*8 if mode=='Q8' else 10*9*11*6+10*9*2*2)
            results.append(dict(kind=kind,mode=mode,exact_int64_and_scaled_output=True,dx_max_abs=dx,dw_max_abs=dw,
                                row_plane_work=audit['logical_row_plane_work']))
    out=dict(status='PASS',scope='independent synthetic CPU integer and declared STE, not model/GPU/quality validation',
             core_sha256=hashlib.sha256(a.core.read_bytes()).hexdigest(),cases=results)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))


if __name__=='__main__':main()
