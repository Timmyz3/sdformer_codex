#!/usr/bin/env python3
"""Fixed, pre-reviewed R16 synthetic CPU matrix. No checkpoint/GT/GPU/train."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
import contextlib
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
import traceback
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
SOURCE=ROOT/'SDformer/third_party/SDformerFlow'
DEPS=ROOT/'research/flow_story_20261011/preflight/cpu_numerics/deps'
sys.path[:0]=[str(DEPS),str(SOURCE)]
import numpy as np
import torch
from spikingjelly.activation_based import functional
from models.STSwinNet_SNN.Spiking_swin_transformer3D import MS_Spiking_Mlp
import compact_current_mlp as core

SCHEMA='flow_current_bn_round16_cpu_v1'
OWN=['compact_current_mlp.py','run_cpu_v1.py','PROTOCOL_V1.md','source_pins_v1.json']
TOLS={'float64':1e-10,'float32':2e-5}
BANDS={'float64':1e-12,'float32':1e-6}


def require(ok,msg):
    if not ok:raise ValueError(msg)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(x):
    a=x.detach().cpu().contiguous().numpy() if isinstance(x,torch.Tensor) else np.ascontiguousarray(x)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def write(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)


def state_sha(model):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def hooks_empty(model):
    return all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def make_case(seed,dtype_name,height,width,mask_name,scenario,batch):
    dtype=getattr(torch,dtype_name);torch.manual_seed(seed)
    model=MS_Spiking_Mlp(4,hidden_features=16,norm_layer='BN',drop=0.,num_steps=10,
        spike_norm='BN',neuron_type='psn',v_th=.1,v_reset=None,
        surrogate_fun='surrogate.ATan()',tau=2.,detach_reset=True)
    functional.set_step_mode(model,'m')
    model=model.to(device='cpu',dtype=dtype).eval().requires_grad_(False)
    with torch.no_grad():
        for sn in (model.sn1.spiking_neuron,model.sn2.spiking_neuron):
            sn.weight.normal_(0.,.3);sn.bias.fill_(.02)
        for wrapper in (model.bn1,model.bn2):
            bn=wrapper.norm_layer;bn.weight.uniform_(.8,1.2);bn.bias.uniform_(-.1,.1)
            bn.track_running_stats=False;bn.running_mean=None;bn.running_var=None;bn.num_batches_tracked=None
        torch.manual_seed(seed+10000)
        u=torch.randn(10,batch,height,width,4,dtype=dtype)
        if scenario in ('constant','near_sn1','near_sn2'):
            sn=model.sn1.spiking_neuron;sn.weight.copy_(torch.eye(10,dtype=dtype));sn.bias.zero_()
        if scenario=='constant':u.fill_(.25)
        if scenario=='low_variance':model.fc1.weight.mul_(1e-7)
        if scenario=='near_sn1':
            parity=(torch.arange(10)[:,None,None,None,None]+torch.arange(height)[None,None,:,None,None]
                    +torch.arange(width)[None,None,None,:,None]+torch.arange(4)[None,None,None,None,:])%2
            u.copy_((parity.to(dtype)*2-1)*2**-20)
        if scenario=='near_sn2':
            u.copy_(((torch.arange(10,dtype=dtype)%2)*2-1)[:,None,None,None,None])
            model.fc1.weight.zero_()
            for row in range(16):model.fc1.weight[row,row%4]=1.
            model.bn1.norm_layer.weight.fill_(1.);model.bn1.norm_layer.bias.zero_()
            model.sn2.spiking_neuron.weight.copy_(torch.eye(10,dtype=dtype))
            delta=((torch.arange(10)%3)-1).to(dtype)[:,None]*2**-24
            model.sn2.spiking_neuron.bias.copy_(-.5/math.sqrt(.25+model.bn1.norm_layer.eps)+delta)
    ny,nx=(height+8)//9,(width+8)//9
    mask=torch.tensor([[mask_name=='FF' or (mask_name=='checkerboard' and (y+x)%2==0)
                        for x in range(nx)] for y in range(ny)],dtype=torch.bool)
    core.validate(model,u,mask)
    return model,u,mask


@contextlib.contextmanager
def observe(model):
    require(hooks_empty(model),'Unexpected pre-existing hooks')
    raw={};counts={};handles=[]
    def capture(label):
        def hook(module,args,out):
            counts[label]=counts.get(label,0)+1
            raw[label+'_input']=args[0].detach().clone()
            raw[label+'_output']=out.detach().clone()
            if label in ('fc1','fc2'):counts[label+'_rows']=counts.get(label+'_rows',0)+args[0].numel()//args[0].shape[-1]
        return hook
    for name in ('fc1','fc2','sn1','sn2','bn1','bn2'):
        handles.append(getattr(model,name).register_forward_hook(capture(name)))
    def mlp_hook(module,args,out):counts['mlp']=counts.get('mlp',0)+1
    handles.append(model.register_forward_hook(mlp_hook))
    for name in ('sn1','sn2'):
        def pre(module,args,label=name):raw[label+'_prethreshold_flat']=args[0].detach().clone()
        handles.append(getattr(model,name).spiking_neuron.surrogate_function.register_forward_pre_hook(pre))
    try:yield raw,counts
    finally:
        for h in handles:h.remove()
        require(hooks_empty(model),'Observer hook leak')


def run_method(model,u,mask,mode):
    before=state_sha(model);input_before=array_sha(u)
    with observe(model) as (raw,counts):
        if mode=='compact':branch,audit=core.compact_mlp(model,u,mask)
        elif mode=='expanded':branch,dense=core.expanded_mlp(model,u,mask);audit={'dense_input':dense}
        elif mode=='direct':branch=model(u);audit={'dense_input':u}
        else:raise ValueError(mode)
    for name in ('sn1','sn2'):
        raw[name+'_prethreshold']=raw.pop(name+'_prethreshold_flat').reshape(raw[name+'_input'].shape)
    raw['branch']=branch.detach().clone();raw['residual']=u+branch
    if mode=='compact':
        for i in (1,2):
            raw['bn%d_output'%i]=audit['BN%d_output'%i]
            raw['bn%d_mean'%i]=audit['BN%d'%i]['mean']
            raw['bn%d_population_variance'%i]=audit['BN%d'%i]['population_variance']
        raw['packed_input']=audit['packed_input'];raw['areas']=audit['areas']
    else:
        raw['dense_input']=audit['dense_input']
        for i in (1,2):
            raw['bn%d_output'%i]=raw['bn%d_output'%i].permute(0,1,3,4,2).contiguous()
            z=raw['fc%d_output'%i]
            raw['bn%d_mean'%i]=z.mean((0,1,2,3))
            raw['bn%d_population_variance'%i]=z.var((0,1,2,3),unbiased=False)
    closure={'state_before':before,'state_after':state_sha(model),'input_unchanged':input_before==array_sha(u),
             'no_running_buffers':all(getattr(w.norm_layer,k) is None for w in (model.bn1,model.bn2)
                 for k in ('running_mean','running_var','num_batches_tracked')),'hooks_empty':hooks_empty(model)}
    require(closure['state_before']==closure['state_after'] and all(closure[k] for k in ('input_unchanged','no_running_buffers','hooks_empty')),'State/input/hook mutation')
    return raw,counts,audit,closure


def geometry(height,width,mask):
    # Independent expectation: real tile dimensions and ceil/3 formula, not core cells.
    representatives=0;fine_pixels=0;coarse_cells=0;area_hist={}
    for ty in range((height+8)//9):
        for tx in range((width+8)//9):
            h,w=min(9,height-ty*9),min(9,width-tx*9)
            if bool(mask[ty,tx]):
                representatives+=h*w;fine_pixels+=h*w;area_hist[1]=area_hist.get(1,0)+h*w
            else:
                n=((h+2)//3)*((w+2)//3);representatives+=n;coarse_cells+=n
                for y in range(0,h,3):
                    for x in range(0,w,3):
                        a=min(3,h-y)*min(3,w-x);area_hist[a]=area_hist.get(a,0)+1
    require(sum(a*n for a,n in area_hist.items())==height*width,'Independent coverage')
    return {'representatives':representatives,'fine_pixels':fine_pixels,'coarse_representatives':coarse_cells,
            'area_histogram':area_hist,'original_spatial_area':height*width}


def comparison(a,b,tol):
    require(a.shape==b.shape and a.dtype==b.dtype,'Comparison shape/dtype')
    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    error=(a-b).abs();limit=tol+tol*b.abs();bad=(error>limit)|~torch.isfinite(error)
    return {'passed':finite and not bool(bad.any()),'numel':a.numel(),'bad_elements':int(bad.sum()),
            'max_abs_error':float(error.max()),'atol':tol,'rtol':tol}


def compare_paths(left,right,cells,height,width,dtype_name,packed_left):
    tol=TOLS[dtype_name];band=BANDS[dtype_name];reports={};flips={}
    continuous=['fc1_input','fc1_output','fc2_input','fc2_output','sn1_input','sn1_prethreshold',
                'sn2_input','sn2_prethreshold','bn1_output','bn2_output','bn1_mean','bn1_population_variance',
                'bn2_mean','bn2_population_variance','branch','residual']
    def align(name):
        a=left[name]
        if packed_left and a.ndim==5 and name not in ('branch','residual'):
            a=core.expand(a,cells,height,width)
        return a
    for name in continuous:reports[name]=comparison(align(name),right[name],tol)
    for name in ('sn1','sn2'):
        a,b=align(name+'_output'),right[name+'_output'];bad=a!=b;indices=bad.nonzero()
        ah,bh=align(name+'_prethreshold'),right[name+'_prethreshold']
        flips[name]={'count':len(indices),'exact_passed':not bool(bad.any()),'margin_band':band,
            'either_in_band':int(((ah.abs()<=band)|(bh.abs()<=band))[bad].sum()),
            'both_in_band':int(((ah.abs()<=band)&(bh.abs()<=band))[bad].sum()),
            'indices':indices.tolist(),'actual_h':ah[bad].tolist(),'reference_h':bh[bad].tolist()}
    return {'passed':all(v['passed'] for v in reports.values()) and all(v['exact_passed'] for v in flips.values()),
            'continuous':reports,'PSN_flips':flips}


def cases():
    for seed,dtype,(h,w),mask,scenario in itertools.product((17,29,43),('float64','float32'),
            ((9,9),(11,14),(3,4)),('FF','CC','checkerboard'),
            ('random','constant','low_variance','near_sn1','near_sn2')):
        yield seed,dtype,h,w,mask,scenario,1
    for dtype,mask in itertools.product(('float64','float32'),('FF','CC','checkerboard')):
        yield 17,dtype,11,14,mask,'random',2


def run_case(spec,out,index):
    seed,dtype,h,w,mask_name,scenario,batch=spec
    model,u,mask=make_case(*spec);initial=state_sha(model)
    saved={'u':u,'mask':mask,**{'state__'+n:t.detach().clone() for n,t in model.state_dict().items()}}
    methods={};counts={};closures={};audits={}
    for mode in ('compact','expanded')+(('direct',) if mask_name=='FF' else ()):
        raw,count,audit,closure=run_method(model,u,mask,mode)
        methods[mode]=raw;counts[mode]=count;closures[mode]=closure;audits[mode]=audit
        saved.update({mode+'__'+key:value for key,value in raw.items()})
    cells=audits['compact']['cells'];saved['cells']=np.asarray(cells,dtype=np.int64)
    geo=geometry(h,w,mask);rep=geo['representatives']
    require(rep==audits['compact']['representatives'],'Independent representative count')
    for mode,count in counts.items():
        expected=10*batch*(rep if mode=='compact' else h*w)
        require(count['fc1_rows']==count['fc2_rows']==expected,'Actual FC rows mismatch')
        require(all(count[n]==1 for n in ('fc1','fc2','sn1','sn2')),'Primitive replay/omission')
        require(count.get('mlp',0)==(0 if mode=='compact' else 1),'Native MLP shortcut or replay')
        require(count.get('bn1',0)==count.get('bn2',0)==(0 if mode=='compact' else 1),'BN path mismatch')
    comparisons={'compact_vs_expanded':compare_paths(methods['compact'],methods['expanded'],cells,h,w,dtype,True)}
    if mask_name=='FF':
        comparisons['compact_vs_direct_native']=compare_paths(methods['compact'],methods['direct'],cells,h,w,dtype,True)
        comparisons['expanded_vs_direct_native']=compare_paths(methods['expanded'],methods['direct'],cells,h,w,dtype,False)
    arrays={k:v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else v for k,v in saved.items()}
    archive=out/('case_%03d.npz'%index);np.savez_compressed(archive,**arrays)
    payload={k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':array_sha(v)} for k,v in arrays.items()}
    result={'index':index,'seed':seed,'dtype':dtype,'height':h,'width':w,'batch':batch,'mask':mask_name,'scenario':scenario,
        'passed':all(v['passed'] for v in comparisons.values()),'state_sha256':initial,'geometry':geo,'actual_calls':counts,
        'closures':closures,'comparisons':comparisons,'BN_population':10*batch*h*w,'barriers':2,
        'FC1_retained_elements':audits['compact']['FC1_retained_elements'],'FC2_retained_elements':audits['compact']['FC2_retained_elements'],
        'array_archive':{'path':archive.name,'sha256':sha(archive),'payloads':payload},'native_fine_shortcut':False}
    write(out/('case_%03d.json'%index),result)
    return result


def rejection_checks():
    tested=[]
    for name in ('wrong_T','nonbool_mask','training','tracked_BN','dtype_mismatch','BN_single_step'):
        m,u,mask=make_case(17,'float32',3,4,'CC','random',1)
        if name=='wrong_T':u=u[:9]
        if name=='nonbool_mask':mask=mask.float()
        if name=='training':m.train()
        if name=='tracked_BN':m.bn1.norm_layer.track_running_stats=True
        if name=='dtype_mismatch':u=u.double()
        if name=='BN_single_step':m.bn1.norm_layer.step_mode='s'
        try:core.validate(m,u,mask)
        except ValueError:tested.append(name)
        else:raise AssertionError('Invalid contract accepted: '+name)
    m,u,mask=make_case(17,'float32',3,4,'CC','random',1)
    try:
        with observe(m):raise RuntimeError('injected before any forward')
    except RuntimeError:pass
    require(hooks_empty(m),'Exception hook cleanup');tested.append('exception_hook_cleanup')
    require(len(list(cases()))==276 and sum(c[4]=='FF' for c in cases())==92,'Fixed matrix changed')
    return {'passed':True,'checks':tested,'numerical_forward_calls':0}


def preflight(freeze_path,expected_sha):
    require(sha(freeze_path)==expected_sha,'Freeze digest')
    freeze=json.loads(Path(freeze_path).read_text());require(freeze['schema']=='flow_current_bn_round16_source_freeze_v1','Freeze schema')
    require(freeze['review_status']=='SOURCE_REVIEW_PASS_BEFORE_NUMERICAL_RUN','Pre-result independent review required')
    require(set(freeze['source_sha256'])==set(OWN),'Source scope')
    for name,digest in freeze['source_sha256'].items():require(sha(HERE/name)==digest,'Own source '+name)
    pins=json.loads((HERE/'source_pins_v1.json').read_text())
    for rel,digest in pins['repository_files'].items():require(sha(ROOT/rel)==digest,'Historical source '+rel)
    for rel,digest in pins['dependency_files'].items():require(sha(DEPS/rel)==digest,'CPU dependency '+rel)
    require(freeze['case_count']==276 and freeze['logical_MLP_calls']==644,'Frozen budget')
    require(torch.get_default_dtype()==torch.float32,'Default dtype');require(not torch.cuda.is_initialized(),'No CUDA initialization')
    return freeze


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,required=True);p.add_argument('--freeze-sha256',required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    freeze=preflight(a.freeze,a.freeze_sha256);require(not a.output_dir.exists(),'Fresh output only')
    a.output_dir.mkdir(parents=True);start=time.monotonic();records=[]
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    result={'schema':SCHEMA,'status':'INCOMPLETE','freeze_sha256':a.freeze_sha256,'torch':torch.__version__,
        'numpy':np.__version__,'device':'cpu','autograd':False,'actual_full_model_forwards':0,'optimizer_steps':0,
        'case_records':[]}
    try:
        with torch.inference_mode():
            result['contract_checks']=rejection_checks()
            for index,spec in enumerate(cases(),1):
                require(time.monotonic()-start<300,'Five-minute execution cap')
                record=run_case(spec,a.output_dir,index);records.append(record)
                result['case_records'].append({'index':index,'path':'case_%03d.json'%index,'sha256':sha(a.output_dir/('case_%03d.json'%index)),
                    'passed':record['passed'],'dtype':record['dtype'],'scenario':record['scenario'],'mask':record['mask']})
                if index%30==0:print(json.dumps({'cases_complete':index,'failures':sum(not r['passed'] for r in records)}),flush=True)
        require(len(records)==276,'Incomplete matrix')
        require(sum(len(r['actual_calls']) for r in records)==644,'Logical MLP budget')
        require(sum(c['fc1_rows'] for r in records for c in r['actual_calls'].values())==473300,'Independent whole-matrix FC budget')
        require(sum(c.get('mlp',0) for r in records for c in r['actual_calls'].values())==368,'Actual native MLP method budget')
        result['status']='CPU_MATRIX_COMPLETE';result['admission']='ADMITTED_FINITE_SYNTHETIC_CPU_ONLY' if all(r['passed'] for r in records) else 'NOT_ADMITTED'
    except Exception:
        result['exception']=traceback.format_exc();result['admission']='NOT_ADMITTED_INCOMPLETE'
    result['elapsed_seconds']=time.monotonic()-start;result['cases_complete']=len(records)
    result['failed_cases']=[r['index'] for r in records if not r['passed']]
    result['actual_logical_MLP_calls']=sum(len(r['actual_calls']) for r in records)
    result['actual_fc_rows_per_layer']=sum(c['fc1_rows'] for r in records for c in r['actual_calls'].values())
    result['actual_PSN_calls_each']=sum(c['sn1'] for r in records for c in r['actual_calls'].values())
    result['actual_native_MLP_method_calls']=sum(c.get('mlp',0) for r in records for c in r['actual_calls'].values())
    write(a.output_dir/'results.json',result)
    print(json.dumps({k:result[k] for k in ('status','admission','cases_complete','actual_logical_MLP_calls','elapsed_seconds')}),flush=True)
    if result['status']!='CPU_MATRIX_COMPLETE':raise SystemExit(2)


if __name__=='__main__':main()
