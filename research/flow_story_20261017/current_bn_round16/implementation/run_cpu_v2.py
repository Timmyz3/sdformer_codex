#!/usr/bin/env python3
"""Single V2 execution; frozen V1 helpers, unchanged cases/tolerances."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
sys.dont_write_bytecode=True
import run_cpu_v1 as v1
import native_dense_bn_v2 as core
torch,np=v1.torch,v1.np
HERE=Path(__file__).resolve().parent
OWN=['native_dense_bn_v2.py','run_cpu_v2.py','PROTOCOL_V2.md']
INHERITED={'compact_current_mlp.py':'ad85fab7c0de64bad5b1fca15baca1af3347e32e226fb17ec18487485cc837ec',
 'run_cpu_v1.py':'854cf198cb3d966e766dfa94c7ba9193973ce623207f1c91b8320469d0f271b5',
 'PROTOCOL_V1.md':'9898028fb66b74d981ba247b2e0a35c209c08b0dd8ffef65108cb6aea40eb8c4',
 'source_pins_v1.json':'972d10ba2303c3f8ff72b980ed2b70a55aa26dc1b8f22adda847666150742598',
 'freeze_v1.json':'7daaf09b64f6ee7c165318d9f6ac1c6bd7f994cde2038de3ceb95dd1c774c5c6'}
RESOURCE_FIELDS=['full_domain_BN_elements','expand_written_elements','BN_output_dense_elements',
 'production_BN1_gather_read_elements','production_BN1_gather_written_elements',
 'diagnostic_BN2_gather_read_elements','cell_exact_audit_read_elements']
require,sha,write=v1.require,v1.sha,v1.write


def run_method(model,u,mask,mode):
    before=v1.state_sha(model);input_before=v1.array_sha(u)
    with v1.observe(model) as (raw,counts):
        if mode=='compact':branch,audit=core.compact_mlp(model,u,mask)
        elif mode=='expanded':branch,dense=core.expanded_mlp(model,u,mask);audit={'dense_input':dense}
        elif mode=='direct':branch=model(u);audit={'dense_input':u}
        else:raise ValueError(mode)
    for name in ('sn1','sn2'):
        raw[name+'_prethreshold']=raw.pop(name+'_prethreshold_flat').reshape(raw[name+'_input'].shape)
    raw['branch']=branch.detach().clone();raw['residual']=u+branch
    if mode=='compact':
        for i in (1,2):
            # Preserve actual BN wrapper capture, then use packed views only for
            # unchanged V1 per-pixel comparison and a separately verified gather.
            raw['bn%d_dense_output'%i]=raw['bn%d_output'%i].permute(0,1,3,4,2).contiguous()
            raw['bn%d_dense_input'%i]=raw['bn%d_input'%i].permute(0,1,3,4,2).contiguous()
            require(torch.equal(raw['bn%d_dense_output'%i],audit['BN%d_dense_output'%i]),'Actual BN output capture')
            require(torch.equal(raw['bn%d_dense_input'%i],audit['BN%d_dense_input'%i]),'Actual BN input capture')
            raw['bn%d_output'%i]=audit['BN%d_output'%i]
            raw['bn%d_mean'%i]=audit['BN%d'%i]['mean']
            raw['bn%d_population_variance'%i]=audit['BN%d'%i]['population_variance']
        raw['packed_input']=audit['packed_input'];raw['areas']=audit['areas']
    else:
        raw['dense_input']=audit['dense_input']
        for i in (1,2):
            raw['bn%d_output'%i]=raw['bn%d_output'%i].permute(0,1,3,4,2).contiguous()
            z=raw['fc%d_output'%i];raw['bn%d_mean'%i]=z.mean((0,1,2,3))
            raw['bn%d_population_variance'%i]=z.var((0,1,2,3),unbiased=False)
    closure={'state_before':before,'state_after':v1.state_sha(model),'input_unchanged':input_before==v1.array_sha(u),
        'no_running_buffers':all(getattr(w.norm_layer,k) is None for w in (model.bn1,model.bn2)
            for k in ('running_mean','running_var','num_batches_tracked')),'hooks_empty':v1.hooks_empty(model)}
    require(before==closure['state_after'] and all(closure[k] for k in ('input_unchanged','no_running_buffers','hooks_empty')),'State/input/hook mutation')
    return raw,counts,audit,closure


def run_case(spec,out,index):
    seed,dtype,h,w,mask_name,scenario,batch=spec
    model,u,mask=v1.make_case(*spec);initial=v1.state_sha(model)
    saved={'u':u,'mask':mask,**{'state__'+n:t.detach().clone() for n,t in model.state_dict().items()}}
    methods={};counts={};closures={};audits={}
    for mode in ('compact','expanded')+(('direct',) if mask_name=='FF' else ()):
        raw,count,audit,closure=run_method(model,u,mask,mode)
        methods[mode]=raw;counts[mode]=count;closures[mode]=closure;audits[mode]=audit
        saved.update({mode+'__'+key:value for key,value in raw.items()})
    audit=audits['compact'];cells=audit['cells'];saved['cells']=np.asarray(cells,dtype=np.int64)
    geo=v1.geometry(h,w,mask);rep=geo['representatives'];require(rep==audit['representatives'],'Independent representative count')
    for mode,count in counts.items():
        expected=10*batch*(rep if mode=='compact' else h*w)
        require(count['fc1_rows']==count['fc2_rows']==expected,'Actual FC rows mismatch')
        require(all(count[n]==1 for n in ('fc1','fc2','sn1','sn2','bn1','bn2')),'Primitive replay/omission')
        require(count.get('mlp',0)==(0 if mode=='compact' else 1),'Native MLP shortcut or replay')
    require(audit['full_domain_BN_elements']==10*batch*h*w*(16+4),'Full-domain BN geometry')
    require(audit['expand_written_elements']==audit['full_domain_BN_elements'],'Expand geometry')
    require(audit['production_BN1_gather_read_elements']==10*batch*rep*16,'Full-axis gather geometry')
    comparisons={'compact_vs_expanded':v1.compare_paths(methods['compact'],methods['expanded'],cells,h,w,dtype,True)}
    if mask_name=='FF':
        comparisons['compact_vs_direct_native']=v1.compare_paths(methods['compact'],methods['direct'],cells,h,w,dtype,True)
        comparisons['expanded_vs_direct_native']=v1.compare_paths(methods['expanded'],methods['direct'],cells,h,w,dtype,False)
    arrays={k:v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else v for k,v in saved.items()}
    archive=out/('case_%03d.npz'%index);np.savez_compressed(archive,**arrays)
    payload={k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':v1.array_sha(v)} for k,v in arrays.items()}
    result={'index':index,'seed':seed,'dtype':dtype,'height':h,'width':w,'batch':batch,'mask':mask_name,'scenario':scenario,
        'passed':all(v['passed'] for v in comparisons.values()) and audit['each_cell_BN1_and_BN2_bitwise_equal'],
        'state_sha256':initial,'geometry':geo,'actual_calls':counts,'closures':closures,'comparisons':comparisons,
        'BN_population':10*batch*h*w,'barriers':2,'BN_mode':'NATIVE_DENSE_CURRENT',
        'cell_exact':{k:audit[k] for k in ('BN1_cell_bitwise_equal','BN2_cell_bitwise_equal','each_cell_BN1_and_BN2_bitwise_equal')},
        'resources':{k:audit[k] for k in RESOURCE_FIELDS},
        'FC1_retained_elements':audit['FC1_retained_elements'],'FC2_retained_elements':audit['FC2_retained_elements'],
        'array_archive':{'path':archive.name,'sha256':sha(archive),'payloads':payload},'native_fine_shortcut':False}
    write(out/('case_%03d.json'%index),result);return result


def preflight(path,expected_sha):
    require(sha(path)==expected_sha,'V2 freeze digest');freeze=json.loads(Path(path).read_text())
    require(freeze['schema']=='flow_current_bn_round16_source_freeze_v2','V2 freeze schema')
    require(freeze['review_status']=='SOURCE_REVIEW_PASS_BEFORE_NUMERICAL_RUN','V2 independent source review required')
    require(set(freeze['source_sha256'])==set(OWN),'V2 source scope')
    for name,digest in freeze['source_sha256'].items():require(sha(HERE/name)==digest,'V2 source '+name)
    for name,digest in INHERITED.items():require(sha(HERE/name)==digest,'Frozen V1 dependency '+name)
    v1.preflight(HERE/'freeze_v1.json',INHERITED['freeze_v1.json'])
    env=json.loads((HERE/'runtime_environment_sources_v1.json').read_text())
    require(sha(HERE/'runtime_environment_sources_v1.json')=='419ee5e0ab8f6f639091db44c5f8c7a873079e419a37e8b80bd5b38e92ef0a29','Runtime environment receipt')
    require(torch.__version__==env['torch_version'] and np.__version__==env['numpy_version'],'Same runtime versions')
    for item in env['source_files'].values():require(sha(item['path'])==item['sha256'],'Same native runtime source')
    require(freeze['case_count']==276 and freeze['logical_MLP_calls']==644,'V2 unchanged budget')
    return freeze


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,required=True);p.add_argument('--freeze-sha256',required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();preflight(a.freeze,a.freeze_sha256)
    require(not a.output_dir.exists(),'Fresh V2 output only');a.output_dir.mkdir(parents=True)
    start=time.monotonic();records=[];torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    result={'schema':'flow_current_bn_round16_cpu_v2','status':'INCOMPLETE','freeze_sha256':a.freeze_sha256,
        'BN_mode':'NATIVE_DENSE_CURRENT','torch':torch.__version__,'numpy':np.__version__,'device':'cpu',
        'autograd':False,'actual_full_model_forwards':0,'optimizer_steps':0,'case_records':[]}
    try:
        with torch.inference_mode():
            result['contract_checks']=v1.rejection_checks()
            for index,spec in enumerate(v1.cases(),1):
                require(time.monotonic()-start<300,'Five-minute execution cap')
                r=run_case(spec,a.output_dir,index);records.append(r)
                result['case_records'].append({'index':index,'path':'case_%03d.json'%index,'sha256':sha(a.output_dir/('case_%03d.json'%index)),
                    'passed':r['passed'],'dtype':r['dtype'],'scenario':r['scenario'],'mask':r['mask']})
                if index%30==0:print(json.dumps({'cases_complete':index,'failures':sum(not x['passed'] for x in records)}),flush=True)
        require(len(records)==276 and sum(len(r['actual_calls']) for r in records)==644,'Complete unchanged matrix')
        require(sum(c['fc1_rows'] for r in records for c in r['actual_calls'].values())==473300,'Whole-matrix FC rows')
        require(sum(c.get('mlp',0) for r in records for c in r['actual_calls'].values())==368,'Native MLP method budget')
        result['status']='CPU_MATRIX_COMPLETE';result['admission']='ADMITTED_FINITE_SYNTHETIC_CPU_ONLY' if all(r['passed'] for r in records) else 'NOT_ADMITTED'
    except Exception:result['exception']=traceback.format_exc();result['admission']='NOT_ADMITTED_INCOMPLETE'
    result['elapsed_seconds']=time.monotonic()-start;result['cases_complete']=len(records)
    result['failed_cases']=[r['index'] for r in records if not r['passed']]
    result['actual_logical_MLP_calls']=sum(len(r['actual_calls']) for r in records)
    result['actual_fc_rows_per_layer']=sum(c['fc1_rows'] for r in records for c in r['actual_calls'].values())
    result['actual_PSN_calls_each']=sum(c['sn1'] for r in records for c in r['actual_calls'].values())
    result['actual_native_MLP_method_calls']=sum(c.get('mlp',0) for r in records for c in r['actual_calls'].values())
    result['actual_native_BN_calls_each']=sum(c['bn1'] for r in records for c in r['actual_calls'].values())
    result['compact_resource_totals']={k:sum(r['resources'][k] for r in records) for k in RESOURCE_FIELDS}
    write(a.output_dir/'results.json',result)
    print(json.dumps({k:result[k] for k in ('status','admission','cases_complete','actual_logical_MLP_calls','elapsed_seconds')}),flush=True)
    if result['status']!='CPU_MATRIX_COMPLETE':raise SystemExit(2)


if __name__=='__main__':main()
