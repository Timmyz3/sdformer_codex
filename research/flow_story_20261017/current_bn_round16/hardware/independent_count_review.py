#!/usr/bin/env python3
"""Independent production/tiny FC geometry, from saved input tensors, no forward."""
import hashlib,itertools,json
from collections import Counter
from pathlib import Path
import numpy as np
import service_model as m

HERE=Path(__file__).resolve().parent;IMPL=HERE.parent/'implementation'
def array_sha(a):
 a=np.ascontiguousarray(a)
 return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def main():
 result_path=IMPL/'actual_v1/results.json';result=json.loads(result_path.read_text());calls=Counter();rows=Counter();checked=0
 specs=[(*s[:2],*s[2],*s[3:],1) for s in itertools.product((17,29,43),('float64','float32'),((9,9),(11,14),(3,4)),('FF','CC','checkerboard'),('random','constant','low_variance','near_sn1','near_sn2'))]
 specs += [(17,d,11,14,k,'random',2) for d,k in itertools.product(('float64','float32'),('FF','CC','checkerboard'))]
 m.require(len(specs)==len(result['case_records'])==276,'Fixed independent matrix')
 for i,(spec,entry) in enumerate(zip(specs,result['case_records']),1):
  casepath=result_path.parent/entry['path'];m.require(m.sha(casepath)==entry['sha256'],'Actual case JSON hash')
  record=json.loads(casepath.read_text());seed,dtype,h,w,kind,scenario,b=spec
  m.require(tuple(record[k] for k in ('seed','dtype','height','width','mask','scenario','batch'))==spec,'Case identity/order')
  points=m.points(h,w,m.masks(h,w,'checker' if kind=='checkerboard' else kind));r=len(points)
  archive=result_path.parent/record['array_archive']['path'];m.require(m.sha(archive)==record['array_archive']['sha256'],'Actual archive hash')
  with np.load(archive,allow_pickle=False) as raw:
   for mode in ('compact','expanded')+(('direct',) if kind=='FF' else ()):
    actual=[]
    for fc in ('fc1','fc2'):
     key=mode+'__'+fc+'_input';a=raw[key];meta=record['array_archive']['payloads'][key]
     m.require(array_sha(a)==meta['sha256'] and list(a.shape)==meta['shape'] and str(a.dtype)==meta['dtype'],'Actual FC input payload')
     actual.append(a.size//a.shape[-1]);checked+=1
    expected=10*b*(r if mode=='compact' else h*w)
    m.require(actual==[expected,expected],'Independent rows from true saved FC input')
    m.require(record['actual_calls'][mode]['fc1_rows']==record['actual_calls'][mode]['fc2_rows']==expected,'Hooks agree with tensor-derived rows')
    for sn in ('sn1','sn2'):m.require(record['actual_calls'][mode][sn]==1,'No PSN replay reported')
    rows[mode]+=expected;calls[mode]+=1
 m.require(dict(rows)=={'compact':152260,'expanded':240780,'direct':80260},'Whole small matrix FC rows')
 m.require(sum(calls.values())==644 and calls['expanded']+calls['direct']==368,'Whole small matrix paths')
 out={'schema':'flow_r16_hardware_independent_FC_input_audit_v1','status':'PASS_GEOMETRY_COUNTS_ONLY','source_sha256':{str(Path(__file__)):m.sha(__file__),str(Path(m.__file__)):m.sha(m.__file__),str(result_path):m.sha(result_path)},
  'case_count':276,'actual_FC_input_payloads_independently_hashed':checked,'FC_rows_each_by_path':dict(rows),'FC_rows_each_total':sum(rows.values()),'path_calls':dict(calls),'logical_MLP_calls':sum(calls.values()),'PSN_calls_each_from_per_case_hook_records':sum(calls.values()),
  'numerical_gate_from_frozen_impl':result['admission'],'new_numerical_forwards':0,
  'scope':'Read and hash actual saved FC input tensors, independently reconstruct true-tail geometry and count rows. No numerical gate is changed; mathematical raw arrays independently audited by data reviewer.'}
 with (HERE/'INDEPENDENT_IMPL_FC_COUNT_REVIEW.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
