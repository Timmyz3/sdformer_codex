#!/usr/bin/env python3
"""R20 mask-generalized, no-score services; no numerical accelerator claim."""
import argparse,hashlib,importlib.util,json,types
from pathlib import Path
HERE=Path(__file__).resolve().parent;DATE=HERE.parents[1]
PRIOR=DATE/'compact_graph_round19/hardware/paired_service.py'
PRIOR_SHA='95c9ba99724a22a74ee7137010e505ce2073838e14dcf93be393957780bac1b2'
PRIOR_FREEZE=DATE/'compact_graph_round19/hardware/FREEZE_V1.json'
PRIOR_FREEZE_SHA='5270006ebb0abb722b683404faea756452a319b8278ab9b9904d38317c7004b6'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,message):
 if not ok:raise ValueError(message)
require(sha(PRIOR)==PRIOR_SHA and sha(PRIOR_FREEZE)==PRIOR_FREEZE_SHA,'Exact final R19 resource model')
spec=importlib.util.spec_from_file_location('_r20_frozen_r19_service',PRIOR);P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
ARMS=('FF','CC','F12','T8')
def validate_mask(mask):
 require(type(mask) is list and len(mask)==4 and all(type(r) is list and len(r)==5 and all(type(v) is bool for v in r) for r in mask),'Boolean 4x5 true spatial mask')
 return [r[:] for r in mask]
def mask_for(arm):
 require(arm in ARMS,'Fixed known no-score diagnostic anchor')
 return [[arm=='FF' or (arm=='F12' and y<3 and x<4) or (arm=='T8' and not(y<3 and x<4)) for x in range(5)] for y in range(4)]
def cover_mask(mask):
 mask=validate_mask(mask);out=[]
 for ty,row in enumerate(mask):
  for tx,fine in enumerate(row):
   sy,sx=ty*9,tx*9;ey,ex=min(sy+9,30),min(sx+9,40);step=1 if fine else 3
   for y in range(sy,ey,step):
    for x in range(sx,ex,step):out.append({'y':y,'x':x,'area':(min(y+step,ey)-y)*(min(x+step,ex)-x),'fine':fine})
 require(sum(c['area'] for c in out)==1200,'All true pixels exactly represented')
 return out
def layer(arm,*,native=False,mask=None):
 require(arm in ARMS or mask is not None,'Known anchor or explicit mask')
 mask=validate_mask(mask) if mask is not None else mask_for(arm)
 require(not native or arm=='FF' and all(all(row) for row in mask),'Unmodified native allfine only')
 before=P.cover;source_hash=sha(PRIOR);private_globals=dict(P.layer.__globals__)
 private_globals['cover']=lambda ignored:cover_mask(mask)
 invoke=types.FunctionType(P.layer.__code__,private_globals,'r20_mask_service',P.layer.__defaults__,P.layer.__closure__)
 result,schedule=invoke('native' if native else 'compact',arm)
 require(P.cover is before and P.layer.__globals__['cover'] is before and sha(PRIOR)==source_hash==PRIOR_SHA,'Frozen R19 helper identity/source unchanged')
 result.update(mask=[r[:] for r in mask],score_or_predictor_ticks=0,selector_ticks=0,
  complete_tile_fine_count=sum(mask[y][x] for y in range(3) for x in range(4)),
  true_tail_tile_fine_count=sum(mask[y][x] for y in range(4) for x in range(5) if y==3 or x==4),
  whole12_FC_rows_each=546000+result['FC_rows_each'])
 return result,schedule
def report():
 rows={'native':layer('FF',native=True)[0]}
 for arm in ARMS:rows[arm]=layer(arm)[0]
 native=rows['native'];comparisons={}
 for arm in ARMS:
  r=rows[arm];comparisons[arm]={
   'native_no_score_service_ticks':native['modeled_blocking_total_service_ticks'],
   'candidate_no_score_service_ticks':r['modeled_blocking_total_service_ticks'],
   'net_local_service_ticks_saved':native['modeled_blocking_total_service_ticks']-r['modeled_blocking_total_service_ticks'],
   'net_local_service_fraction_saved':(native['modeled_blocking_total_service_ticks']-r['modeled_blocking_total_service_ticks'])/native['modeled_blocking_total_service_ticks'],
   'net_PE_reserved_ticks_saved':native['modeled_PE_reserved_ticks']-r['modeled_PE_reserved_ticks'],
   'net_external_bytes_saved':native['external_total_bytes']-r['external_total_bytes'],
   'extra_predictor_service_tick_break_even_budget':native['modeled_blocking_total_service_ticks']-r['modeled_blocking_total_service_ticks'],
   'whole_network_speedup':None,'precision_or_quality_admission':None}
 return {'schema':'flow_r20_true20tile_no_score_service_v1','status':'CONDITIONAL_RESOURCE_MODEL_ONLY',
  'source_sha256':{str(p):sha(p) for p in (Path(__file__),PRIOR,PRIOR_FREEZE,P.HELPER,P.SCORE)},
  'resource_math':'Private copy of frozen R19 layer function globals, changing only the cell cover; no historical module or source mutation.',
  'graphs':rows,'versus_native_no_score':comparisons,
  'scope':['Fixed anchors have no score, no online sorting and no learned predictor; mask transport/true-cell descriptors and pooling remain charged.',
   'F12 and T8 are unequal-budget factor probes, not matched-compute algorithm competitors.',
   'Full-domain native BN, complete T10/C, true tails and explicit expansion/gather are retained. No weighted-BN service substitution.',
   'All timing assumptions, finite capacity/aggregate port checks and unclosed numerical FP/bank-address/RF boundaries are inherited from final R19.',
   'No actual R20 precision/GT results read or implied, no neural forward/GPU/RTL performed; no whole-network schedule or benefit claim.']}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'Exclusive new artifact')
 result=report();a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'sha256':sha(a.output),'summary':result['versus_native_no_score']}))
