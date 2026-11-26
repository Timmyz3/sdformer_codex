#!/usr/bin/env python3
"""Fixed H1 two-slot source work and conditional same-resource service; no model."""
import argparse,hashlib,importlib.util,json,types
from pathlib import Path
HERE=Path(__file__).resolve().parent;DATE=HERE.parents[1]
PRIOR=DATE/'compact_graph_round19/hardware/paired_service.py'
PRIOR_SHA='95c9ba99724a22a74ee7137010e505ce2073838e14dcf93be393957780bac1b2'
FREEZE=DATE/'compact_graph_round19/hardware/FREEZE_V1.json'
FREEZE_SHA='5270006ebb0abb722b683404faea756452a319b8278ab9b9904d38317c7004b6'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,msg):
 if not ok:raise ValueError(msg)
require(sha(PRIOR)==PRIOR_SHA and sha(FREEZE)==FREEZE_SHA,'Frozen same-resource full-domain BN schedule')
s=importlib.util.spec_from_file_location('_r24_frozen_resource_schedule',PRIOR);P=importlib.util.module_from_spec(s);s.loader.exec_module(P)
ARMS=('FF','T2')+tuple(f'fine_{y}_{x}' for y in range(3) for x in range(4))
def rectangles(arm):
 require(arm in ARMS,'Known fixed H1 action');fine=tuple(map(int,arm.split('_')[1:])) if arm.startswith('fine_') else None;out=[]
 for ty in range(4):
  for tx in range(5):
   sy,sx=9*ty,9*tx;ey,ex=min(sy+9,30),min(sx+9,40)
   if arm=='FF' or ty==3 or tx==4 or fine==(ty,tx):
    out.extend([y,y+1,x,x+1,1] for y in range(sy,ey) for x in range(sx,ex))
   else:
    for y in range(sy,ey,3):
     for x in range(sx,ex,3):out.extend(([y,y+1,x,x+3,3],[y+1,y+3,x,x+3,6]))
 return out
def independent_pixel_groups(arm):
 fine=tuple(map(int,arm.split('_')[1:])) if arm.startswith('fine_') else None;groups={}
 for y in range(30):
  for x in range(40):
   ty,tx=y//9,x//9
   if arm=='FF' or ty==3 or tx==4 or fine==(ty,tx):key=(y,x)
   else:key=(y-y%3+(1 if y%3 else 0),x-x%3)
   groups.setdefault(key,[]).append((y,x))
 return groups
def layer(arm,native=False):
 cells=rectangles(arm);groups=independent_pixel_groups(arm)
 require(len(cells)==len(groups) and sum(r[4] for r in cells)==1200,'Unique full1200-pixel support')
 for y,yy,x,xx,area in cells:require(set(groups[(y,x)])=={(r,c) for r in range(y,yy) for c in range(x,xx)} and area==(yy-y)*(xx-x),'Independent rectangle/no-gap/no-overlap check')
 old=P.cover;glob=dict(P.layer.__globals__)
 glob['cover']=lambda ignored:[{'y':y,'x':x,'area':a,'fine':a==1} for y,yy,x,xx,a in cells]
 fn=types.FunctionType(P.layer.__code__,glob,'r24_service',P.layer.__defaults__,P.layer.__closure__)
 out,sch=fn('native' if native else 'compact',arm)
 require(P.cover is old and P.layer.__globals__['cover'] is old and sha(PRIOR)==PRIOR_SHA,'Historical source/globals unmodified')
 P.validate_schedule(sch)
 expected=1200 if arm=='FF' else 444 if arm=='T2' else 507
 require(out['producer_representatives']==expected and out['FC_rows_each']==expected*10,'Complete T10 actual rectangle rows')
 require(out['full_BN_domain_elements']==23040000 and out['work'].get('BN_area_multiply',0)==0,'Full native BN not weighted representative statistics')
 out.update(rectangles=cells,whole12_FC_rows_each=546000+expected*10,score_or_predictor_ticks=0,selector_ticks=0,
  action='Fixed H1 two rectangles within each coarse3x3; all eight true tail tiles fine; no runtime partition score')
 return out
def report():
 rows={'native':layer('FF',True)};rows.update({arm:layer(arm) for arm in ARMS});n=rows['native'];comparison={}
 for arm,r in rows.items():
  if arm=='native':continue
  saved=n['modeled_blocking_total_service_ticks']-r['modeled_blocking_total_service_ticks']
  comparison[arm]={'net_local_service_ticks_saved':saved,'fraction_local_service_saved':saved/n['modeled_blocking_total_service_ticks'],
   'net_external_bytes_saved':n['external_total_bytes']-r['external_total_bytes'],'additional_predictor_break_even_necessary_tick_upper_bound':saved,
   'whole12_MLP_FC_MAC_cancelled_fraction':(12000-r['FC_rows_each'])/12000/12,'whole_network_benefit':None,'quality_admission':None}
 require(len({rows[f'fine_{y}_{x}']['modeled_blocking_total_service_ticks'] for y in range(3) for x in range(4)})==1,'Position-invariant work for this fixed schedule only')
 return {'schema':'flow_r24_fixed_h1_two_slot_service_v1','status':'CONDITIONAL_RESOURCE_MODEL_ONLY','source_sha256':{str(p):sha(p) for p in (Path(__file__),PRIOR,FREEZE,P.HELPER)},
  'graphs':rows,'versus_native_no_score':comparison,'geometry_and_resource_checks':'PASS14 masks: independent pixel groups, true rectangles, fullT, nativeBN, finite capacity/port and barrier schedule checks',
  'planned_actual122':{'native':8,'compact':114,'quality_endpoints':120,'replays':2,'FC_rows_each':16*558000+10*550440+96*551070,'BN_calls':122*78,
   'all_BN_input_elements':122*1414344960,'compact_target_full_BN_elements':114*23040000,'compact_target_BN1_gather_elements':(8*12000+10*4440+96*5070)*1536},
  'limits':['No new model/GT/GPU/RTL. This is the frozen R19 finite schedule with only true region support substituted, not a numerically validated accelerator.',
   'FixedH1 no runtime orientation score; enumeration no deployment selector. Future score/selector and same-cost strong control obligations remain additional.',
   'Pool fullu, representative stores, fullT PSN/FC, full-domain BN and fullZ materialization/three reads, gather, repeat, residual and descriptor transactions all retained.',
   'Final partial eight-point batches are included; representative counts and MAC are not converted directly to cycles. PE reservation includes recurrence and drains.',
   '2MiB activation/512KiB weights,16x1RW activation banks,8x32 PE and serialized64B bus assumptions inherited. Physical addresses/coeffRF ports/FP numeric order/area/whole-network trace unclosed.',
   'T2 cancels5.25%, singleton4.8125% of all12MLP originalFC MAC. Not whole-network PE/latency/energy, no transferred quality, no reopening failedT8+k1.']}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'Exclusive new result')
 out=report();a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'sha256':sha(a.output),'counts':out['planned_actual122'],'anchors':{k:out['versus_native_no_score'][k] for k in ('FF','T2','fine_0_0')}}))
