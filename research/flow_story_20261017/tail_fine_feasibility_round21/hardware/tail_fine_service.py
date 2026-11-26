#!/usr/bin/env python3
"""R21 bounded T8+k1 service obligations. No neural forward or quality input."""
import argparse, hashlib, importlib.util, json
from pathlib import Path
HERE=Path(__file__).resolve().parent; DATE=HERE.parents[1]
PRIOR=DATE/'action_domain_round20/hardware/domain_service.py'
PRIOR_SHA='790da6e3db3804844e7368a2f799f0e788498fa8cc81cf0c2484ba61e767a483'
FREEZE=DATE/'action_domain_round20/hardware/FREEZE_V1.json'
FREEZE_SHA='f1c43c35a9b482a0e6b219d21e988a45067c8b022cd6b877a06937f3b4198859'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,msg):
 if not ok:raise ValueError(msg)
require(sha(PRIOR)==PRIOR_SHA and sha(FREEZE)==FREEZE_SHA,'Frozen same-resource R20 mask schedule')
spec=importlib.util.spec_from_file_location('_r21_fixed_r20_service',PRIOR); P=importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
def independently_cover(mask):
 # Per-pixel grouping, independent of prior cell-cover loop.
 groups={}
 for y in range(30):
  for x in range(40):
   ty,tx=y//9,x//9
   key=(y,x) if mask[ty][tx] else (ty*9+(y%9)//3*3,tx*9+(x%9)//3*3)
   groups.setdefault(key,0);groups[key]+=1
 return groups
def report():
 native,_=P.layer('FF',native=True); t8,_=P.layer('T8'); rows={}; checks=[]
 for y in range(3):
  for x in range(4):
   mask=P.mask_for('T8');mask[y][x]=True; group=independently_cover(mask)
   require(len(group)==408 and sum(group.values())==1200,'True complete-domain 408 representatives')
   old_identity=P.P.cover; row,schedule=P.layer('T8',mask=mask)
   require(P.P.cover is old_identity and sha(PRIOR)==PRIOR_SHA,'No historical source/global mutation')
   require({(v['y'],v['x']):v['area'] for v in P.cover_mask(mask)}==group,'Independent area/position cover')
   require(row['FC_rows_each']==4080 and row['whole12_FC_rows_each']==550080,'Complete T10 FC rows')
   require(row['complete_tile_fine_count']==1 and row['true_tail_tile_fine_count']==8,'One complete plus all eight true tail tiles')
   require(row['full_BN_domain_elements']==23040000 and row['work'].get('BN_area_multiply',0)==0,'Both native full-domain BN retained')
   P.P.validate_schedule(schedule)
   row['actual_predictor_or_selector_implemented']=False
   rows[f'fine_{y}_{x}']=row;checks.append(f'fine_{y}_{x}: true cover/full T/finite capacity/port and barrier checks PASS')
 common=('modeled_PE_reserved_ticks','modeled_PE_FMA_issue_ticks','modeled_blocking_total_service_ticks','external_total_bytes','local_activation_read_bytes','local_activation_write_bytes')
 require(all(len({r[k] for r in rows.values()})==1 for k in common),'Fixed budget is location-independent in this stated schedule only')
 c=rows['fine_0_0']; delta=native['modeled_blocking_total_service_ticks']-c['modeled_blocking_total_service_ticks']
 return {'schema':'flow_r21_tail_fine_k1_conditional_service_v1','status':'CONDITIONAL_RESOURCE_MODEL_ONLY',
  'source_sha256':{str(p):sha(p) for p in (Path(__file__),PRIOR,FREEZE,P.PRIOR,P.PRIOR_FREEZE,P.P.HELPER)},
  'native_no_score':native,'T8_no_score':t8,'T8_plus_one_complete_tile_no_score':rows,
  'independent_geometry_and_resource_checks':checks,
  'versus_native_no_score':{'net_local_service_ticks_saved':delta,'fraction_local_service_saved':delta/native['modeled_blocking_total_service_ticks'],
    'extra_predictor_service_ticks_break_even_necessary_upper_bound':delta,
    'net_external_bytes_saved':native['external_total_bytes']-c['external_total_bytes'],
    'whole12_MLP_FC_MAC_cancelled_fraction':(12000-4080)/12000/12,
    'whole_network_cycles_or_energy':None,'precision_or_quality_admission':None},
  'fixed122_inference_inventory':{'native':8,'compact':114,'quality_endpoints':120,'repeat':2,
    'FC_rows_each':16*558000+10*549360+96*550080,'native_BN_calls':122*78,
    'all_BN_input_elements':122*1414344960,'compact_target_full_BN_elements':114*23040000,
    'compact_target_BN1_gather_elements':(8*12000+10*3360+96*4080)*1536},
  'scope':['Frozen R19/R20 finite resources and serialized service assumptions unchanged; no numerical FP RTL or detailed bank-address proof.',
   'Each tested mask keeps all T/C and all true tails fine, and one of twelve complete9x9 tiles fine. Full native BN, expansion, gather, residual and pool/pack are charged.',
   'These fixed enumeration calls have no predictor or selector. Any deployed score, native-PSN input, vector scaling/group statistics, mask selection and buffering must be added fairly on the same resources.',
   'No model/GT/quality input, no GPU/SSH/new RTL, no transfer of R17 dense quality or R16 compact-expanded equivalence.',
   'Any H8 frame with zero feasible singleton stops this T8+k1 selector family; an oracle-feasible H8 set only permits a later bounded information test, not deployment or held-out success.',
   'Single target cancellation is only5.5% of the twelve MLP original FC MAC, before remaining PE work. No10% net whole-network claim.']}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'Exclusive new result')
 out=report();a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'sha256':sha(a.output),'comparison':out['versus_native_no_score'],'counts':out['fixed122_inference_inventory']}))
