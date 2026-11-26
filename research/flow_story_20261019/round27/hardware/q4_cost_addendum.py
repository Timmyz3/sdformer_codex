#!/usr/bin/env python3
"""Prospective Q4 cost ONLY. No new kernel/model, training, quality or GPU.

R27 remains frozen Q6. A private copy of its cost function changes exactly two
constants: precision tag Q6->Q4 and complete-tile plane count6->4. All other
resources/ports, common work, scale, fullgrid/BN/T and true tails remain fixed.
"""
import argparse, ast, collections, dataclasses, hashlib, inspect, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import unified_service as s
SOURCE_SHA='7583db372a9b5883685e9fb62fbc1f26ecbd1ff1e3aff19bac3baf996f5aa6b8'
RESULT_SHA='f1aca17e9a6f7e3afc6f57260fce313ae8197d61eedb5c5c9c0d65603ac6ad14'
FREEZE_SHA='43c6268d4ca0b0650b66d3978f35b225e9516923b63c963e3b7b9391161016ee'

def build_private_function():
 s.require(s.sha(HERE/'unified_service.py')==SOURCE_SHA,'Frozen R27 model')
 source=inspect.getsource(s.affine_service);tree=ast.parse(source)
 class Narrow(ast.NodeTransformer):
  def __init__(self):self.tag=0;self.planes=0
  def visit_Compare(self,node):
   if isinstance(node.left,ast.Name) and node.left.id=='precision' and len(node.comparators)==1 and isinstance(node.comparators[0],ast.Constant) and node.comparators[0].value=='Q6':
    node.comparators[0].value='Q4';self.tag+=1
   return self.generic_visit(node)
  def visit_keyword(self,node):
   if node.arg=='planes' and isinstance(node.value,ast.Constant) and node.value.value==6:node.value.value=4;self.planes+=1
   return self.generic_visit(node)
 transform=Narrow();tree=transform.visit(tree);ast.fix_missing_locations(tree)
 s.require(transform.tag==transform.planes==1,'Only precision guard and complete plane count change')
 namespace=dict(s.__dict__);old=s.affine_service;exec(compile(tree,'<private_R27_cost_only_Q4>','exec'),namespace)
 s.require(s.affine_service is old,'Frozen model globals/function not changed')
 return namespace['affine_service'],dict(precision_guard_changes=1,fulltile_plane_changes=1,original_globals_mutated=False,private_AST_sha256=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest())

def calculate():
 s.require(s.sha(HERE/'UNIFIED_SERVICE_V1.json')==RESULT_SHA and s.sha(HERE/'SERVICE_FREEZE_V1.json')==FREEZE_SHA,'Original results/freeze fixed')
 raw=s.read(HERE/'UNIFIED_SERVICE_V1.json');inv=raw['inventory'];q4,change=build_private_function()
 variants=[('primary',s.Machine(),next(r for r in raw['results'] if r['scope']=='MLP12_patch4' and r['implementation']=='Q8_strong')['summary'])]
 for case in raw['sensitivity']:
  variants.append((case['label'],s.Machine(**case['machine']),next(r for r in case['results'] if r['implementation']=='Q8_strong')['summary']))
 rows=[];primary_records=[]
 keys=('PE_arithmetic_ticks','PE_reserved_ticks','ticks','DMA_bytes','local_input_bytes','local_output_bytes','weight_SRAM_bytes','jobs')
 for label,m,baseline_total in variants:
  old=collections.Counter();new=collections.Counter();records=[];logical_planes=0
  for op in inv['affines']:
   if op['scope'] not in ('MLP12','patch4'):continue
   p=s.affine_service(op,m,'int8_parallel','Q8');b=s.affine_service(op,m,'bitserial','Q8')
   base=min((p,b),key=lambda x:(x['PE_reserved_ticks'],x['ticks']));candidate=q4(op,m,'bitserial','Q4')
   d=op['row_domains'][0];independent=d['rows']*8-d['complete_rows']*4
   s.require(candidate['logical_row_planes']==independent,'Independent true-tail Q4 row-plane conservation')
   s.require(candidate['rows']==op['rows'] and candidate['source_weight_DMA_bytes']==b['source_weight_DMA_bytes'],'No spatial rows or all8-tail weights removed')
   logical_planes+=independent
   records.append(dict(name=op['name'],scope=op['scope'],ordinary_Q8=base,Q4_cost_only=candidate))
   for key in keys:old[key]+=base.get(key,0);new[key]+=candidate.get(key,0)
  common={k:baseline_total[k]-old[k] for k in keys};s.require(all(v>=0 for v in common.values()),'Shared common graph services unchanged')
  total={k:common[k]+new[k] for k in keys};comparisons={k:dict(baseline=baseline_total[k],Q4_candidate=total[k],reduction_fraction=1-total[k]/baseline_total[k],additional_budget_for10=.9*baseline_total[k]-total[k]) for k in ('PE_arithmetic_ticks','PE_reserved_ticks','ticks','DMA_bytes')}
  s.require(logical_planes==17731680,'All28 Q4 hypothetical row-plane total')
  rows.append(dict(variant=label,machine=dataclasses.asdict(m),common=common,baseline_selected=dict(old),Q4_selected=dict(new),Q4_total=total,comparisons=comparisons,logical_row_planes=logical_planes))
  if label=='primary':primary_records=records
 return dict(schema='flow_r27_prospective_Q4_cost_addendum_v1',status='COST_ONLY_NOT_EXECUTED_OR_QUALITY_QUALIFIED',private_model_change=change,
  source_sha256={str(HERE/n):s.sha(HERE/n) for n in ('q4_cost_addendum.py','unified_service.py','UNIFIED_SERVICE_V1.json','SERVICE_FREEZE_V1.json')},variants=rows,primary_selected_records=primary_records,
  scope='All12MLP+patch4 complete9x9 outputtiles4planes, all true tails8planes; signed two-complement floor truncation, no offset, unchanged currentBN/fullT/fullspatial.',
  actual_kernel_implemented=False,actual_quality_available_for_this_scope=False,model_forwards=0,GPU_used=False,GT_read=False,R27_source_or_protocol_changed=False,
  decision_limit='Positive modeled PE headroom is necessary only. It cannot establish acceptable accuracy, flow-specific allocation, calibrated hardware speedup, or authorize changing the running fixedQ6 recovery.')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();s.require(not a.output.exists(),'Exclusive addendum output')
 result=calculate();a.output.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps([dict(variant=x['variant'],**x['comparisons']['PE_arithmetic_ticks']) for x in result['variants']],indent=2))
