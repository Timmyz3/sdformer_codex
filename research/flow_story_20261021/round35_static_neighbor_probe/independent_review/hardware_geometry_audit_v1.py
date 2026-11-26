#!/usr/bin/env python3
"""Independent static360 pixel/row/plane audit; no neural forward or CUDA."""
from pathlib import Path
import hashlib,importlib.util,json,math
import numpy as np
import torch
P=Path(__file__).resolve().parent;R35=P.parent;ROOT=R35.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
CORE=ROOT/'e0_ordinary/mixed_integer_core.py';core=module('r35_review_core',CORE)
impl=module('r35_review_materializer',R35/'static_contract.py')
contract=read(R35/'ACTION_CONTRACT_V1.json')
HW=ROOT/'round34_deployability/hardware/actual_v2/STATIC360_CONTRACT_V2.json';hw=read(HW)
assert sha(CORE)==contract['core_sha256']=='ce07ae04c9f3d58738ae134cdc1c8276542a41a04584720b5d69359e220d1f9f'
membership=read(ROOT/'round33_predictability/data/MEMBERSHIP_V1.json')
assert set(contract['target_ids'])=={r['target_id'] for r in membership['records']}
assert contract['tile_ids_selection_order']==hw['selection_order']
assert set(hw['sorted_ids'])==set(range(209))|set(range(302,453)) and len(hw['sorted_ids'])==360
bits=np.unpackbits(np.asarray(hw['mask_words'],dtype='<u4').view(np.uint8),bitorder='little')[:910]
assert np.flatnonzero(bits).tolist()==hw['sorted_ids']
gy,gx=np.indices((240,320));complete=(gy<234)&(gx<315)
gold=complete & np.isin((gy//9)*35+gx//9,np.flatnonzero(bits))
assert gold.sum()==29160 and not np.any(gold[234:]) and not np.any(gold[:,315:])
selected,record=impl.materialize(contract,core,True);background,_=impl.materialize(contract,core,False)
expected={};checks=[];total=0;base=0
for name in core.TARGETS:
 g=contract['geometries'][name];t,h,w,k,co=[g[q] for q in ('T','H','W','K','Cout')]
 group=core.group_name(name);actual=selected[group].numpy();reference=gold if name in core.CONVS else np.zeros((h,w),bool)
 assert np.array_equal(actual,reference) and not background[group].any()
 yy,xx=np.indices((h,w));tail=(yy>=h//9*9)|(xx>=w//9*9)
 high=np.broadcast_to(tail|reference,(t,1,h,w)).copy().ravel();ta=np.broadcast_to(tail,(t,1,h,w)).copy().ravel();rows=t*h*w
 tail_t,high_t,mask=core.row_masks(t,1,h,w,'cpu','M',selected[group])
 assert np.array_equal(tail_t.numpy(),ta) and np.array_equal(high_t.numpy(),high)
 high8=int(high.sum());low4=rows-high8;planes={str(b):high8+(low4 if b>=4 else 0) for b in range(8)}
 work=sum(planes.values());ticks=work*math.ceil(k/128)*math.ceil(co/16);base_ticks=(8*int(ta.sum())+4*int((~ta).sum()))*math.ceil(k/128)*math.ceil(co/16)
 v=dict(T=t,H=h,W=w,K=k,Cout=co,rows=rows,tail_rows=int(ta.sum()),restored_complete_rows=int((high&~ta).sum()),high8_rows=high8,low4_rows=low4,
        selected_complete_tiles=mask['selected_complete_tiles'],selected_spatial_mask_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),high8_rows_sha256=hashlib.sha256(high.tobytes()).hexdigest(),row_planes=planes,logical_row_plane_work=work,accepted_bitplane128_core_beats=ticks,background_core_beats=base_ticks)
 if name in core.CONVS:
  assert (high8,low4,int(ta.sum()),ticks)==(322500,445500,30900,183204000)
  assert ticks==hw['per_operator_accepted_beats'] and ticks-base_ticks==48988800
 else:assert int((high&~ta).sum())==0 and ticks==base_ticks
 expected[name]=v;total+=ticks;base+=base_ticks
assert len(expected)==28 and len(core.CONVS)==4 and len(core.LINEARS)==24 and total-base==195955200
# Counterchecks exercise unchanged frozen whole-tile/tail rejection, not a GPU.
bad=selected[core.CONVS[0]].clone();bad[0,0]=False
try:core.row_masks(10,1,240,320,'cpu','M',bad)
except ValueError:checks.append('fragmented_tile_rejected')
else:raise AssertionError('fragment accepted')
bad=selected[core.CONVS[0]].clone();bad[239,319]=True
try:core.row_masks(10,1,240,320,'cpu','M',bad)
except ValueError:checks.append('true_tail_selection_rejected')
else:raise AssertionError('tail selected')
out=dict(schema='r35_independent_hardware_geometry_v1',status='PASS_GEOMETRY_MATERIALIZATION_ROW_PLANE_GOLDEN',bindings={str(p.relative_to(ROOT)):sha(p) for p in [R35/'ACTION_CONTRACT_V1.json',R35/'static_contract.py',CORE,HW]},selected_complete_tiles=360,spatial_selected_pixels_per_conv=29160,expected_affines=expected,all28_core_beats=total,background_all28_core_beats=base,added_four_patch_core_beats=total-base,tests=checks,GPU_used=False,neural_forward_executed=False,new_mask_AEE_known=False,full_network_cycle_claim=False)
(P/'hardware_geometry_audit_v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['expected_affines','bindings']},indent=2))
