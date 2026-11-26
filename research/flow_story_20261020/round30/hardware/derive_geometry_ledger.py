"""Independently enumerate per-pixel domain counts for both real geometries."""
import json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
c=json.loads((HERE/'MASK_CONTRACT_V1.json').read_text())
out={}
for geometry,groups in c['geometries'].items():
 rows=[];totals=dict(affine_rows=0,row_planes_Q4=0,row_planes_M=0,row_planes_Q8=0)
 for group,g in groups.items():
  h,w,k=g['H'],g['W'],g['Q8_complete_tiles']
  spatial_tail=sum(y>=h//9*9 or x>=w//9*9 for y in range(h) for x in range(w))
  r=10*h*w;tail=10*spatial_tail;restored=810*k
  multiplicity=2 if group.endswith('.mlp') else 1
  v=dict(group=group,H=h,W=w,T=10,affines_sharing_mask=multiplicity,
         rows_per_affine=r,tail_rows_per_affine=tail,restored_rows_per_affine=restored,
         high8_rows_per_affine=tail+restored,
         row_planes_Q4=4*r+4*tail,row_planes_M=4*r+4*(tail+restored),row_planes_Q8=8*r)
  rows.append(v)
  totals['affine_rows']+=r*multiplicity
  for key in ('row_planes_Q4','row_planes_M','row_planes_Q8'):totals[key]+=v[key]*multiplicity
 out[geometry]=dict(groups=rows,totals=totals)
result=dict(schema='flow_r30_full_crop_domain_conservation_v1',geometries=out,
            contract_sha256=hashlib.sha256((HERE/'MASK_CONTRACT_V1.json').read_bytes()).hexdigest(),
            scope='28selected affines logical plane rows; this is not full-network cycles or GPU issued padding',
            crop_full_network_cost_claim=False,model_forwards=0,GPU_used=False,GT_read=False)
p=HERE/'FULL_CROP_DOMAIN_LEDGER_V1.json';assert not p.exists()
p.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v['totals'] for k,v in out.items()},indent=2))
