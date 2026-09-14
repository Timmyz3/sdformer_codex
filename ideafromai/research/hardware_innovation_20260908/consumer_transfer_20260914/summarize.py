"""Keep actual raw and consumer endpoints distinct in one compact index."""
from pathlib import Path
import csv,json
H=Path(__file__).resolve().parent
rows=[]
for family in ['bitmap_pipeline','bitmap_block_resident']:
 for stage in ['64','disjoint']:
  for key,r in json.loads((H/family/f'SUMMARY_{stage}.json').read_text()).items():
   mode,bp,repeat=key.split('_')
   rows.append(dict(family=family,workload=stage,endpoint='raw',contexts=1,z_port_bits=208,mode=int(mode[1:]),backpressure=int(bp[1:]),repeat=int(repeat[6:]),cycles=r['service_cycles']))
for family in ['count_rr','bitmap_consumer','bitmap_block_consumer']:
 for stage in ['held','disjoint']:
  data=json.loads((H/family/f'SUMMARY_{stage}.json').read_text())
  assert data['passed']
  for r in data['results']:
   rows.append(dict(family=family,workload='64' if stage=='held' else stage,endpoint='FP32_identity_to_J20_to_I24',contexts=2 if family=='count_rr' else 1,z_port_bits=416 if family=='count_rr' else 208,mode=r['mode'],backpressure=r['stall'],repeat=r['command'],cycles=r['total_cycles']))
with (H/'comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
print(json.dumps(dict(rows=len(rows),note='Actual cycles; resource points and endpoints differ. Not an equal-area ranking.')))
