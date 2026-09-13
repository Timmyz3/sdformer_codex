"""Finite geometric upper bound only; no RTL/performance prediction, no model sweep."""
import json
from pathlib import Path
Q={(y,x) for y in range(2) for x in range(2)}
def fan(s):
 return {(s[0]-p[0],s[1]-p[1]):p for p in Q if 0<=s[0]-p[0]<3 and 0<=s[1]-p[1]<3}
rows=[]
for y in range(4):
 for x in (0,2):
  a=(y,x);b=(y,x+1);fa=fan(a);fb=fan(b);shared=sorted(fa.keys()&fb.keys())
  assert all((fb[k][0]-fa[k][0],fb[k][1]-fa[k][1])==(0,1) for k in shared)
  rows.append({'a':a,'b':b,'a_consumers_per_og':len(fa),'b_consumers_per_og':len(fb),'common_taps':shared,'common_count':len(shared)})
base=sum(r['a_consumers_per_og']+r['b_consumers_per_og'] for r in rows);shared=sum(r['common_count'] for r in rows)
assert base==36 and shared==12
out={'scope':'4x4 native source,2x2 output,3x3 stride1,oneCin4,12O8groups; fixed disjoint horizontal pairs; all sources/weights active, no existing cache reuse','pairs':rows,'unjoined_consumers_per_og':base,'maximum_cancelled_same_weight_consumers_per_og':shared,'all12_og_unjoined':base*12,'all12_og_cancelled_upper':shared*12,'fourW128_reads_unjoined_upper':base*12*4,'fourW128_reads_cancelled_upper':shared*12*4,'fraction_upper':shared/base,'caveats':'Bound shrinks with source/code mismatch,padding,weight mask and existing cache hits; no psum write saving, no source read saving and no cycle claim'}
Path('geometry_bound.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='pairs'},indent=2))
