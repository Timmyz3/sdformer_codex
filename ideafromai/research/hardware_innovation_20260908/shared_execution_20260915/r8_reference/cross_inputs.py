from pathlib import Path
import numpy as np,json
H=Path(__file__).resolve().parent;S=H.parent/'spatial_rr'
def rd(p):return np.asarray([int(s,16) for s in p.read_text().split()],np.uint32)
sets={};words=ids=origins=0
for stage in ['held','disjoint','sequences']:
 aa=[Path(x) for x in (H/f'{stage}.txt').read_text().splitlines()]
 bb=[Path(x) for x in (S/f'{stage}.txt').read_text().splitlines()]
 assert len(aa)==len(bb)
 for i,(a,b) in enumerate(zip(aa,bb)):
  source=np.fromfile(a/'source.bin','<u2').astype(np.uint32)
  identity=np.fromfile(a/'identity_fp32.bin','<u4')
  origin=np.fromfile(a/'origin.bin','<i4').view(np.uint32)
  assert np.array_equal(source,rd(b/'source.hex')),(stage,i,'source')
  assert np.array_equal(identity,rd(b/'identity.hex')),(stage,i,'identity')
  assert np.array_equal(origin,rd(b/'origin.hex')),(stage,i,'origin')
  words+=len(source);ids+=len(identity);origins+=len(origin)
 sets[stage]=len(aa)
r=dict(passed=True,sets=sets,source_words=words,FP32_identity_words=ids,origin32_scalars=origins,order_equal=True,different_functions=True,outputs_deliberately_not_shared=True,small_sets_differ='R8 12 vs spatial 15; do not rank their aggregate small cycles')
(H/'cross_inputs.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
