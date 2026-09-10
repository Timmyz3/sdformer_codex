import sys
sys.dont_write_bytecode=True
assert sys.version_info[:2]==(3,12)
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PARSER=ROOT.parents[1]/'mechanism_rebuild_gh_20260906/scripts'
sys.path.insert(0,str(PARSER))
from screen_threshold_packets import sources,EXPECTED
import numpy as np
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((ROOT/'plan.json').read_text())
name=plan['scope']['module']; spec,S=sources()[name]
assert spec['input_shape']==[10,1,120,160,384]
assert spec['weight_layout']['base_address']==393216
assert spec['weight_layout']['row_bytes']==6144
assert spec['weight_layout']['source_group_count']==24
assert spec['weight_layout']['bank_formula']=='(address//row_bytes)%8'
S=S.reshape(10,19200,384)
lo,hi=plan['scope']['spatial_start'],plan['scope']['spatial_end_exclusive']
with (ROOT/'support.txt').open('x') as f:
 f.write(f'{hi-lo} 10 384 96 {lo}\n')
 for p in range(lo,hi):
  for c in range(384):
   mask=sum(int(S[t,p,c])<<t for t in range(10))
   f.write(f'{mask:03x}\n')
W=9*((7*np.arange(384)[:,None]+11*np.arange(96)[None,:]+3)%25-12)
assert int(np.abs(W).max())==108
with (ROOT/'weights.txt').open('x') as f:
 for row in W:
  f.write(' '.join(str(int(x)) for x in row)+'\n')
Y=S[:,lo:hi,:].astype(np.int64) @ W.astype(np.int64)
with (ROOT/'golden.txt').open('x') as f:
 for p in range(hi-lo):
  for t in range(10):
   f.write(' '.join(str(int(x)) for x in Y[t,p])+'\n')
record={'status':'CAPTURE_SUPPORT_ONLY_NO_WEIGHT_OR_FP_CLAIM','plan_sha256':sha(ROOT/'plan.json'),
 'script_sha256':sha(Path(__file__)),'parser_sha256':sha(PARSER/'screen_threshold_packets.py'),
 'capture_sha256':EXPECTED,'layer_spec':spec,'support_sha256':sha(ROOT/'support.txt'),
 'weights_sha256':sha(ROOT/'weights.txt'),'golden_sha256':sha(ROOT/'golden.txt'),
 'shape':[hi-lo,10,384,96],'source_ones':int(S[:,lo:hi,:].sum()),
 'support_encoding':'p,c order, each value is full-T10 bit mask; no input coefficient quantization claim'}
with (ROOT/'fixture.json').open('x') as f: json.dump(record,f,indent=2);f.write('\n')
print(json.dumps({'status':record['status'],'shape':record['shape'],'source_ones':record['source_ones']}))
