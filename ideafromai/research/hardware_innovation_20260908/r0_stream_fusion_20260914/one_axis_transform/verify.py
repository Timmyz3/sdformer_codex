from pathlib import Path
import json,numpy as np
from prepare import Bt,At,G2
H=Path(__file__).resolve().parent
rows=json.loads((H/'results.json').read_text());assert len(rows)==52
old=json.loads((H.parent.parent/'r0_execution_trials_20260913/winograd/rtl_results.json').read_text())
old_index={(r['fixture'],r['stress']):r for r in old if r['kind']=='direct'}
old_checks=0
for r in rows:
 assert r['cycles']==sum(r['state_cycles'])
 assert r['state_cycles'][14]==r['state_cycles'][15]==0
 if r['kind']=='one_axis':assert r['state_cycles'][10]==0
 if r['kind']=='direct' and r['fixture'].startswith('real_'):
  ref=old_index[(r['fixture'],r['stress'])]
  for f in ('cycles','configuration_cycles','weight_requests','weight_responses','cache_hits','alu_vector_cycles','input_stall','output_stall','request_stall','state_cycles'):
   assert r[f]==ref[f],(r['fixture'],r['stress'],f);old_checks+=1
basis=0
for bit in range(16):
 S=np.zeros((4,4),np.int64);S.flat[bit]=1;V=S@Bt.T
 for tap in range(9):
  W=np.zeros((3,3),np.int64);W.flat[tap]=1;U=W@G2.T
  direct=np.array([[np.sum(S[y:y+3,x:x+3]*W) for x in range(2)] for y in range(2)])
  M=np.array([np.sum(U*V[y:y+3],axis=0) for y in range(2)])
  assert np.array_equal(M@At.T,2*direct);basis+=1
letters=set()
for pattern in range(16):
 d=np.array([(pattern>>x)&1 for x in range(4)]);letters.update(map(int,Bt@d))
assert letters=={-1,0,1,2}
packed_coefficients=0;straddled=0;half_only=0
for path in sorted((H/'fixtures').iterdir()):
 if not path.is_dir():continue
 z=np.load(path/'fixture.npz')
 for kind,key,bits in [('direct','Wq',16),('one_axis','U2',18)]:
  expected=z[key].reshape(6,16,-1).transpose(0,2,1).reshape(-1,16)
  raw=np.frombuffer((path/f'{kind}.bin').read_bytes(),np.uint8)
  words=np.unpackbits(raw,bitorder='little').reshape(-1,bits)
  unsigned=np.sum(words.astype(np.int64)*(1<<np.arange(bits)),axis=1)
  decoded=np.where(unsigned>=(1<<(bits-1)),unsigned-(1<<bits),unsigned)
  assert np.array_equal(decoded,expected.reshape(-1));packed_coefficients+=len(decoded)
  if kind=='one_axis':
   support=np.any(expected[:,:8]!=0,axis=1).astype(int)+2*np.any(expected[:,8:]!=0,axis=1).astype(int)
   for i,mask in enumerate(support):
    if not mask:continue
    first=(i*288+(0 if mask&1 else 144))//256
    last=(i*288+(288 if mask&2 else 144)-1)//256
    assert last-first<=1
    straddled+=last!=first;half_only+=mask!=3
out={'complete':True,'runs':52,'checked_outputs':sum(r['values_checked'] for r in rows),'old_direct_scalar_or_array_checks':old_checks,
 'native_basis_identities':basis,'binary_row_patterns':16,'V_letters':sorted(letters),'packed_coefficients_checked':packed_coefficients,
 'live_U2_vectors_crossing_CR_rows':int(straddled),'live_half_only_U2_vectors':int(half_only),'all_pass':True,
 'scope':'No RTL rerun; exact identity/format and actual counters audit. Actual RTL outputs checked by tb.cpp.'}
(H/'checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
