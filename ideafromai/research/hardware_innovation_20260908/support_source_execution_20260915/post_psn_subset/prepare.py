#!/opt/anaconda3/bin/python3.12
from pathlib import Path
import json, struct
import numpy as np
HERE=Path(__file__).resolve().parent
src=HERE.parents[1]/'support_lut_execution_20260915'/'cases.npz'
d=np.load(src,allow_pickle=False)
indices=list(np.flatnonzero(d['is_real']))+[i for i,n in enumerate(d['case_name']) if str(n) in ['diagnostic_zero','diagnostic_onehot_escape','diagnostic_multibit_escape','diagnostic_signed_dense','diagnostic_negative_gain_constant']]
assert len(indices)==37
rows=[]
with (HERE/'cases.bin').open('wb') as f:
 f.write(struct.pack('<II',0x50534e35,len(indices)))
 f.write(d['A'].astype('<i2').tobytes())
 for i in indices:
  name=str(d['case_name'][i]); y=d['S'][i].astype('int64')@d['W'][i].astype('int64'); u=np.einsum('ts,psh->pth',d['A'].astype('int64'),y)
  g=np.where(d['constant_channels'][i][None,None],d['constant_gate'][i][None],np.where(d['positive_gain'][i][None,None],u>=d['tau'][i][None],u<=d['tau'][i][None]))
  assert np.all((-2**23<=y)&(y<2**23)) and np.all((-2**47<=u)&(u<2**47))
  assert np.all((-2**47<=d['tau'][i])&(d['tau'][i]<2**47))
  assert np.array_equal(y,d['Y'][i]) and np.array_equal(u,d['U'][i]) and np.array_equal(g,d['gold'][i])
  f.write(name.encode().ljust(64,b'\0'));f.write(struct.pack('<I',bool(d['is_real'][i])))
  for arr,ty in [(y,'<i4'),(u,'<i8'),(d['tau'][i],'<i8'),(d['positive_gain'][i],'u1'),(d['constant_channels'][i],'u1'),(d['constant_gate'][i],'u1'),(g,'u1')]:f.write(arr.astype(ty).tobytes())
  rows.append({'case':name,'real':bool(d['is_real'][i]),'Y_min':int(y.min()),'Y_max':int(y.max()),'U_min':int(u.min()),'U_max':int(u.max()),'ties':int((u==d['tau'][i][None]).sum())})
sub=np.array([sum(d['A'][:,h*5+j].astype('int64') for j in range(5) if c>>j&1) if c else np.zeros(10,dtype='int64') for h in range(2) for c in range(32)])
assert sub.min()>=-2**15 and sub.max()<2**15
(HERE/'inputs.json').write_text(json.dumps({'source':str(src),'cases':rows,'A_nonzeros':int(np.count_nonzero(d['A'])),'subset_min':int(sub.min()),'subset_max':int(sub.max()),'Y_U_gate_checked_each':37*30720,'status':'PASS'},separators=(',',':'))+'\n')
print('PASS export',len(rows),'cases; independent Y/U/gate',37*30720)
