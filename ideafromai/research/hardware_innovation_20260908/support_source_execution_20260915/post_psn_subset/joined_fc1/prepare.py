from pathlib import Path
import json
import numpy as np
P=Path(__file__).resolve().parent
src=P.parents[2]/'support_lut_execution_20260915/cases.npz'
d=np.load(src,allow_pickle=False);a=d['A'].astype(np.int64);D=d['D'].astype(np.int64)
positive_sum=np.maximum(a,0).sum(1);negative_sum=np.minimum(a,0).sum(1)
rows=[]
for ci in range(37):
 S=d['S'][ci].astype(np.int64);W=d['W'][ci].astype(np.int64)
 Y=np.matmul(S,W);U=np.matmul(a,Y)
 tau=d['tau'][ci].astype(np.int64);pos=d['positive_gain'][ci].astype(bool);const=d['constant_channels'][ci].astype(bool);cg=d['constant_gate'][ci].astype(bool)
 gold=np.where(const[None,None,:],cg[None,:,:],np.where(pos[None,None,:],U>=tau[None,:,:],U<=tau[None,:,:]))
 assert np.array_equal(Y,d['Y'][ci]) and np.array_equal(U,d['U'][ci]) and np.array_equal(gold,d['gold'][ci])
 response=np.einsum('gkc,gch->gkh',D,W.reshape(6,16,96))
 canonical=np.tile(np.arange(16),(6,1));masks=(D*(1<<np.arange(16))).sum(2)
 for g in range(6):
  for k in range(16):
   for j in range(1,k):
    if np.array_equal(response[g,j],response[g,k]):canonical[g,k]=j;break
 routes={}
 for r,bits in enumerate(S.reshape(320,96)):
  for g in range(6):
   word=int(np.dot(bits[g*16:g*16+16],1<<np.arange(16)));matches=np.flatnonzero(masks[g]==word)
   if len(matches) and word.bit_count()>=2: jobs=[g*32+16+int(canonical[g,int(matches[0])])]
   else:jobs=[g*32+k for k in range(16) if word>>k&1]
   for j in jobs:routes.setdefault(j,set()).add(r)
 updates=0;live_rows=set();coeff_words=0;live_jobs=0
 for j,consumers in routes.items():
  g,k=divmod(j,32);co=W[g*16+k] if k<16 else response[g,k-16]
  coeff_words+=6 if k<16 else 8
  if np.any(co):updates+=len(consumers);live_rows|=consumers;live_jobs+=1
 fullplanes=certplanes=early=0
 for pi in range(32):
  for hg in range(12):
   y=Y[pi,:,hg*8:hg*8+8];e=max(int(abs(v)).bit_length() for v in y.flat);fullplanes+=max(1,e)
   locked=np.broadcast_to(const[hg*8:hg*8+8],(10,8)).copy();th=tau[:,hg*8:hg*8+8];ps=pos[hg*8:hg*8+8]
   for m in range(max(1,e)-1,-1,-1):
    base=np.matmul(a,y>>m)<<m
    lo=base+negative_sum[:,None]*((1<<m)-1);hi=base+positive_sum[:,None]*((1<<m)-1)
    locked|=np.where(ps[None,:],(lo>=th)|(hi<th),(lo>th)|(hi<=th));certplanes+=1
    if locked.all():early+=int(m>0);break
   assert locked.all()
 rows.append({'case':str(d['case_name'][ci]),'real':bool(d['is_real'][ci]),'hblock':int(d['hblock'][ci]),'Y_min':int(Y.min()),'Y_max':int(Y.max()),'U_min':int(U.min()),'U_max':int(U.max()),'expected_updates':updates,'expected_jobs':len(routes),'live_jobs':live_jobs,'expected_coeff_words':coeff_words,'expected_native_mac':len(live_rows)*int(np.count_nonzero(a[:,0])),'expected_full_planes':fullplanes,'expected_cert_planes':certplanes,'expected_early_groups':early})
# All actual A columns have ten nonzero targets; native validity tracks initialized rows, not value nonzero.
assert np.count_nonzero(a)==100
report={'status':'PASS','source':str(src),'cases':rows,'independent_Y_U_gate_values_each':37*30720,'A_nonzeros':100,'config_words_cold':397,'config_words_warm':0,'tau_words':360,'flag_words':9,'D_words':12,'class_words':3,'A_words':13}
(P/'inputs.json').write_text(json.dumps(report,separators=(',',':'))+'\n')
print('PASS independent dense Y/U/gate and mode4/plane trace for',len(rows),'cases')
