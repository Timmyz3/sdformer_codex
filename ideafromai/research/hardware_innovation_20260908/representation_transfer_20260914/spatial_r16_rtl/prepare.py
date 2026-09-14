from pathlib import Path
import json
import numpy as np
H=Path(__file__).resolve().parent
F=H.parent/'spatial_r16_integer'
f=np.load(F/'factors.npz');g=np.load(F/'gold_tiles.npz')
q1=f['q1'].astype(np.int64);q2=f['q2'].astype(np.int64);W=np.einsum('orx,rcy->ocyx',q2,q1)
zlo=np.minimum(q1,0).sum((1,2));zhi=np.maximum(q1,0).sum((1,2))
pbound=(abs(q2)*np.maximum(-zlo,zhi)[None,:,None]).sum((1,2))
assert zlo.min()>=-16384 and zhi.max()<=16383 and pbound.max()<2**31
def hexfile(p,a):
 p.write_text(''.join(f'{int(v)&0xffffffff:08x}\n' for v in np.asarray(a).reshape(-1)))
q1vec=q1.reshape(2,8,288).transpose(0,2,1).reshape(576,8)
q2vec=q2.reshape(12,8,2,8,3).transpose(2,0,3,4,1).reshape(576,8)
(H/'parameters').mkdir(exist_ok=True)
hexfile(H/'parameters/q1.hex',q1vec);hexfile(H/'parameters/q2.hex',q2vec)
hexfile(H/'parameters/consumer.hex',np.stack([f['a_q40'].reshape(12,8),f['b_q20'].reshape(12,8)],axis=1))
def rawlayout(a):
 return a.transpose(1,2,3,0).reshape(12,8,4,10).transpose(0,2,3,1).reshape(480,8)
def native(words,origin):
 src=words.astype(np.int64).copy()
 oy,ox=np.asarray(origin)-1
 for y in range(4):
  for x in range(4):
   if not(0<=oy+y<240 and 0<=ox+x<320):src[:,y,x]=0
 ev=((src[None]>>np.arange(10)[:,None,None,None])&1)
 z=np.zeros((10,16,2,4),np.int64)
 for y in range(2):z[:,:,y,:]=np.einsum('tcyx,rcy->trx',ev[:,:,y:y+3,:],q1)
 p=np.zeros((10,96,2,2),np.int64)
 for x in range(2):p[:,:,:,x]=np.einsum('tryx,orx->toy',z[:,:,:,x:x+3],q2)
 for y in range(2):
  for x in range(2):assert np.array_equal(p[:,:,y,x],np.einsum('tcij,ocij->to',ev[:,:,y:y+3,x:x+3],W))
 return ev,z,p
def profile(ev,z,origin):
 U=A=G=M=V=0
 for s in range(2):
  first=q1[s*8:(s+1)*8]
  for c in range(96):
   for ky in range(3):
    if not np.any(first[:,c,ky]):continue
    G+=1;a=ev[:,c,ky:ky+2,:].reshape(10,8)
    if a.any():A+=1;U+=int(np.logical_or(a[:,0::2],a[:,1::2]).sum())
  zz=z[:,s*8:(s+1)*8]
  rank_live=np.any(zz!=0,axis=(0,2,3))
  for og in range(12):
   live=np.any(q2[og*8:(og+1)*8,s*8:(s+1)*8]!=0,axis=0)&rank_live[:,None]
   V+=int(live.sum())
   for y in range(2):
    for x in range(2):M+=int(((zz[:,:,y,x:x+3]!=0)&live[None]).sum())
 state=[0]*64
 for i,n in {1:80,2:192,3:3072,4:576,5:G,6:A,7:U+A,8:U,9:U,10:576,11:80,12:576,13:960,14:M,15:960,16:480,17:480,18:1}.items():state[i]=n
 oy,ox=np.asarray(origin)-1
 inside=int(sum(0<=oy+y<240 and 0<=ox+x<320 for y in range(4) for x in range(4)))
 return dict(base_cycles=sum(state),source_words=2*96*inside,q1_words=A,q2_words=V,local_gathers=G,q1_issues=U,q2_issues=M,z_vector_reads=U+80,z_scalar_reads=M,z_writes=80+U,psum_reads=960,psum_writes=960,cache_writes=576,base_states=state)
profiles={};records=[]
def emit(name,words,origin,refp=None,refz=None,identity=None,j=None,i24=None):
 ev,z,p=native(words,origin)
 if refp is not None:assert np.array_equal(p,refp)
 if refz is not None:assert np.array_equal(z,refz)
 d=H/'fixtures'/name;d.mkdir(parents=True,exist_ok=True)
 hexfile(d/'source.hex',words);hexfile(d/'origin.hex',np.asarray(origin)-1);hexfile(d/'gold.hex',rawlayout(p))
 packed=np.zeros((2,40,8),np.uint32)
 for stripe in range(2):
  for a in range(40):
   pos=2*(a//10);t=a%10
   lo=z[t,stripe*8:stripe*8+8,pos//4,pos%4]
   hi=z[t,stripe*8:stripe*8+8,(pos+1)//4,(pos+1)%4]
   packed[stripe,a]=(lo&32767)|((hi&32767)<<15)
 hexfile(d/'z.hex',packed)
 if identity is not None:
  hexfile(d/'identity.hex',rawlayout(identity));hexfile(d/'j.hex',rawlayout(j));hexfile(d/'i24.hex',rawlayout(i24))
 profiles[str(d)]=profile(ev,z,origin);records.append(str(d))
for i,t in enumerate(g['tile_ids']):
 emit(f'tile_{int(t)}',g['source_words'][i],g['output_origin_yx'][i],g['p_int'][i],g['z_halo_int'][i],g['identity_fp32_bits'][i],g['J_q20'][i],g['i24'][i])
rng=np.random.default_rng(9471)
cases={'zero':np.zeros((96,4,4),np.uint16),'one':np.full((96,4,4),1023,np.uint16),'random':rng.integers(0,1024,(96,4,4),np.uint16)}
tail=np.zeros((96,4,4),np.uint16);tail[95,3,3]=512;cases['tail']=tail
for name,r,sign in [('rank0_positive',0,1),('rank_negative',int(np.argmin(zlo)),-1)]:
 a=np.zeros((96,4,4),np.uint16)
 for ky in range(3):a[:,ky,:]=np.where(q1[r,:,ky]*sign>0,1023,0)[:,None]
 cases[name]=a
for name,a in cases.items():emit(name,a,[100,100])
poison=g['source_words'][0].copy();poison[:,0,:]=1023;poison[:,:,0]=1023
emit('padding_poison',poison,[0,0],g['p_int'][0],g['z_halo_int'][0])
original=[int(t) for t in g['tile_ids'][:8]]
sets={'small':[str(H/'fixtures'/f'tile_{t}') for t in original]+[str(H/'fixtures'/n) for n in cases]+[str(H/'fixtures/padding_poison')],
 'held':[str(H/'fixtures'/f'tile_{t}') for t in range(128,192)],
 'disjoint':[str(H/'fixtures'/f'tile_{t}') for t in range(4000,4064)]}
for n,rows in sets.items():(H/f'{n}.txt').write_text('\n'.join(rows)+'\n')
(H/'profiles.json').write_text(json.dumps(profiles,separators=(',',':'))+'\n')
(H/'admission.json').write_text(json.dumps(dict(passed=True,q1_range=[int(q1.min()),int(q1.max())],q2_range=[int(q2.min()),int(q2.max())],z_lower=zlo.tolist(),z_upper=zhi.tolist(),p_any_prefix_abs=pbound.tolist(),factor_and_expanded_gold_fixtures=len(records),sets={k:len(v) for k,v in sets.items()}),indent=2)+'\n')
print(json.dumps(dict(fixtures=len(records),sets={k:len(v) for k,v in sets.items()},z_bound=[int(zlo.min()),int(zhi.max())],p_prefix=int(pbound.max()))))
