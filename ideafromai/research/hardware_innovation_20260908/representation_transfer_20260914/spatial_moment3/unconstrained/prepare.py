from pathlib import Path
import json,re
import numpy as np
H=Path(__file__).resolve().parent;N=H.parents[1];F=N/'spatial_winograd_pruning/unconstrained';A=N/'spatial_r16_rtl';Q=N/'quality/q11'
f=np.load(F/'factors.npz');gold=np.load(F/'gold_tiles.npz');q1=f['q1'].astype(np.int64);q2=f['physical_coeff3'].astype(np.int64)
assert bool(f['raw_p_is_unhalved']) and np.max(abs(q2))<4096
phase=np.stack([np.stack([q2[:,:,0],q2[:,:,1],q2[:,:,1]-q2[:,:,0],np.zeros_like(q2[:,:,0])],axis=2),np.stack([np.zeros_like(q2[:,:,0]),q2[:,:,1]-q2[:,:,2],q2[:,:,1],q2[:,:,2]],axis=2)])
W=np.einsum('aorx,rcy->aocyx',phase,q1);assert np.array_equal(W,f['expanded_phase_int32']);lo=np.minimum(q1,0).sum((1,2));hi=np.maximum(q1,0).sum((1,2))
da=np.stack([hi-lo,2*np.maximum(-lo,hi),hi-lo],axis=1);mb=(abs(q2)*da[None]).sum(1)
assert da.max()<32768 and mb.max()<2**31 and max((mb[:,0]+mb[:,1]).max(),(mb[:,1]+mb[:,2]).max())<2**31
names=re.search(r'typedef enum logic \[5:0\] \{([^}]+)\}',(H.parent/'spatial_core.sv').read_text()).group(1).replace('\n','').split(',');stateid={k.strip():i for i,k in enumerate(names)}
def hexfile(path,a):path.write_text(''.join(f'{int(v)&0xffffffff:08x}\n' for v in np.asarray(a).reshape(-1)))
def readhex(path):return np.array([int(v,16) for v in path.read_text().split()],np.uint32)
def order(a):return a.transpose(1,2,3,0).reshape(12,8,4,10).transpose(0,2,3,1).reshape(480,8)
(H/'parameters').mkdir(exist_ok=True)
hexfile(H/'parameters/q1.hex',q1.reshape(2,8,288).transpose(0,2,1))
hexfile(H/'parameters/q2.hex',q2.reshape(12,8,2,8,3).transpose(2,0,3,4,1))
hexfile(H/'parameters/consumer.hex',np.stack([f['a_q40'].reshape(12,8),f['b_q20'].reshape(12,8)],axis=1))
profiles={};records={};total=0
def emit(name,words,origin,identity,ref=None):
 global total
 d=H/'fixtures'/name;d.mkdir(parents=True,exist_ok=True);src=words.astype(np.uint16).copy();sy,sx=np.asarray(origin)-1;inside=0
 for y in range(4):
  for x in range(4):
   if 0<=sy+y<240 and 0<=sx+x<320:inside+=1
   else:src[:,y,x]=0
 ev=((src[None]>>np.arange(10)[:,None,None,None])&1).astype(np.int64)
 z=np.stack([np.einsum('tcyx,rcy->trx',ev[:,:,y:y+3,:],q1) for y in range(2)],axis=2)
 dv=np.stack([z[:,:,:,0]-z[:,:,:,2],z[:,:,:,1]+z[:,:,:,2],z[:,:,:,1]-z[:,:,:,3]],axis=3)
 m=np.einsum('tryx,orx->toyx',dv,q2);p=np.stack([m[:,:,:,0]+m[:,:,:,1],m[:,:,:,1]-m[:,:,:,2]],axis=3)
 for y in range(2):
  for x in range(2):assert np.array_equal(p[:,:,y,x],np.einsum('tcij,ocij->to',ev[:,:,y:y+3,:],W[x]))
 ident=np.asarray(identity,np.uint32).view(np.float32);assert np.isfinite(ident).all()
 j=np.clip(np.rint(ident.astype(np.float64)*(1<<20)),-2**31,2**31-1).astype(np.int64)
 wide=p*f['a_q40'][None,:,None,None].astype(np.int64)+((j+f['b_q20'][None,:,None,None].astype(np.int64))<<20)
 qr=wide//(1<<26);rem=wide-qr*(1<<26);i24=np.clip(qr+((rem>(1<<25))|((rem==(1<<25))&((qr&1)!=0))),-2**23,2**23-1)
 if ref is not None:
  for key,value in [('z_halo_int',z),('p_int',p),('J_q20',j),('wide_int64',wide),('i24',i24)]:assert np.array_equal(value,ref[key]),(name,key)
 hexfile(d/'source.hex',words);hexfile(d/'origin.hex',np.asarray(origin)-1)
 for key,value in [('gold',p),('identity',identity),('j',j),('i24',i24)]:hexfile(d/f'{key}.hex',order(value))
 hexfile(d/'wide.hex',order(wide).astype('<i8').view('<u4').reshape(480,16))
 zp=np.zeros((2,40,8),np.uint32);dp=np.zeros_like(zp)
 for ss in range(2):
  for row in range(40):
   y,x=divmod(row//10,2);t=row%10;rr=slice(ss*8,ss*8+8)
   zp[ss,row]=(z[t,rr,y,2*x]&32767)|((z[t,rr,y,2*x+1]&32767)<<15)
   dp[ss,row]=((dv[t,rr,y,0]&65535)|((dv[t,rr,y,1]&65535)<<16)) if x==0 else (dv[t,rr,y,2]&65535)
 hexfile(d/'z.hex',zp);hexfile(d/'d.hex',dp)
 U=Aq=G=M=V=0
 for ss in range(2):
  first=q1[ss*8:ss*8+8];dd=dv[:,ss*8:ss*8+8];rl=np.any(dd!=0,axis=(0,2,3))
  for c in range(96):
   for ky in range(3):
    if not np.any(first[:,c,ky]):continue
    G+=1;a=ev[:,c,ky:ky+2,:].reshape(10,8)
    if a.any():Aq+=1;U+=int(np.logical_or(a[:,0::2],a[:,1::2]).sum())
  for og in range(12):
   live=np.any(q2[og*8:og*8+8,ss*8:ss*8+8]!=0,axis=0)&rl[:,None]
   V+=int(live.sum());M+=int(((dd!=0)&live[None,:,None,:]).sum())
 counts=dict(ZCLEAR=80,L_START=192,L_LOAD=3072,L_GATHER=576,CHECK=G,QREAD=Aq,TIMESEL=U+Aq,ZREAD=U,ZADD=U,KNEXT=576,ZSCAN=80,VLOAD=576,POSLOAD=480,BASE_MAC=M,STORE=0,DRAIN_READ=480,DRAIN_SEND=480,FINISH=1,
  XDREAD0=40,XDREAD1=40,XD0=40,XD1=40,XD2=40,W_INV0A=480,W_PREAD0=240,W_PADD0=240,W_STORE0=480,W_INV1A=480,W_PREAD1=240,W_PADD1=240,W_STORE1=480)
 states=[0]*64
 for key,value in counts.items():states[stateid[key]]=value
 profiles[str(d)]=dict(base_cycles=sum(states),source_words=2*96*inside,q1_words=Aq,q2_words=V,local_gathers=G,q1_issues=U,q2_issues=M,z_vector_reads=U+160,z_scalar_reads=M,z_writes=U+160,psum_reads=960,psum_writes=960,cache_writes=576,
  transform_issues=120,transform_reads=80,transform_writes=80,reconstruction_issues=960,stripe_add_issues=480,cache_reads=M,exact_halves=0,base_states=states)
 records[name]=str(d);total+=p.size
for i,t in enumerate(gold['tile_ids']):emit(f'tile_{int(t)}',gold['source_words'][i],gold['output_origin_yx'][i],gold['identity_fp32_bits'][i],{k:gold[k][i] for k in ['z_halo_int','p_int','J_q20','wide_int64','i24']})
for original in (A/'small.txt').read_text().splitlines():
 old=Path(original)
 if old.name in records:continue
 words=readhex(old/'source.hex').reshape(96,4,4).astype(np.uint16);origin=readhex(old/'origin.hex').view(np.int32).astype(np.int64)+1
 # Inverse of og,P,T,lane external order into T,O,Y,X.
 ident=readhex(old/'identity.hex').reshape(12,4,10,8).transpose(0,3,1,2).reshape(96,2,2,10).transpose(3,0,1,2)
 emit(old.name,words,origin,ident)
sets={name:[records[Path(x).name] for x in (A/f'{name}.txt').read_text().splitlines()] for name in ['small','held','disjoint']}
seq=np.load(F/'gold_sequences.npz');meta=json.loads((Q/'sequence_tiles.json').read_text());sets['sequences']=[]
assert np.array_equal(q1,np.load(N/'spatial_winograd_inputs/factors.npz')['q1'])
for i,mdata in enumerate(meta):
 emit(f'sequence_{i:02d}',seq['source_words'][i],seq['output_origin_yx'][i],seq['identity_fp32_bits'][i],{key:seq[key][i] for key in ['z_halo_int','p_int','J_q20','wide_int64','i24']});sets['sequences'].append(records[f'sequence_{i:02d}'])
for name,rows in sets.items():(H/f'{name}.txt').write_text('\n'.join(rows)+'\n')
(H/'profiles.json').write_text(json.dumps(profiles,separators=(',',':'))+'\n')
(H/'admission.json').write_text(json.dumps(dict(passed=True,function='unconstrained U2=0 physical3 planes; unhalved p2 and newly rounded half-scale consumer',all_scalar_constraints=0,physical_coeff3_range=[int(q2.min()),int(q2.max())],D_abs=int(da.max()),M_prefix_abs=int(mb.max()),reconstruction_abs=int(max((mb[:,0]+mb[:,1]).max(),(mb[:,1]+mb[:,2]).max())),fixtures=len(profiles),raw_values=total,sets={k:len(v) for k,v in sets.items()},state_ids=stateid,sequence_scope='36 original Q11 upstream source/FP32 identity prefixes, new unconstrained phase3 CPU gold; not new 36-frame control network gold',sequence_metadata=meta),separators=(',',':'))+'\n')
print(json.dumps(dict(passed=True,fixtures=len(profiles),raw_values=total,sets={k:len(v) for k,v in sets.items()})))
