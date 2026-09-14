from pathlib import Path
import json,re
import numpy as np
H=Path(__file__).resolve().parent;N=H.parent;F=N/'spatial_winograd_pruning/moment';A=N/'spatial_r16_rtl';Q=N/'quality/q11'
f=np.load(F/'factors.npz');gold=np.load(F/'gold_tiles.npz');q1=f['q1'].astype(np.int64);q2=f['q2'].astype(np.int64)
assert np.all(q2[:,:,1]==q2[:,:,0]+q2[:,:,2])
W=np.einsum('orx,rcy->ocyx',q2,q1);lo=np.minimum(q1,0).sum((1,2));hi=np.maximum(q1,0).sum((1,2))
q2two=q2[:,:,[0,2]];elo=2*lo;ehi=2*hi;prefix=(abs(q2two)*np.maximum(-elo,ehi)[None,:,None]).sum((1,2))
assert elo.min()>=-32768 and ehi.max()<=32767 and prefix.max()<2**31
names=re.search(r'typedef enum logic \[5:0\] \{([^}]+)\}',(H/'spatial_core.sv').read_text()).group(1).replace('\n','').split(',');stateid={k.strip():i for i,k in enumerate(names)}
def hexfile(path,a):path.write_text(''.join(f'{int(v)&0xffffffff:08x}\n' for v in np.asarray(a).reshape(-1)))
def readhex(path):return np.array([int(v,16) for v in path.read_text().split()],np.uint32)
def order(a):return a.transpose(1,2,3,0).reshape(12,8,4,10).transpose(0,2,3,1).reshape(480,8)
(H/'parameters').mkdir(exist_ok=True)
hexfile(H/'parameters/q1.hex',q1.reshape(2,8,288).transpose(0,2,1))
hexfile(H/'parameters/q2.hex',q2two.reshape(12,8,2,8,2).transpose(2,0,3,4,1))
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
 e=z[:,:,:,:3]+z[:,:,:,1:]
 count=ev[:,:,:,:3]+ev[:,:,:,1:]
 edirect=np.stack([np.einsum('tcyx,rcy->trx',count[:,:,y:y+3,:],q1) for y in range(2)],axis=2)
 assert np.array_equal(e,edirect)
 epad=np.pad(e,((0,0),(0,0),(0,0),(0,1)))
 p=np.stack([np.einsum('trpx,orx->top',e[:,:,:,x:x+2],q2two) for x in range(2)],axis=3)
 native=np.stack([np.einsum('trpx,orx->top',z[:,:,:,x:x+3],q2) for x in range(2)],axis=3)
 assert np.array_equal(p,native)
 for y in range(2):
  for x in range(2):assert np.array_equal(p[:,:,y,x],np.einsum('tcij,ocij->to',ev[:,:,y:y+3,x:x+3],W))
 ident=np.asarray(identity,np.uint32).view(np.float32);assert np.isfinite(ident).all()
 j=np.clip(np.rint(ident.astype(np.float64)*(1<<20)),-2**31,2**31-1).astype(np.int64)
 wide=p*f['a_q40'][None,:,None,None].astype(np.int64)+((j+f['b_q20'][None,:,None,None].astype(np.int64))<<20)
 qr=wide//(1<<26);rem=wide-qr*(1<<26);i24=np.clip(qr+((rem>(1<<25))|((rem==(1<<25))&((qr&1)!=0))),-2**23,2**23-1)
 if ref is not None:
  for key,value in [('z_halo_int',z),('p_int',p),('J_q20',j),('wide_int64',wide),('i24',i24)]:assert np.array_equal(value,ref[key]),(name,key)
 hexfile(d/'source.hex',words);hexfile(d/'origin.hex',np.asarray(origin)-1)
 for key,value in [('gold',p),('identity',identity),('j',j),('i24',i24)]:hexfile(d/f'{key}.hex',order(value))
 hexfile(d/'wide.hex',order(wide).astype('<i8').view('<u4').reshape(480,16))
 zp=np.zeros((2,40,8),np.uint32)
 for ss in range(2):
  for row in range(40):
   y,x=divmod(row//10,2);t=row%10;rr=slice(ss*8,ss*8+8)
   zp[ss,row]=(epad[t,rr,y,2*x]&65535)|((epad[t,rr,y,2*x+1]&65535)<<16)
 hexfile(d/'z.hex',zp)
 # z.hex and legacy z_* ports now contain E16, including zero padding column.
 U=Aq=G=M=V=C2=CNZ=0
 for ss in range(2):
  first=q1[ss*8:ss*8+8];ee=epad[:,ss*8:ss*8+8];rl=np.any(ee!=0,axis=(0,2,3))
  for c in range(96):
   for ky in range(3):
    if not np.any(first[:,c,ky]):continue
    G+=1;a=ev[:,c,ky:ky+2,:]
    cm=np.pad(a[:,:,:3]+a[:,:,1:],((0,0),(0,0),(0,1))).reshape(10,8)
    if cm.any():
     Aq+=1;U+=int(np.logical_or(cm[:,0::2]>0,cm[:,1::2]>0).sum())
     C2+=int((cm==2).sum());CNZ+=int((cm!=0).sum())
  for og in range(12):
   live=np.any(q2two[og*8:og*8+8,ss*8:ss*8+8]!=0,axis=0)&rl[:,None]
   V+=int(live.sum())
   for y in range(2):
    for x in range(2):M+=int(((ee[:,:,y,x:x+2]!=0)&live[None,:,:]).sum())
 counts=dict(ZCLEAR=80,L_START=192,L_LOAD=3072,L_GATHER=576,CHECK=G,QREAD=Aq,TIMESEL=U+Aq,ZREAD=U,ZADD=U,KNEXT=576,ZSCAN=80,VLOAD=384,POSLOAD=960,BASE_MAC=M,STORE=960,DRAIN_READ=480,DRAIN_SEND=480,FINISH=1)
 states=[0]*64
 for key,value in counts.items():states[stateid[key]]=value
 profiles[str(d)]=dict(base_cycles=sum(states),source_words=2*96*inside,q1_words=Aq,q2_words=V,local_gathers=G,q1_issues=U,q2_issues=M,z_vector_reads=U+80,z_scalar_reads=M,z_writes=U+80,psum_reads=960,psum_writes=960,cache_writes=384,
  count_constructs=G,count2_fields=C2,count_nonzero_fields=CNZ,base_states=states)
 # Compare all actual output and original identity values to independently frozen moment3 fixtures.
 old=N/'spatial_moment3/fixtures'/name
 if old.exists():
  for fn in ['source.hex','origin.hex','gold.hex','identity.hex','j.hex','wide.hex','i24.hex']:
   assert (d/fn).read_bytes()==(old/fn).read_bytes(),(name,fn)
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
seq=np.load(Q/'sequence_tiles.npz');meta=json.loads((Q/'sequence_tiles.json').read_text());sets['sequences']=[]
assert np.array_equal(q1,np.load(N/'spatial_winograd_inputs/factors.npz')['q1'])
for i,mdata in enumerate(meta):
 emit(f'sequence_{i:02d}',seq['source_words'][i],seq['output_origin_yx'][i],seq['identity_fp32_bits'][i]);sets['sequences'].append(records[f'sequence_{i:02d}'])
for name,rows in sets.items():(H/f'{name}.txt').write_text('\n'.join(rows)+'\n')
(H/'profiles.json').write_text(json.dumps(profiles,separators=(',',':'))+'\n')
(H/'admission.json').write_text(json.dumps(dict(passed=True,function='same frozen moment via source box2 and ordinary two-tap Q2',all_scalar_constraints=1536,q2_range=[int(q2.min()),int(q2.max())],E_lower=int(elo.min()),E_upper=int(ehi.max()),Q2_prefix_abs=int(prefix.max()),fixtures=len(profiles),raw_values=total,sets={k:len(v) for k,v in sets.items()},state_ids=stateid,sequence_scope='36 original Q11 upstream source/FP32 identity prefixes, new moment CPU gold; not new 36-frame moment network gold',sequence_metadata=meta),separators=(',',':'))+'\n')
print(json.dumps(dict(passed=True,fixtures=len(profiles),raw_values=total,sets={k:len(v) for k,v in sets.items()})))
