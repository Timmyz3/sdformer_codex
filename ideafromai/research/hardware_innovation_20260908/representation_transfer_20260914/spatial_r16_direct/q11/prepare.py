from pathlib import Path
import json
import numpy as np
H=Path(__file__).resolve().parent;D=H.parent;A=D.parent/'spatial_r16_rtl';F=D.parent/'spatial_winograd_inputs'
f=np.load(F/'factors.npz');g=np.load(F/'gold_tiles.npz');w=f['expanded_int32'].astype(np.int64)
assert np.array_equal(w,np.einsum('orx,rcy->ocyx',f['q2'].astype(np.int64),f['q1'].astype(np.int64)))
lo=np.minimum(w,0).sum((1,2,3));hi=np.maximum(w,0).sum((1,2,3));assert lo.min()>=-2**31 and hi.max()<2**31
assert np.all(np.any(w.reshape(12,8,864)!=0,axis=1))
def hexfile(path,a):path.write_text(''.join(f'{int(v)&0xffffffff:08x}\n' for v in np.asarray(a).reshape(-1)))
def readhex(path):return np.array([int(v,16) for v in path.read_text().split()],np.uint32)
def order(a):return a.transpose(1,2,3,0).reshape(12,8,4,10).transpose(0,2,3,1).reshape(480,8)
(H/'parameters').mkdir(exist_ok=True)
hexfile(H/'parameters/weight.hex',w.reshape(12,8,864).transpose(0,2,1))
hexfile(H/'parameters/consumer.hex',np.stack([f['a_q40'].reshape(12,8),f['b_q20'].reshape(12,8)],axis=1))
real={f'tile_{int(t)}':i for i,t in enumerate(g['tile_ids'])};profiles={};base=json.loads((D/'os_profiles.json').read_text())
for name in ['small','held','disjoint']:
 lines=[]
 for original in (A/f'{name}.txt').read_text().splitlines():
  srcdir=Path(original);dest=H/'fixtures'/srcdir.name;lines.append(str(dest))
  if str(dest) in profiles:continue
  dest.mkdir(parents=True,exist_ok=True)
  for field in ['source','origin','identity']:
   path=dest/f'{field}.hex'
   if not path.exists():path.symlink_to(srcdir/f'{field}.hex')
  source=readhex(dest/'source.hex').reshape(96,4,4)&1023;origin=readhex(dest/'origin.hex').view(np.int32).astype(np.int64)
  for y in range(4):
   for x in range(4):
    if not(0<=origin[0]+y<240 and 0<=origin[1]+x<320):source[:,y,x]=0
  ev=((source[None]>>np.arange(10)[:,None,None,None])&1).astype(np.int64)
  p=np.zeros((10,96,2,2),np.int64)
  for y in range(2):
   for x in range(2):p[:,:,y,x]=np.einsum('tcij,ocij->to',ev[:,:,y:y+3,x:x+3],w)
  po=order(p);identity=readhex(dest/'identity.hex').view(np.float32).reshape(480,8)
  assert np.isfinite(identity).all()
  j=np.clip(np.rint(identity.astype(np.float64)*(1<<20)),-2**31,2**31-1).astype(np.int64)
  av=np.repeat(f['a_q40'].reshape(12,1,8),40,axis=1).reshape(480,8).astype(np.int64)
  bv=np.repeat(f['b_q20'].reshape(12,1,8),40,axis=1).reshape(480,8).astype(np.int64)
  wide=po*av+((j+bv)<<20);q=wide//(1<<26);r=wide-q*(1<<26)
  i24=np.clip(q+((r>(1<<25))|((r==(1<<25))&((q&1)!=0))),-2**23,2**23-1)
  if srcdir.name in real:
   i=real[srcdir.name]
   for actual,field in [(po,'p_int'),(j,'J_q20'),(wide,'wide_int64'),(i24,'i24')]:assert np.array_equal(actual,order(g[field][i])),(srcdir.name,field)
   assert np.array_equal(readhex(dest/'identity.hex').reshape(480,8),order(g['identity_fp32_bits'][i]))
  for field,value in [('gold',po),('j',j),('i24',i24)]:hexfile(dest/f'{field}.hex',value)
  hexfile(dest/'wide.hex',wide.astype('<i8').view('<u4').reshape(480,16))
  profiles[str(dest)]=base[original]
 (H/f'{name}.txt').write_text('\n'.join(lines)+'\n')
(H/'profiles.json').write_text(json.dumps(profiles,separators=(',',':'))+'\n')
(H/'admission.json').write_text(json.dumps(dict(passed=True,q2_bits=int(f['q2_bits']),weight_range=[int(w.min()),int(w.max())],raw_prefix_abs=int(np.maximum(-lo,hi).max()),fixtures=len(profiles),real_gold_tiles=len(real),source='actual existing native source and original FP32 identity',same_function='Q11 spatial_winograd_inputs expanded-W32',quantization_changed=False),separators=(',',':'))+'\n')
print(json.dumps(dict(passed=True,fixtures=len(profiles),real_gold_tiles=len(real),raw_prefix_abs=int(np.maximum(-lo,hi).max()))))
