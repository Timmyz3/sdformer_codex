"""Recompute flat-R8 outputs on the spatial captures, with unchanged R8 parameters."""
from pathlib import Path
import json
import numpy as np

H=Path(__file__).resolve().parent
B=H.parents[1]
D=B/'r8_consumer_fusion_20260914/data'
OLD=B/'representation_transfer_20260914/bitmap_rr'
SP=B/'representation_transfer_20260914'
f=np.load(D/'factors.npz');co=np.load(D/'consumer_coefficients.npz')
q1=f['q1'].astype(np.int64).reshape(8,96,3,3)
q2=f['q2'].astype(np.int64).reshape(96,8)
a=co['a_q40'].astype(np.int64);b=co['b_q20'].astype(np.int64)
W=np.einsum('or,rcij->ocij',q2,q1)
lo=np.minimum(q1,0).sum((1,2,3));hi=np.maximum(q1,0).sum((1,2,3))
assert lo.min()>=-4096 and hi.max()<4096
assert (abs(q2)*np.maximum(-lo,hi)[None]).sum(1).max()<2**31
param=H/'parameters';param.mkdir(parents=True,exist_ok=True)
oldparam=OLD/'fixtures/real_0'
assert np.array_equal(np.fromfile(oldparam/'param4.bin','<i4').reshape(864,8),q1.reshape(8,864).T)
assert np.array_equal(np.fromfile(oldparam/'param5.bin','<i4').reshape(12,8,8).transpose(0,2,1).reshape(96,8),q2)
ab=np.fromfile(oldparam/'param7.bin','<i4').reshape(12,2,8)
assert np.array_equal(ab[:,0].reshape(96),a) and np.array_equal(ab[:,1].reshape(96),b)
for kind in [4,5,6,7,8,9,10,11]:
 p=param/f'param{kind}.bin'
 if not p.exists():p.symlink_to(oldparam/p.name)

def ordered(x):
 return x.reshape(10,12,8,4).transpose(1,3,0,2).reshape(480,8)

def compute(words,output_origin,identity):
 words=words.copy();oy,ox=map(int,output_origin)
 for y in range(4):
  for x in range(4):
   if not(0<=oy-1+y<240 and 0<=ox-1+x<320):words[:,y,x]=0
 ev=(words[None]>>np.arange(10)[:,None,None,None])&1
 z=np.zeros((10,8,2,2),np.int64)
 raw=np.zeros((10,96,2,2),np.int64)
 for y in range(2):
  for x in range(2):
   patch=ev[:,:,y:y+3,x:x+3]
   z[:,:,y,x]=np.einsum('tcij,rcij->tr',patch,q1)
   raw[:,:,y,x]=np.einsum('tr,or->to',z[:,:,y,x],q2)
   assert np.array_equal(raw[:,:,y,x],np.einsum('tcij,ocij->to',patch,W))
 j=np.clip(np.rint(identity.astype(np.float64)*2**20),-2**31,2**31-1).astype(np.int64)
 wide=raw*a[None,:,None,None]+((j+b[None,:,None,None])<<20)
 out=wide>>26;rem=wide-(out<<26)
 out+=(rem>2**25)|((rem==2**25)&((out&1)!=0))
 i24=np.clip(out,-2**23,2**23-1)
 return dict(raw=raw,z=z,j=j,wide=wide,i24=i24)

records={};fullmatches=0
def emit(name,words,origin,identity,reference=None,meta=None):
 global fullmatches
 vals=compute(words,origin,identity)
 if reference is not None:
  for key,ref in reference.items():assert np.array_equal(vals[key],ref),(name,key)
  fullmatches+=1
 d=H/'fixtures'/name;d.mkdir(parents=True,exist_ok=True)
 words.astype('<u2').tofile(d/'source.bin')
 (np.asarray(origin,dtype='<i4')-1).tofile(d/'origin.bin')
 ordered(identity).astype('<f4').tofile(d/'identity_fp32.bin')
 for key,file,dtype in [('raw','raw','<i4'),('j','identity','<i4'),('wide','wide','<i8'),('i24','gold','<i4')]:
  ordered(vals[key]).astype(dtype).tofile(d/f'{file}.bin')
 vals['z'].astype('<i4').tofile(d/'latent.bin')
 records[name]=dict(name=name,path=str(d),output_origin_yx=list(map(int,origin)),input_origin_yx=(np.asarray(origin)-1).tolist(),**(meta or {}))
 return vals

src=np.load(D/'first_source_words.npy',mmap_mode='r')
fp=np.load(D/'identity_fp32_full.npy',mmap_mode='r')
gold={k:np.load(D/file,mmap_mode='r') for k,file in [('raw','raw_p_full.npy'),('j','identity_q20_full.npy'),('i24','i24_new_full.npy')]}
oldsmall=np.load(D/'consumer_integer_first8.npz')
ids=list(dict.fromkeys([int(y//2*160+x//2) for y,x in oldsmall['output_origin_yx']]+list(range(128,192))+list(range(4000,4064))+[9664]))
for tile in ids:
 origin=np.array([2*(tile//160),2*(tile%160)],np.int32)
 words=np.zeros((96,4,4),np.uint16)
 for y in range(4):
  for x in range(4):
   yy,xx=origin[0]-1+y,origin[1]-1+x
   if 0<=yy<240 and 0<=xx<320:words[:,y,x]=src[:,yy,xx]
 emit(f'tile_{tile}',words,origin,np.asarray(fp[tile]),{k:ar[tile] for k,ar in gold.items()},dict(tile_id=tile,frame_index=0,file='zurich_city_09_a_0001.npy'))

seq=np.load(SP/'quality/q11/sequence_tiles.npz')
seqmeta=json.loads((SP/'quality/q11/sequence_tiles.json').read_text())
sequence=[];prefix_matches=[]
for i,meta in enumerate(seqmeta):
 name=f'seq{i//2:02d}_tile{meta["tile_id"]}';sequence.append(name)
 vals=emit(name,seq['source_words'][i],seq['output_origin_yx'][i],seq['identity_fp32_bits'][i].copy().view(np.float32),meta=meta)
 if meta['frame_index']==0:
  old=Path(records[f'tile_{meta["tile_id"]}']['path'])
  for file in ['source.bin','identity_fp32.bin','raw.bin','identity.bin','wide.bin','gold.bin']:
   assert (old/file).read_bytes()==(Path(records[name]['path'])/file).read_bytes(),(name,file)
  prefix_matches.append(meta['tile_id'])
 # Only J depends on the unchanged identity; phase3/spatial raw/I24 are not R8 gold.
 assert np.array_equal(vals['j'],seq['J_q20'][i])

rng=np.random.default_rng(61519)
zero_id=np.zeros((10,96,2,2),np.float32)
for name,words in [('zero',np.zeros((96,4,4),np.uint16)),('one',np.full((96,4,4),1023,np.uint16)),('random',rng.integers(0,1024,(96,4,4),dtype=np.uint16))]:
 emit(name,words,[100,100],zero_id)
poison=np.fromfile(Path(records['tile_0']['path'])/'source.bin','<u2').reshape(96,4,4).copy()
poison[:,0,:]=1023;poison[:,:,0]=1023
emit('padding_poison',poison,[0,0],np.asarray(fp[0]),{k:ar[0] for k,ar in gold.items()})
small=[f'tile_{int(y//2*160+x//2)}' for y,x in oldsmall['output_origin_yx']]+['zero','one','random','padding_poison']
sets=dict(small=small,held=[f'tile_{i}' for i in range(128,192)],disjoint=[f'tile_{i}' for i in range(4000,4064)],sequences=sequence,
          swap=[sequence[0],sequence[3],sequence[2],sequence[-1]])
for name,names in sets.items():
 (H/f'{name}.txt').write_text('\n'.join(records[n]['path'] for n in names)+'\n')
(H/'fixtures.json').write_text(json.dumps(records,separators=(',',':'))+'\n')
report=dict(passed=True,target='sttmultires_unet.encoders.swin3d.patch_embed.residual_encoding.resblocks.0.conv2.0',
 input_contract='same native C96x4x4 low10 gate words; output origin=(2*(tile//160),2*(tile%160)), physical source origin=output-1; original FP32 residual block input',
 prefix_evidence='both evaluators import identical data/model_access.py load_parent and replace only same conv2 and r0 post; first-frame captured source/id checked below',
 source_and_identity_matches_old_firstframe_tiles=prefix_matches,
 existing_R8_fullgold_matches=fullmatches,fixtures=len(records),new_sequence_tiles=36,
 new_sequence_R8_raw_J_wide_I24_each=36*3840,R8_gold_from_phase3=False,
 q1_range=[int(q1.min()),int(q1.max())],q2_range=[int(q2.min()),int(q2.max())],
 latent_bounds=[int(lo.min()),int(hi.max())],sets={k:len(v) for k,v in sets.items()},
 quality='fixed deployed flatR8 I24, own valid825 AEE1.3276350226079938; no new GPU evaluation')
(H/'admission.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
