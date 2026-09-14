from pathlib import Path
import json
import numpy as np
H=Path(__file__).resolve().parent;N=H.parents[1]
F=N/'spatial_winograd_inputs';Q=N/'quality/q11'
f=np.load(F/'factors.npz');g=np.load(Q/'sequence_tiles.npz');meta=json.loads((Q/'sequence_tiles.json').read_text())
assert len(meta)==36 and len({x['file'] for x in meta})==18
w=f['expanded_int32'].astype(np.int64);q1=f['q1'].astype(np.int64);q2=f['q2'].astype(np.int64)
assert np.all(np.any(w.reshape(12,8,864)!=0,axis=1))
profiles=json.loads((H/'profiles.json').read_text());records=[]
def hexfile(path,a):path.write_text(''.join(f'{int(v)&0xffffffff:08x}\n' for v in np.asarray(a).reshape(-1)))
def order(a):return a.transpose(1,2,3,0).reshape(12,8,4,10).transpose(0,2,3,1).reshape(480,8)
for i,m in enumerate(meta):
 d=H/'fixtures'/f'sequence_{i:02d}';d.mkdir(parents=True,exist_ok=True)
 source=g['source_words'][i].astype(np.uint16);origin=g['output_origin_yx'][i].astype(np.int64)-1
 actual=source.copy();inside=0
 for y in range(4):
  for x in range(4):
   if 0<=origin[0]+y<240 and 0<=origin[1]+x<320:inside+=1
   else:actual[:,y,x]=0
 ev=((actual[None]>>np.arange(10)[:,None,None,None])&1).astype(np.int64)
 z=np.stack([np.einsum('tcyx,rcy->trx',ev[:,:,y:y+3,:],q1) for y in range(2)],axis=2)
 p=np.stack([np.einsum('trpx,orx->top',z[:,:,:,x:x+3],q2) for x in range(2)],axis=3)
 for y in range(2):
  for x in range(2):assert np.array_equal(p[:,:,y,x],np.einsum('tcij,ocij->to',ev[:,:,y:y+3,x:x+3],w))
 ident=g['identity_fp32_bits'][i].copy().view(np.float32);assert np.isfinite(ident).all()
 j=np.clip(np.rint(ident.astype(np.float64)*(1<<20)),-2**31,2**31-1).astype(np.int64)
 wide=p*f['a_q40'][None,:,None,None].astype(np.int64)+((j+f['b_q20'][None,:,None,None].astype(np.int64))<<20)
 q=wide//(1<<26);rem=wide-q*(1<<26);i24=np.clip(q+((rem>(1<<25))|((rem==(1<<25))&((q&1)!=0))),-2**23,2**23-1)
 for field,value in [('z_halo_int',z),('p_int',p),('J_q20',j),('wide_int64',wide),('i24',i24)]:assert np.array_equal(value,g[field][i]),(i,field)
 hexfile(d/'source.hex',source);hexfile(d/'origin.hex',origin)
 for field,value in [('gold',p),('identity',g['identity_fp32_bits'][i]),('j',j),('i24',i24)]:hexfile(d/f'{field}.hex',order(value))
 hexfile(d/'wide.hex',order(wide).astype('<i8').view('<u4').reshape(480,16))
 bits=np.stack([ev[:,:,y:y+3,x:x+3].reshape(10,864) for y in range(2) for x in range(2)])
 updates=int(bits.sum())*12;reads=int(np.any(bits.reshape(4,10,27,32),axis=3).sum())*12;state=[0]*64
 for k,v in {1:384,2:6144,3:3456,4:540,5:480,6:reads,7:updates,8:480,9:4,10:480,11:480,12:1}.items():state[k]=v
 profiles[str(d)]=dict(base_cycles=sum(state),source_words=4*96*inside,weight_words=updates,local_gathers=3456,add_issues=updates,psum_reads=480,psum_writes=480,psum_clears=0,bitmap_reads=reads,bitmap_writes=540,base_states=state)
 records.append(dict(index=i,fixture=str(d),output_origin_yx=g['output_origin_yx'][i].tolist(),**m))
(H/'sequence.txt').write_text('\n'.join(r['fixture'] for r in records)+'\n')
(H/'sequence_manifest.json').write_text(json.dumps(dict(passed=True,source_npz=str(Q/'sequence_tiles.npz'),source_metadata=str(Q/'sequence_tiles.json'),sequences=18,tiles=36,raw_values=36*3840,checked=['actual source x Q1/Q2','independent expanded W','actual FP32 identity to J','wide','I24','source-derived OS bitmap profile'],records=records),separators=(',',':'))+'\n')
(H/'profiles.json').write_text(json.dumps(profiles,separators=(',',':'))+'\n')
print(json.dumps(dict(passed=True,sequences=18,tiles=36,raw_values=36*3840,base_core_cycles=sum(profiles[r['fixture']]['base_cycles'] for r in records))))
