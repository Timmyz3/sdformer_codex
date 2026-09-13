from pathlib import Path
import json,numpy as np
H=Path(__file__).resolve().parent;R=H.parent.parent
Bt=np.array([[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]],dtype=np.int64)
At=np.array([[1,1,1,0],[0,1,-1,-1]],dtype=np.int64)
G2=np.array([[2,0,0],[1,1,1],[1,-1,1],[0,0,2]],dtype=np.int64)

def make(name,W,raw_source,origin=(0,0)):
 W=np.asarray(W,np.int64);raw_source=np.asarray(raw_source,np.int64)
 assert W.shape==(96,96,3,3) and raw_source.shape==(10,96,4,4)
 assert W.min()>=-32768 and W.max()<=32767
 valid=((np.arange(4)+origin[0]>=0)&(np.arange(4)+origin[0]<240))[:,None]&((np.arange(4)+origin[1]>=0)&(np.arange(4)+origin[1]<320))[None,:]
 S=raw_source*valid[None,None,:,:]
 U2=np.einsum('ix,ocyx->ocyi',G2,W)
 V=np.einsum('ix,tcyx->tcyi',Bt,S)
 gold=np.zeros((10,96,2,2),dtype=np.int64)
 M=np.zeros((10,96,2,4),dtype=np.int64)
 for y in range(2):
  for ky in range(3):
   M[:,:,y,:]+=np.einsum('oci,tci->toi',U2[:,:,ky,:],V[:,:,y+ky,:])
  for x in range(2):gold[:,:,y,x]=np.einsum('ocyx,tcyx->to',W,S[:,:,y:y+3,x:x+3])
 numerator=np.einsum('xi,toyi->toyx',At,M)
 assert np.array_equal(numerator,2*gold) and np.all(numerator%2==0)
 path=H/'fixtures'/name;path.mkdir(parents=True,exist_ok=True)
 (path/'origin.txt').write_text(f'{origin[0]} {origin[1]}\n')
 for kind,C,bits in [('direct',W.reshape(96,864),16),('one_axis',U2.reshape(96,1152),18)]:
  vectors=C.reshape(6,16,-1).transpose(0,2,1).reshape(-1,16)
  packed=np.packbits(((vectors.reshape(-1,1)&((1<<bits)-1))>>np.arange(bits)&1).astype(np.uint8).reshape(-1),bitorder='little')
  (path/f'{kind}.bin').write_bytes(packed.tobytes())
  support=np.any(vectors[:,:8]!=0,axis=1).astype(np.uint8)+2*np.any(vectors[:,8:]!=0,axis=1).astype(np.uint8)
  words=[]
  for i in range(0,len(support),64):words.append(sum(int(v)<<(2*j) for j,v in enumerate(support[i:i+64])))
  (path/f'{kind}.meta').write_text(''.join(f'{v:032x}\n' for v in words))
  (path/f'{kind}.gold').write_text(''.join(f'{int(v)}\n' for v in gold.reshape(10,6,16,4).transpose(1,3,0,2).reshape(-1)))
 words=[]
 for t in range(10):
  for cg in range(12):
   words.append(sum(int(raw_source[t,cg*8+l,y,x])<<(16*l+y*4+x) for l in range(8) for y in range(4) for x in range(4)))
 (path/'input.hex').write_text(''.join(f'{v:032x}\n' for v in words))
 np.savez(path/'fixture.npz',Wq=W,S=raw_source,valid=valid,U2=U2,V=V,gold=gold,origin=origin)
 info={'origin':list(map(int,origin)),'source_bits':int(S.sum()),'weight_range':[int(W.min()),int(W.max())],'U2_range':[int(U2.min()),int(U2.max())],
 'coefficient_bytes':{'direct':165888,'one_axis':248832},'V_histogram':{str(int(v)):int(c) for v,c in zip(*np.unique(V,return_counts=True))},'identity_equal':True}
 (path/'manifest.json').write_text(json.dumps(info,indent=2)+'\n');return info

if __name__=='__main__':
 z=np.load(R/'r0_execution_trials_20260913/data_and_quality/dense_q16.npz')
 origins=z['input_origin_yx'];out={}
 for i in range(8):out[f'real_{i:02d}']=make(f'real_{i:02d}',z['weight_q16'],z['source_bits'][i],tuple(origins[i]))
 rng=np.random.default_rng(914);W=rng.integers(-32768,32768,(96,96,3,3),dtype=np.int64)
 for name,S,origin in [('zero',np.zeros((10,96,4,4),np.int64),(0,0)),('one',np.ones((10,96,4,4),np.int64),(0,0)),('padding_poison',np.ones((10,96,4,4),np.int64),(-1,-1))]:out[name]=make(name,W,S,origin)
 S=np.zeros((10,96,4,4),np.int64);S[0,0,1,1]=1;S[9,95,3,3]=1
 out['isolated']=make('isolated',W,S)
 W[:8]=0;W[16:]=0
 out['half_zero']=make('half_zero',W,np.ones_like(S))
 (H/'fixture_summary.json').write_text(json.dumps(out,indent=2)+'\n');print('FIXTURES',len(out))
