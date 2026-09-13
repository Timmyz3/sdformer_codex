#!/opt/anaconda3/bin/python3.12
"""Static coefficient compiler and independent integer gold, never dynamic RTL assistance."""
from pathlib import Path
import json, numpy as np
D=Path(__file__).resolve().parent
Bt=np.array([[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]],dtype=np.int64)
At=np.array([[1,1,1,0],[0,1,-1,-1]],dtype=np.int64)
G2=np.array([[2,0,0],[1,1,1],[1,-1,1],[0,0,2]],dtype=np.int64)
L=np.kron(At,At); R=np.kron(Bt,Bt)
def rne2(v):
 q=v//4;r=v%4
 return q+((r>2)|((r==2)&((q&1)!=0)))
def hexlines(path,rows):
 path.write_text(''.join(f'{int(x):0{width}x}\n' for x,width in rows))
def make(name,W,S,mask,provenance):
 out=D/'fixtures'/name;out.mkdir(parents=True,exist_ok=True)
 W=np.asarray(W,np.int64);S=np.asarray(S,np.int64)
 assert W.shape==(96,96,3,3) and S.shape==(10,96,4,4)
 assert np.all((S==0)|(S==1)) and np.max(np.abs(W))<32768
 U=np.einsum('ik,ockl,jl->ocij',G2,W,G2).reshape(96,96,16)
 Um=U*mask.repeat(8,axis=0)[:,None,:]
 E=np.einsum('px,ocx,xs->ocps',L,Um,R)
 V=np.einsum('ip,tcpq,jq->tcij',Bt,S,Bt).reshape(10,96,16)
 raw=np.empty((10,96,4),np.int64)
 for a in range(2):
  for b in range(2):raw[:,:,a*2+b]=np.einsum('ocij,tcij->to',W,S[:,:,a:a+3,b:b+3])
 wu=np.einsum('px,ocx,tcx->top',L,U,V)
 assert np.array_equal(wu,raw*4)
 masked_raw=np.einsum('ocps,tcs->top',E,S.reshape(10,96,16))
 assert np.array_equal(masked_raw,np.einsum('px,ocx,tcx->top',L,Um,V))
 golds={'direct':raw,'wino':raw,'masked_wino':rne2(masked_raw),'masked_expanded':rne2(masked_raw)}
 coeffs={'direct':W.reshape(96,864),'wino':U.reshape(96,1536),'masked_wino':Um.reshape(96,1536),'masked_expanded':E.reshape(96,6144)}
 meta={}
 for kind,C in coeffs.items():
  cb=2 if kind=='direct' else 3
  assert np.max(np.abs(C))<(1<<(cb*8-1))
  vectors=C.reshape(6,16,-1).transpose(0,2,1).reshape(-1,16)
  blob=bytearray()
  for vec in vectors:
   for n in vec:blob.extend(int(n).to_bytes(cb,'little',signed=True))
  (out/f'{kind}.bin').write_bytes(blob)
  support=(np.any(vectors[:,:8]!=0,axis=1).astype(np.uint8)|np.any(vectors[:,8:]!=0,axis=1).astype(np.uint8)*2)
  sm=[]
  for i in range(0,len(support),64):sm.append((sum(int(x)<<(2*j) for j,x in enumerate(support[i:i+64])),32))
  hexlines(out/f'{kind}.meta',sm)
  gold=golds[kind]
  assert gold.min()>=-(1<<31) and gold.max()<(1<<31)
  packed=gold.reshape(10,6,16,4).transpose(1,3,0,2).reshape(-1)
  (out/f'{kind}.gold').write_text(''.join(f'{int(x)}\n' for x in packed))
  meta[kind]={'coefficient_bytes':len(blob),'metadata_bytes':len(sm)*16,'zero_vectors':int(np.sum(support==0)),'single_live_half_vectors':int(np.sum((support==1)|(support==2))),'max_abs_coefficient':int(np.max(np.abs(C))),'output_min':int(gold.min()),'output_max':int(gold.max())}
 ib=[]
 for t in range(10):
  for cg in range(12):
   word=0
   for lane in range(8):
    bits=sum(int(x)<<i for i,x in enumerate(S[t,cg*8+lane].reshape(-1)))
    word|=bits<<(lane*16)
   ib.append((word,32))
 hexlines(out/'input.hex',ib)
 np.savez(out/'fixture.npz',Wq=W,S=S,U4=U,mask=mask,E4=E,gold=raw,masked_gold=rne2(masked_raw))
 summary={'provenance':provenance,'shape_S':list(S.shape),'shape_W':list(W.shape),'source_nnz':int(S.sum()),'source_density':float(S.mean()),'v_hist':{str(int(x)):int(n) for x,n in zip(*np.unique(V,return_counts=True))},'modes':meta,'masked_vs_original_nrmse':float(np.linalg.norm(rne2(masked_raw)-raw)/max(np.linalg.norm(raw),1))}
 (out/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
 return summary
if __name__=='__main__':
 rng=np.random.default_rng(913)
 W=rng.integers(-1024,1025,(96,96,3,3),dtype=np.int64)
 mask=np.ones((12,16),np.int64);mask[:,[0,3,12,15]]=0
 cases={'synthetic_zero':np.zeros((10,96,4,4),np.int64),'synthetic_dense':np.ones((10,96,4,4),np.int64),'synthetic_sparse':(rng.random((10,96,4,4))<.04).astype(np.int64)}
 result={name:make(name,W,S,mask,'synthetic diagnostic; deterministic random W and binary input, NOT captured model or AEE') for name,S in cases.items()}
 (D/'fixture_summary.json').write_text(json.dumps(result,indent=2)+'\n')
