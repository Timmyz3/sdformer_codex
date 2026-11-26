"""Read audited R26 binary activations/weights only; no GT or model call."""
import hashlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
RAW=ROOT/'research/flow_story_20261017/window_precision_round26/actual_v1/run_v1'
OUT=HERE.parent/'rtl_results'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def payload(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
assert sha(RAW/'results.json')=='ca16d2b0c491841f45e43a9ace4010de83706585950903ac22da4d2a4a42ce5c'
r=json.loads((RAW/'results.json').read_text());rec=r['forward_records'][0]
archive=RAW/rec['prediction_file'];qpath=RAW/r['quantization_archive']['path']
assert sha(archive)==rec['prediction_file_sha256']
assert sha(qpath)==r['quantization_archive']['sha256']
contract=json.loads((HERE.parent/'MASK_CONTRACT_V1.json').read_text())
group='sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.2.mlp'
selected={tuple(x) for x in contract['evaluation_tiles'][group]}
coords=[(t,y,x) for t in (0,9) for y,x in ((0,0),(0,18),(8,8),(9,9),(26,35),(29,39))]
records=[];cases=[]
with np.load(archive,allow_pickle=False) as a,np.load(qpath,allow_pickle=False) as q:
 for fc,sn in (('fc1','sn1'),('fc2','sn2')):
  spike=a[sn];weight=q[fc+'__q8']
  assert payload(spike)==rec['tensor_records'][sn]['sha256']
  assert spike.shape[-1]==weight.shape[1] and np.isin(spike,[0,1]).all()
  for t,y,x in coords:
   high=y>=27 or x>=36 or (y//9,x//9) in selected
   for start in (0,len(weight)-16):
    for q8 in (high,True):
     v=spike[t,0,y,x].astype(np.int8);w=weight[start:start+16]
     cases.append((len(v),q8,v,w))
     records.append(dict(fc=fc,coordinate=[t,0,y,x],outputs=[start,start+16],
                         mode='Q8' if q8 else 'Q4',input_sha256=payload(v),q8_weight_sha256=payload(w),
                         note='Actual binary operands; layer-local integer check, not full-network re-execution'))
out=OUT/'actual_vectors_v1.txt'
assert not out.exists()
with out.open('w') as f:
 f.write(str(len(cases))+'\n')
 for k,q8,x,w in cases:
  f.write(f'{k} {int(q8)}\n')
  f.write(' '.join(map(str,x.tolist()))+'\n')
  f.write(' '.join(map(str,w.reshape(-1).tolist()))+'\n')
manifest=dict(schema='flow_r30_actual_PE_vectors_v1',cases=len(cases),records=records,
 source_sha256={str(p):sha(p) for p in (RAW/'results.json',archive,qpath,HERE.parent/'MASK_CONTRACT_V1.json',Path(__file__))},
 vectors_sha256=sha(out),model_forwards=0,GT_read=False)
(OUT/'ACTUAL_VECTOR_MANIFEST_V1.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(cases=len(cases),vectors_sha256=manifest['vectors_sha256'],bytes=out.stat().st_size)))
