"""No quality input: export the 56 actual score vectors for RTL validation."""
import json,struct,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=HERE.parent/'actual_g0_v1/scores.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(p)=='a67569d3981ce286c75542c4e2e54a1279e7b6ebbe92f6a09d6c66c6a36e51e1'
r=json.loads(p.read_text());assert r['status']=='G0_CPU_COMPLETE'
lines=[];records=[]
for f in r['frames']:
 for n in r['score_names']:
  scores=[v[n] for v in f['scores']]
  best=max(range(12),key=lambda i:(scores[i],-i));tile=[best//4,best%4]
  assert f['policies'][n]['selected_tile']==tile
  words=[struct.unpack('>Q',struct.pack('>d',x))[0] for x in scores]
  lines.append(' '.join(f'{w:016x}' for w in words))
  records.append({'target_id':f['target_id'],'score':n,'winner':best,'fine_mask':1<<(best+best//4)})
assert len(records)==56
dest=HERE/'results/actual_score_vectors.txt';assert not dest.exists();dest.write_text('\n'.join(lines)+'\n')
(HERE/'results/actual_score_manifest.json').write_text(json.dumps({'score_source_sha256':sha(p),'vectors_sha256':sha(dest),
 'vectors':56,'payload_score_words':672,'model_forwards':0,'quality_read':False,'expected':records},indent=2)+'\n')
