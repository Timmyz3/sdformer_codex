from pathlib import Path
import json,numpy as np
H=Path(__file__).resolve().parent;D=H.parent/'data'
m=json.loads((D/'manifest.json').read_text());assert m['complete'] and m['gold_complete']
source=np.load(D/m['source_words'],mmap_mode='r');assert source.shape==(96,240,320) and source.dtype==np.dtype('<u2')
assert int(source.max())<1024
out={}
for arm,entry in m['arms'].items():
 z=np.load(D/entry['weights']);w=z['weight_q16'];live=z['live']
 assert w.shape==(96,96,3,3) and live.shape==(12,24)
 path=H/'parameters'/arm;path.mkdir(parents=True,exist_ok=True)
 w.reshape(12,8,864).transpose(0,2,1).astype('<i2').tofile(path/'weight.bin')
 live.astype('u1').tofile(path/'mask.bin')
 gold=np.load(D/entry['gold'],mmap_mode='r');assert gold.shape==(19200,10,96,2,2) and gold.dtype==np.dtype('<i4')
 out[arm]={'source':str(D/m['source_words']),'gold':str(D/entry['gold']),'weight':str(path/'weight.bin'),'mask':str(path/'mask.bin')}
(H/'inputs.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'arms':list(out),'source_shape':list(source.shape),'gold_shape':list(gold.shape)}))
