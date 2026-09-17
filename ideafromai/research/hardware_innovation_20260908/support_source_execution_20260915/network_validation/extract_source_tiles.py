"""Select four fixed contiguous P32 groups per recaptured validation frame."""
import argparse
from pathlib import Path
import json
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('capture',type=Path)
p.add_argument('output',type=Path)
a=p.parse_args()
d=np.load(a.capture)
x=d['X_fp32']
starts=np.rint(np.linspace(0,x.shape[2]-32,4)).astype(int)
tiles=[]; names=[]
for i,name in enumerate(d['frame_file']):
    for start in starts:
        q=np.rint(x[i,:,start:start+32,:].astype(np.float64)*65536).astype(np.int64)
        assert q.min()>=-2**23 and q.max()<2**23
        tiles.append(q.transpose(1,2,0).astype(np.int32))
        names.append(str(name).removesuffix('.npy')+f'_p{start}')
np.savez_compressed(a.output,X_q16=np.stack(tiles),case_name=np.array(names),
    frame_file=d['frame_file'],starts=starts,A_fp32=d['A_fp32'],
    bias_fp32=d['bias_fp32'],center_fp32=d['center_fp32'],
    theta_fp32=d['theta_fp32'],D=d['D'])
a.output.with_suffix('.json').write_text(json.dumps(dict(
    frames=d['frame_file'].tolist(),fixed_contiguous_starts=starts.tolist(),P=32,
    cases=names,selection='Four equally spaced starts chosen without activity inspection.',
    split='Two validation frames already reused in exploratory work; not unseen validation.',
    scope='Real continuous source input only; no g/code/Y/PSN oracle sent to DUT.'),indent=2)+'\n')
