from pathlib import Path
import json
import numpy as np
from numeric import Arithmetic,difference
HERE=Path(__file__).resolve().parent;m=Arithmetic(centered=True);r={}
for axis in ['ordinary','lifting_raw']:
    w=Path('/tmp/default_bn_20260912/ready')/axis
    x=np.fromfile(w/'dense.f32',np.float32).reshape(-1,96);c=np.fromfile(w/'coeff.f32',np.float32)
    s=m.statistics(x,c[:96],c[96:192],c[208]);y=m.output(x,s)
    r[axis]=dict(stats=difference(s[:3],np.fromfile(w/'dense_stats.f32',np.float32).reshape(3,96)),output=difference(y,np.fromfile(w/'dense_output.f32',np.float32).reshape(-1,96)))
    assert r[axis]['stats']['bit_differences']==r[axis]['output']['bit_differences']==0
(HERE/'centered_exact_check.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
