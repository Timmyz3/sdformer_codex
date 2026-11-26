#!/usr/bin/env python3
from pathlib import Path
import concurrent.futures,json,subprocess
import numpy as np
P=Path(__file__).resolve().parent;O=P/'actual_v2';r=json.loads((O/'STREAM_EXECUTION_V1.json').read_text());f=np.load(O/'STREAM_FEATURES_V1.npz');x=f['x160'];z=np.sum((x[:,:128:8]==0)&(x[:,1:128:8]==0),axis=1)
indices=[0]+[i for i in range(1,len(z)) if z[i]>0]
def run(i):
 row=r['rows'][i];p=O/(row['id']+'_base128');subprocess.run([str(P/'stream_features_v2'),row['input'],str(p)+'.f32',str(p)+'.json','-','rowband','0'],check=True);d=json.loads(Path(str(p)+'.json').read_text());got=np.fromfile(str(p)+'.f32',dtype=np.float32);mem=np.fromfile(str(p)+'.f32.ios',dtype=np.float32);assert np.array_equal(got,x[i,:128]) and np.array_equal(got,mem[100000:100128]);assert d['feature_ready_tick']==113938624-32*int(z[i]);return dict(identity=row['id'],zero_mass_cells=int(z[i]),actual_full_X128_feature_ticks=d['feature_ready_tick'],actual_IOS_handoff_matches=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(run,indices))
out=dict(schema='r34_base128_guard_service_validation_v2',executed_cases=rows,all120_zero_mass_counts=z.tolist(),full_dimensions=True,other116_service_inferred_from_fixed_work_and_observed_nonzero_guard=True)
(O/'BASE128_GUARD_VALIDATION_V2.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
