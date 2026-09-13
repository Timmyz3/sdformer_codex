from pathlib import Path
import numpy as np,json
from prepare import make,D
src=Path('../data_and_quality/r0_contiguous_t10.npz');msk=Path('../data_and_quality/coordinate25_q16.npz')
z=np.load(src);m=np.load(msk)
results={}
for i,S in enumerate(z['source_bits']):
 name=f'real_{i:02d}'
 results[name]=make(name,z['weight_q16'],S,m['coordinate_live'],'matched-dense stage320 r0.conv2.0 native contiguous capture; '+str(src.resolve())+'; origin='+str(z['output_origin_yx'][i].tolist()))
 q=np.load(D/'fixtures'/name/'fixture.npz')
 assert np.array_equal(q['gold'].reshape(10,96,2,2),z['golden_accum'][i])
 assert np.array_equal(q['masked_gold'].reshape(10,96,2,2),m['golden_rne_q16'][i])
 assert np.array_equal(q['U4'].reshape(96,96,4,4),m['U4_q16'].reshape(96,96,4,4))
 E=m['phase_kernel4_q16'].reshape(96,4,96,16).transpose(0,2,1,3)
 assert np.array_equal(q['E4'],E)
(D/'real_fixture_summary.json').write_text(json.dumps(results,indent=2)+'\n')
