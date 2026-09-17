"""Independent exact L/H interval vs normalized integer comparisons."""
from pathlib import Path
import json
import numpy as np

rng = np.random.default_rng(901515)
total = 0
ties = 0
for m in range(24):
    n = -rng.integers(0,327681,size=4096,dtype=np.int64)
    p = rng.integers(0,327671,size=4096,dtype=np.int64)
    nv = rng.integers(-(1<<20),1<<20,size=4096,dtype=np.int64)
    lo = nv*(1<<m)+n*((1<<m)-1)
    hi = nv*(1<<m)+p*((1<<m)-1)
    for tau in [lo-1,lo,lo+1,hi-1,hi,hi+1,
                np.full_like(lo,-(1<<47)),np.full_like(lo,(1<<47)-1),
                rng.integers(-(1<<47),(1<<47),size=4096,dtype=np.int64)]:
        for positive in [False,True]:
            delta = int(positive)
            lower = nv+n > np.floor_divide(tau+n-delta,1<<m)
            upper = nv+p <= np.floor_divide(tau+p-delta,1<<m)
            ref_lower = lo>=tau if positive else lo>tau
            ref_upper = hi<tau if positive else hi<=tau
            np.testing.assert_array_equal(lower,ref_lower)
            np.testing.assert_array_equal(upper,ref_upper)
            total += len(tau)
            ties += int(((tau==lo)|(tau==hi)).sum())
result = dict(decision_cases=total,boundary_ties=ties,m_range=[0,23],
    tau_range=[-(1<<47),(1<<47)-1],lower_upper_equivalent=True,
    scope='integer algebra only; no measured clocks or RTL cycle claim')
(Path(__file__).resolve().parent/'identity_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
