"""Synthetic math checks before real response analysis, no data/GPU/fit."""
import json
from pathlib import Path
import numpy as np
import analyze as a


def main():
    mask = np.array([[True,True,False,False],[True,False,False,False],
                     [False,False,True,True],[False,False,False,True]])
    u = np.arange(32,dtype=float).reshape(2,4,4)/31
    d = np.flip(u,axis=2)-.3
    count,um = a.grid_labels(u,mask,2); c2,dm = a.grid_labels(d,mask,2)
    expected = 0.
    for yy in range(2):
        for xx in range(2):
            m = mask[yy*2:yy*2+2,xx*2:xx*2+2]
            assert count[yy,xx] == m.sum()
            if m.any():
                uu = u[:,yy*2:yy*2+2,xx*2:xx*2+2][:,m].mean(axis=1)
                dd = d[:,yy*2:yy*2+2,xx*2:xx*2+2][:,m].mean(axis=1)
                assert np.allclose(uu,um[:,yy,xx]) and np.allclose(dd,dm[:,yy,xx])
                expected += m.sum()*np.dot(uu,dd)/mask.sum()
            else:
                assert not um[:,yy,xx].any() and not dm[:,yy,xx].any()
    assert np.isclose(a.contraction(count,um,dm),expected,rtol=0,atol=1e-15)
    # Full support within a cell has opposing signs; averaging products is not
    # the declared separately averaged field contraction.
    x = np.zeros((2,4,4)); x[0,::2,:]=1; x[0,1::2,:]=-1
    v = np.ones((4,4),dtype=bool)
    n,xbar = a.grid_labels(x,v,1)
    assert a.contraction(n,xbar,xbar) == 0 and float(np.mean(np.sum(x*x,axis=0))) == 1
    n4,x4 = a.grid_labels(x,v,4)
    assert a.contraction(n4,x4,x4) == 1
    z = a.rank_metrics([0,1,2,3,4,5],[0,1,2,3,4,5])
    assert z['spearman'] == 1 and z['pairwise_agreement'] == 1 and z['regret_AEE'] == 0
    z = a.rank_metrics([5,4,3,2,1,0],[0,1,2,3,4,5])
    assert z['spearman'] == -1 and z['pairwise_agreement'] == 0 and z['regret_AEE'] == 5
    z = a.rank_metrics([1]*6,[0,1,2,3,4,5])
    assert z['spearman'] is None and z['pairwise_agreement'] == .5 and z['picked_action'] == 0
    assert a.ranks([1,1,2]).tolist() == [1.5,1.5,3.]
    assert 'torch' not in a.sys.modules
    out = Path(__file__).parent/'MATH_CHECKS_V1.json'; assert not out.exists()
    a.write(out,dict(status='PASS',GPU_used=False,GT_arrays_read=False,
        tests=['unequal_support_and_empty_cells','mean_product_not_product_mean',
               'full_resolution_cell_identity','ranking_perfect_reverse_ties'],
        source_sha256={n:a.sha(Path(__file__).parent/n) for n in ('analyze.py','test_math.py')}))
    print(json.dumps(dict(status='PASS',GPU_used=False)))


if __name__ == '__main__':
    main()
