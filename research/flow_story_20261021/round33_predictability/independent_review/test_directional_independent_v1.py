"""Independent formula/observability review of the fixed X160 addendum."""
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROUND = HERE.parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = HERE/"DIRECTIONAL_INDEPENDENT_CPU_V1.json"
    assert not output.exists()
    oldpath = ROUND/"algorithm/predictors.py"
    newpath = ROUND/"directional_branch/predictors.py"
    old, new = module("old128_independent", oldpath), module("new160_independent", newpath)
    assert sha(oldpath) == "35663dd6b999621652d9d185d72b967e33c380e1064a09ec75540d2759cd61c3"

    # Native-sized binary moving blocks give exact count/mass preservation under
    # mirrors while the new correlations must change the mirrored axis sign.
    a = np.zeros((10, 480, 640), dtype=np.float64)
    for y in range(4):
        for x in range(4):
            for t in range(10):
                a[t, y*120+24+t:y*120+40+t, x*160+30+2*t:x*160+54+2*t] = 1
    base = new.voxel_features(a)
    assert np.array_equal(base[:128], old.voxel_features(a))
    c = base[128:].reshape(16, 2)
    assert np.all(np.abs(c) > .05) and np.all(c < 0)
    mirror_cases = []
    for mx, my in [(True, False), (False, True), (True, True)]:
        b = a.copy()
        for y in range(4):
            for x in range(4):
                block = a[:,y*120:(y+1)*120,x*160:(x+1)*160]
                b[:,y*120:(y+1)*120,x*160:(x+1)*160] = block[:,::-1 if my else 1,::-1 if mx else 1]
        feat = new.voxel_features(b)
        assert np.array_equal(feat[:128], base[:128])
        expected = c*np.asarray([-1 if mx else 1, -1 if my else 1])
        assert np.allclose(feat[128:].reshape(16,2), expected, rtol=1e-14, atol=1e-14)
        mirror_cases.append(dict(horizontal=mx,vertical=my,base128_exact=True,
                                 signed_coupling_max_error=float(np.max(np.abs(feat[128:].reshape(16,2)-expected)))))

    # Independently calculate sums with explicit time/row/column loops on a
    # smaller legal tensor. This catches cell-boundary and alignment mistakes.
    rng = np.random.default_rng(10332026)
    toy = rng.normal(size=(10,16,20))
    expected = []
    for cy in range(4):
        for cx in range(4):
            block = toy[:,cy*4:(cy+1)*4,cx*5:(cx+1)*5]
            terms = []
            for t in range(9):
                for y in range(1,3):
                    for x in range(1,4):
                        it = float(block[t+1,y,x]-block[t,y,x])
                        ix = float((block[t+1,y,x+1]+block[t,y,x+1]-block[t+1,y,x-1]-block[t,y,x-1])/4)
                        iy = float((block[t+1,y+1,x]+block[t,y+1,x]-block[t+1,y-1,x]-block[t,y-1,x])/4)
                        terms.append((ix,iy,it))
            for axis in (0,1):
                numerator = math.fsum(v[axis]*v[2] for v in terms)
                den = math.sqrt(math.fsum(v[axis]**2 for v in terms)*math.fsum(v[2]**2 for v in terms))
                expected.append(numerator/den if den>1e-12 else 0)
    calculated = new.directional_couplings(toy)
    assert np.allclose(calculated, expected, rtol=1e-13, atol=1e-14)
    assert np.all(np.abs(calculated)<=1+1e-14)
    assert np.array_equal(new.directional_couplings(np.zeros_like(toy)),np.zeros(32))
    assert np.array_equal(new.directional_couplings(np.ones_like(toy)),np.zeros(32))
    for scale in [3.0,-2.0]:
        assert np.allclose(new.directional_couplings(toy*scale),calculated,rtol=1e-13,atol=1e-14)
    assert np.array_equal(new.directional_couplings(toy*1e-10),np.zeros(32))
    # Verify no cross-cell derivative: perturb another cell, leave first pair.
    changed = toy.copy(); changed[:,4:8,5:10] += rng.normal(size=(10,4,5))*100
    assert np.array_equal(new.directional_couplings(changed)[:2],calculated[:2])
    out = dict(schema="flow_r33_directional_independent_cpu_v1",status="PASS",
        source_sha256={"base128_predictors":sha(oldpath),"directional_predictors":sha(newpath),
                       "addendum":sha(ROUND/"DIRECTIONAL_ADDENDUM_V1.md"),"independent_test":sha(__file__)},
        native_mirror_cases=mirror_cases,native_first_cell_cx_cy=c[0].tolist(),
        explicit_scalar_max_abs_difference=float(np.max(np.abs(calculated-np.asarray(expected)))),
        checks=["native shape frozen128 exact", "three cellwise mirrors change only correct coupling signs",
                "independent scalar spatial/time common ROI calculation", "zero/static inputs and denominator guard",
                "positive/negative amplitude scaling", "no derivative across cell borders"],
        limitations=["Pure synthetic observability checks do not establish event brightness constancy or actual damage predictability.",
                     "The fixed small-denominator guard deliberately breaks amplitude invariance near zero.",
                     "Spatial-temporal correlation features are not claimed novel and must be shared by all ordinary rivals."],
        actual_new_inputs_read=False,new_GT_read=False,new_quality_read=False,real_predictor_fit=False,GPU_used=False)
    output.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"status":"PASS","native_first_cell_cx_cy":c[0].tolist(),"scalar_max_abs_difference":out["explicit_scalar_max_abs_difference"]}))


if __name__ == "__main__":
    main()
