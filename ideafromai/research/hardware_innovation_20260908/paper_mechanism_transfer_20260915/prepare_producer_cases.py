"""Append producer boundary cases; preserve the upstream capture fixtures."""
from pathlib import Path
import sys, struct, json
import numpy as np

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent / 'psn/rtl/gp_slice'))
from prepare_cases import make
from prepare_intersection_cases import encode

D = np.asarray([[(c >> r) & 1 for c in range(8)] for r in range(3)], np.int64)
A = np.asarray([[1,-2,3],[-2,1,1],[0,0,0],[1,0,-1],[-3,2,1],
                [1,1,1],[-1,0,2],[0,1,0],[0,-1,0],[2,1,-2]], np.int64)
W = np.resize(np.array([-1,2,0,-3], np.int8), (2,384))
W[:,383] = [7,-5]
tau = np.resize(np.array([-1,0,1,5,-5], np.int64), (10,2))
extra = []
for pattern in ('remapped_mixed', 'late_C383', 'remapped_empty'):
    codes = np.zeros((16,384), np.uint8)
    if pattern == 'remapped_mixed':
        codes[:] = (np.arange(16*384).reshape(16,384)*3 + np.arange(16)[:,None]) % 8
    elif pattern == 'late_C383':
        codes[:,383] = 1
    for route, dec, coeff in [('packed_class', np.eye(8,dtype=np.int64)[1:], (A@D)[:,1:]),
                              ('packed_time', D, A)]:
        c = make(pattern+'_'+route, codes, W, dec, coeff, tau, [1.25,.75],
                 dict(real=False, route=route, directed=pattern,
                      identity='Producer boundary diagnostic, not network performance'))
        # Bijective relabeling preserves every mathematical source: old live
        # code 1 becomes 0, while old empty code 0 becomes nonzero code 7.
        c['codes'] = ((c['codes'].astype(np.int16)+7)%8).astype(np.uint8)
        c['decode'] = c['decode'][:,(np.arange(8)+1)%8]
        extra.append(c)
old = (H.parent/'consumer_transfer_20260914/gustav_intersection/cases.bin').read_bytes()
count = struct.unpack('<I', old[4:8])[0]
(H/'producer_rtl/cases.bin').write_bytes(b'GPS1'+struct.pack('<I', count+len(extra))+old[8:]+b''.join(map(encode,extra)))
(H/'producer_rtl/directed_cases.json').write_text(json.dumps(
    [dict(name=c['name'], **c['metadata']) for c in extra], indent=2)+'\n')
print(f'{count} original cases + {len(extra)} producer boundary cases')
