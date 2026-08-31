# M162 PAFT ep4 BN-policy valid825 receipt

Both 825-frame evaluations completed on the A800 with strict checkpoint load
(`missing=0`, `unexpected=0`), 105 ATLIF modules and 12 attention modules.

| BN policy | AEE | hardware interpretation |
|---|---:|---|
| `no_running` | 1.309925 | sample-dependent statistics across 78 BN modules; not statically foldable |
| `running` | 1.469151 | frozen inference statistics; eligible for BN folding |

The 0.159226 AEE gap means PAFT is not promoted as a hardware-accuracy win.
The next algorithm action is BN running-stat recalibration with frozen weights,
followed by another `running` valid825.  If recalibration cannot close the gap,
PAFT training must explicitly enforce running-stat/fold consistency.

The PAFT checkpoint still has exactly unit `sn2` thresholds in all 12 FFNs, so
the signed INT8 weight-add identity used by the hardware remains valid.
