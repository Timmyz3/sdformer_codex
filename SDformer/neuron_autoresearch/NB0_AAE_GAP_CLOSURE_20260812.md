# NB0 AAE Gap Closure (2026-08-12)

Machine receipt: `neuron_autoresearch/NB0_AAE_GAP_CLOSURE_20260812.json`.

## Final conclusion

- Formula audit PASS: legacy AAE is 2-D direction angle; benchmark AE uses Barron/Middlebury `(u,v,1)`.
- NB0 is operationally plateaued/overfit. Its budget-30 checkpoint remains AEE rank-1 after equal training at budgets 35 and 40.
- The three local aggregations all remain above official hidden-test AE 4.871; aggregation choice alone does not close the gap.
- The remaining numerical gap is not evidence that local NB0 needs more epochs. Official hidden test and local valid825 are different populations/protocols.

## NB0 equal+10

| budget | AEE | AAE-2D | AE-3D | spikes (G) |
|---:|---:|---:|---:|---:|
| 30 | 1.4454 | 6.5128 | 6.1803 | 126.1156 |
| 35 | 1.4584 | 6.5741 | 6.2463 | 127.0435 |
| 40 | 1.4549 | 6.5222 | 6.2109 | 128.0836 |

## Same-population rank-1 comparison

| route | AEE | AAE-2D | AE-3D | spikes (G) | vs NB0 AEE | vs NB0 AE-3D | vs NB0 spikes |
|---|---:|---:|---:|---:|---:|---:|---:|
| NB0 | 1.4454 | 6.5128 | 6.1803 | 126.1156 | - | - | - |
| H67 | 1.3297 | 5.9004 | 5.6509 | 82.1107 | 8.00% | 8.57% | 34.89% |
| Local5 | 1.3153 | 5.8291 | 5.5379 | 84.4197 | 9.00% | 10.39% | 33.06% |
