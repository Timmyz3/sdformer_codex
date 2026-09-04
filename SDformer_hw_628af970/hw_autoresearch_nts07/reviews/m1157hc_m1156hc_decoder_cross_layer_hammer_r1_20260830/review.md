# M1157HC different-author hammer

## Verdict

M1156's arithmetic is reproducible, but its performance admission is not. The independent replay reproduced every D0-D3 baseline/candidate cycle, update, commit, fill, hit, eviction, timestep flush and phase invariant, including the sum-weighted `125,974,084 / 65,186,002 = 1.932532754501x` result. That replay follows M1105/M672's destination-major matrix-pull order.

The executable frontend has a different order. M514/M523 accepts one source event and emits its 4/6/9 legal destination taps atomically without a global reorder. Therefore, consecutive destination ownership and the resulting one-entry accumulator hit rate are not supplied by the current protocol. The 1.932533x result is downgraded to a destination-major free-reorder upper bound. No run-accumulator, bridge RTL, VCS or EDA is authorized.

## Reconciliation with the existing charged PIDP ledger

M1156 and sealed M712 have identical contributor counts for the same interlaken sample0 calls:

| Layer | Contributors | M1156 free-order local ratio | M712 charged A1/PIDP | Weight identities/cache |
|---|---:|---:|---:|---:|
| D0 | 29,622,568 | 1.979143x | 0.221460x | 384/16 |
| D1 | 30,338,394 | 1.959911x | 0.255834x | 98/16 |
| D2 | 30,328,495 | 1.927627x | 0.251106x | 25/16 |
| D3 | 96,760,057 | 1.915126x | 2.186169x | 13/16 |

This is the same destination-pull/PIDP dataflow collision, not a new independent mechanism. M712 already charges deterministic pulls, bitmap probes, optimistic K8 groups, dense commits and a fully associative 16-entry weight cache. D0-D2 thrash their weight working sets; only D3 fits statically.

For sample0 all-four, full PIDP is `0.432875902x`. Selecting PIDP only for D3 and A1-OSG for D0-D2 is `1.368423824x`. M718's `1.474346419x` and joint-fairness `1.214175731x` are the three-sequence D0+D2+D3 headline subset and are not decoder-complete.

## Missing bridge cost

The deterministic 96-bit inverse scan needs at least 7,488,000 words, or 89.856 MB, over the four calls. A global reorder would materialize at least 31,489,158 grouped updates: 503.827 MB for one 16-byte pass, or 1.008 GB for write plus read. M1156's 240-KiB accounting leaves only 2,270 bytes after its one-entry cache, so an uncharged catalog cannot be hidden there.

The only authorized next step is a new D3-only, statically selected, bridge-inclusive CPU fast-kill. It must keep D0-D2 on A1-OSG and charge actual ingress, bitmap probes, K8 bank conflicts, 15-cycle group service, 13-of-16 weight-cache behavior, control and dense commit. It remains CPU-only until a fresh different-author hammer.
