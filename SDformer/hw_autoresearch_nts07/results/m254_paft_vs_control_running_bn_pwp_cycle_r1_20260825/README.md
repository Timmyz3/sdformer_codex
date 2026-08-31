# M254 paired PAFT versus control running-BN PWP cycle replay

M254 compares the M87 PAFT-ep4 and no-PAFT-control ep4 checkpoints on the
same ten DSEC samples, the same deployable running-BN policy, the same four
bottleneck Conv operators, the same disjoint M77 train-only catalog and the
same M251 fixed12 PWP cycle model.  All 80 source payloads are identity-bound,
rehashed and expanded into the exact Conv3x3 row/partition population.

PAFT reduces bit-sparse vector work by `13.859073%` and fixed12 PWP candidate
work by `13.155264%`.  On the same hardware, candidate cycles fall by
`13.153832%` (`1.151461x` throughput) with separate 144-byte PWP and 96-byte
weight ports, and by `13.266428%` (`1.152956x`) when one 96-byte port is
shared.  PAFT is faster on all 10 paired samples.  Its pattern efficiency is
slightly worse (`-0.810422%`), so the gain comes from reduced activity rather
than a favorable change to the fixed catalog.

The paired valid825 running-BN AEE improves by only `0.573022%` in one seed.
This is an isolated four-Conv algorithm/hardware co-design ablation, not a
multi-seed accuracy claim, PAFT checkpoint INT8/Acc19 admission, integrated
RTL cycle result, energy result, system speedup, paper PPA or headline.
