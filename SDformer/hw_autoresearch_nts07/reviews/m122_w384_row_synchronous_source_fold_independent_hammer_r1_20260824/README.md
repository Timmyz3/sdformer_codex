# M122 W384 row-synchronous source-fold independent hammer

Verdict: the frozen ideal cycle DSE is arithmetically sound and K1 reproduces
all 22 frozen M109 W384 recurrence fields.  Exhaustive independent selection
over all 65,536 masks and K={1,2,4,8} found no lost or duplicated source when
each accepted group is removed from the remaining mask.

K4 projects 351,410,711 candidate cycles and 3.172536901x against the inherited
fixed8 service-island denominator.  Its incremental gain over the K1 candidate
is 1.251265785x, not 3.173x.  The denominator is not an equal-controller,
full-network or system baseline.

The cache table needs careful wording.  The published 1,536 bytes is explicitly
per output block.  If all eight output blocks are resident, storage is 12,288
bytes before implementing four logical reads.  Four-copy realization of
single-read storage would reach 49,152 bytes, although that is an upper-bound
implementation example rather than a required design.  A 1,536-byte total
cache requires strict block-phased reuse and a cache-lifetime schedule, neither
of which M122 proves.

K4 also assumes four 768-bit logical weight reads, four 16-to-1 INT8 selectors
per lane and a 96-lane adder tree in one cycle.  Area, energy, timing and clock
degradation are not modeled.  M123 now independently validates a 16-update
same-address forwarding chain, but it is not integrated with the K4 selector,
cache and adder tree.

Run the strict audit from the hardware root:

```bash
python3 reviews/m122_w384_row_synchronous_source_fold_independent_hammer_r1_20260824/audit_m122_w384_row_synchronous_source_fold.py
```

Safe claim: exact frozen heldout service-island cycle DSE projects 3.173x versus
fixed8 and 1.251x incremental speedup versus K1.  K4 RTL/VCS, a realizable
multi-read cache, macro-inclusive PPA, physical speedup, system speedup and
headline admission remain false.
