# M265 matched-boundary ATLIF module cycles

M265 first overlays four corrections on frozen M258 without changing M258's
core cycle table: the actual maximum tag is 738,658,303; the exact registered-
FIFO producer-stall counts replace the invalid lower-bound wording; M258 only
had a fixed config/release barrier, not config sensitivity; and the population
is 7,318,350 five-beat tile results comprising 36,591,750 ordered result beats.

The new model compares a tile-closed exact-96 Fixed T10 schedule against a
complete rank-3 candidate.  Both consume the same five raw 256-bit beats per
tile, use the same 256-bit config bus and ready trace, and emit the same five
48-bit registered-FIFO result beats.  Rank-3 explicitly executes five-cycle
stage1 and five-cycle CSD stage2 on distinct resources; M37 alone is never
counted as the candidate.  Every one of 45 contexts drains before release.

The JSON and CSV contain the ideal point plus isolated result, ingress, config,
and joint periodic-pressure sweeps.  Ideal matched-boundary cycles are
124,412,490 versus 36,592,605 (`3.399935x`).  The independent M25 exact-96
cross-tile arithmetic lower-bound comparison is `3.333333x`; the difference is
the explicit 17-cycle tile closure.  All speedups are isolated ATLIF analytical
module-cycle ratios.  They are not system speedup, throughput/area, trained
rank-3 accuracy, energy, paper PPA, or headline claims.  Integrated RTL and an
area-matched Synopsys comparison remain future gates.
