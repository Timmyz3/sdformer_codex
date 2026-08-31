# M261 independent hammer of M258

Score: **89/100**. Severity: **P0=0, P1=2, P2=5**.

The frozen population and four ready-profile cycle ratios independently
recompute exactly. The trace has 45 ordered T10 ATLIF contexts, 7,318,350
factor tiles and 36,591,750 output beats per inference. The serial/candidate
cycle pairs are 73,183,590/36,592,065, 73,183,603/39,031,509,
73,183,619/41,819,452 and 73,183,642/48,789,289. Every context pays one
configuration and one release cycle, drains before release, and does not
overlap the next context.

The frozen M31 and M37-r10 result ports are T10-compatible, but their inputs
are not a matched executable boundary: M31 consumes five 256-bit raw beats;
M37-r10 consumes one 384-bit already-produced intermediate and omits candidate
stage1. Configuration is also fixed at one cycle rather than sensitivity
swept. The admitted result is therefore a conditional trace-driven ATLIF
module-cycle sensitivity model, not integrated module throughput.

Three auxiliary corrections are required: the 87.5% and 75% producer-stall
"lower bounds" exceed independently modeled registered-FIFO stalls by 58 and
203 cycles; the reported maximum tag is a conservative composite rather than
the actual maximum; and the README must say 36,591,750 result beats, not that
many five-beat results.

The producer replay is byte-identical, wrong trace SHA fails before output
creation, and docs/359 remains unchanged. System speedup, headline, integrated
RTL, throughput/area, trained accuracy, energy and paper PPA remain false.
