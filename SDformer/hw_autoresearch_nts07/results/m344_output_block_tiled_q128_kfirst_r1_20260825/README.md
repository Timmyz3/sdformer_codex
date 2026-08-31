# M344 fixed-64-KiB output-block-tiled q128 DSE

M344 trades output-block tile width for pattern capacity under one fixed 64 KiB
double-context PWP/weight/pattern cache: q16/O8, q32/O4, q64/O2 and q128/O1.
The product q times O is always 128. A separate two-context 48-bit assignment
descriptor SRAM costs 36,000 bytes.

On the exact M339 runtime cohort, the wide-port systolic points have strict
speedups of 1.470188x, 1.624304x, 1.787374x and 1.970784x. Allowing the next
partition first tile to overlap the current body gives 1.538734x, 1.690179x,
1.854125x and 2.038802x. The q128/O1 worst cache context is 20,224 bytes, or
40,448 bytes for two contexts, so it fits the fixed 64 KiB cache.

Both schedules remain unadmitted cycle bounds. Finite queues, bank conflicts,
RTL cycle match, area normalization, energy and system speedup are open.

