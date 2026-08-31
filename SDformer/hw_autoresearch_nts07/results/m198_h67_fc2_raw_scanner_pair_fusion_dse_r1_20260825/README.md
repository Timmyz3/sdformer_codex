# M198 raw-bitmap scanner pair-fusion DSE

M198 closes M197's preindexed-nonzero oracle.  It scans all 36,480,000 raw
96-bit sn2 beats, including zero beats and token tails, preserves beat order,
forbids cross-window/token same-cycle fill and replays the exact 120 frozen FC2
payloads.

With two buffers, R2/R4/R8 raw-beat scanners take 97,542,123 / 90,222,444 /
87,973,357 cycles, or 1.175344x / 1.270698x / 1.303184x versus the raw-scanned
W1/R1/B2 baseline.  Pair fusion itself adds only 1.033391x / 1.050311x /
1.056571x against the corresponding iso-width W1 scanner.  R8 already shows
strong diminishing returns.

Each M198 point gives its compactor the same maximum descriptor width as its
raw scanner.  The next cheaper architecture must therefore decouple a four
raw-beat scanner from a two-descriptor stable compactor with finite backlog.
No RTL, SRAM timing, physical, FC2, FFN or system speedup is admitted here.
