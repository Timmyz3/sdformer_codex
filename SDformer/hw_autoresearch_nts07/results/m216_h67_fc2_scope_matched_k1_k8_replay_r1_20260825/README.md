# M216 frozen-H67 scope-matched K1/K8 replay

This exact-SHA replay traverses the same 120 frozen H67 ep35 FC2 records in a
single analyzer and changes only the accepted source count per replay group.
Both points retain raw4 scanning, descriptor queue depth 8, two physical D8
windows, the eight-lane group interface, tags, done/stall, terminal close,
handoff, and the M214 same-cycle done-fence load rule.

Across 5,580,000 tokens and 143,894,510 events, K1 requires 429,716,335 cycles
and K8 requires 90,196,785 cycles.  The exact standalone sparse-frontend cycle
speedup is 4.764209001x.  The output-block-weighted event count is 412,900,394
cycles, so even a zero-control-overhead K1 oracle cannot reduce the speedup
below 4.577772855x against this K8 denominator.

The result is calibrated to exact Synopsys VCS recurrence tests but remains an
always-ready frontend-only cycle result.  Weight SRAM response latency,
accumulation, BN/requantization, residual commit, complete FC2/FFN, physical
speedup, and system speedup are outside this artifact.
