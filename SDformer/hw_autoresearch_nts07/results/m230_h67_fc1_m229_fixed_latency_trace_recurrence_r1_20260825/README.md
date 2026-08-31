# M230 H67 FC1 M229 fixed-latency trace recurrence

All 100 frozen binary-FC1 payloads were re-decoded and matched the M225 group, read and service ledger with zero mismatch.  At a two-cycle accepted-request-to-response latency, raw K8 F2/F4 are 1.551620x/2.068357x versus raw K8 F1; logic-only throughput/area is 1.177240x/1.055125x.  Spatial-parent F4 composes to 2.155535x versus raw K1/F1.

These are no-stall trace-recurrence results, not physical SRAM, complete FC1/FFN, or system speedups.  Stage-3 nonbinary FC1 stays conventional.
