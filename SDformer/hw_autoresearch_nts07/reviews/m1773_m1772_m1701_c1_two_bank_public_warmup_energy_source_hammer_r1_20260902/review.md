# M1773 independent source hammer

## Verdict

PASS, 99/100. P0/P1/P2 are all zero. M1774 may be authored; M1773 alone does not authorize or execute the campaign.

## What was independently checked

- Epoch 5943 warms metadata bank0. Public `psum_write_ready` and `row_complete_ready` backpressure keeps it occupied while epoch 5944 is loaded through the same public prep interface into bank1. Both tasks finish before epoch 5945 opens the only SAIF window.
- No active `force`, `release`, hierarchical DUT read/write, `initreg`, unknown-state waiver, timing-check waiver, notifier waiver, or old binary/csrc reuse exists.
- The mapped execution counters clear at each task execution start. The architectural scoreboard and counters therefore refer to the measured epoch 5945 without subtracting warmup activity.
- The SAIF lexer preserves comment delimiters inside strings and strips C block comments only outside strings. Eight independent malformed/TX/duration mutations were rejected.
- A generated full-cardinality fixture exercised all five required tags at 117,690 forms per tag (588,450 total), TX=0, unique top/DUT/scratch hierarchy, and `duration = cycles * 3 ns`.
- The mapped netlist contains nine parent SRAM macros. PTPX fails unless mapped-net and leaf-cell annotations are each exactly 100%; its primary report is the whole mapped top including those macro Liberty models.
- The author suite passes 9/9 under CPython 3.6.8 and 3.10.18. `docs/359` remains at the frozen SHA.

## Boundary

This is a source-only authorization review. It creates no power or energy result. The future campaign is one directed 64-row component workload under UNIT_DELAY functional activity, with separately sealed PrimeTime timing. A distinct result hammer is mandatory before citation.
