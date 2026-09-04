# M1596 — M1591 C1 full-storage macro-area model independent review

Verdict: **PASS, but only as `0.988049 mm² [macro area model]`**.  This is an
area-accounting result, not a timing-integrated or paper-PPA-ready C1 result.

## Independently reproduced

- The frozen storage ledger is `214,912 B`: parent scratch `18,432 B`,
  metadata/reserve `24,448 B`, psum `122,880 B`, and weight `49,152 B`.
- At `2,048 B` per foundry `128×128-bit 1RW` macro, conservative per-class
  rounding gives `9 + 12 + 60 + 24 = 105` macros.  These represent `215,040 B`,
  add only `128 B` rounding overhead, and leave `30,720 B` below the `240 KiB`
  budget.
- M993 reports `147,246.392090 µm²` total and `78,825.243164 µm²` for its nine
  parent macros.  Therefore the logic remainder is `68,421.148926 µm²`, each
  same-foundry macro is `8,758.360351555... µm²`, and linear replication gives
  `988,048.985839333... µm² = 0.988048985839333... mm²`.
- The published result rebuilt byte-for-byte.  The three author tests passed on
  CPython 3.12.  The independent checker passed byte-identically on CPython
  3.6 and 3.12.  The production builder itself intentionally requires a newer
  Python syntax and is not CPython-3.6 compatible.
- Eighteen independent mutations were rejected, including macro undercounts,
  erased rounding, nine-macro area substitution, and forged full-netlist,
  timing, power, throughput/mm², or system-speedup claims.

## Claim boundary

The exact paper-safe wording is:

> Mapping the frozen `214,912 B` C1 storage ledger to 105 instances of the same
> foundry SRAM macro yields `0.988049 mm² [macro area model]` and remains within
> the `240 KiB` capacity budget.

The prior `3 ns` setup result applies only to the existing logic plus nine
parent macros.  The additional 96 macros are area-replicated but not integrated
into a full-storage netlist or timing top.  Consequently M1591 does **not**
admit full-storage timing, power, energy, throughput, throughput/mm², system
speedup, or paper-ready PPA.  Capacity rounding is conservative; the modeled
logic-plus-storage number is not a guaranteed whole-design upper bound because
added decode/interconnect/routing and physical overhead are unmeasured.

One P2 test-quality issue remains: the author overwrite unit test calls the
guard primitive directly instead of invoking the CLI.  M1596 closes the
evidence gap by exercising the real CLI against an occupied output and proving
nonzero exit with unchanged contents.
