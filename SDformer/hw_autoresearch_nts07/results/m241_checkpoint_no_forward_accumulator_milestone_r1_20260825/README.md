# M241 Conv checkpoint/no-forward accumulator milestone r1

## Outcome

M241 implements the standalone chain selected by M238:

`M152 source-major descriptor/order -> M154-style four-bank INT8 macro token -> dense-address, row-interlocked no-forward Acc19 accumulator`

M149 is not instantiated.  The accumulator keeps lazy-valid state and the
runtime overflow guard, but it does not carry M155's four-bank 96-lane
same-address forwarding payload (`4 * 96 * 19 = 7,296` bits).  Same-address
tokens wait behind a one-entry bank/address interlock instead.

Synopsys VCS V-2023.12-SP1 passed with exact-SHA preflight and zero assertion or
integer-miter failures.

## Real checkpoint subset

- H67/Motion ep35, sample 5.
- `sttmultires_unet.resblocks.0.conv1.0`, partition 251, window 7.
- 126 ordered descriptors and 504 destination groups.
- 63 low-half plus 63 high-half descriptors.
- 8 real negative tuples in 2 descriptors.
- 504 exact accumulator writes and 4,032 lane comparisons, 0 mismatch.
- 56 weight macro reads plus 448 cache hits for 504 groups: 9.0x bounded
  weight-read work reduction.  This is not a cycle or energy speedup.
- The selected real source-major subset contains no tail.  Four prefix tails
  are covered separately with the same exact checkpoint-partition weights.

The real high-half descriptors reach local row 303.  Dense addressing maps the
largest tested address to `384 + 303 = 687`, inside a 768-entry logical macro.
The older concatenation form would produce 815, so M241 intentionally uses
`half * ROWS + row` rather than `{half,row}`.

## Fail-closed coverage

- Commit stalls and a same-address interlock.
- Synchronous reset flush of an accepted, uncommitted token.
- Stale sequence, replayed order and wrong checkpoint/cache epoch.
- Three younger-fault atomicity checks: every older full4 token commits once.
- Runtime signed19 overflow: all bank writes are suppressed atomically and
  younger accepted tokens are quarantined.
- Lazy-valid correctness with intentionally stale external accumulator memory.

## Claim boundary

M238's fair target remains `126,581,635 / 75,032,786 = 1.687017659x`.
M241 does not admit that ratio: this run uses 8 representative lanes, one
bounded real context and external SRAM models.  Full 96-lane VCS, the complete
M152 finite trace, matched SRAM macros, DC/STA and energy remain open.

Therefore `physical_speedup=false`, `system_speedup=false`,
`paper_ppa_ready=false`, and `headline=false`.

## Evidence

- Exact VCS run: `results/m241_checkpoint_no_forward_accumulator_directed_vcs_r1_exact_20260825/`
- Frozen vector export: `results/m241_ordered_checkpoint_subset_r1_20260825/`
- Contract: `contracts/m241_checkpoint_no_forward_accumulator_exact_vcs_contract_r1_20260825.json`
- RTL/SVA/TB: `rtl_m241/`, `verif_m241/`, `tb_m241/`

No M241 DC was launched while other Synopsys jobs were active.  `docs/359` was
not modified.
