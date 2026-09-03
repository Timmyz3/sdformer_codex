# M2054 TSBG matched mapped-energy independent source hammer

## Verdict

**FAIL_CLOSED, 42/100, P0/P1/P2 = 4/4/2. No M2054 VCS, SAIF, PTPX, license query, release, or one-shot runner is authorized.**

The overall hybrid idea is usable: exactly one schedule axis can be mapped while
the other stays at the frozen RTL, the UCLI files name distinct mapped scopes,
the first stop is before the next execute clock, and the final UCLI `run` can
resume M2051 through the protocol attacks, reset recovery, scoreboard and final
`PASS_M2051...` token. The reviewed source hashes are nevertheless not runnable
or citable.

## P0 blockers

1. Both filelists omit the mapped Verilog they are named to simulate. The
   `_SCHEDULE_MODE0`/`_SCHEDULE_MODE1` modules referenced by the hybrid adapter
   are undefined.
2. The unpacked-array adapter direction is reversed relative to DC. The mapped
   netlist directly shows original bank-0/lane-0/bit-0 entering
   `mem_rsp_weight[1016]`, while original bank-7/lane-15/bit-0 enters packed
   bit 0. M2054 instead connects bank 0/lane 0 to bit 0. All ascending
   `[0:N]` bank/lane arrays, including the commit lanes, require high-segment
   mapping.
3. The measurement identity is false. In the full40 fixture, global slot 0 is
   sample 0/layer 8/FC1/token 0/G6, with 23 rows, 222 issues and 6912 products;
   the sealed prior execution is 3184 versus 1043 cycles. M2054 labels the
   same default slot as layer 28/G48. Layer 28/G48 is global slot 42.
4. The PTPX TCL links only TT before reading an SDC that names the SSG library
   and SSG operating condition. SSG must be loaded before `read_sdc`, followed
   by an explicit TT override.

## P1 defects

- `+WORKLOAD_SLOT` remains arbitrary while the wrapper always prints slot 0.
- The 383-cycle preload is a literal message rather than an asserted invariant.
- X/Z checks exist only at the two window endpoints, not every active SAIF
  clock.
- axis, design, strip path and measurement cycles are independent free
  environment variables in PTPX and are not reconciled with the final M2051
  receipt or SAIF duration.

## What is already structurally correct

- The M2030 mapped top names and public widths match the adapter declarations.
- In each build, the parameterized generate selects one mapped DUT and one RTL
  DUT.
- Ordinary UCLI selects only
  `...core.dut_base.g_mapped.mapped_implementation`; TSBG selects only
  `...core.dut_tsbg.g_mapped.mapped_implementation`.
- UCLI sequencing is `run` to start stop, enable, `run` to completion stop,
  disable/report, then `run` to M2051 `$finish`.
- PTPX contains black-box, memory-cell, exact-net annotation, exact-leaf
  annotation and `check_power` gates and explicitly marks the result as
  pre-layout standard-cell power with external weight SRAM excluded.

## Required additive successor

Use a fresh namespace. Explicitly put the corresponding mapped netlist in each
filelist; reverse every ascending unpacked bank/lane segment when bridging DC's
flat ports; lock and truthfully label one exact workload; prove 383 preload
cycles; monitor selected mapped outputs for X/Z on every measurement clock;
derive axis/design/UCLI scope/PT strip path/cycles from a single fixed axis;
and load both SSG and TT libraries before applying the TT power corner.

No EDA or license command was run by this review, and `docs/359` remains at
SHA256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
