# M1331 — C2 production SAIF/PTPX read-only gap audit

## Verdict

`NO_GO_DIRECT_PRODUCTION_SAIF_PTPX__MAPPED_ACTIVITY_INPUT_NOT_ADMITTED`

The current C2 evidence cannot directly enter production PrimeTime PX.  This
is an input-admission failure, not merely a missing shell command:

- M1046 proved that the old three-axis mapped netlists compile and that the
  tiny UCLI power mechanism can emit a 2,106-byte SAIF, but its first K1
  production case watchdogs after header acceptance.  It completed zero
  mapped cases and emitted zero production SAIF files.
- M1080 resynthesized a K1-only 168-bit FIFO-payload reset repair.  The reset
  payload survived synthesis, yet mapped case 0 still watchdogs at the same
  boundary.  The independent audit classifies the remaining gap as unreset
  control/payload-valid isolation or synthesis-specific X reconvergence.
- M1293/M1304 later closes directed semantic observability at RTL, but only for
  diagnostic K1.  It contains neither K8 nor equal-bandwidth K1x8, and admits
  no mapped functionality, SAIF, power, or energy.
- A generic `run_ptpx.tcl` exists and contains `read_saif`, `update_power`, and
  `report_power`.  It cannot manufacture the missing activity evidence and is
  not a C2-specific annotation/energy admission contract.

Consequently, launching PTPX now would either have no production SAIF input or
would substitute tiny/RTL-only activity for the K8-versus-K1x8 mapped
comparison.  Both would weaken the frozen coverage boundary.

## Minimum missing artifacts

1. An additive mapped-replay repair for the first failing valid/control cone,
   without `initreg` dependence, applied consistently to headline K8 and
   equal-bandwidth K1x8.  K1 may remain diagnostic.
2. Fresh same-source/same-constraint mapped netlists and SDCs for both headline
   axes after that repair, with exact SHA identities and one frozen power
   corner.
3. Fresh mapped VCS replay of the same five workloads on each headline axis,
   retaining numeric, request/response tuple, weight, unknown, protocol,
   cycle-window, and major-cone gates.
4. Ten DUT-only production SAIF files (fifteen only if K1 is retained), with
   reset/preheader/post-token idle excluded and duration exactly equal to
   measured cycles times 3 ns.
5. A C2-specific PrimeTime PX Tcl/runner pinning netlist, SDC, library and
   operating condition, strip path, annotation coverage, `update_power`,
   `report_power`, and energy = average power × measured activity duration.
6. A fresh exact-SHA single-job namespace with independent source/release
   hammer and fail-closed result extraction.  M1046 and M1080 are consumed and
   must not be retried.

## Claim boundary

This audit launched no EDA or GPU work and does not change C2's admitted
performance claims.  Production SAIF, PTPX power, energy, fair
K8-versus-K1x8 energy, system energy, and paper-ready PPA remain false.
`docs/359` remains frozen at
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
