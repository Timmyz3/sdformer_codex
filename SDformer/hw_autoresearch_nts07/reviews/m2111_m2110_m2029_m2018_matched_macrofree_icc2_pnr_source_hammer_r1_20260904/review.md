# M2111 independent source hammer: M2110 matched macro-free ICC2 P&R

## Verdict

**FAIL — no EDA or license query authorized.** Score: **68/100**; P0/P1/P2 = **3/5/2**.

The overall flow shape is sound: one-shot attempt consumption, two sequential axes, common normalized SDC, exact source-review binding, same-UID collision guard, and a legacy-Milkyway gate before NXTGRD. The current source cannot be released because three fail-closed claims are not actually enforced.

## Blocking attacks

1. `check_routes` returning success is not proof of zero open nets or zero DRC. The Tcl stores only the command status; the parser trusts `route_check_return=1`. A controlled mock containing 999 open nets and 777 DRC violations was accepted.
2. `routed*.spef*` admits `routed.spef_scenario`. A controlled mock with no actual SPEF payload was accepted.
3. `unresolved_count` is a constant zero. The default mismatch query excludes accepted mismatches, per the installed V-2023.12-SP3 command documentation.

The exact minimum repairs are specified in `review.json`. A new identity also needs exact ICC2/lmutil executable pins, complete consumed Milkyway physical-input sealing, per-corner 94-master coverage, and actual—not hardcoded—floorplan/pin/scenario evidence.

## Permitted checks performed

- `bash -n`: PASS
- Python 3.6 compile: PASS
- parser unit tests: 3/3 PASS
- author/contract double seals: PASS
- controlled parser attack: explicit route violations accepted (confirms P0)
- controlled parser attack: scenario manifest without actual SPEF accepted (confirms P0)
- docs/359 SHA remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

No ICC2, StarRC, VCS, GPU, `lmstat`, or other license query was executed.
