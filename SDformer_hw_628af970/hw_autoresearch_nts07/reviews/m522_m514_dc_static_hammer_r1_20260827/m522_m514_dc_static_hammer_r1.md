# M522/M514 logic-only DC receipt-blind static hammer r1

Verdict: **STATIC NO-GO; do not execute the current M522 runner.** Score 68/100, P0=2, P1=5, P2=2.

## What is sound

The design-side chain is coherent. All 12 frozen contract inputs match their SHA256 pins. The M514 directed VCS package and independent receipt-blind hammer both pass their member manifests and outer seals. Their claim boundary is preserved: M514 proves directed functional completeness only, not cycle speedup, area, timing, energy, system performance, or paper PPA.

The intended DC flow also contains the right core mechanisms: `-define SYNTHESIS`, the slow target and fast min library, 3 ns clocking, `ZeroWireload`, a precompile TIM-209/OPT-150 gate before flatten/compile, flattening before mapping, five constraint-class checks, setup/hold nonnegative gates, and required netlist/SDC/DDC/SVF outputs. The future receipt makes only one positive claim—additive decoder-support logic cost—and leaves all speedup/system/PPA claims false.

## P0 blockers

1. **The runner cannot pass its tool preflight.** `m522_expect` requires every checked file to be non-symlink, but `/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell` is a symlink to `snps_shell`. The resolved target is a regular file and has the expected `23a410...` SHA, yet the current runner still exits 10. Minimum repair: point at the resolved regular executable, or separately pin the link target and resolved target SHA; do not weaken non-symlink checks for evidence files.

2. **A post-publication seal failure is not quarantined.** The runner moves staging to the canonical path, sets `m522_complete=1`, and only then checks the two seals. Failure at that point leaves a canonical directory and disables the trap. Minimum repair: verify both seals in staging before the atomic move. If post-move checking remains, track the active directory and quarantine canonical on failure; mark complete only after the last check.

Either defect alone is a STATIC NO-GO. The RTL itself is not rejected; the execution/evidence wrapper is.

## P1 closure items

- Make ideal-clock semantics explicit in Tcl/SDC and receipt instead of relying solely on the DC pre-CTS default.
- Scan a combined precompile transcript plus `check_design` and `check_timing`; the whole-log check after compile is too late to be the precompile hard gate.
- Make the receipt self-contained: source/tool/library hashes, SYNTHESIS/flatten/ideal settings, precompile counts, five constraint counts, and strict finite JSON readback.
- Persist the independently reviewed runner SHA and negative-preflight result. A caller-selected self hash by itself does not freeze the reviewed runner.
- Add an exact file/type/link-target topology gate so unexpected unsealed entries cannot coexist with a passing member manifest.

No DC or VCS command was run, no production file was changed, the canonical M522 output was absent, and `docs/359` remained at `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
