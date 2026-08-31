# M979 — C2 three-axis mapped-gate replay and SAIF source receipt

## Verdict

`PASS_M979_MAPPED_GATE_SAIF_SOURCE__NO_EDA_EXECUTED`, 98/100, P0=0, P1=2.

M979 closes the source boundary requested by M974: a common mapped-gate replay
testbench, DUT-only UCLI capture, future sequential runner, and a validator for
15 independent SAIF files (K1/K8/K1x8 × five frozen cases). No VCS, PT, PTPX,
DC, GPU, or remote task was launched.

## What is gated

- K8 cycles must be `51/131/486/1231/14`; K1x8 must be
  `53/133/499/1246/14`. K1 remains diagnostic.
- Every replay must have zero numeric, request/response tuple, weight,
  accepted-X, and protocol mismatches.
- Capture begins at accepted header and ends one full clock after accepted
  token completion. Reset, pre-header idle, and inter-case idle are excluded.
- Each SAIF must be DUT-only, have duration `cycles × 3 ns`, all `TX=0`, and
  `rst_core TC=0`.
- Cases 0–3 require nonzero clock, header/raw, memory, result/accumulator, and
  token-done cones. Frozen case 4 has no source events, so memory TC may be zero
  and is explicitly reported; forcing it nonzero would manufacture activity.

## Static evidence and boundary

Seven directed tests passed, including negative duration, cycle-anchor, TX,
scope, reset and cone checks. The future M993 runner stops before any tool call
unless independently sealed M990, M991 and M992 authorities are caller-pinned.
After authorization, the canonical attempt-directory `mkdir` is itself the
irreversible consumption point; an interrupt during its sealing leaves the
canonical directory in place and blocks retry. Post-consumption failures are
sealed before quarantine, and seal/move failure retains the original work.

This receipt does **not** prove the flattened mapped-netlist port orientation,
real SAIF syntax/coverage, power, or energy. Those remain fail-closed until the
fresh mapped-gate replay executes. `docs/359` remains unchanged at
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
