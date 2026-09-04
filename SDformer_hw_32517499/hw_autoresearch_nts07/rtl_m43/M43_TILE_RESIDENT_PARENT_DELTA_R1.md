# M43-r1 Tile-Resident Bank-Aware Parent Delta

## Purpose

M43 removes repeated Conv3x3 source work without deleting an output update or
approximating a value.  The primary design reuses a completed partial output
from the left or upper output position within the same 256-feature weight tile.
The previous-timestep parent is retained only as an ablation because its small
additional source-cycle reduction costs a larger state buffer and adds a
temporal dependency.

## Exact arithmetic identity

For a feature tile, let `S` be the current nonzero support and `P` the selected
parent support.  The bias-free signed-INT8 partial convolution is reconstructed
as:

```text
Y_tile(S) = Y_tile(P)
          + sum(i in S \ P) W[i]
          - sum(i in P \ S) W[i]
```

The 27 tile partials are then accumulated.  The parent is a previously computed
tile partial output vector, not an input/source row; therefore the identity does
not mix unrelated destination ownership.  Add and subtract terms are disjoint
and their union is exactly `S xor P`.

## Primary microarchitecture

- Geometry: T10, 768 input/output channels, Conv3x3, 27 feature tiles of at
  most 256 features, eight source banks, and 96 output lanes.
- Each source issue reads one signed-INT8 weight from each active source bank
  and broadcasts its 96-weight row to the output lanes.  Peak issue is eight
  sources by 96 lanes, or 768 signed additions per cycle.
- Bank mapping is `global_feature_index mod 8`; bank row is integer division by
  eight.  The external payload is the independently reviewed M41
  `I_KY_KX_O` layout.
- For each tile row, the selector evaluates zero, left, and up.  It minimizes
  exact finite-bank issue cycles first, signed-delta population second, and uses
  the frozen candidate order for a deterministic tie break.
- The planned RTL has four tagged accumulator contexts.  Descriptor enqueue,
  parent-partial state, final-accumulator state, weight loading, and source-bank
  issue use independent services.  This is a design requirement; the M43-r1
  capacity maximum is not yet an integrated executable-cycle result.
- The signed source descriptor carries add/subtract polarity.  Subtraction is
  applied to the sign-extended INT8 weight before the lane accumulator update.

## Working-set contract

The primary spatial design uses a 49,152-byte double weight buffer, a 5,760-byte
96-lane up-line partial buffer, a 640-byte support line, and 1,408 bytes for four
contexts: 56,960 bytes in total, below the frozen 193,728-byte local residency.
One 96-channel output block across all ten timesteps uses an additional
864,000-byte final-accumulator working buffer at three bytes per value.  This
global buffer requires a real SRAM macro selection and is not included in the
logic-only P8 engine area.

The temporal ablation replaces the up-line partial/support state with one full
spatial frame for 146,560 local bytes in total.  It still fits the frozen local
residency, but it is not the primary architecture because its measured gain is
small relative to the dependency and state cost.

## Frozen source-schedule evidence

The canonical analyzer expands all 40 M40 records, reconciles every valid-pad
Local source-destination pair, selects a parent independently for every
feature-tile/output row, and schedules every signed-delta term through eight
finite source banks for all eight 96-output blocks.  It also binds the four M41
INT8 payloads and the independently reviewed M42 3x headroom gate.

The result may report exact logical-pair reduction, finite-bank source-issue
cycles, utilization, byte traffic, and independent service-capacity gates.  A
zero-visible-overhead compute projection is conditional even when every
individual service has sufficient average capacity.

## Required RTL/VCS milestone

The next implementation must provide:

1. Four or more tagged contexts with fail-closed descriptor and response tags.
2. Separate add/subtract masks, exact signed extension, and 19-bit accumulator
   overflow assertions against the frozen checkpoint trace.
3. A dependency scheduler that respects left/up parent readiness under input,
   weight-response, state-memory, and output backpressure.
4. Source/event conservation and an integer output miter against the M41-r2
   oracle, including zero-delta copies, reset under stall, live-input mutation,
   illegal tags, and queue saturation.
5. Exact cycle counters for command stalls, bank conflicts, parent waits,
   accumulator-memory waits, weight-load waits, and drain.

VCS success is necessary before DC/STA/Formality.  The Synopsys milestone must
use the same 3 ns library/constraint boundary as the P8-L96 and Fixed baselines,
then replace logic-only memory ports with selected SRAM macros for paper PPA.

## Claim boundary

M43-r1 does not admit an integrated cycle count, a measured 3x crossing,
full-network/end-to-end speedup, fixed-point accuracy, SRAM/DRAM timing or
energy, PPA/power, comparison with Prosperity/Phi, a DATE headline, or a
best-paper claim.  Those remain gated on multi-context RTL/VCS, exact integer
output conservation, Synopsys closure, SRAM macros, and full-system replay.
