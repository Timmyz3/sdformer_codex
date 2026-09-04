# M514 C2-D static hammer review r1

Date: 2026-08-27  
Scope: static RTL/TB review only. No VCS, DC, DSE, GPU job, or open-source
simulation was run. `docs/359` was not modified.

## Verdict

`P0_FIX_REQUIRED_BEFORE_EXACT_SHA_VCS_RUNNER`.

The fixed H67 mapping itself is correct: PyTorch `ConvTranspose2d` with
`K3/S2/P1/output_padding1/dilation1` scatters source `(sy,sx)` and kernel
`(ky,kx)` to

```text
dy = 2*sy - 1 + ky
dx = 2*sx - 1 + kx
```

and produces an output extent of `2H x 2W`. Only `ky=0` at `sy=0` and
`kx=0` at `sx=0` are clipped; bottom and right taps remain legal because
output padding is one. Thus top-left/top-only/left-only/interior fanout is
exactly `4/6/6/9`. The RTL implements that mapping.

There is, however, one real ready/valid P0. An illegal successor event can set
`fault_q` while an already accepted tap is stalled. Because `tap_valid` is
masked by `!fault_q`, the module retracts that tap on the next cycle and
violates its own unconditional `p_tap_stable_under_stall` contract. The
current TB attacks the input only while idle, so it cannot expose this case.

Do not seal the current RTL/TB hashes into a final VCS runner. Fix P0-01, add
the directed overlap attack, then write the exact-SHA runner against the new
hashes. After that repair, the runner may be admitted only as
`DIRECTED_FUNCTIONAL_COMPLETENESS`; M514 remains a C2 decoder support adapter,
not a speedup result.

## Static identity

| Item | SHA256 | Lines |
|---|---|---:|
| `rtl_m514/m514_c2_convtranspose_k3s2_polyphase_address_mapper.sv` | `f88abc8ca50f0d40f6f923e4bc0939f75616dd7d9e5b7e882b1c741fd2265b35` | 209 |
| `dc_handoff/tb/tb_m514_c2_convtranspose_k3s2_polyphase_address_mapper.sv` | `52e1c6ba0cca1b478d6e285567168283be4cc727c75fcbd5f5d2564c4678f931` | 253 |

Both inputs were untracked at review time. This review does not imply git or
paper identity admission.

## Independent mapping audit

### Coordinate and boundary proof

For output size `O=(I-1)*2-2+2+1+1=2I`, the maximum destination from the last
source and last kernel tap is
`2*(I-1)-1+2=2I-1`, which is legal. The minimum is `-1` only for source zero
and kernel tap zero. Consequently:

| Source class | Legal ky count | Legal kx count | Fanout |
|---|---:|---:|---:|
| `(sy,sx)=(0,0)` | 2 | 2 | 4 |
| `sy=0, sx>0` | 2 | 3 | 6 |
| `sy>0, sx=0` | 3 | 2 | 6 |
| `sy>0, sx>0` | 3 | 3 | 9 |

RTL mask slots 0/1/4 are gated by `sy!=0`; 0/2/6 are gated by `sx!=0`;
3/5/7/8 are unconditional. This is equivalent to the table.

### Phase-major order

Because `2*sy-1` and `2*sx-1` are odd, even kernel indices produce odd
destination parity while kernel index one produces even parity. Slots are:

1. 0--3: `(ky,kx)` even/even -> destination odd/odd, four taps;
2. 4--5: even/one -> odd/even, two taps;
3. 6--7: one/even -> even/odd, two taps;
4. 8: one/one -> even/even, one tap.

Therefore interior order is exactly `4/2/2/1`, and
`tap_phase_bank={dy[0],dx[0]}` is consistent. Boundary clipping preserves
phase-major ordering among the remaining slots.

### TB total

The five directed sources have fanouts `4 + 6 + 6 + 9 + 9 = 34`; expected
34 is correct. The independently derived bank totals, in TB index order
`00/01/10/11`, are `5/8/8/13`. Same-edge replacement is structurally legal:
`event_ready` rises only when the old final tap is accepted, and the later
`event_accept` nonblocking assignments intentionally override retirement to
load the successor mask and metadata.

## Findings

### P0-01: protocol fault retracts a stalled, already accepted tap

Evidence:

- `tap_valid = busy_q && selected_found && !fault_q`;
- any `event_valid && !event_legal` sets sticky `fault_q`, regardless of the
  current output stall;
- the SVA requires every `tap_valid && !tap_ready` to remain valid and stable
  on the following cycle.

Counterexample: accept a legal event, stall any emitted tap, and present an
out-of-range successor with `event_valid=1`. The antecedent of the stall SVA
is true while `fault_q=0`; `fault_q` becomes one after that edge; next cycle
`tap_valid=0`, so the SVA fails and the accepted tap is lost. `busy_q` and the
pending mask also remain stranded until reset.

Required repair:

1. Make sticky fault stop only new event acceptance; do not mask the output of
   an already accepted event. The simplest contract is
   `tap_valid = busy_q && selected_found`, while `event_ready` remains gated
   by `!fault_q`. Existing pending work then drains exactly once.
2. Add a directed case that stalls a tap, injects an illegal successor, checks
   `protocol_error`, keeps the tap tuple stable, drains all accepted taps, and
   proves that no later event is accepted before reset.
3. Add an assertion that a protocol fault is sticky and that faulted state
   never raises `event_ready`.

An alternative immediate-abort policy would need an explicit non-standard
output protocol and an assertion exception. It is not recommended because it
breaks ordinary ready/valid semantics.

### P1-01: the expected model shares the RTL slot table

`slot_valid` and `slot_to_kernel` reproduce the DUT mask and case statement.
The tuple scoreboard is useful, but this creates a common-mode blind spot.

Recommended repair: generate expected taps independently by nested
`ky=0..2`, `kx=0..2`, compute signed `dy/dx`, filter by
`0<=dy<2H && 0<=dx<2W`, then sort by destination parity class
`11,10,01,00` and a documented within-class order. Do not use DUT slot IDs in
the oracle.

### P1-02: parameter and coordinate-range contract is implicit

The design intentionally rejects any dimension with the coordinate MSB set.
This safely covers frozen H67 dimensions up to `120x160`, but it also rejects
the exactly representable edge `I=2^(COORD_BITS-1)`, whose largest coordinate
is `2^COORD_BITS-1`. No elaboration-time parameter legality check documents
this choice.

Recommended repair: freeze and assert positive tag/channel/time widths,
`COORD_BITS>=2`, and the supported maximum input dimension. Either document
the conservative `<2^(COORD_BITS-1)` contract or widen extent arithmetic and
support the equality edge explicitly. Add H67 maximum-shape directed vectors.

### P1-03: input and replacement protocol coverage is incomplete

The TB proves at least one stall and one replacement, but does not assert
upstream payload stability while `event_valid && !event_ready`, nor the exact
first tuple/no-bubble behavior after replacement.

Recommended repair:

- assert stable event payload under backpressure;
- assert a legal same-edge replacement causes the successor's first tap on
  the next available output cycle with its tag and metadata;
- cover/reset-check `H=W=1`, bottom-right, zero dimension, x-range, y-range,
  oversized dimension, fault while busy, and reset while busy.

### P1-04: internal invariants are not asserted

Add simulation-only invariants for `busy_q -> pending_q!=0`, selected onehot
being onehot-or-zero and a subset of pending, `tap_last_for_event` iff the
selected bit is the sole pending bit, destination bounds, and sticky fault.
These make later exact-SHA regression more diagnostic without changing the
synthesized design.

## Runner admission

Current hashes: `NO_GO_FINAL_RUNNER` because P0-01 can violate the included
stall assertion.

After P0-01 and its overlap test are fixed: `GO_EXACT_SHA_VCS_RUNNER`, subject
to all of the following labels:

- standalone directed RTL/SVA completeness only;
- exact K3/S2/P1/OP1 address mapping and handshake for tested geometry;
- no cycle speedup, system speedup, energy, PPA, PyTorch numerical-output
  equivalence, or DATE-headline admission;
- source, TB, filelist, runner, library, tool version, compile command, and run
  command must all be pinned in the receipt.

