# M514 C2-D static hammer review r2

Date: 2026-08-27  
Scope: independent static RTL/TB review only. No VCS, DC, DSE, GPU job, lint,
or simulator was run. `docs/359` was not modified.

## Verdict

`R1_P0_FIXED__NEW_UPPER_BOUND_P0__FIX_BEFORE_EXACT_SHA_RUNNER`.

The r1 fault/stall defect is correctly repaired. `fault_q` now locks
`event_ready` while `tap_valid` remains a function only of accepted pending
work, so an illegal successor cannot retract an advertised tap. The added
fault-overlap test does force `tap_ready=0` at the attack edge and its new
legal interior event correctly raises the total from 34 to 43 taps.

One new P0 remains in the just-added exact size upper bound. The legality
logic admits `input_height/width == 2^(COORD_BITS-1)`, which is mathematically
correct because the largest output coordinate is still representable. The
simulation assertion, however, computes `(input_height_q << 1)` at only
`COORD_BITS` width. At the admitted equality edge this wraps to zero, so every
legal tap violates the destination-bound assertion. The current TB uses only
sizes 3 and 4 and cannot expose it.

Do not seal the current source/TB hashes into the final exact-SHA VCS runner.
Widen the assertion comparison and add an equality-edge vector first. No
other P0 was found.

## Static identity

| Item | SHA256 | Lines |
|---|---|---:|
| `rtl_m514/m514_c2_convtranspose_k3s2_polyphase_address_mapper.sv` | `7543a25c81f2bfcbf2768f53f4bcba713c139aebae4713e2225a0d2395464511` | 223 |
| `dc_handoff/tb/tb_m514_c2_convtranspose_k3s2_polyphase_address_mapper.sv` | `10392f182c0be2a6f7298cbc61a464541a11dec8ea9f4a7fa8e4c892506d7458` | 266 |

## r1 P0 closure

The repaired output contract is coherent:

- `tap_valid = busy_q && selected_found` keeps all previously accepted work
  visible through a fault;
- `event_ready = !fault_q && event_capacity && event_legal` permanently locks
  new work after a protocol attack;
- accepted taps continue clearing `pending_q`; the final accepted tap clears
  `busy_q` even while fault is sticky;
- an invalid same-edge successor cannot replace a retiring tap because
  `event_ready=0`, while a legal successor still replaces it through the later
  nonblocking assignments in the `event_accept` branch.

The overlap driver is effective. `force_stall` is asserted before a following
negedge sets `tap_ready<=0`; the loop does not proceed until a posedge observes
`tap_valid && !tap_ready`; the illegal event is then driven on the next
negedge. Thus the attack edge necessarily overlaps an advertised stalled tap.
Deasserting `force_stall` on a negedge can race only between zero and one extra
drain stall; it cannot invalidate the attack or tuple count.

## P0-01: admitted size equality wraps the assertion bound

The new legality condition admits exactly one MSB-set dimension:

```text
I = 1 << (COORD_BITS-1)
```

For that size, the last source `s=I-1` and `k=2` produce
`d=2s-1+k=2I-1=2^COORD_BITS-1`, which fits the output coordinate port. The
datapath is therefore correct. But SystemVerilog shift result width is the
left operand width, so

```systemverilog
input_height_q << 1
```

is still `COORD_BITS` wide and evaluates to zero at `I=2^(COORD_BITS-1)`.
Lines 213--214 consequently assert `tap_destination < 0` and fail.

Required fix:

```systemverilog
assert ({1'b0, tap_destination_y}
        < ({1'b0, input_height_q} << 1));
assert ({1'b0, tap_destination_x}
        < ({1'b0, input_width_q} << 1));
```

Equivalent explicitly declared `COORD_BITS+1` output-extent wires are also
acceptable. Add a directed event with TB `COORD_BITS=6`, size 32, source
`(31,31)` and verify all nine taps, including destination `(63,63)`. A second
top-left size-32 event should retain fanout four.

## Other audited points

### PyTorch geometry, fanout and phase

The formula remains `d=2s-1+k` for K3/S2/P1/OP1. Top-left/top-only/left-only/
interior fanout remains `4/6/6/9`; bottom and right do not clip. Slot order
remains destination phase `11/10/01/00` with interior counts `4/2/2/1`.

The first five TB events contribute `4+6+6+9+9=34`; the added interior event
contributes nine, so total 43 is correct. Its additional phase counts are
`00/01/10/11 = 1/2/2/4`; final totals should be `6/10/10/17`.

### Same-edge replacement

`event_capacity` becomes true only when the current final tap is actually
accepted. When a legal successor is also valid, retirement executes first and
the later event-accept assignments replace `busy_q`, `pending_q`, and metadata.
There is no tap reorder or empty output cycle introduced by the state update.

### Ready/valid and data-dependent ready

`event_ready` depends on event dimensions and coordinates. This is legal for
a ready/valid sink provided the source asserts and holds a complete legal
payload independently of ready. The TB driver does so: it sets payload and
valid together on a negedge and holds them through acceptance. There is no
ready-to-valid combinational loop inside M514.

This dependency should be explicit in the interface contract. Add an SVA that
the input payload is stable while `event_valid && !event_ready`; otherwise a
non-conforming upstream can change legality and ready combinationally.

### SVA/driver scheduling

No attack-invalidating active-region race was found. Event payload and valid
change on negedges. `force_stall` changes on a posedge but affects `tap_ready`
only through the following negedge process. Reset deassertion occurs on a
posedge, but four earlier reset edges have already initialized every state;
moving reset release to a negedge would still be cleaner. The drain loop may
wait one harmless extra edge because monitor counters and `busy_q` update in
different scheduling regions.

## P1 findings

1. **Common-mode oracle remains.** `slot_valid` and `slot_to_kernel` mirror
   the RTL mask/case table. Replace or supplement them with signed nested
   `ky/kx` enumeration, coordinate filtering, and independent phase sorting.
2. **Upper-bound coverage is absent.** Add size 32 vectors as described above,
   plus size 1 and bottom/right edge vectors.
3. **Parameter fail-closed is simulation-only.** The legality `initial` block
   is under `ifndef SYNTHESIS`; the fixed H67 parameter contract should also
   be pinned by wrapper/contract and exact SHA rather than relying on this
   check.
4. **Input stability and exact replacement SVA remain absent.** Add payload
   stability under backpressure and a next-cycle successor tag/first-tap
   property. The tuple scoreboard and nonzero replacement counter are useful
   but weaker.
5. **TB scheduling style.** Release reset and toggle `force_stall` on negedges
   or a clocking block to remove benign active-region ordering dependence.

## Runner admission

Current r2 hashes: `NO_GO_FINAL_EXACT_SHA_RUNNER` due to P0-01.

After widening both bound assertions and adding the equality-edge test:
`GO_EXACT_SHA_VCS_RUNNER`. The receipt must remain scoped to standalone
address/handshake completeness and must not admit cycle speedup, system
speedup, numerical ConvTranspose output equivalence, energy, PPA, or a DATE
headline.

