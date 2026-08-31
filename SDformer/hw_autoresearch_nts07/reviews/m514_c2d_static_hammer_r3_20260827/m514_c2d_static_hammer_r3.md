# M514 C2-D final static hammer review r3

Date: 2026-08-27  
Scope: final independent static RTL/TB review only. No VCS, DC, DSE, GPU,
lint, or simulator was run. `docs/359` was not modified.

## Final verdict

`P0_CLOSED__GO_EXACT_SHA_VCS_RUNNER`.

Both prior P0 findings are closed:

1. sticky protocol fault blocks only new events; already accepted pending taps
   retain `tap_valid` and drain under ordinary ready/valid semantics;
2. destination-bound assertions now widen both operands to
   `COORD_BITS+1`, so the exact legal size upper bound no longer wraps.

No new P0 was found. The exact-SHA VCS runner may now be written against the
source identities below. Its admission must remain standalone directed
address/handshake completeness; it does not admit speedup, numerical
ConvTranspose output equivalence, energy, PPA, or a DATE headline.

## Static identity

| Item | SHA256 | Lines |
|---|---|---:|
| `rtl_m514/m514_c2_convtranspose_k3s2_polyphase_address_mapper.sv` | `90c44fc9bde839c3cf325ccc8f45c153bf5d30e18de7f39b26d7a4456b017a9a` | 227 |
| `dc_handoff/tb/tb_m514_c2_convtranspose_k3s2_polyphase_address_mapper.sv` | `6c283bf94d6933e6aa866428f63d6a8b9a2066da2deb39220301f781ec3df47a` | 268 |

## Upper-bound proof

With TB `COORD_BITS=6`, the newly exercised maximum legal input dimension is
32. The legal source `(31,31)` emits all nine taps:

```text
dy = 2*31 - 1 + ky = 61 + ky
dx = 2*31 - 1 + kx = 61 + kx
```

so each coordinate is 61, 62, or 63, and `(ky,kx)=(2,2)` reaches the required
representable maximum `(63,63)`. The widened assertion compares a seven-bit
destination against

```text
{1'b0, 6'd32} << 1 = 7'd64
```

and therefore accepts every legal coordinate while still excluding 64.

The successor attack uses `source_y=32`, `height=32`; both are represented
exactly in six bits and `32 < 32` is false. Its dimensions remain legal, but
its source coordinate is deterministically illegal, so it cannot be accepted
and must set sticky fault.

## Count and phase closure

The first five events remain:

```text
4 + 6 + 6 + 9 + 9 = 34 taps
```

The size-32 bottom-right interior event contributes nine, giving total 43.
An interior event contributes phase-bank counts `00/01/10/11 = 1/2/2/4`.
Adding these to the r2 first-batch counts gives final totals:

```text
00/01/10/11 = 6/10/10/17
```

The tuple scoreboard checks each phase, kernel coordinate, destination,
event-last and stream-last value. Thus observed count 43 with no tuple fatal
also closes those phase totals even though the TB display does not separately
assert the four aggregate numbers.

## Protocol and scheduling closure

- `event_ready` remains payload-dependent. This is legal because the driver
  presents a complete payload with valid independently of ready and holds it
  through acceptance. M514 contains no ready-to-valid combinational loop.
- During legal same-edge replacement, the retiring tap is accepted before the
  later nonblocking event assignments reload pending state and metadata; no
  old tap is reordered or lost.
- During the forced-stall attack, `force_stall` reaches the negedge ready
  driver before the loop is allowed to proceed. The attack is not issued until
  a posedge has observed `tap_valid && !tap_ready`.
- Once fault is set, `event_ready` stays low while pending taps continue to
  accept and clear. The final tap clears busy; fault remains observable.
- Reset-release and force-stall deassertion retain only the benign scheduling
  style noted in r2; neither changes the attack edge, tuple multiset, or final
  state.

## Non-blocking P1 items

1. The expected oracle still mirrors the RTL slot table. An independent
   signed nested-`ky/kx` oracle would improve future regression strength.
2. Add explicit input-payload stability and successor-first-tap SVA when this
   adapter is integrated with a real upstream source.
3. Pin legal parameters in the wrapper/contract because the simulation-only
   elaboration check is not a synthesized protection mechanism.
4. A clocking block or negedge-only reset/control driver would remove benign
   active-region style dependence.

These items do not block the directed exact-SHA runner because the fixed H67
geometry and the current TB behavior have been independently closed here.

## Runner admission boundary

Allowed after this review:

- exact-SHA VCS compile/run runner;
- directed K3/S2/P1/OP1 address, fanout, phase, replacement, stall and
  fault-drain receipt;
- receipt label `DIRECTED_FUNCTIONAL_COMPLETENESS_ONLY`.

Not allowed from M514 alone:

- decoder or system speedup;
- PyTorch finite-width accumulated-output equality;
- energy, area, timing or paper-PPA claims;
- standalone DATE novelty/headline admission.

