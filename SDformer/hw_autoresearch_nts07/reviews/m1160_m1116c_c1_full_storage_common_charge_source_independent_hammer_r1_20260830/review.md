# M1160 — M1116C common-charge source independent hammer

Verdict: **the accounting boundary passes, but the wrapper protocol fails closed before VCS/DC.** The exact byte map and the DC claim labels may be reused; the present wrapper must not be simulated or synthesized as an admitted successor.

## What passed

- All author identities and nested seals resolve. Frozen M935 remains `e834b524...`; the parent wrapper remains `8fd008a3...`; `docs/359` remains `dedde7ce...`.
- The four ranges are contiguous and nonoverlapping: internal parent `18,432 B`, external psum `122,880 B`, external weight `49,152 B`, and external model-only metadata/reserve `24,448 B`. They total `214,912 B`, leaving `30,848 B` below `240 KiB`.
- Exactly nine live parent macros are inside the candidate. No wrapper dummy/tied macro exists. External physical macro count is zero.
- All 59 frozen M935 ports are connected exactly once.
- The filelist is synthesis-only, the SDC has zero exceptions, and the Tcl explicitly reports external area as `UNMODELED_EXCLUDED` and full-214,912-B total area as `NOT_ADMITTED`. It does not mislabel the external common charge as an internal physical macro result.

## P0 — atomic fire was implemented with an unsafe ready-to-valid loop

For a first beat, the wrapper implements:

```text
weight_valid = issue_valid && psum_ready
psum_valid   = issue_valid && weight_ready
```

If both external `ready` inputs are already stable, the two handshakes do fire together. That truth-table property is narrower than a valid ready/valid protocol. Each output `valid` is combinationally driven by the other sink's `ready`, while the contract never requires either sink's `ready` to be independent of its `valid`. A legal valid-dependent-ready environment can create a combinational loop with no fixed point. Constant-ready static tests would not expose it.

## P1 — one outstanding transaction changes the performance recurrence

`service_outstanding_q` blocks every new request until the joined response is consumed. With a one-cycle service response, request/data accepts recur at minimum II=2. Frozen M935 itself advances to a new beat after each accepted issue, so this wrapper is not a transparent realization of the existing raw CPU schedule. This is not a numerical failure because M1116C admits no cycles; it is a hard reason that no later result may inherit `1.7591725402×` without a new matched joint replay.

The requested sticky-fault concern is not the present bug: M935 holds `issue_request_valid` until the joined `issue_data` acceptance, so `service_outstanding_q && !issue_request_valid` does not fire on normal accepted-request flow. It remains a cancellation/fault guard.

## Exact next authorization

Only an additive source-only protocol repair is authorized. The repair must either use a single explicitly atomic bundled service, or latch one request and independently track weight/psum acceptance without duplicate issue. It must freeze ready/valid dependency rules, response latency/hold and early-response behavior, outstanding depth, reset/cancellation, then add TB/SVA attacks for both partial-accept orders, skewed responses, stalls, backpressure, reset and spurious responses. A fresh different-author source hammer is required afterward.

VCS, DC, PT, Formality and PTPX remain unauthorized. No source was changed by this review, no tool was run, and no cycle, speedup, PPA, energy or paper-ready claim was created.
