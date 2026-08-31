# M480 dynamic-BN exact materialization-elision CPU DSE

## Verdict

`GO` as mandatory fair implementation-baseline hygiene; `NO_GO` as a standalone
DATE novelty claim.  The fused schedule removes only the normalized-tensor
write/read pair.  It still charges every raw capture, every current-batch moment
barrier, all 22,080 coefficient pairs, and one complete raw replay.

No RTL was changed.  This result is not a module or system admission.

## Matched result

- Scope: H67/Motion ep35, 12 FFNs and 24 dynamic-BN phases.
- DSE: stored activation width 16/24/32 bit, 32/64/128 B/cycle, one 1R1W
  store, and coefficient service either serialized or overlapped with replay.
- Coverage: 18 summary points and 432 independently charged phase points.
- BN-local cycle speedup: 1.492602x--1.499999x; all 18 points pass the
  predeclared 1.25x implementation-baseline gate.
- Local buffer traffic: exactly 2.000x lower at every point because four
  payload passes become two.
- Peak raw-retention capacity: unchanged (`1.000x`).  At the Q24 point, the
  largest phase still retains 221,184,000 bytes across its barrier.
- Pessimistic serial addition to the M159 fixed FFN ledger gives
  1.031194x--1.173811x; 14/18 configurations exceed 1.05x.  This is not a
  system-speedup claim.

The reference Q24, 64 B/cycle, overlap point is 61,568,856 versus 41,048,856
BN-local cycles (1.499892x), with 5,253,120,000 versus 2,626,560,000 buffer
bytes (2.000x).  Its serial-M159 accounted ratio is 1.083268x.

## Why this is not a standalone contribution

M159/M161 already define the strong dense baseline as updating moments during
the raw write, retaining the global barrier, then reading raw once and
streaming normalized values directly to the consumer.  Therefore, publishing
a weaker baseline that materializes the normalized tensor would inflate the
gain.  M480 makes that comparison explicit and prevents this accounting error;
it does not establish a new sparse-BN mechanism.

## Claim boundary and next gate

The cycle model assumes a bus-wide affine/consumer datapath at each scanned
bandwidth.  Runtime-affine arithmetic, fixed-point accuracy, SRAM macros,
VCS/DC/STA and energy are unimplemented/unpriced.  Before any RTL promotion,
one selected width needs an address-bearing 1R1W schedule and a numerical miter
for the runtime affine path.  A later BN proposal must beat this fused baseline
while retaining raw storage, barrier and replay charges.

Run the receipt-blind audit with:

```bash
python3 audit_m480_dse.py
```
