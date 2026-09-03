# M1942 — M1941 TSBG all-driver-legal TB source hammer

## Verdict

**PASS source-only, 99/100; P0/P1/P2 = 0/0/0.** The additive M1942 testbench implements exactly the repair authorized by the sealed M1941 diagnosis. This review authorizes fresh runner source authoring only; it ran no VCS, simulation, license query, DC, PT, or other EDA.

## Parent binding and exact delta

The required M1941 review is present at SHA256 `33e3456b1b4855c86faa4d2ff2dd34877b94a62b0e8d9924fc2c07577b5d5241`; its inner and outer seals verify. M1941 permits only the `tb_cycle` process header repair.

Relative to frozen M1924, M1942 contains one semantic change:

```systemverilog
-    always_ff @(posedge clk_core) begin
+    always @(posedge clk_core) begin
```

The `posedge clk_core` event, reset/counter body, and nonblocking assignments are unchanged. Three adjacent explanatory comment lines are the only other additions. Deleting those comments and reversing the process keyword yields SHA256 `df99e881e62ef2172f8658d36384d49640dcd86c8785e44cd7fbcfea97f264f1`, exactly the frozen M1924 TB identity.

The filelist changes only its final TB pathname. The adapter, frontend RTL, and SVA remain respectively:

- `cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156`
- `8524f6a7a6d09e1aaab55ee91515bd1fce9ea57fa2a478a9817f637685299a05`
- `e5519a75c14d68dfc273c3a7e9930560fa8a3c7779ab5ed7f22f294a14be58c2`

The 31 fatal checks, one assertion bind, and unique PASS token are unchanged.

## Driver-ownership audit

The top testbench now has no `always_ff` process. `tb_cycle`, which is initialized in the directed `initial` thread, is updated by an ordinary posedge `always`. The scoreboard/statistics process remains the M1924 ordinary posedge `always`, so its deliberate initialization/phase-clear sharing is legal testbench ownership. No shared-initialized top-level variable remains under `always_ff`.

The one remaining `always_ff` in the file belongs to the separate `m1880_directed_scalar_bank_memory` module. That module has no `initial` block, and its state/counters have a single procedural owner. No remaining illegal `always_ff` ownership was found.

## Integrity and boundary

- M1942 TB SHA256: `fc2340ec6cf23a3537fc63b829c26e32d2dec847dd6aa73a5aa9178f6d686a7c`.
- M1942 filelist SHA256: `e08804894411e7c97454ecb7d06912c250db2c732a1c99a73e705c6fdd9ca8bc`.
- `docs/359` remains SHA256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
- No compile or simulation result exists from this review. It admits no speedup, functionality, PPA, energy, system, or paper claim.

## Next gate

A fresh runner may be authored against these exact identities. A different-author runner source hammer and a separate launch-release audit must pass before one fresh VCS attempt. This review does not authorize that attempt.
