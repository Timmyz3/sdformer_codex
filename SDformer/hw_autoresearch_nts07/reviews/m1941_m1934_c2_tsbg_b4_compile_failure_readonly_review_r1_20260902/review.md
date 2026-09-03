# M1941 — M1934 TSBG VCS compile-failure read-only review

## Verdict

**PASS diagnosis, 99/100; P0/P1/P2 = 0/0/0.** M1934 remains permanently `FAILED_OR_INCOMPLETE_DO_NOT_CITE`. This review authorizes only an additive successor TB source; it ran no EDA and authorizes no VCS attempt.

## Unique root cause

The sealed VCS `V-2023.12-SP1_Full64` compile log parses all four source objects and reports exactly one error:

`Error-[ICPD] Illegal combination of drivers` on `tb_cycle`.

The exact ownership conflict is:

- declaration at M1924 line 193;
- clocked driver in `always_ff` at line 226;
- directed initialization `tb_cycle = 0` at line 589.

IEEE 1800 gives `always_ff` an exclusive procedural-writer rule. The sufficient minimal repair is therefore exactly:

```systemverilog
-    always_ff @(posedge clk_core) begin
+    always @(posedge clk_core) begin
```

The `posedge clk_core` event control, reset/counter body, and nonblocking assignments must remain unchanged.

## Other-driver audit

No second illegal `always_ff` shared-writer conflict was found. The scalar-bank state process at line 81 has exclusive ownership. M1924 already converted the scoreboard/statistics process to an ordinary posedge `always` at line 500; its initialization and phase clearing by the single directed `initial` thread are legal testbench shared ownership. The clear task is used at initial/idle reset boundaries in this workload, so it does not identify another compile blocker.

The compile log's `context` keyword diagnostics are warnings, not the failure root. No RTL, adapter, SVA, arithmetic check, attack, fatal check, or PASS token needs to change for this repair.

## Integrity and claim boundary

- M1934 failure directory and consumed-attempt directory both pass inner and outer SHA checks.
- Compile log SHA256: `b645a035dce59129d4372e8d458a3a5fa814de1e930196e80ce5e4c97d70a104`.
- M1924 TB SHA256: `df99e881e62ef2172f8658d36384d49640dcd86c8785e44cd7fbcfea97f264f1`.
- `docs/359` remains SHA256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
- There is no `simv.log` and no valid simulation result. The consumed-attempt receipt is budget accounting, not evidence that simulation executed.

## Next gate

Create an additive successor TB copied from the exact M1924 identity and change only the line-226 process header, with an explanatory comment. A different-author source hammer must prove that one-line semantic delta before a fresh runner, release, or EDA attempt.
