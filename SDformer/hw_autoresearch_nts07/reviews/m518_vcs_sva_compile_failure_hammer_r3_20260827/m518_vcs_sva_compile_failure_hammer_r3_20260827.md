# M518 r5 VCS SVA compile-failure independent hammer

## Verdict

`DIAGNOSTIC_CONFIRMED__R5_SVA_UNMATCHED_PAREN_FAILURE__R6_STATIC_READMISSION_REQUIRED`

The exact statically authorized r5 runner reached Synopsys VCS compilation and failed closed before simulation. This result is diagnostic only and is not a VCS pass.

## Authorization and tool identity

- The sealed r5 static review is intact (`STATIC_GO`, 97/100, P0=0); its member manifest and outer seal both verify.
- The authorized, expected, and observed runner SHA are identical: `854f152ad23bcc3e353953dee93d0b88f24eab2b4f34261bd88c3c3560a7312a`.
- The output is the authorized default canonical r5 result directory.
- All 24 positive SHA bindings match. The isolated wrong-RTL negative control returned the required code 10 without a positive artifact.
- `vcs_id.txt` identifies Synopsys VCS `V-2023.12-SP1_Full64`.

## Exact failure

VCS parsed the r5 RTL, entered the SVA file, and reported exactly one compile error:

`verif_m518/m518_matched_fixed_t10_atlif_assertions.sv:143`, token `;`, in `ap_dense_start_ownership`.

The assertion statement on lines 137--143 contains 8 opening parentheses and 7 closing parentheses. The ternary branches and the ternary grouping close, but the outer `assert property(` does not. Therefore the minimal repair is exactly one inserted byte: add one `)` immediately before the semicolon on current line 143.

Current suffix:

```systemverilog
(raw_owned_internal[0]&&!raw_ready_internal[0]));
```

Required r6 suffix:

```systemverilog
(raw_owned_internal[0]&&!raw_ready_internal[0])));
```

After this insertion the statement has 8 opening and 8 closing parentheses. No RTL, TB, interface, schedule, campaign, or intended behavior change is justified by this audit.

## Fail-closed topology

- `compile.rc = 255`; the runner exits 20 at its compile gate.
- `RUN_FAILED_OR_INCOMPLETE.txt` is present.
- No `simv`, `sim.log`, `sim.rc`, assertion report, positive receipt, or `RUN_COMPLETE` exists.
- The TB was not parsed and simulation never began. Consequently no assertion, numeric-equivalence, cycle, DC, PPA, energy, speedup, system, or headline claim is admitted.

## Required r6 chain

The author may create r6 only as a new identity:

- SVA: exactly the one-byte insertion above, with forward and reverse byte-for-byte mutation proofs.
- RTL/TB/filelist hashes must remain unchanged.
- New r6 contract, exact-SHA runner, canonical r6 result path, author handoff, and static review request.
- The r6 runner must bind this sealed diagnostic review.
- A new independent static review must authorize the exact r6 runner before any VCS execution.

This review does not directly authorize r6 VCS, DC, PT/PTPX, Formality, or any performance claim.

`docs/359_DATE终局冻结_20260813.md` remains SHA256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
