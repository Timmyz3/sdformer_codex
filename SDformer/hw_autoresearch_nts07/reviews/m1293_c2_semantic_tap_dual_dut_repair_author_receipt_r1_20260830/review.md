# M1293 — C2 semantic-tap/dual-DUT additive repair author receipt

## Verdict

`PASS_M1293_SOURCE_REPAIR__DIFFERENT_AUTHOR_HAMMER_REQUIRED__NO_EXECUTION_AUTHORIZED`

M1293 closes the four M1287 P1 source findings in a new additive endpoint/TB,
checker, test and contract namespace.  Every M1279 RTL source remains frozen at
its prior SHA.  The source checker passes and all twelve mutation tests pass.

This is source evidence only.  No SystemVerilog compiler or elaborator, VCS,
DC, PT, PTPX, SAIF, GPU or remote task ran.  Functional reachability and
comparison are now mandatory **in the testbench source**, but are not yet an
executed simulation result.  No fresh RTL release exists until a different-
author receipt-blind hammer passes.

## Repair 1 — endpoint participation and class-aware completion

The new TB has two systems receiving identical header/raw stimulus.  PASS is
inside one guarded 256-cycle atomic block and requires all of the following:

1. both raw packets were accepted;
2. both endpoints accepted at least one bank request;
3. both DUTs produced at least one accepted result;
4. both DUTs produced at least one accepted token-done;
5. request, result and token counts match and all class mismatch counters are
   zero;
6. the first result follows the first bank request and token-done is not before
   that request;
7. qualified endpoint/fault/unknown checks remain clean.

The transaction-class comparison is not a count-only check:

- request class compares the accepted bank vector and per-bank slot;
- result class compares tag, output block, slice, last and all sixteen Acc24
  lanes;
- token-done class compares tag and had-event.

If the qualified endpoint is never reached, result never returns, token-done
never returns, or either DUT differs in a transaction class, only `$fatal` is
reachable.  There is no PASS classification for an unparticipating endpoint.

The complete `transaction_class_compare` block has comment-insensitive token SHA
`833c00145a86b41ea09ca9b405d71d6d284bb4b13dc2ef3563be275b3cffbd9e`.
The complete atomic reachability/PASS block has token SHA
`e651e1cf23cb88abc6fa34172e2aa5442e253ef9e950d186849926ca94a1a539`.

## Repair 2 — exact closed contract and claims

The M1293 checker requires the exact contract top-level key set, exact nested
key sets and exact Boolean types.  Its closed claim boundary states:

- K1 diagnostic axis only;
- no K8 and no equal-bandwidth K1x8;
- no VCS/DC/PT/PTPX/SAIF;
- no single-K1 power admission;
- no fair energy comparison;
- no performance, mapped functionality, system speedup, PPA or headline.

Adding a future key is rejected.  Changing any false claim to true is rejected.
Changing a false Boolean to integer zero is rejected.  The mutation suite
explicitly attacks K8, K1x8, single-K1 power, fair energy, performance, mapped
functionality, system speedup, PPA and headline.

Contract JSON SHA256:
`1c50a862e02aeda009d52850f00ba8befa96c19b6599077e61951b36929299f5`.
Its sidecar and outer-seal-file SHA256 values are respectively
`344604bb7fe3baa5ee7093ed11e80c42c62dbdda2e69bae493b3bc4e2d1e67d1`
and `4951c1133ca49f03589b02dfc64b5b6608f9dc376765f70389838d2e0924a516`.

## Repair 3 — structural valid-qualified endpoint guard

M1293 adds a new endpoint module rather than modifying M1279.  The checker:

- removes block/line comments;
- tokenizes the balanced named `always_comb` block;
- requires its complete normalized token SHA;
- separately requires the seven payload fields, nested valid/payload/accept
  guards, and exact inner qualified bindings.

The protected block token SHA is
`9b6e504a6d7d7bf4cae7dfa1cd005535a37b5a552ba47ecb2a979420df6e173d`.
Replacing either valid or payload-known condition with `1'b1` while leaving the
old expression in a comment changes that structural hash and is rejected.
Harmless whitespace/comments do not change it.

## Repair 4 — exact thirteen tap RHS bindings

The checker requires the exact thirteen kept declarations and exact occurrence
counts.  Eight top-level assignments must have these precise RHS expressions:

| Tap | Exact RHS |
|---|---|
| `tap_core_mem_req_accept` | `core_mem_req_accept` |
| `tap_adapter_core_mem_req_accept` | `adapter_core_mem_req_accept` |
| `tap_core_mem_rsp_accept` | `core_mem_rsp_accept` |
| `tap_adapter_core_mem_rsp_accept` | `adapter_core_mem_rsp_accept` |
| `tap_consistency_fault_now` | `consistency_fault_now` |
| `tap_consistency_fault_q` | `consistency_fault_q` |
| `tap_core_protocol_error` | `core_protocol_error` |
| `tap_adapter_protocol_error` | `adapter_protocol_error` |

The other five taps must have one exact named leaf-port connection.  Direct tap
statements explicitly forbid `$isunknown`, `===`, `!==`, ternary `?`, and X/Z
literals.  The mutation that changes `tap_core_protocol_error` to an
`$isunknown(...) ? 0 : ...` coercion is rejected.

## New source identities

| Source | SHA256 |
|---|---|
| `dc_handoff/tb/m1293_valid_qualified_scalar_bank_endpoint.sv` | `0f33949232f6973c4f05364b331b51192ad39f4d1318402a82616fe570718e88` |
| `dc_handoff/tb/tb_m1293_c2_dual_dut_functional_reachability.sv` | `89f1915b57b9ed5b7ebc72eb8db3dcdea6dfde67db1dae6c92a87ba027e893a9` |
| `dc_handoff/filelists/date_m1293_c2_dual_dut_source_only_vcs.f` | `af7c3e4394d35ba037280c82b6395769ed10203a73b91446ac24f7251a169564` |
| `system_simulator/scripts/check_m1293_c2_semantic_tap_dual_dut_repair_source.py` | `eb88b6c7b3def8c01a429305225d51802d0a91e6918ded44fe87c6f780b28c39` |
| `system_simulator/tests/test_m1293_c2_semantic_tap_dual_dut_repair_source.py` | `a682ed7f6cf00e3a87c9eeec8c648f464d79f90906967eb6a7c929e91f5f5cf1` |

The eleven-member filelist reuses the seven frozen M1279 RTL modules and the
frozen M218/M349 dependencies, then selects only the new M1293 endpoint and TB.
It is a compile intent, not compile/elaboration proof.

## Mechanical checks

- Checker status:
  `PASS_M1293_SOURCE_REPAIR__NO_EXECUTION_AUTHORIZED`.
- Mutation tests: 12/12 PASS in 0.031 s.
- Exact semantic taps: 13; eight direct RHS and five leaf-port RHS.
- Endpoint payload-known fields: 7.
- Dual-DUT transaction classes compared: 3.
- Atomic window: 256 cycles.
- Filelist members: 11.
- Real tool calls: 0.

## Claim and next gate

M1293 admits only this statement:

> The additive source contract structurally requires a known-valid endpoint and
> a same-stimulus K1 dual-DUT diagnostic whose PASS is reachable only after bank
> request, result and token-done participation with class-aware equality.

It does not admit that those events have occurred in simulation.  It does not
admit K8/K1x8 fairness, power, energy, performance, mapped function, timing,
area, system speedup or a paper claim.

The unique next step is a different-author receipt-blind source hammer attacking
all four repaired boundaries.  Only if that hammer passes may the root author
consider a separately sealed, one-shot RTL-only VCS release.  No EDA release is
created here.

`docs/359_DATE终局冻结_20260813.md` was not modified and remains SHA256
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

