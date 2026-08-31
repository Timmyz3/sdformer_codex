# M1279 — C2 semantic-tap dual-DUT source-only receipt

## Verdict

`PASS_SOURCE_ONLY_K1_OBSERVATION_METHOD_GATE__NO_EDA_EXECUTION__NO_POWER_CLAIM`

M1279 closes the source-authoring step required by M1155/M1274.  It adds a
K1-only RTL observation wrapper with thirteen named semantic taps, a
valid-qualified scalar-bank endpoint, and a fully instantiated dual-DUT
128-cycle diagnostic testbench.  Static source checking and eight mutation
tests pass.

This is **not** compile/elaboration proof and is **not** VCS, DC, PT, PTPX,
SAIF, power, energy, performance, PPA, or system evidence.  No EDA, GPU, remote
or checkpoint action was performed.  The wrapper is the diagnostic K1 axis;
it does not instantiate K8 or equal-bandwidth K1x8.  A future K1 power number
therefore cannot be called a fair C2 energy comparison.

## Additive source delivered

Seven tapped modules were created in `rtl_m1279/`.  The source checker removes
only tap ports, tap assignments and tap-only connections, normalizes the
successor module names, and proves the remaining module bodies equal the seven
frozen originals.  No frozen RTL or netlist was overwritten.

| Source | SHA256 |
|---|---|
| `rtl_m1279/m1279_c2_k1_semantic_tap_wrapper.sv` | `1ae6fc8107367817123e12f0b1ff70722de65d129792282b87c0532435334f43` |
| `rtl_m1279/m1279_fc2_bundle_to_8bank_no_reuse_adapter_tapped.sv` | `ed0e31f74bfa3f424ab1783a7aaffe15dab32d3e75bb7b996301837513e3a950` |
| `rtl_m1279/m1279_fc2_descriptor4_source_cap_frontend_tapped.sv` | `9f00a0d6de0d4c7eb01156aac749fc808274d7aa0949d0f3de562cb7ab200826` |
| `rtl_m1279/m1279_fc2_k1_reset_hygiene_registered_release_service_island_tapped.sv` | `43b9684ae9f38c864e67d54322008f8d8767fec6fb11aed2185fbd16788ee743` |
| `rtl_m1279/m1279_fc2_raw4_to_descriptor4_terminal_hint_compactor_tapped.sv` | `4cb345112b62b5b5ab5195c96f42521fade76e1d9845633b2e7e8eafb576818b` |
| `rtl_m1279/m1279_fc2_raw4_to_source_cap_frontend_tapped.sv` | `0f5dd8fc9247fd16160646d27d481c3d2dfe5e1001be0d4c95525675106c96f6` |
| `rtl_m1279/m1279_fc2_reset_hygiene_registered_release_standalone_raw4_acc24_tapped.sv` | `dc2abbc2e2ce8dc955672d267fd0a51ea160ad05eb0b1af29e0ddb61c3de03e7` |
| `dc_handoff/tb/m1279_valid_qualified_scalar_bank_endpoint.sv` | `defe7c86b3aeaf41a8d9b794848895bbaf6409aee28b7aba81b13569f4a983f9` |
| `dc_handoff/tb/tb_m1279_c2_dual_dut_valid_qualified_endpoint.sv` | `f5cc756d7d632a7cda90ba9fcec872295de15fb32e35905c1d9c465b8237d28c` |
| `dc_handoff/filelists/date_m1279_c2_dual_dut_source_only_vcs.f` | `3e24a97d62bf095c2282cd6669a89513135dc459abbda2aae8503d390629d926` |
| `system_simulator/scripts/check_m1279_c2_semantic_tap_dual_dut_source.py` | `84dc1780f3b0dae447c6591b1835aafceb08846c68c316315ec6689762c8d475` |
| `system_simulator/tests/test_m1279_c2_semantic_tap_dual_dut_source.py` | `c6f8fba34e230ed5fd1330880d8f729807e777543bbb812caa44d420248843e1` |

The filelist has eleven members and ends with the diagnostic testbench.  It is
an executable source description, but it has deliberately not been compiled
or elaborated in this milestone.

## Thirteen preserved semantic taps

Every tap is an explicit `(* keep = "true" *) output logic` signal.  Five are
leaf fault-Q signals propagated through named ports; eight are direct
top-level protocol/handshake observations.  Static checks find no anonymous or
hierarchical binding and no tap fan-in to functional control.

1. `tap_frontend_compactor_fault_q`
2. `tap_frontend_paired_sink_fault_q`
3. `tap_core_adapter_fault_q`
4. `tap_service_fault_q`
5. `tap_memory_adapter_fault_q`
6. `tap_core_mem_req_accept`
7. `tap_adapter_core_mem_req_accept`
8. `tap_core_mem_rsp_accept`
9. `tap_adapter_core_mem_rsp_accept`
10. `tap_consistency_fault_now`
11. `tap_consistency_fault_q`
12. `tap_core_protocol_error`
13. `tap_adapter_protocol_error`

## Dual-DUT comparison and fail-closed behavior

The source testbench applies identical stimulus to two endpoints around the
same frozen M349 scalar-bank model:

- the original unqualified endpoint; and
- the valid-qualified endpoint added by M1279.

The qualified endpoint exposes ready as false by default and enables an inner
request only when valid is exactly one and every relevant payload bit is
known.  Unknown valid or payload is quarantined and raises an endpoint fault.
The source uses strict case equality only for detection; it contains no
`force`, `release`, `initreg`, X coercion, `casex`, `casez`, or functional
`set_case_analysis` mechanism.

The diagnostic samples all thirteen taps per DUT plus request/public-fault and
endpoint diagnostics into one 32-bit first-X/union bitmap on every cycle of an
atomic 128-cycle window.  Any X or fault on the qualified path fails closed.
Original-X with a clean qualified path is only a diagnostic classification,
not a mapped-gate or power admission.  Both-clean is likewise a source-window
classification, not mapped proof.

## Checks completed

- Source checker status:
  `PASS_M1279_SOURCE_ONLY__NO_EXECUTION_AUTHORIZED`.
- Normalized frozen-clone equivalence: 7/7.
- Exact semantic tap set: 13/13.
- Dual-DUT instances: 2.
- Atomic diagnostic window: 128 cycles.
- Static/mutation unit tests: 8/8 PASS.
- Real EDA/tool calls recorded by the checker: 0.
- `docs/359_DATE终局冻结_20260813.md` remains unchanged at SHA256
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## Claim boundary

The admitted claim is limited to source structure and observation methodology:

> M1279 provides an additive K1 dual-endpoint diagnostic source topology with
> thirteen stable semantic observation points and fail-closed unknown handling.

It may not be promoted to any of the following: compiled RTL, functionally
passing RTL, mapped function, timing, area, SAIF, power, energy, throughput,
throughput/mm2, PPA, system speedup, K8/K1x8 fairness, or paper headline.
M903 remains the only admitted C2 physical/performance row.

## Unique legal successor DAG

1. A different-author M1280 source hammer checks this exact contract and source
   identity.  It does not run EDA.
2. Only after that hammer passes, a fresh one-shot release may authorize exactly
   one RTL VCS compile/elaboration plus one 128-cycle dual-DUT run.  DC should
   not be bundled into that first gate.
3. An independent result hammer must classify the RTL result.  If clean, a new
   K1 DC namespace must preserve all thirteen observation outputs, followed by
   one mapped dual-DUT diagnostic.
4. Only if the qualified endpoint uniquely removes or localizes the first X may
   the exact root be patched in another additive RTL namespace.  The order is
   RTL VCS, DC, then mapped five-case K1 anchors
   `259/737/3153/7569/14`.
5. If the qualified path retains X, stop the mapped-power line and retain M903
   logic-only evidence.
6. Fair energy is a separate successor, not an extension of the K1 diagnostic:
   add matched ARCH_MODE1 K8 and ARCH_MODE2 equal-bandwidth K1x8 wrappers and
   replays at identical external ports, libraries, and 3 ns boundary.  Frozen
   exact anchors are K8 `[51,131,486,1231,14]` and K1x8
   `[53,133,499,1246,14]`.  Produce ten production SAIFs for the two fair axes
   in one atomic attempt, or all fifteen only if diagnostic K1 is retained.
   Netlist/SDC identity and M974 PT/PTPX rules remain mandatory.

## Contract identity

- Contract JSON SHA256:
  `a93fa602788d6e6fe89f0260fe2f9ce8b3468212ca1c718d8c16c01beaa63bf4`.
- Contract sidecar SHA256:
  `7a49369dde03365e14df07b13e4a575b5845275b8377d6cdf2109f7d306db79b`.
- Outer contract-sidecar SHA256:
  `4dedd24c9eba08eb62c08e5e52cb621eeac68d10e58a2fda6287c27741271af3`.

