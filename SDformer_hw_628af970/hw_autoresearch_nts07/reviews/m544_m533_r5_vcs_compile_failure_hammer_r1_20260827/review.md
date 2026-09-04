# M544 / M533-r5 VCS compile-failure independent hammer

Verdict: **FAIL, 35/100, P0/P1/P2 = 1/1/1.** The consumed result identity is permanently **`FAILED_UNSEALED_DO_NOT_CITE`**. This was a fresh independent read-only review. It ran no VCS, simv, Icarus, Verilator, DC, Formality, PT, PTPX, CPU/GPU experiment, or remote job; it did not alter or seal the partial result directory.

## Exact failure

`compile.log` contains one VCS parse error and no later elaboration or simulation evidence. VCS V-2023.12-SP1 parsed the foundry macro model, macro wrapper, core RTL r2, SVA r2, and then TB r3. It stopped at:

```text
tb_m528_dead_write_only_1rw_product_capture_r3.sv:178
logic [1151:0] packed;
token is 'packed'
SystemVerilog keyword 'packed' is not expected to be used in this context.
```

The source at TB lines 175–185 confirms `packed` is only the local return-packing variable of `oracle_pack_row12`; it is nevertheless a SystemVerilog reserved keyword and is illegal as this identifier. `compile.log` ends with `1 error` and `CPU time: .736 seconds to compile`.

The exact numeric child `compile_rc` is **not persisted** in the result directory. The runner captures it in shell variable `compile_rc`, then on nonzero calls the final resource sampler and exits via `fail`; only stderr outside `compile.log` would have carried `rc=${compile_rc}`. What is durable is sufficient to prove `compile_rc != 0`: the exact syntax error, no `simv`, no `sim.log`, and the runner's compile-failure branch.

## Attempt consumption and failure-receipt gap

The attempt was consumed before compilation: runner line 611 atomically created `results/m533_m528_dead_write_only_1rw_vcs_r3_20260827`, and the frozen release defines that `mkdir` as the sole attempt-consumption point. The directory now exists with eight partial files.

On compile failure, runner lines 753–755 perform a final synchronous resource sample and call `fail`. Cleanup lines 59–76 only stop the monitor, remove the temporary preflight directory, and preserve the shell exit code. They do **not** create `RUN_FAILED`, a machine-readable failure receipt, `SHA256SUMS`, or an outer seal. Consequently the raw directory is neither a PASS result nor a sealed FAIL result. It must not be completed in place, resealed later as if atomic, deleted and retried, or reused as a new identity.

## What passed around the failed compile

- Both collision receipts say `PASS` with zero forbidden-process matches.
- All three prelaunch samples pass the frozen thresholds. Minima are `MemAvailable=412,882,972 KiB`, `SwapFree=57,218,812 KiB`, and commit headroom `69,769,616 KiB`.
- Runtime monitoring contains eight periodic samples plus one final synchronous sample. Session/user failcnt, `under_oom`, and `oom_kill` remain zero. `RESOURCE_FINAL_ACK` records exactly one final acknowledgement at sequence 8. There is no `RESOURCE_VIOLATION`.
- Frozen identities recompute exactly: core r2 `726039db...`, SVA r2 `b9f66feb...`, macro wrapper `8fd008a3...`, binding plan `db4075cb...`, and `docs/359` `dedde7ce...`.

These checks show the launch preconditions and final resource handshake worked. They do not repair the failed compilation and establish no functional RTL result.

## Findings

### M544-P0-01 — the only authorized functional attempt produced no executable simulation

TB r3 is syntactically illegal for VCS at line 178. No `simv`, `sim.log`, functional PASS token, SVA result, or coverage token exists. Therefore M533/M528 has no functional VCS admission from this identity, and no RTL correctness, speedup, PPA, energy, full-network, or paper claim may cite it.

### M544-P1-01 — a consumed compile failure is left as an unsealed partial directory

The runner seals only its success tail. Its compile-failure path writes neither an explicit failure status nor the captured numeric `compile_rc`, and produces neither member hashes nor an outer seal. The partial files can be frozen by this review, but that cannot retroactively convert the original result into an atomic result receipt.

### M544-P2-01 — the 100/100 source chain did not establish parser readiness

The prior source/static/admission chain correctly checked protocol/oracle and launch controls but let an obvious SystemVerilog keyword identifier reach the sole tool attempt. A changed TB must receive a new source identity and a fresh static/admission chain; the old 100/100 review cannot authorize it by inheritance.

## Minimum next identity

1. Create a new TB identity whose only functional-source edit is renaming the local variable `packed` and its references in `oracle_pack_row12`. Do not change oracle behavior.
2. Freeze core RTL r2, SVA r2, foundry macro wrapper/model, binding plan, and all other TB behavior.
3. Create a new runner/result identity. On every post-`mkdir` failure, it must write and double-seal an explicit failure receipt containing phase, child return code, immutable source hashes, resource/collision status, present/absent output inventory, and `FAILED_DO_NOT_CITE`; success and failure must both terminate in sealed states.
4. Run a fresh source-static hammer and a fresh closed launch-admission/release chain for those exact new hashes. Only then may a separately authorized single VCS attempt run.

The original r3 result remains permanently `FAILED_UNSEALED_DO_NOT_CITE` and must never be reused.
