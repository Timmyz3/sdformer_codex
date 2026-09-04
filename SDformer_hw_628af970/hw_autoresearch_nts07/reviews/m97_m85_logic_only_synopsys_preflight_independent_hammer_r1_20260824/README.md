# M97 M85 logic-only Synopsys preflight — independent hammer r1

Date: 2026-08-24  
Review boundary: read-only review of the M97 contract, production DC file list, SDC, Tcl and launcher. No production input was modified and no tool was launched by this review.

## Verdict

**91/100 — GO for the hardened, scoped M85 logic-island DC flow; NO-GO for promoting or sealing the result as complete PWP timing until the remaining SRAM port-cut claim gap is explicit.**

The selected top and source closure are correct: `guarded_wordpacked_pwp_stream` is compiled from exactly the M82 and M85 production RTL files. The synchronous active-high `rst_core` is correctly timed as an ordinary data input, `SYNTHESIS` is defined, the clock is pre-CTS/ideal, and the observed in-progress log confirms one clock, zero macros and effective `ZeroWireload` application. Contract and admission text also correctly prohibit paper-ready, full-frontend, headline and system-speedup claims.

The updated launcher now detects `common_shell_exec -shell dc_shell` and takes an atomic per-output `.launch_lock`; those two requested guard defects are statically closed. Process-tree/lsof evidence supplied by the root reviewer also establishes that the currently running job has one launcher/tee/log writer; multiple backend PIDs are not evidence of two launches.

During this review the launcher was hardened. It now freezes and verifies the M97 contract, DC file list, SDC, Tcl, M82/M85 RTL, VCS preconditions, and setup/min DB SHA identities; it rejects period/operating-condition overrides and records the actual launcher SHA, libraries and corner. The remaining P1 is claim semantics: the contract does not explicitly say that the M85 bank interface is a port-cut/combinational-response abstraction with no address-to-SRAM-to-data timing closure.

The current run began with pre-lock launcher SHA `3720b19e55dbe0c81bc45ff941ff6b563a3878f5f258c751c6a1ef76280a8540`. The root reviewer confirmed that contract/filelist/SDC/Tcl/RTL/DB semantic inputs were not modified during that run. Its post-run receipt must record that exact launch provenance and must not attribute hardened launcher SHA `670dbe8b40f35f2d5ceaa536141c440179acc7ae5a05e2b679910fe97f8e6fc0` retroactively.

## Score

| Category | Score |
|---|---:|
| Production-only source closure, top and parameters | 20/20 |
| Clock, synchronous reset and I/O constraints | 20/20 |
| `SYNTHESIS`, ideal-clock, ZeroWireload and compile semantics | 18/20 |
| Fail-closed identity, corner and concurrency controls | 23/25 |
| Claim boundary and output sealing | 10/15 |
| **Total** | **91/100** |

Open severity count: **P0=0, P1=1, P2=3**.

## Confirmed passes

- Production DC file list contains exactly:

  ```text
  rtl_m82/zero_bubble_elastic_pwp_stream.sv
  rtl_m85/guarded_wordpacked_pwp_stream.sv
  ```

  It contains no SVA, testbench, wrapper, M86 storage, generated netlist or result source.
- Top is hard-coded consistently as `guarded_wordpacked_pwp_stream`; frozen/default `ROW_W=10`, `TAG_W=32`, `BUFFER_WORDS=3680` are protected indirectly by the exact M85 RTL SHA.
- SDC creates a 3.000 ns `core_clk`. `data_inputs` removes only `clk_core`, so `rst_core` retains the same 0.250 ns input delay and 0.100 ns transition as other synchronous inputs. No reset false path or asynchronous reset exception exists.
- Tcl analyzes SystemVerilog with `-define SYNTHESIS`, preventing simulation-only frozen-geometry `$fatal` logic from entering synthesis.
- No propagated-clock command exists; the clock is an ideal pre-CTS clock. The in-progress DC log reports one clock.
- Tcl applies `ZeroWireload`, and the in-progress log reports `OPT-170/171` applications to M85, M82 and generated subdesigns. The log also reports `Macro Count 0`.
- Target/min library roles, `set_fix_hold`, preserved hierarchy and postcompile setup/hold/constraint reports are structurally present.
- Contract/admission consistently state `logic_only=true`, `pre_macro=true`, `paper_ppa_ready=false`, `full_pwp_frontend_ppa=false`, `system_speedup=false/admitted=false`, `headline=false/admitted=false`, and `macros=0`.
- Current launcher lines 28–41 cover `dc_shell`, `dc_shell-t`, `common_shell_exec -shell dc_shell`, existing output refusal and an atomic per-output launch lock. The lock closes same-output TOCTOU between the existence check and directory creation.
- Current launcher rejects noncanonical period/operating-condition values, SHA-checks the complete semantic source/flow/library set, and records launcher/library/corner identities in admission.

## Controls closed during this review

The initial launcher did not freeze contract/filelist/SDC/Tcl or DB identities, and it omitted the `common_shell_exec` backend plus an atomic output lock. Hardened launcher SHA `670dbe8b40f35f2d5ceaa536141c440179acc7ae5a05e2b679910fe97f8e6fc0` closes those launch-time defects:

- exact SHA checks cover contract, production file list, SDC, Tcl, M82, M85, VCS contract/completion, and both DB files;
- canonical `3.000` ns and `ssg0p9v125c` are enforced;
- admission records actual launcher SHA, setup/min library paths and hashes, and operating condition;
- backend detection covers `common_shell_exec -shell dc_shell`;
- output refusal plus atomic `.launch_lock` closes same-output launch races.

These are closed for future launches. The in-progress run retains the explicitly disclosed pre-lock-launcher provenance described above.

## Open findings

### M97-P1-01 — The SRAM port-cut timing abstraction is not explicit enough to prevent timing-claim laundering

The scope excludes SRAM and the SDC comment says synchronous SRAM is excluded, but neither contract nor admission explicitly states that `bank_row_addresses[79:0]` and `bank_words[255:0]` are disconnected output/input timing cuts. `bank_words` receives a generic 0.250 ns input delay; there is no address-to-SRAM-to-data path, macro clock-to-Q, bank routing or one-cycle synchronous-response boundary.

Impact: M85 setup slack/Fmax can be mistaken for a complete PWP lookup path even though the memory round trip is absent. This is the most important remaining claim-boundary loophole.

Required closure: add machine-readable flags such as `combinational_external_bank_model=true`, `address_to_bank_to_data_timing_closed=false`, `synchronous_sram_interface=false`, `complete_pwp_lookup_timing=false`, and carry them into admission, receipt and every summary table.

### M97-P2-01 — ZeroWireload/ideal-clock semantics should be made explicit and machine-checked

The observed log proves that `set_wire_load_model -name ZeroWireload` currently takes effect, and no propagated clock is requested. However, Tcl does not explicitly set `auto_wire_load_selection false` and `set_wire_load_mode top`, nor does sealing parse `report_clocks`/wireload evidence to confirm an unpropagated clock and effective ZeroWireload after compile.

Suggested closure: set those wireload controls explicitly, emit a design/clock attribute report, and have the receipt validate the reported clock period, propagated state and wireload model.

### M97-P2-02 — Completion is weakly sealed

The launcher checks only for four nonempty reports plus the mapped Verilog, then creates an empty `RUN_COMPLETE.txt`. There is no output manifest or machine receipt carrying the claim gates and exact report/netlist SHA identities.

Impact: reports can be detached from `admission.txt`, replaced after completion, or summarized without the required logic-only labels.

Suggested closure: produce a nonempty final receipt and manifest only after all checks pass. Repeat every negative admission flag in the receipt and include hashes for admission, log, reports, netlist, SDC/DDC/SVF where retained, and launch-time inputs.

### M97-P2-03 — Post-run admission checks do not inspect report contents

`pipefail` and the `^Error:` scan are useful, but final admission does not parse postcompile `check_design`, `check_timing`, reference/blackbox status, clock identity, macro count, constraint violators, setup/hold slack or the actual mapped target library.

Impact: `RUN_COMPLETE` means that files exist, not that the intended constraints and cell closure were achieved. A negative-slack run may still be a valid diagnostic, but its status must be explicit.

Suggested closure: a fail-closed receipt builder should extract and report these fields. Keep `paper_ppa_ready=false` regardless; distinguish `tool_completed`, `netlist_link_clean`, `setup_met`, `hold_met` and `constraint_clean` rather than overloading completion.

## Admission after fixes

Even after all findings close, the only admissible statement is:

> M85 production RTL was synthesized as a 3.000 ns, ideal-clock, ZeroWireload, zero-macro logic island under the recorded TSMC28 setup/min libraries.

It is not M86-R3, not a complete PWP frontend, not SRAM-inclusive timing/area, not paper-ready PPA, and not evidence of energy, throughput, FPS or system speedup.
