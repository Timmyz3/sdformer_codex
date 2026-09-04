# M1218 — M1213 C1 R8 VCS timeout failure review

## Verdict

M1213 is a **consumed, recursively sealed failure**.  Compilation,
elaboration, and linking completed, but the one authorized simulation produced
no coverage line and no PASS token before the runner's 1800 s timeout returned
exit 124.  `functional_vcs_verified=false` remains authoritative.

The failure is consistent with a clock-time-advancing unbounded testbench wait.
It does **not** prove an RTL/DUT functional fault.  The sealed log contains no
phase marker or sampled DUT state, so it cannot distinguish the three
unbounded waits in `random_legal_transaction` from the unbounded
`normal_m935_completion` preload wait.  No VCS/EDA rerun was made by M1218.

## Sealed evidence

- Attempt identity exists at
  `.m1213_m1210r8_m1162_c1_common_charge_protocol_vcs_r8_attempt_consumed`;
  `automatic_retry=false` and runner SHA-256 is
  `0eb674169ad79730e41642b2b3c2b3e2571dfc42032f2e96ff8ff05f4b080049`.
- The quarantine's `SHA256SUMS` and outer `SHA256SUMS.seal.sha256` both verify
  recursively.  Key file hashes are: `compile.log`
  `43515f953d360419806317027d68c53ae93e21d553b17c5d6e653f49aa25d158`,
  `sim.log`
  `a94bd19c0734eb943856f2ff86f019f9dc6a131d61150951ec769ac28fd546c3`,
  and failure marker
  `97bada6020144ec894868d2de54d5a4ffbff388e59df29b9ddf8b2843481255f`.
- `compile.log` parses all four frozen sources and ends with
  `1.361 seconds to compile + .744 seconds to elab + .184 seconds to link`;
  an executable `simv` is sealed.  This is compile/elaborate/link PASS only.
- `sim.log` is only 486 bytes.  It contains VCS startup and
  `SVART-AMAXINT: Exceeding INT_MAX assertion attempts`, but no `$fatal`, phase
  token, coverage line, assertion summary, or PASS token.  The failure marker
  records `exit_code=124`.
- The exact runner wraps `./simv -no_save` in `timeout ... 1800s`; timestamps
  span approximately 15:50–16:20 CST, matching that external timeout.

## Liveness localization

All SVA attempts are sampled at `posedge clk_core`; the TB clock is
`always #1.5 clk_core = ~clk_core`.  Saturating an assertion attempt counter is
therefore compatible with billions of time-advancing clock edges while the
main initial thread is suspended.  The warning alone is not evidence of a
zero-delay combinational oscillation.

Four unbounded suspension sites remain:

1. `wait (weight_fire_count == w0 + 1)` in each random transaction;
2. optional `wait (psum_fire_count == p0 + 1)`;
3. `wait (dut.response_accept_w)`;
4. `while (!prep_ready) @(negedge clk_core)` for every normal preload row.

The later normal issue, response, and task-done loops already have watchdogs.
The R7 sealed simulation reached random transaction 1 and failed at 498 ns on
a duplicated request.  R8 changes that path by adding random-window handshake
counters and quiescing both ready inputs after the intended request handshakes;
this makes a later latent liveness hole plausible, but does not prove R8 reached
normal preload because R8 prints progress only after all phases finish.

The specific hypothesis that a wrapper request remains pending *across* the
normal phase reset is contradicted by source structure: `normal_m935_completion`
begins with `reset_dut()`, and `reset_dut()` checks that `request_active_q`,
`weight_request_accepted_q`, and `psum_request_accepted_q` are clear.  Reset also
resets frozen M935.  A pending request can still explain a random-phase wait
before that reset, but it is not an evidenced cross-reset cause.

If the run did reach normal preload, `prep_ready` should be high after clean
reset whenever M935 has no fault/match and a free bank exists.  A permanent low
there would be material, but the present log neither proves the phase nor
captures `fault_q`, `match_active_q`, `prep_active_q`, or bank state.  DUT fault
therefore remains **unproven**, not exonerated and not admitted.

## R7 → R8 delta

R7's filelist reuses the R6 TB source with SHA-256
`0fcc2138ef5d716735eea01dee25a148a5223b1d6adf1e3b2fa464341fbf1345`.
R8 TB SHA-256 is
`060ec9d5ae6085a0dd013160d22f63e21615730384ddaef342eb3fa77e17947b`.
Excluding comments, module/token renaming, and reporting strings, R8 adds only:

- two handshake counters and a random-window enable;
- counting while that window is active;
- ready quiescence immediately after the intended handshakes;
- exact-one checks before and after response completion.

Frozen M528, M935, M1162 and R3 SVA sources are unchanged between releases.

## Minimum successor repair

The next revision should be **source-only observability/liveness hardening**:

1. replace all three random `wait(...)` statements and the preload
   `while(!prep_ready)` with bounded edge-count watchdogs; never use exact
   equality as an unbounded wait predicate;
2. print a unique, flushed phase token before and after each directed group,
   random transaction, reset, preload row range, issue beat, response, and task
   completion;
3. on timeout print the global counters plus wrapper
   `request_active/accepted/response_accept/boundary_fault` and M935
   `fault/match/prep/bank` state;
4. directly after the normal clean reset, bound-check initial `prep_ready`
   before sending row 0.  Only a reproducible low `prep_ready` at this clean
   boundary should promote the investigation to a DUT/M935 fault;
5. do not change RTL to accommodate this failure, do not retry M1213, and do
   not launch a successor until a distinct source hammer and release exist.

M1218 itself authorizes neither source mutation nor VCS.  It is failure-only,
`paper_admission=false`, `speedup=false`, `PPA=false`, and `headline=false`.

