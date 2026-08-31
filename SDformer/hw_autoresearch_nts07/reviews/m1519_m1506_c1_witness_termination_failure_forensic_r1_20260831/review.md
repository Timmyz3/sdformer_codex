# M1519 — M1506 C1 witness/termination failure forensic

## Verdict

M1506 remains FAILED_OR_INCOMPLETE, DO_NOT_CITE, and its one-shot attempt
must not be replayed. This read-only forensic does **not** prove a DUT
functional failure. It also does not admit a DUT functional pass.

The admitted diagnosis is a verification observation seam:

1. the source oracle records a completed legal transaction and prints its
   source-only PASS token;
2. the testbench immediately calls $finish;
3. the bound final witness then reports that its strict registered stage
   machine is only at stage 3 and calls $fatal.

There is a second, independent admission blocker: the source oracle accepts
response_cycle_gap >= 2 and measured a gap of 3 cycles, while the release
runner also requires the exact two-cycle cp_ii2 cover. That cover has zero
matches. Therefore this log could not become a release-safe functional-VCS
receipt even if the terminal witness token were repaired.

## Evidence

| Check | Sealed observation | Interpretation |
|---|---:|---|
| Compile / simv | 1 / 1; no retry | The consumed budget is final |
| Source oracle records | 90 / 90 pass | No source-oracle mismatch appears |
| Source design ledger | 2 issues, 1 commit, 1 row | The source path observed completion |
| Source fault fields | boundary/core/M935 all 0 | No logged DUT fault bit |
| Source terminal order | PASS, then $finish | Termination starts immediately after source PASS |
| Final witness | pass 0, stage 3 | Witness did not reach W_TASK_DONE |
| Witness-local ledger | 1 response, 0 commit, 0 row, 0 task | Strict witness missed the later completion sequence |
| Design ledger on same witness line | 2 issues, 1 commit, 1 row; faults 0 | DUT/source ledger and witness-local ledger disagree |
| Response gap | 3 cycles | Satisfies source lower bound, not exact II=2 |
| cp_nonfirst / cp_ii2 | 1 / 0 matches | Non-first path covered; exact-II2 not covered |

The runner reached LOG_ADMISSION, so its run_tool(simv) return gate had
accepted the simulator process return. Admission then failed first on required
token cardinality because the expected witness-PASS token and pass operands
were absent. If that first gate were bypassed, exact-II2 coverage and the fatal
line would each still reject the log.

The quarantined logs are byte-identical to the minimal raw-build logs:
compile.log is 3,689 bytes and sim.log is 28,690 bytes. Both the consumed
attempt and quarantine inner manifests and outer seals verify.

## Root-cause boundary

The high-confidence statement is:

WITNESS_OBSERVATION_AND_TERMINATION_ORDER_MISMATCH

The M1497 testbench uses blocking counters at posedge, checks completion after
#1ps, prints the source PASS, and immediately calls $finish. The M1337R15
witness is a one-milestone-per-registered-stage observer and only emits its
verdict in a final block. At finalization it has not consumed the completion
sequence that the source/design counters already contain.

Without a waveform and without spending another prohibited run, the evidence
cannot distinguish a same-edge overlap rejected/ignored by the strict witness
from an active/NBA-region visibility race. The witness fault bit is zero, so
the log specifically demonstrates **incompleteness**, not a witness-classified
DUT violation. Changing DUT RTL on this evidence would therefore be an
unsupported response.

## Minimal successor / no-go

M1519 authorizes source-only design of one additive successor; it authorizes no
VCS, simv, or EDA execution.

1. NO_GO: never edit or replay M1506 and never promote any M1506 functional,
   cycle, timing, PPA, energy, speedup, system, or headline claim.
2. Preserve the M1497 source oracle, identities, exact event counts, unknown
   checks, attack checks, and all DUT fault gates. Do not weaken them to obtain
   a PASS.
3. Replace the verification-only terminal seam in a fresh namespace:
   independently accumulate legal milestone events (including legally
   coincident events) and expose an explicit witness_done/pass handshake.
   The TB must wait for that handshake, or fail on a bounded timeout, before
   $finish. Merely adding idle cycles is insufficient if the stage machine
   has already missed an event.
4. Choose the II claim before execution. If the claim is only the proven lower
   bound, admit a gap >= 2 observation and do not call it exact II=2. If exact
   II=2 is required, the present gap-3 stimulus/result remains a valid NO_GO and
   a new successor must actually cover the exact pattern.
5. Before any future one-shot run, require a source-only negative fixture
   reproducing source PASS -> finish -> witness fail, a positive terminal
   handshake fixture, an independent source hammer, and fresh launch authority.

## Claim boundary

dut_functional_failure_proven=false and
dut_functional_pass_admitted=false. All paper-facing claims remain false,
including functional_vcs, cycles_measured, timing_verified, speedup,
ppa, energy, system_speedup, and headline.

No VCS, simv, EDA, license, or retry action was run by M1519. No old evidence or
source file was modified. docs/359 remains
dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4.

