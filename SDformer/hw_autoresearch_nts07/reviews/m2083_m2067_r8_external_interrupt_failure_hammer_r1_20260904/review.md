# M2083 — M2067 R8 external-interrupt failure hammer

Verdict: **PASS failure audit; R8 is permanently FAILED/INCOMPLETE and non-citable.**

The R8 attempt compiled one Synopsys VCS image and produced 210 consecutive,
parser-valid workload logs for slots 0--209.  Slot 210 contains only the VCS
startup preamble.  The parent Python one-shot process exited when its owning
external session was aborted; the orphaned `simv` stopped making progress and
was terminated before quarantine.  No RTL assertion, arithmetic mismatch,
protocol fault, overflow, or source-identity failure was observed in the 210
completed slots, but a partial campaign is not a paper result.

The quarantine is exhaustive and double sealed.  It contains 211 raw slot
logs, `failure.json`, and `RUN_FAILED_OR_INCOMPLETE.txt`; exactly 210 logs have
one `PASS_M2067` token.  The canonical R8 success namespace is absent, the
attempt remains consumed, automatic retry is forbidden, and `docs/359` retains
SHA-256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

A source audit also found a duplicate-launch ownership race in the R8 runner:
namespace checking precedes lock ownership and failure publication does not
prove that the publishing PID created the attempt.  A successor must acquire
an owner lock before namespace checks and may publish failure only after its
own attempt marker is created.  R8 logs must not be inherited into a clean
successor headline result.

