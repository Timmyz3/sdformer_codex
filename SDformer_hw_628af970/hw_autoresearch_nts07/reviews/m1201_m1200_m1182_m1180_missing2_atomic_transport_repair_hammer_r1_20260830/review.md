# M1201 independent hammer: M1200 STOP

Verdict: **STOP; do not execute M1200 transport.**

The frozen source, test, contract, exact-two inventory identities, author
recursive seal, and docs/359 identity are intact.  The declared 10/10 source
tests pass, and isolated attacks confirm rejection of extra/traversal/symlink,
wrong-SHA, preexisting-destination, and M1180-attempt states.  Post-install SHA
and M1180 postcondition failures also remove both published destinations.

Two independent local fault injections nevertheless disprove the stronger
rollback-clean release claim:

1. An archive-cleanup exception raised from the extractor's `finally` block
   occurs outside its publication rollback handler.  The remote process exits
   nonzero while both exact repository destinations and the temporary archive
   remain.
2. An exception after the second `os.link` succeeds but before the destination
   is appended to `published` removes only the first destination and leaves the
   second.  The local helper has the same bookkeeping window.

This is not a remote/GPU result: M1201 made no SSH/SCP call, executed no
transfer/capture/EDA, and consumed no M1180 or M1200 attempt.  There is no
authorized one-shot command or digest environment for M1200.

Required successor: use a monotonic exact-state reconciliation invariant.  A
target may be absent or an exact regular file of the expected size/SHA;
preexisting exact targets are accepted, wrong files and symlinks are rejected,
missing targets are installed, and success requires both exact.  A separate
fresh hammer must review that successor before any transport.

