# M989 atomic one-shot copy-only promotion source

## Verdict

`PASS_M989_ATOMIC_ONE_SHOT_COPY_ONLY_PROMOTION_SOURCE__FUTURE_M990_HAMMER_REQUIRED`

M989 is an additive successor. It does not modify M975, M976, the M962
quarantine, the original runner, or any EDA result.

The M975 publication TOCTOU is closed for cooperating invocations by two
separate atomic identities:

1. after the complete M990/M991/M992 authorization chain is verified, an
   atomic `mkdir` acquires the fixed launch lock;
2. inside that lock, another atomic `mkdir` permanently consumes the one-shot
   attempt before `WORK` exists or any source byte is copied.

`WORK` is fixed and attempt-bound, not PID-derived. A second invocation while
the first is active stops at the lock. Every invocation after lock release
stops at the permanent attempt identity before copy. The complete work tree is
sealed, `TARGET` is checked a second time, and `mv -T` publishes without
directory nesting.

On failure after work creation, the trap adds a failure marker, seals
best-effort, and moves the fixed work tree to a separate forensic quarantine.
It never writes or moves anything into `TARGET`. After successful atomic
publication, `WORK` no longer exists, so the trap cannot alter the target.

## Static concurrency attacks

The production promoter was not executed. An exact source-order audit and a
temporary-filesystem primitive test passed:

- first 32-way wave: one copy winner, 31 stopped before copy;
- second 32-way wave after attempt consumption: zero winners, all 32 stopped;
- target appearing before publication: publication stopped and work isolated;
- `mv -T`: payload appeared directly under the target, with no nested work
  directory;
- simulated post-publication trap condition left the target unchanged.

## Preserved physical boundary

M989 changes no result semantics. M975 remains the authority for the recovered
M962 evidence: setup MET at 3 ns with WNS +0.001795 ns, TNS 0, 100 complete
MET paths, 9 SRAM macros, and 147,246.392090 um2 total cell area.

Hold is not signed off, power and energy are false, and the CPU same-ledger
`1.746753x` is not an RTL-cycle speedup. The current source is not directly
citable. Execution remains blocked until M990 source hammer, M991 release,
M992 release hammer, and M993 execution/result review.
