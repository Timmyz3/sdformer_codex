# M926 request: fresh independent M925 R2 source hammer

Review only the sealed M925 driver, runner and source contract. Do not run or
enumerate the full D0/A1/t0 row and do not create the M925 attempt, result or
failure namespace.

The central attack is process ownership. The future worker must be the actual
Python process in a private `setsid --wait` process group. Normal exit, TERM,
TERM-to-KILL fallback and shell signal/EXIT paths must all reap the root and
prove the group empty before any stage rename, quarantine or seal. RSS must be
aggregated over that real process group, not a shell-function wrapper.

The scientific threshold `9.320783571 s` is a historical M900 failure. M925 is
not a 100x retry; `2715 s` is only an operational safety timeout derived from
the bounded 100K scaling calculation. Host simulator runtime is never
accelerator speedup.

Run the unmodified bounded M896 suite through real 100K, exact-pin M925
`--dry-run-no-work`, negative caller-pin/argument attacks and bounded dummy
process-group attacks. If and only if every check passes, the sole next
authority is for another author to write an inert M927 release. A later fresh
M928 final hammer is still required before one diagnostic invocation.
