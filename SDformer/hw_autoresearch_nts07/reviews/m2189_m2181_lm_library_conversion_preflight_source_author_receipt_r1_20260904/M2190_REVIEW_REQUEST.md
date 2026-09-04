# M2190 independent source-hammer request

Independently review the additive M2189 LM-only source. Do not run LM, EDA,
`lmutil`, a license query, GPU work, M2182, or M2191.

Reproduce the M2181 fourth-child failure, then prove M2189 rejects it. Verify
that the only connected process identities are the root bootstrap/wrapper,
exactly one pinned `lm_shell_exec` identity, and exactly one pinned `Milkyway`
identity below the actual LM identity. Independently mutate an extra wrapper,
actual, Milkyway, `/usr/bin/sleep`, reparenting, environment, executable path,
and an extra observation on an otherwise allowed identity.

Normalize the new TCL identity strings back to M2180/M2182/M2183 and require
byte equality with the frozen M2180 TCL. Verify one non-overwriting frame
conversion, option set/readback order, seven fresh isolated directories,
fresh result/attempt/work/lock, empty M2182 and M2191 censuses, and no design or
P&R command.

Only an exhaustive double-sealed review scoring at least 95 with P0/P1/P2 all
zero may emit `PASS_M2190_M2189_SOURCE_HAMMER__M2191_ONE_SHOT_AUTHORIZED`.
M2182 remains permanently unauthorized. A passing M2190 may authorize exactly
one M2191 library-conversion preflight, one license query, one top-level
`lm_shell`, no retry, and zero P&R.
