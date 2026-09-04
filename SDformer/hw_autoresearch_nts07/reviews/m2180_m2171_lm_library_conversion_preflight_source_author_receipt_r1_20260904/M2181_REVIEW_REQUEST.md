# M2181 independent source-hammer request

Independently review the additive M2180 LM-only library-conversion source. Do
not run `lmutil`, `lm_shell`, `lm_shell_exec`, Milkyway, ICC2, VCS, DC, PT,
Formality, or a GPU job.

Reproduce the source-only suite: exact M2171 and docs/359 identities, native
same-version Library Manager header control and mutation, the accepted
LM-wrapper -> pinned `lm_shell_exec` -> pinned Milkyway process graph, all 12
process mutations, Bash/Python syntax, Tcl option set/readback ordering, and
the absence of design/P&R commands. Verify both the contract and this receipt
exhaustively.

Authorize exactly one M2182 library-conversion preflight only at score >=95
and P0/P1/P2=0/0/0, with status
`PASS_M2181_M2180_SOURCE_HAMMER__M2182_ONE_SHOT_AUTHORIZED`. M2182 may make
one license query and one top-level regular `lm_shell` run; it may create only
the isolated frame NDM and evidence. It must not create a design library,
import RTL, run P&R, or retry automatically.
