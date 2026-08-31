# M750 source/candidate hammer request

Perform a fresh, read-only static review of the M750 macro-integrated DC source
package and launch-now-false candidate. Do not run the runner, `dc_shell`, VCS,
PT, PTPX, Formality, or any other EDA command.

The review must fail closed on any path that reads the foundry behavioral `.v`
in DC, does not link the checksum-pinned macro `.db`, permits a macro count other
than nine, admits an unresolved macro or inferred parent array, weakens the
setup/hold/area report gates, or permits launch before M746/r12 VCS PASS plus a
fresh independent result hammer. Confirm the result and attempt identities are
still absent and `docs/359` retains its frozen SHA.

A PASS only permits another agent to author a future release after the M746
prerequisite exists. It does not authorize DC.
