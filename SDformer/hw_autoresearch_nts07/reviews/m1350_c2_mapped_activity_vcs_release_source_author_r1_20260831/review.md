# M1350 C2 mapped-activity VCS release-source author review

M1350 is an additive source-only successor to M1344.  It does not modify the
sealed runner or the failed M1348 review.  It exact-SHA binds all three layers
of M1348 and closes its seven false negatives.

Future source-hammer, launch-release and final-hammer JSON is loaded with a
duplicate-key-rejecting parser.  Their `claim_boundary` object must equal the
frozen nine-key all-false object; an extra `launch_authorized` field therefore
cannot hide beside the known false claims.

The runner is no longer admitted by raw token counts.  M1350 parses the active
failure and attempt `printf` writers by exact normalized format plus value
expressions, and parses the unique active success-receipt heredoc with Python
AST.  Each writer must contain the same nine SHA identities exactly once with
the canonical live expression.  Comments, dead copies and unrelated strings
cannot replace an active field.

The new suite passes 36/36.  It includes all 27 combinations of three receipt
writers and nine identities, deleting the live field and adding its name back
as a comment.  Duplicate status and extra-claim attacks are rejected.  The
inherited M1344 12/12, M1336 10/10 and M1334 12/12 suites also pass.

This author review permits only a fresh different-author blind hammer.  It
does not authorize a launch release, license query, VCS, simv, SAIF or EDA.
`docs/359` remains unchanged.
