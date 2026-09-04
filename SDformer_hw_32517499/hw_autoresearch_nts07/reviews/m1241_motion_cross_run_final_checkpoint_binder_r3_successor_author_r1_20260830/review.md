# M1241 cross-run binder r3 successor author receipt

M1241 is an additive, source-only security successor to M1234.  It keeps the
exact M1234 result schema and status so the downstream M1233 capture interface
does not fork.  No checkpoint is selected by this package.

The two M1238 blockers are closed in source.  Every path component is opened by
a descriptor walk with `O_NOFOLLOW`.  Old and resume run roots must expose two
different device/inode/type identities, and old/resume configurations must
expose two physical identities and two pinned SHA identities.  Each checkpoint
and profile descriptor walk must pass through the exact frozen run-root
identity, which makes containment physical rather than a string-prefix test.

Profile bytes are read and hashed once.  Strict JSON parsing and the complete
825-sample, load-audit, 105/12 module, eight finite nonnegative metric, and
activity-domain validation finish before a final pathname `lstat` is compared
with the frozen descriptor identity.  After that comparison the result is
assembled only from frozen values; no `resolve()` or other pathname trust is
used.

The controlled suite passes 18/18 under Python 3.10.16.  It includes the two
M1238 reproductions: both parse-time and semantic-validation-time swaps are
rejected, and both final-component and parent-component run-root symlink
collapse are rejected.  Exact ep29/30/32/34, two run/config identities, 825,
four typed zero load counters, 105/12, all eight metric domains, lower-epoch
tie breaking, E0--E8, receipt sealing, fixed schema/status, and source-only
boundaries remain covered.

A fresh different-author source hammer is required before release authoring.
This author did not access a production path, remote host, GPU, real checkpoint,
valid825, EDA, production binder, or hardware rebind.  `docs/359` was not
modified.

