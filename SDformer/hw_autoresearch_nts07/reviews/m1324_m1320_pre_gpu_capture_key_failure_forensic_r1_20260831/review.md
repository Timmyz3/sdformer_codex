# M1324 M1320 pre-GPU failure forensic

The M1249 attempt is permanently consumed and must not be reused.  The local
failure artifacts bind the exact traceback, attempt token and empty temporary
log.  The failure is a single missing runtime field: M1313 contains no
top-level `capture`, while frozen M1227 directly reads
`capture.attention_windows_per_call` before model or GPU work.

The safe repair is not to change M1313 or selection identity.  A source-only
additive successor must first obtain the unchanged binding through M1319 exact
M1313/M1314 validation, then construct a minimal runtime projection containing
the new sealed contract path, exact unchanged cohort, a new result path and
`capture.attention_windows_per_call=100`.  The value 100 is independently
present in the frozen executable M1182 and M1210 R1-compatible bindings.

The successor must own fresh M1324 result, attempt and log namespaces, consume
only its own `O_EXCL` attempt, and retain the no-retry and atomic no-replace log
policy.  It must not call M1319 `execute_once` or M1249 `consume_attempt`, since
both own the already-consumed M1249 namespace.
