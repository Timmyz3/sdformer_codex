# M1187 independent hammer of the M1186 E8 handoff release

Verdict: **PASS (100/100, P0=0, P1=0)**.  This different-author hammer
authorizes only the exact 44-file small transfer, read-only remote preflight,
and one zero-retry invocation of the unchanged M1183 E8 runtime.  It did not
perform any transfer, remote access, GPU work, checkpoint load, range capture,
EDA, or production execution.

The release preserves the exact M1183 runtime SHA, transfers 44 small files
totalling 400,749 bytes (the self contract plus 43 inventoried rows), and
preflights 40 canonical data rows totalling 491,525,120 bytes by remote hash.
The complete M1181 directory, M1183/M1185 authorities, canonical cohort and
seals, profiler, evaluator, both cohort authorities, ep29 identity, protected
docs/359, namespaces, canonical GPU lease, GPU-idle test, and legacy-M511 test
are all bound.

Python 3.10 compilation and all 11 controlled tests passed.  Six controlled
attacks covering a missing row, path drift, data-order drift, shell injection,
self-certification, and namespace redirection fail closed.  `subprocess` uses
fixed argv with `shell=False`; preflight does not consume the attempt marker;
the runtime is invoked once and has no automatic retry.

No resulting range/compression claim is admitted by this review.  The future
production output must be double-sealed and pass a fresh independent result
hammer before any hardware or paper binding.
