# M1714 read-only M1701 salvage audit

Verdict: **PASS_SALVAGE_CANDIDATE_ONLY**. The sealed quarantine contains a complete positive DC candidate: `dc.rc=0`, the Tcl terminal marker exists, setup/hold are MET with +0.0278721/+0.0285582 ns WNS and zero TNS/violations, area is 166,514.31 µm² (+8.91%, within the +10% ceiling), all nine SRAM macros remain bound, and DRC/link reports are clean.

The exact M1701 fatal regex matches only the fixed Synopsys GUI initialization error caused by the stripped HOME environment; the same signature occurs in completed DC campaigns. Broader `error`, `link`, `unresolved`, and `loops` words are Tcl guard source echoed by dc_shell or informational `check_timing` headings. No guard fired: the run reached normal dc_shell termination and wrote `TCL_INTERNAL_COMPLETE.txt`.

This does not promote or rename the quarantine and is not yet paper-citable. Since the mapped identity changed, a newly numbered source must bind the immutable quarantine and run Formality plus independent PT under a separate review/release chain.
