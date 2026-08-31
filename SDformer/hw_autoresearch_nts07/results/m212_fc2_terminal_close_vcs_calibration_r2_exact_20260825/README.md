# M212 terminal-descriptor partial-close VCS calibration r2

M212 adds one exact `descriptor_token_last` sideband to the four-wide
compactor.  A final partial window is closed on the accepted terminal
descriptor edge, while the authoritative registered upstream-done fence is
still accepted normally.  The optimization changes availability timing only;
it does not cross a token or compact-window ownership boundary.

Synopsys VCS passes the broad reference-checked directed regression, a 256-case
continuous-source sweep, and the legal bank48 adversarial case.  The new cover
hits 65 terminal partial closes in the sweep.  The cycle-exact software
recurrence matches all 256 cases with zero mismatches.  Bank48 still accepts
two 48-event packets, emits 192 groups, and completes in 195 cycles.

The exact M210-to-M212 sweep A/B has 36 one-cycle improvements, 220 unchanged
cases, and zero regressions.  This is isolated sparse-frontend cycle evidence,
not a complete-FC2, FFN, physical, system, or headline speedup.  `docs/359` was
not modified.
