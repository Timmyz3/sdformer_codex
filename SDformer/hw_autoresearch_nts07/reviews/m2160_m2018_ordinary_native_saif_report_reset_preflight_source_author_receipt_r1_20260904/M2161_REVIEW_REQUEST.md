# M2161 independent source-hammer request

Perform a read-only, no-EDA review of the fresh M2160 source identity.  M2151
is consumed and must not be edited or retried.

The central P0 question is whether the UCLI sequence matches actual `run` /
SystemVerilog `$stop` control flow and obeys VCS report-before-reset semantics:
the census and window-begin markers occur inside the first run; UCLI then
disables and reports a distinct diagnostic prehistory SAIF before requesting
reset; it rejects `SAIF_REPORT_BEFORE_RESET`, `ignored`, or equivalent reset
warnings; it re-enables and measures only the second 20,292-cycle run; end/pass
occur before the second-run-return marker.  The testbench may attest only
`power_reset_requested=1`; final acceptance must combine warning absence with
the sealed 60,876-ns measurement duration.

Independently mutate and reject all entries in `mutation_matrix.json`.  Prove
that both raw SAIFs receive distinct two-level file seals before parsing, the
prehistory file is diagnostic-only and never annotated, and the measurement
file retains the exact 93,971-record, all-TX-zero, conservation, critical-cone,
ledger, scoreboard, and knownness gates.  Verify one direct M2018 frontend at
`SCHEDULE_MODE=0`, exact slot 42 fixture identity, the M2152 failure-hammer
lineage, contract seals, author-receipt exhaustive seals, tool identities,
docs/359 SHA, zero source-author EDA, and fresh M2162 attempt/result absence.

Only P0/P1/P2=0 and score at least 95 may authorize one M2162 license query,
one VCS compile, one ordinary simv, and exactly two raw SAIF writes.  M2161
must not itself run a license query, VCS, simv, DC, PT/PTPX, ICC2, or GPU.
