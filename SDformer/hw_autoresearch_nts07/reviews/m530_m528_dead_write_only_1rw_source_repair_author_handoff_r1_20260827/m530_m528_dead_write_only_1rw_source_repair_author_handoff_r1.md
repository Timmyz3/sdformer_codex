# M530/M528 DW1RW r2 source-repair author handoff

The repaired r2 source-only identity is complete. No HDL, EDA, CPU, GPU, or
remote job was run, no result directory or VCS launch admission was created,
and the failed M529 r1 identity remains untouched.

The P0 fix introduces a combinational preaccept predicate that prevents a
malformed parent-only nonzero payload from creating any current-beat
architectural event before the sticky fault. Parent operands and overflow are
now used only with a matching authoritative response, and the TB contains a
directed stale-positive held-final followed by a legal parent release.

The TB independently recomputes M504 parents, per-row refcounts, the complete
live bitmap, active rows, and parent-edge totals. It closes dynamic accepted
events and all required counters at task completion. Eleven named normal cover
groups each have an explicit minimum of one, one exact summary token, and
runner-side parsing; six protocol attacks have separate counters.

The contract separates three identities: future functional VCS, later
trace-driven RTL/cycle recurrence plus both 1.50x gates, and the already sealed
M528-r4 CPU DSE prerequisite. Functional VCS cannot claim recurrence, speedup,
PPA, energy, full-network performance, or a headline.

The next action is a fresh independent read-only source-static hammer. Only a
double-sealed PASS with P0=0 and P1=0 permits root to create a separate
one-attempt VCS launch admission.
