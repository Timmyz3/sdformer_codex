# M530/M528 DW1RW r2 independent source-static hammer request

Perform a fresh, read-only source review. Do not run VCS, iverilog,
Verilator, DC, Formality, PT/PTPX, CPU/GPU analysis, or remote work. Do not
modify the author package or `docs/359_DATE终局冻结_20260813.md`.

Fail P0 if a malformed synthetic parent-only nonzero payload can create any
ready/accept/psum/completion/scratch/elision/prefetch/counter event before the
sticky fault. Fail P1 if stale or nonmatching parent-slot data participates in
overflow, or a held final can fault before a matching response is authoritative.

Independently trace the TB cleanroom parent/refcount/live calculation, full
bitmap check, deterministic event counters, and task-completion counter closure.
Require every one of the eleven named normal cover minima and exactly one
coverage-summary token; keep all six protocol attack counters separate. Verify
the directed malformed-parent and held-stale-then-legal-parent tests.

The runner is source for a future functional VCS attempt only. It must bind the
private manifest and exact foundry `.v` with no fallback and must not contain a
trace recurrence or CPU-DSE rerun. The r2 contract must keep functional VCS,
future trace recurrence plus two 1.50x gates, and sealed M528-r4 CPU DSE as
three separate attempt identities. Future DC must bind the `.db`; future
Formality must match nine macro blackboxes/cutpoints under separate admissions.

A static PASS requires P0=0 and P1=0 and a double seal. It does not authorize a
VCS launch; root must create a separate exact one-attempt launch admission.
