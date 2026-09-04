# M890 decoder GTLS source-author handoff

M890 implements a bounded, source-only group/transaction-level exact scheduler. It consumes frozen M785 compressed transactions, retains only dependency-referenced produced tokens, retires them after statically counted last use, applies a closed form only when the resource state makes the proof trivial, and otherwise executes the frozen per-request recurrence. Cycle priority is reconstructed from packed int64 endpoints. The compressed/group IR digest is intentionally distinct from the legacy expanded-address digest.

Author checks passed for synthetic 1K, real D0/A1/t0 1K and 10K against M768/M861, and real 100K against M861. All scheduled endpoints, compressed rows, expanded/commit hashes, terminal readiness, port calendars, same-cycle response-slot semantics and six cycle classes matched. The directed pytest suite passed 9/9.

This is not a decoder cycle result. The sealed full D0/A1/t0 row was not run. The future gate remains end-to-end wall time at most 9.320783571 s, exact M883 numerical identity, and peak RSS at most 512 MiB. The 100K author miter retained two endpoint populations and peaked at 938692 KiB, so it neither proves nor claims the future memory gate.

No production/full population, VCS, DC, PT, Formality, PTPX, GPU, remote or training action was performed. `docs/359` remains frozen.
