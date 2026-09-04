# M2080 — M2067 R7 failure hammer

Verdict: **PASS failure forensics; R7 is permanently no-retry and supplies no paper result.**

R7 consumed one license preflight, one VCS compile, and six simulations. Slots 0–4 passed. Slot 5 (`sample=0`, `layer=19`, `token_start=11996`) emitted all eight expected row/chunk records, then failed the final ledger.

The frozen stats row has `expected_nonzero_codes=0`. R7 nevertheless required both address-check counters to be positive. For an all-zero workload, zero weight-memory requests are correct, so this is a checker false negative rather than proof of an RTL numerical or protocol defect.

Any successor must use a fresh identity, preserve R7 and its quarantine, require zero address checks for zero-source workloads, retain positive-address checks for nonzero workloads, and pass both cases before a full cohort may be authorized.

No R7 cycle ratio, speedup, energy, or system claim is admitted.
