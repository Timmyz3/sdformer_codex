# M208 admission revocation

The original 256-case sweep did not reach the legal 48-event per-packet bank
sum.  Independent VCS review did and exposed a five-bit truncation/deadlock in
M207.  Therefore M208 is no longer sufficient to admit M207 or the M209 frozen
replay.  The recurrence itself models non-truncated counts and is reused only
after the M210 six-bit fix and bank48 regression.
