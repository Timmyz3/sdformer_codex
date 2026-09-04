# M1114 — M1102 C1 production result independent hammer

Verdict: **GO for the raw CPU same-ledger result only**.  The admitted totals are
`434,242,823` candidate cycles and `763,908,050` cycles for both
strongest-zero and same-coordinate-bit, or `1.7591725402×`.

## What was independently checked

- The result and attempt have valid nested atomic seals, exact payload member
  sets, regular files only, no symlinks, one consumed attempt, one published
  result, and no leftover M1102 lock/work/quarantine namespace.
- M1100, M1101, M1102, M1104 and the external M1108 final-launch authority all
  resolve to their frozen hashes.  The result's internal launch-authority field
  intentionally pins the earlier M1104 source hammer; M1108 is the external
  pre-launch trust root and binds the exact M1107 launcher and M1104 chain.
- The ten sample rows independently sum to the three published aggregates.
  Both baselines are byte-for-byte equal in cycle/overhead fields.  All three
  designs retain the same service digest, 50,088 delayed accesses, 33,392
  nominal excess accesses, and the same 960,000-cycle aggregate commit charge.
- Coverage is 812,160 tasks / 2,436,480 task-design work values, including
  12,522 legal work-8 occurrences.  Canonical work is on the eight-block
  lattice, baseline work is equal, and candidate parent counters match the
  frozen conservation constants.
- Capacity arithmetic is `122,880 + 49,152 + 42,880 = 214,912 B`, leaving
  `30,848 B` below the `245,760 B` budget.
- Fifteen attacks were rejected, including member and receipt mutation, extra
  and symlink members, duplicate keys, NaN, sample/aggregate disagreement,
  resealed forged preflight counts, and forged speedup/RTL/paper/capacity flags.

## Claim boundary

Legal: on the frozen H67 four-bottleneck-Conv, ten-sample CPU same-ledger
replay, exact-1RW product capture is `1.7591725402×` versus either frozen
baseline.  This is now citable only with the words **raw CPU same-ledger**.

It is not an RTL-cycle result, a mapped-gate speedup, a decoder-complete or
system speedup, final-checkpoint activity, or paper-ready SRAM/PPA/energy.
The physical SRAM macro timing/power boundary remains open.
