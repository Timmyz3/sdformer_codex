# M2024 independent M2020/M2018 VCS-source review

## Verdict

**PASS, 98/100; P0/P1/P2 = 0/0/0.** M2020 is a logic-free public-name
adapter whose parameter and port header is semantically identical to M2018.
Its five-row filelist contains M2018 exactly once, excludes M1995, and pins the
existing M803 adapter, M1880 SVA, and M1984/M1970 bounded testbench.

The sealed M2019 prerequisite was reverified. After removing comments and
strings, M2018 contains no runtime division or remainder operator, no old
dynamic `active_q`/`sign_q` cube, and no former runtime two-dimensional row
selection. Both compile-time modes map into the same 192-row live map and
12-by-16 selector. The common payload selector remains; this review does not
permit a zero-mux claim.

## Directed scope

The frozen environment elaborates `SOURCE_GROUPS=12` for both MODE=0 and
MODE=1. It requires 48 rows, 576 issues, 9,216 signed products, 24 commits,
576 versus 144 weight bundles, stale/replay fail-closed behavior, two reset
recoveries, and exact `-(-128)=+128` handling. The SVA source contains 24
assert properties and 11 cover properties. This is a directed G12 regression,
not a dynamic production-G48 proof.

## M2025 one-shot authorization

The exact runner is authorized for one bounded attempt:

- exact runner-self, M2024 review, sources, filelist, M2019 review, and
  docs/359 identity gates, with both review-directory seals rechecked;
- fresh result/attempt/work/lock namespace, same-UID Synopsys collision scan,
  and 16-GiB MemAvailable plus commit-headroom gates;
- exactly one `lmstat`, one VCS compile, and one `simv` execution; no retry and
  no other EDA;
- SVA compilation, `global_finish_maxfail=1`, a 180-second external timeout,
  one exact PASS line, ten exact phase pairs, 52/52 load begin/complete rows,
  and zero load timeout;
- sealed failure quarantine and sealed atomic success publication.

The project static test passed 10/10 under Python 3.6 and 3.12. The independent
hammer also passed under both interpreters and rejected 16/16 runner mutations.
`bash -n` passed. This reviewer executed no EDA tool and made no license query.

M2025 success will still be raw directed evidence pending a different-author
result hammer. This review does not admit production G48, function, same-area,
exact-cycle speedup, system speedup, timing, energy, paper, or headline claims.
