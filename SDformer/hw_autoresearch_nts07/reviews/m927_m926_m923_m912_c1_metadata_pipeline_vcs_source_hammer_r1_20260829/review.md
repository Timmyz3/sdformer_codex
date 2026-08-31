# M927 — M926/M923/M912 C1 metadata-pipeline VCS source hammer

## Verdict

`PASS_M927_M926_M923_M912_SOURCE_HAMMER`, **100/100, P0=0, P1=0**.
This is a read-only source-integrity verdict.  The hammer did not invoke the
runner, VCS, `simv`, DC, PT, Formality, any other EDA shell, or a license query.
It created no result, attempt, work identity, or release.

## Closed P1

M924's sole P1 is closed.  M926's recursive completeness checker now walks
with `os.walk(..., followlinks=False)`, prunes symbolic-link directories,
classifies every file entry with `os.lstat`, and admits only
`stat.S_ISREG`.  Neither `Path.rglob()` nor `Path.is_file()` occurs in that
checker.  This is exactly the non-following regular-file set produced by the
runner's `find -P ... -type f` manifest command.

The frozen M919 quarantine independently reproduces as 111 manifest entries
and 111 actual non-symlink regular files, with no missing or extra entry.  Its
two tool-created links are excluded:

- `csrc/_3541518_archive_1.so`
- `simv.vdb/snps/coverage/db/testdata/test/assert.verilog.shape.xml`

For comparison, the frozen M923 `Path.is_file()` predicate follows those two
links and produces 113 entries.  Both the M919 inner manifest and outer seal
verify; the quarantine remains `FAILED_OR_INCOMPLETE_DO_NOT_CITE` and was not
modified.

## Additive runner delta

The byte diff from frozen M923 shows one semantic repair: recursive-seal
completeness enumeration.  All other changes are additive identity and
predecessor gates, M927/M928/result/attempt/work namespaces, and the M926
receipt/final-token names.  The M923 TB and static checker, M912 RTL, frozen
M919 SVA, macro adapter, foundry model and tool binary match their pinned
hashes.  The M926 and frozen M923 contracts both pass their inner and outer
seal checks.  The sealed M924 predecessor remains the expected non-PASS
90/100 review with `M924-P1-001`.

## Preserved functional source contract

The M923 phase repair still waits for `execute_busy`, advances to a core
negedge, and fail-closes unless active and next contexts are both exactly
invalid.  It corrupts backing row 1 to parent 63 with that parent dead, then
requires row 1 to enter active or next context with `relation_ok===0` within
64 iterations before applying the inherited 20-cycle sticky-fault watchdog.
Neither cached `relation_ok` signal is forced.

All six attacks are defined and called exactly once.  The 14 normal coverage
minima, P2 strength, held-final recovery, strict prefetch `{pop,row}` ordering,
and unique PASS/coverage/phase/held-final tokens remain present.  The foundry
`UNIT_DELAY` define is retained, while `+notimingcheck` and `+no_notifier` are
absent.

## Launch boundary

The fixed M926 result, attempt, work and failed-work namespaces were absent at
review time.  Before creating an attempt, the runner requires this fixed-path
recursively sealed M927 and a separately authored, double-sealed M928 release.
M928 must bind the exact SHA-256 values of this review, its manifest and its
outer-seal file, as well as the exact runner/contract/source identity.  The
runner then applies exact process-collision and 64-GiB available-memory gates,
a 600-second simulation timeout, quarantine-on-failure, and recursive result
sealing.

M927 does not author M928 and does not itself authorize a launch.  A separate
release author may now bind this sealed trio.

## Claim boundary

This review establishes source integrity only.  It establishes no functional
VCS result, timing, cycles, speedup, PPA, energy, system result, headline, or
paper-citable claim.
