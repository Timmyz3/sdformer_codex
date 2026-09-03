# M1995 independent failure review of M1992 matched TSBG DC

## Verdict

**PASS failure diagnosis.**  M1992 was consumed exactly once and correctly
failed closed.  Its quarantine and attempt directories both pass their inner
and outer SHA seals.  There is no published M1992 result, no retry namespace,
and no `tsbg_b4` axis directory: the first `ordinary_lru4` analyze failed
before elaboration, so the second axis was never started.

The unique **observed first-cause class** is Synopsys Presto rejecting the
standalone SystemVerilog keyword `context` as an identifier (`VER-294`).  The
first error is M1880 line 207.  Presto then reports the same class at lines
214, 469, 473, 474, and 480 before its error limit.  `VER-40` is the error-limit
termination; `LBR-0` and `UID-4` follow because no WORK design was created.
The adapter's earlier `VER-104` unlabeled-`$fatal` warning is noncausal at this
legal parameter point.  The run ended after 11 seconds at 677 MB, its license
preflight was healthy, and there is no timeout, OOM, allocation, crash, or
license-denial signature.  No functional simulation or synthesized structure
was reached, so this failure is neither evidence of a functional bug nor of a
structural/PPA failure.

This diagnosis does not claim that no later parser or synthesis error can be
revealed after the keyword repair; the successor must still pass fresh VCS and
DC gates.

## Complete token map and exact additive repair

M1880 SHA remains
`8524f6a7a6d09e1aaab55ee91515bd1fce9ea57fa2a478a9817f637685299a05`.
It has exactly 16 standalone `context` tokens:

| Line | Columns | Role |
|---:|:---|:---|
| 207 | 27 | `half_has_source` formal declaration |
| 214 | 40 | formal use in `active_q` index |
| 469 | 22, 35, 53 | reset loop declaration, condition, increment |
| 470 | 31 | reset `context_tag_q` index |
| 473 | 34 | reset `active_q` index |
| 474 | 32 | reset `sign_q` index |
| 480 | 31 | reset `acc_q` index |
| 628 | 26, 39, 57 | bundle-retire loop declaration, condition, increment |
| 629 | 35 | retire `context_tag_q` index |
| 633 | 38 | retire `active_q` index |
| 634 | 36 | retire `sign_q` index |
| 640 | 35 | retire `acc_q` index |

The minimum acceptable M1995 repair is a **new additive source file** whose
only byte-level semantic edit is replacing those 16 standalone tokens with
`ctx`.  M1880 itself must not be modified.  The module/top name remains
`m1880_c2_tsbg_b4_real_channel_signed_frontend`; substrings such as
`load_context`, `expected_context_q`, and `context_tag_q` remain unchanged.
The exact modeled successor SHA is
`2c1a8a7644b359a153decdc3106a8718992d37d54809007b61e184121fcc14fd`.
It is 64 bytes shorter, exactly `16 * (7-3)`.  Six mutations that missed a
rename or changed a substring, top, schedule default, comment, or replacement
name were all rejected.

## Successor gates

A new-numbered one-shot chain is justified because the old namespace is
permanently consumed, but this review does **not** authorize EDA before the
new source and runners receive exact-SHA source reviews.

1. **M1995 source gate:** new file only; exact modeled SHA above; normalized
   `ctx -> context` reconstruction must be byte-identical to M1880; M1880 and
   docs/359 SHAs unchanged; filelist excludes old M1880 and includes the new
   source exactly once.
2. **Fresh VCS gate:** compile M803, exact M1995 source, the existing M1880 SVA,
   and M1984 TB under a new no-retry namespace.  Require the prior arithmetic,
   work-conservation, LRU4, typed-sign, stale/replay, reset-recovery, phase,
   load, and cover gates; a new independent result hammer is mandatory.  The
   old M1990 VCS result cannot be rebound to the new source identity.
3. **Fresh matched-DC source gate:** only after VCS admission.  Preserve the
   two axes, G48 defaults, same top/ports/Tcl/SDC/libraries, `SCHEDULE_MODE`
   as the sole elaboration delta, six-hour per-axis timeout, single-attempt
   quarantine, exact bootstrap whitelist, WNS-min parser, M1866 identity,
   and existing claim boundaries.
4. **Fresh matched-DC result gate:** both analyze/elaborate/compile paths and
   required artifacts must exist; all non-whitelisted errors fail.  Area,
   setup, port equality, and any candidate decision remain pending a different
   independent result review.  Hold, power, energy, exact cycle ratio,
   same-area, G48 dynamic verification, and system speedup remain false.

No EDA tool or license query was launched, and docs/359 was not modified.
