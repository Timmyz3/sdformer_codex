# M924 | M923/M912 C1 metadata-pipeline VCS source hammer

## Verdict

**FAIL CLOSED: 90/100, P0=0, P1=1.**  Status is
`FAIL_M924_M923_M912_SOURCE_HAMMER`; this review does not authorize M925 or an
M923 attempt.  The hammer was read-only and source-only: it did not invoke the
runner, VCS, `simv`, DC, PT, Formality, another Synopsys shell, or a license
query, and it created no attempt, work, result, or release identity.

## P1 blocking finding

The exact runner's `verify_recursive_seal` applies two different file-set
semantics.  Its manifest producer uses `find -P ... -type f`, which excludes
symbolic links.  Its independent completeness check uses `Path.is_file()`,
which follows file-target symbolic links.  The frozen M919 quarantine contains
two tool-generated links already documented by M922:

- `csrc/_3541518_archive_1.so`
- `simv.vdb/snps/coverage/db/testdata/test/assert.verilog.shape.xml`

The sealed manifest has 111 entries and exactly matches the 111 non-symlink
regular files.  The runner's Python expression instead produces 113 entries,
so its line-71 `assert listed == actual` fails with those two paths in
`actual-listed`.  This gate is before M925 validation and before attempt
creation.  Therefore the current exact runner cannot consume an otherwise
valid fixed M919 quarantine and cannot reach an authorized M923 launch.

The quarantine itself has not drifted: its inner manifest, outer seal, exact
manifest SHA, exact outer-seal-file SHA, and the M922 forensic review/manifest/
outer identities all verify.  M922 explicitly records the same two symlinks
and defines the seal as `find -P -type f` regular-file coverage.  The defect is
the new runner's inconsistent completeness predicate, not the frozen evidence.

Required repair: do not modify the M919 quarantine.  Make the runner's Python
set use the same non-following regular-file policy as `find -P -type f` (for
example, exclude `p.is_symlink()`), then issue fresh runner/contract/hammer/
release identities.

## Checks that passed

- Live identities match the presented runner, RTL, frozen M919 SVA, M923 TB,
  static checker, frozen M919 TB, docs/359, and hammer request.  The live M923
  contract SHA is `58a57921a8b1bca0d52c92d774eba898d2a5d7e05a59a3ce4f181a7a05d5cd79`;
  both contract seals verify.  `bash -n` passes for the exact runner.
- Byte comparison proves that the M923 TB prefix and suffix outside
  `attack_wrong_parent_and_dead_live` are identical to M919.  Only that task
  changed (a 2161-byte additive delta).
- The repaired attack waits for `execute_busy`, advances to a core negedge,
  requires active and next contexts to be exactly invalid, checks bank 0,
  forces backing row 1 parent 63 and parent-live 0, and never forces either
  cached `relation_ok_q` bit.
- A 64-iteration capture watchdog requires row 1 in the active or next context
  with the corresponding relation predicate exactly zero.  Only after that
  witness does the test call the inherited 20-cycle sticky-fault watchdog.
- All six attack definitions and calls occur exactly once.  Normal 14-cover,
  P2, held-final, unique PASS/coverage/phase/held tokens, fail-closed labels,
  and foundry `+define+UNIT_DELAY` gates remain present; forbidden
  `+notimingcheck` and `+no_notifier` options remain absent.
- The independent static checker passes.  The fixed M923 result, attempt, work
  pattern, failed-work pattern, and M925 release were absent at audit time.
- Structurally, the runner requires a fixed-path recursively sealed M924 and a
  separate double-sealed M925 before attempt creation.  M925 must bind this
  hammer's exact review, manifest, and outer-seal hashes.  This correct later
  gate does not cure the earlier P1 quarantine-verifier failure.

## Claim boundary

This is source-integrity evidence only.  It establishes no functional VCS
result, timing, cycle count, speedup, PPA, energy, system result, headline, or
paper-citable claim.  M919 remains `FAILED_OR_INCOMPLETE_DO_NOT_CITE`.
