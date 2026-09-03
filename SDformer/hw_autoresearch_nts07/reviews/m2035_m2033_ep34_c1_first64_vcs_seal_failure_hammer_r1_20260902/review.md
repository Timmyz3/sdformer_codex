# M2035 independent M2033 canonical-seal failure review

## Verdict

**PASS, 100/100 for failure classification; P0/P1/P2 = 0/0/0.**  The sole
M2033 production identity was consumed exactly once.  Its VCS compile and
`simv` process both returned zero, the exact full terminal line occurs once,
and no functional error, fatal, assertion failure, watchdog, counter mismatch,
numeric mismatch, or protocol-error token occurs.  Those observations remain
diagnostic only because publication failed before a canonical result was
sealed.

The attempt marker and M2034 authority trees both verify against their double
seals.  The executed runner SHA is
`7a3f7340955edcdb5eb68e28c1b92a6fbf3f2fe2baeba8037f254978322ea41d`,
exactly as bound by M2034 and its launch release.  There is one attempt marker,
one failure quarantine, no leftover private stage, and no canonical M2033
result.

## Failure sequence and root cause

The frozen runner writes `receipt.json` and `RUN_COMPLETE.txt` only after the
zero compile/simulation return codes, exact PASS-cardinality check, and negative
error-token gate.  It then removes only `.vcs.timestamp` and invokes the
canonical `seal_dir`.  That function rejects the tree if any symlink exists.

VCS generated exactly one symlink:

`csrc/_2362104_archive_1.so -> .//../simv.daidir//_2362104_archive_1.so`

It resolves to the regular 573,944-byte file inside the same private result
tree, with SHA-256
`6e63b0e29cf867d67d6eb68fbfd434cbed4b26a6bbf6176d3a20ec22995924c8`.
No input, RTL, TB, fixture, or foundry model is a symlink.  The canonical seal
therefore returned one; the EXIT handler added `FAILED_DO_NOT_CITE` with
`exit_code=1`, retried the same zero-symlink seal unsuccessfully, and moved the
unsealed private stage to the failure quarantine.  The quarantine consequently
has no root `SHA256SUMS` or outer seal.

This is a deterministic packaging-policy incompatibility with a normal VCS
derived artifact, not a functional RTL, fixture, numeric, or protocol failure.
The pending receipt and logs are hashes bound by this review only as the state
observed during forensic inspection; M2035 does not retroactively seal or admit
the old VCS result.

## Permanent boundary of the old attempt

The old M2033 attempt is permanently consumed and is
`FAILED_OR_INCOMPLETE_DO_NOT_CITE`.  It may not be renamed, completed in place,
post-sealed, or presented as a VCS validation result.  Its apparent PASS cannot
enter the paper table.  In particular it does not promote the M1590
`1.694510x` CPU cycle model to RTL cycles, same-area speedup, full-network
performance, or system speedup.

## Narrow successor disposition

A manually initiated successor is technically justified, but this review
authorizes only **authoring** its source package.  It does not authorize VCS,
`simv`, a license query, GPU work, or any execution.

The successor must use a new milestone, runner SHA, source review, launch
release, attempt path, result path, and independent result review.  It must
rerun the same exact frozen RTL/TB/fixture/foundry/UNIT_DELAY workload; it may
not reuse or salvage this quarantine as a result.  The sole packaging repair
permitted before canonical sealing is:

1. enumerate all generated symlinks after simulation and require exactly the
   expected VCS `csrc/_<pid>_archive_1.so` form;
2. require its raw target to be the matching
   `simv.daidir/_<pid>_archive_1.so`, resolve inside the private stage, and
   verify that target is a regular file;
3. record path, raw target, resolved relative target, target SHA-256, and the
   removal action in a regular `removed_vcs_symlinks.json` member;
4. unlink exactly that generated symlink, require zero remaining symlinks, then
   create and verify the canonical double seal before no-replace publication.

Any additional symlink, missing/mismatched target, input change, PASS/counter
change, or other cleanup must fail closed.  The successor claim boundary
remains one real ep34 64-row **mask** tile with synthetic deterministic signed12
values and zero prior psum.  Real weights/real prior psums, timing, power,
energy, the `1.694510x` ratio, and all system/headline claims remain false.

No EDA, simulation, GPU work, or license query was launched by this reviewer;
the runner, RTL, TB, fixture, failed attempt, and docs/359 were not modified.
