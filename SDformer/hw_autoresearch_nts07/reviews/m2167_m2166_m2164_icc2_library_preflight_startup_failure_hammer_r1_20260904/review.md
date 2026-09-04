# M2167 independent failure hammer of the M2166 ICC2 library preflight

## Verdict

**PASS failure diagnosis; M2166 is consumed, permanently non-retriable under
that identity, and noncitable.**  The unique sealed quarantine records exit
code 1 before the execution-contract writer, license client, process monitor,
or ICC2 launch site.  It contains no library-conversion output and proves no
library compatibility, physical implementation, timing, area, or power fact.

The exact cause is a shell startup defect.  The runner creates four directory
operands with plain `mkdir`: `home`, `tmp`, `cache/xdg`, and `cache/library`.
The first two exist in the quarantine, while the `cache` parent and both nested
children do not.  Under `set -e`, the nonzero `mkdir` return exits the runner.
The next statement, which would write `execution_contract.json`, is never
reached; neither are the later `lmutil` and `icc2_shell -no_init` sites.

The attempt marker's `license_queries=1` and
`top_level_icc2_shell_runs=1` fields are the irreversibly reserved one-shot
budget, not observed process counts.  The observed counts are **zero license
queries, zero ICC2 sessions, and zero P&R runs**.  This is established by the
fail-fast statement order together with the exact partial directory state and
the absence of every post-`mkdir` artifact; it is not inferred from a live
process snapshot taken after the failure.

## Seal, attempt, and root-inventory audit

- Exactly one M2166 attempt marker and one M2166 quarantine exist; the canonical
  result path does not.
- Both directories are symlink-free and pass exhaustive inner manifests plus
  outer manifest seals.
- The quarantine has exactly four payload files: its failure marker, the
  293-node repository-root pre-snapshot and log, and the byte-exact copied
  M2135 collateral.  Empty directories are represented by the sealed tree but
  do not appear in the file manifest.
- `RUN_FAILED_OR_INCOMPLETE.txt` says exit code 1 and `retry=false`.  There is no
  `RUN_COMPLETE.txt`, receipt, checker log, execution contract, license log,
  ICC2 log/return code, monitor log/tree/readiness marker, or launch gate.
- The root snapshot is valid JSON with schema
  `m2153_repo_root_inventory_r1_v1`, 293 unique top-level names, and the exact
  matching PASS log.  Its node types are the regular files and directories
  present at the snapshot time.
- The copied and still-preserved repository-root M2135 transcript both hash to
  `0410c14052c0b18c0f1a92246ecec4f109a9e37130b8f95f5cb4587cbcf863d6`.
- Protected `docs/359` remains
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

The seal proves a correctly quarantined startup failure.  It does not turn the
failure into a successful preflight.

## Minimal additive successor

For the exact observed cause, `mkdir -p` semantics are sufficient, but the
successor must make the ordering auditable rather than change only one flag.
Use a fresh M2168 source and fresh attempt/result paths; do not edit M2164 or
retry M2166.

Before any external license/tool command, M2168 should:

1. create the fresh attempt/work/isolated roots;
2. take the repository-root pre-snapshot and copy the frozen prior collateral;
3. use one fail-fast `mkdir -p --` operation to create `home`, `tmp`,
   `cache/xdg`, `cache/library`, `frame_output`, `frame_logs`, and `reports`;
4. require every listed path to be a real directory, not a symlink, and to
   resolve strictly below the fresh isolated root; keep the design `.nlib` and
   generated frame `.ndm` absent because those are ICC2 outputs;
5. write and re-read an execution contract that names those exact paths; and
6. only after all preceding gates pass, issue the single license query, start
   the monitor, release the launch gate, and invoke one `icc2_shell -no_init`
   session.

The source self-tests and independent hammer must mutate the cache parent away,
substitute a symlink, precreate a design/frame output, and move the license or
ICC2 site before the directory/contract gates.  Every mutation must be rejected
without external execution.  `mkdir -p` fixes nested-parent creation; the
postconditions prevent it from silently accepting a hostile or stale path.

## Release sequence and claim boundary

- M2168: additive source and author self-tests only.
- M2169: independent source hammer.  It alone may authorize one M2170
  library-only attempt if score is at least 95 and P0/P1/P2 is 0/0/0.
- M2170: at most one license query and one top-level ICC2 session, no P&R and no
  retry.
- M2171: independent result hammer required even if M2170 reports raw PASS.

M2167 authorizes M2168 source authoring and the M2169 source hammer only.  It
authorizes no license query, ICC2 invocation, or P&R.  No EDA executable,
license client, GPU job, source-under-review edit, paper edit, or `docs/359`
edit was performed by this review.

Score for the failure diagnosis: **100/100**, P0/P1/P2 = **0/0/0**.  This score
does not score M2166 as a successful experiment; M2166 remains permanently
failed and absent from TCAS-II/ISCAS evidence tables.
