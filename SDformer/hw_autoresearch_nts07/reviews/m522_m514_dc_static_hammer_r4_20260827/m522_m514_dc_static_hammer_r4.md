# M522/M514 logic-only DC static hammer r4

Verdict: **STATIC NO-GO — do not execute runner `c50ce62dcda22c61a1263c2d194f41003036d604b4c7bcc1c271d9be82086005`.** Score 92/100, P0=1.

## What r4 closes

The historical VCS root contains exactly two VCS-generated symlinks. R4 now excludes symlinks from the regular-file set and admits only these exact tuples: path, raw link text, in-root resolved target, regular non-symlink target, sealed target membership, and target SHA must all match. The actual VCS root passes 94/94 regular files and 2/2 links. All non-VCS review roots remain zero-symlink.

The exact embedded verifier was replayed in isolation for 16 cases. The real VCS root passes only `historical_vcs_exact2` and fails `zero_symlink`; zero-link review roots show the inverse behavior. A third link, path drift, raw-text drift with the same target, target drift, out-of-root target, dangling target, directory target, unsealed target, and target-SHA drift all fail and leave a full failure inventory.

R4 also explains both earlier pre-tool exits. The first old verifier retained all 94 `./name` spellings and disagreed with normalized actual labels. The second normalized labels but followed exactly the two symlinks through `is_file()`, adding them to the regular set. Both exits precede resource admission, staging, and DC. No old or new M522 canonical, staging, quarantine, `dc.log`, `dc.rc`, receipt, or active DC process exists.

Staging and its EXIT trap are established before all four input-root verifiers. Each verifier prints root/profile/full inventory and stores the same JSON in staging. A successful output would seal all four inventories. The exact tool, RTL, SDC, Tcl, library, constraint, finite-JSON, atomic publication, and narrow claim gates remain present.

## Blocking P0

The literal r4 contract and request require **new DC staging, canonical, and quarantine to remain strictly zero-symlink**. Successful staging and canonical packages are checked with the zero-symlink verifier. The failure trap is not.

`m522_quarantine_incomplete` directly renames a failed staging or post-move canonical directory into quarantine and then writes `RUN_FAILED_OR_INCOMPLETE.txt`. If DC or post-processing created a symlink before a later gate failed, the output verifier would correctly reject the package, but the trap would preserve that symlink in the quarantine. There is no symlink inventory, no unlink/no-follow sanitization, and no post-move zero-symlink check. Thus the requested quarantine invariant is not true for all failed executions.

This cannot publish a false PASS because the quarantine is non-citable. It is still P0 here because zero-symlink quarantine was explicitly listed as an r4 hard gate.

Minimum repair: in the EXIT trap, use Python with no symlink following to record every symlink path and raw link text into a regular JSON file, unlink those symlinks, assert zero symlinks, move to quarantine, and assert zero symlinks again. Then bind the new runner/contract/request SHA in a narrow independent review. No DC should run before that review returns P0=0.

Three nonblocking P1s remain: basename-based seal exclusions are broader than root-relative exclusions; the collision check does not name directly launched `snps_shell`; and receipt gate counts rely on earlier exact shell gates. The ideal-clock report remains declarative P2.

No runner or EDA tool was executed. Production files and `docs/359` were not modified; `docs/359` remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
