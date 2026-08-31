# M839 receipt-blind fresh source hammer: M836 PASS100

## Verdict

`PASS100_M836_SOURCE_CANDIDATE__AUTHORIZE_TRUE_RELEASE_ONLY`, score 100, P0/P1/P2 = 0/0/0.

This authorizes only authorship of a separately sealed M836 true release. It does not authorize invoking the formal runner or launching production.

## Publication-boundary result

M836 closes the M835 P1. Parent, results, and private-stage directories are opened with directory and no-follow flags. Attempt members are created exclusively, recorded by type/dev/inode, and checked against exact bytes through a stage FD that stays open across `renameat2(RENAME_NOREPLACE)`. The same exact population and bytes are rechecked immediately before and after publication; the current results pathname and canonical attempt inode are rebound after publication.

Independent attacks replaced the current results directory after the final prepublication rebind and after publication. Both were rejected; the exact self-publication was removed through the pinned old results FD, and replacement artifacts remained unchanged. Same-inode member mutations were rejected and exactly rolled back. Unlink-recreate and unknown-member attacks were rejected with cleanup refusal, preserving attacker-owned objects. Stage-path replacement, stage/canonical collisions, wrong nonce, prefix changes, alternate canonical paths, and 13 candidate identity mutations all failed closed.

## Regression and frozen semantics

Python 3.10 and Python 3.6 each passed compile, M836 12/12, M832 12/12, M828 12/12, M809 9/9, M815 10/10, self-test, candidate validation, and the clean exact M836→M832→M828→M819→M809 traversal. The traversal stopped at M809 `output.mkdir` with zero scheduled rows, no output, unchanged attempt identity, and restored delegated validators.

The production meaning remains exactly 40 M686 plus 120 M699 records, T10, A1/K1x8/K8, 96 lanes, 245760 B, Acc24, 3 ns, and 192 B/cycle. D1 remains charged/nonheadline; only typed signed K8 versus equal-service K1x8 may headline.

No true release, formal attempt, result, failure, log, production row, VCS, EDA, license, GPU, or remote workload was created or invoked. `docs/359` remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
